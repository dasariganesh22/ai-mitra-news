"""Reliability classifier implementing claim-aware verification and tri-partite separation."""

from typing import Dict, List, Optional, Tuple
from core.logger import get_logger
from core.models import ConflictInfo, NewsFactSheet, ReliabilityState, SourceFact, ToolMetadata
from core.news.dedup_crosscheck import EventDeduplicator
from core.news.fetcher import RawArticle
from core.news.source_resolver import OfficialSourceResolver, is_official_domain

logger = get_logger("reliability_classifier")


class ReliabilityClassifier:
    """Classifies news clusters into grounded reliability states with claim-level evidence."""

    def __init__(self, resolver: Optional[OfficialSourceResolver] = None):
        self.resolver = resolver or OfficialSourceResolver()

    def process_cluster(
        self,
        cluster: List[RawArticle],
        candidate_official_url: Optional[str] = None,
        tool_metadata: Optional[ToolMetadata] = None,
    ) -> NewsFactSheet:
        """Process an event cluster into a verified NewsFactSheet."""
        lead = cluster[0]
        event_id = EventDeduplicator.generate_event_id(cluster)

        # 1. Detect any conflicts across cluster articles
        conflicts: List[ConflictInfo] = EventDeduplicator.detect_conflicts(cluster)

        # 2. Extract initial candidate facts from lead article
        extracted_facts: List[str] = self._extract_key_claims(lead)

        # 3. Check official source verification
        official_url = candidate_official_url or self.resolver.identify_candidate_official_url(lead)
        is_official = False
        claim_support_map: Dict[str, bool] = {}

        if official_url:
            has_supported, claim_support_map, _ = self.resolver.verify_claims_against_official_source(
                official_url, extracted_facts
            )
            is_official = has_supported

        # 4. Check independent secondary sources for cross-checking
        independent_sources = EventDeduplicator.find_independent_sources(cluster)
        is_cross_checked = len(independent_sources) >= 2

        # 5. Determine overall reliability state
        # Priority: Conflicts -> Uncertain; Official evidence -> Official; 2+ independent -> Cross-Checked; else Reported
        if conflicts:
            overall_reliability = ReliabilityState.UNCERTAIN
        elif is_official:
            overall_reliability = ReliabilityState.OFFICIAL
        elif is_cross_checked:
            overall_reliability = ReliabilityState.CROSS_CHECKED
        else:
            # Check for rumor/speculation markers in text
            lower_text = f"{lead.title} {lead.summary}".lower()
            if any(w in lower_text for w in ["rumor", "leak", "unconfirmed", "speculation", "alleged"]):
                overall_reliability = ReliabilityState.UNCERTAIN
            else:
                overall_reliability = ReliabilityState.REPORTED

        # 6. Formulate SourceFact list with claim-aware attribution
        source_facts: List[SourceFact] = []
        for claim in extracted_facts:
            # If claim was individually verified by official source, attribute to official URL
            if claim_support_map.get(claim, False) and official_url:
                source_facts.append(
                    SourceFact(claim=claim, source_name="Official Primary Source", source_url=official_url)
                )
            else:
                source_facts.append(
                    SourceFact(claim=claim, source_name=lead.source_name, source_url=lead.url)
                )

        supporting_urls = [art.url for art in independent_sources if art.url != lead.url]

        return NewsFactSheet(
            event_id=event_id,
            headline=lead.title,
            primary_source_name=lead.source_name,
            primary_source_url=lead.url,
            supporting_sources=supporting_urls,
            reliability=overall_reliability,
            is_official_source=is_official,
            source_reported_facts=source_facts,
            ai_summary=lead.summary if lead.summary else lead.title,
            ai_interpretation=None,  # Generated on-demand in synthesis/tutor phase
            conflicts=conflicts,
            tool_metadata=tool_metadata,
            category=lead.category,
            published_at=lead.published_at,
            updated_at=getattr(lead, "updated_at", None),
            retrieved_at=getattr(lead, "retrieved_at", None),
            event_date=getattr(lead, "event_date", None),
        )

    @staticmethod
    def _extract_key_claims(article: RawArticle) -> List[str]:
        """Extract short, grounded factual statements from the article title and summary."""
        claims: List[str] = [article.title.strip()]
        if article.summary:
            sentences = [s.strip() for s in article.summary.split(".") if len(s.strip()) > 15]
            claims.extend(sentences[:2])
        return claims

