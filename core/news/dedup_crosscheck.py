"""Event-level semantic deduplication and independent source cross-checking."""

import hashlib
import re
from typing import Dict, List, Set, Tuple
from core.logger import get_logger
from core.models import ConflictInfo
from core.news.fetcher import RawArticle

logger = get_logger("dedup_crosscheck")

COMMON_ENTITIES: Set[str] = {
    "google", "deepmind", "openai", "microsoft", "anthropic", "meta", "nvidia",
    "apple", "amazon", "aws", "huggingface", "github", "ai", "artificial intelligence",
}

STOP_WORDS: Set[str] = {
    "the", "and", "for", "that", "this", "with", "from", "have", "been", "will",
    "about", "what", "which", "their", "there", "they", "here", "were", "when",
    "says", "said", "reporting", "reports", "today", "yesterday", "week", "month",
}


def extract_event_tokens(title: str, summary: str) -> Set[str]:
    """Extract distinct non-generic action, noun, and event-specific tokens."""
    combined = f"{title} {summary}".lower()
    words = re.findall(r"\b[a-z0-9_-]{3,}\b", combined)
    # Filter out generic stop words and common entity names to isolate the event action/topic
    event_tokens = {w for w in words if w not in STOP_WORDS and w not in COMMON_ENTITIES}
    return event_tokens


def is_syndicated_reproduction(article_a: RawArticle, article_b: RawArticle) -> bool:
    """Detect whether article B is merely a syndicated, reproduced, or wire copy of article A."""
    text_b = f"{article_b.title} {article_b.summary}".lower()
    source_a_clean = re.sub(r"[^a-z0-9]", "", article_a.source_name.lower())

    # Check for attribution markers referencing the other source
    syndication_markers = [
        f"via {article_a.source_name.lower()}",
        f"originally published on {article_a.source_name.lower()}",
        f"reported by {article_a.source_name.lower()}",
        f"source: {article_a.source_name.lower()}",
    ]
    if any(marker in text_b for marker in syndication_markers):
        return True

    # Exact or near-identical headline match across different secondary aggregators
    norm_title_a = re.sub(r"[^a-z0-9]", "", article_a.title.lower())
    norm_title_b = re.sub(r"[^a-z0-9]", "", article_b.title.lower())
    if norm_title_a == norm_title_b and len(norm_title_a) > 20:
        return True

    return False


def extract_tokens_from_text(text: str) -> Set[str]:
    """Extract distinct non-generic action, noun, and event-specific tokens from text."""
    words = re.findall(r"\b[a-z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS and w not in COMMON_ENTITIES}


class EventDeduplicator:
    """Clusters articles at the meaningful event level and detects cross-source conflicts."""

    @classmethod
    def are_same_event(cls, article_a: RawArticle, article_b: RawArticle) -> bool:
        """Determine if two articles describe the same underlying event.

        Strict Event-Level Rule:
        Avoids merging unrelated events that merely share an entity, company, model, or keyword.
        Must have meaningful event-action token overlap.
        """
        # Title-level event comparison (titles are high-density event summaries)
        title_tokens_a = extract_tokens_from_text(article_a.title)
        title_tokens_b = extract_tokens_from_text(article_b.title)

        if title_tokens_a and title_tokens_b:
            title_intersection = title_tokens_a.intersection(title_tokens_b)
            title_union = title_tokens_a.union(title_tokens_b)
            title_jaccard = len(title_intersection) / len(title_union)
            # If titles share 3+ distinct event tokens and have >= 35% overlap, it is the same event
            if len(title_intersection) >= 3 and title_jaccard >= 0.35:
                return True

        # Combined title + summary comparison
        tokens_a = extract_tokens_from_text(f"{article_a.title} {article_a.summary}")
        tokens_b = extract_tokens_from_text(f"{article_b.title} {article_b.summary}")

        if not tokens_a or not tokens_b:
            return False

        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        jaccard_similarity = len(intersection) / len(union)

        # Require at least 4 shared distinctive event tokens and >= 25% overlap
        return jaccard_similarity >= 0.25 and len(intersection) >= 4

    @classmethod
    def cluster_articles(cls, articles: List[RawArticle]) -> List[List[RawArticle]]:
        """Cluster list of raw articles into distinct event groups."""
        clusters: List[List[RawArticle]] = []

        for article in articles:
            matched_cluster = None
            for cluster in clusters:
                # Compare against the lead article of the cluster
                if cls.are_same_event(cluster[0], article):
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster.append(article)
            else:
                clusters.append([article])

        return clusters

    @classmethod
    def find_independent_sources(cls, cluster: List[RawArticle]) -> List[RawArticle]:
        """Filter out syndicated reproductions so only genuinely independent sources count for cross-checking."""
        independent: List[RawArticle] = []

        for article in cluster:
            is_dup = False
            for existing in independent:
                if is_syndicated_reproduction(existing, article) or is_syndicated_reproduction(article, existing):
                    is_dup = True
                    break
            if not is_dup:
                independent.append(article)

        return independent

    @classmethod
    def detect_conflicts(cls, cluster: List[RawArticle]) -> List[ConflictInfo]:
        """Detect disagreements across sources (e.g. conflicting pricing or dates)."""
        conflicts: List[ConflictInfo] = []
        if len(cluster) < 2:
            return conflicts

        # Detect price conflicts (e.g. '$20' vs '$30')
        prices_by_source: Dict[str, str] = {}
        for article in cluster:
            combined = f"{article.title} {article.summary}"
            price_matches = re.findall(r"\$\d+(?:\.\d+)?(?:\s*(?:billion|million|k|per month|/mo|/month))?", combined, re.IGNORECASE)
            if price_matches:
                prices_by_source[article.source_name] = price_matches[0].lower()

        distinct_prices = set(prices_by_source.values())
        if len(distinct_prices) > 1:
            conflicts.append(
                ConflictInfo(
                    topic="pricing/valuation",
                    conflicting_claims=[f"{src}: {val}" for src, val in prices_by_source.items()],
                    sources_involved=list(prices_by_source.keys()),
                    explanation="Sources report conflicting monetary amounts or pricing figures.",
                )
            )

        return conflicts

    @staticmethod
    def generate_event_id(cluster: List[RawArticle]) -> str:
        """Create a deterministic event ID from the primary article's title and URL."""
        lead = cluster[0]
        base_str = f"{lead.title.strip().lower()}_{lead.url.strip()}"
        return hashlib.sha256(base_str.encode("utf-8")).hexdigest()[:16]
