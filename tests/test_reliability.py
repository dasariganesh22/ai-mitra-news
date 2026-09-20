"""Automated test suite verifying the reliability, deduplication, and fact-checking pipeline."""

from unittest.mock import MagicMock, patch
import pytest

from core.models import ReliabilityState
from core.news.classifier import ReliabilityClassifier
from core.news.content_extractor import ContentExtractor
from core.news.dedup_crosscheck import EventDeduplicator
from core.news.fetcher import NewsFetcher, NewsSourceConfig, RawArticle
from core.news.source_resolver import OfficialSourceResolver, is_official_domain


# ==============================================================================
# 1. Event-Level Deduplication Isolation (Refinement 4)
# ==============================================================================

def test_dedup_does_not_merge_unrelated_events():
    """Verify that two articles sharing an entity but describing different events remain separate.

    Article 1: Google launches Gemma 3 open weights model
    Article 2: Google signs clean energy datacenter contract
    Both contain 'Google', but describe completely distinct actions/events.
    """
    article1 = RawArticle(
        title="Google Releases Gemma 3 Open Weights Model with Multimodal Capabilities",
        url="https://techcrunch.com/google-gemma-3",
        source_name="TechCrunch",
        summary="Google has officially launched Gemma 3, a new family of open weights models featuring multimodal vision.",
        category="ai_around_the_world",
    )

    article2 = RawArticle(
        title="Google Signs Historic Clean Energy Contract for Global Datacenter Expansion",
        url="https://venturebeat.com/google-energy-contract",
        source_name="VentureBeat",
        summary="Google announced a massive partnership to purchase nuclear and solar energy for its upcoming cloud infrastructure.",
        category="ai_around_the_world",
    )

    # 1. are_same_event must return False
    assert EventDeduplicator.are_same_event(article1, article2) is False

    # 2. cluster_articles must place them in two separate clusters
    clusters = EventDeduplicator.cluster_articles([article1, article2])
    assert len(clusters) == 2
    assert clusters[0][0].title == article1.title
    assert clusters[1][0].title == article2.title


def test_duplicate_articles_are_detected_and_clustered():
    """Verify that articles describing the same event from different publishers are merged."""
    art_tc = RawArticle(
        title="Anthropic Launches Claude 3.7 Sonnet Hybrid Reasoning Architecture",
        url="https://techcrunch.com/anthropic-claude-3-7-sonnet",
        source_name="TechCrunch",
        summary="Anthropic has released Claude 3.7 Sonnet featuring combined fast thinking and extended test-time reasoning.",
        category="ai_around_the_world",
    )

    art_vb = RawArticle(
        title="Anthropic Unveils Claude 3.7 Sonnet with Hybrid Reasoning Mode",
        url="https://venturebeat.com/claude-3-7-sonnet-hybrid-reasoning",
        source_name="VentureBeat",
        summary="Claude 3.7 Sonnet from Anthropic introduces simultaneous quick answers and deep reasoning capabilities.",
        category="ai_around_the_world",
    )

    assert EventDeduplicator.are_same_event(art_tc, art_vb) is True
    clusters = EventDeduplicator.cluster_articles([art_tc, art_vb])
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


# ==============================================================================
# 2. Source-Failure Resilience (Refinement 6)
# ==============================================================================

def test_source_failure_does_not_abort_entire_pipeline():
    """Verify that when one news source fails or times out, remaining sources continue processing."""
    failing_source = NewsSourceConfig(
        name="Broken Source",
        url="https://broken-example.com/rss",
        source_type="rss",
        enabled=True,
    )

    working_source = NewsSourceConfig(
        name="Healthy Source",
        url="https://healthy-example.com/rss",
        source_type="rss",
        enabled=True,
    )

    fetcher = NewsFetcher(sources=[failing_source, working_source])

    # Mock _fetch_single_source to fail for source 1 and succeed for source 2
    def mock_fetch_single(source, max_items=5):
        if source.name == "Broken Source":
            raise ConnectionError("Simulated network timeout or unreachable host")
        return [
            RawArticle(
                title="AI Breakthrough Announced",
                url="https://healthy-example.com/article1",
                source_name="Healthy Source",
                summary="Researchers have made a significant breakthrough.",
                category="ai_around_the_world",
            )
        ]

    with patch.object(fetcher, "_fetch_single_source", side_effect=mock_fetch_single):
        results = fetcher.fetch_all()
        # Pipeline must NOT crash and must return articles from the working source
        assert len(results) == 1
        assert results[0].source_name == "Healthy Source"
        assert results[0].title == "AI Breakthrough Announced"


# ==============================================================================
# 3. Official Source Verification at Claim-Level (Refinements 3 & 7)
# ==============================================================================

def test_verified_official_requires_actual_supporting_evidence():
    """Verify that discovering an official URL is NOT sufficient unless retrieved content supports the claim."""
    mock_extractor = MagicMock()
    resolver = OfficialSourceResolver(extractor=mock_extractor)

    article = RawArticle(
        title="OpenAI Releases Revolutionary Model Q",
        url="https://news-aggregator.com/openai-model-q",
        source_name="NewsAggregator",
        summary="A new model called Model Q has been announced by OpenAI.",
        category="ai_around_the_world",
        discovered_links=["https://openai.com/index/unrelated-blog-post"],
    )

    candidate_url = "https://openai.com/index/unrelated-blog-post"

    # Scenario A: Retrieved official content is unrelated and does NOT support the claim
    mock_extractor.extract_from_url.return_value = (
        "Welcome to the OpenAI research archives. Today we discuss general safety principles and historical compute.",
        [],
    )

    has_any, claim_map, _ = resolver.verify_claims_against_official_source(
        candidate_url, claims=["OpenAI Releases Revolutionary Model Q"]
    )
    # Must NOT be marked as supported
    assert has_any is False
    assert claim_map["OpenAI Releases Revolutionary Model Q"] is False

    # Scenario B: Retrieved official content actively contains supporting evidence
    mock_extractor.extract_from_url.return_value = (
        "Today we are officially releasing Model Q. Revolutionary model Q provides breakthrough capabilities across domains.",
        [],
    )

    has_any_b, claim_map_b, _ = resolver.verify_claims_against_official_source(
        candidate_url, claims=["OpenAI Releases Revolutionary Model Q"]
    )
    # Must be marked as supported
    assert has_any_b is True
    assert claim_map_b["OpenAI Releases Revolutionary Model Q"] is True


def test_official_verification_evaluates_at_individual_claim_level():
    """Verify that supporting one claim does NOT automatically treat every claim as supported (Rule 6)."""
    mock_extractor = MagicMock()
    resolver = OfficialSourceResolver(extractor=mock_extractor)

    claims = [
        "OpenAI releases Model Q",                      # Supported
        "Model Q will cost 500 dollars per month",       # NOT supported in official release
    ]

    mock_extractor.extract_from_url.return_value = (
        "Today OpenAI releases Model Q. It is available to all Plus and Pro subscribers starting immediately.",
        [],
    )

    has_any, claim_map, _ = resolver.verify_claims_against_official_source(
        "https://openai.com/index/model-q", claims=claims
    )

    assert has_any is True
    assert claim_map["OpenAI releases Model Q"] is True
    assert claim_map["Model Q will cost 500 dollars per month"] is False


# ==============================================================================
# 4. Cross-Checking & Syndication Filtering (Refinement 5)
# ==============================================================================

def test_cross_check_rejects_syndicated_duplicates():
    """Verify that syndicated or republished copies do NOT count as independent confirmation."""
    lead_article = RawArticle(
        title="Meta Releases Llama 4 Open Source Weights",
        url="https://techcrunch.com/meta-llama-4",
        source_name="TechCrunch",
        summary="Meta has open-sourced Llama 4 today.",
        category="ai_around_the_world",
    )

    # Syndicated copy acknowledging original source
    syndicated_copy = RawArticle(
        title="Meta Releases Llama 4 Open Source Weights",
        url="https://aggregator.net/meta-llama-4",
        source_name="AggregatorNews",
        summary="Originally published on TechCrunch: Meta has open-sourced Llama 4 today.",
        category="ai_around_the_world",
    )

    cluster = [lead_article, syndicated_copy]
    independent = EventDeduplicator.find_independent_sources(cluster)

    # Only 1 independent source should remain
    assert len(independent) == 1
    assert independent[0].source_name == "TechCrunch"

    # Reliability classification must NOT be cross-checked
    classifier = ReliabilityClassifier()
    fact_sheet = classifier.process_cluster(cluster)
    assert fact_sheet.reliability == ReliabilityState.REPORTED


# ==============================================================================
# 5. Conflicting Claims Preservation (Refinement 4 & Plan)
# ==============================================================================

def test_conflicting_reports_preserved_as_uncertain():
    """Verify that conflicting claims across sources are preserved in ConflictInfo and marked uncertain."""
    article_a = RawArticle(
        title="AI Startup Raises Funding at $20 Million Valuation",
        url="https://source-a.com/story",
        source_name="Source A",
        summary="The startup closed a round valuing the company at $20 million.",
        category="ai_around_the_world",
    )

    article_b = RawArticle(
        title="AI Startup Raises Funding at $30 Million Valuation",
        url="https://source-b.com/story",
        source_name="Source B",
        summary="Reports indicate the round valued the company at $30 million.",
        category="ai_around_the_world",
    )

    cluster = [article_a, article_b]
    conflicts = EventDeduplicator.detect_conflicts(cluster)

    assert len(conflicts) == 1
    assert conflicts[0].topic == "pricing/valuation"
    assert "Source A: $20 million" in conflicts[0].conflicting_claims
    assert "Source B: $30 million" in conflicts[0].conflicting_claims

    classifier = ReliabilityClassifier()
    fact_sheet = classifier.process_cluster(cluster)

    # Must be marked as UNCERTAIN due to conflicting claims
    assert fact_sheet.reliability == ReliabilityState.UNCERTAIN
    assert len(fact_sheet.conflicts) == 1


# ==============================================================================
# 6. Source Attribution & Data Model Integrity
# ==============================================================================

def test_source_urls_and_attribution_preserved():
    """Verify that original article URLs and canonical source names are preserved."""
    article = RawArticle(
        title="New Vision Agent Released",
        url="https://venturebeat.com/new-vision-agent",
        source_name="VentureBeat",
        summary="A computer vision agent for automation.",
        category="what_people_are_doing",
    )

    classifier = ReliabilityClassifier()
    fact_sheet = classifier.process_cluster([article])

    assert fact_sheet.primary_source_name == "VentureBeat"
    assert fact_sheet.primary_source_url == "https://venturebeat.com/new-vision-agent"
    assert len(fact_sheet.source_reported_facts) >= 1
    assert fact_sheet.source_reported_facts[0].source_url == "https://venturebeat.com/new-vision-agent"


# ==============================================================================
# 7. Content Extractor Functional & Limit Tests
# ==============================================================================

def test_content_extractor_functional_and_limits():
    """Verify HTML cleaning, script stripping, and 15,000-character limit enforcement."""
    sample_html = """
    <html>
        <head>
            <script>alert('malicious')</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <header><p>Site Header Navigation</p></header>
            <article>
                <h1>Major AI Milestone</h1>
                <p>This is the first paragraph describing the significant AI breakthrough in detail.</p>
                <p>Here is another paragraph containing relevant factual information about the model.</p>
                <a href="https://openai.com/research/milestone">Official Research Paper</a>
            </article>
            <footer><p>Copyright 2026</p></footer>
        </body>
    </html>
    """

    cleaned_text, links = ContentExtractor.extract_from_html(sample_html, max_chars=15000)

    # 1. Scripts and styling removed
    assert "alert('malicious')" not in cleaned_text
    assert "color: red" not in cleaned_text

    # 2. Main content extracted
    assert "Major AI Milestone" in cleaned_text
    assert "significant AI breakthrough" in cleaned_text

    # 3. Discovered links extracted
    assert "https://openai.com/research/milestone" in links

    # 4. Test character limit cap
    huge_html = "<p>" + ("Z" * 20000) + "</p>"
    capped_text, _ = ContentExtractor.extract_from_html(huge_html, max_chars=15000)
    assert len(capped_text) == 15000

