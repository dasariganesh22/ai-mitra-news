"""News ingestion, content extraction, and reliability fact-checking pipeline."""

from core.news.fetcher import NewsFetcher, NewsSourceConfig, RawArticle
from core.news.content_extractor import ContentExtractor
from core.news.source_resolver import OfficialSourceResolver, is_official_domain
from core.news.dedup_crosscheck import EventDeduplicator
from core.news.classifier import ReliabilityClassifier

__all__ = [
    "NewsFetcher",
    "NewsSourceConfig",
    "RawArticle",
    "ContentExtractor",
    "OfficialSourceResolver",
    "is_official_domain",
    "EventDeduplicator",
    "ReliabilityClassifier",
]

