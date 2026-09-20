"""Configurable multi-source news fetcher with isolated failure resilience."""

from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo
import feedparser
from pydantic import BaseModel, Field
from core.logger import get_logger
from core.security import safe_http_get

logger = get_logger("news_fetcher")


class NewsSourceConfig(BaseModel):
    """Configuration definition for an independent news source."""
    name: str = Field(..., description="Human-readable source name")
    url: str = Field(..., description="Feed or endpoint URL")
    source_type: str = Field(default="rss", description="'rss', 'web', or 'api'")
    category: str = Field(default="ai_around_the_world", description="'ai_around_the_world' or 'what_people_are_doing'")
    enabled: bool = Field(default=True, description="Whether this source is currently active")


class RawArticle(BaseModel):
    """Normalized representation of a fetched article before deduplication and verification."""
    title: str
    url: str
    source_name: str
    summary: str
    category: str
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    event_date: Optional[str] = None
    discovered_links: List[str] = Field(default_factory=list)


DEFAULT_NEWS_SOURCES: List[NewsSourceConfig] = [
    NewsSourceConfig(
        name="Google News - Artificial Intelligence",
        url="https://news.google.com/rss/search?q=Artificial+Intelligence&hl=en-US&gl=US&ceid=US:en",
        source_type="rss",
        category="ai_around_the_world",
        enabled=True,
    ),
    NewsSourceConfig(
        name="TechCrunch - Artificial Intelligence",
        url="https://techcrunch.com/category/artificial-intelligence/feed/",
        source_type="rss",
        category="ai_around_the_world",
        enabled=True,
    ),
    NewsSourceConfig(
        name="VentureBeat - AI",
        url="https://venturebeat.com/category/ai/feed/",
        source_type="rss",
        category="ai_around_the_world",
        enabled=True,
    ),
    NewsSourceConfig(
        name="Hacker News - Show HN / AI Projects",
        url="https://hnrss.org/show?q=AI",
        source_type="rss",
        category="what_people_are_doing",
        enabled=True,
    ),
    NewsSourceConfig(
        name="GitHub Trending - AI & LLM Repositories",
        url="https://github.com/trending?since=daily",
        source_type="web",
        category="what_people_are_doing",
        enabled=True,
    ),
]


class NewsFetcher:
    """Orchestrates resilient fetching across all enabled news sources."""

    def __init__(
        self,
        sources: Optional[List[NewsSourceConfig]] = None,
        timezone: str = "Asia/Kolkata",
    ):
        self.sources = sources if sources is not None else DEFAULT_NEWS_SOURCES
        self.timezone = timezone

    def fetch_all(self, max_items_per_source: int = 5) -> List[RawArticle]:
        """Fetch news from all enabled sources.

        Source-Failure Resilience:
        If one source times out, returns malformed feed data, or becomes unreachable,
        the error is safely logged, and the remaining sources continue processing.
        """
        collected_articles: List[RawArticle] = []

        for source in self.sources:
            if not source.enabled:
                continue

            try:
                articles = self._fetch_single_source(source, max_items=max_items_per_source)
                collected_articles.extend(articles)
                logger.info(f"Successfully fetched {len(articles)} items from '{source.name}'.")
            except Exception as exc:
                # Isolated failure: Log warning and continue with remaining sources
                logger.warning(
                    f"Source '{source.name}' failed to fetch ({type(exc).__name__}). Continuing with remaining sources."
                )

        return collected_articles

    def _fetch_single_source(self, source: NewsSourceConfig, max_items: int = 5) -> List[RawArticle]:
        """Fetch and parse articles from a single configured source."""
        # Use safe_http_get with SSRF and size limits
        response = safe_http_get(source.url, timeout=10)

        if source.source_type == "rss":
            return self._parse_rss_feed(response.text, source, max_items=max_items)
        elif source.source_type == "web":
            return self._parse_web_source(response.text, source, max_items=max_items)
        else:
            logger.warning(f"Unsupported source type '{source.source_type}' for source '{source.name}'.")
            return []

    def _parse_rss_feed(
        self,
        feed_content: str,
        source: NewsSourceConfig,
        max_items: int = 5,
        timezone: Optional[str] = None,
    ) -> List[RawArticle]:
        """Parse RSS/Atom feed content using feedparser."""
        parsed_feed = feedparser.parse(feed_content)
        articles: List[RawArticle] = []
        tz_str = timezone or getattr(self, "timezone", "Asia/Kolkata")
        retrieved_now = datetime.now(ZoneInfo(tz_str)).isoformat()

        for entry in parsed_feed.entries[:max_items]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = getattr(entry, "summary", "").strip()
            published = getattr(entry, "published", None)
            updated = getattr(entry, "updated", None)

            if not title or not link:
                continue

            articles.append(
                RawArticle(
                    title=title,
                    url=link,
                    source_name=source.name,
                    summary=summary,
                    category=source.category,
                    published_at=published,
                    updated_at=updated,
                    retrieved_at=retrieved_now,
                    event_date=None,
                )
            )

        return articles

    def _parse_web_source(
        self,
        html_content: str,
        source: NewsSourceConfig,
        max_items: int = 5,
        timezone: Optional[str] = None,
    ) -> List[RawArticle]:
        """Parse non-RSS web sources (e.g. GitHub Trending) with BeautifulSoup."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        articles: List[RawArticle] = []
        tz_str = timezone or getattr(self, "timezone", "Asia/Kolkata")
        retrieved_now = datetime.now(ZoneInfo(tz_str)).isoformat()

        # Target GitHub trending repo boxes
        repo_rows = soup.find_all("article", class_="Box-row")
        for row in repo_rows[:max_items]:
            h2 = row.find("h2")
            if not h2:
                continue
            a_tag = h2.find("a")
            if not a_tag or not a_tag.get("href"):
                continue

            repo_name = " ".join(a_tag.get_text().split())
            repo_url = f"https://github.com{a_tag['href'].strip()}"
            p_desc = row.find("p")
            description = p_desc.get_text(strip=True) if p_desc else "Trending AI Repository"

            articles.append(
                RawArticle(
                    title=f"Trending Project: {repo_name}",
                    url=repo_url,
                    source_name=source.name,
                    summary=description,
                    category=source.category,
                    published_at=None,
                    updated_at=None,
                    retrieved_at=retrieved_now,
                    event_date=None,
                )
            )

        return articles

