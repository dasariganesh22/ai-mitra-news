"""Official source identification and individual-claim verification against retrieved evidence."""

import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from core.logger import get_logger
from core.models import ToolMetadata
from core.news.content_extractor import ContentExtractor
from core.news.fetcher import RawArticle

logger = get_logger("source_resolver")

KNOWN_OFFICIAL_DOMAINS: Set[str] = {
    "openai.com",
    "blog.google",
    "deepmind.google",
    "anthropic.com",
    "github.com",
    "huggingface.co",
    "pypi.org",
    "meta.com",
    "ai.meta.com",
    "microsoft.com",
    "mistral.ai",
    "cohere.com",
}


def is_official_domain(url_or_domain: str) -> bool:
    """Check whether a URL or domain string belongs to an officially recognized organization/repository."""
    if not url_or_domain:
        return False

    try:
        if "://" in url_or_domain:
            hostname = urlparse(url_or_domain).hostname or ""
        else:
            hostname = url_or_domain.split("/")[0]

        hostname = hostname.lower().strip()
        for official in KNOWN_OFFICIAL_DOMAINS:
            if hostname == official or hostname.endswith("." + official):
                return True
        return False
    except Exception:
        return False


class OfficialSourceResolver:
    """Discovers and actively verifies claims against candidate official sources."""

    def __init__(self, extractor: Optional[ContentExtractor] = None):
        self.extractor = extractor or ContentExtractor()

    def identify_candidate_official_url(self, article: RawArticle) -> Optional[str]:
        """Scan article URL, title, and discovered links for an official announcement or repository."""
        # Case A: Article itself originates from an official domain
        if is_official_domain(article.url):
            return article.url

        # Case B: Discovered outgoing link points to an official domain
        for link in article.discovered_links:
            if is_official_domain(link):
                return link

        # Case C: Search for GitHub repository links in summary/text
        github_match = re.search(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", article.summary)
        if github_match:
            return github_match.group(0)

        return None

    def verify_claims_against_official_source(
        self,
        candidate_url: str,
        claims: List[str],
    ) -> Tuple[bool, Dict[str, bool], Optional[str]]:
        """Safely retrieves official content and verifies support at the individual-claim level.

        Rules:
        - An official URL being discovered is NOT sufficient.
        - Support is evaluated at the individual claim level: An official source supporting one claim
          must NOT automatically be treated as supporting every claim.

        Returns:
            (has_any_supported_claims, {claim: is_supported}, retrieved_content_summary)
        """
        if not candidate_url or not is_official_domain(candidate_url):
            return False, {claim: False for claim in claims}, None

        # Step 1: Safely retrieve candidate official content
        logger.info(f"Retrieving candidate official content from '{candidate_url}' for verification.")
        retrieved_text, _ = self.extractor.extract_from_url(candidate_url)

        if not retrieved_text or len(retrieved_text.strip()) < 50:
            logger.warning(f"Could not retrieve readable content from official URL '{candidate_url}'.")
            return False, {claim: False for claim in claims}, None

        # Step 2: Compare each claim individually against retrieved text
        claim_support_map: Dict[str, bool] = {}
        retrieved_lower = retrieved_text.lower()

        for claim in claims:
            is_supported = self._check_single_claim_support(claim, retrieved_lower)
            claim_support_map[claim] = is_supported

        has_any_supported = any(claim_support_map.values())
        return has_any_supported, claim_support_map, retrieved_text[:500]

    def extract_tool_metadata(self, article: RawArticle, official_url: Optional[str]) -> Optional[ToolMetadata]:
        """Extract tool details (repo, official site) if the article announces a software tool/model."""
        text = f"{article.title} {article.summary}"
        # Check for tool/launch indicators
        is_tool = any(kw in text.lower() for kw in ["github", "repo", "released", "launches", "tool", "open-source", "sdk", "model"])
        if not is_tool:
            return None

        # Determine tool name from title
        name_candidate = article.title.split(":")[0].strip()
        if len(name_candidate) > 40:
            name_candidate = " ".join(name_candidate.split()[:4])

        repo_url = None
        if official_url and "github.com" in official_url:
            repo_url = official_url
        elif "github.com" in article.url:
            repo_url = article.url

        return ToolMetadata(
            name=name_candidate,
            official_url=official_url or article.url,
            repo_url=repo_url,
            is_free=True if "open-source" in text.lower() or "free" in text.lower() else None,
            pricing_details="Free / Open Source" if "open-source" in text.lower() else None,
        )

    @staticmethod
    def _check_single_claim_support(claim: str, retrieved_lower_text: str) -> bool:
        """Check if an individual claim has affirmative evidence in the retrieved text.

        Uses root/prefix matching (e.g. 'release' matches 'releasing', 'releases', 'released').
        """
        # Tokenize claim into meaningful keywords
        words = re.findall(r"\b[A-Za-z0-9_-]{2,}\b", claim.lower())
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "have", "been", "will",
            "are", "has", "can", "our", "all", "its", "new", "how", "what", "who", "openai",
            "google", "anthropic", "meta", "microsoft"
        }
        keywords = [w for w in words if w not in stop_words]

        if not keywords:
            return False

        # Stemming / prefix matching
        matched_count = 0
        for kw in keywords:
            # Generate root stem
            stem = kw
            for suffix in ("ing", "ed", "es", "s", "tion", "able"):
                if kw.endswith(suffix) and len(kw) - len(suffix) >= 3:
                    stem = kw[:-len(suffix)]
                    break

            if stem in retrieved_lower_text or kw in retrieved_lower_text:
                matched_count += 1

        match_ratio = matched_count / len(keywords)
        return match_ratio >= 0.50
