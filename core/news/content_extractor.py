"""Safe content extraction from web pages and articles respecting SSRF and size limits."""

import re
from typing import List, Tuple
from bs4 import BeautifulSoup
from core.logger import get_logger
from core.security import enforce_article_content_length, safe_http_get

logger = get_logger("content_extractor")

UNWANTED_TAGS = ["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "svg", "form"]


class ContentExtractor:
    """Safely extracts clean readable text and discovered links from URLs or HTML strings."""

    @classmethod
    def extract_from_url(cls, url: str, timeout: int = 10, max_chars: int = 15000) -> Tuple[str, List[str]]:
        """Fetch a URL securely using safe_http_get and return (cleaned_text, discovered_links)."""
        try:
            response = safe_http_get(url, timeout=timeout)
            html_text = response.text
            return cls.extract_from_html(html_text, base_url=url, max_chars=max_chars)
        except Exception as exc:
            logger.warning(f"Failed safe content extraction from '{url}': {exc}")
            return "", []

    @classmethod
    def extract_from_html(cls, html_content: str, base_url: str = "", max_chars: int = 15000) -> Tuple[str, List[str]]:
        """Parse HTML, strip boilerplate/scripts, extract plain text and outgoing links."""
        if not html_content or not html_content.strip():
            return "", []

        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Extract outgoing links before modifying tree
        discovered_links: List[str] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if href.startswith("http://") or href.startswith("https://"):
                discovered_links.append(href)

        # 2. Decompose unwanted elements
        for tag_name in UNWANTED_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

        # 3. Target main content container if present, else body
        container = soup.find("article") or soup.find("main") or soup.find("body") or soup

        # Extract text from paragraphs and headings
        blocks = []
        for element in container.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li"]):
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 5:  # Keep meaningful headings and sentences
                blocks.append(text)

        if not blocks:
            # Fallback to general text extraction
            raw_text = container.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            cleaned_text = "\n\n".join(lines)
        else:
            cleaned_text = "\n\n".join(blocks)

        # 4. Collapse consecutive whitespace
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        # 5. Enforce configured max character limit
        safe_text = enforce_article_content_length(cleaned_text, max_chars=max_chars)

        return safe_text, list(dict.fromkeys(discovered_links))  # deduplicated links
