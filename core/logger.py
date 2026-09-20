"""Secret-redacting logger for AI Mitra."""

import logging
import os
import re
from typing import List, Pattern

# Regular expressions for identifying common secret patterns
SENSITIVE_PATTERNS: List[Pattern] = [
    # Telegram Bot Token (e.g. 123456789:ABCdefGh...)
    re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,45}\b"),
    # Google/Gemini API Key (e.g. AIzaSy... or standard Google AI keys)
    re.compile(r"\bAIza[0-9A-Za-z-_]{30,45}\b"),
    # New Gemini API Key format (e.g. AQ.Ab8RN...)
    re.compile(r"\bAQ\.[A-Za-z0-9_-]{40,60}\b"),
    # Bearer tokens in headers
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
]


class SecretRedactingFormatter(logging.Formatter):
    """Custom logging formatter that scrubs sensitive secrets from log output."""

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)
        # Dynamically register environment secrets if already set
        self.exact_secrets: List[str] = []
        for key in ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN"):
            val = os.environ.get(key, "").strip()
            if val and len(val) > 6 and "your_" not in val:
                self.exact_secrets.append(val)

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        redacted = original

        # Redact known regex patterns
        for pattern in SENSITIVE_PATTERNS:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)

        # Redact exact secret values known from environment
        for secret in self.exact_secrets:
            redacted = redacted.replace(secret, "[REDACTED_SECRET]")

        return redacted


def get_logger(name: str = "ai_mitra", level: str = "INFO") -> logging.Logger:
    """Obtain or configure a secure logger instance with secret redaction."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler()
        formatter = SecretRedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
