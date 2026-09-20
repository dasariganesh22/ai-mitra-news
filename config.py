"""Centralized application configuration with testable fail-closed security validation."""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class SecurityConfigError(ValueError):
    """Raised when security-critical configuration is missing or invalid."""
    pass


class Settings(BaseModel):
    """Immutable application settings validated at startup."""

    gemini_api_key: str = Field(..., description="Google Gemini API key for intelligence engine")
    gemini_model: str = Field(default="gemini-3.8-flash", description="Gemini model name for synthesis and tutor")
    telegram_bot_token: str = Field(..., description="Telegram Bot Token from @BotFather")
    allowed_telegram_user_ids: List[int] = Field(..., description="Allowlist of Telegram User IDs")

    timezone: str = Field(default="Asia/Kolkata", description="Timezone for scheduled jobs")
    schedule_time: str = Field(default="08:00", description="Daily digest dispatch time in HH:MM")
    digest_lookback_hours: int = Field(default=24, description="Lookback window in hours for news digest")
    database_path: str = Field(default="data/ai_mitra.db", description="Path to SQLite database")
    log_level: str = Field(default="INFO", description="Application log level")

    # Security Boundaries and Limits
    max_user_message_chars: int = Field(default=4000, description="Max allowed length for incoming Telegram messages")
    max_article_extract_chars: int = Field(default=15000, description="Max length for extracted news content")
    http_timeout_seconds: int = Field(default=10, description="Network timeout in seconds for content fetching")
    max_http_content_bytes: int = Field(default=1024 * 1024, description="Max allowed response size (1MB)")
    rate_limit_requests_per_minute: int = Field(default=10, description="Max requests per minute per user")


def parse_allowed_user_ids(raw_value: Optional[str]) -> List[int]:
    """Safely parse comma-separated Telegram user IDs into a list of integers."""
    if not raw_value or not raw_value.strip():
        raise SecurityConfigError("ALLOWED_TELEGRAM_USER_IDS must be provided and non-empty.")

    ids: List[int] = []
    for item in raw_value.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        try:
            parsed_id = int(cleaned)
            if parsed_id <= 0:
                raise ValueError("User ID must be positive")
            ids.append(parsed_id)
        except ValueError as exc:
            raise SecurityConfigError(f"Invalid Telegram User ID '{cleaned}': must be a positive integer.") from exc

    if not ids:
        raise SecurityConfigError("ALLOWED_TELEGRAM_USER_IDS must contain at least one valid user ID.")
    return ids


def validate_and_load_config(env_dict: Optional[Dict[str, str]] = None) -> Settings:
    """Validate and load security-critical configuration from environment or provided dict.

    Fails closed by raising SecurityConfigError if any required security configuration
    is missing, blank, or set to placeholder defaults.
    """
    env = env_dict if env_dict is not None else os.environ

    # 1. Validate Gemini API Key
    gemini_key = env.get("GEMINI_API_KEY", "").strip()
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        raise SecurityConfigError("GEMINI_API_KEY is missing or contains placeholder text.")

    # 2. Validate Telegram Bot Token
    tg_token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not tg_token or tg_token == "your_telegram_bot_token_here":
        raise SecurityConfigError("TELEGRAM_BOT_TOKEN is missing or contains placeholder text.")

    # 3. Validate Allowed Telegram User IDs
    raw_user_ids = env.get("ALLOWED_TELEGRAM_USER_IDS", "").strip()
    if raw_user_ids == "your_telegram_user_id_here":
        raise SecurityConfigError("ALLOWED_TELEGRAM_USER_IDS contains placeholder text.")
    allowed_user_ids = parse_allowed_user_ids(raw_user_ids)

    # 4. Non-critical configuration with sensible defaults
    gemini_model = env.get("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
    timezone = env.get("TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"
    schedule_time = env.get("SCHEDULE_TIME", "08:00").strip() or "08:00"
    raw_lookback = env.get("DIGEST_LOOKBACK_HOURS", "24").strip() or "24"
    try:
        digest_lookback_hours = int(raw_lookback)
    except ValueError:
        digest_lookback_hours = 24
    db_path = env.get("DATABASE_PATH", "data/ai_mitra.db").strip() or "data/ai_mitra.db"
    log_level = env.get("LOG_LEVEL", "INFO").strip() or "INFO"

    return Settings(
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        telegram_bot_token=tg_token,
        allowed_telegram_user_ids=allowed_user_ids,
        timezone=timezone,
        schedule_time=schedule_time,
        digest_lookback_hours=digest_lookback_hours,
        database_path=db_path,
        log_level=log_level,
    )


_CACHED_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or initialize cached settings from current environment."""
    global _CACHED_SETTINGS
    if _CACHED_SETTINGS is None:
        _CACHED_SETTINGS = validate_and_load_config()
    return _CACHED_SETTINGS


def enforce_fail_closed_startup() -> Settings:
    """Explicitly called at application startup to abort process if security configuration is invalid."""
    try:
        return validate_and_load_config()
    except SecurityConfigError as err:
        # Fail closed: Do not allow partial or degraded startup
        raise SystemExit(f"[FATAL SECURITY CONFIGURATION ERROR] {err}") from err

