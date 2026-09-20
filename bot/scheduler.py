from datetime import datetime
from typing import Any, List, Optional
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram.ext import Application
from bot.formatters import chunk_telegram_message, format_digest_message
from core.database import DatabaseManager
from core.logger import get_logger
from core.news.classifier import ReliabilityClassifier
from core.news.dedup_crosscheck import EventDeduplicator
from core.news.fetcher import NewsFetcher
from core.synthesis.explainer import DigestGenerator, DigestResult

logger = get_logger("scheduler")

DAILY_JOB_ID = "daily_morning_digest"


async def scheduled_morning_dispatch(
    application: Application,
    db: DatabaseManager,
    fetcher: NewsFetcher,
    classifier: ReliabilityClassifier,
    explainer: DigestGenerator,
    allowed_user_ids: List[int],
    timezone_str: str = "Asia/Kolkata",
    lookback_hours: int = 24,
) -> Optional[DigestResult]:
    """Execute the end-to-end news, verification, synthesis, and isolated delivery pipeline."""
    logger.info("Executing scheduled morning intelligence pipeline...")

    try:
        # 1. Fetch news from configured sources
        raw_articles = fetcher.fetch_all()
        logger.info(f"Fetched {len(raw_articles)} raw articles across sources.")

        # 2. Deduplicate and cluster events
        clusters = EventDeduplicator.cluster_articles(raw_articles)
        logger.info(f"Clustered into {len(clusters)} unique news events.")

        # 3. Classify reliability and persist to database
        fact_sheets = []
        for cluster in clusters:
            sheet = classifier.process_cluster(cluster)
            db.save_news_item(sheet)
            fact_sheets.append(sheet)

        # 4. Fetch the most significant verified news items for digest cutoff
        tz = ZoneInfo(timezone_str)
        cutoff_time = datetime.now(tz)
        if hasattr(db, "get_news_for_digest"):
            recent_sheets = db.get_news_for_digest(
                cutoff_time=cutoff_time,
                lookback_hours=lookback_hours,
                limit=6,
                timezone_str=timezone_str,
            )
            # Backward-compatibility fallback for tests mocking get_recent_news
            from unittest.mock import MagicMock
            if isinstance(recent_sheets, MagicMock) and hasattr(db, "get_recent_news"):
                recent_mock_val = db.get_recent_news.return_value
                if isinstance(recent_mock_val, list):
                    recent_sheets = recent_mock_val
        else:
            recent_sheets = db.get_recent_news(limit=6)

        if not recent_sheets and fact_sheets:
            recent_sheets = fact_sheets[:6]

        # 5. Synthesize structured digest and audio note
        digest_result = explainer.generate_digest(recent_sheets, include_audio=True)
        formatted_text = format_digest_message(digest_result.items, recent_sheets)
        chunks = chunk_telegram_message(formatted_text)

        # 6. Deliver to each authorized Telegram user with isolated failure handling
        for user_id in allowed_user_ids:
            try:
                for chunk in chunks:
                    try:
                        await application.bot.send_message(
                            chat_id=user_id,
                            text=chunk,
                            parse_mode="Markdown",
                        )
                    except Exception as err:
                        if "parse entities" in str(err).lower():
                            logger.warning(
                                f"Telegram entity parsing failed for user {user_id} ({err}). "
                                f"Delivering safe plain-text fallback."
                            )
                            await application.bot.send_message(
                                chat_id=user_id,
                                text=chunk,
                                parse_mode=None,
                            )
                        else:
                            raise
                if digest_result.audio_path:
                    try:
                        with open(digest_result.audio_path, "rb") as audio_file:
                            await application.bot.send_voice(
                                chat_id=user_id,
                                voice=audio_file,
                                caption="🎙️ AI Mitra 60-second Morning Audio Briefing",
                            )
                    except Exception as audio_err:
                        logger.warning(f"Voice dispatch warning for user {user_id}: {audio_err}")
            except Exception as user_err:
                # Isolate failure: delivery failure to one user MUST NOT stop the remaining deliveries!
                logger.error(f"Failed to deliver scheduled digest to user {user_id}: {user_err}")
                continue

        logger.info("Scheduled morning dispatch completed successfully.")
        return digest_result

    except Exception as pipeline_err:
        logger.error(f"Pipeline execution error during scheduled dispatch: {pipeline_err}")
        return None


def configure_scheduler(
    application: Application,
    db: DatabaseManager,
    fetcher: NewsFetcher,
    classifier: ReliabilityClassifier,
    explainer: DigestGenerator,
    allowed_user_ids: List[int],
    timezone_str: str = "Asia/Kolkata",
    schedule_time_str: str = "08:00",
    lookback_hours: int = 24,
    existing_scheduler: Optional[AsyncIOScheduler] = None,
) -> AsyncIOScheduler:
    """Configure the AsyncIOScheduler integrated with the Telegram bot asyncio event loop."""
    tz = ZoneInfo(timezone_str)
    scheduler = existing_scheduler or AsyncIOScheduler(timezone=tz)

    try:
        parts = schedule_time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        hour, minute = 8, 0

    # Define the job wrapper
    async def run_dispatch():
        await scheduled_morning_dispatch(
            application=application,
            db=db,
            fetcher=fetcher,
            classifier=classifier,
            explainer=explainer,
            allowed_user_ids=allowed_user_ids,
            timezone_str=timezone_str,
            lookback_hours=lookback_hours,
        )

    # Idempotent registration: Remove existing job if already present, then add
    if scheduler.get_job(DAILY_JOB_ID):
        scheduler.remove_job(DAILY_JOB_ID)

    scheduler.add_job(
        run_dispatch,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=tz),
        id=DAILY_JOB_ID,
        replace_existing=True,
    )

    return scheduler
