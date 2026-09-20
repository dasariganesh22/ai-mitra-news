"""Main application runner for AI Mitra.

Initializes database, news aggregation, synthesis pipelines, Telegram bot,
and AsyncIOScheduler with fail-closed security startup.
"""

import sys
from google import genai
from telegram.ext import Application
from bot.handlers import BotHandlers, register_handlers
from bot.middleware import SecurityGate
from bot.scheduler import configure_scheduler
from config import enforce_fail_closed_startup
from core.database import DatabaseManager
from core.logger import get_logger
from core.news.classifier import ReliabilityClassifier
from core.news.content_extractor import ContentExtractor
from core.news.fetcher import NewsFetcher
from core.synthesis.explainer import DigestGenerator
from core.synthesis.tutor import TutorEngine
from core.synthesis.voice import VoiceGenerator

logger = get_logger("ai_mitra_main")


def main() -> None:
    """Run AI Mitra application daemon."""
    logger.info("Initializing AI Mitra with fail-closed security checks...")

    # 1. Enforce fail-closed startup validation
    try:
        settings = enforce_fail_closed_startup()
    except Exception as exc:
        logger.critical(f"Startup halted due to invalid security configuration: {exc}")
        sys.exit(1)

    # 2. Initialize database
    db = DatabaseManager(settings.database_path)

    # 3. Initialize Gemini client
    gemini_client = None
    if settings.gemini_api_key and "your_" not in settings.gemini_api_key:
        try:
            gemini_client = genai.Client(api_key=settings.gemini_api_key)
            logger.info("Gemini AI client initialized successfully.")
        except Exception as err:
            logger.warning(f"Could not initialize Gemini AI client: {err}")

    # 4. Initialize news and synthesis components
    fetcher = NewsFetcher(timezone=settings.timezone)
    classifier = ReliabilityClassifier()
    content_extractor = ContentExtractor()
    voice_gen = VoiceGenerator()
    explainer = DigestGenerator(
        gemini_client=gemini_client,
        voice_generator=voice_gen,
        model_name=settings.gemini_model,
    )
    tutor = TutorEngine(
        gemini_client=gemini_client,
        model_name=settings.gemini_model,
    )

    # 5. Initialize security gate
    security_gate = SecurityGate(
        allowed_user_ids=settings.allowed_telegram_user_ids,
        rate_limit_per_minute=settings.rate_limit_requests_per_minute,
        max_message_chars=settings.max_user_message_chars,
    )

    # 6. Initialize bot handlers
    bot_handlers = BotHandlers(
        db=db,
        digest_generator=explainer,
        tutor_engine=tutor,
        content_extractor=content_extractor,
        gemini_client=gemini_client,
        news_fetcher=fetcher,
        classifier=classifier,
        gemini_model=settings.gemini_model,
        timezone=settings.timezone,
        digest_lookback_hours=settings.digest_lookback_hours,
    )

    # 7. Setup scheduler lifecycle hooks integrated into bot asyncio loop
    scheduler = None

    async def post_init_hook(app: Application) -> None:
        nonlocal scheduler
        logger.info("Setting up daily morning AsyncIOScheduler...")
        scheduler = configure_scheduler(
            application=app,
            db=db,
            fetcher=fetcher,
            classifier=classifier,
            explainer=explainer,
            allowed_user_ids=settings.allowed_telegram_user_ids,
            timezone_str=settings.timezone,
            schedule_time_str=settings.schedule_time,
            lookback_hours=settings.digest_lookback_hours,
        )
        if not scheduler.running:
            scheduler.start()
            logger.info(
                f"Scheduler started. Daily morning digest scheduled for "
                f"{settings.schedule_time} {settings.timezone}."
            )

    async def post_stop_hook(app: Application) -> None:
        nonlocal scheduler
        if scheduler and scheduler.running:
            logger.info("Shutting down AsyncIOScheduler...")
            scheduler.shutdown(wait=False)

    # 8. Build Telegram Application
    builder = Application.builder().token(settings.telegram_bot_token)
    builder.post_init(post_init_hook)
    builder.post_stop(post_stop_hook)
    app = builder.build()

    # 9. Register handlers and security middleware
    register_handlers(app, security_gate, bot_handlers)

    logger.info("AI Mitra Telegram bot starting polling loop...")
    try:
        app.run_polling(drop_pending_updates=True)
    except Exception as exc:
        logger.error(f"Telegram polling loop exited with error: {exc}", exc_info=True)


if __name__ == "__main__":
    main()

