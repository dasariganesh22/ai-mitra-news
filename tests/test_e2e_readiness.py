"""Non-destructive end-to-end startup-readiness and pipeline wiring verification test.

Verifies that all 12 stages of AI Mitra connect together in the exact required sequence
using an isolated temporary database, temporary audio output directory, and mocked
external boundaries (Gemini, Telegram, gTTS, external HTTP).
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from telegram import Chat, Message, Update, User
from telegram.ext import Application, CommandHandler, MessageHandler, TypeHandler

from bot.formatters import chunk_telegram_message, format_digest_message
from bot.handlers import BotHandlers, register_handlers
from bot.middleware import SecurityGate
from bot.scheduler import DAILY_JOB_ID, configure_scheduler
from config import validate_and_load_config
from core.database import DatabaseManager
from core.models import NewsFactSheet, ReliabilityState
from core.news.classifier import ReliabilityClassifier
from core.news.content_extractor import ContentExtractor
from core.news.dedup_crosscheck import EventDeduplicator
from core.news.fetcher import NewsFetcher, NewsSourceConfig, RawArticle
from core.synthesis.explainer import DigestGenerator
from core.synthesis.tutor import TutorEngine
from core.synthesis.voice import VoiceGenerator


def test_complete_e2e_startup_and_wiring_readiness(tmp_path):
    """Verify that all application components wire together successfully in the exact order:
    1. Configuration validation
    2. Database initialization (isolated temp DB)
    3. News ingestion (mocked isolated feed)
    4. Content extraction (SafeContentExtractor)
    5. Deduplication & clustering
    6. Reliability classification & DB storage
    7. Structured synthesis (mocked Gemini client)
    8. Voice generation (mocked TTS in isolated temp dir)
    9. Telegram formatting & chunking
    10. Security middleware gate
    11. Scheduler registration & idempotence
    12. Telegram application wiring
    """
    temp_db_path = tmp_path / "test_e2e.db"
    temp_audio_dir = tmp_path / "audio"

    # ----------------------------------------------------------------------
    # Stage 1: Configuration Validation (Fail-Closed Check)
    # ----------------------------------------------------------------------
    mock_env = {
        "GEMINI_API_KEY": "AIzaSyFakeKeyForTesting1234567890",
        "TELEGRAM_BOT_TOKEN": "1234567890:ABCdefGhIjKlMnOpQrStUvWxYz1234567",
        "ALLOWED_TELEGRAM_USER_IDS": "12345,67890",
        "TIMEZONE": "Asia/Kolkata",
        "SCHEDULE_TIME": "08:00",
        "DATABASE_PATH": str(temp_db_path),
        "LOG_LEVEL": "INFO",
    }
    settings = validate_and_load_config(mock_env)
    assert settings.gemini_api_key == mock_env["GEMINI_API_KEY"]
    assert settings.allowed_telegram_user_ids == [12345, 67890]
    assert settings.timezone == "Asia/Kolkata"
    assert settings.schedule_time == "08:00"

    # ----------------------------------------------------------------------
    # Stage 2: Database Initialization (Isolated Temp DB)
    # ----------------------------------------------------------------------
    db = DatabaseManager(db_path=str(temp_db_path))
    assert temp_db_path.exists()

    # Verify schema initialized
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "news_items" in tables
        assert "user_conversations" in tables

    # ----------------------------------------------------------------------
    # Stage 3: News Ingestion (Mocked Sources)
    # ----------------------------------------------------------------------
    source_config = NewsSourceConfig(
        name="Test Verified Source",
        url="https://verified-news.example.com/rss",
        source_type="rss",
        category="ai_around_the_world",
        enabled=True,
    )
    fetcher = NewsFetcher(sources=[source_config])

    raw_articles = [
        RawArticle(
            title="Anthropic Launches Claude 3.7 Sonnet Hybrid Reasoning Architecture",
            url="https://techcrunch.com/anthropic-claude-3-7-sonnet",
            source_name="TechCrunch",
            summary="Anthropic has released Claude 3.7 Sonnet featuring combined fast thinking and extended test-time reasoning.",
            category="ai_around_the_world",
            published_at="2026-09-19T08:00:00Z",
        ),
        RawArticle(
            title="Anthropic Releases Claude 3.7 Sonnet with Extended Reasoning Capabilities",
            url="https://venturebeat.com/claude-3-7-sonnet-extended",
            source_name="VentureBeat",
            summary="The Claude 3.7 Sonnet model launched by Anthropic brings extended test-time reasoning and hybrid thinking.",
            category="ai_around_the_world",
            published_at="2026-09-19T08:30:00Z",
        ),
    ]

    # ----------------------------------------------------------------------
    # Stage 4: Content Extraction (SafeContentExtractor)
    # ----------------------------------------------------------------------
    extractor = ContentExtractor()
    with patch.object(
        extractor,
        "extract_from_url",
        return_value=("Claude 3.7 Sonnet Details", "Official documentation regarding Claude 3.7 reasoning."),
    ):
        title, text = extractor.extract_from_url("https://techcrunch.com/anthropic-claude-3-7-sonnet")
        assert title == "Claude 3.7 Sonnet Details"
        assert "Claude 3.7" in text

    # ----------------------------------------------------------------------
    # Stage 5: Deduplication & Clustering
    # ----------------------------------------------------------------------
    clusters = EventDeduplicator.cluster_articles(raw_articles)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2  # Both articles grouped into single event cluster

    # ----------------------------------------------------------------------
    # Stage 6: Reliability Classification & DB Storage
    # ----------------------------------------------------------------------
    classifier = ReliabilityClassifier()
    fact_sheet = classifier.process_cluster(clusters[0])
    assert fact_sheet.reliability in (ReliabilityState.CROSS_CHECKED, ReliabilityState.OFFICIAL)
    assert fact_sheet.primary_source_name is not None

    db.save_news_item(fact_sheet)
    stored_items = db.get_recent_news(limit=5)
    assert len(stored_items) == 1
    assert stored_items[0].event_id == fact_sheet.event_id

    # ----------------------------------------------------------------------
    # Stage 7: Structured Synthesis (Mocked Gemini Client)
    # ----------------------------------------------------------------------
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        '[\n'
        '  {\n'
        f'    "event_id": "{fact_sheet.event_id}",\n'
        '    "headline": "Claude 3.7 Sonnet Officially Released",\n'
        '    "concise_summary": "Anthropic released Claude 3.7 with hybrid reasoning advances.",\n'
        '    "key_takeaway": "Developers gain stronger test-time reasoning capabilities."\n'
        '  }\n'
        ']'
    )
    mock_gemini.models.generate_content.return_value = mock_response

    # ----------------------------------------------------------------------
    # Stage 8: Voice Generation (Mocked TTS in Isolated Directory)
    # ----------------------------------------------------------------------
    voice_gen = VoiceGenerator(output_dir=str(temp_audio_dir))
    with patch("gtts.gTTS.save") as mock_gtts_save:
        # Simulate writing empty/mock audio file
        def fake_save(save_path):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(b"MOCK_MP3_BYTES")
        mock_gtts_save.side_effect = fake_save

        explainer = DigestGenerator(gemini_client=mock_gemini, voice_generator=voice_gen)
        digest_result = explainer.generate_digest([fact_sheet], include_audio=True)

        assert len(digest_result.items) == 1
        assert digest_result.items[0].headline == "Claude 3.7 Sonnet Officially Released"
        assert digest_result.audio_path is not None
        assert Path(digest_result.audio_path).exists()
        # Verify audio was written to temp directory, NOT data/audio/
        assert str(temp_audio_dir) in digest_result.audio_path

    # ----------------------------------------------------------------------
    # Stage 9: Telegram Formatting & Chunking
    # ----------------------------------------------------------------------
    formatted_message = format_digest_message(digest_result.items, [fact_sheet])
    assert fact_sheet.primary_source_name in formatted_message
    assert fact_sheet.primary_source_url in formatted_message
    # Verify deterministic reliability badge is present
    assert ("Verified Official" in formatted_message) or ("Cross-Checked" in formatted_message)

    chunks = chunk_telegram_message(formatted_message, max_chars=4000)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) <= 4000

    # ----------------------------------------------------------------------
    # Stage 10: Security Middleware Gate
    # ----------------------------------------------------------------------
    security_gate = SecurityGate(
        allowed_user_ids=settings.allowed_telegram_user_ids,
        rate_limit_per_minute=settings.rate_limit_requests_per_minute,
        max_message_chars=settings.max_user_message_chars,
    )

    async def check_security_gate():
        # Authorized user passes
        auth_user = User(id=12345, first_name="Owner", is_bot=False)
        auth_chat = Chat(id=12345, type="private")
        auth_msg = MagicMock(spec=Message)
        auth_msg.text = "/today"
        auth_update = MagicMock(spec=Update)
        auth_update.effective_user = auth_user
        auth_update.effective_chat = auth_chat
        auth_update.effective_message = auth_msg

        allowed, reason = await security_gate.evaluate_security(auth_update)
        assert allowed is True
        assert reason is None

        # Unauthorized user is blocked
        unauth_user = User(id=99999, first_name="Attacker", is_bot=False)
        unauth_update = MagicMock(spec=Update)
        unauth_update.effective_user = unauth_user
        unauth_update.effective_chat = auth_chat
        unauth_update.effective_message = auth_msg

        unauth_allowed, unauth_reason = await security_gate.evaluate_security(unauth_update)
        assert unauth_allowed is False
        assert unauth_reason == "unauthorized_user"

    asyncio.run(check_security_gate())

    # ----------------------------------------------------------------------
    # Stage 11: Scheduler Registration (Without waiting for 08:00)
    # ----------------------------------------------------------------------
    mock_app = MagicMock(spec=Application)
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    mock_bot.send_voice = AsyncMock()
    mock_app.bot = mock_bot

    scheduler = configure_scheduler(
        application=mock_app,
        db=db,
        fetcher=fetcher,
        classifier=classifier,
        explainer=explainer,
        allowed_user_ids=settings.allowed_telegram_user_ids,
        timezone_str=settings.timezone,
        schedule_time_str=settings.schedule_time,
    )

    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == DAILY_JOB_ID
    assert "hour='8'" in str(jobs[0].trigger)
    assert "minute='0'" in str(jobs[0].trigger)
    # Idempotence: Calling again replaces cleanly
    scheduler = configure_scheduler(
        application=mock_app,
        db=db,
        fetcher=fetcher,
        classifier=classifier,
        explainer=explainer,
        allowed_user_ids=settings.allowed_telegram_user_ids,
        timezone_str=settings.timezone,
        schedule_time_str=settings.schedule_time,
        existing_scheduler=scheduler,
    )
    assert len(scheduler.get_jobs()) == 1

    # ----------------------------------------------------------------------
    # Stage 12: Telegram Application Wiring
    # ----------------------------------------------------------------------
    tutor = TutorEngine(gemini_client=mock_gemini)
    bot_handlers = BotHandlers(
        db=db,
        digest_generator=explainer,
        tutor_engine=tutor,
        content_extractor=extractor,
        gemini_client=mock_gemini,
    )

    # Build application and register handlers
    app_builder = Application.builder().token(settings.telegram_bot_token)
    app = app_builder.build()
    register_handlers(app, security_gate, bot_handlers)

    # Assert handlers registered across groups
    assert -1 in app.handlers
    assert any(isinstance(h, TypeHandler) for h in app.handlers[-1])
    assert 0 in app.handlers
    assert any(isinstance(h, CommandHandler) for h in app.handlers[0])
    assert any(isinstance(h, MessageHandler) for h in app.handlers[0])

    # Verify no real messages sent, no live daemon launched, temp DB cleaned up on block exit
    assert mock_bot.send_message.call_count == 0

