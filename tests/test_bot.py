"""Automated test suite verifying the Telegram Bot interface, middleware security, formatting, and scheduler."""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from telegram import Chat, Message, Update, User
from telegram.ext import Application, ContextTypes

from bot.formatters import (
    chunk_telegram_message,
    format_digest_message,
    format_howto_guide,
    format_village_explanation,
)
from bot.handlers import BotHandlers
from bot.middleware import SecurityGate, TokenBucketRateLimiter
from bot.scheduler import (
    DAILY_JOB_ID,
    configure_scheduler,
    scheduled_morning_dispatch,
)
from core.models import NewsFactSheet, ReliabilityState, SourceFact, ToolMetadata
from core.synthesis.explainer import DigestGenerator, DigestResult, StructuredDigestItem
from core.synthesis.tutor import HowToGuide, TutorEngine, VillageExplanation


# ==============================================================================
# Helper Fixtures & Mock Builders
# ==============================================================================

def create_mock_update(
    user_id: int = 12345,
    chat_id: int = 12345,
    text: str = "/today",
    chat_type: str = "private",
):
    """Create mock telegram Update and Message objects with async reply methods."""
    user = User(id=user_id, first_name="TestUser", is_bot=False)
    chat = Chat(id=chat_id, type=chat_type)
    msg = MagicMock(spec=Message)
    msg.message_id = 1
    msg.text = text
    msg.chat = chat
    msg.from_user = user
    msg.reply_text = AsyncMock()
    msg.reply_voice = AsyncMock()

    update = MagicMock(spec=Update)
    update.effective_user = user
    update.effective_chat = chat
    update.effective_message = msg
    return update, msg


def create_sample_fact_sheet(event_id: str = "event_1", headline: str = "Model Released") -> NewsFactSheet:
    return NewsFactSheet(
        event_id=event_id,
        headline=headline,
        primary_source_name="Official AI Blog",
        primary_source_url="https://example.com/official-announcement",
        supporting_sources=["https://example.com/official-announcement"],
        reliability=ReliabilityState.OFFICIAL,
        is_official_source=True,
        source_reported_facts=[
            SourceFact(
                claim="A new high-performance AI model was released.",
                source_name="Official AI Blog",
                source_url="https://example.com/official-announcement",
            )
        ],
        ai_summary="The model achieves state-of-the-art results on standard benchmarks.",
        tool_metadata=ToolMetadata(
            name="OpenModel",
            official_url="https://example.com/docs",
            repo_url="https://github.com/example/openmodel",
        ),
        category="ai_around_the_world",
    )


# ==============================================================================
# 1. Pre-Business-Logic Authorization & Security Gate Tests
# ==============================================================================

def test_unauthorized_user_never_reaches_handler_logic():
    """Verify an unauthorized user is rejected before handler business logic, Gemini, or DB access."""
    async def run():
        authorized_id = 12345
        unauthorized_id = 99999
        gate = SecurityGate(allowed_user_ids=[authorized_id])

        mock_db = MagicMock()
        mock_explainer = MagicMock()
        mock_tutor = MagicMock()
        mock_gemini = MagicMock()

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=mock_explainer,
            tutor_engine=mock_tutor,
            gemini_client=mock_gemini,
        )

        update, msg = create_mock_update(user_id=unauthorized_id, text="/today")
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # 1. Evaluate security check directly
        is_allowed, reason = await gate.evaluate_security(update)
        assert is_allowed is False
        assert reason == "unauthorized_user"

        # 2. Invoke handler wrapped with gate guard
        guarded_today = gate.guard(handlers.handle_today)
        await guarded_today(update, context)

        # Asserts: Business logic, Gemini, and Database were NEVER invoked
        assert mock_db.get_recent_news.call_count == 0
        assert mock_db.save_chat_message.call_count == 0
        assert mock_explainer.generate_digest.call_count == 0
        assert mock_tutor.explain_in_village_terms.call_count == 0
        assert mock_gemini.models.generate_content.call_count == 0
        # Must send unauthorized rejection
        msg.reply_text.assert_awaited_with("⛔ Unauthorized access.")

    asyncio.run(run())


def test_group_chat_rejected():
    """Verify non-private chat interactions (group, channel) are rejected at the gate."""
    async def run():
        gate = SecurityGate(allowed_user_ids=[12345])
        update, msg = create_mock_update(user_id=12345, chat_type="group", text="/today")

        is_allowed, reason = await gate.evaluate_security(update)
        assert is_allowed is False
        assert reason == "non_private_chat"

    asyncio.run(run())


def test_rate_limiter_blocks_burst_flooding():
    """Verify rapid bursts exceeding configured rate limits are blocked."""
    limiter = TokenBucketRateLimiter(requests_per_minute=3)
    user_id = 12345

    # First 3 requests should pass
    assert limiter.allow_request(user_id) is True
    assert limiter.allow_request(user_id) is True
    assert limiter.allow_request(user_id) is True

    # 4th request must be blocked
    assert limiter.allow_request(user_id) is False


def test_input_length_limit_enforced_in_middleware():
    """Verify messages exceeding maximum length are rejected by the security gate."""
    async def run():
        gate = SecurityGate(allowed_user_ids=[12345], max_message_chars=4000)
        oversized_text = "A" * 4001
        update, msg = create_mock_update(user_id=12345, text=oversized_text)

        is_allowed, reason = await gate.evaluate_security(update)
        assert is_allowed is False
        assert reason == "input_too_long"

    asyncio.run(run())


# ==============================================================================
# 2. Malformed & Unsupported Command Safety
# ==============================================================================

def test_invalid_command_is_rejected_safely():
    """Verify that malformed or unsupported commands are handled safely without crashing,
    touching DB, or calling Gemini.
    """
    async def run():
        gate = SecurityGate(allowed_user_ids=[12345])
        mock_db = MagicMock()
        mock_gemini = MagicMock()
        mock_explainer = MagicMock()
        mock_tutor = MagicMock()

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=mock_explainer,
            tutor_engine=mock_tutor,
            gemini_client=mock_gemini,
        )

        # Case A: /unknown command
        update, msg = create_mock_update(user_id=12345, text="/unknown")
        context = MagicMock()
        await gate.guard(handlers.handle_unknown_command)(update, context)
        assert "Unsupported Command" in msg.reply_text.call_args[0][0]

        # Case B: /explain with missing argument
        update, msg = create_mock_update(user_id=12345, text="/explain")
        context.args = []
        await gate.guard(handlers.handle_explain)(update, context)
        assert "/explain <number>" in msg.reply_text.call_args[0][0]
        assert "Usage" in msg.reply_text.call_args[0][0]

        # Case C: /explain with non-integer argument
        update, msg = create_mock_update(user_id=12345, text="/explain abc")
        context.args = ["abc"]
        await gate.guard(handlers.handle_explain)(update, context)
        assert "Invalid Story Number" in msg.reply_text.call_args[0][0]

        # Case D: /howto with missing tool name
        update, msg = create_mock_update(user_id=12345, text="/howto")
        context.args = []
        await gate.guard(handlers.handle_howto)(update, context)
        assert "/howto <tool_name>" in msg.reply_text.call_args[0][0]
        assert "Usage" in msg.reply_text.call_args[0][0]

        # In all cases, database and Gemini must NOT have been called
        assert mock_gemini.models.generate_content.call_count == 0
        assert mock_db.get_recent_news.call_count == 0

    asyncio.run(run())


# ==============================================================================
# 3. User-Submitted URL Safe Content Pipeline
# ==============================================================================

def test_pasted_url_uses_safe_content_pipeline():
    """Verify that a user-submitted news link is checked via validate_safe_url and
    SafeContentExtractor before synthesis, and blocked URLs are safely rejected.
    """
    async def run():
        gate = SecurityGate(allowed_user_ids=[12345])
        mock_db = MagicMock()
        mock_db.get_user_conversation_history.return_value = []
        mock_extractor = MagicMock()
        mock_tutor = MagicMock()
        mock_tutor.explain_in_village_terms.return_value = VillageExplanation(
            event_id="user_link",
            concept_or_topic="Article Analysis",
            source_facts=["Fact 1"],
            village_analogy="Like a town bulletin board.",
            what_is_it="An informative post.",
            why_it_matters="Relevant to learning.",
            how_it_works="Summarized safely.",
        )

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=MagicMock(),
            tutor_engine=mock_tutor,
            content_extractor=mock_extractor,
        )

        # Sub-case A: SSRF / Private IP URL is rejected immediately
        update, msg = create_mock_update(user_id=12345, text="Look at http://127.0.0.1:8000/secret")
        await gate.guard(handlers.handle_message)(update, MagicMock())
        assert "Blocked Link" in msg.reply_text.call_args[0][0]
        assert mock_extractor.extract_from_url.call_count == 0

        # Sub-case B: Public safe URL is extracted and wrapped
        mock_extractor.extract_from_url.return_value = ("Tech Article", "Readable article text about AI.")
        update, msg = create_mock_update(user_id=12345, text="Can you explain https://example.com/news/article")
        await gate.guard(handlers.handle_message)(update, MagicMock())

        assert mock_extractor.extract_from_url.call_count == 1
        assert mock_tutor.explain_in_village_terms.call_count == 1
        # Verify conversation turn was persisted to DB
        assert mock_db.save_chat_message.call_count == 2

    asyncio.run(run())


# ==============================================================================
# 4. General Conversational AI Doubts
# ==============================================================================

def test_general_conversational_question_supported():
    """Verify that general AI technical questions are answered without requiring digest context."""
    async def run():
        gate = SecurityGate(allowed_user_ids=[12345])
        mock_db = MagicMock()
        mock_db.get_user_conversation_history.return_value = []
        mock_db.get_recent_news.return_value = []  # No news in DB

        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Neural networks are computational models inspired by the human brain."
        mock_gemini.models.generate_content.return_value = mock_response

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=MagicMock(),
            tutor_engine=MagicMock(),
            gemini_client=mock_gemini,
        )

        update, msg = create_mock_update(user_id=12345, text="What is a neural network in simple words?")
        await gate.guard(handlers.handle_message)(update, MagicMock())

        # Asserts
        assert mock_gemini.models.generate_content.call_count == 1
        assert mock_db.save_chat_message.call_count == 2
        # First save: user query
        assert mock_db.save_chat_message.call_args_list[0][1]["role"] == "user"
        # Second save: assistant reply
        assert mock_db.save_chat_message.call_args_list[1][1]["role"] == "assistant"
        assert "Neural networks" in msg.reply_text.call_args[0][0]

    asyncio.run(run())


# ==============================================================================
# 5. Telegram Message Chunking Tests
# ==============================================================================

def test_telegram_message_chunking():
    """Verify that oversized text is split into valid Telegram-sized chunks without
    splitting inside markdown links or code blocks.
    """
    # Create long text with multiple paragraphs and a markdown link near the 4000-char boundary
    para1 = "Paragraph one content. " * 150  # ~3450 chars
    link = "Here is an important reference: [Official AI Portal](https://deepmind.google/technologies/gemini/)."
    para2 = "Paragraph two content with more details. " * 100  # ~4100 chars
    long_text = f"{para1}\n\n{link}\n\n{para2}"

    chunks = chunk_telegram_message(long_text, max_chars=4000)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 4000

    # Verify link was not broken across chunks
    combined = "".join(chunks)
    assert "[Official AI Portal](https://deepmind.google/technologies/gemini/)" in combined
    # Link must exist intact in exactly one chunk
    link_found = any("[Official AI Portal](https://deepmind.google/technologies/gemini/)" in c for c in chunks)
    assert link_found is True


# ==============================================================================
# 6. Formatting & Deterministic Badges Tests
# ==============================================================================

def test_formatting_renders_clickable_source_links_and_badges():
    """Verify deterministic binding of clickable source links and reliability badges."""
    item = StructuredDigestItem(
        event_id="ev_101",
        headline="Breakthrough in Quantum Computing AI",
        concise_summary="Researchers announce error correction milestones.",
        key_takeaway="Accelerates quantum machine learning research.",
    )
    fact_sheet = create_sample_fact_sheet(event_id="ev_101", headline="Breakthrough in Quantum Computing AI")

    formatted = format_digest_message([item], [fact_sheet])

    # Must contain deterministic reliability badge and markdown link
    assert "🛡️ [Verified Official]" in formatted
    assert "[Official AI Blog](https://example.com/official-announcement)" in formatted
    assert "Breakthrough in Quantum Computing AI" in formatted


def test_grounded_fallback_is_telegram_safe():
    """Verify that grounded fallback content containing realistic special characters,
    feed HTML attributes, and complex URLs formats cleanly for Telegram without malformed entities.
    """
    # Construct realistic messy fallback content with characters: _ * [ ] ( ) ~ ` > # + - = | { } . !
    # and raw Google News RSS HTML with target="_blank" and font tags
    messy_headline = "Release: agent_skills_v2.0 *Preview* [x86_64] (Beta) ~ `latest` > #ai + - = | {core} . !"
    messy_html_summary = (
        '<a href="https://news.google.com/rss/articles/CBMi..." target="_blank">'
        'Exclusive: US military had close call after using AI_model for false_intelligence report, sources say'
        '</a>&nbsp;&nbsp;<font color="#6f6f6f">CNN_News & Tech</font>'
    )
    messy_url = (
        "https://news.example.com/rss/articles/article_123?id=99&cat=ai_tech&ref=homepage_(edition)&utm_source=rss_feed#top"
    )

    fact_sheet = NewsFactSheet(
        event_id="evt_messy_fallback",
        headline=messy_headline,
        primary_source_name="CNN_News & Wire [Official]",
        primary_source_url=messy_url,
        supporting_sources=[messy_url],
        reliability=ReliabilityState.CROSS_CHECKED,
        is_official_source=False,
        source_reported_facts=[
            SourceFact(
                claim="Claim with _underscores_ and *stars* and `code` and [brackets]",
                source_name="CNN_News",
                source_url=messy_url,
            )
        ],
        ai_summary=messy_html_summary,
        category="ai_around_the_world",
    )

    fallback_items = DigestGenerator._fallback_grounded_items([fact_sheet])
    assert len(fallback_items) == 1
    fallback_item = fallback_items[0]

    # 1. Verify HTML tags were stripped from the fallback item
    assert "<a href" not in fallback_item.concise_summary
    assert 'target="_blank"' not in fallback_item.concise_summary
    assert "<font" not in fallback_item.concise_summary
    assert "&nbsp;" not in fallback_item.concise_summary

    # 2. Format the message for Telegram
    formatted = format_digest_message([fallback_item], [fact_sheet])

    # 3. Verify that raw unescaped HTML tags are completely removed
    assert "<a href" not in formatted
    assert "<font" not in formatted

    # 4. Verify that the URL is properly sanitized so parentheses inside the URL don't break Markdown link syntax
    assert "ref=homepage_%28edition%29" in formatted
    assert "[CNN\\_News & Wire \\[Official\\]](" in formatted
    assert "🔍 [Cross-Checked]" in formatted

    # 5. Verify entity balance: check that all underscores in text (outside of link URLs) are either escaped (\_) or part of the valid _Options:_ footer
    non_url_text = re.sub(r'\[.*?\]\(.*?\)', '', formatted)
    unescaped_underscores = [
        m.start() for m in re.finditer(r'(?<!\\)_', non_url_text)
    ]
    assert len(unescaped_underscores) == 2
    assert non_url_text[unescaped_underscores[0]:unescaped_underscores[1] + 1] == "_Options:_"

    # 6. Verify delivery via BotHandlers._safe_reply_chunks with entity fallback
    async def verify_delivery():
        msg = MagicMock(spec=Message)
        msg.reply_text = AsyncMock()
        await BotHandlers._safe_reply_chunks(msg, formatted, parse_mode="Markdown")
        assert msg.reply_text.call_count >= 1
        call_kwargs = msg.reply_text.call_args[1]
        assert call_kwargs.get("parse_mode") == "Markdown"

        # Verify fallback on BadRequest entity parse error
        from telegram.error import BadRequest
        failing_msg = MagicMock(spec=Message)
        failing_msg.reply_text = AsyncMock(
            side_effect=[BadRequest("Can't parse entities: can't find end of the entity"), None]
        )
        await BotHandlers._safe_reply_chunks(failing_msg, formatted, parse_mode="Markdown")
        assert failing_msg.reply_text.call_count == 2
        second_call_kwargs = failing_msg.reply_text.call_args[1]
        assert second_call_kwargs.get("parse_mode") is None

    asyncio.run(verify_delivery())


# ==============================================================================
# 7. Command Routing Handlers
# ==============================================================================

def test_today_command_delivers_digest():
    """Verify /today command generates digest and delivers text and audio voice note."""
    async def run():
        mock_db = MagicMock()
        sample_sheet = create_sample_fact_sheet()
        mock_db.get_recent_news.return_value = [sample_sheet]

        mock_explainer = MagicMock()
        mock_explainer.generate_digest.return_value = DigestResult(
            items=[
                StructuredDigestItem(
                    event_id="event_1",
                    headline="Model Released",
                    concise_summary="A new model was released.",
                    key_takeaway="Major milestone.",
                )
            ],
            formatted_text="Digest text",
            audio_path=None,
        )

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=mock_explainer,
            tutor_engine=MagicMock(),
        )

        update, msg = create_mock_update(user_id=12345, text="/today")
        await handlers.handle_today(update, MagicMock())

        assert mock_explainer.generate_digest.call_count == 1
        assert msg.reply_text.call_count >= 1

    asyncio.run(run())


def test_today_command_handles_empty_news_without_fabrication():
    """Verify /today command genuinely reports zero articles when none exist without fabricating demo items."""
    async def run():
        mock_db = MagicMock()
        mock_db.get_recent_news.return_value = []

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = []

        mock_classifier = MagicMock()
        mock_explainer = MagicMock()

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=mock_explainer,
            tutor_engine=MagicMock(),
            news_fetcher=mock_fetcher,
            classifier=mock_classifier,
        )

        update, msg = create_mock_update(user_id=12345, text="/today")
        await handlers.handle_today(update, MagicMock())

        # Explainer must NOT be called with fake items
        assert mock_explainer.generate_digest.call_count == 0
        # User is informed honestly
        assert msg.reply_text.call_count == 1
        reply_call_text = msg.reply_text.call_args[0][0]
        assert "No news items were found across verified sources" in reply_call_text

    asyncio.run(run())


def test_explain_command_invokes_village_analogy():
    """Verify /explain <number> retrieves the corresponding news event and invokes tutor."""
    async def run():
        mock_db = MagicMock()
        sample_sheet = create_sample_fact_sheet()
        mock_db.get_recent_news.return_value = [sample_sheet]

        mock_tutor = MagicMock()
        mock_tutor.explain_in_village_terms.return_value = VillageExplanation(
            event_id="event_1",
            concept_or_topic="Model Released",
            source_facts=["Fact A"],
            village_analogy="Think of this like a village water pump.",
            what_is_it="A strong model.",
            why_it_matters="Saves time.",
            how_it_works="Coordinates actions.",
        )

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=MagicMock(),
            tutor_engine=mock_tutor,
        )

        update, msg = create_mock_update(user_id=12345, text="/explain 1")
        context = MagicMock()
        context.args = ["1"]
        await handlers.handle_explain(update, context)

        assert mock_tutor.explain_in_village_terms.call_count == 1
        assert "village water pump" in msg.reply_text.call_args[0][0]

    asyncio.run(run())


def test_howto_command_invokes_evidence_tutor():
    """Verify /howto <tool> invokes tutor with official documentation grounding."""
    async def run():
        mock_db = MagicMock()
        sample_sheet = create_sample_fact_sheet()
        mock_db.get_recent_news.return_value = [sample_sheet]

        mock_tutor = MagicMock()
        mock_tutor.generate_howto.return_value = HowToGuide(
            tool_name="Ollama",
            official_documentation_url="https://ollama.com",
            is_free=True,
            pricing_details="Free and Open Source",
            verified_prerequisites=["macOS, Linux, or Windows"],
            verified_install_commands=["curl -fsSL https://ollama.com/install.sh | sh"],
            step_by_step_usage=["Run ollama run llama3"],
            caveats_or_unverified=[],
        )

        handlers = BotHandlers(
            db=mock_db,
            digest_generator=MagicMock(),
            tutor_engine=mock_tutor,
        )

        update, msg = create_mock_update(user_id=12345, text="/howto ollama")
        context = MagicMock()
        context.args = ["ollama"]
        await handlers.handle_howto(update, context)

        assert mock_tutor.generate_howto.call_count == 1
        assert "ollama.com" in msg.reply_text.call_args[0][0]

    asyncio.run(run())


# ==============================================================================
# 8. Scheduler Configuration, Idempotence & Pipeline Execution
# ==============================================================================

def test_scheduler_configuration_and_idempotence():
    """Verify scheduler registers 08:00 Asia/Kolkata trigger and idempotent registration."""
    mock_app = MagicMock(spec=Application)
    mock_db = MagicMock()
    mock_fetcher = MagicMock()
    mock_classifier = MagicMock()
    mock_explainer = MagicMock()

    scheduler = configure_scheduler(
        application=mock_app,
        db=mock_db,
        fetcher=mock_fetcher,
        classifier=mock_classifier,
        explainer=mock_explainer,
        allowed_user_ids=[12345, 67890],
        timezone_str="Asia/Kolkata",
        schedule_time_str="08:00",
    )

    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == DAILY_JOB_ID
    assert "hour='8'" in str(jobs[0].trigger)
    assert "minute='0'" in str(jobs[0].trigger)

    # Calling configure_scheduler again must NOT create duplicate jobs
    scheduler2 = configure_scheduler(
        application=mock_app,
        db=mock_db,
        fetcher=mock_fetcher,
        classifier=mock_classifier,
        explainer=mock_explainer,
        allowed_user_ids=[12345, 67890],
        timezone_str="Asia/Kolkata",
        schedule_time_str="08:00",
        existing_scheduler=scheduler,
    )

    jobs2 = scheduler2.get_jobs()
    assert len(jobs2) == 1
    assert jobs2[0].id == DAILY_JOB_ID


def test_scheduler_job_executes_digest_pipeline():
    """Verify that the scheduled job executes the complete news, verification,
    synthesis pipeline, and continues attempting delivery to remaining users even
    if delivery to one user fails.
    """
    async def run():
        mock_app = MagicMock(spec=Application)
        mock_bot = MagicMock()
        mock_app.bot = mock_bot

        user1_id = 11111
        user2_id = 22222

        # Simulate delivery to user1 raising an exception, but user2 succeeding
        async def mock_send_message(chat_id, text, parse_mode):
            if chat_id == user1_id:
                raise ConnectionError("Telegram network error for user 1")
            return MagicMock()

        mock_bot.send_message = AsyncMock(side_effect=mock_send_message)
        mock_bot.send_voice = AsyncMock()

        mock_db = MagicMock()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = []
        mock_classifier = MagicMock()
        mock_explainer = MagicMock()

        mock_explainer.generate_digest.return_value = DigestResult(
            items=[
                StructuredDigestItem(
                    event_id="sched_1",
                    headline="Morning Update",
                    concise_summary="All systems nominal.",
                    key_takeaway="Stay informed.",
                )
            ],
            formatted_text="Morning Digest Text",
            audio_path=None,
        )

        # Execute dispatch to both users
        result = await scheduled_morning_dispatch(
            application=mock_app,
            db=mock_db,
            fetcher=mock_fetcher,
            classifier=mock_classifier,
            explainer=mock_explainer,
            allowed_user_ids=[user1_id, user2_id],
        )

        assert result is not None
        # Verify send_message was called for BOTH user1 and user2
        delivered_chat_ids = [call.kwargs["chat_id"] for call in mock_bot.send_message.call_args_list]
        assert user1_id in delivered_chat_ids
        assert user2_id in delivered_chat_ids

    asyncio.run(run())
