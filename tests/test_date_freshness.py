"""Regression test suite for runtime date/time awareness and recent-news window selection."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo
import pytest

from bot.handlers import BotHandlers
from config import get_settings
from core.database import DatabaseManager, get_effective_article_time, parse_to_timezone
from core.models import NewsFactSheet, ReliabilityState, SourceFact
from core.news.fetcher import RawArticle
from core.synthesis.explainer import DigestGenerator, DigestResult, StructuredDigestItem


def create_fact_sheet(
    event_id: str,
    headline: str,
    published_at: str = None,
    updated_at: str = None,
    retrieved_at: str = None,
    category: str = "ai_around_the_world",
) -> NewsFactSheet:
    return NewsFactSheet(
        event_id=event_id,
        headline=headline,
        primary_source_name="Tech Source",
        primary_source_url=f"https://example.com/{event_id}",
        supporting_sources=[],
        reliability=ReliabilityState.REPORTED,
        source_reported_facts=[
            SourceFact(
                claim=f"Factual claim for {headline}",
                source_name="Tech Source",
                source_url=f"https://example.com/{event_id}",
            )
        ],
        ai_summary=f"Summary for {headline}",
        category=category,
        published_at=published_at,
        updated_at=updated_at,
        retrieved_at=retrieved_at,
    )


# ==============================================================================
# 1. test_runtime_date_uses_configured_timezone
# ==============================================================================

def test_runtime_date_uses_configured_timezone():
    """Mock a known UTC time and verify that AI Mitra converts it to the configured Asia/Kolkata time correctly."""
    utc_time = datetime(2026, 9, 20, 2, 30, 0, tzinfo=timezone.utc)
    kolkata_tz = ZoneInfo("Asia/Kolkata")
    kolkata_time = utc_time.astimezone(kolkata_tz)

    assert kolkata_time.year == 2026
    assert kolkata_time.month == 9
    assert kolkata_time.day == 20
    assert kolkata_time.hour == 8
    assert kolkata_time.minute == 0
    assert kolkata_time.tzinfo == kolkata_tz

    # Verify parser utility correctly converts UTC strings to configured timezone
    parsed = parse_to_timezone("2026-09-20T02:30:00Z", kolkata_tz)
    assert parsed == kolkata_time


# ==============================================================================
# 2. test_today_does_not_require_publication_date_equal_current_date
# ==============================================================================

def test_today_does_not_require_publication_date_equal_current_date(tmp_path):
    """Provide important articles published on the previous day and verify that they can appear in the September 20 digest."""
    db = DatabaseManager(str(tmp_path / "test_prev_day.db"))
    kolkata_tz = ZoneInfo("Asia/Kolkata")
    cutoff_time = datetime(2026, 9, 20, 8, 0, 0, tzinfo=kolkata_tz)

    # Article published on September 19 at 18:00 IST (14 hours before cutoff)
    prev_day_item = create_fact_sheet(
        event_id="evt_prev_day_1",
        headline="Major Model Architecture Published Yesterday Evening",
        published_at="2026-09-19T18:00:00+05:30",
    )
    db.save_news_item(prev_day_item)

    selected = db.get_news_for_digest(cutoff_time=cutoff_time, lookback_hours=24, limit=6)

    assert len(selected) == 1
    assert selected[0].event_id == "evt_prev_day_1"
    assert selected[0].headline == "Major Model Architecture Published Yesterday Evening"


# ==============================================================================
# 3. test_future_articles_are_excluded
# ==============================================================================

def test_future_articles_are_excluded(tmp_path):
    """Provide an article timestamp later than the digest cutoff and verify it is excluded."""
    db = DatabaseManager(str(tmp_path / "test_future.db"))
    kolkata_tz = ZoneInfo("Asia/Kolkata")
    cutoff_time = datetime(2026, 9, 20, 8, 0, 0, tzinfo=kolkata_tz)

    # Valid article published 2 hours before cutoff
    valid_item = create_fact_sheet(
        event_id="evt_valid_morning",
        headline="Morning Announcement",
        published_at="2026-09-20T06:00:00+05:30",
    )
    # Future article published 4 hours AFTER cutoff
    future_item = create_fact_sheet(
        event_id="evt_future_afternoon",
        headline="Future Afternoon Release",
        published_at="2026-09-20T12:00:00+05:30",
    )

    db.save_news_item(valid_item)
    db.save_news_item(future_item)

    selected = db.get_news_for_digest(cutoff_time=cutoff_time, lookback_hours=24, limit=6)

    selected_ids = [item.event_id for item in selected]
    assert "evt_valid_morning" in selected_ids
    assert "evt_future_afternoon" not in selected_ids


# ==============================================================================
# 4. test_recent_window_selection
# ==============================================================================

def test_recent_window_selection(tmp_path):
    """Provide articles across several dates and verify that the recent 24-hour window is preferred."""
    db = DatabaseManager(str(tmp_path / "test_window.db"))
    kolkata_tz = ZoneInfo("Asia/Kolkata")
    cutoff_time = datetime(2026, 9, 20, 8, 0, 0, tzinfo=kolkata_tz)

    # 1. Inside 24h window (10h ago)
    art_24h_1 = create_fact_sheet(
        event_id="evt_24h_1",
        headline="Fresh Breakthrough 10h Ago",
        published_at="2026-09-19T22:00:00+05:30",
    )
    # 2. Inside 24h window (20h ago)
    art_24h_2 = create_fact_sheet(
        event_id="evt_24h_2",
        headline="Fresh Breakthrough 20h Ago",
        published_at="2026-09-19T12:00:00+05:30",
    )
    # 3. 36 hours ago (in 48h window)
    art_48h = create_fact_sheet(
        event_id="evt_48h",
        headline="Older Update 36h Ago",
        published_at="2026-09-18T20:00:00+05:30",
    )
    # 4. 5 days ago
    art_old = create_fact_sheet(
        event_id="evt_old",
        headline="Very Old Article 5 Days Ago",
        published_at="2026-09-15T08:00:00+05:30",
    )

    db.save_news_item(art_24h_1)
    db.save_news_item(art_24h_2)
    db.save_news_item(art_48h)
    db.save_news_item(art_old)

    # Request limit=2: only the two 24h window items should be selected
    selected_top2 = db.get_news_for_digest(cutoff_time=cutoff_time, lookback_hours=24, limit=2)
    assert len(selected_top2) == 2
    assert selected_top2[0].event_id == "evt_24h_1"
    assert selected_top2[1].event_id == "evt_24h_2"

    # Request limit=3: should expand window to include the 48h item
    selected_top3 = db.get_news_for_digest(cutoff_time=cutoff_time, lookback_hours=24, limit=3)
    assert len(selected_top3) == 3
    assert selected_top3[2].event_id == "evt_48h"


# ==============================================================================
# 5. test_old_launch_not_presented_as_new_launch
# ==============================================================================

def test_old_launch_not_presented_as_new_launch():
    """Provide an old product/project with current repository activity and verify that
    synthesis does not describe it as a new launch unless supporting evidence exists.
    """
    trending_item = create_fact_sheet(
        event_id="evt_trending_vllm",
        headline="Trending Project: vllm-project/vllm",
        published_at=None,
        updated_at=None,
        category="what_people_are_doing",
    )
    trending_item.ai_summary = "A high-throughput LLM serving engine showing high developer activity on GitHub today."

    # Test Gemini prompt contains strict anti-false-launch instructions
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
    [
        {
            "event_id": "evt_trending_vllm",
            "headline": "Trending Project: vLLM High-Throughput Serving",
            "concise_summary": "vLLM is currently seeing strong developer activity on GitHub for fast inference.",
            "key_takeaway": "Active project trending in the open-source AI community."
        }
    ]
    """
    mock_gemini.models.generate_content.return_value = mock_response

    generator = DigestGenerator(gemini_client=mock_gemini)
    digest_result = generator.generate_digest([trending_item], include_audio=False)

    # 1. Verify prompt contains anti-false-launch directives
    call_args = mock_gemini.models.generate_content.call_args[1]
    prompt_sent = call_args["contents"]
    assert "Do NOT describe something as 'launched today'" in prompt_sent
    assert "NEVER claim that an existing product or trending repository launched today" in prompt_sent

    # 2. Verify synthesized result does not say "launched today" or "new launch"
    item = digest_result.items[0]
    assert "launched today" not in item.headline.lower()
    assert "launched today" not in item.concise_summary.lower()
    assert "new launch" not in item.concise_summary.lower()

    # 3. Verify fallback behavior also avoids false launch claims
    fallback_items = DigestGenerator._fallback_grounded_items([trending_item])
    assert "launched today" not in fallback_items[0].headline.lower()
    assert "launched today" not in fallback_items[0].key_takeaway.lower()


# ==============================================================================
# 6. test_missing_publication_date_not_assumed_current
# ==============================================================================

def test_missing_publication_date_not_assumed_current(tmp_path):
    """Provide an article with no trustworthy publication timestamp and verify that
    the system does not assume it was published today.
    """
    db = DatabaseManager(str(tmp_path / "test_missing_date.db"))
    kolkata_tz = ZoneInfo("Asia/Kolkata")
    cutoff_time = datetime(2026, 9, 20, 8, 0, 0, tzinfo=kolkata_tz)

    undated_item = create_fact_sheet(
        event_id="evt_undated_1",
        headline="Repository Without Publication Timestamp",
        published_at=None,
        updated_at=None,
        retrieved_at="2026-09-20T08:00:00+05:30",  # fetched today
    )

    dated_item = create_fact_sheet(
        event_id="evt_dated_1",
        headline="Dated Report Within Lookback Window",
        published_at="2026-09-19T23:00:00+05:30",
    )

    # Check precedence: retrieved_at must NOT make the item appear dated/current
    assert get_effective_article_time(undated_item, kolkata_tz) is None
    assert get_effective_article_time(dated_item, kolkata_tz) is not None

    db.save_news_item(undated_item)
    db.save_news_item(dated_item)

    # When limit=1, the dated item in the 24h window MUST be selected, NOT the undated item
    selected = db.get_news_for_digest(cutoff_time=cutoff_time, lookback_hours=24, limit=1)
    assert len(selected) == 1
    assert selected[0].event_id == "evt_dated_1"


# ==============================================================================
# 7. test_today_refreshes_news_before_selection
# ==============================================================================

def test_today_refreshes_news_before_selection(tmp_path):
    """Verify that /today invokes the existing news refresh/fetch pipeline before querying the date-aware digest selection."""
    async def run():
        db = DatabaseManager(str(tmp_path / "test_refresh_today.db"))
        kolkata_tz = ZoneInfo("Asia/Kolkata")
        cutoff_time = datetime(2026, 9, 20, 10, 30, 0, tzinfo=kolkata_tz)

        # Initially, database only has an older 4-day-old article
        old_item = create_fact_sheet(
            event_id="evt_old_stored",
            headline="Four Day Old AI Report",
            published_at="2026-09-16T10:00:00+05:30",
        )
        db.save_news_item(old_item)

        # Mock news fetcher returning a brand-new article available right now (e.g. 1 hour ago)
        fresh_raw_article = RawArticle(
            title="Brand New Breakthrough Just Published",
            url="https://example.com/fresh-breakthrough",
            source_name="Live Feed",
            summary="A brand-new breakthrough was just published 1 hour ago.",
            category="ai_around_the_world",
            published_at="2026-09-20T09:30:00+05:30",
            updated_at=None,
            retrieved_at="2026-09-20T10:30:00+05:30",
        )

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = [fresh_raw_article]

        mock_classifier = MagicMock()
        mock_classifier.process_cluster.return_value = create_fact_sheet(
            event_id="evt_fresh_live",
            headline="Brand New Breakthrough Just Published",
            published_at="2026-09-20T09:30:00+05:30",
        )

        mock_explainer = MagicMock()
        mock_explainer.generate_digest.return_value = DigestResult(
            items=[
                StructuredDigestItem(
                    event_id="evt_fresh_live",
                    headline="Brand New Breakthrough Just Published",
                    concise_summary="A fresh breakthrough occurred.",
                    key_takeaway="Key advancement.",
                )
            ],
            formatted_text="Morning Digest Text",
            audio_path=None,
        )

        handlers = BotHandlers(
            db=db,
            digest_generator=mock_explainer,
            tutor_engine=MagicMock(),
            news_fetcher=mock_fetcher,
            classifier=mock_classifier,
            timezone="Asia/Kolkata",
            digest_lookback_hours=24,
        )

        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update = MagicMock()
        mock_update.effective_message = mock_msg

        # Invoke handle_today
        with patch("bot.handlers.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_time
            await handlers.handle_today(mock_update, MagicMock())

        # Asserts:
        # 1. news_fetcher.fetch_all() MUST have been called to refresh news before selection
        assert mock_fetcher.fetch_all.call_count == 1

        # 2. classifier.process_cluster() MUST have processed the newly fetched article
        assert mock_classifier.process_cluster.call_count == 1

        # 3. explainer.generate_digest() MUST have received the freshly ingested item (inside 24h window)
        assert mock_explainer.generate_digest.call_count == 1
        fact_sheets_passed = mock_explainer.generate_digest.call_args[0][0]
        passed_ids = [fs.event_id for fs in fact_sheets_passed]
        assert "evt_fresh_live" in passed_ids

        # 4. Message was delivered
        assert mock_msg.reply_text.call_count >= 1

    asyncio.run(run())

