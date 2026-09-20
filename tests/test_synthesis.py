"""Automated test suite verifying the synthesis, tutor, and voice generation engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.models import NewsFactSheet, ReliabilityState, SourceFact, ToolMetadata
from core.synthesis.explainer import DigestGenerator, StructuredDigestItem
from core.synthesis.tutor import HowToGuide, TutorEngine, VillageExplanation
from core.synthesis.voice import VoiceGenerator


@pytest.fixture
def sample_fact_sheets():
    """Fixture providing sample verified fact sheets with deterministic metadata."""
    return [
        NewsFactSheet(
            event_id="evt_gemini25",
            headline="Google Launches Gemini 2.5 Architecture",
            primary_source_name="Google DeepMind Blog",
            primary_source_url="https://deepmind.google/technologies/gemini/",
            supporting_sources=["https://blog.google/technology/ai/gemini-2-5/"],
            reliability=ReliabilityState.OFFICIAL,
            is_official_source=True,
            source_reported_facts=[
                SourceFact(
                    claim="Google released Gemini 2.5 with 2M token window.",
                    source_name="Google DeepMind",
                    source_url="https://deepmind.google/technologies/gemini/",
                )
            ],
            ai_summary="Google announced Gemini 2.5 with advanced reasoning and long context.",
            category="ai_around_the_world",
        ),
        NewsFactSheet(
            event_id="evt_claude_hybrid",
            headline="Anthropic Unveils Claude 3.7 Sonnet",
            primary_source_name="TechCrunch",
            primary_source_url="https://techcrunch.com/anthropic-claude-3-7",
            supporting_sources=["https://venturebeat.com/claude-3-7"],
            reliability=ReliabilityState.CROSS_CHECKED,
            is_official_source=False,
            source_reported_facts=[
                SourceFact(
                    claim="Claude 3.7 combines fast inference with hybrid reasoning.",
                    source_name="TechCrunch",
                    source_url="https://techcrunch.com/anthropic-claude-3-7",
                )
            ],
            ai_summary="Anthropic released Claude 3.7 Sonnet with hybrid reasoning capabilities.",
            category="ai_around_the_world",
        ),
    ]


# ==============================================================================
# 1. Structured Gemini Digest Output & Fallback Tests
# ==============================================================================

def test_structured_gemini_digest_output(sample_fact_sheets):
    """Verify that Gemini digest output is parsed and validated by Pydantic."""
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
    [
        {
            "event_id": "evt_gemini25",
            "headline": "Google Releases Gemini 2.5",
            "concise_summary": "Google introduced a new Gemini model with deep thinking and huge context memory.",
            "key_takeaway": "Handles much longer documents with improved reasoning speed."
        },
        {
            "event_id": "evt_claude_hybrid",
            "headline": "Anthropic Debuts Claude 3.7",
            "concise_summary": "Claude 3.7 combines instant replies with optional extended reasoning.",
            "key_takeaway": "Gives developers control over thinking speed versus depth."
        }
    ]
    """
    mock_gemini.models.generate_content.return_value = mock_response

    generator = DigestGenerator(gemini_client=mock_gemini)
    digest_result = generator.generate_digest(sample_fact_sheets, include_audio=False)

    assert len(digest_result.items) == 2
    assert isinstance(digest_result.items[0], StructuredDigestItem)
    assert digest_result.items[0].event_id == "evt_gemini25"
    assert digest_result.items[0].headline == "Google Releases Gemini 2.5"
    assert "improved reasoning speed" in digest_result.items[0].key_takeaway


def test_digest_fallback_when_gemini_fails(sample_fact_sheets):
    """Verify that digest generation falls back to grounded fact sheet data when Gemini fails."""
    mock_gemini = MagicMock()
    # Simulate an API failure or timeout
    mock_gemini.models.generate_content.side_effect = RuntimeError("API service temporarily unavailable")

    with patch("time.sleep"):
        generator = DigestGenerator(gemini_client=mock_gemini)
        digest_result = generator.generate_digest(sample_fact_sheets, include_audio=False)

    # Must complete without crashing
    assert len(digest_result.items) == 2
    # Fallback content must match the verified source summaries in input fact sheets
    assert digest_result.items[0].event_id == "evt_gemini25"
    assert digest_result.items[0].headline == sample_fact_sheets[0].headline
    assert digest_result.items[0].concise_summary == sample_fact_sheets[0].ai_summary
    assert "AI Mitra Morning Digest" in digest_result.formatted_text


def test_gemini_transient_503_uses_bounded_retry_then_fallback(sample_fact_sheets):
    """Verify that transient 503 errors trigger bounded retries with backoff,
    terminate after max retries, fall back to grounded data, and format cleanly.
    """
    mock_gemini = MagicMock()
    # Simulate repeated 503 UNAVAILABLE responses
    mock_gemini.models.generate_content.side_effect = RuntimeError(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}"
    )

    with patch("time.sleep") as mock_sleep:
        generator = DigestGenerator(gemini_client=mock_gemini)
        digest_result = generator.generate_digest(sample_fact_sheets, include_audio=False)

        # 1. Verify exact bounded retries: 1 initial attempt + 2 retries = 3 total attempts
        assert mock_gemini.models.generate_content.call_count == 3

        # 2. Verify retry backoff sleep called twice (1.0s and 2.0s)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0

        # 3. Verify loop terminated cleanly and grounded fallback is used
        assert len(digest_result.items) == 2
        assert digest_result.items[0].event_id == sample_fact_sheets[0].event_id
        assert digest_result.items[0].headline == sample_fact_sheets[0].headline

        # 4. Verify no placeholder/demo items are fabricated
        event_ids = {item.event_id for item in digest_result.items}
        assert "init_event" not in event_ids
        assert "sample_item" not in event_ids

        # 5. Verify Telegram-safe formatting
        from bot.formatters import format_digest_message
        msg = format_digest_message(digest_result.items, sample_fact_sheets)
        assert "AI Mitra Morning Intelligence Digest" in msg
        assert "[Google DeepMind Blog](https://deepmind.google/technologies/gemini/)" in msg


def test_gemini_permanent_error_fails_immediately_without_retry(sample_fact_sheets):
    """Verify that permanent errors (e.g. 401, 404, invalid key) do not waste retries."""
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = RuntimeError("404 NOT_FOUND: Model does not exist")

    with patch("time.sleep") as mock_sleep:
        generator = DigestGenerator(gemini_client=mock_gemini)
        digest_result = generator.generate_digest(sample_fact_sheets, include_audio=False)

        # Permanent error: call_count must be 1 (no retries)
        assert mock_gemini.models.generate_content.call_count == 1
        assert mock_sleep.call_count == 0
        assert len(digest_result.items) == 2


# ==============================================================================
# 2. Deterministic Metadata Binding Tests
# ==============================================================================

def test_deterministic_metadata_binding(sample_fact_sheets):
    """Verify that source URLs, names, and reliability badges are strictly bound in Python."""
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    # Notice: the model does NOT output URLs or badges
    mock_response.text = """
    [
        {
            "event_id": "evt_gemini25",
            "headline": "Model Gemini 2.5 Out",
            "concise_summary": "Summary text here.",
            "key_takeaway": "Takeaway text here."
        }
    ]
    """
    mock_gemini.models.generate_content.return_value = mock_response

    generator = DigestGenerator(gemini_client=mock_gemini)
    digest_result = generator.generate_digest(sample_fact_sheets, include_audio=False)
    text = digest_result.formatted_text

    # 1. Official badge must be present deterministically
    assert "🛡️ Verified Official" in text
    # 2. Cross-checked badge must be present deterministically
    assert "🔍 Cross-Checked" in text
    # 3. Authoritative source links must be bound from fact sheets, NOT invented
    assert "[Source: Google DeepMind Blog](https://deepmind.google/technologies/gemini/)" in text
    assert "[Source: TechCrunch](https://techcrunch.com/anthropic-claude-3-7)" in text


# ==============================================================================
# 3. Evidence-Based /howto Guidance Tests
# ==============================================================================

def test_evidence_based_howto_guidance():
    """Verify that tool guides are strictly grounded in documentation without hallucinating commands."""
    tool_meta = ToolMetadata(
        name="AutoAgent",
        official_url="https://github.com/example/autoagent",
        repo_url="https://github.com/example/autoagent",
        is_free=True,
    )

    # Case A: Documentation containing verified install command and pricing
    docs_with_commands = """
    # AutoAgent
    AutoAgent is an open-source task runner. It is completely free under MIT license.
    ## Installation
    Run: pip install autoagent-cli
    """

    mock_gemini = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = """
    {
        "verified_prerequisites": ["Python 3.10+"],
        "verified_install_commands": ["pip install autoagent-cli"],
        "step_by_step_usage": ["1. Run pip install", "2. Execute autoagent-cli start"]
    }
    """
    mock_gemini.models.generate_content.return_value = mock_resp

    tutor = TutorEngine(gemini_client=mock_gemini)
    guide = tutor.generate_tool_howto(tool_meta, docs_with_commands)

    assert "pip install autoagent-cli" in guide.verified_install_commands
    assert guide.is_free is True
    assert guide.tool_name == "AutoAgent"

    # Case B: Documentation with NO commands specified (must not fabricate commands)
    docs_empty = "This is a theoretical paper about agents with no software package yet."
    guide_empty = tutor.generate_tool_howto(tool_meta, docs_empty)
    # Must record caveat that commands were not found
    assert len(guide_empty.verified_install_commands) == 0
    assert any("not explicitly stated" in c for c in guide_empty.caveats_or_unverified)


# ==============================================================================
# 4. Non-Blocking Voice Generation Tests
# ==============================================================================

def test_voice_failure_does_not_block_digest(sample_fact_sheets):
    """Verify that if voice synthesis raises an exception, the text digest succeeds."""
    mock_voice = MagicMock()
    # Simulate a voice generation failure (e.g. network failure / timeout)
    mock_voice.generate_voice_note.side_effect = ConnectionError("TTS service unreachable")

    generator = DigestGenerator(gemini_client=None, voice_generator=mock_voice)
    digest_result = generator.generate_digest(sample_fact_sheets, include_audio=True)

    # Text digest must still succeed completely!
    assert len(digest_result.items) == 2
    assert "AI Mitra Morning Digest" in digest_result.formatted_text
    assert digest_result.audio_path is None


def test_voice_generator_creates_audio_file(tmp_path):
    """Verify that VoiceGenerator writes audio bytes to the configured directory without requiring network."""
    mock_tts_cls = MagicMock()
    mock_instance = MagicMock()

    # When save() is called on mock gTTS, write dummy audio bytes to the destination file
    def mock_save(filepath):
        Path(filepath).write_bytes(b"\xFF\xFB\x90\x44" + b"\x00" * 500)  # Dummy MP3 header + payload

    mock_instance.save.side_effect = mock_save
    mock_tts_cls.return_value = mock_instance

    voice_gen = VoiceGenerator(output_dir=str(tmp_path / "audio"), tts_cls=mock_tts_cls)
    output_path = voice_gen.generate_voice_note(
        script_text="Good morning! Today we have two major AI announcements from Google and Anthropic.",
        filename_prefix="test_digest",
    )

    assert output_path is not None
    assert Path(output_path).exists()
    assert output_path.endswith(".mp3")
    assert Path(output_path).stat().st_size > 500

