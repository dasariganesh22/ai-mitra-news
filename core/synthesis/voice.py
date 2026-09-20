"""Non-blocking voice note generator for AI Mitra using gTTS."""

import os
import time
from pathlib import Path
from typing import Optional
from gtts import gTTS
from core.logger import get_logger

logger = get_logger("voice_generator")


class VoiceGenerator:
    """Generates audio voice notes for the morning digest with non-blocking error handling."""

    def __init__(self, output_dir: str = "data/audio", tts_cls=gTTS):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tts_cls = tts_cls

    def generate_voice_note(
        self,
        script_text: str,
        filename_prefix: str = "morning_digest",
        lang: str = "en",
    ) -> Optional[str]:
        """Generate a 60-90 second audio voice note (.mp3) from script text.

        Non-Blocking Guarantee:
        If TTS generation encounters any network error, timeout, or exception,
        the failure is safely logged, and None is returned. The caller can then
        deliver the text digest without interruption.
        """
        if not script_text or not script_text.strip():
            logger.warning("Empty script text provided for voice generation. Skipping.")
            return None

        timestamp = int(time.time())
        output_file = self.output_dir / f"{filename_prefix}_{timestamp}.mp3"

        try:
            logger.info(f"Synthesizing voice note (approx {len(script_text.split())} words)...")
            tts = self.tts_cls(text=script_text.strip(), lang=lang)
            tts.save(str(output_file))
            logger.info(f"Voice note successfully saved to '{output_file}'.")
            return str(output_file)
        except Exception as exc:
            logger.warning(f"Voice note generation failed ({type(exc).__name__}: {exc}). Proceeding without audio.")
            return None

