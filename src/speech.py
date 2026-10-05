import io
import logging
import json

from gtts import gTTS

logger = logging.getLogger(__name__)


def text_to_speech_gtts(text, slow=False):
    if not text or not text.strip():
        raise ValueError("Text string is empty. Cannot synthesize speech.")
    try:
        tts = gTTS(text=text, lang="en", slow=slow, timeout=(3, 8))
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        return audio_fp.getvalue()
    except Exception as e:
        logger.exception("gTTS speech synthesis failed.")
        raise RuntimeError(f"Failed to generate speech audio: {e}") from e


def speak_description(text, slow=False):
    try:
        audio_bytes = text_to_speech_gtts(text, slow=slow)
        return {
            "success": True,
            "audio_bytes": audio_bytes,
            "format": "audio/mp3",
            "message": "Speech audio is ready.",
        }
    except Exception as e:
        return {
            "success": False,
            "audio_bytes": None,
            "error": str(e),
            "message": "MP3 speech generation failed. Please use browser speech output.",
        }


def get_web_speech_js(text, autoplay=False, rate=0.9):
    safe_text = json.dumps(text, ensure_ascii=False)
    safe_text = safe_text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    autoplay_script = "speakVisionPulse();" if autoplay else ""
    return f"""
    <div style="font-family:system-ui,sans-serif;display:flex;gap:12px;flex-wrap:wrap">
      <button aria-label="Read the full image description aloud"
        onclick="speakVisionPulse()"
        style="background:#00E5FF;color:#000;padding:12px 24px;font-size:20px;
        font-weight:800;min-height:60px;border:3px solid #FFD700;border-radius:8px">
        Listen
      </button>
      <button aria-label="Stop reading the image description"
        onclick="stopVisionPulse()"
        style="background:#FFD700;color:#000;padding:12px 24px;font-size:20px;
        font-weight:800;min-height:60px;border:3px solid #00E5FF;border-radius:8px">
        Stop
      </button>
      <script>
        const visionPulseText = {safe_text};
        function speakVisionPulse() {{
          if (!("speechSynthesis" in window)) {{
            alert("Speech synthesis is not supported in this browser.");
            return;
          }}
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(visionPulseText);
          utterance.rate = {float(rate)};
          utterance.lang = "en-US";
          const englishVoices = window.speechSynthesis.getVoices()
            .filter((voice) => voice.lang.toLowerCase().startsWith("en"));
          const feminineVoice = englishVoices.find((voice) =>
            /female|samantha|zira|jenny|aria|ava|susan|hazel|siri/i.test(voice.name)
          );
          const englishDefault = englishVoices.find((voice) => voice.default);
          if (feminineVoice || englishDefault) {{
            utterance.voice = feminineVoice || englishDefault;
          }}
          window.speechSynthesis.speak(utterance);
        }}
        function stopVisionPulse() {{
          if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        }}
        {autoplay_script}
      </script>
    </div>
    """
