"""Tests for measured image analysis and the optional real-model fallback."""

import os
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from app import _prepare_uploaded_image
from src.describer import (
    _describe_with_blip,
    _finish_description,
    analyze_image_features,
    describe_image,
)
from src.preprocessing import extract_image_properties
from src.speech import get_web_speech_js

SAMPLE_DIR = Path(__file__).parent / "sample_images"
MODEL_TESTS_ENABLED = os.getenv("VISIONPULSE_RUN_MODEL_TESTS") == "1"


class ImageFeatureTests(unittest.TestCase):
    def read_image(self, filename):
        with Image.open(SAMPLE_DIR / filename) as image:
            return image.convert("RGB")

    def test_clear_photo_has_measured_quality_without_color_predictions(self):
        image = self.read_image("clear_objects.jpg")
        features = analyze_image_features(image)
        self.assertGreater(features["brightness"], 55)
        self.assertGreaterEqual(features["contrast"], 0)
        self.assertNotIn("colors", features)
        self.assertNotIn("color_desc", extract_image_properties(image))

    def test_image_with_visible_text_is_a_valid_image(self):
        image = self.read_image("image_with_text.png")
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)

    def test_dark_image_is_measured_and_flagged(self):
        features = analyze_image_features(self.read_image("dark_image.jpg"))
        self.assertTrue(features["is_dark"])
        self.assertIn("Very dark", features["lighting"])
        self.assertLess(features["brightness"], 55)

    def test_blurry_image_is_measured_and_flagged(self):
        features = analyze_image_features(self.read_image("blurry_image.jpg"))
        self.assertTrue(features["is_blurry"])
        self.assertIn("blurry", features["lighting"])
        self.assertLess(features["blur_variance"], 35)

    def test_corrupt_upload_gets_a_friendly_validation_error(self):
        raw_bytes = (SAMPLE_DIR / "corrupt_image.bin").read_bytes()
        with self.assertRaisesRegex(ValueError, "not a readable image"):
            _prepare_uploaded_image(raw_bytes)

    def test_empty_upload_gets_a_friendly_validation_error(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            _prepare_uploaded_image(b"")


class SpeechOutputTests(unittest.TestCase):
    def test_mp3_speech_is_always_english(self):
        with patch("src.speech.gTTS") as tts:
            tts.return_value.write_to_fp.side_effect = lambda stream: stream.write(b"mp3")
            from src.speech import text_to_speech_gtts

            self.assertEqual(text_to_speech_gtts("A red flower."), b"mp3")
        tts.assert_called_once_with(
            text="A red flower.", lang="en", slow=False, timeout=(3, 8)
        )

    def test_browser_speech_escapes_quotes_and_newlines(self):
        text = 'Read "this" line\nthen this line.'
        component = get_web_speech_js(text, rate=0.8)
        self.assertIn(json.dumps(text, ensure_ascii=False), component)
        self.assertIn('utterance.lang = "en-US";', component)
        self.assertIn("const feminineVoice = englishVoices.find", component)
        self.assertIn("/female|samantha|zira|jenny|aria|ava|susan|hazel|siri/i", component)
        self.assertIn("utterance.voice = feminineVoice || englishDefault;", component)
        self.assertNotIn("hi-IN", component)
        self.assertNotIn("te-IN", component)
        self.assertIn('aria-label="Read the full image description aloud"', component)
        self.assertIn('aria-label="Stop reading the image description"', component)


class DescriptionHonestyTests(unittest.TestCase):
    def test_blip_requests_a_longer_caption_for_more_visible_detail(self):
        with Image.open(SAMPLE_DIR / "clear_objects.jpg") as source:
            image = source.convert("RGB")
        captioner = Mock(
            return_value=[{"generated_text": "A flower beside several leaves."}]
        )
        result = _describe_with_blip(image, lambda: captioner)
        self.assertEqual(result, "A flower beside several leaves.")
        captioner.assert_called_once_with(
            image,
            prompt="Describe the visible objects and scene without mentioning colors: ",
            generate_kwargs={"max_new_tokens": 64, "num_beams": 4},
        )

    def test_high_confidence_description_is_not_claimed_to_be_perfect(self):
        with Image.open(SAMPLE_DIR / "clear_objects.jpg") as source:
            image = source.convert("RGB")
        result = _finish_description(
            {
                "summary": "A flower with layered petals stands in front of several leaves.",
                "objects": ["flower", "leaves"],
                "text_in_image": "Not clear",
                "lighting": "",
                "safety_notes": "Not assessed",
                "people_count_estimate": "Unclear",
                "confidence": 0.99,
                "is_unsure": False,
            },
            analyze_image_features(image),
            "Gemini",
        )
        self.assertIn("flower", result["summary"])
        self.assertNotIn("Main colors:", result["full_description"])
        self.assertNotIn("colors", result)
        self.assertNotIn("100%", result["full_description"])
        self.assertNotIn("I'm not sure", result["full_description"])

    def test_offline_caption_does_not_repeat_backend_disclaimer(self):
        with Image.open(SAMPLE_DIR / "clear_objects.jpg") as source:
            image = source.convert("RGB")
        result = _finish_description(
            {"summary": "A flower with layered petals beside several leaves."},
            analyze_image_features(image),
            "BLIP",
        )
        self.assertIn("flower", result["full_description"])
        self.assertNotIn("Main colors:", result["full_description"])
        self.assertNotIn("colors", result)
        self.assertNotIn("Offline caption", result["full_description"])

    def test_low_confidence_suppresses_unverified_scene_details(self):
        with Image.open(SAMPLE_DIR / "clear_objects.jpg") as source:
            image = source.convert("RGB")
        features = analyze_image_features(image)
        result = _finish_description(
            {
                "summary": "",
                "objects": [],
                "text_in_image": "",
                "lighting": "",
                "safety_notes": "",
                "people_count_estimate": "",
                "confidence": 0.1,
                "is_unsure": True,
            },
            features,
            "Gemini",
        )
        self.assertIn("Vision model confidence is low", result["summary"])
        self.assertNotIn("I'm not sure", result["full_description"])
        self.assertEqual(result["objects"], [])
        self.assertNotIn("Main colors:", result["full_description"])
        self.assertNotIn("colors", result)
        self.assertNotIn("Safety notes:", result["full_description"])
        self.assertNotIn("People estimate:", result["full_description"])
        self.assertIn("model is unsure", result["text_in_image"])
        self.assertIn("not possible", result["safety_notes"])


@unittest.skipUnless(
    MODEL_TESTS_ENABLED,
    "Set VISIONPULSE_RUN_MODEL_TESTS=1 to run real BLIP inference.",
)
class RealBlipIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.describer import load_blip_pipeline

        cls.captioner = load_blip_pipeline()

    def describe_fixture(self, filename):
        with Image.open(SAMPLE_DIR / filename) as image:
            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                return describe_image(
                    image.convert("RGB"),
                    fallback_loader=lambda: self.captioner,
                )

    def test_clear_photo_caption_matches_visible_flower(self):
        result = self.describe_fixture("clear_objects.jpg")
        self.assertIn("flower", result["summary"].lower())

    def test_text_image_does_not_claim_ocr_in_offline_mode(self):
        result = self.describe_fixture("image_with_text.png")
        self.assertEqual(
            result["text_in_image"], "Not analyzed in offline captioning mode."
        )

    def test_dark_image_description_states_detail_limit_without_guessing(self):
        result = self.describe_fixture("dark_image.jpg")
        self.assertIn("very dark", result["summary"].lower())
        self.assertNotIn("I'm not sure", result["full_description"])
        self.assertNotIn("flower", result["summary"].lower())
        self.assertIn("very dark", result["full_description"].lower())

    def test_blurry_image_description_states_detail_limit_without_guessing(self):
        result = self.describe_fixture("blurry_image.jpg")
        self.assertIn("blurry", result["summary"].lower())
        self.assertNotIn("I'm not sure", result["full_description"])
        self.assertNotIn("painting", result["summary"].lower())
        self.assertNotIn("sun", result["summary"].lower())
        self.assertIn("blurry", result["full_description"].lower())


if __name__ == "__main__":
    unittest.main()
