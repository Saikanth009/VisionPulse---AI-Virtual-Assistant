"""Vision-language image description with measurable image-quality analysis."""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-2.5-flash"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_IMAGE_SIDE = 1024
logger = logging.getLogger(__name__)

VISION_PROMPT = """
In English, give a clear, detailed description of the visible scene in three to
five sentences. Start with the overall scene and main subject, then describe
observable details such as shape, position, foreground and background, nearby
objects, and visible activity. Do not name or guess colors. Mention readable
text only when it is clearly legible. Do not infer a purpose, identity,
relationship, or hidden detail, and do not add details just to make the
description longer. If the image does not support a reliable description, say
what cannot be determined. Never identify a person by name.
Return only a JSON object with these fields:
summary (three to five concise English sentences), objects (list of clearly visible
main object names only), text_in_image (readable text if clearly visible,
otherwise "Not clear"),
lighting (a short quality note only if very dark or blurry),
safety_notes (short note only if an obvious hazard is visible, otherwise "Not assessed"),
people_count_estimate ("Unclear" unless clearly countable),
confidence (number from 0 to 1), and is_unsure (boolean).
""".strip()

VISION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "objects": {"type": "ARRAY", "items": {"type": "STRING"}},
        "text_in_image": {"type": "STRING"},
        "lighting": {"type": "STRING"},
        "safety_notes": {"type": "STRING"},
        "people_count_estimate": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "is_unsure": {"type": "BOOLEAN"},
    },
    "required": [
        "summary",
        "objects",
        "text_in_image",
        "lighting",
        "safety_notes",
        "people_count_estimate",
        "confidence",
        "is_unsure",
    ],
}


def prepare_image(image: Image.Image) -> Image.Image:
    """Correct EXIF orientation, convert to RGB, and bound the longest side."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image


def analyze_image_features(image: Image.Image) -> dict:
    """Return pixel-derived brightness, contrast, and blur measurements."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 55:
        lighting = f"Very dark (measured mean brightness {brightness:.0f}/255)."
    elif brightness < 85:
        lighting = f"Dim (measured mean brightness {brightness:.0f}/255)."
    elif brightness > 205:
        lighting = f"Bright (measured mean brightness {brightness:.0f}/255)."
    else:
        lighting = f"Measured mean brightness {brightness:.0f}/255."
    lighting += f" Measured contrast is {contrast:.0f}/255."
    if blur_variance < 35:
        lighting += f" The image appears blurry (Laplacian variance {blur_variance:.1f})."

    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur_variance": blur_variance,
        "lighting": lighting,
        "is_dark": brightness < 55,
        "is_blurry": blur_variance < 35,
    }


def _clean_strings(value, limit=8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value[:limit] if isinstance(item, str) and item.strip()]


def _parse_model_json(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The vision model returned an empty response.")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("The vision model response was not valid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError("The vision model response must be a JSON object.")
    return result


def _describe_with_gemini(image: Image.Image, api_key: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[VISION_PROMPT, image],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=VISION_RESPONSE_SCHEMA,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    return _parse_model_json(getattr(response, "text", ""))


def load_blip_pipeline():
    """Create the offline image-captioning pipeline (cache it in the Streamlit app)."""
    os.environ["USE_TF"] = "0"
    from transformers import pipeline

    return pipeline("image-to-text", model=BLIP_MODEL_NAME)


def _describe_with_blip(image: Image.Image, fallback_loader: Callable | None) -> str:
    if fallback_loader is None:
        captioner = load_blip_pipeline()
    else:
        captioner = fallback_loader()
    results = captioner(
        image,
        prompt="Describe the visible objects and scene without mentioning colors: ",
        generate_kwargs={"max_new_tokens": 64, "num_beams": 4},
    )
    if not results or not isinstance(results[0], dict):
        raise ValueError("The offline caption model returned no caption.")
    caption = results[0].get("generated_text")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("The offline caption model returned an empty caption.")
    return caption.strip()


def _finish_description(
    model_result: dict,
    features: dict,
    backend: str,
) -> dict:
    quality_limited = features["is_dark"] or features["is_blurry"]
    summary = model_result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "A reliable description is unavailable for this image."
        model_result["is_unsure"] = True
    summary = summary.strip()

    try:
        confidence = float(model_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    uncertain = backend == "Gemini" and (
        model_result.get("is_unsure") is True
        or not math.isfinite(confidence)
        or confidence < 0.45
    )
    details_limited = quality_limited or uncertain

    if quality_limited:
        conditions = []
        if features["is_dark"]:
            conditions.append("very dark")
        if features["is_blurry"]:
            conditions.append("blurry")
        condition_text = " and ".join(conditions)
        summary = f"The image is {condition_text}, so subject details are limited."
    elif uncertain:
        if summary == "A reliable description is unavailable for this image.":
            summary = "Vision model confidence is low; a detailed description is unavailable."
        else:
            summary = f"Possible description: {summary}"

    caution = []
    if uncertain:
        caution.append("Vision model confidence is low; details may be inaccurate.")
    if features["is_dark"]:
        caution.append(
            "The image is very dark; scene details may be difficult to determine."
        )
    if features["is_blurry"]:
        caution.append(
            "The image appears blurry; fine details may be difficult to determine."
        )
    if backend == "BLIP":
        objects = []
        text_in_image = "Not analyzed in offline captioning mode."
        people_count = "Not estimated."
        safety_notes = "Not assessed in offline captioning mode."
    elif details_limited:
        objects = []
        reason = "the image quality is poor" if quality_limited else "the vision model is unsure"
        text_in_image = f"Text could not be read reliably because {reason}."
        people_count = f"Unclear because {reason}."
        safety_notes = f"A reliable hazard assessment is not possible because {reason}."
    else:
        objects = _clean_strings(model_result.get("objects"))
        text_in_image = model_result.get("text_in_image")
        if not isinstance(text_in_image, str) or not text_in_image.strip():
            text_in_image = "Text content is unclear or was not returned by the vision model."
        people_count = model_result.get("people_count_estimate")
        if not isinstance(people_count, str) or not people_count.strip():
            people_count = "Unclear."
        safety_notes = model_result.get("safety_notes")
        if not isinstance(safety_notes, str) or not safety_notes.strip():
            safety_notes = (
                "Hazard assessment was not returned by the vision model; "
                "do not treat this as a safety check."
            )

    lighting = features["lighting"]
    model_lighting = model_result.get("lighting")
    if (
        backend == "Gemini"
        and not details_limited
        and isinstance(model_lighting, str)
        and model_lighting.strip()
    ):
        lighting = f"{model_lighting.strip()} {lighting}"

    full_description = summary
    if caution:
        full_description += " " + " ".join(caution)

    return {
        "summary": summary,
        "objects": objects,
        "text_in_image": text_in_image,
        "lighting": lighting,
        "people_count_estimate": people_count,
        "safety_notes": safety_notes,
        "full_description": full_description,
    }


def describe_image(
    pil_image: Image.Image,
    *,
    fallback_loader: Callable | None = None,
) -> dict:
    """Describe an image using Gemini or BLIP and return accessible structured fields."""
    if not isinstance(pil_image, Image.Image):
        raise TypeError("describe_image expects a PIL image.")
    image = prepare_image(pil_image)
    if image.width < 1 or image.height < 1:
        raise ValueError("The image has invalid dimensions.")
    features = analyze_image_features(image)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            model_result = _describe_with_gemini(image, api_key)
            return _finish_description(model_result, features, "Gemini")
        except Exception as exc:
            logger.warning(
                "Gemini description failed (%s); attempting offline BLIP fallback.",
                type(exc).__name__,
            )

    caption = _describe_with_blip(image, fallback_loader)
    return _finish_description(
        {
            "summary": caption,
            "objects": [],
            "text_in_image": "Not analyzed in offline captioning mode.",
            "people_count_estimate": "",
            "safety_notes": "",
            "confidence": 0.0,
            "is_unsure": True,
        },
        features,
        "BLIP",
    )
