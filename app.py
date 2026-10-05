"""VisionPulse Streamlit accessibility application."""

import hashlib
import io

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.describer import describe_image, load_blip_pipeline
from src.speech import get_web_speech_js, speak_description

BUILD_VERSION = "VisionPulse 2.0.0 — vision-language"
MAX_IMAGE_SIDE = 1024
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}

st.set_page_config(
    page_title="VisionPulse — Visual Assistant",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        background: #121212 !important;
        color: #ffffff !important;
        font-family: "Segoe UI", Roboto, Arial, sans-serif !important;
        font-size: 20px !important;
    }
    h1, h2, h3, h4 { color: #00E5FF !important; font-weight: 800 !important; }
    p, li, label, [data-testid="stMarkdownContainer"] { font-size: 20px !important; }
    div.stButton > button, [data-testid="stFileUploader"] button {
        background: #00E5FF !important; color: #000000 !important;
        font-size: 20px !important; font-weight: 800 !important;
        min-height: 60px !important; border: 3px solid #FFD700 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section { border: 2px solid #00E5FF; }
    [data-testid="stAudio"] { min-height: 60px; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading offline caption model…")
def get_offline_captioner():
    """Load the BLIP model once; image descriptions are never cached."""
    return load_blip_pipeline()


def _prepare_uploaded_image(raw_bytes: bytes) -> Image.Image:
    if not raw_bytes:
        raise ValueError("The selected file is empty.")
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image_format = (image.format or "").upper()
        if image_format not in SUPPORTED_FORMATS:
            raise ValueError("Unsupported image type. Please upload a JPG, PNG, or WEBP image.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This file is not a readable image or it is corrupted.") from exc


def _analysis_controls():
    st.sidebar.markdown(f"### {BUILD_VERSION}")
    st.sidebar.markdown("#### English speech")
    speech_rate = st.sidebar.slider(
        "Speech rate", min_value=0.5, max_value=1.5, value=0.9, step=0.1,
        help="Adjust the browser voice speed.",
    )
    st.sidebar.caption(
        "Gemini mode sends the uploaded image to Google's Gemini API. "
        "Without an API key, VisionPulse uses the offline BLIP captioner."
    )
    return speech_rate


def main():
    speech_rate = _analysis_controls()
    st.markdown(
        "<h1 style='text-align:center;font-size:42px'>👁️ VISIONPULSE</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align:center;color:#FFD700 !important'>"
        "AI Visual Assistant for People with Low Vision</h2>",
        unsafe_allow_html=True,
    )
    st.write(
        "Upload an image for a brief description of the visible subject and its colors."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported image formats: JPG, PNG, and WEBP.",
    )
    if uploaded_file is None:
        st.session_state.pop("analysis_hash", None)
        st.session_state.pop("description_data", None)
        st.session_state.pop("audio_data", None)
        st.session_state.pop("audio_key", None)
        st.info("Choose an image to begin.")
        return

    raw_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        image = _prepare_uploaded_image(raw_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.image(image, width="stretch", caption="Uploaded image")

    if st.session_state.get("analysis_hash") != image_hash:
        st.session_state.pop("description_data", None)
        st.session_state.pop("audio_data", None)
        st.session_state.pop("audio_key", None)
        with st.spinner("Analyzing this image…"):
            try:
                result = describe_image(image, fallback_loader=get_offline_captioner)
            except Exception as exc:
                st.error(
                    "VisionPulse could not analyze this image. Check your internet "
                    "connection for the first BLIP download, or verify the Gemini API key. "
                    f"Details: {exc}"
                )
                return
            st.session_state["description_data"] = result
            st.session_state["analysis_hash"] = image_hash

    result = st.session_state.get("description_data")
    if not result:
        return

    st.markdown("### Description")
    st.markdown(
        f"<div style='background:#1E1E1E;border:3px solid #FFD700;"
        f"border-radius:12px;padding:20px;color:white;font-size:22px;line-height:1.6'>"
        f"{_escape_html(result['full_description'])}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Main colors")
    st.write(", ".join(result["colors"]) if result["colors"] else "Not measured")

    audio_key = image_hash
    if st.session_state.get("audio_key") != audio_key:
        with st.spinner("Preparing spoken audio…"):
            audio_result = speak_description(
                result["full_description"]
            )
        st.session_state["audio_key"] = audio_key
        st.session_state["audio_data"] = (
            audio_result["audio_bytes"] if audio_result["success"] else None
        )
        st.session_state["audio_error"] = (
            audio_result["message"] if not audio_result["success"] else None
        )

    st.markdown("### Listen")
    st.components.v1.html(
        get_web_speech_js(
            result["full_description"],
            rate=speech_rate,
        ),
        height=90,
    )
    audio_bytes = st.session_state.get("audio_data")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning(
            "The MP3 fallback is unavailable right now. Use the browser Listen button. "
            + (st.session_state.get("audio_error") or "")
        )


def _escape_html(text: str) -> str:
    import html

    return html.escape(text, quote=True).replace("\n", "<br>")


if __name__ == "__main__":
    main()
