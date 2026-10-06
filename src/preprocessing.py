"""
VisionPulse — AI Visual Assistant for People with Low Vision
Image Preprocessing Module (src/preprocessing.py)

Handles image loading, format validation, resizing, pixel normalization,
tensor preparation, and per-image visual quality extraction (contrast and brightness).
"""

import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from src.config import IMAGE_SIZE, CHANNELS


def load_image(image_source):
    """
    Load an image from a file path, file-like object, bytes, or numpy array.
    Returns a PIL Image in RGB mode.
    """
    if image_source is None:
        raise ValueError("No image provided. Please upload a valid image file.")

    try:
        if isinstance(image_source, (str, bytes)):
            if isinstance(image_source, bytes):
                image = Image.open(io.BytesIO(image_source))
            else:
                image = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        elif isinstance(image_source, np.ndarray):
            # Convert OpenCV BGR array to RGB PIL Image if needed
            if len(image_source.shape) == 3 and image_source.shape[2] == 3:
                rgb_array = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_array)
            else:
                image = Image.fromarray(image_source)
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            image = Image.open(image_source)

        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Invalid or corrupted image format. Details: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


def validate_image(image_source, allowed_formats=("JPG", "JPEG", "PNG", "WEBP")):
    """
    Validates file extension/format and image integrity.
    """
    try:
        img = load_image(image_source)
        width, height = img.size

        if width <= 0 or height <= 0:
            return False, "Image has invalid zero or negative dimensions."
        if width * height > 50000000:  # 50 Megapixels safety cap
            return False, "Image resolution exceeds supported maximum dimensions."

        return True, "Image validation successful."
    except Exception as e:
        return False, f"Validation failed: {str(e)}"


def resize_image(image, target_size=IMAGE_SIZE):
    """
    Resizes image to target dimensions (width, height) using high-quality bilinear interpolation.
    """
    if isinstance(image, Image.Image):
        # PIL size order is (width, height)
        return image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    elif isinstance(image, np.ndarray):
        # cv2 resize order is (width, height)
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    else:
        raise TypeError("Unsupported image type for resize.")


def normalize_image(image_array):
    """
    Normalizes pixel values from uint8 range [0, 255] to float32 range [0.0, 1.0].
    """
    array_float = image_array.astype(np.float32)
    if array_float.max() > 1.0:
        array_float /= 255.0
    return array_float


def preprocess_image(image_source, target_size=IMAGE_SIZE):
    """
    Full preprocessing pipeline: Load -> Resize -> Convert to Array -> Normalize.
    Returns normalized NumPy array of shape (height, width, channels).
    """
    img = load_image(image_source)
    img_resized = resize_image(img, target_size)
    img_array = np.array(img_resized, dtype=np.float32)
    img_normalized = normalize_image(img_array)
    return img_normalized


def prepare_input(image_source, target_size=IMAGE_SIZE):
    """
    Prepares input tensor for CNN model inference.
    Returns 4D batch tensor of shape (1, height, width, channels).
    """
    processed = preprocess_image(image_source, target_size)
    input_tensor = np.expand_dims(processed, axis=0)
    return input_tensor


def extract_image_properties(image_source):
    """
    Extracts measurable visual properties (brightness, contrast, and orientation).
    """
    img = load_image(image_source)
    width, height = img.size
    aspect_ratio = width / float(height)

    # Convert to grayscale array for brightness & contrast metric
    gray_array = np.array(img.convert("L"), dtype=np.float32)
    brightness = float(np.mean(gray_array))
    contrast = float(np.std(gray_array))

    # Brightness description
    if brightness < 65:
        brightness_desc = "darkly illuminated with low lighting"
    elif brightness > 185:
        brightness_desc = "brightly lit with high illumination"
    else:
        brightness_desc = "well-lit with balanced lighting"

    # Contrast description
    if contrast < 35:
        contrast_desc = "soft visual contrast"
    else:
        contrast_desc = "distinct high contrast against the background"

    # Orientation
    if aspect_ratio > 1.25:
        orientation = "landscape orientation"
    elif aspect_ratio < 0.8:
        orientation = "portrait orientation"
    else:
        orientation = "square framing"

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "brightness_desc": brightness_desc,
        "contrast_desc": contrast_desc,
        "orientation": orientation,
    }
