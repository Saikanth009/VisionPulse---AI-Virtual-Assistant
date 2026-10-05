"""
VisionPulse — AI Visual Assistant for People with Low Vision
Image Preprocessing Module (src/preprocessing.py)

Handles image loading, format validation, resizing, pixel normalization,
tensor preparation, and per-image visual quality extraction (dominant colors, contrast, brightness).
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


def analyze_dominant_colors(pil_img):
    """
    Extracts the dominant color palette and tone description of an image.
    """
    # Downsample image to speed up color analysis
    small_img = pil_img.resize((50, 50))
    np_img = np.array(small_img)

    r_mean = float(np.mean(np_img[:, :, 0]))
    g_mean = float(np.mean(np_img[:, :, 1]))
    b_mean = float(np.mean(np_img[:, :, 2]))

    # Calculate color dominance
    total = r_mean + g_mean + b_mean + 1e-5
    r_ratio = r_mean / total
    g_ratio = g_mean / total
    b_ratio = b_mean / total

    # Determine dominant tones
    if g_ratio > 0.40 and g_ratio > r_ratio and g_ratio > b_ratio:
        color_desc = "lush green and natural tones"
    elif b_ratio > 0.40 and b_ratio > r_ratio:
        color_desc = "cool blue and aquatic hues"
    elif r_ratio > 0.42 and g_ratio > 0.25:
        color_desc = "warm orange, yellow, or earthy tones"
    elif r_ratio > 0.40:
        color_desc = "rich red or warm crimson tones"
    elif r_mean > 200 and g_mean > 200 and b_mean > 200:
        color_desc = "bright white and light silver background"
    elif r_mean < 70 and g_mean < 70 and b_mean < 70:
        color_desc = "dark, black, or deep grey tones"
    elif abs(r_mean - g_mean) < 15 and abs(g_mean - b_mean) < 15:
        color_desc = "sleek neutral grey and metallic tones"
    else:
        color_desc = "balanced multi-colored tones"

    return color_desc


def extract_image_properties(image_source):
    """
    Extracts comprehensive visual properties (brightness, contrast, aspect ratio, color palette)
    to generate unique per-picture visual descriptions for low-vision accessibility.
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

    # Dominant Color Analysis
    color_desc = analyze_dominant_colors(img)

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "brightness_desc": brightness_desc,
        "contrast_desc": contrast_desc,
        "orientation": orientation,
        "color_desc": color_desc
    }
