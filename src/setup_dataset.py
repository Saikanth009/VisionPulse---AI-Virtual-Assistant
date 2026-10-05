"""
VisionPulse — AI Visual Assistant for People with Low Vision
Dataset Setup & Generator Utility (src/setup_dataset.py)

Creates structured sample image datasets inside data/train, data/validation, and data/test
across all 10 target classes to enable rapid model training and testing without slow network downloads.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    CLASS_NAMES, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    IMAGE_HEIGHT, IMAGE_WIDTH
)

# Visual color mapping for distinct visual patterns
CLASS_COLORS = {
    "airplane": (135, 206, 235),     # Sky Blue
    "automobile": (220, 20, 60),     # Crimson Red
    "bird": (34, 139, 34),           # Forest Green
    "cat": (255, 165, 0),            # Orange
    "deer": (139, 69, 19),           # Saddle Brown
    "dog": (210, 105, 30),           # Chocolate
    "frog": (0, 255, 127),           # Spring Green
    "horse": (160, 82, 45),          # Sienna
    "ship": (0, 105, 148),           # Deep Ocean Blue
    "truck": (112, 128, 144)         # Slate Gray
}


def generate_class_sample_image(class_name, idx):
    """
    Generates a 32x32 sample image with distinct visual patterns and noise for a given class.
    """
    base_color = CLASS_COLORS.get(class_name, (128, 128, 128))
    img_array = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    # Fill base color with slight per-sample variation
    r_var = int(np.random.randint(-15, 15))
    g_var = int(np.random.randint(-15, 15))
    b_var = int(np.random.randint(-15, 15))

    r = np.clip(base_color[0] + r_var, 0, 255)
    g = np.clip(base_color[1] + g_var, 0, 255)
    b = np.clip(base_color[2] + b_var, 0, 255)

    img_array[:, :, 0] = r
    img_array[:, :, 1] = g
    img_array[:, :, 2] = b

    # Add geometric shapes / pattern signals
    # Convert color elements to int
    circle_color = (int(255 - r), int(255 - g), int(255 - b))
    cv2.circle(img_array, (16 + (idx % 5), 16 + (idx % 3)), 8, circle_color, -1)
    cv2.rectangle(img_array, (4, 4), (28, 28), (255, 255, 255), 1)

    # Add subtle Gaussian noise
    noise = np.random.normal(0, 8, img_array.shape).astype(np.int16)
    noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img)


def setup_local_dataset(samples_per_train=50, samples_per_val=15, samples_per_test=15):
    """
    Populates data/train, data/validation, and data/test with sample images.
    """
    print("==================================================")
    print("VisionPulse Dataset Generator — Creating Local Splits")
    print("==================================================")

    for split_name, split_path, count in [
        ("Train", TRAIN_DATA_PATH, samples_per_train),
        ("Validation", VAL_DATA_PATH, samples_per_val),
        ("Test", TEST_DATA_PATH, samples_per_test)
    ]:
        print(f"Generating {split_name} split ({count} images per class)...")
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(split_path, cls)
            os.makedirs(cls_dir, exist_ok=True)

            for i in range(count):
                img = generate_class_sample_image(cls, i)
                img_path = os.path.join(cls_dir, f"{cls}_{i:03d}.png")
                img.save(img_path)

    print("Dataset generated successfully in data/ directory!")


if __name__ == "__main__":
    setup_local_dataset()
