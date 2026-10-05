"""
VisionPulse — AI Visual Assistant for People with Low Vision
Configuration Module (src/config.py)
Centralized hyperparameters, paths, class labels, and metadata.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = str(BASE_DIR)

# Dataset Directories
DATASET_PATH = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA_PATH = os.path.join(DATASET_PATH, "train")
VAL_DATA_PATH = os.path.join(DATASET_PATH, "validation")
TEST_DATA_PATH = os.path.join(DATASET_PATH, "test")

# Legacy ten-class baseline experiment checkpoint
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "visionpulse_cnn_baseline.keras")

# Output Artifacts Paths
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_PATH = os.path.join(OUTPUTS_PATH, "plots")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_PATH, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_PATH, "classification_report.txt")
ACCURACY_LOSS_PLOT_PATH = os.path.join(PLOTS_PATH, "accuracy_loss.png")

# Ensure required output directories exist
for path in [MODEL_DIR, OUTPUTS_PATH, PLOTS_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]:
    os.makedirs(path, exist_ok=True)

# Hyperparameters
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
SEED = 42

# Target Class Names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Low-Vision Accessible Class Descriptions & Guidance
CLASS_DESCRIPTIONS = {
    "airplane": {
        "display_name": "Airplane / Aircraft",
        "category": "transportation",
        "spoken_name": "an airplane in the sky or on a runway",
        "accessibility_note": "Large aerial transport vehicle. Ensure clear open space if outdoors."
    },
    "automobile": {
        "display_name": "Automobile / Car",
        "category": "transportation",
        "spoken_name": "a passenger car or automobile",
        "accessibility_note": "Motor vehicle nearby. Exercise caution near roadways and driveways."
    },
    "bird": {
        "display_name": "Bird",
        "category": "animal",
        "spoken_name": "a bird outdoors",
        "accessibility_note": "Small winged animal, often perching on branches, railings, or ground."
    },
    "cat": {
        "display_name": "Cat",
        "category": "animal / pet",
        "spoken_name": "a cat or pet animal",
        "accessibility_note": "Domestic animal nearby. Watch floor level to avoid tripping."
    },
    "deer": {
        "display_name": "Deer",
        "category": "wildlife",
        "spoken_name": "a deer in a natural or park setting",
        "accessibility_note": "Wild animal in outdoor environment. Maintain a safe distance."
    },
    "dog": {
        "display_name": "Dog",
        "category": "animal / pet",
        "spoken_name": "a dog or domestic pet",
        "accessibility_note": "Canine companion present. Listen for barking or movement."
    },
    "frog": {
        "display_name": "Frog",
        "category": "amphibian",
        "spoken_name": "a frog or small amphibian",
        "accessibility_note": "Small amphibian typically near moist garden or water areas."
    },
    "horse": {
        "display_name": "Horse",
        "category": "animal",
        "spoken_name": "a horse",
        "accessibility_note": "Large domesticated animal. Listen for hoof steps and movements."
    },
    "ship": {
        "display_name": "Ship / Watercraft",
        "category": "transportation",
        "spoken_name": "a ship or vessel on water",
        "accessibility_note": "Water transport craft visible over open water or harbour."
    },
    "truck": {
        "display_name": "Truck / Heavy Vehicle",
        "category": "transportation",
        "spoken_name": "a large truck or cargo vehicle",
        "accessibility_note": "Large commercial motor vehicle. Be mindful of traffic safety."
    }
}
