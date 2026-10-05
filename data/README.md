# VisionPulse Dataset Structure

This directory contains the dataset used for training, validating, and testing the VisionPulse CNN visual recognition model.

## Directory Layout
- `train/`: Training split containing subdirectories for each visual class.
- `validation/`: Validation split for hyperparameter tuning and model checkpointing.
- `test/`: Test split for final performance evaluation (accuracy, precision, recall, confusion matrix).

## Default Dataset Details
VisionPulse is configured by default for everyday visual objects recognition (10 classes):
1. `Airplane` / Aerial Vehicle
2. `Automobile` / Passenger Vehicle
3. `Bird` / Wildlife
4. `Cat` / Domestic Pet
5. `Deer` / Wildlife
6. `Dog` / Domestic Pet
7. `Frog` / Small Animal
8. `Horse` / Domesticated Animal
9. `Ship` / Water Vessel
10. `Truck` / Heavy Transport Vehicle

## Automatic Setup
If local images are not placed in `train/`, running `python src/train.py` will automatically download and partition the standard dataset (CIFAR-10 / everyday object categories) into the `train/`, `validation/`, and `test/` folders.

You can also place your own custom image dataset inside `train/`, `validation/`, and `test/` using class subfolders:
```text
data/
├── train/
│   ├── laptop/
│   ├── mobile_phone/
│   └── cup/
├── validation/
└── test/
```
