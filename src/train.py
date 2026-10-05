"""
VisionPulse — AI Visual Assistant for People with Low Vision
Training Pipeline Module (src/train.py)

Loads dataset, performs data augmentation, builds and compiles the CNN model,
trains with callbacks, saves trained Keras model, and plots loss/accuracy history.
This is a legacy ten-class baseline experiment; the Streamlit app does not use it.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks

# Add project root to Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    INPUT_SHAPE, CLASS_NAMES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    PLOTS_PATH, ACCURACY_LOSS_PLOT_PATH, SEED
)
from src.model import build_cnn, compile_model


def load_dataset():
    """
    Loads dataset from local data directory or downloads and prepares standard dataset (CIFAR-10)
    if local custom data folders are empty.
    Populates data/train, data/validation, and data/test.
    """
    print("==================================================")
    print("VisionPulse Training Pipeline — Dataset Loading")
    print("==================================================")

    # Check if custom train dataset directory exists and has subdirectories
    has_custom_train = (
        os.path.exists(TRAIN_DATA_PATH) and
        len([d for d in os.listdir(TRAIN_DATA_PATH) if os.path.isdir(os.path.join(TRAIN_DATA_PATH, d))]) > 0
    )

    if has_custom_train:
        print(f"Loading custom image dataset from '{TRAIN_DATA_PATH}'...")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            TRAIN_DATA_PATH,
            image_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            seed=SEED
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            VAL_DATA_PATH,
            image_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            seed=SEED
        )
        # Normalize pixel values
        train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
        val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
        return train_ds, val_ds, None

    else:
        print("Local data/train folder is empty. Downloading standard CIFAR-10 benchmark dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize [0, 255] to [0.0, 1.0]
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # One-hot encode targets
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(CLASS_NAMES))

        # Create Validation Split (20% of training set)
        val_samples = int(len(x_train) * 0.2)
        x_val = x_train[:val_samples]
        y_val_cat = y_train_cat[:val_samples]
        x_train_split = x_train[val_samples:]
        y_train_cat_split = y_train_cat[val_samples:]

        print(f"Dataset Split Summary:")
        print(f" - Training samples:   {len(x_train_split)}")
        print(f" - Validation samples: {len(x_val)}")
        print(f" - Testing samples:    {len(x_test)}")
        print(f" - Number of Classes:  {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")

        return (x_train_split, y_train_cat_split), (x_val, y_val_cat), (x_test, y_test_cat)


def get_data_augmentation():
    """
    Constructs Data Augmentation sequential pipeline to reduce overfitting.
    Includes random flip, rotation, and zoom.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="Data_Augmentation")


def plot_training_history(history, save_path=ACCURACY_LOSS_PLOT_PATH):
    """
    Plots Training vs Validation Accuracy and Loss curves and saves image artifact.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#00E5FF", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s-", label="Validation Accuracy", color="#FFD700", linewidth=2)
    ax1.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Accuracy Score", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#FF5252", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s-", label="Validation Loss", color="#AB47BC", linewidth=2)
    ax2.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=12)
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("VisionPulse CNN Model Training History", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training history curves saved successfully to '{save_path}'.")


def train_model():
    """
    Main training execution function.
    """
    # 1. Load Dataset
    train_data, val_data, test_data = load_dataset()

    # 2. Build & Compile Model
    print("\nBuilding CNN Architecture...")
    model = build_cnn(input_shape=INPUT_SHAPE, num_classes=len(CLASS_NAMES))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    # 3. Define Callbacks
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    cb_list = [cb_early_stop, cb_checkpoint, cb_reduce_lr]

    # 4. Train Model
    print(f"\nStarting CNN Model Training for {EPOCHS} epochs (Batch Size: {BATCH_SIZE})...")

    if isinstance(train_data, tuple):
        x_tr, y_tr = train_data
        x_v, y_v = val_data

        # Train with Data Augmentation
        data_aug = get_data_augmentation()
        train_dataset_augmented = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        train_dataset_augmented = (
            train_dataset_augmented
            .shuffle(buffer_size=10000)
            .batch(BATCH_SIZE)
            .map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        val_dataset = (
            tf.data.Dataset.from_tensor_slices((x_v, y_v))
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        history = model.fit(
            train_dataset_augmented,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=cb_list
        )
    else:
        history = model.fit(
            train_data,
            epochs=EPOCHS,
            validation_data=val_data,
            callbacks=cb_list
        )

    # 5. Save Final Model explicitly
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nTrained CNN Model saved successfully to '{MODEL_PATH}'.")

    # 6. Plot Training Curves
    plot_training_history(history)

    print("\nTraining completed cleanly!")
    return model, history


if __name__ == "__main__":
    train_model()
