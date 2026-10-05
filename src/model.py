"""
VisionPulse — AI Visual Assistant for People with Low Vision
CNN Model Architecture Module (src/model.py)

Implements the Convolutional Neural Network (CNN) feature extraction and classification pipeline.
Layers used: Conv2D, ReLU activation, MaxPooling2D, Flatten, Dense, Dropout, Softmax.
This ten-class classifier is retained only for the documented baseline experiment.
"""

import tensorflow as tf
from keras import layers, models, Sequential
from src.config import INPUT_SHAPE, CLASS_NAMES, LEARNING_RATE


def build_cnn(input_shape=INPUT_SHAPE, num_classes=len(CLASS_NAMES)):
    """
    Constructs a multi-layer Convolutional Neural Network (CNN) for visual feature recognition.

    Architecture:
    1. Conv2D (32 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
    2. Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
    3. Conv2D (128 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
    4. Flatten layer to convert spatial 2D feature maps to 1D vector
    5. Dense layer (128 units, ReLU)
    6. Dropout (0.5 rate) for regularization against overfitting
    7. Dense Output layer (num_classes units, Softmax)
    """
    model = Sequential([
        # First Convolutional Block
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_1"),
        layers.MaxPooling2D((2, 2), name="maxpooling_1"),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_2"),
        layers.MaxPooling2D((2, 2), name="maxpooling_2"),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2d_3"),
        layers.MaxPooling2D((2, 2), name="maxpooling_3"),

        # Feature Vector Flattening & Fully Connected Dense Layers
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_features"),
        layers.Dropout(0.5, name="dropout_regularization"),
        layers.Dense(num_classes, activation="softmax", name="softmax_classifier")
    ], name="VisionPulse_CNN")

    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compiles the Keras CNN model using Adam optimizer and Categorical Cross-Entropy loss.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model


def get_model_summary(model):
    """
    Returns string representation of model summary for display / viva documentation.
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)
