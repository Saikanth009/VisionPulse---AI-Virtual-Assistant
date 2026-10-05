"""
VisionPulse — AI Visual Assistant for People with Low Vision
Model Evaluation Module (src/evaluate.py)

Evaluates the trained CNN model performance metrics (Accuracy, Precision, Recall, F1-Score)
on test data and generates Confusion Matrix visualization and Classification Report artifacts.
This evaluates the legacy ten-class baseline experiment, not the app's VLM descriptions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Add project root to Python path for script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    MODEL_PATH, CLASS_NAMES, CONFUSION_MATRIX_PATH, CLASSIFICATION_REPORT_PATH, TEST_DATA_PATH
)
from src.predict import load_trained_model


def generate_confusion_matrix(y_true, y_pred, save_path=CONFUSION_MATRIX_PATH):
    """
    Plots high-contrast confusion matrix heatmap and saves image artifact.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=True, annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("VisionPulse CNN Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Category", fontsize=12, fontweight="bold")
    plt.ylabel("Actual True Category", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix image saved successfully to '{save_path}'.")


def generate_classification_report_file(y_true, y_pred, save_path=CLASSIFICATION_REPORT_PATH):
    """
    Generates scikit-learn text classification report and writes to text artifact.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    header = (
        "===============================================================\n"
        "VisionPulse CNN — Model Evaluation & Classification Report\n"
        "===============================================================\n\n"
        f"Overall Accuracy:  {acc * 100:.2f}%\n"
        f"Weighted Precision: {prec * 100:.2f}%\n"
        f"Weighted Recall:    {rec * 100:.2f}%\n"
        f"Weighted F1-Score:  {f1 * 100:.2f}%\n\n"
        "Detailed Per-Class Performance:\n"
        "---------------------------------------------------------------\n"
    )

    full_report = header + report_text
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"Classification report saved successfully to '{save_path}'.")
    return full_report, acc, prec, rec, f1


def evaluate_model():
    """
    Main evaluation pipeline. Loads trained CNN model and test dataset, performs inference,
    calculates metrics, and saves evaluation artifacts.
    """
    print("==================================================")
    print("VisionPulse — Model Performance Evaluation")
    print("==================================================")

    model = load_trained_model(MODEL_PATH)

    # Load test split from local test path or fallback
    has_custom_test = (
        os.path.exists(TEST_DATA_PATH) and
        len([d for d in os.listdir(TEST_DATA_PATH) if os.path.isdir(os.path.join(TEST_DATA_PATH, d))]) > 0
    )

    if has_custom_test:
        print(f"Loading test dataset from local folder '{TEST_DATA_PATH}'...")
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            TEST_DATA_PATH,
            image_size=(32, 32),
            batch_size=64,
            shuffle=False
        )
        x_test_list = []
        y_test_list = []
        for x_b, y_b in test_ds:
            x_test_list.append(x_b.numpy() / 255.0)
            y_test_list.append(y_b.numpy())
        x_test = np.concatenate(x_test_list, axis=0)
        y_true = np.concatenate(y_test_list, axis=0)
    else:
        print("Loading test dataset from standard benchmark split...")
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype("float32") / 255.0
        y_true = y_test.flatten()

    print(f"Running inference on {len(x_test)} test samples...")
    y_pred_probs = model.predict(x_test, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate & plot artifacts
    generate_confusion_matrix(y_true, y_pred)
    report_text, acc, prec, rec, f1 = generate_classification_report_file(y_true, y_pred)

    print("\n---------------------------------------------------")
    print(f"Summary Evaluation Metrics:")
    print(f" - Test Accuracy:  {acc * 100:.2f}%")
    print(f" - Precision:      {prec * 100:.2f}%")
    print(f" - Recall:         {rec * 100:.2f}%")
    print(f" - F1-Score:       {f1 * 100:.2f}%")
    print("---------------------------------------------------")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "report_text": report_text
    }


if __name__ == "__main__":
    evaluate_model()
