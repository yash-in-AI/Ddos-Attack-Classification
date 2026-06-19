# =============================================================================
# src/model_training.py
# AI-Powered DDoS IDS — Model Training, Evaluation & Comparison
#
# PURPOSE:
#   Train five different classifiers on the preprocessed dataset, evaluate
#   them, generate a feature importance analysis, and persist each model
#   as a .pkl file so the Streamlit app can load them for live inference.
#
# MODELS COMPARED:
#   1. Random Forest    — ensemble of decision trees, robust, interpretable
#   2. Decision Tree    — single tree, fast, highly explainable
#   3. SVM              — effective in high-dimensional spaces
#   4. Logistic Regression — linear baseline, fast, interpretable
#   5. Neural Network   — deep learning via TensorFlow/Keras
# =============================================================================

import os
import time
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

# TensorFlow / Keras for the Neural Network model
# Wrapped in try/except so the rest of the module still works if TF isn't installed
try:
    import tensorflow as tf
    import keras
    from keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not found. Neural Network model will be skipped.")

from src.data_preprocessing import prepare_training_data

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTraining")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR   = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# 1. SKLEARN MODEL DEFINITIONS
# ===========================================================================

def get_sklearn_models() -> dict:
    """
    Return a dictionary of {model_name: sklearn_estimator} objects.

    Design notes:
      - Random Forest uses 100 trees; n_jobs=-1 uses all CPU cores (faster).
      - Decision Tree max_depth=20 avoids overfitting pure memorisation.
      - SVM with RBF kernel is powerful but slow on large datasets; we use
        probability=True so we can get confidence scores.
      - Logistic Regression with max_iter=1000 for convergence on many features.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,       # Grow trees until leaves are pure (default)
            min_samples_split=5,
            n_jobs=-1,            # Parallelise across all CPU cores
            random_state=42,
            class_weight="balanced",  # Handles imbalanced DDoS vs. BENIGN ratio
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        ),
        "SVM": SVC(
            kernel="rbf",         # Radial Basis Function kernel
            C=1.0,                # Regularisation strength
            gamma="scale",        # Kernel coefficient auto-scaled by feature variance
            probability=True,     # Enable predict_proba() for confidence scores
            random_state=42,
            class_weight="balanced",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",       # Works well for multi-class problems
            multi_class="auto",
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    return models


# ===========================================================================
# 2. NEURAL NETWORK BUILDER
# ===========================================================================

def build_neural_network(input_dim: int, num_classes: int) -> "keras.Model":
    """
    Build a simple feed-forward neural network for multi-class traffic classification.

    Architecture:
      Input(n_features)
        → Dense(256, ReLU)  → BatchNorm → Dropout(0.3)
        → Dense(128, ReLU)  → BatchNorm → Dropout(0.2)
        → Dense(64,  ReLU)
        → Dense(num_classes, Softmax)

    Design rationale:
      - BatchNormalization stabilises training on the high-variance network features.
      - Dropout is a regularisation technique that randomly deactivates neurons
        during training, preventing the model from memorising the training data.
      - Softmax on the output layer converts raw scores to probabilities that
        sum to 1.0 across all traffic classes.

    Args:
        input_dim:   Number of input features (matches FEATURE_COLUMNS length).
        num_classes: Number of unique traffic classes in the dataset.

    Returns:
        Compiled Keras model.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed. Cannot build neural network.")

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Hidden layer 1: 256 neurons with ReLU activation
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layer 2: 128 neurons
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Hidden layer 3: 64 neurons (bottleneck representation)
        layers.Dense(64, activation="relu"),

        # Output layer: one neuron per class, Softmax for probability distribution
        layers.Dense(num_classes, activation="softmax"),
    ], name="DDoS_Classifier")

    # Compile with Adam optimiser (adaptive learning rate) and sparse categorical
    # cross-entropy (efficient for integer-encoded labels)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    return model


# ===========================================================================
# 3. TRAINING AND EVALUATION
# ===========================================================================

def train_and_evaluate(data: dict) -> dict:
    """
    Train all models, evaluate on the test set, and collect metrics.

    Args:
        data: Dict returned by prepare_training_data() containing X_train,
              X_test, y_train, y_test, encoder, scaler, feature_names.

    Returns:
        results dict: {model_name: {accuracy, f1, precision, recall, train_time, ...}}
    """
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    encoder = data["encoder"]
    feature_names = data["feature_names"]
    num_classes = len(encoder.classes_)

    results = {}
    sklearn_models = get_sklearn_models()

    # -------------------------------------------------------------------
    # Train sklearn models
    # -------------------------------------------------------------------
    for name, model in sklearn_models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {name}")

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time

        # Measure prediction speed (important for real-time IDS)
        pred_start = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = time.perf_counter() - pred_start

        # Collect metrics
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        logger.info(f"  Accuracy:   {acc:.4f}")
        logger.info(f"  F1 Score:   {f1:.4f}")
        logger.info(f"  Train time: {train_time:.2f}s | Pred time: {pred_time:.4f}s")
        logger.info(f"\nClassification Report:\n"
                    f"{classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0)}")

        # Save model to disk
        model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"  Model saved: {model_path}")

        results[name] = {
            "accuracy":   acc,
            "f1":         f1,
            "precision":  precision,
            "recall":     recall,
            "train_time": round(train_time, 3),
            "pred_time":  round(pred_time * 1000, 2),  # ms per batch
            "model_path": model_path,
            "y_pred":     y_pred,
        }

        # Plot confusion matrix for this model
        _plot_confusion_matrix(y_test, y_pred, encoder.classes_, name)

    # -------------------------------------------------------------------
    # Train Neural Network (TensorFlow)
    # -------------------------------------------------------------------
    if TF_AVAILABLE:
        logger.info(f"\n{'='*60}")
        logger.info("Training: Neural Network")

        nn_model = build_neural_network(X_train.shape[1], num_classes)

        start_time = time.perf_counter()
        history = nn_model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=512,
            validation_split=0.1,
            callbacks=[
                # Stop training early if validation loss stops improving (prevents overfitting)
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                # Reduce learning rate when training plateaus
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
            ],
            verbose=1,
        )
        train_time = time.perf_counter() - start_time

        pred_start = time.perf_counter()
        y_pred_proba = nn_model.predict(X_test, verbose=0)
        y_pred_nn = np.argmax(y_pred_proba, axis=1)
        pred_time = time.perf_counter() - pred_start

        acc       = accuracy_score(y_test, y_pred_nn)
        f1        = f1_score(y_test, y_pred_nn, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred_nn, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred_nn, average="weighted", zero_division=0)

        logger.info(f"  NN Accuracy:  {acc:.4f}")
        logger.info(f"  NN F1 Score:  {f1:.4f}")

        # Save Keras model in the native SavedModel format
        nn_path = os.path.join(MODEL_DIR, "neural_network.keras")
        nn_model.save(nn_path)
        logger.info(f"  Neural network saved to: {nn_path}")

        # Also save training history for plotting in the dashboard
        _plot_nn_history(history)
        _plot_confusion_matrix(y_test, y_pred_nn, encoder.classes_, "Neural Network")

        results["Neural Network"] = {
            "accuracy":   acc,
            "f1":         f1,
            "precision":  precision,
            "recall":     recall,
            "train_time": round(train_time, 3),
            "pred_time":  round(pred_time * 1000, 2),
            "model_path": nn_path,
            "y_pred":     y_pred_nn,
        }

    # -------------------------------------------------------------------
    # Feature importance (from Random Forest — best source for this)
    # -------------------------------------------------------------------
    if "Random Forest" in results:
        rf_model = joblib.load(results["Random Forest"]["model_path"])
        _plot_feature_importance(rf_model, feature_names)

    # -------------------------------------------------------------------
    # Save results summary to JSON (used by Streamlit dashboard)
    # -------------------------------------------------------------------
    summary = {name: {k: v for k, v in metrics.items() if k != "y_pred"}
               for name, metrics in results.items()}

    summary_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nModel comparison summary saved to {summary_path}")

    return results


# ===========================================================================
# 4. VISUALISATION HELPERS
# ===========================================================================

def _plot_confusion_matrix(y_true, y_pred, class_names, model_name: str):
    """
    Generate and save a confusion matrix heatmap.

    A confusion matrix shows:
      - Diagonal: correct predictions (True Positives)
      - Off-diagonal: misclassifications

    For an IDS, a False Negative (attack classified as BENIGN) is far more
    dangerous than a False Positive (BENIGN classified as attack).
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    path = os.path.join(RESULTS_DIR, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {path}")


def _plot_feature_importance(rf_model, feature_names: list, top_n: int = 20):
    """
    Extract and plot the top N most important features from the Random Forest.

    Feature importance in Random Forest is computed as the mean decrease in
    impurity (Gini) across all trees — a higher value means the feature
    contributed more to distinguishing attack from normal traffic.

    This is a KEY component for explaining the model during your presentation.
    You can say: "Our model relies heavily on SYN Flag Count and Flow Bytes/s,
    which are known indicators of SYN Flood and UDP Flood attacks respectively."
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]   # Top-N descending

    top_features = [feature_names[i] for i in indices]
    top_scores   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_features[::-1], top_scores[::-1], color="steelblue")
    ax.set_xlabel("Feature Importance (Mean Decrease in Gini Impurity)")
    ax.set_title(f"Top {top_n} Most Important Features (Random Forest)", fontsize=13)

    # Add value labels on the bars
    for bar, score in zip(bars, top_scores[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance chart saved: {path}")

    # Also save raw values as CSV for the Streamlit dashboard to load
    fi_df = pd.DataFrame({"feature": top_features, "importance": top_scores})
    fi_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)


def _plot_nn_history(history):
    """
    Plot the Neural Network training/validation loss and accuracy curves.
    These curves tell us:
      - If training loss << validation loss → overfitting
      - If both are high → underfitting
      - Smooth convergence → good training
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Neural Network — Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Neural Network — Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "nn_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"NN training history saved: {path}")


# ===========================================================================
# 5. ENTRY POINT
# ===========================================================================

def run_training_pipeline(dataset_path: str):
    """
    Full end-to-end training pipeline. Call this once to train all models.

    Args:
        dataset_path: Path to the CICIDS2017 or NSL-KDD CSV file.
    """
    logger.info("=" * 60)
    logger.info("Starting Full Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Prepare data
    data = prepare_training_data(dataset_path)

    # Step 2: Train and evaluate all models
    results = train_and_evaluate(data)

    # Step 3: Print final comparison table
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    header = f"{'Model':<22} {'Accuracy':>9} {'F1':>9} {'Train(s)':>10} {'Pred(ms)':>10}"
    logger.info(header)
    logger.info("-" * 62)
    for name, metrics in results.items():
        logger.info(
            f"{name:<22} {metrics['accuracy']:>9.4f} {metrics['f1']:>9.4f} "
            f"{metrics['train_time']:>10.2f} {metrics['pred_time']:>10.2f}"
        )

    return results


if __name__ == "__main__":
    # USAGE: python -m src.model_training
    # Change the path below to your actual dataset location.
    run_training_pipeline("data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
