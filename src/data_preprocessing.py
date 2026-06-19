# =============================================================================
# src/data_preprocessing.py
# AI-Powered DDoS IDS — Data Preprocessing Pipeline
#
# PURPOSE:
#   This module handles ALL data preparation before feeding it to an ML model.
#   It works for two scenarios:
#     1. Training: loading a labelled CSV dataset (CICIDS2017, NSL-KDD, etc.)
#     2. Inference: transforming a single live-captured packet or a batch CSV
#        into the exact same feature vector the model was trained on.
#
# PIPELINE OVERVIEW:
#   Raw CSV / Packet  →  Clean  →  Encode Labels  →  Select Features  →  Scale
# =============================================================================

import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Preprocessor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Paths where the fitted scaler and encoder are saved after training.
# During live inference, these are reloaded so the same transformation is applied.
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# The subset of columns we use from CICIDS2017 / NSL-KDD.
# These are well-known discriminative features for DDoS detection.
# Adjust this list if you use a different dataset with different column names.
FEATURE_COLUMNS = [
    "Flow Duration",          # Total duration of the flow in microseconds
    "Total Fwd Packets",      # Number of packets sent in the forward direction
    "Total Backward Packets", # Number of packets in the backward direction
    "Total Length of Fwd Packets",  # Total payload size forward
    "Total Length of Bwd Packets",  # Total payload size backward
    "Fwd Packet Length Max",  # Largest single packet in forward direction
    "Fwd Packet Length Min",  # Smallest packet forward
    "Fwd Packet Length Mean", # Average packet size forward
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",           # Bandwidth — a key DDoS indicator
    "Flow Packets/s",         # Packet rate — spikes signal SYN/UDP floods
    "Flow IAT Mean",          # Inter-Arrival Time average (flooding = very low IAT)
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd PSH Flags",          # TCP PSH flag (data push)
    "Bwd PSH Flags",
    "Fwd URG Flags",          # TCP URG flag (urgent pointer)
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",         # TCP flag counts — abnormal combos indicate attacks
    "SYN Flag Count",         # High SYN with no ACK = SYN Flood
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_forward",  # TCP window size — truncated in SYN floods
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Idle Mean",
    "Idle Std",
]

# Label column name in the dataset
LABEL_COLUMN = "Label"


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk. Handles common issues like leading/trailing
    spaces in column names (a known quirk of CICIDS2017 exports).

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame with cleaned column names.
    """
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    # CICIDS2017 has a notorious issue: column names often have leading spaces.
    # Strip whitespace from ALL column names to prevent KeyError surprises.
    df.columns = df.columns.str.strip()

    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Label distribution:\n{df[LABEL_COLUMN].value_counts()}")
    return df


# ===========================================================================
# 2. DATA CLEANING
# ===========================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove or fix rows/columns that would corrupt model training.

    Steps performed:
      - Drop fully-duplicate rows
      - Replace infinity values (common in rate-based features like Flow Bytes/s
        when Flow Duration = 0) with NaN, then drop those rows
      - Drop columns that are entirely NaN
      - Report how many rows were removed

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    initial_rows = len(df)
    logger.info("Starting data cleaning...")

    # Remove exact duplicate rows — these add no information
    df = df.drop_duplicates()
    logger.info(f"Dropped duplicates. Rows remaining: {len(df)}")

    # CICIDS2017 specific: Flow Bytes/s and Flow Packets/s contain 'Infinity'
    # values when the flow duration is zero. Replace with NaN then drop.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS, how="any")
    logger.info(f"Dropped rows with NaN/Inf in feature columns. Rows remaining: {len(df)}")

    # Drop any column that is entirely NaN (won't contribute to training)
    df = df.dropna(axis=1, how="all")

    removed = initial_rows - len(df)
    logger.info(f"Cleaning complete. Removed {removed} rows ({removed/initial_rows*100:.1f}%)")
    return df


# ===========================================================================
# 3. LABEL ENCODING
# ===========================================================================

def encode_labels(df: pd.DataFrame, fit: bool = True,
                  encoder: LabelEncoder = None):
    """
    Convert string labels ('BENIGN', 'DDoS', 'DoS Hulk', etc.) into integers
    that sklearn models can consume.

    Args:
        df:      DataFrame containing a LABEL_COLUMN column.
        fit:     If True, fit a new LabelEncoder and save it.
                 If False, use a pre-fitted encoder (for inference).
        encoder: Pre-fitted LabelEncoder (used when fit=False).

    Returns:
        Tuple of (encoded_labels array, fitted_encoder).

    HOW IT WORKS:
        LabelEncoder assigns each unique class an integer:
          BENIGN → 0, DDoS → 1, DoS Hulk → 2, etc.
        We save this mapping so inference time decoding gives back human-readable names.
    """
    if fit:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(df[LABEL_COLUMN])
        logger.info(f"Label encoding complete. Classes: {list(encoder.classes_)}")

        # Persist the encoder so the app can decode predictions back to strings
        os.makedirs("models", exist_ok=True)
        joblib.dump(encoder, ENCODER_PATH)
        logger.info(f"LabelEncoder saved to {ENCODER_PATH}")
    else:
        if encoder is None:
            encoder = joblib.load(ENCODER_PATH)
        encoded = encoder.transform(df[LABEL_COLUMN])

    return encoded, encoder


# ===========================================================================
# 4. FEATURE SELECTION & VALIDATION
# ===========================================================================

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns in FEATURE_COLUMNS that actually exist in this
    particular dataset. This makes the code robust to slight schema differences
    between CICIDS2017 and NSL-KDD.

    Args:
        df: Full DataFrame with all columns.

    Returns:
        DataFrame with only the feature columns present in both our list and df.
    """
    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing   = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing:
        logger.warning(f"Features not found in dataset (will be skipped): {missing}")

    logger.info(f"Using {len(available)} features for training/inference.")
    return df[available]


# ===========================================================================
# 5. FEATURE SCALING
# ===========================================================================

def scale_features(X: np.ndarray, fit: bool = True,
                   scaler: StandardScaler = None):
    """
    Apply StandardScaler: transforms each feature to zero mean and unit variance.

      z = (x - mean) / std

    Why this matters for our models:
      - SVM is HIGHLY sensitive to feature scale. Without scaling, a feature
        like 'Flow Bytes/s' (values in millions) will dominate features like
        'SYN Flag Count' (values 0 or 1).
      - Logistic Regression and Neural Networks also converge faster on scaled data.
      - Tree-based models (Random Forest, Decision Tree) are scale-invariant,
        but scaling doesn't hurt them.

    Args:
        X:      Feature matrix (numpy array).
        fit:    True = fit a new scaler (training). False = use pre-fitted (inference).
        scaler: Pre-fitted scaler for inference mode.

    Returns:
        Tuple of (scaled_X, scaler).
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"StandardScaler fitted and saved to {SCALER_PATH}")
    else:
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


# ===========================================================================
# 6. FULL TRAINING PIPELINE (orchestrator function)
# ===========================================================================

def prepare_training_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    End-to-end pipeline for preparing training and test splits.

    Orchestrates all steps above:
      Load → Clean → Select Features → Encode Labels → Scale → Split

    Args:
        filepath:     Path to the raw dataset CSV.
        test_size:    Fraction of data reserved for testing (default 20%).
        random_state: Seed for reproducibility.

    Returns:
        dict with keys: X_train, X_test, y_train, y_test, encoder, scaler, feature_names
    """
    # Step 1: Load
    df = load_dataset(filepath)

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Select feature columns
    X_df = select_features(df)
    feature_names = list(X_df.columns)   # Save for feature importance plotting later

    # Step 4: Encode target labels
    y, encoder = encode_labels(df, fit=True)

    # Step 5: Scale features
    X = X_df.values
    X_scaled, scaler = scale_features(X, fit=True)

    # Step 6: Train/test split — stratify ensures each class is proportionally
    # represented in both splits (important for imbalanced DDoS datasets)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    logger.info("Pipeline complete. Data ready for model training.")

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "encoder": encoder,
        "scaler":  scaler,
        "feature_names": feature_names,
    }


# ===========================================================================
# 7. INFERENCE PIPELINE (for live packets / batch upload)
# ===========================================================================

def preprocess_single_record(record: dict) -> np.ndarray:
    """
    Transform a single raw network flow record (from live capture or uploaded CSV)
    into a model-ready feature vector using the SAVED scaler.

    Args:
        record: Dict mapping feature names to their values for one flow.
                Example: {"SYN Flag Count": 1, "Flow Bytes/s": 50000, ...}

    Returns:
        A 2D numpy array shaped (1, n_features) ready for model.predict().

    IMPORTANT: The scaler must have been fitted during training first.
               It is loaded automatically from SCALER_PATH.
    """
    # Load the persisted scaler
    scaler = joblib.load(SCALER_PATH)

    # Build a row using only the features the model was trained on,
    # in the EXACT same order. Missing features default to 0.
    row = []
    for col in FEATURE_COLUMNS:
        val = record.get(col, 0)
        # Replace infinity values coming from live capture division-by-zero
        if np.isinf(val) or np.isnan(val):
            val = 0.0
        row.append(float(val))

    X = np.array(row).reshape(1, -1)   # (1, n_features)
    X_scaled = scaler.transform(X)

    return X_scaled


def preprocess_batch_csv(filepath: str) -> np.ndarray:
    """
    Preprocess an uploaded CSV for batch prediction (no labels required).
    Used by the Streamlit 'Upload CSV' feature.

    Args:
        filepath: Path to the uploaded CSV file.

    Returns:
        Scaled feature matrix ready for batch prediction.
    """
    df = load_dataset(filepath)
    df = clean_data(df)
    X_df = select_features(df)

    # Fill any remaining NaN with column median (safe for inference)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X_df.values)

    return X_scaled, list(X_df.columns)
