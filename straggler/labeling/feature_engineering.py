"""
features.py
-----------
Builds the final feature matrix used by all PLABS ML/DL models.

Feature groups (from Table 2 in the paper):
  A – Task progress & timing     (map.csv)
  B – Worker system resources    (workers.csv join)
  C – Job-level progress         (jobDetails.csv join)
  D – Cluster-level resources    (cluster.csv join)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os


# ── feature column definitions ────────────────────────────────────────────────

FEATURE_COLS = [
    # A – Task
    "progress",
    "executionTimeSec",
    "elapsedTimeSec",

    # B – Worker
    "w_cpu_percent",
    "w_cpu_count",
    "w_mem_percent",
    "w_net_upload",
    "w_net_download",

    # C – Job Details
    "mapProgress",
    "mapsRunning",
    "mapsCompleted",
    "mapsPending",
    "mapsTotal",

    # D – Cluster
    "cpuUsage",
    "memoryUsage",
    "availableMB",
    "allocatedVirtualCores",
]

LABEL_COL = "straggler"
SEQ_LEN   = 10   # time-steps for LSTM / CNN


def extract_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
    scaler_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Return (X, y, fitted_scaler).

    X shape: (n_samples, n_features)
    y shape: (n_samples,)  — binary 0/1
    """
    # Keep only rows that have the label
    df = df.dropna(subset=[LABEL_COL]).copy()

    # Select and coerce feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    X_df = df[available].copy()

    # Fill NaN with column median
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    # Add derived features
    if "executionTimeSec" in X_df and "elapsedTimeSec" in X_df:
        X_df["exec_elapsed_ratio"] = (
            X_df["executionTimeSec"] / (X_df["elapsedTimeSec"] + 1e-9)
        ).clip(0, 10)
    if "w_cpu_percent" in X_df and "cpuUsage" in X_df:
        X_df["cpu_delta"] = X_df["w_cpu_percent"] - X_df["cpuUsage"]

    X = X_df.values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)

    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    elif scaler is not None:
        X = scaler.transform(X)

    return X, y, scaler


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat (n, f) feature matrix into overlapping windows
    (n - seq_len, seq_len, f) for LSTM and 1-D CNN.
    """
    n = len(X)
    if n <= seq_len:
        raise ValueError(f"Need at least {seq_len+1} samples; got {n}")
    Xs = np.stack([X[i: i + seq_len] for i in range(n - seq_len)])
    ys = y[seq_len:]          # label is the last step in the window
    return Xs, ys


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the list of feature names actually present."""
    base = [c for c in FEATURE_COLS if c in df.columns]
    extra = []
    if "executionTimeSec" in df and "elapsedTimeSec" in df:
        extra.append("exec_elapsed_ratio")
    if "w_cpu_percent" in df and "cpuUsage" in df:
        extra.append("cpu_delta")
    return base + extra
