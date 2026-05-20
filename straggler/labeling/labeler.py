"""
labeler.py
----------
Implements the PLABS Labeler component (Algorithm 1, Section IV-B).

A map task is labelled a *straggler* (label = 1) when its executionTimeSec
exceeds mean + threshold * std for all RUNNING tasks in the same job.
Only RUNNING rows are considered (SUCCEEDED tasks already finished normally).

Reference thresholds from the paper:
  - k = 1.5  (mild)  — used as default
  - k = 2.0  (strict)
"""

import pandas as pd
import numpy as np


def label_stragglers(
    df: pd.DataFrame,
    threshold_k: float = 1.5,
    exec_col: str = "executionTimeSec",
    state_col: str = "state",
    job_col: str = "jobId",
    label_col: str = "straggler",
) -> pd.DataFrame:
    """
    Add a binary 'straggler' column to *df*.

    Rules
    -----
    1. Group by jobId.
    2. Among RUNNING tasks, compute mean µ and std σ of executionTimeSec.
    3. straggler = 1  if executionTimeSec > µ + k·σ
                  0  otherwise
    4. Rows with NaN executionTimeSec are dropped.

    Returns a copy of df with the new column.
    """
    df = df.copy()
    df[exec_col] = pd.to_numeric(df[exec_col], errors="coerce")
    df = df.dropna(subset=[exec_col]).reset_index(drop=True)

    df[label_col] = 0  # default: normal

    for jid, grp in df.groupby(job_col):
        # only RUNNING tasks contribute to the statistics
        running_mask = grp[state_col].str.upper() == "RUNNING"
        running_exec = grp.loc[running_mask, exec_col]

        if running_exec.empty or running_exec.std() == 0:
            continue

        mu = running_exec.mean()
        sigma = running_exec.std()
        threshold = mu + threshold_k * sigma

        straggler_idx = grp.index[grp[exec_col] > threshold]
        df.loc[straggler_idx, label_col] = 1

    n_strag = df[label_col].sum()
    pct = 100 * n_strag / len(df) if len(df) else 0
    print(f"  Labeler (k={threshold_k}): {n_strag:,} stragglers "
          f"/ {len(df):,} tasks ({pct:.1f}%)")
    return df


def label_by_progress_lag(
    df: pd.DataFrame,
    lag_ratio: float = 0.75,
    progress_col: str = "progress",
    job_col: str = "jobId",
    label_col: str = "straggler_prog",
) -> pd.DataFrame:
    """
    Alternative labelling: a task is a straggler if its progress is less than
    *lag_ratio* × median progress of all RUNNING tasks in the same job
    at the same polling snapshot.
    This is the AUtool+ approach described in Section IV-B (progress-based).
    """
    df = df.copy()
    df[progress_col] = pd.to_numeric(df[progress_col], errors="coerce")
    df[label_col] = 0

    for jid, grp in df.groupby(job_col):
        running = grp[grp["state"].str.upper() == "RUNNING"]
        if running.empty:
            continue
        median_prog = running[progress_col].median()
        if median_prog <= 0:
            continue
        slow_idx = running.index[
            running[progress_col] < lag_ratio * median_prog
        ]
        df.loc[slow_idx, label_col] = 1

    return df


def add_labels(
    df: pd.DataFrame,
    k: float = 1.5,
    also_progress: bool = True,
) -> pd.DataFrame:
    """
    Convenience function: add both execution-time and progress-based labels.
    The primary label used for training is 'straggler' (execution-time based).
    """
    df = label_stragglers(df, threshold_k=k)
    if also_progress:
        df = label_by_progress_lag(df)
        # Consensus label: straggler if either method fires
        df["straggler_consensus"] = (
            (df["straggler"] | df["straggler_prog"]).astype(int)
        )
    return df
