"""
detector.py
-----------
Implements Algorithm 1 from the paper — the PLABS real-time detection loop.

Algorithm 1  Symptom detection for stragglers in Hadoop
─────────────────────────────────────────────────────────
Input:  Jp  – Job progress (%)
        Ji  – List of jobs
        Mp  – Map progress (%)
        Ei  – List of ExecutionTimeSec per map
        Ti  – List of timestamps
        L   – Labeler
Output: αi  – List of detected stragglers

1:  Initialize the DetectorModel
2:  while Jp < 100.0 do
3:    for each J in Ji do
4:      // Labeler gets Mp, Ei, Ti from data matrix
5:      L ← Mp, Ei, Ti
6:      O ← DetectorModel(L)
7:      // Get αnew from O
8:      αi ← O
9:    end for
10:   // Update Jp, Mp, Ei, Ti
11:   Jp←Jpnew, Mp←Mpnew, Ei←Enew, Ti←Tnew, αi←αnew
12: end while
─────────────────────────────────────────────────────────

In our implementation:
 - "DetectorModel" is any trained sklearn / keras model loaded from disk.
 - We replay the dataset in timestamp order to simulate the polling loop.
 - RabbitMQ / InfluxDB integration stubs are included; they are activated
   when RABBITMQ_HOST and INFLUXDB_URL env-vars are set.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Callable


# ── optional message-queue / TSDB integration ─────────────────────────────────

def _try_publish_rabbitmq(payload: dict) -> None:
    host = os.environ.get("RABBITMQ_HOST")
    if not host:
        return
    try:
        import pika
        conn = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        ch = conn.channel()
        ch.queue_declare(queue="plabs.stragglers", durable=True)
        ch.basic_publish(
            exchange="",
            routing_key="plabs.stragglers",
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        conn.close()
    except Exception as e:
        print(f"  [RabbitMQ] {e}")


def _try_write_influxdb(record: dict) -> None:
    url = os.environ.get("INFLUXDB_URL")
    token = os.environ.get("INFLUXDB_TOKEN", "")
    org = os.environ.get("INFLUXDB_ORG", "plabs")
    bucket = os.environ.get("INFLUXDB_BUCKET", "stragglers")
    if not url:
        return
    try:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        client = InfluxDBClient(url=url, token=token, org=org)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        p = (Point("straggler_event")
             .tag("jobId",  record.get("jobId", ""))
             .tag("mapId",  record.get("mapId", ""))
             .field("executionTimeSec", float(record.get("executionTimeSec", 0)))
             .field("straggler", int(record.get("predicted", 0))))
        write_api.write(bucket=bucket, record=p)
        client.close()
    except Exception as e:
        print(f"  [InfluxDB] {e}")


# ── main detector class ───────────────────────────────────────────────────────

class PLABSDetector:
    """
    Wraps a trained estimator and implements Algorithm 1's polling loop
    against a DataFrame that represents the live data stream.

    Parameters
    ----------
    model       : trained sklearn or keras model (must have .predict())
    scaler      : fitted StandardScaler (or None)
    feature_cols: list of column names to use as features
    threshold   : classification threshold (for probabilistic models)
    poll_cb     : optional callback(stragglers_df) called each polling round
    """

    def __init__(
        self,
        model,
        scaler=None,
        feature_cols: list[str] | None = None,
        threshold: float = 0.5,
        poll_cb: Callable | None = None,
    ):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols or []
        self.threshold = threshold
        self.poll_cb = poll_cb

        self._is_keras = hasattr(model, "predict") and not hasattr(model, "predict_proba")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[:, 1]
            return (proba >= self.threshold).astype(int)
        # keras model
        proba = self.model.predict(X, verbose=0).ravel()
        return (proba >= self.threshold).astype(int)

    def run(
        self,
        stream_df: pd.DataFrame,
        job_progress_col: str = "job_progress",
        map_progress_col: str = "mapProgress",
        exec_time_col: str = "executionTimeSec",
        timestamp_col: str = "timestamp",
        job_col: str = "jobId",
        map_id_col: str = "mapId",
        real_time: bool = False,
        poll_interval_s: float = 1.0,
    ) -> pd.DataFrame:
        """
        Replay *stream_df* in timestamp order (Algorithm 1).

        Returns a DataFrame of detected straggler events.
        """
        df = stream_df.copy().sort_values(timestamp_col)
        detected_events = []

        # unique poll timestamps simulate one Hadoop polling cycle
        if timestamp_col not in df.columns:
            raise ValueError(f"Column '{timestamp_col}' not in DataFrame")

        poll_times = sorted(df[timestamp_col].dropna().unique())
        n_polls = len(poll_times)

        print(f"\n{'─'*60}")
        print(f" PLABS Real-time Detection Loop  ({n_polls} polling rounds)")
        print(f"{'─'*60}")

        all_stragglers: list[str] = []   # αi  in Algorithm 1

        for i, t in enumerate(poll_times):
            # Algorithm 1, line 2: while Jp < 100.0
            snapshot = df[df[timestamp_col] == t]

            # Approximate job progress from the snapshot
            if "job_progress" in snapshot.columns:
                jp = snapshot["job_progress"].mean()
            elif map_progress_col in snapshot.columns:
                jp = pd.to_numeric(
                    snapshot[map_progress_col], errors="coerce"
                ).mean()
            else:
                jp = float(i) / n_polls * 100

            if jp >= 100.0:
                print(f"  Poll {i+1}: Jp={jp:.1f}% → job complete, stopping.")
                break

            # Algorithm 1, lines 3-9: for each J in Ji
            avail_cols = [c for c in self.feature_cols if c in snapshot.columns]
            if not avail_cols:
                continue

            X_snap = snapshot[avail_cols].fillna(0).values.astype(np.float32)
            if self.scaler is not None:
                n_expected = self.scaler.n_features_in_
                if X_snap.shape[1] < n_expected:
                    pad = np.zeros((X_snap.shape[0], n_expected - X_snap.shape[1]),
                                   dtype=np.float32)
                    X_snap = np.hstack([X_snap, pad])
                elif X_snap.shape[1] > n_expected:
                    X_snap = X_snap[:, :n_expected]
                X_snap = self.scaler.transform(X_snap)

            preds = self._predict(X_snap)
            snapshot = snapshot.copy()
            snapshot["predicted_straggler"] = preds

            # Algorithm 1, line 7-8: αnew ← O
            new_stragglers = snapshot[snapshot["predicted_straggler"] == 1]
            if map_id_col in new_stragglers.columns:
                new_ids = new_stragglers[map_id_col].tolist()
            else:
                new_ids = [f"task_{j}" for j in new_stragglers.index.tolist()]

            # Accumulate (Algorithm 1, line 11)
            all_stragglers = list(set(all_stragglers + new_ids))

            detected_events.append({
                "poll_time":     t,
                "poll_round":    i + 1,
                "job_progress":  jp,
                "tasks_seen":    len(snapshot),
                "new_stragglers": len(new_stragglers),
                "cumulative_stragglers": len(all_stragglers),
            })

            if new_stragglers.shape[0] > 0:
                print(f"  Poll {i+1:4d} | Jp={jp:6.1f}% | "
                      f"Detected {len(new_stragglers)} straggler(s)  "
                      f"(cumulative: {len(all_stragglers)})")
                # publish to RabbitMQ / InfluxDB if configured
                for _, row in new_stragglers.iterrows():
                    rec = row.to_dict()
                    rec["predicted"] = 1
                    _try_publish_rabbitmq(rec)
                    _try_write_influxdb(rec)

                if self.poll_cb:
                    self.poll_cb(new_stragglers)

            # Algorithm 1, line 10-11: update Jp, Mp, Ei, Ti
            if real_time:
                time.sleep(poll_interval_s)

        print(f"\n  Detection complete.  Total unique stragglers: {len(all_stragglers)}")
        return pd.DataFrame(detected_events)


# ── loader helper ─────────────────────────────────────────────────────────────

def load_detector(model_path: str, scaler_path: str | None = None,
                  feature_cols: list[str] | None = None) -> PLABSDetector:
    """
    Load a saved sklearn model (.pkl) or keras model (.keras/.h5)
    and wrap it in a PLABSDetector.
    """
    if model_path.endswith(".pkl"):
        model = joblib.load(model_path)
    else:
        from tensorflow import keras as _keras
        model = _keras.models.load_model(model_path)

    scaler = joblib.load(scaler_path) if scaler_path else None
    return PLABSDetector(model, scaler=scaler, feature_cols=feature_cols or [])
