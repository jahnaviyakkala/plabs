"""
pipeline.py
-----------
End-to-end PLABS pipeline:

  Step 1 – Load & merge all 5 CSVs
  Step 2 – Label stragglers (Algorithm 1 Labeler)
  Step 3 – Feature engineering + train/test split
  Step 4 – Train 5 ML models
  Step 5 – Train 2 DL models (CNN + LSTM)
  Step 6 – Evaluate all 7 models + produce plots
  Step 7 – Run Algorithm 1 real-time detection loop (replay)
  Step 8 – Save metrics CSV + HTML summary
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# local imports
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all_benchmarks
from labeler import add_labels
from features import extract_features, build_sequences, get_feature_names, FEATURE_COLS
from models.ml_models import train_ml_models, evaluate_model
from models.dl_models import train_dl_models
from detector import PLABSDetector, load_detector
from evaluator import (
    summary_table, plot_metric_bars, plot_confusion_matrix,
    plot_roc_curves, plot_feature_importance,
    plot_detection_timeline, plot_benchmark_comparison
)


# ── config ────────────────────────────────────────────────────────────────────

DATASET_ROOT = os.environ.get(
    "AUTOOL_DATASET",
    os.path.join(os.path.dirname(__file__), "..", "..", "AUtool-Dataset")
)
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "plots")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

TEST_SIZE  = 0.2
SEQ_LEN    = 10
STRAGGLER_K = 1.5   # labeller threshold multiplier


# ── helpers ───────────────────────────────────────────────────────────────────

def _header(msg: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {msg}")
    print(f"{'═'*60}")


# ── pipeline steps ────────────────────────────────────────────────────────────

def step1_load(dataset_root: str) -> pd.DataFrame:
    _header("Step 1 – Load & merge AUtool-Dataset")
    df = load_all_benchmarks(dataset_root)
    print(f"  Shape: {df.shape}")
    print(f"  Benchmarks: {df['benchmark'].value_counts().to_dict()}")
    return df


def step2_label(df: pd.DataFrame) -> pd.DataFrame:
    _header("Step 2 – Label stragglers")
    df = add_labels(df, k=STRAGGLER_K, also_progress=True)
    print(f"  Class distribution:\n{df['straggler'].value_counts()}")
    return df


def step3_features(df: pd.DataFrame):
    _header("Step 3 – Feature engineering")
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, scaler = extract_features(df, fit_scaler=True, scaler_path=SCALER_PATH)
    feat_names = get_feature_names(df)
    print(f"  Feature matrix: {X.shape}  |  Features: {feat_names}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")
    print(f"  Train pos%: {y_tr.mean()*100:.1f}  Test pos%: {y_te.mean()*100:.1f}")

    # sequences for DL
    Xs_tr, ys_tr = build_sequences(X_tr, y_tr, SEQ_LEN)
    Xs_te, ys_te = build_sequences(X_te, y_te, SEQ_LEN)

    return X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te, ys_tr, ys_te, scaler, feat_names


def step4_train_ml(X_tr, y_tr, X_te, y_te) -> list[dict]:
    _header("Step 4 – Train ML models (DT, RF, SVM, KNN, GBT)")
    return train_ml_models(X_tr, y_tr, X_te, y_te,
                           save_dir=MODEL_DIR, use_smote=True)


def step5_train_dl(Xs_tr, ys_tr, Xs_te, ys_te) -> list[dict]:
    _header("Step 5 – Train DL models (CNN, LSTM)")
    return train_dl_models(Xs_tr, ys_tr, Xs_te, ys_te,
                           save_dir=MODEL_DIR, epochs=25, batch_size=64)


def step6_evaluate(ml_results, dl_results, X_te, y_te,
                   feat_names: list[str]) -> None:
    _header("Step 6 – Evaluation & plots")
    all_results = ml_results + dl_results
    tbl = summary_table(all_results)
    print(f"\n{tbl.to_string()}")

    # save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tbl.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))
    print(f"\n  Metrics saved → outputs/metrics.csv")

    # bar chart
    p = plot_metric_bars(all_results, save_dir=PLOT_DIR)
    print(f"  Plot → {p}")

    # per-model confusion matrix + ROC
    probas = {}
    for name in ["RandomForest", "GradientBoosting", "DecisionTree", "KNN", "SVM"]:
        pkl = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(pkl):
            continue
        clf = joblib.load(pkl)
        y_pred = clf.predict(X_te)
        plot_confusion_matrix(y_te, y_pred, name, save_dir=PLOT_DIR)
        if hasattr(clf, "predict_proba"):
            probas[name] = clf.predict_proba(X_te)[:, 1]

    if len(probas) > 1:
        p = plot_roc_curves(probas, y_te, save_dir=PLOT_DIR)
        print(f"  ROC curves → {p}")

    # feature importance for RF
    rf_path = os.path.join(MODEL_DIR, "RandomForest.pkl")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        fi_names = feat_names[:rf.n_features_in_]
        p = plot_feature_importance(rf, fi_names, "RandomForest", PLOT_DIR)
        if p:
            print(f"  Feature importance → {p}")


def step7_detection_loop(df: pd.DataFrame, scaler) -> None:
    _header("Step 7 – Algorithm 1: Real-time detection loop (replay)")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    # load best ML model (prefer GBT or RF)
    for model_name in ["GradientBoosting", "RandomForest", "DecisionTree"]:
        mp = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if os.path.exists(mp):
            model = joblib.load(mp)
            print(f"  Using {model_name} as DetectorModel")
            break
    else:
        print("  No trained ML model found, skipping detection loop.")
        return

    detector = PLABSDetector(
        model=model,
        scaler=scaler,
        feature_cols=feat_cols,
        threshold=0.5,
    )

    # run on PageRank subset (most interesting benchmark)
    bench_df = df[df["benchmark"] == "PageRank"].copy()
    if bench_df.empty:
        bench_df = df.copy()

    # pick first job for the demo replay
    first_job = bench_df["jobId"].iloc[0]
    job_df = bench_df[bench_df["jobId"] == first_job].copy()

    print(f"  Replaying job: {first_job}  ({len(job_df)} snapshots)")

    events_df = detector.run(
        job_df,
        real_time=False,   # set True to insert time.sleep pauses
    )

    p = plot_detection_timeline(events_df, save_dir=PLOT_DIR)
    if p:
        print(f"  Timeline plot → {p}")

    events_df.to_csv(os.path.join(OUTPUT_DIR, "detection_events.csv"), index=False)
    print(f"  Events saved → outputs/detection_events.csv")


def step8_html_report(ml_results, dl_results) -> None:
    _header("Step 8 – HTML summary report")
    all_results = ml_results + dl_results
    df = pd.DataFrame(all_results)
    rows = ""
    for _, row in df.iterrows():
        style = 'style="background:#e8f5e9"' if row.get("f1", 0) > 0.8 else ""
        rows += f"""<tr {style}>
          <td>{row.get('model','')}</td>
          <td>{row.get('accuracy',0):.4f}</td>
          <td>{row.get('precision',0):.4f}</td>
          <td>{row.get('recall',0):.4f}</td>
          <td><b>{row.get('f1',0):.4f}</b></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>PLABS Straggler Detection – Results</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;color:#333}}
  h1{{color:#1565C0}}h2{{color:#1976D2;border-bottom:2px solid #90CAF9;padding-bottom:4px}}
  table{{border-collapse:collapse;width:100%}}
  th{{background:#1565C0;color:#fff;padding:8px 12px}}
  td{{padding:7px 12px;border-bottom:1px solid #ddd;text-align:center}}
  tr:hover{{background:#f5f5f5}}
  .img-grid{{display:flex;flex-wrap:wrap;gap:16px;margin-top:16px}}
  .img-grid img{{max-width:45%;border:1px solid #ccc;border-radius:6px}}
</style>
</head><body>
<h1>PLABS – Pluggable AI-based Real-time Straggler Detection</h1>
<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>Model Performance</h2>
<table>
  <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
  {rows}
</table>

<h2>Plots</h2>
<div class="img-grid">
  <img src="plots/metric_bars.png" alt="Metric bars">
  <img src="plots/roc_curves.png" alt="ROC curves">
  <img src="plots/feat_imp_RandomForest.png" alt="Feature importance">
  <img src="plots/detection_timeline.png" alt="Detection timeline">
</div>

<h2>Algorithm 1 – Detection Loop</h2>
<p>The real-time detection loop was replayed against the PageRank benchmark.
Detection events are saved in <code>outputs/detection_events.csv</code>.</p>
</body></html>"""

    path = os.path.join(OUTPUT_DIR, "report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  HTML report → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(PLOT_DIR,   exist_ok=True)

    df = step1_load(DATASET_ROOT)
    df = step2_label(df)
    (X_tr, X_te, y_tr, y_te,
     Xs_tr, Xs_te, ys_tr, ys_te,
     scaler, feat_names) = step3_features(df)

    ml_results = step4_train_ml(X_tr, y_tr, X_te, y_te)
    dl_results = step5_train_dl(Xs_tr, ys_tr, Xs_te, ys_te)

    step6_evaluate(ml_results, dl_results, X_te, y_te, feat_names)
    step7_detection_loop(df, scaler)
    step8_html_report(ml_results, dl_results)

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  ALL DONE in {elapsed:.1f}s")
    print(f"  Results → outputs/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
