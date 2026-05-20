import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def summary_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    for col in ["accuracy", "precision", "recall", "f1"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    return df.set_index("model") if "model" in df.columns else df

def get_metrics_dict(name: str, y_true, y_pred, y_proba=None):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
