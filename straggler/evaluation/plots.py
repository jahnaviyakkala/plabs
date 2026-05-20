import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc

def plot_metric_bars(results: list[dict], save_dir: str = "outputs/plots") -> str:
    """
    Generate 4 subplots for Accuracy, Precision, Recall, and F1 score.
    Matching the style of the user-provided image.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results)
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    titles = ["Accuracy", "Precision", "Recall", "F1 score"]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Map model names to numbers 1-7 for the X-axis as in the reference image
    model_names = df["model"].tolist()
    x = np.arange(1, len(model_names) + 1)
    
    color = "#84CDE5" # Light blue/teal color from the image
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        values = df[m]
        ax.bar(x, values, color=color, edgecolor="gray", alpha=0.9)
        
        ax.set_ylabel(titles[i])
        ax.set_xlabel(labels[i])
        ax.set_xticks(x)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
    plt.tight_layout()
    path = os.path.join(save_dir, "metrics_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

def plot_roc_curves(models_proba: dict, y_test: np.ndarray, save_dir: str = "outputs/plots") -> str:
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, proba in models_proba.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={roc_auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – PLABS Models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_detection_timeline(events_df, save_dir: str = "outputs/plots") -> str:
    os.makedirs(save_dir, exist_ok=True)
    if events_df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(events_df["poll_round"],
                    events_df["cumulative_stragglers"],
                    alpha=0.25, color="#DD8452")
    ax.plot(events_df["poll_round"],
            events_df["cumulative_stragglers"],
            color="#DD8452", lw=2, label="Cumulative stragglers")
    ax2 = ax.twinx()
    ax2.plot(events_df["poll_round"],
             events_df["job_progress"],
             color="#4C72B0", lw=1.5, linestyle="--", label="Job progress %")
    ax.set_xlabel("Poll round")
    ax.set_ylabel("Stragglers detected (cumulative)")
    ax2.set_ylabel("Job progress (%)")
    ax.set_title("Algorithm 1 – Real-time Detection Timeline")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "detection_timeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
