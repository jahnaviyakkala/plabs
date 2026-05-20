import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str = "outputs/plots") -> str:
    os.makedirs(save_dir, exist_ok=True)
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Straggler"],
                yticklabels=["Normal", "Straggler"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    fig.tight_layout()
    path = os.path.join(save_dir, f"cm_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
