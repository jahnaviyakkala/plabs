import os
import joblib
from detection.models.ml.logistic import LogisticModel
from detection.models.ml.naive_bayes import NaiveBayesModel
from detection.models.ml.knn import KNNModel
from evaluation.metrics import get_metrics_dict

def train_all_ml(X_train, y_train, X_test, y_test, save_dir="outputs/models"):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    # ── 1. Model Initialization ──────────────────────────────────────────────
    print(f"🚀 Initializing ML Models for training...")
    try:
        models = [
            LogisticModel(),
            NaiveBayesModel(),
            KNNModel()
        ]
    except Exception as e:
        print(f"❌ Critical Error during model initialization: {e}")
        raise

    # ── 2. Training Loop ─────────────────────────────────────────────────────
    for model in models:
        try:
            print(f"  [TRAIN] {model.name}...")
            model.train(X_train, y_train)
            
            print(f"  [EVAL]  {model.name}...")
            y_pred = model.predict(X_test)
            metrics = get_metrics_dict(model.name, y_test, y_pred)
            results.append(metrics)
            
            save_path = os.path.join(save_dir, f"{model.name}.pkl")
            model.save(save_path)
            print(f"  [SAVE]  {model.name} saved to {save_path}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to train/save {model.name}: {e}")
            continue
            
    # ── 3. Summary ───────────────────────────────────────────────────────────
    print(f"✅ Training Complete. {len(results)}/{len(models)} models saved.")
    return results
