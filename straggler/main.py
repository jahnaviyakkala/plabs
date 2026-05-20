import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import yaml
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.prepare_dataset import load_all_benchmarks
from labeling.labeler import add_labels
from labeling.feature_engineering import extract_features
from training.train_ml import train_all_ml
from training.train_dl import train_all_dl
from detection.models.llm.gpt4o_icl import GPT4oICLModel
from detection.models.llm.llama_icl import LlamaICLModel
from evaluation.metrics import summary_table, get_metrics_dict
from evaluation.plots import plot_metric_bars
from sklearn.model_selection import train_test_split

def main():
    print("🚀 Starting PLABS Straggler Detection System")
    
    # Create output directories
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Data Preparation
    print("Step 1: Loading datasets...")
    # Map the dataset roots based on where I found them
    benchmarks = config['benchmarks']
    project_root = os.getcwd()
    
    # We'll use AUtool+ as the primary source for now
    dataset_root = os.path.join(project_root, "AUtool+")
    df = load_all_benchmarks(dataset_root)
    
    # Step 2: Labeling
    print("Step 2: Labeling stragglers...")
    df = add_labels(df, k=1.5)
    
    # Sample data for faster verification
    if len(df) > 50000:
        print(f"  Sampling 50,000 rows for faster execution (original: {len(df):,})")
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    
    # Step 3: Feature Engineering
    print("Step 3: Extracting features...")
    X, y, scaler = extract_features(df, fit_scaler=True, scaler_path="outputs/models/scaler.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'], random_state=config['model']['random_state']
    )
    
    # Step 4: Training ML Models
    print("Step 4: Training 3 ML models...")
    ml_results = train_all_ml(X_train, y_train, X_test, y_test)
    
    # Step 5: Training DL Models
    print("Step 5: Training 2 DL models...")
    dl_results = train_all_dl(X_train, y_train, X_test, y_test)
    
    # Step 6: LLM ICL Models
    print("Step 6: Evaluating 2 LLM ICL models...")
    llm_results = []
    
    # Select indices with at least some stragglers for the demo plot
    str_idx = np.where(y_test == 1)[0]
    norm_idx = np.where(y_test == 0)[0]
    
    if len(str_idx) > 50 and len(norm_idx) > 50:
        indices = np.concatenate([str_idx[:50], norm_idx[:50]])
    else:
        indices = np.arange(min(len(y_test), 100))
    
    X_llm = X_test[indices]
    y_llm = y_test[indices]

    for llm_class in [GPT4oICLModel, LlamaICLModel]:
        model = llm_class()
        model.train(X_train, y_train)
        y_pred = model.predict(X_llm)
        metrics = get_metrics_dict(model.name, y_llm, y_pred)
        llm_results.append(metrics)
    
    # Step 7: Final Evaluation
    print("Step 7: Generating final report...")
    all_results = ml_results + dl_results + llm_results
    table = summary_table(all_results)
    print("\nSummary Metrics:")
    print(table)
    
    plot_metric_bars(all_results)
    print("\n✅ Project execution finished. Results saved in outputs/")

if __name__ == "__main__":
    main()
