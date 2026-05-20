import os
from detection.models.dl.mlp import MLPModel
from detection.models.dl.softmax import SoftmaxModel
from detection.models.dl.lstm import LSTMModel
from detection.models.dl.rnn import RNNModel
from evaluation.metrics import get_metrics_dict

def train_all_dl(X_train, y_train, X_test, y_test, save_dir="outputs/models"):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    models = [
        MLPModel(input_dim=X_train.shape[1]),
        SoftmaxModel(input_dim=X_train.shape[1]),
        LSTMModel(input_dim=X_train.shape[1]),
        RNNModel(input_dim=X_train.shape[1])
    ]
    
    for model in models:
        print(f"  Training {model.name}...")
        model.train(X_train, y_train, epochs=10)
        y_pred = model.predict(X_test)
        metrics = get_metrics_dict(model.name, y_test, y_pred)
        results.append(metrics)
        model.save(os.path.join(save_dir, f"{model.name}.keras"))
        
    return results
