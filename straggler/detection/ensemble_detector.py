class EnsembleDetector:
    def __init__(self, weights={'model': 0.5, 'slowdown': 0.3, 'history': 0.2}):
        self.weights = weights
        self.history_scores = {} # Mock historical data: {app_id: score}

    def get_ensemble_score(self, model_proba, task_rate, avg_rate, app_id):
        """
        score = w1*model_prob + w2*slowdown + w3*history_score
        """
        # 1. Model Probability (Already normalized 0-1)
        w1 = self.weights['model']
        s1 = model_proba

        # 2. Slowdown Signal (Now aligned with PeerAnalyzer: avg_rate / task_rate)
        w2 = self.weights['slowdown']
        # If task is average or faster, slowdown score is low. 
        # If task is 5x slower (slowdown=5.0), s2 = 1.0
        slowdown_factor = (avg_rate / task_rate) if task_rate > 0 else 1.0
        s2 = min(slowdown_factor / 5.0, 1.0) if slowdown_factor > 1.2 else 0.0

        # 3. History Score (Mock lookup)
        w3 = self.weights['history']
        s3 = self.history_scores.get(app_id, 0.5) 

        ensemble_score = (w1 * s1) + (w2 * s2) + (w3 * s3)
        
        # Binary prediction based on threshold
        is_straggler = 1 if ensemble_score > 0.65 else 0
        
        return is_straggler, ensemble_score

if __name__ == "__main__":
    ensemble = EnsembleDetector()
    # Scenario: Model says 0.8 prob, but task rate is normal (1.0 compared to 1.0 avg)
    pred, score = ensemble.get_ensemble_score(0.8, 1.0, 1.0, "app_1")
    print(f"Scenario 1 - Score: {score:.2f}, Predict: {pred}")

    # Scenario: Model says 0.4 prob (low), but task rate is very slow (0.1 compared to 1.0 avg)
    pred, score = ensemble.get_ensemble_score(0.4, 0.1, 1.0, "app_1")
    print(f"Scenario 2 - Score: {score:.2f}, Predict: {pred}")
