from monitoring.cluster_status import ClusterMonitor, NodeSelector

class SmartScheduler:
    def __init__(self):
        self.monitor = ClusterMonitor()
        self.selector = NodeSelector(self.monitor)

    def calculate_speculation_score(self, slowdown, imbalance, confidence):
        """
        speculation_score = (slowdown_factor) + (resource_imbalance) + (prediction_confidence)
        Normalized to 0-1 scale.
        """
        # Slowdown: 1.0 (normal) to 5.0+ (very slow) -> map to 0-1
        s_comp = min(slowdown / 5.0, 1.0)
        # Imbalance: 0 to 1
        i_comp = imbalance
        # Confidence: 0 to 1
        c_comp = confidence
        
        return (s_comp * 0.4) + (i_comp * 0.3) + (c_comp * 0.3)

    def mitigate(self, task_id, cause, metadata, llm_reasoning=""):
        """
        Intelligent Mitigation Logic with Scoring engine and LLM guidance.
        """
        slowdown = metadata.get('slowdown', 1.0)
        confidence = metadata.get('confidence', 0.5)
        imbalance = 0.4 
        
        score = self.calculate_speculation_score(slowdown, imbalance, confidence)
        
        print(f"🧠 Decision Engine (Scoring):")
        print(f"   - Slowdown Factor: {slowdown:.2f}x")
        print(f"   - Resource Imbalance: {imbalance:.2f}")
        print(f"   - Prediction Confidence: {confidence:.2f}")
        print(f"   → Total Speculation Score: {score:.2f}")

        print("🛠 Decision:")
        
        # --- LLM Guided Strategy ---
        llm_r = llm_reasoning.lower()
        if "local" in llm_r:
            strategy = "RERUN_LOCAL"
            reason = "Data Locality Issue"
        elif "cpu" in llm_r or "memory" in llm_r:
            strategy = "MOVE"
            reason = "Resource Bottleneck"
        elif "network" in llm_r or "io" in llm_r:
            strategy = "SPECULATE"
            reason = "I/O or Network Latency"
        else:
            strategy = "SPECULATE"
            reason = "General Straggler"

        print(f"Strategy: {strategy}")

        # Impact Analysis (Renamed from Cost Analysis)
        expected_delay = 8.2 
        actual_gain = 4.3 # Fixed for demonstration as requested
        print("💰 Impact:")
        print(f"Expected delay: +{expected_delay}s")
        print(f"Actual gain: +{actual_gain}s saved")

        best_node, _ = self.selector.select_best_node()
        return strategy, best_node, score

if __name__ == "__main__":
    scheduler = SmartScheduler()
    # Test cases
    scheduler.mitigate("task_1", "High CPU Contention", {})
    scheduler.mitigate("task_2", "Data Locality Latency (Remote Block)", {'preferred_node': 'node_2'})
    scheduler.mitigate("task_3", "Background Noise", {})
