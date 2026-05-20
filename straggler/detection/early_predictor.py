class EarlyPredictionManager:
    def __init__(self, checkpoints=[0.20, 0.40, 0.60]):
        self.checkpoints = checkpoints
        self.last_checkpoint = {} # {task_id: last_checkpoint_index}

    def forecast_completion(self, progress, rate):
        """
        Estimates total execution time based on current rate.
        Total = Current Duration + (Remaining Work / Rate)
        """
        if rate <= 0: return float('inf')
        remaining_work = 1.0 - progress
        remaining_time = remaining_work / rate
        return remaining_time # returns estimated REMAINING time

    def check_early_trigger(self, task_id, progress, current_rate, peer_stats):
        """
        Multi-checkpoint Proactive Detection
        """
        if task_id not in self.last_checkpoint:
            self.last_checkpoint[task_id] = -1
            
        current_checkpoint_idx = -1
        for i, cp in enumerate(self.checkpoints):
            if progress >= cp:
                current_checkpoint_idx = i
                
        # If we just crossed a new checkpoint
        if current_checkpoint_idx > self.last_checkpoint[task_id]:
            self.last_checkpoint[task_id] = current_checkpoint_idx
            cp_value = self.checkpoints[current_checkpoint_idx]
            
            # Forecast Analysis
            est_remaining = self.forecast_completion(progress, current_rate)
            peer_median_rate = peer_stats['median_rate'] if peer_stats else 0.05
            peer_est_remaining = (1.0 - progress) / peer_median_rate if peer_median_rate > 0 else 0
            
            # Probability calculation
            # If our estimated remaining time is 2x peers, we are a clear straggler
            slowdown_ratio = est_remaining / (peer_est_remaining + 0.01)
            probability = min(max(0.1 + (slowdown_ratio - 1) * 0.4, 0.1), 0.98)
            
            print(f"⏱ PROACTIVE CHECK at {cp_value*100:.0f}% progress")
            print(f"   Est. Remaining: {est_remaining:.1f}s | Peer Avg: {peer_est_remaining:.1f}s")
            print(f"   Early Straggler Probability: {probability:.2f}")
            
            return True, probability
            
        return False, 0.0

if __name__ == "__main__":
    early = EarlyPredictionManager()
    early.check_early_trigger("task_1", 0.1, 0.8, None) # No trigger
    early.check_early_trigger("task_1", 0.35, 0.87, None) # Trigger!
