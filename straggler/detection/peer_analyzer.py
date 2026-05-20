import numpy as np

class PeerAnalyzer:
    def __init__(self):
        # job_id -> list of (task_duration, progress_rate)
        self.peer_data = {}

    def update(self, job_id, duration, rate):
        if job_id not in self.peer_data:
            self.peer_data[job_id] = []
        self.peer_data[job_id].append({'duration': duration, 'rate': rate})
        # Keep only recent peers to avoid stale data
        if len(self.peer_data[job_id]) > 50:
            self.peer_data[job_id].pop(0)

    def get_peer_stats(self, job_id):
        if job_id not in self.peer_data or len(self.peer_data[job_id]) < 2:
            return None
        
        durations = [d['duration'] for d in self.peer_data[job_id]]
        rates = [d['rate'] for d in self.peer_data[job_id]]
        
        return {
            'median_duration': np.median(durations),
            'median_rate': np.median(rates),
            'std_rate': np.std(rates)
        }

    def calculate_rate_slowdown(self, job_id, task_rate):
        stats = self.get_peer_stats(job_id)
        if not stats:
            return 1.0
        
        avg_rate = stats['median_rate']
        # If task is faster than average, slowdown is < 1.0
        # If task is 5x slower than average, slowdown is 5.0
        return (avg_rate / task_rate) if task_rate > 0 else 5.0

    def calculate_slowdown(self, job_id, task_duration):
        stats = self.get_peer_stats(job_id)
        if not stats:
            return 1.0
        
        median = stats['median_duration']
        return task_duration / median if median > 0 else 1.0
