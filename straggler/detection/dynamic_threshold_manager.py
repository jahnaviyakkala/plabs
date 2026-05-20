import numpy as np
from collections import deque

class DynamicThresholdManager:
    """
    Manages performance thresholds dynamically based on moving statistics.
    Replaces static 'is > 80%' rules with 'is anomalous relative to recent history'.
    """
    def __init__(self, window_size=50):
        self.history = {}
        self.window_size = window_size
        self.default_sensitivities = {
            'cpu': 2.0,
            'memory': 2.0,
            'disk_read': 2.5,
            'net_recv': 2.5,
            'progress_rate': 2.0
        }

    def update(self, metric_name, value):
        """Add a new observation to the historical window for a metric."""
        if metric_name not in self.history:
            self.history[metric_name] = deque(maxlen=self.window_size)
        
        if value is not None:
            self.history[metric_name].append(float(value))

    def get_stats(self, metric_name):
        """Retrieve mean and std_dev for a metric."""
        if metric_name not in self.history or len(self.history[metric_name]) < 5:
            return None, None
        
        data = np.array(self.history[metric_name])
        return np.mean(data), np.std(data)

    def is_anomalous(self, metric_name, value, sensitivity=None):
        """
        Check if a value is statistically anomalous (Z-Score method).
        Threshold = Mean + Sensitivity * StdDev
        """
        mean, std = self.get_stats(metric_name)
        if mean is None or std == 0:
            return False
        
        s = sensitivity or self.default_sensitivities.get(metric_name, 2.0)
        z_score = abs(value - mean) / std
        
        return z_score > s

    def get_current_threshold(self, metric_name, sensitivity=None):
        """Return the current dynamic threshold for a metric."""
        mean, std = self.get_stats(metric_name)
        if mean is None:
            return None
        
        s = sensitivity or self.default_sensitivities.get(metric_name, 2.0)
        return mean + (s * std)

    def get_summary(self):
        """Generate a human-readable summary of current dynamic thresholds."""
        summary = {}
        for metric in self.history:
            thresh = self.get_current_threshold(metric)
            if thresh is not None:
                summary[metric] = round(thresh, 2)
        return summary
