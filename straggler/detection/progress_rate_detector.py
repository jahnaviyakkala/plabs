import numpy as np
import pandas as pd

class ProgressRateDetector:
    def __init__(self, threshold=0.75):
        """
        threshold: sensitivity of straggler detection. 
                   If task_rate < threshold * avg_rate, it's a straggler.
        """
        self.threshold = threshold

    def detect(self, df):
        """
        Detects stragglers based on progress rate compared to cluster average.
        Requires 'progress' and 'time_passed' (or execution_time) columns.
        """
        # Ensure required columns exist
        if 'progress' not in df.columns:
            # If not present, simulate or derive from other metrics
            if 'total_bytes' in df.columns and 'bytes_read' in df.columns:
                df['progress'] = df['bytes_read'] / df['total_bytes']
            else:
                # Default mock progress if unavailable
                df['progress'] = np.random.uniform(0.1, 0.9, size=len(df))

        if 'execution_time' not in df.columns:
            df['execution_time'] = df['time_passed'] if 'time_passed' in df.columns else 1.0

        # Avoid division by zero
        df['execution_time'] = df['execution_time'].replace(0, 0.001)

        # Calculate progress rate
        df['progress_rate'] = df['progress'] / df['execution_time']

        # Group by job_id to calculate cluster (job) average
        if 'job_id' in df.columns:
            avg_rates = df.groupby('job_id')['progress_rate'].transform('mean')
        else:
            avg_rates = df['progress_rate'].mean()

        # Prediction logic
        df['predicted_straggler'] = (df['progress_rate'] < self.threshold * avg_rates).astype(int)
        
        return df

if __name__ == "__main__":
    # Example usage
    data = {
        'task_id': [1, 2, 3, 4, 5],
        'job_id': [101, 101, 101, 101, 101],
        'progress': [0.5, 0.6, 0.1, 0.55, 0.5],
        'execution_time': [10, 10, 10, 11, 10]
    }
    df = pd.DataFrame(data)
    detector = ProgressRateDetector(threshold=0.7)
    results = detector.detect(df)
    print("Progress Rate Detection Results:")
    print(results[['task_id', 'progress_rate', 'predicted_straggler']])
