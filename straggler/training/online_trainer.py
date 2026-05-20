import numpy as np

class OnlineTrainer:
    def __init__(self, model, buffer_size=1000, update_threshold=100):
        self.model = model
        self.buffer_size = buffer_size
        self.update_threshold = update_threshold
        self.data_buffer_X = []
        self.data_buffer_y = []
        self.total_samples_seen = 0

    def add_data(self, X_new, y_new):
        """Add new real-time data to the buffer and trigger fine-tuning if enough samples."""
        self.data_buffer_X.append(X_new)
        self.data_buffer_y.append(y_new)
        self.total_samples_seen += 1

        # Keep only recent data (Concept Drift handling / Data Decay)
        if len(self.data_buffer_X) > self.buffer_size:
            self.data_buffer_X = self.data_buffer_X[-self.buffer_size:]
            self.data_buffer_y = self.data_buffer_y[-self.buffer_size:]

        # Periodically fine-tune
        if len(self.data_buffer_X) >= self.update_threshold and self.total_samples_seen % self.update_threshold == 0:
            self.fine_tune()

    def fine_tune(self):
        """Fine-tune the model with recent buffered data."""
        model_name = self.model.__class__.__name__
        print(f"🔄 OnlineTrainer: Fine-tuning {model_name} with {len(self.data_buffer_X)} recent samples...")
        X_train = np.array(self.data_buffer_X)
        y_train = np.array(self.data_buffer_y)
        
        # Call model's train method (incremental if model supports it, or full retraining on small buffer)
        self.model.train(X_train, y_train, epochs=2)
        print(f"✅ Model {self.model.name} updated successfully.")

if __name__ == "__main__":
    from detection.models.ml.logistic import LogisticModel
    mock_model = LogisticModel()
    trainer = OnlineTrainer(mock_model, buffer_size=50, update_threshold=20)
    
    # Simulate data stream
    for i in range(100):
        trainer.add_data(np.random.rand(5), np.random.randint(0, 2))
