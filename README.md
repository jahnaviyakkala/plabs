PLABS+ : Enhanced AI-Based Real-Time Straggler Detection and Mitigation Framework for Hadoop

PLABS+ is an enhanced intelligent framework for real-time straggler detection, root-cause analysis, and adaptive mitigation in Hadoop clusters using Machine Learning, Deep Learning, and Large Language Models (LLMs).

This project extends traditional Hadoop speculative execution by introducing:

Early straggler prediction
Dynamic threshold-based detection
Online learning
LLM-based root cause analysis
Adaptive speculative execution
Real-time monitoring pipeline
🚀 Features
✅ Real-Time Monitoring
Monitors Hadoop/YARN task execution continuously
Collects:
CPU usage
Memory usage
Disk I/O
Network statistics
Task progress
Execution time
Data locality
✅ Advanced Straggler Detection

Implements multiple detection approaches:

Machine Learning Models
Logistic Regression
Naive Bayes
K-Nearest Neighbors (KNN)
Softmax Regression
Multilayer Perceptron (MLP)
Detection Techniques
Progress-rate-based detection
Trend analysis
Dynamic thresholding
Early straggler prediction
✅ LLM-Based Root Cause Analysis

Uses:

LLaMA 3.1 (ICL)
GPT-4o (ICL)

to analyze:

system metrics
Hadoop logs
execution patterns

and identify:

CPU bottlenecks
memory pressure
disk I/O saturation
network delay
data locality issues
✅ Adaptive Speculative Execution

Instead of blindly duplicating tasks, the framework dynamically selects mitigation strategies:

Task migration
Data-local rerun
Resource-aware scheduling
Speculative execution

based on detected root causes.

✅ Online Learning
Continuously updates models using streaming data
Handles concept drift in changing cluster conditions
Supports adaptive system behavior
🏗️ System Architecture
Monitoring Agents
        ↓
RabbitMQ Streaming Pipeline
        ↓
Feature Processing + Labeling
        ↓
ML/DL Straggler Detection
        ↓
LLM Root Cause Analysis
        ↓
Decision Engine
        ↓
Adaptive Mitigation
        ↓
Feedback Loop + Online Training
🧠 Core Algorithms
1. Advanced Dynamic Labeler

Generates real-time binary labels using adaptive CPU, memory, and locality thresholds.

2. Early Straggler Detection

Detects potential stragglers during the first 20–30% of execution using:

progress rate
execution trend
3. Dynamic Threshold Detection

Uses:

Threshold = μ + Kσ

for adaptive execution-time anomaly detection.

4. LLM-Based Root Cause Analysis

Uses in-context learning (ICL) with structured prompts to generate:

straggler classification
root cause
confidence score
recommended mitigation action
5. Adaptive Speculative Execution

Applies targeted mitigation instead of static duplication strategies.

🛠️ Tech Stack
Languages
Python
ML/DL
Scikit-learn
NumPy
Pandas
Messaging & Monitoring
RabbitMQ
InfluxDB
Prometheus
Distributed Systems
Hadoop
YARN
LLM Integration
GPT-4o
LLaMA 3.1
📂 Project Structure
plabs/
│
├── monitoring/
│   ├── monitor1.py
│   ├── cluster_status.py
│
├── detection/
│   ├── ensemble_detector.py
│   ├── root_cause_analyzer.py
│
├── training/
│   ├── online_trainer.py
│   ├── train_mlp.py
│   ├── train_dl.py
│
├── scheduler/
│   ├── scheduler.py
│
├── outputs/
│   ├── plots/
│   ├── models/
│
├── datasets/
│
├── README.md
⚙️ Installation
Clone Repository
git clone https://github.com/jahnaviyakkala/plabs.git
cd plabs
Install Dependencies
pip install -r requirements.txt
Start RabbitMQ
sudo systemctl start rabbitmq-server
Start Monitoring Agent
python3 monitor1.py
Run Detection Pipeline
python3 labeler1.py
📊 Sample Output
STRAGGLER CASE

Job: application_12345_0003

Progress Rate: 0.0057 vs Avg: 0.0276
Slowdown: 4.9x
Confidence: 0.8

Root Cause:
Data Locality Latency

LLM Insight:
Disk I/O saturation causing execution stall

Decision:
RERUN_LOCAL

Actual Gain:
+4.3s saved
📈 Evaluation Metrics

The framework evaluates:

Accuracy
Precision
Recall
F1-score
Detection latency
Execution time reduction
Resource overhead
🔥 Novel Contributions
Hybrid ML + LLM architecture
Early straggler prediction
Dynamic threshold adaptation
Explainable root-cause analysis
Adaptive speculative execution
Real-time streaming pipeline
Online model recalibration
📌 Future Improvements
Graph Neural Networks (GNNs)
Reinforcement-learning-based scheduling
Kubernetes integration
Multi-cluster orchestration
Full FastAPI + React dashboard
👥 Authors
Swathi Sri
Madhumitha
Jahnavi Y

Department of Artificial Intelligence and Data Science
IIITDM Kurnool

📜 License

This project is developed for academic and research purposes
