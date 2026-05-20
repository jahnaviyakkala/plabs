# PLABS+: Enhanced AI-Based Real-Time Straggler Detection and Mitigation for Hadoop

**PLABS+** extends the *Pluggable AI-based Real-Time Stragglers Detection Framework* (Liu et al., 2026) with automated mitigation and explainable root-cause analysis. The framework closes two gaps left by the original design: detected stragglers were never acted upon, and there was no human-readable explanation of why they occurred.

> Based on: Liu, X., Li, Y., Ranjan, R., Jha, D.N. (2026). *High-Confidence Computing*, 6, 100341. https://doi.org/10.1016/j.hcc.2025.100341

---

## What this does

When a Hadoop MapReduce job runs, a single slow task (a *straggler*) can hold up the entire job — because MapReduce waits for every task to finish before declaring completion. PLABS+ monitors all running tasks in real time, detects stragglers early, infers why they are slow, and automatically reschedules them to the most appropriate node.

```
Monitor (AUtool+)
    ↓
Label (Advanced Labeler — dynamic thresholds)
    ↓
Detect (Dynamic Threshold + Early Prediction + MLP + LLM)
    ↓
Decide (Adaptive Decision Engine — weighted signal fusion)
    ↓
Explain (Llama 3.1 ICL — root-cause report)
    ↓
Mitigate (Adaptive Speculative Execution via YARN)
    ↓
Feedback (InfluxDB — model recalibration)
```

---

## Key results

| Model | Accuracy | Precision | Recall | F1 | Latency |
|---|---|---|---|---|---|
| Logistic Regression | 0.96 | 0.95 | 0.96 | 0.95 | < 1 ms |
| Naive Bayes | 0.96 | 0.96 | 0.95 | 0.95 | < 1 ms |
| K-Nearest Neighbors | 0.88 | 0.87 | 0.88 | 0.87 | 50–200 ms |
| Softmax Regression | 0.98 | 0.98 | 0.97 | 0.97 | < 1 ms |
| **MLP (primary)** | **0.99** | **1.00** | **0.99** | **0.99** | **2–10 ms** |
| Llama 3.1 (ICL) | 0.72 | 0.61 | 0.92 | 0.58 | 50–300 ms |
| GPT-4o (ICL) | 0.73 | 0.20 | 0.40 | 0.54 | 200–500 ms |

- **Early detection is 2.5× faster** than waiting for an absolute time threshold to be crossed.
- **Layered detection raises accuracy from 45.2% → 95.4%** as techniques are stacked.
- LLMs are not used for binary classification — they generate per-straggler diagnostic reports (root cause, confidence, recommended action).

---

## Detection techniques

### 1. Advanced Labeler
Assigns ground-truth straggler labels using a sliding-window dynamic threshold. CPU and memory thresholds are recomputed from recent cluster history rather than fixed at configuration time. A task is labelled a straggler if:

```
Label = 1  if  (CpuPercent < θ_cpu)  OR  (MemPercent < θ_mem)  OR  (block not local)
Label = 0  otherwise
```

where `θ_cpu` and `θ_mem` are the rolling window means, not static values.

### 2. Dynamic threshold detection
Execution times are maintained in a sliding window. The detection boundary self-adjusts every cycle:

```
Threshold = μ + K·σ     (K = 1.5)
```

Tasks exceeding this boundary are flagged as stragglers.

### 3. Early straggler prediction
Monitors the first **30% of task execution** using two signals:

- **Progress rate**: `R_i = progress / elapsed_time`
- **Trend**: slope of `R_i` over a sliding window

A task is flagged early if `R_i < 0.8 · R̄` (below 80% of the window average) **or** if the trend is negative (decelerating). This delivers the 2.5× detection speedup.

### 4. LLM-based root-cause analysis
Llama 3.1 is queried via in-context learning (ICL) — **no fine-tuning**. Each prompt contains:
- System context (Hadoop analyst role)
- 5 few-shot examples from the InfluxDB annotation library
- Current task metrics as structured key-value text
- JSON output schema: `{straggler, cause, confidence, recommendation}`

Example output:
```
Task:        map_20241101_0003_m_000012
Root cause:  Hardware heterogeneity
Confidence:  0.87
Explanation: Task assigned to worker-07 (2 vCPU, 8 GB) vs cluster
             average (4 vCPU, 16 GB). ExecutionTimeSec = 142s vs
             median 67s. CpuPercent = 98% sustained.
Action:      Migrate via speculative execution to a higher-capacity
             node, or permanently exclude worker-07 from
             compute-intensive scheduling.
```

### 5. Adaptive speculative execution
Instead of blindly duplicating the task on a random node, the MLP infers the root cause and selects the targeted YARN action:

| Root cause | Action | Why |
|---|---|---|
| CPU saturation | Migrate to lowest-CPU node | Frees compute cycles |
| Memory pressure | Migrate to max-free-RAM node | Eliminates GC/swap overhead |
| Poor data locality | Rerun on block's home node | Turns remote I/O into local I/O |
| Network delay | Assign to proximal node | Shortens transfer path |
| Unknown | Speculative clone on any healthy node | General fallback |

At most 10% of tasks run speculatively at once (`speculativecap = 0.1`).

---

## System architecture

Ten integrated layers:

| Layer | Name | Role |
|---|---|---|
| 1 | Hadoop execution | Job submission, YARN containers, MapReduce tasks |
| 2 | AUtool+ monitoring | HA Agent (REST API, 2 s) + RU Agents (Prometheus, all 30 nodes) |
| 3 | RabbitMQ streaming | Fault-tolerant message bus, decouples agents from consumers |
| 4 | Advanced Labeler + InfluxDB | Dynamic binary labeling, time-series storage, CSV export |
| 5 | Multi-technique detection | Dynamic threshold + Early detector + MLP + Llama 3.1 ICL |
| 6 | Adaptive Decision Engine | Weighted signal fusion → final straggler score |
| 7 | Adaptive Speculative Execution | Cause-targeted YARN migration |
| 8 | LLM RCA | Human-readable diagnostic reports |
| 9 | Output | Reduced latency, explainability, resource reports |
| 10 | Feedback loop | Event logs → threshold recalibration → model update |

---

## Experimental setup

- **Cluster**: 30-node private cloud, Ubuntu 20.04.6, Apache Hadoop 3.2.1
- **Nodes**: 1 master (8 vCPU, 16 GB), 29 workers (2–4 vCPU, 8–16 GB), 1 monitoring node
- **Heterogeneity**: 1 worker deliberately under-provisioned (2 vCPU, 8 GB) to simulate production variance
- **Block size**: 1 GB (equal task inputs, isolating resource effects)
- **Dataset**: [AUtool public dataset](https://github.com/Xyl0014742/AUtool-Dataset)
- **Benchmarks**: WordCount, Grep, PageRank, Histogram-Movies, Histogram-Ratings

---

## Project structure

```
plabs/
├── monitoring/
│   ├── monitor1.py          # HA Agent + RU Agent orchestrator
│   └── cluster_status.py    # Cluster health snapshot
├── detection/
│   ├── ensemble_detector.py # Multi-technique detection pipeline
│   └── root_cause_analyzer.py  # LLM ICL integration
├── training/
│   ├── train_mlp.py         # MLP training on AUtool dataset
│   ├── train_dl.py          # Softmax + other DL models
│   └── online_trainer.py    # Feedback loop recalibration
├── scheduler/
│   └── scheduler.py         # Adaptive Speculative Execution controller
├── datasets/                # AUtool dataset (CSV exports from InfluxDB)
├── outputs/
│   ├── models/              # Saved model weights
│   └── plots/               # Evaluation graphs
└── README.md
```

---

## Installation

**Prerequisites**: Python 3.9+, RabbitMQ, InfluxDB, Prometheus, Hadoop 3.2.1, YARN

```bash
# Clone
git clone https://github.com/jahnaviyakkala/plabs.git
cd plabs

# Install Python dependencies
pip install -r requirements.txt

# Start RabbitMQ
sudo systemctl start rabbitmq-server

# Start monitoring agents (run on each node)
python3 monitoring/monitor1.py

# Run the full detection + labeling pipeline
python3 detection/ensemble_detector.py
```

For LLM-based RCA with Llama 3.1, a local vLLM endpoint is required (24 GB GPU). GPT-4o can be substituted via the OpenAI API by setting `OPENAI_API_KEY` in your environment.

---

## Hadoop configuration

Add these to `mapred-site.xml` to enable speculative execution:

```xml
<property>
  <name>mapreduce.map.speculative</name>
  <value>true</value>
</property>
<property>
  <name>mapreduce.reduce.speculative</name>
  <value>true</value>
</property>
<property>
  <name>mapreduce.job.speculative.speculativecap</name>
  <value>0.1</value>
</property>
<property>
  <name>mapreduce.job.speculative.slowtaskthreshold</name>
  <value>1.0</value>
</property>
```

---

## Sample output

```
STRAGGLER DETECTED
──────────────────────────────────────────
Job:           application_12345_0003
Task:          map_20241101_0003_m_000012
Progress rate: 0.0057  (cluster avg: 0.0276)
Slowdown:      4.9×
Confidence:    0.80

Root cause:    Data locality latency
LLM insight:   Disk I/O saturation — block assigned to remote node
Decision:      RERUN_LOCAL
Time saved:    +4.3 s
──────────────────────────────────────────
```

---

## Tech stack

| Category | Tools |
|---|---|
| Languages | Python |
| ML/DL | Scikit-learn, NumPy, Pandas |
| Monitoring | Prometheus, InfluxDB |
| Messaging | RabbitMQ |
| Distributed systems | Apache Hadoop 3.2.1, YARN |
| LLMs | Llama 3.1 (local vLLM), GPT-4o (cloud API) |

---

## Novel contributions

1. **Advanced Labeler** — sliding-window dynamic thresholds replace the original fixed resource-sufficiency cutoffs
2. **Early straggler prediction** — detects at-risk tasks in the first 30% of execution, 2.5× earlier than conventional methods
3. **Adaptive Decision Engine** — fuses four detection signals for 95.4% composite accuracy
4. **LLM-based explainability** — Llama 3.1 via ICL generates per-straggler diagnostic reports without fine-tuning
5. **Cause-targeted speculative execution** — maps inferred root cause to the specific YARN corrective action, replacing blind task duplication
6. **Feedback loop** — every resolved event recalibrates thresholds and updates model weights at runtime

---

## Future work

- **Fine-tune Llama 3.1** on the AUtool dataset using LoRA/QLoRA to close the binary classification gap (currently 0.72 vs MLP's 0.99)
- **Adaptive monitoring interval** — event-driven polling (0.5 s during bursts, 5 s during stable periods) to reduce overhead by ~30–50%
- **Large-scale stress testing** — 100–500 node clusters with deliberate node failures and network throttling
- **Reinforcement learning scheduler** — proactive task placement using Q-learning/PPO to prevent stragglers before they form
- **Kubernetes integration** and multi-cluster orchestration

---

## Authors

Swathi Sri · Madhumitha · Jahnavi Y  
Department of AI & DS, IIITDM Kurnool
Academic Year 2025–2026

---

## Reference

Liu, X., Li, Y., Ranjan, R., & Jha, D.N. (2026). Pluggable AI-based real-time stragglers detection framework in Hadoop. *High-Confidence Computing*, 6, 100341. https://doi.org/10.1016/j.hcc.2025.100341

---

*Developed for academic and research purposes.*
