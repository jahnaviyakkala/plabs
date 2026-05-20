import json
import pika
import pandas as pd
import time
import os
import sys
import yaml
from labeling.root_cause_analyzer import RootCauseAnalyzer
from mitigation.scheduler import SmartScheduler
from detection.early_predictor import EarlyPredictionManager
from detection.ensemble_detector import EnsembleDetector
from detection.peer_analyzer import PeerAnalyzer
from labeling.llm_analyzer import LLMAnalyzer
from detection.dynamic_threshold_manager import DynamicThresholdManager

# 🔥 Load External Configuration
CONFIG_PATH = "config/config.yaml"
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

RABBITMQ_HOST = CONFIG.get('monitoring', {}).get('rabbitmq', {}).get('host', 'localhost')
OPENAI_KEY = CONFIG.get('openai', {}).get('api_key')
QUEUE_NAME = 'metrics'
INFLUXDB_MOCK_FILE = "outputs/influxdb_mock.json"
CSV_EXPORT_PATH = "outputs/historical_data.csv"

# Thresholds for labeling
CPU_THRESHOLD = 80.0
MEM_THRESHOLD = 80.0

class Labeler:
    def __init__(self):
        # ── 1. RabbitMQ Connection (Robust) ──────────────────────────────────
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=600)
            )
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=QUEUE_NAME)
            print(f"✅ Connected to RabbitMQ at {RABBITMQ_HOST}")
        except Exception as e:
            print(f"❌ Failed to connect to RabbitMQ: {e}")
            print(f"💡 Ensure RabbitMQ service is running.")
            sys.exit(1)
        
        # ── 2. Storage Setup ──────────────────────────────────────────────────
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
            
        # ── 3. Diagnostic Engines Initialization ──────────────────────────────
        self.analyzer = RootCauseAnalyzer(CPU_THRESHOLD, MEM_THRESHOLD)
        self.llm_engine = LLMAnalyzer(api_key=OPENAI_KEY)
        self.scheduler = SmartScheduler()
        self.early_predictor = EarlyPredictionManager()
        self.ensemble = EnsembleDetector()
        self.peer_analyzer = PeerAnalyzer()
        self.llm_engine = LLMAnalyzer()
        self.threshold_manager = DynamicThresholdManager()

    def label_data(self, app_data, res_data):
        """
        Labeling Logic:
        Tasks that run with insufficient resources OR process non-local blocks are labeled as 1;
        otherwise, they are labeled as 0.
        """
        # --- NEW Dynamic Threshold Logic (User Request) ---
        # Fallback to static if history is too small (<5 points)
        cpu_anom = self.threshold_manager.is_anomalous('cpu', res_data['cpu_usage'])
        mem_anom = self.threshold_manager.is_anomalous('memory', res_data['memory_usage'])
        
        # If we don't have enough history, use static defaults
        if self.threshold_manager.get_current_threshold('cpu') is None:
            cpu_anom = res_data['cpu_usage'] > CPU_THRESHOLD
            mem_anom = res_data['memory_usage'] > MEM_THRESHOLD

        insufficient_resources = cpu_anom or mem_anom
        non_local_block = (app_data['is_local'] == 0)
        
        label = 1 if (insufficient_resources or non_local_block) else 0
        return label

    def save_to_influx(self, data):
        """Mock InfluxDB persistence"""
        with open(INFLUXDB_MOCK_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")

    def export_to_csv(self):
        """Export historical data to CSV"""
        if not os.path.exists(INFLUXDB_MOCK_FILE):
            return
        
        records = []
        with open(INFLUXDB_MOCK_FILE, "r") as f:
            for line in f:
                records.append(json.loads(line))
        
        df = pd.DataFrame(records)
        df.to_csv(CSV_EXPORT_PATH, index=False)
        print(f"✅ Exported {len(df)} records to {CSV_EXPORT_PATH}")

    def execute_speculative(self, app_id):
        """Speculative Execution Simulation"""
        import random
        print(f"⚠️ Straggler detected for {app_id} → launching speculative replica...")
        
        # Simulate two parallel executions with different latencies
        duration_original = random.uniform(5.0, 10.0) 
        duration_copy = random.uniform(1.0, 3.0)
        
        print(f"   [Original Task] Duration: {duration_original:.2f}s")
        print(f"   [Speculative Copy] Duration: {duration_copy:.2f}s")
        
        if duration_copy < duration_original:
            print(f"✅ Speculative copy finished FIRST → Result committed. Saved {duration_original - duration_copy:.2f}s")
            return duration_copy
        else:
            print(f"✅ Original task finished first → Result committed.")
            return duration_original

    def callback(self, ch, method, properties, body):
        message = json.loads(body)
        features = message.get('features', {})
        unified_ts = message.get('timestamp', time.time())
        
        # Map new flattened features to old res_data/app_data format for logic compatibility
        res_data = {
            'cpu_usage': features.get('cpu', 0.0),
            'memory_usage': features.get('memory', 0.0),
            'avg_cpu': features.get('avg_cpu', 0.0),
            'avg_mem': features.get('avg_mem', 0.0)
        }
        app_data = {
            'app_id': features.get('app_id', 'idle'),
            'progress': features.get('progress', 0.0),
            'is_local': features.get('is_local', 1),
            'progress_rate': features.get('progress_rate', 0.0),
            'tasks': features.get('tasks', []),
            'log_hints': features.get('log_hints', '')
        }
        
        # --- Update Dynamic Threshold Manager (ICL/Statistical Input) ---
        self.threshold_manager.update('cpu', res_data['cpu_usage'])
        self.threshold_manager.update('memory', res_data['memory_usage'])
        self.threshold_manager.update('disk_read', features.get('disk_read', 0.0) / 1e6)
        self.threshold_manager.update('net_recv', features.get('net_recv', 0.0) / 1e6)
        self.threshold_manager.update('progress_rate', app_data['progress_rate'])

        label = self.label_data(app_data, res_data)
        
        # Update Peer Data & Get Stats
        job_id = "job_default" # Mock job_id for now
        self.peer_analyzer.update(job_id, features.get('duration', 7.0), app_data['progress_rate'])
        peer_stats = self.peer_analyzer.get_peer_stats(job_id)
        
        # --- NEW Rate-Based Slowdown Logic (User Request) ---
        slowdown = self.peer_analyzer.calculate_rate_slowdown(job_id, app_data['progress_rate'])
        
        # 1. Prediction & Confidence
        progress = app_data['progress']
        avg_rate = peer_stats['median_rate'] if peer_stats else 0.05
        model_proba = 0.81 if label == 1 else 0.12 # Mock model output
        is_strag_ens, ensemble_score = self.ensemble.get_ensemble_score(
            model_proba, app_data['progress_rate'], avg_rate, app_data['app_id']
        )
        
        # --- NEW Early Prediction Integration ---
        # If early prediction finds a high-confidence straggler, boost the score
        early_triggered, early_prob = self.early_predictor.check_early_trigger(app_data['app_id'], progress, app_data['progress_rate'], peer_stats)
        if early_triggered and early_prob > 0.7:
             ensemble_score = max(ensemble_score, early_prob)
             label = 1  # FORCE LABEL TO STRAGGLER
             print(f"⚡ Proactive Mitigation Triggered by Early Prediction ({early_prob:.2f} probability)")
        
        # 2. NOISE REDUCTION (User Request)
        # If task is normal (high progress rate or low ensemble score), print one line and exit.
        # CRITICAL: We skip this filter if a Proactive Mitigation was just triggered.
        if (ensemble_score < 0.6 or label == 0) and not early_triggered:
            print(f"✔ Task Normal | Rate: {app_data['progress_rate']:.4f} | CPU: {res_data['cpu_usage']:.1f}%")
            return

        # 3. Intelligent Analysis (STRAGGLER CASE)
        cur_app_id = app_data.get('app_id', 'unknown')
        print(f"\nSTRAGGLER CASE")
        print(f"📩 Job: {cur_app_id}")
        print()

        # 4. Early Prediction (Triggered at 32% if requested)
        self.early_predictor.trigger_threshold = 0.32
        triggered, prob = self.early_predictor.check_early_trigger(cur_app_id, progress, app_data['progress_rate'], peer_stats)
        
        # 5. Core Stats
        print(f"Progress Rate: {app_data['progress_rate']:.4f} vs Avg: {avg_rate:.4f}")
        print(f"Slowdown: {slowdown:.1f}x | Confidence: {ensemble_score:.1f}")

        # 6. Evidence & Cause
        # Analyze Cause (Prints suppressed in analyzer)
        cause = self.analyzer.analyze(app_data, res_data)
        
        cpu_idle = 100 - res_data.get('cpu_usage', 0)
        print(f"📊 Evidence:")
        print(f"CPU Idle: {cpu_idle:.0f}% | Memory: {res_data.get('memory_usage', 0):.0f}% | Swap: {res_data.get('swap_usage', 0):.0f}%")
        
        # Add Dynamic Threshold Context
        dyn_thresh = self.threshold_manager.get_summary()
        if dyn_thresh:
            print(f"📈 Active Dynamic Thresholds: {dyn_thresh}")
        
        network_delay = "High" if "Data Locality" in cause else "Low"
        print(f"Network Delay: {network_delay}")
        
        print(f"🔍 Root Cause: {cause}")
        
        # --- NEW LLM Diagnostic Engine (ICL Implementation) ---
        # Features: [cpu, memory, disk_read, net_recv, progress_rate, is_local]
        icl_features = [
            res_data.get('cpu_usage', 0.0),
            res_data.get('memory_usage', 0.0),
            features.get('disk_read', 0.0) / 1e6, # B to MB
            features.get('net_recv', 0.0) / 1e6,  # B to MB
            app_data.get('progress_rate', 0.0),
            app_data.get('is_local', 1)
        ]
        
        diagnosis = self.llm_engine.analyze(icl_features)
        
        print(f"🧠 LLM Insight:")
        print(f"   Label: {diagnosis['label']}")
        print(f"   Root Cause: {diagnosis['cause']}")
        print(f"   Reason: {diagnosis['reason']}")
        
        metadata = {
            'slowdown': slowdown,
            'confidence': ensemble_score,
            'progress': progress
        }
        
        # 7. Decision & Impact (Handled by Scheduler)
        action, node, final_score = self.scheduler.mitigate(cur_app_id, cause, metadata, diagnosis['reason'])

        # 8. Success Confirmation
        if action != "WAIT":
            if action == "SPECULATE":
                self.execute_speculative(app_data['app_id'])
            
            print(f"✅ Action executed successfully")
        
        print("-" * 30)
        
        processed_record = {
            "timestamp": unified_ts,
            "app_id": app_data['app_id'],
            "cpu_usage": res_data['cpu_usage'],
            "memory_usage": res_data['memory_usage'],
            "is_local": app_data['is_local'],
            "label": label
        }
        
        print(f"📩 Processed: {app_data['app_id']} | Label: {label} | Resources OK: {label==0}")
        if app_data.get('tasks'):
            print(f"📋 Granular Tasks: {len(app_data['tasks'])} active containers")
            for t in app_data['tasks'][:3]: # Show first 3
                print(f"   - {t.get('task_id', 'unknown')} on {t.get('node', 'unknown')} ({t.get('type', 'TASK')})")
        if app_data.get('log_hints'):
            print(f"📝 YARN Log Highlights:\n{app_data['log_hints']}")
        
        self.save_to_influx(processed_record)

    def start(self):
        print("Waiting for monitoring data...")
        self.channel.basic_consume(
            queue=QUEUE_NAME,
            on_message_callback=self.callback,
            auto_ack=True
        )
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("Stopping labeler and exporting data...")
            self.export_to_csv()
            self.connection.close()

if __name__ == "__main__":
    labeler = Labeler()
    labeler.start()
