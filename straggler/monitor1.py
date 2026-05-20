import psutil
import time
import json
import pika
import subprocess
import random
from collections import deque
from prometheus_client import start_http_server, Gauge

# ================= CONFIG =================
INTERVAL = 2.0
RABBITMQ_HOST = 'localhost'
QUEUE_NAME = 'metrics'
PROMETHEUS_PORT = 8000

# ================= PROMETHEUS =================
CPU_GAUGE = Gauge('node_cpu_usage', 'CPU usage percentage')
MEM_GAUGE = Gauge('node_memory_usage', 'Memory usage percentage')
DISK_READ_GAUGE = Gauge('disk_read_bytes', 'Disk read bytes')
NET_RECV_GAUGE = Gauge('network_receive_bytes', 'Network received bytes')

# ================= HADOOP MONITOR =================
class HadoopMonitorAgent:
    def get_active_apps(self):
        try:
            output = subprocess.check_output("yarn application -list", shell=True).decode()
            return [line.split()[0] for line in output.split('\n') if 'application_' in line]
        except:
            return []

    def get_app_progress(self, app_id):
        try:
            output = subprocess.check_output(f"yarn application -status {app_id}", shell=True).decode()
            for line in output.split('\n'):
                if 'Progress :' in line:
                    return float(line.split(':')[1].strip().replace('%', '')) / 100.0
            return 0.0
        except:
            return 0.0

    def parse_logs_for_tasks(self, app_id):
        """Historical task tracking from logs if real-time fails"""
        try:
            output = subprocess.check_output(f"yarn logs -applicationId {app_id} | grep 'Container: container_' | head -n 20", shell=True).decode()
            tasks = []
            for line in output.split('\n'):
                if 'Container: container_' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        tasks.append({"task_id": parts[1], "node": parts[3], "type": "TASK", "progress": 1.0})
            return tasks
        except:
            return []

    def get_containers(self, app_id):
        """Track ALL tasks/containers inside job"""
        try:
            attempt_id = app_id.replace("application_", "appattempt_") + "_000001"
            output = subprocess.check_output(f"yarn container -list {attempt_id}", shell=True).decode()
            containers = []
            for line in output.split('\n'):
                if 'container_' in line:
                    parts = line.split()
                    containers.append(parts[0])
            return containers
        except:
            return []

    def get_container_details(self, container_id):
        """Collect: execution time, node assigned, resource usage"""
        try:
            output = subprocess.check_output(f"yarn container -status {container_id}", shell=True).decode()
            node = "unknown"
            for line in output.split('\n'):
                if 'Node :' in line:
                    node = line.split(':')[1].strip()
            return {"container_id": container_id, "node": node}
        except:
            return {"container_id": container_id, "node": "unknown"}

# ================= MAIN AGENT =================
class MonitoringAgent:
    def __init__(self, mode="SIMULATION"):
        self.mode = mode
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=QUEUE_NAME)

        self.hadoop_monitor = HadoopMonitorAgent() if mode == "REAL" else None

        # Tracking
        self.progress = 0.0
        self.task_start_times = {}

        # Rolling history for dynamic threshold
        self.cpu_hist = deque(maxlen=10)
        self.mem_hist = deque(maxlen=10)

    # ================= RESOURCE METRICS =================
    def get_resource_metrics(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()

        # Update Prometheus
        CPU_GAUGE.set(cpu)
        MEM_GAUGE.set(mem)
        DISK_READ_GAUGE.set(disk.read_bytes)
        NET_RECV_GAUGE.set(net.bytes_recv)

        # Store history
        self.cpu_hist.append(cpu)
        self.mem_hist.append(mem)

        avg_cpu = sum(self.cpu_hist)/len(self.cpu_hist)
        avg_mem = sum(self.mem_hist)/len(self.mem_hist)

        return {
            "cpu": cpu,
            "memory": mem,
            "disk_read": disk.read_bytes,
            "disk_write": disk.write_bytes,
            "net_sent": net.bytes_sent,
            "net_recv": net.bytes_recv,
            "cpu_spike": cpu > avg_cpu * 1.5,
            "mem_pressure": mem > 85,
            "avg_cpu": avg_cpu,
            "avg_mem": avg_mem
        }

    # ================= APPLICATION METRICS =================
    def get_application_metrics(self):
        current_time = time.time()

        if self.mode == "REAL":
            apps = self.hadoop_monitor.get_active_apps()
            if not apps:
                return {
                    "app_id": "idle", 
                    "progress": 0.0, 
                    "duration": 0.0,
                    "progress_rate": 0.0,
                    "is_local": 1, 
                    "tasks": []
                }

            app_id = apps[0]
            progress = self.hadoop_monitor.get_app_progress(app_id)
            
            # Record start time
            if app_id not in self.task_start_times:
                self.task_start_times[app_id] = current_time
            
            # Track Granular Tasks (Containers)
            containers = self.hadoop_monitor.get_containers(app_id)
            tasks = []
            if containers:
                for cid in containers:
                    details = self.hadoop_monitor.get_container_details(cid)
                    tasks.append({
                        "task_id": cid,
                        "node": details['node'],
                        "progress": progress,
                        "type": "AM" if "_000001" in str(cid) else "TASK"
                    })
            
            # If no real-time containers, try log parsing (especially if job finished)
            if not tasks:
                tasks = self.hadoop_monitor.parse_logs_for_tasks(app_id)

            # Mandatory Log highlights for verification
            logs = ""
            try:
                # Catch some interesting task events
                logs = subprocess.check_output(f"yarn logs -applicationId {app_id} | grep -iE 'Task|Container' | tail -n 5", shell=True).decode()
            except: pass

            return {
                "app_id": app_id,
                "progress": progress,
                "duration": current_time - self.task_start_times[app_id],
                "progress_rate": progress / (current_time - self.task_start_times[app_id]) if (current_time - self.task_start_times[app_id]) > 0 else 0,
                "tasks": tasks,
                "is_local": 1,
                "log_hints": logs
            }

        else:
            # Multi-Task Simulation
            sim_ids = ["application_12345_0001", "application_12345_0002", "application_12345_0003"]
            
            # Initialize states if needed
            if not hasattr(self, 'sim_progress'):
                self.sim_progress = {sid: 0.0 for sid in sim_ids}
                self.sim_task_index = 0
                # CRITICAL: Reset start times for all simulation tasks to NOW
                for sid in sim_ids:
                    self.task_start_times[sid] = current_time

            # Select current task to report
            app_id = sim_ids[self.sim_task_index % len(sim_ids)]
            self.sim_task_index += 1
            
            # Update progress based on task "personality" (Normal tasks are now faster)
            if app_id.endswith("0001"): # Normal Fast
                self.sim_progress[app_id] += random.uniform(0.06, 0.09)
            elif app_id.endswith("0002"): # Normal Moderate
                self.sim_progress[app_id] += random.uniform(0.04, 0.06)
            else: # Potential Straggler (Slow/Stalling)
                self.sim_progress[app_id] += random.uniform(0.005, 0.02)
                
            if self.sim_progress[app_id] > 1.0:
                self.sim_progress[app_id] = 0.0
                self.task_start_times[app_id] = current_time
            
            if app_id not in self.task_start_times:
                self.task_start_times[app_id] = current_time
            
            progress = self.sim_progress[app_id]
            duration = current_time - self.task_start_times[app_id]
            progress_rate = progress / duration if duration > 0 else 0
            is_local = 1 if random.random() > 0.1 else 0

            return {
                "app_id": app_id,
                "progress": progress,
                "duration": duration,
                "progress_rate": progress_rate,
                "is_local": is_local,
                "tasks": [{"task_id": f"sim_task_{app_id[-4:]}", "node": "node_0", "progress": progress, "type": "TASK"}]
            }

    # ================= MAIN LOOP =================
    def run(self):
        print(f"🚀 Monitoring Agent started [{self.mode}]")

        try:
            start_http_server(PROMETHEUS_PORT)
        except:
            pass

        try:
            while True:
                res = self.get_resource_metrics()
                app = self.get_application_metrics()

                unified = {
                    "timestamp": time.time(),
                    "features": {
                        **res,
                        **app
                    }
                }

                self.channel.basic_publish(
                    exchange='',
                    routing_key=QUEUE_NAME,
                    body=json.dumps(unified)
                )

                print(f"📡 {app['app_id']} | Prog: {app['progress']:.2f} | Rate: {app['progress_rate']:.4f}")

                time.sleep(INTERVAL)

        except KeyboardInterrupt:
            print("Stopping...")
            self.connection.close()


# ================= ENTRY =================
if __name__ == "__main__":
    import sys
    mode = "REAL" if "--real" in sys.argv else "SIMULATION"
    agent = MonitoringAgent(mode=mode)
    agent.run()