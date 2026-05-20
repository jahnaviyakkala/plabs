import psutil
import time
import json
import pika
import subprocess

# 🔥 CHANGE THIS if using another laptop
RABBITMQ_IP = "localhost"

# Connect to RabbitMQ
connection = pika.BlockingConnection(
    pika.ConnectionParameters(RABBITMQ_IP)
)
channel = connection.channel()
channel.queue_declare(queue='metrics')

while True:
    try:
        # 🔥 Remove old output from HDFS
        subprocess.run([
            "hdfs", "dfs", "-rm", "-r", "-f", "/user/output"
        ])

        # 🔥 Start time
        start_time = time.time()

        # 🔥 Run Hadoop WordCount job
        subprocess.run([
            "hadoop",
            "jar",
            "share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.6.jar",
            "wordcount",
            "/user/input",
            "/user/output"
        ])

        # 🔥 End time
        end_time = time.time()

        execution_time = end_time - start_time

        # 🔥 Collect REAL system metrics
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        data = {
            "cpu": cpu,
            "memory": memory,
            "execution_time": execution_time
        }

        # 🔥 Send to RabbitMQ
        channel.basic_publish(
            exchange='',
            routing_key='metrics',
            body=json.dumps(data)
        )

        print("Sent:", data)

        time.sleep(2)

    except Exception as e:
        print("Error:", e)
        time.sleep(5)
