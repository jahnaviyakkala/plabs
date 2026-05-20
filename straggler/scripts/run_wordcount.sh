#!/bin/bash

# Configuration
HADOOP_EXAMPLES="/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.6.jar"
INPUT_DIR="/user/swathi_hadoop/input"
OUTPUT_DIR="/user/swathi_hadoop/output_$(date +%s)"

echo "🚀 Starting WordCount Job..."
hadoop jar $HADOOP_EXAMPLES wordcount $INPUT_DIR $OUTPUT_DIR &

echo "📡 Starting Real-Time Monitoring Agent..."
python3 monitor1.py --real
