import os
import json
import numpy as np

# Optional cloud dependency
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class LLMAnalyzer:
    def __init__(self, api_key=None):
        # Priority: 1. Passed Arg | 2. Env Var
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.offline_mode = False
        
        # Priority 1: Key provided | Priority 2: Key in ENV
        if not self.api_key:
            print("⚠️  OPENAI_API_KEY not found. Running in Local Semantic Mode.")
            self.offline_mode = True
            self.client = None
        elif not HAS_OPENAI:
            print("⚠️  'openai' library not detected despite API key presence. Falling back to Local Mode.")
            self.offline_mode = True
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.offline_mode = False
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI client: {e}")
                self.offline_mode = True
                self.client = None
        
        # Gold Examples for In-Context Learning (ICL) and Local Fallback
        # Features: [cpu, memory, disk_read, net_recv, progress_rate, is_local]
        self.examples = [
            {
                "features": [95.0, 40.0, 10.0, 5.0, 0.05, 1],
                "label": "Straggler",
                "root_cause": "High CPU Contention",
                "reason": "CPU usage is near saturation (95%) while memory and I/O are low. The extremely low progress rate confirms the task is starved of cycles."
            },
            {
                "features": [20.0, 92.0, 5.0, 2.0, 0.08, 1],
                "label": "Straggler",
                "root_cause": "Memory Pressure / Swapping",
                "reason": "Memory usage is above 90% with low CPU. High swap activity is likely stalling execution despite available CPU cycles."
            },
            {
                "features": [15.0, 20.0, 2.0, 85.0, 0.12, 0],
                "label": "Straggler",
                "root_cause": "Data Locality Latency",
                "reason": "Task is reading remote blocks (is_local=0) and net_recv is high. Network I/O bottleneck is preventing the task from progressing at peer speed."
            },
            {
                "features": [30.0, 35.0, 90.0, 10.0, 0.15, 1],
                "label": "Straggler",
                "root_cause": "Disk I/O Bottleneck",
                "reason": "Disk read throughput is extremely high (90MB/s) while other resources are healthy. The task is stalled on local disk wait."
            },
            {
                "features": [25.0, 30.0, 15.0, 10.0, 0.45, 1],
                "label": "Normal",
                "root_cause": "None",
                "reason": "All resource metrics are within healthy ranges and the progress rate is consistent with cluster averages."
            }
        ]

    def dynamic_retrieval(self, current_features, top_k=3):
        """Similarity-based retrieval (k-Nearest Neighbors)."""
        current_features = np.array(current_features)
        distances = []
        for ex in self.examples:
            distances.append(np.linalg.norm(current_features - np.array(ex["features"])))
            
        indices = np.argsort(distances)[:top_k]
        return [self.examples[i] for i in indices]

    def build_prompt(self, features):
        """Constructs the Few-Shot prompt for GPT-4o."""
        retrieved_examples = self.dynamic_retrieval(features)
        prompt = "You are a Hadoop Performance Expert. Analyze these metrics:\n"
        for i, ex in enumerate(retrieved_examples):
            prompt += f"Example {i+1}: Input {ex['features']} -> Label: {ex['label']}, Cause: {ex['root_cause']}, Reason: {ex['reason']}\n"
        prompt += f"Current Input: {features}\n"
        prompt += "Return strictly: Label: ..., Cause: ..., Reason: ..."
        return prompt

    def call_llm(self, prompt):
        """Calls OpenAI GPT-4o."""
        if self.offline_mode:
            return None
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=10.0 # Add timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API ERROR: {str(e)}"

    def analyze(self, features):
        """
        Master diagnostic function with Local Fallback support.
        """
        # --- Local Mode (Initial Check or Manual Force) ---
        if self.offline_mode:
            closest = self.dynamic_retrieval(features, top_k=1)[0]
            return {
                "label": closest["label"],
                "cause": f"[Local Reasoning] {closest['root_cause']}",
                "reason": closest["reason"]
            }
        
        # --- Cloud Mode (GPT-4o) ---
        prompt = self.build_prompt(features)
        raw_response = self.call_llm(prompt)
        
        if raw_response and "API ERROR" not in raw_response:
            res = self.parse_output(raw_response)
            res["cause"] = f"[Cloud AI] {res.get('cause', 'Unknown')}"
            return res
            
        # --- Fail-over to Local on API Error ---
        closest = self.dynamic_retrieval(features, top_k=1)[0]
        return {
            "label": closest["label"],
            "cause": f"[Local Fallback] {closest['root_cause']}",
            "reason": closest["reason"]
        }

    def parse_output(self, response):
        """Parses structured LLM response."""
        res = {"label": "Unknown", "cause": "Unknown", "reason": ""}
        for line in response.split('\n'):
            if ":" in line:
                k, v = line.split(":", 1)
                res[k.strip().lower()] = v.strip()
        return res

if __name__ == "__main__":
    # Example usage / Sanity check
    analyzer = LLMAnalyzer()
    test_features = [90.0, 30.0, 5.0, 2.0, 0.04, 1]
    diagnosis = analyzer.analyze(test_features)
    print(json.dumps(diagnosis, indent=2))
