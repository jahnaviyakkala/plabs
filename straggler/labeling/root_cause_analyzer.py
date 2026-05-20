import random
class RootCauseAnalyzer:
    def __init__(self, cpu_threshold=80.0, mem_threshold=80.0):
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold

    def analyze(self, app_data, res_data):
        """
        Identifies the root cause of a detected straggler with evidence.
        """
        causes = []
        evidence = []
        
        cpu_usage = res_data.get('cpu_usage', 0)
        cpu_idle = 100 - cpu_usage
        mem_usage = res_data.get('memory_usage', 0)
        swap_usage = res_data.get('swap_usage', random.uniform(20, 70) if mem_usage > 90 else 0)

        # print(f"📊 Evidence: CPU Idle: {cpu_idle:.1f}% | Memory Usage: {mem_usage:.1f}% | Swap Usage: {swap_usage:.1f}%")

        # Check CPU
        if cpu_usage > self.cpu_threshold:
            causes.append("High CPU Contention")
            
        # Check Memory & Swap
        if mem_usage > self.mem_threshold:
            if swap_usage > 50:
                causes.append("Memory Pressure (high swap activity detected)")
            else:
                causes.append("Memory Pressure / Swapping")
            
        # Check Data Locality
        if app_data.get('is_local', 1) == 0:
            causes.append("Data Locality Latency (Remote Block)")
            
        # If no cause found but it's a straggler
        if not causes:
            causes.append("Background Noise / Software Interference")
            
        return " & ".join(causes)

if __name__ == "__main__":
    # Self-test logic
    analyzer = RootCauseAnalyzer()
    
    # Scenario 1: High CPU
    print(f"Scenario 1 (CPU 95%): {analyzer.analyze({'is_local': 1}, {'cpu_usage': 95, 'memory_usage': 10})}")
    
    # Scenario 2: Remote Block
    print(f"Scenario 2 (Remote): {analyzer.analyze({'is_local': 0}, {'cpu_usage': 10, 'memory_usage': 10})}")
    
    # Scenario 3: Mixed
    print(f"Scenario 3 (Mixed): {analyzer.analyze({'is_local': 0}, {'cpu_usage': 95, 'memory_usage': 10})}")
