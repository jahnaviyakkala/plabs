class ResourceAgent:
    def __init__(self):
        pass

    def collect_resource_usage(self):
        # In a real cluster, this would use psutil or /proc
        return {"cpu_usage": 45.2, "memory_usage": 1024}
