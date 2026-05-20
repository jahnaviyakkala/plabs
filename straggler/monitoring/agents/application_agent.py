class ApplicationAgent:
    def __init__(self):
        pass

    def collect_metrics(self):
        # In a real Hadoop cluster, this would query JMX or JobHistoryServer
        return {"jobId": "job_123", "mapProgress": 0.5}
