class DuplicateManager:
    def __init__(self, replica_cap=2, cluster_congested=False):
        self.replica_cap = replica_cap
        self.active_replicas = {} # {task_id: [replica_ids]}
        self.replica_progress = {} # {replica_id: progress}
        self.cluster_congested = cluster_congested

    def should_launch_replica(self, task_id):
        """Manage replica caps and backoff if cluster is congested."""
        if self.cluster_congested:
            print(f"⚠️  DuplicateManager: Cluster congested. Backing off for {task_id}.")
            return False
            
        current_count = len(self.active_replicas.get(task_id, []))
        if current_count >= self.replica_cap:
            print(f"⏹️  DuplicateManager: Replica cap reached for {task_id} ({self.replica_cap}).")
            return False
        return True

    def register_replica(self, task_id, replica_id):
        if task_id not in self.active_replicas:
            self.active_replicas[task_id] = []
        self.active_replicas[task_id].append(replica_id)
        self.replica_progress[replica_id] = 0

    def update_and_check_cancellation(self, task_id, replica_id, progress):
        """
        Kill slower copy when faster one progresses beyond threshold.
        """
        self.replica_progress[replica_id] = progress
        
        # Check if any sister replica has already progressed significantly
        sister_replicas = self.active_replicas.get(task_id, [])
        for sister in sister_replicas:
            if sister != replica_id:
                sister_prog = self.replica_progress.get(sister, 0)
                # If sister is 30% further ahead, consider killing this one
                if sister_prog > progress + 30:
                    print(f"✂️  DuplicateManager: Killing slow replica {replica_id} (Sister {sister} at {sister_prog}%).")
                    return True # Should cancel
        return False

if __name__ == "__main__":
    manager = DuplicateManager(replica_cap=2)
    # Scenario: Normal launch
    if manager.should_launch_replica("task_1"):
        manager.register_replica("task_1", "rep_1a")
        manager.register_replica("task_1", "rep_1b")
    
    # Check cancellation
    manager.update_and_check_cancellation("task_1", "rep_1a", 80)
    should_kill = manager.update_and_check_cancellation("task_1", "rep_1b", 20)
    print(f"Should kill rep_1b? {should_kill}")
