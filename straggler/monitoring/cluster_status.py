import random

class ClusterMonitor:
    def __init__(self, node_count=3):
        # Mock cluster state
        self.nodes = {f"node_{i}": self._generate_mock_status() for i in range(node_count)}

    def _generate_mock_status(self):
        return {
            "cpu_idle": random.uniform(50, 95),
            "mem_free": random.uniform(50, 95),
            "network_latency": random.uniform(1, 10), # ms
            "rack_id": random.choice(["rack_a", "rack_b"]),
        }

    def get_node_status(self, node_id):
        return self.nodes.get(node_id)

    def get_all_nodes(self):
        return self.nodes

class NodeSelector:
    def __init__(self, cluster_monitor):
        self.monitor = cluster_monitor

    def select_best_node(self, target_block_node=None, current_rack=None):
        """
        score(node) = low_cpu + low_memory + data_locality + network_proximity
        """
        all_nodes = self.monitor.get_all_nodes()
        scores = {}

        for node_id, status in all_nodes.items():
            # Normalize scores (Higher is better)
            cpu_score = status['cpu_idle'] / 100.0
            mem_score = status['mem_free'] / 100.0
            
            # Data Locality Bonus
            locality_score = 1.0 if (node_id == target_block_node) else 0.0
            
            # Network Proximity (simplified: same rack bonus)
            network_score = 0.5 if (status['rack_id'] == current_rack) else 0.0
            
            # Final gravity score
            total_score = cpu_score + mem_score + locality_score + network_score
            scores[node_id] = total_score
            
        best_node = max(scores, key=scores.get)
        return best_node, scores[best_node]

if __name__ == "__main__":
    monitor = ClusterMonitor()
    selector = NodeSelector(monitor)
    best, score = selector.select_best_node(target_block_node="node_1")
    print(f"Cluster Status: {monitor.get_all_nodes()}")
    print(f"Selected Best Node: {best} with score {score:.2f}")
