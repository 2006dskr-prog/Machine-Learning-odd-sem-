import numpy as np


class MetricsTracker:
    def __init__(self, num_vehicles=6):
        self.num_vehicles = num_vehicles
        self.total_attempted = 0
        self.total_delivered = 0
        self.semantic_utility = 0.0
        self.aoi_history = []
        self.weighted_aoi_history = []
        self.delivery_counts = None

        # Per-source importance tracking (for semantic-weighted AoI)
        self.source_importance_sum = np.zeros(num_vehicles, dtype=np.float32)
        self.source_message_count = np.zeros(num_vehicles, dtype=np.float32)

    def record_transmission(self, attempted: int, delivered: int, utility: float):
        self.total_attempted += attempted
        self.total_delivered += delivered
        self.semantic_utility += utility

    def record_aoi(self, avg_aoi: float):
        self.aoi_history.append(float(avg_aoi))

    def record_weighted_aoi(self, weighted_aoi: float):
        self.weighted_aoi_history.append(float(weighted_aoi))

    def record_deliveries(self, delivered_per_agent: np.ndarray):
        if self.delivery_counts is None:
            self.delivery_counts = delivered_per_agent.astype(np.float32).copy()
        else:
            self.delivery_counts += delivered_per_agent

    def record_message(self, source_id: int, importance: float):
        self.source_importance_sum[source_id] += float(importance)
        self.source_message_count[source_id] += 1.0

    def get_source_weights(self) -> np.ndarray:
        """Average importance per source. Falls back to 1.0 if no messages."""
        counts = np.maximum(self.source_message_count, 1.0)
        weights = self.source_importance_sum / counts
        weights[weights == 0] = 1.0
        return weights

    def get_pdr(self) -> float:
        return float(self.total_delivered / max(1, self.total_attempted))

    def get_avg_aoi(self) -> float:
        return float(np.mean(self.aoi_history)) if self.aoi_history else 0.0

    def get_avg_weighted_aoi(self) -> float:
        return float(np.mean(self.weighted_aoi_history)) if self.weighted_aoi_history else 0.0

    def get_semantic_weighted_aoi(self, aoi_matrix: np.ndarray) -> float:
        """Weight AoI[i, j] by source i's average importance across all receivers j."""
        weights = self.get_source_weights()
        weighted = aoi_matrix * weights[:, np.newaxis]
        denom = weights.sum() * aoi_matrix.shape[1]
        return float(weighted.sum() / (denom + 1e-8))

    def get_fairness(self) -> float:
        if self.delivery_counts is None:
            return 0.0
        n = len(self.delivery_counts)
        if n == 0 or np.sum(self.delivery_counts) == 0:
            return 0.0
        numerator = float(np.sum(self.delivery_counts) ** 2)
        denominator = float(n * np.sum(self.delivery_counts ** 2))
        return numerator / max(1e-8, denominator)