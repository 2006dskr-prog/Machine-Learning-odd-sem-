import numpy as np


class AoITracker:
    def __init__(self, num_vehicles: int):
        self.num_vehicles = num_vehicles
        self.aoi_matrix = np.zeros((num_vehicles, num_vehicles), dtype=np.float32)
        self.last_update_time = np.zeros((num_vehicles, num_vehicles), dtype=np.float32)

    def step(self, dt: float):
        self.aoi_matrix += dt

    def update_aoi(self, source_id: int, receiver_id: int,
                   current_time: float, generation_time: float):
        self.aoi_matrix[source_id, receiver_id] = current_time - generation_time
        self.last_update_time[source_id, receiver_id] = generation_time

    def get_average_aoi(self) -> float:
        return float(np.mean(self.aoi_matrix))

    def get_semantic_weighted_aoi(self, semantic_weights: np.ndarray) -> float:
        weighted_sum = np.sum(self.aoi_matrix * semantic_weights[:, np.newaxis])
        total_weight = np.sum(semantic_weights) * self.num_vehicles
        return float(weighted_sum / (total_weight + 1e-8))