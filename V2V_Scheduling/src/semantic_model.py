class SemanticModel:
    def __init__(self, safety_p: float, traffic_p: float, routine_p: float):
        self.importance_map = {
            "safety": float(safety_p),
            "traffic": float(traffic_p),
            "routine": float(routine_p),
        }

    def get_importance(self, class_type: str) -> float:
        return self.importance_map.get(class_type, 0.0)

    def calculate_utility(self, messages_delivered) -> float:
        return sum(m.importance for m in messages_delivered)