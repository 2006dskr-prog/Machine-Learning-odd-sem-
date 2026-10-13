import numpy as np


class BaseScheduler:
    def schedule(self, env, capacity: int):
        raise NotImplementedError


class RandomScheduler(BaseScheduler):
    def schedule(self, env, capacity: int):
        requests = [v.id for v in env.vehicles if len(v.message_queue) > 0]
        np.random.shuffle(requests)
        return requests[:capacity]


class RoundRobinScheduler(BaseScheduler):
    def __init__(self):
        self.last_scheduled = -1

    def schedule(self, env, capacity: int):
        requests = [v.id for v in env.vehicles if len(v.message_queue) > 0]
        if not requests:
            return []
        num_vehicles = env.num_vehicles
        sorted_requests = sorted(
            requests,
            key=lambda x: (x - self.last_scheduled - 1) % num_vehicles
        )
        granted = sorted_requests[:capacity]
        if granted:
            self.last_scheduled = granted[-1]
        return granted


class AoIScheduler(BaseScheduler):
    def schedule(self, env, capacity: int):
        requests = [v.id for v in env.vehicles if len(v.message_queue) > 0]
        if not requests:
            return []
        sorted_requests = sorted(
            requests,
            key=lambda x: env.vehicles[x].message_queue[0].get_age(env.current_time),
            reverse=True
        )
        return sorted_requests[:capacity]


class SemanticScheduler(BaseScheduler):
    def schedule(self, env, capacity: int):
        requests = [v.id for v in env.vehicles if len(v.message_queue) > 0]
        if not requests:
            return []
        sorted_requests = sorted(
            requests,
            key=lambda x: max(m.importance for m in env.vehicles[x].message_queue),
            reverse=True
        )
        return sorted_requests[:capacity]