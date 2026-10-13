import yaml
from src.environment import V2VEnv
from src.scheduler import (
    RandomScheduler, RoundRobinScheduler, AoIScheduler, SemanticScheduler
)


def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_capacity_respected():
    config = load_config()
    env = V2VEnv(config)

    for sched in [RandomScheduler(), RoundRobinScheduler(),
                  AoIScheduler(), SemanticScheduler()]:
        env.reset(seed=0)
        for _ in range(50):
            granted = sched.schedule(env, env.capacity)
            assert len(granted) <= env.capacity
            actions = [2 if i in granted else 0 for i in range(env.num_vehicles)]
            env.step(actions)
    print("Scheduler capacity test passed.")


if __name__ == "__main__":
    test_capacity_respected()