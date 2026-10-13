import yaml
import numpy as np
from src.environment import V2VEnv


def load_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_environment_runs():
    config = load_config()
    env = V2VEnv(config)

    obs, info = env.reset(seed=42)
    assert obs.shape == (config['simulation']['num_vehicles'], 5)

    total_reward = 0.0
    for step in range(100):
        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(np.sum(rewards))
        assert info['attempted'] <= config['simulation']['transmission_resources']
        assert rewards.shape == (config['simulation']['num_vehicles'],)

    print(f"Environment ran 100 steps successfully. Total Reward: {total_reward:.2f}")
    print(f"Final Metrics -> Avg AoI: {info['avg_aoi']:.2f}, "
          f"PDR: {info['pdr']:.2f}, Utility: {info['semantic_utility']:.2f}")


if __name__ == "__main__":
    test_environment_runs()