import os
import yaml
import numpy as np
import torch

from src.environment import V2VEnv
from src.mappo import MAPPO
from src.scheduler import (
    RandomScheduler, RoundRobinScheduler, AoIScheduler, SemanticScheduler
)


def evaluate_rule_based(env, scheduler, episodes=10):
    aoi, pdr, utility, fairness = [], [], [], []
    for ep in range(episodes):
        env.reset(seed=ep)
        done = False
        info = {}
        while not done:
            granted = scheduler.schedule(env, env.capacity)
            action_array = np.zeros(env.num_vehicles, dtype=int)
            for vid in granted:
                if len(env.vehicles[vid].message_queue) > 0:
                    action_array[vid] = 2
            _, _, terminated, truncated, info = env.step(action_array)
            done = terminated or truncated
        aoi.append(info['avg_aoi'])
        pdr.append(info['pdr'])
        utility.append(info['semantic_utility'])
        fairness.append(info['fairness'])
    return {
        'aoi': float(np.mean(aoi)),
        'pdr': float(np.mean(pdr)),
        'utility': float(np.mean(utility)),
        'fairness': float(np.mean(fairness)),
    }


def evaluate_mappo(config, checkpoint_path="outputs/checkpoints/best_mappo.pth", episodes=10):
    env = V2VEnv(config)
    obs_dim = 5
    gs_dim = (config['simulation']['num_vehicles'] * 5
              + config['simulation']['num_vehicles'] ** 2)
    mappo = MAPPO(obs_dim, gs_dim, 3, config)

    if not os.path.exists(checkpoint_path):
        print(f"[warn] checkpoint not found: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location=mappo.device, weights_only=False)
    mappo.actor.load_state_dict(ckpt['actor'])
    mappo.critic.load_state_dict(ckpt['critic'])

    aoi, pdr, utility, fairness = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        info = {}
        while not done:
            actions, _, _ = mappo.select_action(obs, env._get_global_state(), deterministic=True)
            obs, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        aoi.append(info['avg_aoi'])
        pdr.append(info['pdr'])
        utility.append(info['semantic_utility'])
        fairness.append(info['fairness'])
    return {
        'aoi': float(np.mean(aoi)),
        'pdr': float(np.mean(pdr)),
        'utility': float(np.mean(utility)),
        'fairness': float(np.mean(fairness)),
    }


def main():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = V2VEnv(config)

    schedulers = {
        "Random": RandomScheduler(),
        "Round-Robin": RoundRobinScheduler(),
        "AoI-based": AoIScheduler(),
        "Semantic-priority": SemanticScheduler(),
    }

    results = {}
    for name, sched in schedulers.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_rule_based(env, sched)

    print("\nEvaluating MAPPO...")
    mappo_results = evaluate_mappo(config)
    if mappo_results is not None:
        results["Proposed MAPPO"] = mappo_results

    print("\n--- Baseline Results ---")
    header = f"{'Method':<20} | {'Avg AoI':>8} | {'PDR':>6} | {'Utility':>9} | {'Fairness':>9}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        print(f"{name:<20} | {res['aoi']:>8.2f} | {res['pdr']:>6.3f} | "
              f"{res['utility']:>9.2f} | {res['fairness']:>9.3f}")


if __name__ == "__main__":
    main()