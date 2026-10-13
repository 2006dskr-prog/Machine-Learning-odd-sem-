import yaml
import numpy as np
import torch

from src.environment import V2VEnv
from src.mappo import MAPPO
from src.scheduler import (
    RandomScheduler, RoundRobinScheduler, AoIScheduler, SemanticScheduler
)


def run_rule(env, scheduler, seeds):
    aoi, w_aoi, pdr, util, fair = [], [], [], [], []
    for seed in seeds:
        env.reset(seed=seed)
        done = False
        info = {}
        while not done:
            granted = scheduler.schedule(env, env.capacity)
            actions = np.zeros(env.num_vehicles, dtype=int)
            for vid in granted:
                if len(env.vehicles[vid].message_queue) > 0:
                    actions[vid] = 2
            _, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        aoi.append(info['avg_aoi'])
        w_aoi.append(info['weighted_aoi'])
        pdr.append(info['pdr'])
        util.append(info['semantic_utility'])
        fair.append(info['fairness'])
    return dict(
        aoi=np.mean(aoi),
        aoi_std=np.std(aoi),
        weighted_aoi=np.mean(w_aoi),
        pdr=np.mean(pdr),
        util=np.mean(util),
        fair=np.mean(fair),
    )


def run_mappo(config, ckpt_path, seeds):
    env = V2VEnv(config)
    obs_dim = 6
    gs_dim = config['simulation']['num_vehicles'] * 6 + config['simulation']['num_vehicles'] ** 2
    mappo = MAPPO(obs_dim, gs_dim, 3, config)
    ckpt = torch.load(ckpt_path, map_location=mappo.device, weights_only=False)
    mappo.actor.load_state_dict(ckpt['actor'])
    mappo.critic.load_state_dict(ckpt['critic'])

    aoi, w_aoi, pdr, util, fair = [], [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                actions, _, _ = mappo.select_action(
                    obs, env._get_global_state(), deterministic=True
                )
            obs, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        aoi.append(info['avg_aoi'])
        w_aoi.append(info['weighted_aoi'])
        pdr.append(info['pdr'])
        util.append(info['semantic_utility'])
        fair.append(info['fairness'])
    return dict(
        aoi=np.mean(aoi),
        aoi_std=np.std(aoi),
        weighted_aoi=np.mean(w_aoi),
        pdr=np.mean(pdr),
        util=np.mean(util),
        fair=np.mean(fair),
    )


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    test_seeds = list(range(50_000, 50_020))

    env = V2VEnv(config)
    results = {}

    for name, sched in [
        ("Random", RandomScheduler()),
        ("Round-Robin", RoundRobinScheduler()),
        ("AoI-based", AoIScheduler()),
        ("Semantic-priority", SemanticScheduler()),
    ]:
        print(f"Evaluating {name}...")
        results[name] = run_rule(env, sched, test_seeds)

    print("Evaluating MAPPO...")
    results["MAPPO"] = run_mappo(config, "outputs/checkpoints/best_mappo.pth", test_seeds)

    print()
    header = (f"{'Method':<22} | {'AoI':>6} | {'+/-':>5} | {'W-AoI':>7} | "
              f"{'PDR':>5} | {'Utility':>8} | {'Fairness':>9}")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<22} | {r['aoi']:>6.2f} | {r['aoi_std']:>5.2f} | "
              f"{r['weighted_aoi']:>7.2f} | {r['pdr']:>5.3f} | "
              f"{r['util']:>8.2f} | {r['fair']:>9.3f}")


if __name__ == "__main__":
    main()