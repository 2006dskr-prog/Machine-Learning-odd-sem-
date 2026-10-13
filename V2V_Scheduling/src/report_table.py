"""Prints a Markdown-ready table for the final report."""

import yaml
import numpy as np
import torch

from src.environment import V2VEnv
from src.mappo import MAPPO
from src.scheduler import (
    RandomScheduler, RoundRobinScheduler, AoIScheduler, SemanticScheduler
)


def eval_rule(env, sched, seeds):
    aoi, util, fair, pdr = [], [], [], []
    for s in seeds:
        env.reset(seed=s)
        d = False
        info = {}
        while not d:
            granted = sched.schedule(env, env.capacity)
            a = np.zeros(env.num_vehicles, dtype=int)
            for vid in granted:
                if len(env.vehicles[vid].message_queue) > 0:
                    a[vid] = 2
            _, _, t, tr, info = env.step(a)
            d = t or tr
        aoi.append(info['avg_aoi']); util.append(info['semantic_utility'])
        fair.append(info['fairness']); pdr.append(info['pdr'])
    return [np.mean(aoi), np.mean(util), np.mean(fair), np.mean(pdr)]


def eval_mappo(config, ckpt, seeds):
    env = V2VEnv(config)
    m = MAPPO(6, config['simulation']['num_vehicles'] * 6 +
              config['simulation']['num_vehicles'] ** 2, 3, config)
    c = torch.load(ckpt, map_location=m.device, weights_only=False)
    m.actor.load_state_dict(c['actor']); m.critic.load_state_dict(c['critic'])

    aoi, util, fair, pdr = [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        d = False
        info = {}
        while not d:
            with torch.no_grad():
                a, _, _ = m.select_action(obs, env._get_global_state(), deterministic=True)
            obs, _, t, tr, info = env.step(a)
            d = t or tr
        aoi.append(info['avg_aoi']); util.append(info['semantic_utility'])
        fair.append(info['fairness']); pdr.append(info['pdr'])
    return [np.mean(aoi), np.mean(util), np.mean(fair), np.mean(pdr)]


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    seeds = list(range(60000, 60020))

    env = V2VEnv(cfg)
    rows = {}
    for name, sched in [("Random", RandomScheduler()),
                        ("Round-Robin", RoundRobinScheduler()),
                        ("AoI-based", AoIScheduler()),
                        ("Semantic-priority", SemanticScheduler())]:
        rows[name] = eval_rule(env, sched, seeds)
    rows["MAPPO (proposed)"] = eval_mappo(cfg, "outputs/checkpoints/best_mappo.pth", seeds)

    print("\n| Method | AoI ↓ | Utility ↑ | Fairness ↑ | PDR ↑ |")
    print("|---|---|---|---|---|")
    for name, (aoi, util, fair, pdr) in rows.items():
        print(f"| {name} | {aoi:.2f} | {util:.2f} | {fair:.3f} | {pdr:.3f} |")


if __name__ == "__main__":
    main()