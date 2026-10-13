"""
Evaluate the main MAPPO checkpoint and all ablation variants
on identical held-out seeds, and print a Markdown-ready table.
"""
import os
import yaml
import numpy as np
import torch

from src.environment import V2VEnv
from src.mappo import MAPPO


CHECKPOINTS = {
    "Full model (semantic + AoI)":      "outputs/checkpoints/best_mappo.pth",
    "Ablation: no semantic reward":     "outputs/checkpoints/ablation_no_semantic.pth",
    "Ablation: no AoI reward":          "outputs/checkpoints/ablation_no_aoi.pth",
    "Ablation: retrained (full)":       "outputs/checkpoints/ablation_full.pth",
}


def evaluate(config, ckpt_path, seeds):
    if not os.path.exists(ckpt_path):
        return None

    env = V2VEnv(config)
    obs_dim = 6
    gs_dim = (config['simulation']['num_vehicles'] * 6
              + config['simulation']['num_vehicles'] ** 2)
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
        aoi=np.mean(aoi), aoi_std=np.std(aoi),
        weighted_aoi=np.mean(w_aoi),
        pdr=np.mean(pdr),
        util=np.mean(util),
        fair=np.mean(fair),
    )


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # Held-out seeds not used in training or earlier eval
    test_seeds = list(range(70000, 70020))

    print("Evaluating all checkpoints on held-out seeds...\n")

    rows = {}
    for name, ckpt in CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} not found")
            continue
        print(f"  -> {name}")
        rows[name] = evaluate(config, ckpt, test_seeds)

    print("\n--- Ablation Results ---\n")
    header = (f"{'Variant':<35} | {'AoI ↓':>6} | {'W-AoI ↓':>8} | "
              f"{'PDR ↑':>6} | {'Utility ↑':>10} | {'Fair ↑':>7}")
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        print(f"{name:<35} | {r['aoi']:>6.2f} | {r['weighted_aoi']:>8.2f} | "
              f"{r['pdr']:>6.3f} | {r['util']:>10.2f} | {r['fair']:>7.3f}")

    # Markdown block for report
    print("\n--- Markdown (paste into report) ---\n")
    print("| Variant | AoI ↓ | Weighted AoI ↓ | PDR ↑ | Utility ↑ | Fairness ↑ |")
    print("|---|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['aoi']:.2f} | {r['weighted_aoi']:.2f} | "
              f"{r['pdr']:.3f} | {r['util']:.2f} | {r['fair']:.3f} |")


if __name__ == "__main__":
    main()