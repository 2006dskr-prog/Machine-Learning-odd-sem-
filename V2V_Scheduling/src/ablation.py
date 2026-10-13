import os
import yaml
import numpy as np
import torch

from src.environment import V2VEnv
from src.mappo import MAPPO
from src.buffer import RolloutBuffer


def make_config(base, overrides):
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for section, values in overrides.items():
        if isinstance(values, dict):
            cfg[section] = {**cfg.get(section, {}), **values}
        else:
            cfg[section] = values
    return cfg


def train_variant(name, config, episodes=1500, rollout_steps=400):
    print(f"\n===== Training: {name} =====")
    env = V2VEnv(config)
    obs_dim = 6
    gs_dim = (config['simulation']['num_vehicles'] * 6
              + config['simulation']['num_vehicles'] ** 2)
    mappo = MAPPO(obs_dim, gs_dim, 3, config)
    buffer = RolloutBuffer()
    best_aoi = float('inf')
    ckpt_path = f"outputs/checkpoints/{name}.pth"

    for episode in range(episodes):
        local_obs, _ = env.reset(seed=episode)
        gs = env._get_global_state()

        for _ in range(rollout_steps):
            actions, log_probs, values = mappo.select_action(local_obs, gs)
            v = float(values.reshape(-1)[0])
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_gs = env._get_global_state()
            done = terminated or truncated
            buffer.add(local_obs, gs, actions, log_probs, rewards, v, float(done))
            local_obs, gs = next_obs, next_gs
            if done:
                break

        mappo.update(buffer)
        buffer.clear()

        if episode % 200 == 0:
            eval_aoi = []
            for ep in range(5):
                eo, _ = env.reset(seed=50000 + ep)
                d = False
                while not d:
                    with torch.no_grad():
                        a, _, _ = mappo.select_action(eo, env._get_global_state(), deterministic=True)
                    eo, _, t, tr, inf = env.step(a)
                    d = t or tr
                eval_aoi.append(inf['avg_aoi'])
            avg = float(np.mean(eval_aoi))
            print(f"  [{name}] ep {episode} | eval AoI: {avg:.2f}")

    torch.save({'actor': mappo.actor.state_dict(),
                'critic': mappo.critic.state_dict()}, ckpt_path)
    print(f"  Saved {ckpt_path}")


def main():
    with open("configs/default.yaml") as f:
        base = yaml.safe_load(f)

    variants = {
        "ablation_full":        {},
        "ablation_no_semantic": {"reward": {"alpha_semantic": 0.0}},
        "ablation_no_aoi":      {"reward": {"beta_aoi": 0.0}},
    }

    for name, overrides in variants.items():
        cfg = make_config(base, overrides)
        train_variant(name, cfg, episodes=1500)


if __name__ == "__main__":
    main()