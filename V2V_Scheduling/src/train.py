import os
import yaml
import numpy as np
import torch

from src.environment import V2VEnv
from src.mappo import MAPPO
from src.buffer import RolloutBuffer


def train():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    env = V2VEnv(config)
    obs_dim = 6
    global_state_dim = (config['simulation']['num_vehicles'] * 6
                        + config['simulation']['num_vehicles'] ** 2)
    action_dim = 3
    num_agents = config['simulation']['num_vehicles']

    mappo = MAPPO(obs_dim, global_state_dim, action_dim, config)
    buffer = RolloutBuffer()

    total_episodes = config['training']['total_episodes']
    rollout_steps = config['training']['rollout_steps']
    best_aoi = float('inf')

    history = {
        'episode': [],
        'train_reward': [],
        'eval_aoi': [],
        'eval_pdr': [],
        'eval_utility': [],
    }

    for episode in range(total_episodes):
        local_obs, _ = env.reset(seed=episode)
        global_state = env._get_global_state()
        episode_reward = 0.0
        info = {}

        for _ in range(rollout_steps):
            actions, log_probs, values = mappo.select_action(local_obs, global_state)
            value_scalar = float(values.reshape(-1)[0])

            next_local_obs, per_agent_rewards, terminated, truncated, info = env.step(actions)
            next_global_state = env._get_global_state()
            done = terminated or truncated

            buffer.add(
                local_obs=local_obs,
                global_state=global_state,
                action=actions,
                log_prob=log_probs,
                reward=per_agent_rewards,
                value=value_scalar,
                done=float(done),
            )

            local_obs = next_local_obs
            global_state = next_global_state
            episode_reward += float(np.sum(per_agent_rewards))

            if done:
                break

        mappo.update(buffer)
        buffer.clear()

        # ---- periodic evaluation ----
        if episode % 100 == 0:
            eval_aoi, eval_pdr, eval_util = [], [], []
            last_eval_obs = None

            for eval_ep in range(10):
                eval_obs, _ = env.reset(seed=9999 + eval_ep)
                done = False
                eval_info = {}
                while not done:
                    eval_actions, _, _ = mappo.select_action(
                        eval_obs, env._get_global_state(), deterministic=True
                    )
                    eval_obs, _, terminated, truncated, eval_info = env.step(eval_actions)
                    done = terminated or truncated
                    last_eval_obs = eval_obs
                eval_aoi.append(eval_info['avg_aoi'])
                eval_pdr.append(eval_info['pdr'])
                eval_util.append(eval_info['semantic_utility'])

            avg_eval_aoi = float(np.mean(eval_aoi))
            avg_eval_pdr = float(np.mean(eval_pdr))
            avg_eval_util = float(np.mean(eval_util))

            print(f"Episode {episode} | Train Reward: {episode_reward:.2f} | "
                  f"Eval AoI: {avg_eval_aoi:.2f} | Eval PDR: {avg_eval_pdr:.2f} | "
                  f"Eval Utility: {avg_eval_util:.2f}")

            if last_eval_obs is not None:
                with torch.no_grad():
                    sample_obs = torch.tensor(last_eval_obs, dtype=torch.float32).to(mappo.device)
                    sample_probs = mappo.actor(sample_obs)
                    print(f"  Action probs for vehicle 0: "
                          f"{sample_probs[0].cpu().numpy().round(3)}")

            history['episode'].append(episode)
            history['train_reward'].append(episode_reward)
            history['eval_aoi'].append(avg_eval_aoi)
            history['eval_pdr'].append(avg_eval_pdr)
            history['eval_utility'].append(avg_eval_util)

            if episode >= 100 and avg_eval_aoi < best_aoi:
                best_aoi = avg_eval_aoi
                torch.save({
                    'actor': mappo.actor.state_dict(),
                    'critic': mappo.critic.state_dict(),
                    'aoi': best_aoi,
                    'episode': episode,
                }, "outputs/checkpoints/best_mappo.pth")
                print(f"  --> New best model saved (AoI: {best_aoi:.2f})")

        elif episode % 10 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | "
                  f"Avg AoI: {info.get('avg_aoi', 0.0):.2f} | "
                  f"PDR: {info.get('pdr', 0.0):.2f}")

    np.save("outputs/logs/history.npy", history)
    print("\nTraining complete.")
    print(f"Best eval AoI: {best_aoi:.2f}")


if __name__ == "__main__":
    train()