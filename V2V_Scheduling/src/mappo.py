import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks import Actor, CentralizedCritic


class MAPPO:
    def __init__(self, obs_dim, global_state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hs = config['training']['hidden_size']
        self.actor = Actor(obs_dim, action_dim, hs).to(self.device)
        self.critic = CentralizedCritic(global_state_dim, hs).to(self.device)

        lr = config['training']['learning_rate']
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = config['training']['gamma']
        self.gae_lambda = config['training']['gae_lambda']
        self.clip_epsilon = config['training']['clip_epsilon']
        self.entropy_coef = config['training']['entropy_coefficient']
        self.value_coef = config['training']['value_coefficient']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.update_epochs = config['training']['update_epochs']
        self.num_agents = config['simulation']['num_vehicles']

    def select_action(self, local_obs, global_state, deterministic=False):
        local_obs_t = torch.tensor(local_obs, dtype=torch.float32).to(self.device)
        global_state_t = torch.tensor(global_state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.actor(local_obs_t)

            # Action masking: if queue is empty, only action 0 is valid.
            # Index 2 in the observation is the normalized queue size.
            queue_sizes = local_obs_t[:, 2] * float(self.config['simulation']['max_queue_size'])
            no_msg = (queue_sizes < 0.5)

            action_mask = torch.ones_like(probs)
            action_mask[no_msg, 1:] = 0.0
            probs = probs * action_mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            if deterministic:
                actions = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()

            log_probs = torch.log(
                probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8
            )
            value = self.critic(global_state_t)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.cpu().numpy()

    def compute_gae(self, rewards, values, dones):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        T = rewards.shape[0]
        values_b = np.tile(values, (1, self.num_agents))
        dones_b = np.tile(dones, (1, self.num_agents))

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values_b[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones_b[t]) - values_b[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones_b[t]) * last_gae
        returns = advantages + values_b
        return advantages, returns

    def update(self, buffer):
        data = buffer.get_tensors(self.device)

        rewards_np = data['rewards'].cpu().numpy()
        values_np = data['values'].cpu().numpy()
        dones_np = data['dones'].cpu().numpy()

        advantages, returns = self.compute_gae(rewards_np, values_np, dones_np)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        T = data['local_obs'].shape[0]
        local_obs_flat = data['local_obs'].reshape(T * self.num_agents, -1)
        actions_flat = data['actions'].reshape(T * self.num_agents)
        old_log_probs_flat = data['log_probs'].reshape(T * self.num_agents)
        advantages_flat = torch.tensor(
            advantages.reshape(-1), dtype=torch.float32
        ).to(self.device)

        returns_joint = torch.tensor(
            returns.mean(axis=1), dtype=torch.float32
        ).to(self.device)

        last_stats = {}
        for _ in range(self.update_epochs):
            probs = self.actor(local_obs_flat)

            # Re-apply masking during the update as well
            queue_sizes = local_obs_flat[:, 2] * float(self.config['simulation']['max_queue_size'])
            no_msg = (queue_sizes < 0.5)
            mask = torch.ones_like(probs)
            mask[no_msg, 1:] = 0.0
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_flat)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_flat
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_values = self.critic(data['global_states']).squeeze(-1)
            critic_loss = nn.MSELoss()(critic_values, returns_joint)

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            last_stats = {
                'actor_loss': float(actor_loss.item()),
                'critic_loss': float(critic_loss.item()),
                'entropy': float(entropy.item()),
            }

        return last_stats