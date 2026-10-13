import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.local_obs = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, local_obs, global_state, action, log_prob, reward, value, done):
        self.local_obs.append(np.asarray(local_obs, dtype=np.float32))
        self.global_states.append(np.asarray(global_state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.int64))
        self.log_probs.append(np.asarray(log_prob, dtype=np.float32))
        self.rewards.append(np.asarray(reward, dtype=np.float32))
        self.values.append(float(value))
        self.dones.append(float(done))

    def clear(self):
        self.local_obs = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def get_tensors(self, device="cpu"):
        return {
            "local_obs":     torch.tensor(np.stack(self.local_obs), dtype=torch.float32).to(device),
            "global_states": torch.tensor(np.stack(self.global_states), dtype=torch.float32).to(device),
            "actions":       torch.tensor(np.stack(self.actions), dtype=torch.long).to(device),
            "log_probs":     torch.tensor(np.stack(self.log_probs), dtype=torch.float32).to(device),
            "rewards":       torch.tensor(np.stack(self.rewards), dtype=torch.float32).to(device),
            "values":        torch.tensor(np.array(self.values), dtype=torch.float32).to(device),
            "dones":         torch.tensor(np.array(self.dones), dtype=torch.float32).to(device),
        }