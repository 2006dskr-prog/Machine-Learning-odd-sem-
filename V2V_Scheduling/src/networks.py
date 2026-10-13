import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Shared decentralized policy for all vehicles."""
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


class CentralizedCritic(nn.Module):
    """Centralized value network using the joint global state."""
    def __init__(self, global_state_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, global_state):
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)