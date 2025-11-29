import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim_total: int,
        act_dim_total: int,
        context_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        in_dim = obs_dim_total + act_dim_total + context_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, acts, context):
        x = torch.cat([obs, acts, context], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
