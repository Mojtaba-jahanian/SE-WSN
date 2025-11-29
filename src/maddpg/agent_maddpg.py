import copy
import torch
import torch.nn.functional as F

from .actor import Actor
from .critic import Critic


class MADDPG:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        context_dim: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.actors_target = [copy.deepcopy(a).to(device) for a in self.actors]
        self.actor_opt = [
            torch.optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors
        ]

        self.critic = Critic(
            obs_dim_total=n_agents * obs_dim,
            act_dim_total=n_agents * act_dim,
            context_dim=context_dim,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_actions(self, obs_list, noise_scale: float = 0.1):
        actions = []
        for i, obs in enumerate(obs_list):
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            a = self.actors[i](o)
            if noise_scale > 0.0:
                a = a + noise_scale * torch.randn_like(a)
            actions.append(a.clamp(-1.0, 1.0))
        return actions

    def update(self, buffer, batch_size: int):
        batch = buffer.sample(batch_size)

        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        obs_next = batch["obs_next"].to(self.device)
        context = batch["context"].to(self.device)
        context_next = batch["context_next"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Critic update
        with torch.no_grad():
            obs_next_list = obs_next.view(-1, self.n_agents, self.obs_dim)
            next_acts = []
            for i in range(self.n_agents):
                a_t = self.actors_target[i](obs_next_list[:, i, :])
                next_acts.append(a_t)
            next_acts_cat = torch.cat(next_acts, dim=-1)
            q_next = self.critic_target(obs_next, next_acts_cat, context_next)
            y = rews.mean(dim=-1, keepdim=True) + self.gamma * (1 - dones) * q_next

        q = self.critic(obs, acts, context)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        obs_list = obs.view(-1, self.n_agents, self.obs_dim)
        cur_acts = []
        for i in range(self.n_agents):
            a_i = self.actors[i](obs_list[:, i, :])
            cur_acts.append(a_i)
        cur_acts_cat = torch.cat(cur_acts, dim=-1)

        actor_loss = -self.critic(obs, cur_acts_cat, context).mean()
        for opt in self.actor_opt:
            opt.zero_grad()
        actor_loss.backward()
        for opt in self.actor_opt:
            opt.step()

        self._soft_update()
        return critic_loss.item(), actor_loss.item()

    def _soft_update(self):
        with torch.no_grad():
            for i in range(self.n_agents):
                for p, p_t in zip(
                    self.actors[i].parameters(), self.actors_target[i].parameters()
                ):
                    p_t.data.mul_(1 - self.tau)
                    p_t.data.add_(self.tau * p.data)

            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - self.tau)
                p_t.data.add_(self.tau * p.data)
