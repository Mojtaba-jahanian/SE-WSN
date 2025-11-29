import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        obs_dim_total: int,
        act_dim_total: int,
        context_dim: int,
        n_agents: int,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim_total), dtype=np.float32)
        self.acts = np.zeros((max_size, act_dim_total), dtype=np.float32)
        self.rews = np.zeros((max_size, n_agents), dtype=np.float32)
        self.obs_next = np.zeros((max_size, obs_dim_total), dtype=np.float32)
        self.context = np.zeros((max_size, context_dim), dtype=np.float32)
        self.context_next = np.zeros((max_size, context_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def store(
        self,
        obs,
        acts,
        rews,
        obs_next,
        context,
        context_next,
        done,
    ):
        idx = self.ptr % self.max_size
        self.obs[idx] = obs
        self.acts[idx] = acts
        self.rews[idx] = rews
        self.obs_next[idx] = obs_next
        self.context[idx] = context
        self.context_next[idx] = context_next
        self.dones[idx] = done

        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.from_numpy(self.obs[idxs]),
            acts=torch.from_numpy(self.acts[idxs]),
            rews=torch.from_numpy(self.rews[idxs]),
            obs_next=torch.from_numpy(self.obs_next[idxs]),
            context=torch.from_numpy(self.context[idxs]),
            context_next=torch.from_numpy(self.context_next[idxs]),
            dones=torch.from_numpy(self.dones[idxs]),
        )
        return batch
