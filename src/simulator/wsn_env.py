import numpy as np


class WSNEnv:
    def __init__(self, n_nodes: int, obs_dim: int = 6, act_dim: int = 4):
        self.n_nodes = n_nodes
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.t = 0

    def reset(self):
        self.t = 0
        self.obs = [
            np.random.rand(self.obs_dim).astype(np.float32)
            for _ in range(self.n_nodes)
        ]
        return self.obs

    def step(self, actions):
        actions = np.clip(actions, -1.0, 1.0)
        self.t += 1

        energy_cost = np.linalg.norm(actions, axis=1)

        smooth_cost = np.zeros(self.n_nodes, dtype=np.float32)
        for i in range(self.n_nodes - 1):
            diff = np.linalg.norm(actions[i] - actions[i + 1])
            smooth_cost[i] += diff
            smooth_cost[i + 1] += diff

        rewards = -0.5 * energy_cost - 0.1 * smooth_cost

        self.obs = [
            (o + 0.01 * np.random.randn(self.obs_dim)).astype(np.float32)
            for o in self.obs
        ]

        done = self.t >= 50
        info = {}
        return self.obs, rewards.astype(np.float32), done, info
