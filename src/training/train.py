import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from maddpg.agent_maddpg import MADDPG
from maddpg.buffer import ReplayBuffer
from simulator.wsn_env import WSNEnv


def main():
    n_nodes = 10
    obs_dim = 6
    act_dim = 4
    context_dim = 8
    buffer_size = int(1e5)

    n_episodes = 10
    batch_size = 64

    env = WSNEnv(n_nodes=n_nodes, obs_dim=obs_dim, act_dim=act_dim)
    maddpg = MADDPG(
        n_agents=n_nodes,
        obs_dim=obs_dim,
        act_dim=act_dim,
        context_dim=context_dim,
        device="cpu",
    )

    buffer = ReplayBuffer(
        max_size=buffer_size,
        obs_dim_total=n_nodes * obs_dim,
        act_dim_total=n_nodes * act_dim,
        context_dim=context_dim,
        n_agents=n_nodes,
    )

    for ep in trange(n_episodes, desc="Training episodes"):
        obs_list = env.reset()
        done = False
        ep_reward = 0.0
        context = np.zeros(context_dim, dtype=np.float32)

        while not done:
            actions = maddpg.select_actions(obs_list, noise_scale=0.1)
            actions_np = np.stack([a.detach().cpu().numpy() for a in actions])

            obs_next_list, rewards, done, _ = env.step(actions_np)

            obs_flat = np.concatenate(obs_list, axis=0)
            obs_next_flat = np.concatenate(obs_next_list, axis=0)
            acts_flat = actions_np.reshape(-1)

            buffer.store(
                obs_flat,
                acts_flat,
                rewards,
                obs_next_flat,
                context,
                context,
                float(done),
            )
            ep_reward += rewards.mean()
            obs_list = obs_next_list

            if buffer.size > batch_size:
                maddpg.update(buffer, batch_size)

        print(f"Episode {ep} | Mean reward: {ep_reward:.3f}")

    save_dir = PROJECT_ROOT / "results" / "trained_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    for agent_id, actor in enumerate(maddpg.actors):
        actor_path = save_dir / f"actor_{agent_id}.pth"
        torch.save(actor.state_dict(), actor_path)

    critic_path = save_dir / "critic_centralized.pth"
    torch.save(maddpg.critic.state_dict(), critic_path)

    print(f"Models saved to {save_dir}")
    print("Training run finished.")


if __name__ == "__main__":
    main()
