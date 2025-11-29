# src/training/train.py
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

# --- مسیرهای پروژه ---
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]       # .../src
PROJECT_ROOT = THIS_FILE.parents[2]   # ریشه پروژه

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from maddpg.agent_maddpg import MADDPG
from maddpg.buffer import ReplayBuffer
from simulator.wsn_env import WSNEnv


def main():
    # --------- تنظیمات پایه WSN / MADDPG ----------
    n_nodes = 10
    obs_dim = 6
    act_dim = 4
    context_dim = 8
    buffer_size = int(1e5)

    n_episodes = 10          # برای تست، بعداً می‌تونید 1000 کنید
    batch_size = 64

    # محیط شبیه‌ساز WSN
    env = WSNEnv(n_nodes=n_nodes, obs_dim=obs_dim, act_dim=act_dim)

    # کنترل‌کننده MADDPG
    maddpg = MADDPG(
        n_agents=n_nodes,
        obs_dim=obs_dim,
        act_dim=act_dim,
        context_dim=context_dim,
        device="cpu",
    )

    # بافر تجربه
    buffer = ReplayBuffer(
        max_size=buffer_size,
        obs_dim_total=n_nodes * obs_dim,
        act_dim_total=n_nodes * act_dim,
        context_dim=context_dim,
        n_agents=n_nodes,
    )

    # --------- لیست‌های لاگ سطح اپیزود ----------
    episode_rewards = []

    energy_log = []            # هر عنصر: آرایه زمان-محور از میانگین انرژی
    pdr_log = []               # هر عنصر: آرایه زمان-محور از PDR
    link_quality_log = []      # هر عنصر: آرایه زمان-محور از میانگین LQ
    actions_log = []           # هر عنصر: لیست [step] از ماتریس actions (nodes × act_dim)
    duty_cycle_log = []        # هر عنصر: آرایه زمان-محور از میانگین duty-cycle
    next_hop_log = []          # هر عنصر: آرایه زمان-محور از مقدار اکشن next-hop (به‌صورت پیوسته)
    power_level_log = []       # هر عنصر: آرایه زمان-محور از میانگین power-level
    failure_pred_log = []      # هر عنصر: آرایه زمان-محور از میانگین P_fail پیش‌بینی‌شده
    cluster_role_log = []      # هر عنصر: آرایه زمان-محور از میانگین cluster-role action

    # --------- حلقه آموزش ----------
    for ep in trange(n_episodes, desc="Training episodes"):
        obs_list = env.reset()
        done = False
        ep_reward = 0.0
        context = np.zeros(context_dim, dtype=np.float32)

        # لاگ سطح گام (برای این اپیزود)
        ep_energy_series = []
        ep_pdr_series = []
        ep_lq_series = []
        ep_duty_series = []
        ep_nh_series = []
        ep_power_series = []
        ep_failpred_series = []
        ep_cluster_series = []
        ep_actions_series = []

        while not done:
            # --------- انتخاب اکشن‌ها ----------
            actions = maddpg.select_actions(obs_list, noise_scale=0.1)
            actions_np = np.stack([a.detach().cpu().numpy() for a in actions])

            # --------- یک گام در محیط ----------
            obs_next_list, rewards, done, info = env.step(actions_np)

            # --- فلت‌کردن برای بافر ---
            obs_flat = np.concatenate(obs_list, axis=0)
            obs_next_flat = np.concatenate(obs_next_list, axis=0)
            acts_flat = actions_np.reshape(-1)

            buffer.store(
                obs_flat,
                acts_flat,
                rewards,
                obs_next_flat,
                context,
                context,   # در نسخه‌ی کامل: context_next واقعی از Temporal-GNN
                float(done),
            )

            ep_reward += rewards.mean()
            obs_list = obs_next_list

            # --------- استخراج آمار برای لاگ ----------
            # obs_list: لیست طول n_nodes، هر کدام (obs_dim,)
            obs_array = np.stack(obs_list)        # (n_nodes, obs_dim)

            # فرض بر اساس مقاله:
            # x_i = [E_i, LQ_i, tau_i, d_i, sigma_i, P_fail_or_linkrisk]
            energies = obs_array[:, 0]
            link_qualities = obs_array[:, 1]
            # اگر ستون 4 همان P_fail باشد:
            if obs_array.shape[1] >= 5:
                fail_pred = obs_array[:, 4]
            else:
                fail_pred = np.zeros_like(energies)

            mean_energy = float(energies.mean())
            mean_lq = float(link_qualities.mean())
            mean_failpred = float(fail_pred.mean())

            ep_energy_series.append(mean_energy)
            ep_lq_series.append(mean_lq)
            ep_failpred_series.append(mean_failpred)

            # PDR از info (اگر موجود نبود -> NaN)
            pdr = info.get("pdr", np.nan) if isinstance(info, dict) else np.nan
            ep_pdr_series.append(float(pdr))

            # اکشن‌ها را هم به‌صورت خام ذخیره می‌کنیم
            ep_actions_series.append(actions_np.copy())

            # فرض: ساختار اکشن: [next-hop, power, duty-cycle, cluster-role]
            next_hop_vals = actions_np[:, 0]
            power_vals = actions_np[:, 1]
            duty_vals = actions_np[:, 2]
            cluster_vals = actions_np[:, 3]

            ep_nh_series.append(float(next_hop_vals.mean()))
            ep_power_series.append(float(power_vals.mean()))
            ep_duty_series.append(float(duty_vals.mean()))
            ep_cluster_series.append(float(cluster_vals.mean()))

            # --------- آپدیت MADDPG ----------
            if buffer.size > batch_size:
                maddpg.update(buffer, batch_size)

        # --- پایان اپیزود: لاگ‌ها را جمع‌بندی می‌کنیم ---
        print(f"Episode {ep} | Mean reward: {ep_reward:.3f}")
        episode_rewards.append(ep_reward)

        energy_log.append(np.array(ep_energy_series, dtype=np.float32))
        pdr_log.append(np.array(ep_pdr_series, dtype=np.float32))
        link_quality_log.append(np.array(ep_lq_series, dtype=np.float32))
        failure_pred_log.append(np.array(ep_failpred_series, dtype=np.float32))
        duty_cycle_log.append(np.array(ep_duty_series, dtype=np.float32))
        next_hop_log.append(np.array(ep_nh_series, dtype=np.float32))
        power_level_log.append(np.array(ep_power_series, dtype=np.float32))
        cluster_role_log.append(np.array(ep_cluster_series, dtype=np.float32))

        actions_log.append(ep_actions_series)

    # --------- ذخیره مدل‌ها ----------
    save_dir = PROJECT_ROOT / "results" / "trained_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    for agent_id, actor in enumerate(maddpg.actors):
        actor_path = save_dir / f"actor_{agent_id}.pth"
        torch.save(actor.state_dict(), actor_path)

    critic_path = save_dir / "critic_centralized.pth"
    torch.save(maddpg.critic.state_dict(), critic_path)

    # --------- ذخیره لاگ‌ها ----------
    logs_dir = PROJECT_ROOT / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1) پاداش اپیزودها
    np.save(logs_dir / "episode_rewards.npy", np.array(episode_rewards, dtype=np.float32))

    # 2) لاگ‌هایی که طول اپیزودها ممکن است متفاوت باشد -> به‌صورت object ذخیره می‌کنیم
    np.save(logs_dir / "energy_log.npy", np.array(energy_log, dtype=object))
    np.save(logs_dir / "pdr_log.npy", np.array(pdr_log, dtype=object))
    np.save(logs_dir / "link_quality_log.npy", np.array(link_quality_log, dtype=object))
    np.save(logs_dir / "failure_pred_log.npy", np.array(failure_pred_log, dtype=object))
    np.save(logs_dir / "duty_cycle_log.npy", np.array(duty_cycle_log, dtype=object))
    np.save(logs_dir / "next_hop_log.npy", np.array(next_hop_log, dtype=object))
    np.save(logs_dir / "power_level_log.npy", np.array(power_level_log, dtype=object))
    np.save(logs_dir / "cluster_role_log.npy", np.array(cluster_role_log, dtype=object))

    # actions: هر اپیزود -> لیست steps، هر step -> (n_nodes, act_dim)
    np.save(logs_dir / "actions_log.npy", np.array(actions_log, dtype=object))

    print(f"Models saved to {save_dir}")
    print(f"Logs saved to {logs_dir}")
    print("Training run finished.")


if __name__ == "__main__":
    main()
