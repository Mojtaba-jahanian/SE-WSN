import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------- مسیرهای پروژه --------
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]       # .../src
PROJECT_ROOT = THIS_FILE.parents[2]   # root

LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


def load_object_array(path):
    """کمک‌تابع برای npyهای نوع object (سری‌های زمان‌محور)."""
    return np.load(path, allow_pickle=True)


def per_episode_mean(series_array):
    """
    series_array: آرایه‌ای از طول num_episodes
    که هر عنصرش آرایه‌ی زمان‌محور است.
    خروجی: میانگین هر اپیزود (num_episodes,)
    """
    means = []
    for ep_series in series_array:
        arr = np.asarray(ep_series, dtype=float)
        if arr.size == 0:
            means.append(np.nan)
        else:
            means.append(float(arr.mean()))
    return np.array(means, dtype=float)


def plot_metric_vs_episode(values, ylabel, filename, title=None):
    eps = np.arange(len(values))

    plt.figure()
    plt.plot(eps, values, marker="o")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    out_path = FIG_DIR / filename
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    # ---------- 1) پاداش ----------
    rewards_path = LOG_DIR / "episode_rewards.npy"
    if rewards_path.exists():
        episode_rewards = np.load(rewards_path)
        plot_metric_vs_episode(
            episode_rewards,
            ylabel="Mean reward",
            filename="fig_training_reward.png",
            title="Training Reward per Episode",
        )
    else:
        print("episode_rewards.npy not found")

    # ---------- 2) انرژی ----------
    energy = load_object_array(LOG_DIR / "energy_log.npy")
    energy_mean = per_episode_mean(energy)
    plot_metric_vs_episode(
        energy_mean,
        ylabel="Avg residual energy",
        filename="fig_energy_per_episode.png",
        title="Average Residual Energy per Episode",
    )

    # ---------- 3) PDR ----------
    pdr = load_object_array(LOG_DIR / "pdr_log.npy")
    pdr_mean = per_episode_mean(pdr)
    plot_metric_vs_episode(
        pdr_mean,
        ylabel="PDR",
        filename="fig_pdr_per_episode.png",
        title="Packet Delivery Ratio per Episode",
    )

    # ---------- 4) کیفیت لینک ----------
    lq = load_object_array(LOG_DIR / "link_quality_log.npy")
    lq_mean = per_episode_mean(lq)
    plot_metric_vs_episode(
        lq_mean,
        ylabel="Link quality (avg)",
        filename="fig_link_quality_per_episode.png",
        title="Average Link Quality per Episode",
    )

    # ---------- 5) ریسک خرابی ----------
    fail = load_object_array(LOG_DIR / "failure_pred_log.npy")
    fail_mean = per_episode_mean(fail)
    plot_metric_vs_episode(
        fail_mean,
        ylabel="Predicted failure risk",
        filename="fig_failure_risk_per_episode.png",
        title="Predicted Node-Failure Risk per Episode",
    )

    # ---------- 6) Duty-cycle ----------
    duty = load_object_array(LOG_DIR / "duty_cycle_log.npy")
    duty_mean = per_episode_mean(duty)
    plot_metric_vs_episode(
        duty_mean,
        ylabel="Duty cycle (avg)",
        filename="fig_duty_cycle_per_episode.png",
        title="Average Duty Cycle per Episode",
    )

    # ---------- 7) Next-hop signal ----------
    nh = load_object_array(LOG_DIR / "next_hop_log.npy")
    nh_mean = per_episode_mean(nh)
    plot_metric_vs_episode(
        nh_mean,
        ylabel="Next-hop action (avg)",
        filename="fig_next_hop_per_episode.png",
        title="Average Next-Hop Control Signal per Episode",
    )

    # ---------- 8) Power level ----------
    power = load_object_array(LOG_DIR / "power_level_log.npy")
    power_mean = per_episode_mean(power)
    plot_metric_vs_episode(
        power_mean,
        ylabel="Tx power level (avg)",
        filename="fig_power_level_per_episode.png",
        title="Average Transmission Power per Episode",
    )

    # ---------- 9) Cluster-role ----------
    cluster = load_object_array(LOG_DIR / "cluster_role_log.npy")
    cluster_mean = per_episode_mean(cluster)
    plot_metric_vs_episode(
        cluster_mean,
        ylabel="Cluster-role action (avg)",
        filename="fig_cluster_role_per_episode.png",
        title="Average Cluster-Role Control per Episode",
    )

    # ---------- 10) Heatmap اکشن‌ها (اختیاری) ----------
    # از actions_log فقط یکی از اپیزودها را به عنوان مثال رسم می‌کنیم
    # ---------- 10) Heatmap اکشن‌ها ----------
    actions_path = LOG_DIR / "actions_log.npy"
    if actions_path.exists():
        all_actions = load_object_array(actions_path)

        if len(all_actions) > 0 and all_actions[0] is not None:
            ep0_raw = all_actions[0]

            # حذف stepهای None یا خالی
            ep0_raw = [step for step in ep0_raw if step is not None]

            # تبدیل هر step به آرایه float
            cleaned_steps = []
            for step in ep0_raw:
                try:
                    arr = np.asarray(step, dtype=float)
                    cleaned_steps.append(arr.flatten())
                except Exception:
                    continue

            if len(cleaned_steps) > 0:
                action_matrix = np.stack(cleaned_steps, axis=0)

                plt.figure()
                plt.imshow(action_matrix.T, aspect="auto", cmap="viridis")
                plt.colorbar(label="Action value")
                plt.xlabel("Step")
                plt.ylabel("Node × Action-dim index")
                plt.title("Action Heatmap – Episode 0")
                plt.tight_layout()

                out_path = FIG_DIR / "fig_actions_heatmap_ep0.png"
                plt.savefig(out_path, dpi=300)
                print(f"Saved: {out_path}")
                plt.close()
            else:
                print("No valid action steps to plot heatmap.")
        else:
            print("actions_log.npy exists but contains no usable data.")
    else:
        print("actions_log.npy not found.")


if __name__ == "__main__":
    main()
