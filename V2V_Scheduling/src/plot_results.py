import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training():
    path = "outputs/logs/history.npy"
    if not os.path.exists(path):
        print(f"[skip] {path} not found — training curves will not be plotted.")
        return

    hist = np.load(path, allow_pickle=True).item()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(hist['episode'], hist['train_reward'], color='tab:blue')
    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Mean episode reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist['episode'], hist['eval_aoi'], color='tab:red')
    axes[1].axhline(5.30, ls='--', color='gray', label='Round-Robin')
    axes[1].axhline(13.96, ls=':', color='black', label='Semantic-priority')
    axes[1].set_title("Evaluation AoI")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average AoI")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(hist['episode'], hist['eval_utility'], color='tab:green')
    axes[2].axhline(111.73, ls='--', color='gray', label='AoI-based')
    axes[2].set_title("Evaluation Semantic Utility")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Utility")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/figures/training_curves.png", dpi=150)
    plt.close()
    print("Saved outputs/figures/training_curves.png")


def plot_comparison():
    methods = ["Random", "Round-Robin", "AoI-based", "Semantic-priority", "MAPPO"]

    # Values from your final_eval run
    aoi = [5.34, 5.30, 11.34, 13.96, 8.95]
    utility = [109.34, 107.74, 111.73, 106.32, 175.64]
    fairness = [0.943, 0.949, 0.492, 0.346, 0.672]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, aoi, width, label='AoI (lower better)', color='tab:red')
    ax.bar(x, utility, width, label='Utility (higher better)', color='tab:green')
    ax.bar(x + width, [f * 100 for f in fairness], width,
           label='Fairness × 100 (higher better)', color='tab:blue')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_title("Scheduler Comparison: AoI vs. Semantic Utility vs. Fairness")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("outputs/figures/comparison.png", dpi=150)
    plt.close()
    print("Saved outputs/figures/comparison.png")


def plot_radar():
    """Radar plot showing each scheduler's normalized performance."""
    import numpy as np

    methods = ["Random", "Round-Robin", "AoI-based", "Semantic-priority", "MAPPO"]
    aoi = np.array([5.34, 5.30, 11.34, 13.96, 8.95])
    utility = np.array([109.34, 107.74, 111.73, 106.32, 175.64])
    fairness = np.array([0.943, 0.949, 0.492, 0.346, 0.672])

    # Normalize to [0, 1] where 1 is best
    aoi_norm = 1 - (aoi - aoi.min()) / (aoi.max() - aoi.min() + 1e-8)
    util_norm = (utility - utility.min()) / (utility.max() - utility.min() + 1e-8)
    fair_norm = fairness  # already in [0, 1]

    labels = ['AoI\n(higher=fresher)', 'Utility', 'Fairness']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, m in enumerate(methods):
        vals = [aoi_norm[i], util_norm[i], fair_norm[i]]
        vals += vals[:1]
        ax.plot(angles, vals, label=m, linewidth=2)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Normalized Scheduler Performance", pad=20)

    plt.tight_layout()
    plt.savefig("outputs/figures/radar.png", dpi=150)
    plt.close()
    print("Saved outputs/figures/radar.png")

def plot_ablation():
    variants = [
        "Full model",
        "No semantic\nreward",
        "No AoI\nreward",
        "Retrained\nfull",
    ]
    aoi      = [7.97, 11.98, 12.32, 11.33]
    utility  = [194.23, 153.56, 144.68, 140.49]
    fairness = [0.702, 0.434, 0.416, 0.443]

    x = np.arange(len(variants))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, aoi, width, label='AoI ↓', color='tab:red')
    ax.bar(x, utility, width, label='Utility ↑', color='tab:green')
    ax.bar(x + width, [f * 100 for f in fairness], width,
           label='Fairness ×100 ↑', color='tab:blue')

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_title("Ablation Study: Effect of Removing Each Reward Component")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("outputs/figures/ablation.png", dpi=150)
    plt.close()
    print("Saved outputs/figures/ablation.png")

if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)
    plot_training()
    plot_comparison()
    plot_radar()
    plot_ablation()
    print("\nAll figures generated. See outputs/figures/")