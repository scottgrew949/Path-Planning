"""
training_curves.py — plot Q-learning training curves from training_data.csv.
Run after the C++ binary has written training_data.csv:
    python training_curves.py
"""

import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = Path(__file__).parent.parent.parent / "qlearning_training.csv"

def load(path: Path):
    episodes, rewards, steps, goal_reached, epsilons = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            steps.append(int(row["steps"]))
            goal_reached.append(int(row["goal_reached"]))
            epsilons.append(float(row["epsilon"]))
    return episodes, rewards, steps, goal_reached, epsilons

def rolling_mean(values: list, window: int) -> list:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run the C++ binary first.")
        sys.exit(1)

    episodes, rewards, steps, goal_reached, epsilons = load(CSV_PATH)
    window = max(1, len(episodes) // 50)  # ~2% of episodes per window

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Q-Learning Training Curves", fontsize=14, fontweight="bold")

    # ---- Reward per episode ----
    ax = axes[0]
    ax.plot(episodes, rewards, color="#b0c4de", linewidth=0.6, alpha=0.7, label="Raw")
    ax.plot(episodes, rolling_mean(rewards, window), color="#1f77b4", linewidth=1.8,
            label=f"Rolling mean (w={window})")
    ax.set_ylabel("Total Reward")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    ax.set_title("Reward per Episode")

    # ---- Steps to goal ----
    ax = axes[1]
    ax.plot(episodes, steps, color="#f0c080", linewidth=0.6, alpha=0.7, label="Raw")
    ax.plot(episodes, rolling_mean(steps, window), color="#d67000", linewidth=1.8,
            label=f"Rolling mean (w={window})")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=8)
    ax.set_title("Steps to Goal per Episode")

    # ---- Epsilon decay ----
    ax = axes[2]
    ax.plot(episodes, epsilons, color="#2ca02c", linewidth=1.8)
    ax.set_ylabel("Epsilon (ε)")
    ax.set_xlabel("Episode")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_title("Epsilon Decay")

    plt.tight_layout()
    out_path = Path(__file__).parent / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
