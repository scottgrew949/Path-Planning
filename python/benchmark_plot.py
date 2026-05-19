"""
benchmark_plot.py — visualise benchmark_results.csv produced by statistical_benchmark.py.
Usage:
    python python/benchmark_plot.py [--input benchmark_results.csv] [--output-dir .]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ALGO_COLORS = {
    "Expert":  "#2ca02c",   # green
    "BC":      "#1f77b4",   # blue
    "DAgger":  "#9467bd",   # purple
    "Random":  "#7f7f7f",   # gray
}
ALGO_ORDER = ["Expert", "BC", "DAgger", "Random"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    present = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    df = df[df["algorithm"].isin(present)]
    return df, present


def add_bar_labels(ax, bars, fmt="{:.1f}%", offset_frac=0.01):
    y_max = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * offset_frac,
            fmt.format(h),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )


# ------------------------------------------------------------------
# Figure 1 — success rate
# ------------------------------------------------------------------
def plot_success_rate(df: pd.DataFrame, present: list, out_dir: Path):
    rates = []
    for algo in present:
        sub = df[df["algorithm"] == algo]
        rate = sub["success"].mean() * 100
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Algorithm Success Rate (N=1000 seeds)", fontsize=13, fontweight="bold")

    colors = [ALGO_COLORS[a] for a in present]
    bars = ax.bar(present, rates, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    ax.set_ylim(0, min(115, max(rates) * 1.15 + 5))
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Algorithm")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    add_bar_labels(ax, bars, fmt="{:.1f}%")

    plt.tight_layout()
    out = out_dir / "benchmark_success_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved benchmark_success_rate.png")


# ------------------------------------------------------------------
# Figure 2 — mean path length (successful only)
# ------------------------------------------------------------------
def plot_path_length(df: pd.DataFrame, present: list, out_dir: Path):
    means, stds, labels = [], [], []
    for algo in present:
        sub = df[(df["algorithm"] == algo) & (df["path_length"] != -1)]
        if sub.empty:
            continue
        means.append(sub["path_length"].mean())
        stds.append(sub["path_length"].std())
        labels.append(algo)

    if not labels:
        print("No successful episodes found — skipping benchmark_path_length.png")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Mean Path Length — Successful Episodes Only", fontsize=13, fontweight="bold")

    colors = [ALGO_COLORS[a] for a in labels]
    bars = ax.bar(
        labels, means, yerr=stds, color=colors, width=0.5,
        edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "capsize": 4, "capthick": 1.5, "ecolor": "#333333"},
    )

    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.18)
    ax.set_ylabel("Mean Path Length (steps)")
    ax.set_xlabel("Algorithm")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    add_bar_labels(ax, bars, fmt="{:.1f}", offset_frac=0.015)

    plt.tight_layout()
    out = out_dir / "benchmark_path_length.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved benchmark_path_length.png")


# ------------------------------------------------------------------
# Figure 3 — classical vs RL
# ------------------------------------------------------------------
def plot_classical_vs_rl(df: pd.DataFrame, present: list, out_dir: Path):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Classical vs RL: Per-Seed Path Length Distribution", fontsize=13, fontweight="bold")

    # --- Left: success rate bar chart, Expert labelled "Classical Proxy" ---
    display_names = {"Expert": "Classical\nProxy"}
    plot_labels = [display_names.get(a, a) for a in present]
    rates = [df[df["algorithm"] == a]["success"].mean() * 100 for a in present]
    colors = [ALGO_COLORS[a] for a in present]

    bars = ax_left.bar(plot_labels, rates, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    ax_left.set_ylim(0, min(115, max(rates) * 1.15 + 5))
    ax_left.set_ylabel("Success Rate (%)")
    ax_left.set_xlabel("Algorithm")
    ax_left.set_title("Success Rate")
    ax_left.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax_left.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax_left.set_axisbelow(True)

    y_max = ax_left.get_ylim()[1]
    for bar, rate in zip(bars, rates):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.01,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # --- Right: per-seed scatter, path_length vs seed index ---
    seeds_sorted = sorted(df["seed"].unique())
    seed_index = {s: i for i, s in enumerate(seeds_sorted)}

    for algo in present:
        sub = df[df["algorithm"] == algo].copy()
        sub = sub[sub["path_length"] != -1]
        if sub.empty:
            continue
        xs = [seed_index[s] for s in sub["seed"]]
        ys = sub["path_length"].tolist()
        ax_right.scatter(
            xs, ys,
            color=ALGO_COLORS[algo], alpha=0.3, s=8,
            label=display_names.get(algo, algo),
        )

    ax_right.set_xlabel("Seed Index")
    ax_right.set_ylabel("Path Length")
    ax_right.set_title("Per-Seed Path Length (successful only)")
    ax_right.grid(linewidth=0.5, alpha=0.4)
    ax_right.set_axisbelow(True)
    ax_right.legend(fontsize=9, markerscale=2)

    plt.tight_layout()
    out = out_dir / "benchmark_classical_vs_rl.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved benchmark_classical_vs_rl.png")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent.parent / "benchmark_results.csv"),
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent),
        help="Directory to write PNG files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run statistical_benchmark.py first.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    df, present = load(input_path)

    if not present:
        print("No recognised algorithms found in CSV.")
        sys.exit(1)

    plot_success_rate(df, present, out_dir)
    plot_path_length(df, present, out_dir)
    plot_classical_vs_rl(df, present, out_dir)


if __name__ == "__main__":
    main()
