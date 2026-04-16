"""
visualize.py

Generate figures from scored evaluation results.

Usage:
    python analysis/visualize.py --input results/combined_scored.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

TIER_ORDER = ["high_confidence", "low_confidence", "adversarial"]
TIER_LABELS = {
    "high_confidence": "High Confidence",
    "low_confidence": "Low Confidence",
    "adversarial": "Adversarial",
}
TIER_COLORS = {
    "high_confidence": "#4C9BE8",
    "low_confidence": "#E8A24C",
    "adversarial": "#E85C4C",
}

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150


def plot_hedge_counts(df: pd.DataFrame) -> None:
    """Box plot of hedge phrase counts by tier."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tiers_present = [t for t in TIER_ORDER if t in df["tier"].unique()]
    data = [df[df["tier"] == t]["hedge_phrase_count"].dropna().values for t in tiers_present]
    labels = [TIER_LABELS[t] for t in tiers_present]
    colors = [TIER_COLORS[t] for t in tiers_present]

    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Hedge Phrase Count by Prompt Tier", fontweight="bold")
    ax.set_ylabel("Hedge Phrase Count")
    ax.set_xlabel("Prompt Tier")

    outpath = FIGURES_DIR / "hedge_counts_by_tier.png"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_flag_rates(df: pd.DataFrame) -> None:
    """Bar chart of sycophancy and confabulation flag rates by tier."""
    tiers_present = [t for t in TIER_ORDER if t in df["tier"].unique()]

    syc_rates = [df[df["tier"] == t]["sycophancy_flag"].mean() * 100 for t in tiers_present]
    conf_rates = [df[df["tier"] == t]["confabulation_flag"].mean() * 100 for t in tiers_present]
    labels = [TIER_LABELS[t] for t in tiers_present]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, syc_rates, width, label="Sycophancy", color="#E85C4C", alpha=0.8)
    bars2 = ax.bar(x + width / 2, conf_rates, width, label="Confabulation Markers", color="#9B59B6", alpha=0.8)

    ax.set_title("Alignment Risk Flags by Prompt Tier", fontweight="bold")
    ax.set_ylabel("Flagged Responses (%)")
    ax.set_xlabel("Prompt Tier")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.bar_label(bars1, fmt="%.0f%%", padding=2, fontsize=9)
    ax.bar_label(bars2, fmt="%.0f%%", padding=2, fontsize=9)

    outpath = FIGURES_DIR / "flag_rates_by_tier.png"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_per_prompt_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of hedge count per prompt, averaged across runs."""
    pivot = (
        df.groupby(["tier", "topic"])["hedge_phrase_count"]
        .mean()
        .reset_index()
        .pivot(index="topic", columns="tier", values="hedge_phrase_count")
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Avg Hedge Phrase Count"},
    )
    ax.set_title("Average Hedge Phrases per Prompt and Tier", fontweight="bold")
    ax.set_ylabel("Prompt Topic")
    ax.set_xlabel("")

    outpath = FIGURES_DIR / "per_prompt_heatmap.png"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def main():
    set_style()
    parser = argparse.ArgumentParser(description="Visualize Claude eval results")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records. Generating figures...")

    plot_hedge_counts(df)
    plot_flag_rates(df)
    plot_per_prompt_heatmap(df)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
