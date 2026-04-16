"""
analyze.py

Aggregate analysis of scored evaluation results.
Computes cross-tier comparisons and statistical tests.

Usage:
    python analysis/analyze.py --input results/combined_scored.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_scored(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["tier", "prompt_id", "hedge_phrase_count", "sycophancy_flag", "confabulation_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns (did you run scorer.py first?): {missing}")
    return df


def compare_tiers(df: pd.DataFrame) -> None:
    """
    Compare hedge phrase counts between high and low confidence tiers.
    Tests the hypothesis: low confidence prompts elicit more hedging.
    """
    high = df[df["tier"] == "high_confidence"]["hedge_phrase_count"].dropna()
    low = df[df["tier"] == "low_confidence"]["hedge_phrase_count"].dropna()

    if len(high) == 0 or len(low) == 0:
        print("Not enough data for tier comparison.")
        return

    t_stat, p_val = stats.mannwhitneyu(high, low, alternative="less")

    print("\nHedge phrase count comparison: high vs low confidence tiers")
    print(f"  High confidence -- mean: {high.mean():.2f}, median: {high.median():.1f}")
    print(f"  Low confidence  -- mean: {low.mean():.2f}, median: {low.median():.1f}")
    print(f"  Mann-Whitney U test (hypothesis: high < low): U={t_stat:.1f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("  Result: Significant. Model hedges more on low-confidence prompts.")
    else:
        print("  Result: Not significant. Model does not reliably hedge more on uncertain topics.")


def sycophancy_by_tier(df: pd.DataFrame) -> None:
    """Report sycophancy rates across tiers."""
    print("\nSycophancy flag rates by tier:")
    for tier in sorted(df["tier"].unique()):
        tier_df = df[df["tier"] == tier]
        rate = tier_df["sycophancy_flag"].mean()
        n = len(tier_df)
        print(f"  {tier}: {rate * 100:.1f}% ({int(rate * n)}/{n} responses)")


def confabulation_by_tier(df: pd.DataFrame) -> None:
    """Report confabulation marker rates across tiers."""
    print("\nConfabulation marker rates by tier:")
    for tier in sorted(df["tier"].unique()):
        tier_df = df[df["tier"] == tier]
        rate = tier_df["confabulation_flag"].mean()
        n = len(tier_df)
        print(f"  {tier}: {rate * 100:.1f}% ({int(rate * n)}/{n} responses)")


def consistency_analysis(df: pd.DataFrame) -> None:
    """
    For prompts run multiple times, measure response consistency.
    High variance in hedge counts for the same prompt suggests instability.
    """
    print("\nResponse consistency (std dev of hedge count per prompt):")
    consistency = (
        df.groupby(["tier", "prompt_id"])["hedge_phrase_count"]
        .std()
        .reset_index()
        .rename(columns={"hedge_phrase_count": "hedge_std"})
    )
    print(consistency.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Analyze scored Claude eval results")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    df = load_scored(args.input)
    print(f"Loaded {len(df)} scored records.")

    compare_tiers(df)
    sycophancy_by_tier(df)
    confabulation_by_tier(df)
    consistency_analysis(df)


if __name__ == "__main__":
    main()
