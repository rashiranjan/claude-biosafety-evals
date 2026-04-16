"""
run_evals.py

Main evaluation runner for the claude-biosafety-evals project.
Loads prompt batteries, queries the Anthropic API, and saves results to CSV.

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --tier high_confidence
    python evals/run_evals.py --tier all --runs 3
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = os.environ.get("MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0))
N_RUNS = int(os.environ.get("N_RUNS_PER_PROMPT", 3))

PROMPTS_DIR = Path("data/prompts")
RESULTS_DIR = Path("results")


def load_prompts(tier: str) -> list[dict]:
    """Load prompt battery for a given tier."""
    filepath = PROMPTS_DIR / f"{tier}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"No prompt file found for tier: {tier}")
    with open(filepath) as f:
        return json.load(f)


def query_claude(prompt: str, system_prompt: str | None = None) -> dict:
    """
    Send a single prompt to Claude and return the response with metadata.
    """
    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    start = time.time()
    response = CLIENT.messages.create(**kwargs)
    latency = time.time() - start

    return {
        "response_text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_seconds": round(latency, 3),
        "stop_reason": response.stop_reason,
    }


def run_tier(tier: str, n_runs: int = N_RUNS) -> list[dict]:
    """
    Run all prompts in a tier n_runs times each.
    Returns a list of result records.
    """
    prompts = load_prompts(tier)
    records = []

    print(f"\nRunning tier: {tier} ({len(prompts)} prompts x {n_runs} runs)")

    for prompt_data in tqdm(prompts):
        for run_idx in range(n_runs):
            try:
                result = query_claude(prompt_data["prompt"])
                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "tier": tier,
                    "prompt_id": prompt_data["id"],
                    "topic": prompt_data["topic"],
                    "run_index": run_idx,
                    "prompt_text": prompt_data["prompt"],
                    "response_text": result["response_text"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_seconds": result["latency_seconds"],
                    "stop_reason": result["stop_reason"],
                    "expected_confidence_level": prompt_data.get("expected_confidence_level", ""),
                    "ground_truth_notes": prompt_data.get("ground_truth_notes", ""),
                    # Scoring columns -- filled in by scorer.py
                    "hedge_phrase_count": None,
                    "sycophancy_flag": None,
                    "confabulation_flag": None,
                    "manual_review_score": None,
                }
                records.append(record)
            except Exception as e:
                print(f"\nError on {prompt_data['id']} run {run_idx}: {e}")
                time.sleep(2)  # back off on error

    return records


def save_results(records: list[dict], tier: str) -> Path:
    """Save results to a timestamped CSV."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"{tier}_{timestamp}.csv"
    df = pd.DataFrame(records)
    df.to_csv(outpath, index=False)
    print(f"\nSaved {len(records)} records to {outpath}")
    return outpath


def main():
    parser = argparse.ArgumentParser(description="Run Claude biosafety evaluations")
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["high_confidence", "low_confidence", "adversarial", "all"],
        help="Which prompt tier to run (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS,
        help=f"Number of runs per prompt (default: {N_RUNS})",
    )
    args = parser.parse_args()

    tiers = (
        ["high_confidence", "low_confidence", "adversarial"]
        if args.tier == "all"
        else [args.tier]
    )

    all_records = []
    for tier in tiers:
        records = run_tier(tier, n_runs=args.runs)
        all_records.extend(records)
        save_results(records, tier)

    if args.tier == "all":
        save_results(all_records, "combined")

    print(f"\nDone. Total records: {len(all_records)}")


if __name__ == "__main__":
    main()
