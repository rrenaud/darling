import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize(df: pd.DataFrame, eval_dir: str) -> dict:
    summary = {}

    summary["mean_distinct"] = np.mean(df["partition_scores"].map(len))
    summary["mean_quality"] = np.mean(df["quality"])
    summary["mean_utility"] = np.mean(df["utility"])

    summary["var_distinct"] = np.var(df["partition_scores"].map(len))
    summary["var_quality"] = np.var(df["quality"])
    summary["var_utility"] = np.var(df["utility"])

    # plot distribution of distinct and utility scores
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df["partition_scores"].map(len), bins=30, color='blue', alpha=0.7)
    plt.title("Distribution of Distinct Scores")
    plt.xlabel("Distinct Score")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    plt.hist(df["utility"], bins=30, color='green', alpha=0.7)
    plt.title("Distribution of Utility Scores")
    plt.xlabel("Utility Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f'{eval_dir}/distribution_plot.pdf')
    plt.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    df = pd.read_json(os.path.join(eval_dir, "scores.jsonl"), lines=True)
    summary = summarize(df, eval_dir)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
