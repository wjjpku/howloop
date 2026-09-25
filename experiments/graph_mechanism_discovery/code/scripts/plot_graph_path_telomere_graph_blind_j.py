from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "raw_mse": "Raw hidden MSE",
    "diag_whitened": "Diagonal-whitened hidden loss",
    "bias_only": "Bias only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    colors = {
        "raw_mse": "#2f6f9f",
        "diag_whitened": "#d97904",
        "bias_only": "#6c757d",
    }
    for result in payload["results"]:
        label = result["partition"]
        accuracy = result["curves"]["learned_J_plus_full_Block2"][
            "accuracy_by_cycle"
        ]
        axis.plot(
            range(1, len(accuracy) + 1),
            accuracy,
            label=LABELS[label],
            color=colors[label],
            linewidth=1.8,
        )
    reference = payload["results"][0]["curves"]
    oracle = reference["exact_H7_plus_full_Block2"]["accuracy_by_cycle"]
    no_j = reference["no_J_plus_full_Block2"]["accuracy_by_cycle"]
    axis.plot(
        range(1, len(oracle) + 1),
        oracle,
        label="Exact H7 oracle",
        color="#2a9d55",
        linewidth=2.2,
    )
    axis.plot(
        range(1, len(no_j) + 1),
        no_j,
        label="No J",
        color="#a6a6a6",
        linewidth=1.3,
        linestyle="--",
    )
    axis.axhline(
        0.125,
        color="#8b1e3f",
        linestyle=":",
        linewidth=1.5,
        label="Random baseline (1/8)",
    )
    for cycle in range(8, len(oracle) + 1, 8):
        axis.axvline(cycle, color="#eeeeee", linewidth=0.8, zorder=0)
    axis.set(
        title=(
            "Graph-blind affine J on strictly held-out single 8-cycles\n"
            "High values at multiples of 8 are orbit wrap-around, not progress"
        ),
        xlabel="Continuation loop",
        ylabel="Exact next-node accuracy",
        xlim=(1, len(oracle)),
        ylim=(-0.02, 1.03),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
