from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "baseline_b2": "baseline",
    "attn2_mlp05_b2": "attn LR 2x / MLP 0.5x",
    "block2fast_b2": "Block1 0.5x / Block2 1.5x",
    "beta2_099_b2": "Adam beta2=0.99",
    "beta1_08_b2": "Adam beta1=0.8",
    "layers1_b1": "1 block / loop",
    "layers3_b3": "3 blocks / loop",
}

ORDER = [
    "baseline_b2",
    "attn2_mlp05_b2",
    "block2fast_b2",
    "beta2_099_b2",
    "beta1_08_b2",
    "layers1_b1",
    "layers3_b3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = json.loads(
        (args.result_dir / "aggregate" / "aggregate.json").read_text()
    )
    with (args.result_dir / "aggregate" / "seed_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        seed_rows = list(csv.DictReader(handle))

    component = np.array(
        [
            aggregate["configs"][config]["component_loop_success_count"]
            for config in ORDER
        ],
        dtype=float,
    )
    portable = np.array(
        [
            aggregate["configs"][config][
                "portable_transition_success_count"
            ]
            for config in ORDER
        ],
        dtype=float,
    )
    matrix = np.zeros((len(ORDER), 5), dtype=float)
    for row in seed_rows:
        config = row["config"]
        if config not in ORDER:
            continue
        seed = int(row["name"].rsplit("seed", 1)[1])
        matrix[ORDER.index(config), seed] = (
            1.0 if row["component_loop_pass"] == "True" else 0.0
        )

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.2),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    y = np.arange(len(ORDER))
    height = 0.34
    left.barh(
        y - height / 2,
        component,
        height,
        color="#277da1",
        label="Gate B+C+D",
    )
    left.barh(
        y + height / 2,
        portable,
        height,
        color="#90be6d",
        label="portable transition (Gate C)",
    )
    left.axvline(2, color="#333333", linestyle="--", linewidth=1)
    left.set(
        yticks=y,
        yticklabels=[DISPLAY_NAMES[config] for config in ORDER],
        xlim=(0, 5),
        xlabel="successful seeds out of 5",
        title="Component-level success frequency",
    )
    left.invert_yaxis()
    left.set_xticks(range(6))
    left.grid(axis="x", alpha=0.2)
    left.legend(loc="lower right", frameon=False)

    right.imshow(
        matrix,
        cmap=plt.matplotlib.colors.ListedColormap(["#eeeeee", "#277da1"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    right.set(
        xticks=np.arange(5),
        xticklabels=[f"seed {seed}" for seed in range(5)],
        yticks=np.arange(len(ORDER)),
        yticklabels=[DISPLAY_NAMES[config] for config in ORDER],
        title="Paired Gate B+C+D outcomes",
    )
    right.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    right.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
    right.grid(which="minor", color="white", linewidth=2)
    right.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(len(ORDER)):
        for seed in range(5):
            right.text(
                seed,
                row_index,
                "pass" if matrix[row_index, seed] else "—",
                ha="center",
                va="center",
                color="white" if matrix[row_index, seed] else "#777777",
                fontsize=8,
                fontweight="bold" if matrix[row_index, seed] else "normal",
            )

    fig.suptitle(
        "Graph permutation composition: d64, D=6, L=6, final-only loss",
        fontsize=13,
    )
    fig.tight_layout()
    output = args.result_dir / "hparam_component_frequency.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
