"""Visualize how different J composition laws reshape compact-bank parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", required=True, help="label=artifact.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banks = []
    for item in args.bank:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        alpha = state["shared_diagonal_scale"].float()
        shared_update = state["shared_A"].float() @ state["shared_B"].float()
        stage_updates = {
            age: state[f"stage_A.{age}"].float() @ state[f"stage_B.{age}"].float()
            for age in range(2, 9)
        }
        effective_updates = {
            age: shared_update + stage_updates[age] for age in range(2, 9)
        }
        diagonal = torch.diag(alpha)
        cross_ratios = []
        for age in range(3, 9):
            first, second = effective_updates[age], effective_updates[age - 1]
            exact = (diagonal + first) @ (diagonal + second)
            cross_ratios.append(float((first @ second).norm() / exact.norm()))
        summary = json.loads((path.parent / "summary.json").read_text())
        learned = {
            row["split"]: float(row["accuracy_mean"])
            for row in summary["random_trajectory_summary"]
            if row["condition"] == "learned"
        }
        banks.append(
            {
                "label": label,
                "path": str(path),
                "alpha": alpha,
                "shared_update_norm": float(shared_update.norm()),
                "stage_update_norm_mean": float(
                    torch.stack([value.norm() for value in stage_updates.values()]).mean()
                ),
                "bias_norm": float(state["shared_bias"].float().norm()),
                "second_order_ratio_mean": float(np.mean(cross_ratios)),
                "single": learned["single_J"],
                "mixed": np.mean(
                    [learned["focused_mixture_a"], learned["focused_mixture_b"]]
                ),
                "hard": learned["hard_boundary_T05_R5"],
            }
        )

    product_alpha = banks[0]["alpha"]
    order = torch.argsort(product_alpha)
    rows = []
    for bank in banks:
        alpha = bank["alpha"]
        rows.append(
            {
                key: value
                for key, value in bank.items()
                if key not in {"alpha"}
            }
            | {
                "alpha_mean": float(alpha.mean()),
                "alpha_std": float(alpha.std(unbiased=False)),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "alpha_relative_change_from_product": float(
                    (alpha - product_alpha).norm() / product_alpha.norm()
                ),
                "alpha_cosine_with_product": float(
                    torch.nn.functional.cosine_similarity(alpha, product_alpha, dim=0)
                ),
            }
        )

    labels = [row["label"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for bank in banks:
        axes[0, 0].plot(
            bank["alpha"][order].numpy(), label=bank["label"], linewidth=1.5
        )
    axes[0, 0].set_title("shared diagonal alpha (sorted by product model)")
    axes[0, 0].set_xlabel("hidden dimension")
    axes[0, 0].set_ylabel("diagonal coefficient")
    axes[0, 0].legend(fontsize=8)

    x = np.arange(len(labels))
    width = 0.25
    axes[0, 1].bar(x - width, [row["shared_update_norm"] for row in rows], width, label="shared U")
    axes[0, 1].bar(x, [row["stage_update_norm_mean"] for row in rows], width, label="mean stage U")
    axes[0, 1].bar(x + width, [row["bias_norm"] for row in rows], width, label="bias")
    axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 1].set_title("component Frobenius/L2 norms")
    axes[0, 1].legend()

    axes[1, 0].bar(labels, [row["second_order_ratio_mean"] for row in rows])
    axes[1, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 0].set_ylabel("mean ||U_i U_k|| / ||W_i W_k||")
    axes[1, 0].set_title("size of omitted second-order interaction")

    axes[1, 1].bar(x - width, [row["single"] for row in rows], width, label="single")
    axes[1, 1].bar(x, [row["mixed"] for row in rows], width, label="mixed")
    axes[1, 1].bar(x + width, [row["hard"] for row in rows], width, label="five consecutive")
    axes[1, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 1].set_ylim(0.75, 1)
    axes[1, 1].set_title("accuracy under each bank's trained law")
    axes[1, 1].legend()
    figure.suptitle("Compact J parameter adaptation across composition laws")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190)
    plt.close(figure)
    args.json_output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
