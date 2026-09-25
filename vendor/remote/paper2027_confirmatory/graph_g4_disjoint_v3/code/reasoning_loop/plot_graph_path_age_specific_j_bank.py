"""Render the seven learned J maps and their functional controls."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--routing-summary", type=Path, required=True)
    parser.add_argument("--adjacent-summary", type=Path)
    parser.add_argument("--identity-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def matrix_bank(payload):
    state = payload["state_dict"]
    weights, biases, diagonals = {}, {}, {}
    for age in range(2, 9):
        prefix = f"maps.{age}."
        if payload.get("map_architecture", "diagonal_lora") == "full_affine":
            weights[age] = state[prefix + "weight"].float()
            diagonal = torch.diagonal(weights[age])
        else:
            diagonal = state[prefix + "diagonal_scale"].float()
            weights[age] = torch.diag(diagonal) + (
                state[prefix + "A"].float() @ state[prefix + "B"].float()
            )
        biases[age] = state[prefix + "bias"].float()
        diagonals[age] = diagonal
    return weights, biases, diagonals


def render_parameters(out_dir: Path, weights, biases, diagonals, architecture: str) -> None:
    identity = torch.eye(next(iter(weights.values())).shape[0])
    updates = [weight - identity for weight in weights.values()]
    bound = float(torch.quantile(torch.cat([x.abs().flatten() for x in updates]), 0.995))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    image = None
    for axis, age in zip(axes.flat, range(2, 9), strict=False):
        image = axis.imshow(
            (weights[age] - identity).T.numpy(),
            cmap="RdBu_r", vmin=-bound, vmax=bound,
            interpolation="nearest", aspect="auto",
        )
        axis.set_title(f"$J_{{{age}\\to {age-1}}}-I$")
        axis.set_xlabel("input dimension")
        axis.set_ylabel("output dimension")
    axes.flat[-1].axis("off")
    fig.colorbar(image, ax=list(axes.flat[:-1]), shrink=0.75, label="coefficient")
    fig.suptitle(f"Seven age-specific rollback matrices ({architecture})")
    fig.savefig(out_dir / "01_seven_J_heatmaps.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    for age in range(2, 9):
        axes[0].plot((diagonals[age] - 1).numpy(), lw=0.8, label=f"J{age}")
        axes[1].plot(biases[age].numpy(), lw=0.8, label=f"J{age}")
    axes[0].set_title("diagonal_scale - 1")
    axes[1].set_title("bias")
    for axis in axes:
        axis.axhline(0, color="black", lw=0.5)
        axis.set_xlabel("hidden dimension")
        axis.set_ylabel("coefficient")
        axis.legend(ncol=7, fontsize=8)
    fig.savefig(out_dir / "02_diagonal_and_bias_by_age.png", dpi=220)
    plt.close(fig)


def render_relations(out_dir: Path, weights) -> None:
    identity = torch.eye(next(iter(weights.values())).shape[0])
    ages = list(range(2, 9))
    cosine = np.eye(7)
    distance = np.zeros((7, 7))
    for left, right in itertools.product(ages, repeat=2):
        l, r = weights[left], weights[right]
        cosine[left - 2, right - 2] = float(
            torch.nn.functional.cosine_similarity(
                (l - identity).flatten()[None], (r - identity).flatten()[None]
            )
        )
        distance[left - 2, right - 2] = float(
            (l - r).norm() / ((l.norm() + r.norm()) / 2).clamp_min(1e-12)
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, values, title, cmap in (
        (axes[0], cosine, "cosine of J-I", "viridis"),
        (axes[1], distance, "relative function distance", "magma"),
    ):
        image = axis.imshow(values, cmap=cmap, aspect="equal")
        axis.set_xticks(range(7), ages)
        axis.set_yticks(range(7), ages)
        axis.set_xlabel("J source age")
        axis.set_ylabel("J source age")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(out_dir / "03_pairwise_J_relations.png", dpi=220)
    plt.close(fig)


def load_summary(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summary", payload.get("random_trajectory_summary", [])), payload.get(
        "products", payload.get("product_evaluation", [])
    )


def render_function(out_dir: Path, routing_path: Path, adjacent_path, identity_path) -> None:
    routing, canonical_products = load_summary(routing_path)
    conditions = ["learned", "wrong_stage", "reverse_stage", "shared_J8", "identity", "exact"]
    splits = ["train_like", "longer_unseen"]
    lookup = {(row["split"], row["condition"]): row for row in routing}
    x = np.arange(len(conditions))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for offset, split in zip((-0.18, 0.18), splits, strict=True):
        axes[0].bar(
            x + offset,
            [lookup[(split, c)]["accuracy_mean"] for c in conditions],
            width=0.36,
            label=split,
        )
        axes[1].bar(
            x + offset,
            [lookup[(split, c)]["cross_entropy_mean"] for c in conditions],
            width=0.36,
            label=split,
        )
    for axis, title, ylabel in (
        (axes[0], "Final accuracy on random mixed trajectories", "accuracy"),
        (axes[1], "Final CE on the same trajectories", "cross entropy"),
    ):
        axis.set_xticks(x, conditions, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend()
    fig.savefig(out_dir / "04_routing_controls.png", dpi=220)
    plt.close(fig)

    variants = [("canonical init", canonical_products)]
    if adjacent_path is not None:
        variants.append(("adjacent init", load_summary(adjacent_path)[1]))
    if identity_path is not None:
        variants.append(("identity init", load_summary(identity_path)[1]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for label, rows in variants:
        steps = sorted({int(row["rollback_steps"]) for row in rows})
        r2_means = [
            np.mean([row["r2"] for row in rows if int(row["rollback_steps"]) == step])
            for step in steps
        ]
        relative_means = [
            np.mean(
                [row["relative_error"] for row in rows if int(row["rollback_steps"]) == step]
            )
            for step in steps
        ]
        if label != "identity init":
            axes[0].plot(steps, r2_means, marker="o", label=label)
        axes[1].plot(steps, relative_means, marker="o", label=label)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylim(-0.05, 1.0)
    axes[0].set_ylabel("mean hidden-state $R^2$")
    axes[0].set_title("Compositional rollback geometry")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative hidden-state error (log scale)")
    axes[1].set_title("Cold-start products become unstable")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("number of multiplied rollback matrices")
    fig.savefig(out_dir / "05_product_compositionality.png", dpi=220)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.bank_artifact, map_location="cpu", weights_only=False)
    weights, biases, diagonals = matrix_bank(payload)
    render_parameters(
        args.out_dir,
        weights,
        biases,
        diagonals,
        payload.get("map_architecture", "diagonal_lora"),
    )
    render_relations(args.out_dir, weights)
    render_function(
        args.out_dir,
        args.routing_summary,
        args.adjacent_summary,
        args.identity_summary,
    )


if __name__ == "__main__":
    main(parse_args())
