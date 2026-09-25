"""Compare gauge-invariant matrix components of three seven-stage J banks."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


AGES = tuple(range(2, 9))


@dataclass
class BankMatrices:
    label: str
    path: Path
    diagonal: np.ndarray
    shared: np.ndarray
    stages: dict[int, np.ndarray]
    bias: np.ndarray
    weights: dict[int, np.ndarray]
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        action="append",
        nargs=2,
        metavar=("LABEL", "ARTIFACT"),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_bank(label: str, path: Path) -> BankMatrices:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("map_architecture") != "shared_diagonal_stage_lora":
        raise ValueError(f"{label} is not a shared/stage LoRA bank")
    state = payload["state_dict"]
    diagonal = np.diag(state["shared_diagonal_scale"].double().numpy())
    shared = (
        state["shared_A"].double().numpy()
        @ state["shared_B"].double().numpy()
    )
    stages = {
        age: (
            state[f"stage_A.{age}"].double().numpy()
            @ state[f"stage_B.{age}"].double().numpy()
        )
        for age in AGES
    }
    bias = state["shared_bias"].double().numpy()
    weights = {age: diagonal + shared + stages[age] for age in AGES}
    return BankMatrices(label, path, diagonal, shared, stages, bias, weights, payload)


def frobenius(value: np.ndarray) -> float:
    return float(np.linalg.norm(value))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    return float(left_flat @ right_flat / max(denominator, 1e-12))


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-12))


def svd_energy(value: np.ndarray) -> dict[str, float | int]:
    singular = np.linalg.svd(value, compute_uv=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-30)
    return {
        "rank90": int(np.searchsorted(cumulative, 0.90) + 1),
        "rank99": int(np.searchsorted(cumulative, 0.99) + 1),
        "top1_energy": float(cumulative[0]),
        "top8_energy": float(cumulative[min(7, len(cumulative) - 1)]),
    }


def compose(bank: BankMatrices, source_age: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    dimension = bank.diagonal.shape[0]
    weight = np.eye(dimension)
    bias = np.zeros(dimension)
    for source in range(source_age, source_age - steps, -1):
        next_weight = bank.weights[source]
        bias = bias @ next_weight + bank.bias
        weight = weight @ next_weight
    return weight, bias


def bank_metrics(bank: BankMatrices) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dimension = bank.diagonal.shape[0]
    identity = np.eye(dimension)
    diagonal_update = bank.diagonal - identity
    shared_update = diagonal_update + bank.shared
    full_updates = {age: bank.weights[age] - identity for age in AGES}
    stage_rows: list[dict[str, Any]] = []
    for age in AGES:
        singular = np.linalg.svd(bank.weights[age], compute_uv=False)
        update_svd = svd_energy(full_updates[age])
        stage_rows.append(
            {
                "bank": bank.label,
                "age": age,
                "diagonal_update_norm": frobenius(diagonal_update),
                "shared_AB_norm": frobenius(bank.shared),
                "stage_U_norm": frobenius(bank.stages[age]),
                "full_update_norm": frobenius(full_updates[age]),
                "bias_norm": frobenius(bank.bias),
                "minimum_singular_value": float(singular.min()),
                "maximum_singular_value": float(singular.max()),
                "condition_number": float(singular.max() / max(singular.min(), 1e-12)),
                **{f"update_{key}": value for key, value in update_svd.items()},
            }
        )
    pair_cosines = [
        cosine(full_updates[left], full_updates[right])
        for index, left in enumerate(AGES)
        for right in AGES[index + 1 :]
    ]
    commutators = []
    for index, left in enumerate(AGES):
        for right in AGES[index + 1 :]:
            commutator = bank.weights[left] @ bank.weights[right] - bank.weights[right] @ bank.weights[left]
            denominator = frobenius(full_updates[left]) * frobenius(full_updates[right])
            commutators.append(frobenius(commutator) / max(denominator, 1e-12))
    centered = np.stack([full_updates[age] for age in AGES])
    centered = centered.reshape(len(AGES), -1)
    centered -= centered.mean(axis=0, keepdims=True)
    stage_singular = np.linalg.svd(centered, compute_uv=False)
    stage_energy = stage_singular**2
    stage_cumulative = np.cumsum(stage_energy) / max(float(stage_energy.sum()), 1e-30)
    metrics = {
        "bank": bank.label,
        "artifact": str(bank.path),
        "rank": int(bank.payload["rank"]),
        "stage_rank": int(bank.payload["stage_rank"]),
        "parameter_count": int(bank.payload["parameter_count"]),
        "initialization": bank.payload.get("initialization"),
        "training_start_age": bank.payload.get("training_start_age"),
        "diagonal_update_norm": frobenius(diagonal_update),
        "shared_AB_norm": frobenius(bank.shared),
        "shared_update_norm": frobenius(shared_update),
        "mean_stage_U_norm": float(np.mean([frobenius(bank.stages[age]) for age in AGES])),
        "mean_full_update_norm": float(np.mean([frobenius(full_updates[age]) for age in AGES])),
        "bias_norm": frobenius(bank.bias),
        "stage_fraction_of_full_update": float(
            np.mean([frobenius(bank.stages[age]) / max(frobenius(full_updates[age]), 1e-12) for age in AGES])
        ),
        "shared_fraction_of_full_update": float(
            np.mean([frobenius(shared_update) / max(frobenius(full_updates[age]), 1e-12) for age in AGES])
        ),
        "pairwise_full_update_cosine_mean": float(np.mean(pair_cosines)),
        "pairwise_full_update_cosine_min": float(np.min(pair_cosines)),
        "normalized_commutator_mean": float(np.mean(commutators)),
        "normalized_commutator_max": float(np.max(commutators)),
        "stage_variation_top1_energy": float(stage_cumulative[0]),
        "stage_variation_top2_energy": float(stage_cumulative[1]),
        "stage_variation_top3_energy": float(stage_cumulative[2]),
    }
    return metrics, stage_rows


def comparisons(banks: list[BankMatrices]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    dimension = banks[0].diagonal.shape[0]
    identity = np.eye(dimension)
    for left_index, left in enumerate(banks):
        for right in banks[left_index + 1 :]:
            pair = f"{left.label} -> {right.label}"
            for age in AGES:
                left_update = left.weights[age] - identity
                right_update = right.weights[age] - identity
                delta = right.weights[age] - left.weights[age]
                energy = svd_energy(delta)
                stage_rows.append(
                    {
                        "comparison": pair,
                        "age": age,
                        "diagonal_relative_change": relative_error(right.diagonal, left.diagonal),
                        "shared_AB_relative_change": relative_error(right.shared, left.shared),
                        "stage_U_relative_change": relative_error(right.stages[age], left.stages[age]),
                        "stage_U_cosine": cosine(left.stages[age], right.stages[age]),
                        "full_weight_relative_change": relative_error(right.weights[age], left.weights[age]),
                        "full_update_relative_change": relative_error(right_update, left_update),
                        "full_update_cosine": cosine(left_update, right_update),
                        "bias_relative_change": relative_error(right.bias, left.bias),
                        **{f"delta_weight_{key}": value for key, value in energy.items()},
                    }
                )
            for steps in range(1, 6):
                left_weight, left_bias = compose(left, 8, steps)
                right_weight, right_bias = compose(right, 8, steps)
                product_rows.append(
                    {
                        "comparison": pair,
                        "steps": steps,
                        "product_weight_relative_change": relative_error(right_weight, left_weight),
                        "product_update_relative_change": relative_error(right_weight - identity, left_weight - identity),
                        "product_update_cosine": cosine(left_weight - identity, right_weight - identity),
                        "product_bias_relative_change": relative_error(right_bias, left_bias),
                    }
                )
    return stage_rows, product_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def component_figure(banks: list[BankMatrices], path: Path) -> None:
    dimension = banks[0].diagonal.shape[0]
    identity = np.eye(dimension)
    shared_bound = max(np.quantile(np.abs(bank.shared), 0.995) for bank in banks)
    stage_rms = {
        bank.label: np.sqrt(np.mean(np.stack([bank.stages[age] ** 2 for age in AGES]), axis=0))
        for bank in banks
    }
    stage_bound = max(np.quantile(value, 0.995) for value in stage_rms.values())
    figure, axes = plt.subplots(len(banks), 4, figsize=(18, 4.2 * len(banks)), constrained_layout=True)
    for row, bank in enumerate(banks):
        axes[row, 0].plot(np.diag(bank.diagonal - identity), linewidth=1)
        axes[row, 0].axhline(0, color="black", linewidth=0.6)
        axes[row, 0].set_title(f"{bank.label}: diagonal - 1")
        image_shared = axes[row, 1].imshow(bank.shared, cmap="RdBu_r", vmin=-shared_bound, vmax=shared_bound)
        axes[row, 1].set_title("shared AB")
        image_stage = axes[row, 2].imshow(stage_rms[bank.label], cmap="viridis", vmin=0, vmax=stage_bound)
        axes[row, 2].set_title("RMS of seven stage AiBi")
        axes[row, 3].plot(bank.bias, linewidth=1)
        axes[row, 3].axhline(0, color="black", linewidth=0.6)
        axes[row, 3].set_title("shared bias")
    figure.colorbar(image_shared, ax=axes[:, 1], shrink=0.75, label="coefficient")
    figure.colorbar(image_stage, ax=axes[:, 2], shrink=0.75, label="RMS coefficient")
    rank_labels = ", ".join(
        f"r{bank.payload['rank']}/s{bank.payload['stage_rank']}" for bank in banks
    )
    figure.suptitle(
        f"Gauge-invariant components of product J banks ({rank_labels})", fontsize=16
    )
    figure.savefig(path, dpi=185)
    plt.close(figure)


def stage_heatmaps(banks: list[BankMatrices], path: Path) -> None:
    dimension = banks[0].diagonal.shape[0]
    identity = np.eye(dimension)
    updates = [bank.weights[age] - identity for bank in banks for age in AGES]
    bound = max(np.quantile(np.abs(value), 0.997) for value in updates)
    figure, axes = plt.subplots(len(banks), len(AGES), figsize=(22, 3.5 * len(banks)), constrained_layout=True)
    for row, bank in enumerate(banks):
        for column, age in enumerate(AGES):
            axes[row, column].imshow(bank.weights[age] - identity, cmap="RdBu_r", vmin=-bound, vmax=bound)
            axes[row, column].set_title(f"{bank.label}\nJ{age} - I")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle("All seven learned rollback matrices under matched color scale", fontsize=16)
    figure.savefig(path, dpi=175)
    plt.close(figure)


def summary_figure(
    banks: list[BankMatrices],
    metrics: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    labels = [bank.label for bank in banks]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    names = ["diagonal_update_norm", "shared_AB_norm", "mean_stage_U_norm", "bias_norm"]
    x = np.arange(len(labels))
    width = 0.19
    for index, name in enumerate(names):
        axes[0, 0].bar(x + (index - 1.5) * width, [item[name] for item in metrics], width, label=name)
    axes[0, 0].set_xticks(x, labels, rotation=15)
    axes[0, 0].set_title("component Frobenius norms")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].bar(x - width / 2, [item["pairwise_full_update_cosine_mean"] for item in metrics], width, label="stage cosine")
    axes[0, 1].bar(x + width / 2, [item["normalized_commutator_mean"] for item in metrics], width, label="normalized commutator")
    axes[0, 1].set_xticks(x, labels, rotation=15)
    axes[0, 1].set_title("stage similarity and non-commutativity")
    axes[0, 1].legend(fontsize=8)
    for bank in banks:
        spectra = [np.linalg.svd(bank.weights[age] - np.eye(bank.diagonal.shape[0]), compute_uv=False) for age in AGES]
        axes[0, 2].plot(np.mean(spectra, axis=0)[:80], label=bank.label)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("mean spectrum of J_i - I")
    axes[0, 2].set_xlabel("singular-value index")
    axes[0, 2].legend(fontsize=8)
    pairs = sorted({row["comparison"] for row in comparison_rows})
    for pair in pairs:
        selected = [row for row in comparison_rows if row["comparison"] == pair]
        axes[1, 0].plot(AGES, [row["full_update_relative_change"] for row in selected], "o-", label=pair)
        axes[1, 1].plot(AGES, [row["full_update_cosine"] for row in selected], "o-", label=pair)
        axes[1, 2].plot(AGES, [row["delta_weight_rank90"] for row in selected], "o-", label=pair)
    axes[1, 0].set_title("relative change of each full update")
    axes[1, 1].set_title("cosine between each full update")
    axes[1, 2].set_title("rank needed for 90% of delta-W energy")
    for axis in axes[1]:
        axis.set_xlabel("rollback source age")
        axis.legend(fontsize=7)
    figure.suptitle("How WSD changes the seven fixed-H1 J matrices", fontsize=16)
    figure.savefig(path, dpi=185)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    banks = [load_bank(label, Path(path)) for label, path in args.bank]
    if len(banks) < 2:
        raise ValueError("at least two banks are required")
    dimensions = {bank.diagonal.shape[0] for bank in banks}
    if len(dimensions) != 1:
        raise ValueError("all banks must have the same hidden dimension")
    bank_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for bank in banks:
        metric, rows = bank_metrics(bank)
        bank_rows.append(metric)
        stage_rows.extend(rows)
    comparison_rows, product_rows = comparisons(banks)
    write_csv(args.out_dir / "bank_metrics.csv", bank_rows)
    write_csv(args.out_dir / "stage_metrics.csv", stage_rows)
    write_csv(args.out_dir / "bank_comparisons_by_stage.csv", comparison_rows)
    write_csv(args.out_dir / "product_comparisons.csv", product_rows)
    component_figure(banks, args.out_dir / "component_comparison.png")
    stage_heatmaps(banks, args.out_dir / "all_stage_update_heatmaps.png")
    summary_figure(banks, bank_rows, comparison_rows, args.out_dir / "matrix_comparison_summary.png")
    summary = {
        "status": "complete",
        "comparison_rule": "compare AB and AiBi products, never raw LoRA factors",
        "banks": bank_rows,
        "stage_comparisons": comparison_rows,
        "product_comparisons": product_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "banks": bank_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
