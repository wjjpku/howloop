"""Track compact seven-J matrix geometry across curriculum checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


AGES = tuple(range(2, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", required=True, help="label=artifact.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def materialize(payload: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    state = payload["state_dict"]
    architecture = payload["map_architecture"]
    if architecture != "shared_diagonal_stage_lora":
        raise ValueError("trajectory audit expects shared_diagonal_stage_lora")
    diagonal = torch.diag(state["shared_diagonal_scale"].double())
    shared = state["shared_A"].double() @ state["shared_B"].double()
    weights = {
        age: (
            diagonal
            + shared
            + state[f"stage_A.{age}"].double() @ state[f"stage_B.{age}"].double()
        ).numpy()
        for age in AGES
    }
    bias = state["shared_bias"].double().numpy()
    return weights, {age: bias.copy() for age in AGES}


def augmented(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    result = np.eye(weight.shape[0] + 1)
    result[:-1, :-1] = weight
    result[-1, :-1] = bias
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for item in args.bank:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        weights, biases = materialize(payload)
        loaded.append((label, path, payload, weights, biases))
    reference_state = loaded[0][2]["state_dict"]
    identity = np.eye(next(iter(loaded[0][3].values())).shape[0])
    checkpoint_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    for label, path, payload, weights, biases in loaded:
        state = payload["state_dict"]
        deltas = np.stack(
            [
                np.vstack((weights[age] - identity, biases[age][None, :])).reshape(-1)
                for age in AGES
            ]
        )
        singular = np.linalg.svd(deltas, compute_uv=False)
        cosines = deltas @ deltas.T / np.outer(
            np.linalg.norm(deltas, axis=1), np.linalg.norm(deltas, axis=1)
        )
        commutators = []
        augmented_maps = {age: augmented(weights[age], biases[age]) for age in AGES}
        augmented_identity = np.eye(identity.shape[0] + 1)
        for left_index, left_age in enumerate(AGES):
            for right_age in AGES[left_index + 1 :]:
                left, right = augmented_maps[left_age], augmented_maps[right_age]
                commutators.append(
                    np.linalg.norm(left @ right - right @ left, "fro")
                    / (
                        np.linalg.norm(left - augmented_identity, "fro")
                        * np.linalg.norm(right - augmented_identity, "fro")
                    )
                )
        one_step_singular = np.concatenate(
            [np.linalg.svd(weights[age], compute_uv=False) for age in AGES]
        )
        relative_change_numerator = 0.0
        relative_change_denominator = 0.0
        for key, value in state.items():
            current = value.double()
            reference = reference_state[key].double()
            relative_change_numerator += float((current - reference).square().sum())
            relative_change_denominator += float(reference.square().sum())
        checkpoint_rows.append(
            {
                "checkpoint": label,
                "path": str(path),
                "parameter_relative_change_from_parent": float(
                    np.sqrt(relative_change_numerator / relative_change_denominator)
                ),
                "diagonal_mean": float(state["shared_diagonal_scale"].double().mean()),
                "diagonal_std": float(state["shared_diagonal_scale"].double().std(unbiased=False)),
                "diagonal_min": float(state["shared_diagonal_scale"].double().min()),
                "diagonal_max": float(state["shared_diagonal_scale"].double().max()),
                "shared_update_norm": float(
                    (state["shared_A"].double() @ state["shared_B"].double()).norm()
                ),
                "stage_update_norm_mean": float(
                    np.mean(
                        [
                            float(
                                (
                                    state[f"stage_A.{age}"].double()
                                    @ state[f"stage_B.{age}"].double()
                                ).norm()
                            )
                            for age in AGES
                        ]
                    )
                ),
                "bias_norm": float(state["shared_bias"].double().norm()),
                "seven_map_shared_pc1_energy": float(singular[0] ** 2 / np.sum(singular**2)),
                "pairwise_delta_cosine_mean": float(
                    cosines[np.triu_indices_from(cosines, k=1)].mean()
                ),
                "affine_commutator_mean": float(np.mean(commutators)),
                "affine_commutator_max": float(np.max(commutators)),
                "one_step_singular_min": float(one_step_singular.min()),
                "one_step_singular_median": float(np.median(one_step_singular)),
                "one_step_singular_max": float(one_step_singular.max()),
            }
        )
        product = np.eye(identity.shape[0])
        product_bias = np.zeros(identity.shape[0])
        for rollback_steps, age in enumerate(reversed(AGES), start=1):
            product_bias = product_bias @ weights[age] + biases[age]
            product = product @ weights[age]
            spectrum = np.linalg.svd(product, compute_uv=False)
            product_rows.append(
                {
                    "checkpoint": label,
                    "rollback_steps": rollback_steps,
                    "source_ages": "-".join(str(value) for value in range(8, 8 - rollback_steps, -1)),
                    "bias_norm": float(np.linalg.norm(product_bias)),
                    "singular_min": float(spectrum[-1]),
                    "singular_q10": float(np.quantile(spectrum, 0.1)),
                    "singular_median": float(np.median(spectrum)),
                    "singular_q90": float(np.quantile(spectrum, 0.9)),
                    "singular_max": float(spectrum[0]),
                    "mean_log_singular": float(np.mean(np.log(spectrum.clip(1e-300)))),
                    "condition_number": float(spectrum[0] / spectrum[-1]),
                }
            )
    write_csv(args.out_dir / "checkpoint_geometry.csv", checkpoint_rows)
    write_csv(args.out_dir / "ordered_product_spectra.csv", product_rows)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    x = np.arange(len(checkpoint_rows))
    labels = [row["checkpoint"] for row in checkpoint_rows]
    axes[0, 0].plot(x, [row["seven_map_shared_pc1_energy"] for row in checkpoint_rows], "o-", label="shared PC1")
    axes[0, 0].plot(x, [row["pairwise_delta_cosine_mean"] for row in checkpoint_rows], "o-", label="mean cosine")
    axes[0, 0].set_title("common rollback direction")
    axes[0, 1].plot(x, [row["affine_commutator_mean"] for row in checkpoint_rows], "o-", label="mean")
    axes[0, 1].plot(x, [row["affine_commutator_max"] for row in checkpoint_rows], "o-", label="max")
    axes[0, 1].set_title("order sensitivity (normalized commutator)")
    for label in labels:
        selected = [row for row in product_rows if row["checkpoint"] == label]
        axes[1, 0].plot(
            [row["rollback_steps"] for row in selected],
            [row["singular_median"] for row in selected],
            "o-", label=label,
        )
        axes[1, 1].plot(
            [row["rollback_steps"] for row in selected],
            [row["singular_min"] for row in selected],
            "o-", label=label,
        )
    axes[1, 0].set_title("ordered-product median singular value")
    axes[1, 1].set_title("ordered-product minimum singular value")
    axes[1, 1].set_yscale("log")
    for axis in axes[0]:
        axis.set_xticks(x, labels, rotation=25, ha="right")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7)
    for axis in axes[1]:
        axis.set_xlabel("consecutive rollback maps, H8 downward")
    figure.tight_layout()
    figure.savefig(args.out_dir / "checkpoint_matrix_trajectory.png", bbox_inches="tight")
    plt.close(figure)
    result = {
        "status": "complete",
        "checkpoint_order": labels,
        "geometry": checkpoint_rows,
        "claim_boundary": (
            "Matrix spectra and commutators are state-independent operator diagnostics; "
            "they do not by themselves establish task-relevant functional equivalence."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
