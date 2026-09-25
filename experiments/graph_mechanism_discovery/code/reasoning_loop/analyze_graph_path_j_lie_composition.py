"""Test whether product-trained affine J maps form a smooth shared flow.

The static part embeds every row-vector affine map h -> hW + b into a
homogeneous (d+1)x(d+1) matrix.  It measures invertibility, principal matrix
logs, BCH approximations, and non-commutativity.  The causal part swaps two
adjacent rollback maps and asks whether two subsequent backbone loops still
produce the correct task answer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_composition_laws import (
    forward_steps,
    load_bank,
    state_metrics,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


AGES = tuple(range(2, 9))
PAIR_SOURCE_AGES = tuple(range(3, 9))
FUNCTIONAL_MODES = ("ordered", "swapped", "first_twice", "second_twice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=829001)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def homogeneous(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Homogeneous matrix for row vectors: [h,1] M = [hW+b,1]."""
    dimension = weight.shape[0]
    result = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    result[:dimension, :dimension] = weight
    result[dimension, :dimension] = bias
    result[dimension, dimension] = 1.0
    return result


def affine_arrays(bank, age: int) -> tuple[np.ndarray, np.ndarray]:
    weight, bias = bank.affine(age)
    return (
        weight.detach().cpu().double().numpy(),
        bias.detach().cpu().double().numpy(),
    )


def relative_norm(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(value.norm() / reference.norm().clamp_min(1e-30))


def imaginary_ratio(value: torch.Tensor) -> float:
    return float(value.imag.norm() / value.norm().clamp_min(1e-30))


def commutator(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right - right @ left


def principal_log_via_eigendecomposition(matrix: torch.Tensor) -> torch.Tensor:
    """Principal log for a diagonalizable matrix, checked by reconstruction."""
    eigenvalues, eigenvectors = torch.linalg.eig(matrix.to(torch.complex128))
    return (
        eigenvectors
        @ torch.diag(torch.log(eigenvalues))
        @ torch.linalg.inv(eigenvectors)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def static_analysis(bank, *, device: torch.device) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, torch.Tensor],
    dict[str, float],
]:
    matrices: dict[int, torch.Tensor] = {}
    logs: dict[int, torch.Tensor] = {}
    stage_rows: list[dict[str, Any]] = []
    for age in AGES:
        weight, bias = affine_arrays(bank, age)
        matrix = torch.as_tensor(
            homogeneous(weight, bias), device=device, dtype=torch.float64
        )
        generator = principal_log_via_eigendecomposition(matrix)
        reconstructed = torch.matrix_exp(generator)
        singular = np.linalg.svd(weight, compute_uv=False)
        determinant_sign, log_absolute_determinant = np.linalg.slogdet(weight)
        eigenvalues = np.linalg.eigvals(weight)
        negative_real_eigenvalues = int(
            np.sum((np.abs(eigenvalues.imag) < 1e-8) & (eigenvalues.real < 0))
        )
        matrices[age] = matrix.to(torch.complex128)
        logs[age] = generator
        stage_rows.append(
            {
                "source_age": age,
                "minimum_singular_value": float(singular[-1]),
                "maximum_singular_value": float(singular[0]),
                "condition_number": float(singular[0] / singular[-1]),
                "determinant_sign": float(determinant_sign),
                "log_absolute_determinant": float(log_absolute_determinant),
                "negative_real_eigenvalue_count": negative_real_eigenvalues,
                "principal_log_imaginary_ratio": imaginary_ratio(generator),
                "principal_log_reconstruction_error": relative_norm(
                    reconstructed - matrix, matrix
                ),
                "distance_from_identity": relative_norm(
                    matrix - torch.eye(matrix.shape[0], device=device, dtype=matrix.dtype),
                    torch.eye(matrix.shape[0], device=device, dtype=matrix.dtype),
                ),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for source_age in PAIR_SOURCE_AGES:
        next_age = source_age - 1
        left = matrices[source_age]
        right = matrices[next_age]
        ordered = left @ right
        swapped = right @ left
        generator_left = logs[source_age]
        generator_right = logs[next_age]
        first_commutator = commutator(generator_left, generator_right)
        bch1 = generator_left + generator_right
        bch2 = bch1 + 0.5 * first_commutator
        bch3 = bch2 + (
            commutator(generator_left, first_commutator)
            + commutator(generator_right, -first_commutator)
        ) / 12.0
        approximation_rows: dict[str, float] = {}
        for name, approximation_generator in (
            ("log_add", bch1),
            ("bch_order2", bch2),
            ("bch_order3", bch3),
        ):
            approximation = torch.matrix_exp(approximation_generator)
            approximation_rows[f"{name}_relative_error"] = relative_norm(
                approximation - ordered, ordered
            )
            approximation_rows[f"{name}_imaginary_ratio"] = imaginary_ratio(
                approximation
            )
        pair_rows.append(
            {
                "first_source_age": source_age,
                "second_source_age": next_age,
                "actual_matrix_commutator_ratio": relative_norm(
                    ordered - swapped, ordered
                ),
                "log_commutator_ratio": relative_norm(
                    first_commutator, bch1
                ),
                **approximation_rows,
            }
        )
    dimension = next(iter(matrices.values())).shape[0]
    identity = torch.eye(dimension, device=device, dtype=torch.complex128)

    def energy_fractions(values: list[torch.Tensor], prefix: str) -> dict[str, float]:
        flattened = torch.stack([value.reshape(-1) for value in values])
        singular = torch.linalg.svdvals(flattened)
        energy = singular.square()
        total = energy.sum().clamp_min(1e-30)
        return {
            f"{prefix}_rank1_energy": float(energy[:1].sum() / total),
            f"{prefix}_rank2_energy": float(energy[:2].sum() / total),
            f"{prefix}_rank3_energy": float(energy[:3].sum() / total),
        }

    shared_direction_metrics = {
        **energy_fractions(
            [matrices[age] - identity for age in AGES], "matrix_delta"
        ),
        **energy_fractions([logs[age] for age in AGES], "principal_log"),
    }
    return stage_rows, pair_rows, matrices, shared_direction_metrics


def apply_sequence(bank, state: torch.Tensor, ages: tuple[int, int]) -> torch.Tensor:
    positions = tuple(range(state.shape[1]))
    for age in ages:
        state = bank.rollback(state, source_age=age, positions=positions)
    return state


@torch.no_grad()
def functional_swap_analysis(
    *,
    model,
    cfg,
    bank,
    phase_positions: list[int],
    device: torch.device,
    examples: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch size")
    set_seed(seed)
    storage: dict[int, dict[str, list[torch.Tensor]]] = {
        age: {mode: [] for mode in FUNCTIONAL_MODES}
        | {"current": [], "successors": []}
        for age in PAIR_SOURCE_AGES
    }
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        current = path_targets[:, cfg.max_depth - 1]
        for source_age in PAIR_SOURCE_AGES:
            source = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=source_age,
                phase_position=phase_positions[source_age],
            )
            sequences = {
                "ordered": (source_age, source_age - 1),
                "swapped": (source_age - 1, source_age),
                "first_twice": (source_age, source_age),
                "second_twice": (source_age - 1, source_age - 1),
            }
            for mode, ages in sequences.items():
                storage[source_age][mode].append(
                    apply_sequence(bank, source, ages).cpu()
                )
            storage[source_age]["current"].append(current.cpu())
            storage[source_age]["successors"].append(successors.cpu())

    model_cpu = model.cpu().eval()
    jump = phase_positions[2] - phase_positions[1]
    rows: list[dict[str, Any]] = []
    for source_age, values in storage.items():
        current = torch.cat(values["current"])
        successors = torch.cat(values["successors"])
        ordered = torch.cat(values["ordered"])
        target = advance_nodes(successors, current, steps=2 * jump)
        mode_ages = {
            "ordered": (source_age, source_age - 1),
            "swapped": (source_age - 1, source_age),
            "first_twice": (source_age, source_age),
            "second_twice": (source_age - 1, source_age - 1),
        }
        for mode in FUNCTIONAL_MODES:
            state = torch.cat(values[mode])
            immediate = logits_from_raw_state(model_cpu, state).argmax(-1)
            roundtrip_state = forward_steps(
                model_cpu, state, start_age=source_age - 2, steps=2
            )
            roundtrip = logits_from_raw_state(model_cpu, roundtrip_state).argmax(-1)
            metrics = state_metrics(state, ordered)
            rows.append(
                {
                    "source_age": source_age,
                    "first_J": mode_ages[mode][0],
                    "second_J": mode_ages[mode][1],
                    "mode": mode,
                    "state_relative_error_to_ordered": metrics["relative_error"],
                    "state_r2_to_ordered": metrics["r2"],
                    "immediate_current_accuracy": float(
                        immediate.eq(current).float().mean()
                    ),
                    "roundtrip_task_accuracy": float(
                        roundtrip.eq(target).float().mean()
                    ),
                    "examples": examples,
                }
            )
    return rows


def make_plot(
    stage_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    functional_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ages = [row["source_age"] for row in stage_rows]
    axes[0, 0].semilogy(
        ages,
        [row["minimum_singular_value"] for row in stage_rows],
        "o-",
        label="minimum singular value",
    )
    axes[0, 0].semilogy(
        ages,
        [row["condition_number"] for row in stage_rows],
        "s-",
        label="condition number",
    )
    axes[0, 0].set_title("J is strongly contracting / ill-conditioned")
    axes[0, 0].set_xlabel("source age")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        ages,
        [row["principal_log_imaginary_ratio"] for row in stage_rows],
        "o-",
        label="imaginary fraction of principal log",
    )
    axes[0, 1].axhline(0, color="black", linewidth=0.7)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("A real one-parameter flow is not a good fit")
    axes[0, 1].set_xlabel("source age")
    axes[0, 1].set_ylabel("fraction")

    pair_ages = [row["first_source_age"] for row in pair_rows]
    for key, label in (
        ("log_add_relative_error", "exp(Gi+Gj)"),
        ("bch_order2_relative_error", "BCH order 2"),
        ("bch_order3_relative_error", "BCH order 3"),
    ):
        axes[1, 0].plot(
            pair_ages, [row[key] for row in pair_rows], "o-", label=label
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Low-order BCH approximation error")
    axes[1, 0].set_xlabel("first source age in Ji Ji-1")
    axes[1, 0].set_ylabel("relative matrix error")
    axes[1, 0].legend(fontsize=8)

    for mode in FUNCTIONAL_MODES:
        selected = [row for row in functional_rows if row["mode"] == mode]
        axes[1, 1].plot(
            [row["source_age"] for row in selected],
            [row["roundtrip_task_accuracy"] for row in selected],
            "o-",
            label=mode,
        )
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_title("Causal order test: rollback twice, then F twice")
    axes[1, 1].set_xlabel("starting source age")
    axes[1, 1].set_ylabel("task accuracy")
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("D8L8 seed0: is product-J a smooth compositional flow?", fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    bank, bank_payload = load_bank(
        args.bank_artifact, dimension=cfg.d_model, device=device
    )
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase["trajectory_positions_including_initial"]
    ]
    stage_rows, pair_rows, _, shared_direction_metrics = static_analysis(
        bank, device=device
    )
    functional_rows = functional_swap_analysis(
        model=model,
        cfg=cfg,
        bank=bank,
        phase_positions=phase_positions,
        device=device,
        examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    write_csv(args.out_dir / "stage_log_metrics.csv", stage_rows)
    write_csv(args.out_dir / "pair_bch_metrics.csv", pair_rows)
    write_csv(args.out_dir / "functional_order_metrics.csv", functional_rows)
    make_plot(
        stage_rows,
        pair_rows,
        functional_rows,
        args.out_dir / "lie_composition_summary.png",
    )
    ordered_accuracy = float(
        np.mean(
            [
                row["roundtrip_task_accuracy"]
                for row in functional_rows
                if row["mode"] == "ordered"
            ]
        )
    )
    swapped_accuracy = float(
        np.mean(
            [
                row["roundtrip_task_accuracy"]
                for row in functional_rows
                if row["mode"] == "swapped"
            ]
        )
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "J_training_loss": bank_payload.get("training_loss"),
        "rollback_composition": bank_payload.get("rollback_composition") or "product",
        "examples_per_pair": args.examples,
        "principal_log_imaginary_ratio_mean": float(
            np.mean([row["principal_log_imaginary_ratio"] for row in stage_rows])
        ),
        "minimum_singular_value_mean": float(
            np.mean([row["minimum_singular_value"] for row in stage_rows])
        ),
        "negative_determinant_stage_count": int(
            np.sum([row["determinant_sign"] < 0 for row in stage_rows])
        ),
        "ordered_roundtrip_accuracy_mean": ordered_accuracy,
        "swapped_roundtrip_accuracy_mean": swapped_accuracy,
        "order_accuracy_drop": ordered_accuracy - swapped_accuracy,
        **shared_direction_metrics,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
