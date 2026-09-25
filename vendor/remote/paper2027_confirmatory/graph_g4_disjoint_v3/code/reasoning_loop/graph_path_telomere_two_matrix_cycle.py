"""Test the phase-matched cycle F,F,J1,J2 on frozen D8L8.

J1 and J2 are distinct position-shared affine maps:

    J1: H8(current) -> H7(current)
    J2: H7(current) -> H6(current)

The closed-loop trajectory starts at H6, executes two full shared stacks F,
then applies J1 followed immediately by J2, and repeats.  This is deliberately
different from applying a power of one shared J.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_transition_matrices import (
    affine_metrics,
    fit_affine,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine, run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)


Affine = tuple[torch.Tensor, torch.Tensor]
Reset = Callable[[torch.Tensor], torch.Tensor]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate F,F,J1,J2 phase cycling.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--canonical-artifact", type=Path, required=True)
    parser.add_argument("--canonical-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration-examples", type=int, default=1024)
    parser.add_argument("--heldout-examples", type=int, default=1024)
    parser.add_argument("--evaluation-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--continuation-loops", type=int, default=128)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_interfaces(
    *,
    model: torch.nn.Module,
    cfg: Any,
    phase_positions: list[int],
    device: torch.device,
    examples: int,
    batch_size: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch size")
    set_seed(seed)
    chunks: dict[int, list[torch.Tensor]] = {6: [], 7: [], 8: []}
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        current = path_targets[:, cfg.max_depth - 1]
        for age in chunks:
            chunks[age].append(
                _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=age,
                    phase_position=phase_positions[age],
                ).float()
            )
    return {age: torch.cat(values) for age, values in chunks.items()}


def vector_affine(affine: Affine) -> VectorAffine:
    weight, bias = affine
    return VectorAffine(
        weight=weight,
        bias=bias,
        update_rank=weight.shape[0],
        fit_dimension=weight.shape[0],
        retained_fit_energy=1.0,
    )


def compressed_operator(
    affine: Affine,
    *,
    rank: int,
    gauge_seed: int,
) -> tuple[DiagonalIdentityLoRAJ, float]:
    module = DiagonalIdentityLoRAJ(
        dimension=affine[0].shape[0], rank=rank
    ).to(affine[0].device)
    retained = module.initialize_from_affine_svd(
        vector_affine(affine), gauge_seed=gauge_seed
    )
    return module.frozen(), retained


def affine_from_module(module: DiagonalIdentityLoRAJ) -> Affine:
    return (
        torch.diag(module.diagonal_scale.float())
        + module.A.float() @ module.B.float(),
        module.bias.float(),
    )


def compose(first: Affine, second: Affine) -> Affine:
    """Return second(first(x)) for row-vector affine maps."""
    return first[0] @ second[0], first[1] @ second[0] + second[1]


def apply_affine(state: torch.Tensor, affine: Affine) -> torch.Tensor:
    return state.float() @ affine[0] + affine[1]


def fit_rows(
    heldout: dict[int, torch.Tensor],
    maps: dict[str, Affine],
) -> list[dict[str, Any]]:
    pairs = {
        "J1_H8_to_H7": (8, 7),
        "J2_H7_to_H6": (7, 6),
        "pair_H8_to_H6": (8, 6),
        "J1_wrong_H7_to_H6": (7, 6),
        "J2_wrong_H8_to_H7": (8, 7),
    }
    selected_map = {
        "J1_H8_to_H7": "J1",
        "J2_H7_to_H6": "J2",
        "pair_H8_to_H6": "J1_then_J2",
        "J1_wrong_H7_to_H6": "J1",
        "J2_wrong_H8_to_H7": "J2",
    }
    rows = []
    for test, (source_age, target_age) in pairs.items():
        metrics = affine_metrics(
            heldout[source_age],
            heldout[target_age],
            *maps[selected_map[test]],
        )
        rows.append(
            {
                "test": test,
                "map": selected_map[test],
                "source_age": source_age,
                "target_age": target_age,
                **metrics,
            }
        )
    return rows


def rolling_last(values: list[float], threshold: float, window: int = 8) -> int:
    rolling = np.convolve(np.asarray(values), np.ones(window) / window, mode="valid")
    selected = np.nonzero(rolling >= threshold)[0]
    return int(selected[-1] + window) if selected.size else 0


def segment(values: list[float], start: int, stop: int) -> float | None:
    selected = values[start:stop]
    return float(np.mean(selected)) if selected else None


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    cfg: Any,
    phase_positions: list[int],
    device: torch.device,
    resets: dict[str, tuple[Reset | None, int]],
    examples: int,
    batch_size: int,
    continuation_loops: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if examples % batch_size:
        raise ValueError("evaluation examples must divide batch size")
    set_seed(seed)
    correct = {
        label: torch.zeros(continuation_loops, dtype=torch.long)
        for label in resets
    }
    total = 0
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        states = {
            label: _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=endpoint,
                age=start_age,
                phase_position=phase_positions[start_age],
            )
            for label, (_, start_age) in resets.items()
        }
        for cycle in range(1, continuation_loops + 1):
            target = advance_nodes(successors, endpoint, steps=cycle)
            for label, (reset, _) in resets.items():
                step = run_one_loop(
                    model,
                    states[label],
                    loop_index=cfg.max_loops + cycle - 1,
                )
                correct[label][cycle - 1] += step.logits.argmax(dim=-1).eq(target).sum().cpu()
                state = step.state
                if cycle % 2 == 0:
                    if label == "exact_H6_every2":
                        state = _aligned_state_at_age(
                            model=model,
                            cfg=cfg,
                            successors=successors,
                            current=target,
                            age=6,
                            phase_position=phase_positions[6],
                        )
                    elif reset is not None:
                        state = reset(state)
                states[label] = state
        total += batch_size
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label in resets:
        values = (correct[label].float() / total).tolist()
        curve_rows.extend(
            {
                "condition": label,
                "cycle": cycle,
                "accuracy": value,
                "examples": total,
            }
            for cycle, value in enumerate(values, start=1)
        )
        summary_rows.append(
            {
                "condition": label,
                "auc_1_24": segment(values, 0, 24),
                "auc_25_48": segment(values, 24, 48),
                "auc_49_64": segment(values, 48, 64),
                "auc_65_96": segment(values, 64, 96),
                "auc_97_128": segment(values, 96, 128),
                "final_accuracy": values[-1],
                "last_rolling8_at_least_90": rolling_last(values, 0.9),
                "last_rolling8_at_least_50": rolling_last(values, 0.5),
                "odd_intermediate_accuracy_mean": float(np.mean(values[0::2])),
                "even_macro_readout_accuracy_mean": float(np.mean(values[1::2])),
                "even_macro_auc_first_16": segment(values[1::2], 0, 16),
                "even_macro_auc_17_32": segment(values[1::2], 16, 32),
                "even_macro_auc_33_48": segment(values[1::2], 32, 48),
                "even_macro_auc_49_64": segment(values[1::2], 48, 64),
                "last_even_loop_rolling8_at_least_90": 2
                * rolling_last(values[1::2], 0.9),
                "last_even_loop_rolling8_at_least_50": 2
                * rolling_last(values[1::2], 0.5),
            }
        )
    return curve_rows, summary_rows


def draw(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = list(dict.fromkeys(row["condition"] for row in rows))
    values = np.asarray(
        [
            [row["accuracy"] for row in rows if row["condition"] == label]
            for label in labels
        ]
    )
    fig, axis = plt.subplots(figsize=(15, 5.8), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("continuation loop")
    axis.set_title("F,F,J1,J2: accuracy by cycle")
    fig.colorbar(image, ax=axis, label="accuracy")
    fig.savefig(out_dir / "01_two_matrix_cycle.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase_payload["trajectory_positions_including_initial"]
    ]
    calibration = collect_interfaces(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        examples=args.calibration_examples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    heldout = collect_interfaces(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        examples=args.heldout_examples,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )
    full_j1 = fit_affine(calibration[8], calibration[7], args.ridge)
    full_j2 = fit_affine(calibration[7], calibration[6], args.ridge)
    diagonal_j1, retained_j1 = compressed_operator(
        full_j1, rank=args.rank, gauge_seed=args.seed + 101
    )
    diagonal_j2, retained_j2 = compressed_operator(
        full_j2, rank=args.rank, gauge_seed=args.seed + 102
    )
    diag_j1 = affine_from_module(diagonal_j1)
    diag_j2 = affine_from_module(diagonal_j2)
    maps = {
        "J1": diag_j1,
        "J2": diag_j2,
        "J1_then_J2": compose(diag_j1, diag_j2),
    }
    mapping_rows = [
        {"parameterization": "diagonal_rank", **row}
        for row in fit_rows(heldout, maps)
    ]
    full_maps = {
        "J1": full_j1,
        "J2": full_j2,
        "J1_then_J2": compose(full_j1, full_j2),
    }
    mapping_rows.extend(
        {"parameterization": "full_affine", **row}
        for row in fit_rows(heldout, full_maps)
    )
    sequential = apply_affine(apply_affine(heldout[8], diag_j1), diag_j2)
    composed = apply_affine(heldout[8], maps["J1_then_J2"])
    composition_relative_error = float(
        (sequential - composed).norm() / sequential.norm().clamp_min(1e-12)
    )

    artifact_checkpoint, positions, canonical_modules, artifact_payload = load_task_lora_modules(
        args.canonical_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("canonical J and backbone checkpoint differ")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("expected all-position canonical J")
    canonical = canonical_modules[args.canonical_label]
    resets: dict[str, tuple[Reset | None, int]] = {
        "no_reset_start_H6": (None, 6),
        "exact_H6_every2": (None, 6),
        "full_J1_J2": (
            lambda state: apply_affine(apply_affine(state, full_j1), full_j2),
            6,
        ),
        "diag_J1_J2": (
            lambda state: diagonal_j2(diagonal_j1(state)),
            6,
        ),
        "diag_product_once": (
            lambda state: apply_affine(state, maps["J1_then_J2"]),
            6,
        ),
        "diag_J2_J1_wrong_order": (
            lambda state: diagonal_j1(diagonal_j2(state)),
            6,
        ),
        "diag_J1_J1": (
            lambda state: diagonal_j1(diagonal_j1(state)),
            6,
        ),
        "diag_J2_J2": (
            lambda state: diagonal_j2(diagonal_j2(state)),
            6,
        ),
        "canonical_J_J": (
            lambda state: canonical(canonical(state)),
            6,
        ),
        "diag_J1_J2_wrong_start_H8": (
            lambda state: diagonal_j2(diagonal_j1(state)),
            8,
        ),
    }
    # The artifact contains three independently optimized matrices for the
    # same frozen seed0 model.  Ordered pairs test whether alternating two
    # genuinely different trained Js changes drift even without phase-specific
    # training.  These are not mislabeled as phase specialists.
    for first_label, first_module in canonical_modules.items():
        for second_label, second_module in canonical_modules.items():
            first_seed = first_label.rsplit("seed", 1)[-1]
            second_seed = second_label.rsplit("seed", 1)[-1]
            resets[f"trained_{first_seed}_then_{second_seed}"] = (
                lambda state, first=first_module, second=second_module: second(
                    first(state)
                ),
                6,
            )
    curve_rows, summary_rows = evaluate(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        resets=resets,
        examples=args.evaluation_examples,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        seed=args.seed + 2,
    )
    write_csv(args.out_dir / "mapping_metrics.csv", mapping_rows)
    write_csv(args.out_dir / "two_matrix_curves.csv", curve_rows)
    write_csv(args.out_dir / "two_matrix_summary.csv", summary_rows)
    draw(args.out_dir, curve_rows)
    torch.save(
        {
            "kind": "graph_path_two_matrix_cycle",
            "checkpoint": str(args.checkpoint),
            "rank": args.rank,
            "J1_role": "H8_to_H7_same_graph_current_position",
            "J2_role": "H7_to_H6_same_graph_current_position",
            "J1_state_dict": {
                key: value.cpu() for key, value in diagonal_j1.state_dict().items()
            },
            "J2_state_dict": {
                key: value.cpu() for key, value in diagonal_j2.state_dict().items()
            },
        },
        args.out_dir / "J1_J2.pt",
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": artifact_payload.get("backbone_loss_description", "not recorded"),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "cycle": "start H6; F,F,J1(H8->H7),J2(H7->H6); repeat",
        "calibration_examples": args.calibration_examples,
        "heldout_examples": args.heldout_examples,
        "evaluation_examples": args.evaluation_examples,
        "rank": args.rank,
        "retained_offdiagonal_energy": {"J1": retained_j1, "J2": retained_j2},
        "composition_relative_error": composition_relative_error,
        "mapping_metrics": mapping_rows,
        "closed_loop_metrics": summary_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
