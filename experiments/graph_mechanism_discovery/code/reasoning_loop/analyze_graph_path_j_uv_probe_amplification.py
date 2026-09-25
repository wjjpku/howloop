"""Freeze clean age probes and amplify the bottom-output directions of J.

The script separates two interventions that are easy to conflate:

1. ``coordinate_scale`` directly rescales the post-J state coordinates in the
   consensus bottom-output space.  Its effect on a linear probe is an analytic
   calibration check, not circuit evidence.
2. ``bottom_floor`` raises each stage J's bottom singular values, restoring the
   stage-specific U_bottom -> V_bottom channel.  Random rank-matched operators
   are rescaled on the clean training states to match the realized state RMS.

Both the full-state and consensus-V-bottom age probes are trained only on clean
training graph seeds, frozen, and evaluated on disjoint graph seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_anti_compression import (
    build_stage_delta,
    random_orthonormal,
)
from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    AGES,
    age_metrics,
    atomic_json,
    fit_ridge_regression,
    load_bank_and_bases,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--train-graph-seeds", type=int, nargs="+", default=(887101, 887102, 887103)
    )
    parser.add_argument(
        "--test-graph-seeds", type=int, nargs="+", default=(887104, 887105)
    )
    parser.add_argument("--examples-per-seed", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--singular-floors", type=float, nargs="+", default=(0.1, 0.25, 0.5, 0.75, 1.0)
    )
    parser.add_argument(
        "--coordinate-scales", type=float, nargs="+", default=(0.0, 0.5, 1.5, 2.0)
    )
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=887001)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.03)
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def vector_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=-1))))


@torch.no_grad()
def collect_states(
    *,
    model,
    cfg,
    bank,
    graph_seeds: Sequence[int],
    examples_per_seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if examples_per_seed % batch_size:
        raise ValueError("examples-per-seed must be divisible by batch-size")
    answer_position = cfg.seq_len - 1
    positions = tuple(range(cfg.seq_len))
    store: dict[str, list[np.ndarray]] = defaultdict(list)
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        for _ in range(examples_per_seed // batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            current = start
            for source_age in range(1, 9):
                state = model.apply_loop(state, loop_index=source_age - 1)
                current = advance_nodes(successors, current, steps=1)
                if source_age < 2:
                    continue
                post_j = bank.rollback(
                    state, source_age=source_age, positions=positions
                )
                store["pre_answer"].append(
                    state[:, answer_position].float().cpu().numpy()
                )
                store["post_answer"].append(
                    post_j[:, answer_position].float().cpu().numpy()
                )
                store["source_age"].append(
                    np.full(batch_size, source_age, dtype=np.int64)
                )
                store["logical_age"].append(
                    np.full(batch_size, source_age - 1, dtype=np.float64)
                )
                store["graph_seed"].append(
                    np.full(batch_size, graph_seed, dtype=np.int64)
                )
        print(
            json.dumps(
                {
                    "event": "graph_seed_complete",
                    "graph_seed": int(graph_seed),
                    "observations": int(examples_per_seed * len(AGES)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {key: np.concatenate(parts, axis=0) for key, parts in store.items()}


def fit_clean_probes(
    train: dict[str, np.ndarray],
    *,
    output_basis: np.ndarray,
    ridge: float,
) -> dict[str, dict[str, np.ndarray]]:
    target = train["logical_age"][:, None]
    features = {
        "full": train["post_answer"],
        "bottom_output": train["post_answer"] @ output_basis,
    }
    probes: dict[str, dict[str, np.ndarray]] = {}
    for family, values in features.items():
        weight, bias = fit_ridge_regression(values, target, ridge)
        probes[family] = {"weight": weight, "bias": bias}
    return probes


def probe_prediction(
    state: np.ndarray,
    *,
    family: str,
    probe: dict[str, np.ndarray],
    output_basis: np.ndarray,
) -> np.ndarray:
    features = state if family == "full" else state @ output_basis
    return (features @ probe["weight"] + probe["bias"]).reshape(-1)


def make_operator_deltas(
    *,
    weights: dict[int, np.ndarray],
    train: dict[str, np.ndarray],
    rank: int,
    floors: Sequence[float],
    random_draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for floor in floors:
        bottom_deltas: dict[int, np.ndarray] = {}
        bottom_stats = []
        for age in AGES:
            delta, stats = build_stage_delta(
                weights[age], rank=rank, mode="bottom_floor", target=float(floor)
            )
            bottom_deltas[age] = delta
            bottom_stats.append(stats)
        conditions.append(
            {
                "name": f"bottom{rank}_floor{floor:g}",
                "family": "bottom_floor",
                "dose": float(floor),
                "random_draw": -1,
                "deltas": bottom_deltas,
                "random_train_rms_scale": {},
                "operator_delta_fro_mean": float(
                    np.mean([row["delta_fro"] for row in bottom_stats])
                ),
            }
        )
        for draw in range(random_draws):
            random_deltas: dict[int, np.ndarray] = {}
            random_input_same_output_deltas: dict[int, np.ndarray] = {}
            rms_scales: dict[int, float] = {}
            same_output_rms_scales: dict[int, float] = {}
            fro_norms = []
            same_output_fro_norms = []
            for age in AGES:
                rng = np.random.default_rng(seed + 100000 * draw + 1000 * age + rank)
                random_delta, _ = build_stage_delta(
                    weights[age],
                    rank=rank,
                    mode="random_matched",
                    target=float(floor),
                    rng=rng,
                )
                age_mask = train["source_age"].astype(int) == age
                pre = train["pre_answer"][age_mask]
                target_rms = vector_rms(pre @ bottom_deltas[age])
                raw_rms = vector_rms(pre @ random_delta)
                scale = target_rms / max(raw_rms, 1e-30)
                random_deltas[age] = random_delta * scale
                rms_scales[age] = float(scale)
                fro_norms.append(float(np.linalg.norm(random_deltas[age], ord="fro")))
                _, singular, right_t = np.linalg.svd(weights[age])
                coefficients = np.maximum(float(floor) - singular[-rank:], 0.0)
                same_output_delta = (
                    random_orthonormal(weights[age].shape[0], rank, rng)
                    @ np.diag(coefficients)
                    @ right_t[-rank:, :]
                )
                same_output_raw_rms = vector_rms(pre @ same_output_delta)
                same_output_scale = target_rms / max(same_output_raw_rms, 1e-30)
                random_input_same_output_deltas[age] = (
                    same_output_delta * same_output_scale
                )
                same_output_rms_scales[age] = float(same_output_scale)
                same_output_fro_norms.append(
                    float(
                        np.linalg.norm(
                            random_input_same_output_deltas[age], ord="fro"
                        )
                    )
                )
            conditions.append(
                {
                    "name": f"random{rank}_state_rms_floor{floor:g}_d{draw}",
                    "family": "random_floor",
                    "dose": float(floor),
                    "random_draw": draw,
                    "deltas": random_deltas,
                    "random_train_rms_scale": rms_scales,
                    "operator_delta_fro_mean": float(np.mean(fro_norms)),
                }
            )
            conditions.append(
                {
                    "name": f"randomU_same_stageV{rank}_floor{floor:g}_d{draw}",
                    "family": "random_input_same_bottom_output",
                    "dose": float(floor),
                    "random_draw": draw,
                    "deltas": random_input_same_output_deltas,
                    "random_train_rms_scale": same_output_rms_scales,
                    "operator_delta_fro_mean": float(
                        np.mean(same_output_fro_norms)
                    ),
                }
            )
    return conditions


def apply_stage_deltas(
    data: dict[str, np.ndarray], deltas: dict[int, np.ndarray]
) -> np.ndarray:
    modified = data["post_answer"].copy()
    source_age = data["source_age"].astype(int)
    for age, delta in deltas.items():
        mask = source_age == age
        modified[mask] += data["pre_answer"][mask] @ delta
    return modified


def metric_rows(
    *,
    condition: dict[str, Any],
    modified: np.ndarray,
    clean: np.ndarray,
    data: dict[str, np.ndarray],
    probes: dict[str, dict[str, np.ndarray]],
    clean_predictions: dict[str, np.ndarray],
    output_basis: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delta_state = modified - clean
    for source_age in (0, *AGES):
        mask = (
            np.ones(data["source_age"].shape[0], dtype=bool)
            if source_age == 0
            else data["source_age"].astype(int) == source_age
        )
        target = data["logical_age"][mask]
        for probe_family, probe in probes.items():
            prediction = probe_prediction(
                modified[mask],
                family=probe_family,
                probe=probe,
                output_basis=output_basis,
            )
            clean_prediction = clean_predictions[probe_family][mask]
            shift = prediction - clean_prediction
            metrics = age_metrics(prediction, target)
            if source_age != 0:
                # R2 is undefined when every example has the same stage-age target.
                metrics["r2"] = float("nan")
            rows.append(
                {
                    "condition": condition["name"],
                    "condition_family": condition["family"],
                    "dose": condition["dose"],
                    "random_draw": condition["random_draw"],
                    "probe_family": probe_family,
                    "source_age": source_age,
                    "observations": int(mask.sum()),
                    **metrics,
                    "mean_predicted_age": float(prediction.mean()),
                    "mean_target_age": float(target.mean()),
                    "mean_signed_error": float((prediction - target).mean()),
                    "mean_probe_shift": float(shift.mean()),
                    "probe_shift_rms": float(np.sqrt(np.mean(np.square(shift)))),
                    "probe_shift_positive_fraction": float(np.mean(shift > 0)),
                    "state_delta_vector_rms": vector_rms(delta_state[mask]),
                    "state_delta_in_consensus_V_vector_rms": vector_rms(
                        delta_state[mask] @ output_basis
                    ),
                    "operator_delta_fro_mean": condition.get(
                        "operator_delta_fro_mean", 0.0
                    ),
                }
            )
    return rows


def aggregate_random_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["condition_family"],
                row["dose"],
                row["probe_family"],
                row["source_age"],
            )
        ].append(row)
    metrics = (
        "r2",
        "rmse",
        "rounded_accuracy",
        "mean_predicted_age",
        "mean_signed_error",
        "mean_probe_shift",
        "probe_shift_rms",
        "probe_shift_positive_fraction",
        "state_delta_vector_rms",
        "state_delta_in_consensus_V_vector_rms",
    )
    output: list[dict[str, Any]] = []
    for key, parts in sorted(grouped.items()):
        family, dose, probe_family, source_age = key
        result: dict[str, Any] = {
            "condition_family": family,
            "dose": dose,
            "probe_family": probe_family,
            "source_age": source_age,
            "draws": len(parts),
        }
        for metric in metrics:
            values = np.asarray([float(part[metric]) for part in parts])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
        output.append(result)
    return output


def plot_results(rows: Sequence[dict[str, Any]], path: Path) -> None:
    aggregate = aggregate_random_rows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    colors = {"full": "#4C78A8", "bottom_output": "#F58518"}
    labels = {"full": "full-256 probe", "bottom_output": "bottom-V rank-8 probe"}
    for probe_family in ("full", "bottom_output"):
        bottom = sorted(
            (
                row
                for row in aggregate
                if row["condition_family"] == "bottom_floor"
                and row["probe_family"] == probe_family
                and int(row["source_age"]) == 0
            ),
            key=lambda row: float(row["dose"]),
        )
        random = sorted(
            (
                row
                for row in aggregate
                if row["condition_family"] == "random_floor"
                and row["probe_family"] == probe_family
                and int(row["source_age"]) == 0
            ),
            key=lambda row: float(row["dose"]),
        )
        x = np.asarray([0.0, *[float(row["dose"]) for row in bottom]])
        axes[0].plot(
            x,
            [0.0, *[float(row["mean_probe_shift_mean"]) for row in bottom]],
            marker="o",
            color=colors[probe_family],
            label=labels[probe_family],
        )
        if random:
            random_x = np.asarray([float(row["dose"]) for row in random])
            random_y = np.asarray([float(row["mean_probe_shift_mean"]) for row in random])
            random_sem = np.asarray([float(row["mean_probe_shift_sem"]) for row in random])
            axes[0].fill_between(
                random_x,
                random_y - random_sem,
                random_y + random_sem,
                color=colors[probe_family],
                alpha=0.14,
            )
            axes[0].plot(random_x, random_y, linestyle="--", color=colors[probe_family])
        axes[1].plot(
            x,
            [
                next(
                    float(row["r2_mean"])
                    for row in aggregate
                    if row["condition_family"] == "baseline"
                    and row["probe_family"] == probe_family
                    and int(row["source_age"]) == 0
                ),
                *[float(row["r2_mean"]) for row in bottom],
            ],
            marker="o",
            color=colors[probe_family],
            label=labels[probe_family],
        )
    same_output = sorted(
        (
            row
            for row in aggregate
            if row["condition_family"] == "random_input_same_bottom_output"
            and row["probe_family"] == "bottom_output"
            and int(row["source_age"]) == 0
        ),
        key=lambda row: float(row["dose"]),
    )
    if same_output:
        same_x = np.asarray([float(row["dose"]) for row in same_output])
        same_shift = np.asarray(
            [float(row["mean_probe_shift_mean"]) for row in same_output]
        )
        same_shift_sem = np.asarray(
            [float(row["mean_probe_shift_sem"]) for row in same_output]
        )
        axes[0].errorbar(
            same_x,
            same_shift,
            yerr=same_shift_sem,
            marker="s",
            linestyle=":",
            color="#B279A2",
            label="bottom-V probe; random U, same stage V",
        )
        axes[1].plot(
            same_x,
            [float(row["r2_mean"]) for row in same_output],
            marker="s",
            linestyle=":",
            color="#B279A2",
            label="bottom-V probe; random U, same stage V",
        )
    coordinate = [
        row
        for row in aggregate
        if row["condition_family"] == "coordinate_scale"
        and row["probe_family"] == "bottom_output"
        and int(row["source_age"]) == 0
    ]
    coordinate_points = [(1.0, 0.0)] + [
        (float(row["dose"]), float(row["mean_probe_shift_mean"]))
        for row in coordinate
    ]
    coordinate_points.sort()
    axes[2].plot(
        [point[0] for point in coordinate_points],
        [point[1] for point in coordinate_points],
        marker="o",
        color=colors["bottom_output"],
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        xlabel="bottom singular-value floor",
        ylabel="mean frozen-probe shift (years)",
        title="Operator intervention\nsolid=bottom, dashed=random mean",
    )
    axes[1].set(
        xlabel="bottom singular-value floor",
        ylabel="held-out age R2",
        title="Probe calibration under intervention",
    )
    axes[1].set_yscale("symlog", linthresh=0.25)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set(
        xlabel="post-J consensus-V coordinate scale",
        ylabel="mean bottom-V probe shift (years)",
        title="Direct coordinate rescaling\n(analytic calibration only)",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if set(args.train_graph_seeds) & set(args.test_graph_seeds):
        raise ValueError("train and test graph seeds must be disjoint")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    atomic_json(
        manifest_path,
        {
            "status": "running",
            "pid": os.getpid(),
            "started_unix_time": time.time(),
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
        },
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank, weights, bases, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (args.rank,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    output_basis = bases["bottom_output"][args.rank]
    train = collect_states(
        model=model,
        cfg=cfg,
        bank=bank,
        graph_seeds=tuple(args.train_graph_seeds),
        examples_per_seed=args.examples_per_seed,
        batch_size=args.batch_size,
        device=device,
    )
    test = collect_states(
        model=model,
        cfg=cfg,
        bank=bank,
        graph_seeds=tuple(args.test_graph_seeds),
        examples_per_seed=args.examples_per_seed,
        batch_size=args.batch_size,
        device=device,
    )
    probes = fit_clean_probes(train, output_basis=output_basis, ridge=args.ridge)
    clean_predictions = {
        family: probe_prediction(
            test["post_answer"],
            family=family,
            probe=probe,
            output_basis=output_basis,
        )
        for family, probe in probes.items()
    }
    conditions: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "family": "baseline",
            "dose": 1.0,
            "random_draw": -1,
        }
    ]
    conditions.extend(
        make_operator_deltas(
            weights=weights,
            train=train,
            rank=args.rank,
            floors=tuple(args.singular_floors),
            random_draws=args.random_draws,
            seed=args.seed,
        )
    )
    for scale in args.coordinate_scales:
        conditions.append(
            {
                "name": f"consensus_V_coordinate_scale{scale:g}",
                "family": "coordinate_scale",
                "dose": float(scale),
                "random_draw": -1,
            }
        )

    rows: list[dict[str, Any]] = []
    projector = output_basis @ output_basis.T
    for condition in conditions:
        if condition["family"] in {
            "bottom_floor",
            "random_floor",
            "random_input_same_bottom_output",
        }:
            modified = apply_stage_deltas(test, condition["deltas"])
        elif condition["family"] == "coordinate_scale":
            scale = float(condition["dose"])
            modified = test["post_answer"] + (scale - 1.0) * (
                test["post_answer"] @ projector
            )
        else:
            modified = test["post_answer"].copy()
        rows.extend(
            metric_rows(
                condition=condition,
                modified=modified,
                clean=test["post_answer"],
                data=test,
                probes=probes,
                clean_predictions=clean_predictions,
                output_basis=output_basis,
            )
        )
    aggregate = aggregate_random_rows(rows)
    write_csv(args.out_dir / "intervention_metrics.csv", rows)
    write_csv(args.out_dir / "intervention_summary.csv", aggregate)
    np.savez_compressed(
        args.out_dir / "frozen_probes.npz",
        bottom_output_basis=output_basis,
        full_weight=probes["full"]["weight"],
        full_bias=probes["full"]["bias"],
        bottom_output_weight=probes["bottom_output"]["weight"],
        bottom_output_bias=probes["bottom_output"]["bias"],
    )
    plot_results(rows, args.out_dir / "uv_probe_amplification.png")
    baseline = {
        row["probe_family"]: row
        for row in aggregate
        if row["condition_family"] == "baseline" and int(row["source_age"]) == 0
    }
    result = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_recorded_checkpoint": bank_payload.get("checkpoint"),
        "train_graph_seeds": list(args.train_graph_seeds),
        "test_graph_seeds": list(args.test_graph_seeds),
        "examples_per_seed": args.examples_per_seed,
        "train_observations": int(train["logical_age"].shape[0]),
        "test_observations": int(test["logical_age"].shape[0]),
        "rank": args.rank,
        "ridge": args.ridge,
        "random_draws": args.random_draws,
        "clean_probe_metrics": {
            family: {
                "r2": float(row["r2_mean"]),
                "rmse": float(row["rmse_mean"]),
                "rounded_accuracy": float(row["rounded_accuracy_mean"]),
            }
            for family, row in baseline.items()
        },
        "coordinate_scale_identity": (
            "For the bottom-output probe T(h)=((h V)c)+b, replacing hV by s(hV) "
            "changes T by (s-1)(hV)c exactly; this condition is calibration, not circuit evidence."
        ),
        "operator_intervention": (
            "Each J stage has its bottom singular values raised to the requested floor. "
            "Random rank-matched deltas are rescaled per stage using clean training states "
            "to match the realized answer-state RMS, then evaluated on disjoint test seeds. "
            "A stricter control also keeps the same stage bottom-output V space while "
            "randomizing the input-read directions."
        ),
        "claim_boundary": (
            "Clean held-out R2 establishes readability only. A selective frozen-probe shift "
            "under bottom-channel restoration versus state-RMS-matched random operators supports "
            "a causal link to that readout, but the probe remains non-unique and an OOD readout "
            "does not by itself establish the downstream attention circuit."
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    atomic_json(args.out_dir / "summary.json", result)
    atomic_json(
        manifest_path,
        {
            **result,
            "completed_unix_time": time.time(),
            "out_dir": str(args.out_dir),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
