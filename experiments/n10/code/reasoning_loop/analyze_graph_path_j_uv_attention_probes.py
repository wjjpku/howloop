"""Compare the bottom input/output spaces of J as probes of its next attention call.

For each canonical natural state H_a, a=2..8, this script applies the trained
rollback map J_a and instruments the immediately following F call.  Equal-rank
probes compare the consensus bottom-input space U_B with the consensus
bottom-output space V_B.  Targets include logical age, B2.H0 answer-query,
the correct graph token's key/value, destination attention mass, and routing
margin.  Leave-one-graph-seed-out evaluation and random rank-matched spaces
keep the result at the localization level rather than treating one fitted
probe as a unique mechanism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
import torch.nn.functional as F

from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    atomic_json,
    fit_ridge_regression,
    load_bank_and_bases,
    orthonormal_random,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--graph-seeds",
        type=int,
        nargs="+",
        default=(886101, 886102, 886103, 886104, 886105),
    )
    parser.add_argument("--examples-per-seed", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=886001)
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


def _target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    mask = F.one_hot(target, num_classes=logits.shape[-1]).bool()
    wrong = logits.masked_fill(mask, float("-inf")).max(-1).values
    return correct - wrong


def _routing_margin(
    q: torch.Tensor,
    k: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    logits = torch.einsum("bd,bnd->bn", q.float(), k.float()) / math.sqrt(q.shape[-1])
    correct = logits.gather(1, current[:, None]).squeeze(1)
    mask = F.one_hot(current, num_classes=logits.shape[-1]).bool()
    wrong = logits.masked_fill(mask, float("-inf")).max(-1).values
    return correct - wrong


@torch.no_grad()
def collect_events(
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
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"][0]
    destination_positions = torch.as_tensor(
        groups["destination"], dtype=torch.long, device=device
    )
    positions = tuple(range(cfg.seq_len))
    store: dict[str, list[np.ndarray]] = defaultdict(list)
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        for batch_index in range(examples_per_seed // batch_size):
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
                pre_j = state
                post_j = bank.rollback(
                    pre_j, source_age=source_age, positions=positions
                )
                logits, trace = run_instrumented_state(
                    model,
                    post_j,
                    loop_indices=(source_age - 1,),
                )
                b2 = trace.sites[1]
                batch = torch.arange(batch_size, device=device)
                graph_positions = destination_positions[None, :].expand(batch_size, -1)
                correct_positions = destination_positions[current]
                q = b2.q[:, 0, answer]
                k_graph = b2.k[:, 0][batch[:, None], graph_positions]
                k_correct = b2.k[:, 0][batch, correct_positions]
                v_correct = b2.v[:, 0][batch, correct_positions]
                destination_mass = b2.attention_pattern[:, 0, answer][
                    batch, correct_positions
                ]
                next_current = advance_nodes(successors, current, steps=1)
                values: dict[str, torch.Tensor] = {
                    "pre_answer": pre_j[:, answer],
                    "post_answer": post_j[:, answer],
                    "post_correct_destination": post_j[batch, correct_positions],
                    "logical_age": torch.full(
                        (batch_size,), source_age - 1, device=device
                    ),
                    "source_age": torch.full(
                        (batch_size,), source_age, device=device
                    ),
                    "current": current,
                    "next_current": next_current,
                    "b2h0_q": q,
                    "b2h0_k_correct": k_correct,
                    "b2h0_v_correct": v_correct,
                    "destination_mass": destination_mass,
                    "routing_margin": _routing_margin(q, k_graph, current),
                    "next_task_margin": _target_margin(logits.float(), next_current),
                }
                for key, value in values.items():
                    store[key].append(value.float().cpu().numpy())
                store["graph_seed"].append(
                    np.full(batch_size, graph_seed, dtype=np.int64)
                )
                store["batch_index"].append(
                    np.full(batch_size, batch_index, dtype=np.int64)
                )
        print(
            json.dumps(
                {
                    "event": "graph_seed_complete",
                    "graph_seed": int(graph_seed),
                    "observations": int(examples_per_seed * 7),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {key: np.concatenate(values, axis=0) for key, values in store.items()}


def one_hot_labels(data: dict[str, np.ndarray]) -> np.ndarray:
    age = np.eye(7, dtype=np.float64)[data["logical_age"].astype(int) - 1]
    current = np.eye(8, dtype=np.float64)[data["current"].astype(int)]
    next_current = np.eye(8, dtype=np.float64)[data["next_current"].astype(int)]
    return np.concatenate((age, current, next_current), axis=1)


def projected_features(
    values: np.ndarray,
    *,
    family: str,
    basis: np.ndarray | None,
) -> np.ndarray:
    if family == "full":
        return values
    if basis is None:
        raise ValueError(f"{family} requires a basis")
    if family.startswith("complement_"):
        return values - (values @ basis) @ basis.T
    return values @ basis


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    residual = prediction - target
    centered = target - target.mean(axis=0, keepdims=True)
    r2 = 1.0 - float(np.square(residual).sum()) / max(
        float(np.square(centered).sum()), 1e-30
    )
    result = {
        "r2": r2,
        "rmse": float(np.sqrt(np.square(residual).mean())),
    }
    if target.ndim == 2 and target.shape[1] > 1:
        numerator = np.sum(prediction * target, axis=1)
        denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
        result["mean_cosine"] = float(
            np.mean(numerator / np.maximum(denominator, 1e-30))
        )
    return result


def classification_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(prediction.argmax(-1) == target.astype(int)))


def feature_specs(
    *,
    bases: dict[str, dict[int, np.ndarray]],
    random_bases: Sequence[np.ndarray],
    rank: int,
) -> list[tuple[str, int, np.ndarray | None]]:
    return [
        ("full", -1, None),
        ("bottom_input", -1, bases["bottom_input"][rank]),
        ("bottom_output", -1, bases["bottom_output"][rank]),
        ("complement_bottom_input", -1, bases["bottom_input"][rank]),
        ("complement_bottom_output", -1, bases["bottom_output"][rank]),
        *[("random", draw, basis) for draw, basis in enumerate(random_bases)],
    ]


TARGETS_BY_LOCUS: dict[str, tuple[str, ...]] = {
    "pre_answer": (
        "logical_age",
        "current",
        "b2h0_q",
        "destination_mass",
        "routing_margin",
        "next_task_margin",
    ),
    "post_answer": (
        "logical_age",
        "current",
        "b2h0_q",
        "destination_mass",
        "routing_margin",
        "next_task_margin",
    ),
    "post_correct_destination": (
        "logical_age",
        "current",
        "next_current",
        "b2h0_k_correct",
        "b2h0_v_correct",
    ),
}


def run_cross_validated_probes(
    *,
    data: dict[str, np.ndarray],
    bases: dict[str, dict[int, np.ndarray]],
    random_bases: Sequence[np.ndarray],
    rank: int,
    ridge: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nuisance = one_hot_labels(data)
    graph_seeds = np.unique(data["graph_seed"].astype(int))
    specs = feature_specs(bases=bases, random_bases=random_bases, rank=rank)
    for held_out_seed in graph_seeds:
        train_seed = data["graph_seed"].astype(int) != held_out_seed
        test_seed = ~train_seed
        for source_age_filter in (0, 2, 3, 4, 5, 6, 7, 8):
            eligible = (
                np.ones_like(train_seed, dtype=bool)
                if source_age_filter == 0
                else data["source_age"].astype(int) == source_age_filter
            )
            train = train_seed & eligible
            test = test_seed & eligible
            for locus, targets in TARGETS_BY_LOCUS.items():
                raw_features = data[locus]
                for target_name in targets:
                    if source_age_filter and target_name == "logical_age":
                        continue
                    target = data[target_name]
                    is_classification = target_name in {"current", "next_current"}
                    if is_classification:
                        fit_target = np.eye(8, dtype=np.float64)[target.astype(int)]
                    else:
                        fit_target = target[:, None] if target.ndim == 1 else target
                    nuisance_prediction = None
                    nuisance_metrics: dict[str, float] = {}
                    if not is_classification and target_name != "logical_age":
                        nuisance_weight, nuisance_bias = fit_ridge_regression(
                            nuisance[train], fit_target[train], ridge
                        )
                        nuisance_prediction = (
                            nuisance[test] @ nuisance_weight + nuisance_bias
                        )
                        nuisance_metrics = regression_metrics(
                            nuisance_prediction, fit_target[test]
                        )
                    for family, random_draw, basis in specs:
                        features = projected_features(
                            raw_features, family=family, basis=basis
                        )
                        weight, bias = fit_ridge_regression(
                            features[train], fit_target[train], ridge
                        )
                        prediction = features[test] @ weight + bias
                        row: dict[str, Any] = {
                            "held_out_graph_seed": int(held_out_seed),
                            "source_age": source_age_filter,
                            "locus": locus,
                            "target": target_name,
                            "feature_family": family,
                            "rank": raw_features.shape[1] if family == "full" else rank,
                            "random_draw": random_draw,
                            "train_observations": int(train.sum()),
                            "test_observations": int(test.sum()),
                        }
                        if is_classification:
                            row["classification_accuracy"] = classification_accuracy(
                                prediction, target[test]
                            )
                        else:
                            metrics = regression_metrics(prediction, fit_target[test])
                            row.update(metrics)
                            if nuisance_prediction is not None:
                                combined = np.concatenate((nuisance, features), axis=1)
                                combined_weight, combined_bias = fit_ridge_regression(
                                    combined[train], fit_target[train], ridge
                                )
                                combined_prediction = (
                                    combined[test] @ combined_weight + combined_bias
                                )
                                combined_metrics = regression_metrics(
                                    combined_prediction, fit_target[test]
                                )
                                row["nuisance_r2"] = nuisance_metrics["r2"]
                                row["nuisance_plus_feature_r2"] = combined_metrics["r2"]
                                row["incremental_r2_over_nuisance"] = (
                                    combined_metrics["r2"] - nuisance_metrics["r2"]
                                )
                        rows.append(row)
    return rows


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "r2",
        "rmse",
        "mean_cosine",
        "classification_accuracy",
        "nuisance_r2",
        "nuisance_plus_feature_r2",
        "incremental_r2_over_nuisance",
    )
    seed_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["held_out_graph_seed"],
            row["source_age"],
            row["locus"],
            row["target"],
            row["feature_family"],
            row["rank"],
        )
        seed_grouped[key].append(row)
    seed_rows: list[dict[str, Any]] = []
    for key, parts in seed_grouped.items():
        seed, source_age, locus, target, family, rank = key
        result: dict[str, Any] = {
            "held_out_graph_seed": seed,
            "source_age": source_age,
            "locus": locus,
            "target": target,
            "feature_family": family,
            "rank": rank,
            "random_draws_averaged": len(parts) if family == "random" else 0,
        }
        for metric in metrics:
            values = [float(part[metric]) for part in parts if metric in part]
            if values:
                result[metric] = float(np.mean(values))
        seed_rows.append(result)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[
            (
                row["source_age"],
                row["locus"],
                row["target"],
                row["feature_family"],
                row["rank"],
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, parts in sorted(grouped.items()):
        source_age, locus, target, family, rank = key
        result = {
            "source_age": source_age,
            "locus": locus,
            "target": target,
            "feature_family": family,
            "rank": rank,
            "graph_seeds": len(parts),
        }
        for metric in metrics:
            values = np.asarray(
                [float(part[metric]) for part in parts if metric in part],
                dtype=np.float64,
            )
            if values.size:
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_sem"] = (
                    float(values.std(ddof=1) / np.sqrt(values.size))
                    if values.size > 1
                    else 0.0
                )
        output.append(result)
    return output


def _lookup(
    rows: Sequence[dict[str, Any]], locus: str, target: str, family: str
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["locus"] == locus
        and int(row["source_age"]) == 0
        and row["target"] == target
        and row["feature_family"] == family
    )


def plot_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    families = ("bottom_input", "bottom_output", "random")
    labels = ("U bottom", "V bottom", "random-8")
    colors = ("#4C78A8", "#F58518", "#9D9D9D")

    for axis, locus, targets, title in (
        (
            axes[0, 0],
            "post_answer",
            ("logical_age", "b2h0_q"),
            "Post-J answer state",
        ),
        (
            axes[0, 1],
            "post_answer",
            ("destination_mass", "routing_margin", "next_task_margin"),
            "Downstream routing and task margins",
        ),
        (
            axes[1, 0],
            "post_correct_destination",
            ("b2h0_k_correct", "b2h0_v_correct"),
            "Correct graph token at B2.H0",
        ),
    ):
        width = 0.24
        x = np.arange(len(targets))
        for offset, (family, label, color) in enumerate(zip(families, labels, colors)):
            values = [float(_lookup(rows, locus, target, family)["r2_mean"]) for target in targets]
            errors = [float(_lookup(rows, locus, target, family)["r2_sem"]) for target in targets]
            axis.bar(x + (offset - 1) * width, values, width, label=label, color=color, yerr=errors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [target.replace("b2h0_", "") for target in targets], rotation=15)
        axis.set_title(title)
        axis.set_ylabel("held-out R2")
        axis.grid(alpha=0.2, axis="y")

    targets = ("b2h0_q", "destination_mass", "routing_margin", "next_task_margin")
    width = 0.24
    x = np.arange(len(targets))
    for offset, (family, label, color) in enumerate(zip(families, labels, colors)):
        values = [
            float(_lookup(rows, "post_answer", target, family)["incremental_r2_over_nuisance_mean"])
            for target in targets
        ]
        errors = [
            float(_lookup(rows, "post_answer", target, family)["incremental_r2_over_nuisance_sem"])
            for target in targets
        ]
        axes[1, 1].bar(x + (offset - 1) * width, values, width, label=label, color=color, yerr=errors)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(x, [target.replace("b2h0_", "") for target in targets], rotation=15)
    axes[1, 1].set(title="Incremental R2 beyond age/current/next-node labels", ylabel="delta R2")
    axes[1, 1].grid(alpha=0.2, axis="y")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.examples_per_seed % args.batch_size:
        raise ValueError("examples-per-seed must divide by batch-size")
    if len(args.graph_seeds) < 2:
        raise ValueError("at least two graph seeds are required for held-out evaluation")
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
            "out_dir": str(args.out_dir),
            "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
    bank, _, bases, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (args.rank,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    rng = np.random.default_rng(args.seed + 91)
    random_bases = [
        orthonormal_random(cfg.d_model, args.rank, rng)
        for _ in range(args.random_draws)
    ]
    data = collect_events(
        model=model,
        cfg=cfg,
        bank=bank,
        graph_seeds=tuple(args.graph_seeds),
        examples_per_seed=args.examples_per_seed,
        batch_size=args.batch_size,
        device=device,
    )
    fold_rows = run_cross_validated_probes(
        data=data,
        bases=bases,
        random_bases=random_bases,
        rank=args.rank,
        ridge=args.ridge,
    )
    summary_rows = aggregate_rows(fold_rows)
    write_csv(args.out_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.out_dir / "probe_summary.csv", summary_rows)
    np.savez_compressed(
        args.out_dir / "probe_bases.npz",
        bottom_input=bases["bottom_input"][args.rank],
        bottom_output=bases["bottom_output"][args.rank],
        **{f"random_{draw}": basis for draw, basis in enumerate(random_bases)},
    )
    plot_summary(summary_rows, args.out_dir / "uv_attention_probe_summary.png")

    u_basis = bases["bottom_input"][args.rank]
    v_basis = bases["bottom_output"][args.rank]
    principal_cosines = np.linalg.svd(u_basis.T @ v_basis, compute_uv=False)
    highlights: dict[str, Any] = {
        "post_answer_logical_age_r2_U": _lookup(
            summary_rows, "post_answer", "logical_age", "bottom_input"
        )["r2_mean"],
        "post_answer_logical_age_r2_V": _lookup(
            summary_rows, "post_answer", "logical_age", "bottom_output"
        )["r2_mean"],
        "post_answer_B2H0_Q_r2_U": _lookup(
            summary_rows, "post_answer", "b2h0_q", "bottom_input"
        )["r2_mean"],
        "post_answer_B2H0_Q_r2_V": _lookup(
            summary_rows, "post_answer", "b2h0_q", "bottom_output"
        )["r2_mean"],
        "post_destination_B2H0_K_r2_U": _lookup(
            summary_rows,
            "post_correct_destination",
            "b2h0_k_correct",
            "bottom_input",
        )["r2_mean"],
        "post_destination_B2H0_K_r2_V": _lookup(
            summary_rows,
            "post_correct_destination",
            "b2h0_k_correct",
            "bottom_output",
        )["r2_mean"],
        "post_destination_B2H0_V_r2_U": _lookup(
            summary_rows,
            "post_correct_destination",
            "b2h0_v_correct",
            "bottom_input",
        )["r2_mean"],
        "post_destination_B2H0_V_r2_V": _lookup(
            summary_rows,
            "post_correct_destination",
            "b2h0_v_correct",
            "bottom_output",
        )["r2_mean"],
    }
    result = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_recorded_checkpoint": bank_payload.get("checkpoint"),
        "graph_seeds": list(args.graph_seeds),
        "examples_per_seed": args.examples_per_seed,
        "observations": int(data["graph_seed"].shape[0]),
        "rank": args.rank,
        "random_draws": args.random_draws,
        "ridge": args.ridge,
        "input_output_overlap": float(np.square(u_basis.T @ v_basis).sum() / args.rank),
        "principal_cosines": principal_cosines.tolist(),
        "highlights": highlights,
        "claim_boundary": (
            "Fixed-subspace held-out probes localize variables available in U_bottom or V_bottom. "
            "They do not establish that the fitted readout is unique or that the subspace is a sufficient circuit."
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    atomic_json(args.out_dir / "summary.json", result)
    atomic_json(
        manifest_path,
        {
            "status": "complete",
            "pid": os.getpid(),
            "completed_unix_time": time.time(),
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "out_dir": str(args.out_dir),
            "actual_device": str(device),
            "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
