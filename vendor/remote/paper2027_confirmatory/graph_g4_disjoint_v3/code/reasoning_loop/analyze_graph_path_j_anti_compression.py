"""Test whether J rollback depends on suppressing its weakest singular channels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional

from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    AGES,
    atomic_json,
    load_bank_and_bases,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import (
    MAX_AGE,
    MIN_AGE,
    sample_equivalent_word_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--probe-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=(836001, 836002))
    parser.add_argument("--back-counts", type=int, nargs="+", default=(8, 12))
    parser.add_argument("--path-pairs-per-k", type=int, default=1)
    parser.add_argument("--word-seed", type=int, default=836701)
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional exact condition-name subset; baseline is added automatically.",
    )
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def random_orthonormal(
    dimension: int, rank: int, rng: np.random.Generator
) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
    return basis[:, :rank]


def build_stage_delta(
    weight: np.ndarray,
    *,
    rank: int,
    mode: str,
    target: float | None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return delta W for row-vector x @ (W + delta), preserving bias."""
    left, singular, right_t = np.linalg.svd(weight)
    bottom_singular = singular[-rank:]
    if mode == "bottom_zero":
        coefficients = -bottom_singular
        input_basis = left[:, -rank:]
        output_basis_t = right_t[-rank:, :]
    else:
        if target is None:
            raise ValueError("lift modes require a target")
        coefficients = np.maximum(target - bottom_singular, 0.0)
        if mode == "bottom_floor":
            input_basis = left[:, -rank:]
            output_basis_t = right_t[-rank:, :]
        elif mode == "top_matched":
            input_basis = left[:, :rank]
            output_basis_t = right_t[:rank, :]
        elif mode == "random_matched":
            if rng is None:
                raise ValueError("random_matched requires an RNG")
            input_basis = random_orthonormal(weight.shape[0], rank, rng)
            output_basis_t = random_orthonormal(weight.shape[1], rank, rng).T
        else:
            raise ValueError(f"unknown operator intervention mode: {mode}")
    delta = input_basis @ np.diag(coefficients) @ output_basis_t
    return delta, {
        "delta_fro": float(np.linalg.norm(delta, ord="fro")),
        "bottom_singular_mean": float(bottom_singular.mean()),
        "bottom_singular_max": float(bottom_singular.max()),
        "coefficient_mean": float(coefficients.mean()),
        "lifted_singular_count": float(np.count_nonzero(coefficients > 0.0)),
    }


def make_conditions(
    weights: dict[int, np.ndarray], *, random_draws: int
) -> list[dict[str, Any]]:
    specifications: list[tuple[str, int, str, float | None, int]] = []
    dimension = next(iter(weights.values())).shape[0]
    for target in (0.1, 0.2, 0.3, 0.4, 0.5):
        specifications.append(
            (f"all_floor{target:g}", dimension, "bottom_floor", target, -1)
        )
    for rank in (4, 8, 16, 32):
        specifications.extend(
            [
                (f"bottom{rank}_zero", rank, "bottom_zero", None, -1),
                (f"bottom{rank}_floor1", rank, "bottom_floor", 1.0, -1),
                (f"random{rank}_floor1_d0", rank, "random_matched", 1.0, 0),
            ]
        )
    for target in (0.25, 0.5, 0.75):
        specifications.append(
            (f"bottom16_floor{target:g}", 16, "bottom_floor", target, -1)
        )
    for rank in (4, 8):
        for target in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75):
            specifications.append(
                (f"bottom{rank}_floor{target:g}", rank, "bottom_floor", target, -1)
            )
            for draw in range(random_draws):
                specifications.append(
                    (
                        f"random{rank}_floor{target:g}_d{draw}",
                        rank,
                        "random_matched",
                        target,
                        draw,
                    )
                )
    for target in (0.5, 1.0):
        specifications.append(
            (f"top16_matched_floor{target:g}", 16, "top_matched", target, -1)
        )
    for draw in range(random_draws):
        for target in (0.25, 0.5, 0.75, 1.0):
            name = f"random16_floor{target:g}_d{draw}"
            if draw == 0 and target == 1.0:
                continue
            specifications.append(
                (name, 16, "random_matched", target, draw)
            )
    conditions: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "rank": 0,
            "mode": "baseline",
            "target": None,
            "random_draw": -1,
            "deltas": {},
            "delta_fro_mean": 0.0,
        }
    ]
    for name, rank, mode, target, draw in specifications:
        deltas: dict[int, np.ndarray] = {}
        stats = []
        for age in AGES:
            rng = (
                np.random.default_rng(840000 + 1000 * draw + 37 * age + rank)
                if mode == "random_matched"
                else None
            )
            deltas[age], stage_stats = build_stage_delta(
                weights[age], rank=rank, mode=mode, target=target, rng=rng
            )
            stats.append(stage_stats)
        conditions.append(
            {
                "name": name,
                "rank": rank,
                "mode": mode,
                "target": target,
                "random_draw": draw,
                "deltas": deltas,
                "delta_fro_mean": float(
                    np.mean([value["delta_fro"] for value in stats])
                ),
                "bottom_singular_mean": float(
                    np.mean([value["bottom_singular_mean"] for value in stats])
                ),
                "coefficient_mean": float(
                    np.mean([value["coefficient_mean"] for value in stats])
                ),
                "lifted_singular_count_mean": float(
                    np.mean([value["lifted_singular_count"] for value in stats])
                ),
            }
        )
    return conditions


@torch.no_grad()
def execute_word(
    *,
    model,
    bank: AgeSpecificJBank,
    h1: torch.Tensor,
    h1_current: torch.Tensor,
    successors: torch.Tensor,
    actions: Sequence[int],
    positions: tuple[int, ...],
    deltas: dict[int, torch.Tensor],
    age_weight: torch.Tensor,
    age_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    state = h1
    current = h1_current
    logical_age = MIN_AGE
    age_absolute_error = 0.0
    age_signed_error = 0.0
    age_rounded_correct = 0.0
    source_closer = 0.0
    age_count = 0
    index = list(positions)
    for action in actions:
        if action == 1:
            state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
        else:
            source_age = logical_age
            input_state = state
            state = bank.rollback(state, source_age=source_age, positions=positions)
            if deltas:
                state[:, index] = (
                    state[:, index].float()
                    + input_state[:, index].float() @ deltas[source_age]
                ).to(state.dtype)
            logical_age -= 1
            prediction = state[:, -1].float() @ age_weight + age_bias
            target = torch.full_like(prediction, float(logical_age))
            source = torch.full_like(prediction, float(source_age))
            age_absolute_error += float((prediction - target).abs().sum())
            age_signed_error += float((prediction - target).sum())
            age_rounded_correct += float(prediction.round().eq(target).sum())
            source_closer += float(
                (prediction - source).abs().lt((prediction - target).abs()).sum()
            )
            age_count += prediction.numel()
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("trajectory left H1..H8")
    if logical_age != MAX_AGE:
        raise RuntimeError("trajectory did not end at H8")
    return state, current, {
        "age_absolute_error": age_absolute_error,
        "age_signed_error": age_signed_error,
        "age_rounded_correct": age_rounded_correct,
        "source_closer": source_closer,
        "age_count": float(age_count),
    }


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    conditions: list[dict[str, Any]],
    age_weight: torch.Tensor,
    age_bias: torch.Tensor,
    examples: int,
    batch_size: int,
    graph_seeds: Sequence[int],
    back_counts: Sequence[int],
    path_pairs_per_k: int,
    word_seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("examples must divide by batch size")
    rng = np.random.default_rng(word_seed)
    words: list[tuple[int, str, tuple[int, ...]]] = []
    for back_count in back_counts:
        for pair_index in range(path_pairs_per_k):
            source_age = AGES[pair_index % len(AGES)]
            if back_count == 1:
                actions = (
                    (1,) * (source_age - MIN_AGE)
                    + (-1,)
                    + (1,) * (MAX_AGE - source_age + 1)
                )
                words.append(
                    (back_count, f"k1_p{pair_index}_source{source_age}", actions)
                )
                continue
            left, right = sample_equivalent_word_pair(
                rng=rng,
                back_count=back_count,
                mandatory_source_age=source_age,
            )
            words.extend(
                [
                    (back_count, f"k{back_count}_p{pair_index}_left", left),
                    (back_count, f"k{back_count}_p{pair_index}_right", right),
                ]
            )
    tensor_conditions = []
    for condition in conditions:
        tensor_conditions.append(
            {
                **condition,
                "deltas": {
                    age: torch.as_tensor(delta, dtype=torch.float32, device=device)
                    for age, delta in condition["deltas"].items()
                },
            }
        )
    slots: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    positions = tuple(range(cfg.seq_len))
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        for _ in range(examples // batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            h1 = model.apply_loop(raw, loop_index=0)
            h1_current = advance_nodes(successors, start, steps=1)
            for back_count, _, actions in words:
                for condition in tensor_conditions:
                    state, target, age_stats = execute_word(
                        model=model,
                        bank=bank,
                        h1=h1,
                        h1_current=h1_current,
                        successors=successors,
                        actions=actions,
                        positions=positions,
                        deltas=condition["deltas"],
                        age_weight=age_weight,
                        age_bias=age_bias,
                    )
                    logits = logits_from_raw_state(model, state).float()
                    prediction = logits.argmax(-1)
                    correct_logits = logits.gather(1, target[:, None]).squeeze(1)
                    distractor = logits.clone()
                    distractor.scatter_(1, target[:, None], -torch.inf)
                    slot = slots[(back_count, condition["name"])]
                    slot["correct"] += float(prediction.eq(target).sum())
                    slot["count"] += batch_size
                    slot["ce"] += float(
                        torch_functional.cross_entropy(logits, target, reduction="sum")
                    )
                    slot["margin"] += float(
                        (correct_logits - distractor.max(-1).values).sum()
                    )
                    slot["answer_rms"] += float(
                        state[:, -1].float().square().mean(-1).sqrt().sum()
                    )
                    for key, value in age_stats.items():
                        slot[key] += value
    lookup = {condition["name"]: condition for condition in conditions}
    rows = []
    for (back_count, name), slot in sorted(slots.items()):
        condition = lookup[name]
        count = slot["count"]
        age_count = slot["age_count"]
        rows.append(
            {
                "back_count": back_count,
                "condition": name,
                "mode": condition["mode"],
                "rank": condition["rank"],
                "target_singular_floor": condition["target"],
                "random_draw": condition["random_draw"],
                "delta_fro_mean": condition["delta_fro_mean"],
                "bottom_singular_mean": condition.get("bottom_singular_mean", ""),
                "lifted_singular_count_mean": condition.get(
                    "lifted_singular_count_mean", ""
                ),
                "accuracy": slot["correct"] / count,
                "ce": slot["ce"] / count,
                "correct_margin": slot["margin"] / count,
                "answer_rms": slot["answer_rms"] / count,
                "post_J_age_mae": slot["age_absolute_error"] / age_count,
                "post_J_age_signed_error": slot["age_signed_error"] / age_count,
                "post_J_age_rounded_accuracy": slot["age_rounded_correct"] / age_count,
                "post_J_probe_closer_to_source_fraction": slot["source_closer"] / age_count,
                "final_examples": int(count),
                "post_J_observations": int(age_count),
            }
        )
    return rows


def plot_results(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    longest = max(int(row["back_count"]) for row in rows)
    selected = [row for row in rows if int(row["back_count"]) == longest]
    baseline = next(row for row in selected if row["condition"] == "baseline")
    bottom = sorted(
        [
            row
            for row in selected
            if row["mode"] == "bottom_floor" and int(row["rank"]) == 16
        ],
        key=lambda row: float(row["target_singular_floor"]),
    )
    random_rows = [
        row
        for row in selected
        if row["mode"] == "random_matched" and int(row["rank"]) == 16
    ]
    random_grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in random_rows:
        random_grouped[float(row["target_singular_floor"])].append(row)
    targets = [0.0] + [float(row["target_singular_floor"]) for row in bottom]
    axes[0].plot(
        targets,
        [float(baseline["accuracy"])] + [float(row["accuracy"]) for row in bottom],
        marker="o",
        label="true bottom-16",
    )
    random_targets = sorted(random_grouped)
    axes[0].plot(
        [0.0] + random_targets,
        [float(baseline["accuracy"])]
        + [float(np.mean([float(row["accuracy"]) for row in random_grouped[target]])) for target in random_targets],
        marker="s",
        label="random rank-16 matched",
    )
    axes[0].set(
        title=f"Final accuracy after anti-compression (k={longest})",
        xlabel="target singular floor",
        ylabel="accuracy",
        ylim=(0, 1.03),
    )
    axes[0].legend()
    axes[1].plot(
        targets,
        [float(baseline["post_J_age_signed_error"])]
        + [float(row["post_J_age_signed_error"]) for row in bottom],
        marker="o",
        label="true bottom-16",
    )
    axes[1].plot(
        [0.0] + random_targets,
        [float(baseline["post_J_age_signed_error"])]
        + [float(np.mean([float(row["post_J_age_signed_error"]) for row in random_grouped[target]])) for target in random_targets],
        marker="s",
        label="random rank-16 matched",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        title="Age probe after J (positive means too old)",
        xlabel="target singular floor",
        ylabel="predicted age minus target age",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.examples % args.batch_size:
        raise ValueError("examples must divide by batch size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            "status": "running",
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "probe_artifact": str(args.probe_artifact),
            "device_requested": args.device,
            "pid": os.getpid(),
        },
    )
    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    bank, weights, _, _, _ = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (4, 8, 16, 32),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    probe = np.load(args.probe_artifact)
    age_weight = torch.as_tensor(
        probe["post_J_full_age_weight"], dtype=torch.float32, device=device
    )
    age_bias = torch.as_tensor(
        probe["post_J_full_age_bias"], dtype=torch.float32, device=device
    )
    conditions = make_conditions(weights, random_draws=args.random_draws)
    if args.conditions:
        requested = {"baseline", *args.conditions}
        available = {condition["name"] for condition in conditions}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown requested conditions: {missing}")
        conditions = [
            condition for condition in conditions if condition["name"] in requested
        ]
    rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        conditions=conditions,
        age_weight=age_weight,
        age_bias=age_bias,
        examples=args.examples,
        batch_size=args.batch_size,
        graph_seeds=tuple(args.graph_seeds),
        back_counts=tuple(args.back_counts),
        path_pairs_per_k=args.path_pairs_per_k,
        word_seed=args.word_seed,
        device=device,
    )
    write_csv(args.out_dir / "anti_compression_metrics.csv", rows)
    plot_results(rows, args.out_dir / "anti_compression.png")
    result = {
        "status": "complete",
        "actual_device": str(device),
        "examples_per_graph_seed": args.examples,
        "graph_seeds": list(args.graph_seeds),
        "back_counts": list(args.back_counts),
        "conditions": len(conditions),
        "claim_boundary": (
            "This is a targeted operator-surgery test. A selective change versus matched random deltas supports a causal role for the weak singular channels; "
            "generic failure under equally sized perturbations does not."
        ),
    }
    atomic_json(args.out_dir / "summary.json", result)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            **result,
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "probe_artifact": str(args.probe_artifact),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
