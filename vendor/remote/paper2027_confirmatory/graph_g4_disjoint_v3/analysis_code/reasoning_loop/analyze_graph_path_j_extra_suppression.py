"""Test whether stronger removal of J's harmful low-gain directions extends lifespan.

The sign convention is explicit: the selected bottom singular channels carry
harmful old-phase residue, so reducing their gain is a cleaning intervention.
Random low-rank negative updates are matched either in parameter norm or in the
actual hidden-state perturbation at every rollback call.
"""

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
import torch.nn.functional as F

from reasoning_loop.analyze_graph_path_j_anti_compression import random_orthonormal
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--probe-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--graph-seeds", type=int, nargs="+", default=(856101, 856102, 856103)
    )
    parser.add_argument(
        "--back-counts", type=int, nargs="+", default=(16, 24, 32, 48, 64)
    )
    parser.add_argument("--path-pairs-per-k", type=int, default=1)
    parser.add_argument("--word-seed", type=int, default=856701)
    parser.add_argument("--random-draws", type=int, default=2)
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional exact condition subset; baseline is always included.",
    )
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


def build_suppression_delta(
    weight: np.ndarray,
    *,
    rank: int,
    residual_scale: float,
    mode: str,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return delta W that reduces selected gain under row-vector h @ W.

    ``residual_scale=0`` zeros the selected bottom singular values.  Random
    controls use the identical signed coefficient vector in random orthonormal
    input/output bases, so the update Frobenius norm is exactly matched.
    """
    if not 0.0 <= residual_scale <= 1.0:
        raise ValueError("residual_scale must be in [0, 1]")
    left, singular, right_t = np.linalg.svd(weight)
    selected = singular[-rank:]
    coefficients = (residual_scale - 1.0) * selected
    if mode == "bottom":
        input_basis = left[:, -rank:]
        output_basis_t = right_t[-rank:, :]
    elif mode == "random":
        if rng is None:
            raise ValueError("random mode requires an RNG")
        input_basis = random_orthonormal(weight.shape[0], rank, rng)
        output_basis_t = random_orthonormal(weight.shape[1], rank, rng).T
    else:
        raise ValueError(f"unknown suppression mode: {mode}")
    delta = input_basis @ np.diag(coefficients) @ output_basis_t
    return delta, {
        "delta_fro": float(np.linalg.norm(delta, ord="fro")),
        "selected_singular_mean": float(selected.mean()),
        "coefficient_mean": float(coefficients.mean()),
    }


def match_effect_per_example(
    random_effect: torch.Tensor, target_effect: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale random effect to target RMS independently for each example."""
    dims = tuple(range(1, random_effect.ndim))
    random_rms = random_effect.float().square().mean(dim=dims).sqrt()
    target_rms = target_effect.float().square().mean(dim=dims).sqrt()
    scale = target_rms / random_rms.clamp_min(1e-12)
    shape = (scale.shape[0],) + (1,) * (random_effect.ndim - 1)
    return random_effect * scale.reshape(shape), scale


def make_conditions(
    weights: dict[int, np.ndarray], *, random_draws: int
) -> list[dict[str, Any]]:
    specs: list[tuple[str, int, float, str, int, bool]] = []
    for rank in (4, 8, 16):
        scales = (0.0,) if rank != 8 else (0.5, 0.25, 0.0)
        for scale in scales:
            scale_label = f"{scale:g}".replace(".", "p")
            specs.append(
                (f"bottom{rank}_scale{scale_label}", rank, scale, "bottom", -1, False)
            )
        for draw in range(random_draws):
            specs.append(
                (f"random{rank}_scale0_d{draw}", rank, 0.0, "random", draw, False)
            )
            specs.append(
                (
                    f"random{rank}_scale0_state_d{draw}",
                    rank,
                    0.0,
                    "random",
                    draw,
                    True,
                )
            )

    conditions: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "rank": 0,
            "residual_scale": 1.0,
            "mode": "baseline",
            "draw": -1,
            "state_effect_matched": False,
            "deltas": {},
            "reference_deltas": {},
            "delta_fro_mean": 0.0,
        }
    ]
    true_cache: dict[tuple[int, float], dict[int, np.ndarray]] = {}
    for rank in (4, 8, 16):
        for scale in (0.0, 0.25, 0.5):
            true_cache[(rank, scale)] = {
                age: build_suppression_delta(
                    weights[age], rank=rank, residual_scale=scale, mode="bottom"
                )[0]
                for age in AGES
            }
    for name, rank, scale, mode, draw, state_matched in specs:
        deltas: dict[int, np.ndarray] = {}
        stats: list[dict[str, float]] = []
        for age in AGES:
            rng = (
                np.random.default_rng(857000 + 1000 * draw + 37 * age + rank)
                if mode == "random"
                else None
            )
            delta, stat = build_suppression_delta(
                weights[age],
                rank=rank,
                residual_scale=scale,
                mode=mode,
                rng=rng,
            )
            deltas[age] = delta
            stats.append(stat)
        conditions.append(
            {
                "name": name,
                "rank": rank,
                "residual_scale": scale,
                "mode": mode,
                "draw": draw,
                "state_effect_matched": state_matched,
                "deltas": deltas,
                "reference_deltas": true_cache[(rank, scale)] if state_matched else {},
                "delta_fro_mean": float(np.mean([s["delta_fro"] for s in stats])),
                "selected_singular_mean": float(
                    np.mean([s["selected_singular_mean"] for s in stats])
                ),
            }
        )
    return conditions


def build_words(
    *, back_counts: Sequence[int], path_pairs_per_k: int, seed: int
) -> list[tuple[int, tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    words: list[tuple[int, tuple[int, ...]]] = []
    for back_count in back_counts:
        for pair_index in range(path_pairs_per_k):
            mandatory = AGES[pair_index % len(AGES)]
            left, right = sample_equivalent_word_pair(
                rng=rng,
                back_count=int(back_count),
                mandatory_source_age=mandatory,
            )
            words.extend(((int(back_count), left), (int(back_count), right)))
    return words


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
    reference_deltas: dict[int, torch.Tensor],
    age_weight: torch.Tensor,
    age_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    state = h1
    current = h1_current
    logical_age = MIN_AGE
    index = list(positions)
    stats = defaultdict(float)
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
                random_or_true = input_state[:, index].float() @ deltas[source_age]
                scale = torch.ones(input_state.shape[0], device=input_state.device)
                if reference_deltas:
                    target_effect = (
                        input_state[:, index].float() @ reference_deltas[source_age]
                    )
                    random_or_true, scale = match_effect_per_example(
                        random_or_true, target_effect
                    )
                state[:, index] = (
                    state[:, index].float() + random_or_true
                ).to(state.dtype)
                stats["effect_rms_sum"] += float(
                    random_or_true.square().mean(dim=(1, 2)).sqrt().sum()
                )
                stats["effect_scale_sum"] += float(scale.sum())
                stats["effect_count"] += float(input_state.shape[0])
            logical_age -= 1
            prediction = state[:, -1].float() @ age_weight + age_bias
            target_age = torch.full_like(prediction, float(logical_age))
            stats["age_abs_sum"] += float((prediction - target_age).abs().sum())
            stats["age_signed_sum"] += float((prediction - target_age).sum())
            stats["age_count"] += float(prediction.numel())
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("trajectory left H1..H8")
    if logical_age != MAX_AGE:
        raise RuntimeError("trajectory did not end at H8")
    return state, current, dict(stats)


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
    words = build_words(
        back_counts=back_counts,
        path_pairs_per_k=path_pairs_per_k,
        seed=word_seed,
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
                "reference_deltas": {
                    age: torch.as_tensor(delta, dtype=torch.float32, device=device)
                    for age, delta in condition["reference_deltas"].items()
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
            for back_count, actions in words:
                for condition in tensor_conditions:
                    state, target, extra = execute_word(
                        model=model,
                        bank=bank,
                        h1=h1,
                        h1_current=h1_current,
                        successors=successors,
                        actions=actions,
                        positions=positions,
                        deltas=condition["deltas"],
                        reference_deltas=condition["reference_deltas"],
                        age_weight=age_weight,
                        age_bias=age_bias,
                    )
                    logits = logits_from_raw_state(model, state).float()
                    prediction = logits.argmax(-1)
                    correct = logits.gather(1, target[:, None]).squeeze(1)
                    distractor = logits.masked_fill(
                        F.one_hot(target, num_classes=logits.shape[-1]).bool(),
                        float("-inf"),
                    ).max(-1).values
                    slot = slots[(back_count, condition["name"])]
                    slot["correct"] += float(prediction.eq(target).sum())
                    slot["count"] += float(batch_size)
                    slot["ce"] += float(F.cross_entropy(logits, target, reduction="sum"))
                    slot["margin"] += float((correct - distractor).sum())
                    slot["answer_rms"] += float(
                        state[:, -1].float().square().mean(-1).sqrt().sum()
                    )
                    for key, value in extra.items():
                        slot[key] += value

    lookup = {condition["name"]: condition for condition in conditions}
    rows: list[dict[str, Any]] = []
    for (back_count, name), slot in sorted(slots.items()):
        condition = lookup[name]
        count = slot["count"]
        age_count = max(slot["age_count"], 1.0)
        effect_count = max(slot["effect_count"], 1.0)
        rows.append(
            {
                "back_count": back_count,
                "condition": name,
                "mode": condition["mode"],
                "rank": condition["rank"],
                "residual_scale": condition["residual_scale"],
                "state_effect_matched": condition["state_effect_matched"],
                "random_draw": condition["draw"],
                "delta_fro_mean": condition["delta_fro_mean"],
                "accuracy": slot["correct"] / count,
                "ce": slot["ce"] / count,
                "correct_margin": slot["margin"] / count,
                "answer_rms": slot["answer_rms"] / count,
                "post_J_age_mae": slot["age_abs_sum"] / age_count,
                "post_J_age_signed_error": slot["age_signed_sum"] / age_count,
                "mean_effect_rms": slot["effect_rms_sum"] / effect_count,
                "mean_effect_scale": slot["effect_scale_sum"] / effect_count,
                "final_examples": int(count),
            }
        )
    return rows


def plot_results(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    selected_names = [
        "baseline",
        "bottom4_scale0",
        "bottom8_scale0p5",
        "bottom8_scale0",
        "bottom16_scale0",
        "random8_scale0_state_d0",
    ]
    for name in selected_names:
        subset = sorted(
            [row for row in rows if row["condition"] == name],
            key=lambda row: int(row["back_count"]),
        )
        if not subset:
            continue
        x = [int(row["back_count"]) for row in subset]
        axes[0].plot(x, [float(row["accuracy"]) for row in subset], marker="o", label=name)
        axes[1].plot(
            x,
            [float(row["post_J_age_signed_error"]) for row in subset],
            marker="o",
            label=name,
        )
    axes[0].set(title="Does extra suppression extend lifespan?", xlabel="J calls k", ylabel="final accuracy", ylim=(0, 1.03))
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(title="Age probe after J", xlabel="J calls k", ylabel="predicted age - target age")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "sign_convention": "larger harmful-direction residue is worse; J suppression is cleaning",
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
    age_weight = torch.as_tensor(probe["post_J_full_age_weight"], dtype=torch.float32, device=device)
    age_bias = torch.as_tensor(probe["post_J_full_age_bias"], dtype=torch.float32, device=device)
    conditions = make_conditions(weights, random_draws=args.random_draws)
    if args.conditions:
        requested = {"baseline", *args.conditions}
        available = {condition["name"] for condition in conditions}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown conditions: {missing}")
        conditions = [c for c in conditions if c["name"] in requested]
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
    write_csv(args.out_dir / "extra_suppression_metrics.csv", rows)
    plot_results(rows, args.out_dir / "extra_suppression.png")
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "back_counts": list(args.back_counts),
        "conditions": [condition["name"] for condition in conditions],
        "sign_convention": "larger harmful-direction residue is worse; J suppression is cleaning",
        "claim_boundary": "Selective lifespan extension over state-effect-matched random suppression supports a natural limiting role; otherwise evidence remains restricted to artificial under-reset.",
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            **summary,
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "probe_artifact": str(args.probe_artifact),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
