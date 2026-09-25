"""Compress low-singular V output directions after every graph-path J call."""

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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=(886001, 886002))
    parser.add_argument("--back-counts", type=int, nargs="+", default=(8, 12))
    parser.add_argument("--path-pairs-per-k", type=int, default=1)
    parser.add_argument("--word-seed", type=int, default=886701)
    parser.add_argument(
        "--taus", type=float, nargs="+", default=(0.1, 0.2, 0.3, 0.4, 0.5)
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


def build_v_output_projector(
    weight: np.ndarray, *, tau: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Project onto output singular directions whose singular value is below tau."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    _, singular, right_t = np.linalg.svd(weight)
    selected = singular < tau
    output_basis = right_t[selected].T
    projector = output_basis @ output_basis.T
    return projector, {
        "selected_count": float(np.count_nonzero(selected)),
        "selected_singular_mean": (
            float(singular[selected].mean()) if np.any(selected) else 0.0
        ),
        "projector_fro": float(np.linalg.norm(projector, ord="fro")),
    }


def compress_v_output(value: torch.Tensor, projector: torch.Tensor) -> torch.Tensor:
    """Hard-remove the selected row-vector output components."""
    return value.float() - value.float() @ projector


def make_conditions(
    weights: dict[int, np.ndarray], *, taus: Sequence[float]
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "tau": None,
            "projectors": {},
            "selected_count_mean": 0.0,
            "selected_singular_mean": 0.0,
        }
    ]
    for tau in taus:
        projectors: dict[int, np.ndarray] = {}
        stats = []
        for age in AGES:
            projectors[age], stage_stats = build_v_output_projector(
                weights[age], tau=float(tau)
            )
            stats.append(stage_stats)
        conditions.append(
            {
                "name": f"vzero_tau{tau:g}",
                "tau": float(tau),
                "projectors": projectors,
                "selected_count_mean": float(
                    np.mean([value["selected_count"] for value in stats])
                ),
                "selected_count_min": int(
                    min(value["selected_count"] for value in stats)
                ),
                "selected_count_max": int(
                    max(value["selected_count"] for value in stats)
                ),
                "selected_singular_mean": float(
                    np.mean([value["selected_singular_mean"] for value in stats])
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
    projectors: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    state = h1
    current = h1_current
    logical_age = MIN_AGE
    removed_answer_rms = 0.0
    removed_answer_fraction = 0.0
    intervention_count = 0.0
    index = list(positions)
    for action in actions:
        if action == 1:
            state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
        else:
            source_age = logical_age
            state = bank.rollback(state, source_age=source_age, positions=positions)
            if projectors:
                post_j = state[:, index].float()
                component = post_j @ projectors[source_age]
                answer = post_j[:, -1]
                answer_component = component[:, -1]
                state[:, index] = (post_j - component).to(state.dtype)
                removed_answer_rms += float(
                    answer_component.square().mean(-1).sqrt().sum()
                )
                removed_answer_fraction += float(
                    (
                        answer_component.norm(dim=-1)
                        / answer.norm(dim=-1).clamp_min(1e-12)
                    ).sum()
                )
                intervention_count += float(answer.shape[0])
            logical_age -= 1
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("trajectory left H1..H8")
    if logical_age != MAX_AGE:
        raise RuntimeError("trajectory did not end at H8")
    return state, current, {
        "removed_answer_rms": removed_answer_rms,
        "removed_answer_fraction": removed_answer_fraction,
        "intervention_count": intervention_count,
    }


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    conditions: list[dict[str, Any]],
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
                "projectors": {
                    age: torch.as_tensor(value, dtype=torch.float32, device=device)
                    for age, value in condition["projectors"].items()
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
                    state, target, compression_stats = execute_word(
                        model=model,
                        bank=bank,
                        h1=h1,
                        h1_current=h1_current,
                        successors=successors,
                        actions=actions,
                        positions=positions,
                        projectors=condition["projectors"],
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
                    for key, value in compression_stats.items():
                        slot[key] += value

    lookup = {condition["name"]: condition for condition in conditions}
    rows = []
    for (back_count, name), slot in sorted(slots.items()):
        condition = lookup[name]
        count = slot["count"]
        intervention_count = slot["intervention_count"]
        rows.append(
            {
                "back_count": back_count,
                "condition": name,
                "tau": condition["tau"],
                "selected_count_mean": condition["selected_count_mean"],
                "selected_count_min": condition.get("selected_count_min", 0),
                "selected_count_max": condition.get("selected_count_max", 0),
                "selected_singular_mean": condition["selected_singular_mean"],
                "accuracy": slot["correct"] / count,
                "ce": slot["ce"] / count,
                "correct_margin": slot["margin"] / count,
                "answer_rms": slot["answer_rms"] / count,
                "post_J_removed_answer_rms": (
                    slot["removed_answer_rms"] / intervention_count
                    if intervention_count
                    else 0.0
                ),
                "post_J_removed_answer_fraction": (
                    slot["removed_answer_fraction"] / intervention_count
                    if intervention_count
                    else 0.0
                ),
                "final_examples": int(count),
                "post_J_interventions": int(intervention_count),
            }
        )
    return rows


def plot_accuracy(rows: list[dict[str, Any]], path: Path) -> None:
    series = {
        "baseline": {"label": "raw J", "color": "#111827", "linewidth": 2.8},
        "vzero_tau0.1": {"label": r"$\tau=0.1$", "color": "#2563eb"},
        "vzero_tau0.2": {"label": r"$\tau=0.2$", "color": "#0891b2"},
        "vzero_tau0.3": {"label": r"$\tau=0.3$", "color": "#16a34a"},
        "vzero_tau0.4": {"label": r"$\tau=0.4$", "color": "#f59e0b"},
        "vzero_tau0.5": {"label": r"$\tau=0.5$", "color": "#dc2626"},
    }
    figure, axis = plt.subplots(figsize=(10.5, 6.3), dpi=220)
    for name, style in series.items():
        selected = sorted(
            (int(row["back_count"]), float(row["accuracy"]))
            for row in rows
            if row["condition"] == name
        )
        if not selected:
            continue
        axis.plot(
            [loop for loop, _ in selected],
            [accuracy for _, accuracy in selected],
            label=style["label"],
            color=style["color"],
            linewidth=style.get("linewidth", 2.1),
            marker="o",
            markersize=4.2,
        )
    axis.set_title("$V$-output compression over cumulative loops")
    axis.set_xlabel("Cumulative $J$ calls (loop count)")
    axis.set_ylabel("Final accuracy")
    axis.set_xlim(1, 24)
    axis.set_ylim(0.0, 1.025)
    axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.grid(alpha=0.22)
    axis.legend(loc="lower left", ncol=2, frameon=True, fontsize=9.5)
    axis.text(
        0.995,
        0.025,
        r"After each $J_i$, components in $V_{i,\sigma<\tau}$ are set to zero"
        "\nD8L8 seed0; 5 graph seeds; 128 examples/seed",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color="#374151",
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.examples % args.batch_size:
        raise ValueError("examples must divide by batch size")
    if sorted(set(args.taus)) != list(args.taus):
        raise ValueError("taus must be unique and sorted")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            "status": "running",
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "device_requested": args.device,
            "pid": os.getpid(),
            "intervention": "post-J hard removal of V directions with sigma < tau",
        },
    )
    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    bank, weights, _, _, _ = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (4, 8),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    conditions = make_conditions(weights, taus=tuple(args.taus))
    rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        conditions=conditions,
        examples=args.examples,
        batch_size=args.batch_size,
        graph_seeds=tuple(args.graph_seeds),
        back_counts=tuple(args.back_counts),
        path_pairs_per_k=args.path_pairs_per_k,
        word_seed=args.word_seed,
        device=device,
    )
    write_csv(args.out_dir / "v_output_compression_metrics.csv", rows)
    plot_accuracy(rows, args.out_dir / "acc_vs_loops_v_output_compression.png")
    result = {
        "status": "complete",
        "actual_device": str(device),
        "examples_per_graph_seed": args.examples,
        "graph_seeds": list(args.graph_seeds),
        "back_counts": list(args.back_counts),
        "taus": list(args.taus),
        "conditions": len(conditions),
        "intervention": "post-J hard removal of V directions with sigma < tau",
        "claim_boundary": (
            "This directly tests whether downstream computation tolerates removal of the selected post-J V-output components. "
            "It does not by itself identify which attention or MLP component consumes them."
        ),
    }
    if device.type == "cuda":
        result["peak_cuda_reserved_gib"] = float(
            torch.cuda.max_memory_reserved(device) / (1024**3)
        )
    atomic_json(args.out_dir / "summary.json", result)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            **result,
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
