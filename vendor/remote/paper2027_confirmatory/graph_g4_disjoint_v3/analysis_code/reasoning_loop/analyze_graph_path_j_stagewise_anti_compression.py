"""Directly test weak-channel age gating for every one of the seven J maps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.analyze_graph_path_j_anti_compression import build_stage_delta
from reasoning_loop.analyze_graph_path_j_attention_circuit import target_margin
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


@dataclass(frozen=True)
class StageCondition:
    name: str
    rank: int
    floor: float
    mode: str
    draw: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--probe-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--graph-seeds", type=int, nargs="+", default=(874001, 874002, 874003)
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8))
    parser.add_argument(
        "--floors", type=float, nargs="+", default=(0.1, 0.25, 0.5, 0.75, 1.0)
    )
    parser.add_argument("--random-draws", type=int, default=2)
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def build_conditions(
    *, ranks: Sequence[int], floors: Sequence[float], random_draws: int
) -> list[StageCondition]:
    conditions = [StageCondition("baseline", 0, 0.0, "baseline", -1)]
    for rank in ranks:
        for floor in floors:
            suffix = str(floor).replace(".", "p")
            conditions.append(
                StageCondition(
                    f"bottom{rank}_floor{suffix}",
                    int(rank),
                    float(floor),
                    "bottom",
                    -1,
                )
            )
            for draw in range(random_draws):
                conditions.append(
                    StageCondition(
                        f"random{rank}_floor{suffix}_d{draw}_statematch",
                        int(rank),
                        float(floor),
                        "random",
                        draw,
                    )
                )
    return conditions


def _per_example_norm(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().sum(-1).sqrt()


def make_stage_deltas(
    *,
    weights: dict[int, np.ndarray],
    conditions: Sequence[StageCondition],
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    deltas: dict[str, dict[int, torch.Tensor]] = {}
    for condition in conditions:
        if condition.mode == "baseline":
            deltas[condition.name] = {}
            continue
        stage: dict[int, torch.Tensor] = {}
        for source_age in AGES:
            rng = (
                np.random.default_rng(
                    875000 + condition.draw * 1000 + source_age * 31 + condition.rank
                )
                if condition.mode == "random"
                else None
            )
            value, _ = build_stage_delta(
                weights[source_age],
                rank=condition.rank,
                mode=(
                    "bottom_floor" if condition.mode == "bottom" else "random_matched"
                ),
                target=condition.floor,
                rng=rng,
            )
            stage[source_age] = torch.as_tensor(
                value, dtype=torch.float32, device=device
            )
        deltas[condition.name] = stage
    return deltas


@torch.no_grad()
def apply_direct_j(
    *,
    bank: AgeSpecificJBank,
    input_state: torch.Tensor,
    source_age: int,
    positions: tuple[int, ...],
    condition: StageCondition,
    deltas: dict[str, dict[int, torch.Tensor]],
    bottom_reference: dict[tuple[int, float], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    state = bank.rollback(input_state, source_age=source_age, positions=positions)
    if condition.mode == "baseline":
        return state, {"delta_rms": 0.0, "match_scale_mean": 1.0}
    index = list(positions)
    effect = input_state[:, index].float() @ deltas[condition.name][source_age]
    scale = torch.ones(input_state.shape[0], device=input_state.device)
    if condition.mode == "random":
        reference = (
            input_state[:, index].float()
            @ bottom_reference[(condition.rank, condition.floor)]
        )
        scale = _per_example_norm(reference) / _per_example_norm(effect).clamp_min(1e-12)
        effect = effect * scale[:, None, None]
    state[:, index] = (state[:, index].float() + effect).to(state.dtype)
    return state, {
        "delta_rms": float(effect.square().mean().sqrt()),
        "match_scale_mean": float(scale.mean()),
    }


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    weights: dict[int, np.ndarray],
    conditions: Sequence[StageCondition],
    age_weight: torch.Tensor,
    age_bias: torch.Tensor,
    examples: int,
    batch_size: int,
    graph_seeds: Sequence[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch size")
    positions = tuple(range(cfg.seq_len))
    deltas = make_stage_deltas(weights=weights, conditions=conditions, device=device)
    rows: list[dict[str, Any]] = []
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        print(json.dumps({"event": "stagewise_seed_start", "graph_seed": graph_seed}), flush=True)
        slots: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for _ in range(examples // batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            state = model.apply_loop(raw, loop_index=0)
            current = advance_nodes(successors, start, steps=1)
            natural_states: dict[int, torch.Tensor] = {}
            natural_currents: dict[int, torch.Tensor] = {}
            for source_age in AGES:
                state = model.apply_loop(state, loop_index=source_age - 1)
                current = advance_nodes(successors, current, steps=1)
                natural_states[source_age] = state
                natural_currents[source_age] = current
            for source_age in AGES:
                input_state = natural_states[source_age]
                input_current = natural_currents[source_age]
                bottom_reference: dict[tuple[int, float], torch.Tensor] = {}
                for rank in {condition.rank for condition in conditions if condition.rank}:
                    for floor in {condition.floor for condition in conditions if condition.rank == rank}:
                        name = f"bottom{rank}_floor{str(floor).replace('.', 'p')}"
                        bottom_reference[(rank, floor)] = deltas[name][source_age]
                for condition in conditions:
                    post_j, stats = apply_direct_j(
                        bank=bank,
                        input_state=input_state,
                        source_age=source_age,
                        positions=positions,
                        condition=condition,
                        deltas=deltas,
                        bottom_reference=bottom_reference,
                    )
                    target_age = source_age - 1
                    age_prediction = post_j[:, -1].float() @ age_weight + age_bias
                    post_logits = logits_from_raw_state(model, post_j).float()
                    post_current_accuracy = post_logits.argmax(-1).eq(input_current)
                    continuation = post_j
                    continuation_current = input_current
                    immediate_correct = None
                    for logical_age in range(target_age, 8):
                        continuation = model.apply_loop(
                            continuation, loop_index=logical_age
                        )
                        continuation_current = advance_nodes(
                            successors, continuation_current, steps=1
                        )
                        if immediate_correct is None:
                            immediate_logits = logits_from_raw_state(
                                model, continuation
                            ).float()
                            immediate_correct = immediate_logits.argmax(-1).eq(
                                continuation_current
                            )
                    final_logits = logits_from_raw_state(model, continuation).float()
                    slot = slots[(source_age, condition.name)]
                    slot["count"] += batch_size
                    slot["post_current_correct"] += float(post_current_accuracy.sum())
                    slot["immediate_correct"] += float(immediate_correct.sum())
                    slot["final_correct"] += float(
                        final_logits.argmax(-1).eq(continuation_current).sum()
                    )
                    slot["final_margin"] += float(
                        target_margin(final_logits, continuation_current).sum()
                    )
                    slot["age_error"] += float(
                        (age_prediction - target_age).sum()
                    )
                    slot["age_abs_error"] += float(
                        (age_prediction - target_age).abs().sum()
                    )
                    slot["delta_rms"] += stats["delta_rms"] * batch_size
                    slot["match_scale"] += stats["match_scale_mean"] * batch_size
        lookup = {condition.name: condition for condition in conditions}
        for (source_age, name), slot in slots.items():
            condition = lookup[name]
            count = slot["count"]
            rows.append(
                {
                    "graph_seed": int(graph_seed),
                    "source_age": source_age,
                    "target_age": source_age - 1,
                    "condition": name,
                    "mode": condition.mode,
                    "rank": condition.rank,
                    "floor": condition.floor,
                    "random_draw": condition.draw,
                    "post_J_current_accuracy": slot["post_current_correct"] / count,
                    "next_F_accuracy": slot["immediate_correct"] / count,
                    "final_H8_accuracy": slot["final_correct"] / count,
                    "final_margin": slot["final_margin"] / count,
                    "post_J_age_signed_error": slot["age_error"] / count,
                    "post_J_age_mae": slot["age_abs_error"] / count,
                    "delta_rms": slot["delta_rms"] / count,
                    "match_scale_mean": slot["match_scale"] / count,
                    "examples": int(count),
                }
            )
        print(json.dumps({"event": "stagewise_seed_complete", "graph_seed": graph_seed}), flush=True)
    return rows


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["source_age"]), str(row["condition"]))].append(row)
    output: list[dict[str, Any]] = []
    for (source_age, condition), parts in sorted(groups.items()):
        result: dict[str, Any] = {
            "source_age": source_age,
            "target_age": source_age - 1,
            "condition": condition,
            "mode": parts[0]["mode"],
            "rank": parts[0]["rank"],
            "floor": parts[0]["floor"],
            "random_draw": parts[0]["random_draw"],
            "graph_seeds": len(parts),
        }
        for metric in (
            "post_J_current_accuracy",
            "next_F_accuracy",
            "final_H8_accuracy",
            "final_margin",
            "post_J_age_signed_error",
            "post_J_age_mae",
            "delta_rms",
            "match_scale_mean",
        ):
            values = np.asarray([float(part[metric]) for part in parts])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sem"] = float(
                values.std(ddof=1) / np.sqrt(len(values))
            ) if len(values) > 1 else 0.0
        output.append(result)
    return output


def plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180, sharex=True, sharey="row")
    for column, source_age in enumerate((2, 4, 6, 8)):
        selected = [row for row in rows if int(row["source_age"]) == source_age]
        baseline = next(row for row in selected if row["condition"] == "baseline")
        for rank, marker in ((4, "o"), (8, "s")):
            bottom = sorted(
                [row for row in selected if row["mode"] == "bottom" and int(row["rank"]) == rank],
                key=lambda row: float(row["floor"]),
            )
            x = [0.0] + [float(row["floor"]) for row in bottom]
            axes[0, column].plot(
                x,
                [float(baseline["post_J_age_signed_error_mean"])]
                + [float(row["post_J_age_signed_error_mean"]) for row in bottom],
                marker=marker,
                label=f"bottom-{rank}",
            )
            axes[1, column].plot(
                x,
                [float(baseline["final_H8_accuracy_mean"])]
                + [float(row["final_H8_accuracy_mean"]) for row in bottom],
                marker=marker,
                label=f"bottom-{rank}",
            )
        axes[0, column].set_title(f"J{source_age-1}: H{source_age}->H{source_age-1}")
        axes[0, column].axhline(0, color="black", linewidth=0.7)
        axes[1, column].set_ylim(0, 1.03)
        axes[1, column].set_xlabel("singular floor")
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    axes[0, 0].set_ylabel("post-J predicted age error")
    axes[1, 0].set_ylabel("final H8 accuracy")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            "status": "running",
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "probe_artifact": str(args.probe_artifact),
            "pid": os.getpid(),
        },
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        fraction = float(os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.12"))
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    bank, weights, _, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        tuple(args.ranks),
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
    conditions = build_conditions(
        ranks=tuple(args.ranks),
        floors=tuple(args.floors),
        random_draws=args.random_draws,
    )
    rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        weights=weights,
        conditions=conditions,
        age_weight=age_weight,
        age_bias=age_bias,
        examples=args.examples,
        batch_size=args.batch_size,
        graph_seeds=tuple(args.graph_seeds),
        device=device,
    )
    summary_rows = aggregate(rows)
    write_csv(args.out_dir / "stagewise_rows.csv", rows)
    write_csv(args.out_dir / "stagewise_summary.csv", summary_rows)
    plot(summary_rows, args.out_dir / "stagewise_anti_compression.png")
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "probe_artifact": str(args.probe_artifact),
        "graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "ranks": list(args.ranks),
        "floors": list(args.floors),
        "random_draws": args.random_draws,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "claim_boundary": (
            "This directly covers J1..J7 one stage at a time. A stage-general age gate "
            "requires dose-monotone post-J age contamination and selective final-task "
            "damage versus per-example state-effect-matched random operator deltas."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
