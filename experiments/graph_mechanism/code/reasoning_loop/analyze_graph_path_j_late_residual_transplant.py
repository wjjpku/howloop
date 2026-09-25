"""Bidirectionally transplant natural late post-J residual errors.

For a late rollback event, find an earlier visit in the same trajectory with
the same graph, original query, graph-current node, and logical age.  A healthy
profile fitted on independent random graphs removes the predictable coordinate
component.  The experiment then isolates the *excess residual* between late
and matched-young states in the shared bottom-output subspace.

Two causal directions are tested from the exact same event:

1. clean the late state by subtracting the excess residual;
2. age the young state by injecting the same excess residual.

Random and top-subspace excess residuals are matched per example to the bottom
hidden perturbation norm.  Branches are scored only on matched examples, and
the branch suffix is identical, so graph progress and target generation cannot
change across intervention conditions.
"""

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

from reasoning_loop.analyze_graph_path_j_attention_circuit import (
    build_words,
    final_state_from_trace,
    target_margin,
)
from reasoning_loop.analyze_graph_path_j_centered_residual_telomere import (
    HealthyProfile,
    _healthy_post_j_states,
    _per_example_norm,
    evaluate_healthy_profiles,
    fit_healthy_profiles,
    write_csv,
)
from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    atomic_json,
    load_bank_and_bases,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import (
    MAX_AGE,
    MIN_AGE,
    action_semantics,
)


@dataclass
class RollbackEvent:
    action_index: int
    rollback_index: int
    source_age: int
    target_age: int
    state: torch.Tensor
    current: torch.Tensor


@dataclass(frozen=True)
class BranchCondition:
    name: str
    receiver: str
    family: str
    direction: str
    strength: float
    draw: int = -1
    state_effect_match: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--profile-mode", choices=("mean", "linear"), default="linear")
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--calibration-seed", type=int, default=865001)
    parser.add_argument("--profile-validation-examples", type=int, default=512)
    parser.add_argument("--profile-validation-seed", type=int, default=865501)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--graph-seeds", type=int, nargs="+", default=(866001, 866002, 866003)
    )
    parser.add_argument("--back-counts", type=int, nargs="+", default=(24, 32, 48))
    parser.add_argument("--path-pairs-per-k", type=int, default=2)
    parser.add_argument("--word-seed", type=int, default=866701)
    parser.add_argument(
        "--rollback-checkpoints", type=int, nargs="+", default=(8, 16, 24, 32, 48)
    )
    parser.add_argument("--random-draws", type=int, default=2)
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional exact condition names. late and young baselines are always included.",
    )
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args(argv)


def build_conditions(
    *, strengths: Sequence[float], random_draws: int
) -> list[BranchCondition]:
    conditions = [
        BranchCondition("late_baseline", "late", "baseline", "none", 0.0),
        BranchCondition("young_reference", "young", "baseline", "none", 0.0),
    ]
    for strength in strengths:
        suffix = str(strength).replace(".", "p")
        conditions.extend(
            (
                BranchCondition(
                    f"late_bottom_excess_clean{suffix}",
                    "late",
                    "bottom",
                    "clean",
                    float(strength),
                ),
                BranchCondition(
                    f"young_bottom_excess_inject{suffix}",
                    "young",
                    "bottom",
                    "inject",
                    float(strength),
                ),
                BranchCondition(
                    f"late_top_excess_clean{suffix}_statematch",
                    "late",
                    "top",
                    "clean",
                    float(strength),
                    state_effect_match=True,
                ),
            )
        )
        for draw in range(random_draws):
            conditions.extend(
                (
                    BranchCondition(
                        f"late_random{draw}_excess_clean{suffix}_statematch",
                        "late",
                        f"random{draw}",
                        "clean",
                        float(strength),
                        draw=draw,
                        state_effect_match=True,
                    ),
                    BranchCondition(
                        f"young_random{draw}_excess_inject{suffix}_statematch",
                        "young",
                        f"random{draw}",
                        "inject",
                        float(strength),
                        draw=draw,
                        state_effect_match=True,
                    ),
                )
            )
    return conditions


@torch.no_grad()
def collect_rollback_events(
    *,
    model,
    bank: AgeSpecificJBank,
    h1: torch.Tensor,
    h1_current: torch.Tensor,
    successors: torch.Tensor,
    actions: Sequence[int],
    positions: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, list[RollbackEvent]]:
    state = h1
    current = h1_current
    logical_age = MIN_AGE
    rollback_index = 0
    events: list[RollbackEvent] = []
    for action_index, action in enumerate(actions):
        if action == 1:
            state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
        elif action == -1:
            source_age = logical_age
            state = bank.rollback(state, source_age=source_age, positions=positions)
            logical_age -= 1
            rollback_index += 1
            events.append(
                RollbackEvent(
                    action_index=action_index,
                    rollback_index=rollback_index,
                    source_age=source_age,
                    target_age=logical_age,
                    state=state,
                    current=current,
                )
            )
        else:
            raise ValueError("actions must be +1 or -1")
    if logical_age != MAX_AGE:
        raise RuntimeError("trajectory did not end at H8")
    return state, current, events


def gather_matched_young(
    events: Sequence[RollbackEvent], event_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    event = events[event_index]
    batch_size = event.state.shape[0]
    source = torch.full(
        (batch_size,), event_index, dtype=torch.long, device=event.state.device
    )
    for earlier_index in range(event_index):
        earlier = events[earlier_index]
        if earlier.target_age != event.target_age:
            continue
        matches = earlier.current.eq(event.current)
        source[matches & source.eq(event_index)] = earlier_index
    stack = torch.stack([candidate.state for candidate in events], dim=0)
    batch = torch.arange(batch_size, device=event.state.device)
    return stack[source, batch], source.ne(event_index)


def excess_ambient_residual(
    *,
    late: torch.Tensor,
    young: torch.Tensor,
    profile: HealthyProfile,
    profile_mode: str,
) -> torch.Tensor:
    excess_coordinates = profile.residual(late, profile_mode) - profile.residual(
        young, profile_mode
    )
    return excess_coordinates @ profile.basis.T


def make_branch_state(
    *,
    late: torch.Tensor,
    young: torch.Tensor,
    source_age: int,
    condition: BranchCondition,
    profiles: dict[str, dict[int, HealthyProfile]],
    profile_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    receiver = late if condition.receiver == "late" else young
    if condition.family == "baseline":
        return receiver, {
            "delta_rms": 0.0,
            "match_scale_mean": 1.0,
            "match_scale_max": 1.0,
        }
    profile = profiles[condition.family][source_age]
    excess = excess_ambient_residual(
        late=late,
        young=young,
        profile=profile,
        profile_mode=profile_mode,
    )
    sign = -1.0 if condition.direction == "clean" else 1.0
    delta = sign * condition.strength * excess
    scale = torch.ones(receiver.shape[0], device=receiver.device)
    if condition.state_effect_match:
        bottom_excess = excess_ambient_residual(
            late=late,
            young=young,
            profile=profiles["bottom"][source_age],
            profile_mode=profile_mode,
        )
        target = condition.strength * bottom_excess
        scale = _per_example_norm(target) / _per_example_norm(delta).clamp_min(1e-12)
        delta = delta * scale[:, None, None]
    return (receiver.float() + delta).to(receiver.dtype), {
        "delta_rms": float(delta.square().mean().sqrt()),
        "match_scale_mean": float(scale.mean()),
        "match_scale_max": float(scale.max()),
    }


def _destination_mass(trace, current: torch.Tensor, cfg) -> torch.Tensor:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = torch.as_tensor(
        groups["destination"], dtype=torch.long, device=current.device
    )
    batch = torch.arange(current.shape[0], device=current.device)
    pattern = trace.sites[1].attention_pattern[:, 0, answer_position]
    return pattern[batch, destination_positions[current]]


@torch.no_grad()
def execute_suffix(
    *,
    model,
    bank: AgeSpecificJBank,
    state: torch.Tensor,
    current: torch.Tensor,
    successors: torch.Tensor,
    start_age: int,
    actions: Sequence[int],
    positions: tuple[int, ...],
    cfg,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    logical_age = start_age
    alive = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    survival = torch.zeros(state.shape[0], dtype=torch.float32, device=state.device)
    correct_sum = torch.zeros_like(survival)
    forward_count = 0
    first_destination_mass: torch.Tensor | None = None
    for action in actions:
        if action == 1:
            if forward_count == 0:
                _, trace = run_instrumented_state(
                    model, state, loop_indices=(logical_age,)
                )
                first_destination_mass = _destination_mass(trace, current, cfg)
                state = final_state_from_trace(model, trace)
            else:
                state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
            logits = logits_from_raw_state(model, state).float()
            correct = logits.argmax(-1).eq(current)
            alive = alive & correct
            survival += alive.float()
            correct_sum += correct.float()
            forward_count += 1
        elif action == -1:
            state = bank.rollback(state, source_age=logical_age, positions=positions)
            logical_age -= 1
        else:
            raise ValueError("actions must be +1 or -1")
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("suffix left H1..H8")
    if logical_age != MAX_AGE:
        raise RuntimeError("suffix did not end at H8")
    if first_destination_mass is None:
        raise RuntimeError("selected rollback event was not followed by an F")
    return state, current, {
        "survival": survival,
        "forward_accuracy_auc": correct_sum / max(forward_count, 1),
        "first_destination_mass": first_destination_mass,
        "suffix_forward_count": float(forward_count),
    }


def _aggregate(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, parts in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, group, strict=True))
        numeric = sorted(
            {
                key
                for part in parts
                for key, value in part.items()
                if key not in keys and isinstance(value, (int, float))
            }
        )
        for key in numeric:
            values = [float(part[key]) for part in parts if key in part]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        result["rows"] = len(parts)
        output.append(result)
    return output


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    profiles: dict[str, dict[int, HealthyProfile]],
    conditions: Sequence[BranchCondition],
    profile_mode: str,
    graph_seeds: Sequence[int],
    examples: int,
    batch_size: int,
    words: Sequence[tuple[int, str, tuple[int, ...]]],
    rollback_checkpoints: set[int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch size")
    positions = tuple(range(cfg.seq_len))
    rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        print(
            json.dumps(
                {
                    "event": "late_residual_seed_start",
                    "graph_seed": int(graph_seed),
                    "conditions": len(conditions),
                    "words": len(words),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for batch_index in range(examples // batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            h1 = model.apply_loop(raw, loop_index=0)
            h1_current = advance_nodes(successors, start, steps=1)
            for back_count, word, actions in words:
                baseline_state, baseline_target, events = collect_rollback_events(
                    model=model,
                    bank=bank,
                    h1=h1,
                    h1_current=h1_current,
                    successors=successors,
                    actions=actions,
                    positions=positions,
                )
                baseline_logits = logits_from_raw_state(model, baseline_state).float()
                for event_index, event in enumerate(events):
                    if event.rollback_index not in rollback_checkpoints:
                        continue
                    suffix = actions[event.action_index + 1 :]
                    if not suffix or suffix[0] != 1:
                        continue
                    young, matched = gather_matched_young(events, event_index)
                    matched_count = int(matched.sum())
                    match_rows.append(
                        {
                            "graph_seed": int(graph_seed),
                            "batch": batch_index,
                            "back_count": back_count,
                            "word": word,
                            "rollback_checkpoint": event.rollback_index,
                            "source_age": event.source_age,
                            "target_age": event.target_age,
                            "matched_fraction": float(matched.float().mean()),
                            "matched_examples": matched_count,
                        }
                    )
                    if matched_count == 0:
                        continue
                    for condition in conditions:
                        branch_state, stats = make_branch_state(
                            late=event.state,
                            young=young,
                            source_age=event.source_age,
                            condition=condition,
                            profiles=profiles,
                            profile_mode=profile_mode,
                        )
                        final_state, final_target, trajectory = execute_suffix(
                            model=model,
                            bank=bank,
                            state=branch_state,
                            current=event.current,
                            successors=successors,
                            start_age=event.target_age,
                            actions=suffix,
                            positions=positions,
                            cfg=cfg,
                        )
                        if not torch.equal(final_target, baseline_target):
                            raise RuntimeError("branch suffix changed the graph target")
                        logits = logits_from_raw_state(model, final_state).float()
                        selected_logits = logits[matched]
                        selected_target = final_target[matched]
                        rows.append(
                            {
                                "graph_seed": int(graph_seed),
                                "batch": batch_index,
                                "back_count": back_count,
                                "word": word,
                                "rollback_checkpoint": event.rollback_index,
                                "source_age": event.source_age,
                                "target_age": event.target_age,
                                "condition": condition.name,
                                "receiver": condition.receiver,
                                "family": condition.family,
                                "direction": condition.direction,
                                "strength": condition.strength,
                                "state_effect_match": condition.state_effect_match,
                                "matched_fraction": float(matched.float().mean()),
                                "matched_examples": matched_count,
                                "accuracy": float(
                                    selected_logits.argmax(-1)
                                    .eq(selected_target)
                                    .float()
                                    .mean()
                                ),
                                "baseline_fullword_accuracy": float(
                                    baseline_logits.argmax(-1)
                                    .eq(baseline_target)
                                    .float()
                                    .mean()
                                ),
                                "margin": float(
                                    target_margin(selected_logits, selected_target).mean()
                                ),
                                "ce": float(F.cross_entropy(selected_logits, selected_target)),
                                "survival_steps": float(
                                    trajectory["survival"][matched].mean()
                                ),
                                "forward_accuracy_auc": float(
                                    trajectory["forward_accuracy_auc"][matched].mean()
                                ),
                                "first_destination_mass": float(
                                    trajectory["first_destination_mass"][matched].mean()
                                ),
                                "suffix_forward_count": float(
                                    trajectory["suffix_forward_count"]
                                ),
                                **stats,
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "late_residual_seed_complete",
                    "graph_seed": int(graph_seed),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows, match_rows


def plot_results(rows: Sequence[dict[str, Any]], path: Path) -> None:
    selected = (
        "late_baseline",
        "young_reference",
        "late_bottom_excess_clean1p0",
        "young_bottom_excess_inject1p0",
        "late_random0_excess_clean1p0_statematch",
        "young_random0_excess_inject1p0_statematch",
    )
    checkpoints = sorted({int(row["rollback_checkpoint"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    for condition in selected:
        values = [row for row in rows if row["condition"] == condition]
        if not values:
            continue
        lookup = {int(row["rollback_checkpoint"]): row for row in values}
        x = [checkpoint for checkpoint in checkpoints if checkpoint in lookup]
        axes[0].plot(
            x,
            [float(lookup[value]["accuracy_mean"]) for value in x],
            marker="o",
            label=condition,
        )
        axes[1].plot(
            x,
            [float(lookup[value]["survival_steps_mean"]) for value in x],
            marker="o",
            label=condition,
        )
    axes[0].set(title="Final suffix accuracy", xlabel="rollback checkpoint", ylabel="accuracy", ylim=(0, 1.03))
    axes[1].set(title="Remaining correct F steps", xlabel="rollback checkpoint", ylabel="survival steps")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
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
    bank, _, common_bases, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (args.rank,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    rng = np.random.default_rng(867001)
    basis_arrays = {
        "bottom": common_bases["bottom_output"][args.rank],
        "top": common_bases["top_output"][args.rank],
    }
    for draw in range(args.random_draws):
        value, _ = np.linalg.qr(rng.standard_normal((cfg.d_model, args.rank)))
        basis_arrays[f"random{draw}"] = value[:, : args.rank]
    fit_states = _healthy_post_j_states(
        model=model,
        bank=bank,
        cfg=cfg,
        examples=args.calibration_examples,
        batch_size=args.batch_size,
        seed=args.calibration_seed,
        device=device,
    )
    profiles, fit_rows = fit_healthy_profiles(
        healthy_states=fit_states,
        bases=basis_arrays,
        ridge=args.ridge,
        device=device,
    )
    del fit_states
    validation_states = _healthy_post_j_states(
        model=model,
        bank=bank,
        cfg=cfg,
        examples=args.profile_validation_examples,
        batch_size=args.batch_size,
        seed=args.profile_validation_seed,
        device=device,
    )
    profile_rows = [{"split": "fit", **row} for row in fit_rows]
    profile_rows.extend(
        evaluate_healthy_profiles(
            healthy_states=validation_states,
            profiles=profiles,
            split="held_out_graphs",
            device=device,
        )
    )
    del validation_states
    conditions = build_conditions(
        strengths=tuple(args.strengths), random_draws=args.random_draws
    )
    if args.conditions:
        requested = {"late_baseline", "young_reference", *args.conditions}
        available = {condition.name for condition in conditions}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown requested conditions: {missing}")
        conditions = [condition for condition in conditions if condition.name in requested]
    words = build_words(
        back_counts=tuple(args.back_counts),
        path_pairs_per_k=args.path_pairs_per_k,
        seed=args.word_seed,
    )
    rows, match_rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        profiles=profiles,
        conditions=conditions,
        profile_mode=args.profile_mode,
        graph_seeds=tuple(args.graph_seeds),
        examples=args.examples,
        batch_size=args.batch_size,
        words=words,
        rollback_checkpoints=set(args.rollback_checkpoints),
        device=device,
    )
    aggregate = _aggregate(
        rows, ("back_count", "rollback_checkpoint", "condition")
    )
    match_summary = _aggregate(
        match_rows, ("back_count", "rollback_checkpoint")
    )
    write_csv(args.out_dir / "healthy_profile_fit.csv", profile_rows)
    write_csv(args.out_dir / "branch_rows.csv", rows)
    write_csv(args.out_dir / "branch_summary.csv", aggregate)
    write_csv(args.out_dir / "match_rows.csv", match_rows)
    write_csv(args.out_dir / "match_summary.csv", match_summary)
    plot_results(aggregate, args.out_dir / "late_residual_transplant.png")
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "rank": args.rank,
        "ridge": args.ridge,
        "profile_mode": args.profile_mode,
        "calibration_examples": args.calibration_examples,
        "calibration_seed": args.calibration_seed,
        "profile_validation_examples": args.profile_validation_examples,
        "profile_validation_seed": args.profile_validation_seed,
        "graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "back_counts": list(args.back_counts),
        "rollback_checkpoints": list(args.rollback_checkpoints),
        "path_pairs_per_k": args.path_pairs_per_k,
        "conditions": [condition.name for condition in conditions],
        "words": [
            {
                "back_count": back_count,
                "label": label,
                "actions": list(actions),
                "semantics": action_semantics(actions).__dict__,
            }
            for back_count, label, actions in words
        ],
        "match_definition": (
            "earliest earlier post-J event in the same action word with identical "
            "graph, original query, graph-current node, and target logical age"
        ),
        "excess_definition": (
            "profile residual(late)-profile residual(young), not raw young-coordinate copying"
        ),
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "claim_boundary": (
            "A natural telomere driver requires selective late-state rescue and young-state "
            "damage versus state-effect-matched controls, together with improved remaining "
            "correct-step survival and destination routing."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
