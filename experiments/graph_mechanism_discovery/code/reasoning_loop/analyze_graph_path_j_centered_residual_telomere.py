"""Causally test graph-blind post-J residual cleaning and amplification.

The weak output directions of the seven rollback maps are hypothesized to gate
old loop-phase residue.  Previous experiments changed the operator singular
values or copied a younger donor's coordinates.  Neither operation directly
removes the *abnormal post-J residual* in the current state.

This experiment first fits a healthy, stage-specific post-J profile on an
independent set of random permutation graphs.  At evaluation time it never
reads a graph-matched young donor.  Instead, after every legal rollback J_a it
estimates the deviation from the healthy profile in a fixed rank-r subspace and
either shrinks or amplifies that deviation.  Random and top-subspace controls
are matched per example to the actual hidden-state perturbation norm.
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
    target_margin,
)
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
    action_semantics,
)


@dataclass(frozen=True)
class HealthyProfile:
    """Graph-blind prediction of healthy coordinates for one stage/subspace."""

    basis: torch.Tensor
    mean_by_position: torch.Tensor
    complement_weight: torch.Tensor
    position_bias: torch.Tensor

    def predict(self, state: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mean":
            return self.mean_by_position.unsqueeze(0).expand(state.shape[0], -1, -1)
        if mode != "linear":
            raise ValueError(f"unknown healthy-profile mode: {mode}")
        coordinates = state.float() @ self.basis
        complement = state.float() - coordinates @ self.basis.T
        return complement @ self.complement_weight + self.position_bias.unsqueeze(0)

    def residual(self, state: torch.Tensor, mode: str) -> torch.Tensor:
        return state.float() @ self.basis - self.predict(state, mode)

    def ambient_residual(self, state: torch.Tensor, mode: str) -> torch.Tensor:
        return self.residual(state, mode) @ self.basis.T


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    profile_mode: str
    direction: str
    strength: float
    draw: int = -1
    state_effect_match: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--probe-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--calibration-seed", type=int, default=861001)
    parser.add_argument("--profile-validation-examples", type=int, default=512)
    parser.add_argument("--profile-validation-seed", type=int, default=861501)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--graph-seeds", type=int, nargs="+", default=(862001, 862002, 862003)
    )
    parser.add_argument(
        "--back-counts", type=int, nargs="+", default=(16, 24, 32, 48)
    )
    parser.add_argument("--path-pairs-per-k", type=int, default=2)
    parser.add_argument("--word-seed", type=int, default=862701)
    parser.add_argument("--random-draws", type=int, default=2)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0)
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional exact condition names. Baseline is always included.",
    )
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


@torch.no_grad()
def _healthy_post_j_states(
    *,
    model,
    bank: AgeSpecificJBank,
    cfg,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Collect one clean natural-prefix rollback for every source age."""

    if examples % batch_size:
        raise ValueError("calibration examples must be divisible by batch size")
    stores: dict[int, list[np.ndarray]] = {age: [] for age in AGES}
    positions = tuple(range(cfg.seq_len))
    set_seed(seed)
    total_batches = examples // batch_size
    print(
        json.dumps(
            {
                "event": "healthy_profile_collection_start",
                "seed": seed,
                "examples": examples,
                "batches": total_batches,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for batch_index in range(total_batches):
        tokens, _, _, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        state = model.apply_loop(raw, loop_index=0)
        for source_age in AGES:
            # ``state`` is H_(source_age-1) at loop entry.  The loop_index is
            # zero based, so the transition into H_source_age uses index
            # source_age-1 (H1 used index 0 above).
            state = model.apply_loop(state, loop_index=source_age - 1)
            rolled = bank.rollback(
                state, source_age=source_age, positions=positions
            )
            stores[source_age].append(rolled.float().cpu().numpy())
        if (batch_index + 1) % max(1, total_batches // 4) == 0:
            print(
                json.dumps(
                    {
                        "event": "healthy_profile_collection_progress",
                        "seed": seed,
                        "batch": batch_index + 1,
                        "total_batches": total_batches,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {age: np.concatenate(parts, axis=0) for age, parts in stores.items()}


def _fit_profile(
    states: np.ndarray,
    basis: np.ndarray,
    *,
    ridge: float,
    device: torch.device,
) -> HealthyProfile:
    """Fit healthy subspace coordinates from the orthogonal complement.

    The predictor is shared across graph instances and uses only the current
    token's complement coordinates plus a token-position bias.  It therefore
    cannot retrieve a graph-matched donor or mix information across tokens.
    """

    examples, seq_len, dimension = states.shape
    rank = basis.shape[1]
    coordinates = states @ basis
    ambient_component = coordinates @ basis.T
    complement = states - ambient_component
    x_hidden = complement.reshape(examples * seq_len, dimension).astype(np.float64)
    y = coordinates.reshape(examples * seq_len, rank).astype(np.float64)
    positions = np.tile(np.arange(seq_len), examples)
    one_hot = np.eye(seq_len, dtype=np.float64)[positions]
    design = np.concatenate((x_hidden, one_hot), axis=1)
    gram = design.T @ design
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    coefficient = np.linalg.solve(
        gram + np.eye(gram.shape[0]) * ridge * max(scale, 1e-12),
        design.T @ y,
    )
    return HealthyProfile(
        basis=torch.as_tensor(basis, dtype=torch.float32, device=device),
        mean_by_position=torch.as_tensor(
            coordinates.mean(axis=0), dtype=torch.float32, device=device
        ),
        complement_weight=torch.as_tensor(
            coefficient[:dimension], dtype=torch.float32, device=device
        ),
        position_bias=torch.as_tensor(
            coefficient[dimension:], dtype=torch.float32, device=device
        ),
    )


def fit_healthy_profiles(
    *,
    healthy_states: dict[int, np.ndarray],
    bases: dict[str, np.ndarray],
    ridge: float,
    device: torch.device,
) -> tuple[dict[str, dict[int, HealthyProfile]], list[dict[str, Any]]]:
    profiles: dict[str, dict[int, HealthyProfile]] = {
        family: {} for family in bases
    }
    rows: list[dict[str, Any]] = []
    for family, basis in bases.items():
        for source_age, states in healthy_states.items():
            profile = _fit_profile(states, basis, ridge=ridge, device=device)
            profiles[family][source_age] = profile
            sample = torch.as_tensor(states, dtype=torch.float32, device=device)
            for mode in ("mean", "linear"):
                residual = profile.residual(sample, mode)
                rows.append(
                    {
                        "family": family,
                        "source_age": source_age,
                        "profile_mode": mode,
                        "coordinate_rms": float(
                            (sample @ profile.basis).square().mean().sqrt()
                        ),
                        "healthy_residual_rms": float(
                            residual.square().mean().sqrt()
                        ),
                        "observations": int(states.shape[0] * states.shape[1]),
                    }
                )
    return profiles, rows


@torch.no_grad()
def evaluate_healthy_profiles(
    *,
    healthy_states: dict[int, np.ndarray],
    profiles: dict[str, dict[int, HealthyProfile]],
    split: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, stage_profiles in profiles.items():
        for source_age, profile in stage_profiles.items():
            sample = torch.as_tensor(
                healthy_states[source_age], dtype=torch.float32, device=device
            )
            for mode in ("mean", "linear"):
                residual = profile.residual(sample, mode)
                rows.append(
                    {
                        "split": split,
                        "family": family,
                        "source_age": source_age,
                        "profile_mode": mode,
                        "coordinate_rms": float(
                            (sample @ profile.basis).square().mean().sqrt()
                        ),
                        "healthy_residual_rms": float(
                            residual.square().mean().sqrt()
                        ),
                        "observations": int(sample.shape[0] * sample.shape[1]),
                    }
                )
    return rows


def build_conditions(
    *, strengths: Sequence[float], random_draws: int
) -> list[Condition]:
    conditions = [Condition("baseline", "baseline", "none", "none", 0.0)]
    for mode in ("mean", "linear"):
        for strength in strengths:
            suffix = str(strength).replace(".", "p")
            conditions.append(
                Condition(
                    f"bottom_{mode}_shrink{suffix}",
                    "bottom",
                    mode,
                    "shrink",
                    float(strength),
                )
            )
    for strength in strengths:
        suffix = str(strength).replace(".", "p")
        conditions.append(
            Condition(
                f"bottom_linear_expand{suffix}",
                "bottom",
                "linear",
                "expand",
                float(strength),
            )
        )
        conditions.append(
            Condition(
                f"top_linear_shrink{suffix}_statematch",
                "top",
                "linear",
                "shrink",
                float(strength),
                state_effect_match=True,
            )
        )
        for draw in range(random_draws):
            conditions.extend(
                (
                    Condition(
                        f"random{draw}_linear_shrink{suffix}_statematch",
                        f"random{draw}",
                        "linear",
                        "shrink",
                        float(strength),
                        draw=draw,
                        state_effect_match=True,
                    ),
                    Condition(
                        f"random{draw}_linear_expand{suffix}_statematch",
                        f"random{draw}",
                        "linear",
                        "expand",
                        float(strength),
                        draw=draw,
                        state_effect_match=True,
                    ),
                )
            )
    return conditions


def _per_example_norm(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().sum(-1).sqrt()


def residual_delta(
    *,
    state: torch.Tensor,
    profile: HealthyProfile,
    profile_mode: str,
    direction: str,
    strength: float,
) -> torch.Tensor:
    sign = -1.0 if direction == "shrink" else 1.0
    if direction not in {"shrink", "expand"}:
        raise ValueError(f"unknown residual direction: {direction}")
    return sign * float(strength) * profile.ambient_residual(state, profile_mode)


def apply_condition(
    *,
    state: torch.Tensor,
    source_age: int,
    condition: Condition,
    profiles: dict[str, dict[int, HealthyProfile]],
) -> tuple[torch.Tensor, dict[str, float]]:
    if condition.family == "baseline":
        return state, {
            "delta_rms": 0.0,
            "match_scale_mean": 1.0,
            "match_scale_max": 1.0,
        }
    profile = profiles[condition.family][source_age]
    delta = residual_delta(
        state=state,
        profile=profile,
        profile_mode=condition.profile_mode,
        direction=condition.direction,
        strength=condition.strength,
    )
    scale = torch.ones(state.shape[0], device=state.device)
    if condition.state_effect_match:
        bottom_delta = residual_delta(
            state=state,
            profile=profiles["bottom"][source_age],
            profile_mode=condition.profile_mode,
            direction=condition.direction,
            strength=condition.strength,
        )
        scale = _per_example_norm(bottom_delta) / _per_example_norm(delta).clamp_min(
            1e-12
        )
        delta = delta * scale[:, None, None]
    result = (state.float() + delta).to(state.dtype)
    return result, {
        "delta_rms": float(delta.square().mean().sqrt()),
        "match_scale_mean": float(scale.mean()),
        "match_scale_max": float(scale.max()),
    }


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
    condition: Condition,
    profiles: dict[str, dict[int, HealthyProfile]],
    age_weight: torch.Tensor,
    age_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    state = h1
    current = h1_current
    logical_age = MIN_AGE
    rollback_index = 0
    step_rows: list[dict[str, float]] = []
    for action in actions:
        if action == 1:
            state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
        elif action == -1:
            source_age = logical_age
            state = bank.rollback(state, source_age=source_age, positions=positions)
            logical_age -= 1
            rollback_index += 1
            baseline_prediction = state[:, -1].float() @ age_weight + age_bias
            before_bottom = profiles["bottom"][source_age].residual(
                state, "linear"
            )
            state, stats = apply_condition(
                state=state,
                source_age=source_age,
                condition=condition,
                profiles=profiles,
            )
            after_prediction = state[:, -1].float() @ age_weight + age_bias
            after_bottom = profiles["bottom"][source_age].residual(state, "linear")
            step_rows.append(
                {
                    "rollback_index": float(rollback_index),
                    "source_age": float(source_age),
                    "target_age": float(logical_age),
                    "age_error_before": float(
                        (baseline_prediction - logical_age).mean()
                    ),
                    "age_error_after": float((after_prediction - logical_age).mean()),
                    "bottom_residual_rms_before": float(
                        before_bottom.square().mean().sqrt()
                    ),
                    "bottom_residual_rms_after": float(
                        after_bottom.square().mean().sqrt()
                    ),
                    **stats,
                }
            )
        else:
            raise ValueError("actions must be +1 or -1")
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("trajectory left H1..H8")
    if logical_age != MAX_AGE:
        raise RuntimeError("trajectory did not end at H8")
    return state, current, step_rows


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
    conditions: Sequence[Condition],
    age_weight: torch.Tensor,
    age_bias: torch.Tensor,
    graph_seeds: Sequence[int],
    examples: int,
    batch_size: int,
    words: Sequence[tuple[int, str, tuple[int, ...]]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch size")
    positions = tuple(range(cfg.seq_len))
    behavior_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        print(
            json.dumps(
                {
                    "event": "centered_residual_seed_start",
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
                for condition in conditions:
                    state, target, local_steps = execute_word(
                        model=model,
                        bank=bank,
                        h1=h1,
                        h1_current=h1_current,
                        successors=successors,
                        actions=actions,
                        positions=positions,
                        condition=condition,
                        profiles=profiles,
                        age_weight=age_weight,
                        age_bias=age_bias,
                    )
                    logits = logits_from_raw_state(model, state).float()
                    behavior_rows.append(
                        {
                            "graph_seed": int(graph_seed),
                            "batch": batch_index,
                            "back_count": back_count,
                            "word": word,
                            "condition": condition.name,
                            "family": condition.family,
                            "profile_mode": condition.profile_mode,
                            "direction": condition.direction,
                            "strength": condition.strength,
                            "random_draw": condition.draw,
                            "state_effect_match": condition.state_effect_match,
                            "accuracy": float(
                                logits.argmax(-1).eq(target).float().mean()
                            ),
                            "margin": float(target_margin(logits, target).mean()),
                            "ce": float(F.cross_entropy(logits, target)),
                            "answer_rms": float(
                                state[:, -1].float().square().mean().sqrt()
                            ),
                            "examples": int(target.numel()),
                        }
                    )
                    for local in local_steps:
                        step_rows.append(
                            {
                                "graph_seed": int(graph_seed),
                                "batch": batch_index,
                                "back_count": back_count,
                                "word": word,
                                "condition": condition.name,
                                "family": condition.family,
                                "profile_mode": condition.profile_mode,
                                "direction": condition.direction,
                                "strength": condition.strength,
                                **local,
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "centered_residual_seed_complete",
                    "graph_seed": int(graph_seed),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return behavior_rows, step_rows


def plot_results(rows: Sequence[dict[str, Any]], path: Path) -> None:
    selected_names = [
        "baseline",
        "bottom_linear_shrink0p5",
        "bottom_linear_shrink1p0",
        "bottom_linear_expand0p5",
        "random0_linear_shrink1p0_statematch",
        "top_linear_shrink1p0_statematch",
    ]
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=180)
    for name in selected_names:
        selected = [row for row in rows if row["condition"] == name]
        if not selected:
            continue
        axis.plot(
            [int(row["back_count"]) for row in selected],
            [float(row["accuracy_mean"]) for row in selected],
            marker="o",
            label=name,
        )
    axis.set(
        xlabel="number of J calls",
        ylabel="final accuracy",
        ylim=(0, 1.03),
        title="Graph-blind centered post-J residual intervention",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
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
    bank, _, common_bases, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (args.rank,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    rng = np.random.default_rng(863001)
    basis_arrays = {
        "bottom": common_bases["bottom_output"][args.rank],
        "top": common_bases["top_output"][args.rank],
    }
    for draw in range(args.random_draws):
        value, _ = np.linalg.qr(rng.standard_normal((cfg.d_model, args.rank)))
        basis_arrays[f"random{draw}"] = value[:, : args.rank]
    healthy_states = _healthy_post_j_states(
        model=model,
        bank=bank,
        cfg=cfg,
        examples=args.calibration_examples,
        batch_size=args.batch_size,
        seed=args.calibration_seed,
        device=device,
    )
    profiles, calibration_rows = fit_healthy_profiles(
        healthy_states=healthy_states,
        bases=basis_arrays,
        ridge=args.ridge,
        device=device,
    )
    del healthy_states
    validation_states = _healthy_post_j_states(
        model=model,
        bank=bank,
        cfg=cfg,
        examples=args.profile_validation_examples,
        batch_size=args.batch_size,
        seed=args.profile_validation_seed,
        device=device,
    )
    calibration_rows = [
        {"split": "fit", **row} for row in calibration_rows
    ] + evaluate_healthy_profiles(
        healthy_states=validation_states,
        profiles=profiles,
        split="held_out_graphs",
        device=device,
    )
    del validation_states
    probe = np.load(args.probe_artifact)
    age_weight = torch.as_tensor(
        probe["post_J_full_age_weight"], dtype=torch.float32, device=device
    )
    age_bias = torch.as_tensor(
        probe["post_J_full_age_bias"], dtype=torch.float32, device=device
    )
    conditions = build_conditions(
        strengths=tuple(args.strengths), random_draws=args.random_draws
    )
    if args.conditions:
        requested = {"baseline", *args.conditions}
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
    behavior_rows, step_rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        profiles=profiles,
        conditions=conditions,
        age_weight=age_weight,
        age_bias=age_bias,
        graph_seeds=tuple(args.graph_seeds),
        examples=args.examples,
        batch_size=args.batch_size,
        words=words,
        device=device,
    )
    behavior_summary = _aggregate(behavior_rows, ("back_count", "condition"))
    step_summary = _aggregate(
        step_rows, ("back_count", "condition", "source_age")
    )
    write_csv(args.out_dir / "healthy_profile_fit.csv", calibration_rows)
    write_csv(args.out_dir / "behavior_rows.csv", behavior_rows)
    write_csv(args.out_dir / "behavior_summary.csv", behavior_summary)
    write_csv(args.out_dir / "step_rows.csv", step_rows)
    write_csv(args.out_dir / "step_summary.csv", step_summary)
    plot_results(behavior_summary, args.out_dir / "centered_residual_intervention.png")
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "probe_artifact": str(args.probe_artifact),
        "rank": args.rank,
        "ridge": args.ridge,
        "calibration_examples": args.calibration_examples,
        "calibration_seed": args.calibration_seed,
        "profile_validation_examples": args.profile_validation_examples,
        "profile_validation_seed": args.profile_validation_seed,
        "evaluation_graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "back_counts": list(args.back_counts),
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
        "profile_definition": (
            "stage-specific graph-blind post-J coordinate predictor from the same "
            "token's orthogonal-complement residual plus token-position bias"
        ),
        "state_effect_control": (
            "random/top deltas are rescaled per example to match the bottom-space "
            "ambient hidden perturbation norm"
        ),
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "claim_boundary": (
            "Selective rescue by bottom-residual shrinkage supports a natural limiting "
            "role only if it beats state-effect-matched random/top controls. Expansion "
            "is a sufficiency-style stress test, not proof of a unique telomere."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
