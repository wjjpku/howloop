"""Trace and causally localize how J's common compressed directions affect F.

The experiment compares the trained J bank with singular-value anti-compression
and matched random operator controls.  All conditions use the same random graph,
start node, and F/J action word.  The clean J trajectory is the donor for causal
patching, so donor and receiver always have the same task content and differ only
in controller state.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.analyze_graph_path_j_anti_compression import (
    build_stage_delta,
)
from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    AGES,
    atomic_json,
    load_bank_and_bases,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalTrace,
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
    sample_equivalent_word_pair,
)


@dataclass(frozen=True)
class PatchCase:
    label: str
    interventions: tuple[FunctionalIntervention, ...]


@dataclass
class ForwardRecord:
    trace: FunctionalTrace
    state: torch.Tensor
    logical_age_before: int
    follows_j: bool
    rollback_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=(846001, 846002))
    parser.add_argument("--back-counts", type=int, nargs="+", default=(8, 12))
    parser.add_argument("--path-pairs-per-k", type=int, default=1)
    parser.add_argument("--word-seed", type=int, default=846701)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--singular-floor", type=float, default=0.5)
    parser.add_argument("--random-draws", type=int, default=2)
    parser.add_argument(
        "--patch-scope", choices=("post_j", "all_f"), default="post_j"
    )
    parser.add_argument(
        "--patch-directions",
        choices=("rescue", "disrupt"),
        nargs="+",
        default=("rescue",),
    )
    parser.add_argument(
        "--patch-labels",
        nargs="*",
        help="Optional exact PatchCase labels; default evaluates every case.",
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


def final_state_from_trace(model, trace: FunctionalTrace) -> torch.Tensor:
    if not trace.sites:
        raise ValueError("trace has no sites")
    state = trace.sites[-1].hidden_out
    return model.outer_norm(state) if model.outer_norm is not None else state


def relative_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (value.float() - reference.float()).square().mean().sqrt()
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-8)
    return float(numerator / denominator)


def mean_cosine(value: torch.Tensor, reference: torch.Tensor) -> float:
    left = value.float().reshape(value.shape[0], -1)
    right = reference.float().reshape(reference.shape[0], -1)
    return float(F.cosine_similarity(left, right, dim=-1, eps=1e-8).mean())


def attention_js(value: torch.Tensor, reference: torch.Tensor) -> float:
    left = value.float().clamp_min(1e-12)
    right = reference.float().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    divergence = 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )
    return float(divergence.mean())


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    mask = F.one_hot(target, num_classes=logits.shape[-1]).bool()
    wrong = logits.masked_fill(mask, float("-inf")).max(dim=-1).values
    return correct - wrong


def build_patch_cases(cfg) -> tuple[PatchCase, ...]:
    groups = explicit_depth_position_groups(cfg.node_count)
    graph = groups["graph"]
    answer = groups["answer"]
    all_positions = tuple(range(cfg.seq_len))
    cases: list[PatchCase] = []
    for site, block_label, query_positions in (
        (0, "B1", graph),
        (1, "B2", answer),
    ):
        for head in range(cfg.n_heads):
            common = dict(site=site, mode="patch", heads=(head,))
            component_interventions = {
                "q": FunctionalIntervention(
                    component="q", positions=query_positions, **common
                ),
                "k": FunctionalIntervention(
                    component="k", positions=all_positions, **common
                ),
                "v": FunctionalIntervention(
                    component="v", positions=all_positions, **common
                ),
            }
            for component in ("q", "k", "v"):
                cases.append(
                    PatchCase(
                        f"{block_label}.H{head}.{component}",
                        (component_interventions[component],),
                    )
                )
            cases.append(
                PatchCase(
                    f"{block_label}.H{head}.qk",
                    (
                        component_interventions["q"],
                        component_interventions["k"],
                    ),
                )
            )
            cases.append(
                PatchCase(
                    f"{block_label}.H{head}.qkv",
                    (
                        component_interventions["q"],
                        component_interventions["k"],
                        component_interventions["v"],
                    ),
                )
            )
            cases.append(
                PatchCase(
                    f"{block_label}.H{head}.pattern",
                    (
                        FunctionalIntervention(
                            component="attention_pattern",
                            positions=query_positions,
                            **common,
                        ),
                    ),
                )
            )
            cases.append(
                PatchCase(
                    f"{block_label}.H{head}.context",
                    (
                        FunctionalIntervention(
                            component="head_context",
                            positions=query_positions,
                            **common,
                        ),
                    ),
                )
            )
        for component in ("attention_out", "mlp_out"):
            cases.append(
                PatchCase(
                    f"{block_label}.{component}",
                    (
                        FunctionalIntervention(
                            site=site,
                            component=component,  # type: ignore[arg-type]
                            mode="patch",
                            positions=query_positions,
                        ),
                    ),
                )
            )
    return tuple(cases)


def build_words(
    *, back_counts: Sequence[int], path_pairs_per_k: int, seed: int
) -> list[tuple[int, str, tuple[int, ...]]]:
    if path_pairs_per_k < 1:
        raise ValueError("path_pairs_per_k must be positive")
    rng = np.random.default_rng(seed)
    words: list[tuple[int, str, tuple[int, ...]]] = []
    for back_count in back_counts:
        for pair_index in range(path_pairs_per_k):
            mandatory = AGES[pair_index % len(AGES)]
            left, right = sample_equivalent_word_pair(
                rng=rng,
                back_count=int(back_count),
                mandatory_source_age=mandatory,
            )
            words.extend(
                (
                    (int(back_count), f"k{back_count}_p{pair_index}_left", left),
                    (int(back_count), f"k{back_count}_p{pair_index}_right", right),
                )
            )
    return words


def make_operator_deltas(
    weights: dict[int, np.ndarray],
    *,
    rank: int,
    target: float,
    random_draws: int,
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    result: dict[str, dict[int, torch.Tensor]] = {"anti": {}}
    for age in AGES:
        delta, _ = build_stage_delta(
            weights[age], rank=rank, mode="bottom_floor", target=target
        )
        result["anti"][age] = torch.as_tensor(
            delta, dtype=torch.float32, device=device
        )
    for draw in range(random_draws):
        name = f"random_parameter_matched_d{draw}"
        result[name] = {}
        for age in AGES:
            delta, _ = build_stage_delta(
                weights[age],
                rank=rank,
                mode="random_matched",
                target=target,
                rng=np.random.default_rng(847000 + 1000 * draw + 31 * age + rank),
            )
            result[name][age] = torch.as_tensor(
                delta, dtype=torch.float32, device=device
            )
    if random_draws:
        # This control is rescaled at application time to match ||h Delta W||.
        result["random_state_effect_matched"] = result[
            "random_parameter_matched_d0"
        ]
    return result


def apply_rollback(
    *,
    bank: AgeSpecificJBank,
    state: torch.Tensor,
    source_age: int,
    positions: tuple[int, ...],
    deltas: dict[int, torch.Tensor] | None,
    reference_deltas: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, float]:
    result = bank.rollback(state, source_age=source_age, positions=positions)
    if not deltas:
        return result, 1.0
    index = list(positions)
    effect = state[:, index].float() @ deltas[source_age]
    scale = 1.0
    if reference_deltas is not None:
        reference = state[:, index].float() @ reference_deltas[source_age]
        scale = float(
            reference.square().sum().sqrt()
            / effect.square().sum().sqrt().clamp_min(1e-12)
        )
        effect = effect * scale
    result[:, index] = (result[:, index].float() + effect).to(result.dtype)
    return result, scale


@torch.no_grad()
def clean_rollout(
    *,
    model,
    bank: AgeSpecificJBank,
    h1: torch.Tensor,
    actions: Sequence[int],
    positions: tuple[int, ...],
) -> tuple[list[ForwardRecord], torch.Tensor]:
    state = h1
    age = MIN_AGE
    rollback_count = 0
    follows_j = False
    records: list[ForwardRecord] = []
    for action in actions:
        if action == 1:
            _, trace = run_instrumented_state(
                model, state, loop_indices=(age,)
            )
            state = final_state_from_trace(model, trace)
            records.append(
                ForwardRecord(
                    trace=trace,
                    state=state,
                    logical_age_before=age,
                    follows_j=follows_j,
                    rollback_count=rollback_count,
                )
            )
            age += 1
            follows_j = False
        elif action == -1:
            state, _ = apply_rollback(
                bank=bank,
                state=state,
                source_age=age,
                positions=positions,
                deltas=None,
            )
            age -= 1
            rollback_count += 1
            follows_j = True
        else:
            raise ValueError("actions must be +1 or -1")
    if action_semantics(actions).end_age != age or age != MAX_AGE:
        raise RuntimeError("clean rollout ended at the wrong logical age")
    return records, state


@torch.no_grad()
def receiver_rollout(
    *,
    model,
    bank: AgeSpecificJBank,
    h1: torch.Tensor,
    actions: Sequence[int],
    positions: tuple[int, ...],
    deltas: dict[int, torch.Tensor] | None,
    donor_records: Sequence[ForwardRecord] | None = None,
    patch_case: PatchCase | None = None,
    input_subspace_basis: torch.Tensor | None = None,
    patch_scope: str = "post_j",
    reference_deltas: dict[int, torch.Tensor] | None = None,
    collect_records: bool = False,
) -> tuple[torch.Tensor, list[ForwardRecord], list[float]]:
    state = h1
    age = MIN_AGE
    rollback_count = 0
    follows_j = False
    forward_index = 0
    records: list[ForwardRecord] = []
    state_effect_scales: list[float] = []
    for action in actions:
        if action == 1:
            should_patch = (
                (patch_case is not None or input_subspace_basis is not None)
                and (patch_scope == "all_f" or follows_j)
            )
            donor = None
            interventions: Sequence[FunctionalIntervention] = ()
            if should_patch:
                if donor_records is None:
                    raise ValueError("patching requires donor records")
                donor = donor_records[forward_index].trace
                if input_subspace_basis is not None:
                    donor_state = donor.sites[0].hidden_in
                    difference = donor_state.float() - state.float()
                    state = (
                        state.float()
                        + (difference @ input_subspace_basis)
                        @ input_subspace_basis.transpose(0, 1)
                    ).to(state.dtype)
                if patch_case is not None:
                    interventions = patch_case.interventions
            _, trace = run_instrumented_state(
                model,
                state,
                loop_indices=(age,),
                interventions=interventions,
                donor_trace=donor,
            )
            state = final_state_from_trace(model, trace)
            if collect_records:
                records.append(
                    ForwardRecord(
                        trace=trace,
                        state=state,
                        logical_age_before=age,
                        follows_j=follows_j,
                        rollback_count=rollback_count,
                    )
                )
            age += 1
            follows_j = False
            forward_index += 1
        elif action == -1:
            state, scale = apply_rollback(
                bank=bank,
                state=state,
                source_age=age,
                positions=positions,
                deltas=deltas,
                reference_deltas=reference_deltas,
            )
            state_effect_scales.append(scale)
            age -= 1
            rollback_count += 1
            follows_j = True
        else:
            raise ValueError("actions must be +1 or -1")
    if age != MAX_AGE:
        raise RuntimeError("receiver rollout ended at the wrong logical age")
    return state, records, state_effect_scales


def _select(value: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    return value[:, :, list(positions)]


def activation_rows_for_pair(
    *,
    clean_records: Sequence[ForwardRecord],
    receiver_records: Sequence[ForwardRecord],
    condition: str,
    cfg,
    input_current_by_forward: Sequence[torch.Tensor],
    output_current_by_forward: Sequence[torch.Tensor],
    graph_seed: int,
    word_label: str,
    back_count: int,
    batch_index: int,
) -> list[dict[str, Any]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    rows: list[dict[str, Any]] = []
    if len(clean_records) != len(receiver_records):
        raise ValueError("clean and receiver traces do not align")
    for forward_index, (clean, receiver) in enumerate(
        zip(clean_records, receiver_records, strict=True)
    ):
        if not receiver.follows_j:
            continue
        input_current = input_current_by_forward[forward_index]
        output_current = output_current_by_forward[forward_index]
        for site in range(2):
            clean_site = clean.trace.sites[site]
            receiver_site = receiver.trace.sites[site]
            query_positions = groups["graph"] if site == 0 else groups["answer"]
            key_positions = tuple(range(cfg.seq_len))
            for head in range(cfg.n_heads):
                q_clean = _select(clean_site.q[:, head : head + 1], query_positions)
                q_recv = _select(receiver_site.q[:, head : head + 1], query_positions)
                k_clean = _select(clean_site.k[:, head : head + 1], key_positions)
                k_recv = _select(receiver_site.k[:, head : head + 1], key_positions)
                v_clean = _select(clean_site.v[:, head : head + 1], key_positions)
                v_recv = _select(receiver_site.v[:, head : head + 1], key_positions)
                p_clean = _select(
                    clean_site.attention_pattern[:, head : head + 1],
                    query_positions,
                )
                p_recv = _select(
                    receiver_site.attention_pattern[:, head : head + 1],
                    query_positions,
                )
                c_clean = _select(
                    clean_site.head_context[:, head : head + 1], query_positions
                )
                c_recv = _select(
                    receiver_site.head_context[:, head : head + 1], query_positions
                )
                row: dict[str, Any] = {
                    "graph_seed": graph_seed,
                    "batch": batch_index,
                    "word": word_label,
                    "back_count": back_count,
                    "condition": condition,
                    "rollback_index": receiver.rollback_count,
                    "logical_age_before_F": receiver.logical_age_before,
                    "block": site + 1,
                    "head": head,
                    "q_relative_rms": relative_rms(q_recv, q_clean),
                    "q_cosine": mean_cosine(q_recv, q_clean),
                    "k_relative_rms": relative_rms(k_recv, k_clean),
                    "k_cosine": mean_cosine(k_recv, k_clean),
                    "v_relative_rms": relative_rms(v_recv, v_clean),
                    "v_cosine": mean_cosine(v_recv, v_clean),
                    "pattern_js": attention_js(p_recv, p_clean),
                    "context_relative_rms": relative_rms(c_recv, c_clean),
                    "context_cosine": mean_cosine(c_recv, c_clean),
                    "attention_out_relative_rms": relative_rms(
                        receiver_site.attention_out[:, list(query_positions)],
                        clean_site.attention_out[:, list(query_positions)],
                    ),
                    "mlp_hidden_relative_rms": relative_rms(
                        receiver_site.mlp_hidden[:, list(query_positions)],
                        clean_site.mlp_hidden[:, list(query_positions)],
                    ),
                    "mlp_hidden_sign_flip": float(
                        (
                            receiver_site.mlp_hidden[:, list(query_positions)].gt(0)
                            != clean_site.mlp_hidden[:, list(query_positions)].gt(0)
                        )
                        .float()
                        .mean()
                    ),
                    "mlp_out_relative_rms": relative_rms(
                        receiver_site.mlp_out[:, list(query_positions)],
                        clean_site.mlp_out[:, list(query_positions)],
                    ),
                    "hidden_out_relative_rms": relative_rms(
                        receiver_site.hidden_out[:, list(query_positions)],
                        clean_site.hidden_out[:, list(query_positions)],
                    ),
                }
                if site == 1:
                    answer = groups["answer"][0]
                    source_positions = torch.as_tensor(
                        groups["source"], device=input_current.device
                    )
                    destination_positions = torch.as_tensor(
                        groups["destination"], device=input_current.device
                    )
                    lookup_source = source_positions[input_current]
                    lookup_destination = destination_positions[input_current]
                    next_source = source_positions[output_current]
                    next_destination = destination_positions[output_current]
                    batch = torch.arange(
                        input_current.shape[0], device=input_current.device
                    )
                    clean_pattern = clean_site.attention_pattern[:, head, answer]
                    recv_pattern = receiver_site.attention_pattern[:, head, answer]
                    def mass(pattern: torch.Tensor, position: torch.Tensor) -> float:
                        return float(pattern[batch, position].mean())

                    row.update(
                        {
                            "lookup_source_mass_clean": mass(
                                clean_pattern, lookup_source
                            ),
                            "lookup_source_mass_receiver": mass(
                                recv_pattern, lookup_source
                            ),
                            "lookup_source_mass_delta": mass(
                                recv_pattern - clean_pattern, lookup_source
                            ),
                            "lookup_destination_mass_clean": mass(
                                clean_pattern, lookup_destination
                            ),
                            "lookup_destination_mass_receiver": mass(
                                recv_pattern, lookup_destination
                            ),
                            "lookup_destination_mass_delta": mass(
                                recv_pattern - clean_pattern, lookup_destination
                            ),
                            "next_source_mass_clean": mass(
                                clean_pattern, next_source
                            ),
                            "next_source_mass_receiver": mass(
                                recv_pattern, next_source
                            ),
                            "next_destination_mass_clean": mass(
                                clean_pattern, next_destination
                            ),
                            "next_destination_mass_receiver": mass(
                                recv_pattern, next_destination
                            ),
                            "lookup_destination_argmax_clean": float(
                                clean_pattern.argmax(-1)
                                .eq(lookup_destination)
                                .float()
                                .mean()
                            ),
                            "lookup_destination_argmax_receiver": float(
                                recv_pattern.argmax(-1)
                                .eq(lookup_destination)
                                .float()
                                .mean()
                            ),
                        }
                    )
                    role_groups = {
                        "bos": (0,),
                        "edge_marker": groups["edge_marker"],
                        "source": groups["source"],
                        "destination": groups["destination"],
                        "query_metadata": groups["query_metadata"],
                        "answer": groups["answer"],
                    }
                    for role, role_positions in role_groups.items():
                        clean_mass = clean_pattern[:, list(role_positions)].sum(-1)
                        recv_mass = recv_pattern[:, list(role_positions)].sum(-1)
                        row[f"role_{role}_mass_clean"] = float(clean_mass.mean())
                        row[f"role_{role}_mass_receiver"] = float(recv_mass.mean())
                        row[f"role_{role}_mass_delta"] = float(
                            (recv_mass - clean_mass).mean()
                        )
                rows.append(row)
    return rows


def aggregate_numeric(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, group, strict=True))
        numeric_keys = sorted(
            {
                key
                for row in parts
                for key, value in row.items()
                if key not in keys and isinstance(value, (int, float))
            }
        )
        for key in numeric_keys:
            values = [float(row[key]) for row in parts if key in row]
            result[key] = float(np.mean(values))
        result["rows"] = len(parts)
        output.append(result)
    return output


def currents_by_forward(
    *, successors: torch.Tensor, h1_current: torch.Tensor, actions: Sequence[int]
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    current = h1_current
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for action in actions:
        if action == 1:
            inputs.append(current)
            current = advance_nodes(successors, current, steps=1)
            outputs.append(current)
    return inputs, outputs, current


def behavior_row(
    *,
    state: torch.Tensor,
    model,
    target: torch.Tensor,
    graph_seed: int,
    batch_index: int,
    word: str,
    back_count: int,
    condition: str,
    patch_label: str,
    direction: str,
) -> dict[str, Any]:
    logits = logits_from_raw_state(model, state).float()
    margin = target_margin(logits, target)
    return {
        "graph_seed": graph_seed,
        "batch": batch_index,
        "word": word,
        "back_count": back_count,
        "condition": condition,
        "patch_label": patch_label,
        "direction": direction,
        "accuracy": float(logits.argmax(-1).eq(target).float().mean()),
        "margin": float(margin.mean()),
        "ce": float(F.cross_entropy(logits, target)),
        "answer_rms": float(state[:, -1].float().square().mean(-1).sqrt().mean()),
        "examples": int(target.numel()),
    }


def plot_activation(rows: Sequence[dict[str, Any]], path: Path) -> None:
    anti = [row for row in rows if row["condition"] == "anti"]
    if not anti:
        return
    metrics = ("pattern_js", "context_relative_rms", "mlp_out_relative_rms")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
    labels = [f"B{row['block']}H{row['head']}" for row in anti]
    for axis, metric in zip(axes, metrics, strict=True):
        values = [float(row[metric]) for row in anti]
        order = np.argsort(values)[::-1]
        axis.bar(np.arange(len(order)), np.asarray(values)[order])
        axis.set_xticks(np.arange(len(order)), np.asarray(labels)[order], rotation=45)
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Anti-compression drift in the F immediately following J")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_patches(rows: Sequence[dict[str, Any]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["patch_label"] not in {"none"}
        and row["direction"] in {"rescue", "disrupt"}
    ]
    if not selected:
        return
    selected = sorted(selected, key=lambda row: float(row["accuracy"]), reverse=True)
    figure, axis = plt.subplots(figsize=(12, max(5, len(selected) * 0.24)), dpi=180)
    labels = [f"{row['direction']}:{row['patch_label']}" for row in selected]
    axis.barh(np.arange(len(selected)), [float(row["accuracy"]) for row in selected])
    axis.set_yticks(np.arange(len(selected)), labels)
    axis.invert_yaxis()
    axis.set(xlabel="final accuracy", xlim=(0, 1.02))
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    operator_deltas: dict[str, dict[int, torch.Tensor]],
    words: Sequence[tuple[int, str, tuple[int, ...]]],
    graph_seeds: Sequence[int],
    examples: int,
    batch_size: int,
    patch_cases: Sequence[PatchCase],
    patch_scope: str,
    patch_directions: Sequence[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch_size")
    positions = tuple(range(cfg.seq_len))
    activation_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        for batch_index in range(examples // batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            h1 = model.apply_loop(raw, loop_index=0)
            h1_current = advance_nodes(successors, start, steps=1)
            for back_count, word_label, actions in words:
                clean_records, clean_state = clean_rollout(
                    model=model,
                    bank=bank,
                    h1=h1,
                    actions=actions,
                    positions=positions,
                )
                (
                    input_current_by_forward,
                    output_current_by_forward,
                    final_target,
                ) = currents_by_forward(
                    successors=successors,
                    h1_current=h1_current,
                    actions=actions,
                )
                patch_rows.append(
                    behavior_row(
                        state=clean_state,
                        model=model,
                        target=final_target,
                        graph_seed=int(graph_seed),
                        batch_index=batch_index,
                        word=word_label,
                        back_count=back_count,
                        condition="baseline",
                        patch_label="none",
                        direction="none",
                    )
                )
                unpatched: dict[str, tuple[torch.Tensor, list[ForwardRecord]]] = {}
                for condition, deltas in operator_deltas.items():
                    reference = (
                        operator_deltas["anti"]
                        if condition == "random_state_effect_matched"
                        else None
                    )
                    state, records, scales = receiver_rollout(
                        model=model,
                        bank=bank,
                        h1=h1,
                        actions=actions,
                        positions=positions,
                        deltas=deltas,
                        reference_deltas=reference,
                        collect_records=True,
                    )
                    unpatched[condition] = (state, records)
                    patch_rows.append(
                        behavior_row(
                            state=state,
                            model=model,
                            target=final_target,
                            graph_seed=int(graph_seed),
                            batch_index=batch_index,
                            word=word_label,
                            back_count=back_count,
                            condition=condition,
                            patch_label="none",
                            direction="none",
                        )
                    )
                    activation_rows.extend(
                        activation_rows_for_pair(
                            clean_records=clean_records,
                            receiver_records=records,
                            condition=condition,
                            cfg=cfg,
                            input_current_by_forward=input_current_by_forward,
                            output_current_by_forward=output_current_by_forward,
                            graph_seed=int(graph_seed),
                            word_label=word_label,
                            back_count=back_count,
                            batch_index=batch_index,
                        )
                    )
                    if scales:
                        scale_rows.append(
                            {
                                "graph_seed": int(graph_seed),
                                "batch": batch_index,
                                "word": word_label,
                                "condition": condition,
                                "mean_state_effect_scale": float(np.mean(scales)),
                                "min_state_effect_scale": float(np.min(scales)),
                                "max_state_effect_scale": float(np.max(scales)),
                            }
                        )
                for patch_case in patch_cases:
                    if "rescue" in patch_directions:
                        state, _, _ = receiver_rollout(
                            model=model,
                            bank=bank,
                            h1=h1,
                            actions=actions,
                            positions=positions,
                            deltas=operator_deltas["anti"],
                            donor_records=clean_records,
                            patch_case=patch_case,
                            patch_scope=patch_scope,
                        )
                        patch_rows.append(
                            behavior_row(
                                state=state,
                                model=model,
                                target=final_target,
                                graph_seed=int(graph_seed),
                                batch_index=batch_index,
                                word=word_label,
                                back_count=back_count,
                                condition="anti",
                                patch_label=patch_case.label,
                                direction="rescue",
                            )
                        )
                    if "disrupt" in patch_directions:
                        anti_records = unpatched["anti"][1]
                        state, _, _ = receiver_rollout(
                            model=model,
                            bank=bank,
                            h1=h1,
                            actions=actions,
                            positions=positions,
                            deltas=None,
                            donor_records=anti_records,
                            patch_case=patch_case,
                            patch_scope=patch_scope,
                        )
                        patch_rows.append(
                            behavior_row(
                                state=state,
                                model=model,
                                target=final_target,
                                graph_seed=int(graph_seed),
                                batch_index=batch_index,
                                word=word_label,
                                back_count=back_count,
                                condition="baseline",
                                patch_label=patch_case.label,
                                direction="disrupt",
                            )
                        )
    return activation_rows, patch_rows, scale_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.rank < 1:
        raise ValueError("rank must be positive")
    if args.singular_floor <= 0:
        raise ValueError("singular_floor must be positive")
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
        fraction = float(os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.20"))
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("experiment is fixed to D8L8 with two shared blocks")
    bank, weights, _, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (args.rank,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    operator_deltas = make_operator_deltas(
        weights,
        rank=args.rank,
        target=args.singular_floor,
        random_draws=args.random_draws,
        device=device,
    )
    words = build_words(
        back_counts=args.back_counts,
        path_pairs_per_k=args.path_pairs_per_k,
        seed=args.word_seed,
    )
    patch_cases = list(build_patch_cases(cfg))
    if args.patch_labels:
        requested = set(args.patch_labels)
        available = {case.label for case in patch_cases}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown patch labels: {missing}")
        patch_cases = [case for case in patch_cases if case.label in requested]
    activation_rows, patch_rows, scale_rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        operator_deltas=operator_deltas,
        words=words,
        graph_seeds=args.graph_seeds,
        examples=args.examples,
        batch_size=args.batch_size,
        patch_cases=patch_cases,
        patch_scope=args.patch_scope,
        patch_directions=args.patch_directions,
        device=device,
    )
    activation_summary = aggregate_numeric(
        activation_rows, ("condition", "block", "head")
    )
    patch_summary = aggregate_numeric(
        patch_rows,
        ("back_count", "condition", "patch_label", "direction"),
    )
    write_csv(args.out_dir / "activation_rows.csv", activation_rows)
    write_csv(args.out_dir / "activation_summary.csv", activation_summary)
    write_csv(args.out_dir / "patch_rows.csv", patch_rows)
    write_csv(args.out_dir / "patch_summary.csv", patch_summary)
    write_csv(args.out_dir / "state_effect_scales.csv", scale_rows)
    plot_activation(activation_summary, args.out_dir / "attention_component_drift.png")
    plot_patches(
        [row for row in patch_summary if int(row["back_count"]) == max(args.back_counts)],
        args.out_dir / "causal_patch_accuracy.png",
    )
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "rank": args.rank,
        "singular_floor": args.singular_floor,
        "graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "words": [
            {
                "back_count": back_count,
                "label": label,
                "actions": list(actions),
                "semantics": action_semantics(actions).__dict__,
            }
            for back_count, label, actions in words
        ],
        "patch_scope": args.patch_scope,
        "patch_directions": list(args.patch_directions),
        "patch_cases": [case.label for case in patch_cases],
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "claim_boundary": (
            "Activation drift is descriptive. A component is called a mediator only when "
            "clean same-graph/same-current patching rescues anti-compressed trajectories "
            "and the reverse patch disrupts clean trajectories on held-out graphs."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
