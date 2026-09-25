"""Measure state-conditioned head/MLP causal-effect drift between J banks.

The backbone is frozen.  Therefore any change in effective component ablation
effects is caused by the intervention trajectory induced by the J bank, not by
changed Transformer weights.  This is a causal role comparison, not a claim of
a complete or unique circuit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import (
    action_semantics,
    sample_equivalent_word_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=829501)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_bank(path: Path, device: torch.device, dimension: int) -> AgeSpecificJBank:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=dimension,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=str(payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    return bank.frozen()


def _schedule_specs() -> dict[str, tuple[int, ...]]:
    rng = np.random.default_rng(829617)
    k8_left, k8_right = sample_equivalent_word_pair(
        rng=rng, back_count=8, mandatory_source_age=8
    )
    k12_left, k12_right = sample_equivalent_word_pair(
        rng=rng, back_count=12, mandatory_source_age=8
    )
    return {
        "natural_F7": (1,) * 7,
        "k8_left": k8_left,
        "k8_right": k8_right,
        "k12_left": k12_left,
        "k12_right": k12_right,
    }


def _components(cfg) -> list[tuple[str, FunctionalIntervention]]:
    answer = explicit_depth_position_groups(cfg.node_count)["answer"]
    result: list[tuple[str, FunctionalIntervention]] = []
    for block_index in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            result.append(
                (
                    f"B{block_index + 1}.H{head}.context_answer",
                    FunctionalIntervention(
                        site=block_index,
                        component="head_context",
                        mode="zero",
                        positions=answer,
                        heads=(head,),
                    ),
                )
            )
        result.append(
            (
                f"B{block_index + 1}.MLP.out_answer",
                FunctionalIntervention(
                    site=block_index,
                    component="mlp_out",
                    mode="zero",
                    positions=answer,
                ),
            )
        )
    return result


@torch.no_grad()
def _execute(
    *,
    model,
    bank: AgeSpecificJBank,
    state: torch.Tensor,
    current: torch.Tensor,
    successors: torch.Tensor,
    actions: Sequence[int],
    positions: tuple[int, ...],
    ablate_f_event: int | None = None,
    intervention: FunctionalIntervention | None = None,
    attention_rows: list[dict[str, Any]] | None = None,
    attention_context: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_rows is not None and attention_context is None:
        raise ValueError("attention capture requires identifying context")
    if attention_rows is not None and ablate_f_event is not None:
        raise ValueError("attention capture is only defined for clean execution")
    logical_age = 1
    f_event = 0
    groups = explicit_depth_position_groups(model.cfg.node_count)
    attention_sources = (
        "edge_marker",
        "source",
        "destination",
        "query",
        "start",
        "depth",
        "answer",
    )
    answer = groups["answer"]
    for action_index, action in enumerate(actions, start=1):
        if action == 1:
            f_event += 1
            if attention_rows is not None:
                _, trace = run_instrumented_state(
                    model,
                    state,
                    loop_indices=(logical_age,),
                )
                state = trace.sites[-1].hidden_out
                if model.outer_norm is not None:
                    state = model.outer_norm(state)
                for site in trace.sites:
                    for head in range(model.cfg.n_heads):
                        row = {
                            **attention_context,
                            "f_event": f_event,
                            "action_index": action_index,
                            "logical_age_before_F": logical_age,
                            "block": site.block_index + 1,
                            "head": head,
                            "answer_context_rms": float(
                                site.head_context[:, head, answer]
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            ),
                        }
                        for source_name in attention_sources:
                            source_index = torch.as_tensor(
                                groups[source_name], device=state.device
                            )
                            row[f"attention_to_{source_name}"] = float(
                                site.attention_pattern[
                                    :, head, answer[0], source_index
                                ]
                                .float()
                                .sum(dim=-1)
                                .mean()
                            )
                        attention_rows.append(row)
            elif ablate_f_event == f_event:
                if intervention is None:
                    raise ValueError("targeted F event requires an intervention")
                _, trace = run_instrumented_state(
                    model,
                    state,
                    loop_indices=(logical_age,),
                    interventions=(intervention,),
                )
                # run_instrumented_state returns readout logits and a block trace,
                # whereas the F/J executor must continue from the post-block hidden.
                state = trace.sites[-1].hidden_out
                if model.outer_norm is not None:
                    state = model.outer_norm(state)
            else:
                state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
        else:
            state = bank.rollback(
                state, source_age=logical_age, positions=positions
            )
            logical_age -= 1
    if logical_age != 8:
        raise RuntimeError("circuit schedule must end at H8")
    return logits_from_raw_state(model, state), current


def _aggregate_attention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "bank",
        "schedule",
        "f_event",
        "action_index",
        "logical_age_before_F",
        "block",
        "head",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    metrics = (
        "answer_context_rms",
        "attention_to_edge_marker",
        "attention_to_source",
        "attention_to_destination",
        "attention_to_query",
        "attention_to_start",
        "attention_to_depth",
        "attention_to_answer",
    )
    output: list[dict[str, Any]] = []
    for key, values in grouped.items():
        item = dict(zip(keys, key, strict=True))
        item["observations"] = len(values)
        for metric in metrics:
            item[metric] = float(np.mean([float(value[metric]) for value in values]))
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            row["schedule"],
            row["bank"],
            row["f_event"],
            row["block"],
            row["head"],
        ),
    )


def _compare_attention_roles(
    summary: list[dict[str, Any]], labels: list[str]
) -> list[dict[str, Any]]:
    if len(labels) < 2:
        return []
    reference = labels[0]
    keys = ("schedule", "f_event", "block", "head")
    lookup = {
        (row["bank"], *(row[key] for key in keys)): row for row in summary
    }
    metrics = (
        "answer_context_rms",
        "attention_to_edge_marker",
        "attention_to_source",
        "attention_to_destination",
        "attention_to_query",
        "attention_to_start",
        "attention_to_depth",
        "attention_to_answer",
    )
    output: list[dict[str, Any]] = []
    for candidate in labels[1:]:
        for row in summary:
            if row["bank"] != reference:
                continue
            candidate_row = lookup[
                (candidate, *(row[key] for key in keys))
            ]
            item = {
                "reference_bank": reference,
                "candidate_bank": candidate,
                **{key: row[key] for key in keys},
                "logical_age_before_F": row["logical_age_before_F"],
            }
            for metric in metrics:
                item[f"reference_{metric}"] = row[metric]
                item[f"candidate_{metric}"] = candidate_row[metric]
                item[f"delta_{metric}"] = candidate_row[metric] - row[metric]
            output.append(item)
    return output


def _f_event_phases(actions: Sequence[int]) -> dict[int, tuple[int, int]]:
    logical_age = 1
    f_event = 0
    result = {}
    for action_index, action in enumerate(actions, start=1):
        if action == 1:
            f_event += 1
            result[f_event] = (action_index, logical_age)
            logical_age += 1
        else:
            logical_age -= 1
    return result


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "bank",
        "schedule",
        "f_event",
        "action_index",
        "logical_age_before_F",
        "component",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    metrics = (
        "clean_accuracy",
        "ablated_accuracy",
        "accuracy_drop",
        "clean_margin",
        "ablated_margin",
        "margin_drop",
        "shared_correct_margin_drop",
        "shared_correct_count",
    )
    output: list[dict[str, Any]] = []
    for key, values in grouped.items():
        item = dict(zip(keys, key, strict=True))
        item["observations"] = len(values)
        for metric in metrics:
            array = np.asarray([float(value[metric]) for value in values])
            item[metric] = float(
                np.nanmean(array)
                if metric == "shared_correct_margin_drop"
                else np.mean(array)
            )
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            row["schedule"], row["bank"], row["f_event"], row["component"]
        ),
    )


def _plot(summary: list[dict[str, Any]], out_dir: Path) -> None:
    for schedule in sorted({row["schedule"] for row in summary}):
        rows = [row for row in summary if row["schedule"] == schedule]
        components = sorted({row["component"] for row in rows})
        banks = sorted({row["bank"] for row in rows})
        figure, axes = plt.subplots(
            len(banks), 1, figsize=(15, 3.6 * len(banks)), dpi=180, squeeze=False
        )
        for axis, bank in zip(axes[:, 0], banks, strict=True):
            selected = [row for row in rows if row["bank"] == bank]
            events = sorted({int(row["f_event"]) for row in selected})
            matrix = np.asarray(
                [
                    [
                        next(
                            row["margin_drop"]
                            for row in selected
                            if row["component"] == component
                            and int(row["f_event"]) == event
                        )
                        for event in events
                    ]
                    for component in components
                ]
            )
            image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
            axis.set_yticks(range(len(components)), components, fontsize=7)
            axis.set_xticks(range(len(events)), events)
            axis.set_xlabel("effective F event")
            axis.set_title(f"{bank}: target-margin drop under zero ablation")
            figure.colorbar(image, ax=axis, fraction=0.02)
        figure.suptitle(schedule)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(out_dir / f"circuit_drift_{schedule}.png", bbox_inches="tight")
        plt.close(figure)


def _compare_banks(
    summary: list[dict[str, Any]], labels: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(labels) < 2:
        return [], []
    reference = labels[0]
    lookup = {
        (row["bank"], row["schedule"], int(row["f_event"]), row["component"]): row
        for row in summary
    }
    comparison: list[dict[str, Any]] = []
    schedule_summary: list[dict[str, Any]] = []
    for candidate in labels[1:]:
        for schedule in sorted({row["schedule"] for row in summary}):
            values = []
            for (bank, current_schedule, event, component), before in lookup.items():
                if bank != reference or current_schedule != schedule:
                    continue
                after = lookup[(candidate, schedule, event, component)]
                row = {
                    "reference_bank": reference,
                    "candidate_bank": candidate,
                    "schedule": schedule,
                    "f_event": event,
                    "action_index": before["action_index"],
                    "logical_age_before_F": before["logical_age_before_F"],
                    "component": component,
                    "reference_margin_drop": before["margin_drop"],
                    "candidate_margin_drop": after["margin_drop"],
                    "margin_drop_delta": after["margin_drop"] - before["margin_drop"],
                    "absolute_margin_drop_delta": abs(
                        after["margin_drop"] - before["margin_drop"]
                    ),
                    "reference_accuracy_drop": before["accuracy_drop"],
                    "candidate_accuracy_drop": after["accuracy_drop"],
                    "accuracy_drop_delta": after["accuracy_drop"] - before["accuracy_drop"],
                    "reference_shared_correct_margin_drop": before[
                        "shared_correct_margin_drop"
                    ],
                    "candidate_shared_correct_margin_drop": after[
                        "shared_correct_margin_drop"
                    ],
                    "shared_correct_margin_drop_delta": (
                        after["shared_correct_margin_drop"]
                        - before["shared_correct_margin_drop"]
                    ),
                    "shared_correct_count_per_batch": before["shared_correct_count"],
                }
                comparison.append(row)
                values.append(row)
            before_values = np.asarray([row["reference_margin_drop"] for row in values])
            after_values = np.asarray([row["candidate_margin_drop"] for row in values])
            before_shared = np.asarray(
                [row["reference_shared_correct_margin_drop"] for row in values]
            )
            after_shared = np.asarray(
                [row["candidate_shared_correct_margin_drop"] for row in values]
            )
            finite_shared = np.isfinite(before_shared) & np.isfinite(after_shared)
            schedule_summary.append(
                {
                    "reference_bank": reference,
                    "candidate_bank": candidate,
                    "schedule": schedule,
                    "component_event_cells": len(values),
                    "causal_profile_pearson": float(
                        np.corrcoef(before_values, after_values)[0, 1]
                    ),
                    "margin_drop_mae": float(np.mean(np.abs(after_values - before_values))),
                    "margin_drop_relative_l2": float(
                        np.linalg.norm(after_values - before_values)
                        / max(np.linalg.norm(before_values), 1e-12)
                    ),
                    "largest_absolute_drift": float(
                        np.max(np.abs(after_values - before_values))
                    ),
                    "shared_correct_causal_profile_pearson": float(
                        np.corrcoef(
                            before_shared[finite_shared], after_shared[finite_shared]
                        )[0, 1]
                    )
                    if int(finite_shared.sum()) >= 2
                    else float("nan"),
                    "shared_correct_margin_drop_mae": float(
                        np.mean(
                            np.abs(
                                after_shared[finite_shared]
                                - before_shared[finite_shared]
                            )
                        )
                    )
                    if bool(finite_shared.any())
                    else float("nan"),
                    "shared_correct_count_per_batch": float(
                        np.mean([row["shared_correct_count_per_batch"] for row in values])
                    ),
                }
            )
    return comparison, schedule_summary


def _plot_differences(comparison: list[dict[str, Any]], out_dir: Path) -> None:
    if not comparison:
        return
    for candidate in sorted({row["candidate_bank"] for row in comparison}):
        for schedule in sorted({row["schedule"] for row in comparison}):
            rows = [
                row for row in comparison
                if row["candidate_bank"] == candidate and row["schedule"] == schedule
            ]
            components = sorted({row["component"] for row in rows})
            events = sorted({int(row["f_event"]) for row in rows})
            for metric, suffix, population in (
                ("margin_drop_delta", "difference", "all examples"),
                (
                    "shared_correct_margin_drop_delta",
                    "shared_correct_difference",
                    "examples answered correctly by both banks",
                ),
            ):
                matrix = np.asarray(
                    [
                        [
                            next(
                                row[metric] for row in rows
                                if row["component"] == component
                                and int(row["f_event"]) == event
                            )
                            for event in events
                        ]
                        for component in components
                    ],
                    dtype=float,
                )
                if not np.isfinite(matrix).any():
                    continue
                finite = np.abs(matrix[np.isfinite(matrix)])
                limit = max(float(np.quantile(finite, 0.98)), 1e-8)
                figure, axis = plt.subplots(figsize=(15, 5.8), dpi=180)
                image = axis.imshow(
                    matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
                )
                axis.set_yticks(range(len(components)), components, fontsize=7)
                axis.set_xticks(range(len(events)), events)
                axis.set_xlabel("effective F event")
                axis.set_title(
                    f"{candidate} - reference causal margin effect: {schedule}\n"
                    f"population: {population}"
                )
                figure.colorbar(image, ax=axis, fraction=0.025)
                figure.tight_layout()
                figure.savefig(
                    out_dir / f"circuit_drift_{suffix}_{candidate}_{schedule}.png",
                    bbox_inches="tight",
                )
                plt.close(figure)


def _compare_equivalent_paths(
    summary: list[dict[str, Any]], labels: list[str]
) -> list[dict[str, Any]]:
    lookup = {
        (row["bank"], row["schedule"], int(row["f_event"]), row["component"]): row
        for row in summary
    }
    output: list[dict[str, Any]] = []
    for label in labels:
        for left_schedule, right_schedule in (
            ("k8_left", "k8_right"),
            ("k12_left", "k12_right"),
        ):
            left_rows = [
                row for row in summary
                if row["bank"] == label and row["schedule"] == left_schedule
            ]
            left_values = []
            right_values = []
            for left in left_rows:
                key = (
                    label,
                    right_schedule,
                    int(left["f_event"]),
                    left["component"],
                )
                if key not in lookup:
                    continue
                left_values.append(left["margin_drop"])
                right_values.append(lookup[key]["margin_drop"])
            left_array = np.asarray(left_values)
            right_array = np.asarray(right_values)
            output.append(
                {
                    "bank": label,
                    "left_schedule": left_schedule,
                    "right_schedule": right_schedule,
                    "matched_component_event_cells": len(left_array),
                    "causal_profile_pearson": float(
                        np.corrcoef(left_array, right_array)[0, 1]
                    ),
                    "margin_drop_mae": float(np.mean(np.abs(left_array - right_array))),
                    "margin_drop_relative_l2": float(
                        np.linalg.norm(left_array - right_array)
                        / max(np.linalg.norm(left_array), 1e-12)
                    ),
                }
            )
    return output


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if len(args.bank_artifacts) != len(args.labels):
        raise ValueError("bank artifacts and labels must have equal lengths")
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    banks = {
        label: _load_bank(path, device, cfg.d_model)
        for label, path in zip(args.labels, args.bank_artifacts, strict=True)
    }
    positions = tuple(range(cfg.seq_len))
    schedules = _schedule_specs()
    components = _components(cfg)
    rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    for batch_index in range(args.examples // args.batch_size):
        tokens, _, successors, start = fixed_depth_batch(
            cfg, args.batch_size, device, path_positions=cfg.max_depth
        )
        raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        h1 = model.apply_loop(raw, loop_index=0)
        h1_current = advance_nodes(successors, start, steps=1)
        for schedule, actions in schedules.items():
            clean: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for label, bank in banks.items():
                clean[label] = _execute(
                    model=model,
                    bank=bank,
                    state=h1,
                    current=h1_current,
                    successors=successors,
                    actions=actions,
                    positions=positions,
                    attention_rows=attention_rows,
                    attention_context={
                        "bank": label,
                        "schedule": schedule,
                        "batch": batch_index,
                    },
                )
            targets = [value[1] for value in clean.values()]
            if not all(target.eq(targets[0]).all() for target in targets[1:]):
                raise RuntimeError("bank conditions produced different targets")
            target = targets[0]
            shared_correct = torch.stack(
                [logits.argmax(-1).eq(target) for logits, _ in clean.values()]
            ).all(dim=0)
            phases = _f_event_phases(actions)
            for label, bank in banks.items():
                clean_logits = clean[label][0]
                clean_margin = target_margin(clean_logits.float(), target)
                for f_event, (action_index, logical_age) in phases.items():
                    for component, intervention in components:
                        ablated_logits, ablated_target = _execute(
                            model=model,
                            bank=bank,
                            state=h1,
                            current=h1_current,
                            successors=successors,
                            actions=actions,
                            positions=positions,
                            ablate_f_event=f_event,
                            intervention=intervention,
                        )
                        if not ablated_target.eq(target).all():
                            raise RuntimeError("ablation changed graph target")
                        ablated_margin = target_margin(ablated_logits.float(), target)
                        margin_drop = clean_margin - ablated_margin
                        rows.append(
                            {
                                "bank": label,
                                "schedule": schedule,
                                "batch": batch_index,
                                "f_event": f_event,
                                "action_index": action_index,
                                "logical_age_before_F": logical_age,
                                "component": component,
                                "clean_accuracy": float(
                                    clean_logits.argmax(-1).eq(target).float().mean()
                                ),
                                "ablated_accuracy": float(
                                    ablated_logits.argmax(-1).eq(target).float().mean()
                                ),
                                "accuracy_drop": float(
                                    clean_logits.argmax(-1).eq(target).float().mean()
                                    - ablated_logits.argmax(-1).eq(target).float().mean()
                                ),
                                "clean_margin": float(clean_margin.mean()),
                                "ablated_margin": float(ablated_margin.mean()),
                                "margin_drop": float(margin_drop.mean()),
                                "shared_correct_margin_drop": (
                                    float(margin_drop[shared_correct].mean())
                                    if bool(shared_correct.any())
                                    else float("nan")
                                ),
                                "shared_correct_count": int(shared_correct.sum()),
                            }
                        )
    summary = _aggregate(rows)
    comparison, schedule_comparison = _compare_banks(summary, list(args.labels))
    equivalent_path_comparison = _compare_equivalent_paths(summary, list(args.labels))
    attention_summary = _aggregate_attention(attention_rows)
    attention_comparison = _compare_attention_roles(
        attention_summary, list(args.labels)
    )
    _write_csv(args.out_dir / "circuit_ablation_per_batch.csv", rows)
    _write_csv(args.out_dir / "circuit_ablation_summary.csv", summary)
    _write_csv(args.out_dir / "circuit_drift_comparison.csv", comparison)
    _write_csv(
        args.out_dir / "circuit_drift_schedule_comparison.csv", schedule_comparison
    )
    _write_csv(
        args.out_dir / "equivalent_path_circuit_comparison.csv",
        equivalent_path_comparison,
    )
    _write_csv(args.out_dir / "clean_attention_roles_per_batch.csv", attention_rows)
    _write_csv(args.out_dir / "clean_attention_roles_summary.csv", attention_summary)
    _write_csv(
        args.out_dir / "clean_attention_role_comparison.csv", attention_comparison
    )
    _plot(summary, args.out_dir)
    _plot_differences(comparison, args.out_dir)
    top_drift = sorted(
        comparison,
        key=lambda row: row["absolute_margin_drop_delta"],
        reverse=True,
    )[:50]
    top_shared_correct_drift = sorted(
        (
            row
            for row in comparison
            if np.isfinite(float(row["shared_correct_margin_drop_delta"]))
        ),
        key=lambda row: abs(float(row["shared_correct_margin_drop_delta"])),
        reverse=True,
    )[:50]
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "banks": {
            label: str(path)
            for label, path in zip(args.labels, args.bank_artifacts, strict=True)
        },
        "behavior": "final H8 graph prediction after a fixed H1-start F/J word",
        "intervention": "zero one answer-position head context or block MLP output at one effective F event",
        "claim_boundary": "component causal-effect drift only; no completeness, minimality, or uniqueness claim",
        "schedules": {
            name: {
                "actions": "".join("F" if action == 1 else "J" for action in actions),
                "semantics": action_semantics(actions).__dict__,
            }
            for name, actions in schedules.items()
        },
        "examples": args.examples,
        "components": [component for component, _ in components],
        "reference_bank": args.labels[0],
        "schedule_causal_profile_comparison": schedule_comparison,
        "equivalent_path_causal_profile_comparison": equivalent_path_comparison,
        "attention_role_evidence": (
            "correlational clean attention mass by source-position group; use with, "
            "not instead of, answer-position head-context zero ablation"
        ),
        "top_50_component_event_drifts": top_drift,
        "top_50_shared_correct_component_event_drifts": top_shared_correct_drift,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
