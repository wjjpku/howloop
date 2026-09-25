"""Localize natural computational age with bidirectional attention mediation.

For a late post-J event, this experiment finds the earliest earlier post-J
state in the same trajectory with the same graph current node and logical age.
The next F therefore has identical task semantics in the late and young
branches.  It patches one internal attention component only in that first F:

* young -> late tests rescue;
* late -> young tests disruption.

The remaining suffix is then executed without intervention.  Final accuracy,
consecutive correct F steps, suffix accuracy AUC, and Block-2 destination
attention are measured only on examples with a valid young match.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.analyze_graph_path_j_attention_circuit import (
    PatchCase,
    build_patch_cases,
    build_words,
    final_state_from_trace,
    target_margin,
)
from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    atomic_json,
    load_bank_and_bases,
)
from reasoning_loop.analyze_graph_path_j_late_residual_transplant import (
    collect_rollback_events,
    gather_matched_young,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import MAX_AGE, MIN_AGE


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--graph-seeds", type=int, nargs="+", default=(881001, 881002, 881003)
    )
    parser.add_argument("--back-counts", type=int, nargs="+", default=(32, 48))
    parser.add_argument("--path-pairs-per-k", type=int, default=1)
    parser.add_argument("--word-seed", type=int, default=881701)
    parser.add_argument(
        "--rollback-checkpoints", type=int, nargs="+", default=(16, 24, 32, 48)
    )
    parser.add_argument(
        "--patch-labels",
        nargs="+",
        default=(
            "B1.H1.qkv",
            "B2.H0.q",
            "B2.H0.k",
            "B2.H0.v",
            "B2.H0.qk",
            "B2.H0.qkv",
            "B2.H0.pattern",
            "B2.H0.context",
            "B2.H1.qkv",
            "B2.H1.pattern",
            "B2.H1.context",
            "B2.attention_out",
            "B2.mlp_out",
        ),
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


def _destination_masses(
    trace: FunctionalTrace, current: torch.Tensor, cfg
) -> list[torch.Tensor]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"][0]
    destinations = torch.as_tensor(
        groups["destination"], dtype=torch.long, device=current.device
    )
    batch = torch.arange(current.shape[0], device=current.device)
    pattern = trace.sites[1].attention_pattern[:, :, answer]
    positions = destinations[current]
    return [pattern[batch, head, positions] for head in range(pattern.shape[1])]


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
    patch_case: PatchCase | None = None,
    donor_trace: FunctionalTrace | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor | float | list[torch.Tensor]],
]:
    logical_age = start_age
    alive = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    survival = torch.zeros(state.shape[0], dtype=torch.float32, device=state.device)
    correct_sum = torch.zeros_like(survival)
    forward_count = 0
    destination_masses: list[torch.Tensor] | None = None
    for action in actions:
        if action == 1:
            if forward_count == 0:
                _, trace = run_instrumented_state(
                    model,
                    state,
                    loop_indices=(logical_age,),
                    interventions=() if patch_case is None else patch_case.interventions,
                    donor_trace=donor_trace,
                )
                destination_masses = _destination_masses(trace, current, cfg)
                state = final_state_from_trace(model, trace)
            else:
                state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
            logits = logits_from_raw_state(model, state).float()
            correct = logits.argmax(-1).eq(current)
            alive &= correct
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
    if destination_masses is None:
        raise RuntimeError("selected rollback was not followed by F")
    return state, current, {
        "survival_steps": survival,
        "forward_accuracy_auc": correct_sum / max(forward_count, 1),
        "suffix_forward_count": float(forward_count),
        "destination_masses": destination_masses,
    }


def _weighted_seed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("graph_seed", "back_count", "rollback_checkpoint", "condition", "direction")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    metric_names = (
        "accuracy",
        "margin",
        "ce",
        "survival_steps",
        "forward_accuracy_auc",
        "destination_mass_h0",
        "destination_mass_h1",
        "destination_mass_h2",
        "destination_mass_h3",
        "suffix_forward_count",
    )
    for key, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        weights = np.asarray([float(part["matched_examples"]) for part in parts])
        result = dict(zip(keys, key, strict=True))
        result["matched_examples"] = int(weights.sum())
        for metric in metric_names:
            values = np.asarray([float(part[metric]) for part in parts])
            result[metric] = float(np.average(values, weights=weights))
        output.append(result)
    return output


def _condition_summary(seed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("back_count", "rollback_checkpoint", "condition", "direction")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "accuracy",
        "margin",
        "survival_steps",
        "forward_accuracy_auc",
        "destination_mass_h0",
        "destination_mass_h1",
        "destination_mass_h2",
        "destination_mass_h3",
    )
    for key, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, key, strict=True))
        for metric in metrics:
            values = np.asarray([float(part[metric]) for part in parts])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        result["graph_seeds"] = len(parts)
        result["matched_examples"] = int(sum(int(part["matched_examples"]) for part in parts))
        output.append(result)
    return output


def _mediation_summary(seed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row["graph_seed"]),
            int(row["back_count"]),
            int(row["rollback_checkpoint"]),
            str(row["condition"]),
        ): row
        for row in seed_rows
    }
    delta_rows: list[dict[str, Any]] = []
    metrics = (
        "accuracy",
        "margin",
        "survival_steps",
        "forward_accuracy_auc",
        "destination_mass_h0",
        "destination_mass_h1",
    )
    for row in seed_rows:
        direction = str(row["direction"])
        if direction not in {"young_to_late_rescue", "late_to_young_disrupt"}:
            continue
        baseline = "late_baseline" if direction == "young_to_late_rescue" else "young_reference"
        reference = lookup[
            (
                int(row["graph_seed"]),
                int(row["back_count"]),
                int(row["rollback_checkpoint"]),
                baseline,
            )
        ]
        delta = {
            "graph_seed": int(row["graph_seed"]),
            "back_count": int(row["back_count"]),
            "rollback_checkpoint": int(row["rollback_checkpoint"]),
            "condition": str(row["condition"]),
            "direction": direction,
        }
        for metric in metrics:
            delta[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
        delta_rows.append(delta)
    keys = ("back_count", "rollback_checkpoint", "condition", "direction")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, parts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, key, strict=True))
        for metric in metrics:
            values = np.asarray([float(part[f"delta_{metric}"]) for part in parts])
            result[f"delta_{metric}_mean"] = float(values.mean())
            result[f"delta_{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        result["graph_seeds"] = len(parts)
        output.append(result)
    return output


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    patch_cases: Sequence[PatchCase],
    words: Sequence[tuple[int, str, tuple[int, ...]]],
    rollback_checkpoints: set[int],
    graph_seeds: Sequence[int],
    examples: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch_size")
    positions = tuple(range(cfg.seq_len))
    rows: list[dict[str, Any]] = []
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        print(json.dumps({"event": "seed_start", "graph_seed": int(graph_seed)}), flush=True)
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
                    if matched_count == 0:
                        continue
                    _, late_trace = run_instrumented_state(
                        model, event.state, loop_indices=(event.target_age,)
                    )
                    _, young_trace = run_instrumented_state(
                        model, young, loop_indices=(event.target_age,)
                    )
                    branches: list[
                        tuple[str, str, torch.Tensor, PatchCase | None, FunctionalTrace | None]
                    ] = [
                        ("late_baseline", "baseline", event.state, None, None),
                        ("young_reference", "baseline", young, None, None),
                    ]
                    for case in patch_cases:
                        branches.extend(
                            (
                                (
                                    f"{case.label}.young_to_late",
                                    "young_to_late_rescue",
                                    event.state,
                                    case,
                                    young_trace,
                                ),
                                (
                                    f"{case.label}.late_to_young",
                                    "late_to_young_disrupt",
                                    young,
                                    case,
                                    late_trace,
                                ),
                            )
                        )
                    for condition, direction, receiver, case, donor in branches:
                        final_state, final_target, trajectory = execute_suffix(
                            model=model,
                            bank=bank,
                            state=receiver,
                            current=event.current,
                            successors=successors,
                            start_age=event.target_age,
                            actions=suffix,
                            positions=positions,
                            cfg=cfg,
                            patch_case=case,
                            donor_trace=donor,
                        )
                        if not torch.equal(final_target, baseline_target):
                            raise RuntimeError("branch suffix changed graph target")
                        logits = logits_from_raw_state(model, final_state).float()[matched]
                        target = final_target[matched]
                        masses = trajectory["destination_masses"]
                        assert isinstance(masses, list)
                        rows.append(
                            {
                                "graph_seed": int(graph_seed),
                                "batch": batch_index,
                                "back_count": int(back_count),
                                "word": word,
                                "rollback_checkpoint": event.rollback_index,
                                "source_age": event.source_age,
                                "target_age": event.target_age,
                                "condition": condition,
                                "direction": direction,
                                "matched_fraction": float(matched.float().mean()),
                                "matched_examples": matched_count,
                                "accuracy": float(logits.argmax(-1).eq(target).float().mean()),
                                "margin": float(target_margin(logits, target).mean()),
                                "ce": float(F.cross_entropy(logits, target)),
                                "survival_steps": float(trajectory["survival_steps"][matched].mean()),
                                "forward_accuracy_auc": float(
                                    trajectory["forward_accuracy_auc"][matched].mean()
                                ),
                                "destination_mass_h0": float(masses[0][matched].mean()),
                                "destination_mass_h1": float(masses[1][matched].mean()),
                                "destination_mass_h2": float(masses[2][matched].mean()),
                                "destination_mass_h3": float(masses[3][matched].mean()),
                                "suffix_forward_count": float(trajectory["suffix_forward_count"]),
                                "baseline_fullword_accuracy": float(
                                    baseline_logits.argmax(-1)
                                    .eq(baseline_target)
                                    .float()
                                    .mean()
                                ),
                            }
                        )
        print(json.dumps({"event": "seed_complete", "graph_seed": int(graph_seed)}), flush=True)
    return rows


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
        torch.cuda.set_per_process_memory_fraction(fraction, device=torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    bank, _, _, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (8,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    available = {case.label: case for case in build_patch_cases(cfg)}
    missing = sorted(set(args.patch_labels) - set(available))
    if missing:
        raise ValueError(f"unknown patch labels: {missing}")
    patch_cases = [available[label] for label in args.patch_labels]
    words = build_words(
        back_counts=tuple(args.back_counts),
        path_pairs_per_k=args.path_pairs_per_k,
        seed=args.word_seed,
    )
    rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        patch_cases=patch_cases,
        words=words,
        rollback_checkpoints=set(args.rollback_checkpoints),
        graph_seeds=tuple(args.graph_seeds),
        examples=args.examples,
        batch_size=args.batch_size,
        device=device,
    )
    seed_rows = _weighted_seed_rows(rows)
    condition_rows = _condition_summary(seed_rows)
    mediation_rows = _mediation_summary(seed_rows)
    write_csv(args.out_dir / "branch_rows.csv", rows)
    write_csv(args.out_dir / "seed_summary.csv", seed_rows)
    write_csv(args.out_dir / "condition_summary.csv", condition_rows)
    write_csv(args.out_dir / "mediation_summary.csv", mediation_rows)
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "back_counts": list(args.back_counts),
        "path_pairs_per_k": args.path_pairs_per_k,
        "rollback_checkpoints": list(args.rollback_checkpoints),
        "patch_cases": [case.label for case in patch_cases],
        "match_definition": (
            "earliest earlier post-J event in the same action word with identical "
            "graph, original query, graph-current node, and target logical age"
        ),
        "intervention_scope": (
            "one attention component in only the first F after the selected post-J event; "
            "the remaining suffix is unpatched"
        ),
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "claim_boundary": (
            "Bidirectional component mediation localizes a causal carrier at the selected "
            "post-J boundary. It does not by itself prove that the component is sufficient "
            "for the full distributed computational-age state."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
