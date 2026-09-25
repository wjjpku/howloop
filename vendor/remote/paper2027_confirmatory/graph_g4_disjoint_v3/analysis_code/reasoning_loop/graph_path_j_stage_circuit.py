"""Causally localize how stage-specific J residuals alter frozen-loop behavior.

The clean trajectory starts from H8, applies the ordered adjacent product
J8,J7,J6,J5,J4, and then runs the frozen executor from logical age H3 through
H8.  The corrupt trajectory replaces every J with the learned shared component
D+U0+b.  Discovery uses bidirectional patching and a batch-shuffled donor.
The selected sparse head-context circuit is then tested on a disjoint split
against its complement and size-matched random subsets.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_j_interval_closure import (
    apply_adjacent_sequence,
    apply_shared_sequence,
    load_adjacent_bank,
    ordered_source_ages,
    phase_jump,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age


Direction = Literal["patch_in", "patch_out"]


@dataclass(frozen=True)
class PatchNode:
    label: str
    site: int
    component: str
    positions: tuple[int, ...] | None = None
    heads: tuple[int, ...] | None = None

    def intervention(self) -> FunctionalIntervention:
        return FunctionalIntervention(
            site=self.site,
            component=self.component,  # type: ignore[arg-type]
            mode="patch",
            positions=self.positions,
            heads=self.heads,
        )


def branch_nodes(
    *,
    loop_indices: Sequence[int],
    n_layers: int,
) -> tuple[PatchNode, ...]:
    result: list[PatchNode] = []
    for loop_offset, loop_index in enumerate(loop_indices):
        for block in range(n_layers):
            site = loop_offset * n_layers + block
            for component in ("attention_out", "mlp_out"):
                result.append(
                    PatchNode(
                        label=f"H{loop_index}_F.B{block + 1}.{component}.all",
                        site=site,
                        component=component,
                    )
                )
    return tuple(result)


def head_context_nodes(
    *,
    loop_indices: Sequence[int],
    n_layers: int,
    n_heads: int,
    position_groups: dict[str, tuple[int, ...]],
) -> tuple[PatchNode, ...]:
    result: list[PatchNode] = []
    for loop_offset, loop_index in enumerate(loop_indices):
        for block in range(n_layers):
            site = loop_offset * n_layers + block
            for head in range(n_heads):
                for group, positions in position_groups.items():
                    result.append(
                        PatchNode(
                            label=(
                                f"H{loop_index}_F.B{block + 1}.H{head}."
                                f"{group}.head_context"
                            ),
                            site=site,
                            component="head_context",
                            positions=positions,
                            heads=(head,),
                        )
                    )
    return tuple(result)


def mlp_group_nodes(
    *,
    loop_indices: Sequence[int],
    n_layers: int,
    position_groups: dict[str, tuple[int, ...]],
) -> tuple[PatchNode, ...]:
    result: list[PatchNode] = []
    for loop_offset, loop_index in enumerate(loop_indices):
        for block in range(n_layers):
            site = loop_offset * n_layers + block
            for group, positions in position_groups.items():
                result.append(
                    PatchNode(
                        label=f"H{loop_index}_F.B{block + 1}.{group}.mlp_out",
                        site=site,
                        component="mlp_out",
                        positions=positions,
                    )
                )
    return tuple(result)


def effect_from_margins(
    *,
    clean: float,
    corrupt: float,
    patched: float,
    direction: Direction,
) -> float:
    denominator = clean - corrupt
    if abs(denominator) < 1e-8:
        return float("nan")
    if direction == "patch_in":
        return (patched - corrupt) / denominator
    if direction == "patch_out":
        return (clean - patched) / denominator
    raise ValueError(f"unknown patch direction: {direction}")


def select_circuit(
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    scored = []
    for row in rows:
        score = min(
            float(row["patch_in_recovery"]),
            float(row["patch_out_damage"]),
        ) - max(0.0, float(row["shuffled_recovery"]))
        scored.append({**dict(row), "selection_score": score})
    return sorted(
        scored,
        key=lambda row: (-float(row["selection_score"]), str(row["label"])),
    )[:top_k]


def attention_only(nodes: Sequence[PatchNode]) -> tuple[PatchNode, ...]:
    return tuple(node for node in nodes if node.component == "head_context")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rolled_trace(trace: FunctionalTrace) -> FunctionalTrace:
    result = copy.deepcopy(trace)
    for site in result.sites:
        for key, value in vars(site).items():
            if isinstance(value, torch.Tensor):
                setattr(site, key, value.roll(1, dims=0))
    result.logits_by_loop = result.logits_by_loop.roll(1, dims=1)
    return result


def _target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    mask = torch.nn.functional.one_hot(target, num_classes=logits.shape[-1]).bool()
    strongest_other = logits.masked_fill(mask, float("-inf")).max(dim=-1).values
    return correct - strongest_other


def _behavior(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(target).float().mean()),
        "target_margin": float(_target_margin(logits.float(), target).mean()),
        "target_probability": float(
            logits.softmax(dim=-1).gather(1, target[:, None]).mean()
        ),
    }


@dataclass
class PreparedBatch:
    target: torch.Tensor
    clean_state: torch.Tensor
    corrupt_state: torch.Tensor
    clean_logits: torch.Tensor
    corrupt_logits: torch.Tensor
    clean_trace: FunctionalTrace
    corrupt_trace: FunctionalTrace


@torch.no_grad()
def prepare_batch(
    *,
    model,
    cfg,
    adjacent,
    phase_positions: Sequence[int],
    positions: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    source_age: int,
    target_age: int,
) -> PreparedBatch:
    _, path_targets, successors, _ = fixed_depth_batch(
        cfg, batch_size, device, path_positions=cfg.max_depth
    )
    current = path_targets[:, cfg.max_depth - 1]
    source = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=current,
        age=source_age,
        phase_position=phase_positions[source_age],
    )
    steps = source_age - target_age
    clean_state = apply_adjacent_sequence(
        source,
        bank=adjacent,
        source_ages=ordered_source_ages(source_age, target_age),
        positions=positions,
    )
    corrupt_state = apply_shared_sequence(
        source,
        bank=adjacent,
        steps=steps,
        positions=positions,
    )
    loop_indices = tuple(range(target_age, source_age))
    clean_logits, clean_trace = run_instrumented_state(
        model, clean_state, loop_indices=loop_indices
    )
    corrupt_logits, corrupt_trace = run_instrumented_state(
        model, corrupt_state, loop_indices=loop_indices
    )
    target = advance_nodes(
        successors,
        current,
        steps=phase_jump(phase_positions) * steps,
    )
    return PreparedBatch(
        target=target,
        clean_state=clean_state,
        corrupt_state=corrupt_state,
        clean_logits=clean_logits,
        corrupt_logits=corrupt_logits,
        clean_trace=clean_trace,
        corrupt_trace=corrupt_trace,
    )


@torch.no_grad()
def patched_behavior(
    *,
    model,
    receiver_state: torch.Tensor,
    loop_indices: Sequence[int],
    nodes: Sequence[PatchNode],
    donor_trace: FunctionalTrace,
    target: torch.Tensor,
) -> dict[str, float]:
    logits, _ = run_instrumented_state(
        model,
        receiver_state,
        loop_indices=loop_indices,
        interventions=tuple(node.intervention() for node in nodes),
        donor_trace=donor_trace,
    )
    return _behavior(logits, target)


def _effect_row(
    *,
    label: str,
    family: str,
    direction: Direction,
    clean: dict[str, float],
    corrupt: dict[str, float],
    patched: dict[str, float],
    batch_index: int,
) -> dict[str, Any]:
    return {
        "family": family,
        "label": label,
        "direction": direction,
        "batch": batch_index,
        "clean_accuracy": clean["accuracy"],
        "corrupt_accuracy": corrupt["accuracy"],
        "patched_accuracy": patched["accuracy"],
        "clean_margin": clean["target_margin"],
        "corrupt_margin": corrupt["target_margin"],
        "patched_margin": patched["target_margin"],
        "normalized_effect": effect_from_margins(
            clean=clean["target_margin"],
            corrupt=corrupt["target_margin"],
            patched=patched["target_margin"],
            direction=direction,
        ),
    }


@torch.no_grad()
def discover(
    *,
    model,
    cfg,
    adjacent,
    phase_positions: Sequence[int],
    positions: tuple[int, ...],
    device: torch.device,
    branch: Sequence[PatchNode],
    candidates: Sequence[PatchNode],
    source_age: int,
    target_age: int,
    examples: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("discovery examples must be divisible by batch size")
    set_seed(seed)
    loop_indices = tuple(range(target_age, source_age))
    rows: list[dict[str, Any]] = []
    for batch_index in range(examples // batch_size):
        batch = prepare_batch(
            model=model,
            cfg=cfg,
            adjacent=adjacent,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            batch_size=batch_size,
            source_age=source_age,
            target_age=target_age,
        )
        clean = _behavior(batch.clean_logits, batch.target)
        corrupt = _behavior(batch.corrupt_logits, batch.target)
        shuffled_clean = _rolled_trace(batch.clean_trace)
        for family, nodes in (("branch", branch), ("candidate", candidates)):
            for node in nodes:
                patch_in = patched_behavior(
                    model=model,
                    receiver_state=batch.corrupt_state,
                    loop_indices=loop_indices,
                    nodes=(node,),
                    donor_trace=batch.clean_trace,
                    target=batch.target,
                )
                rows.append(
                    _effect_row(
                        label=node.label,
                        family=family,
                        direction="patch_in",
                        clean=clean,
                        corrupt=corrupt,
                        patched=patch_in,
                        batch_index=batch_index,
                    )
                )
                shuffled = patched_behavior(
                    model=model,
                    receiver_state=batch.corrupt_state,
                    loop_indices=loop_indices,
                    nodes=(node,),
                    donor_trace=shuffled_clean,
                    target=batch.target,
                )
                shuffled_row = _effect_row(
                    label=node.label,
                    family=family,
                    direction="patch_in",
                    clean=clean,
                    corrupt=corrupt,
                    patched=shuffled,
                    batch_index=batch_index,
                )
                shuffled_row["direction"] = "shuffled_patch_in"
                rows.append(shuffled_row)
                patch_out = patched_behavior(
                    model=model,
                    receiver_state=batch.clean_state,
                    loop_indices=loop_indices,
                    nodes=(node,),
                    donor_trace=batch.corrupt_trace,
                    target=batch.target,
                )
                rows.append(
                    _effect_row(
                        label=node.label,
                        family=family,
                        direction="patch_out",
                        clean=clean,
                        corrupt=corrupt,
                        patched=patch_out,
                        batch_index=batch_index,
                    )
                )
    return rows


def aggregate_discovery(
    raw_rows: Sequence[dict[str, Any]],
    *,
    family: str,
) -> list[dict[str, Any]]:
    labels = sorted({str(row["label"]) for row in raw_rows if row["family"] == family})
    result: list[dict[str, Any]] = []
    for label in labels:
        chosen = [row for row in raw_rows if row["family"] == family and row["label"] == label]
        by_direction = {
            direction: [
                float(row["normalized_effect"])
                for row in chosen
                if row["direction"] == direction
            ]
            for direction in ("patch_in", "patch_out", "shuffled_patch_in")
        }
        result.append(
            {
                "family": family,
                "label": label,
                "patch_in_recovery": float(np.nanmean(by_direction["patch_in"])),
                "patch_out_damage": float(np.nanmean(by_direction["patch_out"])),
                "shuffled_recovery": float(np.nanmean(by_direction["shuffled_patch_in"])),
                "batches": len(by_direction["patch_in"]),
            }
        )
    return result


def _random_subsets(
    nodes: Sequence[PatchNode],
    *,
    size: int,
    count: int,
    seed: int,
) -> list[tuple[PatchNode, ...]]:
    if size > len(nodes):
        raise ValueError("random subset cannot exceed candidate count")
    rng = np.random.default_rng(seed)
    return [
        tuple(nodes[int(index)] for index in rng.choice(len(nodes), size=size, replace=False))
        for _ in range(count)
    ]


def random_control_pool(
    candidates: Sequence[PatchNode],
    selected: Sequence[PatchNode],
) -> tuple[PatchNode, ...]:
    selected_labels = {node.label for node in selected}
    return tuple(node for node in candidates if node.label not in selected_labels)


@torch.no_grad()
def validate_circuit(
    *,
    model,
    cfg,
    adjacent,
    phase_positions: Sequence[int],
    positions: tuple[int, ...],
    device: torch.device,
    candidates: Sequence[PatchNode],
    selected: Sequence[PatchNode],
    source_age: int,
    target_age: int,
    examples: int,
    batch_size: int,
    random_subsets: int,
    seed: int,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("validation examples must be divisible by batch size")
    set_seed(seed)
    loop_indices = tuple(range(target_age, source_age))
    complement = random_control_pool(candidates, selected)
    random_sets = _random_subsets(
        complement, size=len(selected), count=random_subsets, seed=seed + 99
    )
    rows: list[dict[str, Any]] = []
    for batch_index in range(examples // batch_size):
        batch = prepare_batch(
            model=model,
            cfg=cfg,
            adjacent=adjacent,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            batch_size=batch_size,
            source_age=source_age,
            target_age=target_age,
        )
        clean = _behavior(batch.clean_logits, batch.target)
        corrupt = _behavior(batch.corrupt_logits, batch.target)
        shuffled = _rolled_trace(batch.clean_trace)
        conditions: list[tuple[str, Sequence[PatchNode], torch.Tensor, FunctionalTrace, Direction]] = [
            ("circuit_only", selected, batch.corrupt_state, batch.clean_trace, "patch_in"),
            ("selected_patch_out", selected, batch.clean_state, batch.corrupt_trace, "patch_out"),
            ("all_candidates", candidates, batch.corrupt_state, batch.clean_trace, "patch_in"),
            ("complement_only", complement, batch.corrupt_state, batch.clean_trace, "patch_in"),
            ("shuffled_circuit", selected, batch.corrupt_state, shuffled, "patch_in"),
        ]
        conditions.extend(
            (
                f"random_{index:02d}",
                subset,
                batch.corrupt_state,
                batch.clean_trace,
                "patch_in",
            )
            for index, subset in enumerate(random_sets)
        )
        rows.extend(
            [
                {
                    "condition": "clean",
                    "direction": "baseline",
                    "batch": batch_index,
                    "accuracy": clean["accuracy"],
                    "target_margin": clean["target_margin"],
                    "normalized_effect": 1.0,
                    "node_labels": "",
                },
                {
                    "condition": "corrupt_shared_only",
                    "direction": "baseline",
                    "batch": batch_index,
                    "accuracy": corrupt["accuracy"],
                    "target_margin": corrupt["target_margin"],
                    "normalized_effect": 0.0,
                    "node_labels": "",
                },
            ]
        )
        for condition, nodes, receiver, donor, direction in conditions:
            patched = patched_behavior(
                model=model,
                receiver_state=receiver,
                loop_indices=loop_indices,
                nodes=nodes,
                donor_trace=donor,
                target=batch.target,
            )
            rows.append(
                {
                    "condition": condition,
                    "direction": direction,
                    "batch": batch_index,
                    "accuracy": patched["accuracy"],
                    "target_margin": patched["target_margin"],
                    "normalized_effect": effect_from_margins(
                        clean=clean["target_margin"],
                        corrupt=corrupt["target_margin"],
                        patched=patched["target_margin"],
                        direction=direction,
                    ),
                    "node_labels": "|".join(node.label for node in nodes),
                }
            )
    return rows


def aggregate_validation(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        result.append(
            {
                "condition": condition,
                "accuracy": float(np.mean([float(row["accuracy"]) for row in selected])),
                "target_margin": float(np.mean([float(row["target_margin"]) for row in selected])),
                "normalized_effect": float(
                    np.nanmean([float(row["normalized_effect"]) for row in selected])
                ),
                "batches": len(selected),
            }
        )
    return result


def _mediation_interventions(node: PatchNode, component: str) -> tuple[FunctionalIntervention, ...]:
    base = dict(site=node.site, mode="patch", heads=node.heads)
    if component == "q":
        return (FunctionalIntervention(component="q", positions=node.positions, **base),)
    if component in {"k", "v"}:
        return (FunctionalIntervention(component=component, positions=None, **base),)  # type: ignore[arg-type]
    if component == "attention_pattern":
        return (
            FunctionalIntervention(
                component="attention_pattern", positions=node.positions, **base
            ),
        )
    if component == "qkv":
        return (
            FunctionalIntervention(component="q", positions=node.positions, **base),
            FunctionalIntervention(component="k", positions=None, **base),
            FunctionalIntervention(component="v", positions=None, **base),
        )
    raise ValueError(f"unknown mediation component: {component}")


@torch.no_grad()
def qkv_mediation(
    *,
    model,
    cfg,
    adjacent,
    phase_positions: Sequence[int],
    positions: tuple[int, ...],
    device: torch.device,
    selected: Sequence[PatchNode],
    source_age: int,
    target_age: int,
    examples: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("mediation examples must be divisible by batch size")
    set_seed(seed)
    loop_indices = tuple(range(target_age, source_age))
    rows: list[dict[str, Any]] = []
    for batch_index in range(examples // batch_size):
        batch = prepare_batch(
            model=model,
            cfg=cfg,
            adjacent=adjacent,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            batch_size=batch_size,
            source_age=source_age,
            target_age=target_age,
        )
        clean = _behavior(batch.clean_logits, batch.target)
        corrupt = _behavior(batch.corrupt_logits, batch.target)
        for node in selected:
            for component in ("q", "k", "v", "attention_pattern", "qkv"):
                logits, _ = run_instrumented_state(
                    model,
                    batch.corrupt_state,
                    loop_indices=loop_indices,
                    interventions=_mediation_interventions(node, component),
                    donor_trace=batch.clean_trace,
                )
                behavior = _behavior(logits, batch.target)
                rows.append(
                    {
                        "label": node.label,
                        "component": component,
                        "batch": batch_index,
                        "accuracy": behavior["accuracy"],
                        "target_margin": behavior["target_margin"],
                        "normalized_recovery": effect_from_margins(
                            clean=clean["target_margin"],
                            corrupt=corrupt["target_margin"],
                            patched=behavior["target_margin"],
                            direction="patch_in",
                        ),
                    }
                )
    return rows


def aggregate_mediation(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(str(row["label"]), str(row["component"])) for row in rows})
    for label, component in keys:
        chosen = [row for row in rows if row["label"] == label and row["component"] == component]
        result.append(
            {
                "label": label,
                "component": component,
                "accuracy": float(np.mean([float(row["accuracy"]) for row in chosen])),
                "target_margin": float(np.mean([float(row["target_margin"]) for row in chosen])),
                "normalized_recovery": float(
                    np.nanmean([float(row["normalized_recovery"]) for row in chosen])
                ),
                "batches": len(chosen),
            }
        )
    return result


def circuit_decision(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {str(row["condition"]): row for row in rows}
    random_effects = [
        float(row["normalized_effect"])
        for row in rows
        if str(row["condition"]).startswith("random_")
    ]
    circuit = float(by_condition["circuit_only"]["normalized_effect"])
    patch_out = float(by_condition["selected_patch_out"]["normalized_effect"])
    shuffled = float(by_condition["shuffled_circuit"]["normalized_effect"])
    complement = float(by_condition["complement_only"]["normalized_effect"])
    random_p95 = float(np.quantile(random_effects, 0.95))
    baseline_accuracy_gap = float(by_condition["clean"]["accuracy"]) - float(
        by_condition["corrupt_shared_only"]["accuracy"]
    )
    passed = (
        baseline_accuracy_gap >= 0.10
        and circuit >= 0.50
        and patch_out >= 0.30
        and shuffled <= 0.20
        and circuit >= random_p95 + 0.10
        and complement <= 0.30
    )
    return {
        "decision": (
            "invalid_baseline"
            if baseline_accuracy_gap < 0.10
            else ("supported" if passed else "not_supported")
        ),
        "clean_corrupt_accuracy_gap": baseline_accuracy_gap,
        "circuit_only_recovery": circuit,
        "selected_patch_out_damage": patch_out,
        "shuffled_circuit_recovery": shuffled,
        "complement_only_recovery": complement,
        "random_subset_p95_recovery": random_p95,
        "gates": {
            "clean_corrupt_accuracy_gap_at_least": 0.10,
            "circuit_only_at_least": 0.50,
            "patch_out_at_least": 0.30,
            "shuffled_recovery_at_most": 0.20,
            "circuit_over_random_p95_at_least": 0.10,
            "complement_at_most": 0.30,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--adjacent-bank", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=827001)
    parser.add_argument("--source-age", type=int, default=8)
    parser.add_argument("--target-age", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--discovery-examples", type=int, default=128)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-subsets", type=int, default=20)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.source_age - args.target_age != 5:
        raise ValueError("pre-registered circuit interval is fixed to five rollback steps")
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
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(value) for value in phase_payload["trajectory_positions_including_initial"]]
    adjacent, adjacent_payload = load_adjacent_bank(
        args.adjacent_bank, dimension=cfg.d_model, device=device
    )
    positions = tuple(range(cfg.seq_len))
    loop_indices = tuple(range(args.target_age, args.source_age))
    all_groups = explicit_depth_position_groups(cfg.node_count)
    selected_groups = {
        name: all_groups[name] for name in ("graph", "query_metadata", "answer")
    }
    branch = branch_nodes(loop_indices=loop_indices, n_layers=cfg.n_layers)
    candidates = (
        head_context_nodes(
            loop_indices=loop_indices,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            position_groups=selected_groups,
        )
        + mlp_group_nodes(
            loop_indices=loop_indices,
            n_layers=cfg.n_layers,
            position_groups=selected_groups,
        )
    )
    discovery_raw = discover(
        model=model,
        cfg=cfg,
        adjacent=adjacent,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        branch=branch,
        candidates=candidates,
        source_age=args.source_age,
        target_age=args.target_age,
        examples=args.discovery_examples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    branch_summary = aggregate_discovery(discovery_raw, family="branch")
    candidate_summary = aggregate_discovery(discovery_raw, family="candidate")
    selected_rows = select_circuit(candidate_summary, top_k=args.top_k)
    by_label = {node.label: node for node in candidates}
    selected_nodes = tuple(by_label[str(row["label"])] for row in selected_rows)
    validation_raw = validate_circuit(
        model=model,
        cfg=cfg,
        adjacent=adjacent,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        candidates=candidates,
        selected=selected_nodes,
        source_age=args.source_age,
        target_age=args.target_age,
        examples=args.validation_examples,
        batch_size=args.batch_size,
        random_subsets=args.random_subsets,
        seed=args.seed + 1,
    )
    validation_summary = aggregate_validation(validation_raw)
    mediation_raw = qkv_mediation(
        model=model,
        cfg=cfg,
        adjacent=adjacent,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        selected=attention_only(selected_nodes),
        source_age=args.source_age,
        target_age=args.target_age,
        examples=args.validation_examples,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )
    mediation_summary = aggregate_mediation(mediation_raw)
    _write_csv(args.out_dir / "discovery_raw.csv", discovery_raw)
    _write_csv(args.out_dir / "branch_discovery.csv", branch_summary)
    _write_csv(args.out_dir / "candidate_discovery.csv", candidate_summary)
    _write_csv(args.out_dir / "selected_circuit.csv", selected_rows)
    _write_csv(args.out_dir / "validation_raw.csv", validation_raw)
    _write_csv(args.out_dir / "validation_summary.csv", validation_summary)
    _write_csv(args.out_dir / "qkv_mediation_raw.csv", mediation_raw)
    _write_csv(args.out_dir / "qkv_mediation_summary.csv", mediation_summary)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "adjacent_bank": str(args.adjacent_bank),
        "adjacent_bank_architecture": adjacent_payload.get("architecture"),
        "clean_condition": "ordered J8,J7,J6,J5,J4 product applied to H8",
        "corrupt_condition": "shared D+U0+b applied five times to H8",
        "executor": "five frozen loops from logical age H3 through H8",
        "discovery_examples": args.discovery_examples,
        "validation_examples": args.validation_examples,
        "selection": "top-k attention-head-context or MLP-output nodes by min(patch-in recovery, patch-out damage) minus positive shuffled recovery",
        "selected_circuit": selected_rows,
        "validation_summary": validation_summary,
        "qkv_mediation_summary": mediation_summary,
        "circuit_decision": circuit_decision(validation_summary),
        "claim_boundary": "effective-loop role on seed0; no physical-parameter generality claim",
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
