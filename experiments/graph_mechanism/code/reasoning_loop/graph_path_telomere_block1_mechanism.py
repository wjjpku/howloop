from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_diag_rank48_circuit import (
    _metrics,
    _rolled_trace,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


@dataclass(frozen=True)
class InterventionSpec:
    label: str
    family: str
    interventions: tuple[FunctionalIntervention, ...]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _relative_change(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    numerator = (value.float() - reference.float()).flatten(1).norm(dim=-1)
    denominator = reference.float().flatten(1).norm(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def _directional_progress(
    base: torch.Tensor, donor: torch.Tensor, patched: torch.Tensor
) -> torch.Tensor:
    base_flat = base.float().flatten(1)
    delta = donor.float().flatten(1) - base_flat
    moved = patched.float().flatten(1) - base_flat
    return (moved * delta).sum(dim=-1) / delta.square().sum(dim=-1).clamp_min(1e-8)


def _preference(
    logits: torch.Tensor, *, preferred: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    return logits.gather(1, preferred[:, None]).squeeze(1) - logits.gather(
        1, reference[:, None]
    ).squeeze(1)


def _mean_cosine(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            value.float().flatten(1), reference.float().flatten(1), dim=-1
        ).mean()
    )


def _lookup_diagnostics(
    *,
    trace: FunctionalTrace,
    reference: FunctionalTrace,
    current: torch.Tensor,
    lookup_head: int,
) -> dict[str, float]:
    site = trace.sites[1]
    base = reference.sites[1]
    batch = torch.arange(current.shape[0], device=current.device)
    answer = site.q.shape[2] - 1
    current_destination = 3 + 3 * current
    destination_positions = 3 + 3 * torch.arange(
        int((answer - 3) / 3), device=current.device
    )
    pattern = site.attention_pattern[:, lookup_head]
    top_destination = pattern[:, answer, destination_positions].argmax(dim=-1)
    return {
        "B2_lookup_q_relative_change": float(
            _relative_change(
                site.q[:, lookup_head, answer],
                base.q[:, lookup_head, answer],
            ).mean()
        ),
        "B2_lookup_q_cosine_to_full": _mean_cosine(
            site.q[:, lookup_head, answer],
            base.q[:, lookup_head, answer],
        ),
        "B2_lookup_context_relative_change": float(
            _relative_change(
                site.head_context[:, lookup_head, answer],
                base.head_context[:, lookup_head, answer],
            ).mean()
        ),
        "B2_current_destination_attention": float(
            pattern[batch, answer, current_destination].mean()
        ),
        "B2_top_destination_is_current": float(
            top_destination.eq(current).float().mean()
        ),
    }


def _position_groups(cfg) -> dict[str, tuple[int, ...] | None]:
    groups = explicit_depth_position_groups(cfg.node_count)
    return {
        "answer": groups["answer"],
        "graph": groups["graph"],
        "edge_marker": groups["edge_marker"],
        "source": groups["source"],
        "destination": groups["destination"],
        "metadata": groups["query_metadata"],
        "query": groups["query"],
        "start": groups["start"],
        "depth": groups["depth"],
        "all": None,
    }


def _bypass_specs(cfg) -> list[InterventionSpec]:
    result: list[InterventionSpec] = []
    for group, positions in _position_groups(cfg).items():
        attention = FunctionalIntervention(
            site=0, component="attention_out", mode="zero", positions=positions
        )
        mlp = FunctionalIntervention(
            site=0, component="mlp_out", mode="zero", positions=positions
        )
        result.extend(
            (
                InterventionSpec(
                    f"B1.attention_update_zero_{group}", "bypass", (attention,)
                ),
                InterventionSpec(
                    f"B1.mlp_update_zero_{group}", "bypass", (mlp,)
                ),
                InterventionSpec(
                    f"B1.full_update_zero_{group}",
                    "bypass",
                    (attention, mlp),
                ),
            )
        )
    for head in range(cfg.n_heads):
        for group, positions in _position_groups(cfg).items():
            result.append(
                InterventionSpec(
                    f"B1.H{head}.context_zero_{group}",
                    "head_context",
                    (
                        FunctionalIntervention(
                            site=0,
                            component="head_context",
                            mode="zero",
                            positions=positions,
                            heads=(head,),
                        ),
                    ),
                )
            )
    return result


def _semantic_edge_specs(
    *, cfg, current: torch.Tensor
) -> list[InterventionSpec]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    dynamic = {
        "current_source": (2 + 3 * current)[:, None],
        "current_destination": (3 + 3 * current)[:, None],
        "matched_unrelated_destination": (
            3 + 3 * ((current + 3) % cfg.node_count)
        )[:, None],
    }
    fixed = {
        "answer_self": groups["answer"],
        "start": groups["start"],
        "depth": groups["depth"],
        "query": groups["query"],
        "all_graph": groups["graph"],
    }
    result: list[InterventionSpec] = []
    for head in range(cfg.n_heads):
        for role, sources in fixed.items():
            result.append(
                InterventionSpec(
                    f"B1.H{head}.edge_zero_{role}",
                    "semantic_edge",
                    (
                        FunctionalIntervention(
                            site=0,
                            component="attention_pattern",
                            mode="zero",
                            positions=answer,
                            source_positions=sources,
                            heads=(head,),
                            renormalize=True,
                        ),
                    ),
                )
            )
        for role, sources in dynamic.items():
            result.append(
                InterventionSpec(
                    f"B1.H{head}.edge_zero_{role}",
                    "semantic_edge",
                    (
                        FunctionalIntervention(
                            site=0,
                            component="attention_pattern",
                            mode="zero",
                            positions=answer,
                            dynamic_source_positions=sources,
                            heads=(head,),
                            renormalize=True,
                        ),
                    ),
                )
            )
    return result


def _graph_record_edge_specs(cfg) -> list[InterventionSpec]:
    """Ablate query-specific B1 routes used to assemble each edge record.

    Each graph edge is serialized as ``[EDGE, source, destination]``.  The
    destination position is the position queried by the Block2 lookup head,
    so these interventions distinguish copying the paired source, retaining
    the destination itself, and using the edge marker.
    """

    groups = explicit_depth_position_groups(cfg.node_count)
    relations = {
        "record_paired_edge_marker": tuple(
            zip(groups["destination"], groups["edge_marker"], strict=True)
        ),
        "record_paired_source": tuple(
            zip(groups["destination"], groups["source"], strict=True)
        ),
        "record_destination_self": tuple(
            zip(groups["destination"], groups["destination"], strict=True)
        ),
    }
    result: list[InterventionSpec] = []
    for head in range(cfg.n_heads):
        for role, pairs in relations.items():
            result.append(
                InterventionSpec(
                    f"B1.H{head}.edge_zero_{role}",
                    "graph_record_edge",
                    tuple(
                        FunctionalIntervention(
                            site=0,
                            component="attention_pattern",
                            mode="zero",
                            positions=(query_position,),
                            source_positions=(source_position,),
                            heads=(head,),
                            renormalize=True,
                        )
                        for query_position, source_position in pairs
                    ),
                )
            )
    return result


def _interchange_specs(cfg, *, lookup_head: int) -> list[InterventionSpec]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    result = [
        InterventionSpec(
            "B1.input_answer",
            "current_interchange",
            (
                FunctionalIntervention(
                    site=0, component="block_input", mode="patch", positions=answer
                ),
            ),
        ),
        InterventionSpec(
            "B1.input_graph",
            "current_interchange_control",
            (
                FunctionalIntervention(
                    site=0, component="block_input", mode="patch", positions=graph
                ),
            ),
        ),
        InterventionSpec(
            "B1.post_attention_answer",
            "current_interchange",
            (
                FunctionalIntervention(
                    site=0, component="residual_mid", mode="patch", positions=answer
                ),
            ),
        ),
        InterventionSpec(
            "B1.attention_update_answer",
            "current_interchange_component",
            (
                FunctionalIntervention(
                    site=0, component="attention_out", mode="patch", positions=answer
                ),
            ),
        ),
        InterventionSpec(
            "B1.mlp_update_answer",
            "current_interchange_component",
            (
                FunctionalIntervention(
                    site=0, component="mlp_out", mode="patch", positions=answer
                ),
            ),
        ),
        InterventionSpec(
            "B2.input_answer",
            "current_interchange",
            (
                FunctionalIntervention(
                    site=1, component="block_input", mode="patch", positions=answer
                ),
            ),
        ),
        InterventionSpec(
            "B2.input_graph",
            "current_interchange_control",
            (
                FunctionalIntervention(
                    site=1, component="block_input", mode="patch", positions=graph
                ),
            ),
        ),
        InterventionSpec(
            f"B2.H{lookup_head}.q_answer",
            "lookup_positive_control",
            (
                FunctionalIntervention(
                    site=1,
                    component="q",
                    mode="patch",
                    positions=answer,
                    heads=(lookup_head,),
                ),
            ),
        ),
        InterventionSpec(
            f"B2.H{lookup_head}.pattern_answer_graph",
            "lookup_positive_control",
            (
                FunctionalIntervention(
                    site=1,
                    component="attention_pattern",
                    mode="patch",
                    positions=answer,
                    source_positions=graph,
                    heads=(lookup_head,),
                    renormalize=True,
                ),
            ),
        ),
        InterventionSpec(
            f"B2.H{lookup_head}.context_answer",
            "lookup_positive_control",
            (
                FunctionalIntervention(
                    site=1,
                    component="head_context",
                    mode="patch",
                    positions=answer,
                    heads=(lookup_head,),
                ),
            ),
        ),
    ]
    for head in range(cfg.n_heads):
        result.append(
            InterventionSpec(
                f"B1.H{head}.context_answer",
                "current_interchange_component",
                (
                    FunctionalIntervention(
                        site=0,
                        component="head_context",
                        mode="patch",
                        positions=answer,
                        heads=(head,),
                    ),
                ),
            )
        )
    return result


def _semantic_mass(
    *, trace: FunctionalTrace, spec: InterventionSpec
) -> float | None:
    if not spec.interventions or any(
        item.component != "attention_pattern" for item in spec.interventions
    ):
        return None
    pattern = trace.sites[0].attention_pattern
    selected: list[torch.Tensor] = []
    for item in spec.interventions:
        heads = item.heads or tuple(range(pattern.shape[1]))
        positions = item.positions or tuple(range(pattern.shape[2]))
        if item.dynamic_source_positions is not None:
            sources = item.dynamic_source_positions.to(pattern.device)
            batch = torch.arange(pattern.shape[0], device=pattern.device)
            for head in heads:
                for position in positions:
                    for column in range(sources.shape[1]):
                        selected.append(
                            pattern[batch, head, position, sources[:, column]]
                        )
        else:
            sources = item.source_positions or tuple(range(pattern.shape[3]))
            for head in heads:
                for position in positions:
                    selected.append(
                        pattern[:, head, position, list(sources)].sum(dim=-1)
                    )
    return float(torch.stack(selected).mean())


def _append_per_example(
    rows: list[dict[str, Any]],
    *,
    cycle: int,
    condition: str,
    family: str,
    logits: torch.Tensor,
    trace: FunctionalTrace,
    successors: torch.Tensor,
    current: torch.Tensor,
    alternate_current: torch.Tensor,
    target: torch.Tensor,
    alternate_target: torch.Tensor,
    lookup_head: int,
) -> None:
    prediction = logits.argmax(dim=-1)
    margin = target_margin(logits.float(), target)
    alternate_margin = target_margin(logits.float(), alternate_target)
    batch = torch.arange(logits.shape[0], device=logits.device)
    answer = trace.sites[1].attention_pattern.shape[2] - 1
    current_destination = 3 + 3 * current
    alternate_destination = 3 + 3 * alternate_current
    pattern = trace.sites[1].attention_pattern[:, lookup_head]
    for sample in range(logits.shape[0]):
        rows.append(
            {
                "cycle": cycle,
                "sample": sample,
                "condition": condition,
                "family": family,
                "successors": " ".join(
                    str(int(value)) for value in successors[sample].tolist()
                ),
                "current": int(current[sample]),
                "alternate_current": int(alternate_current[sample]),
                "target": int(target[sample]),
                "alternate_target": int(alternate_target[sample]),
                "prediction": int(prediction[sample]),
                "correct": int(prediction[sample].eq(target[sample])),
                "switched_to_alternate": int(
                    prediction[sample].eq(alternate_target[sample])
                ),
                "target_margin": float(margin[sample]),
                "alternate_target_margin": float(alternate_margin[sample]),
                "B2_current_destination_attention": float(
                    pattern[sample, answer, current_destination[sample]]
                ),
                "B2_alternate_destination_attention": float(
                    pattern[sample, answer, alternate_destination[sample]]
                ),
            }
        )


def _component_geometry_rows(
    *, cycle: int, trace: FunctionalTrace, cfg
) -> list[dict[str, Any]]:
    site = trace.sites[0]
    result: list[dict[str, Any]] = []
    for group, positions in _position_groups(cfg).items():
        if positions is None:
            index = tuple(range(site.hidden_in.shape[1]))
        else:
            index = positions
        hidden = site.hidden_in[:, index]
        attention = site.attention_out[:, index]
        mlp = site.mlp_out[:, index]
        total = site.hidden_out[:, index] - hidden
        result.append(
            {
                "cycle": cycle,
                "component": "B1.update",
                "position_group": group,
                "input_norm": float(hidden.float().norm(dim=-1).mean()),
                "attention_update_norm": float(attention.float().norm(dim=-1).mean()),
                "mlp_update_norm": float(mlp.float().norm(dim=-1).mean()),
                "total_update_norm": float(total.float().norm(dim=-1).mean()),
                "attention_to_input_ratio": float(
                    attention.float().norm(dim=-1).mean()
                    / hidden.float().norm(dim=-1).mean().clamp_min(1e-8)
                ),
                "mlp_to_input_ratio": float(
                    mlp.float().norm(dim=-1).mean()
                    / hidden.float().norm(dim=-1).mean().clamp_min(1e-8)
                ),
            }
        )
    answer = cfg.seq_len - 1
    for head in range(cfg.n_heads):
        result.append(
            {
                "cycle": cycle,
                "component": f"B1.H{head}.context",
                "position_group": "answer",
                "context_norm": float(
                    site.head_context[:, head, answer].float().norm(dim=-1).mean()
                ),
                "answer_self_attention": float(
                    site.attention_pattern[:, head, answer, answer].mean()
                ),
            }
        )
    return result


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "checkpoint": str(args.checkpoint),
        "trajectory_mode": args.trajectory_mode,
        "physical_gpu": args.physical_gpu,
        "prelaunch_used_mib": args.prelaunch_used_mib,
        "prelaunch_free_mib": args.prelaunch_free_mib,
        "declared_peak_gib": args.declared_peak_gib,
        "reserve_gib": args.reserve_gib,
        "shared_gpu": args.shared_gpu,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, operators, operator_payload = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if operator_payload.get("placement") != "loop_boundary":
        raise ValueError("Block1 mechanism audit requires loop-boundary J")
    operator = operators[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ) or operator.rank != 48:
        raise ValueError("Block1 mechanism audit requires diagonal rank-48 J")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("J must act on every token position")
    set_seed(args.seed)
    tokens, path_targets, successors, start = fixed_depth_batch(
        cfg, args.batch_size, device, path_positions=cfg.max_depth
    )
    if args.trajectory_mode == "controlled_continuation":
        phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
        phase_positions = [
            int(value)
            for value in phase["trajectory_positions_including_initial"]
        ]
        jump = phase_positions[3] - phase_positions[2]
        if jump != 1:
            raise ValueError("this Block1 audit is preregistered for one-hop loops")
        endpoint = path_targets[:, cfg.max_depth - 1]
        alternate_endpoint = (endpoint + 1) % cfg.node_count
        trajectory = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        alternate_trajectory = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=alternate_endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
    elif args.trajectory_mode == "native_prefix":
        endpoint = start
        alternate_endpoint = (start + 1) % cfg.node_count
        alternate_tokens, _, _, _ = fixed_depth_batch(
            cfg,
            args.batch_size,
            device,
            path_positions=cfg.max_depth,
            successors=successors,
            start=alternate_endpoint,
        )
        trajectory = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        alternate_trajectory = (
            model.token_embed(alternate_tokens) + model.pos_embed.unsqueeze(0)
        )
        if max(args.cycles) > cfg.max_loops:
            raise ValueError("native-prefix cycles cannot exceed trained loops")
    else:
        raise ValueError(f"unknown trajectory mode: {args.trajectory_mode}")

    requested = set(args.cycles)
    baseline_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    interchange_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    bypass_specs = _bypass_specs(cfg)
    interchange_specs = _interchange_specs(cfg, lookup_head=args.lookup_head)

    for cycle in range(1, max(requested) + 1):
        current = advance_nodes(successors, endpoint, steps=cycle - 1)
        target = advance_nodes(successors, endpoint, steps=cycle)
        alternate_current = advance_nodes(
            successors, alternate_endpoint, steps=cycle - 1
        )
        alternate_target = advance_nodes(successors, alternate_endpoint, steps=cycle)
        loop_index = (
            cfg.max_loops + cycle - 1
            if args.trajectory_mode == "controlled_continuation"
            else cycle - 1
        )
        if cycle in requested:
            if args.trajectory_mode == "controlled_continuation":
                controlled = operator(trajectory)
                alternate_controlled = operator(alternate_trajectory)
            else:
                controlled = trajectory
                alternate_controlled = alternate_trajectory
            logits, trace = run_instrumented_state(
                model, controlled, loop_indices=(loop_index,)
            )
            alternate_logits, alternate_trace = run_instrumented_state(
                model, alternate_controlled, loop_indices=(loop_index,)
            )
            baseline_rows.extend(
                (
                    {
                        "cycle": cycle,
                        "run": "base_current",
                        **_metrics(logits, target),
                    },
                    {
                        "cycle": cycle,
                        "run": "alternate_current",
                        **_metrics(alternate_logits, alternate_target),
                    },
                )
            )
            _append_per_example(
                per_example_rows,
                cycle=cycle,
                condition="full",
                family="baseline",
                logits=logits,
                trace=trace,
                successors=successors,
                current=current,
                alternate_current=alternate_current,
                target=target,
                alternate_target=alternate_target,
                lookup_head=args.lookup_head,
            )
            stages = (
                ("loop_input", controlled),
                ("B1_post_attention", trace.sites[0].residual_mid),
                ("B1_post_mlp", trace.sites[0].hidden_out),
                ("B2_post_attention", trace.sites[1].residual_mid),
                ("B2_post_mlp", trace.sites[1].hidden_out),
            )
            for stage, state in stages:
                readout = logits_from_raw_state(model, state)
                stage_rows.append(
                    {
                        "cycle": cycle,
                        "stage": stage,
                        "current_accuracy": float(
                            readout.argmax(dim=-1).eq(current).float().mean()
                        ),
                        "current_margin": float(
                            target_margin(readout.float(), current).mean()
                        ),
                        "successor_accuracy": float(
                            readout.argmax(dim=-1).eq(target).float().mean()
                        ),
                        "successor_margin": float(
                            target_margin(readout.float(), target).mean()
                        ),
                    }
                )
            geometry_rows.extend(
                _component_geometry_rows(cycle=cycle, trace=trace, cfg=cfg)
            )

            semantic_specs = _semantic_edge_specs(
                cfg=cfg, current=current
            ) + _graph_record_edge_specs(cfg)
            for spec in bypass_specs + semantic_specs:
                patched_logits, patched_trace = run_instrumented_state(
                    model,
                    controlled,
                    loop_indices=(loop_index,),
                    interventions=spec.interventions,
                )
                intervention_rows.append(
                    {
                        "cycle": cycle,
                        "component": spec.label,
                        "family": spec.family,
                        "removed_attention_mass": _semantic_mass(
                            trace=trace, spec=spec
                        ),
                        **_metrics(patched_logits, target),
                        **_lookup_diagnostics(
                            trace=patched_trace,
                            reference=trace,
                            current=current,
                            lookup_head=args.lookup_head,
                        ),
                    }
                )
                _append_per_example(
                    per_example_rows,
                    cycle=cycle,
                    condition=spec.label,
                    family=spec.family,
                    logits=patched_logits,
                    trace=patched_trace,
                    successors=successors,
                    current=current,
                    alternate_current=alternate_current,
                    target=target,
                    alternate_target=alternate_target,
                    lookup_head=args.lookup_head,
                )

            base_preference = _preference(
                logits, preferred=alternate_target, reference=target
            )
            alternate_preference = _preference(
                alternate_logits, preferred=alternate_target, reference=target
            )
            shuffled_trace = _rolled_trace(alternate_trace)
            for spec in interchange_specs:
                for donor_name, donor in (
                    ("matched_same_graph", alternate_trace),
                    ("batch_shuffled", shuffled_trace),
                ):
                    patched_logits, patched_trace = run_instrumented_state(
                        model,
                        controlled,
                        loop_indices=(loop_index,),
                        interventions=spec.interventions,
                        donor_trace=donor,
                    )
                    patched_preference = _preference(
                        patched_logits,
                        preferred=alternate_target,
                        reference=target,
                    )
                    preference_denominator = alternate_preference - base_preference
                    preference_progress = torch.where(
                        preference_denominator.abs().gt(1e-8),
                        (patched_preference - base_preference)
                        / preference_denominator,
                        torch.full_like(preference_denominator, float("nan")),
                    )
                    interchange_rows.append(
                        {
                            "cycle": cycle,
                            "component": spec.label,
                            "family": spec.family,
                            "donor": donor_name,
                            "base_target_accuracy": float(
                                patched_logits.argmax(dim=-1).eq(target).float().mean()
                            ),
                            "alternate_target_accuracy": float(
                                patched_logits.argmax(dim=-1)
                                .eq(alternate_target)
                                .float()
                                .mean()
                            ),
                            "base_target_margin": float(
                                target_margin(patched_logits.float(), target).mean()
                            ),
                            "alternate_target_margin": float(
                                target_margin(
                                    patched_logits.float(), alternate_target
                                ).mean()
                            ),
                            "alternate_preference_progress": float(
                                preference_progress.mean()
                            ),
                            "B2_lookup_q_progress_to_alternate": float(
                                _directional_progress(
                                    trace.sites[1].q[:, args.lookup_head, -1],
                                    alternate_trace.sites[1].q[
                                        :, args.lookup_head, -1
                                    ],
                                    patched_trace.sites[1].q[
                                        :, args.lookup_head, -1
                                    ],
                                ).mean()
                            ),
                            "B2_lookup_context_progress_to_alternate": float(
                                _directional_progress(
                                    trace.sites[1].head_context[
                                        :, args.lookup_head, -1
                                    ],
                                    alternate_trace.sites[1].head_context[
                                        :, args.lookup_head, -1
                                    ],
                                    patched_trace.sites[1].head_context[
                                        :, args.lookup_head, -1
                                    ],
                                ).mean()
                            ),
                            **_lookup_diagnostics(
                                trace=patched_trace,
                                reference=trace,
                                current=current,
                                lookup_head=args.lookup_head,
                            ),
                        }
                    )
                    _append_per_example(
                        per_example_rows,
                        cycle=cycle,
                        condition=f"{spec.label}:{donor_name}",
                        family=spec.family,
                        logits=patched_logits,
                        trace=patched_trace,
                        successors=successors,
                        current=current,
                        alternate_current=alternate_current,
                        target=target,
                        alternate_target=alternate_target,
                        lookup_head=args.lookup_head,
                    )

        if args.trajectory_mode == "controlled_continuation":
            trajectory = _controlled_loop(
                loop_runner=run_one_loop,
                model=model,
                state=trajectory,
                loop_index=loop_index,
                positions=positions,
                operator=operator,
                placement="loop_boundary",
            ).state
            alternate_trajectory = _controlled_loop(
                loop_runner=run_one_loop,
                model=model,
                state=alternate_trajectory,
                loop_index=loop_index,
                positions=positions,
                operator=operator,
                placement="loop_boundary",
            ).state
        else:
            trajectory = run_one_loop(
                model, trajectory, loop_index=loop_index
            ).state
            alternate_trajectory = run_one_loop(
                model, alternate_trajectory, loop_index=loop_index
            ).state

    _write(args.out_dir / "baseline.csv", baseline_rows)
    _write(args.out_dir / "stage_readout.csv", stage_rows)
    _write(args.out_dir / "bypass_and_semantic_edges.csv", intervention_rows)
    _write(args.out_dir / "current_interchange.csv", interchange_rows)
    _write(args.out_dir / "component_geometry.csv", geometry_rows)
    _write(args.out_dir / "per_example.csv", per_example_rows)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "trajectory_mode": args.trajectory_mode,
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller": (
            "J(h)=h*D+(hA)B+b"
            if args.trajectory_mode == "controlled_continuation"
            else "not applied in native prefix"
        ),
        "controller_rank": (
            operator.rank
            if args.trajectory_mode == "controlled_continuation"
            else None
        ),
        "controller_parameters": (
            operator.parameter_count
            if args.trajectory_mode == "controlled_continuation"
            else None
        ),
        "controller_placement": (
            "loop boundary after Block2 FFN"
            if args.trajectory_mode == "controlled_continuation"
            else "not applied in native prefix"
        ),
        "controller_training_loss": (
            "successor CE at every controlled continuation loop; no hidden MSE"
            if args.trajectory_mode == "controlled_continuation"
            else "not applicable; J was not executed"
        ),
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "examples": args.batch_size,
        "data_seed": args.seed,
        "cycles": sorted(requested),
        "lookup_head": args.lookup_head,
        "raw_per_example_rows": len(per_example_rows),
        "gpu_peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu_peak_allocated_gib": result["gpu_peak_allocated_gib"],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block1 mechanism and current-interface causal audit."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--lookup-head", type=int, required=True)
    parser.add_argument(
        "--trajectory-mode",
        choices=("controlled_continuation", "native_prefix"),
        default="controlled_continuation",
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.05)
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--prelaunch-used-mib", type=int, default=None)
    parser.add_argument("--prelaunch-free-mib", type=int, default=None)
    parser.add_argument("--declared-peak-gib", type=float, default=None)
    parser.add_argument("--reserve-gib", type=float, default=None)
    parser.add_argument(
        "--shared-gpu", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run_experiment(parse_args()), indent=2, sort_keys=True))
