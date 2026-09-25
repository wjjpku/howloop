from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
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
    FunctionalSiteTrace,
    FunctionalTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_rejuvenation_circuit import (
    MatchedAgeBatch,
    attention_function_rows,
    behavior_metrics,
    normalized_component_metrics,
)
from reasoning_loop.graph_path_rejuvenation_multimodel import (
    AffineMap,
    apply_affine_to_state,
)
from reasoning_loop.graph_path_telomere_overloop import (
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_telomere_rejuvenator import (
    FlattenedAffine,
    PositionwiseAffine,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


@dataclass(frozen=True)
class Seed1Trajectory:
    terminal: torch.Tensor
    young1: torch.Tensor
    young2: torch.Tensor
    j1_pre: torch.Tensor
    post1: torch.Tensor
    j2_pre_bad: torch.Tensor
    oracle1: torch.Tensor
    oracle2: torch.Tensor
    successors: torch.Tensor
    current0: torch.Tensor
    target1: torch.Tensor
    target2: torch.Tensor
    aligned_second_ages: dict[int, torch.Tensor]

    def first_batch(self) -> MatchedAgeBatch:
        return MatchedAgeBatch(
            terminal=self.terminal,
            young=self.young1,
            successors=self.successors,
            current=self.current0,
            target=self.target1,
            endpoint=self.current0,
        )

    def second_batch(self) -> MatchedAgeBatch:
        return MatchedAgeBatch(
            terminal=self.j2_pre_bad,
            young=self.young2,
            successors=self.successors,
            current=self.target1,
            target=self.target2,
            endpoint=self.target1,
        )


@dataclass(frozen=True)
class PatchNode:
    label: str
    site: int
    component: str
    head: int | None = None
    pattern_scope: str | None = None

    def intervention(
        self,
        *,
        cfg: GraphPathConfig,
        current: torch.Tensor,
        mode: str = "patch",
    ) -> FunctionalIntervention:
        answer = cfg.seq_len - 1
        if self.component == "attention_pattern":
            fields: dict[str, Any] = {
                "positions": (answer,),
                "heads": None if self.head is None else (self.head,),
                "renormalize": True,
            }
            if self.pattern_scope == "current_destination":
                fields["dynamic_source_positions"] = (
                    3 + 3 * current
                ).unsqueeze(1)
            elif self.pattern_scope == "graph":
                fields["source_positions"] = explicit_depth_position_groups(
                    cfg.node_count
                )["graph"]
            else:
                raise ValueError("attention pattern nodes need a valid scope")
            return FunctionalIntervention(
                site=self.site,
                component="attention_pattern",
                mode=mode,  # type: ignore[arg-type]
                **fields,
            )
        return FunctionalIntervention(
            site=self.site,
            component=self.component,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            positions=(answer,),
            heads=None if self.head is None else (self.head,),
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_saved_affine(
    path: Path,
    *,
    device: torch.device,
) -> tuple[AffineMap, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    raw = payload["affine"]
    base = PositionwiseAffine(
        weight=raw["weight"].to(device=device, dtype=torch.float32),
        bias=raw["bias"].to(device=device, dtype=torch.float32),
    )
    if raw["structure"] == "flattened_affine":
        affine: AffineMap = FlattenedAffine(
            affine=base,
            position_count=int(raw["position_count"]),
            feature_count=int(raw["feature_count"]),
        )
    elif raw["structure"] == "positionwise_affine":
        affine = base
    else:
        raise ValueError(f"unknown affine structure: {raw['structure']}")
    return affine, payload


@torch.no_grad()
def collect_seed1_trajectory(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    affine: AffineMap,
    positions: Sequence[int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Seed1Trajectory:
    reference_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    if reference_age != 2 or jump != 2:
        raise ValueError("this targeted experiment requires the seed1 age2/jump2 candidate")
    set_seed(seed)
    tokens, targets, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth + 2 * jump,
    )
    terminal = cache_states_with_initial(
        model,
        tokens,
        loops=cfg.max_loops,
    )[-1]
    reference_start = advance_nodes(
        successors,
        start,
        steps=cfg.max_depth - reference_position,
    )
    next_reference_start = advance_nodes(
        successors,
        reference_start,
        steps=jump,
    )
    reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=reference_start,
    )
    next_reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=next_reference_start,
    )
    young1 = cache_states_with_initial(
        model,
        reference_tokens,
        loops=reference_age,
    )[reference_age]
    young2 = cache_states_with_initial(
        model,
        next_reference_tokens,
        loops=reference_age,
    )[reference_age]
    j1_pre = apply_affine_to_state(
        terminal,
        positions=positions,
        affine=affine,
    )
    post1 = apply_shared_stack(
        model,
        j1_pre,
        loop_index=cfg.max_loops,
    )
    j2_pre_bad = apply_affine_to_state(
        post1,
        positions=positions,
        affine=affine,
    )
    oracle1 = terminal.clone()
    oracle1[:, list(positions)] = young1[:, list(positions)]
    oracle2 = j2_pre_bad.clone()
    oracle2[:, list(positions)] = young2[:, list(positions)]

    # For the second desired current f^10(start), age a must start at
    # f^(10 - 2a)(start). Ages 0..5 therefore need no inverse permutation.
    aligned_second_ages: dict[int, torch.Tensor] = {}
    desired_position = cfg.max_depth + jump
    for age in range(0, desired_position // jump + 1):
        aligned_start = advance_nodes(
            successors,
            start,
            steps=desired_position - jump * age,
        )
        aligned_tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
            successors=successors,
            start=aligned_start,
        )
        aligned_second_ages[age] = cache_states_with_initial(
            model,
            aligned_tokens,
            loops=max(1, age),
        )[age]

    return Seed1Trajectory(
        terminal=terminal,
        young1=young1,
        young2=young2,
        j1_pre=j1_pre,
        post1=post1,
        j2_pre_bad=j2_pre_bad,
        oracle1=oracle1,
        oracle2=oracle2,
        successors=successors,
        current0=targets[:, cfg.max_depth - 1],
        target1=targets[:, cfg.max_depth + jump - 1],
        target2=targets[:, cfg.max_depth + 2 * jump - 1],
        aligned_second_ages=aligned_second_ages,
    )


def _metric_row(
    logits: torch.Tensor,
    *,
    batch: MatchedAgeBatch,
    **labels: Any,
) -> dict[str, Any]:
    return {
        **labels,
        **behavior_metrics(
            logits,
            target=batch.target,
            endpoint=batch.endpoint,
        ),
    }


@torch.no_grad()
def stage_rows(
    *,
    model: LoopedGraphPathTransformer,
    batch: MatchedAgeBatch,
    initial_states: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
) -> list[dict[str, Any]]:
    rows = []
    for run, trace in traces.items():
        site1, site2 = trace.sites
        stages = (
            ("input", initial_states[run]),
            ("B1_post_attention", site1.residual_mid),
            ("B1_post_mlp", site1.hidden_out),
            ("B2_post_attention", site2.residual_mid),
            ("B2_post_mlp", site2.hidden_out),
        )
        for stage, state in stages:
            rows.append(
                _metric_row(
                    logits_from_raw_state(model, state),
                    batch=batch,
                    run=run,
                    stage=stage,
                )
            )
    return rows


def _rolled_trace(trace: FunctionalTrace) -> FunctionalTrace:
    def roll(site: FunctionalSiteTrace) -> FunctionalSiteTrace:
        return FunctionalSiteTrace(
            loop_index=site.loop_index,
            block_index=site.block_index,
            hidden_in=site.hidden_in.roll(1, dims=0),
            q=site.q.roll(1, dims=0),
            k=site.k.roll(1, dims=0),
            v=site.v.roll(1, dims=0),
            attention_pattern=site.attention_pattern.roll(1, dims=0),
            head_context=site.head_context.roll(1, dims=0),
            attention_out=site.attention_out.roll(1, dims=0),
            residual_mid=site.residual_mid.roll(1, dims=0),
            mlp_hidden=site.mlp_hidden.roll(1, dims=0),
            mlp_out=site.mlp_out.roll(1, dims=0),
            hidden_out=site.hidden_out.roll(1, dims=0),
        )

    return FunctionalTrace(
        sites=[roll(site) for site in trace.sites],
        logits_by_loop=trace.logits_by_loop.roll(1, dims=0),
    )


def seed1_path_pairs() -> tuple[tuple[PatchNode, PatchNode], ...]:
    b1_h1 = PatchNode("B1.H1.context", 0, "head_context", head=1)
    b1_attn = PatchNode("B1.attention_out", 0, "attention_out")
    b1_mlp = PatchNode("B1.MLP.out", 0, "mlp_out")
    b2_h2_q = PatchNode("B2.H2.q", 1, "q", head=2)
    b2_h2_graph = PatchNode(
        "B2.H2.pattern_graph",
        1,
        "attention_pattern",
        head=2,
        pattern_scope="graph",
    )
    b2_h2_edge = PatchNode(
        "B2.H2.pattern_current_destination",
        1,
        "attention_pattern",
        head=2,
        pattern_scope="current_destination",
    )
    b2_h2_context = PatchNode("B2.H2.context", 1, "head_context", head=2)
    b2_input = PatchNode("B2.block_input", 1, "block_input")
    b2_attn = PatchNode("B2.attention_out", 1, "attention_out")
    b2_residual = PatchNode("B2.residual_mid", 1, "residual_mid")
    b2_mlp = PatchNode("B2.MLP.out", 1, "mlp_out")
    return (
        (b1_h1, b2_h2_q),
        (b1_attn, b2_h2_q),
        (b1_mlp, b2_h2_q),
        (
            b1_mlp,
            PatchNode("B2.H0.q", 1, "q", head=0),
        ),
        (
            b1_mlp,
            PatchNode("B2.H1.q", 1, "q", head=1),
        ),
        (
            b1_mlp,
            PatchNode("B2.H3.q", 1, "q", head=3),
        ),
        (
            b1_mlp,
            PatchNode("B2.H0.context", 1, "head_context", head=0),
        ),
        (
            b1_mlp,
            PatchNode("B2.H1.context", 1, "head_context", head=1),
        ),
        (b1_mlp, b2_h2_context),
        (
            b1_mlp,
            PatchNode("B2.H3.context", 1, "head_context", head=3),
        ),
        (b1_mlp, b2_input),
        (b1_mlp, b2_attn),
        (b1_mlp, b2_residual),
        (b1_mlp, b2_mlp),
        (b2_h2_q, b2_h2_graph),
        (b2_h2_q, b2_h2_edge),
        (b2_h2_graph, b2_h2_context),
        (b2_h2_edge, b2_h2_context),
        (b2_h2_context, b2_residual),
        (b2_h2_context, b2_mlp),
        (PatchNode("B2.H0.context", 1, "head_context", head=0), b2_mlp),
        (PatchNode("B2.H1.context", 1, "head_context", head=1), b2_mlp),
    )


def second_rescue_nodes() -> tuple[PatchNode, ...]:
    return (
        PatchNode("B1.H1.context", 0, "head_context", head=1),
        PatchNode("B1.attention_out", 0, "attention_out"),
        PatchNode("B1.MLP.out", 0, "mlp_out"),
        PatchNode("B2.block_input", 1, "block_input"),
        PatchNode("B2.H0.q", 1, "q", head=0),
        PatchNode("B2.H1.q", 1, "q", head=1),
        PatchNode("B2.H0.context", 1, "head_context", head=0),
        PatchNode("B2.H1.context", 1, "head_context", head=1),
        PatchNode("B2.H2.q", 1, "q", head=2),
        PatchNode("B2.H3.q", 1, "q", head=3),
        PatchNode(
            "B2.H2.pattern_graph",
            1,
            "attention_pattern",
            head=2,
            pattern_scope="graph",
        ),
        PatchNode(
            "B2.H2.pattern_current_destination",
            1,
            "attention_pattern",
            head=2,
            pattern_scope="current_destination",
        ),
        PatchNode("B2.H2.context", 1, "head_context", head=2),
        PatchNode("B2.H3.context", 1, "head_context", head=3),
        PatchNode("B2.attention_out", 1, "attention_out"),
        PatchNode("B2.residual_mid", 1, "residual_mid"),
        PatchNode("B2.MLP.out", 1, "mlp_out"),
    )


def target_margin_rescue_metrics(
    *,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    target: torch.Tensor,
    endpoint: torch.Tensor,
) -> dict[str, float]:
    basic = behavior_metrics(
        patched_logits,
        target=target,
        endpoint=endpoint,
    )
    clean = target_margin(clean_logits, target)
    corrupt = target_margin(corrupt_logits, target)
    patched = target_margin(patched_logits, target)
    denominator = clean - corrupt
    valid = (
        target.ne(endpoint)
        & clean_logits.argmax(dim=-1).eq(target)
        & denominator.abs().gt(1e-6)
    )
    recovery = (patched - corrupt) / denominator
    basic["controlled_valid_count"] = int(valid.sum())
    basic["target_margin_recovery"] = (
        float(recovery[valid].mean())
        if bool(valid.any())
        else float("nan")
    )
    return basic


@torch.no_grad()
def first_use_path_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    clean_state: torch.Tensor,
    corrupt_state: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, FunctionalTrace]]:
    corrupt_logits, corrupt_trace = run_instrumented_state(
        model,
        corrupt_state,
        loop_indices=(cfg.max_loops,),
    )
    clean_logits, clean_trace = run_instrumented_state(
        model,
        clean_state,
        loop_indices=(cfg.max_loops,),
    )
    rows = []
    for sender, receiver in seed1_path_pairs():
        sender_logits, sender_corrupt_trace = run_instrumented_state(
            model,
            clean_state,
            loop_indices=(cfg.max_loops,),
            interventions=(
                sender.intervention(cfg=cfg, current=batch.current),
            ),
            donor_trace=corrupt_trace,
        )
        mediated_logits, _ = run_instrumented_state(
            model,
            clean_state,
            loop_indices=(cfg.max_loops,),
            interventions=(
                receiver.intervention(cfg=cfg, current=batch.current),
            ),
            donor_trace=sender_corrupt_trace,
        )
        direct_receiver_logits, _ = run_instrumented_state(
            model,
            clean_state,
            loop_indices=(cfg.max_loops,),
            interventions=(
                receiver.intervention(cfg=cfg, current=batch.current),
            ),
            donor_trace=corrupt_trace,
        )
        for condition, logits in (
            ("sender_total_patchout", sender_logits),
            ("sender_to_receiver_path_patchout", mediated_logits),
            ("receiver_total_patchout", direct_receiver_logits),
        ):
            rows.append(
                {
                    "sender": sender.label,
                    "receiver": receiver.label,
                    "condition": condition,
                    **normalized_component_metrics(
                        clean_logits=clean_logits,
                        corrupt_logits=corrupt_logits,
                        patched_logits=logits,
                        target=batch.target,
                        endpoint=batch.endpoint,
                    ),
                }
            )
    return rows, {"terminal": corrupt_trace, "J1": clean_trace}


@torch.no_grad()
def second_use_rescue_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    clean_state: torch.Tensor,
    corrupt_state: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, FunctionalTrace]]:
    corrupt_logits, corrupt_trace = run_instrumented_state(
        model,
        corrupt_state,
        loop_indices=(cfg.max_loops + 1,),
    )
    clean_logits, clean_trace = run_instrumented_state(
        model,
        clean_state,
        loop_indices=(cfg.max_loops + 1,),
    )
    shuffled_clean = _rolled_trace(clean_trace)
    rows = []
    for node in second_rescue_nodes():
        intervention = node.intervention(cfg=cfg, current=batch.current)
        rescued_logits, _ = run_instrumented_state(
            model,
            corrupt_state,
            loop_indices=(cfg.max_loops + 1,),
            interventions=(intervention,),
            donor_trace=clean_trace,
        )
        shuffled_logits, _ = run_instrumented_state(
            model,
            corrupt_state,
            loop_indices=(cfg.max_loops + 1,),
            interventions=(intervention,),
            donor_trace=shuffled_clean,
        )
        for condition, logits in (
            ("matched_oracle_into_bad", rescued_logits),
            ("shuffled_oracle_into_bad", shuffled_logits),
        ):
            rows.append(
                {
                    "node": node.label,
                    "condition": condition,
                    **target_margin_rescue_metrics(
                        clean_logits=clean_logits,
                        corrupt_logits=corrupt_logits,
                        patched_logits=logits,
                        target=batch.target,
                        endpoint=batch.endpoint,
                    ),
                }
            )
    return rows, {"J2_bad": corrupt_trace, "oracle_age2": clean_trace}


def _relative_mse(value: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (value.float() - reference.float()).square().mean()
    denominator = (
        reference.float() - reference.float().mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-12)
    return float(numerator / denominator)


def _alignment_row(
    value: torch.Tensor,
    reference: torch.Tensor,
    *,
    stage: str,
    reference_name: str,
) -> dict[str, Any]:
    value_flat = value.float().flatten(1)
    reference_flat = reference.float().flatten(1)
    return {
        "stage": stage,
        "reference": reference_name,
        "relative_mse": _relative_mse(value, reference),
        "cosine_similarity": float(
            F.cosine_similarity(value_flat, reference_flat, dim=1).mean()
        ),
        "relative_energy_error": float(
            (value_flat - reference_flat).square().sum(dim=1).mean()
            / reference_flat.square().sum(dim=1).mean().clamp_min(1e-12)
        ),
        "value_norm": float(value_flat.norm(dim=1).mean()),
        "reference_norm": float(reference_flat.norm(dim=1).mean()),
    }


@torch.no_grad()
def state_age_alignment_rows(
    *,
    trajectory: Seed1Trajectory,
    positions: Sequence[int],
) -> list[dict[str, Any]]:
    index = list(positions)
    rows = [
        _alignment_row(
            trajectory.j1_pre[:, index],
            trajectory.young1[:, index],
            stage="J1_terminal",
            reference_name="aligned_age2_current_f8",
        ),
        _alignment_row(
            trajectory.post1[:, index],
            trajectory.aligned_second_ages[3][:, index],
            stage="after_first_execution",
            reference_name="aligned_age3_current_f10",
        ),
        _alignment_row(
            trajectory.j2_pre_bad[:, index],
            trajectory.young2[:, index],
            stage="J1_reused_after_first_execution",
            reference_name="aligned_age2_current_f10",
        ),
    ]
    for age, reference in trajectory.aligned_second_ages.items():
        rows.append(
            _alignment_row(
                trajectory.j2_pre_bad[:, index],
                reference[:, index],
                stage="J1_reused_after_first_execution",
                reference_name=f"aligned_age{age}_current_f10",
            )
        )

    first_update = trajectory.j1_pre[:, index] - trajectory.terminal[:, index]
    second_update = trajectory.j2_pre_bad[:, index] - trajectory.post1[:, index]
    desired_second = trajectory.young2[:, index] - trajectory.post1[:, index]
    rows.extend(
        (
            _alignment_row(
                second_update,
                desired_second,
                stage="second_J_update",
                reference_name="desired_age3_to_age2_update",
            ),
            _alignment_row(
                second_update,
                first_update,
                stage="second_J_update",
                reference_name="realized_first_J_update",
            ),
        )
    )
    return rows


def _tensor_alignment(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    return (
        _relative_mse(value, reference),
        float(
            F.cosine_similarity(
                value.float().flatten(1),
                reference.float().flatten(1),
                dim=1,
            ).mean()
        ),
    )


@torch.no_grad()
def internal_alignment_rows(
    *,
    cfg: GraphPathConfig,
    traces: dict[str, FunctionalTrace],
) -> list[dict[str, Any]]:
    answer = cfg.seq_len - 1
    pairs = (
        ("J1", "oracle1"),
        ("J2_bad", "oracle2"),
    )
    components = (
        ("B1_H1_context", 0, "head_context", 1),
        ("B1_MLP_hidden", 0, "mlp_hidden", None),
        ("B1_MLP_out", 0, "mlp_out", None),
        ("B2_H2_q", 1, "q", 2),
        ("B2_H2_pattern", 1, "attention_pattern", 2),
        ("B2_H2_context", 1, "head_context", 2),
        ("B2_MLP_hidden", 1, "mlp_hidden", None),
        ("B2_MLP_out", 1, "mlp_out", None),
    )
    rows = []
    for run, reference_run in pairs:
        for label, site_index, component, head in components:
            site = traces[run].sites[site_index]
            reference_site = traces[reference_run].sites[site_index]
            value = getattr(site, component)
            reference = getattr(reference_site, component)
            if component == "attention_pattern":
                graph = explicit_depth_position_groups(cfg.node_count)["graph"]
                value = value[:, head, answer, list(graph)]
                reference = reference[:, head, answer, list(graph)]
            elif head is not None:
                value = value[:, head, answer]
                reference = reference[:, head, answer]
            else:
                value = value[:, answer]
                reference = reference[:, answer]
            relative_mse, cosine = _tensor_alignment(value, reference)
            rows.append(
                {
                    "run": run,
                    "reference_run": reference_run,
                    "component": label,
                    "relative_mse": relative_mse,
                    "cosine_similarity": cosine,
                }
            )
    return rows


def _topk_jaccard(
    value: torch.Tensor,
    reference: torch.Tensor,
    *,
    top_k: int,
) -> float:
    value_score = value.float().abs().mean(dim=0)
    reference_score = reference.float().abs().mean(dim=0)
    count = min(top_k, value_score.numel())
    value_top = set(value_score.topk(count).indices.tolist())
    reference_top = set(reference_score.topk(count).indices.tolist())
    return len(value_top & reference_top) / len(value_top | reference_top)


@torch.no_grad()
def mlp_overlap_rows(
    *,
    cfg: GraphPathConfig,
    traces: dict[str, FunctionalTrace],
    top_ks: Sequence[int],
) -> list[dict[str, Any]]:
    answer = cfg.seq_len - 1
    pairs = (
        ("J1", "oracle1"),
        ("J2_bad", "oracle2"),
        ("J1", "J2_bad"),
        ("oracle1", "oracle2"),
    )
    rows = []
    for run, reference_run in pairs:
        for site in range(cfg.n_layers):
            value = traces[run].sites[site].mlp_hidden[:, answer]
            reference = traces[reference_run].sites[site].mlp_hidden[:, answer]
            for top_k in top_ks:
                rows.append(
                    {
                        "run": run,
                        "reference_run": reference_run,
                        "block": site + 1,
                        "top_k": int(top_k),
                        "topk_neuron_jaccard": _topk_jaccard(
                            value,
                            reference,
                            top_k=int(top_k),
                        ),
                    }
                )
    return rows


def _find_row(
    rows: Sequence[dict[str, Any]],
    **criteria: Any,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    )


@torch.no_grad()
def analyze_seed(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    affine: AffineMap,
    positions: Sequence[int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    trajectory = collect_seed1_trajectory(
        model=model,
        cfg=cfg,
        candidate=candidate,
        affine=affine,
        positions=positions,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    first_batch = trajectory.first_batch()
    second_batch = trajectory.second_batch()
    first_path, first_core_traces = first_use_path_rows(
        model=model,
        cfg=cfg,
        batch=first_batch,
        clean_state=trajectory.j1_pre,
        corrupt_state=trajectory.terminal,
    )
    _, oracle1_trace = run_instrumented_state(
        model,
        trajectory.oracle1,
        loop_indices=(cfg.max_loops,),
    )
    second_rescue, second_core_traces = second_use_rescue_rows(
        model=model,
        cfg=cfg,
        batch=second_batch,
        clean_state=trajectory.oracle2,
        corrupt_state=trajectory.j2_pre_bad,
    )
    traces = {
        **first_core_traces,
        "oracle1": oracle1_trace,
        **second_core_traces,
        "oracle2": second_core_traces["oracle_age2"],
    }
    first_stages = stage_rows(
        model=model,
        batch=first_batch,
        initial_states={
            "terminal": trajectory.terminal,
            "J1": trajectory.j1_pre,
            "oracle1": trajectory.oracle1,
        },
        traces={
            "terminal": traces["terminal"],
            "J1": traces["J1"],
            "oracle1": traces["oracle1"],
        },
    )
    second_stages = stage_rows(
        model=model,
        batch=second_batch,
        initial_states={
            "J2_bad": trajectory.j2_pre_bad,
            "oracle2": trajectory.oracle2,
        },
        traces={
            "J2_bad": traces["J2_bad"],
            "oracle2": traces["oracle2"],
        },
    )
    first_attention = attention_function_rows(
        cfg=cfg,
        batch=first_batch,
        traces={
            "terminal": traces["terminal"],
            "J1": traces["J1"],
            "oracle1": traces["oracle1"],
        },
    )
    second_attention = attention_function_rows(
        cfg=cfg,
        batch=second_batch,
        traces={
            "J2_bad": traces["J2_bad"],
            "oracle2": traces["oracle2"],
        },
    )
    state_alignment = state_age_alignment_rows(
        trajectory=trajectory,
        positions=positions,
    )
    internal_alignment = internal_alignment_rows(cfg=cfg, traces=traces)
    mlp_overlap = mlp_overlap_rows(
        cfg=cfg,
        traces=traces,
        top_ks=(16, 32, 64, 128),
    )
    summary = {
        "seed": seed,
        "sample_size": batch_size,
        "first_use": {
            "pre_J_accuracy": _find_row(
                first_stages, run="J1", stage="input"
            )["accuracy"],
            "post_stack_accuracy": _find_row(
                first_stages, run="J1", stage="B2_post_mlp"
            )["accuracy"],
            "oracle_post_stack_accuracy": _find_row(
                first_stages, run="oracle1", stage="B2_post_mlp"
            )["accuracy"],
            "B2_H2_current_destination_attention": _find_row(
                first_attention,
                run="J1",
                block=2,
                head=2,
                key_role="current_destination",
            )["attention"],
        },
        "second_use": {
            "pre_J_accuracy": _find_row(
                second_stages, run="J2_bad", stage="input"
            )["accuracy"],
            "post_stack_accuracy": _find_row(
                second_stages, run="J2_bad", stage="B2_post_mlp"
            )["accuracy"],
            "oracle_post_stack_accuracy": _find_row(
                second_stages, run="oracle2", stage="B2_post_mlp"
            )["accuracy"],
            "bad_B2_H2_current_destination_attention": _find_row(
                second_attention,
                run="J2_bad",
                block=2,
                head=2,
                key_role="current_destination",
            )["attention"],
            "oracle_B2_H2_current_destination_attention": _find_row(
                second_attention,
                run="oracle2",
                block=2,
                head=2,
                key_role="current_destination",
            )["attention"],
        },
        "second_J_to_age2_relative_mse": _find_row(
            state_alignment,
            stage="J1_reused_after_first_execution",
            reference="aligned_age2_current_f10",
        )["relative_mse"],
        "best_aligned_age_by_cosine_after_second_J": max(
            (
                row
                for row in state_alignment
                if row["stage"] == "J1_reused_after_first_execution"
                and row["reference"].startswith("aligned_age")
            ),
            key=lambda row: row["cosine_similarity"],
        )["reference"],
        "best_aligned_age_cosine_after_second_J": max(
            float(row["cosine_similarity"])
            for row in state_alignment
            if row["stage"] == "J1_reused_after_first_execution"
            and row["reference"].startswith("aligned_age")
        ),
    }
    return {
        "summary": summary,
        "first_path_rows": first_path,
        "second_rescue_rows": second_rescue,
        "first_stage_rows": first_stages,
        "second_stage_rows": second_stages,
        "first_attention_rows": first_attention,
        "second_attention_rows": second_attention,
        "state_age_alignment_rows": state_alignment,
        "internal_alignment_rows": internal_alignment,
        "mlp_overlap_rows": mlp_overlap,
    }


def _mean(values: Sequence[float]) -> float:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return float(tensor[torch.isfinite(tensor)].mean())


def aggregate_summaries(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries = [run["summary"] for run in runs]
    return {
        "evaluation_seeds": [summary["seed"] for summary in summaries],
        "sample_size_per_seed": summaries[0]["sample_size"],
        "first_use_post_stack_accuracy_mean": _mean(
            [
                summary["first_use"]["post_stack_accuracy"]
                for summary in summaries
            ]
        ),
        "second_use_post_stack_accuracy_mean": _mean(
            [
                summary["second_use"]["post_stack_accuracy"]
                for summary in summaries
            ]
        ),
        "second_use_oracle_accuracy_mean": _mean(
            [
                summary["second_use"]["oracle_post_stack_accuracy"]
                for summary in summaries
            ]
        ),
        "second_J_to_age2_relative_mse_mean": _mean(
            [summary["second_J_to_age2_relative_mse"] for summary in summaries]
        ),
        "best_aligned_age_by_cosine_after_second_J": [
            summary["best_aligned_age_by_cosine_after_second_J"]
            for summary in summaries
        ],
        "best_aligned_age_cosine_after_second_J_mean": _mean(
            [
                summary["best_aligned_age_cosine_after_second_J"]
                for summary in summaries
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint-specific D8L8-seed1 J1 path patching and "
            "second-use failure localization."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Independent evaluation-data seed; repeat the flag for replications.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seed or [2026074101, 2026075101]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise ValueError("cuda memory fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    affine, matrix_payload = load_saved_affine(args.matrix, device=device)
    positions = tuple(int(value) for value in matrix_payload["positions"])
    if tuple(positions) != (cfg.seq_len - 1,):
        raise ValueError("the seed1 matrix must edit only the answer position")
    if Path(matrix_payload["checkpoint"]) != args.checkpoint:
        raise ValueError("matrix artifact does not match the requested checkpoint")
    candidate = matrix_payload["candidate"]
    run_results = []
    for seed in seeds:
        result = analyze_seed(
            model=model,
            cfg=cfg,
            candidate=candidate,
            affine=affine,
            positions=positions,
            batch_size=args.batch_size,
            device=device,
            seed=int(seed),
        )
        run_dir = args.out_dir / f"eval_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for key, value in result.items():
            if key == "summary":
                continue
            _write_csv(run_dir / f"{key}.csv", value)
        (run_dir / "summary.json").write_text(
            json.dumps(result["summary"], indent=2),
            encoding="utf-8",
        )
        run_results.append(result)
        print(f"done evaluation seed {seed}", flush=True)
    aggregate = {
        "model": "D8L8-seed1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "initialization_seed": checkpoint_payload.get("initialization_seed"),
        "config": asdict(cfg),
        "matrix": str(args.matrix),
        "matrix_selected_ridge": matrix_payload["selected_ridge"],
        "matrix_selection_metric": matrix_payload["ridge_selection_metric"],
        "matrix_is_fixed_across_evaluation_seeds": True,
        "matrix_is_never_refit_in_this_experiment": True,
        "position_group": matrix_payload["position_group"],
        "positions": positions,
        **aggregate_summaries(run_results),
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
