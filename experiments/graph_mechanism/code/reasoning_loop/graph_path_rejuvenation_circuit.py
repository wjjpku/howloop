from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    logit_difference,
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
    set_seed,
)
from reasoning_loop.graph_path_telomere_overloop import (
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_telomere_rejuvenator import (
    PositionwiseAffine,
    fit_positionwise_affine,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


@dataclass(frozen=True)
class MatchedAgeBatch:
    terminal: torch.Tensor
    young: torch.Tensor
    successors: torch.Tensor
    current: torch.Tensor
    target: torch.Tensor
    endpoint: torch.Tensor


@dataclass(frozen=True)
class CircuitNode:
    label: str
    site: int
    component: str
    head: int | None = None

    def intervention(
        self,
        *,
        answer_position: int,
    ) -> FunctionalIntervention:
        return FunctionalIntervention(
            site=self.site,
            component=self.component,  # type: ignore[arg-type]
            mode="patch",
            positions=(answer_position,),
            heads=None if self.head is None else (self.head,),
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_matched_age_batch(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> MatchedAgeBatch:
    reference_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    set_seed(seed)
    tokens, targets, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth + jump,
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
    reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=reference_start,
    )
    young = cache_states_with_initial(
        model,
        reference_tokens,
        loops=max(1, reference_age),
    )[reference_age]
    endpoint = targets[:, cfg.max_depth - 1]
    target = targets[:, cfg.max_depth + jump - 1]
    return MatchedAgeBatch(
        terminal=terminal,
        young=young,
        successors=successors,
        current=endpoint,
        target=target,
        endpoint=endpoint,
    )


def fit_initial_affine(
    batch: MatchedAgeBatch,
    *,
    ridge: float,
) -> PositionwiseAffine:
    return fit_positionwise_affine(
        batch.terminal[:, -1:].float(),
        batch.young[:, -1:].float(),
        ridge=ridge,
    )


def affine_update_matrix(
    affine: PositionwiseAffine,
) -> torch.Tensor:
    if affine.weight.shape[0] != 1 or affine.bias.shape[0] != 1:
        raise ValueError("the init map must act on one answer position")
    weight = affine.weight[0]
    bias = affine.bias[0]
    if weight.shape[0] != weight.shape[1]:
        raise ValueError("the init map must be square")
    identity = torch.eye(
        weight.shape[0],
        device=weight.device,
        dtype=weight.dtype,
    )
    return torch.cat((weight - identity, bias.unsqueeze(0)), dim=0)


def homogeneous_matrix_from_update(
    update_matrix: torch.Tensor,
) -> torch.Tensor:
    input_count, output_count = update_matrix.shape
    if input_count != output_count + 1:
        raise ValueError("expected a [d+1, d] affine update matrix")
    matrix = torch.eye(
        input_count,
        device=update_matrix.device,
        dtype=update_matrix.dtype,
    )
    matrix[:, :-1] += update_matrix
    return matrix


def truncated_update(
    left: torch.Tensor,
    singular_values: torch.Tensor,
    right: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    if not 0 <= rank <= singular_values.numel():
        raise ValueError("rank is outside the SVD range")
    if rank == 0:
        return torch.zeros(
            left.shape[0],
            right.shape[1],
            device=left.device,
            dtype=left.dtype,
        )
    return (
        left[:, :rank]
        * singular_values[:rank].unsqueeze(0)
    ) @ right[:rank]


def age_subspace_update(
    full_update: torch.Tensor,
    *,
    mean_delta: torch.Tensor,
    directions: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Project the realized affine update into an r-dimensional age subspace."""
    if not 0 <= rank <= directions.shape[0]:
        raise ValueError("rank is outside the age-subspace range")
    if rank == 0:
        result = torch.zeros_like(full_update)
        result[-1] = mean_delta
        return result
    basis = directions[:rank]
    projector = basis.T @ basis
    identity = torch.eye(
        projector.shape[0],
        device=projector.device,
        dtype=projector.dtype,
    )
    result = full_update @ projector
    result[-1] += mean_delta @ (identity - projector)
    return result


def centered_age_mode_update(
    full_update: torch.Tensor,
    *,
    mean_delta: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    projector = direction[:, None] @ direction[None, :]
    result = full_update @ projector
    result[-1] -= mean_delta @ projector
    return result


def random_rank_matched_update(
    singular_values: torch.Tensor,
    *,
    input_count: int,
    output_count: int,
    rank: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if rank == 0:
        return torch.zeros(
            input_count,
            output_count,
            device=singular_values.device,
            dtype=singular_values.dtype,
        )
    left = torch.randn(
        input_count,
        rank,
        device=singular_values.device,
        dtype=singular_values.dtype,
        generator=generator,
    )
    right = torch.randn(
        output_count,
        rank,
        device=singular_values.device,
        dtype=singular_values.dtype,
        generator=generator,
    )
    left = torch.linalg.qr(left, mode="reduced").Q
    right = torch.linalg.qr(right, mode="reduced").Q
    return (
        left * singular_values[:rank].unsqueeze(0)
    ) @ right.T


def apply_update_to_state(
    state: torch.Tensor,
    update_matrix: torch.Tensor,
    *,
    mode: str = "matched",
) -> torch.Tensor:
    if mode not in {"matched", "shuffled", "reverse"}:
        raise ValueError(f"unknown update mode: {mode}")
    answer = state[:, -1].float()
    augmented = torch.cat(
        (
            answer,
            torch.ones(
                answer.shape[0],
                1,
                device=answer.device,
                dtype=answer.dtype,
            ),
        ),
        dim=1,
    )
    update = augmented @ update_matrix
    if mode == "shuffled":
        update = update.roll(1, dims=0)
    elif mode == "reverse":
        update = -update
    result = state.clone()
    result[:, -1] = result[:, -1] + update.to(result.dtype)
    return result


def relative_state_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    numerator = (prediction.float() - target.float()).square().mean()
    denominator = (
        target.float() - target.float().mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-12)
    return float(numerator / denominator)


def behavior_metrics(
    logits: torch.Tensor,
    *,
    target: torch.Tensor,
    endpoint: torch.Tensor,
) -> dict[str, float]:
    valid = target.ne(endpoint)
    count = int(valid.sum())
    if count == 0:
        return {
            "valid_count": 0,
            "accuracy": float("nan"),
            "endpoint_accuracy": float("nan"),
            "probability": float("nan"),
            "margin": float("nan"),
        }
    prediction = logits.argmax(dim=-1)
    probability = logits.softmax(dim=-1).gather(
        1, target[:, None]
    ).squeeze(1)
    margin = logit_difference(logits, target, endpoint)
    return {
        "valid_count": count,
        "accuracy": float(prediction[valid].eq(target[valid]).float().mean()),
        "endpoint_accuracy": float(
            prediction[valid].eq(endpoint[valid]).float().mean()
        ),
        "probability": float(probability[valid].mean()),
        "margin": float(margin[valid].mean()),
    }


@torch.no_grad()
def evaluate_state(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    state: torch.Tensor,
    batch: MatchedAgeBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = apply_shared_stack(
        model,
        state,
        loop_index=cfg.max_loops,
    )
    logits = logits_from_raw_state(model, output)
    return logits, behavior_metrics(
        logits,
        target=batch.target,
        endpoint=batch.endpoint,
    )


def normalized_component_metrics(
    *,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    target: torch.Tensor,
    endpoint: torch.Tensor,
) -> dict[str, float]:
    valid = (
        target.ne(endpoint)
        & clean_logits.argmax(dim=-1).eq(target)
        & corrupt_logits.argmax(dim=-1).eq(endpoint)
    )
    clean_margin = logit_difference(clean_logits, target, endpoint)
    corrupt_margin = logit_difference(corrupt_logits, target, endpoint)
    patched_margin = logit_difference(patched_logits, target, endpoint)
    denominator = clean_margin - corrupt_margin
    valid = valid & denominator.abs().gt(1e-6)
    recovery = (patched_margin - corrupt_margin) / denominator
    basic = behavior_metrics(
        patched_logits,
        target=target,
        endpoint=endpoint,
    )
    basic["controlled_valid_count"] = int(valid.sum())
    basic["recovery"] = (
        float(recovery[valid].mean())
        if bool(valid.any())
        else float("nan")
    )
    basic["necessity"] = 1.0 - basic["recovery"]
    return basic


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


def circuit_nodes(cfg: GraphPathConfig) -> list[CircuitNode]:
    nodes = []
    effective_loop = cfg.max_loops + 1
    for site in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            nodes.append(
                CircuitNode(
                    label=(
                        f"L{effective_loop}.B{site + 1}."
                        f"H{head}.context@answer"
                    ),
                    site=site,
                    component="head_context",
                    head=head,
                )
            )
        nodes.append(
            CircuitNode(
                label=(
                    f"L{effective_loop}.B{site + 1}.MLP.out@answer"
                ),
                site=site,
                component="mlp_out",
            )
        )
    return nodes


def node_interventions(
    nodes: Sequence[CircuitNode],
    *,
    answer_position: int,
) -> tuple[FunctionalIntervention, ...]:
    return tuple(
        node.intervention(answer_position=answer_position)
        for node in nodes
    )


@torch.no_grad()
def run_component_hybrid(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    initial_state: torch.Tensor,
    patched_nodes: Sequence[CircuitNode],
    donor_trace: FunctionalTrace,
) -> torch.Tensor:
    logits, _ = run_instrumented_state(
        model,
        initial_state,
        loop_indices=(cfg.max_loops,),
        interventions=node_interventions(
            patched_nodes,
            answer_position=cfg.seq_len - 1,
        ),
        donor_trace=donor_trace,
    )
    return logits


def _metric_row(
    metrics: dict[str, float],
    **fields: Any,
) -> dict[str, Any]:
    return {**fields, **metrics}


@torch.no_grad()
def component_localization_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    rejuvenated_state: torch.Tensor,
) -> tuple[
    list[dict[str, Any]],
    torch.Tensor,
    FunctionalTrace,
    torch.Tensor,
    FunctionalTrace,
]:
    old_logits, old_trace = run_instrumented_state(
        model,
        batch.terminal,
        loop_indices=(cfg.max_loops,),
    )
    clean_logits, clean_trace = run_instrumented_state(
        model,
        rejuvenated_state,
        loop_indices=(cfg.max_loops,),
    )
    shuffled_trace = _rolled_trace(clean_trace)
    rows = []
    for node in circuit_nodes(cfg):
        intervention = (node.intervention(answer_position=cfg.seq_len - 1),)
        patch_in, _ = run_instrumented_state(
            model,
            batch.terminal,
            loop_indices=(cfg.max_loops,),
            interventions=intervention,
            donor_trace=clean_trace,
        )
        shuffled, _ = run_instrumented_state(
            model,
            batch.terminal,
            loop_indices=(cfg.max_loops,),
            interventions=intervention,
            donor_trace=shuffled_trace,
        )
        patch_out, _ = run_instrumented_state(
            model,
            rejuvenated_state,
            loop_indices=(cfg.max_loops,),
            interventions=intervention,
            donor_trace=old_trace,
        )
        for condition, logits in (
            ("patch_in", patch_in),
            ("patch_in_shuffled", shuffled),
            ("patch_out", patch_out),
        ):
            metrics = normalized_component_metrics(
                clean_logits=clean_logits,
                corrupt_logits=old_logits,
                patched_logits=logits,
                target=batch.target,
                endpoint=batch.endpoint,
            )
            rows.append(
                _metric_row(
                    metrics,
                    node=node.label,
                    site=node.site,
                    component=node.component,
                    head="" if node.head is None else node.head,
                    condition=condition,
                )
            )
    return rows, old_logits, old_trace, clean_logits, clean_trace


@torch.no_grad()
def discover_component_circuit(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    rejuvenated_state: torch.Tensor,
    target_recovery: float,
    min_accuracy: float = 0.90,
) -> tuple[list[CircuitNode], list[dict[str, Any]]]:
    nodes = circuit_nodes(cfg)
    old_logits, old_trace = run_instrumented_state(
        model,
        batch.terminal,
        loop_indices=(cfg.max_loops,),
    )
    clean_logits, _ = run_instrumented_state(
        model,
        rejuvenated_state,
        loop_indices=(cfg.max_loops,),
    )
    selected: list[CircuitNode] = []
    history = []
    while len(selected) < len(nodes):
        candidates = [node for node in nodes if node not in selected]
        scored = []
        for candidate in candidates:
            trial = [*selected, candidate]
            patched_to_old = [node for node in nodes if node not in trial]
            logits = run_component_hybrid(
                model=model,
                cfg=cfg,
                initial_state=rejuvenated_state,
                patched_nodes=patched_to_old,
                donor_trace=old_trace,
            )
            metrics = normalized_component_metrics(
                clean_logits=clean_logits,
                corrupt_logits=old_logits,
                patched_logits=logits,
                target=batch.target,
                endpoint=batch.endpoint,
            )
            scored.append((metrics["recovery"], candidate, metrics))
        _, best, best_metrics = max(
            scored,
            key=lambda item: (
                float("-inf")
                if not np.isfinite(item[0])
                else item[0]
            ),
        )
        selected.append(best)
        history.append(
            _metric_row(
                best_metrics,
                step=len(selected),
                added_node=best.label,
                selected_nodes=";".join(node.label for node in selected),
            )
        )
        if (
            best_metrics["recovery"] >= target_recovery
            and best_metrics["accuracy"] >= min_accuracy
        ):
            break
    return selected, history


@torch.no_grad()
def validate_component_circuit(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    rejuvenated_state: torch.Tensor,
    selected: Sequence[CircuitNode],
    random_subsets: int,
    exhaustive_subsets: bool,
    seed: int,
) -> list[dict[str, Any]]:
    nodes = circuit_nodes(cfg)
    old_logits, old_trace = run_instrumented_state(
        model,
        batch.terminal,
        loop_indices=(cfg.max_loops,),
    )
    clean_logits, clean_trace = run_instrumented_state(
        model,
        rejuvenated_state,
        loop_indices=(cfg.max_loops,),
    )
    rows = []

    def evaluate(
        label: str,
        *,
        initial_state: torch.Tensor,
        patched_nodes: Sequence[CircuitNode],
        donor_trace: FunctionalTrace,
    ) -> None:
        logits = run_component_hybrid(
            model=model,
            cfg=cfg,
            initial_state=initial_state,
            patched_nodes=patched_nodes,
            donor_trace=donor_trace,
        )
        metrics = normalized_component_metrics(
            clean_logits=clean_logits,
            corrupt_logits=old_logits,
            patched_logits=logits,
            target=batch.target,
            endpoint=batch.endpoint,
        )
        rows.append(
            _metric_row(
                metrics,
                condition=label,
                node_count=len(selected),
                nodes=";".join(node.label for node in selected),
            )
        )

    evaluate(
        "circuit_only",
        initial_state=rejuvenated_state,
        patched_nodes=[node for node in nodes if node not in selected],
        donor_trace=old_trace,
    )
    evaluate(
        "complement_only",
        initial_state=rejuvenated_state,
        patched_nodes=selected,
        donor_trace=old_trace,
    )
    evaluate(
        "selected_patch_in",
        initial_state=batch.terminal,
        patched_nodes=selected,
        donor_trace=clean_trace,
    )
    evaluate(
        "all_candidate_changes_removed",
        initial_state=rejuvenated_state,
        patched_nodes=nodes,
        donor_trace=old_trace,
    )
    for removed in selected:
        kept = [node for node in selected if node != removed]
        evaluate(
            f"leave_out:{removed.label}",
            initial_state=rejuvenated_state,
            patched_nodes=[node for node in nodes if node not in kept],
            donor_trace=old_trace,
        )
    generator = torch.Generator().manual_seed(seed)
    for sample in range(random_subsets):
        order = torch.randperm(len(nodes), generator=generator)
        random_selected = [
            nodes[int(index)] for index in order[: len(selected)]
        ]
        logits = run_component_hybrid(
            model=model,
            cfg=cfg,
            initial_state=rejuvenated_state,
            patched_nodes=[
                node for node in nodes if node not in random_selected
            ],
            donor_trace=old_trace,
        )
        metrics = normalized_component_metrics(
            clean_logits=clean_logits,
            corrupt_logits=old_logits,
            patched_logits=logits,
            target=batch.target,
            endpoint=batch.endpoint,
        )
        rows.append(
            _metric_row(
                metrics,
                condition=f"random_circuit_only_{sample}",
                node_count=len(selected),
                nodes=";".join(
                    node.label for node in random_selected
                ),
            )
        )
    if exhaustive_subsets:
        for subset_index, indices in enumerate(
            itertools.combinations(range(len(nodes)), len(selected))
        ):
            subset = [nodes[index] for index in indices]
            logits = run_component_hybrid(
                model=model,
                cfg=cfg,
                initial_state=rejuvenated_state,
                patched_nodes=[
                    node for node in nodes if node not in subset
                ],
                donor_trace=old_trace,
            )
            metrics = normalized_component_metrics(
                clean_logits=clean_logits,
                corrupt_logits=old_logits,
                patched_logits=logits,
                target=batch.target,
                endpoint=batch.endpoint,
            )
            rows.append(
                _metric_row(
                    metrics,
                    condition=f"exhaustive_subset_{subset_index}",
                    node_count=len(selected),
                    nodes=";".join(node.label for node in subset),
                )
            )
    return rows


@torch.no_grad()
def qkv_mediation_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    rejuvenated_state: torch.Tensor,
) -> list[dict[str, Any]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    old_logits, old_trace = run_instrumented_state(
        model,
        batch.terminal,
        loop_indices=(cfg.max_loops,),
    )
    clean_logits, _ = run_instrumented_state(
        model,
        rejuvenated_state,
        loop_indices=(cfg.max_loops,),
    )
    rows = []
    specs: list[tuple[str, str, dict[str, Any]]] = [
        ("q_answer", "q", {"positions": answer}),
        ("k_answer", "k", {"positions": answer}),
        ("v_answer", "v", {"positions": answer}),
        ("k_graph", "k", {"positions": graph}),
        ("v_graph", "v", {"positions": graph}),
        (
            "pattern_answer_graph",
            "attention_pattern",
            {
                "positions": answer,
                "source_positions": graph,
                "renormalize": True,
            },
        ),
        ("context_answer", "head_context", {"positions": answer}),
    ]
    for site in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            for role, component, fields in specs:
                intervention = FunctionalIntervention(
                    site=site,
                    component=component,  # type: ignore[arg-type]
                    mode="patch",
                    heads=(head,),
                    **fields,
                )
                patched, _ = run_instrumented_state(
                    model,
                    rejuvenated_state,
                    loop_indices=(cfg.max_loops,),
                    interventions=(intervention,),
                    donor_trace=old_trace,
                )
                metrics = normalized_component_metrics(
                    clean_logits=clean_logits,
                    corrupt_logits=old_logits,
                    patched_logits=patched,
                    target=batch.target,
                    endpoint=batch.endpoint,
                )
                rows.append(
                    _metric_row(
                        metrics,
                        site=site,
                        block=site + 1,
                        head=head,
                        role=role,
                        component=component,
                        condition="old_into_rejuvenated",
                    )
                )
    return rows


def _dynamic_attention_value(
    pattern: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    head: int,
    answer_position: int,
) -> float:
    batch = torch.arange(pattern.shape[0], device=pattern.device)
    return float(
        pattern[batch, head, answer_position, key_positions].mean()
    )


@torch.no_grad()
def attention_function_rows(
    *,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    traces: dict[str, FunctionalTrace],
) -> list[dict[str, Any]]:
    current = batch.current
    next_node = batch.successors.gather(
        1, current[:, None]
    ).squeeze(1)
    slots = {
        "current_source": 2 + 3 * current,
        "current_destination": 3 + 3 * current,
        "next_source": 2 + 3 * next_node,
        "next_destination": 3 + 3 * next_node,
        "bos": torch.zeros_like(current),
        "query": torch.full_like(current, cfg.seq_len - 4),
        "start": torch.full_like(current, cfg.seq_len - 3),
        "depth": torch.full_like(current, cfg.seq_len - 2),
        "answer_self": torch.full_like(current, cfg.seq_len - 1),
    }
    rows = []
    answer = cfg.seq_len - 1
    graph_positions = torch.as_tensor(
        explicit_depth_position_groups(cfg.node_count)["graph"],
        device=current.device,
    )
    for run_name, trace in traces.items():
        for site, site_trace in enumerate(trace.sites):
            for head in range(cfg.n_heads):
                pattern = site_trace.attention_pattern
                graph_mass = float(
                    pattern[:, head, answer, graph_positions].sum(-1).mean()
                )
                for role, positions in slots.items():
                    rows.append(
                        {
                            "run": run_name,
                            "site": site,
                            "block": site + 1,
                            "head": head,
                            "key_role": role,
                            "attention": _dynamic_attention_value(
                                pattern,
                                positions,
                                head=head,
                                answer_position=answer,
                            ),
                            "total_graph_attention": graph_mass,
                        }
                    )
    return rows


@torch.no_grad()
def stage_readout_rows(
    *,
    model: LoopedGraphPathTransformer,
    batch: MatchedAgeBatch,
    initial_states: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
) -> list[dict[str, Any]]:
    rows = []
    for run_name, trace in traces.items():
        states = (
            ("input_to_loop9", initial_states[run_name]),
            ("B1_post_attention", trace.sites[0].residual_mid),
            ("B1_post_mlp", trace.sites[0].hidden_out),
            ("B2_post_attention", trace.sites[1].residual_mid),
            ("B2_post_mlp", trace.sites[1].hidden_out),
        )
        for stage, state in states:
            metrics = behavior_metrics(
                logits_from_raw_state(model, state),
                target=batch.target,
                endpoint=batch.endpoint,
            )
            rows.append(
                _metric_row(
                    metrics,
                    run=run_name,
                    stage=stage,
                )
            )
    return rows


@torch.no_grad()
def mlp_neuron_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    discovery_batch: MatchedAgeBatch,
    discovery_state: torch.Tensor,
    evaluation_batch: MatchedAgeBatch,
    evaluation_state: torch.Tensor,
    top_ks: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    _, discovery_old = run_instrumented_state(
        model,
        discovery_batch.terminal,
        loop_indices=(cfg.max_loops,),
    )
    _, discovery_clean = run_instrumented_state(
        model,
        discovery_state,
        loop_indices=(cfg.max_loops,),
    )
    old_logits, evaluation_old = run_instrumented_state(
        model,
        evaluation_batch.terminal,
        loop_indices=(cfg.max_loops,),
    )
    clean_logits, _ = run_instrumented_state(
        model,
        evaluation_state,
        loop_indices=(cfg.max_loops,),
    )
    generator = torch.Generator(device=evaluation_state.device).manual_seed(
        seed
    )
    rows = []
    for site in range(cfg.n_layers):
        delta = (
            discovery_clean.sites[site].mlp_hidden[:, -1]
            - discovery_old.sites[site].mlp_hidden[:, -1]
        )
        activation_score = delta.abs().mean(dim=0)
        all_neurons = torch.arange(
            activation_score.numel(),
            device=activation_score.device,
        )
        ranked = activation_score.argsort(descending=True)
        conditions: list[
            tuple[str, torch.Tensor, int, torch.Tensor]
        ] = [("all", all_neurons, all_neurons.numel(), ranked)]
        for requested_k in top_ks:
            count = min(int(requested_k), activation_score.numel())
            top = ranked[:count]
            remaining_mask = torch.ones(
                activation_score.numel(),
                dtype=torch.bool,
                device=activation_score.device,
            )
            remaining_mask[top] = False
            remaining = torch.where(remaining_mask)[0]
            random = remaining[
                torch.randperm(
                    remaining.numel(),
                    device=remaining.device,
                    generator=generator,
                )[:count]
            ]
            conditions.extend(
                (
                    (f"top{count}_removed", top, count, top),
                    (f"random{count}_removed", random, count, top),
                    (
                        f"top{count}_only",
                        remaining,
                        count,
                        top,
                    ),
                    (
                        f"random{count}_only",
                        all_neurons[
                            ~torch.isin(all_neurons, random)
                        ],
                        count,
                        top,
                    ),
                )
            )
        for condition, neurons, reported_count, top in conditions:
            intervention = FunctionalIntervention(
                site=site,
                component="mlp_hidden",
                mode="patch",
                positions=(cfg.seq_len - 1,),
                neurons=tuple(int(item) for item in neurons),
            )
            patched, _ = run_instrumented_state(
                model,
                evaluation_state,
                loop_indices=(cfg.max_loops,),
                interventions=(intervention,),
                donor_trace=evaluation_old,
            )
            metrics = normalized_component_metrics(
                clean_logits=clean_logits,
                corrupt_logits=old_logits,
                patched_logits=patched,
                target=evaluation_batch.target,
                endpoint=evaluation_batch.endpoint,
            )
            rows.append(
                _metric_row(
                    metrics,
                    site=site,
                    block=site + 1,
                    condition=condition,
                    neuron_count=reported_count,
                    top_neurons=(
                        " ".join(str(int(item)) for item in top)
                        if condition.startswith("top")
                        else ""
                    ),
                )
            )
    return rows


@torch.no_grad()
def analyze(
    *,
    checkpoint: Path,
    lifespan_summary_path: Path,
    out_dir: Path,
    device: torch.device,
    calibration_size: int,
    validation_size: int,
    discovery_size: int,
    evaluation_size: int,
    ridge_values: Sequence[float],
    ranks: Sequence[int],
    mode_count: int,
    circuit_target_recovery: float,
    random_circuit_subsets: int,
    exhaustive_component_subsets: bool,
    mlp_top_ks: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    lifespan = json.loads(
        lifespan_summary_path.read_text(encoding="utf-8")
    )
    candidate = lifespan["candidate"]
    calibration = collect_matched_age_batch(
        model=model,
        cfg=cfg,
        candidate=candidate,
        batch_size=calibration_size,
        device=device,
        seed=seed,
    )
    validation = collect_matched_age_batch(
        model=model,
        cfg=cfg,
        candidate=candidate,
        batch_size=validation_size,
        device=device,
        seed=seed + 1,
    )
    ridge_rows = []
    maps = {}
    for ridge in ridge_values:
        affine = fit_initial_affine(calibration, ridge=ridge)
        maps[ridge] = affine
        state = validation.terminal.clone()
        state[:, -1:] = affine(state[:, -1:])
        _, metrics = evaluate_state(
            model=model,
            cfg=cfg,
            state=state,
            batch=validation,
        )
        ridge_rows.append(
            _metric_row(
                metrics,
                ridge=ridge,
                state_relative_mse=relative_state_mse(
                    state[:, -1],
                    validation.young[:, -1],
                ),
                update_frobenius_norm=float(
                    affine_update_matrix(affine).norm()
                ),
            )
        )
    best_ridge_row = min(
        ridge_rows,
        key=lambda row: (
            -float(row["accuracy"]),
            float(row["state_relative_mse"]),
            float(row["update_frobenius_norm"]),
        ),
    )
    best_ridge = float(best_ridge_row["ridge"])
    best_affine = maps[best_ridge]
    full_update = affine_update_matrix(best_affine)
    left, singular_values, right = torch.linalg.svd(
        full_update,
        full_matrices=False,
    )
    true_delta = (
        calibration.young[:, -1].float()
        - calibration.terminal[:, -1].float()
    )
    mean_delta = true_delta.mean(dim=0)
    centered_delta = true_delta - mean_delta
    _, age_singular_values, age_directions = torch.linalg.svd(
        centered_delta,
        full_matrices=False,
    )

    evaluation = collect_matched_age_batch(
        model=model,
        cfg=cfg,
        candidate=candidate,
        batch_size=evaluation_size,
        device=device,
        seed=seed + 3,
    )
    rank_rows = []
    generator = torch.Generator(device=device).manual_seed(seed + 4)
    allowed_ranks = sorted(
        {max(0, min(int(rank), cfg.d_model)) for rank in ranks}
        | {cfg.d_model}
    )
    for rank in allowed_ranks:
        top = truncated_update(
            left,
            singular_values,
            right,
            rank=rank,
        )
        random_update = random_rank_matched_update(
            singular_values,
            input_count=full_update.shape[0],
            output_count=full_update.shape[1],
            rank=rank,
            generator=generator,
        )
        age_update = age_subspace_update(
            full_update,
            mean_delta=mean_delta,
            directions=age_directions,
            rank=rank,
        )
        random_basis = torch.randn(
            cfg.d_model,
            rank,
            device=device,
            generator=generator,
        )
        random_directions = (
            torch.linalg.qr(random_basis, mode="reduced").Q.T
            if rank
            else age_directions[:0]
        )
        random_age_update = age_subspace_update(
            full_update,
            mean_delta=mean_delta,
            directions=random_directions,
            rank=rank,
        )
        for condition, update, mode in (
            ("operator_top_rank", top, "matched"),
            ("operator_complement", full_update - top, "matched"),
            ("operator_random_rank_matched", random_update, "matched"),
            ("age_subspace", age_update, "matched"),
            ("age_subspace_complement", full_update - age_update, "matched"),
            ("age_subspace_shuffled", age_update, "shuffled"),
            ("age_subspace_reverse", age_update, "reverse"),
            ("random_output_subspace", random_age_update, "matched"),
        ):
            state = apply_update_to_state(
                evaluation.terminal,
                update,
                mode=mode,
            )
            _, metrics = evaluate_state(
                model=model,
                cfg=cfg,
                state=state,
                batch=evaluation,
            )
            rank_rows.append(
                _metric_row(
                    metrics,
                    rank=rank,
                    condition=condition,
                    state_relative_mse=relative_state_mse(
                        state[:, -1],
                        evaluation.young[:, -1],
                    ),
                    retained_operator_energy=(
                        float(
                            singular_values[:rank].square().sum()
                            / singular_values.square().sum()
                        )
                        if rank
                        else 0.0
                    ),
                    retained_age_variance=(
                        float(
                            age_singular_values[:rank].square().sum()
                            / age_singular_values.square().sum()
                        )
                        if rank
                        else 0.0
                    ),
                )
            )
    full_state = apply_update_to_state(
        evaluation.terminal,
        full_update,
    )
    _, full_metrics = evaluate_state(
        model=model,
        cfg=cfg,
        state=full_state,
        batch=evaluation,
    )
    oracle_state = evaluation.terminal.clone()
    oracle_state[:, -1] = evaluation.young[:, -1]
    _, oracle_metrics = evaluate_state(
        model=model,
        cfg=cfg,
        state=oracle_state,
        batch=evaluation,
    )
    _, terminal_metrics = evaluate_state(
        model=model,
        cfg=cfg,
        state=evaluation.terminal,
        batch=evaluation,
    )

    minimal_rank = cfg.d_model
    for row in rank_rows:
        if (
            row["condition"] == "age_subspace"
            and float(row["accuracy"])
            >= min(0.90, float(full_metrics["accuracy"]) - 0.02)
        ):
            minimal_rank = int(row["rank"])
            break
    selected_update = age_subspace_update(
        full_update,
        mean_delta=mean_delta,
        directions=age_directions,
        rank=minimal_rank,
    )

    mode_rows = []
    evaluated_modes = min(mode_count, cfg.d_model)
    for mode_index in range(evaluated_modes):
        centered_mode = centered_age_mode_update(
            full_update,
            mean_delta=mean_delta,
            direction=age_directions[mode_index],
        )
        mean_update = torch.zeros_like(full_update)
        mean_update[-1] = mean_delta
        for condition, update in (
            ("age_mode_only", mean_update + centered_mode),
            ("leave_age_mode_out", full_update - centered_mode),
        ):
            state = apply_update_to_state(
                evaluation.terminal,
                update,
            )
            _, metrics = evaluate_state(
                model=model,
                cfg=cfg,
                state=state,
                batch=evaluation,
            )
            mode_rows.append(
                _metric_row(
                    metrics,
                    mode=mode_index,
                    condition=condition,
                    age_singular_value=float(
                        age_singular_values[mode_index]
                    ),
                    explained_age_variance=float(
                        age_singular_values[mode_index].square()
                        / age_singular_values.square().sum()
                    ),
                )
            )

    discovery = collect_matched_age_batch(
        model=model,
        cfg=cfg,
        candidate=candidate,
        batch_size=discovery_size,
        device=device,
        seed=seed + 2,
    )
    discovery_state = apply_update_to_state(
        discovery.terminal,
        full_update,
    )
    component_rows, _, old_trace, _, clean_trace = (
        component_localization_rows(
            model=model,
            cfg=cfg,
            batch=evaluation,
            rejuvenated_state=full_state,
        )
    )
    selected_nodes, circuit_search_rows = discover_component_circuit(
        model=model,
        cfg=cfg,
        batch=discovery,
        rejuvenated_state=discovery_state,
        target_recovery=circuit_target_recovery,
    )
    circuit_validation_rows = validate_component_circuit(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=full_state,
        selected=selected_nodes,
        random_subsets=random_circuit_subsets,
        exhaustive_subsets=exhaustive_component_subsets,
        seed=seed + 5,
    )
    oracle_circuit_validation_rows = validate_component_circuit(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=oracle_state,
        selected=selected_nodes,
        random_subsets=0,
        exhaustive_subsets=False,
        seed=seed + 15,
    )
    qkv_rows = qkv_mediation_rows(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=full_state,
    )
    oracle_qkv_rows = qkv_mediation_rows(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=oracle_state,
    )
    oracle_logits, oracle_trace = run_instrumented_state(
        model,
        oracle_state,
        loop_indices=(cfg.max_loops,),
    )
    del oracle_logits
    attention_rows = attention_function_rows(
        cfg=cfg,
        batch=evaluation,
        traces={
            "terminal": old_trace,
            "rejuvenated": clean_trace,
            "oracle_young": oracle_trace,
        },
    )
    stage_rows = stage_readout_rows(
        model=model,
        batch=evaluation,
        initial_states={
            "terminal": evaluation.terminal,
            "rejuvenated": full_state,
            "oracle_young": oracle_state,
        },
        traces={
            "terminal": old_trace,
            "rejuvenated": clean_trace,
            "oracle_young": oracle_trace,
        },
    )
    neuron_rows = mlp_neuron_rows(
        model=model,
        cfg=cfg,
        discovery_batch=discovery,
        discovery_state=discovery_state,
        evaluation_batch=evaluation,
        evaluation_state=full_state,
        top_ks=mlp_top_ks,
        seed=seed + 6,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ridge_selection_rows.csv", ridge_rows)
    _write_csv(out_dir / "rank_intervention_rows.csv", rank_rows)
    _write_csv(out_dir / "singular_mode_rows.csv", mode_rows)
    _write_csv(out_dir / "component_localization_rows.csv", component_rows)
    _write_csv(out_dir / "circuit_search_rows.csv", circuit_search_rows)
    _write_csv(
        out_dir / "circuit_validation_rows.csv",
        circuit_validation_rows,
    )
    _write_csv(
        out_dir / "oracle_circuit_transfer_rows.csv",
        oracle_circuit_validation_rows,
    )
    _write_csv(out_dir / "qkv_mediation_rows.csv", qkv_rows)
    _write_csv(
        out_dir / "oracle_qkv_mediation_rows.csv",
        oracle_qkv_rows,
    )
    _write_csv(out_dir / "attention_function_rows.csv", attention_rows)
    _write_csv(out_dir / "stage_readout_rows.csv", stage_rows)
    _write_csv(out_dir / "mlp_neuron_rows.csv", neuron_rows)
    matrix_artifact = out_dir / "optimal_init_matrix.pt"
    torch.save(
        {
            "format_version": 1,
            "checkpoint": str(checkpoint),
            "model_name": "D8_L8_seed1",
            "candidate": candidate,
            "selected_ridge": best_ridge,
            "full_update_matrix": full_update.cpu(),
            "homogeneous_matrix": homogeneous_matrix_from_update(
                full_update
            ).cpu(),
            "svd_left": left.cpu(),
            "singular_values": singular_values.cpu(),
            "svd_right": right.cpu(),
            "mean_age_delta": mean_delta.cpu(),
            "age_singular_values": age_singular_values.cpu(),
            "age_directions": age_directions.cpu(),
            "minimal_sufficient_rank": minimal_rank,
            "minimal_rank_update_matrix": selected_update.cpu(),
        },
        matrix_artifact,
    )
    circuit_only = next(
        row
        for row in circuit_validation_rows
        if row["condition"] == "circuit_only"
    )
    complement_only = next(
        row
        for row in circuit_validation_rows
        if row["condition"] == "complement_only"
    )
    oracle_circuit_only = next(
        row
        for row in oracle_circuit_validation_rows
        if row["condition"] == "circuit_only"
    )
    oracle_complement_only = next(
        row
        for row in oracle_circuit_validation_rows
        if row["condition"] == "complement_only"
    )
    random_recoveries = [
        float(row["recovery"])
        for row in circuit_validation_rows
        if str(row["condition"]).startswith("random_circuit_only")
    ]
    exhaustive_rows = [
        row
        for row in circuit_validation_rows
        if str(row["condition"]).startswith("exhaustive_subset")
    ]
    selected_labels = {node.label for node in selected_nodes}
    faithful_alternatives = [
        row
        for row in exhaustive_rows
        if float(row["recovery"]) >= 0.90
        and float(row["accuracy"]) >= 0.90
    ]
    disjoint_rows = [
        row
        for row in exhaustive_rows
        if selected_labels.isdisjoint(str(row["nodes"]).split(";"))
    ]
    summary = {
        "model": "D8_L8_seed1",
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": {
            "node_count": cfg.node_count,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_mlp": cfg.d_mlp,
            "physical_blocks": cfg.n_layers,
            "trained_loops": cfg.max_loops,
            "evaluated_effective_loop": cfg.max_loops + 1,
        },
        "candidate": candidate,
        "selected_ridge": best_ridge,
        "matrix_shape": [cfg.d_model + 1, cfg.d_model + 1],
        "matrix_artifact": matrix_artifact.name,
        "terminal_behavior": terminal_metrics,
        "oracle_young_answer_behavior": oracle_metrics,
        "full_matrix_behavior": full_metrics,
        "minimal_sufficient_rank": minimal_rank,
        "minimal_rank_behavior": next(
            row
            for row in rank_rows
            if row["condition"] == "age_subspace"
            and int(row["rank"]) == minimal_rank
        ),
        "operator_singular_energy_at_minimal_rank": float(
            singular_values[:minimal_rank].square().sum()
            / singular_values.square().sum()
        ),
        "age_variance_at_minimal_rank": float(
            age_singular_values[:minimal_rank].square().sum()
            / age_singular_values.square().sum()
        ),
        "selected_circuit_nodes": [
            node.label for node in selected_nodes
        ],
        "circuit_only": circuit_only,
        "complement_only": complement_only,
        "oracle_young_circuit_transfer": {
            "circuit_only": oracle_circuit_only,
            "complement_only": oracle_complement_only,
        },
        "random_circuit_recovery_mean": (
            float(np.mean(random_recoveries))
            if random_recoveries
            else None
        ),
        "random_circuit_recovery_max": (
            float(np.max(random_recoveries))
            if random_recoveries
            else None
        ),
        "alternative_circuit_search": {
            "subset_size": len(selected_nodes),
            "subsets_evaluated": len(exhaustive_rows),
            "faithful_subset_count": len(faithful_alternatives),
            "best_disjoint_recovery": (
                max(float(row["recovery"]) for row in disjoint_rows)
                if disjoint_rows
                else None
            ),
            "best_disjoint_accuracy": (
                max(float(row["accuracy"]) for row in disjoint_rows)
                if disjoint_rows
                else None
            ),
            "top_faithful_subsets": [
                {
                    "nodes": row["nodes"],
                    "accuracy": row["accuracy"],
                    "recovery": row["recovery"],
                }
                for row in sorted(
                    faithful_alternatives,
                    key=lambda item: float(item["recovery"]),
                    reverse=True,
                )[:10]
            ],
        },
        "sample_sizes": {
            "calibration": calibration_size,
            "ridge_validation": validation_size,
            "circuit_discovery": discovery_size,
            "final_evaluation": evaluation_size,
        },
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lifespan-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--discovery-size", type=int, default=256)
    parser.add_argument("--evaluation-size", type=int, default=512)
    parser.add_argument(
        "--ridge-values",
        type=float,
        nargs="+",
        default=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 32, 64, 128),
    )
    parser.add_argument("--mode-count", type=int, default=64)
    parser.add_argument("--circuit-target-recovery", type=float, default=0.90)
    parser.add_argument("--random-circuit-subsets", type=int, default=32)
    parser.add_argument(
        "--exhaustive-component-subsets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mlp-top-ks",
        type=int,
        nargs="+",
        default=(8, 16, 32, 64, 128, 256, 512),
    )
    parser.add_argument("--seed", type=int, default=20267101)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    summary = analyze(
        checkpoint=args.checkpoint,
        lifespan_summary_path=args.lifespan_summary,
        out_dir=args.out_dir,
        device=device,
        calibration_size=args.calibration_size,
        validation_size=args.validation_size,
        discovery_size=args.discovery_size,
        evaluation_size=args.evaluation_size,
        ridge_values=args.ridge_values,
        ranks=args.ranks,
        mode_count=args.mode_count,
        circuit_target_recovery=args.circuit_target_recovery,
        random_circuit_subsets=args.random_circuit_subsets,
        exhaustive_component_subsets=args.exhaustive_component_subsets,
        mlp_top_ks=args.mlp_top_ks,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
