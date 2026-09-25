from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.typed_relation_composition import (
    RelationBatch,
    TypedRelationConfig,
    TypedRelationModel,
    TypedRelationWorkspaceCell,
    make_relation_batch,
    relation_visibility,
)
from reasoning_loop.typed_relation_train import pick_device


HeadAblation = Mapping[int, set[int]]


def manual_attention(
    cell: TypedRelationWorkspaceCell,
    workspace: torch.Tensor,
    static_edges: torch.Tensor,
    visible_edges: torch.Tensor,
    *,
    ablate_heads: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = cell.attention
    query = cell.norm_attention(workspace).unsqueeze(1)
    edge_values = cell.norm_attention(static_edges)
    key_value = torch.cat((query, edge_values), dim=1)
    d_model = query.shape[-1]
    n_heads = attention.num_heads
    head_dim = d_model // n_heads
    q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3, dim=0)
    if attention.in_proj_bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = attention.in_proj_bias.chunk(3, dim=0)

    def split_heads(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            value.shape[0],
            value.shape[1],
            n_heads,
            head_dim,
        ).transpose(1, 2)

    q = split_heads(F.linear(query, q_weight, q_bias))
    k = split_heads(F.linear(key_value, k_weight, k_bias))
    v = split_heads(F.linear(key_value, v_weight, v_bias))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    padding_mask = torch.cat(
        (
            torch.zeros(
                (workspace.shape[0], 1),
                dtype=torch.bool,
                device=workspace.device,
            ),
            ~visible_edges,
        ),
        dim=1,
    )
    scores = scores.masked_fill(padding_mask[:, None, None, :], -torch.inf)
    weights = scores.softmax(dim=-1)
    contexts = torch.matmul(weights, v)
    if ablate_heads:
        contexts = contexts.clone()
        for head in ablate_heads:
            if not 0 <= head < n_heads:
                raise ValueError("head index is out of range")
            contexts[:, head] = 0.0
    joined = contexts.transpose(1, 2).reshape(
        workspace.shape[0],
        1,
        d_model,
    )
    return attention.out_proj(joined).squeeze(1), weights, contexts


@torch.no_grad()
def run_intervened(
    model: TypedRelationModel,
    batch: RelationBatch,
    visibility: torch.Tensor,
    *,
    initial_workspace: torch.Tensor | None = None,
    start_loop: int = 0,
    stop_loop: int | None = None,
    zero_attention_loops: set[int] | None = None,
    zero_mlp_loops: set[int] | None = None,
    ablate_heads: HeadAblation | None = None,
    cell_schedule: Sequence[int] | None = None,
    return_trace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    final_loop = model.cfg.loops if stop_loop is None else stop_loop
    if not 0 <= start_loop < final_loop <= model.cfg.loops:
        raise ValueError("loop range must be nonempty and configured")
    expected = (
        batch.batch_size,
        model.cfg.loops,
        2 * model.cfg.node_count,
    )
    if visibility.shape != expected:
        raise ValueError("visibility has the wrong shape")
    static_edges = model.encode_edges(batch)
    workspace = (
        model._initial_workspace(batch.query)
        if initial_workspace is None
        else initial_workspace
    )
    if workspace.shape != (batch.batch_size, model.cfg.d_model):
        raise ValueError("initial_workspace has the wrong shape")
    zero_attention_loops = zero_attention_loops or set()
    zero_mlp_loops = zero_mlp_loops or set()
    ablate_heads = ablate_heads or {}
    if cell_schedule is not None and len(cell_schedule) != final_loop - start_loop:
        raise ValueError("cell_schedule must cover every executed loop")
    logits: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    trace: dict[str, torch.Tensor] = {}
    for loop_index in range(start_loop, final_loop):
        schedule_offset = loop_index - start_loop
        cell_index = (
            loop_index
            if cell_schedule is None
            else int(cell_schedule[schedule_offset])
        )
        cell = model.cell_for_loop(cell_index)
        attention_out, weights, contexts = manual_attention(
            cell,
            workspace,
            static_edges,
            visibility[:, loop_index],
            ablate_heads=ablate_heads.get(loop_index),
        )
        if loop_index in zero_attention_loops:
            attention_out = torch.zeros_like(attention_out)
        residual_mid = workspace + attention_out
        mlp_out = cell.mlp(cell.norm_mlp(residual_mid))
        if loop_index in zero_mlp_loops:
            mlp_out = torch.zeros_like(mlp_out)
        workspace = residual_mid + mlp_out
        states.append(workspace)
        logits.append(model.readout(model.readout_norm(workspace)))
        if return_trace:
            prefix = f"loop{loop_index}"
            trace[f"{prefix}.attention_out"] = attention_out.detach()
            trace[f"{prefix}.attention_weights"] = weights.detach()
            trace[f"{prefix}.head_contexts"] = contexts.detach()
            trace[f"{prefix}.mlp_out"] = mlp_out.detach()
            trace[f"{prefix}.workspace_out"] = workspace.detach()
    return torch.stack(logits, dim=1), torch.stack(states, dim=1), trace


def _mean_logit_margin(logits: torch.Tensor, target: torch.Tensor) -> float:
    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    distractor_logits = logits.masked_fill(
        torch.nn.functional.one_hot(
            target,
            num_classes=logits.shape[1],
        ).bool(),
        -torch.inf,
    )
    return float((target_logit - distractor_logits.max(dim=1).values).mean().item())


@torch.no_grad()
def enumerate_component_circuits(
    model: TypedRelationModel,
    batch: RelationBatch,
    visibility: torch.Tensor,
    *,
    train_loops: int,
    tie_masks_across_loops: bool,
) -> list[dict[str, Any]]:
    if not 1 <= train_loops <= model.cfg.loops:
        raise ValueError("train_loops must be configured")
    bits_per_loop = model.cfg.n_heads + 1
    bit_count = bits_per_loop if tie_masks_across_loops else train_loops * bits_per_loop
    all_heads = set(range(model.cfg.n_heads))
    target = batch.targets[:, 1]
    rows: list[dict[str, Any]] = []
    for bits in itertools.product((False, True), repeat=bit_count):
        if tie_masks_across_loops:
            masks = [bits] * train_loops
        else:
            masks = [
                bits[index * bits_per_loop : (index + 1) * bits_per_loop]
                for index in range(train_loops)
            ]
        kept_heads = [
            [head for head in range(model.cfg.n_heads) if mask[head]]
            for mask in masks
        ]
        kept_mlps = [bool(mask[-1]) for mask in masks]
        ablate_heads = {
            loop_index: all_heads - set(heads)
            for loop_index, heads in enumerate(kept_heads)
        }
        zero_mlp_loops = {
            loop_index
            for loop_index, keep in enumerate(kept_mlps)
            if not keep
        }
        logits, _, _ = run_intervened(
            model,
            batch,
            visibility,
            stop_loop=train_loops,
            ablate_heads=ablate_heads,
            zero_mlp_loops=zero_mlp_loops,
        )
        endpoint_logits = logits[:, -1]
        effective_nodes = sum(map(len, kept_heads)) + sum(kept_mlps)
        if model.shared_cell is not None:
            parameter_nodes = len(set().union(*map(set, kept_heads))) + int(
                any(kept_mlps)
            )
        else:
            parameter_nodes = effective_nodes
        rows.append(
            {
                "n_heads": model.cfg.n_heads,
                "kept_heads": kept_heads,
                "kept_mlps": kept_mlps,
                "effective_nodes": effective_nodes,
                "parameter_nodes": parameter_nodes,
                "endpoint_accuracy": float(
                    endpoint_logits.argmax(dim=-1)
                    .eq(target)
                    .float()
                    .mean()
                    .item()
                ),
                "mean_logit_margin": _mean_logit_margin(endpoint_logits, target),
            }
        )
    return rows


@torch.no_grad()
def cell_schedule_table(
    model: TypedRelationModel,
    batch: RelationBatch,
    visibility: torch.Tensor,
    *,
    train_loops: int,
) -> list[dict[str, Any]]:
    if train_loops != 2:
        raise ValueError("cell schedule controls currently require two loops")
    schedules = (
        ((0, 1), "normal"),
        ((0, 0), "repeat_first"),
        ((1, 1), "repeat_second"),
        ((1, 0), "swapped"),
    )
    target = batch.targets[:, 1]
    rows: list[dict[str, Any]] = []
    for schedule, name in schedules:
        logits, _, _ = run_intervened(
            model,
            batch,
            visibility,
            stop_loop=train_loops,
            cell_schedule=schedule,
        )
        endpoint_logits = logits[:, -1]
        rows.append(
            {
                "schedule_name": name,
                "cell_schedule": list(schedule),
                "endpoint_accuracy": float(
                    endpoint_logits.argmax(dim=-1)
                    .eq(target)
                    .float()
                    .mean()
                    .item()
                ),
                "mean_logit_margin": _mean_logit_margin(endpoint_logits, target),
            }
        )
    return rows


def relation_attention_metrics(
    attention_by_loop: torch.Tensor,
    *,
    batch: RelationBatch,
    node_count: int,
) -> dict[str, torch.Tensor]:
    if attention_by_loop.ndim != 5:
        raise ValueError(
            "attention must have [loop, batch, head, query, key] axes"
        )
    if attention_by_loop.shape[0] < 2 or attention_by_loop.shape[1] != batch.batch_size:
        raise ValueError("attention does not cover the two stages")
    if attention_by_loop.shape[-2] != 1:
        raise ValueError("workspace attention must have one query token")
    weights = attention_by_loop[:2, :, :, 0]
    expected_keys = 1 + 2 * node_count
    if weights.shape[-1] != expected_keys:
        raise ValueError("attention key axis does not match typed edges")
    first = batch.targets[:, 0]
    f_then_g = batch.composition_order == "f_then_g"
    correct_indices = torch.stack(
        (
            1 + batch.query + (0 if f_then_g else node_count),
            1 + first + (node_count if f_then_g else 0),
        ),
        dim=0,
    )
    wrong_relation_indices = torch.stack(
        (
            1 + batch.query + (node_count if f_then_g else 0),
            1 + first + (0 if f_then_g else node_count),
        ),
        dim=0,
    )
    gather_correct = correct_indices[:, :, None, None].expand(
        2,
        batch.batch_size,
        weights.shape[2],
        1,
    )
    gather_wrong = wrong_relation_indices[:, :, None, None].expand_as(
        gather_correct
    )
    correct = weights.gather(-1, gather_correct).squeeze(-1)
    wrong_relation = weights.gather(-1, gather_wrong).squeeze(-1)
    f_range = slice(1, 1 + node_count)
    g_range = slice(1 + node_count, 1 + 2 * node_count)
    relation_ranges = (f_range, g_range) if f_then_g else (g_range, f_range)
    incorrect_means: list[torch.Tensor] = []
    relation_totals: list[torch.Tensor] = []
    wrong_totals: list[torch.Tensor] = []
    for loop_index, relation_slice in enumerate(relation_ranges):
        relation_weight = weights[loop_index, :, :, relation_slice]
        relation_total = relation_weight.sum(dim=-1)
        relation_totals.append(relation_total)
        incorrect_means.append(
            (relation_total - correct[loop_index]) / (node_count - 1)
        )
        wrong_slice = relation_ranges[1 - loop_index]
        wrong_totals.append(
            weights[loop_index, :, :, wrong_slice].sum(dim=-1)
        )
    incorrect = torch.stack(incorrect_means, dim=0)
    return {
        "correct_relation_attention": correct.mean(dim=1),
        "incorrect_same_relation_attention": incorrect.mean(dim=1),
        "wrong_relation_attention": wrong_relation.mean(dim=1),
        "correct_relation_total_attention": torch.stack(
            relation_totals,
            dim=0,
        ).mean(dim=1),
        "wrong_relation_total_attention": torch.stack(
            wrong_totals,
            dim=0,
        ).mean(dim=1),
        "typed_edge_selectivity": (correct - incorrect).mean(dim=1),
    }


def _accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    readout_index: int = -1,
) -> float:
    return float(
        logits[:, readout_index]
        .argmax(dim=-1)
        .eq(target)
        .float()
        .mean()
        .item()
    )


@torch.no_grad()
def phase_circuit_table(
    model: TypedRelationModel,
    donor: RelationBatch,
    receiver: RelationBatch,
    *,
    train_loops: int,
) -> dict[str, Any]:
    if train_loops != 2 or model.cfg.loops < 2:
        raise ValueError("typed phase analysis requires exactly two trained loops")
    if donor.batch_size != receiver.batch_size:
        raise ValueError("donor and receiver batch sizes must match")
    if donor.composition_order != receiver.composition_order:
        raise ValueError("donor and receiver must share composition_order")
    device = donor.query.device
    full = relation_visibility(
        "full",
        batch_size=donor.batch_size,
        loops=model.cfg.loops,
        node_count=model.cfg.node_count,
        device=device,
        composition_order=donor.composition_order,
    )
    aligned = relation_visibility(
        "aligned",
        batch_size=donor.batch_size,
        loops=model.cfg.loops,
        node_count=model.cfg.node_count,
        device=device,
        composition_order=donor.composition_order,
    )
    swapped = relation_visibility(
        "swapped",
        batch_size=donor.batch_size,
        loops=model.cfg.loops,
        node_count=model.cfg.node_count,
        device=device,
        composition_order=donor.composition_order,
    )
    baseline_logits, donor_states, trace = run_intervened(
        model,
        donor,
        full,
        return_trace=True,
    )
    endpoint = donor.targets[:, 1]
    baseline_endpoint = _accuracy(
        baseline_logits,
        endpoint,
        readout_index=1,
    )
    component_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for loop_index in range(train_loops):
        for component in ("attention", "mlp"):
            kwargs = (
                {"zero_attention_loops": {loop_index}}
                if component == "attention"
                else {"zero_mlp_loops": {loop_index}}
            )
            logits, _, _ = run_intervened(
                model,
                donor,
                full,
                **kwargs,
            )
            accuracy = _accuracy(logits, endpoint, readout_index=1)
            component_rows.append(
                {
                    "loop": loop_index + 1,
                    "component": component,
                    "endpoint_accuracy": accuracy,
                    "endpoint_drop": baseline_endpoint - accuracy,
                }
            )
        for head in range(model.cfg.n_heads):
            logits, _, _ = run_intervened(
                model,
                donor,
                full,
                ablate_heads={loop_index: {head}},
            )
            accuracy = _accuracy(logits, endpoint, readout_index=1)
            head_rows.append(
                {
                    "loop": loop_index + 1,
                    "head": head,
                    "endpoint_accuracy": accuracy,
                    "endpoint_drop": baseline_endpoint - accuracy,
                }
            )

    attention = relation_attention_metrics(
        torch.stack(
            [
                trace["loop0.attention_weights"],
                trace["loop1.attention_weights"],
            ],
            dim=0,
        ),
        batch=donor,
        node_count=model.cfg.node_count,
    )
    aligned_logits, _, _ = run_intervened(model, donor, aligned)
    swapped_logits, _, _ = run_intervened(model, donor, swapped)

    donor_workspace = donor_states[:, 0]
    second_mapping = (
        receiver.g if donor.composition_order == "f_then_g" else receiver.f
    )
    hybrid_target = second_mapping.gather(
        1,
        donor.targets[:, 0, None],
    ).squeeze(1)

    def hybrid_continue(**kwargs: Any) -> torch.Tensor:
        logits, _, _ = run_intervened(
            model,
            receiver,
            full,
            initial_workspace=donor_workspace,
            start_loop=1,
            stop_loop=2,
            **kwargs,
        )
        return logits

    hybrid_logits = hybrid_continue()
    hybrid_accuracy = _accuracy(hybrid_logits, hybrid_target)
    hybrid_component_rows: list[dict[str, Any]] = []
    for component in ("attention", "mlp"):
        kwargs = (
            {"zero_attention_loops": {1}}
            if component == "attention"
            else {"zero_mlp_loops": {1}}
        )
        accuracy = _accuracy(hybrid_continue(**kwargs), hybrid_target)
        hybrid_component_rows.append(
            {
                "component": component,
                "hybrid_accuracy": accuracy,
                "hybrid_drop": hybrid_accuracy - accuracy,
            }
        )
    hybrid_head_rows: list[dict[str, Any]] = []
    for head in range(model.cfg.n_heads):
        accuracy = _accuracy(
            hybrid_continue(ablate_heads={1: {head}}),
            hybrid_target,
        )
        hybrid_head_rows.append(
            {
                "head": head,
                "hybrid_accuracy": accuracy,
                "hybrid_drop": hybrid_accuracy - accuracy,
            }
        )

    per_loop_max_drop = {
        loop: max(
            float(row["endpoint_drop"])
            for row in component_rows
            if int(row["loop"]) == loop
        )
        for loop in (1, 2)
    }
    extra_loop_accuracy = (
        _accuracy(baseline_logits, endpoint, readout_index=2)
        if model.cfg.loops >= 3
        else None
    )
    typed_selectivity = attention["typed_edge_selectivity"]
    scorecard = {
        "behavior_gate": (
            baseline_endpoint >= 0.99
            and _accuracy(
                baseline_logits,
                donor.targets[:, 0],
                readout_index=0,
            )
            >= 0.99
        ),
        "both_loops_causally_necessary": min(per_loop_max_drop.values()) >= 0.20,
        "relation_order_gate": (
            _accuracy(aligned_logits, endpoint, readout_index=1) >= 0.99
            and _accuracy(swapped_logits, endpoint, readout_index=1)
            <= 1.0 / model.cfg.node_count + 0.05
        ),
        "attention_phase_specialization": bool(
            typed_selectivity.max(dim=1).values.min().item() >= 0.10
        ),
        "stage_boundary_gate": hybrid_accuracy >= 0.90,
        "stable_operator_repeat_rejected": (
            extra_loop_accuracy is not None
            and baseline_endpoint - extra_loop_accuracy >= 0.50
        ),
        "per_loop_max_component_drop": per_loop_max_drop,
    }
    return {
        "examples": donor.batch_size,
        "baseline_endpoint_accuracy": baseline_endpoint,
        "loop1_first_relation_accuracy": _accuracy(
            baseline_logits,
            donor.targets[:, 0],
            readout_index=0,
        ),
        "extra_loop_endpoint_accuracy": extra_loop_accuracy,
        "aligned_endpoint_accuracy": _accuracy(
            aligned_logits,
            endpoint,
            readout_index=1,
        ),
        "swapped_endpoint_accuracy": _accuracy(
            swapped_logits,
            endpoint,
            readout_index=1,
        ),
        "component_rows": component_rows,
        "head_rows": head_rows,
        "attention": {
            name: value.detach().cpu().tolist()
            for name, value in attention.items()
        },
        "hybrid_accuracy": hybrid_accuracy,
        "hybrid_component_rows": hybrid_component_rows,
        "hybrid_head_rows": hybrid_head_rows,
        "scorecard": scorecard,
    }


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[TypedRelationModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = TypedRelationModel(
        TypedRelationConfig.from_dict(payload["config"])
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.no_grad()
def analyze_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    examples: int,
    seed: int,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    composition_order = payload.get("composition_order", "f_then_g")
    donor = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    receiver = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "seed": int(payload["seed"]),
        "step": int(payload["step"]),
        "architecture": model.cfg.architecture,
        "objective": payload["objective"],
        "train_visibility": payload["train_visibility"],
        "composition_order": composition_order,
        "phase": phase_circuit_table(
            model,
            donor,
            receiver,
            train_loops=int(payload["train_loops"]),
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causally localize typed-relation phase-specific reuse."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=91_017)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze_checkpoint(
        args.checkpoint,
        args.out_dir,
        device=pick_device(args.device),
        examples=args.examples,
        seed=args.seed,
    )
    print(
        json.dumps(result["phase"]["scorecard"], indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
