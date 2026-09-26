from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    _all_targets,
    checkpoint_loss_mode,
    fixed_depth_batch,
    load_checkpoint,
    logit_difference,
    normalized_recovery,
    paired_fixed_depth_batch,
    target_margin,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    MultiHeadSelfAttention,
    TransformerBlock,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_temporal_intervention import (
    cache_raw_states,
    logits_from_raw_state,
    replace_answer_state,
)


Component = Literal[
    "block_input",
    "q",
    "k",
    "v",
    "attention_pattern",
    "head_context",
    "attention_out",
    "residual_mid",
    "mlp_hidden",
    "mlp_out",
]


@dataclass(frozen=True)
class FunctionalIntervention:
    site: int
    component: Component
    mode: Literal["zero", "patch", "onehot"]
    positions: tuple[int, ...] | None = None
    source_positions: tuple[int, ...] | None = None
    heads: tuple[int, ...] | None = None
    neurons: tuple[int, ...] | None = None
    dynamic_source_positions: torch.Tensor | None = None
    renormalize: bool = False


@dataclass
class FunctionalSiteTrace:
    loop_index: int
    block_index: int
    hidden_in: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention_pattern: torch.Tensor
    head_context: torch.Tensor
    attention_out: torch.Tensor
    residual_mid: torch.Tensor
    mlp_hidden: torch.Tensor
    mlp_out: torch.Tensor
    hidden_out: torch.Tensor


@dataclass
class FunctionalTrace:
    sites: list[FunctionalSiteTrace]
    logits_by_loop: torch.Tensor


def explicit_depth_position_groups(node_count: int) -> dict[str, tuple[int, ...]]:
    if node_count < 1:
        raise ValueError("node_count must be positive")
    edge = tuple(1 + 3 * node for node in range(node_count))
    source = tuple(position + 1 for position in edge)
    destination = tuple(position + 2 for position in edge)
    query = 1 + 3 * node_count
    return {
        "edge_marker": edge,
        "source": source,
        "destination": destination,
        "query": (query,),
        "start": (query + 1,),
        "depth": (query + 2,),
        "query_metadata": (query, query + 1, query + 2),
        "answer": (query + 3,),
        "graph": tuple(
            position
            for triple in zip(edge, source, destination, strict=True)
            for position in triple
        ),
    }


def paired_graph_batch(
    cfg: GraphPathConfig,
    batch_size: int,
    device: torch.device,
    *,
    path_positions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create two different graphs with the same start and fixed-depth query."""
    start = torch.randint(0, cfg.node_count, (batch_size,), device=device)
    tokens_a, targets_a, successors_a, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        start=start,
    )
    tokens_b, targets_b, successors_b, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        start=start,
    )
    identical = successors_a.eq(successors_b).all(dim=1)
    while bool(identical.any()):
        count = int(identical.sum())
        replacement, _, replacement_successors, _ = fixed_depth_batch(
            cfg,
            count,
            device,
            path_positions=path_positions,
            start=start[identical],
        )
        tokens_b[identical] = replacement
        successors_b[identical] = replacement_successors
        # Recompute the targets from the replacement graph.
        current = start[identical]
        for position in range(path_positions):
            current = replacement_successors.gather(
                1, current[:, None]
            ).squeeze(1)
            targets_b[identical, position] = current
        identical = successors_a.eq(successors_b).all(dim=1)
    return tokens_a, targets_a, tokens_b, targets_b


def _selected(
    interventions: Sequence[FunctionalIntervention],
    *,
    site: int,
    component: Component,
) -> list[FunctionalIntervention]:
    return [
        item
        for item in interventions
        if item.site == site and item.component == component
    ]


def _validate_patch(
    item: FunctionalIntervention,
    donor: torch.Tensor | None,
) -> None:
    if item.mode == "patch" and donor is None:
        raise ValueError("patch intervention requires a donor trace")


def _apply_head_tensor(
    value: torch.Tensor,
    *,
    site: int,
    component: Component,
    interventions: Sequence[FunctionalIntervention],
    donor: torch.Tensor | None,
) -> torch.Tensor:
    """Intervene on [batch, head, position, feature] tensors."""
    chosen = _selected(interventions, site=site, component=component)
    if not chosen:
        return value
    result = value.clone()
    for item in chosen:
        _validate_patch(item, donor)
        if item.mode == "onehot":
            raise ValueError(f"onehot mode is only valid for attention_pattern")
        if item.source_positions is not None or item.dynamic_source_positions is not None:
            raise ValueError(f"{component} has no attention-source axis")
        if item.neurons is not None or item.renormalize:
            raise ValueError(f"invalid {component} intervention fields")
        heads = range(result.shape[1]) if item.heads is None else item.heads
        positions = (
            range(result.shape[2]) if item.positions is None else item.positions
        )
        head_index = torch.as_tensor(tuple(heads), device=result.device)
        position_index = torch.as_tensor(tuple(positions), device=result.device)
        if bool((head_index < 0).any()) or bool(
            (head_index >= result.shape[1]).any()
        ):
            raise ValueError("head index is out of range")
        if bool((position_index < 0).any()) or bool(
            (position_index >= result.shape[2]).any()
        ):
            raise ValueError("position index is out of range")
        replacement = 0.0 if item.mode == "zero" else donor
        if item.mode == "zero":
            result[:, head_index[:, None], position_index[None, :], :] = 0.0
        else:
            assert isinstance(replacement, torch.Tensor)
            result[:, head_index[:, None], position_index[None, :], :] = (
                replacement[
                    :, head_index[:, None], position_index[None, :], :
                ]
            )
    return result


def _apply_attention_pattern(
    value: torch.Tensor,
    *,
    site: int,
    interventions: Sequence[FunctionalIntervention],
    donor: torch.Tensor | None,
) -> torch.Tensor:
    chosen = _selected(
        interventions, site=site, component="attention_pattern"
    )
    if not chosen:
        return value
    result = value.clone()
    for item in chosen:
        _validate_patch(item, donor)
        if item.neurons is not None:
            raise ValueError("attention patterns do not have neurons")
        if (
            item.source_positions is not None
            and item.dynamic_source_positions is not None
        ):
            raise ValueError("use fixed or dynamic source positions, not both")
        heads = (
            tuple(range(result.shape[1]))
            if item.heads is None
            else item.heads
        )
        positions = (
            tuple(range(result.shape[2]))
            if item.positions is None
            else item.positions
        )
        if item.dynamic_source_positions is not None:
            sources = item.dynamic_source_positions.to(result.device)
            if sources.ndim != 2 or sources.shape[0] != result.shape[0]:
                raise ValueError(
                    "dynamic source positions must have shape [batch, count]"
                )
            if bool((sources < 0).any()) or bool(
                (sources >= result.shape[3]).any()
            ):
                raise ValueError("dynamic source position is out of range")
            for head in heads:
                for position in positions:
                    batch = torch.arange(result.shape[0], device=result.device)
                    if item.mode == "onehot":
                        if sources.shape[1] != 1:
                            raise ValueError(
                                "onehot requires exactly one source per example"
                            )
                        result[:, head, position, :] = 0.0
                        result[batch, head, position, sources[:, 0]] = 1.0
                        continue
                    for column in range(sources.shape[1]):
                        source = sources[:, column]
                        if item.mode == "zero":
                            result[batch, head, position, source] = 0.0
                        else:
                            assert donor is not None
                            result[batch, head, position, source] = donor[
                                batch, head, position, source
                            ]
        else:
            sources = (
                tuple(range(result.shape[3]))
                if item.source_positions is None
                else item.source_positions
            )
            for head in heads:
                for position in positions:
                    if item.mode == "onehot":
                        if len(sources) != 1:
                            raise ValueError(
                                "onehot requires exactly one fixed source"
                            )
                        result[:, head, position, :] = 0.0
                        result[:, head, position, sources[0]] = 1.0
                        continue
                    for source in sources:
                        if item.mode == "zero":
                            result[:, head, position, source] = 0.0
                        else:
                            assert donor is not None
                            result[:, head, position, source] = donor[
                                :, head, position, source
                            ]
        if item.renormalize and item.mode != "onehot":
            for head in heads:
                for position in positions:
                    denominator = result[:, head, position].sum(
                        dim=-1, keepdim=True
                    )
                    if bool(denominator.le(0).any()):
                        raise ValueError("cannot renormalize an empty attention row")
                    result[:, head, position] /= denominator
    return result


def _apply_token_tensor(
    value: torch.Tensor,
    *,
    site: int,
    component: Component,
    interventions: Sequence[FunctionalIntervention],
    donor: torch.Tensor | None,
) -> torch.Tensor:
    chosen = _selected(interventions, site=site, component=component)
    if not chosen:
        return value
    result = value.clone()
    for item in chosen:
        _validate_patch(item, donor)
        if item.mode == "onehot":
            raise ValueError(f"onehot mode is only valid for attention_pattern")
        if (
            item.heads is not None
            or item.source_positions is not None
            or item.dynamic_source_positions is not None
            or item.neurons is not None
            or item.renormalize
        ):
            raise ValueError(f"invalid fields for {component}")
        positions = (
            tuple(range(result.shape[1]))
            if item.positions is None
            else item.positions
        )
        for position in positions:
            if item.mode == "zero":
                result[:, position] = 0.0
            else:
                assert donor is not None
                result[:, position] = donor[:, position]
    return result


def _apply_mlp_hidden(
    value: torch.Tensor,
    *,
    site: int,
    interventions: Sequence[FunctionalIntervention],
    donor: torch.Tensor | None,
) -> torch.Tensor:
    chosen = _selected(interventions, site=site, component="mlp_hidden")
    if not chosen:
        return value
    result = value.clone()
    for item in chosen:
        _validate_patch(item, donor)
        if item.mode == "onehot":
            raise ValueError("onehot mode is only valid for attention_pattern")
        if (
            item.heads is not None
            or item.source_positions is not None
            or item.dynamic_source_positions is not None
            or item.renormalize
        ):
            raise ValueError("invalid fields for mlp_hidden")
        positions = (
            tuple(range(result.shape[1]))
            if item.positions is None
            else item.positions
        )
        neurons = (
            tuple(range(result.shape[2]))
            if item.neurons is None
            else item.neurons
        )
        position_index = torch.as_tensor(positions, device=result.device)
        neuron_index = torch.as_tensor(neurons, device=result.device)
        if item.mode == "zero":
            result[:, position_index[:, None], neuron_index[None, :]] = 0.0
        else:
            assert donor is not None
            result[:, position_index[:, None], neuron_index[None, :]] = donor[
                :, position_index[:, None], neuron_index[None, :]
            ]
    return result


def _attention_parts(
    attention: MultiHeadSelfAttention,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = x.shape
    qkv = attention.qkv(x).view(
        batch,
        seq_len,
        3,
        attention.n_heads,
        attention.d_head,
    )
    q, k, v = qkv.unbind(dim=2)
    return (
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )


def _attention_pattern(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    seq_len = q.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    causal_mask = torch.ones(
        seq_len, seq_len, dtype=torch.bool, device=q.device
    ).triu(diagonal=1)
    return scores.masked_fill(causal_mask, float("-inf")).softmax(dim=-1)


def _project_context(
    attention: MultiHeadSelfAttention,
    context: torch.Tensor,
) -> torch.Tensor:
    batch, _, seq_len, _ = context.shape
    joined = context.transpose(1, 2).contiguous().view(
        batch, seq_len, -1
    )
    return attention.out_proj(joined)


@torch.no_grad()
def run_instrumented_state(
    model: LoopedGraphPathTransformer,
    initial_state: torch.Tensor,
    *,
    loop_indices: Sequence[int],
    interventions: Sequence[FunctionalIntervention] = (),
    donor_trace: FunctionalTrace | None = None,
) -> tuple[torch.Tensor, FunctionalTrace]:
    if model.block_style != "legacy":
        raise ValueError("functional circuit analysis supports legacy blocks")
    if not loop_indices:
        raise ValueError("loop_indices must not be empty")
    model.eval()
    x = initial_state
    sites: list[FunctionalSiteTrace] = []
    logits_by_loop: list[torch.Tensor] = []
    site_index = 0
    for loop_index in loop_indices:
        for block_index in model.active_block_indices(loop_index):
            block = model.blocks[block_index]
            if not isinstance(block, TransformerBlock):
                raise TypeError("legacy TransformerBlock required")
            donor_site = (
                None if donor_trace is None else donor_trace.sites[site_index]
            )
            x = _apply_token_tensor(
                x,
                site=site_index,
                component="block_input",
                interventions=interventions,
                donor=(
                    None if donor_site is None else donor_site.hidden_in
                ),
            )
            hidden_in = x
            q, k, v = _attention_parts(block.attn, block.ln_1(x))
            q = _apply_head_tensor(
                q,
                site=site_index,
                component="q",
                interventions=interventions,
                donor=None if donor_site is None else donor_site.q,
            )
            k = _apply_head_tensor(
                k,
                site=site_index,
                component="k",
                interventions=interventions,
                donor=None if donor_site is None else donor_site.k,
            )
            v = _apply_head_tensor(
                v,
                site=site_index,
                component="v",
                interventions=interventions,
                donor=None if donor_site is None else donor_site.v,
            )
            pattern = _attention_pattern(q, k)
            pattern = _apply_attention_pattern(
                pattern,
                site=site_index,
                interventions=interventions,
                donor=(
                    None
                    if donor_site is None
                    else donor_site.attention_pattern
                ),
            )
            context = torch.matmul(pattern, v)
            context = _apply_head_tensor(
                context,
                site=site_index,
                component="head_context",
                interventions=interventions,
                donor=(
                    None if donor_site is None else donor_site.head_context
                ),
            )
            attention_out = _project_context(block.attn, context)
            if block.inner_norm_style == "ouro_sandwich_rms":
                attention_out = block.attn_out_norm(attention_out)
            attention_out = _apply_token_tensor(
                attention_out,
                site=site_index,
                component="attention_out",
                interventions=interventions,
                donor=(
                    None if donor_site is None else donor_site.attention_out
                ),
            )
            residual_skip = _apply_token_tensor(
                x, site=site_index, component="residual_skip",
                interventions=interventions,
                donor=None if donor_site is None else donor_site.hidden_in,
            )
            residual_mid = residual_skip + attention_out
            residual_mid = _apply_token_tensor(
                residual_mid,
                site=site_index,
                component="residual_mid",
                interventions=interventions,
                donor=(
                    None if donor_site is None else donor_site.residual_mid
                ),
            )
            normalized = block.ln_2(residual_mid)
            mlp_hidden = block.mlp[1](block.mlp[0](normalized))
            mlp_hidden = _apply_mlp_hidden(
                mlp_hidden,
                site=site_index,
                interventions=interventions,
                donor=(
                    None if donor_site is None else donor_site.mlp_hidden
                ),
            )
            mlp_out = block.mlp[3](block.mlp[2](mlp_hidden))
            if block.inner_norm_style == "ouro_sandwich_rms":
                mlp_out = block.mlp_out_norm(mlp_out)
            mlp_out = _apply_token_tensor(
                mlp_out,
                site=site_index,
                component="mlp_out",
                interventions=interventions,
                donor=None if donor_site is None else donor_site.mlp_out,
            )
            x = residual_mid + mlp_out
            if block.residual_projector is not None:
                x = hidden_in + block.residual_projector(
                    hidden_in, x - hidden_in
                )
            sites.append(
                FunctionalSiteTrace(
                    loop_index=loop_index,
                    block_index=block_index,
                    hidden_in=hidden_in,
                    q=q,
                    k=k,
                    v=v,
                    attention_pattern=pattern,
                    head_context=context,
                    attention_out=attention_out,
                    residual_mid=residual_mid,
                    mlp_hidden=mlp_hidden,
                    mlp_out=mlp_out,
                    hidden_out=x,
                )
            )
            site_index += 1
        if model.outer_norm is not None:
            x = model.outer_norm(x)
        logits_by_loop.append(logits_from_raw_state(model, x))
    stacked = torch.stack(logits_by_loop, dim=1)
    return stacked[:, -1], FunctionalTrace(
        sites=sites,
        logits_by_loop=stacked,
    )


@torch.no_grad()
def run_instrumented(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    max_loops: int,
    interventions: Sequence[FunctionalIntervention] = (),
    donor_trace: FunctionalTrace | None = None,
) -> tuple[torch.Tensor, FunctionalTrace]:
    initial = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    return run_instrumented_state(
        model,
        initial,
        loop_indices=tuple(range(max_loops)),
        interventions=interventions,
        donor_trace=donor_trace,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _site_label(site: FunctionalSiteTrace) -> str:
    return f"L{site.loop_index + 1}.B{site.block_index + 1}"


def _state_logits(
    model: LoopedGraphPathTransformer, state: torch.Tensor
) -> torch.Tensor:
    return logits_from_raw_state(model, state)


def _path_state_metrics(
    logits: torch.Tensor,
    all_targets: torch.Tensor,
    *,
    endpoint_position: int,
) -> tuple[int, float, float]:
    prediction = logits.argmax(dim=-1)
    accuracies: list[torch.Tensor] = []
    endpoint = all_targets[:, endpoint_position]
    for position in range(all_targets.shape[1]):
        target = all_targets[:, position]
        valid = (
            torch.ones_like(target, dtype=torch.bool)
            if position == endpoint_position
            else target.ne(endpoint)
        )
        accuracies.append(
            prediction.eq(target)[valid].float().mean()
            if bool(valid.any())
            else torch.tensor(float("nan"), device=logits.device)
        )
    stacked = torch.stack(accuracies)
    finite = torch.nan_to_num(stacked, nan=-1.0)
    best = int(finite.argmax())
    return (
        best,
        float(stacked[best]),
        float(target_margin(logits, endpoint).mean()),
    )


def _pair_score(
    *,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    clean_target: torch.Tensor,
    corrupt_target: torch.Tensor,
) -> dict[str, float]:
    valid = (
        clean_target.ne(corrupt_target)
        & clean_logits.argmax(-1).eq(clean_target)
        & corrupt_logits.argmax(-1).eq(corrupt_target)
    )
    clean_diff = logit_difference(clean_logits, clean_target, corrupt_target)
    corrupt_diff = logit_difference(
        corrupt_logits, clean_target, corrupt_target
    )
    patched_diff = logit_difference(
        patched_logits, clean_target, corrupt_target
    )
    recovery = normalized_recovery(clean_diff, corrupt_diff, patched_diff)
    selected = valid & torch.isfinite(recovery)
    return {
        "valid_count": int(selected.sum()),
        "recovery": (
            float(recovery[selected].mean())
            if bool(selected.any())
            else float("nan")
        ),
        "clean_target_accuracy": (
            float(patched_logits.argmax(-1)[valid].eq(clean_target[valid]).float().mean())
            if bool(valid.any())
            else float("nan")
        ),
        "corrupt_target_accuracy": (
            float(
                patched_logits.argmax(-1)[valid]
                .eq(corrupt_target[valid])
                .float()
                .mean()
            )
            if bool(valid.any())
            else float("nan")
        ),
    }


def _patch_specifications(
    groups: dict[str, tuple[int, ...]],
    pair_type: str,
) -> list[tuple[str, Component, dict[str, Any]]]:
    answer = groups["answer"]
    graph = groups["graph"]
    common: list[tuple[str, Component, dict[str, Any]]] = [
        ("context_answer", "head_context", {"positions": answer}),
        ("attention_output_answer", "attention_out", {"positions": answer}),
        ("mlp_hidden_answer", "mlp_hidden", {"positions": answer}),
        ("mlp_output_answer", "mlp_out", {"positions": answer}),
    ]
    if pair_type == "query":
        return [
            ("q_answer", "q", {"positions": answer}),
            (
                "pattern_answer_graph",
                "attention_pattern",
                {
                    "positions": answer,
                    "source_positions": graph,
                    "renormalize": True,
                },
            ),
            *common,
        ]
    if pair_type == "graph":
        return [
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
            *common,
        ]
    raise ValueError(f"unknown pair type: {pair_type}")


def _roll_trace(trace: FunctionalTrace) -> FunctionalTrace:
    return FunctionalTrace(
        sites=[
            FunctionalSiteTrace(
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
            for site in trace.sites
        ],
        logits_by_loop=trace.logits_by_loop.roll(1, dims=0),
    )


@torch.no_grad()
def _patch_rows_for_pair(
    *,
    model: LoopedGraphPathTransformer,
    clean_tokens: torch.Tensor,
    clean_targets: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    corrupt_targets: torch.Tensor,
    max_loops: int,
    pair_type: str,
    endpoint_index: int,
) -> list[dict[str, Any]]:
    clean_logits, clean_trace = run_instrumented(
        model, clean_tokens, max_loops=max_loops
    )
    corrupt_logits, _ = run_instrumented(
        model, corrupt_tokens, max_loops=max_loops
    )
    shuffled_trace = _roll_trace(clean_trace)
    clean_target = clean_targets[:, endpoint_index]
    corrupt_target = corrupt_targets[:, endpoint_index]
    groups = explicit_depth_position_groups(model.cfg.node_count)
    rows: list[dict[str, Any]] = []
    for site_index, site in enumerate(clean_trace.sites):
        for role, component, fields in _patch_specifications(
            groups, pair_type
        ):
            intervention = FunctionalIntervention(
                site=site_index,
                component=component,
                mode="patch",
                **fields,
            )
            patched_logits, _ = run_instrumented(
                model,
                corrupt_tokens,
                max_loops=max_loops,
                interventions=(intervention,),
                donor_trace=clean_trace,
            )
            shuffled_logits, _ = run_instrumented(
                model,
                corrupt_tokens,
                max_loops=max_loops,
                interventions=(intervention,),
                donor_trace=shuffled_trace,
            )
            score = _pair_score(
                clean_logits=clean_logits,
                corrupt_logits=corrupt_logits,
                patched_logits=patched_logits,
                clean_target=clean_target,
                corrupt_target=corrupt_target,
            )
            shuffle_score = _pair_score(
                clean_logits=clean_logits,
                corrupt_logits=corrupt_logits,
                patched_logits=shuffled_logits,
                clean_target=clean_target,
                corrupt_target=corrupt_target,
            )
            rows.append(
                {
                    "pair_type": pair_type,
                    "site": _site_label(site),
                    "loop": site.loop_index + 1,
                    "block": site.block_index + 1,
                    "role_probe": role,
                    "component": component,
                    **score,
                    "shuffle_recovery": shuffle_score["recovery"],
                    "specific_recovery": score["recovery"]
                    - shuffle_score["recovery"],
                }
            )
    return rows


def _random_neurons(
    scores: torch.Tensor,
    top_neurons: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    remaining = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    remaining[top_neurons] = False
    candidates = torch.where(remaining)[0]
    order = torch.randperm(
        candidates.numel(), generator=generator, device=scores.device
    )
    return candidates[order[: top_neurons.numel()]]


@torch.no_grad()
def _mlp_neuron_rows(
    *,
    model: LoopedGraphPathTransformer,
    calibration: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    evaluation: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    max_loops: int,
    endpoint_index: int,
    top_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    calibration_clean, _, calibration_corrupt, _ = calibration
    _, calibration_clean_trace = run_instrumented(
        model, calibration_clean, max_loops=max_loops
    )
    _, calibration_corrupt_trace = run_instrumented(
        model, calibration_corrupt, max_loops=max_loops
    )
    clean_tokens, clean_targets, corrupt_tokens, corrupt_targets = evaluation
    clean_logits, clean_trace = run_instrumented(
        model, clean_tokens, max_loops=max_loops
    )
    corrupt_logits, corrupt_trace = run_instrumented(
        model, corrupt_tokens, max_loops=max_loops
    )
    clean_target = clean_targets[:, endpoint_index]
    corrupt_target = corrupt_targets[:, endpoint_index]
    answer = explicit_depth_position_groups(model.cfg.node_count)["answer"]
    generator = torch.Generator(device=clean_tokens.device).manual_seed(seed)
    rows: list[dict[str, Any]] = []
    for site_index, site in enumerate(clean_trace.sites):
        calibration_delta = (
            calibration_clean_trace.sites[site_index].mlp_hidden[:, -1]
            - calibration_corrupt_trace.sites[site_index].mlp_hidden[:, -1]
        )
        scores = calibration_delta.abs().mean(dim=0)
        selected_k = min(top_k, scores.numel())
        top_neurons = scores.topk(selected_k).indices
        random_neurons = _random_neurons(
            scores, top_neurons, generator=generator
        )
        all_neurons = torch.arange(scores.numel(), device=scores.device)
        mask = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
        mask[top_neurons] = False
        complement_neurons = all_neurons[mask]
        conditions = (
            ("all", all_neurons),
            (f"top{selected_k}", top_neurons),
            (f"random{selected_k}", random_neurons),
            ("complement", complement_neurons),
        )
        for condition, neurons in conditions:
            intervention = FunctionalIntervention(
                site=site_index,
                component="mlp_hidden",
                mode="patch",
                positions=answer,
                neurons=tuple(int(item) for item in neurons),
            )
            patched_logits, _ = run_instrumented(
                model,
                corrupt_tokens,
                max_loops=max_loops,
                interventions=(intervention,),
                donor_trace=clean_trace,
            )
            score = _pair_score(
                clean_logits=clean_logits,
                corrupt_logits=corrupt_logits,
                patched_logits=patched_logits,
                clean_target=clean_target,
                corrupt_target=corrupt_target,
            )
            rows.append(
                {
                    "site": _site_label(site),
                    "loop": site.loop_index + 1,
                    "block": site.block_index + 1,
                    "condition": condition,
                    "neuron_count": int(neurons.numel()),
                    **score,
                    "top_neuron_ids": (
                        " ".join(str(int(item)) for item in top_neurons)
                        if condition == f"top{selected_k}"
                        else ""
                    ),
                }
            )
    return rows


def _resolve_donor_states(
    *,
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    start: torch.Tensor,
) -> list[dict[str, Any]]:
    states = cache_raw_states(model, tokens, max_loop=model.cfg.max_loops)
    all_targets = _all_targets(start, targets)
    endpoint_position = model.cfg.max_depth
    resolved: list[dict[str, Any]] = []
    for loop_index, state in enumerate(states):
        logits = logits_from_raw_state(model, state)
        best, accuracy, _ = _path_state_metrics(
            logits,
            all_targets,
            endpoint_position=endpoint_position,
        )
        if accuracy >= 0.80 and best < endpoint_position:
            resolved.append(
                {
                    "loop_index": loop_index,
                    "position": best,
                    "accuracy": accuracy,
                    "state": state,
                }
            )
    return resolved


def _current_from_position(
    tokens: torch.Tensor,
    targets: torch.Tensor,
    position: int,
) -> torch.Tensor:
    return tokens[:, -3] if position == 0 else targets[:, position - 1]


@torch.no_grad()
def _select_transition_pair(
    *,
    model: LoopedGraphPathTransformer,
    batch_size: int,
    path_positions: int,
    device: torch.device,
) -> dict[str, Any] | None:
    donor_tokens, donor_targets, _, donor_start = fixed_depth_batch(
        model.cfg, batch_size, device, path_positions=path_positions
    )
    receiver_tokens, _, receiver_successors, _ = fixed_depth_batch(
        model.cfg, batch_size, device, path_positions=path_positions
    )
    candidates = _resolve_donor_states(
        model=model,
        tokens=donor_tokens,
        targets=donor_targets,
        start=donor_start,
    )
    receiver_states = cache_raw_states(
        model, receiver_tokens, max_loop=model.cfg.max_loops
    )
    best: dict[str, Any] | None = None
    for donor in candidates:
        current = _current_from_position(
            donor_tokens, donor_targets, donor["position"]
        )
        target = receiver_successors.gather(1, current[:, None]).squeeze(1)
        for receiver_index, receiver_state in enumerate(receiver_states):
            patched = replace_answer_state(receiver_state, donor["state"])
            logits, _ = run_instrumented_state(
                model,
                patched,
                loop_indices=(receiver_index + 1,),
            )
            accuracy = float(logits.argmax(-1).eq(target).float().mean())
            item = {
                "donor_loop_index": donor["loop_index"],
                "donor_position": donor["position"],
                "donor_mapping_accuracy": donor["accuracy"],
                "receiver_loop_index": receiver_index,
                "selection_next_accuracy": accuracy,
            }
            if best is None or accuracy > best["selection_next_accuracy"]:
                best = item
    return best


def _dynamic_edge_positions(
    current: torch.Tensor,
    *,
    random_control: bool,
    node_count: int,
) -> torch.Tensor:
    node = (current + 1) % node_count if random_control else current
    base = 1 + 3 * node
    return torch.stack((base, base + 1, base + 2), dim=1)


@torch.no_grad()
def _transition_rows(
    *,
    model: LoopedGraphPathTransformer,
    selection: dict[str, Any] | None,
    batch_size: int,
    path_positions: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if selection is None:
        return [], {"status": "no_resolved_pretarget_state"}
    donor_tokens, donor_targets, _, donor_start = fixed_depth_batch(
        model.cfg, batch_size, device, path_positions=path_positions
    )
    receiver_tokens, _, receiver_successors, _ = fixed_depth_batch(
        model.cfg, batch_size, device, path_positions=path_positions
    )
    donor_states = cache_raw_states(
        model,
        donor_tokens,
        max_loop=selection["donor_loop_index"] + 1,
    )
    receiver_states = cache_raw_states(
        model,
        receiver_tokens,
        max_loop=selection["receiver_loop_index"] + 1,
    )
    current = _current_from_position(
        donor_tokens, donor_targets, selection["donor_position"]
    )
    target = receiver_successors.gather(1, current[:, None]).squeeze(1)
    patched = replace_answer_state(
        receiver_states[selection["receiver_loop_index"]],
        donor_states[selection["donor_loop_index"]],
    )
    logical_loop = selection["receiver_loop_index"] + 1
    baseline_logits, baseline_trace = run_instrumented_state(
        model, patched, loop_indices=(logical_loop,)
    )
    baseline_accuracy = float(
        baseline_logits.argmax(-1).eq(target).float().mean()
    )
    baseline_margin = float(target_margin(baseline_logits, target).mean())
    if baseline_accuracy < 0.40:
        return [], {
            **selection,
            "status": "failed_heldout_transition_gate",
            "heldout_next_accuracy": baseline_accuracy,
        }
    groups = explicit_depth_position_groups(model.cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    current_edge = _dynamic_edge_positions(
        current, random_control=False, node_count=model.cfg.node_count
    )
    random_edge = _dynamic_edge_positions(
        current, random_control=True, node_count=model.cfg.node_count
    )
    rows: list[dict[str, Any]] = []
    for site_index, site in enumerate(baseline_trace.sites):
        specifications: list[tuple[str, Component, dict[str, Any]]] = [
            ("q_answer", "q", {"positions": answer}),
            ("k_graph", "k", {"positions": graph}),
            ("v_graph", "v", {"positions": graph}),
            (
                "pattern_current_edge",
                "attention_pattern",
                {
                    "positions": answer,
                    "dynamic_source_positions": current_edge,
                    "renormalize": True,
                },
            ),
            (
                "pattern_random_edge",
                "attention_pattern",
                {
                    "positions": answer,
                    "dynamic_source_positions": random_edge,
                    "renormalize": True,
                },
            ),
            ("context_answer", "head_context", {"positions": answer}),
            ("attention_output_answer", "attention_out", {"positions": answer}),
            ("mlp_hidden_answer", "mlp_hidden", {"positions": answer}),
            ("mlp_output_answer", "mlp_out", {"positions": answer}),
        ]
        for role, component, fields in specifications:
            intervention = FunctionalIntervention(
                site=site_index,
                component=component,
                mode="zero",
                **fields,
            )
            logits, _ = run_instrumented_state(
                model,
                patched,
                loop_indices=(logical_loop,),
                interventions=(intervention,),
            )
            accuracy = float(logits.argmax(-1).eq(target).float().mean())
            margin = float(target_margin(logits, target).mean())
            rows.append(
                {
                    "site": _site_label(site),
                    "block": site.block_index + 1,
                    "role_probe": role,
                    "component": component,
                    "baseline_accuracy": baseline_accuracy,
                    "ablated_accuracy": accuracy,
                    "accuracy_drop": baseline_accuracy - accuracy,
                    "baseline_margin": baseline_margin,
                    "ablated_margin": margin,
                    "margin_drop": baseline_margin - margin,
                }
            )
        for head in range(model.cfg.n_heads):
            for role, dynamic_positions in (
                ("head_pattern_current_edge", current_edge),
                ("head_pattern_random_edge", random_edge),
            ):
                intervention = FunctionalIntervention(
                    site=site_index,
                    component="attention_pattern",
                    mode="zero",
                    positions=answer,
                    heads=(head,),
                    dynamic_source_positions=dynamic_positions,
                    renormalize=True,
                )
                logits, _ = run_instrumented_state(
                    model,
                    patched,
                    loop_indices=(logical_loop,),
                    interventions=(intervention,),
                )
                accuracy = float(
                    logits.argmax(-1).eq(target).float().mean()
                )
                rows.append(
                    {
                        "site": _site_label(site),
                        "block": site.block_index + 1,
                        "role_probe": role,
                        "component": f"head_{head}",
                        "baseline_accuracy": baseline_accuracy,
                        "ablated_accuracy": accuracy,
                        "accuracy_drop": baseline_accuracy - accuracy,
                        "baseline_margin": baseline_margin,
                        "ablated_margin": float("nan"),
                        "margin_drop": float("nan"),
                    }
                )
    return rows, {
        **selection,
        "status": "passed_heldout_transition_gate",
        "heldout_next_accuracy": baseline_accuracy,
        "heldout_next_margin": baseline_margin,
    }


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    set_seed(seed)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    path_positions = max(cfg.max_depth + 1, cfg.max_loops + 1)
    endpoint_index = cfg.max_depth - 1
    groups = explicit_depth_position_groups(cfg.node_count)

    baseline_tokens, baseline_targets, _, baseline_start = fixed_depth_batch(
        cfg, batch_size, device, path_positions=path_positions
    )
    baseline_logits, baseline_trace = run_instrumented(
        model, baseline_tokens, max_loops=cfg.max_loops
    )
    endpoint_target = baseline_targets[:, endpoint_index]
    baseline_accuracy = float(
        baseline_logits.argmax(-1).eq(endpoint_target).float().mean()
    )
    baseline_margin = float(
        target_margin(baseline_logits, endpoint_target).mean()
    )

    all_targets = _all_targets(baseline_start, baseline_targets)
    progression_rows: list[dict[str, Any]] = []
    for site in baseline_trace.sites:
        for stage, state in (
            ("input", site.hidden_in),
            ("post_attention", site.residual_mid),
            ("post_mlp", site.hidden_out),
        ):
            best, accuracy, endpoint_margin = _path_state_metrics(
                _state_logits(model, state),
                all_targets,
                endpoint_position=cfg.max_depth,
            )
            progression_rows.append(
                {
                    "site": _site_label(site),
                    "loop": site.loop_index + 1,
                    "block": site.block_index + 1,
                    "stage": stage,
                    "best_path_position": best,
                    "best_path_accuracy": accuracy,
                    "endpoint_margin": endpoint_margin,
                }
            )
    _write_csv(run_dir / "branch_progression_rows.csv", progression_rows)

    position_rows: list[dict[str, Any]] = []
    position_groups = {
        key: value
        for key, value in groups.items()
        if key not in {"graph", "query_metadata"}
    }
    for site_index, site in enumerate(baseline_trace.sites):
        for component in ("attention_out", "mlp_out"):
            for group, positions in position_groups.items():
                intervention = FunctionalIntervention(
                    site=site_index,
                    component=component,
                    mode="zero",
                    positions=positions,
                )
                logits, _ = run_instrumented(
                    model,
                    baseline_tokens,
                    max_loops=cfg.max_loops,
                    interventions=(intervention,),
                )
                accuracy = float(
                    logits.argmax(-1).eq(endpoint_target).float().mean()
                )
                margin = float(target_margin(logits, endpoint_target).mean())
                position_rows.append(
                    {
                        "site": _site_label(site),
                        "loop": site.loop_index + 1,
                        "block": site.block_index + 1,
                        "component": component,
                        "position_group": group,
                        "baseline_accuracy": baseline_accuracy,
                        "ablated_accuracy": accuracy,
                        "accuracy_drop": baseline_accuracy - accuracy,
                        "baseline_margin": baseline_margin,
                        "ablated_margin": margin,
                        "margin_drop": baseline_margin - margin,
                    }
                )
    _write_csv(run_dir / "position_ablation_rows.csv", position_rows)

    query_pair = paired_fixed_depth_batch(
        cfg, batch_size, device, path_positions=path_positions
    )
    query_evaluation = (
        query_pair[0],
        query_pair[1],
        query_pair[2],
        query_pair[3],
    )
    graph_evaluation = paired_graph_batch(
        cfg, batch_size, device, path_positions=path_positions
    )
    patch_rows = _patch_rows_for_pair(
        model=model,
        clean_tokens=query_evaluation[0],
        clean_targets=query_evaluation[1],
        corrupt_tokens=query_evaluation[2],
        corrupt_targets=query_evaluation[3],
        max_loops=cfg.max_loops,
        pair_type="query",
        endpoint_index=endpoint_index,
    )
    patch_rows.extend(
        _patch_rows_for_pair(
            model=model,
            clean_tokens=graph_evaluation[0],
            clean_targets=graph_evaluation[1],
            corrupt_tokens=graph_evaluation[2],
            corrupt_targets=graph_evaluation[3],
            max_loops=cfg.max_loops,
            pair_type="graph",
            endpoint_index=endpoint_index,
        )
    )
    _write_csv(run_dir / "internal_patching_rows.csv", patch_rows)

    calibration_pair = paired_fixed_depth_batch(
        cfg, batch_size, device, path_positions=path_positions
    )
    neuron_rows = _mlp_neuron_rows(
        model=model,
        calibration=(
            calibration_pair[0],
            calibration_pair[1],
            calibration_pair[2],
            calibration_pair[3],
        ),
        evaluation=query_evaluation,
        max_loops=cfg.max_loops,
        endpoint_index=endpoint_index,
        top_k=top_k,
        seed=seed + 7919,
    )
    _write_csv(run_dir / "mlp_neuron_rows.csv", neuron_rows)

    transition_selection = _select_transition_pair(
        model=model,
        batch_size=batch_size,
        path_positions=path_positions,
        device=device,
    )
    transition_rows, transition_summary = _transition_rows(
        model=model,
        selection=transition_selection,
        batch_size=batch_size,
        path_positions=path_positions,
        device=device,
    )
    _write_csv(run_dir / "transition_function_rows.csv", transition_rows)

    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "data_seed": payload.get("data_seed"),
        "config": asdict(cfg),
        "loss_mode": checkpoint_loss_mode(checkpoint),
        "trained_loops": cfg.max_loops,
        "physical_blocks": cfg.n_layers,
        "effective_sites": cfg.max_loops * cfg.n_layers,
        "baseline": {
            "endpoint_accuracy": baseline_accuracy,
            "endpoint_margin": baseline_margin,
        },
        "functional_tests": {
            "position_ablation": (
                "zero one branch update only at one token group"
            ),
            "query_patching": (
                "same graph, different start; clean-to-corrupt component patch"
            ),
            "graph_patching": (
                "same start, different graph; clean-to-corrupt component patch"
            ),
            "mlp_neurons": (
                f"calibration-ranked top {top_k}, held-out patch, random and complement controls"
            ),
            "transition": transition_summary,
        },
        "evidence_scope": (
            "attention maps and logit lens are descriptive; functional labels "
            "require causal patching/ablation and controls"
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    name, raw_path = text.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-grained attention/MLP functional circuit analysis."
    )
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size and top-k must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{args.out_dir} is nonempty; pass --force to reuse it"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = []
    for name, checkpoint in args.run:
        summary = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            top_k=args.top_k,
        )
        summaries.append(summary)
        print(json.dumps(summary, allow_nan=False), flush=True)
    (args.out_dir / "comparison_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
