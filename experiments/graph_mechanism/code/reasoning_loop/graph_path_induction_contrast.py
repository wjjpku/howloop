from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    _all_targets,
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalSiteTrace,
    FunctionalTrace,
    _pair_score,
    _path_state_metrics,
    _roll_trace,
    _state_logits,
    explicit_depth_position_groups,
    run_instrumented,
)
from reasoning_loop.graph_path_loop import (
    LoopedGraphPathTransformer,
    TransformerBlock,
    pick_device,
    set_seed,
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


def _accuracy_margin(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    return (
        float(logits.argmax(-1).eq(target).float().mean()),
        float(target_margin(logits, target).mean()),
    )


def _dynamic_token_position(
    node: torch.Tensor,
    *,
    field: str,
) -> torch.Tensor:
    offsets = {"marker": 0, "source": 1, "destination": 2}
    if field not in offsets:
        raise ValueError(f"unknown edge field: {field}")
    return 1 + 3 * node + offsets[field]


def _dynamic_edge_positions(node: torch.Tensor) -> torch.Tensor:
    marker = _dynamic_token_position(node, field="marker")
    return torch.stack((marker, marker + 1, marker + 2), dim=1)


def _gather_answer_attention(
    pattern: torch.Tensor,
    source_position: torch.Tensor,
) -> torch.Tensor:
    """Return [batch, head] attention from the answer token to a source."""
    if pattern.ndim != 4 or source_position.ndim != 1:
        raise ValueError("invalid attention/source ranks")
    batch, heads, _, _ = pattern.shape
    if source_position.shape[0] != batch:
        raise ValueError("source batch does not match attention batch")
    batch_index = torch.arange(batch, device=pattern.device)[:, None]
    head_index = torch.arange(heads, device=pattern.device)[None, :]
    return pattern[
        batch_index,
        head_index,
        pattern.shape[2] - 1,
        source_position[:, None],
    ]


def _semantic_position(
    *,
    model: LoopedGraphPathTransformer,
    state: torch.Tensor,
    all_targets: torch.Tensor,
) -> tuple[int, float]:
    # Restrict the semantic decoder to start..requested endpoint. Permutation
    # cycles can make later generated path positions repeat the same node and
    # otherwise spuriously win the logit-lens argmax.
    decoded_targets = all_targets[:, : model.cfg.max_depth + 1]
    position, accuracy, _ = _path_state_metrics(
        _state_logits(model, state),
        decoded_targets,
        endpoint_position=model.cfg.max_depth,
    )
    return position, accuracy


def _node_at_position(
    all_targets: torch.Tensor,
    position: int,
) -> torch.Tensor:
    if not 0 <= position < all_targets.shape[1]:
        raise ValueError("semantic position is outside generated path")
    return all_targets[:, position]


def _ov_copy_logits(
    *,
    model: LoopedGraphPathTransformer,
    site: FunctionalSiteTrace,
    head: int,
    source_position: torch.Tensor,
) -> torch.Tensor:
    block = model.blocks[site.block_index]
    if not isinstance(block, TransformerBlock):
        raise TypeError("legacy TransformerBlock required")
    batch = site.v.shape[0]
    batch_index = torch.arange(batch, device=site.v.device)
    value = site.v[batch_index, head, source_position]
    start = head * block.attn.d_head
    stop = start + block.attn.d_head
    head_weight = block.attn.out_proj.weight[:, start:stop]
    residual_update = F.linear(value, head_weight)
    return model.unembed(residual_update)[:, : model.cfg.node_count]


@torch.no_grad()
def _induction_fingerprint_rows(
    *,
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    successors: torch.Tensor,
    start: torch.Tensor,
) -> list[dict[str, Any]]:
    endpoint_target = targets[:, model.cfg.max_depth - 1]
    baseline_logits, baseline_trace = run_instrumented(
        model, tokens, max_loops=model.cfg.max_loops
    )
    baseline_endpoint_accuracy, baseline_endpoint_margin = _accuracy_margin(
        baseline_logits, endpoint_target
    )
    all_targets = _all_targets(start, targets)
    answer = explicit_depth_position_groups(model.cfg.node_count)["answer"]
    rows: list[dict[str, Any]] = []
    for site_index, site in enumerate(baseline_trace.sites):
        if site.block_index != 1:
            continue
        position, mapping_accuracy = _semantic_position(
            model=model,
            state=site.hidden_in,
            all_targets=all_targets,
        )
        current = _node_at_position(all_targets, position)
        next_target = successors.gather(1, current[:, None]).squeeze(1)
        source_position = _dynamic_token_position(current, field="source")
        destination_position = _dynamic_token_position(
            current, field="destination"
        )
        random_destination = _dynamic_token_position(
            (current + 3) % model.cfg.node_count,
            field="destination",
        )
        source_mass = _gather_answer_attention(
            site.attention_pattern, source_position
        )
        destination_mass = _gather_answer_attention(
            site.attention_pattern, destination_position
        )
        random_mass = _gather_answer_attention(
            site.attention_pattern, random_destination
        )
        baseline_post_attention = _accuracy_margin(
            _state_logits(model, site.residual_mid), next_target
        )
        baseline_post_mlp = _accuracy_margin(
            _state_logits(model, site.hidden_out), next_target
        )
        for head in range(model.cfg.n_heads):
            ov_logits = _ov_copy_logits(
                model=model,
                site=site,
                head=head,
                source_position=destination_position,
            )
            ov_accuracy, ov_margin = _accuracy_margin(
                ov_logits, next_target
            )
            other_heads = tuple(
                item for item in range(model.cfg.n_heads) if item != head
            )
            interventions = (
                FunctionalIntervention(
                    site=site_index,
                    component="attention_pattern",
                    mode="onehot",
                    positions=answer,
                    heads=(head,),
                    dynamic_source_positions=destination_position[:, None],
                ),
                FunctionalIntervention(
                    site=site_index,
                    component="head_context",
                    mode="zero",
                    positions=answer,
                    heads=other_heads,
                ),
            )
            clamped_logits, clamped_trace = run_instrumented(
                model,
                tokens,
                max_loops=model.cfg.max_loops,
                interventions=interventions,
            )
            clamped_site = clamped_trace.sites[site_index]
            clamp_attention_accuracy, clamp_attention_margin = (
                _accuracy_margin(
                    _state_logits(model, clamped_site.residual_mid),
                    next_target,
                )
            )
            clamp_mlp_accuracy, clamp_mlp_margin = _accuracy_margin(
                _state_logits(model, clamped_site.hidden_out),
                next_target,
            )
            clamp_endpoint_accuracy, clamp_endpoint_margin = _accuracy_margin(
                clamped_logits, endpoint_target
            )
            rows.append(
                {
                    "site": _site_label(site),
                    "loop": site.loop_index + 1,
                    "block": site.block_index + 1,
                    "head": head,
                    "decoded_input_position": position,
                    "decoded_input_accuracy": mapping_accuracy,
                    "source_match_attention": float(
                        source_mass[:, head].mean()
                    ),
                    "destination_copy_attention": float(
                        destination_mass[:, head].mean()
                    ),
                    "random_destination_attention": float(
                        random_mass[:, head].mean()
                    ),
                    "attention_match_gap": float(
                        (
                            destination_mass[:, head]
                            - random_mass[:, head]
                        ).mean()
                    ),
                    "ov_copy_next_accuracy": ov_accuracy,
                    "ov_copy_next_margin": ov_margin,
                    "baseline_post_attention_next_accuracy": (
                        baseline_post_attention[0]
                    ),
                    "baseline_post_mlp_next_accuracy": baseline_post_mlp[0],
                    "clamp_post_attention_next_accuracy": (
                        clamp_attention_accuracy
                    ),
                    "clamp_post_attention_next_margin": (
                        clamp_attention_margin
                    ),
                    "clamp_post_mlp_next_accuracy": clamp_mlp_accuracy,
                    "clamp_post_mlp_next_margin": clamp_mlp_margin,
                    "baseline_endpoint_accuracy": baseline_endpoint_accuracy,
                    "baseline_endpoint_margin": baseline_endpoint_margin,
                    "clamp_endpoint_accuracy": clamp_endpoint_accuracy,
                    "clamp_endpoint_margin": clamp_endpoint_margin,
                }
            )
        all_head_intervention = FunctionalIntervention(
            site=site_index,
            component="attention_pattern",
            mode="onehot",
            positions=answer,
            dynamic_source_positions=destination_position[:, None],
        )
        all_logits, all_trace = run_instrumented(
            model,
            tokens,
            max_loops=model.cfg.max_loops,
            interventions=(all_head_intervention,),
        )
        all_site = all_trace.sites[site_index]
        all_attention = _accuracy_margin(
            _state_logits(model, all_site.residual_mid), next_target
        )
        all_mlp = _accuracy_margin(
            _state_logits(model, all_site.hidden_out), next_target
        )
        all_endpoint = _accuracy_margin(all_logits, endpoint_target)
        rows.append(
            {
                "site": _site_label(site),
                "loop": site.loop_index + 1,
                "block": site.block_index + 1,
                "head": "all",
                "decoded_input_position": position,
                "decoded_input_accuracy": mapping_accuracy,
                "source_match_attention": float(source_mass.mean()),
                "destination_copy_attention": float(
                    destination_mass.mean()
                ),
                "random_destination_attention": float(random_mass.mean()),
                "attention_match_gap": float(
                    (destination_mass - random_mass).mean()
                ),
                "ov_copy_next_accuracy": float("nan"),
                "ov_copy_next_margin": float("nan"),
                "baseline_post_attention_next_accuracy": (
                    baseline_post_attention[0]
                ),
                "baseline_post_mlp_next_accuracy": baseline_post_mlp[0],
                "clamp_post_attention_next_accuracy": all_attention[0],
                "clamp_post_attention_next_margin": all_attention[1],
                "clamp_post_mlp_next_accuracy": all_mlp[0],
                "clamp_post_mlp_next_margin": all_mlp[1],
                "baseline_endpoint_accuracy": baseline_endpoint_accuracy,
                "baseline_endpoint_margin": baseline_endpoint_margin,
                "clamp_endpoint_accuracy": all_endpoint[0],
                "clamp_endpoint_margin": all_endpoint[1],
            }
        )
    return rows


@torch.no_grad()
def _path_bundle_rows(
    *,
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    start: torch.Tensor,
    max_bundle_edges: int,
) -> list[dict[str, Any]]:
    endpoint_target = targets[:, model.cfg.max_depth - 1]
    baseline_logits, baseline_trace = run_instrumented(
        model, tokens, max_loops=model.cfg.max_loops
    )
    baseline_endpoint = _accuracy_margin(
        baseline_logits, endpoint_target
    )
    all_targets = _all_targets(start, targets)
    answer = explicit_depth_position_groups(model.cfg.node_count)["answer"]
    rows: list[dict[str, Any]] = []
    for site_index, site in enumerate(baseline_trace.sites):
        if site.block_index != 1:
            continue
        position, mapping_accuracy = _semantic_position(
            model=model,
            state=site.hidden_in,
            all_targets=all_targets,
        )
        current = _node_at_position(all_targets, position)
        next_target = _node_at_position(all_targets, position + 1)
        conditions: list[tuple[str, torch.Tensor]] = []
        for offset in range(max_bundle_edges):
            node = _node_at_position(all_targets, position + offset)
            conditions.append(
                (f"path_edge_offset_{offset}", _dynamic_edge_positions(node))
            )
        bundle = torch.cat(
            [positions for _, positions in conditions], dim=1
        )
        conditions.append(("path_edge_bundle", bundle))
        conditions.append(
            (
                "random_edge",
                _dynamic_edge_positions(
                    (current + 3) % model.cfg.node_count
                ),
            )
        )
        for condition, source_positions in conditions:
            intervention = FunctionalIntervention(
                site=site_index,
                component="attention_pattern",
                mode="zero",
                positions=answer,
                dynamic_source_positions=source_positions,
                renormalize=True,
            )
            logits, trace = run_instrumented(
                model,
                tokens,
                max_loops=model.cfg.max_loops,
                interventions=(intervention,),
            )
            changed_site = trace.sites[site_index]
            immediate = _accuracy_margin(
                _state_logits(model, changed_site.hidden_out), next_target
            )
            endpoint = _accuracy_margin(logits, endpoint_target)
            rows.append(
                {
                    "site": _site_label(site),
                    "loop": site.loop_index + 1,
                    "decoded_input_position": position,
                    "decoded_input_accuracy": mapping_accuracy,
                    "condition": condition,
                    "ablated_edge_count": source_positions.shape[1] // 3,
                    "baseline_endpoint_accuracy": baseline_endpoint[0],
                    "ablated_endpoint_accuracy": endpoint[0],
                    "endpoint_accuracy_drop": (
                        baseline_endpoint[0] - endpoint[0]
                    ),
                    "baseline_endpoint_margin": baseline_endpoint[1],
                    "ablated_endpoint_margin": endpoint[1],
                    "endpoint_margin_drop": baseline_endpoint[1] - endpoint[1],
                    "post_mlp_next_accuracy": immediate[0],
                    "post_mlp_next_margin": immediate[1],
                }
            )
    return rows


@torch.no_grad()
def _writer_dependency_rows(
    *,
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    start: torch.Tensor,
) -> list[dict[str, Any]]:
    endpoint_target = targets[:, model.cfg.max_depth - 1]
    baseline_logits, baseline_trace = run_instrumented(
        model, tokens, max_loops=model.cfg.max_loops
    )
    baseline_endpoint = _accuracy_margin(
        baseline_logits, endpoint_target
    )
    all_targets = _all_targets(start, targets)
    destinations = explicit_depth_position_groups(
        model.cfg.node_count
    )["destination"]
    rows: list[dict[str, Any]] = []
    for b2_index, b2_site in enumerate(baseline_trace.sites):
        if b2_site.block_index != 1:
            continue
        b1_index = next(
            index
            for index, candidate in enumerate(baseline_trace.sites)
            if candidate.loop_index == b2_site.loop_index
            and candidate.block_index == 0
        )
        position, mapping_accuracy = _semantic_position(
            model=model,
            state=b2_site.hidden_in,
            all_targets=all_targets,
        )
        current = _node_at_position(all_targets, position)
        destination_position = _dynamic_token_position(
            current, field="destination"
        )
        baseline_mass = float(
            _gather_answer_attention(
                b2_site.attention_pattern, destination_position
            ).mean()
        )
        conditions = {
            "none": (),
            "zero_B1_attention_destination": (
                FunctionalIntervention(
                    site=b1_index,
                    component="attention_out",
                    mode="zero",
                    positions=destinations,
                ),
            ),
            "zero_B1_mlp_destination": (
                FunctionalIntervention(
                    site=b1_index,
                    component="mlp_out",
                    mode="zero",
                    positions=destinations,
                ),
            ),
            "zero_B1_attention_and_mlp_destination": (
                FunctionalIntervention(
                    site=b1_index,
                    component="attention_out",
                    mode="zero",
                    positions=destinations,
                ),
                FunctionalIntervention(
                    site=b1_index,
                    component="mlp_out",
                    mode="zero",
                    positions=destinations,
                ),
            ),
        }
        for condition, interventions in conditions.items():
            if not interventions:
                logits, trace = baseline_logits, baseline_trace
            else:
                logits, trace = run_instrumented(
                    model,
                    tokens,
                    max_loops=model.cfg.max_loops,
                    interventions=interventions,
                )
            changed_b2 = trace.sites[b2_index]
            mass = float(
                _gather_answer_attention(
                    changed_b2.attention_pattern, destination_position
                ).mean()
            )
            endpoint = _accuracy_margin(logits, endpoint_target)
            rows.append(
                {
                    "loop": b2_site.loop_index + 1,
                    "b1_site": _site_label(baseline_trace.sites[b1_index]),
                    "b2_site": _site_label(b2_site),
                    "decoded_input_position": position,
                    "decoded_input_accuracy": mapping_accuracy,
                    "condition": condition,
                    "baseline_destination_attention": baseline_mass,
                    "destination_attention": mass,
                    "destination_attention_drop": baseline_mass - mass,
                    "baseline_endpoint_accuracy": baseline_endpoint[0],
                    "endpoint_accuracy": endpoint[0],
                    "endpoint_accuracy_drop": (
                        baseline_endpoint[0] - endpoint[0]
                    ),
                    "baseline_endpoint_margin": baseline_endpoint[1],
                    "endpoint_margin": endpoint[1],
                    "endpoint_margin_drop": baseline_endpoint[1] - endpoint[1],
                }
            )
    return rows


def _depth_patch_specifications(
    answer: tuple[int, ...],
) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        ("q_answer", "q", {"positions": answer}),
        (
            "attention_pattern_answer",
            "attention_pattern",
            {"positions": answer},
        ),
        (
            "attention_output_answer",
            "attention_out",
            {"positions": answer},
        ),
        ("mlp_hidden_answer", "mlp_hidden", {"positions": answer}),
        ("mlp_output_answer", "mlp_out", {"positions": answer}),
    ]


@torch.no_grad()
def _depth_phase_patch_rows(
    *,
    model: LoopedGraphPathTransformer,
    long_tokens: torch.Tensor,
    targets: torch.Tensor,
    short_depth: int,
) -> list[dict[str, Any]]:
    short_tokens = long_tokens.clone()
    short_tokens[:, -2] = model.cfg.depth_token_base + short_depth - 1
    long_logits, long_trace = run_instrumented(
        model, long_tokens, max_loops=model.cfg.max_loops
    )
    short_logits, short_trace = run_instrumented(
        model, short_tokens, max_loops=model.cfg.max_loops
    )
    shuffled_long_trace = _roll_trace(long_trace)
    long_target = targets[:, model.cfg.max_depth - 1]
    short_target = targets[:, short_depth - 1]
    answer = explicit_depth_position_groups(model.cfg.node_count)["answer"]
    rows: list[dict[str, Any]] = []
    for site_index, site in enumerate(short_trace.sites):
        if site.block_index != 1:
            continue
        for role, component, fields in _depth_patch_specifications(answer):
            intervention = FunctionalIntervention(
                site=site_index,
                component=component,  # type: ignore[arg-type]
                mode="patch",
                **fields,
            )
            patched_logits, _ = run_instrumented(
                model,
                short_tokens,
                max_loops=model.cfg.max_loops,
                interventions=(intervention,),
                donor_trace=long_trace,
            )
            shuffled_logits, _ = run_instrumented(
                model,
                short_tokens,
                max_loops=model.cfg.max_loops,
                interventions=(intervention,),
                donor_trace=shuffled_long_trace,
            )
            score = _pair_score(
                clean_logits=long_logits,
                corrupt_logits=short_logits,
                patched_logits=patched_logits,
                clean_target=long_target,
                corrupt_target=short_target,
            )
            shuffled = _pair_score(
                clean_logits=long_logits,
                corrupt_logits=short_logits,
                patched_logits=shuffled_logits,
                clean_target=long_target,
                corrupt_target=short_target,
            )
            rows.append(
                {
                    "site": _site_label(site),
                    "loop": site.loop_index + 1,
                    "block": site.block_index + 1,
                    "long_depth": model.cfg.max_depth,
                    "short_depth": short_depth,
                    "role_probe": role,
                    "component": component,
                    "head": "all",
                    **score,
                    "shuffle_recovery": shuffled["recovery"],
                    "specific_recovery": (
                        score["recovery"] - shuffled["recovery"]
                    ),
                }
            )
        for head in range(model.cfg.n_heads):
            for role, component in (
                ("q_answer_head", "q"),
                ("attention_pattern_answer_head", "attention_pattern"),
            ):
                intervention = FunctionalIntervention(
                    site=site_index,
                    component=component,  # type: ignore[arg-type]
                    mode="patch",
                    positions=answer,
                    heads=(head,),
                )
                patched_logits, _ = run_instrumented(
                    model,
                    short_tokens,
                    max_loops=model.cfg.max_loops,
                    interventions=(intervention,),
                    donor_trace=long_trace,
                )
                shuffled_logits, _ = run_instrumented(
                    model,
                    short_tokens,
                    max_loops=model.cfg.max_loops,
                    interventions=(intervention,),
                    donor_trace=shuffled_long_trace,
                )
                score = _pair_score(
                    clean_logits=long_logits,
                    corrupt_logits=short_logits,
                    patched_logits=patched_logits,
                    clean_target=long_target,
                    corrupt_target=short_target,
                )
                shuffled = _pair_score(
                    clean_logits=long_logits,
                    corrupt_logits=short_logits,
                    patched_logits=shuffled_logits,
                    clean_target=long_target,
                    corrupt_target=short_target,
                )
                rows.append(
                    {
                        "site": _site_label(site),
                        "loop": site.loop_index + 1,
                        "block": site.block_index + 1,
                        "long_depth": model.cfg.max_depth,
                        "short_depth": short_depth,
                        "role_probe": role,
                        "component": component,
                        "head": head,
                        **score,
                        "shuffle_recovery": shuffled["recovery"],
                        "specific_recovery": (
                            score["recovery"] - shuffled["recovery"]
                        ),
                    }
                )
    return rows


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    seed: int,
    max_bundle_edges: int,
) -> dict[str, Any]:
    set_seed(seed)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if cfg.n_layers != 2 or cfg.block_schedule != "all_blocks":
        raise ValueError("induction contrast requires the two-block all_blocks model")
    path_positions = cfg.max_depth + max_bundle_edges + 2
    tokens, targets, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
    )
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    induction_rows = _induction_fingerprint_rows(
        model=model,
        tokens=tokens,
        targets=targets,
        successors=successors,
        start=start,
    )
    _write_csv(run_dir / "induction_fingerprint_rows.csv", induction_rows)
    bundle_rows = _path_bundle_rows(
        model=model,
        tokens=tokens,
        targets=targets,
        start=start,
        max_bundle_edges=max_bundle_edges,
    )
    _write_csv(run_dir / "path_bundle_ablation_rows.csv", bundle_rows)
    writer_rows = _writer_dependency_rows(
        model=model,
        tokens=tokens,
        targets=targets,
        start=start,
    )
    _write_csv(run_dir / "writer_dependency_rows.csv", writer_rows)
    short_depth = max(1, cfg.max_depth - 2)
    depth_rows = _depth_phase_patch_rows(
        model=model,
        long_tokens=tokens,
        targets=targets,
        short_depth=short_depth,
    )
    _write_csv(run_dir / "depth_phase_patch_rows.csv", depth_rows)

    baseline_logits, _ = run_instrumented(
        model, tokens, max_loops=cfg.max_loops
    )
    endpoint_target = targets[:, cfg.max_depth - 1]
    baseline_accuracy, baseline_margin = _accuracy_margin(
        baseline_logits, endpoint_target
    )
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "data_seed": payload.get("data_seed"),
        "config": asdict(cfg),
        "loss_mode": "final_only",
        "trained_loops": cfg.max_loops,
        "physical_blocks": cfg.n_layers,
        "effective_block_visits": cfg.max_loops * cfg.n_layers,
        "baseline": {
            "endpoint_accuracy": baseline_accuracy,
            "endpoint_margin": baseline_margin,
        },
        "tests": {
            "canonical_induction_fingerprint": (
                "answer-to-current-destination QK attention, head OV copy, "
                "and one-hot destination clamp"
            ),
            "path_bundle_dependency": (
                "current edge versus future path edges and matched random edge"
            ),
            "two_block_writer_dependency": (
                "B1 destination writes causally changing B2 matching"
            ),
            "depth_phase_patching": (
                f"same graph/start, depth {short_depth} versus {cfg.max_depth}"
            ),
        },
        "claim_boundary": (
            "attention mass and direct OV scores are fingerprints; causal "
            "clamps, ablations, and matched/shuffled patches support roles"
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
        description=(
            "Causal contrast between graph-path loop circuits and canonical "
            "two-layer induction fingerprints."
        )
    )
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-bundle-edges", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_bundle_edges < 1:
        raise ValueError("batch size and bundle edge count must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{args.out_dir} is not empty; pass --force to overwrite runs"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = []
    for name, checkpoint in args.run:
        summaries.append(
            analyze_checkpoint(
                name=name,
                checkpoint=checkpoint,
                out_dir=args.out_dir,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
                max_bundle_edges=args.max_bundle_edges,
            )
        )
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
