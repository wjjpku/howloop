from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalSiteTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
)
from reasoning_loop.graph_path_telomere_rejuvenator import (
    FlattenedAffine,
    PositionwiseAffine,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
    _apply_answer_map,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scope_for_block(
    cfg: GraphPathConfig,
    block_index: int,
) -> tuple[str, tuple[int, ...]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    if block_index == 0:
        return "graph", groups["graph"]
    if block_index == 1:
        return "answer", groups["answer"]
    raise ValueError("D8L8 circuit analysis expects two physical blocks")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    return float(
        F.cosine_similarity(left_flat, right_flat, dim=-1).mean()
    )


def _relative_mse(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean()
    centered = right.float() - right.float().mean(
        dim=0,
        keepdim=True,
    )
    return float(numerator / centered.square().mean().clamp_min(1e-12))


def _load_age_map(
    path: Path,
    *,
    device: torch.device,
) -> tuple[FlattenedAffine, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_simple_one_step_telomere":
        raise ValueError("operator artifact has the wrong kind")
    weight = payload["weight"].to(device)
    bias = payload["bias"].to(device)
    affine = PositionwiseAffine(
        weight=weight.unsqueeze(0),
        bias=bias.unsqueeze(0),
    )
    return (
        FlattenedAffine(
            affine=affine,
            position_count=1,
            feature_count=weight.shape[0],
        ),
        payload,
    )


def _add_metric(
    bucket: dict[str, float],
    prefix: str,
    metrics: dict[str, float],
) -> None:
    count = int(metrics["valid_count"])
    if count == 0:
        return
    bucket[f"{prefix}_correct"] += float(metrics["accuracy"]) * count
    bucket[f"{prefix}_margin"] += float(metrics["margin"]) * count
    bucket[f"{prefix}_count"] += count


def _metric_bucket(prefixes: tuple[str, ...]) -> dict[str, float]:
    return {
        f"{prefix}_{field}": 0.0
        for prefix in prefixes
        for field in ("correct", "margin", "count")
    }


def _final_metric(
    bucket: dict[str, float],
    prefix: str,
    field: str,
) -> float:
    count = bucket[f"{prefix}_count"]
    if count == 0:
        return float("nan")
    numerator = (
        bucket[f"{prefix}_correct"]
        if field == "accuracy"
        else bucket[f"{prefix}_margin"]
    )
    return numerator / count


def _intervention(
    *,
    site: int,
    component: str,
    positions: tuple[int, ...],
    head: int | None,
    mode: str,
) -> FunctionalIntervention:
    return FunctionalIntervention(
        site=site,
        component=component,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        positions=positions,
        heads=None if head is None else (head,),
    )


def _activation_rows(
    *,
    cycle: int,
    learned_site: FunctionalSiteTrace,
    oracle_site: FunctionalSiteTrace,
    positions: tuple[int, ...],
) -> list[dict[str, Any]]:
    index = torch.as_tensor(
        positions,
        device=learned_site.hidden_in.device,
    )
    rows = []
    for head in range(learned_site.head_context.shape[1]):
        learned_context = learned_site.head_context[:, head, index]
        oracle_context = oracle_site.head_context[:, head, index]
        learned_pattern = learned_site.attention_pattern[
            :, head, index
        ]
        oracle_pattern = oracle_site.attention_pattern[:, head, index]
        rows.append(
            {
                "cycle": cycle,
                "block": learned_site.block_index + 1,
                "component": "head",
                "head": head,
                "context_cosine": _cosine(
                    learned_context,
                    oracle_context,
                ),
                "attention_pattern_cosine": _cosine(
                    learned_pattern,
                    oracle_pattern,
                ),
                "update_norm_ratio": float(
                    learned_context.float().flatten(1).norm(dim=-1).mean()
                    / oracle_context.float()
                    .flatten(1)
                    .norm(dim=-1)
                    .mean()
                    .clamp_min(1e-12)
                ),
            }
        )
    for component, learned, oracle in (
        (
            "attention_out",
            learned_site.attention_out[:, index],
            oracle_site.attention_out[:, index],
        ),
        (
            "mlp_out",
            learned_site.mlp_out[:, index],
            oracle_site.mlp_out[:, index],
        ),
    ):
        rows.append(
            {
                "cycle": cycle,
                "block": learned_site.block_index + 1,
                "component": component,
                "head": -1,
                "context_cosine": _cosine(learned, oracle),
                "attention_pattern_cosine": float("nan"),
                "update_norm_ratio": float(
                    learned.float().flatten(1).norm(dim=-1).mean()
                    / oracle.float()
                    .flatten(1)
                    .norm(dim=-1)
                    .mean()
                    .clamp_min(1e-12)
                ),
            }
        )
    return rows


@torch.no_grad()
def analyze_circuit(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    operator_path: Path,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    cycles: int,
    seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, _ = load_checkpoint(checkpoint, device)
    if cfg.n_layers != 2:
        raise ValueError("analysis requires two physical blocks")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    age_map, operator_payload = _load_age_map(
        operator_path,
        device=device,
    )
    answer_positions = explicit_depth_position_groups(
        cfg.node_count
    )["answer"]
    jump = phase_positions[3] - phase_positions[2]

    behavior_buckets = {
        (cycle, state): _metric_bucket(("baseline",))
        for cycle in range(1, cycles + 1)
        for state in ("learned", "oracle")
    }
    component_buckets: dict[
        tuple[int, int, str, int],
        dict[str, float],
    ] = {}
    activation_parts: list[dict[str, Any]] = []
    routing_parts: list[dict[str, Any]] = []
    state_parts: list[dict[str, Any]] = []
    position_groups = explicit_depth_position_groups(cfg.node_count)

    set_seed(seed)
    for batch_index in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * cycles,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        learned_state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        oracle_state = learned_state.clone()

        for cycle in range(1, cycles + 1):
            current = all_targets[
                :, cfg.max_depth + jump * (cycle - 1)
            ]
            target = all_targets[:, cfg.max_depth + jump * cycle]
            state_parts.append(
                {
                    "batch": batch_index,
                    "cycle": cycle,
                    "answer_relative_mse": _relative_mse(
                        learned_state[:, list(answer_positions)],
                        oracle_state[:, list(answer_positions)],
                    ),
                    "all_state_relative_mse": _relative_mse(
                        learned_state,
                        oracle_state,
                    ),
                }
            )
            learned_logits, learned_trace = run_instrumented_state(
                model,
                learned_state,
                loop_indices=(cfg.max_loops + cycle - 1,),
            )
            oracle_logits, oracle_trace = run_instrumented_state(
                model,
                oracle_state,
                loop_indices=(cfg.max_loops + cycle - 1,),
            )
            learned_metrics = _masked_metrics(
                learned_logits,
                target,
                endpoint=endpoint,
            )
            oracle_metrics = _masked_metrics(
                oracle_logits,
                target,
                endpoint=endpoint,
            )
            _add_metric(
                behavior_buckets[(cycle, "learned")],
                "baseline",
                learned_metrics,
            )
            _add_metric(
                behavior_buckets[(cycle, "oracle")],
                "baseline",
                oracle_metrics,
            )
            for state_name, trace in (
                ("learned", learned_trace),
                ("oracle", oracle_trace),
            ):
                second_block = trace.sites[1]
                pattern = second_block.attention_pattern[
                    :, 2, answer_positions[0]
                ]
                top_position = pattern.argmax(dim=-1)
                for role in ("edge_marker", "source", "destination"):
                    ordered_positions = torch.as_tensor(
                        position_groups[role],
                        device=device,
                    )
                    correct_position = ordered_positions[current]
                    batch_index_tensor = torch.arange(
                        current.shape[0],
                        device=device,
                    )
                    routing_parts.append(
                        {
                            "batch": batch_index,
                            "cycle": cycle,
                            "state": state_name,
                            "role": role,
                            "attention_mass": float(
                                pattern[
                                    batch_index_tensor,
                                    correct_position,
                                ].mean()
                            ),
                            "argmax_hit_rate": float(
                                top_position.eq(correct_position)
                                .float()
                                .mean()
                            ),
                        }
                    )

            for site_index, (
                learned_site,
                oracle_site,
            ) in enumerate(
                zip(
                    learned_trace.sites,
                    oracle_trace.sites,
                    strict=True,
                )
            ):
                scope, positions = _scope_for_block(
                    cfg,
                    learned_site.block_index,
                )
                for row in _activation_rows(
                    cycle=cycle,
                    learned_site=learned_site,
                    oracle_site=oracle_site,
                    positions=positions,
                ):
                    activation_parts.append(
                        {
                            "batch": batch_index,
                            "scope": scope,
                            **row,
                        }
                    )
                components = [
                    ("attention_out", -1, positions, "attention_out"),
                    ("mlp_out", -1, positions, "mlp_out"),
                    *[
                        (
                            "head_context",
                            head,
                            positions,
                            f"head{head}",
                        )
                        for head in range(cfg.n_heads)
                    ],
                ]
                if learned_site.block_index == 1:
                    all_positions = tuple(range(cfg.seq_len))
                    for head in range(cfg.n_heads):
                        components.extend(
                            (
                                (
                                    "q",
                                    head,
                                    positions,
                                    f"head{head}_q_answer",
                                ),
                                (
                                    "k",
                                    head,
                                    all_positions,
                                    f"head{head}_k_all",
                                ),
                                (
                                    "v",
                                    head,
                                    all_positions,
                                    f"head{head}_v_all",
                                ),
                                (
                                    "attention_pattern",
                                    head,
                                    positions,
                                    f"head{head}_pattern_answer",
                                ),
                            )
                        )
                for (
                    component,
                    head,
                    intervention_positions,
                    label,
                ) in components:
                    key = (
                        cycle,
                        learned_site.block_index + 1,
                        label,
                        head,
                    )
                    bucket = component_buckets.setdefault(
                        key,
                        _metric_bucket(
                            (
                                "learned",
                                "oracle",
                                "learned_zero",
                                "oracle_zero",
                                "oracle_patch",
                            )
                        ),
                    )
                    _add_metric(bucket, "learned", learned_metrics)
                    _add_metric(bucket, "oracle", oracle_metrics)
                    learned_zero_logits, _ = run_instrumented_state(
                        model,
                        learned_state,
                        loop_indices=(cfg.max_loops + cycle - 1,),
                        interventions=(
                            _intervention(
                                site=site_index,
                                component=component,
                                positions=intervention_positions,
                                head=None if head < 0 else head,
                                mode="zero",
                            ),
                        ),
                    )
                    oracle_zero_logits, _ = run_instrumented_state(
                        model,
                        oracle_state,
                        loop_indices=(cfg.max_loops + cycle - 1,),
                        interventions=(
                            _intervention(
                                site=site_index,
                                component=component,
                                positions=intervention_positions,
                                head=None if head < 0 else head,
                                mode="zero",
                            ),
                        ),
                    )
                    patched_logits, _ = run_instrumented_state(
                        model,
                        learned_state,
                        loop_indices=(cfg.max_loops + cycle - 1,),
                        interventions=(
                            _intervention(
                                site=site_index,
                                component=component,
                                positions=intervention_positions,
                                head=None if head < 0 else head,
                                mode="patch",
                            ),
                        ),
                        donor_trace=oracle_trace,
                    )
                    _add_metric(
                        bucket,
                        "learned_zero",
                        _masked_metrics(
                            learned_zero_logits,
                            target,
                            endpoint=endpoint,
                        ),
                    )
                    _add_metric(
                        bucket,
                        "oracle_zero",
                        _masked_metrics(
                            oracle_zero_logits,
                            target,
                            endpoint=endpoint,
                        ),
                    )
                    _add_metric(
                        bucket,
                        "oracle_patch",
                        _masked_metrics(
                            patched_logits,
                            target,
                            endpoint=endpoint,
                        ),
                    )

            learned_state = learned_trace.sites[-1].hidden_out
            oracle_state = oracle_trace.sites[-1].hidden_out
            if cycle < cycles:
                learned_state = _apply_answer_map(
                    learned_state,
                    answer_positions=answer_positions,
                    age_map=age_map,
                    mode="matched",
                )
                oracle_state = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=target,
                    age=2,
                    phase_position=phase_positions[2],
                )

    behavior_rows = []
    for (cycle, state), bucket in sorted(behavior_buckets.items()):
        behavior_rows.append(
            {
                "cycle": cycle,
                "state": state,
                "accuracy": _final_metric(
                    bucket,
                    "baseline",
                    "accuracy",
                ),
                "margin": _final_metric(bucket, "baseline", "margin"),
                "valid_count": int(bucket["baseline_count"]),
            }
        )

    component_rows = []
    for (
        cycle,
        block,
        component,
        head,
    ), bucket in sorted(component_buckets.items()):
        learned = _final_metric(bucket, "learned", "accuracy")
        oracle = _final_metric(bucket, "oracle", "accuracy")
        learned_zero = _final_metric(
            bucket,
            "learned_zero",
            "accuracy",
        )
        oracle_zero = _final_metric(
            bucket,
            "oracle_zero",
            "accuracy",
        )
        patched = _final_metric(
            bucket,
            "oracle_patch",
            "accuracy",
        )
        denominator = oracle - learned
        component_rows.append(
            {
                "cycle": cycle,
                "block": block,
                "component": component,
                "head": head,
                "learned_accuracy": learned,
                "oracle_accuracy": oracle,
                "learned_zero_accuracy": learned_zero,
                "learned_zero_drop": learned - learned_zero,
                "oracle_zero_accuracy": oracle_zero,
                "oracle_zero_drop": oracle - oracle_zero,
                "oracle_patch_accuracy": patched,
                "oracle_patch_gain": patched - learned,
                "oracle_patch_normalized_recovery": (
                    (patched - learned) / denominator
                    if abs(denominator) > 1e-12
                    else float("nan")
                ),
            }
        )

    activation_rows = []
    grouped: dict[
        tuple[Any, ...],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in activation_parts:
        grouped[
            (
                row["cycle"],
                row["block"],
                row["scope"],
                row["component"],
                row["head"],
            )
        ].append(row)
    for key, parts in sorted(grouped.items()):
        activation_rows.append(
            {
                "cycle": key[0],
                "block": key[1],
                "scope": key[2],
                "component": key[3],
                "head": key[4],
                **{
                    metric: sum(float(row[metric]) for row in parts)
                    / len(parts)
                    for metric in (
                        "context_cosine",
                        "attention_pattern_cosine",
                        "update_norm_ratio",
                    )
                },
            }
        )

    state_rows = []
    grouped_states: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in state_parts:
        grouped_states[int(row["cycle"])].append(row)
    for cycle, parts in sorted(grouped_states.items()):
        state_rows.append(
            {
                "cycle": cycle,
                "answer_relative_mse": sum(
                    float(row["answer_relative_mse"]) for row in parts
                )
                / len(parts),
                "all_state_relative_mse": sum(
                    float(row["all_state_relative_mse"])
                    for row in parts
                )
                / len(parts),
            }
        )

    routing_rows = []
    grouped_routing: dict[
        tuple[int, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in routing_parts:
        grouped_routing[
            (int(row["cycle"]), str(row["state"]), str(row["role"]))
        ].append(row)
    for key, parts in sorted(grouped_routing.items()):
        routing_rows.append(
            {
                "cycle": key[0],
                "state": key[1],
                "role": key[2],
                "attention_mass": sum(
                    float(row["attention_mass"]) for row in parts
                )
                / len(parts),
                "argmax_hit_rate": sum(
                    float(row["argmax_hit_rate"]) for row in parts
                )
                / len(parts),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "behavior_rows.csv", behavior_rows)
    _write_csv(out_dir / "state_rows.csv", state_rows)
    _write_csv(out_dir / "activation_similarity_rows.csv", activation_rows)
    _write_csv(out_dir / "component_causal_rows.csv", component_rows)
    _write_csv(out_dir / "head2_routing_rows.csv", routing_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "operator": str(operator_path),
        "operator_training_relation": operator_payload[
            "training_relation"
        ],
        "device": str(device),
        "examples": batch_size * batches,
        "cycles": cycles,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
        "files": {
            "behavior": "behavior_rows.csv",
            "states": "state_rows.csv",
            "activation_similarity": "activation_similarity_rows.csv",
            "component_causal": "component_causal_rows.csv",
            "head2_routing": "head2_routing_rows.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare D8L8 simple-R and exact-young circuits across the "
            "success-to-failure boundary."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20262730)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_circuit(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        operator_path=args.operator,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        cycles=args.cycles,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
