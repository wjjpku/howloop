from __future__ import annotations

import argparse
import csv
import json
import math
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
    FunctionalTrace,
    _path_state_metrics,
    _roll_trace,
    _state_logits,
    explicit_depth_position_groups,
    run_instrumented,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _accuracy_margin(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    return (
        float(logits.argmax(-1).eq(target).float().mean()),
        float(target_margin(logits, target).mean()),
    )


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel():
        raise ValueError("cosine vectors must have the same length")
    if float(left.norm()) == 0.0 or float(right.norm()) == 0.0:
        return float("nan")
    return float(F.cosine_similarity(left, right, dim=0))


def _strongly_used(
    *,
    zero_accuracy_drop: float,
    zero_margin_drop: float,
    shuffle_accuracy_drop: float,
    shuffle_margin_drop: float,
) -> bool:
    zero_pass = zero_accuracy_drop >= 0.03 or zero_margin_drop >= 0.5
    shuffle_pass = (
        shuffle_accuracy_drop >= 0.03 or shuffle_margin_drop >= 0.5
    )
    return zero_pass and shuffle_pass


def _site_scope(
    *,
    block_index: int,
    groups: dict[str, tuple[int, ...]],
) -> tuple[str, tuple[int, ...]]:
    if block_index == 0:
        return "graph", groups["graph"]
    if block_index == 1:
        return "answer", groups["answer"]
    raise ValueError("five-model analysis expects exactly two physical blocks")


def _random_neurons(
    score: torch.Tensor,
    top: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    available = torch.ones(score.numel(), dtype=torch.bool, device=score.device)
    available[top] = False
    candidates = torch.where(available)[0]
    order = torch.randperm(
        candidates.numel(),
        generator=generator,
        device=score.device,
    )
    return candidates[order[: top.numel()]]


def _metric_drop(
    baseline: tuple[float, float],
    changed_logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float, float, float]:
    accuracy, margin = _accuracy_margin(changed_logits, target)
    return (
        accuracy,
        margin,
        baseline[0] - accuracy,
        baseline[1] - margin,
    )


def _attention_group_mass(
    pattern: torch.Tensor,
    *,
    head: int,
    query_positions: tuple[int, ...],
    source_positions: tuple[int, ...],
) -> float:
    query = torch.as_tensor(query_positions, device=pattern.device)
    source = torch.as_tensor(source_positions, device=pattern.device)
    selected = pattern[:, head][:, query[:, None], source[None, :]]
    return float(selected.sum(dim=-1).mean())


def _path_position_accuracy(
    logits: torch.Tensor,
    all_targets: torch.Tensor,
    *,
    path_position: int,
    endpoint_position: int,
) -> float:
    prediction = logits.argmax(dim=-1)
    target = all_targets[:, path_position]
    endpoint = all_targets[:, endpoint_position]
    valid = (
        torch.ones_like(target, dtype=torch.bool)
        if path_position == endpoint_position
        else target.ne(endpoint)
    )
    return (
        float(prediction[valid].eq(target[valid]).float().mean())
        if bool(valid.any())
        else float("nan")
    )


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if cfg.n_layers != 2:
        raise ValueError("five-model analysis requires two physical blocks")
    if top_k < 1 or top_k > cfg.d_mlp:
        raise ValueError("top_k must be in [1, d_mlp]")

    tokens, targets, _, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
    )
    target = targets[:, cfg.max_depth - 1]
    baseline_logits, trace = run_instrumented(
        model,
        tokens,
        max_loops=cfg.max_loops,
    )
    shuffled_trace = _roll_trace(trace)
    baseline = _accuracy_margin(baseline_logits, target)
    all_targets = _all_targets(start, targets)
    clean_local_paths: dict[int, tuple[int, float]] = {}
    for site_index, site in enumerate(trace.sites):
        best, accuracy, _ = _path_state_metrics(
            _state_logits(model, site.hidden_out),
            all_targets,
            endpoint_position=cfg.max_depth,
        )
        clean_local_paths[site_index] = (best, accuracy)
    groups = explicit_depth_position_groups(cfg.node_count)
    attention_sources = (
        "edge_marker",
        "source",
        "destination",
        "query_metadata",
        "answer",
    )

    head_rows: list[dict[str, Any]] = []
    head_position_rows: list[dict[str, Any]] = []
    head_vectors: dict[tuple[int, int, int], torch.Tensor] = {}
    for site_index, site in enumerate(trace.sites):
        scope, positions = _site_scope(
            block_index=site.block_index,
            groups=groups,
        )
        position_index = torch.as_tensor(positions, device=device)
        for head in range(cfg.n_heads):
            context = site.head_context[:, head, position_index]
            activation_rms = float(context.square().mean().sqrt())
            mean_vector = context.mean(dim=(0, 1)).detach().cpu()
            head_vectors[(site.block_index, head, site.loop_index)] = mean_vector
            zero = FunctionalIntervention(
                site=site_index,
                component="head_context",
                mode="zero",
                positions=positions,
                heads=(head,),
            )
            shuffled = FunctionalIntervention(
                site=site_index,
                component="head_context",
                mode="patch",
                positions=positions,
                heads=(head,),
            )
            zero_logits, _ = run_instrumented(
                model,
                tokens,
                max_loops=cfg.max_loops,
                interventions=(zero,),
            )
            shuffled_logits, _ = run_instrumented(
                model,
                tokens,
                max_loops=cfg.max_loops,
                interventions=(shuffled,),
                donor_trace=shuffled_trace,
            )
            zero_metrics = _metric_drop(baseline, zero_logits, target)
            shuffle_metrics = _metric_drop(
                baseline, shuffled_logits, target
            )
            row: dict[str, Any] = {
                "run": name,
                "loop": site.loop_index + 1,
                "block": site.block_index + 1,
                "head": head,
                "scope": scope,
                "activation_rms": activation_rms,
                "baseline_accuracy": baseline[0],
                "baseline_margin": baseline[1],
                "zero_accuracy": zero_metrics[0],
                "zero_margin": zero_metrics[1],
                "zero_accuracy_drop": zero_metrics[2],
                "zero_margin_drop": zero_metrics[3],
                "shuffle_accuracy": shuffle_metrics[0],
                "shuffle_margin": shuffle_metrics[1],
                "shuffle_accuracy_drop": shuffle_metrics[2],
                "shuffle_margin_drop": shuffle_metrics[3],
                "strongly_used": int(
                    _strongly_used(
                        zero_accuracy_drop=zero_metrics[2],
                        zero_margin_drop=zero_metrics[3],
                        shuffle_accuracy_drop=shuffle_metrics[2],
                        shuffle_margin_drop=shuffle_metrics[3],
                    )
                ),
            }
            for source_name in attention_sources:
                row[f"attention_to_{source_name}"] = _attention_group_mass(
                    site.attention_pattern,
                    head=head,
                    query_positions=positions,
                    source_positions=groups[source_name],
                )
            head_rows.append(row)

    head_position_scopes = (
        "edge_marker",
        "source",
        "destination",
        "query",
        "start",
        "depth",
        "answer",
    )
    for site_index, site in enumerate(trace.sites):
        for position_scope in head_position_scopes:
            positions = groups[position_scope]
            position_index = torch.as_tensor(positions, device=device)
            for head in range(cfg.n_heads):
                activation_rms = float(
                    site.head_context[:, head, position_index]
                    .square()
                    .mean()
                    .sqrt()
                )
                zero = FunctionalIntervention(
                    site=site_index,
                    component="head_context",
                    mode="zero",
                    positions=positions,
                    heads=(head,),
                )
                shuffled = FunctionalIntervention(
                    site=site_index,
                    component="head_context",
                    mode="patch",
                    positions=positions,
                    heads=(head,),
                )
                zero_logits, zero_trace = run_instrumented(
                    model,
                    tokens,
                    max_loops=cfg.max_loops,
                    interventions=(zero,),
                )
                shuffled_logits, shuffled_intervention_trace = run_instrumented(
                    model,
                    tokens,
                    max_loops=cfg.max_loops,
                    interventions=(shuffled,),
                    donor_trace=shuffled_trace,
                )
                zero_metrics = _metric_drop(
                    baseline,
                    zero_logits,
                    target,
                )
                shuffle_metrics = _metric_drop(
                    baseline,
                    shuffled_logits,
                    target,
                )
                clean_local_position, clean_local_accuracy = (
                    clean_local_paths[site_index]
                )
                zero_local_logits = _state_logits(
                    model,
                    zero_trace.sites[site_index].hidden_out,
                )
                shuffled_local_logits = _state_logits(
                    model,
                    shuffled_intervention_trace.sites[
                        site_index
                    ].hidden_out,
                )
                zero_local_position, _, _ = _path_state_metrics(
                    zero_local_logits,
                    all_targets,
                    endpoint_position=cfg.max_depth,
                )
                shuffle_local_position, _, _ = _path_state_metrics(
                    shuffled_local_logits,
                    all_targets,
                    endpoint_position=cfg.max_depth,
                )
                zero_local_accuracy = _path_position_accuracy(
                    zero_local_logits,
                    all_targets,
                    path_position=clean_local_position,
                    endpoint_position=cfg.max_depth,
                )
                shuffle_local_accuracy = _path_position_accuracy(
                    shuffled_local_logits,
                    all_targets,
                    path_position=clean_local_position,
                    endpoint_position=cfg.max_depth,
                )
                zero_local_drop = (
                    clean_local_accuracy - zero_local_accuracy
                )
                shuffle_local_drop = (
                    clean_local_accuracy - shuffle_local_accuracy
                )
                head_position_rows.append(
                    {
                        "run": name,
                        "loop": site.loop_index + 1,
                        "block": site.block_index + 1,
                        "head": head,
                        "position_scope": position_scope,
                        "activation_rms": activation_rms,
                        "zero_accuracy_drop": zero_metrics[2],
                        "zero_margin_drop": zero_metrics[3],
                        "shuffle_accuracy_drop": shuffle_metrics[2],
                        "shuffle_margin_drop": shuffle_metrics[3],
                        "strongly_used": int(
                            _strongly_used(
                                zero_accuracy_drop=zero_metrics[2],
                                zero_margin_drop=zero_metrics[3],
                                shuffle_accuracy_drop=shuffle_metrics[2],
                                shuffle_margin_drop=shuffle_metrics[3],
                            )
                        ),
                        "clean_local_path_position": clean_local_position,
                        "clean_local_path_accuracy": clean_local_accuracy,
                        "zero_local_best_path_position": (
                            zero_local_position
                        ),
                        "zero_local_path_accuracy": zero_local_accuracy,
                        "zero_local_path_accuracy_drop": zero_local_drop,
                        "shuffle_local_best_path_position": (
                            shuffle_local_position
                        ),
                        "shuffle_local_path_accuracy": (
                            shuffle_local_accuracy
                        ),
                        "shuffle_local_path_accuracy_drop": (
                            shuffle_local_drop
                        ),
                        "strongly_locally_used": int(
                            zero_local_drop >= 0.03
                            and shuffle_local_drop >= 0.03
                        ),
                    }
                )

    head_pair_rows: list[dict[str, Any]] = []
    for block in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            for left in range(cfg.max_loops):
                for right in range(left + 1, cfg.max_loops):
                    head_pair_rows.append(
                        {
                            "run": name,
                            "block": block + 1,
                            "head": head,
                            "left_loop": left + 1,
                            "right_loop": right + 1,
                            "loop_distance": right - left,
                            "mean_context_cosine": _cosine(
                                head_vectors[(block, head, left)],
                                head_vectors[(block, head, right)],
                            ),
                        }
                    )

    generator = torch.Generator(device=device).manual_seed(seed + 17)
    neuron_rows: list[dict[str, Any]] = []
    neuron_scores: dict[tuple[int, int], torch.Tensor] = {}
    top_sets: dict[tuple[int, int], torch.Tensor] = {}
    random_sets: dict[tuple[int, int], torch.Tensor] = {}
    site_lookup: dict[tuple[int, int], int] = {}
    for site_index, site in enumerate(trace.sites):
        scope, positions = _site_scope(
            block_index=site.block_index,
            groups=groups,
        )
        position_index = torch.as_tensor(positions, device=device)
        score = site.mlp_hidden[:, position_index].abs().mean(dim=(0, 1))
        top = score.topk(top_k).indices
        random = _random_neurons(score, top, generator=generator)
        key = (site.block_index, site.loop_index)
        neuron_scores[key] = score.detach().cpu()
        top_sets[key] = top
        random_sets[key] = random
        site_lookup[key] = site_index

        top_intervention = FunctionalIntervention(
            site=site_index,
            component="mlp_hidden",
            mode="zero",
            positions=positions,
            neurons=tuple(int(item) for item in top),
        )
        random_intervention = FunctionalIntervention(
            site=site_index,
            component="mlp_hidden",
            mode="zero",
            positions=positions,
            neurons=tuple(int(item) for item in random),
        )
        top_logits, _ = run_instrumented(
            model,
            tokens,
            max_loops=cfg.max_loops,
            interventions=(top_intervention,),
        )
        random_logits, _ = run_instrumented(
            model,
            tokens,
            max_loops=cfg.max_loops,
            interventions=(random_intervention,),
        )
        top_metrics = _metric_drop(baseline, top_logits, target)
        random_metrics = _metric_drop(baseline, random_logits, target)
        neuron_rows.append(
            {
                "run": name,
                "loop": site.loop_index + 1,
                "block": site.block_index + 1,
                "scope": scope,
                "top_k": top_k,
                "mean_activation_score": float(score.mean()),
                "top_activation_score": float(score[top].mean()),
                "random_activation_score": float(score[random].mean()),
                "top_neuron_ids": " ".join(str(int(item)) for item in top),
                "random_neuron_ids": " ".join(
                    str(int(item)) for item in random
                ),
                "top_zero_accuracy_drop": top_metrics[2],
                "top_zero_margin_drop": top_metrics[3],
                "random_zero_accuracy_drop": random_metrics[2],
                "random_zero_margin_drop": random_metrics[3],
                "specific_accuracy_drop": top_metrics[2] - random_metrics[2],
                "specific_margin_drop": top_metrics[3] - random_metrics[3],
            }
        )

    neuron_pair_rows: list[dict[str, Any]] = []
    for block in range(cfg.n_layers):
        for left in range(cfg.max_loops):
            for right in range(left + 1, cfg.max_loops):
                left_set = set(int(item) for item in top_sets[(block, left)])
                right_set = set(int(item) for item in top_sets[(block, right)])
                neuron_pair_rows.append(
                    {
                        "run": name,
                        "block": block + 1,
                        "scope": "graph" if block == 0 else "answer",
                        "left_loop": left + 1,
                        "right_loop": right + 1,
                        "loop_distance": right - left,
                        "top_neuron_jaccard": _jaccard(
                            left_set,
                            right_set,
                        ),
                        "activation_score_cosine": _cosine(
                            neuron_scores[(block, left)],
                            neuron_scores[(block, right)],
                        ),
                    }
                )

    transfer_rows: list[dict[str, Any]] = []
    for block in range(cfg.n_layers):
        scope, positions = _site_scope(block_index=block, groups=groups)
        for donor_loop in range(cfg.max_loops):
            top = top_sets[(block, donor_loop)]
            random = random_sets[(block, donor_loop)]
            for receiver_loop in range(cfg.max_loops):
                receiver_site = site_lookup[(block, receiver_loop)]
                top_intervention = FunctionalIntervention(
                    site=receiver_site,
                    component="mlp_hidden",
                    mode="zero",
                    positions=positions,
                    neurons=tuple(int(item) for item in top),
                )
                random_intervention = FunctionalIntervention(
                    site=receiver_site,
                    component="mlp_hidden",
                    mode="zero",
                    positions=positions,
                    neurons=tuple(int(item) for item in random),
                )
                top_logits, _ = run_instrumented(
                    model,
                    tokens,
                    max_loops=cfg.max_loops,
                    interventions=(top_intervention,),
                )
                random_logits, _ = run_instrumented(
                    model,
                    tokens,
                    max_loops=cfg.max_loops,
                    interventions=(random_intervention,),
                )
                top_metrics = _metric_drop(baseline, top_logits, target)
                random_metrics = _metric_drop(
                    baseline, random_logits, target
                )
                transfer_rows.append(
                    {
                        "run": name,
                        "block": block + 1,
                        "scope": scope,
                        "donor_loop": donor_loop + 1,
                        "receiver_loop": receiver_loop + 1,
                        "loop_distance": abs(receiver_loop - donor_loop),
                        "top_accuracy_drop": top_metrics[2],
                        "top_margin_drop": top_metrics[3],
                        "random_accuracy_drop": random_metrics[2],
                        "random_margin_drop": random_metrics[3],
                        "specific_accuracy_drop": (
                            top_metrics[2] - random_metrics[2]
                        ),
                        "specific_margin_drop": (
                            top_metrics[3] - random_metrics[3]
                        ),
                    }
                )

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "head_loop_rows.csv", head_rows)
    _write_csv(
        run_dir / "head_position_loop_rows.csv",
        head_position_rows,
    )
    _write_csv(run_dir / "head_loop_pair_rows.csv", head_pair_rows)
    _write_csv(run_dir / "neuron_loop_rows.csv", neuron_rows)
    _write_csv(run_dir / "neuron_loop_pair_rows.csv", neuron_pair_rows)
    _write_csv(run_dir / "neuron_transfer_rows.csv", transfer_rows)
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": asdict(cfg),
        "loss_mode": "final_only" if payload.get("aux_loss", 0.0) == 0.0 else "mixed",
        "baseline": {
            "endpoint_accuracy": baseline[0],
            "endpoint_margin": baseline[1],
        },
        "analysis": {
            "batch_size": batch_size,
            "top_k": top_k,
            "b1_scope": "graph positions",
            "b2_scope": "answer position",
            "head_strong_use_gate": (
                "zero and batch-shuffled replacement each cause "
                "accuracy drop >=0.03 or margin drop >=0.5"
            ),
            "head_position_scopes": list(head_position_scopes),
            "head_local_trajectory_gate": (
                "exploratory follow-up: zero and batch-shuffled replacement "
                "each reduce accuracy on the clean site's decoded path "
                "position by at least 0.03"
            ),
            "neuron_selection": "mean absolute held-out activation",
            "neuron_control": "matched random non-top set",
        },
    }
    if device.type == "cuda":
        summary["cuda_peak_memory_mib"] = {
            "allocated": torch.cuda.max_memory_allocated(device) / 2**20,
            "reserved": torch.cuda.max_memory_reserved(device) / 2**20,
        }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must have NAME=CHECKPOINT format")
    name, checkpoint = value.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("run must have NAME=CHECKPOINT format")
    return name, Path(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-loop head and MLP activation reuse."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for offset, (name, checkpoint) in enumerate(args.run):
        run_summary = args.out_dir / name / "summary.json"
        if run_summary.exists() and not args.force:
            summaries.append(json.loads(run_summary.read_text()))
            continue
        summaries.append(
            analyze_checkpoint(
                name=name,
                checkpoint=checkpoint,
                out_dir=args.out_dir,
                device=device,
                batch_size=args.batch_size,
                top_k=args.top_k,
                seed=args.seed + offset,
            )
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
