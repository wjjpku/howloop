from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_jump_controller import (
    JumpMode,
    _aggregate_rows,
    _validate_modes,
    _write_csv,
    apply_vector_map,
    behavior_metrics,
    collect_jump_pair_batch,
    controller_position_groups,
)
from reasoning_loop.graph_path_jump_controller_causal_switch import (
    _load_controller,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed


def semantic_graph_positions(
    *,
    cfg,
    all_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    groups = explicit_depth_position_groups(cfg.node_count)
    device = all_targets.device
    edge_positions = torch.tensor(groups["edge_marker"], device=device)
    source_positions = torch.tensor(groups["source"], device=device)
    destination_positions = torch.tensor(groups["destination"], device=device)
    endpoint = all_targets[:, cfg.max_depth]
    one = all_targets[:, cfg.max_depth + 1]
    two = all_targets[:, cfg.max_depth + 2]
    random_node = (two + 3) % cfg.node_count
    collision = (
        random_node.eq(endpoint)
        | random_node.eq(one)
        | random_node.eq(two)
    )
    while bool(collision.any()):
        random_node = random_node.clone()
        random_node[collision] = (
            random_node[collision] + 1
        ) % cfg.node_count
        collision = (
            random_node.eq(endpoint)
            | random_node.eq(one)
            | random_node.eq(two)
        )

    result: dict[str, torch.Tensor] = {}
    for semantic_name, node in (
        ("current", endpoint),
        ("one", one),
        ("two", two),
        ("random", random_node),
    ):
        result[f"edge_{semantic_name}"] = edge_positions[node][:, None]
        result[f"source_{semantic_name}"] = source_positions[node][:, None]
        result[f"destination_{semantic_name}"] = destination_positions[node][
            :, None
        ]
    return result


def attention_routing_rows(
    *,
    trace: FunctionalTrace,
    cfg,
    all_targets: torch.Tensor,
    batch_index: int,
    controller_seed: int,
    mode: str,
) -> list[dict[str, Any]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"][0]
    graph = list(groups["graph"])
    query_work = list(
        groups["query_metadata"] + groups["answer"]
    )
    semantic_positions = semantic_graph_positions(
        cfg=cfg,
        all_targets=all_targets,
    )
    rows: list[dict[str, Any]] = []
    batch = torch.arange(all_targets.shape[0], device=all_targets.device)
    for site_index, site in enumerate(trace.sites):
        for head in range(cfg.n_heads):
            pattern = site.attention_pattern[:, head, answer]
            top_position = pattern.argmax(dim=-1)
            row: dict[str, Any] = {
                "batch": batch_index,
                "controller_seed": controller_seed,
                "mode": mode,
                "site": site_index,
                "block": site.block_index + 1,
                "head": head,
                "route_condition": (
                    f"seed{controller_seed}.{mode}."
                    f"B{site.block_index + 1}.H{head}"
                ),
                "graph_mass": float(pattern[:, graph].sum(dim=-1).mean()),
                "query_work_mass": float(
                    pattern[:, query_work].sum(dim=-1).mean()
                ),
                "attention_entropy": float(
                    -(
                        pattern.float()
                        * pattern.float().clamp_min(1e-12).log()
                    )
                    .sum(dim=-1)
                    .mean()
                ),
            }
            for name, positions in semantic_positions.items():
                selected = positions[:, 0]
                row[f"mass_{name}"] = float(
                    pattern[batch, selected].mean()
                )
                row[f"top_is_{name}"] = float(
                    top_position.eq(selected).float().mean()
                )
            rows.append(row)
    return rows


@torch.no_grad()
def run_semantic_routing(
    *,
    checkpoint: Path,
    controller_path: Path,
    out_dir: Path,
    device_name: str,
    controller_seeds: tuple[int, ...],
    eval_batch_size: int,
    eval_batches: int,
    eval_seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("JUMP_CONTROLLER_CUDA_MEMORY_FRACTION", "0.06")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)

    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    two_mode = JumpMode(
        name="two",
        reference_age=1,
        reference_path_before=2,
        programmed_jump=2,
    )
    _validate_modes(cfg, two_mode=two_mode)
    all_positions = controller_position_groups(cfg)["all"]
    answer = explicit_depth_position_groups(cfg.node_count)["answer"]
    loop_indices = (cfg.max_loops,)
    controllers = {
        seed: (
            _load_controller(
                path=controller_path,
                name=f"seed{seed}_J_one_rank8",
                device=device,
            ),
            _load_controller(
                path=controller_path,
                name=f"seed{seed}_J_two_task_lambda10_pre1",
                device=device,
            ),
        )
        for seed in controller_seeds
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(eval_seed)
    routing_rows: list[dict[str, Any]] = []
    onehot_rows: list[dict[str, Any]] = []
    for batch_index in range(eval_batches):
        pair = collect_jump_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=eval_batch_size,
            device=device,
            two_mode=two_mode,
        )
        semantic_positions = semantic_graph_positions(
            cfg=cfg,
            all_targets=pair.all_targets,
        )
        for controller_seed in controller_seeds:
            one_controller, two_controller = controllers[controller_seed]
            states = {
                "J_one": apply_vector_map(
                    pair.terminal,
                    positions=all_positions,
                    controller=one_controller,
                ),
                "J_two": apply_vector_map(
                    pair.terminal,
                    positions=all_positions,
                    controller=two_controller,
                ),
                "oracle_one": pair.one_target_state,
                "oracle_two": pair.two_target_state,
            }
            traces: dict[str, FunctionalTrace] = {}
            for mode, state in states.items():
                _, trace = run_instrumented_state(
                    model,
                    state,
                    loop_indices=loop_indices,
                )
                traces[mode] = trace
                routing_rows.extend(
                    attention_routing_rows(
                        trace=trace,
                        cfg=cfg,
                        all_targets=pair.all_targets,
                        batch_index=batch_index,
                        controller_seed=controller_seed,
                        mode=mode,
                    )
                )

            for mode in ("J_one", "J_two"):
                state = states[mode]
                for site in range(cfg.n_layers):
                    for head in range(cfg.n_heads):
                        for source_role in (
                            "edge_current",
                            "source_current",
                            "destination_current",
                            "edge_one",
                            "source_one",
                            "destination_one",
                            "destination_random",
                        ):
                            intervention = FunctionalIntervention(
                                site=site,
                                component="attention_pattern",
                                mode="onehot",
                                positions=answer,
                                heads=(head,),
                                dynamic_source_positions=(
                                    semantic_positions[source_role]
                                ),
                            )
                            logits, _ = run_instrumented_state(
                                model,
                                state,
                                loop_indices=loop_indices,
                                interventions=(intervention,),
                            )
                            row: dict[str, Any] = {
                                "batch": batch_index,
                                "controller_seed": controller_seed,
                                "mode": mode,
                                "site": site,
                                "block": site + 1,
                                "head": head,
                                "source_role": source_role,
                                "condition": (
                                    f"{mode}.B{site + 1}.H{head}."
                                    f"onehot_{source_role}"
                                ),
                            }
                            row.update(
                                behavior_metrics(
                                    logits,
                                    pair.all_targets,
                                    endpoint_position=cfg.max_depth,
                                )
                            )
                            onehot_rows.append(row)

    routing_summary = _aggregate_rows(
        routing_rows,
        key="route_condition",
    )
    # Keep every controller seed, block, head and source role distinct.
    for row in onehot_rows:
        row["run_condition"] = (
            f"seed{row['controller_seed']}.{row['condition']}"
        )
    onehot_summary = _aggregate_rows(
        onehot_rows,
        key="run_condition",
    )
    _write_csv(out_dir / "routing_rows.csv", routing_rows)
    _write_csv(out_dir / "routing_summary.csv", routing_summary)
    _write_csv(out_dir / "onehot_rows.csv", onehot_rows)
    _write_csv(out_dir / "onehot_summary.csv", onehot_summary)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "controller_path": str(controller_path),
        "controller_seeds": list(controller_seeds),
        "config": asdict(cfg),
        "loss_placement": "final_only",
        "trained_loop_count": cfg.max_loops,
        "evaluated_receiver_age": cfg.max_loops,
        "evaluated_extra_loops": 1,
        "shared_physical_blocks": cfg.n_layers,
        "effective_depth_at_training_horizon": cfg.n_layers * cfg.max_loops,
        "evaluation_examples_per_controller": eval_batch_size * eval_batches,
        "evaluation_seed": eval_seed,
        "two_mode": asdict(two_mode),
        "routing_summary": routing_summary,
        "onehot_summary": onehot_summary,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure which semantic graph edge each controller routes through "
            "and causally force individual heads to read chosen edges."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--controller-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=16401)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_semantic_routing(
        checkpoint=args.checkpoint,
        controller_path=args.controller_path,
        out_dir=args.out_dir,
        device_name=args.device,
        controller_seeds=tuple(args.controller_seeds),
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        eval_seed=args.eval_seed,
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
