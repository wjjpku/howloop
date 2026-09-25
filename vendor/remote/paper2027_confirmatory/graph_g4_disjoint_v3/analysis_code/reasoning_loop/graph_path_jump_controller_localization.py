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
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_jump_controller import (
    JumpMode,
    _aggregate_rows,
    _validate_modes,
    _write_csv,
    apply_vector_map,
    behavior_metrics,
    collect_jump_pair_batch,
    component_similarity,
    controller_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    run_one_loop,
)
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


def localization_groups(cfg) -> dict[str, tuple[int, ...]]:
    explicit = explicit_depth_position_groups(cfg.node_count)
    base = controller_position_groups(cfg)
    groups = {
        "answer": base["answer"],
        "registers": base["registers"],
        "query_work": base["query_work"],
        "graph": explicit["graph"],
        "graph_answer": base["graph_answer"],
        "all": base["all"],
    }
    all_positions = set(base["all"])
    for name in ("answer", "registers", "query_work", "graph"):
        groups[f"all_except_{name}"] = tuple(
            sorted(all_positions.difference(groups[name]))
        )
    return groups


def _load_controllers(
    *,
    path: Path,
    names: tuple[str, ...],
    device: torch.device,
) -> dict[str, VectorAffine]:
    payload = torch.load(path, map_location=device)
    missing = [name for name in names if name not in payload]
    if missing:
        raise KeyError(f"controller names are missing: {missing}")
    result = {}
    for name in names:
        item = payload[name]
        weight = item["weight"].to(device)
        result[name] = VectorAffine(
            weight=weight,
            bias=item["bias"].to(device),
            update_rank=weight.shape[0],
            fit_dimension=weight.shape[0],
            retained_fit_energy=1.0,
        )
    return result


@torch.no_grad()
def run_localization(
    *,
    checkpoint: Path,
    controller_path: Path,
    controller_names: tuple[str, ...],
    out_dir: Path,
    device_name: str,
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
    controllers = _load_controllers(
        path=controller_path,
        names=controller_names,
        device=device,
    )
    groups = localization_groups(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(eval_seed)
    rows: list[dict[str, Any]] = []
    for batch_index in range(eval_batches):
        pair = collect_jump_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=eval_batch_size,
            device=device,
            two_mode=two_mode,
        )
        one_oracle = run_one_loop(
            model,
            pair.one_target_state,
            loop_index=cfg.max_loops,
        )
        two_oracle = run_one_loop(
            model,
            pair.two_target_state,
            loop_index=cfg.max_loops,
        )
        conditions: list[tuple[str, str, torch.Tensor]] = [
            ("identity", "control", pair.terminal),
            ("exact_one", "oracle", pair.one_target_state),
            ("exact_two", "oracle", pair.two_target_state),
        ]
        for name, controller in controllers.items():
            mapped = apply_vector_map(
                pair.terminal,
                positions=groups["all"],
                controller=controller,
            )
            conditions.append((f"{name}__all", name, mapped))
            generator = torch.Generator(device=device)
            generator.manual_seed(
                eval_seed + 1009 * batch_index + len(conditions)
            )
            permutation = torch.randperm(
                mapped.shape[0],
                generator=generator,
                device=device,
            )
            conditions.append(
                (
                    f"{name}__all_batch_shuffled",
                    name,
                    mapped[permutation],
                )
            )
            for group_name, positions in groups.items():
                if group_name == "all":
                    continue
                conditions.append(
                    (
                        f"{name}__{group_name}",
                        name,
                        apply_vector_map(
                            pair.terminal,
                            positions=positions,
                            controller=controller,
                        ),
                    )
                )

        for condition, controller_name, state in conditions:
            preloop = behavior_metrics(
                logits_from_raw_state(model, state),
                pair.all_targets,
                endpoint_position=cfg.max_depth,
            )
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops,
            )
            row: dict[str, Any] = {
                "batch": batch_index,
                "condition": condition,
                "controller": controller_name,
            }
            row.update(
                {
                    f"preloop_{key}": value
                    for key, value in preloop.items()
                }
            )
            row.update(
                behavior_metrics(
                    step.logits,
                    pair.all_targets,
                    endpoint_position=cfg.max_depth,
                )
            )
            row.update(
                component_similarity(
                    step,
                    one_oracle,
                    cfg=cfg,
                    prefix="to_one_oracle",
                )
            )
            row.update(
                component_similarity(
                    step,
                    two_oracle,
                    cfg=cfg,
                    prefix="to_two_oracle",
                )
            )
            rows.append(row)

    condition_summary = _aggregate_rows(rows, key="condition")
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "controller_path": str(controller_path),
        "controller_names": list(controller_names),
        "config": asdict(cfg),
        "loss_placement": "final_only",
        "trained_loop_count": cfg.max_loops,
        "evaluated_receiver_age": cfg.max_loops,
        "evaluated_extra_loops": 1,
        "shared_physical_blocks": cfg.n_layers,
        "effective_depth_at_training_horizon": (
            cfg.n_layers * cfg.max_loops
        ),
        "two_mode": asdict(two_mode),
        "evaluation_examples": eval_batch_size * eval_batches,
        "evaluation_seed": eval_seed,
        "groups": {name: list(value) for name, value in groups.items()},
        "condition_summary": condition_summary,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    _write_csv(out_dir / "localization_per_batch.csv", rows)
    _write_csv(out_dir / "condition_summary.csv", condition_summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Localize learned one-vs-two-hop affine controllers and rule out "
            "pre-loop answer writing."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument(
        "--controller-names",
        nargs="+",
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=8401)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_localization(
        checkpoint=args.checkpoint,
        controller_path=args.controller_path,
        controller_names=tuple(args.controller_names),
        out_dir=args.out_dir,
        device_name=args.device,
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        eval_seed=args.eval_seed,
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
