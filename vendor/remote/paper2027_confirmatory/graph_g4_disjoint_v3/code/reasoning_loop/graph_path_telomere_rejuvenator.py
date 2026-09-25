from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_lifespan_extension import (
    _accumulate,
    _blank_bucket,
    _write_csv,
    extension_position_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


@dataclass(frozen=True)
class PositionwiseAffine:
    weight: torch.Tensor
    bias: torch.Tensor

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError("state must have [batch, position, feature]")
        if state.shape[1:] != self.bias.shape:
            raise ValueError("state shape does not match affine map")
        return (
            torch.einsum("bpd,pde->bpe", state, self.weight)
            + self.bias.unsqueeze(0)
        )


@dataclass(frozen=True)
class FlattenedAffine:
    affine: PositionwiseAffine
    position_count: int
    feature_count: int

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[1:] != (
            self.position_count,
            self.feature_count,
        ):
            raise ValueError("state shape does not match flattened map")
        flat = state.reshape(state.shape[0], 1, -1)
        return self.affine(flat).reshape_as(state)


def fit_positionwise_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> PositionwiseAffine:
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(
            "source and target must share [sample, position, feature] shape"
        )
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    source = source.float()
    target = target.float()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    gram = torch.einsum(
        "npd,npe->pde",
        centered_source,
        centered_source,
    )
    cross = torch.einsum(
        "npd,npe->pde",
        centered_source,
        centered_target,
    )
    feature_count = source.shape[-1]
    scale = (
        gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        .clamp_min(1e-6)
    )
    identity = torch.eye(
        feature_count,
        device=source.device,
        dtype=source.dtype,
    ).unsqueeze(0)
    regularized = gram + ridge * scale[:, None, None] * identity
    weight = torch.linalg.solve(regularized, cross)
    bias = target_mean - torch.einsum(
        "pd,pde->pe",
        source_mean,
        weight,
    )
    return PositionwiseAffine(weight=weight, bias=bias)


def _relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    numerator = (prediction - target).square().mean()
    denominator = (
        target - target.mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-12)
    return float(numerator / denominator)


@torch.no_grad()
def collect_rejuvenator_pairs(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    positions: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    reference_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    if jump not in {1, 2}:
        raise ValueError("programmed jump must be one or two")
    collected: dict[str, list[torch.Tensor]] = {
        "initial_source": [],
        "initial_target": [],
        "cycle_source": [],
        "cycle_target": [],
    }
    set_seed(seed)
    for _ in range(batches):
        tokens, _, successors, start = fixed_depth_batch(
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
        reference_states = cache_states_with_initial(
            model,
            reference_tokens,
            loops=reference_age + 1,
        )
        next_reference = cache_states_with_initial(
            model,
            next_reference_tokens,
            loops=max(1, reference_age),
        )[reference_age]
        collected["initial_source"].append(
            terminal[:, list(positions)]
        )
        collected["initial_target"].append(
            reference_states[reference_age][:, list(positions)]
        )
        collected["cycle_source"].append(
            reference_states[reference_age + 1][
                :, list(positions)
            ]
        )
        collected["cycle_target"].append(
            next_reference[:, list(positions)]
        )
    return {
        key: torch.cat(value, dim=0)
        for key, value in collected.items()
    }


def fit_rejuvenator(
    pairs: dict[str, torch.Tensor],
    *,
    ridge: float,
) -> tuple[
    PositionwiseAffine | FlattenedAffine,
    PositionwiseAffine | FlattenedAffine,
]:
    position_count = pairs["initial_source"].shape[1]
    feature_count = pairs["initial_source"].shape[2]
    flatten = position_count <= 4

    def fit_pair(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> PositionwiseAffine | FlattenedAffine:
        if not flatten:
            return fit_positionwise_affine(
                source,
                target,
                ridge=ridge,
            )
        affine = fit_positionwise_affine(
            source.reshape(source.shape[0], 1, -1),
            target.reshape(target.shape[0], 1, -1),
            ridge=ridge,
        )
        return FlattenedAffine(
            affine=affine,
            position_count=position_count,
            feature_count=feature_count,
        )

    initial = fit_pair(
        pairs["initial_source"],
        pairs["initial_target"],
    )
    cycle = fit_pair(
        pairs["cycle_source"],
        pairs["cycle_target"],
    )
    return initial, cycle


@torch.no_grad()
def collect_dagger_cycle_pairs(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    positions: tuple[int, ...],
    initial_map: PositionwiseAffine | FlattenedAffine,
    cycle_map: PositionwiseAffine | FlattenedAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    sources = []
    targets = []
    index = list(positions)
    set_seed(seed)
    for _ in range(batches):
        tokens, _, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * extra_loops,
        )
        state = cache_states_with_initial(
            model,
            tokens,
            loops=cfg.max_loops,
        )[-1]
        state = state.clone()
        state[:, index] = initial_map(state[:, index])
        state = apply_shared_stack(
            model,
            state,
            loop_index=cfg.max_loops,
        )
        for extra_loop in range(2, extra_loops + 1):
            current_position = (
                cfg.max_depth + jump * (extra_loop - 1)
            )
            reference_start = advance_nodes(
                successors,
                start,
                steps=current_position - reference_position,
            )
            reference_tokens, _, _, _ = fixed_depth_batch(
                cfg,
                batch_size,
                device,
                path_positions=cfg.max_depth,
                successors=successors,
                start=reference_start,
            )
            reference = cache_states_with_initial(
                model,
                reference_tokens,
                loops=max(1, reference_age),
            )[reference_age]
            sources.append(state[:, index])
            targets.append(reference[:, index])
            state = state.clone()
            state[:, index] = cycle_map(state[:, index])
            state = apply_shared_stack(
                model,
                state,
                loop_index=cfg.max_loops + extra_loop - 1,
            )
    return torch.cat(sources, dim=0), torch.cat(targets, dim=0)


@torch.no_grad()
def refine_cycle_map_dagger(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    positions: tuple[int, ...],
    initial_map: PositionwiseAffine | FlattenedAffine,
    cycle_map: PositionwiseAffine | FlattenedAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    iterations: int,
    ridge: float,
    seed: int,
) -> tuple[
    PositionwiseAffine | FlattenedAffine,
    list[dict[str, float]],
]:
    history = []
    refined = cycle_map
    for iteration in range(iterations):
        source, target = collect_dagger_cycle_pairs(
            model=model,
            cfg=cfg,
            candidate=candidate,
            positions=positions,
            initial_map=initial_map,
            cycle_map=refined,
            device=device,
            batch_size=batch_size,
            batches=batches,
            extra_loops=extra_loops,
            seed=seed + iteration,
        )
        before = _relative_mse(refined(source), target)
        pairs = {
            "initial_source": source,
            "initial_target": target,
            "cycle_source": source,
            "cycle_target": target,
        }
        _, refined = fit_rejuvenator(pairs, ridge=ridge)
        after = _relative_mse(refined(source), target)
        history.append(
            {
                "iteration": float(iteration + 1),
                "training_relative_mse_before": before,
                "training_relative_mse_after": after,
                "training_pairs": float(source.shape[0]),
            }
        )
    return refined, history


@torch.no_grad()
def evaluate_rejuvenator(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    positions: tuple[int, ...],
    initial_map: PositionwiseAffine | FlattenedAffine,
    cycle_map: PositionwiseAffine | FlattenedAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    jump = int(candidate["programmed_jump"])
    conditions = (
        "baseline",
        "learned",
        "learned_shuffled",
        "learned_reverse",
    )
    accumulators = {
        (condition, extra_loop): _blank_bucket()
        for condition in conditions
        for extra_loop in range(1, extra_loops + 1)
    }
    set_seed(seed)
    for _ in range(batches):
        tokens, targets, _, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * extra_loops,
        )
        terminal = cache_states_with_initial(
            model,
            tokens,
            loops=cfg.max_loops,
        )[-1]
        all_targets = _all_targets(start, targets)
        endpoint = all_targets[:, cfg.max_depth]
        states = {
            condition: terminal.clone()
            for condition in conditions
        }
        index = list(positions)
        for extra_loop in range(1, extra_loops + 1):
            mapper = initial_map if extra_loop == 1 else cycle_map
            target = all_targets[
                :, cfg.max_depth + jump * extra_loop
            ]
            for condition in conditions:
                state = states[condition]
                if condition != "baseline":
                    prediction = mapper(state[:, index])
                    if condition == "learned_shuffled":
                        prediction = prediction.roll(1, dims=0)
                    elif condition == "learned_reverse":
                        prediction = 2 * state[:, index] - prediction
                    state = state.clone()
                    state[:, index] = prediction
                state = apply_shared_stack(
                    model,
                    state,
                    loop_index=cfg.max_loops + extra_loop - 1,
                )
                states[condition] = state
                metrics = _masked_metrics(
                    logits_from_raw_state(model, state),
                    target,
                    endpoint=endpoint,
                )
                _accumulate(
                    accumulators[(condition, extra_loop)],
                    metrics,
                )
    rows = []
    for condition in conditions:
        for extra_loop in range(1, extra_loops + 1):
            bucket = accumulators[(condition, extra_loop)]
            count = int(bucket["count"])
            rows.append(
                {
                    "condition": condition,
                    "extra_loop": extra_loop,
                    "target_offset": jump * extra_loop,
                    "accuracy": (
                        bucket["correct"] / count
                        if count
                        else float("nan")
                    ),
                    "probability": (
                        bucket["probability"] / count
                        if count
                        else float("nan")
                    ),
                    "margin": (
                        bucket["margin"] / count
                        if count
                        else float("nan")
                    ),
                    "valid_count": count,
                }
            )
    return rows


def select_rejuvenator_group(
    lifespan_summary: dict[str, Any],
) -> str:
    minimal = lifespan_summary["minimal_four_loop_replacement"]
    if minimal is not None:
        return str(minimal["group"])
    return str(lifespan_summary["best_exact_replacement"]["group"])


def _curve(
    rows: list[dict[str, Any]],
    condition: str,
) -> list[float]:
    return [
        float(row["accuracy"])
        for row in sorted(
            (
                row
                for row in rows
                if row["condition"] == condition
            ),
            key=lambda row: int(row["extra_loop"]),
        )
    ]


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    lifespan_summary_path: Path,
    out_dir: Path,
    device: torch.device,
    calibration_batch_size: int,
    calibration_batches: int,
    heldout_batch_size: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    ridge: float,
    dagger_iterations: int,
    dagger_batch_size: int,
    dagger_batches: int,
    seed: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    lifespan_summary = json.loads(
        lifespan_summary_path.read_text(encoding="utf-8")
    )
    candidate = lifespan_summary["candidate"]
    group = select_rejuvenator_group(lifespan_summary)
    positions = extension_position_groups(cfg)[group]
    calibration = collect_rejuvenator_pairs(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=seed,
    )
    initial_map, cycle_map = fit_rejuvenator(
        calibration,
        ridge=ridge,
    )
    cycle_map, dagger_history = refine_cycle_map_dagger(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        initial_map=initial_map,
        cycle_map=cycle_map,
        device=device,
        batch_size=dagger_batch_size,
        batches=dagger_batches,
        extra_loops=extra_loops,
        iterations=dagger_iterations,
        ridge=ridge,
        seed=seed + 10,
    )
    heldout = collect_rejuvenator_pairs(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        device=device,
        batch_size=heldout_batch_size,
        batches=1,
        seed=seed + 1,
    )
    fit_metrics = {
        "initial_relative_mse": _relative_mse(
            initial_map(heldout["initial_source"]),
            heldout["initial_target"],
        ),
        "cycle_relative_mse": _relative_mse(
            cycle_map(heldout["cycle_source"]),
            heldout["cycle_target"],
        ),
    }
    rows = evaluate_rejuvenator(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        initial_map=initial_map,
        cycle_map=cycle_map,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=seed + 2,
    )
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "learned_rejuvenator_rows.csv", rows)
    learned_curve = _curve(rows, "learned")
    shuffled_curve = _curve(rows, "learned_shuffled")
    reverse_curve = _curve(rows, "learned_reverse")
    baseline_curve = _curve(rows, "baseline")
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "candidate": candidate,
        "group": group,
        "position_count": len(positions),
        "map_structure": (
            "flattened_affine"
            if len(positions) <= 4
            else "positionwise_affine"
        ),
        "ridge": ridge,
        "calibration_samples": (
            calibration_batch_size * calibration_batches
        ),
        "heldout_samples": heldout_batch_size,
        "evaluation_samples": (
            evaluation_batch_size * evaluation_batches
        ),
        "fit_metrics": fit_metrics,
        "dagger_iterations": dagger_iterations,
        "dagger_history": dagger_history,
        "accuracy_by_extra_loop": learned_curve,
        "mean_accuracy": sum(learned_curve) / len(learned_curve),
        "baseline_accuracy_by_extra_loop": baseline_curve,
        "shuffled_accuracy_by_extra_loop": shuffled_curve,
        "reverse_accuracy_by_extra_loop": reverse_curve,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path, Path]:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,LIFESPAN_SUMMARY"
        )
    paths = parts[1].split(",", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,LIFESPAN_SUMMARY"
        )
    return parts[0], Path(paths[0]), Path(paths[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit donor-free high-dimensional recurrent-state rejuvenators "
            "and evaluate their closed-loop generalization."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--heldout-batch-size", type=int, default=512)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--dagger-iterations", type=int, default=3)
    parser.add_argument("--dagger-batch-size", type=int, default=128)
    parser.add_argument("--dagger-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = {}
    for run_index, (name, checkpoint, lifespan_summary) in enumerate(
        args.run
    ):
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            lifespan_summary_path=lifespan_summary,
            out_dir=args.out_dir,
            device=device,
            calibration_batch_size=args.calibration_batch_size,
            calibration_batches=args.calibration_batches,
            heldout_batch_size=args.heldout_batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            evaluation_batches=args.evaluation_batches,
            extra_loops=args.extra_loops,
            ridge=args.ridge,
            dagger_iterations=args.dagger_iterations,
            dagger_batch_size=args.dagger_batch_size,
            dagger_batches=args.dagger_batches,
            seed=args.seed + 1000 * run_index,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
