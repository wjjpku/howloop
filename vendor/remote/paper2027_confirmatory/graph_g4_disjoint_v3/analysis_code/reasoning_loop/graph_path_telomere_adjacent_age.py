from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import (
    ReducedRankFamily,
    VectorAffine,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    _curve_summary,
    _weighted_position_samples,
    evaluate_shared_maps,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


@dataclass
class StreamingUpdateRegression:
    """Sufficient statistics for affine residual regression y = x + xM + b."""

    count: int
    sum_source: torch.Tensor
    sum_update: torch.Tensor
    source_gram: torch.Tensor
    source_update_cross: torch.Tensor

    @classmethod
    def empty(
        cls,
        dimension: int,
        *,
        device: torch.device,
    ) -> StreamingUpdateRegression:
        vector = torch.zeros(dimension, device=device, dtype=torch.float64)
        matrix = torch.zeros(
            dimension,
            dimension,
            device=device,
            dtype=torch.float64,
        )
        return cls(
            count=0,
            sum_source=vector.clone(),
            sum_update=vector.clone(),
            source_gram=matrix.clone(),
            source_update_cross=matrix.clone(),
        )

    def add(self, source: torch.Tensor, target: torch.Tensor) -> None:
        if source.shape != target.shape or source.ndim != 2:
            raise ValueError("source and target must share [sample, feature]")
        if source.shape[1] != self.sum_source.numel():
            raise ValueError("feature dimension does not match accumulator")
        x = source.to(dtype=torch.float64)
        delta = target.to(dtype=torch.float64) - x
        self.count += int(x.shape[0])
        self.sum_source += x.sum(dim=0)
        self.sum_update += delta.sum(dim=0)
        self.source_gram += x.transpose(0, 1) @ x
        self.source_update_cross += x.transpose(0, 1) @ delta

    def fit_family(self, *, ridge: float) -> ReducedRankFamily:
        if self.count <= self.sum_source.numel():
            raise ValueError("regression needs more samples than features")
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        count = float(self.count)
        source_mean = self.sum_source / count
        update_mean = self.sum_update / count
        centered_gram = self.source_gram - count * torch.outer(
            source_mean,
            source_mean,
        )
        centered_cross = self.source_update_cross - count * torch.outer(
            source_mean,
            update_mean,
        )
        scale = centered_gram.diagonal().mean().clamp_min(1e-6)
        regularized = centered_gram + ridge * scale * torch.eye(
            centered_gram.shape[0],
            device=centered_gram.device,
            dtype=centered_gram.dtype,
        )
        update_ols = torch.linalg.solve(regularized, centered_cross)
        fitted_output_gram = (
            update_ols.transpose(0, 1)
            @ centered_gram
            @ update_ols
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(
            fitted_output_gram
        )
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        eigenvectors = eigenvectors[:, order]
        return ReducedRankFamily(
            source_mean=source_mean.float(),
            target_mean=(source_mean + update_mean).float(),
            update_ols=update_ols.float(),
            output_right_vectors=eigenvectors.float(),
            fitted_singular_values=eigenvalues.sqrt().float(),
        )


@torch.no_grad()
def fit_adjacent_age_maps(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    answer_repeat: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    min_source_age: int,
    max_source_age: int,
    ranks: Sequence[int],
    ridge: float,
    seed: int,
) -> tuple[dict[str, VectorAffine], dict[str, Any]]:
    stats = StreamingUpdateRegression.empty(cfg.d_model, device=device)
    set_seed(seed)
    graph_count = 0
    pair_count_by_age: dict[int, int] = {}
    for _ in range(batches):
        _, targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = targets[:, cfg.max_depth - 1]
        interface_by_age: dict[int, torch.Tensor] = {}
        for age in range(min_source_age - 1, max_source_age + 1):
            hidden = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=age,
                phase_position=phase_positions[age],
            )
            interface_by_age[age] = run_one_loop(
                model,
                hidden,
                loop_index=cfg.max_loops,
            ).block2_hidden_pre_intervention
        for age in range(min_source_age, max_source_age + 1):
            source = _weighted_position_samples(
                interface_by_age[age],
                positions,
                answer_position=answer_position,
                answer_repeat=answer_repeat,
            )
            target = _weighted_position_samples(
                interface_by_age[age - 1],
                positions,
                answer_position=answer_position,
                answer_repeat=answer_repeat,
            )
            stats.add(source, target)
            pair_count_by_age[age] = (
                pair_count_by_age.get(age, 0) + int(source.shape[0])
            )
        graph_count += batch_size
    family = stats.fit_family(ridge=ridge)
    maps = {
        f"adjacent_w{answer_repeat}_r{int(rank)}": (
            family.map_for_rank(int(rank))
        )
        for rank in ranks
    }
    metadata = {
        "graphs": graph_count,
        "sample_pairs": stats.count,
        "sample_pairs_by_source_age": pair_count_by_age,
        "source_ages": list(range(min_source_age, max_source_age + 1)),
        "answer_repeat": answer_repeat,
        "ridge": ridge,
        "loss": "pooled adjacent-age shared positionwise state MSE only",
        "excluded": [
            "task CE",
            "power loss",
            "closed-loop loss",
            "DAgger rollouts",
        ],
    }
    return maps, metadata


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    min_source_age: int,
    max_source_age: int,
    ranks: Sequence[int],
    answer_repeat: int,
    ridge: float,
    calibration_seed: int,
    evaluation_seed: int,
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
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("experiment is fixed to the D8L8 two-block model")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    if not (
        3
        <= min_source_age
        <= max_source_age
        < len(phase_positions)
    ):
        raise ValueError("adjacent source ages must lie in the cached ladder")
    positions = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    answer_position = explicit_depth_position_groups(
        cfg.node_count
    )["answer"][0]
    maps, training = fit_adjacent_age_maps(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        answer_repeat=answer_repeat,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        min_source_age=min_source_age,
        max_source_age=max_source_age,
        ranks=ranks,
        ridge=ridge,
        seed=calibration_seed,
    )
    closed_rows = evaluate_shared_maps(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        maps=maps,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=evaluation_seed,
        executor_head=2,
    )
    curves = _curve_summary(closed_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "graph_path_telomere_adjacent_age",
            "positions": positions,
            "maps": {
                label: {
                    "weight": age_map.weight.cpu(),
                    "bias": age_map.bias.cpu(),
                    "rank": age_map.update_rank,
                    "retained_fit_energy": age_map.retained_fit_energy,
                }
                for label, age_map in maps.items()
            },
        },
        out_dir / "adjacent_age_maps.pt",
    )
    from reasoning_loop.graph_path_telomere_shared_position_dagger import (
        _write_csv,
    )

    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": {
            "node_count": cfg.node_count,
            "max_depth": cfg.max_depth,
            "max_loops": cfg.max_loops,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_mlp": cfg.d_mlp,
        },
        "device": str(device),
        "phase_positions": phase_positions,
        "shared_map_positions": list(positions),
        "training": training,
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "extra_loops": extra_loops,
        "closed_loop": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "maps": "adjacent_age_maps.pt",
            "closed_loop": "closed_loop_rows.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one shared affine R from pooled adjacent natural ages, then "
            "test R composed with the model loop without DAgger or task loss."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument("--min-source-age", type=int, default=3)
    parser.add_argument("--max-source-age", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=(32, 256))
    parser.add_argument("--answer-repeat", type=int, default=28)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=92101)
    parser.add_argument("--evaluation-seed", type=int, default=92102)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        extra_loops=args.extra_loops,
        min_source_age=args.min_source_age,
        max_source_age=args.max_source_age,
        ranks=args.ranks,
        answer_repeat=args.answer_repeat,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
