from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    Permutation,
    audit_partition,
    cycle_type,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_robust_j import (
    TrainableVectorAffine,
    WeightedAffineStats,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_unit_j import exact_interfaces


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratified_permutation_split(
    *,
    node_count: int,
    seed: int,
) -> tuple[list[Permutation], list[Permutation], dict[str, dict[str, int]]]:
    groups: dict[tuple[int, ...], list[Permutation]] = {}
    for permutation in itertools.permutations(range(node_count)):
        groups.setdefault(cycle_type(permutation), []).append(permutation)
    rng = random.Random(seed)
    train: list[Permutation] = []
    heldout: list[Permutation] = []
    distribution: dict[str, dict[str, int]] = {}
    give_extra_to_train = False
    for group in sorted(groups):
        values = groups[group]
        rng.shuffle(values)
        midpoint = len(values) // 2
        if len(values) % 2:
            give_extra_to_train = not give_extra_to_train
            midpoint += int(give_extra_to_train)
        train.extend(values[:midpoint])
        heldout.extend(values[midpoint:])
        label = "+".join(map(str, group))
        distribution[label] = {
            "train": midpoint,
            "heldout": len(values) - midpoint,
        }
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout, distribution


def sample_graph_current_batch(
    pool: torch.Tensor,
    *,
    batch_size: int,
    node_count: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(
        0,
        pool.shape[0],
        (batch_size,),
        generator=generator,
    )
    current = torch.randint(
        0,
        node_count,
        (batch_size,),
        generator=generator,
    )
    return (
        pool[indices].to(device=device, non_blocking=True),
        current.to(device=device, non_blocking=True),
    )


@dataclass
class Calibration:
    initial_map: VectorAffine
    bias: torch.Tensor
    position_feature_weights: torch.Tensor
    target_scale: torch.Tensor
    source_target_relative_mse: float
    graphs: int


@torch.no_grad()
def calibrate(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    train_pool: torch.Tensor,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    ridge: float,
    variance_floor: float,
) -> Calibration:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    stats = WeightedAffineStats(cfg.d_model)
    position_sum = torch.zeros(
        len(positions),
        cfg.d_model,
        dtype=torch.float64,
    )
    position_square_sum = torch.zeros_like(position_sum)
    delta_sum = torch.zeros(cfg.d_model, dtype=torch.float64)
    squared_error = 0.0
    target_variance_sum = 0.0
    count = 0
    for _ in range(batches):
        successors, current = sample_graph_current_batch(
            train_pool,
            batch_size=batch_size,
            node_count=cfg.node_count,
            generator=generator,
            device=device,
        )
        interfaces = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=current,
            ages=(7, 8),
            loop_index=cfg.max_loops,
        )
        source = interfaces[8]
        target = interfaces[7]
        flattened_source = source.flatten(0, 1)
        flattened_target = target.flatten(0, 1)
        stats.add(
            flattened_source,
            flattened_target,
            torch.ones(
                flattened_source.shape[0],
                device=device,
                dtype=torch.float32,
            ),
            group="random_graph_H8_to_H7",
        )
        target_cpu = target.double().cpu()
        position_sum += target_cpu.sum(dim=0)
        position_square_sum += target_cpu.square().sum(dim=0)
        delta_sum += (
            target.double() - source.double()
        ).sum(dim=(0, 1)).cpu()
        squared_error += float(
            (source.float() - target.float()).square().sum().item()
        )
        target_variance_sum += float(
            (
                target.float()
                - target.float().mean(dim=(0, 1), keepdim=True)
            )
            .square()
            .sum()
            .item()
        )
        count += source.shape[0]
    initial_map = stats.fit(ridge=ridge, device=device)
    position_mean = position_sum / count
    position_variance = (
        position_square_sum / count - position_mean.square()
    ).clamp_min(0.0)
    per_position_mean = position_variance.mean(dim=1, keepdim=True)
    denominator = position_variance + variance_floor * per_position_mean.clamp_min(
        1e-8
    )
    weights = denominator.reciprocal()
    weights /= weights.mean().clamp_min(1e-8)
    feature_rows = count * len(positions)
    return Calibration(
        initial_map=initial_map,
        bias=(delta_sum / feature_rows).to(device=device, dtype=torch.float32),
        position_feature_weights=weights.to(
            device=device,
            dtype=torch.float32,
        ),
        target_scale=torch.tensor(
            target_variance_sum / (feature_rows * cfg.d_model),
            device=device,
            dtype=torch.float32,
        ).clamp_min(1e-8),
        source_target_relative_mse=(
            squared_error / max(target_variance_sum, 1e-8)
        ),
        graphs=count,
    )


class BiasOnlyAffine(torch.nn.Module):
    def __init__(self, *, dimension: int, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", torch.eye(dimension, device=bias.device))
        self.bias = torch.nn.Parameter(bias.detach().clone())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() + self.bias

    def frozen(self) -> VectorAffine:
        return VectorAffine(
            weight=self.weight.detach().clone(),
            bias=self.bias.detach().clone(),
            update_rank=0,
            fit_dimension=self.weight.shape[0],
            retained_fit_energy=0.0,
        )


def hidden_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_scale: torch.Tensor,
    position_feature_weights: torch.Tensor | None,
) -> torch.Tensor:
    error = (prediction.float() - target.float()).square()
    if position_feature_weights is not None:
        error = error * position_feature_weights
    return error.mean() / target_scale


@torch.no_grad()
def collect_onpolicy_pairs(
    *,
    module: torch.nn.Module,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    successors: torch.Tensor,
    current: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=current,
        age=8,
        phase_position=phase_positions[8],
    )
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    live_current = current
    for cycle in range(1, horizon + 1):
        captured: list[torch.Tensor] = []

        def transform(value: torch.Tensor) -> torch.Tensor:
            captured.append(value.detach())
            return module(value)

        step = run_one_loop(
            model,
            state,
            loop_index=cfg.max_loops + cycle - 1,
            block2_position_transform=(positions, transform),
        )
        oracle = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=live_current,
            ages=(7,),
            loop_index=cfg.max_loops + cycle - 1,
        )[7]
        sources.append(captured[0])
        targets.append(oracle)
        state = step.state
        live_current = advance_nodes(successors, live_current, steps=1)
    return torch.cat(sources, dim=0), torch.cat(targets, dim=0)


def train_variant(
    *,
    label: str,
    module: torch.nn.Module,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    train_pool: torch.Tensor,
    device: torch.device,
    rounds: int,
    batch_size: int,
    horizons: Sequence[int],
    epochs_per_round: int,
    mini_batch_size: int,
    learning_rate: float,
    target_scale: torch.Tensor,
    position_feature_weights: torch.Tensor | None,
    seed: int,
) -> tuple[VectorAffine, list[dict[str, Any]]]:
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        round_seed = seed + 1000 * round_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(round_seed)
        successors, current = sample_graph_current_batch(
            train_pool,
            batch_size=batch_size,
            node_count=cfg.node_count,
            generator=generator,
            device=device,
        )
        horizon = int(horizons[(round_index - 1) % len(horizons)])
        source, target = collect_onpolicy_pairs(
            module=module,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=current,
            horizon=horizon,
        )
        optimizer_generator = torch.Generator(device=device)
        optimizer_generator.manual_seed(round_seed + 17)
        loss_sum = 0.0
        gradient_sum = 0.0
        optimizer_steps = 0
        for _ in range(epochs_per_round):
            order = torch.randperm(
                source.shape[0],
                generator=optimizer_generator,
                device=device,
            )
            for start in range(0, source.shape[0], mini_batch_size):
                index = order[start : start + mini_batch_size]
                prediction = module(source[index])
                loss = hidden_loss(
                    prediction,
                    target[index],
                    target_scale=target_scale,
                    position_feature_weights=position_feature_weights,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                loss_sum += float(loss.detach())
                gradient_sum += float(gradient)
                optimizer_steps += 1
        with torch.no_grad():
            final_loss = hidden_loss(
                module(source),
                target,
                target_scale=target_scale,
                position_feature_weights=position_feature_weights,
            )
        rows.append(
            {
                "variant": label,
                "round": round_index,
                "graph_seed": round_seed,
                "graphs": batch_size,
                "horizon": horizon,
                "state_pairs": int(source.shape[0]),
                "optimizer_steps": optimizer_steps,
                "mean_training_loss": loss_sum / optimizer_steps,
                "final_onpolicy_loss": float(final_loss),
                "mean_preclip_gradient_norm": gradient_sum / optimizer_steps,
                "learning_rate": learning_rate,
            }
        )
    return module.frozen(), rows


def curve_windows(accuracy: list[float]) -> dict[str, float]:
    def mean(start: int, stop: int) -> float:
        values = accuracy[start:stop]
        return float(sum(values) / len(values)) if values else float("nan")

    result = {
        "auc_1_8": mean(0, 8),
        "auc_1_24": mean(0, 24),
        "auc_25_48": mean(24, 48),
        "auc_49_64": mean(48, 64),
    }
    for cycle in (1, 8, 24, 48, 64):
        if cycle <= len(accuracy):
            result[f"cycle_{cycle}"] = float(accuracy[cycle - 1])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=190001)
    parser.add_argument("--calibration-seed", type=int, default=190002)
    parser.add_argument("--training-seed", type=int, default=190003)
    parser.add_argument("--evaluation-seed", type=int, default=190004)
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=(8, 16, 24, 32),
    )
    parser.add_argument("--epochs-per-round", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--variance-floor", type=float, default=0.1)
    parser.add_argument("--evaluation-train-graphs", type=int, default=512)
    parser.add_argument("--evaluation-heldout-graphs", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=3.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            0.04,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if (
        cfg.node_count != 8
        or cfg.max_depth != 8
        or cfg.max_loops != 8
        or cfg.n_layers != 2
    ):
        raise ValueError("experiment requires the frozen D8L8 N8 two-block model")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    train_permutations, heldout_permutations, split_distribution = (
        stratified_permutation_split(
            node_count=cfg.node_count,
            seed=args.split_seed,
        )
    )
    if set(train_permutations) & set(heldout_permutations):
        raise RuntimeError("train and held-out permutation pools overlap")
    train_pool = torch.tensor(train_permutations, dtype=torch.long)
    calibration = calibrate(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        train_pool=train_pool,
        device=device,
        batch_size=args.calibration_batch_size,
        batches=args.calibration_batches,
        seed=args.calibration_seed,
        ridge=args.ridge,
        variance_floor=args.variance_floor,
    )
    identity = torch.eye(cfg.d_model, device=device)
    modules: dict[str, torch.nn.Module] = {
        "raw_mse": TrainableVectorAffine(calibration.initial_map).to(device),
        "diag_whitened": TrainableVectorAffine(calibration.initial_map).to(device),
        "bias_only": BiasOnlyAffine(
            dimension=cfg.d_model,
            bias=calibration.bias,
        ).to(device),
    }
    maps: dict[str, VectorAffine] = {}
    training_rows: list[dict[str, Any]] = []
    for label, module in modules.items():
        feature_weights = (
            calibration.position_feature_weights
            if label == "diag_whitened"
            else None
        )
        trained, rows = train_variant(
            label=label,
            module=module,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            train_pool=train_pool,
            device=device,
            rounds=args.rounds,
            batch_size=args.training_batch_size,
            horizons=args.horizons,
            epochs_per_round=args.epochs_per_round,
            mini_batch_size=args.mini_batch_size,
            learning_rate=args.learning_rate,
            target_scale=calibration.target_scale,
            position_feature_weights=feature_weights,
            seed=args.training_seed,
        )
        maps[label] = trained
        training_rows.extend(rows)

    sampled_train, sampled_heldout, matched_distribution = stratified_samples(
        train_permutations,
        heldout_permutations,
        count=max(
            args.evaluation_train_graphs,
            args.evaluation_heldout_graphs,
        ),
        seed=args.evaluation_seed,
    )
    evaluations: list[dict[str, Any]] = []
    for label, age_map in maps.items():
        for partition, sample in (
            (
                "train_pool",
                sampled_train[: args.evaluation_train_graphs],
            ),
            (
                "strict_heldout_pool",
                sampled_heldout[: args.evaluation_heldout_graphs],
            ),
        ):
            result = audit_partition(
                label=f"{label}:{partition}",
                permutations=sample,
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                age_map=age_map,
                device=device,
                batch_size=args.evaluation_batch_size,
                continuation_loops=args.continuation_loops,
            )
            learned_accuracy = result["curves"][
                "learned_J_plus_full_Block2"
            ]["accuracy_by_cycle"]
            result["primary_windows"] = curve_windows(learned_accuracy)
            evaluations.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "graph_path_telomere_graph_blind_j",
            "checkpoint": str(args.checkpoint),
            "positions": positions,
            "maps": {
                label: {
                    "weight": age_map.weight.detach().cpu(),
                    "bias": age_map.bias.detach().cpu(),
                }
                for label, age_map in maps.items()
            },
        },
        args.out_dir / "graph_blind_j_maps.pt",
    )
    write_csv(args.out_dir / "training_rounds.csv", training_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            "frozen backbone final-only; J uses hidden-state age loss only;"
            " no successor labels, logits, or graph CE"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": args.continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "positions": positions,
        "J_parameter_count": cfg.d_model * cfg.d_model + cfg.d_model,
        "split": {
            "seed": args.split_seed,
            "train_permutations": len(train_permutations),
            "heldout_permutations": len(heldout_permutations),
            "overlap": 0,
            "cycle_type_distribution": split_distribution,
        },
        "calibration": {
            "graphs": calibration.graphs,
            "seed": args.calibration_seed,
            "ridge": args.ridge,
            "variance_floor": args.variance_floor,
            "natural_H8_to_H7_relative_mse": (
                calibration.source_target_relative_mse
            ),
            "target_scale": float(calibration.target_scale),
        },
        "training": {
            "successor_labels_used": False,
            "logits_used": False,
            "graph_CE_used": False,
            "rounds": args.rounds,
            "graphs_per_round": args.training_batch_size,
            "horizons": list(args.horizons),
            "epochs_per_round": args.epochs_per_round,
            "mini_batch_size": args.mini_batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.training_seed,
            "rows": training_rows,
        },
        "evaluation": {
            "seed": args.evaluation_seed,
            "matched_cycle_type_distribution": matched_distribution,
            "train_graphs_per_variant": args.evaluation_train_graphs,
            "heldout_graphs_per_variant": args.evaluation_heldout_graphs,
            "all_currents_per_graph": cfg.node_count,
            "results": evaluations,
        },
        "operator_diagnostics": {
            label: {
                "weight_minus_identity_frobenius": float(
                    (age_map.weight - identity).float().norm().item()
                ),
                "bias_norm": float(age_map.bias.float().norm().item()),
            }
            for label, age_map in maps.items()
        },
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "prelaunch_used_mib": args.prelaunch_used_mib,
            "prelaunch_free_mib": args.prelaunch_free_mib,
            "declared_peak_gib": args.declared_peak_gib,
            "reserve_gib": args.reserve_gib,
            "shared_gpu": args.shared_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "artifacts": {
            "maps": "graph_blind_j_maps.pt",
            "training_rounds": "training_rounds.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    compact = {
        result["partition"]: result["primary_windows"]
        for result in evaluations
    }
    print(
        json.dumps(
            {
                "status": "complete",
                "split": summary["split"],
                "primary_windows": compact,
                "gpu_runtime": summary["gpu_runtime"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    set_seed(0)
    main()
