from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.graph_path_twohop_reprogram_j import (
    _all_permutations_excluding,
    _permutation_digest,
    _sha256,
    _write_csv,
    distinct_step_mask,
    strict_unseen_permutations,
    validate_d8l8_onehop_config,
    write_summary_atomic,
)


@dataclass(frozen=True)
class StatePairBatch:
    source: torch.Tensor
    target: torch.Tensor
    current_node: torch.Tensor
    one_node: torch.Tensor
    two_node: torch.Tensor
    age: torch.Tensor


def select_active_state_pairs(
    states: torch.Tensor,
    path_targets: torch.Tensor,
    ages: torch.Tensor,
) -> StatePairBatch:
    if states.ndim != 4:
        raise ValueError("states must have [age, batch, position, feature] shape")
    if path_targets.ndim != 2 or path_targets.shape[0] != states.shape[1]:
        raise ValueError("path targets must have [batch, path] shape")
    if ages.ndim != 1 or ages.shape[0] != states.shape[1]:
        raise ValueError("ages must have one entry per example")
    max_age = min(states.shape[0] - 2, path_targets.shape[1] - 2)
    if bool((ages < 1).any()) or bool((ages > max_age).any()):
        raise ValueError(f"ages must lie in [1, {max_age}]")
    batch_index = torch.arange(states.shape[1], device=states.device)
    target_index = ages - 1
    return StatePairBatch(
        source=states[ages, batch_index],
        target=states[ages + 1, batch_index],
        current_node=path_targets[batch_index, target_index],
        one_node=path_targets[batch_index, target_index + 1],
        two_node=path_targets[batch_index, target_index + 2],
        age=ages,
    )


def fit_full_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> VectorAffine:
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must share [sample, feature] shape")
    if source.shape[0] <= source.shape[1]:
        raise ValueError("affine fit needs more samples than features")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    source = source.float()
    target = target.float()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    gram = centered_source.transpose(0, 1) @ centered_source
    cross = centered_source.transpose(0, 1) @ centered_target
    scale = gram.diagonal().mean().clamp_min(1e-6)
    weight = torch.linalg.solve(
        gram
        + ridge
        * scale
        * torch.eye(source.shape[1], device=source.device, dtype=source.dtype),
        cross,
    )
    bias = target_mean - source_mean @ weight
    return VectorAffine(
        weight=weight,
        bias=bias,
        update_rank=source.shape[1],
        fit_dimension=source.shape[1],
        retained_fit_energy=1.0,
    )


def _cat(parts: list[StatePairBatch]) -> StatePairBatch:
    return StatePairBatch(
        source=torch.cat([part.source for part in parts]),
        target=torch.cat([part.target for part in parts]),
        current_node=torch.cat([part.current_node for part in parts]),
        one_node=torch.cat([part.one_node for part in parts]),
        two_node=torch.cat([part.two_node for part in parts]),
        age=torch.cat([part.age for part in parts]),
    )


def _subset(batch: StatePairBatch, mask: torch.Tensor) -> StatePairBatch:
    return StatePairBatch(
        source=batch.source[mask],
        target=batch.target[mask],
        current_node=batch.current_node[mask],
        one_node=batch.one_node[mask],
        two_node=batch.two_node[mask],
        age=batch.age[mask],
    )


@torch.no_grad()
def collect_pairs(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    starts: list[int],
    ages: list[int],
    device: torch.device,
    batch_size: int,
) -> StatePairBatch:
    parts: list[StatePairBatch] = []
    for offset in range(0, len(permutations), batch_size):
        stop = min(offset + batch_size, len(permutations))
        successors = torch.tensor(
            permutations[offset:stop], dtype=torch.long, device=device
        )
        start = torch.tensor(starts[offset:stop], dtype=torch.long, device=device)
        age = torch.tensor(ages[offset:stop], dtype=torch.long, device=device)
        tokens, targets, _, _ = fixed_depth_batch(
            cfg,
            stop - offset,
            device,
            path_positions=cfg.max_depth + 2,
            successors=successors,
            start=start,
        )
        states = torch.stack(
            cache_states_with_initial(model, tokens, loops=cfg.max_loops)
        )
        parts.append(select_active_state_pairs(states, targets, age))
    return _cat(parts)


def build_calibration_pairs(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    excluded: set[tuple[int, ...]],
    examples: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[StatePairBatch, set[tuple[int, ...]]]:
    candidates = _all_permutations_excluding(cfg.node_count, excluded)
    generator = random.Random(seed)
    permutations = generator.choices(candidates, k=examples)
    starts = [generator.randrange(cfg.node_count) for _ in range(examples)]
    ages = [generator.randrange(1, 7) for _ in range(examples)]
    return (
        collect_pairs(
            model=model,
            cfg=cfg,
            permutations=permutations,
            starts=starts,
            ages=ages,
            device=device,
            batch_size=batch_size,
        ),
        set(permutations),
    )


def build_all_pairs(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    device: torch.device,
    batch_size: int,
) -> StatePairBatch:
    examples = [
        (permutation, start, age)
        for permutation in permutations
        for start in range(cfg.node_count)
        for age in range(1, 7)
    ]
    return collect_pairs(
        model=model,
        cfg=cfg,
        permutations=[row[0] for row in examples],
        starts=[row[1] for row in examples],
        ages=[row[2] for row in examples],
        device=device,
        batch_size=batch_size,
    )


@torch.no_grad()
def evaluate_map(
    *,
    model: nn.Module,
    controller: VectorAffine,
    dataset: StatePairBatch,
    batch_size: int,
) -> dict[str, float]:
    totals = {
        "n": 0,
        "distinct_n": 0,
        "mapped_current": 0,
        "mapped_one": 0,
        "mapped_two": 0,
        "target_one": 0,
        "post_one": 0,
        "post_two": 0,
        "distinct_post_two": 0,
        "mse_num": 0.0,
        "identity_mse_num": 0.0,
        "mse_den": 0.0,
    }
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = slice(offset, min(offset + batch_size, dataset.source.shape[0]))
        source = dataset.source[index]
        target = dataset.target[index]
        mapped = controller(source)
        mapped_prediction = logits_from_raw_state(model, mapped).argmax(dim=-1)
        target_prediction = logits_from_raw_state(model, target).argmax(dim=-1)
        output = model.apply_loop(
            mapped.to(dtype=source.dtype), loop_index=model.cfg.max_loops
        )
        post_prediction = logits_from_raw_state(model, output).argmax(dim=-1)
        current = dataset.current_node[index]
        one = dataset.one_node[index]
        two = dataset.two_node[index]
        distinct = distinct_step_mask(current, one, two)
        totals["n"] += source.shape[0]
        totals["distinct_n"] += int(distinct.sum())
        totals["mapped_current"] += int(mapped_prediction.eq(current).sum())
        totals["mapped_one"] += int(mapped_prediction.eq(one).sum())
        totals["mapped_two"] += int(mapped_prediction.eq(two).sum())
        totals["target_one"] += int(target_prediction.eq(one).sum())
        totals["post_one"] += int(post_prediction.eq(one).sum())
        totals["post_two"] += int(post_prediction.eq(two).sum())
        totals["distinct_post_two"] += int(
            post_prediction[distinct].eq(two[distinct]).sum()
        )
        totals["mse_num"] += float((mapped - target).square().sum())
        totals["identity_mse_num"] += float((source.float() - target).square().sum())
        centered = target.float() - target.float().mean(dim=0, keepdim=True)
        totals["mse_den"] += float(centered.square().sum())
    n = max(totals["n"], 1)
    distinct_n = max(totals["distinct_n"], 1)
    return {
        "examples": float(totals["n"]),
        "relative_state_mse": totals["mse_num"] / max(totals["mse_den"], 1e-12),
        "identity_relative_state_mse": totals["identity_mse_num"]
        / max(totals["mse_den"], 1e-12),
        "mapped_current_accuracy": totals["mapped_current"] / n,
        "mapped_one_accuracy": totals["mapped_one"] / n,
        "mapped_two_accuracy": totals["mapped_two"] / n,
        "target_state_one_accuracy": totals["target_one"] / n,
        "post_one_accuracy": totals["post_one"] / n,
        "post_two_accuracy": totals["post_two"] / n,
        "distinct_post_two_accuracy": totals["distinct_post_two"] / distinct_n,
    }


def _fit_map(dataset: StatePairBatch, ridge: float) -> VectorAffine:
    return fit_full_affine(
        dataset.source.reshape(-1, dataset.source.shape[-1]),
        dataset.target.reshape(-1, dataset.target.shape[-1]),
        ridge=ridge,
    )


def run_control(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    calibration_seed: int,
    calibration_examples: int,
    val_permutations: int,
    eval_permutations: int,
    batch_size: int,
    ridges: tuple[float, ...],
    eval_seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "initializing",
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "calibration_seed": calibration_seed,
    }
    write_summary_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30, device=torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    model.requires_grad_(False)
    set_seed(calibration_seed)
    formal_eval = strict_unseen_permutations(
        cfg.node_count, set(), count=eval_permutations, seed=eval_seed
    )
    validation = strict_unseen_permutations(
        cfg.node_count,
        set(formal_eval),
        count=val_permutations,
        seed=eval_seed + 1,
    )
    calibration, train_seen = build_calibration_pairs(
        model=model,
        cfg=cfg,
        excluded=set(formal_eval) | set(validation),
        examples=calibration_examples,
        seed=calibration_seed + 31_003,
        device=device,
        batch_size=batch_size,
    )
    validation_data = build_all_pairs(
        model=model,
        cfg=cfg,
        permutations=validation,
        device=device,
        batch_size=batch_size,
    )
    candidate_rows: list[dict[str, Any]] = []
    shared_maps: dict[float, VectorAffine] = {}
    for ridge in ridges:
        controller = _fit_map(calibration, ridge)
        shared_maps[ridge] = controller
        metrics = evaluate_map(
            model=model,
            controller=controller,
            dataset=validation_data,
            batch_size=batch_size,
        )
        row = {"family": "shared", "age": "all", "ridge": ridge, **metrics}
        candidate_rows.append(row)
        print(json.dumps({"event": "validation", **row}), flush=True)
    best_ridge = max(
        ridges,
        key=lambda ridge: next(
            row["distinct_post_two_accuracy"]
            for row in candidate_rows
            if row["family"] == "shared" and row["ridge"] == ridge
        ),
    )
    age_maps: dict[int, VectorAffine] = {}
    age_ridges: dict[int, float] = {}
    for age in range(1, 7):
        calibration_age = _subset(calibration, calibration.age.eq(age))
        validation_age = _subset(validation_data, validation_data.age.eq(age))
        choices: list[tuple[float, VectorAffine, dict[str, float]]] = []
        for ridge in ridges:
            controller = _fit_map(calibration_age, ridge)
            metrics = evaluate_map(
                model=model,
                controller=controller,
                dataset=validation_age,
                batch_size=batch_size,
            )
            candidate_rows.append(
                {"family": "age_specific", "age": age, "ridge": ridge, **metrics}
            )
            choices.append((ridge, controller, metrics))
        selected = max(choices, key=lambda row: row[2]["distinct_post_two_accuracy"])
        age_ridges[age], age_maps[age] = selected[0], selected[1]
    formal_data = build_all_pairs(
        model=model,
        cfg=cfg,
        permutations=formal_eval,
        device=device,
        batch_size=batch_size,
    )
    shared_formal = evaluate_map(
        model=model,
        controller=shared_maps[best_ridge],
        dataset=formal_data,
        batch_size=batch_size,
    )
    per_age_rows: list[dict[str, Any]] = []
    for age in range(1, 7):
        age_data = _subset(formal_data, formal_data.age.eq(age))
        shared_metrics = evaluate_map(
            model=model,
            controller=shared_maps[best_ridge],
            dataset=age_data,
            batch_size=batch_size,
        )
        specific_metrics = evaluate_map(
            model=model,
            controller=age_maps[age],
            dataset=age_data,
            batch_size=batch_size,
        )
        per_age_rows.append(
            {"family": "shared", "age": age, "ridge": best_ridge, **shared_metrics}
        )
        per_age_rows.append(
            {
                "family": "age_specific",
                "age": age,
                "ridge": age_ridges[age],
                **specific_metrics,
            }
        )
    age_specific_mean = sum(
        row["distinct_post_two_accuracy"]
        for row in per_age_rows
        if row["family"] == "age_specific"
    ) / 6
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "calibration_seed": calibration_seed,
        "calibration_examples": calibration_examples,
        "unique_calibration_permutations": len(train_seen),
        "strict_eval_permutation_sha256": _permutation_digest(formal_eval),
        "strict_eval_permutations": eval_permutations,
        "selected_shared_ridge": best_ridge,
        "selected_age_ridges": age_ridges,
        "shared_formal": shared_formal,
        "age_specific_distinct_two_mean": age_specific_mean,
        "candidate_rows": candidate_rows,
        "per_age_rows": per_age_rows,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    torch.save(
        {
            "shared": {
                "weight": shared_maps[best_ridge].weight.cpu(),
                "bias": shared_maps[best_ridge].bias.cpu(),
            },
            "age_specific": {
                age: {"weight": controller.weight.cpu(), "bias": controller.bias.cpu()}
                for age, controller in age_maps.items()
            },
        },
        out_dir / "state_successor_maps.pt",
    )
    _write_csv(out_dir / "candidate_rows.csv", candidate_rows)
    _write_csv(out_dir / "per_age_rows.csv", per_age_rows)
    write_summary_atomic(out_dir / "summary.json", summary)
    report = [
        "# D8L8 seed3 状态后继 affine 容量对照",
        "",
        f"- 真实 h_(t+1) 的一步 readout：{shared_formal['target_state_one_accuracy']:.4f}",
        f"- 共享 affine 映射后的一步 readout：{shared_formal['mapped_one_accuracy']:.4f}",
        f"- 共享 affine 接 F 后的 strict distinct 两跳：{shared_formal['distinct_post_two_accuracy']:.4f}",
        f"- 分年龄 affine 接 F 后的 strict distinct 两跳均值：{age_specific_mean:.4f}",
        f"- affine / identity 的相对 state MSE：{shared_formal['relative_state_mse']:.4f} / {shared_formal['identity_relative_state_mse']:.4f}",
        "",
        "该对照用闭式 ridge regression 拟合 h_t -> h_(t+1)，再接同一个冻结 F。",
    ]
    (out_dir / "REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": str(out_dir / "summary.json"),
        }
    )
    write_summary_atomic(out_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--calibration-examples", type=int, default=4096)
    parser.add_argument("--val-permutations", type=int, default=128)
    parser.add_argument("--eval-permutations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ridges", type=float, nargs="+", default=(0.001, 0.01, 0.1))
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_control(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_seed=args.calibration_seed,
        calibration_examples=args.calibration_examples,
        val_permutations=args.val_permutations,
        eval_permutations=args.eval_permutations,
        batch_size=args.batch_size,
        ridges=tuple(args.ridges),
        eval_seed=args.eval_seed,
    )
    print(json.dumps({
        "shared": summary["shared_formal"],
        "age_specific_mean": summary["age_specific_distinct_two_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
