from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.long_cycle_graph_path import make_single_cycle_successors


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    name, raw_path = text.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    return name, Path(raw_path)


def wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("successes and total are inconsistent")
    z = 1.959963984540054
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


@torch.no_grad()
def _accuracy_matrix(
    *,
    model: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    batch_size: int,
    batches: int,
    evaluated_loops: int,
    path_positions: int,
    single_cycle: bool,
) -> torch.Tensor:
    correct = torch.zeros(
        evaluated_loops,
        path_positions,
        dtype=torch.float64,
        device=device,
    )
    total = 0
    for _ in range(batches):
        successors = (
            make_single_cycle_successors(
                batch_size=batch_size,
                node_count=cfg.node_count,
                device=device,
            )
            if single_cycle
            else None
        )
        tokens, targets, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
            successors=successors,
        )
        logits = model.forward_all(
            tokens,
            max_loops=evaluated_loops,
        )["logits_by_loop"]
        predictions = logits.argmax(dim=-1)
        for loop_index in range(evaluated_loops):
            correct[loop_index] += predictions[:, loop_index, None].eq(
                targets
            ).sum(dim=0)
        total += batch_size
    return correct.div(total)


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint, device)
    evaluated_loops = cfg.max_loops + extra_loops
    path_positions = max(cfg.max_depth, evaluated_loops)
    uniform_accuracy = _accuracy_matrix(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        batches=batches,
        evaluated_loops=evaluated_loops,
        path_positions=path_positions,
        single_cycle=False,
    )
    accuracy = _accuracy_matrix(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        batches=batches,
        evaluated_loops=evaluated_loops,
        path_positions=path_positions,
        single_cycle=True,
    )
    trained_diagonal = torch.stack(
        [
            accuracy[loop_index, loop_index]
            for loop_index in range(cfg.max_loops)
        ]
    )
    overloop_diagonal = torch.stack(
        [
            uniform_accuracy[loop_index, loop_index]
            for loop_index in range(cfg.max_loops, evaluated_loops)
        ]
    )
    uniform_trained_diagonal = torch.stack(
        [
            uniform_accuracy[loop_index, loop_index]
            for loop_index in range(cfg.max_loops)
        ]
    )
    best_accuracy, best_position = accuracy.max(dim=1)
    endpoint_accuracy = accuracy[
        cfg.max_loops - 1,
        cfg.max_depth - 1,
    ]
    matched_min = trained_diagonal.min()
    matched_mean = trained_diagonal.mean()
    stride_pass = bool(
        endpoint_accuracy >= 0.99
        and matched_min >= 0.80
    )
    strong_stride_pass = bool(
        endpoint_accuracy >= 0.99
        and matched_min >= 0.95
    )
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "data_seed": payload.get("data_seed"),
        "config": payload["config"],
        "examples_per_distribution": batch_size * batches,
        "primary_distribution": (
            "uniform random single-cycle permutations; f^1..f^6 are distinct"
        ),
        "endpoint_accuracy": float(endpoint_accuracy),
        "matched_step_accuracy": trained_diagonal.cpu().tolist(),
        "matched_step_min_accuracy": float(matched_min),
        "matched_step_mean_accuracy": float(matched_mean),
        "stride_pass": stride_pass,
        "strong_stride_pass": strong_stride_pass,
        "overloop_matched_accuracy": overloop_diagonal.cpu().tolist(),
        "overloop_matched_mean_accuracy": (
            float(overloop_diagonal.mean())
            if overloop_diagonal.numel()
            else None
        ),
        "best_position_by_loop": (best_position + 1).cpu().tolist(),
        "best_accuracy_by_loop": best_accuracy.cpu().tolist(),
        "accuracy_by_loop_and_path_position": accuracy.cpu().tolist(),
        "uniform_permutation": {
            "endpoint_accuracy": float(
                uniform_accuracy[
                    cfg.max_loops - 1,
                    cfg.max_depth - 1,
                ]
            ),
            "matched_step_accuracy": uniform_trained_diagonal.cpu().tolist(),
            "matched_step_min_accuracy": float(
                uniform_trained_diagonal.min()
            ),
            "matched_step_mean_accuracy": float(
                uniform_trained_diagonal.mean()
            ),
            "accuracy_by_loop_and_path_position": (
                uniform_accuracy.cpu().tolist()
            ),
        },
        "gate_definition": {
            "stride_pass": (
                "on single-cycle held-out permutations, endpoint accuracy "
                ">= 0.99 and every trained loop t has accuracy >= 0.80 for "
                "f^t(start)"
            ),
            "strong_stride_pass": (
                "on single-cycle held-out permutations, endpoint accuracy "
                ">= 0.99 and every trained loop t has accuracy >= 0.95 for "
                "f^t(start)"
            ),
        },
    }


def aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    stride_count = sum(bool(run["stride_pass"]) for run in runs)
    strong_count = sum(bool(run["strong_stride_pass"]) for run in runs)
    total = len(runs)
    return {
        "run_count": total,
        "stride_success_count": stride_count,
        "stride_success_rate": stride_count / total,
        "stride_success_wilson95": wilson_interval(stride_count, total),
        "strong_stride_success_count": strong_count,
        "strong_stride_success_rate": strong_count / total,
        "strong_stride_success_wilson95": wilson_interval(strong_count, total),
        "endpoint_accuracy_mean": sum(
            float(run["endpoint_accuracy"]) for run in runs
        )
        / total,
        "matched_step_min_accuracy_mean": sum(
            float(run["matched_step_min_accuracy"]) for run in runs
        )
        / total,
        "matched_step_mean_accuracy_mean": sum(
            float(run["matched_step_mean_accuracy"]) for run in runs
        )
        / total,
        "overloop_matched_mean_accuracy_mean": sum(
            float(run["overloop_matched_mean_accuracy"]) for run in runs
        )
        / total,
    }


def write_rows(path: Path, runs: Sequence[dict[str, Any]]) -> None:
    rows = [
        {
            "name": run["name"],
            "checkpoint_step": run["checkpoint_step"],
            "initialization_seed": run["initialization_seed"],
            "data_seed": run["data_seed"],
            "physical_blocks": run["config"]["n_layers"],
            "endpoint_accuracy": run["endpoint_accuracy"],
            "matched_step_min_accuracy": run["matched_step_min_accuracy"],
            "matched_step_mean_accuracy": run["matched_step_mean_accuracy"],
            "stride_pass": run["stride_pass"],
            "strong_stride_pass": run["strong_stride_pass"],
            "overloop_matched_mean_accuracy": run[
                "overloop_matched_mean_accuracy"
            ],
        }
        for run in runs
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavioral screen for emergent one-step graph-path loops."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--extra-loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.batches < 1 or args.extra_loops < 0:
        raise ValueError("batch size and batches must be positive; extra loops nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    runs = [
        analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            extra_loops=args.extra_loops,
        )
        for name, checkpoint in args.run
    ]
    payload = {
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_rows(args.out_dir / "rows.csv", runs)
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
