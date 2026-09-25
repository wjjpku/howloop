from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.twohop_in_context import (
    TwoHopConfig,
    TwoHopTransformer,
    accuracy_margin,
    build_twohop_model,
    count_parameters,
    cross_entropy_by_depth,
    make_twohop_batch,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    period: int
    description: str


MODEL_SPECS = {
    "S6": ModelSpec(
        name="S6",
        architecture="standard",
        period=6,
        description="six independent blocks",
    ),
    "P1x6": ModelSpec(
        name="P1x6",
        architecture="periodic",
        period=1,
        description="one block reused at all six effective depths",
    ),
    "P2x3": ModelSpec(
        name="P2x3",
        architecture="periodic",
        period=2,
        description="two-block stack repeated three times",
    ),
    "P3x2": ModelSpec(
        name="P3x2",
        architecture="periodic",
        period=3,
        description="three-block stack repeated twice",
    ),
}

EVAL_MODES = ("topological", "adjacent", "grouped", "reversed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train compute-matched standard and periodic Transformers on "
            "two-hop in-context retrieval with distractors."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=["S6", "P2x3"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--entity-count", type=int, default=64)
    parser.add_argument("--chain-count", type=int, default=5)
    parser.add_argument("--total-depth", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--final-eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overloop-depth", type=int, default=12)
    parser.add_argument("--init-seed-base", type=int, default=17000)
    parser.add_argument("--data-seed-base", type=int, default=27000)
    parser.add_argument("--eval-seed-base", type=int, default=37000)
    parser.add_argument(
        "--save-eval-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--expected-peak-mib", type=int, default=4096)
    parser.add_argument("--reserve-mib", type=int, default=16384)
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.05,
        help="Per-process CUDA memory cap as a fraction of the visible GPU.",
    )
    parser.add_argument(
        "--shared-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/twohop_standard_vs_periodic"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def config_for_spec(args: argparse.Namespace, spec: ModelSpec) -> TwoHopConfig:
    if args.total_depth != 6:
        if spec.name == "S6":
            period = args.total_depth
        elif args.total_depth % spec.period == 0:
            period = spec.period
        else:
            raise ValueError(
                f"{spec.name} period {spec.period} does not divide depth {args.total_depth}"
            )
    else:
        period = spec.period
    return TwoHopConfig(
        entity_count=args.entity_count,
        chain_count=args.chain_count,
        total_depth=args.total_depth,
        architecture=spec.architecture,
        period=period,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
    )


def cosine_lr(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type="cuda" if device.type == "cuda" else device.type,
        dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


@torch.inference_mode()
def evaluate_model(
    model: TwoHopTransformer,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    order_mode: str,
    active_depth: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    generator = generator_for(device, seed)
    depth_count = active_depth
    correct_by_depth = torch.zeros(depth_count, dtype=torch.float64)
    loss_by_depth = torch.zeros(depth_count, dtype=torch.float64)
    margin_by_depth = torch.zeros(depth_count, dtype=torch.float64)
    target_probability_by_depth = torch.zeros(depth_count, dtype=torch.float64)
    distractor_probability_by_depth = torch.zeros(
        depth_count,
        dtype=torch.float64,
    )
    examples = 0
    for _ in range(batches):
        batch = make_twohop_batch(
            model.cfg,
            batch_size,
            device=device,
            generator=generator,
            order_mode=order_mode,
        )
        with _autocast_context(device, amp_enabled):
            result = model.forward_all(
                batch.tokens,
                active_depth=active_depth,
            )
            logits_by_depth = result["logits_by_depth"]
            if not isinstance(logits_by_depth, torch.Tensor):
                raise RuntimeError("logits_by_depth is not a tensor")
            losses = cross_entropy_by_depth(logits_by_depth, batch.labels)
        probabilities = torch.softmax(logits_by_depth.float(), dim=-1)
        predictions = logits_by_depth.argmax(dim=-1)
        correct_by_depth += predictions.eq(
            batch.labels[:, None]
        ).sum(dim=0).double().cpu()
        loss_by_depth += losses.sum(dim=0).double().cpu()
        correct_logits = logits_by_depth.gather(
            2,
            batch.labels[:, None, None].expand(-1, depth_count, 1),
        ).squeeze(2)
        competitors = logits_by_depth.clone()
        competitors.scatter_(
            2,
            batch.labels[:, None, None].expand(-1, depth_count, 1),
            torch.finfo(logits_by_depth.dtype).min,
        )
        margin_by_depth += (
            correct_logits - competitors.max(dim=2).values
        ).sum(dim=0).double().cpu()
        target_probability_by_depth += probabilities.gather(
            2,
            batch.labels[:, None, None].expand(-1, depth_count, 1),
        ).squeeze(2).sum(dim=0).double().cpu()
        end_entities = batch.chains[:, :, 2]
        end_probabilities = probabilities.gather(
            2,
            end_entities[:, None, :].expand(-1, depth_count, -1),
        )
        target_mask = F.one_hot(
            batch.target_indices,
            num_classes=model.cfg.chain_count,
        ).bool()
        distractor_probability_by_depth += (
            end_probabilities.masked_fill(
                target_mask[:, None],
                0.0,
            ).sum(dim=2)
            / (model.cfg.chain_count - 1)
        ).sum(dim=0).double().cpu()
        examples += batch_size
    return {
        "order_mode": order_mode,
        "active_depth": active_depth,
        "examples": examples,
        "depth_accuracy": (correct_by_depth / examples).tolist(),
        "depth_loss": (loss_by_depth / examples).tolist(),
        "depth_margin": (margin_by_depth / examples).tolist(),
        "depth_target_probability": (
            target_probability_by_depth / examples
        ).tolist(),
        "depth_distractor_probability": (
            distractor_probability_by_depth / examples
        ).tolist(),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    # A completed optimization without a readable checkpoint is a failure.
    torch.load(path, map_location="cpu", weights_only=False)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"depth_accuracy", "depth_loss", "depth_margin"}
        }
        for name in ("depth_accuracy", "depth_loss", "depth_margin"):
            for depth, value in enumerate(row[name], start=1):
                flat[f"{name}_{depth}"] = value
        flat_rows.append(flat)
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def train_one(
    args: argparse.Namespace,
    spec: ModelSpec,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    cfg = config_for_spec(args, spec)
    run_name = f"{spec.name}_seed{seed}"
    run_dir = args.out_dir / "runs" / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    init_seed = args.init_seed_base + seed
    data_seed = args.data_seed_base + seed
    eval_seed = args.eval_seed_base + seed
    set_seed(init_seed)
    model = build_twohop_model(cfg, seed=init_seed, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    train_generator = generator_for(device, data_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_step = -1
    start_time = time.time()

    def record(step: int, train_loss: float | None, lr: float) -> dict[str, Any]:
        metrics = evaluate_model(
            model,
            device=device,
            batch_size=args.eval_batch_size,
            batches=args.eval_batches,
            seed=eval_seed,
            order_mode="topological",
            active_depth=cfg.total_depth,
            amp_enabled=args.amp,
        )
        row = {
            "step": step,
            "lr": lr,
            "train_loss": train_loss,
            "elapsed_sec": time.time() - start_time,
            "depth_accuracy": metrics["depth_accuracy"],
            "depth_loss": metrics["depth_loss"],
            "depth_margin": metrics["depth_margin"],
        }
        history.append(row)
        return row

    initial_row = record(0, None, 0.0)
    best_state = cpu_state_dict(model)
    best_metrics = initial_row
    best_step = 0
    for step in range(1, args.steps + 1):
        model.train()
        lr = cosine_lr(
            step - 1,
            base_lr=args.lr,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = make_twohop_batch(
            cfg,
            args.batch_size,
            device=device,
            generator=train_generator,
            order_mode="topological",
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, args.amp):
            logits = model(batch.tokens)
            loss = F.cross_entropy(logits, batch.labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_eval = (
            step == 1
            or step % args.eval_every == 0
            or step == args.steps
        )
        if should_eval:
            row = record(step, float(loss.item()), lr)
            final_acc = row["depth_accuracy"][-1]
            final_margin = row["depth_margin"][-1]
            assert best_metrics is not None
            best_acc = best_metrics["depth_accuracy"][-1]
            best_margin = best_metrics["depth_margin"][-1]
            if final_acc > best_acc or (
                final_acc == best_acc and final_margin > best_margin
            ):
                best_state = cpu_state_dict(model)
                best_metrics = row
                best_step = step
            if args.save_eval_checkpoints:
                atomic_torch_save(
                    {
                        "config": cfg.to_dict(),
                        "model_spec": asdict(spec),
                        "seed": seed,
                        "step": step,
                        "state_dict": cpu_state_dict(model),
                        "metrics": row,
                    },
                    run_dir / "checkpoints" / f"step_{step:06d}.pt",
                )
            if step % args.print_every == 0 or step in {1, args.steps}:
                print(
                    f"{run_name} step={step:6d} loss={loss.item():.5f} "
                    f"acc={final_acc:.4f} margin={final_margin:.4f}",
                    flush=True,
                )

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a checkpoint")
    final_state = cpu_state_dict(model)
    final_checkpoint = run_dir / "final.pt"
    best_checkpoint = run_dir / "best.pt"
    checkpoint_common = {
        "config": cfg.to_dict(),
        "model_spec": asdict(spec),
        "seed": seed,
        "init_seed": init_seed,
        "data_seed": data_seed,
        "eval_seed": eval_seed,
        "history": history,
    }
    atomic_torch_save(
        {
            **checkpoint_common,
            "step": args.steps,
            "state_dict": final_state,
            "metrics": history[-1],
            "selection": "final",
        },
        final_checkpoint,
    )
    atomic_torch_save(
        {
            **checkpoint_common,
            "step": best_step,
            "state_dict": best_state,
            "metrics": best_metrics,
            "selection": "best_id_accuracy_then_margin",
        },
        best_checkpoint,
    )

    model.load_state_dict(best_state)
    final_evaluation: dict[str, Any] = {}
    for mode_index, mode in enumerate(EVAL_MODES):
        final_evaluation[mode] = evaluate_model(
            model,
            device=device,
            batch_size=args.eval_batch_size,
            batches=args.final_eval_batches,
            seed=eval_seed + 1000 + mode_index,
            order_mode=mode,
            active_depth=cfg.total_depth,
            amp_enabled=args.amp,
        )
    overloop = None
    if cfg.architecture == "periodic" and args.overloop_depth > cfg.total_depth:
        overloop = evaluate_model(
            model,
            device=device,
            batch_size=args.eval_batch_size,
            batches=args.final_eval_batches,
            seed=eval_seed + 2000,
            order_mode="topological",
            active_depth=args.overloop_depth,
            amp_enabled=args.amp,
        )
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "run_name": run_name,
        "model": spec.name,
        "description": spec.description,
        "seed": seed,
        "config": cfg.to_dict(),
        "parameter_count": count_parameters(model),
        "effective_depth": cfg.total_depth,
        "unique_parameter_blocks": cfg.unique_block_count,
        "parameter_schedule": [
            cfg.parameter_index(depth)
            for depth in range(cfg.total_depth)
        ],
        "best_step": best_step,
        "best_metrics": best_metrics,
        "evaluation": final_evaluation,
        "overloop": overloop,
        "checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "history_path": str(run_dir / "history.json"),
        "runtime": {
            "device": str(device),
            "physical_gpu": args.physical_gpu,
            "shared_gpu": args.shared_gpu,
            "expected_peak_mib": args.expected_peak_mib,
            "observed_peak_mib": peak_mib,
            "reserve_mib": args.reserve_mib,
            "elapsed_sec": time.time() - start_time,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_history(run_dir / "history.json", history)
    return summary


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    evaluation = summary["evaluation"]
    row = {
        "run_name": summary["run_name"],
        "model": summary["model"],
        "seed": summary["seed"],
        "parameter_count": summary["parameter_count"],
        "effective_depth": summary["effective_depth"],
        "unique_parameter_blocks": summary["unique_parameter_blocks"],
        "best_step": summary["best_step"],
        "id_accuracy": evaluation["topological"]["depth_accuracy"][-1],
        "id_margin": evaluation["topological"]["depth_margin"][-1],
        "adjacent_accuracy": evaluation["adjacent"]["depth_accuracy"][-1],
        "grouped_accuracy": evaluation["grouped"]["depth_accuracy"][-1],
        "reversed_accuracy": evaluation["reversed"]["depth_accuracy"][-1],
        "prefix_accuracy": json.dumps(
            evaluation["topological"]["depth_accuracy"]
        ),
        "checkpoint": summary["checkpoint"],
        "observed_peak_mib": summary["runtime"]["observed_peak_mib"],
        "elapsed_sec": summary["runtime"]["elapsed_sec"],
    }
    if summary["overloop"] is not None:
        row["overloop_accuracy"] = json.dumps(
            summary["overloop"]["depth_accuracy"]
        )
    else:
        row["overloop_accuracy"] = ""
    return row


def write_suite_summary(
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_summary(summary) for summary in summaries]
    with (args.out_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    aggregate: dict[str, Any] = {}
    numeric_fields = (
        "id_accuracy",
        "id_margin",
        "adjacent_accuracy",
        "grouped_accuracy",
        "reversed_accuracy",
        "best_step",
        "observed_peak_mib",
        "elapsed_sec",
    )
    for model_name, model_rows in by_model.items():
        aggregate[model_name] = {
            "n_seeds": len(model_rows),
            **{
                f"{field}_{suffix}": function(
                    [float(row[field]) for row in model_rows]
                )
                for field in numeric_fields
                for suffix, function in (
                    ("mean", mean),
                    (
                        "std",
                        lambda values: pstdev(values)
                        if len(values) > 1
                        else 0.0,
                    ),
                )
            },
        }
    payload = {
        "question": (
            "How does a periodic weight-sharing constraint change the circuit "
            "for two-hop in-context retrieval at fixed effective depth?"
        ),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runs": rows,
        "aggregate_by_model": aggregate,
        "behavior_gate": {
            "required_id_accuracy": 0.98,
            "eligible_runs": [
                row["run_name"]
                for row in rows
                if row["id_accuracy"] >= 0.98
            ],
            "ineligible_runs": [
                row["run_name"]
                for row in rows
                if row["id_accuracy"] < 0.98
            ],
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Two-hop standard vs periodic Transformer",
        "",
        "All models use the same six effective depths, online training examples, "
        "optimizer, final-only objective, and paired common-module initialization.",
        "",
        "| model | seeds | params | ID acc | adjacent | grouped | reversed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in args.models:
        model_rows = by_model.get(model_name, [])
        if not model_rows:
            continue
        lines.append(
            f"| {model_name} | {len(model_rows)} | "
            f"{int(model_rows[0]['parameter_count']):,} | "
            f"{mean(row['id_accuracy'] for row in model_rows):.4f} | "
            f"{mean(row['adjacent_accuracy'] for row in model_rows):.4f} | "
            f"{mean(row['grouped_accuracy'] for row in model_rows):.4f} | "
            f"{mean(row['reversed_accuracy'] for row in model_rows):.4f} |"
        )
    lines.extend(
        [
            "",
            "Only runs with ID accuracy at least 0.98 are eligible for circuit comparison.",
            "Attention patterns are localization evidence only; causal results are produced by "
            "`reasoning_loop.twohop_circuit`.",
            "",
        ]
    )
    (args.out_dir / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.models:
        raise ValueError("at least one model is required")
    device = pick_device(args.device)
    if device.type == "cuda":
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise ValueError("cuda-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_specs": {
            name: asdict(MODEL_SPECS[name])
            for name in args.models
        },
        "device": str(device),
        "fairness_contract": (
            "same effective depth, data stream, optimizer, objective, and "
            "paired common-module initialization"
        ),
    }
    (args.out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    summaries: list[dict[str, Any]] = []
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        for seed in args.seeds:
            print(f"=== {model_name} seed={seed} on {device} ===", flush=True)
            summaries.append(train_one(args, spec, seed, device))
    write_suite_summary(args, summaries)
    print(f"wrote outputs under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
