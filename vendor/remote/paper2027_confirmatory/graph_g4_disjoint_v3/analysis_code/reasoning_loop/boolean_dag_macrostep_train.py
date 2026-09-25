from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from reasoning_loop.boolean_dag_data import UNKNOWN, BooleanDAGConfig, make_boolean_dag_batch
from reasoning_loop.boolean_dag_macrostep import (
    DepthConditionedBooleanDAGTransformer,
    macrostep_intermediate_loss,
    macrostep_targets,
    sample_phase_balanced_macrostep_programs,
    sample_macrostep_programs,
)
from reasoning_loop.boolean_dag_model import BooleanDAGModelConfig, count_parameters
from reasoning_loop.graph_path_loop import cosine_lr, pick_device, set_seed


TASK_VERSION = "boolean_dag_macrostep_v1"
DATA_STREAM_SEED_OFFSET = 60_700_000


@dataclass(frozen=True)
class MacroStepTrainConfig:
    condition: Literal["conditioned", "no_instruction"] = "conditioned"
    program_sampling: Literal["random", "phase_balanced"] = "random"
    steps: int = 20_000
    batch_size: int = 512
    eval_batch_size: int = 1020
    eval_batches: int = 8
    eval_every: int = 1000
    print_every: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    out_dir: Path = Path("results/boolean_dag_macrostep")
    run_name: str | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if self.condition not in {"conditioned", "no_instruction"}:
            raise ValueError("condition must be conditioned or no_instruction")
        if self.program_sampling not in {"random", "phase_balanced"}:
            raise ValueError("program_sampling must be random or phase_balanced")
        if self.steps < 1 or self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("steps must be positive and batch_size must be positive even")


def _root_indices(batch) -> torch.Tensor:
    return batch.root_mask.long().argmax(dim=1)


def _first_batch_manifest(program, batch, *, seed: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensors = {
        "program.increments": program.increments,
        "program.active_mask": program.active_mask,
        "program.cumulative_depths": program.cumulative_depths,
        "program.lengths": program.lengths,
        "program.total_depths": program.total_depths,
        **{f"batch.{name}": value for name, value in batch.__dict__.items()},
    }
    for name, value in sorted(tensors.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    preview = []
    for row in range(min(16, len(program.lengths))):
        length = int(program.lengths[row])
        preview.append(program.increments[row, :length].detach().cpu().tolist())
    return {
        "data_stream_seed": seed,
        "first_batch_sha256": digest.hexdigest(),
        "first_programs": preview,
        "heldout_exclusion": "exclude any active program beginning with a heldout program",
        "graph_depth_policy": "every graph has exact depth eval_max_depth",
    }


@torch.no_grad()
def evaluate_macrostep_train_distribution(
    model: DepthConditionedBooleanDAGTransformer,
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    device: torch.device,
    batch_size: int,
    batches: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(20_260_721)
    totals = {
        "state_correct": 0.0,
        "state_count": 0,
        "exact_correct": 0.0,
        "exact_count": 0,
        "frontier_correct": 0.0,
        "frontier_count": 0,
        "root_resolved_correct": 0.0,
        "root_resolved_count": 0,
        "root_unknown_correct": 0.0,
        "root_unknown_count": 0,
    }
    increment_correct = torch.zeros(4, device=device)
    increment_count = torch.zeros(4, device=device)
    for _ in range(batches):
        program = sample_macrostep_programs(
            batch_size=batch_size,
            max_loops=model_cfg.steps,
            device=device,
            generator=generator,
            exclude_holdout=True,
        )
        batch = make_boolean_dag_batch(
            data_cfg,
            batch_size,
            device,
            depths=torch.full(
                (batch_size,),
                data_cfg.eval_max_depth,
                device=device,
            ),
            generator=generator,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model.forward_all(
                batch,
                depth_increments=program.increments,
            )["logits_by_loop"]
        predictions = logits.argmax(dim=-1)
        targets = macrostep_targets(batch, program.cumulative_depths)
        active_nodes = program.active_mask[:, :, None].expand_as(targets)
        totals["state_correct"] += float(predictions.eq(targets)[active_nodes].sum())
        totals["state_count"] += int(active_nodes.sum())
        exact = predictions.eq(targets).all(dim=-1) & program.active_mask
        totals["exact_correct"] += float(exact.sum())
        totals["exact_count"] += int(program.active_mask.sum())
        previous = torch.cat(
            (
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                program.cumulative_depths[:, :-1],
            ),
            dim=1,
        )
        frontier = (
            program.active_mask[:, :, None]
            & batch.levels[:, None, :].gt(previous[:, :, None])
            & batch.levels[:, None, :].le(program.cumulative_depths[:, :, None])
        )
        totals["frontier_correct"] += float(predictions.eq(targets)[frontier].sum())
        totals["frontier_count"] += int(frontier.sum())
        rows = torch.arange(batch_size, device=device)
        final_loop = program.lengths - 1
        root_indices = _root_indices(batch)
        final_root = logits[rows, final_loop, root_indices].argmax(dim=-1)
        final_root_target = targets[rows, final_loop, root_indices]
        resolved_root = final_root_target.ne(UNKNOWN)
        unknown_root = ~resolved_root
        totals["root_resolved_correct"] += float(
            final_root.eq(final_root_target)[resolved_root].sum()
        )
        totals["root_resolved_count"] += int(resolved_root.sum())
        totals["root_unknown_correct"] += float(
            final_root.eq(final_root_target)[unknown_root].sum()
        )
        totals["root_unknown_count"] += int(unknown_root.sum())
        state_correct = predictions.eq(targets).sum(dim=-1)
        for increment in range(1, 5):
            mask = program.active_mask & program.increments.eq(increment)
            increment_correct[increment - 1] += state_correct[mask].sum()
            increment_count[increment - 1] += mask.sum() * data_cfg.node_count
    safe = lambda correct, count: correct / count if count else float("nan")
    return {
        "state_accuracy": safe(totals["state_correct"], totals["state_count"]),
        "exact_state_accuracy": safe(totals["exact_correct"], totals["exact_count"]),
        "frontier_accuracy": safe(totals["frontier_correct"], totals["frontier_count"]),
        "final_root_resolved_accuracy": safe(
            totals["root_resolved_correct"], totals["root_resolved_count"]
        ),
        "final_root_unknown_accuracy": safe(
            totals["root_unknown_correct"], totals["root_unknown_count"]
        ),
        "state_accuracy_by_increment": (
            increment_correct / increment_count.clamp_min(1)
        ).cpu().tolist(),
    }


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _plot_training(path: Path, history: list[dict[str, Any]]) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")
    axes[0].plot(steps, [row["train_loss"] for row in history])
    axes[0].set(title="Training loss", xlabel="optimizer step", ylabel="loss")
    axes[1].plot(steps, [row["eval_state_accuracy"] for row in history])
    axes[1].set(
        title="Cumulative-state accuracy",
        xlabel="optimizer step",
        ylabel="accuracy",
        ylim=(0, 1.02),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def train_macrostep(
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    train_cfg: MacroStepTrainConfig,
) -> Path:
    if 4 * model_cfg.steps > data_cfg.eval_max_depth:
        raise ValueError("data config cannot realize the maximum sampled cumulative depth")
    run_name = train_cfg.run_name or f"{train_cfg.condition}_seed{train_cfg.seed}"
    run_dir = train_cfg.out_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not train_cfg.force:
        raise FileExistsError(f"{run_dir} exists; pass force=True to overwrite artifacts")
    run_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_dir).free < 256 * 1024**2:
        raise RuntimeError(f"insufficient free space for checkpointing in {run_dir}")
    set_seed(train_cfg.seed)
    device = pick_device(train_cfg.device)
    use_instruction = train_cfg.condition == "conditioned"
    model = DepthConditionedBooleanDAGTransformer(
        data_cfg,
        model_cfg,
        use_instruction=use_instruction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )
    history: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}
    data_manifest: dict[str, Any] | None = None
    data_stream_seed = DATA_STREAM_SEED_OFFSET + train_cfg.seed
    data_generator = torch.Generator(device=device).manual_seed(data_stream_seed)
    sample_programs = (
        sample_phase_balanced_macrostep_programs
        if train_cfg.program_sampling == "phase_balanced"
        else sample_macrostep_programs
    )
    start_time = time.time()
    for step in range(1, train_cfg.steps + 1):
        model.train()
        lr = cosine_lr(
            step - 1,
            base_lr=train_cfg.lr,
            total_steps=train_cfg.steps,
            warmup_steps=train_cfg.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        program = sample_programs(
            batch_size=train_cfg.batch_size,
            max_loops=model_cfg.steps,
            device=device,
            generator=data_generator,
            exclude_holdout=True,
        )
        batch = make_boolean_dag_batch(
            data_cfg,
            train_cfg.batch_size,
            device,
            depths=torch.full(
                (train_cfg.batch_size,),
                data_cfg.eval_max_depth,
                device=device,
            ),
            generator=data_generator,
        )
        if data_manifest is None:
            data_manifest = _first_batch_manifest(
                program,
                batch,
                seed=data_stream_seed,
            )
            (run_dir / "data_manifest.json").write_text(
                json.dumps(data_manifest, indent=2),
                encoding="utf-8",
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=train_cfg.amp and device.type == "cuda",
        ):
            logits = model.forward_all(
                batch,
                depth_increments=program.increments,
            )["logits_by_loop"]
            loss, pieces = macrostep_intermediate_loss(logits, batch, program)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        if step == 1 or step % train_cfg.eval_every == 0 or step == train_cfg.steps:
            last_metrics = evaluate_macrostep_train_distribution(
                model,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                device=device,
                batch_size=train_cfg.eval_batch_size,
                batches=train_cfg.eval_batches,
                amp=train_cfg.amp,
            )
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.detach().cpu()),
                "train_frontier_ce": float(pieces["frontier_ce"].detach().cpu()),
                "train_persistence_ce": float(pieces["persistence_ce"].detach().cpu()),
                "train_unknown_ce": float(pieces["unknown_ce"].detach().cpu()),
                "eval_state_accuracy": last_metrics["state_accuracy"],
                "eval_exact_state_accuracy": last_metrics["exact_state_accuracy"],
                "eval_frontier_accuracy": last_metrics["frontier_accuracy"],
                "eval_final_root_resolved_accuracy": last_metrics[
                    "final_root_resolved_accuracy"
                ],
                "eval_final_root_unknown_accuracy": last_metrics[
                    "final_root_unknown_accuracy"
                ],
                "eval_state_accuracy_by_increment": json.dumps(
                    last_metrics["state_accuracy_by_increment"]
                ),
                "elapsed_sec": time.time() - start_time,
            }
            history.append(row)
            _write_history(run_dir / "history.csv", history)
            if step == 1 or step % train_cfg.print_every == 0 or step == train_cfg.steps:
                print(
                    f"[{train_cfg.condition} step={step:05d}] "
                    f"loss={row['train_loss']:.4f} "
                    f"state={row['eval_state_accuracy']:.3f} "
                    f"frontier={row['eval_frontier_accuracy']:.3f}",
                    flush=True,
                )
    checkpoint = {
        "task_version": TASK_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "train_config": {**asdict(train_cfg), "out_dir": str(train_cfg.out_dir)},
        "condition": train_cfg.condition,
        "use_instruction": use_instruction,
        "step": train_cfg.steps,
        "seed": train_cfg.seed,
        "parameter_count": count_parameters(model),
        "metrics": last_metrics,
        "data_manifest": data_manifest,
    }
    checkpoint_path = run_dir / "final.pt"
    temporary_checkpoint = run_dir / "final.pt.tmp"
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint_path)
    summary = {
        "task_version": TASK_VERSION,
        "condition": train_cfg.condition,
        "use_instruction": use_instruction,
        "seed": train_cfg.seed,
        "steps": train_cfg.steps,
        "parameter_count": checkpoint["parameter_count"],
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "final_metrics": last_metrics,
        "data_manifest": data_manifest,
        "elapsed_sec": time.time() - start_time,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot_training(run_dir / "training_curves.png", history)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train depth-conditioned Boolean-DAG macro-steps.")
    parser.add_argument(
        "--condition",
        choices=["conditioned", "no_instruction"],
        required=True,
    )
    parser.add_argument(
        "--program-sampling",
        choices=["random", "phase_balanced"],
        default="random",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1020)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/boolean_dag_macrostep_seed0_20260710"),
    )
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_macrostep(
        data_cfg=BooleanDAGConfig(
            node_count=40,
            leaf_count=6,
            train_max_depth=16,
            eval_max_depth=16,
        ),
        model_cfg=BooleanDAGModelConfig(
            d_model=256,
            n_heads=8,
            d_mlp=1024,
            steps=4,
        ),
        train_cfg=MacroStepTrainConfig(
            condition=args.condition,
            program_sampling=args.program_sampling,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_batches=args.eval_batches,
            eval_every=args.eval_every,
            print_every=args.print_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            seed=args.seed,
            device=args.device,
            amp=args.amp,
            out_dir=args.out_dir,
            run_name=args.run_name,
            force=args.force,
        ),
    )


if __name__ == "__main__":
    main()
