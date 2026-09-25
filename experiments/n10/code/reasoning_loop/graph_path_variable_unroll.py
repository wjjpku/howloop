from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_loop import (
    LoopedGraphPathTransformer,
    cosine_lr,
    count_parameters,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_stepwise import (
    StepwiseGraphPathConfig,
    make_stepwise_batch,
)


def validate_depths(
    depths: Sequence[int],
    *,
    max_train_depth: int,
) -> tuple[int, ...]:
    if max_train_depth < 1:
        raise ValueError("max_train_depth must be positive")
    normalized = tuple(int(depth) for depth in depths)
    if not normalized:
        raise ValueError("depths must contain at least one horizon")
    if any(depth < 1 for depth in normalized):
        raise ValueError("depths must be positive")
    if len(set(normalized)) != len(normalized):
        raise ValueError("depths must be unique")
    if max(normalized) > max_train_depth:
        raise ValueError("depths must not exceed max_train_depth")
    return tuple(sorted(normalized))


def _validate_optional_depths(
    depths: Sequence[int],
    *,
    max_depth: int,
) -> tuple[int, ...]:
    if not depths:
        return ()
    return validate_depths(depths, max_train_depth=max_depth)


def validate_depth_weights(
    depths: Sequence[int],
    weights: Sequence[float],
) -> tuple[float, ...]:
    if len(depths) != len(weights):
        raise ValueError("depths and weights must have the same length")
    normalized = tuple(float(weight) for weight in weights)
    if any(weight < 0 for weight in normalized):
        raise ValueError("depth weights must be nonnegative")
    total = sum(normalized)
    if total <= 0:
        raise ValueError("depth weights must have positive total mass")
    return tuple(weight / total for weight in normalized)


def balanced_depth_assignment(
    depths: Sequence[int],
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    normalized = tuple(int(depth) for depth in depths)
    if not normalized:
        raise ValueError("depths must contain at least one horizon")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    repeats = (batch_size + len(normalized) - 1) // len(normalized)
    base = torch.tensor(normalized, dtype=torch.long, device=device).repeat(repeats)
    permutation = torch.randperm(
        base.numel(),
        device=device,
        generator=generator,
    )
    return base[permutation[:batch_size]]


def weighted_depth_assignment(
    depths: Sequence[int],
    weights: Sequence[float],
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    normalized_depths = tuple(int(depth) for depth in depths)
    if not normalized_depths:
        raise ValueError("depths must contain at least one horizon")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    probabilities = validate_depth_weights(normalized_depths, weights)
    depth_tensor = torch.tensor(
        normalized_depths,
        dtype=torch.long,
        device=device,
    )
    probability_tensor = torch.tensor(
        probabilities,
        dtype=torch.float32,
        device=device,
    )
    indices = torch.multinomial(
        probability_tensor,
        batch_size,
        replacement=True,
        generator=generator,
    )
    return depth_tensor[indices]


def select_horizon_logits(
    logits_by_loop: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    if logits_by_loop.ndim != 3:
        raise ValueError("logits_by_loop must have shape [batch, loops, classes]")
    if depths.ndim != 1 or depths.shape[0] != logits_by_loop.shape[0]:
        raise ValueError("depths must have shape [batch]")
    if depths.numel() and (int(depths.min()) < 1 or int(depths.max()) > logits_by_loop.shape[1]):
        raise ValueError("depths must index an available loop")
    batch_index = torch.arange(logits_by_loop.shape[0], device=logits_by_loop.device)
    return logits_by_loop[batch_index, depths.to(logits_by_loop.device) - 1]


def variable_horizon_loss(
    logits_by_loop: torch.Tensor,
    targets_by_pos: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    if targets_by_pos.ndim != 2 or targets_by_pos.shape[0] != logits_by_loop.shape[0]:
        raise ValueError("targets_by_pos must have shape [batch, positions]")
    if depths.numel() and int(depths.max()) > targets_by_pos.shape[1]:
        raise ValueError("targets_by_pos do not cover every requested depth")
    selected_logits = select_horizon_logits(logits_by_loop, depths)
    batch_index = torch.arange(targets_by_pos.shape[0], device=targets_by_pos.device)
    selected_targets = targets_by_pos[
        batch_index,
        depths.to(targets_by_pos.device) - 1,
    ]
    return F.cross_entropy(selected_logits, selected_targets)


def analytic_start_shortcut(node_count: int, depth: int) -> float:
    if node_count < 1 or depth < 1:
        raise ValueError("node_count and depth must be positive")
    divisor_count = sum(depth % cycle_length == 0 for cycle_length in range(1, node_count + 1))
    return divisor_count / node_count


def _depth_subset_summary(
    native_accuracy: torch.Tensor,
    depths: Sequence[int],
) -> tuple[float | None, float | None]:
    if not depths:
        return None, None
    values = native_accuracy[
        torch.tensor(depths, device=native_accuracy.device, dtype=torch.long) - 1
    ]
    return float(values.mean().cpu()), float(values.min().cpu())


def _format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


@torch.no_grad()
def evaluate_variable_horizon(
    model: torch.nn.Module,
    cfg: StepwiseGraphPathConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    max_eval_depth: int,
    train_depths: Sequence[int],
    interpolation_depths: Sequence[int],
    extrapolation_depths: Sequence[int],
    amp_enabled: bool,
) -> dict[str, Any]:
    if batch_size < 1 or batches < 1 or max_eval_depth < 1:
        raise ValueError("evaluation counts and max_eval_depth must be positive")
    partitions = {
        "train_depths": validate_depths(train_depths, max_train_depth=max_eval_depth),
        "interpolation_depths": _validate_optional_depths(
            interpolation_depths,
            max_depth=max_eval_depth,
        ),
        "extrapolation_depths": _validate_optional_depths(
            extrapolation_depths,
            max_depth=max_eval_depth,
        ),
    }
    flattened = [depth for depths in partitions.values() for depth in depths]
    if len(set(flattened)) != len(flattened):
        raise ValueError("depth partitions must be disjoint")

    model.eval()
    correct = torch.zeros(max_eval_depth, max_eval_depth, device=device)
    probability = torch.zeros_like(correct)
    loss_sum = torch.zeros_like(correct)
    count = 0
    autocast_device = "cuda" if device.type == "cuda" else device.type
    for _ in range(batches):
        tokens, targets_by_pos, _, _ = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=max_eval_depth,
        )
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(  # type: ignore[attr-defined]
                tokens,
                max_loops=max_eval_depth,
            )["logits_by_loop"]
        probs_by_loop = logits_by_loop.softmax(dim=-1)
        predictions = logits_by_loop.argmax(dim=-1)
        for loop_idx in range(max_eval_depth):
            logits = logits_by_loop[:, loop_idx, :]
            probs = probs_by_loop[:, loop_idx, :]
            for pos_idx in range(max_eval_depth):
                target = targets_by_pos[:, pos_idx]
                correct[loop_idx, pos_idx] += (
                    predictions[:, loop_idx].eq(target).float().sum()
                )
                probability[loop_idx, pos_idx] += (
                    probs.gather(1, target[:, None]).squeeze(1).sum()
                )
                loss_sum[loop_idx, pos_idx] += F.cross_entropy(
                    logits,
                    target,
                    reduction="sum",
                )
        count += batch_size

    accuracy = correct / count
    mean_probability = probability / count
    mean_loss = loss_sum / count
    native_accuracy = accuracy.diagonal()
    best_accuracy, best_position = accuracy.max(dim=1)
    train_mean, train_min = _depth_subset_summary(
        native_accuracy,
        partitions["train_depths"],
    )
    interpolation_mean, interpolation_min = _depth_subset_summary(
        native_accuracy,
        partitions["interpolation_depths"],
    )
    extrapolation_mean, extrapolation_min = _depth_subset_summary(
        native_accuracy,
        partitions["extrapolation_depths"],
    )

    return {
        "examples": count,
        "max_eval_depth": max_eval_depth,
        "acc_to_pos": accuracy.cpu().tolist(),
        "mean_prob_to_pos": mean_probability.cpu().tolist(),
        "mean_loss_to_pos": mean_loss.cpu().tolist(),
        "native_accuracy_by_depth": native_accuracy.cpu().tolist(),
        "best_position_by_loop": [int(value) + 1 for value in best_position.cpu().tolist()],
        "best_accuracy_by_loop": best_accuracy.cpu().tolist(),
        "train_depths": list(partitions["train_depths"]),
        "interpolation_depths": list(partitions["interpolation_depths"]),
        "extrapolation_depths": list(partitions["extrapolation_depths"]),
        "analytic_start_shortcut_by_depth": [
            analytic_start_shortcut(cfg.node_count, depth)
            for depth in range(1, max_eval_depth + 1)
        ],
        "trained_native_mean_accuracy": train_mean,
        "trained_native_min_accuracy": train_min,
        "interpolation_native_mean_accuracy": interpolation_mean,
        "interpolation_native_min_accuracy": interpolation_min,
        "extrapolation_native_mean_accuracy": extrapolation_mean,
        "extrapolation_native_min_accuracy": extrapolation_min,
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _save_plots(run_dir: Path, metrics: dict[str, Any]) -> None:
    native = np.asarray(metrics["native_accuracy_by_depth"], dtype=np.float32)
    shortcut = np.asarray(
        metrics["analytic_start_shortcut_by_depth"],
        dtype=np.float32,
    )
    depth = np.arange(1, len(native) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depth, native, marker="o", label="native loop t -> f^t")
    ax.plot(depth, shortcut, marker="x", linestyle=":", label="start-node shortcut")
    ax.set_xlabel("depth / recurrent applications")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(depth)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "native_depth_accuracy.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(metrics["acc_to_pos"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(matrix.shape[1]), [str(index) for index in depth])
    ax.set_yticks(range(matrix.shape[0]), [str(index) for index in depth])
    ax.set_xlabel("actual path position k")
    ax.set_ylabel("readout loop t")
    fig.colorbar(image, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(run_dir / "loop_by_path_accuracy.png", dpi=180)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a no-depth looped graph-path model with one final-only "
            "endpoint label selected at a balanced per-example horizon."
        )
    )
    parser.add_argument("--node-count", type=int, default=16)
    parser.add_argument("--max-train-depth", type=int, default=8)
    parser.add_argument("--max-eval-depth", type=int, default=12)
    parser.add_argument("--train-depths", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument(
        "--train-depth-weights",
        type=float,
        nargs="+",
        help="Optional sampling probabilities aligned with --train-depths.",
    )
    parser.add_argument("--interpolation-depths", type=int, nargs="*", default=[3, 5, 7])
    parser.add_argument("--extrapolation-depths", type=int, nargs="*", default=[9, 10, 11, 12])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--block-schedule",
        choices=["all_blocks", "first_block_once"],
        default="all_blocks",
        help=(
            "all_blocks repeats every block at every loop; first_block_once "
            "runs block 0 only on the first loop and repeats blocks 1..N."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-eval-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--arm-name", default="sparse")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/global_depth_supervision_20260717/variable_unroll"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.max_train_depth < 1 or args.max_eval_depth < args.max_train_depth:
        parser.error("max_eval_depth must be at least max_train_depth >= 1")
    if args.steps < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("training and batch counts must be positive")
    if args.eval_batches < 1 or args.eval_every < 1 or args.print_every < 1:
        parser.error("evaluation and printing counts must be positive")
    train_depths = validate_depths(
        args.train_depths,
        max_train_depth=args.max_train_depth,
    )
    interpolation_depths = _validate_optional_depths(
        args.interpolation_depths,
        max_depth=args.max_eval_depth,
    )
    extrapolation_depths = _validate_optional_depths(
        args.extrapolation_depths,
        max_depth=args.max_eval_depth,
    )
    flattened = train_depths + interpolation_depths + extrapolation_depths
    if len(set(flattened)) != len(flattened):
        parser.error("train, interpolation, and extrapolation depths must be disjoint")
    args.train_depths = train_depths
    args.train_depth_weights = (
        validate_depth_weights(train_depths, args.train_depth_weights)
        if args.train_depth_weights is not None
        else None
    )
    args.interpolation_depths = interpolation_depths
    args.extrapolation_depths = extrapolation_depths
    return args


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    cfg: StepwiseGraphPathConfig,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, Any],
    parameter_count: int,
    train_depth_histogram: dict[str, int],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "metrics": metrics,
        "parameter_count": parameter_count,
        "task": "graph_path_variable_unroll",
        "objective": "one_endpoint_label_per_example",
        "loss_terms_per_example": 1,
        "train_depths": list(args.train_depths),
        "train_depth_weights": (
            list(args.train_depth_weights)
            if args.train_depth_weights is not None
            else None
        ),
        "interpolation_depths": list(args.interpolation_depths),
        "extrapolation_depths": list(args.extrapolation_depths),
        "max_train_depth": args.max_train_depth,
        "max_eval_depth": args.max_eval_depth,
        "seed": args.seed,
        "arm_name": args.arm_name,
        "init_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "train_depth_histogram": train_depth_histogram,
    }


def train_one(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = StepwiseGraphPathConfig(
        node_count=args.node_count,
        max_depth=args.max_eval_depth,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        max_loops=args.max_train_depth,
        dropout=args.dropout,
        block_schedule=args.block_schedule,
    )
    depth_name = "-".join(str(depth) for depth in args.train_depths)
    run_name = args.run_name or (
        f"variable_{args.arm_name}_N{args.node_count}_D{args.max_train_depth}_"
        f"d{args.d_model}_B{args.n_layers}_H{depth_name}_seed{args.seed}"
    )
    run_dir = args.out_dir / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    model = LoopedGraphPathTransformer(cfg).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device)
        initial_cfg = StepwiseGraphPathConfig(**initial["config"])
        if asdict(initial_cfg) != asdict(cfg):
            raise ValueError(
                f"checkpoint config {initial_cfg} does not match requested config {cfg}"
            )
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    parameter_count = count_parameters(model)
    autocast_device = "cuda" if device.type == "cuda" else device.type
    history: list[dict[str, Any]] = []
    train_depth_histogram = {str(depth): 0 for depth in args.train_depths}
    best_score = -1.0
    best_step = 0
    start_time = time.time()
    metadata = {
        "run_name": run_name,
        "task": "graph_path_variable_unroll",
        "objective": "one_endpoint_label_per_example",
        "loss_terms_per_example": 1,
        "config": asdict(cfg),
        "args": vars(args),
        "parameter_count": parameter_count,
        "device": str(device),
        "init_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "run_dir": str(run_dir),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

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
        tokens, targets_by_pos, _, _ = make_stepwise_batch(
            cfg,
            args.batch_size,
            device,
            path_positions=args.max_train_depth,
        )
        sampled_depths = (
            balanced_depth_assignment(
                args.train_depths,
                args.batch_size,
                device,
            )
            if args.train_depth_weights is None
            else weighted_depth_assignment(
                args.train_depths,
                args.train_depth_weights,
                args.batch_size,
                device,
            )
        )
        for depth in args.train_depths:
            train_depth_histogram[str(depth)] += int(sampled_depths.eq(depth).sum())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(
                tokens,
                max_loops=args.max_train_depth,
            )["logits_by_loop"]
            loss = variable_horizon_loss(
                logits_by_loop,
                targets_by_pos,
                sampled_depths,
            )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_variable_horizon(
                model,
                cfg,
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                max_eval_depth=args.max_eval_depth,
                train_depths=args.train_depths,
                interpolation_depths=args.interpolation_depths,
                extrapolation_depths=args.extrapolation_depths,
                amp_enabled=args.amp,
            )
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.detach().cpu()),
                "elapsed_sec": time.time() - start_time,
                **metrics,
            }
            history.append(row)
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            if args.save_checkpoints and args.save_eval_checkpoints:
                atomic_torch_save(
                    _checkpoint_payload(
                        model=model,
                        cfg=cfg,
                        args=args,
                        step=step,
                        metrics=metrics,
                        parameter_count=parameter_count,
                        train_depth_histogram=dict(train_depth_histogram),
                    ),
                    run_dir / f"checkpoint_step_{step:05d}.pt",
                )
            score = metrics["trained_native_mean_accuracy"]
            if score > best_score:
                best_score = score
                best_step = step
                if args.save_checkpoints:
                    atomic_torch_save(
                        _checkpoint_payload(
                            model=model,
                            cfg=cfg,
                            args=args,
                            step=step,
                            metrics=metrics,
                            parameter_count=parameter_count,
                            train_depth_histogram=dict(train_depth_histogram),
                        ),
                        run_dir / "best.pt",
                    )
            if step == 1 or step % args.print_every == 0 or step == args.steps:
                native = " ".join(
                    f"D{depth}:{metrics['native_accuracy_by_depth'][depth - 1]:.3f}"
                    for depth in range(1, args.max_eval_depth + 1)
                )
                print(
                    f"[{args.arm_name} step={step:05d}] "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"train_mean={metrics['trained_native_mean_accuracy']:.3f} "
                    f"interp_mean={_format_optional_metric(metrics['interpolation_native_mean_accuracy'])} "
                    f"extra_mean={_format_optional_metric(metrics['extrapolation_native_mean_accuracy'])} "
                    f"{native}",
                    flush=True,
                )

    final_metrics = evaluate_variable_horizon(
        model,
        cfg,
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 4),
        max_eval_depth=args.max_eval_depth,
        train_depths=args.train_depths,
        interpolation_depths=args.interpolation_depths,
        extrapolation_depths=args.extrapolation_depths,
        amp_enabled=args.amp,
    )
    summary = {
        "run_name": run_name,
        "task": "graph_path_variable_unroll",
        "objective": "one_endpoint_label_per_example",
        "loss_terms_per_example": 1,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "arm_name": args.arm_name,
        "init_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "train_depths": list(args.train_depths),
        "train_depth_weights": (
            list(args.train_depth_weights)
            if args.train_depth_weights is not None
            else None
        ),
        "interpolation_depths": list(args.interpolation_depths),
        "extrapolation_depths": list(args.extrapolation_depths),
        "max_train_depth": args.max_train_depth,
        "max_eval_depth": args.max_eval_depth,
        "best_step": best_step,
        "best_trained_native_mean_accuracy": best_score,
        "train_depth_histogram": train_depth_histogram,
        "final_metrics": final_metrics,
        "run_dir": str(run_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_checkpoints:
        atomic_torch_save(
            _checkpoint_payload(
                model=model,
                cfg=cfg,
                args=args,
                step=args.steps,
                metrics=final_metrics,
                parameter_count=parameter_count,
                train_depth_histogram=train_depth_histogram,
            ),
            run_dir / "final.pt",
        )
    _save_plots(run_dir, final_metrics)
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(
        f"device={device} arm={args.arm_name} train_depths={args.train_depths} "
        f"max_eval_depth={args.max_eval_depth}",
        flush=True,
    )
    train_one(args, device)


if __name__ == "__main__":
    main()
