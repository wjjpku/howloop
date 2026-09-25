from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
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
    count_parameters,
    pick_device,
    set_seed,
)


@dataclass(frozen=True)
class LongCycleConfig:
    node_count: int = 32
    d_model: int = 256
    n_heads: int = 4
    d_mlp: int = 1024
    n_layers: int = 2
    max_loops: int = 6
    dropout: float = 0.0
    outer_norm_groups: int = 0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ValueError("node_count must be at least 2")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_loops < 1:
            raise ValueError("max_loops must be positive")
        if self.outer_norm_groups < 0 or (
            self.outer_norm_groups and self.d_model % self.outer_norm_groups
        ):
            raise ValueError("outer_norm_groups must be zero or divide d_model")

    @property
    def edge_token(self) -> int:
        return self.node_count

    @property
    def query_token(self) -> int:
        return self.node_count + 1

    @property
    def answer_token(self) -> int:
        return self.node_count + 2

    @property
    def bos_token(self) -> int:
        return self.node_count + 3

    @property
    def vocab_size(self) -> int:
        return self.node_count + 4

    @property
    def seq_len(self) -> int:
        return 1 + 3 * self.node_count + 3


@dataclass(frozen=True)
class LongCycleBatch:
    tokens: torch.Tensor
    targets_by_step: torch.Tensor
    successors: torch.Tensor
    start: torch.Tensor


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 256
    eval_batch_size: int = 512
    eval_batches: int = 8
    eval_every: int = 500
    print_every: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_loops: int = 30
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    out_dir: Path = Path("results/long_cycle_group_rmsnorm_20260714")
    force: bool = False


def make_single_cycle_successors(
    *,
    batch_size: int,
    node_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    order = torch.rand(
        batch_size,
        node_count,
        device=device,
        generator=generator,
    ).argsort(dim=-1)
    successors = torch.empty_like(order)
    successors.scatter_(1, order, order.roll(shifts=-1, dims=1))
    return successors


def trace_successors(
    successors: torch.Tensor,
    start: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    current = start
    trajectory = []
    for _ in range(steps):
        current = successors.gather(1, current[:, None]).squeeze(1)
        trajectory.append(current)
    return torch.stack(trajectory, dim=1)


def rolling_horizon(
    rolling_accuracy: Sequence[float],
    *,
    threshold: float = 0.99,
) -> int:
    horizon = 0
    for accuracy in rolling_accuracy:
        if accuracy < threshold:
            break
        horizon += 1
    return horizon


def make_long_cycle_batch(
    cfg: LongCycleConfig,
    *,
    batch_size: int,
    device: torch.device,
    trajectory_steps: int,
    generator: torch.Generator | None = None,
) -> LongCycleBatch:
    if trajectory_steps >= cfg.node_count:
        raise ValueError("trajectory_steps must be smaller than node_count")
    successors = make_single_cycle_successors(
        batch_size=batch_size,
        node_count=cfg.node_count,
        device=device,
        generator=generator,
    )
    source = torch.arange(cfg.node_count, device=device).expand(batch_size, -1)
    edge_order = torch.rand(
        batch_size,
        cfg.node_count,
        device=device,
        generator=generator,
    ).argsort(dim=-1)
    shuffled_source = source.gather(1, edge_order)
    shuffled_destination = successors.gather(1, shuffled_source)

    edge_triplets = torch.empty(
        batch_size,
        cfg.node_count,
        3,
        dtype=torch.long,
        device=device,
    )
    edge_triplets[:, :, 0] = cfg.edge_token
    edge_triplets[:, :, 1] = shuffled_source
    edge_triplets[:, :, 2] = shuffled_destination

    start = torch.randint(
        cfg.node_count,
        (batch_size,),
        device=device,
        generator=generator,
    )
    targets_by_step = trace_successors(successors, start, steps=trajectory_steps)
    tokens = torch.empty(batch_size, cfg.seq_len, dtype=torch.long, device=device)
    tokens[:, 0] = cfg.bos_token
    tokens[:, 1 : 1 + 3 * cfg.node_count] = edge_triplets.reshape(batch_size, -1)
    tokens[:, -3] = cfg.query_token
    tokens[:, -2] = start
    tokens[:, -1] = cfg.answer_token
    return LongCycleBatch(
        tokens=tokens,
        targets_by_step=targets_by_step,
        successors=successors,
        start=start,
    )


def final_only_loss(
    logits_by_loop: torch.Tensor,
    targets_by_step: torch.Tensor,
) -> torch.Tensor:
    trained_loops = logits_by_loop.shape[1]
    if targets_by_step.shape[1] < trained_loops:
        raise ValueError("targets_by_step is shorter than the trained loop count")
    return F.cross_entropy(
        logits_by_loop[:, -1, :],
        targets_by_step[:, trained_loops - 1],
    )


@torch.no_grad()
def evaluate_overloop(
    model: LoopedGraphPathTransformer,
    cfg: LongCycleConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    eval_loops: int,
    amp_enabled: bool,
    generator_seed: int,
) -> dict[str, Any]:
    if eval_loops >= cfg.node_count:
        raise ValueError("eval_loops must be smaller than node_count")
    model.eval()
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    correct_by_position = torch.zeros(eval_loops, eval_loops, device=device)
    probability_by_position = torch.zeros_like(correct_by_position)
    entropy_sum = torch.zeros(eval_loops, device=device)
    margin_sum = torch.zeros(eval_loops, device=device)
    state_norm_sum = torch.zeros(eval_loops, device=device)
    update_norm_sum = torch.zeros(eval_loops, device=device)
    update_state_ratio_sum = torch.zeros(eval_loops, device=device)
    count = 0
    autocast_device = "cuda" if device.type == "cuda" else device.type

    for _ in range(batches):
        batch = make_long_cycle_batch(
            cfg,
            batch_size=batch_size,
            device=device,
            trajectory_steps=eval_loops,
            generator=generator,
        )
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            out = model.forward_all(
                batch.tokens,
                max_loops=eval_loops,
                return_dynamics=True,
            )
        logits = out["logits_by_loop"].float()
        probabilities = logits.softmax(dim=-1)
        predictions = logits.argmax(dim=-1)
        target_index = batch.targets_by_step[:, None, :].expand(-1, eval_loops, -1)
        correct_by_position += predictions[:, :, None].eq(target_index).float().sum(dim=0)
        probability_by_position += probabilities.gather(2, target_index).sum(dim=0)

        entropy_sum += (
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        ).sum(dim=0)
        top2 = logits.topk(2, dim=-1).values
        margin_sum += (top2[:, :, 0] - top2[:, :, 1]).sum(dim=0)

        incoming = out["incoming_answer_states_by_loop"].float()
        recurrent = out["recurrent_answer_states_by_loop"].float()
        update = recurrent - incoming
        state_norm_sum += recurrent.norm(dim=-1).sum(dim=0)
        update_norm = update.norm(dim=-1)
        update_norm_sum += update_norm.sum(dim=0)
        update_state_ratio_sum += (
            update_norm / incoming.norm(dim=-1).clamp_min(1e-8)
        ).sum(dim=0)
        count += batch_size

    path_accuracy = correct_by_position / count
    path_probability = probability_by_position / count
    loop_index = torch.arange(eval_loops, device=device)
    rolling_accuracy_tensor = path_accuracy[loop_index, loop_index]
    endpoint_index = cfg.max_loops - 1
    endpoint_accuracy = path_accuracy[:, endpoint_index]
    best_accuracy, best_position = path_accuracy.max(dim=1)
    rolling_accuracy = [float(value) for value in rolling_accuracy_tensor.cpu()]
    return {
        "rolling_accuracy": rolling_accuracy,
        "rolling_horizon": rolling_horizon(rolling_accuracy),
        "endpoint_accuracy": [float(value) for value in endpoint_accuracy.cpu()],
        "best_position": [int(value) + 1 for value in best_position.cpu()],
        "best_accuracy": [float(value) for value in best_accuracy.cpu()],
        "path_position_accuracy": path_accuracy.cpu().tolist(),
        "path_position_probability": path_probability.cpu().tolist(),
        "entropy": (entropy_sum / count).cpu().tolist(),
        "top1_margin": (margin_sum / count).cpu().tolist(),
        "state_norm": (state_norm_sum / count).cpu().tolist(),
        "update_norm": (update_norm_sum / count).cpu().tolist(),
        "update_state_ratio": (update_state_ratio_sum / count).cpu().tolist(),
        "example_count": count,
        "eval_loops": eval_loops,
        "generator_seed": generator_seed,
    }


def condition_name(outer_norm_groups: int) -> str:
    return f"outer_G{outer_norm_groups}"


def summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    final_eval = summary["final_eval"]
    row = {
        "condition": summary["condition"],
        "seed": summary["seed"],
        "parameter_count": summary["parameter_count"],
        "rolling_horizon": final_eval["rolling_horizon"],
    }
    rolling = final_eval["rolling_accuracy"]
    endpoint = final_eval["endpoint_accuracy"]
    for loop in (6, 8, 10, 20, 30):
        row[f"loop{loop}_rolling_accuracy"] = rolling[loop - 1] if len(rolling) >= loop else None
        row[f"loop{loop}_endpoint_accuracy"] = endpoint[loop - 1] if len(endpoint) >= loop else None
    return row


def _cosine_lr(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _plot_run(run_dir: Path, history: list[dict[str, Any]], final_eval: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["step"] for row in history],
        [row["trained_loop_accuracy"] for row in history],
        marker="o",
        markersize=3,
        label="trained-loop accuracy",
    )
    axis.set(xlabel="train step", ylabel="accuracy", ylim=(0.0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "training_curve.png", dpi=180)
    plt.close(figure)

    matrix = np.asarray(final_eval["path_position_accuracy"], dtype=np.float32)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.axhline(5.5, color="white", linestyle="--", linewidth=1)
    axis.set(
        xlabel="true path position",
        ylabel="readout loop",
        title="Accuracy to every path position",
    )
    axis.set_xticks(np.arange(matrix.shape[1]), np.arange(1, matrix.shape[1] + 1))
    axis.set_yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
    figure.colorbar(image, ax=axis, label="accuracy")
    figure.tight_layout()
    figure.savefig(run_dir / "path_accuracy_heatmap.png", dpi=180)
    plt.close(figure)


def train_condition(
    model_cfg: LongCycleConfig,
    train_cfg: TrainConfig,
) -> Path:
    name = condition_name(model_cfg.outer_norm_groups)
    run_dir = train_cfg.out_dir / f"{name}_seed{train_cfg.seed}"
    final_checkpoint = run_dir / "final.pt"
    if final_checkpoint.exists() and not train_cfg.force:
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(train_cfg.device)
    set_seed(train_cfg.seed)
    model = LoopedGraphPathTransformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )
    parameter_count = count_parameters(model)
    train_generator = torch.Generator(device=device).manual_seed(71_400_000 + train_cfg.seed)
    history: list[dict[str, Any]] = []
    started = time.time()
    autocast_device = "cuda" if device.type == "cuda" else device.type

    metadata = {
        "condition": name,
        "model_config": asdict(model_cfg),
        "train_config": {**asdict(train_cfg), "out_dir": str(train_cfg.out_dir)},
        "parameter_count": parameter_count,
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for step in range(1, train_cfg.steps + 1):
        model.train()
        lr = _cosine_lr(
            step - 1,
            base_lr=train_cfg.lr,
            total_steps=train_cfg.steps,
            warmup_steps=train_cfg.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = make_long_cycle_batch(
            model_cfg,
            batch_size=train_cfg.batch_size,
            device=device,
            trajectory_steps=model_cfg.max_loops,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=train_cfg.amp and device.type == "cuda",
        ):
            logits = model.forward_all(batch.tokens)["logits_by_loop"]
            loss = final_only_loss(logits, batch.targets_by_step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        if train_cfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        else:
            grad_norm = torch.zeros((), device=device)
        optimizer.step()

        should_evaluate = step == 1 or step == train_cfg.steps or step % train_cfg.eval_every == 0
        if should_evaluate:
            metrics = evaluate_overloop(
                model,
                model_cfg,
                device=device,
                batch_size=train_cfg.eval_batch_size,
                batches=train_cfg.eval_batches,
                eval_loops=model_cfg.max_loops,
                amp_enabled=train_cfg.amp,
                generator_seed=91_400_000 + train_cfg.seed,
            )
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.detach().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "trained_loop_accuracy": metrics["rolling_accuracy"][model_cfg.max_loops - 1],
                "trained_endpoint_accuracy": metrics["endpoint_accuracy"][model_cfg.max_loops - 1],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            _write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            if step == 1 or step == train_cfg.steps or step % train_cfg.print_every == 0:
                print(
                    f"[{name} seed={train_cfg.seed} step={step:05d}] "
                    f"loss={row['train_loss']:.4f} acc={row['trained_loop_accuracy']:.4f}",
                    flush=True,
                )

    final_eval = evaluate_overloop(
        model,
        model_cfg,
        device=device,
        batch_size=train_cfg.eval_batch_size,
        batches=max(train_cfg.eval_batches, 2),
        eval_loops=train_cfg.eval_loops,
        amp_enabled=train_cfg.amp,
        generator_seed=191_400_000 + train_cfg.seed,
    )
    summary = {
        "condition": name,
        "seed": train_cfg.seed,
        "parameter_count": parameter_count,
        "model_config": asdict(model_cfg),
        "train_config": {**asdict(train_cfg), "out_dir": str(train_cfg.out_dir)},
        "final_eval": final_eval,
        "elapsed_seconds": time.time() - started,
        "run_dir": str(run_dir),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": {**asdict(train_cfg), "out_dir": str(train_cfg.out_dir)},
            "step": train_cfg.steps,
            "metrics": final_eval,
            "parameter_count": parameter_count,
        },
        final_checkpoint,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot_run(run_dir, history, final_eval)
    return run_dir


def write_comparison(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(summary) for summary in summaries]
    if rows:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"runs": summaries, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    for summary in summaries:
        values = summary["final_eval"]["rolling_accuracy"]
        axis.plot(np.arange(1, len(values) + 1), values, label=summary["condition"])
    axis.axvline(6, color="black", linestyle="--", linewidth=1, label="trained loop")
    axis.set(xlabel="loop", ylabel="accuracy to f^loop", ylim=(0.0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "rolling_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for summary in summaries:
        values = summary["final_eval"]["best_position"]
        axis.plot(np.arange(1, len(values) + 1), values, label=summary["condition"])
    axis.plot([1, summaries[0]["final_eval"]["eval_loops"]], [1, summaries[0]["final_eval"]["eval_loops"]], color="black", linestyle=":", label="ideal")
    axis.axvline(6, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="loop", ylabel="best-matching path position")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "best_position.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(summaries),
        1,
        figsize=(10, max(3.2 * len(summaries), 4)),
        squeeze=False,
    )
    for axis, summary in zip(axes[:, 0], summaries):
        matrix = np.asarray(summary["final_eval"]["path_position_accuracy"], dtype=np.float32)
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.axhline(5.5, color="white", linestyle="--", linewidth=1)
        axis.set(ylabel="readout loop", title=summary["condition"])
    axes[-1, 0].set_xlabel("true path position")
    figure.colorbar(image, ax=axes[:, 0].tolist(), label="accuracy", fraction=0.02)
    figure.subplots_adjust(left=0.09, right=0.9, top=0.96, bottom=0.07, hspace=0.35)
    figure.savefig(out_dir / "path_accuracy_heatmaps.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for summary in summaries:
        metrics = summary["final_eval"]
        loops = np.arange(1, len(metrics["state_norm"]) + 1)
        axes[0].plot(loops, metrics["state_norm"], label=summary["condition"])
        axes[1].plot(loops, metrics["update_state_ratio"], label=summary["condition"])
    axes[0].set(xlabel="loop", ylabel="answer-state L2 norm")
    axes[1].set(xlabel="loop", ylabel="effective update / incoming state")
    for axis in axes:
        axis.axvline(6, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(out_dir / "state_dynamics.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final-only looped Transformers on a single-cycle graph task."
    )
    parser.add_argument("--node-count", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--train-loops", type=int, default=6)
    parser.add_argument("--outer-groups", type=int, nargs="+", default=[0, 1, 4, 16])
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-loops", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/long_cycle_group_rmsnorm_20260714"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model_cfg = LongCycleConfig(
        node_count=args.node_count,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        max_loops=args.train_loops,
    )
    train_cfg = TrainConfig(
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
        eval_loops=args.eval_loops,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        out_dir=args.out_dir,
        force=args.force,
    )
    summaries = []
    for groups in args.outer_groups:
        run_dir = train_condition(
            replace(base_model_cfg, outer_norm_groups=groups),
            train_cfg,
        )
        summaries.append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
    write_comparison(args.out_dir, summaries)


if __name__ == "__main__":
    main()
