from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from reasoning_loop.boolean_dag_data import (
    BooleanDAGBatch,
    BooleanDAGConfig,
    make_boolean_dag_batch,
    wavefront_targets,
)
from reasoning_loop.boolean_dag_model import (
    BooleanDAGModelConfig,
    build_boolean_dag_model,
    count_parameters,
)
from reasoning_loop.graph_path_loop import cosine_lr, pick_device, set_seed


TASK_VERSION = "boolean_dag_v2_conditional_balance"


@dataclass(frozen=True)
class TrainConfig:
    architecture: Literal["looped", "periodic2", "standard"] = "looped"
    loss_mode: Literal["final", "intermediate"] = "final"
    steps: int = 20_000
    batch_size: int = 512
    eval_batch_size: int = 1024
    eval_batches: int = 16
    eval_every: int = 500
    print_every: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    out_dir: Path = Path("results/boolean_dag_wavefront")
    run_name: str | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in {"looped", "periodic2", "standard"}:
            raise ValueError("architecture must be looped, periodic2, or standard")
        if self.loss_mode not in {"final", "intermediate"}:
            raise ValueError("loss_mode must be final or intermediate")
        if self.steps < 1 or self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("steps must be positive and batch_size must be positive and even")
        if self.eval_batch_size < 2 or self.eval_batch_size % 2 or self.eval_batches < 1:
            raise ValueError("evaluation batch settings must be positive with an even batch size")


def _root_indices(batch: BooleanDAGBatch) -> torch.Tensor:
    return batch.root_mask.long().argmax(dim=1)


def final_only_loss(logits_by_step: torch.Tensor, batch: BooleanDAGBatch) -> torch.Tensor:
    if logits_by_step.ndim != 4 or logits_by_step.shape[0] != batch.batch_size:
        raise ValueError("logits_by_step must have shape [batch, step, node, 3]")
    root_indices = _root_indices(batch)
    final_logits = logits_by_step[:, -1].gather(
        1,
        root_indices[:, None, None].expand(-1, 1, logits_by_step.shape[-1]),
    ).squeeze(1)
    return F.cross_entropy(final_logits, batch.root_values + 1)


def stratified_group_mean(
    losses: torch.Tensor,
    kinds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if losses.shape != kinds.shape or losses.shape != targets.shape or losses.shape != mask.shape:
        raise ValueError("losses, kinds, targets, and mask must have identical shapes")
    selected_losses = losses[mask]
    if selected_losses.numel() == 0:
        raise ValueError("stratified group is empty")
    cell_indices = (kinds[mask] * 3 + targets[mask]).long()
    sums = losses.new_zeros(12).scatter_add_(0, cell_indices, selected_losses)
    counts = losses.new_zeros(12).scatter_add_(
        0,
        cell_indices,
        torch.ones_like(selected_losses),
    )
    populated = counts > 0
    return (sums[populated] / counts[populated]).mean()


def intermediate_state_loss(
    logits_by_step: torch.Tensor,
    batch: BooleanDAGBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits_by_step.ndim != 4 or logits_by_step.shape[-1] != 3:
        raise ValueError("logits_by_step must have shape [batch, step, node, 3]")
    targets = wavefront_targets(batch, readouts=logits_by_step.shape[1])
    per_node = F.cross_entropy(
        logits_by_step.reshape(-1, 3),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    step_losses: list[torch.Tensor] = []
    frontier_losses: list[torch.Tensor] = []
    persistence_losses: list[torch.Tensor] = []
    unknown_losses: list[torch.Tensor] = []
    for step in range(logits_by_step.shape[1]):
        readout = step + 1
        frontier = batch.levels.eq(readout)
        persistence = batch.levels.lt(readout)
        unknown = batch.levels.gt(readout)
        groups: list[torch.Tensor] = []
        if frontier.any():
            frontier_loss = stratified_group_mean(
                per_node[:, step], batch.kinds, targets[:, step], frontier
            )
            groups.append(frontier_loss)
            frontier_losses.append(frontier_loss)
        if persistence.any():
            persistence_loss = stratified_group_mean(
                per_node[:, step], batch.kinds, targets[:, step], persistence
            )
            groups.append(persistence_loss)
            persistence_losses.append(persistence_loss)
        if unknown.any():
            unknown_loss = stratified_group_mean(
                per_node[:, step], batch.kinds, targets[:, step], unknown
            )
            groups.append(unknown_loss)
            unknown_losses.append(unknown_loss)
        step_losses.append(torch.stack(groups).mean())
    zero = logits_by_step.new_zeros(())
    return torch.stack(step_losses).mean(), {
        "frontier_ce": torch.stack(frontier_losses).mean() if frontier_losses else zero,
        "persistence_ce": (
            torch.stack(persistence_losses).mean() if persistence_losses else zero
        ),
        "unknown_ce": torch.stack(unknown_losses).mean() if unknown_losses else zero,
    }


@torch.no_grad()
def evaluate_train_depths(
    model: torch.nn.Module,
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    device: torch.device,
    batch_size: int,
    batches: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    eval_generator = torch.Generator(device=device).manual_seed(20_260_710)
    correct_by_step = torch.zeros(model_cfg.steps, device=device)
    count_by_step = torch.zeros(model_cfg.steps, device=device)
    correct_by_depth_step = torch.zeros(
        data_cfg.train_max_depth, model_cfg.steps, device=device
    )
    count_by_depth_step = torch.zeros_like(correct_by_depth_step)
    state_correct = torch.zeros(model_cfg.steps, device=device)
    state_count = torch.zeros(model_cfg.steps, device=device)
    for _ in range(batches):
        batch = make_boolean_dag_batch(
            data_cfg,
            batch_size,
            device,
            generator=eval_generator,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model.forward_all(batch)["logits_by_step"]
        root_indices = _root_indices(batch)
        root_logits = logits.gather(
            2,
            root_indices[:, None, None, None].expand(
                -1, model_cfg.steps, 1, logits.shape[-1]
            ),
        ).squeeze(2)
        root_correct = root_logits.argmax(dim=-1).eq((batch.root_values + 1)[:, None])
        correct_by_step += root_correct.sum(dim=0)
        count_by_step += batch_size
        for depth in range(1, data_cfg.train_max_depth + 1):
            mask = batch.depths.eq(depth)
            correct_by_depth_step[depth - 1] += root_correct[mask].sum(dim=0)
            count_by_depth_step[depth - 1] += mask.sum()
        state_targets = wavefront_targets(batch, readouts=model_cfg.steps)
        state_correct += logits.argmax(dim=-1).eq(state_targets).sum(dim=(0, 2))
        state_count += batch_size * data_cfg.node_count
    root_accuracy = correct_by_step / count_by_step.clamp_min(1)
    depth_step_accuracy = correct_by_depth_step / count_by_depth_step.clamp_min(1)
    return {
        "root_accuracy_by_step": root_accuracy.cpu().tolist(),
        "depth_step_accuracy": depth_step_accuracy.cpu().tolist(),
        "final_root_accuracy": float(root_accuracy[-1].cpu()),
        "depth4_step1_accuracy": float(depth_step_accuracy[-1, 0].cpu()),
        "native_state_accuracy_by_step": (state_correct / state_count.clamp_min(1)).cpu().tolist(),
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
    axes[0].set(title="Training loss", xlabel="step", ylabel="loss")
    axes[1].plot(steps, [row["eval_final_root_accuracy"] for row in history])
    axes[1].set(title="Held-out root accuracy", xlabel="step", ylabel="accuracy", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def make_run_name(
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    train_cfg: TrainConfig,
) -> str:
    return (
        f"boolean_dag_{train_cfg.architecture}_{train_cfg.loss_mode}_N{data_cfg.node_count}"
        f"_D{data_cfg.train_max_depth}_d{model_cfg.d_model}_L{model_cfg.steps}"
        f"_outerG{model_cfg.outer_norm_groups}_innerG{model_cfg.inner_norm_groups}"
        f"_seed{train_cfg.seed}"
    )


def train_boolean_dag(
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    train_cfg: TrainConfig,
) -> Path:
    run_name = train_cfg.run_name or make_run_name(data_cfg, model_cfg, train_cfg)
    run_dir = train_cfg.out_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not train_cfg.force:
        raise FileExistsError(f"{run_dir} exists; pass force=True to overwrite artifacts")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(train_cfg.seed)
    device = pick_device(train_cfg.device)
    model = build_boolean_dag_model(
        architecture=train_cfg.architecture,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )
    history: list[dict[str, Any]] = []
    start_time = time.time()
    last_metrics: dict[str, Any] = {}
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
        batch = make_boolean_dag_batch(data_cfg, train_cfg.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=train_cfg.amp and device.type == "cuda",
        ):
            logits = model.forward_all(batch)["logits_by_step"]
            if train_cfg.loss_mode == "final":
                loss = final_only_loss(logits, batch)
                pieces = {
                    "frontier_ce": loss.detach(),
                    "persistence_ce": loss.detach().new_zeros(()),
                    "unknown_ce": loss.detach().new_zeros(()),
                }
            else:
                loss, pieces = intermediate_state_loss(logits, batch)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step == 1 or step % train_cfg.eval_every == 0 or step == train_cfg.steps:
            last_metrics = evaluate_train_depths(
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
                "eval_final_root_accuracy": last_metrics["final_root_accuracy"],
                "eval_depth4_step1_accuracy": last_metrics["depth4_step1_accuracy"],
                "eval_root_accuracy_by_step": json.dumps(last_metrics["root_accuracy_by_step"]),
                "eval_depth_step_accuracy": json.dumps(last_metrics["depth_step_accuracy"]),
                "elapsed_sec": time.time() - start_time,
            }
            history.append(row)
            _write_history(run_dir / "history.csv", history)
            if step == 1 or step % train_cfg.print_every == 0 or step == train_cfg.steps:
                print(
                    f"[{train_cfg.architecture}/{train_cfg.loss_mode} step={step:05d}] "
                    f"loss={row['train_loss']:.4f} final_acc={row['eval_final_root_accuracy']:.3f} "
                    f"D4_step1={row['eval_depth4_step1_accuracy']:.3f}",
                    flush=True,
                )

    checkpoint = {
        "task_version": TASK_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "train_config": {**asdict(train_cfg), "out_dir": str(train_cfg.out_dir)},
        "step": train_cfg.steps,
        "architecture": train_cfg.architecture,
        "loss_mode": train_cfg.loss_mode,
        "seed": train_cfg.seed,
        "parameter_count": count_parameters(model),
        "metrics": last_metrics,
    }
    torch.save(checkpoint, run_dir / "final.pt")
    summary = {
        "task_version": TASK_VERSION,
        "architecture": train_cfg.architecture,
        "loss_mode": train_cfg.loss_mode,
        "seed": train_cfg.seed,
        "steps": train_cfg.steps,
        "parameter_count": checkpoint["parameter_count"],
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "final_metrics": last_metrics,
        "elapsed_sec": time.time() - start_time,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_training(run_dir / "training_curves.png", history)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformers on permutation-randomized Boolean DAGs.")
    parser.add_argument(
        "--architecture",
        choices=["looped", "periodic2", "standard"],
        default="looped",
    )
    parser.add_argument("--loss-mode", choices=["final", "intermediate"], default="final")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--model-steps", type=int, default=4)
    parser.add_argument("--outer-norm-groups", type=int, default=0)
    parser.add_argument("--inner-norm-groups", type=int, default=0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/boolean_dag_wavefront"))
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_boolean_dag(
        data_cfg=BooleanDAGConfig(),
        model_cfg=BooleanDAGModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            steps=args.model_steps,
            outer_norm_groups=args.outer_norm_groups,
            inner_norm_groups=args.inner_norm_groups,
            rms_norm_eps=args.rms_norm_eps,
        ),
        train_cfg=TrainConfig(
            architecture=args.architecture,
            loss_mode=args.loss_mode,
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
