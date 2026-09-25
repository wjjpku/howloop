from __future__ import annotations

import argparse
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

from reasoning_loop.boolean_dag_data import BooleanDAGBatch, BooleanDAGConfig, make_boolean_dag_batch
from reasoning_loop.graph_path_loop import cosine_lr, pick_device, set_seed
from reasoning_loop.ouro_boolean_dag import OuroBooleanDAG, OuroBooleanDAGConfig


@dataclass(frozen=True)
class OuroDAGTrainConfig:
    loss_mode: Literal["final", "uniform", "ouro"] = "final"
    steps: int = 10_000
    batch_size: int = 128
    eval_batch_size: int = 256
    eval_batches: int = 4
    eval_every: int = 500
    print_every: int = 100
    save_every: int = 5_000
    lr: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    entropy_beta_initial: float = 0.1
    entropy_beta_final: float = 0.05
    entropy_switch_step: int = 5_000
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    output_dir: Path = Path("results/ouro_boolean_dag")
    run_name: str | None = None
    force: bool = False

    def __post_init__(self) -> None:
        sizes = (self.steps, self.batch_size, self.eval_batch_size, self.eval_batches)
        if any(value < 1 for value in sizes):
            raise ValueError("training and evaluation sizes must be positive")
        if self.batch_size % 2 or self.eval_batch_size % 2:
            raise ValueError("balanced Boolean DAG batches must have even sizes")
        if self.loss_mode not in {"final", "uniform", "ouro"}:
            raise ValueError("loss_mode must be final, uniform, or ouro")
        if self.entropy_beta_initial < 0 or self.entropy_beta_final < 0:
            raise ValueError("entropy beta values must be nonnegative")


def final_root_loss(logits_by_step: torch.Tensor, batch: BooleanDAGBatch) -> torch.Tensor:
    root_indices = batch.root_mask.long().argmax(dim=1)
    root_logits = logits_by_step[:, -1].gather(
        1,
        root_indices[:, None, None].expand(-1, 1, logits_by_step.shape[-1]),
    ).squeeze(1)
    return F.cross_entropy(root_logits, batch.root_values + 1)


def _root_step_logits(logits_by_step: torch.Tensor, batch: BooleanDAGBatch) -> torch.Tensor:
    root_indices = batch.root_mask.long().argmax(dim=1)
    return logits_by_step.gather(
        2,
        root_indices[:, None, None, None].expand(
            -1, logits_by_step.shape[1], 1, logits_by_step.shape[-1]
        ),
    ).squeeze(2)


def root_step_losses(logits_by_step: torch.Tensor, batch: BooleanDAGBatch) -> torch.Tensor:
    root_logits = _root_step_logits(logits_by_step, batch)
    targets = (batch.root_values + 1)[:, None].expand(-1, root_logits.shape[1])
    return F.cross_entropy(
        root_logits.flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).view_as(targets)


def uniform_root_loss(
    logits_by_step: torch.Tensor,
    batch: BooleanDAGBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    step_losses = root_step_losses(logits_by_step, batch)
    per_step = step_losses.mean(dim=0)
    loss = per_step.mean()
    return loss, {"task_loss": loss.detach(), "per_step_losses": per_step.detach()}


def ouro_root_loss(
    output: dict[str, torch.Tensor],
    batch: BooleanDAGBatch,
    *,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "exit_probs" not in output:
        raise ValueError("Ouro loss requires a model with an exit gate")
    step_losses = root_step_losses(output["logits_by_step"], batch)
    root_indices = batch.root_mask.long().argmax(dim=1)
    exit_probs = output["exit_probs"].gather(
        2,
        root_indices[:, None, None].expand(-1, output["exit_probs"].shape[1], 1),
    ).squeeze(2)
    task_loss = (exit_probs * step_losses).sum(dim=1).mean()
    entropy = -(exit_probs * exit_probs.clamp_min(1e-10).log()).sum(dim=1).mean()
    loss = task_loss - beta * entropy
    step_numbers = torch.arange(
        1,
        exit_probs.shape[1] + 1,
        device=exit_probs.device,
        dtype=exit_probs.dtype,
    )
    return loss, {
        "task_loss": task_loss.detach(),
        "entropy": entropy.detach(),
        "expected_depth": (exit_probs * step_numbers[None]).sum(dim=1).mean().detach(),
        "per_step_losses": step_losses.mean(dim=0).detach(),
        "mean_exit_probs": exit_probs.mean(dim=0).detach(),
    }


@torch.no_grad()
def evaluate_ouro_boolean_dag(
    model: OuroBooleanDAG,
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: OuroBooleanDAGConfig,
    device: torch.device,
    batch_size: int,
    batches: int,
    amp: bool,
    max_steps: int | None = None,
) -> dict[str, Any]:
    model.eval()
    steps = model_cfg.steps if max_steps is None else max_steps
    generator = torch.Generator(device=device).manual_seed(20_260_715)
    depth_step_correct = torch.zeros(data_cfg.eval_max_depth, steps, device=device)
    depth_counts = torch.zeros(data_cfg.eval_max_depth, device=device)
    full_node_correct = torch.zeros(steps, device=device)
    full_node_count = 0
    expected_correct = torch.zeros(data_cfg.eval_max_depth, device=device)
    expected_depth_sum = torch.zeros(data_cfg.eval_max_depth, device=device)
    for depth in range(1, data_cfg.eval_max_depth + 1):
        requested_depths = torch.full((batch_size,), depth, device=device, dtype=torch.long)
        for _ in range(batches):
            batch = make_boolean_dag_batch(
                data_cfg,
                batch_size,
                device,
                depths=requested_depths,
                generator=generator,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                output = model.forward_all(batch, max_steps=steps)
                logits = output["logits_by_step"]
            root_indices = batch.root_mask.long().argmax(dim=1)
            root_logits = logits.gather(
                2,
                root_indices[:, None, None, None].expand(-1, steps, 1, 3),
            ).squeeze(2)
            depth_step_correct[depth - 1] += root_logits.argmax(dim=-1).eq(
                (batch.root_values + 1)[:, None]
            ).sum(dim=0)
            depth_counts[depth - 1] += batch_size
            full_node_correct += logits.argmax(dim=-1).eq(
                (batch.values + 1)[:, None, :]
            ).sum(dim=(0, 2))
            full_node_count += batch_size * data_cfg.node_count
            if "exit_probs" in output:
                root_exit_probs = output["exit_probs"].gather(
                    2,
                    root_indices[:, None, None].expand(-1, steps, 1),
                ).squeeze(2)
                root_correct = root_logits.argmax(dim=-1).eq(
                    (batch.root_values + 1)[:, None]
                )
                expected_correct[depth - 1] += (
                    root_exit_probs * root_correct.to(root_exit_probs.dtype)
                ).sum()
                step_numbers = torch.arange(
                    1, steps + 1, device=device, dtype=root_exit_probs.dtype
                )
                expected_depth_sum[depth - 1] += (
                    root_exit_probs * step_numbers[None]
                ).sum()
    depth_step_accuracy = depth_step_correct / depth_counts[:, None].clamp_min(1)
    train_rows = depth_step_accuracy[: data_cfg.train_max_depth]
    ood_rows = depth_step_accuracy[data_cfg.train_max_depth :]
    result = {
        "depth_step_accuracy": depth_step_accuracy.cpu().tolist(),
        "root_accuracy_by_step": depth_step_accuracy.mean(dim=0).cpu().tolist(),
        "train_final_accuracy": float(train_rows[:, -1].mean().cpu()),
        "ood_final_accuracy": float(ood_rows[:, -1].mean().cpu()) if ood_rows.numel() else None,
        "full_node_accuracy_by_step": (full_node_correct / full_node_count).cpu().tolist(),
        "examples_per_depth": batch_size * batches,
    }
    if model.exit_gate is not None:
        result["expected_exit_accuracy_by_depth"] = (
            expected_correct / depth_counts.clamp_min(1)
        ).cpu().tolist()
        result["mean_expected_depth_by_depth"] = (
            expected_depth_sum / depth_counts.clamp_min(1)
        ).cpu().tolist()
    return result


def _plot_history(path: Path, history: list[dict[str, Any]]) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(steps, [row["train_loss"] for row in history], marker="o")
    axes[0].set(xlabel="step", ylabel="final-root CE", title="Training loss")
    axes[1].plot(
        steps,
        [row["metrics"]["train_final_accuracy"] for row in history],
        marker="o",
        label="train depths",
    )
    axes[1].plot(
        steps,
        [row["metrics"]["ood_final_accuracy"] for row in history],
        marker="o",
        label="OOD depths",
    )
    axes[1].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[1].set(xlabel="step", ylabel="accuracy", ylim=(0, 1.02), title="Final-loop root accuracy")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _checkpoint(
    model: OuroBooleanDAG,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    data_cfg: BooleanDAGConfig,
    model_cfg: OuroBooleanDAGConfig,
    train_cfg: OuroDAGTrainConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "train_config": {**asdict(train_cfg), "output_dir": str(train_cfg.output_dir)},
        "metrics": metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def train_ouro_boolean_dag(
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: OuroBooleanDAGConfig,
    train_cfg: OuroDAGTrainConfig,
) -> Path:
    name = train_cfg.run_name or (
        f"ouro_dag_N{data_cfg.node_count}_D{data_cfg.train_max_depth}"
        f"_d{model_cfg.d_model}_B{model_cfg.n_layers}_L{model_cfg.steps}_seed{train_cfg.seed}"
    )
    run_dir = Path(train_cfg.output_dir) / name
    if run_dir.exists() and any(run_dir.iterdir()) and not train_cfg.force:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(train_cfg.device)
    set_seed(train_cfg.seed)
    model = OuroBooleanDAG(data_cfg, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )
    history: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}
    started = time.time()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
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
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=train_cfg.amp and device.type == "cuda",
        ):
            output = model.forward_all(batch)
            logits = output["logits_by_step"]
            if train_cfg.loss_mode == "final":
                loss = final_root_loss(logits, batch)
                diagnostics: dict[str, torch.Tensor] = {"task_loss": loss.detach()}
            elif train_cfg.loss_mode == "uniform":
                loss, diagnostics = uniform_root_loss(logits, batch)
            else:
                beta = (
                    train_cfg.entropy_beta_initial
                    if step <= train_cfg.entropy_switch_step
                    else train_cfg.entropy_beta_final
                )
                loss, diagnostics = ouro_root_loss(output, batch, beta=beta)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        should_eval = step == 1 or step % train_cfg.eval_every == 0 or step == train_cfg.steps
        if should_eval:
            last_metrics = evaluate_ouro_boolean_dag(
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
                "grad_norm": float(grad_norm.detach().cpu()),
                "elapsed_sec": time.time() - started,
                "metrics": last_metrics,
                "diagnostics": {
                    name: value.detach().float().cpu().tolist()
                    for name, value in diagnostics.items()
                },
            }
            history.append(row)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            _plot_history(run_dir / "training_curves.png", history)
            print(
                f"[step={step:05d}] loss={row['train_loss']:.4f} "
                f"train={last_metrics['train_final_accuracy']:.3f} "
                f"ood={_format_optional_float(last_metrics['ood_final_accuracy'])} "
                f"loops={[round(value, 3) for value in last_metrics['root_accuracy_by_step']]}",
                flush=True,
            )
        if train_cfg.save_every and step % train_cfg.save_every == 0:
            torch.save(
                _checkpoint(
                    model,
                    optimizer,
                    step=step,
                    data_cfg=data_cfg,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    metrics=last_metrics,
                ),
                run_dir / f"checkpoint_step_{step:07d}.pt",
            )

    payload = _checkpoint(
        model,
        optimizer,
        step=train_cfg.steps,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        metrics=last_metrics,
    )
    torch.save(payload, run_dir / "final.pt")
    summary = {
        "parameter_count": payload["parameter_count"],
        "step": train_cfg.steps,
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "final_metrics": last_metrics,
        "elapsed_sec": time.time() - started,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ouro-style recurrent stacks on Boolean DAGs.")
    parser.add_argument(
        "--model-size",
        choices=("40m", "40m-shallow", "100m", "100m-shallow"),
        default="40m",
    )
    parser.add_argument("--loss-mode", choices=("final", "uniform", "ouro"), default="final")
    parser.add_argument("--node-count", type=int, default=48)
    parser.add_argument("--leaf-count", type=int, default=8)
    parser.add_argument("--train-max-depth", type=int, default=8)
    parser.add_argument("--eval-max-depth", type=int, default=12)
    parser.add_argument(
        "--balanced-output-xor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--balanced-logic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--recurrent-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--entropy-beta-initial", type=float, default=0.1)
    parser.add_argument("--entropy-beta-final", type=float, default=0.05)
    parser.add_argument("--entropy-switch-step", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presets = {
        "40m": OuroBooleanDAGConfig.forty_million,
        "40m-shallow": OuroBooleanDAGConfig.shallow_forty_million,
        "100m": OuroBooleanDAGConfig.hundred_million,
        "100m-shallow": OuroBooleanDAGConfig.shallow_hundred_million,
    }
    model_cfg = presets[args.model_size]()
    model_cfg = OuroBooleanDAGConfig(
        **{
            **asdict(model_cfg),
            "steps": args.recurrent_steps,
            "use_exit_gate": args.loss_mode == "ouro",
        }
    )
    run_dir = train_ouro_boolean_dag(
        data_cfg=BooleanDAGConfig(
            node_count=args.node_count,
            leaf_count=args.leaf_count,
            train_max_depth=args.train_max_depth,
            eval_max_depth=args.eval_max_depth,
            balanced_output_xor=args.balanced_output_xor,
            balanced_logic=args.balanced_logic,
        ),
        model_cfg=model_cfg,
        train_cfg=OuroDAGTrainConfig(
            loss_mode=args.loss_mode,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_batches=args.eval_batches,
            eval_every=args.eval_every,
            print_every=args.print_every,
            save_every=args.save_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            entropy_beta_initial=args.entropy_beta_initial,
            entropy_beta_final=args.entropy_beta_final,
            entropy_switch_step=args.entropy_switch_step,
            seed=args.seed,
            device=args.device,
            amp=args.amp,
            output_dir=args.output_dir,
            run_name=args.run_name,
            force=args.force,
        ),
    )
    print(run_dir, flush=True)


if __name__ == "__main__":
    main()
