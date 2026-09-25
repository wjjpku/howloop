from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    TransformerBlock,
    ce_by_loop,
    count_parameters,
    evaluate,
    make_graph_path_batch,
    pick_device,
    set_seed,
)


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate the orthogonal factor of a matrix gradient for Muon."""
    if g.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2D tensor")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x.to(dtype=g.dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params: list[nn.Parameter],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.ndim != 2:
                    raise ValueError("Muon param group should contain only 2D parameters")
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, ns_steps)
                fan_out, fan_in = param.shape
                scale = max(1.0, fan_out / max(1, fan_in)) ** 0.5
                param.add_(update, alpha=-lr * scale)
        return loss


class HybridOptimizer:
    def __init__(self, optimizers: list[Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


class StandardGraphPathTransformer(nn.Module):
    def __init__(self, cfg: GraphPathConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward_all(
        self,
        tokens: torch.Tensor,
        *,
        max_loops: int | None = None,
        return_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        readouts = self.cfg.n_layers if max_loops is None else min(max_loops, self.cfg.n_layers)
        x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        logits_by_readout: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx < readouts:
                final_state = self.ln_final(x[:, -1, :])
                logits_by_readout.append(self.unembed(final_state)[:, : self.cfg.node_count])
                if return_states:
                    states.append(final_state)
        out = {"logits_by_loop": torch.stack(logits_by_readout, dim=1)}
        if return_states:
            out["states_by_loop"] = torch.stack(states, dim=1)
        return out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_all(tokens)["logits_by_loop"][:, -1, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare graph-path S6 against L1x6.")
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--models", nargs="+", choices=["S6", "L1x6"], default=["S6", "L1x6"])
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.01)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/graph_path_s6_vs_l1x6_20260707"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def cosine_lr(step: int, *, base_lr: float, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    keys = ["model", "optimizer", "step", "lr", "muon_lr", "train_loss", "elapsed_sec"]
    max_readout = len(history[-1]["loop_accuracy"])
    for idx in range(max_readout):
        keys.append(f"eval_acc_readout_{idx + 1}")
        keys.append(f"eval_loss_readout_{idx + 1}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            flat = {key: row.get(key, "") for key in keys}
            for idx, value in enumerate(row["loop_accuracy"]):
                flat[f"eval_acc_readout_{idx + 1}"] = value
            for idx, value in enumerate(row["loop_loss"]):
                flat[f"eval_loss_readout_{idx + 1}"] = value
            writer.writerow(flat)


def build_model(name: str, cfg: GraphPathConfig) -> tuple[nn.Module, GraphPathConfig]:
    if name == "L1x6":
        run_cfg = GraphPathConfig(**{**asdict(cfg), "n_layers": 1, "max_loops": 6})
        return LoopedGraphPathTransformer(run_cfg), run_cfg
    if name == "S6":
        run_cfg = GraphPathConfig(**{**asdict(cfg), "n_layers": 6, "max_loops": 6})
        return StandardGraphPathTransformer(run_cfg), run_cfg
    raise ValueError(f"unknown model name: {name}")


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> Optimizer | HybridOptimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("blocks.") and param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    optimizers: list[Optimizer] = []
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=args.muon_lr,
                momentum=args.muon_momentum,
                weight_decay=args.muon_weight_decay,
                nesterov=args.muon_nesterov,
                ns_steps=args.muon_ns_steps,
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay,
            )
        )
    return HybridOptimizer(optimizers)


def set_optimizer_lrs(optimizer: Optimizer | HybridOptimizer, *, adamw_lr: float, muon_lr: float) -> None:
    for group in optimizer.param_groups:
        if "momentum" in group and "ns_steps" in group:
            group["lr"] = muon_lr
        else:
            group["lr"] = adamw_lr


def train_model(
    name: str,
    cfg: GraphPathConfig,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    run_label = name if args.optimizer == "adamw" else f"{name}_{args.optimizer}"
    run_dir = args.out_dir / run_label
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"{run_dir} exists. Pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    init_offset = {"L1x6": 1001, "S6": 2002}[name]
    set_seed(args.seed + init_offset)
    model, run_cfg = build_model(name, cfg)
    model = model.to(device)
    optimizer = make_optimizer(model, args)
    param_count = count_parameters(model)
    metadata = {
        "model": name,
        "config": asdict(run_cfg),
        "args": vars(args),
        "optimizer": args.optimizer,
        "parameter_count": param_count,
        "device": str(device),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    set_seed(args.seed + 4242)
    history: list[dict[str, Any]] = []
    best_acc = -1.0
    best_step = 0
    start_time = time.time()
    autocast_device = "cuda" if device.type == "cuda" else device.type
    for step in range(1, args.steps + 1):
        model.train()
        lr = cosine_lr(step - 1, base_lr=args.lr, total_steps=args.steps, warmup_steps=args.warmup_steps)
        muon_lr = cosine_lr(
            step - 1,
            base_lr=args.muon_lr,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
        set_optimizer_lrs(optimizer, adamw_lr=lr, muon_lr=muon_lr)
        tokens, target, _, _ = make_graph_path_batch(run_cfg, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=6)["logits_by_loop"]
            loss = ce_by_loop(logits_by_loop, target)[:, -1].mean()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,  # type: ignore[arg-type]
                run_cfg,
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                max_loops=6,
                amp_enabled=args.amp,
            )
            final_acc = metrics["loop_accuracy"][-1]
            row = {
                "model": name,
                "optimizer": args.optimizer,
                "step": step,
                "lr": lr,
                "muon_lr": muon_lr if args.optimizer == "muon" else "",
                "train_loss": float(loss.detach().cpu()),
                "elapsed_sec": time.time() - start_time,
                **metrics,
            }
            history.append(row)
            write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if final_acc > best_acc:
                best_acc = final_acc
                best_step = step
                if args.save_checkpoints:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": asdict(run_cfg),
                            "model_name": name,
                            "optimizer": args.optimizer,
                            "step": step,
                            "metrics": metrics,
                            "parameter_count": param_count,
                        },
                        run_dir / "best.pt",
                    )
            if step == 1 or step % args.print_every == 0 or step == args.steps:
                accs = " ".join(f"R{i + 1}:{acc:.3f}" for i, acc in enumerate(metrics["loop_accuracy"]))
                print(
                    f"[{name} step={step:05d}] loss={float(loss.detach().cpu()):.4f} "
                    f"final_acc={final_acc:.3f} {accs}",
                    flush=True,
                )

    final_metrics = evaluate(
        model,  # type: ignore[arg-type]
        run_cfg,
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 64),
        max_loops=6,
        amp_enabled=args.amp,
    )
    summary = {
        "model": name,
        "run_label": run_label,
        "optimizer": args.optimizer,
        "parameter_count": param_count,
        "best_step": best_step,
        "best_final_accuracy": best_acc,
        "final_metrics": final_metrics,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_checkpoints:
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(run_cfg),
                "model_name": name,
                "optimizer": args.optimizer,
                "step": args.steps,
                "metrics": final_metrics,
                "parameter_count": param_count,
            },
            run_dir / "final.pt",
        )
    plot_single_run(run_dir, history, summary)
    return summary


def plot_single_run(run_dir: Path, history: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    steps = [row["step"] for row in history]
    max_readout = len(history[-1]["loop_accuracy"])
    plt.figure(figsize=(9, 5))
    for idx in range(max_readout):
        plt.plot(steps, [row["loop_accuracy"][idx] for row in history], label=f"readout {idx + 1}")
    plt.xlabel("train step")
    plt.ylabel("eval accuracy")
    plt.ylim(0, 1.02)
    plt.title(f"{summary.get('run_label', summary['model'])}: accuracy by readout")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_by_readout_over_training.png", dpi=180)
    plt.close()

    depth_readout = np.array(summary["final_metrics"]["depth_loop_accuracy"], dtype=np.float32)
    plt.figure(figsize=(10, 5))
    im = plt.imshow(depth_readout, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="accuracy")
    plt.xticks(range(max_readout), [str(i + 1) for i in range(max_readout)])
    plt.yticks(range(depth_readout.shape[0]), [str(i + 1) for i in range(depth_readout.shape[0])])
    plt.xlabel("readout index")
    plt.ylabel("query depth")
    plt.title(f"{summary.get('run_label', summary['model'])}: final depth x readout accuracy")
    for y in range(depth_readout.shape[0]):
        for x in range(depth_readout.shape[1]):
            value = depth_readout[y, x]
            plt.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "final_depth_readout_accuracy_heatmap.png", dpi=180)
    plt.close()


def plot_comparison(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    histories = {}
    for summary in summaries:
        path = Path(summary["run_dir"]) / "history.json"
        histories[summary.get("run_label", summary["model"])] = json.loads(path.read_text(encoding="utf-8"))

    plt.figure(figsize=(9, 5))
    for name, history in histories.items():
        plt.plot(
            [row["step"] for row in history],
            [row["loop_accuracy"][-1] for row in history],
            marker="o",
            markersize=3,
            label=name,
        )
    plt.xlabel("train step")
    plt.ylabel("final-readout eval accuracy")
    plt.ylim(0, 1.02)
    plt.title("Graph-path convergence: S6 vs L1x6")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "s6_vs_l1x6_final_accuracy_convergence.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for name, history in histories.items():
        plt.plot(
            [row["step"] for row in history],
            [row["train_loss"] for row in history],
            marker="o",
            markersize=3,
            label=name,
        )
    plt.xlabel("train step")
    plt.ylabel("train final CE")
    plt.title("Graph-path training loss: S6 vs L1x6")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "s6_vs_l1x6_train_loss_convergence.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(
        1,
        len(summaries),
        figsize=(6.4 * len(summaries), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(summaries) == 1:
        axes = [axes]
    for ax, summary in zip(axes, summaries):
        arr = np.array(summary["final_metrics"]["depth_loop_accuracy"], dtype=np.float32)
        im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(summary.get("run_label", summary["model"]))
        ax.set_xticks(range(arr.shape[1]), [str(i + 1) for i in range(arr.shape[1])])
        ax.set_yticks(range(arr.shape[0]), [str(i + 1) for i in range(arr.shape[0])])
        ax.set_xlabel("readout index")
        ax.set_ylabel("query depth")
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                value = arr[y, x]
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, label="accuracy", shrink=0.85, location="right")
    fig.suptitle("Depth x readout accuracy at final checkpoint")
    fig.savefig(out_dir / "s6_vs_l1x6_depth_readout_heatmaps.png", dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    by_name = {summary["model"]: summary for summary in summaries}
    lines = [
        "# Graph-path S6 vs L1x6",
        "",
        "## Setup",
        "",
        "- Task: in-context random permutation graph path.",
        "- Loss: final-only cross entropy at readout 6.",
        f"- node_count: {args.node_count}",
        f"- max_depth: {args.max_depth}",
        f"- d_model: {args.d_model}",
        f"- d_mlp: {args.d_mlp}",
        f"- n_heads: {args.n_heads}",
        f"- steps: {args.steps}",
        f"- weight_decay: {args.weight_decay}",
        "",
        "S6 and L1x6 have matched compute depth but not matched parameter count.",
        "",
        "## Final Metrics",
        "",
        "| model | params | best step | best final acc | final acc | earliest correct readout |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["S6", "L1x6"]:
        summary = by_name[name]
        metrics = summary["final_metrics"]
        lines.append(
            f"| {name} | {summary['parameter_count']:,} | {summary['best_step']} | "
            f"{summary['best_final_accuracy']:.4f} | {metrics['loop_accuracy'][-1]:.4f} | "
            f"{metrics['earliest_correct_loop']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![final accuracy](s6_vs_l1x6_final_accuracy_convergence.png)",
            "",
            "![depth readout](s6_vs_l1x6_depth_readout_heatmaps.png)",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    cfg = GraphPathConfig(
        node_count=args.node_count,
        max_depth=args.max_depth,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=1,
        max_loops=6,
        dropout=args.dropout,
    )
    print(f"device={device} cfg={cfg}", flush=True)
    summaries = []
    for name in args.models:
        summary = train_model(name, cfg, args, device=device)
        summaries.append(summary)
        (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        if len(summaries) >= 1:
            plot_comparison(args.out_dir, summaries)
            if len(summaries) == 2 and {item["model"] for item in summaries} == {"S6", "L1x6"}:
                write_report(args.out_dir, args, summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
