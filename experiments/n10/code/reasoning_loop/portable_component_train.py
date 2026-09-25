"""Train and evaluate a portable latent-interface objective.

The portable objective asks a shared recurrent stack to consume a state made on
one random graph and apply the successor operation from a different graph.
This is a training-time analogue of the state-transplant intervention used in
the circuit analysis: the state should carry the current node, while the
receiver context should supply the transition rule.
"""

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

from reasoning_loop.graph_path_loop import LoopedGraphPathTransformer, count_parameters, pick_device, set_seed
from reasoning_loop.graph_path_stepwise import (
    StepwiseGraphPathConfig,
    make_stepwise_batch,
    pushforward_successor,
)


def apply_shared_stack(model: LoopedGraphPathTransformer, x: torch.Tensor) -> torch.Tensor:
    for block in model.blocks:
        x = block(x)
    return x


def readout(model: LoopedGraphPathTransformer, x: torch.Tensor) -> torch.Tensor:
    state = model.ln_final(x[:, -1, :])
    return model.unembed(state)[:, : model.cfg.node_count]


def run_with_raw_states(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    loops: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    logits: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for _ in range(loops):
        x = apply_shared_stack(model, x)
        states.append(x)
        logits.append(readout(model, x))
    return torch.stack(logits, dim=1), states


def ce_to_positions(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch, loops, _ = logits.shape
    return F.cross_entropy(
        logits.reshape(batch * loops, -1),
        targets[:, :loops].reshape(batch * loops),
        reduction="none",
    ).view(batch, loops)


def portable_transition_loss(
    model: LoopedGraphPathTransformer,
    cfg: StepwiseGraphPathConfig,
    tokens_a: torch.Tensor,
    targets_a: torch.Tensor,
    tokens_b: torch.Tensor,
    successors_b: torch.Tensor,
    *,
    loop_index: int,
) -> torch.Tensor:
    """Cross-context one-step loss with gradients through both donor and receiver."""
    logits_a, states_a = run_with_raw_states(model, tokens_a, loop_index)
    _, states_b = run_with_raw_states(model, tokens_b, loop_index)
    donor_state = states_a[-1][:, -1, :]
    receiver_state = states_b[-1].clone()
    patched_state = receiver_state.clone()
    patched_state[:, -1, :] = donor_state
    next_state = apply_shared_stack(model, patched_state)
    next_logits = readout(model, next_state)
    current = targets_a[:, loop_index - 1]
    target = successors_b.gather(1, current[:, None]).squeeze(1)
    return F.cross_entropy(next_logits, target)


def transition_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    successors: torch.Tensor,
    *,
    transition_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    first = F.cross_entropy(logits[:, 0], targets[:, 0])
    final = F.cross_entropy(logits[:, -1], targets[:, logits.shape[1] - 1])
    probs = logits[:, :-1].softmax(dim=-1)
    shifted = torch.stack(
        [pushforward_successor(probs[:, idx], successors) for idx in range(probs.shape[1])],
        dim=1,
    ).detach()
    transition = -(shifted * logits[:, 1:].log_softmax(dim=-1)).sum(dim=-1).mean()
    return first + final + transition_weight * transition, {
        "first_ce": first.detach(),
        "final_ce": final.detach(),
        "transition_ce": transition.detach(),
        "portable_ce": logits.new_zeros(()),
    }


def loss_for_mode(
    mode: str,
    model: LoopedGraphPathTransformer,
    cfg: StepwiseGraphPathConfig,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    successors: torch.Tensor,
    *,
    portable_weight: float,
    transition_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, _ = run_with_raw_states(model, tokens, cfg.max_loops)
    if mode == "final":
        final = F.cross_entropy(logits[:, -1], targets[:, cfg.max_loops - 1])
        return final, {
            "first_ce": F.cross_entropy(logits[:, 0], targets[:, 0]).detach(),
            "final_ce": final.detach(),
            "transition_ce": logits.new_zeros(()),
            "portable_ce": logits.new_zeros(()),
        }
    if mode == "transition":
        return transition_loss(logits, targets, successors, transition_weight=transition_weight)
    if mode != "portable":
        raise ValueError(f"unknown mode: {mode}")
    first = F.cross_entropy(logits[:, 0], targets[:, 0])
    final = F.cross_entropy(logits[:, -1], targets[:, cfg.max_loops - 1])
    loop_index = random.randint(1, cfg.max_loops - 1)
    tokens_b, _, successors_b, _ = make_stepwise_batch(cfg, tokens.shape[0], tokens.device)
    portable = portable_transition_loss(
        model,
        cfg,
        tokens,
        targets,
        tokens_b,
        successors_b,
        loop_index=loop_index,
    )
    return first + final + portable_weight * portable, {
        "first_ce": first.detach(),
        "final_ce": final.detach(),
        "transition_ce": logits.new_zeros(()),
        "portable_ce": portable.detach(),
    }


@torch.no_grad()
def evaluate(
    model: LoopedGraphPathTransformer,
    cfg: StepwiseGraphPathConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    overloop: int,
) -> dict[str, Any]:
    model.eval()
    loop_correct = torch.zeros(cfg.max_loops, device=device)
    final_correct = torch.zeros((), device=device)
    portable_correct = torch.zeros(cfg.max_loops - 1, device=device)
    total = torch.zeros((), device=device)
    for _ in range(batches):
        tokens_a, targets_a, _, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=max(cfg.max_depth, overloop)
        )
        tokens_b, _, successors_b, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=max(cfg.max_depth, overloop)
        )
        logits, states_a = run_with_raw_states(model, tokens_a, cfg.max_loops)
        pred = logits.argmax(dim=-1)
        loop_correct += pred.eq(targets_a[:, : cfg.max_loops]).float().sum(dim=0)
        final_correct += pred[:, -1].eq(targets_a[:, cfg.max_loops - 1]).float().sum()
        for idx in range(cfg.max_loops - 1):
            _, states_b = run_with_raw_states(model, tokens_b, idx + 1)
            patched = states_b[-1].clone()
            patched[:, -1, :] = states_a[idx][:, -1, :]
            next_logits = readout(model, apply_shared_stack(model, patched))
            current = targets_a[:, idx]
            target = successors_b.gather(1, current[:, None]).squeeze(1)
            portable_correct[idx] += next_logits.argmax(dim=-1).eq(target).float().sum()
        total += batch_size

    overloop_acc = torch.zeros(overloop, device=device)
    for _ in range(batches):
        tokens, targets, _, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=overloop
        )
        logits, _ = run_with_raw_states(model, tokens, overloop)
        overloop_acc += logits.argmax(dim=-1).eq(targets[:, :overloop]).float().sum(dim=0)
    overloop_acc /= total
    model.train()
    return {
        "loop_accuracy": (loop_correct / total).detach().cpu().tolist(),
        "final_accuracy": float((final_correct / total).detach().cpu()),
        "portable_one_step_accuracy": (portable_correct / total).detach().cpu().tolist(),
        "overloop_accuracy": overloop_acc.detach().cpu().tolist(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: Path,
    *,
    model: LoopedGraphPathTransformer,
    optimizer: torch.optim.Optimizer,
    cfg: StepwiseGraphPathConfig,
    mode: str,
    seed: int,
    step: int,
    total_steps: int,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "mode": mode,
            "seed": seed,
            "step": step,
            "total_steps": total_steps,
            "history": history,
            "parameter_count": count_parameters(model),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: LoopedGraphPathTransformer,
    optimizer: torch.optim.Optimizer,
    expected_cfg: StepwiseGraphPathConfig,
    expected_seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_config = asdict(expected_cfg)
    if payload.get("config") != expected_config:
        raise ValueError(
            f"config mismatch: checkpoint={payload.get('config')} expected={expected_config}"
        )
    if payload.get("seed") != expected_seed:
        raise ValueError(
            f"seed mismatch: checkpoint={payload.get('seed')} expected={expected_seed}"
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng_state"])
    return int(payload["step"]), list(payload.get("history", []))


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one(
    mode: str,
    cfg: StepwiseGraphPathConfig,
    args: argparse.Namespace,
    *,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    default_run_name = (
        f"{mode}_N{cfg.node_count}_D{cfg.max_depth}_d{cfg.d_model}"
        f"_L{cfg.max_loops}_seed{seed}"
    )
    run_dir = out_dir / (args.run_name or default_run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = LoopedGraphPathTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    start_step = 0
    if args.resume is not None:
        start_step, history = load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_cfg=cfg,
            expected_seed=seed,
        )
    stop_step = args.steps if args.stop_step is None else args.stop_step
    if not 0 <= start_step < stop_step <= args.steps:
        raise ValueError("require 0 <= start_step < stop_step <= steps")
    started = time.time()
    best = max((float(row["final_accuracy"]) for row in history), default=-1.0)
    best_step = max(
        (int(row["step"]) for row in history if float(row["final_accuracy"]) == best),
        default=0,
    )
    for step in range(start_step + 1, stop_step + 1):
        lr = cosine_lr(step - 1, args.steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tokens, targets, successors, _ = make_stepwise_batch(cfg, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = loss_for_mode(
            mode, model, cfg, tokens, targets, successors,
            portable_weight=args.portable_weight,
            transition_weight=args.transition_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, cfg, device=device, batch_size=args.eval_batch_size,
                batches=args.eval_batches, overloop=args.overloop,
            )
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "first_ce": float(pieces["first_ce"].cpu()),
                "final_ce": float(pieces["final_ce"].cpu()),
                "transition_ce": float(pieces["transition_ce"].cpu()),
                "portable_ce": float(pieces["portable_ce"].cpu()),
                "elapsed_sec": time.time() - started,
                "loop_accuracy": json.dumps(metrics["loop_accuracy"]),
                "final_accuracy": metrics["final_accuracy"],
                "portable_one_step_accuracy": json.dumps(metrics["portable_one_step_accuracy"]),
                "overloop_accuracy": json.dumps(metrics["overloop_accuracy"]),
            }
            history.append(row)
            write_csv(run_dir / "history.csv", history)
            score = metrics["final_accuracy"]
            if score > best:
                best, best_step = score, step
                torch.save({
                    "model": model.state_dict(), "config": asdict(cfg), "step": step,
                    "loss_mode": mode, "parameter_count": count_parameters(model),
                }, run_dir / "best.pt")
            if step == 1 or step % args.print_every == 0 or step == args.steps:
                print(
                    f"[{mode} seed={seed} step={step:05d}] loss={row['loss']:.4f} "
                    f"final={row['final_accuracy']:.3f} "
                    f"loops={','.join(f'{x:.2f}' for x in metrics['loop_accuracy'])} "
                    f"portable={','.join(f'{x:.2f}' for x in metrics['portable_one_step_accuracy'])}",
                    flush=True,
                )
        checkpoint_due = (
            args.checkpoint_every > 0 and step % args.checkpoint_every == 0
        ) or step == stop_step
        if checkpoint_due:
            save_training_checkpoint(
                run_dir / f"checkpoint_step_{step:07d}.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                mode=mode,
                seed=seed,
                step=step,
                total_steps=args.steps,
                history=history,
            )
    final_metrics = evaluate(
        model, cfg, device=device, batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 4), overloop=args.overloop,
    )
    torch.save({
        "model": model.state_dict(), "config": asdict(cfg), "step": stop_step,
        "loss_mode": mode, "parameter_count": count_parameters(model),
    }, run_dir / "final.pt")
    summary = {
        "mode": mode, "seed": seed, "steps": stop_step,
        "total_steps": args.steps,
        "start_step": start_step,
        "parameter_count": count_parameters(model), "best_step": best_step,
        "final_metrics": final_metrics, "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_summary(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    modes = sorted({row["mode"] for row in summaries})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode in modes:
        rows = [row for row in summaries if row["mode"] == mode]
        loops = np.asarray([row["final_metrics"]["overloop_accuracy"] for row in rows])
        portable = np.asarray([row["final_metrics"]["portable_one_step_accuracy"] for row in rows])
        axes[0].plot(np.arange(1, loops.shape[1] + 1), loops.mean(0), marker="o", label=mode)
        axes[0].fill_between(
            np.arange(1, loops.shape[1] + 1), loops.mean(0) - loops.std(0),
            loops.mean(0) + loops.std(0), alpha=0.12,
        )
        axes[1].plot(np.arange(1, portable.shape[1] + 1), portable.mean(0), marker="o", label=mode)
    axes[0].set_title("Overloop accuracy: target f^t")
    axes[0].set_xlabel("effective loop")
    axes[0].set_ylabel("accuracy")
    axes[1].set_title("Portable one-step accuracy")
    axes[1].set_xlabel("donor state position")
    axes[1].set_ylabel("accuracy")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "portable_component_comparison.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare endpoint, transition, and portable-component objectives.")
    parser.add_argument("--modes", nargs="+", default=["final", "transition", "portable"], choices=["final", "transition", "portable"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--loops", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--overloop", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--portable-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("results/p4_portable_component_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cfg = StepwiseGraphPathConfig(
        node_count=args.node_count, max_depth=args.max_depth, d_model=args.d_model,
        n_heads=args.n_heads, d_mlp=args.d_mlp, n_layers=args.n_layers,
        max_loops=args.loops,
    )
    print(f"device={device} cfg={cfg} modes={args.modes} seeds={args.seeds}", flush=True)
    summaries = [
        train_one(mode, cfg, args, seed=seed, device=device, out_dir=args.out_dir)
        for seed in args.seeds
        for mode in args.modes
    ]
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    plot_summary(summaries, args.out_dir)


if __name__ == "__main__":
    main()
