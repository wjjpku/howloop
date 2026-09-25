from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from reasoning_loop.graph_path_loop import LoopedGraphPathTransformer, pick_device, set_seed
from reasoning_loop.graph_path_stepwise import StepwiseGraphPathConfig, make_stepwise_batch


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"run spec must be NAME=/path/to/checkpoint.pt, got {text!r}")
    name, path = text.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal state-transplant diagnostic for no-depth stepwise graph-path models. "
            "Patch donor answer-token residual state after loop t into a receiver graph "
            "context, then test whether subsequent loops follow the receiver graph from "
            "the donor current node."
        )
    )
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/final.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-loops", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--max-delta", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026070917)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def apply_shared_stack(model: LoopedGraphPathTransformer, x: torch.Tensor) -> torch.Tensor:
    for block in model.blocks:
        x = block(x)
    return x


def logits_from_raw_state(model: LoopedGraphPathTransformer, x: torch.Tensor) -> torch.Tensor:
    final_state = model.ln_final(x[:, -1, :])
    return model.unembed(final_state)[:, : model.cfg.node_count]


def run_to_loop(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    loops: int,
) -> torch.Tensor:
    x = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    for _ in range(loops):
        x = apply_shared_stack(model, x)
    return x


def targets_after_receiver_successors(
    *,
    receiver_successors: torch.Tensor,
    start_node: torch.Tensor,
    max_delta: int,
) -> torch.Tensor:
    targets = [start_node]
    current = start_node
    for _ in range(max_delta):
        current = receiver_successors.gather(1, current[:, None]).squeeze(1)
        targets.append(current)
    return torch.stack(targets, dim=1)


def update_metric_sums(
    *,
    logits: torch.Tensor,
    target: torch.Tensor,
    correct_sum: torch.Tensor,
    prob_sum: torch.Tensor,
    row: int,
    col: int,
) -> None:
    probs = logits.softmax(dim=-1)
    pred = logits.argmax(dim=-1)
    correct_sum[row, col] += pred.eq(target).float().sum()
    prob_sum[row, col] += probs.gather(1, target[:, None]).squeeze(1).sum()


@torch.no_grad()
def analyze_run(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    patch_loops: list[int],
    max_delta: int,
    batch_size: int,
    batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = StepwiseGraphPathConfig(**ckpt["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    patch_loops = sorted(patch_loops)
    if min(patch_loops) < 1:
        raise ValueError("--patch-loops are 1-indexed and must be >= 1")

    path_positions = max(max(patch_loops) + max_delta, cfg.max_depth)
    n_rows = len(patch_loops)
    n_cols = max_delta + 1
    transplant_correct = torch.zeros(n_rows, n_cols, device=device)
    transplant_prob = torch.zeros(n_rows, n_cols, device=device)
    patched_stay_receiver_correct = torch.zeros(n_rows, n_cols, device=device)
    patched_stay_receiver_prob = torch.zeros(n_rows, n_cols, device=device)
    clean_correct = torch.zeros(n_rows, n_cols, device=device)
    clean_prob = torch.zeros(n_rows, n_cols, device=device)
    count = 0

    for _ in range(batches):
        receiver_tokens, receiver_targets, receiver_successors, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=path_positions
        )
        donor_tokens, donor_targets, _, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=path_positions
        )
        count += batch_size

        for row_idx, patch_loop in enumerate(patch_loops):
            donor_x = run_to_loop(model, donor_tokens, loops=patch_loop)
            receiver_x = run_to_loop(model, receiver_tokens, loops=patch_loop)
            patched_x = receiver_x.clone()
            patched_x[:, -1, :] = donor_x[:, -1, :]

            donor_current = donor_targets[:, patch_loop - 1]
            transplant_targets = targets_after_receiver_successors(
                receiver_successors=receiver_successors,
                start_node=donor_current,
                max_delta=max_delta,
            )
            receiver_targets_from_patch_loop = receiver_targets[
                :, patch_loop - 1 : patch_loop + max_delta
            ]

            clean_x = receiver_x
            for delta in range(n_cols):
                patched_logits = logits_from_raw_state(model, patched_x)
                clean_logits = logits_from_raw_state(model, clean_x)
                update_metric_sums(
                    logits=patched_logits,
                    target=transplant_targets[:, delta],
                    correct_sum=transplant_correct,
                    prob_sum=transplant_prob,
                    row=row_idx,
                    col=delta,
                )
                update_metric_sums(
                    logits=patched_logits,
                    target=receiver_targets_from_patch_loop[:, delta],
                    correct_sum=patched_stay_receiver_correct,
                    prob_sum=patched_stay_receiver_prob,
                    row=row_idx,
                    col=delta,
                )
                update_metric_sums(
                    logits=clean_logits,
                    target=receiver_targets_from_patch_loop[:, delta],
                    correct_sum=clean_correct,
                    prob_sum=clean_prob,
                    row=row_idx,
                    col=delta,
                )
                if delta < max_delta:
                    patched_x = apply_shared_stack(model, patched_x)
                    clean_x = apply_shared_stack(model, clean_x)

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return (tensor / count).detach().cpu().numpy()

    transplant_acc = to_numpy(transplant_correct)
    transplant_mean_prob = to_numpy(transplant_prob)
    patched_stay_receiver_acc = to_numpy(patched_stay_receiver_correct)
    patched_stay_receiver_mean_prob = to_numpy(patched_stay_receiver_prob)
    clean_acc = to_numpy(clean_correct)
    clean_mean_prob = to_numpy(clean_prob)

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap(
        transplant_acc,
        patch_loops=patch_loops,
        max_delta=max_delta,
        path=run_dir / "state_transplant_acc_to_transplanted_variable.png",
        title=f"{name}: patched state follows donor current node through receiver graph",
        label="acc to receiver-successor(donor-current)",
    )
    save_heatmap(
        patched_stay_receiver_acc,
        patch_loops=patch_loops,
        max_delta=max_delta,
        path=run_dir / "state_transplant_acc_to_original_receiver_path.png",
        title=f"{name}: patched state still follows original receiver path",
        label="acc to original receiver path",
    )
    save_heatmap(
        clean_acc,
        patch_loops=patch_loops,
        max_delta=max_delta,
        path=run_dir / "clean_continuation_acc.png",
        title=f"{name}: clean receiver continuation sanity check",
        label="clean acc",
    )

    rows: list[dict[str, Any]] = []
    for row_idx, patch_loop in enumerate(patch_loops):
        for delta in range(n_cols):
            rows.append(
                {
                    "model": name,
                    "checkpoint": str(checkpoint),
                    "loss_mode": ckpt.get("loss_mode", ""),
                    "checkpoint_step": ckpt.get("step", ""),
                    "patch_loop": patch_loop,
                    "delta_after_patch": delta,
                    "readout_loop": patch_loop + delta,
                    "examples": count,
                    "transplant_acc": float(transplant_acc[row_idx, delta]),
                    "transplant_mean_prob": float(transplant_mean_prob[row_idx, delta]),
                    "patched_stay_receiver_acc": float(patched_stay_receiver_acc[row_idx, delta]),
                    "patched_stay_receiver_mean_prob": float(
                        patched_stay_receiver_mean_prob[row_idx, delta]
                    ),
                    "clean_receiver_acc": float(clean_acc[row_idx, delta]),
                    "clean_receiver_mean_prob": float(clean_mean_prob[row_idx, delta]),
                }
            )

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "config": asdict(cfg),
        "loss_mode": ckpt.get("loss_mode", ""),
        "examples": count,
        "patch_loops": patch_loops,
        "max_delta": max_delta,
        "transplant_acc": transplant_acc.tolist(),
        "transplant_mean_prob": transplant_mean_prob.tolist(),
        "patched_stay_receiver_acc": patched_stay_receiver_acc.tolist(),
        "clean_receiver_acc": clean_acc.tolist(),
        "mean_transplant_acc_delta0": float(transplant_acc[:, 0].mean()),
        "mean_transplant_acc_delta1_to_3": float(
            transplant_acc[:, 1 : min(max_delta, 3) + 1].mean()
            if max_delta >= 1
            else transplant_acc[:, 0].mean()
        ),
        "mean_clean_acc": float(clean_acc.mean()),
    }
    (run_dir / "state_transplant_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, rows


def save_heatmap(
    values: np.ndarray,
    *,
    patch_loops: list[int],
    max_delta: int,
    path: Path,
    title: str,
    label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(max_delta + 1), [str(i) for i in range(max_delta + 1)])
    ax.set_yticks(range(len(patch_loops)), [str(i) for i in patch_loops])
    ax.set_xlabel("delta after patch")
    ax.set_ylabel("patched after loop t")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_combined_bar(out_dir: Path, summaries: dict[str, Any]) -> None:
    names = list(summaries)
    delta0 = [summaries[name]["mean_transplant_acc_delta0"] for name in names]
    delta13 = [summaries[name]["mean_transplant_acc_delta1_to_3"] for name in names]
    clean = [summaries[name]["mean_clean_acc"] for name in names]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    ax.bar(x - width, delta0, width, label="patched acc delta0")
    ax.bar(x, delta13, width, label="patched acc delta1-3")
    ax.bar(x + width, clean, width, label="clean receiver acc")
    ax.axhline(1 / 8, color="black", linestyle=":", linewidth=1, label="N=8 random")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Answer-state transplant causal abstraction score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_state_transplant_scores.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    run_specs = [parse_run_spec(text) for text in args.run]
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for name, checkpoint in run_specs:
        summary, rows = analyze_run(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            patch_loops=args.patch_loops,
            max_delta=args.max_delta,
            batch_size=args.batch_size,
            batches=args.batches,
        )
        summaries[name] = summary
        all_rows.extend(rows)
        print("done", name, flush=True)
    write_csv(args.out_dir / "state_transplant_rows.csv", all_rows)
    (args.out_dir / "state_transplant_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2), encoding="utf-8"
    )
    save_combined_bar(args.out_dir, summaries)
    print(json.dumps({"models": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
