from __future__ import annotations

import argparse
import csv
import json
import random
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

from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    make_graph_path_batch,
    pick_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe looped graph-path checkpoints for intermediate path variables."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--probe-train-samples", type=int, default=32768)
    parser.add_argument("--probe-eval-samples", type=int, default=16384)
    parser.add_argument("--probe-steps", type=int, default=800)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> tuple[LoopedGraphPathTransformer, GraphPathConfig, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = GraphPathConfig(**ckpt["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def collect_eval_tensors(
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
) -> dict[str, torch.Tensor]:
    logits_chunks: list[torch.Tensor] = []
    state_chunks: list[torch.Tensor] = []
    depth_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    start_chunks: list[torch.Tensor] = []
    for _ in range(batches):
        tokens, target, depth, targets_by_depth = make_graph_path_batch(cfg, batch_size, device)
        out = model.forward_all(tokens, max_loops=cfg.max_loops, return_states=True)
        logits_chunks.append(out["logits_by_loop"].detach().cpu())
        state_chunks.append(out["states_by_loop"].detach().cpu())
        depth_chunks.append(depth.detach().cpu())
        target_chunks.append(targets_by_depth.detach().cpu())
        start_chunks.append(tokens[:, -3].detach().cpu())
    return {
        "logits": torch.cat(logits_chunks, dim=0),
        "states": torch.cat(state_chunks, dim=0),
        "query_depth": torch.cat(depth_chunks, dim=0),
        "targets_by_depth": torch.cat(target_chunks, dim=0),
        "start": torch.cat(start_chunks, dim=0),
    }


def lm_head_step_tables(tensors: dict[str, torch.Tensor], cfg: GraphPathConfig) -> dict[str, Any]:
    logits = tensors["logits"]
    pred = logits.argmax(dim=-1)
    query_depth = tensors["query_depth"]
    targets_by_depth = tensors["targets_by_depth"]
    start = tensors["start"]

    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    penult_rows: list[dict[str, Any]] = []

    for query_d in range(1, cfg.max_depth + 1):
        mask = query_depth.eq(query_d)
        if not mask.any():
            continue
        for loop_idx in range(cfg.max_loops):
            step_accs = []
            for step_k in range(1, cfg.max_depth + 1):
                acc = pred[mask, loop_idx].eq(targets_by_depth[mask, step_k - 1]).float().mean().item()
                step_accs.append(acc)
                rows.append(
                    {
                        "query_depth": query_d,
                        "loop": loop_idx + 1,
                        "step_label": step_k,
                        "acc": acc,
                    }
                )
            target_acc = step_accs[query_d - 1]
            if query_d == 1:
                penult_label = 0
                penult_acc = pred[mask, loop_idx].eq(start[mask]).float().mean().item()
            else:
                penult_label = query_d - 1
                penult_acc = step_accs[query_d - 2]
            best_idx = int(np.argmax(step_accs)) + 1
            best_rows.append(
                {
                    "query_depth": query_d,
                    "loop": loop_idx + 1,
                    "best_step_label": best_idx,
                    "best_step_acc": step_accs[best_idx - 1],
                    "target_acc": target_acc,
                    "penultimate_label": penult_label,
                    "penultimate_acc": penult_acc,
                }
            )
            target_rows.append({"query_depth": query_d, "loop": loop_idx + 1, "target_acc": target_acc})
            penult_rows.append(
                {
                    "query_depth": query_d,
                    "loop": loop_idx + 1,
                    "penultimate_label": penult_label,
                    "penultimate_acc": penult_acc,
                }
            )
    return {
        "all_step_rows": rows,
        "best_rows": best_rows,
        "target_rows": target_rows,
        "penultimate_rows": penult_rows,
    }


class MultiStepProbe(nn.Module):
    def __init__(self, d_model: int, max_depth: int, node_count: int) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.node_count = node_count
        self.linear = nn.Linear(d_model, max_depth * node_count)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.linear(states)
        return logits.view(states.shape[0], self.max_depth, self.node_count)


@torch.no_grad()
def collect_states_for_probe(
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    *,
    device: torch.device,
    samples: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    query_depths: list[torch.Tensor] = []
    remaining = samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        tokens, _, depth, targets_by_depth = make_graph_path_batch(cfg, current_batch, device)
        out = model.forward_all(tokens, max_loops=cfg.max_loops, return_states=True)
        states.append(out["states_by_loop"].detach().cpu())
        targets.append(targets_by_depth.detach().cpu())
        query_depths.append(depth.detach().cpu())
        remaining -= current_batch
    return {
        "states": torch.cat(states, dim=0),
        "targets_by_depth": torch.cat(targets, dim=0),
        "query_depth": torch.cat(query_depths, dim=0),
    }


def train_probe_for_loop(
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    eval_states: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    cfg: GraphPathConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[float], MultiStepProbe]:
    probe = MultiStepProbe(cfg.d_model, cfg.max_depth, cfg.node_count).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    train_states = train_states.to(device)
    train_targets = train_targets.to(device)
    n = train_states.shape[0]
    for _ in range(args.probe_steps):
        idx = torch.randint(0, n, (min(args.probe_batch_size, n),), device=device)
        logits = probe(train_states[idx])
        loss = 0.0
        for step_k in range(cfg.max_depth):
            loss = loss + F.cross_entropy(logits[:, step_k, :], train_targets[idx, step_k])
        loss = loss / cfg.max_depth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(eval_states.to(device))
        pred = logits.argmax(dim=-1).cpu()
    accs = [
        float(pred[:, step_k].eq(eval_targets[:, step_k]).float().mean().item())
        for step_k in range(cfg.max_depth)
    ]
    return accs, probe


def run_linear_probes(
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train = collect_states_for_probe(
        model,
        cfg,
        device=device,
        samples=args.probe_train_samples,
        batch_size=args.eval_batch_size,
    )
    eval_data = collect_states_for_probe(
        model,
        cfg,
        device=device,
        samples=args.probe_eval_samples,
        batch_size=args.eval_batch_size,
    )
    rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for loop_idx in range(cfg.max_loops):
        accs, probe = train_probe_for_loop(
            train["states"][:, loop_idx, :],
            train["targets_by_depth"],
            eval_data["states"][:, loop_idx, :],
            eval_data["targets_by_depth"],
            cfg=cfg,
            args=args,
            device=device,
        )
        with torch.no_grad():
            pred = probe(eval_data["states"][:, loop_idx, :].to(device)).argmax(dim=-1).cpu()
        for step_k, acc in enumerate(accs, start=1):
            rows.append({"loop": loop_idx + 1, "step_label": step_k, "probe_acc": acc})
        for query_d in range(1, cfg.max_depth + 1):
            mask = eval_data["query_depth"].eq(query_d)
            for step_k in range(1, cfg.max_depth + 1):
                acc = pred[mask, step_k - 1].eq(eval_data["targets_by_depth"][mask, step_k - 1]).float().mean().item()
                query_rows.append(
                    {
                        "query_depth": query_d,
                        "loop": loop_idx + 1,
                        "step_label": step_k,
                        "probe_acc": float(acc),
                    }
                )
    return {"probe_rows": rows, "probe_query_rows": query_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def heatmap_from_rows(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    row_key: str,
    col_key: str,
    title: str,
    path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    row_vals = sorted({int(r[row_key]) for r in rows})
    col_vals = sorted({int(r[col_key]) for r in rows})
    arr = np.full((len(row_vals), len(col_vals)), np.nan, dtype=np.float32)
    row_index = {v: i for i, v in enumerate(row_vals)}
    col_index = {v: i for i, v in enumerate(col_vals)}
    for row in rows:
        arr[row_index[int(row[row_key])], col_index[int(row[col_key])]] = float(row[value_key])
    plt.figure(figsize=(1.1 * len(col_vals) + 3.0, 0.65 * len(row_vals) + 2.8))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    plt.colorbar(im, label=value_key)
    plt.xticks(range(len(col_vals)), [str(v) for v in col_vals])
    plt.yticks(range(len(row_vals)), [str(v) for v in row_vals])
    plt.xlabel(col_key)
    plt.ylabel(row_key)
    plt.title(title)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            if not np.isnan(arr[y, x]):
                plt.text(x, y, f"{arr[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_query_depth_panel(rows: list[dict[str, Any]], *, value_key: str, path: Path, title: str) -> None:
    query_depths = sorted({int(r["query_depth"]) for r in rows})
    step_labels = sorted({int(r.get("step_label", r.get("penultimate_label", 0))) for r in rows if int(r.get("step_label", r.get("penultimate_label", 0))) > 0})
    loops = sorted({int(r["loop"]) for r in rows})
    fig, axes = plt.subplots(1, len(query_depths), figsize=(4.2 * len(query_depths), 3.8), squeeze=False)
    for ax, query_d in zip(axes[0], query_depths):
        subset = [r for r in rows if int(r["query_depth"]) == query_d]
        row_values = step_labels
        arr = np.full((len(row_values), len(loops)), np.nan, dtype=np.float32)
        for row in subset:
            step = int(row.get("step_label", row.get("penultimate_label", 0)))
            if step <= 0:
                continue
            arr[row_values.index(step), loops.index(int(row["loop"]))] = float(row[value_key])
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(f"query depth {query_d}")
        ax.set_xlabel("loop")
        ax.set_xticks(range(len(loops)), [str(v) for v in loops])
        ax.set_yticks(range(len(row_values)), [f"f^{v}(s)" for v in row_values])
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if not np.isnan(arr[y, x]):
                    ax.text(x, y, f"{arr[y, x]:.2f}", ha="center", va="center", color="white", fontsize=7)
    axes[0, 0].set_ylabel("decoded step")
    fig.colorbar(im, ax=axes.ravel().tolist(), label=value_key, shrink=0.8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    model, cfg, ckpt = load_model(args.checkpoint, device)

    eval_tensors = collect_eval_tensors(
        model,
        cfg,
        device=device,
        batch_size=args.eval_batch_size,
        batches=args.eval_batches,
    )
    lm_tables = lm_head_step_tables(eval_tensors, cfg)
    probe_tables = run_linear_probes(model, cfg, device=device, args=args)

    write_csv(args.out_dir / "lm_head_step_acc_by_query_depth.csv", lm_tables["all_step_rows"])
    write_csv(args.out_dir / "lm_head_best_step_by_query_loop.csv", lm_tables["best_rows"])
    write_csv(args.out_dir / "lm_head_target_acc.csv", lm_tables["target_rows"])
    write_csv(args.out_dir / "lm_head_penultimate_acc.csv", lm_tables["penultimate_rows"])
    write_csv(args.out_dir / "linear_probe_step_acc.csv", probe_tables["probe_rows"])
    write_csv(args.out_dir / "linear_probe_step_acc_by_query_depth.csv", probe_tables["probe_query_rows"])

    heatmap_from_rows(
        lm_tables["target_rows"],
        value_key="target_acc",
        row_key="query_depth",
        col_key="loop",
        title="LM-head accuracy for requested final target f^d(s)",
        path=args.out_dir / "lm_head_target_acc_heatmap.png",
    )
    heatmap_from_rows(
        lm_tables["penultimate_rows"],
        value_key="penultimate_acc",
        row_key="query_depth",
        col_key="loop",
        title="LM-head accuracy for penultimate state f^(d-1)(s)",
        path=args.out_dir / "lm_head_penultimate_acc_heatmap.png",
    )
    heatmap_from_rows(
        probe_tables["probe_rows"],
        value_key="probe_acc",
        row_key="step_label",
        col_key="loop",
        title="Linear probe accuracy: f^k(s) decoded from answer-position state",
        path=args.out_dir / "linear_probe_step_acc_heatmap.png",
    )
    plot_query_depth_panel(
        lm_tables["all_step_rows"],
        value_key="acc",
        path=args.out_dir / "lm_head_all_steps_by_query_depth.png",
        title="LM head: which path step is directly predicted at each loop?",
    )
    plot_query_depth_panel(
        probe_tables["probe_query_rows"],
        value_key="probe_acc",
        path=args.out_dir / "linear_probe_all_steps_by_query_depth.png",
        title="Linear probes: which path variables are present in h_loop?",
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_metrics": ckpt.get("metrics"),
        "config": asdict(cfg),
        "lm_head": lm_tables,
        "linear_probe": probe_tables,
    }
    (args.out_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"config": asdict(cfg), "checkpoint_step": ckpt.get("step")}, indent=2), flush=True)
    print("wrote", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
