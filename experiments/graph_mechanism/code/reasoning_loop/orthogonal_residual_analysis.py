from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from reasoning_loop.program_path_loop import (
    LoopedProgramPathTransformer,
    ProgramPathConfig,
    canonical_model_state_dict,
    make_program_path_batch,
)


def _grouped(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x.float().reshape(*x.shape[:-1], groups, x.shape[-1] // groups)


def _parallel_component(
    hidden: torch.Tensor,
    update: torch.Tensor,
    *,
    groups: int,
    eps: float,
) -> torch.Tensor:
    grouped_hidden = _grouped(hidden, groups)
    grouped_update = _grouped(update, groups)
    coefficient = (grouped_hidden * grouped_update).sum(dim=-1, keepdim=True)
    coefficient = coefficient / (
        grouped_hidden.square().sum(dim=-1, keepdim=True) + eps
    )
    return (coefficient * grouped_hidden).reshape_as(update)


def _mean_rows(rows_by_index: dict[int, list[dict[str, float]]]) -> list[dict[str, float]]:
    averaged: list[dict[str, float]] = []
    for effective_index in sorted(rows_by_index):
        rows = rows_by_index[effective_index]
        keys = rows[0].keys()
        averaged.append(
            {
                key: float(sum(row[key] for row in rows) / len(rows))
                for key in keys
            }
        )
    return averaged


def _write_csv(path: Path, rows: list[dict[str, float]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _plot_metric(
    rows: list[dict[str, float]],
    *,
    keys: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    x = [row["effective_block"] for row in rows]
    plt.figure(figsize=(9, 5))
    for key, label in zip(keys, labels, strict=True):
        plt.plot(x, [row[key] for row in rows], marker="o", label=label)
    plt.xlabel("effective block depth")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(keys) > 1:
        plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyze_checkpoint(
    *,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    max_eval_loop: int,
    seed: int,
) -> dict[str, Any]:
    if batch_size < 1 or batches < 1 or max_eval_loop < 1:
        raise ValueError("batch_size, batches, and max_eval_loop must be positive")
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = ProgramPathConfig(**payload["config"])
    if cfg.outer_norm_groups:
        raise ValueError("orthogonal residual analysis expects outer_norm_groups=0")
    model = LoopedProgramPathTransformer(cfg).to(device)
    model.load_state_dict(canonical_model_state_dict(payload["model"]), strict=True)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    block_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    projector_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles: list[Any] = []

    def block_hook(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        block_calls.append((args[0], output))

    def projector_pre_hook(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        projector_calls.append((args[0], args[1]))

    for block in model.blocks:
        handles.append(block.register_forward_hook(block_hook))
        if block.residual_projector is not None:
            handles.append(
                block.residual_projector.register_forward_pre_hook(projector_pre_hook)
            )

    rows_by_index: dict[int, list[dict[str, float]]] = defaultdict(list)
    loss_values: list[float] = []
    accuracy_sums = torch.zeros(max_eval_loop, device=device)
    accuracy_count = 0
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    try:
        for _ in range(batches):
            block_calls.clear()
            projector_calls.clear()
            model.zero_grad(set_to_none=True)
            tokens, _, _, targets_by_depth = make_program_path_batch(
                cfg, batch_size, device
            )
            target = targets_by_depth[:, cfg.max_depth - 1]
            query_start = 1 + 4 * cfg.relation_count * cfg.node_count
            tokens[:, query_start + 2] = cfg.depth_token_base + cfg.max_depth - 1

            logits_by_loop = model.forward_all(tokens, max_loops=max_eval_loop)[
                "logits_by_loop"
            ]
            loss = F.cross_entropy(logits_by_loop[:, -1, :], target)
            loss.backward()
            loss_values.append(float(loss.detach().cpu()))
            accuracy_sums += logits_by_loop.argmax(dim=-1).eq(target[:, None]).float().sum(0)
            accuracy_count += batch_size

            expected_calls = max_eval_loop * cfg.n_layers
            if len(block_calls) != expected_calls:
                raise RuntimeError(
                    f"expected {expected_calls} block calls, observed {len(block_calls)}"
                )
            if cfg.residual_projection_groups and len(projector_calls) != expected_calls:
                raise RuntimeError(
                    f"expected {expected_calls} projector calls, observed {len(projector_calls)}"
                )

            for call_index, (block_input, block_output) in enumerate(block_calls):
                projected_update = block_output - block_input
                if cfg.residual_projection_groups:
                    projection_hidden, raw_update = projector_calls[call_index]
                    groups = cfg.residual_projection_groups
                    parallel = _parallel_component(
                        projection_hidden,
                        raw_update,
                        groups=groups,
                        eps=cfg.rms_norm_eps,
                    )
                    update_shrink_ratio = float(
                        (
                            projected_update.float().norm()
                            / raw_update.float().norm().clamp_min(cfg.rms_norm_eps)
                        )
                        .detach()
                        .cpu()
                    )
                else:
                    groups = 1
                    raw_update = projected_update
                    parallel = _parallel_component(
                        block_input,
                        raw_update,
                        groups=groups,
                        eps=cfg.rms_norm_eps,
                    )
                    update_shrink_ratio = 1.0

                if block_output.grad is None:
                    raise RuntimeError("missing retained gradient for a block output")
                effective_block = call_index + 1
                rows_by_index[effective_block].append(
                    {
                        "effective_block": float(effective_block),
                        "loop": float(call_index // cfg.n_layers + 1),
                        "block_in_loop": float(call_index % cfg.n_layers + 1),
                        "gradient_norm": float(
                            block_output.grad.float().norm(dim=-1).mean().detach().cpu()
                        ),
                        "parallel_energy_fraction": float(
                            (
                                parallel.float().square().sum()
                                / raw_update.float().square().sum().clamp_min(
                                    cfg.rms_norm_eps
                                )
                            )
                            .detach()
                            .cpu()
                        ),
                        "update_shrink_ratio": update_shrink_ratio,
                        "input_norm": float(
                            block_input.float().norm(dim=-1).mean().detach().cpu()
                        ),
                        "output_norm": float(
                            block_output.float().norm(dim=-1).mean().detach().cpu()
                        ),
                        "update_norm": float(
                            projected_update.float().norm(dim=-1).mean().detach().cpu()
                        ),
                        "update_state_ratio": float(
                            (
                                projected_update.float().norm(dim=-1)
                                / block_input.float().norm(dim=-1).clamp_min(cfg.rms_norm_eps)
                            )
                            .mean()
                            .detach()
                            .cpu()
                        ),
                        "input_output_cosine": float(
                            F.cosine_similarity(
                                block_input.float(), block_output.float(), dim=-1
                            )
                            .mean()
                            .detach()
                            .cpu()
                        ),
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    rows = _mean_rows(rows_by_index)
    gradient_fields = [
        "effective_block",
        "loop",
        "block_in_loop",
        "gradient_norm",
    ]
    dynamics_fields = [
        "effective_block",
        "loop",
        "block_in_loop",
        "parallel_energy_fraction",
        "update_shrink_ratio",
        "input_norm",
        "output_norm",
        "update_norm",
        "update_state_ratio",
        "input_output_cosine",
    ]
    _write_csv(out_dir / "gradient_flow.csv", rows, gradient_fields)
    _write_csv(out_dir / "projection_dynamics.csv", rows, dynamics_fields)
    _plot_metric(
        rows,
        keys=["gradient_norm"],
        labels=["gradient norm"],
        ylabel="mean token gradient norm",
        title="Final-loss gradient by effective block depth",
        path=out_dir / "gradient_norm_by_effective_block.png",
    )
    _plot_metric(
        rows,
        keys=["parallel_energy_fraction", "update_shrink_ratio"],
        labels=["parallel energy fraction", "projected/raw update norm"],
        ylabel="fraction / ratio",
        title="Orthogonal projection diagnostics",
        path=out_dir / "projection_fraction_by_effective_block.png",
    )
    _plot_metric(
        rows,
        keys=["input_norm", "output_norm", "update_norm"],
        labels=["input state", "output state", "projected update"],
        ylabel="mean token norm",
        title="State and update norms",
        path=out_dir / "state_norm_by_effective_block.png",
    )
    _plot_metric(
        rows,
        keys=["input_output_cosine"],
        labels=["input-output cosine"],
        ylabel="cosine similarity",
        title="Adjacent effective-block state cosine",
        path=out_dir / "state_cosine_by_effective_block.png",
    )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "residual_projection_groups": cfg.residual_projection_groups,
        "max_eval_loop": max_eval_loop,
        "depth": cfg.max_depth,
        "examples": accuracy_count,
        "mean_final_loss": float(sum(loss_values) / len(loss_values)),
        "target_accuracy_by_loop": [
            float(value) for value in (accuracy_sums / accuracy_count).detach().cpu()
        ],
        "effective_block_metrics": rows,
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze gradients and blockwise orthogonal residual dynamics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--max-eval-loop", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    analyze_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        batches=args.batches,
        max_eval_loop=args.max_eval_loop,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
