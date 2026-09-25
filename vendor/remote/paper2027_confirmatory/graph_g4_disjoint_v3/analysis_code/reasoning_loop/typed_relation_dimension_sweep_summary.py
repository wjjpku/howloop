from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_models(
    trajectories: list[dict[str, str]],
    geometry: list[dict[str, str]],
    *,
    accuracy_gate: float = 0.99,
) -> list[dict[str, Any]]:
    by_checkpoint: dict[str, list[dict[str, str]]] = defaultdict(list)
    geometry_by_checkpoint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trajectories:
        if row["condition"] == "trained_cell":
            by_checkpoint[row["checkpoint"]].append(row)
    for row in geometry:
        geometry_by_checkpoint[row["checkpoint"]].append(row)
    summary: list[dict[str, Any]] = []
    for checkpoint, rows in by_checkpoint.items():
        by_loop = {int(row["loop"]): row for row in rows}
        if 2 not in by_loop or 4 not in by_loop:
            continue
        d2_accuracy = float(by_loop[2]["accuracy"])
        if d2_accuracy < accuracy_gate:
            continue
        geometry_loop = {
            int(row["loop"]): row for row in geometry_by_checkpoint[checkpoint]
        }
        if 3 not in geometry_loop or 4 not in geometry_loop:
            continue
        node_count = int(rows[0]["node_count"])
        d_model = int(rows[0]["d_model"])
        checkpoint_path = Path(checkpoint)
        payload = (
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint_path.exists()
            else {}
        )
        checkpoint_history = payload.get("history", [])
        bridge_accuracy = (
            float(checkpoint_history[-1]["loop1_first_relation_accuracy"])
            if checkpoint_history
            else float("nan")
        )
        risk_terms = []
        for loop in (3, 4):
            row = geometry_loop[loop]
            boundary = max(float(row["local_boundary_distance"]), 1e-12)
            risk_terms.append(float(row["absolute_sensitive_step"]) / boundary)
        summary.append(
            {
                "checkpoint": checkpoint,
                "model_seed": int(rows[0]["model_seed"]),
                "node_count": node_count,
                "d_model": d_model,
                "rank_fraction": (node_count - 1) / d_model,
                "d2_accuracy": d2_accuracy,
                "d3_accuracy": float(by_loop[3]["accuracy"]),
                "d4_accuracy": float(by_loop[4]["accuracy"]),
                "d8_accuracy": (
                    float(by_loop[8]["accuracy"]) if 8 in by_loop else float("nan")
                ),
                "d2_margin": float(by_loop[2]["margin"]),
                "d1_bridge_accuracy": bridge_accuracy,
                "d4_margin": float(by_loop[4]["margin"]),
                "d2_to_d4_drop": d2_accuracy - float(by_loop[4]["accuracy"]),
                "trained_sensitive_fraction_d3": float(
                    geometry_loop[3]["sensitive_energy_fraction"]
                ),
                "random_sensitive_fraction_d3": float(
                    geometry_loop[3]["random_sensitive_energy_fraction"]
                ),
                "two_step_local_risk": sum(risk_terms),
            }
        )
    return summary


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "PingFang SC",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    rank = np.asarray([row["rank_fraction"] for row in rows])
    random_energy = np.asarray(
        [row["random_sensitive_fraction_d3"] for row in rows]
    )
    d4_drop = np.asarray([row["d2_to_d4_drop"] for row in rows])
    risk = np.asarray([row["two_step_local_risk"] for row in rows])
    bridge = np.asarray([row["d1_bridge_accuracy"] for row in rows])
    axes[0].scatter(rank, random_energy, color="#1565C0", s=52, alpha=0.85)
    limit = max(float(rank.max()), float(random_energy.max())) * 1.08
    axes[0].plot([0, limit], [0, limit], linestyle="--", color="gray")
    axes[0].set(
        xlabel="理论敏感维度比例 (V-1)/d",
        ylabel="随机 residual 的实测敏感能量",
        xlim=(0, limit),
        ylim=(0, limit),
        title="随机方向服从维度比例预测",
    )
    random_mae = float(np.mean(np.abs(random_energy - rank)))
    axes[0].text(
        0.04,
        0.92,
        f"MAE = {random_mae:.4f}",
        transform=axes[0].transAxes,
        fontsize=9,
    )

    scatter = axes[1].scatter(
        rank,
        d4_drop,
        c=bridge,
        cmap="coolwarm",
        s=72,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[1].set(
        xlabel="(V-1)/d",
        ylabel="D2→D4 accuracy drop",
        title="维度比例不是唯一变量：D1 circuit 也重要",
    )
    rank_correlation = float(np.corrcoef(rank, d4_drop)[0, 1])
    axes[1].text(
        0.04,
        0.92,
        f"Pearson r = {rank_correlation:.2f}",
        transform=axes[1].transAxes,
        fontsize=9,
    )
    fig.colorbar(scatter, ax=axes[1], label="D1 bridge accuracy")

    axes[2].scatter(risk, d4_drop, color="#C62828", s=58, alpha=0.85)
    top_indices = np.argsort(d4_drop)[-3:]
    for index in top_indices:
        row = rows[int(index)]
        label = f"V={row['node_count']}, d={row['d_model']}, s={row['model_seed']}"
        axes[2].annotate(
            label,
            (risk[index], d4_drop[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[2].set(
        xlabel="Σ |敏感步长| / 局部边界距离",
        ylabel="D2→D4 accuracy drop",
        title="统一风险量：步长相对于 margin buffer",
    )
    risk_correlation = float(np.corrcoef(risk, d4_drop)[0, 1])
    axes[2].text(
        0.04,
        0.92,
        f"Pearson r = {risk_correlation:.2f}",
        transform=axes[2].transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    random_error = [
        abs(row["random_sensitive_fraction_d3"] - row["rank_fraction"])
        for row in rows
    ]
    result: dict[str, Any] = {
        "model_count": len(rows),
        "config_count": len(
            {(row["node_count"], row["d_model"]) for row in rows}
        ),
        "mean_absolute_random_projection_error": statistics.mean(random_error),
    }
    if len(rows) >= 3:
        result["correlation_rank_fraction_vs_d4_drop"] = float(
            np.corrcoef(
                [row["rank_fraction"] for row in rows],
                [row["d2_to_d4_drop"] for row in rows],
            )[0, 1]
        )
        result["correlation_local_risk_vs_d4_drop"] = float(
            np.corrcoef(
                [row["two_step_local_risk"] for row in rows],
                [row["d2_to_d4_drop"] for row in rows],
            )[0, 1]
        )
        result["correlation_d1_bridge_vs_d4_drop"] = float(
            np.corrcoef(
                [row["d1_bridge_accuracy"] for row in rows],
                [row["d2_to_d4_drop"] for row in rows],
            )[0, 1]
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--accuracy-gate", type=float, default=0.99)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = summarize_models(
        _read(args.trajectories),
        _read(args.geometry),
        accuracy_gate=args.accuracy_gate,
    )
    if not rows:
        raise RuntimeError("no converged checkpoints passed the accuracy gate")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "dimension_sweep.csv", rows)
    plot(rows, args.out_dir / "dimension_sweep.png")
    summary = aggregate(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
