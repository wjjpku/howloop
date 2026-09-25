from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence


def segments(values: list[float]) -> dict[str, float]:
    def mean(start: int, stop: int) -> float:
        selected = values[start:stop]
        return fmean(selected) if selected else float("nan")

    return {
        "auc_1_24": mean(0, 24),
        "auc_25_48": mean(24, 48),
        "auc_49_64": mean(48, 64),
        "auc_65_96": mean(64, 96),
        "auc_97_128": mean(96, 128),
        "accuracy_loop64": values[63] if len(values) >= 64 else float("nan"),
        "accuracy_loop128": values[127] if len(values) >= 128 else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def controller_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    curves_by_run: dict[str, dict[str, list[float]]] = {}
    for summary_path in sorted((root / "controllers").glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        run = summary_path.parent.name
        curves = payload["random_graph_evaluation"]["curves"]
        task_curves = {
            label: [float(value) for value in item["accuracy_by_cycle"]]
            for label, item in curves.items()
            if label.startswith("task_")
        }
        curves_by_run[run] = task_curves
        for label, values in task_curves.items():
            variant = payload["variants"][label]
            rows.append(
                {
                    "run": run,
                    "variant": label,
                    "checkpoint": payload["checkpoint"],
                    "checkpoint_step": payload.get("checkpoint_step"),
                    "backbone_loss": payload.get("loss_placement"),
                    "task_step_size": payload.get("task_step_size", 1),
                    "placement": payload["controller_placement"],
                    "parameterization": payload["parameterization"],
                    "rank": variant["rank"],
                    "controller_seed": variant["initialization_seed"],
                    "parameters": variant["parameter_count"],
                    "initialization": payload["initialization_mode"],
                    "state_loss_weight": payload["state_loss_weight"],
                    "max_training_horizon": payload["max_training_horizon"],
                    **segments(values),
                }
            )
    return rows, curves_by_run


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "auc_1_24",
        "auc_25_48",
        "auc_49_64",
        "auc_65_96",
        "auc_97_128",
        "accuracy_loop64",
        "accuracy_loop128",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run"]), []).append(row)
    result: list[dict[str, Any]] = []
    for run, items in sorted(grouped.items()):
        base = {
            key: items[0][key]
            for key in (
                "run",
                "checkpoint_step",
                "task_step_size",
                "placement",
                "parameterization",
                "rank",
                "parameters",
                "initialization",
                "state_loss_weight",
                "max_training_horizon",
            )
        }
        base["controller_seeds"] = len(items)
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            valid = [value for value in values if not math.isnan(value)]
            base[f"{metric}_mean"] = fmean(valid) if valid else float("nan")
            base[f"{metric}_std"] = pstdev(valid) if valid else float("nan")
        result.append(base)
    return result


def strict_curves(root: Path) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for path in sorted((root / "audits").glob("*/strict_unseen/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        curves = payload["strictly_unseen"]["curves"]
        result[path.parent.parent.name] = {
            label: [float(value) for value in item["accuracy_by_cycle"]]
            for label, item in curves.items()
            if label.startswith("task_")
        }
    return result


def curve_aggregate_rows(
    curves: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, variants in sorted(curves.items()):
        metrics = [segments(values) for values in variants.values()]
        if not metrics:
            continue
        row: dict[str, Any] = {"run": run, "controller_seeds": len(metrics)}
        for metric in metrics[0]:
            values = [float(item[metric]) for item in metrics]
            row[f"{metric}_mean"] = fmean(values)
            row[f"{metric}_std"] = pstdev(values)
        rows.append(row)
    return rows


def plot_strict(curves: dict[str, dict[str, list[float]]], out_dir: Path) -> None:
    if not curves:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    colors = {"final_seed0_ce_h64": "#0072B2", "final_seed3_ce_h64": "#009E73"}
    names = {
        "final_seed0_ce_h64": "backbone seed 0 (CE only after loop 8)",
        "final_seed3_ce_h64": "backbone seed 3 (CE only after loop 8)",
    }
    fig, ax = plt.subplots(figsize=(7.3, 4.0))
    for run, items in curves.items():
        if run not in names:
            continue
        if not items:
            continue
        matrix = np.asarray(list(items.values()), dtype=float)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        x = np.arange(1, len(mean) + 1)
        color = colors.get(run)
        ax.plot(x, mean, lw=2.0, color=color, label=names.get(run, run))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
    ax.axvline(64, color="#777777", ls="--", lw=1, label="max training horizon")
    ax.axhline(0.125, color="#D55E00", ls=":", lw=1, label="1/8 chance")
    ax.set(xlabel="Continuation loop", ylabel="Accuracy", xlim=(1, 128), ylim=(0, 1.03))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"strict_backbone_curves.{suffix}", dpi=220)
    plt.close(fig)


def plot_sweeps(
    controller_variants: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    ranks = [
        row
        for row in controller_variants
        if (
            row["run"] == "final_seed0_ce_h64"
            or (row["run"].startswith("final_seed0_rank") and "_ce_h64" in row["run"])
        )
        and int(row["rank"]) in {8, 16, 32, 48, 64, 96}
        and int(row["controller_seed"]) == 211001
    ]
    horizons = [
        row
        for row in controller_variants
        if row["run"] == "final_seed0_ce_h64"
        or row["run"].startswith("final_seed0_rank48_ce_h")
    ]
    horizons = [
        row for row in horizons if int(row["controller_seed"]) == 211001
    ]
    if not ranks and not horizons:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    if ranks:
        ranks.sort(key=lambda row: int(row["rank"]))
        axes[0].plot([r["rank"] for r in ranks], [r["auc_49_64"] for r in ranks], marker="o", label="AUC 49--64")
        axes[0].plot([r["rank"] for r in ranks], [r["auc_65_96"] for r in ranks], marker="s", label="AUC 65--96")
    axes[0].set(xlabel="Low-rank width", ylabel="Accuracy AUC", ylim=(0, 1.03))
    axes[0].grid(alpha=.2); axes[0].legend(frameon=False, fontsize=8)
    if horizons:
        horizons.sort(key=lambda row: int(row["max_training_horizon"]))
        axes[1].plot([r["max_training_horizon"] for r in horizons], [r["auc_49_64"] for r in horizons], marker="o", label="AUC 49--64")
        axes[1].plot([r["max_training_horizon"] for r in horizons], [r["auc_65_96"] for r in horizons], marker="s", label="AUC 65--96")
    axes[1].set(xlabel="Maximum training horizon", ylabel="Accuracy AUC", ylim=(0, 1.03))
    axes[1].grid(alpha=.2); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"rank_horizon_sweeps.{suffix}", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, random_curves = controller_rows(args.root)
    aggregates = aggregate_rows(rows)
    strict = strict_curves(args.root)
    strict_aggregates = curve_aggregate_rows(strict)
    write_csv(args.out_dir / "controller_variants.csv", rows)
    write_csv(args.out_dir / "controller_aggregates.csv", aggregates)
    write_csv(args.out_dir / "strict_unseen_aggregates.csv", strict_aggregates)
    plot_strict(strict, args.out_dir)
    plot_sweeps(rows, args.out_dir)
    result = {
        "status": "complete",
        "controller_runs": len(aggregates),
        "controller_variants": len(rows),
        "strict_runs": sorted(strict),
        "strict_aggregates": strict_aggregates,
        "random_curve_runs": sorted(random_curves),
        "aggregates": aggregates,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate canonical DiagLoRA-J runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
