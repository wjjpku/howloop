from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.boolean_dag_data import BooleanDAGConfig
from reasoning_loop.boolean_dag_model import BooleanDAGModelConfig
from reasoning_loop.boolean_dag_norm_dynamics import analyze_norm_checkpoint
from reasoning_loop.boolean_dag_train import TrainConfig, train_boolean_dag
from reasoning_loop.graph_path_loop import pick_device

@dataclass(frozen=True)
class ExperimentCondition:
    name: str
    outer_norm_groups: int
    inner_norm_groups: int


def experiment_conditions() -> list[ExperimentCondition]:
    return [
        ExperimentCondition("outer_G1_inner_LN", 1, 0),
        ExperimentCondition("outer_G4_inner_LN", 4, 0),
        ExperimentCondition("outer_G16_inner_LN", 16, 0),
        ExperimentCondition("outer_G4_inner_G1", 4, 1),
        ExperimentCondition("outer_G4_inner_G4", 4, 4),
        ExperimentCondition("outer_G4_inner_G16", 4, 16),
    ]


def baseline_for_condition(name: str) -> str:
    outer_treatments = {"outer_G4_inner_LN", "outer_G16_inner_LN"}
    inner_treatments = {
        "outer_G4_inner_G1",
        "outer_G4_inner_G4",
        "outer_G4_inner_G16",
    }
    if name in outer_treatments:
        return "outer_G1_inner_LN"
    if name in inner_treatments:
        return "outer_G4_inner_LN"
    raise ValueError(f"{name} is a baseline rather than a treatment")


def assess_pilot_condition(
    treatment: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, Any]:
    required = {
        "loop4_accuracy",
        "loop4_correct_probability",
        "overloop_degradation",
        "late_effective_update_norm",
        "late_tangential_fraction",
    }
    for label, metrics in (("treatment", treatment), ("baseline", baseline)):
        missing = required.difference(metrics)
        if missing:
            raise ValueError(f"{label} is missing metrics: {sorted(missing)}")

    accuracy_preserved = treatment["loop4_accuracy"] >= baseline["loop4_accuracy"] - 0.01
    confidence_preserved = (
        treatment["loop4_correct_probability"]
        >= baseline["loop4_correct_probability"] - 0.03
    )
    absolute_gain = (
        baseline["overloop_degradation"] - treatment["overloop_degradation"]
    )
    relative_improvement = (
        treatment["overloop_degradation"]
        <= 0.70 * baseline["overloop_degradation"]
    )
    overloop_improved = absolute_gain >= 0.05 or relative_improvement
    dynamics_improved = (
        treatment["late_effective_update_norm"]
        <= 0.90 * baseline["late_effective_update_norm"]
        or treatment["late_tangential_fraction"]
        <= 0.90 * baseline["late_tangential_fraction"]
    )
    checks = {
        "accuracy_preserved": bool(accuracy_preserved),
        "confidence_preserved": bool(confidence_preserved),
        "overloop_improved": bool(overloop_improved),
        "dynamics_improved": bool(dynamics_improved),
    }
    return {
        **checks,
        "absolute_overloop_gain": float(absolute_gain),
        "passes": all(checks.values()),
    }


def build_comparison_rows(
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {condition.name for condition in experiment_conditions()}
    missing = expected.difference(summaries)
    if missing:
        raise ValueError(f"missing condition summaries: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for condition in experiment_conditions():
        summary = summaries[condition.name]
        metrics = summary["pilot_metrics"]
        row = {
            "condition": condition.name,
            "outer_norm_groups": condition.outer_norm_groups,
            "inner_norm_groups": condition.inner_norm_groups,
            "parameter_count": summary.get("parameter_count"),
            **{key: float(value) for key, value in metrics.items()},
        }
        if condition.name != "outer_G1_inner_LN":
            baseline_name = baseline_for_condition(condition.name)
            assessment = assess_pilot_condition(
                metrics,
                summaries[baseline_name]["pilot_metrics"],
            )
            row.update({"baseline": baseline_name, **assessment})
        else:
            row.update(
                {
                    "baseline": "",
                    "accuracy_preserved": "",
                    "confidence_preserved": "",
                    "overloop_improved": "",
                    "dynamics_improved": "",
                    "absolute_overloop_gain": "",
                    "passes": "",
                }
            )
        rows.append(row)
    return rows


def _mean_train_depth_curve(summary: dict[str, Any]) -> np.ndarray:
    accuracy = np.asarray(summary["root_accuracy"], dtype=float)
    train_max_depth = int(summary.get("data_config", {}).get("train_max_depth", 4))
    return accuracy[: min(train_max_depth, accuracy.shape[0])].mean(axis=0)


def _mean_dynamics_curve(
    summary: dict[str, Any],
    key: str,
) -> tuple[list[int], list[float]]:
    rows = summary["dynamics_rows"]
    train_max_depth = int(summary.get("data_config", {}).get("train_max_depth", 4))
    loops = sorted({int(row["loop"]) for row in rows})
    values = [
        float(
            np.mean(
                [
                    float(row[key])
                    for row in rows
                    if int(row["loop"]) == loop
                    and int(row["depth"]) <= train_max_depth
                ]
            )
        )
        for loop in loops
    ]
    return loops, values


def _save_overloop_plot(
    summaries: dict[str, dict[str, Any]],
    *,
    trained_horizon: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition in experiment_conditions():
        curve = _mean_train_depth_curve(summaries[condition.name])
        loops = np.arange(1, len(curve) + 1)
        ax.plot(loops, curve, marker="o", linewidth=1.6, label=condition.name)
    ax.axvline(trained_horizon, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="loop readout",
        ylabel="mean root accuracy, depths 1-4",
        ylim=(0, 1.02),
        title="In-distribution accuracy through trained and extra loops",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _save_dynamics_comparison(
    summaries: dict[str, dict[str, Any]],
    *,
    trained_horizon: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for condition in experiment_conditions():
        loops, effective = _mean_dynamics_curve(
            summaries[condition.name], "effective_update_norm"
        )
        _, tangential = _mean_dynamics_curve(
            summaries[condition.name], "tangential_fraction"
        )
        axes[0].plot(loops, effective, marker="o", linewidth=1.4, label=condition.name)
        axes[1].plot(loops, tangential, marker="o", linewidth=1.4, label=condition.name)
    for ax, title, ylabel in (
        (axes[0], "Effective recurrent update", "mean update norm"),
        (axes[1], "Direction-changing update", "group-tangential energy fraction"),
    ):
        ax.axvline(trained_horizon, color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="loop", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _save_heatmap_grid(
    summaries: dict[str, dict[str, Any]],
    *,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    image = None
    for ax, condition in zip(axes.flat, experiment_conditions()):
        summary = summaries[condition.name]
        values = np.asarray(summary["root_accuracy"], dtype=float)
        depths = summary.get("depths", list(range(1, values.shape[0] + 1)))
        image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(condition.name, fontsize=10)
        ax.set_xlabel("loop")
        ax.set_ylabel("DAG depth")
        ax.set_xticks(range(values.shape[1]), range(1, values.shape[1] + 1))
        ax.set_yticks(range(values.shape[0]), depths)
    fig.subplots_adjust(
        left=0.06,
        right=0.89,
        bottom=0.08,
        top=0.93,
        wspace=0.16,
        hspace=0.24,
    )
    if image is not None:
        colorbar_axis = fig.add_axes((0.915, 0.16, 0.015, 0.68))
        fig.colorbar(image, cax=colorbar_axis, label="root accuracy")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _save_training_plot(out_dir: Path) -> bool:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    found = False
    for condition in experiment_conditions():
        history_path = out_dir / "runs" / condition.name / "history.csv"
        if not history_path.exists():
            continue
        with history_path.open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        if not history:
            continue
        found = True
        steps = [int(row["step"]) for row in history]
        axes[0].plot(
            steps,
            [float(row["train_loss"]) for row in history],
            label=condition.name,
        )
        axes[1].plot(
            steps,
            [float(row["eval_final_root_accuracy"]) for row in history],
            label=condition.name,
        )
    if not found:
        plt.close(fig)
        return False
    axes[0].set(title="Training loss", xlabel="step", ylabel="loss")
    axes[1].set(
        title="Held-out loop-4 root accuracy",
        xlabel="step",
        ylabel="accuracy",
        ylim=(0, 1.02),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "training_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    return True


def _report_text(rows: list[dict[str, Any]], *, has_training_plot: bool) -> str:
    lines = [
        "# 1M Group RMSNorm 六组实验报告",
        "",
        "## 实验配置",
        "",
        "所有模型使用同一个约 1M 参数的单层共享 Boolean-DAG Transformer、4 个训练 loop、final-only loss。所有条件都有 outer RMSNorm；outer sweep 保留内部 LayerNorm，inner sweep 固定 outer G=4。",
        "",
        "| condition | params | loop4 acc | loop4 prob | late retention | degradation | late delta norm | late tangent | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        passed = "-" if row["passes"] == "" else ("yes" if row["passes"] else "no")
        lines.append(
            f"| {row['condition']} | {row['parameter_count']} | "
            f"{row['loop4_accuracy']:.3f} | {row['loop4_correct_probability']:.3f} | "
            f"{row['overloop_retention']:.3f} | {row['overloop_degradation']:.3f} | "
            f"{row['late_effective_update_norm']:.3f} | "
            f"{row['late_tangential_fraction']:.3f} | {passed} |"
        )
    lines.extend(["", "## 主要图表", ""])
    if has_training_plot:
        lines.extend(["![训练曲线](training_comparison.png)", ""])
    lines.extend(
        [
            "![六组 depth-loop 准确率](accuracy_heatmaps.png)",
            "",
            "![Overloop 对比](overloop_comparison.png)",
            "",
            "![状态动力学对比](dynamics_comparison.png)",
            "",
            "## 预注册判定",
            "",
        ]
    )
    treatment_rows = [row for row in rows if row["passes"] != ""]
    passing = [row["condition"] for row in treatment_rows if row["passes"]]
    if passing:
        lines.append(
            "通过本轮筛选的配置：" + "、".join(f"`{name}`" for name in passing) + "。"
        )
    else:
        lines.append("没有配置同时通过准确率、overloop、动力学和置信度四项筛选。")
    lines.extend(
        [
            "",
            "注意：overloop 稳定不等于更深 DAG 的算法泛化；两者必须分别阅读热力图。单 seed 结果只能决定是否值得继续做多 seed。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_comparison_artifacts(
    *,
    summaries: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    out_dir: Path,
    trained_horizon: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"comparison_rows": rows, "conditions": summaries}, indent=2),
        encoding="utf-8",
    )
    _save_overloop_plot(
        summaries,
        trained_horizon=trained_horizon,
        path=out_dir / "overloop_comparison.png",
    )
    _save_dynamics_comparison(
        summaries,
        trained_horizon=trained_horizon,
        path=out_dir / "dynamics_comparison.png",
    )
    _save_heatmap_grid(summaries, path=out_dir / "accuracy_heatmaps.png")
    has_training_plot = _save_training_plot(out_dir)
    (out_dir / "REPORT_CN.md").write_text(
        _report_text(rows, has_training_plot=has_training_plot), encoding="utf-8"
    )


def run_experiment(
    *,
    out_dir: Path,
    steps: int,
    batch_size: int,
    eval_every: int,
    analysis_batches: int,
    analysis_batch_size: int,
    seed: int,
    device: torch.device,
    amp: bool,
    force: bool,
) -> dict[str, dict[str, Any]]:
    data_cfg = BooleanDAGConfig()
    summaries: dict[str, dict[str, Any]] = {}
    for condition in experiment_conditions():
        print(f"[condition] {condition.name}", flush=True)
        model_cfg = BooleanDAGModelConfig(
            d_model=256,
            n_heads=8,
            d_mlp=1024,
            steps=4,
            outer_norm_groups=condition.outer_norm_groups,
            inner_norm_groups=condition.inner_norm_groups,
        )
        run_dir = out_dir / "runs" / condition.name
        checkpoint = run_dir / "final.pt"
        if force or not checkpoint.exists():
            run_dir = train_boolean_dag(
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                train_cfg=TrainConfig(
                    architecture="looped",
                    loss_mode="final",
                    steps=steps,
                    batch_size=batch_size,
                    eval_batch_size=max(256, min(1024, batch_size * 2)),
                    eval_batches=8,
                    eval_every=eval_every,
                    print_every=eval_every,
                    warmup_steps=min(500, max(1, steps // 10)),
                    seed=seed,
                    device=str(device),
                    amp=amp,
                    out_dir=out_dir / "runs",
                    run_name=condition.name,
                    force=force,
                ),
            )
            checkpoint = run_dir / "final.pt"
        analysis_dir = out_dir / "analyses" / condition.name
        analysis_summary = analysis_dir / "summary.json"
        if force or not analysis_summary.exists():
            summary = analyze_norm_checkpoint(
                checkpoint=checkpoint,
                out_dir=analysis_dir,
                depths=list(range(1, 9)),
                readouts=12,
                batches=analysis_batches,
                batch_size=analysis_batch_size,
                device=device,
                amp=amp,
            )
        else:
            summary = json.loads(analysis_summary.read_text(encoding="utf-8"))
        summaries[condition.name] = summary
        (out_dir / "partial_summary.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )

    rows = build_comparison_rows(summaries)
    write_comparison_artifacts(
        summaries=summaries,
        rows=rows,
        out_dir=out_dir,
        trained_horizon=4,
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare inner and outer Group RMSNorm.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--analysis-batches", type=int, default=4)
    parser.add_argument("--analysis-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        out_dir=args.out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        analysis_batches=args.analysis_batches,
        analysis_batch_size=args.analysis_batch_size,
        seed=args.seed,
        device=pick_device(args.device),
        amp=args.amp,
        force=args.force,
    )


if __name__ == "__main__":
    main()
