"""Quantify and annotate recurrent ridges in a Parity loop-depth sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minimum-analysis-length", type=int, default=3)
    parser.add_argument("--maximum-cycle-index", type=int, default=3)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in ("length", "target_loop", "loop", "relative_loop", "examples"):
                row[key] = int(row[key])
            for key in (
                "exact_match",
                "parity_token_accuracy",
                "correct_parity_probability",
                "mean_parity_margin",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def recurrence_step(length: int, cycle_index: int) -> int:
    return (2 * cycle_index + 1) * (length + 1) - 1


def analyze(
    rows: list[dict[str, Any]],
    *,
    minimum_length: int,
    maximum_cycle_index: int,
    train_max_length: int,
) -> dict[str, Any]:
    lookup = {
        (str(row["variant"]), int(row["length"]), int(row["loop"])): row
        for row in rows
    }
    variants = sorted({str(row["variant"]) for row in rows}, key=lambda x: x != "raw")
    maximum_loop = max(int(row["loop"]) for row in rows)
    maximum_length = max(int(row["length"]) for row in rows)
    output: dict[str, Any] = {
        "ridge_hypothesis": "t_m(n)=(2m+1)(n+1)-1, m=0,1,2,...",
        "successive_ridge_period": "2(n+1) recurrent loops",
        "metric": "strict answer-region exact match",
        "variants": {},
    }
    for variant in variants:
        primary_pairs: list[tuple[int, int, float]] = []
        neighborhood: dict[str, dict[str, float]] = {}
        for split, start, stop in (
            ("trained_lengths", minimum_length, min(train_max_length, maximum_length)),
            ("OOD_lengths", max(train_max_length + 1, minimum_length), maximum_length),
        ):
            offset_values: dict[int, list[float]] = {offset: [] for offset in range(-3, 4)}
            if start <= stop:
                for length in range(start, stop + 1):
                    for offset in offset_values:
                        loop = length + offset
                        row = lookup.get((variant, length, loop))
                        if row is not None:
                            offset_values[offset].append(float(row["exact_match"]))
            neighborhood[split] = {
                str(offset): statistics.mean(values) if values else float("nan")
                for offset, values in offset_values.items()
            }

        for length in range(minimum_length, maximum_length + 1):
            candidates = [
                (loop, float(lookup[(variant, length, loop)]["exact_match"]))
                for loop in range(max(1, length - 3), min(maximum_loop, length + 3) + 1)
            ]
            best_accuracy = max(accuracy for _, accuracy in candidates)
            best_loop = min(
                loop for loop, accuracy in candidates if accuracy == best_accuracy
            )
            primary_pairs.append((length, best_loop, best_accuracy))
        x = np.asarray([length for length, _, _ in primary_pairs], dtype=np.float64)
        y = np.asarray([loop for _, loop, _ in primary_pairs], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, deg=1)
        predicted = slope * x + intercept
        total = float(np.square(y - y.mean()).sum())
        residual = float(np.square(y - predicted).sum())

        recurrence_rows: list[dict[str, Any]] = []
        for cycle_index in range(maximum_cycle_index + 1):
            formula_values: list[float] = []
            local_best_values: list[float] = []
            best_offsets: list[int] = []
            used_lengths: list[int] = []
            for length in range(minimum_length, maximum_length + 1):
                center = recurrence_step(length, cycle_index)
                if center > maximum_loop:
                    continue
                used_lengths.append(length)
                formula_values.append(
                    float(lookup[(variant, length, center)]["exact_match"])
                )
                candidates = [
                    (loop, float(lookup[(variant, length, loop)]["exact_match"]))
                    for loop in range(max(1, center - 2), min(maximum_loop, center + 2) + 1)
                ]
                best_loop, best_accuracy = max(candidates, key=lambda item: item[1])
                local_best_values.append(best_accuracy)
                best_offsets.append(best_loop - center)
            recurrence_rows.append(
                {
                    "cycle_index_m": cycle_index,
                    "formula": f"t={2 * cycle_index + 1}(n+1)-1",
                    "length_range": [min(used_lengths), max(used_lengths)],
                    "n_lengths": len(used_lengths),
                    "mean_accuracy_exactly_on_formula": statistics.mean(formula_values),
                    "mean_best_accuracy_within_plus_minus_2": statistics.mean(local_best_values),
                    "exact_formula_is_local_peak_fraction": (
                        sum(offset == 0 for offset in best_offsets) / len(best_offsets)
                    ),
                    "mean_local_peak_offset": statistics.mean(best_offsets),
                    "local_peak_offset_counts": {
                        str(offset): count
                        for offset, count in sorted(Counter(best_offsets).items())
                    },
                }
            )
        output["variants"][variant] = {
            "primary_ridge_local_fit": {
                "length_range": [minimum_length, maximum_length],
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": 1.0 - residual / total if total > 0 else None,
                "mean_absolute_loop_offset": float(np.abs(y - x).mean()),
                "exact_t_equals_n_fraction": float(np.mean(y == x)),
                "mean_local_peak_accuracy": statistics.mean(
                    accuracy for _, _, accuracy in primary_pairs
                ),
            },
            "target_neighborhood_mean_accuracy": neighborhood,
            "recurrence_ridges": recurrence_rows,
        }
    output["interpretation"] = {
        "behavioral_finding": (
            "The readout is phase-localized, with recurrent accuracy echoes whose "
            "period scales linearly with sequence length."
        ),
        "controller_finding": (
            "J phase-locks later echoes to the same length-dependent formula; it "
            "does not turn the solution into an absorbing fixed point."
        ),
        "claim_boundary": (
            "These ridges support a token-propagation or traveling-wave hypothesis. "
            "Causal state patching and component interventions are still required "
            "to identify the internal circuit and propagation direction."
        ),
    }
    return output


def plot(rows: list[dict[str, Any]], out_dir: Path, maximum_cycle_index: int) -> None:
    variants = sorted({str(row["variant"]) for row in rows}, key=lambda x: x != "raw")
    lengths = sorted({int(row["length"]) for row in rows})
    loops = sorted({int(row["loop"]) for row in rows})
    lookup = {
        (str(row["variant"]), int(row["length"]), int(row["loop"])): float(
            row["exact_match"]
        )
        for row in rows
    }
    colors = ("white", "#ff6b6b", "#67e8f9", "#fbbf24")
    figure, axes = plt.subplots(1, len(variants), figsize=(16, 6.3), sharex=True, sharey=True)
    if len(variants) == 1:
        axes = np.asarray([axes])
    for axis, variant in zip(axes, variants, strict=True):
        matrix = np.asarray(
            [
                [lookup[(variant, length, loop)] for loop in loops]
                for length in lengths
            ]
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(loops[0] - 0.5, loops[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        for cycle_index in range(maximum_cycle_index + 1):
            valid_lengths = [
                length
                for length in lengths
                if recurrence_step(length, cycle_index) <= loops[-1]
            ]
            if not valid_lengths:
                continue
            axis.plot(
                [recurrence_step(length, cycle_index) for length in valid_lengths],
                valid_lengths,
                color=colors[cycle_index % len(colors)],
                linestyle="--",
                linewidth=1.25,
                label=f"m={cycle_index}: t={2 * cycle_index + 1}(n+1)-1",
            )
        axis.axhline(20.5, color="white", linewidth=0.8, alpha=0.8)
        axis.set_title(f"{variant}: strict exact match")
        axis.set_xlabel("executed recurrent loops")
        axis.set_ylabel("input length n")
        axis.legend(loc="upper left", fontsize=8, framealpha=0.78)
        figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    figure.suptitle("Parity recurrence echoes: predicted phase ridges")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_recurrence_ridges.png", dpi=220)
    plt.close(figure)


def write_report(analysis: dict[str, Any], run_summary: dict[str, Any], path: Path) -> None:
    raw = analysis["variants"]["raw"]
    controlled = analysis["variants"].get("J")
    lines = [
        "# Parity loop-depth 热力图分析",
        "",
        "## 设置",
        "",
        (
            f"- backbone seed={run_summary['backbone_seed']}，step={run_summary['checkpoint_step']}，"
            f"d_model={run_summary['model']['d_model']}，heads={run_summary['model']['n_heads']}，"
            f"共享物理 block={run_summary['model']['shared_physical_layers']}。"
        ),
        (
            f"- 输入长度 {run_summary['evaluation']['lengths'][0]}–{run_summary['evaluation']['lengths'][1]}，"
            f"执行 loop {run_summary['evaluation']['loops'][0]}–{run_summary['evaluation']['loops'][1]}，"
            f"每格 {run_summary['evaluation']['examples_per_cell']} 个固定随机样本。"
        ),
        "- backbone 只在每个样本的最终 T(n)=n 处接受 CE；热力图读取每一个中间 loop。",
        "",
        "## 现象",
        "",
        (
            f"- 主正确带严格落在 t=n：raw 的局部峰斜率="
            f"{raw['primary_ridge_local_fit']['slope']:.3f}，截距="
            f"{raw['primary_ridge_local_fit']['intercept']:.3f}，R²="
            f"{raw['primary_ridge_local_fit']['r_squared']:.3f}。"
        ),
        "- 正确答案不是稳定终态；离开 t=n 后会迅速熄灭，之后又周期性重新出现。",
        "- 回波公式为 t_m(n)=(2m+1)(n+1)-1，相邻回波相距 2(n+1) 个 loop。",
    ]
    for variant, values in (("raw", raw), ("J", controlled)):
        if values is None:
            continue
        lines.append(f"- {variant} 的回波统计：")
        for ridge in values["recurrence_ridges"]:
            lines.append(
                f"  - m={ridge['cycle_index_m']}，{ridge['formula']}："
                f"公式线上平均 ACC={ridge['mean_accuracy_exactly_on_formula']:.4f}，"
                f"公式点是 ±2 局部峰的比例="
                f"{ridge['exact_formula_is_local_peak_fraction']:.3f}，"
                f"平均峰偏移={ridge['mean_local_peak_offset']:+.3f} loop。"
            )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "这已经是很强的逐步、按位置传播的行为证据：计算具有与序列长度成正比的传播时间，而且存在像往返传播一样的周期回波。J 的主要作用更像把老化后漂移的计算相位重新锁准，而不是把答案冻结住。",
            "",
            "但热力图仍不是完整的因果 circuit 证明。下一步应在匹配 loop 上做 token-position state patching：把传播波前前方、波前位置、波后方的状态分别替换，并配合 attention-head/MLP 消融，确认信息确实以每 loop 一个位置的速度移动。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze(
        rows,
        minimum_length=args.minimum_analysis_length,
        maximum_cycle_index=args.maximum_cycle_index,
        train_max_length=20,
    )
    (args.out_dir / "recurrence_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot(rows, args.out_dir, args.maximum_cycle_index)
    write_report(analysis, run_summary, args.out_dir / "PARITY_LOOP_DEPTH_ANALYSIS_ZH.md")
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
