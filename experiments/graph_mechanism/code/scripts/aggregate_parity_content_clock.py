#!/usr/bin/env python3
"""Aggregate the three-backbone by three-data-seed Parity Stage-A campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_ARCHITECTURE = {
    "task": "parity",
    "d_model": 256,
    "attention_heads": 64,
    "mlp_width": 1024,
    "shared_physical_layers": 1,
    "token_embedding_injection": "initial_only",
    "position_embedding": "none",
    "position_injection": "initial_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def classify_cross_seed(passes: dict[int, bool]) -> str:
    values = list(passes.values())
    if len(values) != 3:
        raise ValueError("formal aggregation requires exactly three backbone seeds")
    if all(values):
        return "replicated_all_seeds"
    if any(values):
        return "heterogeneous"
    return "rejected_all_seeds"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        "run_manifest.json",
        "summary.json",
        "factorial_summary.csv",
        "factorial_interaction.csv",
        "factorial_per_example.csv.gz",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"{run_dir}: missing artifacts {missing}")
    summary = _load_json(run_dir / "summary.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    if summary.get("status") != "complete" or manifest.get("status") != "complete":
        raise ValueError(f"{run_dir}: run is not complete")
    if summary.get("analysis") != "parity_content_clock_stage_a":
        raise ValueError(f"{run_dir}: wrong analysis kind")
    if summary.get("controller") is not None:
        raise ValueError(f"{run_dir}: Stage A must not use a controller")
    for name, expected in REQUIRED_ARCHITECTURE.items():
        observed = summary.get("architecture", {}).get(name)
        if observed != expected:
            raise ValueError(
                f"{run_dir}: architecture {name} expected {expected!r}, got {observed!r}"
            )
    checkpoint_hash = str(summary.get("checkpoint_sha256"))
    if checkpoint_hash != str(manifest.get("checkpoint_sha256")):
        raise ValueError(f"{run_dir}: manifest checkpoint hash mismatch")
    backbone_seed = int(summary["backbone_seed"])
    causal_seed = int(summary["protocol"]["causal_seed"])
    if backbone_seed != int(manifest["backbone_seed"]):
        raise ValueError(f"{run_dir}: manifest backbone seed mismatch")
    if causal_seed != int(manifest["causal_seed"]):
        raise ValueError(f"{run_dir}: manifest causal seed mismatch")
    if not _read_csv(run_dir / "factorial_summary.csv"):
        raise ValueError(f"{run_dir}: empty factorial summary")
    if not _read_csv(run_dir / "factorial_interaction.csv"):
        raise ValueError(f"{run_dir}: empty factorial interaction")
    return summary, manifest


def _majority(values: Sequence[bool]) -> bool:
    if len(values) != 3:
        raise ValueError("each backbone needs exactly three causal-data seeds")
    return sum(bool(value) for value in values) >= 2


def _conclusion(status: str) -> str:
    if status == "replicated_all_seeds":
        return "三颗冻结 backbone 都支持局部的 parity 内容—时钟组合律。"
    if status == "heterogeneous":
        return "内容—时钟组合律只在部分冻结 backbone 成立，属于 checkpoint-dependent 机制。"
    return "现有二维轨迹不能解释为可组合的 parity 内容—时钟状态。"


def _plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    metrics = (
        ("mean_content_shift_error_from_pi", 0.50, "content shift error"),
        ("mean_clock_shift_error", 0.50, "clock shift error"),
        (
            "mean_absolute_factorial_interaction_error",
            0.35,
            "factorial interaction",
        ),
        (
            "continued_advantage_over_random",
            0.0,
            "random error - phase error",
        ),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))
    for axis, (name, threshold, title) in zip(axes, metrics):
        for seed in (0, 1, 2):
            selected = [row for row in rows if int(row["backbone_seed"]) == seed]
            values = [float(row[name]) for row in selected]
            x = np.full(len(values), seed, dtype=float) + np.linspace(-0.08, 0.08, len(values))
            axis.scatter(x, values, s=32, label=f"seed {seed}")
            axis.scatter([seed], [np.mean(values)], marker="_", s=180, color="black")
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_xticks((0, 1, 2))
        axis.set_xlabel("backbone seed")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def aggregate_runs(
    run_dirs: Sequence[Path], out_dir: Path
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_pairs: set[tuple[int, int]] = set()
    checkpoint_hashes: dict[int, set[str]] = {}
    run_rows: list[dict[str, Any]] = []
    per_backbone_passes: dict[int, list[bool]] = {}
    for run_dir in sorted(Path(path) for path in run_dirs):
        summary, _ = _validate_run(run_dir)
        backbone_seed = int(summary["backbone_seed"])
        causal_seed = int(summary["protocol"]["causal_seed"])
        pair = (backbone_seed, causal_seed)
        if pair in seen_pairs:
            raise ValueError(f"duplicate backbone/data seed pair: {pair}")
        seen_pairs.add(pair)
        checkpoint_hashes.setdefault(backbone_seed, set()).add(
            str(summary["checkpoint_sha256"])
        )
        decision = summary["decision"]
        passed = bool(decision["checkpoint_data_seed_pass"])
        per_backbone_passes.setdefault(backbone_seed, []).append(passed)
        run_rows.append(
            {
                "backbone_seed": backbone_seed,
                "causal_seed": causal_seed,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "checkpoint_data_seed_pass": passed,
                "mean_content_shift_error_from_pi": decision[
                    "mean_content_shift_error_from_pi"
                ],
                "mean_clock_shift_error": decision["mean_clock_shift_error"],
                "mean_absolute_factorial_interaction_error": decision[
                    "mean_absolute_factorial_interaction_error"
                ],
                "mean_phase_continued_target_error_calls_1_2": decision[
                    "mean_phase_continued_target_error_calls_1_2"
                ],
                "median_random_continued_target_error_calls_1_2": decision[
                    "median_random_continued_target_error_calls_1_2"
                ],
                "continued_advantage_over_random": decision[
                    "median_random_continued_target_error_calls_1_2"
                ]
                - decision["mean_phase_continued_target_error_calls_1_2"],
                "phase_opposite_target_readout_match": decision[
                    "phase_opposite_target_readout_match"
                ],
                "random_opposite_target_readout_match": decision[
                    "random_opposite_target_readout_match"
                ],
                "run_dir": str(run_dir),
            }
        )
    if set(per_backbone_passes) != {0, 1, 2}:
        raise ValueError("formal aggregation requires backbone seeds 0, 1, and 2")
    for seed in (0, 1, 2):
        if len(per_backbone_passes[seed]) != 3:
            raise ValueError(f"backbone seed {seed} requires exactly three data seeds")
        if len(checkpoint_hashes[seed]) != 1:
            raise ValueError(f"backbone seed {seed} has inconsistent checkpoint hashes")
    checkpoint_passes = {
        seed: _majority(per_backbone_passes[seed]) for seed in (0, 1, 2)
    }
    status = classify_cross_seed(checkpoint_passes)
    decision_rows = [
        {
            "backbone_seed": seed,
            "checkpoint_sha256": next(iter(checkpoint_hashes[seed])),
            "passing_data_seeds": sum(per_backbone_passes[seed]),
            "total_data_seeds": 3,
            "checkpoint_passes": checkpoint_passes[seed],
        }
        for seed in (0, 1, 2)
    ]
    _write_csv(out_dir / "multiseed_cells.csv", run_rows)
    _write_csv(out_dir / "multiseed_decisions.csv", decision_rows)
    _plot(run_rows, out_dir / "content_clock_multiseed.png")
    conclusion = _conclusion(status)
    report_lines = [
        "# Parity 内容—时钟 Stage A 结果",
        "",
        conclusion,
        "",
        "## 观察",
        "",
        "每颗 backbone 使用三个独立 causal-data seed；几何平面只由对应 backbone 的 discovery 数据拟合。下表不把 data seed 当作独立 backbone。",
        "",
        "| backbone seed | 通过的数据 seed | checkpoint 结论 |",
        "|---:|---:|---|",
    ]
    for row in decision_rows:
        report_lines.append(
            f"| {row['backbone_seed']} | {row['passing_data_seeds']}/3 | "
            f"{'通过' if row['checkpoint_passes'] else '不通过'} |"
        )
    report_lines.extend(
        [
            "",
            "## 因果判据",
            "",
            "每个 run 同时检验 parity 半圈位移、call 相位位移、二者的 circular difference-in-differences、连续两次冻结执行后的轨迹恢复，以及等范数随机二维平面控制。checkpoint 需要三个 data seed 中至少两个整体通过。",
            "",
            "## 证据边界",
            "",
            "即使三颗 checkpoint 都通过，这也只识别 answer token 的局部内容—时钟状态；输入 bit 如何被组合成 parity 仍属于 Stage B，不能据此声称完整 parity algorithm 已被找到。",
            "",
            "## 下一步",
            "",
            (
                "进入 Stage B，优先用 prefix-parity 与 Hamming-count 匹配的 counterfactual interchange 区分两种计算路线。"
                if status != "rejected_all_seeds"
                else "停止内容—时钟叙事，重新把二维轨迹建模为 class-conditioned dynamical manifold。"
            ),
        ]
    )
    (out_dir / "PARITY_CONTENT_CLOCK_STAGE_A_ZH.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    ledger = [
        "# Claim Ledger Update — Parity Stage A",
        "",
        f"- Stage-A status: `{status}`.",
        f"- Allowed answer-state wording: {conclusion}",
        "- Full parity algorithm: remains open; Stage B causal-variable tests are required.",
        "- J timing claim: unchanged; no controller was used in Stage A.",
    ]
    (out_dir / "CLAIM_LEDGER_UPDATE.md").write_text(
        "\n".join(ledger) + "\n", encoding="utf-8"
    )
    result = {
        "status": "complete",
        "analysis": "parity_content_clock_stage_a_multiseed",
        "checkpoint_passes": checkpoint_passes,
        "cross_seed_status": status,
        "conclusion_zh": conclusion,
        "run_count": len(run_rows),
        "backbone_count": 3,
        "data_seeds_per_backbone": 3,
        "files": {
            "cells": "multiseed_cells.csv",
            "decisions": "multiseed_decisions.csv",
            "plot": "content_clock_multiseed.png",
            "report": "PARITY_CONTENT_CLOCK_STAGE_A_ZH.md",
            "claim_ledger": "CLAIM_LEDGER_UPDATE.md",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    result = aggregate_runs(args.run_dirs, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
