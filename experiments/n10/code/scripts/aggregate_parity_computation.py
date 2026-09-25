#!/usr/bin/env python3
"""Aggregate the preregistered multi-seed Parity computation study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFIX_PROBE_THRESHOLD = 0.80
PREFIX_RECOVERY_THRESHOLD = 0.25
PREFIX_RANDOM_ADVANTAGE = 0.15
COUNT_CONTROL_FLIP_MAX = 0.20
ANSWER_COUNT_TRANSFER_THRESHOLD = 0.25
COUNT_LATER_GAIN_THRESHOLD = 0.10
COUNT_RANDOM_ADVANTAGE = 0.10
COUNT_EVENTUAL_FLIP_MAX = 0.20
JOINT_GAIN_THRESHOLD = 0.20
JOINT_CONTROL_ADVANTAGE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_run(metrics: Mapping[str, Any]) -> str:
    prefix = (
        float(metrics["prefix_probe"]) >= PREFIX_PROBE_THRESHOLD
        and float(metrics["prefix_recovery"]) >= PREFIX_RECOVERY_THRESHOLD
        and float(metrics["prefix_advantage"]) >= PREFIX_RANDOM_ADVANTAGE
        and float(metrics["count_control_flip_rate"]) <= COUNT_CONTROL_FLIP_MAX
        and bool(metrics["restoration_connected"])
    )
    count = (
        float(metrics["answer_count_transfer"]) >= ANSWER_COUNT_TRANSFER_THRESHOLD
        and float(metrics["count_later_gain"]) >= COUNT_LATER_GAIN_THRESHOLD
        and float(metrics["count_random_advantage"]) >= COUNT_RANDOM_ADVANTAGE
        and float(metrics["count_eventual_flip_rate"]) <= COUNT_EVENTUAL_FLIP_MAX
    )
    if prefix and count:
        return "mixed_localized"
    if prefix:
        return "prefix"
    if count:
        return "count"
    distributed = (
        float(metrics["joint_gain_over_single"]) >= JOINT_GAIN_THRESHOLD
        and float(metrics["joint_advantage_over_controls"]) >= JOINT_CONTROL_ADVANTAGE
    )
    return "distributed" if distributed else "unresolved"


def decide_checkpoint(run_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(run_metrics) != 3:
        raise ValueError("checkpoint decision requires exactly three causal-data seeds")
    seeds = [int(row["causal_seed"]) for row in run_metrics]
    if len(set(seeds)) != 3:
        raise ValueError("causal-data seeds must be distinct")
    decisions = [classify_run(row) for row in run_metrics]
    counts = Counter(decisions)
    localized_prefix = counts["prefix"] + counts["mixed_localized"]
    localized_count = counts["count"] + counts["mixed_localized"]
    if localized_prefix >= 2 and localized_count >= 2:
        decision = "mixed_localized"
    elif localized_prefix >= 2:
        decision = "prefix"
    elif localized_count >= 2:
        decision = "count"
    elif counts["distributed"] >= 2:
        decision = "distributed"
    else:
        decision = "unresolved"
    return {
        "decision": decision,
        "run_decisions": dict(counts),
        "causal_seeds": sorted(seeds),
    }


def _mean(rows: Iterable[Mapping[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field, "") not in {"", "nan", "NaN"}]
    return float(np.mean(values)) if values else float("nan")


def _selected(
    rows: Sequence[Mapping[str, str]], **conditions: Any
) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in conditions.items())
    ]


def _final_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [row for row in rows if int(row["evaluation_call"]) == int(row["length"])]


def _condition_margin_delta(
    rows: Sequence[Mapping[str, str]],
    *,
    condition_prefix: str,
    evaluation_offset: int,
) -> float:
    by_cell: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    keys = ("length", "site", "family", "variable", "evaluation_call")
    for row in rows:
        if int(row["continuation_calls"]) != evaluation_offset:
            continue
        key = tuple(row[field] for field in keys)
        condition = row["condition"]
        if condition == "untouched_receiver" or condition.startswith(condition_prefix):
            by_cell[key][condition] = float(row["predicted_oriented_margin"])
    deltas = []
    for values in by_cell.values():
        untouched = values.get("untouched_receiver")
        tested = [value for name, value in values.items() if name.startswith(condition_prefix)]
        if untouched is not None and tested:
            deltas.extend(abs(value - untouched) for value in tested)
    return float(np.mean(deltas)) if deltas else 0.0


def extract_run_metrics(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete run: {run_dir}")
    selection = json.loads((run_dir / "selection.json").read_text())
    atlas = read_csv(run_dir / "atlas.csv")
    causal = read_csv(run_dir / "causal_summary.csv")
    restoration = read_csv(run_dir / "restoration_scan.csv")
    joint = read_csv(run_dir / "joint_patch_summary.csv")

    prefix_probe_rows = [
        row
        for row in atlas
        if row["site"] in selection["prefix_sites"] and row["variable"] == "prefix_parity"
    ]
    prefix_probe = max(float(row["validation_score"]) for row in prefix_probe_rows)
    answer_count_rows = [
        row
        for row in atlas
        if row["site"] in selection["answer_sites"]
        and row["variable"] in {"total_count", "total_count_mod4"}
    ]
    answer_count_transfer = max(float(row["score_above_permutation"]) for row in answer_count_rows)

    final = _final_rows(causal)
    prefix_patch = _selected(
        final,
        role="prefix",
        family="parity_flip",
        variable="parity",
        condition="parity_subspace",
    )
    prefix_random = [
        row
        for row in final
        if row["role"] == "prefix"
        and row["family"] == "parity_flip"
        and row["variable"] == "parity"
        and row["condition"].startswith("parity_random_")
    ]
    prefix_recovery = _mean(prefix_patch, "normalized_recovery")
    prefix_advantage = prefix_recovery - _mean(prefix_random, "normalized_recovery")
    count_control = _selected(
        final,
        role="prefix",
        family="count_shift",
        variable="parity",
        condition="parity_subspace",
    )
    count_control_flip_rate = 1.0 - _mean(count_control, "receiver_label_match")

    corrupt_scan = [
        row
        for row in restoration
        if row["direction"] == "corrupt_to_clean" and row["token_role"] == "intermediate"
    ]
    peaks = []
    for call in sorted({int(row["call"]) for row in corrupt_scan}):
        selected_rows = [row for row in corrupt_scan if int(row["call"]) == call]
        if selected_rows:
            best = max(selected_rows, key=lambda row: float(row["mean_normalized_recovery"]))
            if float(best["mean_normalized_recovery"]) > 0.10:
                peaks.append(float(best["token_fraction"]))
    restoration_connected = len(peaks) >= 2 and all(
        later + 0.05 >= earlier for earlier, later in zip(peaks, peaks[1:])
    )

    count_rows = [
        row
        for row in causal
        if row["role"] == "answer"
        and row["family"] == "count_shift"
        and row["variable"] == "count_given_parity"
    ]
    count_instant = _condition_margin_delta(
        count_rows, condition_prefix="count_given_parity_subspace", evaluation_offset=0
    )
    count_later = _condition_margin_delta(
        count_rows, condition_prefix="count_given_parity_subspace", evaluation_offset=1
    )
    count_random = _condition_margin_delta(
        count_rows, condition_prefix="count_given_parity_random_", evaluation_offset=1
    )
    count_later_gain = count_later - count_instant
    count_random_advantage = count_later - count_random
    count_final = _selected(
        _final_rows(count_rows), condition="count_given_parity_subspace"
    )
    count_eventual_flip_rate = 1.0 - _mean(count_final, "receiver_label_match")

    ordered = [row for row in joint if row["condition"] == "ordered"]
    single = [row for row in ordered if int(row["set_size"]) == 1]
    multi = [row for row in ordered if int(row["set_size"]) >= 2]
    best_single = max([float(row["mean_normalized_recovery"]) for row in single], default=0.0)
    best_multi_row = max(
        multi, key=lambda row: float(row["mean_normalized_recovery"]), default=None
    )
    if best_multi_row is None:
        joint_gain = 0.0
        joint_advantage = 0.0
    else:
        best_multi = float(best_multi_row["mean_normalized_recovery"])
        set_size = best_multi_row["set_size"]
        controls = [
            float(row["mean_normalized_recovery"])
            for row in joint
            if row["set_size"] == set_size and row["condition"] != "ordered"
        ]
        joint_gain = best_multi - best_single
        joint_advantage = best_multi - max(controls, default=0.0)
    metrics = {
        "backbone_seed": int(manifest["backbone_seed"]),
        "causal_seed": int(manifest["causal_seed"]),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_sha256": json.dumps(manifest["source_sha256"], sort_keys=True),
        "prefix_probe": prefix_probe,
        "prefix_recovery": prefix_recovery,
        "prefix_advantage": prefix_advantage,
        "count_control_flip_rate": count_control_flip_rate,
        "restoration_connected": restoration_connected,
        "answer_count_transfer": answer_count_transfer,
        "count_later_gain": count_later_gain,
        "count_random_advantage": count_random_advantage,
        "count_eventual_flip_rate": count_eventual_flip_rate,
        "joint_gain_over_single": joint_gain,
        "joint_advantage_over_controls": joint_advantage,
    }
    metrics["run_decision"] = classify_run(metrics)
    return metrics


def validate_campaign(metrics: Sequence[Mapping[str, Any]]) -> None:
    if len(metrics) != 9:
        raise ValueError(f"formal campaign requires nine runs, found {len(metrics)}")
    by_backbone: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_backbone[int(row["backbone_seed"])].append(row)
    if set(by_backbone) != {0, 1, 2}:
        raise ValueError("formal campaign requires backbone seeds 0, 1, and 2")
    for seed, rows in by_backbone.items():
        if len(rows) != 3 or len({int(row["causal_seed"]) for row in rows}) != 3:
            raise ValueError(f"backbone {seed} does not have three distinct data seeds")
        if len({str(row["checkpoint_sha256"]) for row in rows}) != 1:
            raise ValueError(f"backbone {seed} checkpoint hash changed across runs")
        if len({str(row["source_sha256"]) for row in rows}) != 1:
            raise ValueError(f"backbone {seed} source hashes changed across runs")


def aggregate(root: Path) -> dict[str, Any]:
    run_dirs = sorted(path.parent for path in root.glob("seed*/data*/summary.json"))
    metrics = [extract_run_metrics(path) for path in run_dirs]
    validate_campaign(metrics)
    by_backbone: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_backbone[int(row["backbone_seed"])].append(row)
    checkpoint_rows = []
    for seed, rows in sorted(by_backbone.items()):
        decision = decide_checkpoint(rows)
        checkpoint_rows.append(
            {
                "backbone_seed": seed,
                "decision": decision["decision"],
                "run_decisions": json.dumps(decision["run_decisions"], sort_keys=True),
                "prefix_probe": float(np.mean([float(row["prefix_probe"]) for row in rows])),
                "prefix_recovery": float(np.mean([float(row["prefix_recovery"]) for row in rows])),
                "prefix_advantage": float(np.mean([float(row["prefix_advantage"]) for row in rows])),
                "answer_count_transfer": float(np.mean([float(row["answer_count_transfer"]) for row in rows])),
                "count_later_gain": float(np.mean([float(row["count_later_gain"]) for row in rows])),
                "joint_gain_over_single": float(np.mean([float(row["joint_gain_over_single"]) for row in rows])),
            }
        )
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_csv(aggregate_dir / "decision_table.csv", metrics)
    write_csv(aggregate_dir / "checkpoint_summary.csv", checkpoint_rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(3)
    ax.bar(x - 0.25, [row["prefix_advantage"] for row in checkpoint_rows], width=0.25, label="prefix over random")
    ax.bar(x, [row["count_later_gain"] for row in checkpoint_rows], width=0.25, label="count later gain")
    ax.bar(x + 0.25, [row["joint_gain_over_single"] for row in checkpoint_rows], width=0.25, label="joint over single")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"seed {seed}" for seed in range(3)])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(aggregate_dir / "parity_mechanism_multiseed.png", dpi=180)
    plt.close(fig)
    decisions = {int(row["backbone_seed"]): str(row["decision"]) for row in checkpoint_rows}
    lines = [
        "# Parity 计算机制 Stage B",
        "",
        "## 一句话结论",
        "",
        "本结论由预注册的三类判据产生；probe 只负责定位，机制归类取决于冻结执行后的 counterfactual 行为。",
        "",
        "## Checkpoint 判定",
        "",
        "| backbone | mechanism |",
        "|---:|---|",
        *[f"| {seed} | {decision} |" for seed, decision in decisions.items()],
        "",
        "`unresolved` 不会被自动改写为 distributed；不同 checkpoint 的分歧也不会被跨 seed 平均掩盖。",
        "",
        "## 博客允许的表述",
        "",
    ]
    unique = set(decisions.values())
    if unique == {"prefix"}:
        lines.append("三个 checkpoint 都支持局部 prefix-parity route，可把它作为 toy-model 主线。")
    elif unique == {"count"}:
        lines.append("三个 checkpoint 都支持 answer-token count-to-parity route，可把它作为 toy-model 主线。")
    elif len(unique) > 1:
        lines.append("checkpoint 采用不同路线；博客主线应是相同任务可以收敛到不同小算法。")
    else:
        lines.append("现有因果测试没有唯一识别算法；博客必须保留机制未决。")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本实验只覆盖三个固定 input-once toy checkpoints，不支持对一般 Transformer 或真实语言模型外推。",
        ]
    )
    (aggregate_dir / "PARITY_COMPUTATION_STAGE_B_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"status": "complete", "run_count": len(metrics), "checkpoint_decisions": decisions}
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(aggregate(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
