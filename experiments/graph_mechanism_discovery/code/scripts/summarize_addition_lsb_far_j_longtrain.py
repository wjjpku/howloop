#!/usr/bin/env python3
"""Summarize the two long-trained LSB-first Addition J controllers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    reference_protocol: dict[str, Any] | None = None

    for label in args.labels:
        selection_dir = args.root / "selection" / label
        selection = json.loads((selection_dir / "selection.json").read_text())
        aggregate = read_csv(selection_dir / "snapshot_aggregate_metrics.csv")
        source = next(row for row in aggregate if int(row["snapshot_update"]) == 5376)
        selected_update = int(selection["selected_snapshot_update"])
        selected = next(
            row for row in aggregate if int(row["snapshot_update"]) == selected_update
        )
        controller = torch.load(
            selection_dir / "selected_controller.pt",
            map_location="cpu",
            weights_only=False,
        )
        parameter_count = sum(
            int(tensor.numel()) for tensor in controller["controller_state_dict"].values()
        )
        protocol = {
            "validation_lengths": selection["validation_lengths"],
            "validation_seed": selection["validation_seed"],
            "examples_per_length": selection["examples_per_length"],
            "id_gate": selection["id_gate"],
        }
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"selection protocol mismatch for {label}")

        audit_root = args.root / "audits" / label
        audit_specs = {
            "id": ("id_l1to10_n1024", range(1, 11)),
            "exposed": ("exposed_l11to20_n1024", range(11, 21)),
            "heldout": ("heldout_l21to30_n512", range(21, 31)),
        }
        range_metrics: dict[str, dict[str, float]] = {}
        by_length: dict[tuple[str, int], dict[str, float]] = {}
        for range_name, (directory, expected_lengths) in audit_specs.items():
            rows = read_csv(audit_root / directory / "endpoint_accuracy.csv")
            expected = set(expected_lengths)
            for row in rows:
                variant = row["variant"]
                length = int(row["length"])
                if length not in expected:
                    raise ValueError(f"unexpected length {length} in {directory}")
                metrics = {
                    "digit_em": float(row["supervised_digit_exact_match"]),
                    "bit_accuracy": float(row["supervised_digit_bit_accuracy"]),
                }
                by_length[(variant, length)] = metrics
                endpoint_rows.append({
                    "label": label,
                    "rank": int(controller["rank"]),
                    "range": range_name,
                    "variant": variant,
                    "length": length,
                    "examples": int(row["examples"]),
                    **metrics,
                })
            for variant in ("raw", "full"):
                values = [by_length[(variant, length)] for length in expected_lengths]
                range_metrics[f"{range_name}_{variant}"] = {
                    "mean_digit_em": mean(item["digit_em"] for item in values),
                    "mean_bit_accuracy": mean(item["bit_accuracy"] for item in values),
                }

        summaries.append({
            "label": label,
            "rank": int(controller["rank"]),
            "parameter_count": parameter_count,
            "source_update": 5376,
            "selected_update": selected_update,
            "target_total_updates": int(controller["snapshot_total_updates"]),
            "source_selection_mean_digit_em": float(source["mean_digit_em"]),
            "selected_selection_mean_digit_em": float(selected["mean_digit_em"]),
            "selection_improvement": (
                float(selected["mean_digit_em"]) - float(source["mean_digit_em"])
            ),
            "source_selection_m10_em": float(source["id_gate_em"]),
            "selected_selection_m10_em": float(selected["id_gate_em"]),
            "high_sample_m10_full_em": by_length[("full", 10)]["digit_em"],
            "high_sample_m10_raw_em": by_length[("raw", 10)]["digit_em"],
            "high_sample_full_digit_em_by_length": {
                str(length): by_length[("full", length)]["digit_em"]
                for length in range(1, 31)
            },
            "range_metrics": range_metrics,
        })

    comparison_rows = []
    for item in summaries:
        comparison_rows.append({
            "label": item["label"],
            "rank": item["rank"],
            "parameter_count": item["parameter_count"],
            "source_update": item["source_update"],
            "selected_update": item["selected_update"],
            "source_selection_mean_digit_em": item["source_selection_mean_digit_em"],
            "selected_selection_mean_digit_em": item["selected_selection_mean_digit_em"],
            "selection_improvement": item["selection_improvement"],
            "high_sample_id_full_mean_em": item["range_metrics"]["id_full"]["mean_digit_em"],
            "high_sample_exposed_full_mean_em": item["range_metrics"]["exposed_full"]["mean_digit_em"],
            "high_sample_heldout_full_mean_em": item["range_metrics"]["heldout_full"]["mean_digit_em"],
        })

    best = max(
        summaries,
        key=lambda item: (
            item["range_metrics"]["exposed_full"]["mean_digit_em"],
            item["range_metrics"]["heldout_full"]["mean_digit_em"],
            -item["parameter_count"],
        ),
    )
    result = {
        "status": "complete",
        "backbone": {
            "task": "Addition LSB-to-MSB",
            "causal": True,
            "position_embedding": "none",
            "training_lengths": [1, 10],
            "checkpoint_step": 80000,
            "shared_physical_block_layers": 3,
            "target_loop_rule": "T(m)=m",
        },
        "controller_training": {
            "parameterization": "diagonal plus low-rank affine J",
            "controller_exposed_lengths": [10, 20],
            "source_total_updates": 5376,
            "target_total_updates": 20000,
            "additional_updates": 14624,
            "loss": "final-only supervised-digit CE at T(m)=m",
            "post_final_j": False,
            "optimizer_resume": "fresh AdamW from each selected 5376-step controller",
        },
        "selection_protocol": reference_protocol,
        "candidates": summaries,
        "best_high_sample_exposed_candidate": best["label"],
        "claim_boundary": (
            "Lengths 21..30 were excluded from this continuation training and snapshot "
            "selection, but had been inspected in the preceding experiment, so they are "
            "held-out for this continuation decision rather than globally unopened."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.out_dir / "rank_comparison.csv", comparison_rows)
    write_csv(args.out_dir / "endpoint_length_metrics.csv", endpoint_rows)

    lines = [
        "# Addition LSB causal baseline：rank48 / rank96 的 20k 长训结果",
        "",
        "## 结论",
        "",
        (
            f"两个 rank 都从 5,376 步继续训练到 20,000 步；最终以独立数据在全部快照中选点。"
            f"高样本 m11..20 复评下，当前更优的是 **rank{best['rank']}**。"
            "两者都选中最后一步，说明原来的 5,376 步预算明确不足；"
            "但本实验尚未证明 20,000 步已经完全饱和。"
        ),
        "",
        "| rank | J 参数 | 独立选中步数 | 选点均值：5,376步 | 选点均值：最优快照 | 提升/百分点 | 高样本 m1..10 | 高样本 m11..20 | 高样本 m21..30 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['rank']} | {item['parameter_count']:,} | {item['selected_update']:,} | "
            f"{pct(item['source_selection_mean_digit_em'])} | "
            f"{pct(item['selected_selection_mean_digit_em'])} | "
            f"{100.0 * item['selection_improvement']:.2f} | "
            f"{pct(item['range_metrics']['id_full']['mean_digit_em'])} | "
            f"{pct(item['range_metrics']['exposed_full']['mean_digit_em'])} | "
            f"{pct(item['range_metrics']['heldout_full']['mean_digit_em'])} |"
        )
    lines.extend([
        "",
        "“选点均值”是新 seed 上 m10..20、每个长度 1,024 个样本的 supervised-digit exact match；"
        "m10 必须达到 99.5%。高样本复评另用一个 seed：m1..20 每长度 1,024 个样本，"
        "m21..30 每长度 512 个样本。",
        "",
        "表中的 m1..10 是把同一个远训 J **无条件**施加到全部短长度后的结果；raw 主模型在该段为 100%。"
        "这说明远端 J 应按长度或计算阶段启用，而不是全长度常开。",
        "",
        "| rank | m20 | m21 | m22 | m23 | m24 | m25 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in summaries:
        values = item["high_sample_full_digit_em_by_length"]
        lines.append(
            f"| {item['rank']} | {pct(values['20'])} | {pct(values['21'])} | "
            f"{pct(values['22'])} | {pct(values['23'])} | {pct(values['24'])} | "
            f"{pct(values['25'])} |"
        )
    lines.extend([
        "",
        "m26..30 基本归零，因此这里观察到的是有限的边界外推，不是无界长度泛化。",
        "",
        "## 实验边界",
        "",
        "- 主模型固定为 Addition LSB→MSB、causal、NoPE、训练长度 m=1..10、checkpoint 80,000。",
        "- 主模型有 3 个共享 physical block layers；推理有效深度由 T(m)=m 决定。",
        "- J 为 `D + AB + b`，只在循环接口使用；不使用 post-final J。",
        "- J 的训练长度为 m=10..20，损失只放在最终 T(m)=m 的受监督数字位。",
        "- m21..30 没有参与本轮继续训练或快照选择，但在此前实验中已经看过，因此只能称为本轮决策的 held-out 复评，不能称为全项目从未打开的 test set。",
        "- 这些结果证明的是固定主模型上的行为恢复与长度外推改善；是否为 circuit awakening 仍需额外的 matched causal circuit intervention。",
        "",
    ])
    (args.out_dir / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
