#!/usr/bin/env python3
"""Summarize joint m1..20 continuation of the two Addition J ranks."""

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
        continuation_dir = args.root / "continuations" / label
        selection_dir = args.root / "selection" / label
        audit_dir = args.root / "audits" / label
        continuation = json.loads((continuation_dir / "summary.json").read_text())
        selection = json.loads((selection_dir / "selection.json").read_text())
        aggregate = read_csv(selection_dir / "snapshot_aggregate_metrics.csv")
        source = next(row for row in aggregate if int(row["snapshot_update"]) == 20_000)
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

        audit_specs = {
            "trained_1_10": ("trained_l1to10_n1024", range(1, 11)),
            "trained_11_20": ("trained_l11to20_n1024", range(11, 21)),
            "heldout_21_30": ("heldout_l21to30_n512", range(21, 31)),
        }
        range_metrics: dict[str, dict[str, float]] = {}
        by_length: dict[tuple[str, int], dict[str, float]] = {}
        for range_name, (directory, expected_lengths) in audit_specs.items():
            rows = read_csv(audit_dir / directory / "endpoint_accuracy.csv")
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

        counts = {
            int(length): int(count)
            for length, count in controller[
                "controller_logical_length_example_counts"
            ].items()
        }
        no_gradient_updates_to_selected = counts.get(1, 0) // 32
        summaries.append({
            "label": label,
            "rank": int(controller["rank"]),
            "parameter_count": parameter_count,
            "source_update": 20_000,
            "selected_update": selected_update,
            "target_total_updates": 40_000,
            "source_selection_mean_digit_em": float(source["mean_digit_em"]),
            "selected_selection_mean_digit_em": float(selected["mean_digit_em"]),
            "selection_improvement": (
                float(selected["mean_digit_em"]) - float(source["mean_digit_em"])
            ),
            "source_selection_m10_em": float(source["id_gate_em"]),
            "selected_selection_m10_em": float(selected["id_gate_em"]),
            "selected_training_min_length": int(controller["controller_logical_min_length"]),
            "selected_training_max_length": int(controller["controller_logical_max_length"]),
            "no_gradient_m1_updates_to_selected": no_gradient_updates_to_selected,
            "full_digit_em_by_length": {
                str(length): by_length[("full", length)]["digit_em"]
                for length in range(1, 31)
            },
            "range_metrics": range_metrics,
            "continuation_summary": continuation,
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
            "high_sample_m1to10_full_mean_em": item["range_metrics"]["trained_1_10_full"]["mean_digit_em"],
            "high_sample_m11to20_full_mean_em": item["range_metrics"]["trained_11_20_full"]["mean_digit_em"],
            "high_sample_m21to30_full_mean_em": item["range_metrics"]["heldout_21_30_full"]["mean_digit_em"],
            "no_gradient_m1_updates_to_selected": item["no_gradient_m1_updates_to_selected"],
        })

    best = max(
        summaries,
        key=lambda item: (
            item["selected_selection_mean_digit_em"],
            item["range_metrics"]["heldout_21_30_full"]["mean_digit_em"],
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
            "checkpoint_step": 80_000,
            "shared_physical_block_layers": 3,
            "target_loop_rule": "T(m)=m",
        },
        "controller_training": {
            "parameterization": "diagonal plus low-rank affine J",
            "controller_data_lengths": [1, 20],
            "uniform_per_length_sampling": True,
            "source_total_updates": 20_000,
            "target_total_updates": 40_000,
            "additional_updates": 20_000,
            "loss": "final-only supervised-digit CE at T(m)=m",
            "post_final_j": False,
            "structural_noop": "m=1 has no inter-loop J application and therefore no J gradient",
        },
        "selection_protocol": reference_protocol,
        "candidates": summaries,
        "best_joint_m1to20_candidate": best["label"],
        "claim_boundary": (
            "All m=1..20 are included in data, online evaluation, and snapshot selection. "
            "At m=1, T(1)=1 means the inter-loop J is not invoked, so its loss cannot "
            "update J. Lengths 21..30 are excluded from this continuation and selection "
            "but were inspected in the preceding experiment."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.out_dir / "rank_comparison.csv", comparison_rows)
    write_csv(args.out_dir / "endpoint_length_metrics.csv", endpoint_rows)

    lines = [
        "# Addition LSB causal baseline：m1..20 联合训练到 40k",
        "",
        "## 结论",
        "",
        "两个 rank 均从各自选中的 20k J 继续到 40k；训练数据、在线评估和独立快照选择都覆盖 m1..20。",
        "",
        "| rank | J 参数 | 独立选中步数 | m1..20：20k源快照 | m1..20：联合最优 | 提升/百分点 | 高样本 m1..10 | 高样本 m11..20 | 高样本 m21..30 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['rank']} | {item['parameter_count']:,} | {item['selected_update']:,} | "
            f"{pct(item['source_selection_mean_digit_em'])} | "
            f"{pct(item['selected_selection_mean_digit_em'])} | "
            f"{100.0 * item['selection_improvement']:.2f} | "
            f"{pct(item['range_metrics']['trained_1_10_full']['mean_digit_em'])} | "
            f"{pct(item['range_metrics']['trained_11_20_full']['mean_digit_em'])} | "
            f"{pct(item['range_metrics']['heldout_21_30_full']['mean_digit_em'])} |"
        )
    lines.extend([
        "",
        "独立选点使用新 seed，在每个 m=1..20 上评估 1,024 个样本，并要求 m10≥99.5%。"
        "最终高样本复评再使用另一个 seed；m1..20 每长度 1,024 个样本，m21..30 每长度 512 个样本。",
        "",
        "## 实验边界",
        "",
        "- J 是循环间接口，因此 T(1)=1 的 m1 没有 J 调用：它进入数据、loss统计和选点，但不能产生 J 梯度。",
        "- m2..20 均能对 J 反传；artifact 会正确记录训练最小/最大长度为1和20，并记录m1 no-op次数。",
        "- 主模型冻结；损失只放在最终 T(m)=m 的受监督数字位；不使用 post-final J。",
        "- m21..30 没有参与本轮训练或选点，但此前已经观察过，因此不是全项目从未打开的 test set。",
        "- 本结果衡量的是联合准确率和有限外推，不单独证明 circuit awakening。",
        "",
    ])
    (args.out_dir / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
