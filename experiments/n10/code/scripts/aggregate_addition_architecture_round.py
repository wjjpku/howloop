from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


CONFIG_ORDER = (
    "l3h8t11_ref10k",
    "l2h8t11",
    "l3h4t11",
    "l2h4t11",
    "l3h8t8",
    "l3h8t6",
    "l2h4t8",
)
CONFIG_LABELS = {
    "l3h8t11_ref10k": "3L8H-T11 ref",
    "l2h8t11": "2L8H-T11",
    "l3h4t11": "3L4H-T11",
    "l2h4t11": "2L4H-T11",
    "l3h8t8": "3L8H-T8",
    "l3h8t6": "3L8H-T6",
    "l2h4t8": "2L4H-T8",
}
CONFIG_META = {
    "l3h8t11_ref10k": (3, 8, 11),
    "l2h8t11": (2, 8, 11),
    "l3h4t11": (3, 4, 11),
    "l2h4t11": (2, 4, 11),
    "l3h8t8": (3, 8, 8),
    "l3h8t6": (3, 8, 6),
    "l2h4t8": (2, 4, 8),
}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def _parameter_count(block_layers: int) -> int:
    # The dense attention matrices do not change shape with head count.
    return 1_792 + 789_760 * block_layers + 512 + 1_542


def _load_payloads(round_root: Path, reference_root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    payloads: dict[str, dict[str, list[dict[str, Any]]]] = {}
    reference = []
    for seed in range(3):
        path = reference_root / f"seed{seed}" / "step_010000" / "summary.json"
        value = json.loads(path.read_text())
        if value.get("status") != "complete":
            raise ValueError(f"incomplete reference: {path}")
        reference.append(value)
    payloads["l3h8t11_ref10k"] = {"aligned": reference, "registered": reference}

    for config in CONFIG_ORDER[1:]:
        payloads[config] = {"aligned": [], "registered": []}
        for seed in range(3):
            for rule in ("aligned", "registered"):
                path = round_root / "eval" / f"{config}_seed{seed}_{rule}" / "summary.json"
                value = json.loads(path.read_text())
                if value.get("status") != "complete":
                    raise ValueError(f"incomplete result: {path}")
                if int(value["backbone_seed"]) != seed:
                    raise ValueError(f"seed mismatch: {path}")
                payloads[config][rule].append(value)
    return payloads


def _payload_curves(payload: dict[str, Any]) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["rows"]:
        by_length[int(row["length"])].append(row)
    target: dict[int, dict[str, float]] = {}
    nearby: dict[int, dict[str, float]] = {}
    for length, rows in by_length.items():
        target[length] = next(row for row in rows if int(row["step_offset"]) == 0)
        nearby[length] = max(
            [row for row in rows if -2 <= int(row["step_offset"]) <= 2],
            key=lambda row: float(row["actual_answer_exact_match"]),
        )
    return target, nearby


def _horizon(curve: dict[int, float], threshold: float = 0.90) -> int:
    horizon = 9
    for length in range(10, 41):
        if curve.get(length, -1.0) < threshold:
            break
        horizon = length
    return horizon


def aggregate(payloads: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for config in CONFIG_ORDER:
        layers, heads, train_loops = CONFIG_META[config]
        curves: dict[str, list[tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]]] = {
            "aligned": [],
            "registered": [],
        }
        for rule in ("aligned", "registered"):
            curves[rule] = [_payload_curves(payload) for payload in payloads[config][rule]]
            for length in range(10, 41):
                for kind_index, kind in enumerate(("target", "nearby")):
                    source = [pair[kind_index][length] for pair in curves[rule]]
                    for metric in (
                        "strict_exact_match",
                        "actual_answer_exact_match",
                        "carry_accuracy",
                        "actual_answer_token_accuracy",
                        "mean_correct_low_order_digits",
                        "mean_actual_sequence_min_margin",
                    ):
                        values = [float(row[metric]) for row in source]
                        mean, std = _mean_std(values)
                        curve_rows.append(
                            {
                                "config": config,
                                "label": CONFIG_LABELS[config],
                                "rule": rule,
                                "readout": kind,
                                "length": length,
                                "metric": metric,
                                "mean": mean,
                                "std": std,
                                "min": min(values),
                                "max": max(values),
                            }
                        )

        aligned_targets = [pair[0] for pair in curves["aligned"]]
        aligned_nearby = [pair[1] for pair in curves["aligned"]]
        registered_targets = [pair[0] for pair in curves["registered"]]
        id_values = [
            float(curve[10]["actual_answer_exact_match"])
            for curve in aligned_targets
        ]
        id_strict = [
            float(curve[10]["strict_exact_match"])
            for curve in aligned_targets
        ]
        id_carry = [
            float(curve[10]["carry_accuracy"])
            for curve in aligned_targets
        ]
        id_token = [
            float(curve[10]["actual_answer_token_accuracy"])
            for curve in aligned_targets
        ]
        id_frontier = [
            float(curve[10]["mean_correct_low_order_digits"])
            for curve in aligned_targets
        ]
        id_margin = [
            float(curve[10]["mean_actual_sequence_min_margin"])
            for curve in aligned_targets
        ]
        aligned_horizons = [
            _horizon({n: float(row["actual_answer_exact_match"]) for n, row in curve.items()})
            for curve in aligned_targets
        ]
        nearby_horizons = [
            _horizon({n: float(row["actual_answer_exact_match"]) for n, row in curve.items()})
            for curve in aligned_nearby
        ]
        registered_n10 = [
            float(curve[10]["actual_answer_exact_match"])
            for curve in registered_targets
        ]
        config_rows.append(
            {
                "config": config,
                "label": CONFIG_LABELS[config],
                "block_layers": layers,
                "heads": heads,
                "train_loops": train_loops,
                "effective_depth": layers * train_loops,
                "parameter_count": _parameter_count(layers),
                "id_aligned_actual_em_mean": fmean(id_values),
                "id_aligned_actual_em_std": stdev(id_values),
                "id_aligned_actual_em_min": min(id_values),
                "id_aligned_strict_em_mean": fmean(id_strict),
                "id_aligned_carry_accuracy_mean": fmean(id_carry),
                "id_aligned_carry_accuracy_min": min(id_carry),
                "id_aligned_token_accuracy_mean": fmean(id_token),
                "id_aligned_low_order_frontier_mean": fmean(id_frontier),
                "id_aligned_sequence_min_margin_mean": fmean(id_margin),
                "registered_n10_actual_em_mean": fmean(registered_n10),
                "aligned_horizon90_mean": fmean(aligned_horizons),
                "aligned_horizon90_min": min(aligned_horizons),
                "nearby_horizon90_mean": fmean(nearby_horizons),
                "eligible_all_seeds_id98": min(id_values) >= 0.98,
            }
        )
    return config_rows, curve_rows


def make_figure(config_rows: Sequence[dict[str, Any]], curve_rows: Sequence[dict[str, Any]], path: Path) -> None:
    lookup = {
        (row["config"], row["rule"], row["readout"], int(row["length"]), row["metric"]): row
        for row in curve_rows
    }
    colors = plt.cm.tab10(np.arange(len(CONFIG_ORDER)))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(CONFIG_ORDER))
    width = 0.25
    for offset, field, label, color in (
        (-width, "id_aligned_actual_em_mean", "sum-digit EM", "#4c78a8"),
        (0.0, "id_aligned_token_accuracy_mean", "sum-digit token acc", "#f58518"),
        (width, "id_aligned_carry_accuracy_mean", "natural carry acc", "#54a24b"),
    ):
        ax.bar(
            x + offset,
            [float(row[field]) for row in config_rows],
            width,
            color=color,
            alpha=0.88,
            label=label,
        )
    ax.set_xticks(x, [CONFIG_LABELS[c] for c in CONFIG_ORDER], rotation=30, ha="right")
    ax.set(title="What has been learned by 10k updates?", ylabel="n=10 accuracy", ylim=(-0.02, 1.02))
    ax.legend(fontsize=8, frameon=False)

    for ax, metric, title in (
        (axes[0, 1], "actual_answer_exact_match", "Endpoint-aligned full sum EM"),
        (axes[1, 0], "actual_answer_token_accuracy", "Endpoint-aligned sum-digit token accuracy"),
        (axes[1, 1], "carry_accuracy", "Registered T(n)=n+1 natural carry"),
    ):
        for color, config in zip(colors, CONFIG_ORDER):
            values = [
                lookup[(
                    config,
                    "registered" if metric == "carry_accuracy" else "aligned",
                    "target",
                    n,
                    metric,
                )]["mean"]
                for n in range(10, 41)
            ]
            ax.plot(range(10, 41), values, color=color, lw=1.8, label=CONFIG_LABELS[config])
        ax.axvline(10, color="0.3", ls=":")
        ax.set(title=title, xlabel="logical length n", ylabel="accuracy", ylim=(-0.02, 1.02))
    axes[1, 1].legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Addition fixed-n10: 10k architecture/loop screening", fontsize=15)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_report(config_rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Addition architecture round",
        "",
        "All new models use 10,000 optimizer updates, batch 64, the first 10,000 updates of the released 100,001-step schedule, fixed logical length n=10, and final-only full answer-region CE. Eligibility requires every independently trained backbone seed to reach at least 0.98 sum-digit EM at its trained endpoint.",
        "",
        "| config | params | depth | ID EM | token acc | carry acc | low-order frontier | H@90 | eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in config_rows:
        lines.append(
            f"| {row['label']} | {int(row['parameter_count']):,} | {int(row['effective_depth'])} | "
            f"{float(row['id_aligned_actual_em_mean']):.3f} | {float(row['id_aligned_token_accuracy_mean']):.3f} | "
            f"{float(row['id_aligned_carry_accuracy_mean']):.3f} | "
            f"{float(row['id_aligned_low_order_frontier_mean']):.2f} | "
            f"{float(row['aligned_horizon90_mean']):.1f} | "
            f"{'yes' if row['eligible_all_seeds_id98'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payloads = _load_payloads(args.round_root, args.reference_root)
    config_rows, curve_rows = aggregate(payloads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "config_summary.csv", config_rows)
    _write_csv(args.out_dir / "curves.csv", curve_rows)
    make_figure(config_rows, curve_rows, args.out_dir / "architecture_round.png")
    report = make_report(config_rows)
    (args.out_dir / "REPORT.md").write_text(report)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(
            {"status": "complete", "configs": config_rows, "curves": curve_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
