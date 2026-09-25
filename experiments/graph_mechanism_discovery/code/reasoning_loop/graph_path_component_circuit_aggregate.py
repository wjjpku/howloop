from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _all_at_least(
    rows: Sequence[dict[str, Any]],
    key: str,
    threshold: float,
) -> bool:
    return bool(rows) and min(float(row[key]) for row in rows) >= threshold


def _all_at_most(
    rows: Sequence[dict[str, Any]],
    key: str,
    threshold: float,
) -> bool:
    return bool(rows) and max(float(row[key]) for row in rows) <= threshold


def evaluate_component_circuit_gates(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    operator_rows = [dict(row) for row in rows if row["group"] == "operator"]
    control_rows = [dict(row) for row in rows if row["group"] != "operator"]
    seed_count = len({int(row["seed"]) for row in operator_rows})
    behavior_pass = (
        seed_count >= 5
        and _all_at_least(operator_rows, "rolling_min_accuracy", 0.99)
    )
    stable_executor_role_pass = (
        behavior_pass
        and all(int(row["executor_block"]) == 1 for row in operator_rows)
        and _all_at_least(operator_rows, "executor_local_mean_drop", 0.20)
        and _all_at_least(operator_rows, "executor_transplant_drop", 0.20)
    )
    stable_initializer_role_pass = (
        behavior_pass
        and _all_at_least(operator_rows, "initializer_loop1_drop", 0.40)
        and _all_at_most(operator_rows, "initializer_later_max_drop", 0.05)
    )
    localized_count = sum(
        float(row["executor_attention_selectivity"]) >= 0.20
        for row in operator_rows
    )
    attention_localization_support = (
        seed_count >= 5 and localized_count >= 4
    )
    negative_control_pass = (
        bool(control_rows)
        and _all_at_most(control_rows, "executor_local_mean_drop", 0.05)
        and _all_at_most(control_rows, "executor_transplant_drop", 0.05)
    )
    if (
        stable_executor_role_pass
        and stable_initializer_role_pass
        and attention_localization_support
        and negative_control_pass
    ):
        classification = "staged_successor_circuit_supported"
    elif (
        stable_executor_role_pass
        and stable_initializer_role_pass
        and not control_rows
    ):
        classification = "staged_successor_circuit_without_controls"
    elif operator_rows and seed_count < 5:
        classification = "insufficient_seed_stability"
    else:
        classification = "component_role_not_established"
    return {
        "operator": {
            "seed_count": seed_count,
            "behavior_pass": behavior_pass,
            "stable_executor_role_pass": stable_executor_role_pass,
            "stable_initializer_role_pass": stable_initializer_role_pass,
            "attention_localization_support": attention_localization_support,
            "attention_localized_seed_count": localized_count,
            "minimums": {
                key: (
                    min(float(row[key]) for row in operator_rows)
                    if operator_rows
                    else None
                )
                for key in (
                    "rolling_min_accuracy",
                    "initializer_loop1_drop",
                    "executor_local_mean_drop",
                    "executor_transplant_drop",
                    "executor_attention_selectivity",
                )
            },
            "initializer_later_max": (
                max(
                    float(row["initializer_later_max_drop"])
                    for row in operator_rows
                )
                if operator_rows
                else None
            ),
        },
        "control_count": len(control_rows),
        "negative_control_pass": negative_control_pass,
        "classification": classification,
    }


def summarize_component_run(
    summary: dict[str, Any],
    *,
    group: str,
    seed: int,
    name: str,
) -> dict[str, Any]:
    local_drop = np.asarray(summary["local_transition_drop"], dtype=np.float64)
    if local_drop.ndim != 3:
        raise ValueError("local_transition_drop must have loop, block, and head axes")
    transplant_drop = np.asarray(
        summary["transplant_next_step"]["head_ablation_accuracy_drop"],
        dtype=np.float64,
    )
    selectivity = np.asarray(
        summary["attention_alignment"]["answer_destination_selectivity"],
        dtype=np.float64,
    )
    if transplant_drop.shape != local_drop.shape[1:]:
        raise ValueError("transplant and local head axes do not match")
    if selectivity.shape != local_drop.shape:
        raise ValueError("attention and local drop axes do not match")

    initializer_head = int(np.argmax(local_drop[0, 0]))
    executor_block, executor_head = np.unravel_index(
        int(np.argmax(transplant_drop)),
        transplant_drop.shape,
    )
    rolling = np.asarray(summary["baseline_rolling_accuracy"], dtype=np.float64)
    return {
        "group": group,
        "seed": seed,
        "name": name,
        "rolling_min_accuracy": float(rolling.min()),
        "initializer_block": 0,
        "initializer_head": initializer_head,
        "initializer_loop1_drop": float(local_drop[0, 0, initializer_head]),
        "initializer_later_max_drop": float(
            local_drop[1:, 0, initializer_head].max()
            if local_drop.shape[0] > 1
            else 0.0
        ),
        "executor_block": int(executor_block),
        "executor_head": int(executor_head),
        "executor_local_mean_drop": float(
            local_drop[:, executor_block, executor_head].mean()
        ),
        "executor_transplant_drop": float(
            transplant_drop[executor_block, executor_head]
        ),
        "executor_attention_selectivity": float(
            selectivity[:, executor_block, executor_head].mean()
        ),
        "transplant_baseline_accuracy": float(
            summary["transplant_next_step"]["baseline_accuracy"]
        ),
    }


def parse_spec(text: str) -> tuple[str, int, str, Path]:
    if "=" not in text or ":" not in text.split("=", 1)[0]:
        raise ValueError("run spec must be GROUP:SEED[:NAME]=summary.json")
    identity, path_text = text.split("=", 1)
    pieces = identity.split(":", 2)
    if len(pieces) < 2:
        raise ValueError("run spec must include group and seed")
    group = pieces[0]
    seed = int(pieces[1])
    name = pieces[2] if len(pieces) == 3 else f"{group}_seed{seed}"
    return group, seed, name, Path(path_text)


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    labels = [str(row["name"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(rows)), 5))
    ax.bar(
        x - width,
        [row["initializer_loop1_drop"] for row in rows],
        width,
        label="initializer drop at loop 1",
    )
    ax.bar(
        x,
        [row["executor_local_mean_drop"] for row in rows],
        width,
        label="executor local drop",
    )
    ax.bar(
        x + width,
        [row["executor_transplant_drop"] for row in rows],
        width,
        label="executor transplant drop",
    )
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("accuracy drop")
    ax.set_ylim(-0.05, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate graph successor component-circuit results."
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in args.run:
        group, seed, name, path = parse_spec(spec)
        rows.append(
            summarize_component_run(
                json.loads(path.read_text(encoding="utf-8")),
                group=group,
                seed=seed,
                name=name,
            )
        )
    gates = evaluate_component_circuit_gates(rows)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps({"rows": rows, "gates": gates}, indent=2),
        encoding="utf-8",
    )
    _write_rows(args.out_dir / "rows.csv", rows)
    _save_figure(args.out_dir / "component_role_drops.png", rows)


if __name__ == "__main__":
    main()
