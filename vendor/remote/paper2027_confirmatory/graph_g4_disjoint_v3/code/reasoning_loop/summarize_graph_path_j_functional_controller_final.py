"""Build compact final tables and a reader-facing overview for the seven-J study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expanded_eval = args.root / "eval_multiseed_all_k2_to_k16_final" / "aggregate.csv"
    functional_source = (
        expanded_eval
        if expanded_eval.exists()
        else args.root / "eval_multiseed_all_final" / "aggregate.csv"
    )
    functional_all = _read_csv(functional_source)
    functional = [
        row
        for row in functional_all
        if row["family"] == "equivalent_words"
        and int(row["back_count"])
        in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32}
    ]
    _write_csv(args.out_dir / "functional_composition.csv", functional)

    long_all = _read_csv(
        args.root / "dynamic_parent_main_stage24" / "long_schedule_comparison.csv"
    )
    long_rows = [
        row
        for row in long_all
        if row["bank"] in {"parent", "main", "stage24"}
        and row["schedule"] in {"FFJ", "F2_then_FJ", "FFFFJJJJ", "F7_then_FJ"}
    ]
    _write_csv(args.out_dir / "long_loop.csv", long_rows)

    age_all = _read_csv(
        args.root / "functional_age_probe_all_final" / "functional_age_probe.csv"
    )
    age_keys = {
        ("refit_natural", "natural", "held_out_graphs"),
        ("frozen_natural", "stage24", "post_J_held_out_graphs"),
        ("controlled_stage24", "stage24", "post_J_held_out_graphs"),
        (
            "controlled_unseen_paths_stage24",
            "stage24",
            "post_J_unseen_action_words_and_graphs",
        ),
        (
            "controlled_joint_banks_unseen_paths",
            "stage24",
            "post_J_unseen_action_words_and_graphs",
        ),
    }
    age_rows = [
        row
        for row in age_all
        if (row["probe"], row["test_bank"], row["test_domain"]) in age_keys
    ]
    _write_csv(args.out_dir / "age_probe.csv", age_rows)

    ablation = _read_csv(
        args.root / "circuit_drift_all_final" / "circuit_ablation_summary.csv"
    )
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in ablation:
        if row["bank"] not in {"parent", "stage24"}:
            continue
        if row["schedule"] not in {"natural_F7", "k8_left", "k8_right", "k12_left", "k12_right"}:
            continue
        grouped[(row["bank"], row["schedule"], row["component"])].append(
            float(row["shared_correct_margin_drop"])
        )
    circuit_rows = [
        {
            "bank": bank,
            "schedule": schedule,
            "component": component,
            "shared_correct_margin_drop_mean_over_F_events": float(np.mean(values)),
            "F_events": len(values),
        }
        for (bank, schedule, component), values in sorted(grouped.items())
    ]
    _write_csv(args.out_dir / "circuit_component_load.csv", circuit_rows)

    matrix = json.loads(
        (args.root / "stage24_vs_parent_matrix_math" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    semantics = json.loads((args.root / "semantics_audit.json").read_text(encoding="utf-8"))

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    for bank in ("parent", "main", "stage16", "stage24"):
        selected = sorted(
            [row for row in functional if row["bank"] == bank],
            key=lambda row: int(row["back_count"]),
        )
        axes[0, 0].plot(
            [int(row["back_count"]) for row in selected],
            [float(row["accuracy_mean"]) for row in selected],
            marker="o",
            label=bank,
        )
    axes[0, 0].axhline(0.95, color="gray", linestyle=":")
    axes[0, 0].set(title="Unseen action-word composition", xlabel="J calls", ylabel="accuracy", ylim=(0, 1.03))
    axes[0, 0].legend()

    schedules = ["FFJ", "F2_then_FJ", "FFFFJJJJ"]
    x = np.arange(len(schedules))
    width = 0.25
    for index, bank in enumerate(("parent", "main", "stage24")):
        lookup = {(row["bank"], row["schedule"]): row for row in long_rows}
        axes[0, 1].bar(
            x + (index - 1) * width,
            [float(lookup[(bank, schedule)]["H24_accuracy"]) for schedule in schedules],
            width,
            label=bank,
        )
    axes[0, 1].set_xticks(x, schedules)
    axes[0, 1].set(title="Long schedules at graph step 24", ylabel="accuracy", ylim=(0, 1.03))
    axes[0, 1].legend()

    age_labels = [
        "natural→natural",
        "natural→J state",
        "stage24→J state",
        "stage24→unseen path",
        "joint→unseen path",
    ]
    axes[1, 0].bar(
        np.arange(len(age_rows)),
        [float(row["rounded_accuracy"]) for row in age_rows],
    )
    axes[1, 0].set_xticks(np.arange(len(age_rows)), age_labels, rotation=20, ha="right")
    axes[1, 0].set(title="Linear logical-age readout", ylabel="rounded accuracy", ylim=(0, 1.03))

    circuit_lookup = {
        (row["bank"], row["schedule"], row["component"]): float(
            row["shared_correct_margin_drop_mean_over_F_events"]
        )
        for row in circuit_rows
    }
    circuit_schedules = ["k8_left", "k8_right", "k12_left", "k12_right"]
    for index, bank in enumerate(("parent", "stage24")):
        axes[1, 1].bar(
            np.arange(len(circuit_schedules)) + (index - 0.5) * 0.36,
            [
                circuit_lookup[(bank, schedule, "B2.H0.context_answer")]
                for schedule in circuit_schedules
            ],
            0.36,
            label=bank,
        )
    axes[1, 1].set_xticks(np.arange(len(circuit_schedules)), circuit_schedules)
    axes[1, 1].set(
        title="Causal load of successor-routing head B2.H0",
        ylabel="shared-correct margin drop",
    )
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    figure.savefig(args.out_dir / "functional_controller_overview.png", bbox_inches="tight")
    plt.close(figure)

    stage24_row = next(
        row
        for row in functional
        if row["bank"] == "stage24" and int(row["back_count"]) == 24
    )
    parent_row = next(
        row
        for row in functional
        if row["bank"] == "parent" and int(row["back_count"]) == 24
    )
    result = {
        "status": "complete",
        "functional_evaluation_source": str(functional_source),
        "selected_checkpoint": "stage24",
        "stage24_k24_accuracy": float(stage24_row["accuracy_mean"]),
        "stage24_k24_path_minimum": float(stage24_row["accuracy_min"]),
        "parent_k24_accuracy": float(parent_row["accuracy_mean"]),
        "reliable_lifespan_accuracy_at_least_0.95": 16,
        "state_target_audit": semantics,
        "matrix_checkpoint_stability": matrix["checkpoint_stability"],
        "claim_boundary": (
            "One frozen D8L8 seed0 backbone on random N=8 permutation graphs; "
            "finite task-functional control, not a global inverse or infinite overloop."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
