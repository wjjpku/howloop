from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


RUN_PATTERN = re.compile(
    r"^stage_(?P<arm>l1|shared2|unshared2)_p(?P<p>\d+)_"
    r"d(?P<width>\d+)_seed(?P<seed>\d+)$"
)
ARMS = ("l1", "shared2", "unshared2")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_rows(root: Path) -> list[dict[str, Any]]:
    circuit_by_run: dict[str, dict[str, Any]] = {}
    analysis_root = root / "analysis"
    if analysis_root.exists():
        for path in sorted(analysis_root.glob("*/summary.json")):
            circuit_by_run[path.parent.name] = _load_json(path)

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_root = root / arm
        if not arm_root.exists():
            continue
        for path in sorted(arm_root.glob("*/summary.json")):
            summary = _load_json(path)
            run_name = str(summary.get("run_name", path.parent.name))
            match = RUN_PATTERN.fullmatch(run_name)
            if match is None:
                continue
            heldout = summary["final_metrics"]["heldout"]
            matrix = heldout["accuracy_matrix"]
            diagnostics = summary.get("diagnostics", {})
            circuit = circuit_by_run.get(run_name, {})
            scorecard = circuit.get("scorecard", {})
            endpoint = float(heldout["trained_endpoint_accuracy"])
            extra_endpoint = heldout.get("extra_loop_endpoint_accuracy")
            rows.append(
                {
                    "run_name": run_name,
                    "arm": match.group("arm"),
                    "p": int(match.group("p")),
                    "width": int(match.group("width")),
                    "seed": int(match.group("seed")),
                    "parameter_count": int(summary["parameter_count"]),
                    "best_step": int(summary["best_step"]),
                    "heldout_endpoint_accuracy": endpoint,
                    "loop1_endpoint_accuracy": float(matrix[0][1]),
                    "loop1_partial_sum_accuracy": float(
                        heldout["loop1_sum_accuracy"]
                    ),
                    "extra_loop_endpoint_accuracy": (
                        float(extra_endpoint)
                        if extra_endpoint is not None
                        else None
                    ),
                    "extra_loop_damage": (
                        endpoint - float(extra_endpoint)
                        if extra_endpoint is not None
                        else None
                    ),
                    "hybrid_transplant_accuracy": (
                        float(diagnostics["hybrid_transplant_accuracy"])
                        if diagnostics.get("status") == "ok"
                        and diagnostics.get("hybrid_transplant_accuracy") is not None
                        else None
                    ),
                    "hybrid_transplant_c_only_accuracy": (
                        float(diagnostics["hybrid_transplant_c_only_accuracy"])
                        if diagnostics.get("status") == "ok"
                        and diagnostics.get("hybrid_transplant_c_only_accuracy")
                        is not None
                        else None
                    ),
                    "shuffled_workspace_hybrid_accuracy": (
                        float(diagnostics["shuffled_workspace_hybrid_accuracy"])
                        if diagnostics.get("status") == "ok"
                        and diagnostics.get(
                            "shuffled_workspace_hybrid_accuracy"
                        )
                        is not None
                        else None
                    ),
                    "reset_before_second_loop_accuracy": (
                        float(diagnostics["reset_before_second_loop_accuracy"])
                        if diagnostics.get("status") == "ok"
                        and diagnostics.get(
                            "reset_before_second_loop_accuracy"
                        )
                        is not None
                        else None
                    ),
                    "circuit_behavior_gate": scorecard.get("behavior_gate"),
                    "circuit_stage_boundary_gate": scorecard.get(
                        "stage_boundary_gate"
                    ),
                    "circuit_stage_boundary_c_only_gate": scorecard.get(
                        "stage_boundary_c_only_gate"
                    ),
                    "both_loops_causally_necessary": scorecard.get(
                        "both_loops_causally_necessary"
                    ),
                    "stable_operator_repeat_rejected": scorecard.get(
                        "stable_operator_repeat_rejected"
                    ),
                    "summary_path": str(path.resolve()),
                }
            )
    return rows


def _aggregate_arm_width(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), int(row["width"]))].append(row)
    result: list[dict[str, Any]] = []
    for (arm, width), group in sorted(grouped.items()):
        endpoints = [float(row["heldout_endpoint_accuracy"]) for row in group]
        extra_damage = [
            float(row["extra_loop_damage"])
            for row in group
            if row["extra_loop_damage"] is not None
        ]
        hybrid = [
            float(row["hybrid_transplant_accuracy"])
            for row in group
            if row["hybrid_transplant_accuracy"] is not None
        ]
        result.append(
            {
                "arm": arm,
                "width": width,
                "seeds": sorted(int(row["seed"]) for row in group),
                "seed_count": len(group),
                "endpoint_mean": _mean(endpoints),
                "endpoint_min": min(endpoints),
                "endpoint_max": max(endpoints),
                "loop1_endpoint_mean": _mean(
                    [float(row["loop1_endpoint_accuracy"]) for row in group]
                ),
                "loop1_partial_sum_mean": _mean(
                    [float(row["loop1_partial_sum_accuracy"]) for row in group]
                ),
                "extra_loop_damage_mean": _mean(extra_damage),
                "extra_loop_damage_min": min(extra_damage)
                if extra_damage
                else None,
                "hybrid_transplant_mean": _mean(hybrid),
                "parameter_counts": sorted(
                    {int(row["parameter_count"]) for row in group}
                ),
            }
        )
    return result


def _paired_contrasts(
    rows: list[dict[str, Any]],
    *,
    min_confirmation_seeds: int,
) -> list[dict[str, Any]]:
    by_width_arm_seed: dict[int, dict[str, dict[int, dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for row in rows:
        by_width_arm_seed[int(row["width"])][str(row["arm"])][
            int(row["seed"])
        ] = row

    contrasts: list[dict[str, Any]] = []
    for width, arm_map in sorted(by_width_arm_seed.items()):
        seed_sets = [set(arm_map.get(arm, {})) for arm in ARMS]
        paired = sorted(set.intersection(*seed_sets)) if seed_sets else []
        triples = [
            {
                arm: arm_map[arm][seed]
                for arm in ARMS
            }
            for seed in paired
        ]
        gaps = [
            float(group["shared2"]["heldout_endpoint_accuracy"])
            - float(group["l1"]["heldout_endpoint_accuracy"])
            for group in triples
        ]
        shared_endpoint = [
            float(group["shared2"]["heldout_endpoint_accuracy"])
            for group in triples
        ]
        unshared_endpoint = [
            float(group["unshared2"]["heldout_endpoint_accuracy"])
            for group in triples
        ]
        l1_endpoint = [
            float(group["l1"]["heldout_endpoint_accuracy"])
            for group in triples
        ]
        repeat_damage = [
            float(group["shared2"]["extra_loop_damage"])
            for group in triples
            if group["shared2"]["extra_loop_damage"] is not None
        ]
        hybrid = [
            float(group["shared2"]["hybrid_transplant_accuracy"])
            for group in triples
            if group["shared2"]["hybrid_transplant_accuracy"] is not None
        ]
        same_budget = bool(triples) and all(
            int(group["l1"]["parameter_count"])
            == int(group["shared2"]["parameter_count"])
            for group in triples
        )
        untied_has_more_parameters = bool(triples) and all(
            int(group["unshared2"]["parameter_count"])
            > int(group["shared2"]["parameter_count"])
            for group in triples
        )
        circuit_complete = bool(triples) and all(
            group["shared2"]["both_loops_causally_necessary"] is not None
            and group["shared2"]["stable_operator_repeat_rejected"] is not None
            for group in triples
        )
        circuit_gate = circuit_complete and all(
            bool(group["shared2"]["both_loops_causally_necessary"])
            and bool(group["shared2"]["stable_operator_repeat_rejected"])
            for group in triples
        )
        stage_boundary_complete = len(hybrid) == len(triples) and bool(triples)
        behavioral_window = bool(triples) and (
            max(l1_endpoint) <= 0.80
            and min(shared_endpoint) >= 0.98
            and min(unshared_endpoint) >= 0.98
            and min(gaps) >= 0.20
        )
        repeat_rejection = (
            len(repeat_damage) == len(triples)
            and bool(triples)
            and min(repeat_damage) >= 0.10
            and statistics.fmean(repeat_damage) >= 0.20
        )
        stage_boundary = (
            stage_boundary_complete and min(hybrid) >= 0.90
        )
        pilot_gate = (
            behavioral_window
            and same_budget
            and untied_has_more_parameters
            and repeat_rejection
            and stage_boundary
            and circuit_gate
        )
        contrasts.append(
            {
                "width": width,
                "paired_seeds": paired,
                "paired_seed_count": len(paired),
                "l1_endpoint_mean": _mean(l1_endpoint),
                "shared2_endpoint_mean": _mean(shared_endpoint),
                "unshared2_endpoint_mean": _mean(unshared_endpoint),
                "shared2_minus_l1_endpoint_mean": _mean(gaps),
                "shared2_minus_l1_endpoint_min": min(gaps) if gaps else None,
                "shared2_repeat_damage_mean": _mean(repeat_damage),
                "shared2_repeat_damage_min": (
                    min(repeat_damage) if repeat_damage else None
                ),
                "shared2_hybrid_transplant_mean": _mean(hybrid),
                "same_parameter_budget": same_budget,
                "untied_has_more_parameters": untied_has_more_parameters,
                "behavioral_window": behavioral_window,
                "repeat_rejection": repeat_rejection,
                "stage_boundary": stage_boundary,
                "circuit_gate": circuit_gate,
                "pilot_gate": pilot_gate,
                "confirmation_gate": (
                    pilot_gate and len(paired) >= min_confirmation_seeds
                ),
            }
        )
    return contrasts


def _write_seed_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        key for key in rows[0] if key != "summary_path"
    ] + ["summary_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    arm_width_rows: list[dict[str, Any]],
    selected_width: int | None,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"l1": "#4C78A8", "shared2": "#F58518", "unshared2": "#54A24B"}
    labels = {
        "l1": "one shared application",
        "shared2": "two shared applications",
        "unshared2": "two untied applications",
    }
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for arm in ARMS:
        group = [row for row in arm_width_rows if row["arm"] == arm]
        if not group:
            continue
        axis.plot(
            [row["width"] for row in group],
            [row["endpoint_mean"] for row in group],
            marker="o",
            linewidth=2,
            color=colors[arm],
            label=labels[arm],
        )
        axis.fill_between(
            [row["width"] for row in group],
            [row["endpoint_min"] for row in group],
            [row["endpoint_max"] for row in group],
            color=colors[arm],
            alpha=0.12,
        )
    if selected_width is not None:
        axis.axvline(
            selected_width,
            color="#B279A2",
            linestyle="--",
            linewidth=1.5,
            label=f"selected width = {selected_width}",
        )
    axis.axhline(0.98, color="0.5", linestyle=":", linewidth=1)
    axis.set(
        xlabel="workspace width",
        ylabel="held-out endpoint accuracy",
        ylim=(-0.02, 1.03),
    )
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def aggregate_stage_runs(
    root: Path,
    *,
    out_dir: Path | None = None,
    min_confirmation_seeds: int = 3,
) -> dict[str, Any]:
    if min_confirmation_seeds < 1:
        raise ValueError("min_confirmation_seeds must be positive")
    rows = _discover_rows(root)
    arm_width = _aggregate_arm_width(rows)
    contrasts = _paired_contrasts(
        rows,
        min_confirmation_seeds=min_confirmation_seeds,
    )
    candidates = [
        row for row in contrasts if row["pilot_gate"]
    ]
    selected = (
        sorted(
            candidates,
            key=lambda row: (
                -float(row["shared2_minus_l1_endpoint_mean"]),
                int(row["width"]),
            ),
        )[0]["width"]
        if candidates
        else None
    )
    result = {
        "root": str(root.resolve()),
        "run_count": len(rows),
        "min_confirmation_seeds": min_confirmation_seeds,
        "selected_width": selected,
        "seed_rows": rows,
        "arm_width_rows": arm_width,
        "width_contrasts": contrasts,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "aggregate.json").write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        _write_seed_csv(rows, out_dir / "seed_rows.csv")
        _plot_summary(
            arm_width,
            int(selected) if selected is not None else None,
            out_dir / "width_scan.png",
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the preregistered virtual-depth width scan and select "
            "only paired one-layer/shared-two/untied-two windows."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-confirmation-seeds", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = aggregate_stage_runs(
        args.root,
        out_dir=args.out_dir,
        min_confirmation_seeds=args.min_confirmation_seeds,
    )
    print(
        json.dumps(
            {
                "run_count": result["run_count"],
                "selected_width": result["selected_width"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
