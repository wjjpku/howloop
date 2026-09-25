from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: str | int | float | None) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except ValueError:
        return float("nan")


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _max(values: list[float]) -> float:
    usable = _finite(values)
    return max(usable) if usable else float("nan")


def _median(values: list[float]) -> float:
    usable = _finite(values)
    return statistics.median(usable) if usable else float("nan")


def _family_seed(name: str) -> tuple[str, int]:
    if "_seed" in name:
        family, raw_seed = name.rsplit("_seed", 1)
    elif "_pairseed" in name:
        family, raw_seed = name.rsplit("_pairseed", 1)
    else:
        raise ValueError(f"cannot parse run name: {name}")
    return family, int(raw_seed)


def _site_bundle_metrics(
    rows: list[dict[str, str]],
    fingerprint_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, float]], int]:
    baseline_next = {
        row["site"]: _float(row["baseline_post_mlp_next_accuracy"])
        for row in fingerprint_rows
        if row["head"] == "all"
    }
    by_site: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_site[row["site"]][row["condition"]] = row
    metrics: dict[str, dict[str, float]] = {}
    local_pass_count = 0
    for site, conditions in by_site.items():
        random = conditions.get("random_edge")
        current = conditions.get("path_edge_offset_0")
        future = [
            row
            for condition, row in conditions.items()
            if condition.startswith("path_edge_offset_")
            and condition != "path_edge_offset_0"
        ]
        bundle = conditions.get("path_edge_bundle")
        random_drop = (
            _float(random["endpoint_accuracy_drop"])
            if random is not None
            else float("nan")
        )
        current_drop = (
            _float(current["endpoint_accuracy_drop"])
            if current is not None
            else float("nan")
        )
        future_drop = _max(
            [_float(row["endpoint_accuracy_drop"]) for row in future]
        )
        bundle_drop = (
            _float(bundle["endpoint_accuracy_drop"])
            if bundle is not None
            else float("nan")
        )
        current_next_drop = float("nan")
        random_next_drop = float("nan")
        if site in baseline_next:
            if current is not None:
                current_next_drop = baseline_next[site] - _float(
                    current["post_mlp_next_accuracy"]
                )
            if random is not None:
                random_next_drop = baseline_next[site] - _float(
                    random["post_mlp_next_accuracy"]
                )
        local_specificity = current_next_drop - random_next_drop
        if math.isfinite(local_specificity) and local_specificity >= 0.10:
            local_pass_count += 1
        metrics[site] = {
            "current_edge_endpoint_specificity": current_drop - random_drop,
            "future_edge_endpoint_specificity": future_drop - random_drop,
            "bundle_endpoint_specificity": bundle_drop - random_drop,
            "current_edge_next_specificity": local_specificity,
        }
    return metrics, local_pass_count


def summarize_run(
    run_dir: Path,
    *,
    prior_executor_runs: set[str],
) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text())
    name = summary["name"]
    family, seed = _family_seed(name)
    fingerprint = _read_csv(run_dir / "induction_fingerprint_rows.csv")
    bundle = _read_csv(run_dir / "path_bundle_ablation_rows.csv")
    writer = _read_csv(run_dir / "writer_dependency_rows.csv")
    depth = _read_csv(run_dir / "depth_phase_patch_rows.csv")
    numeric_heads = [
        row
        for row in fingerprint
        if row["head"] != "all"
        and _float(row["decoded_input_accuracy"]) >= 0.50
    ]
    bundle_by_site, local_site_count = _site_bundle_metrics(
        bundle, fingerprint
    )
    writer_drop_by_loop: dict[int, float] = {}
    writer_component_drop_by_loop: dict[int, dict[str, float]] = defaultdict(
        dict
    )
    for row in writer:
        loop = int(row["loop"])
        condition = row["condition"]
        drop = _float(row["endpoint_accuracy_drop"])
        writer_component_drop_by_loop[loop][condition] = drop
        if condition == "zero_B1_attention_and_mlp_destination":
            writer_drop_by_loop[loop] = drop
    canonical_sites: set[tuple[str, str]] = set()
    for row in numeric_heads:
        site = row["site"]
        loop = int(row["loop"])
        local_specificity = bundle_by_site.get(site, {}).get(
            "current_edge_next_specificity", float("nan")
        )
        if (
            _float(row["attention_match_gap"]) >= 0.02
            and _float(row["ov_copy_next_accuracy"]) >= 0.50
            and _float(row["clamp_post_attention_next_accuracy"]) >= 0.50
            and local_specificity >= 0.10
            and writer_drop_by_loop.get(loop, 0.0) >= 0.10
        ):
            canonical_sites.add((site, row["head"]))

    def depth_max(*components: str) -> float:
        return _max(
            [
                _float(row["specific_recovery"])
                for row in depth
                if row["component"] in components
                and row.get("head", "all") == "all"
            ]
        )

    head_depth: dict[tuple[str, str], float] = {}
    for row in depth:
        if row.get("head", "all") == "all":
            continue
        key = (row["site"], row["head"])
        head_depth[key] = max(
            head_depth.get(key, float("-inf")),
            _float(row["specific_recovery"]),
        )
    induction_effective_heads = {
        (row["site"], row["head"])
        for row in numeric_heads
        if _float(row["attention_match_gap"]) >= 0.02
        and _float(row["ov_copy_next_accuracy"]) >= 0.50
        and _float(row["clamp_post_attention_next_accuracy"]) >= 0.50
    }
    phase_effective_heads = {
        key for key, value in head_depth.items() if value >= 0.05
    }
    same_site_multifunction = (
        induction_effective_heads & phase_effective_heads
    )
    induction_physical_heads = {
        head for _, head in induction_effective_heads
    }
    phase_physical_heads = {head for _, head in phase_effective_heads}
    dual_writer_loops = [
        loop
        for loop, values in writer_component_drop_by_loop.items()
        if values.get("zero_B1_attention_destination", 0.0) >= 0.10
        and values.get("zero_B1_mlp_destination", 0.0) >= 0.10
    ]
    writer_synergies = [
        values["zero_B1_attention_and_mlp_destination"]
        - max(
            values.get("zero_B1_attention_destination", 0.0),
            values.get("zero_B1_mlp_destination", 0.0),
        )
        for values in writer_component_drop_by_loop.values()
        if "zero_B1_attention_and_mlp_destination" in values
    ]

    def writer_max(condition: str, field: str) -> float:
        return _max(
            [
                _float(row[field])
                for row in writer
                if row["condition"] == condition
            ]
        )

    return {
        "run": name,
        "family": family,
        "seed": seed,
        "endpoint_accuracy": summary["baseline"]["endpoint_accuracy"],
        "max_qk_destination_match_gap": _max(
            [_float(row["attention_match_gap"]) for row in numeric_heads]
        ),
        "max_ov_copy_next_accuracy": _max(
            [_float(row["ov_copy_next_accuracy"]) for row in numeric_heads]
        ),
        "max_single_head_clamp_next_accuracy": _max(
            [
                _float(row["clamp_post_attention_next_accuracy"])
                for row in numeric_heads
            ]
        ),
        "max_all_head_clamp_next_accuracy": _max(
            [
                _float(row["clamp_post_attention_next_accuracy"])
                for row in fingerprint
                if row["head"] == "all"
            ]
        ),
        "canonical_induction_site_count": len(canonical_sites),
        "prior_edge_specific_executor": int(name in prior_executor_runs),
        "strong_canonical_induction_run": int(
            bool(canonical_sites) and name in prior_executor_runs
        ),
        "local_edge_specific_site_count": local_site_count,
        "same_site_induction_phase_head_count": len(
            same_site_multifunction
        ),
        "physical_induction_phase_head_count": len(
            induction_physical_heads & phase_physical_heads
        ),
        "B1_dual_attention_mlp_writer_loop_count": len(
            dual_writer_loops
        ),
        "max_B1_writer_joint_synergy": _max(writer_synergies),
        "max_mlp_rescued_clamp_next_accuracy": _max(
            [
                _float(row["clamp_post_mlp_next_accuracy"])
                - _float(row["clamp_post_attention_next_accuracy"])
                for row in numeric_heads
            ]
        ),
        "max_mlp_rescued_clamp_next_margin": _max(
            [
                _float(row["clamp_post_mlp_next_margin"])
                - _float(row["clamp_post_attention_next_margin"])
                for row in numeric_heads
            ]
        ),
        "max_current_edge_next_specificity": _max(
            [
                values["current_edge_next_specificity"]
                for values in bundle_by_site.values()
            ]
        ),
        "max_current_edge_endpoint_specificity": _max(
            [
                values["current_edge_endpoint_specificity"]
                for values in bundle_by_site.values()
            ]
        ),
        "max_future_edge_endpoint_specificity": _max(
            [
                values["future_edge_endpoint_specificity"]
                for values in bundle_by_site.values()
            ]
        ),
        "max_bundle_endpoint_specificity": _max(
            [
                values["bundle_endpoint_specificity"]
                for values in bundle_by_site.values()
            ]
        ),
        "max_depth_q_specific_recovery": depth_max("q"),
        "max_depth_attention_specific_recovery": depth_max(
            "attention_pattern", "attention_out"
        ),
        "max_depth_mlp_specific_recovery": depth_max(
            "mlp_hidden", "mlp_out"
        ),
        "max_depth_head_specific_recovery": _max(
            list(head_depth.values())
        ),
        "max_B1_attention_destination_endpoint_drop": writer_max(
            "zero_B1_attention_destination", "endpoint_accuracy_drop"
        ),
        "max_B1_mlp_destination_endpoint_drop": writer_max(
            "zero_B1_mlp_destination", "endpoint_accuracy_drop"
        ),
        "max_B1_joint_destination_endpoint_drop": writer_max(
            "zero_B1_attention_and_mlp_destination",
            "endpoint_accuracy_drop",
        ),
        "max_B1_joint_destination_match_drop": writer_max(
            "zero_B1_attention_and_mlp_destination",
            "destination_attention_drop",
        ),
    }


def summarize_families(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        families[row["family"]].append(row)
    output: list[dict[str, Any]] = []
    count_fields = {
        "canonical_induction_site_count",
        "local_edge_specific_site_count",
        "same_site_induction_phase_head_count",
        "physical_induction_phase_head_count",
        "B1_dual_attention_mlp_writer_loop_count",
        "prior_edge_specific_executor",
        "strong_canonical_induction_run",
    }
    excluded = {"run", "family", "seed"}
    for family, rows in sorted(families.items()):
        item: dict[str, Any] = {
            "family": family,
            "run_count": len(rows),
            "runs_with_canonical_induction_site": sum(
                int(row["canonical_induction_site_count"] > 0)
                for row in rows
            ),
            "runs_with_local_edge_specific_site": sum(
                int(row["local_edge_specific_site_count"] > 0)
                for row in rows
            ),
            "runs_with_prior_edge_specific_executor": sum(
                int(row["prior_edge_specific_executor"]) for row in rows
            ),
            "runs_with_strong_canonical_induction": sum(
                int(row["strong_canonical_induction_run"]) for row in rows
            ),
            "runs_with_same_site_induction_phase_head": sum(
                int(row["same_site_induction_phase_head_count"] > 0)
                for row in rows
            ),
            "same_site_induction_phase_head_count_median": _median(
                [
                    _float(row["same_site_induction_phase_head_count"])
                    for row in rows
                ]
            ),
            "runs_with_physical_induction_phase_head": sum(
                int(row["physical_induction_phase_head_count"] > 0)
                for row in rows
            ),
            "physical_induction_phase_head_count_median": _median(
                [
                    _float(row["physical_induction_phase_head_count"])
                    for row in rows
                ]
            ),
            "runs_with_B1_dual_attention_mlp_writer_loop": sum(
                int(row["B1_dual_attention_mlp_writer_loop_count"] > 0)
                for row in rows
            ),
            "B1_dual_attention_mlp_writer_loop_count_median": _median(
                [
                    _float(
                        row["B1_dual_attention_mlp_writer_loop_count"]
                    )
                    for row in rows
                ]
            ),
        }
        for field in rows[0]:
            if field in excluded or field in count_fields:
                continue
            item[f"{field}_median"] = _median(
                [_float(row[field]) for row in rows]
            )
        output.append(item)
    return output


def _plot(run_rows: list[dict[str, Any]], path: Path) -> None:
    families = sorted({row["family"] for row in run_rows})
    colors = plt.get_cmap("tab10")
    metrics = [
        (
            "max_qk_destination_match_gap",
            "QK destination-vs-random gap",
        ),
        ("max_ov_copy_next_accuracy", "OV local-next accuracy"),
        (
            "max_single_head_clamp_next_accuracy",
            "canonical clamp local-next accuracy",
        ),
        (
            "max_future_edge_endpoint_specificity",
            "future-edge endpoint specificity",
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    rng = np.random.default_rng(20260726)
    for axis, (field, title) in zip(axes.flat, metrics, strict=True):
        for index, family in enumerate(families):
            values = [
                _float(row[field])
                for row in run_rows
                if row["family"] == family
                and math.isfinite(_float(row[field]))
            ]
            x = index + rng.normal(0, 0.04, len(values))
            axis.scatter(
                x,
                values,
                color=colors(index),
                alpha=0.8,
                label=family if axis is axes.flat[0] else None,
            )
            if values:
                axis.hlines(
                    statistics.median(values),
                    index - 0.25,
                    index + 0.25,
                    color="black",
                    linewidth=2,
                )
        axis.set_xticks(range(len(families)), families, rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate graph-path versus induction causal contrasts."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--functional-role-csv",
        type=Path,
        default=None,
        help=(
            "Optional prior causal role table. Runs with an A-level "
            "successor edge lookup/executor are used as the cross-graph "
            "closure gate for a strong induction claim."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = sorted(
        path.parent
        for path in args.raw_dir.rglob("summary.json")
        if path.parent != args.raw_dir
    )
    prior_executor_runs: set[str] = set()
    if args.functional_role_csv is not None:
        for row in _read_csv(args.functional_role_csv):
            if (
                row.get("candidate_function")
                == "successor edge lookup/executor"
                and row.get("evidence_grade") == "A"
            ):
                prior_executor_runs.add(row["run"])
    run_rows = [
        summarize_run(path, prior_executor_runs=prior_executor_runs)
        for path in run_dirs
    ]
    if not run_rows:
        raise FileNotFoundError(f"no run summaries under {args.raw_dir}")
    family_rows = summarize_families(run_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "run_summary.csv", run_rows)
    _write_csv(args.out_dir / "family_summary.csv", family_rows)
    _plot(run_rows, args.out_dir / "induction_contrast_by_family.png")
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {"runs": run_rows, "families": family_rows},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
