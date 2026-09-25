from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


INDUCTION_METRICS = (
    "canonical_induction_site_count",
    "local_edge_specific_site_count",
    "same_site_induction_phase_head_count",
    "physical_induction_phase_head_count",
    "B1_dual_attention_mlp_writer_loop_count",
    "max_qk_destination_match_gap",
    "max_ov_copy_next_accuracy",
    "max_single_head_clamp_next_accuracy",
    "max_current_edge_next_specificity",
    "max_future_edge_endpoint_specificity",
    "max_bundle_endpoint_specificity",
    "max_depth_head_specific_recovery",
    "max_B1_mlp_destination_endpoint_drop",
    "max_mlp_rescued_clamp_next_margin",
)

COMPONENT_METRICS = (
    "role_count",
    "multifunctional",
    "overloop_destabilizer",
    "max_skip_endpoint_drop",
    "max_repeat_endpoint_drop",
    "semantic_shift_site_count",
    "max_overloop_skip_improvement",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: str | int | float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _paired_rows(
    rows: list[dict[str, str]],
    *,
    contrast: str,
    positive_family: str,
    reference_family: str,
    metrics: tuple[str, ...],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    indexed = {
        (
            row["family"],
            *(row[field] for field in key_fields),
        ): row
        for row in rows
    }
    keys = sorted(
        {
            tuple(row[field] for field in key_fields)
            for row in rows
            if row["family"] == positive_family
        }
        & {
            tuple(row[field] for field in key_fields)
            for row in rows
            if row["family"] == reference_family
        }
    )
    output: list[dict[str, Any]] = []
    for key in keys:
        positive = indexed[(positive_family, *key)]
        reference = indexed[(reference_family, *key)]
        for metric in metrics:
            positive_value = _float(positive[metric])
            reference_value = _float(reference[metric])
            output.append(
                {
                    "contrast": contrast,
                    **dict(zip(key_fields, key, strict=True)),
                    "positive_family": positive_family,
                    "reference_family": reference_family,
                    "metric": metric,
                    "positive_value": positive_value,
                    "reference_value": reference_value,
                    "delta": positive_value - reference_value,
                }
            )
    return output


def _summaries(
    rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row["delta"])
        if math.isfinite(value):
            grouped[
                tuple(str(row[field]) for field in group_fields)
            ].append(value)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        output.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "pair_count": len(values),
                "median_delta": statistics.median(values),
                "positive_pair_count": sum(value > 0 for value in values),
                "negative_pair_count": sum(value < 0 for value in values),
                "zero_pair_count": sum(value == 0 for value in values),
                "min_delta": min(values),
                "max_delta": max(values),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired graph horizon compression/stretch effects."
    )
    parser.add_argument(
        "--compression-induction-run-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stretch-induction-run-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compression-component-rows",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stretch-component-rows",
        type=Path,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    induction_rows = _paired_rows(
        _read_csv(args.compression_induction_run_summary),
        contrast="compression",
        positive_family="D8_L6",
        reference_family="D8_L8",
        metrics=INDUCTION_METRICS,
        key_fields=("seed",),
    )
    induction_rows.extend(
        _paired_rows(
            _read_csv(args.stretch_induction_run_summary),
            contrast="stretch",
            positive_family="D6_L8",
            reference_family="D6_L6",
            metrics=INDUCTION_METRICS,
            key_fields=("seed",),
        )
    )
    component_rows = _paired_rows(
        _read_csv(args.compression_component_rows),
        contrast="compression",
        positive_family="D8_L6",
        reference_family="D8_L8",
        metrics=COMPONENT_METRICS,
        key_fields=("seed", "component"),
    )
    component_rows.extend(
        _paired_rows(
            _read_csv(args.stretch_component_rows),
            contrast="stretch",
            positive_family="D6_L8",
            reference_family="D6_L6",
            metrics=COMPONENT_METRICS,
            key_fields=("seed", "component"),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "paired_induction_deltas.csv", induction_rows)
    _write_csv(
        args.out_dir / "paired_induction_delta_summary.csv",
        _summaries(
            induction_rows,
            group_fields=("contrast", "metric"),
        ),
    )
    _write_csv(args.out_dir / "paired_component_deltas.csv", component_rows)
    _write_csv(
        args.out_dir / "paired_component_delta_summary.csv",
        _summaries(
            component_rows,
            group_fields=("contrast", "component", "metric"),
        ),
    )


if __name__ == "__main__":
    main()
