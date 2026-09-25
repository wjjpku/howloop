from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


VECTOR_SPECS = {
    "effective_branch_ablation_accuracy": (
        "effective_branch_ablation_rows.csv",
        "site",
        "accuracy_drop",
    ),
    "effective_branch_ablation_margin": (
        "effective_branch_ablation_rows.csv",
        "site",
        "margin_drop",
    ),
    "effective_branch_patch_in": (
        "effective_branch_patching_rows.csv",
        "site",
        "patch_in_recovery",
    ),
    "effective_branch_patch_out": (
        "effective_branch_patching_rows.csv",
        "site",
        "patch_out_effect",
    ),
    "effective_head_ablation_accuracy": (
        "effective_head_ablation_rows.csv",
        "site",
        "accuracy_drop",
    ),
    "effective_head_ablation_margin": (
        "effective_head_ablation_rows.csv",
        "site",
        "margin_drop",
    ),
    "tied_head_ablation_accuracy": (
        "tied_head_ablation_rows.csv",
        "head",
        "accuracy_drop",
    ),
    "tied_head_ablation_margin": (
        "tied_head_ablation_rows.csv",
        "head",
        "margin_drop",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--macro-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--component-prefix", default="full_seed")
    parser.add_argument("--natural-prefix", default="natural_seed")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv_vector(
    directory: Path,
    *,
    filename: str,
    key_column: str,
    value_column: str,
) -> tuple[list[str], np.ndarray]:
    with (directory / filename).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row[key_column])
    keys = [row[key_column] for row in rows]
    values = np.asarray(
        [float(row[value_column]) for row in rows],
        dtype=np.float64,
    )
    return keys, values


def finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(a) & np.isfinite(b)
    return a[finite], b[finite]


def similarities(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = finite_pair(a, b)
    if len(a) < 2:
        return {
            "pearson": float("nan"),
            "cosine": float("nan"),
            "normalized_l2": float("nan"),
        }
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    pearson_denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    cosine_denominator = np.linalg.norm(a) * np.linalg.norm(b)
    scale = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    return {
        "pearson": (
            float(np.dot(a_centered, b_centered) / pearson_denominator)
            if pearson_denominator > 0
            else float("nan")
        ),
        "cosine": (
            float(np.dot(a, b) / cosine_denominator)
            if cosine_denominator > 0
            else float("nan")
        ),
        "normalized_l2": (
            float(np.linalg.norm(a - b) / scale)
            if scale > 0
            else float("nan")
        ),
    }


def load_summary(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def jaccard(left: list[str], right: list[str]) -> float:
    a = set(left)
    b = set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def vector_rows(
    *,
    left_name: str,
    left_dir: Path,
    right_name: str,
    right_dir: Path,
    comparison: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, (filename, key_column, value_column) in VECTOR_SPECS.items():
        left_keys, left = read_csv_vector(
            left_dir,
            filename=filename,
            key_column=key_column,
            value_column=value_column,
        )
        right_keys, right = read_csv_vector(
            right_dir,
            filename=filename,
            key_column=key_column,
            value_column=value_column,
        )
        if left_keys != right_keys:
            raise ValueError(f"site mismatch for {metric}: {left_name}, {right_name}")
        rows.append(
            {
                "comparison": comparison,
                "left": left_name,
                "right": right_name,
                "metric": metric,
                "sites": len(left_keys),
                **similarities(left, right),
            }
        )

    left_summary = load_summary(left_dir)
    right_summary = load_summary(right_dir)
    rows.extend(
        [
            {
                "comparison": comparison,
                "left": left_name,
                "right": right_name,
                "metric": "selected_branch_jaccard",
                "sites": None,
                "pearson": float("nan"),
                "cosine": jaccard(
                    left_summary["branch_circuit"]["selected"],
                    right_summary["branch_circuit"]["selected"],
                ),
                "normalized_l2": float("nan"),
            },
            {
                "comparison": comparison,
                "left": left_name,
                "right": right_name,
                "metric": "selected_head_jaccard",
                "sites": None,
                "pearson": float("nan"),
                "cosine": jaccard(
                    left_summary["attention_head_circuit"]["selected"],
                    right_summary["attention_head_circuit"]["selected"],
                ),
                "normalized_l2": float("nan"),
            },
        ]
    )
    return rows


def macro_rows(
    *,
    macro_root: Path,
    left_name: str,
    right_name: str,
    comparison: str,
) -> list[dict[str, Any]]:
    left = load_summary(macro_root / left_name)
    right = load_summary(macro_root / right_name)
    rows = []
    for metric, key in (
        ("physical_block_trajectory", "expected_physical_block_accuracy"),
        ("macro_component_trajectory", "expected_macro_accuracy"),
        ("endpoint_overloop_trajectory", "endpoint_accuracy_by_macro_loop"),
    ):
        values = similarities(
            np.asarray(left[key], dtype=np.float64),
            np.asarray(right[key], dtype=np.float64),
        )
        rows.append(
            {
                "comparison": comparison,
                "left": left_name,
                "right": right_name,
                "metric": metric,
                "sites": len(left[key]),
                **values,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def aggregate(
    paired_rows: list[dict[str, Any]],
    natural_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = sorted({str(row["metric"]) for row in paired_rows})
    for metric in metrics:
        paired = [row for row in paired_rows if row["metric"] == metric]
        natural = [row for row in natural_rows if row["metric"] == metric]
        metric_summary: dict[str, Any] = {}
        for key in ("pearson", "cosine", "normalized_l2"):
            paired_values = finite_values(paired, key)
            natural_values = finite_values(natural, key)
            metric_summary[key] = {
                "paired_mean": (
                    float(paired_values.mean()) if len(paired_values) else None
                ),
                "paired_values": paired_values.tolist(),
                "natural_pair_mean": (
                    float(natural_values.mean()) if len(natural_values) else None
                ),
                "natural_pair_median": (
                    float(np.median(natural_values))
                    if len(natural_values)
                    else None
                ),
                "natural_pair_10pct": (
                    float(np.quantile(natural_values, 0.10))
                    if len(natural_values)
                    else None
                ),
                "natural_pair_90pct": (
                    float(np.quantile(natural_values, 0.90))
                    if len(natural_values)
                    else None
                ),
            }
        result[metric] = metric_summary
    return result


def main() -> None:
    args = parse_args()
    paired_rows: list[dict[str, Any]] = []
    natural_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        natural_name = f"{args.natural_prefix}{seed}"
        component_name = f"{args.component_prefix}{seed}"
        paired_rows.extend(
            vector_rows(
                left_name=natural_name,
                left_dir=args.analysis_root / natural_name,
                right_name=component_name,
                right_dir=args.analysis_root / component_name,
                comparison="paired_natural_component",
            )
        )
        if args.macro_root is not None:
            paired_rows.extend(
                macro_rows(
                    macro_root=args.macro_root,
                    left_name=natural_name,
                    right_name=component_name,
                    comparison="paired_natural_component",
                )
            )
    for left_seed, right_seed in itertools.combinations(args.seeds, 2):
        left_name = f"{args.natural_prefix}{left_seed}"
        right_name = f"{args.natural_prefix}{right_seed}"
        natural_rows.extend(
            vector_rows(
                left_name=left_name,
                left_dir=args.analysis_root / left_name,
                right_name=right_name,
                right_dir=args.analysis_root / right_name,
                comparison="natural_cross_seed",
            )
        )
        if args.macro_root is not None:
            natural_rows.extend(
                macro_rows(
                    macro_root=args.macro_root,
                    left_name=left_name,
                    right_name=right_name,
                    comparison="natural_cross_seed",
                )
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "paired_similarity_rows.csv", paired_rows)
    write_csv(args.out_dir / "natural_variability_rows.csv", natural_rows)
    summary = {
        "seeds": args.seeds,
        "component_prefix": args.component_prefix,
        "natural_prefix": args.natural_prefix,
        "interpretation": (
            "Paired similarity must be interpreted against natural cross-seed "
            "similarity. Correlation does not replace the underlying causal "
            "ablation, patching, circuit-only, and complement tests."
        ),
        "metrics": aggregate(paired_rows, natural_rows),
    }
    (args.out_dir / "circuit_similarity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
