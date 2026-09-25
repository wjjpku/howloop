from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--condition", default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def run_dir(root: Path, *, family: str, condition: str, seed: int) -> Path:
    model_dir = f"graphpath_N8_D8_d256_B2_L8_seed{seed}"
    if family == "natural":
        return root / f"D8_L8_seed{seed}" / model_dir
    return root / condition / f"D8_L8_{condition}_seed{seed}" / model_dir


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_step(
    history: list[dict[str, Any]],
    predicate: Any,
) -> int | None:
    for row in history:
        if predicate(row):
            return int(row["step"])
    return None


def component_mean(row: dict[str, Any]) -> float:
    values = row.get("trajectory_accuracy_by_loop")
    return float(np.mean(values[:4])) if values is not None else float("nan")


def depth8_component_mean(row: dict[str, Any]) -> float:
    values = row.get("trajectory_depth_loop_accuracy")
    return float(np.mean(values[-1][:4])) if values is not None else float("nan")


def summarize_run(
    *,
    family: str,
    condition: str,
    seed: int,
    path: Path,
) -> dict[str, Any]:
    summary = read_json(path / "summary.json")
    history = read_json(path / "history.json")
    final_metrics = summary["final_metrics"]
    final_endpoint = float(final_metrics["loop_accuracy"][-1])
    final_component = [
        float(value)
        for value in final_metrics.get(
            "trajectory_accuracy_by_loop",
            [float("nan")] * 8,
        )
    ]
    depth8_component = [
        float(value)
        for value in final_metrics.get(
            "trajectory_depth_loop_accuracy",
            [[float("nan")] * 8],
        )[-1]
    ]
    return {
        "family": family,
        "condition": "final_only" if family == "natural" else condition,
        "seed": seed,
        "best_step": int(summary["best_step"]),
        "best_endpoint_accuracy": float(summary["best_final_accuracy"]),
        "final_endpoint_accuracy": final_endpoint,
        "final_component_mean_loops1_4": float(np.mean(final_component[:4])),
        "final_depth8_component_mean_loops1_4": float(
            np.mean(depth8_component[:4])
        ),
        "final_component_accuracy_by_loop": final_component,
        "final_depth8_component_accuracy_by_loop": depth8_component,
        "first_endpoint_099_step": first_step(
            history,
            lambda row: float(row["loop_accuracy"][-1]) >= 0.99,
        ),
        "first_endpoint_and_component_095_step": first_step(
            history,
            lambda row: (
                float(row["loop_accuracy"][-1]) >= 0.99
                and component_mean(row) >= 0.95
            ),
        ),
        "first_endpoint_and_depth8_component_095_step": first_step(
            history,
            lambda row: (
                float(row["loop_accuracy"][-1]) >= 0.99
                and depth8_component_mean(row) >= 0.95
            ),
        ),
        "run_dir": str(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar = {
            key: value
            for key, value in row.items()
            if not isinstance(value, list)
        }
        for key in (
            "final_component_accuracy_by_loop",
            "final_depth8_component_accuracy_by_loop",
        ):
            prefix = key.removesuffix("_by_loop")
            for loop, value in enumerate(row[key], start=1):
                scalar[f"{prefix}_loop{loop}"] = value
        scalar_rows.append(scalar)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def aggregate(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    selected = [row for row in rows if row["family"] == family]
    keys = (
        "best_endpoint_accuracy",
        "final_endpoint_accuracy",
        "final_component_mean_loops1_4",
        "final_depth8_component_mean_loops1_4",
    )
    result: dict[str, Any] = {"seeds": [row["seed"] for row in selected]}
    for key in keys:
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": (
                float(finite.std(ddof=1))
                if len(finite) > 1
                else (0.0 if len(finite) == 1 else None)
            ),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    for key in (
        "first_endpoint_099_step",
        "first_endpoint_and_component_095_step",
        "first_endpoint_and_depth8_component_095_step",
    ):
        values = [row[key] for row in selected if row[key] is not None]
        result[key] = {
            "successes": len(values),
            "median": float(np.median(values)) if values else None,
            "values": values,
        }
    return result


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.append(
            summarize_run(
                family="natural",
                condition=args.condition,
                seed=seed,
                path=run_dir(
                    args.natural_root,
                    family="natural",
                    condition=args.condition,
                    seed=seed,
                ),
            )
        )
        rows.append(
            summarize_run(
                family="component",
                condition=args.condition,
                seed=seed,
                path=run_dir(
                    args.component_root,
                    family="component",
                    condition=args.condition,
                    seed=seed,
                ),
            )
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "training_seed_rows.csv", rows)
    payload = {
        "condition": args.condition,
        "natural": aggregate(rows, "natural"),
        "component": aggregate(rows, "component"),
        "rows": rows,
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
