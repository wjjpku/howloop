"""Summarize endpoint behavior and low-gain geometry for J interventions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-bank", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weights(path: Path) -> list[np.ndarray]:
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    diagonal = np.diag(state["shared_diagonal_scale"].double().numpy())
    shared = (
        state["shared_A"].double().numpy()
        @ state["shared_B"].double().numpy()
    )
    return [
        diagonal
        + shared
        + state[f"stage_A.{age}"].double().numpy()
        @ state[f"stage_B.{age}"].double().numpy()
        for age in range(2, 9)
    ]


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.T @ right, "fro") ** 2 / left.shape[1])


def condition_name(label: str, artifact_index: int) -> str:
    if artifact_index == 0:
        return "original"
    stem = label.rsplit("/", 1)[-1]
    return re.sub(r"_seed\d+$", "", stem)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.evaluation_csv.open(newline="", encoding="utf-8") as handle:
        evaluation = list(csv.DictReader(handle))

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in evaluation:
        grouped[int(row["artifact_index"])].append(row)

    original_svd = [np.linalg.svd(value, full_matrices=True) for value in weights(args.original_bank)]
    artifact_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for artifact_index, rows in sorted(grouped.items()):
        label = rows[0]["artifact_label"]
        path = args.original_bank if artifact_index == 0 else Path(rows[0]["artifact"])
        if artifact_index != 0 and not path.exists():
            path = args.evaluation_csv.parent.parent / "artifacts" / path.name
        condition = condition_name(label, artifact_index)
        macro = float(np.mean([float(row["accuracy_mean"]) for row in rows]))
        examples = int(sum(int(row["trajectories"]) * int(row["examples_per_trajectory"]) for row in rows))
        artifact_rows.append(
            {
                "artifact_index": artifact_index,
                "condition": condition,
                "artifact": str(path),
                "macro_accuracy": macro,
                "examples": examples,
                "correct_examples_rounded": int(round(macro * examples)),
            }
        )
        stage_geometry: list[dict[str, float]] = []
        for stage, value in enumerate(weights(path), start=1):
            left, singular, right_t = np.linalg.svd(value, full_matrices=True)
            original_left, _, original_right_t = original_svd[stage - 1]
            row = {
                "input_bottom8_overlap_original": overlap(
                    original_left[:, -8:], left[:, -8:]
                ),
                "output_bottom8_overlap_original": overlap(
                    original_right_t.T[:, -8:], right_t.T[:, -8:]
                ),
                "singular_min": float(singular[-1]),
                "bottom8_singular_mean": float(singular[-8:].mean()),
            }
            stage_geometry.append(row)
            geometry_rows.append(
                {
                    "artifact_index": artifact_index,
                    "condition": condition,
                    "user_J": stage,
                    **row,
                }
            )
        for key in stage_geometry[0]:
            artifact_rows[-1][f"{key}_mean"] = float(
                np.mean([row[key] for row in stage_geometry])
            )

    write_csv(args.out_dir / "artifact_summary.csv", artifact_rows)
    write_csv(args.out_dir / "gate_geometry_by_artifact.csv", geometry_rows)

    condition_rows: list[dict[str, Any]] = []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artifact_rows:
        by_condition[row["condition"]].append(row)
    for condition, rows in sorted(by_condition.items()):
        values = np.asarray([row["macro_accuracy"] for row in rows], dtype=float)
        condition_rows.append(
            {
                "condition": condition,
                "artifact_draws": len(rows),
                "macro_accuracy_mean": float(values.mean()),
                "macro_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "macro_accuracy_min": float(values.min()),
                "macro_accuracy_max": float(values.max()),
                "examples_per_artifact": rows[0]["examples"],
                "correct_examples_per_artifact_mean": float(
                    np.mean([row["correct_examples_rounded"] for row in rows])
                ),
                "input_bottom8_overlap_original_mean": float(
                    np.mean([row["input_bottom8_overlap_original_mean"] for row in rows])
                ),
                "output_bottom8_overlap_original_mean": float(
                    np.mean([row["output_bottom8_overlap_original_mean"] for row in rows])
                ),
                "bottom8_singular_mean": float(
                    np.mean([row["bottom8_singular_mean_mean"] for row in rows])
                ),
            }
        )
    write_csv(args.out_dir / "condition_summary.csv", condition_rows)
    (args.out_dir / "derived_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "original_bank": str(args.original_bank),
                "evaluation_csv": str(args.evaluation_csv),
                "condition_rows": condition_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
