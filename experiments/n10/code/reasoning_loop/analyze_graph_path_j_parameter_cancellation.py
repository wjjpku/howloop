"""Measure how D, shared AB, and stage AiBi create J's low-gain directions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 16))
    return parser.parse_args(argv)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1)
    b = right.reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def _gain(component: np.ndarray, basis: np.ndarray) -> float:
    return float(np.linalg.norm(basis.T @ component, "fro") / np.sqrt(basis.shape[1]))


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.T @ right, "fro") ** 2 / left.shape[1])


def _svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(matrix, full_matrices=True)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(path: Path, ranks: Sequence[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    diagonal_values = state["shared_diagonal_scale"].double().numpy()
    diagonal = np.diag(diagonal_values)
    diagonal_mean = np.eye(diagonal.shape[0]) * float(diagonal_values.mean())
    shared = (
        state["shared_A"].double().numpy()
        @ state["shared_B"].double().numpy()
    )
    skeleton = diagonal + shared
    skeleton_u, skeleton_s, _ = _svd(skeleton)
    rows: list[dict[str, Any]] = []
    for source_age in range(2, 9):
        stage = (
            state[f"stage_A.{source_age}"].double().numpy()
            @ state[f"stage_B.{source_age}"].double().numpy()
        )
        full = skeleton + stage
        full_u, full_s, _ = _svd(full)
        no_shared_s = _svd(diagonal + stage)[1]
        mean_diagonal_s = _svd(diagonal_mean + shared + stage)[1]
        for rank in ranks:
            full_bottom = full_u[:, -rank:]
            skeleton_bottom = skeleton_u[:, -rank:]
            diagonal_effect = full_bottom.T @ diagonal
            shared_effect = full_bottom.T @ shared
            skeleton_effect = full_bottom.T @ skeleton
            stage_effect = full_bottom.T @ stage
            rows.append(
                {
                    "source_age": source_age,
                    "user_J": source_age - 1,
                    "rank": int(rank),
                    "diagonal_gain_rms": _gain(diagonal, full_bottom),
                    "shared_AB_gain_rms": _gain(shared, full_bottom),
                    "shared_skeleton_gain_rms": _gain(skeleton, full_bottom),
                    "stage_AiBi_gain_rms": _gain(stage, full_bottom),
                    "full_J_gain_rms": _gain(full, full_bottom),
                    "cosine_D_vs_shared_AB_effect": _cosine(
                        diagonal_effect, shared_effect
                    ),
                    "cosine_skeleton_vs_stage_effect": _cosine(
                        skeleton_effect, stage_effect
                    ),
                    "bottom_input_overlap_with_skeleton": _overlap(
                        full_bottom, skeleton_bottom
                    ),
                    "full_singular_min": float(full_s[-1]),
                    "D_plus_stage_singular_min": float(no_shared_s[-1]),
                    "Dmean_plus_AB_plus_stage_singular_min": float(
                        mean_diagonal_s[-1]
                    ),
                }
            )
    selected = [row for row in rows if int(row["rank"]) == 8]
    summary = {
        "status": "complete",
        "bank_artifact": str(path),
        "interpretation": (
            "For row states, each full-J bottom input basis U_bottom is evaluated "
            "through D, shared AB, their sum, stage AiBi, and the full matrix. "
            "Strong negative component-effect cosine means low gain is made by "
            "directional cancellation rather than by small component weights."
        ),
        "rank8": {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "diagonal_gain_rms",
                "shared_AB_gain_rms",
                "shared_skeleton_gain_rms",
                "stage_AiBi_gain_rms",
                "full_J_gain_rms",
                "cosine_D_vs_shared_AB_effect",
                "cosine_skeleton_vs_stage_effect",
                "bottom_input_overlap_with_skeleton",
                "D_plus_stage_singular_min",
            )
        },
    }
    return rows, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = analyze(args.bank_artifact, args.ranks)
    _write_csv(args.out_dir / "parameter_cancellation_by_stage.csv", rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
