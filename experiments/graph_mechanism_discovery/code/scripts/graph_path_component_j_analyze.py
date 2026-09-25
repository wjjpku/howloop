from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


LOW_RANK_FAMILIES = (
    "lora",
    "identity_low_rank",
    "scalar_low_rank",
    "diagonal_low_rank",
    "weight_low_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        nargs=2,
        metavar=("LABEL", "ROOT"),
        required=True,
    )
    parser.add_argument("--initial-j", type=Path, required=True)
    parser.add_argument("--validation", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _load_map(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    item = payload["maps"]["task"]
    return item["weight"].double(), item["bias"].double()


def _spectral_metrics(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    initial_weight: torch.Tensor,
    initial_bias: torch.Tensor,
) -> dict[str, float | int]:
    eigenvalues = torch.linalg.eigvals(weight)
    magnitudes = eigenvalues.abs()
    angles = torch.angle(eigenvalues)
    singular = torch.linalg.svdvals(weight)
    sign, logabsdet = torch.linalg.slogdet(weight)
    identity = torch.eye(weight.shape[0], dtype=weight.dtype)
    return {
        "eigen_abs_mean": float(magnitudes.mean()),
        "eigen_abs_median": float(magnitudes.median()),
        "eigen_abs_max": float(magnitudes.max()),
        "eigen_angle_abs_median": float(angles.abs().median()),
        "eigen_angle_abs_p95": float(
            torch.quantile(
                angles.abs(),
                torch.tensor(0.95, dtype=angles.dtype),
            )
        ),
        "eigen_near_zero_0p1": int(magnitudes.lt(0.1).sum()),
        "eigen_near_one_0p1": int((eigenvalues - 1).abs().lt(0.1).sum()),
        "singular_max": float(singular.max()),
        "singular_min": float(singular.min()),
        "singular_effective_rank_1e3": int(
            singular.gt(singular.max() * 1e-3).sum()
        ),
        "logabsdet": float(logabsdet),
        "det_sign": float(sign),
        "weight_frobenius": float(torch.linalg.vector_norm(weight)),
        "identity_distance_relative": float(
            torch.linalg.vector_norm(weight - identity)
            / torch.linalg.vector_norm(identity)
        ),
        "initial_weight_delta_relative": float(
            torch.linalg.vector_norm(weight - initial_weight)
            / torch.linalg.vector_norm(initial_weight)
        ),
        "initial_bias_delta_relative": float(
            torch.linalg.vector_norm(bias - initial_bias)
            / torch.linalg.vector_norm(initial_bias).clamp_min(1e-12)
        ),
    }


def _initial_retained_energy(
    weight: torch.Tensor,
    *,
    parameterization: str,
    rank: int,
) -> float:
    if parameterization in ("full", "lora"):
        return 1.0
    identity = torch.eye(weight.shape[0], dtype=weight.dtype)
    if parameterization == "identity_low_rank":
        residual = weight - identity
    elif parameterization == "scalar_low_rank":
        residual = weight - (
            torch.trace(weight) / weight.shape[0]
        ) * identity
    elif parameterization == "diagonal_low_rank":
        residual = weight - torch.diag(torch.diagonal(weight))
    elif parameterization == "weight_low_rank":
        residual = weight
    else:
        raise ValueError(f"unknown parameterization: {parameterization}")
    energy = torch.linalg.svdvals(residual).square()
    return float(energy[:rank].sum() / energy.sum().clamp_min(1e-12))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_low_rank(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for family in LOW_RANK_FAMILIES:
        group = sorted(
            (
                row
                for row in rows
                if row["task_parameterization"] == family
            ),
            key=lambda row: row["task_rank"],
        )
        if not group:
            continue
        axis.plot(
            [row["trainable_J_parameters"] for row in group],
            [row["auc_49_64"] for row in group],
            marker="o",
            label=family,
        )
        for row in group:
            axis.annotate(
                f"r{row['task_rank']}",
                (row["trainable_J_parameters"], row["auc_49_64"]),
                xytext=(2, 3),
                textcoords="offset points",
                fontsize=7,
            )
    baseline = next(
        (
            row["auc_49_64"]
            for row in rows
            if row["name"] == "baseline_no_finetune"
        ),
        None,
    )
    if baseline is not None:
        axis.axhline(
            baseline,
            color="black",
            linestyle="--",
            linewidth=1,
            label="initial dense J",
        )
    axis.set_xlabel("Trainable J parameters")
    axis.set_ylabel("Screen AUC, loops 49–64")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_top(rows: list[dict[str, Any]], path: Path) -> None:
    selected = sorted(
        rows,
        key=lambda row: row["selection_score"],
        reverse=True,
    )[:24]
    selected.reverse()
    figure, axis = plt.subplots(figsize=(9.2, 7.2))
    colors = [
        "#4477AA"
        if row["task_parameterization"] == "full"
        else "#EE6677"
        for row in selected
    ]
    axis.barh(
        [f"{row['stage']}:{row['name']}" for row in selected],
        [row["auc_49_64"] for row in selected],
        color=colors,
    )
    axis.set_xlabel("AUC, loops 49–64")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _load_validation(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["ranking"]:
            result[row["name"]] = row
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial_weight, initial_bias = _load_map(args.initial_j)
    validation = _load_validation(args.validation)
    rows: list[dict[str, Any]] = []
    for stage, root_text in args.stage:
        root = Path(root_text)
        progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
        for item in progress:
            if item["status"] != "complete":
                continue
            artifact = root / item["name"] / "unit_j_maps.pt"
            weight, bias = _load_map(artifact)
            row = {
                "stage": stage,
                **item,
                "initial_residual_retained_energy": _initial_retained_energy(
                    initial_weight,
                    parameterization=item["task_parameterization"],
                    rank=item["task_rank"],
                ),
                **_spectral_metrics(
                    weight,
                    bias,
                    initial_weight=initial_weight,
                    initial_bias=initial_bias,
                ),
            }
            validated = validation.get(item["name"])
            if validated is not None:
                row.update(
                    {
                        "validation_auc_1_24_mean": validated[
                            "auc_1_24_mean"
                        ],
                        "validation_auc_25_48_mean": validated[
                            "auc_25_48_mean"
                        ],
                        "validation_auc_49_64_mean": validated[
                            "auc_49_64_mean"
                        ],
                        "validation_auc_49_64_std": validated[
                            "auc_49_64_std"
                        ],
                        "validation_score_mean": validated[
                            "selection_score_mean"
                        ],
                    }
                )
            rows.append(row)
    serializable_rows = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        for row in rows
    ]
    _write_csv(args.out_dir / "candidate_metrics.csv", serializable_rows)
    (args.out_dir / "candidate_metrics.json").write_text(
        json.dumps(serializable_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_low_rank(
        [row for row in serializable_rows if row["stage"] == "screen"],
        args.out_dir / "low_rank_tradeoff.png",
    )
    _plot_top(serializable_rows, args.out_dir / "top_candidates.png")
    print(f"wrote {len(rows)} candidate records to {args.out_dir}")


if __name__ == "__main__":
    main()
