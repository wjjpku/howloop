from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-j", type=Path, required=True)
    parser.add_argument("--full64-j", type=Path, required=True)
    parser.add_argument("--balanced-j", type=Path, required=True)
    parser.add_argument("--late-j", type=Path, required=True)
    parser.add_argument("--full64-summary", type=Path, required=True)
    parser.add_argument("--late-summary", type=Path, required=True)
    parser.add_argument("--schur-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    item = payload["maps"]["task"]
    return (
        item["weight"].double().numpy(),
        item["bias"].double().numpy(),
    )


def global_metrics(weight: np.ndarray, bias: np.ndarray) -> dict:
    eigenvalues, eigenvectors = sla.eig(weight)
    singular = sla.svdvals(weight)
    frobenius = sla.norm(weight, "fro")
    commutator = weight.T @ weight - weight @ weight.T
    return {
        "weight_frobenius_norm": float(frobenius),
        "bias_norm": float(sla.norm(bias)),
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "operator_norm": float(singular[0]),
        "operator_to_spectral_radius": float(
            singular[0] / np.max(np.abs(eigenvalues))
        ),
        "minimum_singular_value": float(singular[-1]),
        "stable_rank": float(np.sum(singular**2) / singular[0] ** 2),
        "normalized_nonnormality": float(
            sla.norm(commutator, "fro") / frobenius**2
        ),
        "eigenvector_condition_number": float(
            np.linalg.cond(eigenvectors)
        ),
        "near_zero_eigenvalues": int(np.sum(np.abs(eigenvalues) < 0.1)),
        "near_one_eigenvalues": int(
            np.sum(np.abs(eigenvalues - 1) < 0.1)
        ),
        "absolute_angle_p95": float(
            np.quantile(np.abs(np.angle(eigenvalues)), 0.95)
        ),
        "log_abs_determinant": float(np.log(singular).sum()),
    }


def delta_metrics(
    reference: tuple[np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray],
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    reference_weight, reference_bias = reference
    target_weight, target_bias = target
    delta_weight = target_weight - reference_weight
    delta_bias = target_bias - reference_bias
    left, singular, right = sla.svd(delta_weight, full_matrices=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / np.sum(energy)
    ranks = {
        str(threshold): int(np.searchsorted(cumulative, threshold) + 1)
        for threshold in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
    }
    commutator = (
        reference_weight @ target_weight
        - target_weight @ reference_weight
    )
    metrics = {
        "weight_frobenius_norm": float(sla.norm(delta_weight, "fro")),
        "relative_weight_frobenius_norm": float(
            sla.norm(delta_weight, "fro")
            / sla.norm(reference_weight, "fro")
        ),
        "weight_operator_norm": float(singular[0]),
        "bias_norm": float(sla.norm(delta_bias)),
        "relative_bias_norm": float(
            sla.norm(delta_bias) / sla.norm(reference_bias)
        ),
        "stable_rank": float(np.sum(energy) / singular[0] ** 2),
        "energy_ranks": ranks,
        "normalized_commutator_to_delta": float(
            sla.norm(commutator, "fro")
            / (
                sla.norm(reference_weight, "fro")
                * sla.norm(delta_weight, "fro")
            )
        ),
        "singular_values": singular.tolist(),
    }
    return metrics, (left, singular, right)


def subspace_overlap(
    first: tuple[np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    first_left, _, first_right = first
    second_left, _, second_right = second
    output = {}
    for rank in (4, 8, 16, 32, 64):
        left_cosines = sla.svdvals(
            first_left[:, :rank].T @ second_left[:, :rank]
        )
        right_cosines = sla.svdvals(
            first_right[:rank] @ second_right[:rank].T
        )
        output[str(rank)] = {
            "left_mean_squared_cosine": float(
                np.mean(left_cosines**2)
            ),
            "right_mean_squared_cosine": float(
                np.mean(right_cosines**2)
            ),
            "left_max_principal_angle_degrees": float(
                np.degrees(
                    np.arccos(np.clip(left_cosines.min(), -1, 1))
                )
            ),
            "right_max_principal_angle_degrees": float(
                np.degrees(
                    np.arccos(np.clip(right_cosines.min(), -1, 1))
                )
            ),
        }
    return output


def segment(values: list[float], start: int, stop: int) -> float:
    return float(np.mean(values[start - 1 : stop]))


def causal_rows(summary: dict, segments: list[tuple[int, int]]) -> list[dict]:
    result = summary["results"][0]
    rows = []
    for condition, curve in result["curves"].items():
        values = curve["nonendpoint_accuracy_by_cycle"]
        row = {"condition": condition}
        for start, stop in segments:
            if len(values) >= stop:
                row[f"auc_{start}_{stop}"] = segment(values, start, stop)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def condition_value(
    rows: list[dict],
    condition: str,
    metric: str,
) -> float:
    return next(
        float(row[metric])
        for row in rows
        if row["condition"] == condition
    )


def plot_results(
    *,
    maps: dict[str, tuple[np.ndarray, np.ndarray]],
    deltas: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    full_rows: list[dict],
    late_rows: list[dict],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    for label, color in (("original", "#888888"), ("late", "#c44e52")):
        eigenvalues = sla.eigvals(maps[label][0])
        axes[0, 0].scatter(
            eigenvalues.real,
            eigenvalues.imag,
            s=12,
            alpha=0.55,
            label=label,
            color=color,
        )
    axes[0, 0].axhline(0, color="black", linewidth=0.6)
    axes[0, 0].axvline(0, color="black", linewidth=0.6)
    axes[0, 0].set_title("Task-map eigenvalues")
    axes[0, 0].set_xlabel("real")
    axes[0, 0].set_ylabel("imaginary")
    axes[0, 0].legend()

    for label, color in (
        ("full64_minus_original", "#4c72b0"),
        ("late_minus_full64", "#c44e52"),
    ):
        singular = deltas[label][1]
        axes[0, 1].semilogy(
            np.arange(1, len(singular) + 1),
            singular,
            label=label,
            color=color,
        )
    axes[0, 1].set_title("Singular spectrum of learned corrections")
    axes[0, 1].set_xlabel("mode rank")
    axes[0, 1].set_ylabel("singular value")
    axes[0, 1].legend()

    ranks = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    for prefix, linestyle, label in (
        ("top", "-", "retain top-k correction"),
        ("tail_after", "--", "remove top-k correction"),
    ):
        values = [
            condition_value(
                full_rows,
                f"{prefix}{rank}",
                "auc_49_64",
            )
            for rank in ranks
        ]
        axes[1, 0].plot(
            ranks,
            values,
            marker="o",
            markersize=3,
            linestyle=linestyle,
            label=label,
        )
    axes[1, 0].axhline(
        condition_value(full_rows, "base_J", "auc_49_64"),
        color="#888888",
        linewidth=1,
        label="base J",
    )
    axes[1, 0].axhline(
        condition_value(full_rows, "full_J", "auc_49_64"),
        color="#c44e52",
        linewidth=1,
        label="full target J",
    )
    axes[1, 0].set_title("Full64 correction: strict-unseen AUC 49-64")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("nonendpoint AUC")
    axes[1, 0].legend(fontsize=8)

    for metric, color in (
        ("auc_65_80", "#4c72b0"),
        ("auc_81_96", "#c44e52"),
    ):
        values = [
            condition_value(late_rows, f"top{rank}", metric)
            for rank in ranks
        ]
        axes[1, 1].plot(
            ranks,
            values,
            marker="o",
            markersize=3,
            label=metric,
            color=color,
        )
    axes[1, 1].set_title("Late-life correction: retain top-k modes")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("nonendpoint AUC")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def schur_rows(summary: dict) -> list[dict]:
    metadata = summary["delta_schur_blocks"]
    curves = summary["result"]["curves"]
    rows = []
    for label in metadata["blocks_descending_energy"]:
        rows.append(
            {
                "block": label,
                "energy_fraction": metadata["block_energy_fraction"][label],
                "only_auc_49_64": segment(
                    curves[f"only_{label}"][
                        "nonendpoint_accuracy_by_cycle"
                    ],
                    49,
                    64,
                ),
                "without_auc_49_64": segment(
                    curves[f"without_{label}"][
                        "nonendpoint_accuracy_by_cycle"
                    ],
                    49,
                    64,
                ),
            }
        )
    return rows


def plot_schur_blocks(
    summary: dict,
    rows: list[dict],
    path: Path,
) -> None:
    curves = summary["result"]["curves"]
    base = segment(
        curves["base_J"]["nonendpoint_accuracy_by_cycle"],
        49,
        64,
    )
    target = segment(
        curves["full_J"]["nonendpoint_accuracy_by_cycle"],
        49,
        64,
    )
    labels = [row["block"].replace("_to_", "→") for row in rows]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(
        x,
        [row["energy_fraction"] for row in rows],
        color="#4c72b0",
    )
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("fraction of delta-W Frobenius energy")
    axes[0].set_title("Full64 correction in original-J Schur blocks")

    width = 0.38
    axes[1].bar(
        x - width / 2,
        [row["only_auc_49_64"] for row in rows],
        width,
        label="base + only block",
        color="#55a868",
    )
    axes[1].bar(
        x + width / 2,
        [row["without_auc_49_64"] for row in rows],
        width,
        label="target - block",
        color="#c44e52",
    )
    axes[1].axhline(base, color="#888888", linewidth=1, label="base J")
    axes[1].axhline(target, color="#4c72b0", linewidth=1, label="target J")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_ylabel("strict-unseen nonendpoint AUC 49-64")
    axes[1].set_title("Schur-block causal sufficiency and necessity")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        "original": load_map(args.original_j),
        "full64": load_map(args.full64_j),
        "balanced": load_map(args.balanced_j),
        "late": load_map(args.late_j),
    }
    full_metrics, full_svd = delta_metrics(
        maps["original"],
        maps["full64"],
    )
    balanced_metrics, balanced_svd = delta_metrics(
        maps["full64"],
        maps["balanced"],
    )
    late_metrics, late_svd = delta_metrics(
        maps["full64"],
        maps["late"],
    )
    metrics = {
        "global": {
            label: global_metrics(*value)
            for label, value in maps.items()
        },
        "deltas": {
            "full64_minus_original": full_metrics,
            "balanced_minus_full64": balanced_metrics,
            "late_minus_full64": late_metrics,
        },
        "subspace_overlap": {
            "full64_vs_balanced_extension": subspace_overlap(
                full_svd,
                balanced_svd,
            ),
            "full64_vs_late_extension": subspace_overlap(
                full_svd,
                late_svd,
            ),
            "balanced_vs_late_extension": subspace_overlap(
                balanced_svd,
                late_svd,
            ),
        },
    }
    (args.out_dir / "matrix_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    full_summary = json.loads(args.full64_summary.read_text(encoding="utf-8"))
    late_summary = json.loads(args.late_summary.read_text(encoding="utf-8"))
    full_rows = causal_rows(
        full_summary,
        [(1, 24), (25, 48), (49, 64)],
    )
    late_rows = causal_rows(
        late_summary,
        [(1, 24), (25, 48), (49, 64), (65, 80), (81, 96)],
    )
    write_csv(args.out_dir / "full64_delta_causal_auc.csv", full_rows)
    write_csv(args.out_dir / "late_delta_causal_auc.csv", late_rows)
    plot_results(
        maps=maps,
        deltas={
            "full64_minus_original": full_svd,
            "late_minus_full64": late_svd,
        },
        full_rows=full_rows,
        late_rows=late_rows,
        path=args.out_dir / "current_best_J_mechanism.png",
    )
    if args.schur_summary is not None:
        schur_summary = json.loads(
            args.schur_summary.read_text(encoding="utf-8")
        )
        block_rows = schur_rows(schur_summary)
        write_csv(args.out_dir / "delta_schur_block_causal_auc.csv", block_rows)
        plot_schur_blocks(
            schur_summary,
            block_rows,
            args.out_dir / "delta_schur_block_causality.png",
        )


if __name__ == "__main__":
    main()
