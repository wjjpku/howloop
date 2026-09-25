from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _segment(values: np.ndarray, start: int, stop: int) -> float:
    return float(values[start:stop].mean())


def _reliable_prefix(values: np.ndarray, threshold: float) -> int:
    failed = np.flatnonzero(values < threshold)
    return int(failed[0]) if len(failed) else int(len(values))


def run(args: argparse.Namespace) -> dict[str, Any]:
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))
    audit = json.loads(args.strict_unseen_summary.read_text(encoding="utf-8"))
    curves = audit["strictly_unseen"]["curves"]
    labels = sorted(
        label
        for label in curves
        if label.startswith("task_diagonal_lora_r")
    )
    if not labels:
        raise ValueError("no diagonal-LoRA variants found in strict-unseen audit")
    values = np.asarray([curves[label]["accuracy_by_cycle"] for label in labels], dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    raw = np.asarray(curves["identity_no_J"]["accuracy_by_cycle"], dtype=np.float64)
    reference = np.asarray(curves["reference_affine"]["accuracy_by_cycle"], dtype=np.float64)
    loops = np.arange(1, len(mean) + 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, loop in enumerate(loops):
        row: dict[str, Any] = {
            "continuation_loop": int(loop),
            "seen_during_training": bool(loop <= 8),
            "J_mean_accuracy": float(mean[index]),
            "J_std_population": float(std[index]),
            "J_min_accuracy": float(values[:, index].min()),
            "J_max_accuracy": float(values[:, index].max()),
            "raw_no_J_accuracy": float(raw[index]),
            "one_step_affine_initializer_accuracy": float(reference[index]),
        }
        for label, curve in zip(labels, values, strict=True):
            row[label] = float(curve[index])
        rows.append(row)
    with (args.out_dir / "length_generalization_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for axis, stop, title in (
        (axes[0], len(loops), "Full extrapolation range"),
        (axes[1], min(32, len(loops)), "Near-boundary detail"),
    ):
        visible = loops <= stop
        axis.axvspan(1, min(8, stop), color="#B8E0C2", alpha=0.28, label="train: K<=8")
        if stop > 8:
            axis.axvspan(8, stop, color="#F3C3C3", alpha=0.14, label="unseen length: K>8")
        axis.plot(loops[visible], raw[visible], color="#777777", linestyle="--", linewidth=1.5, label="raw frozen F")
        axis.plot(loops[visible], reference[visible], color="#A45DC7", linestyle=":", linewidth=1.5, label="one-step affine init")
        for index, curve in enumerate(values):
            axis.plot(loops[visible], curve[visible], color="#165DFF", alpha=0.22, linewidth=0.9, label="individual J seeds" if index == 0 else None)
        axis.plot(loops[visible], mean[visible], color="#165DFF", linewidth=2.5, label="strict-h8 J mean")
        axis.fill_between(loops[visible], (mean - std)[visible], (mean + std)[visible], color="#165DFF", alpha=0.18, linewidth=0)
        axis.axvline(8, color="black", linewidth=1.1)
        axis.set_xlim(1, stop)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("number K of repeated J->F continuations")
        axis.set_ylabel("successor accuracy on strictly unseen permutations")
        axis.set_title(title)
        axis.grid(alpha=0.18)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8, loc="lower left")
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"strict_h8_length_generalization.{suffix}", dpi=210 if suffix == "png" else None)
    plt.close(fig)

    segments = {
        "auc_1_8_ID": _segment(mean, 0, min(8, len(mean))),
        "auc_9_16_OOD": _segment(mean, 8, min(16, len(mean))),
        "auc_17_32_OOD": _segment(mean, 16, min(32, len(mean))),
        "auc_33_64_OOD": _segment(mean, 32, min(64, len(mean))),
        "auc_65_128_OOD": _segment(mean, 64, min(128, len(mean))),
    }
    result = {
        "status": "complete",
        "protocol": {
            "checkpoint": training["checkpoint"],
            "backbone_loss": training["loss_placement"],
            "controller": training["controller"],
            "controller_placement": training["controller_placement"],
            "rank": training["ranks"],
            "controller_seeds": training["initialization_seeds"],
            "curriculum": training.get("curriculum"),
            "training_stages": training["stages"],
            "maximum_training_composition_length": max(max(stage["horizons"]) for stage in training["stages"]),
            "state_loss_weight": training["state_loss_weight"],
            "strictly_unseen_permutations": audit["sampling"]["strictly_unseen_permutations"],
            "all_starts_per_permutation": audit["sampling"]["all_starts_per_permutation"],
            "evaluation_loops": audit["continuation_loops"],
        },
        "labels": labels,
        "segments": segments,
        "accuracy_at": {
            str(loop): {
                "mean": float(mean[loop - 1]),
                "std_population": float(std[loop - 1]),
                "values": [float(value) for value in values[:, loop - 1]],
                "raw": float(raw[loop - 1]),
            }
            for loop in (1, 2, 4, 8, 9, 12, 16, 24, 32, 48, 64, 96, 128)
            if loop <= len(mean)
        },
        "reliable_prefix": {
            str(threshold): _reliable_prefix(mean, threshold)
            for threshold in (0.99, 0.95, 0.90)
        },
        "files": {
            "curve_csv": "length_generalization_curve.csv",
            "figure_png": "strict_h8_length_generalization.png",
            "figure_pdf": "strict_h8_length_generalization.pdf",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--strict-unseen-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
