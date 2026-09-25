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


def _controller_curves(
    payload: dict[str, Any], *, prefix: str
) -> tuple[list[str], np.ndarray]:
    curves = payload["strictly_unseen"]["curves"]
    full_labels = sorted(
        label
        for label in curves
        if label.startswith(prefix + "task_diagonal_lora_r48_seed")
    )
    labels = [label.removeprefix(prefix) for label in full_labels]
    return labels, np.asarray(
        [curves[label]["accuracy_by_cycle"] for label in full_labels],
        dtype=np.float64,
    )


def _prefix(values: np.ndarray, threshold: float) -> int:
    failed = np.flatnonzero(values < threshold)
    return int(failed[0]) if len(failed) else int(len(values))


def _auc(values: np.ndarray, start: int, stop: int) -> float:
    return float(values[start:stop].mean())


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    before_labels, before = _controller_curves(payload, prefix="before__")
    after_labels, after = _controller_curves(payload, prefix="after__")
    if before_labels != after_labels or before.shape != after.shape:
        raise ValueError("before/after audits do not contain matched controller seeds")
    before_mean, before_std = before.mean(0), before.std(0)
    after_mean, after_std = after.mean(0), after.std(0)
    curves_after = payload["strictly_unseen"]["curves"]
    raw = np.asarray(curves_after["identity_no_J"]["accuracy_by_cycle"], dtype=np.float64)
    loops = np.arange(1, len(after_mean) + 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, loop in enumerate(loops):
        row: dict[str, Any] = {
            "continuation_loop": int(loop),
            "training_support": bool(loop <= 8),
            "before_mean": float(before_mean[index]),
            "before_std_population": float(before_std[index]),
            "after_mean": float(after_mean[index]),
            "after_std_population": float(after_std[index]),
            "after_minus_before": float(after_mean[index] - before_mean[index]),
            "raw_no_J": float(raw[index]),
        }
        for label, values_before, values_after in zip(before_labels, before, after, strict=True):
            row[f"before_{label}"] = float(values_before[index])
            row[f"after_{label}"] = float(values_after[index])
        rows.append(row)
    with (args.out_dir / "before_after_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for axis, stop, title in (
        (axes[0], len(loops), "Full extrapolation range"),
        (axes[1], min(32, len(loops)), "Near-boundary detail"),
    ):
        visible = loops <= stop
        axis.axvspan(1, min(8, stop), color="#B8E0C2", alpha=0.28, label="J train support K<=8")
        if stop > 8:
            axis.axvspan(8, stop, color="#F3C3C3", alpha=0.14, label="unseen K>8")
        axis.plot(
            loops[visible],
            raw[visible],
            color="#888888",
            linestyle="--",
            linewidth=1.3,
            label="no-J continuation from H8 (hidden age H8+K)",
        )
        axis.plot(
            loops[visible],
            before_mean[visible],
            color="#6E8BFF",
            linewidth=2.0,
            label=args.before_label,
        )
        axis.fill_between(loops[visible], (before_mean-before_std)[visible], (before_mean+before_std)[visible], color="#6E8BFF", alpha=0.13)
        axis.plot(
            loops[visible],
            after_mean[visible],
            color="#E4572E",
            linewidth=2.5,
            label=args.after_label,
        )
        axis.fill_between(loops[visible], (after_mean-after_std)[visible], (after_mean+after_std)[visible], color="#E4572E", alpha=0.18)
        axis.axvline(8, color="black", linewidth=1.0)
        axis.set_xlim(1, stop)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("continuation count K after H8")
        axis.set_ylabel("successor accuracy on matched permutations")
        axis.set_title(title)
        axis.grid(alpha=0.18)
    axes[0].legend(fontsize=8, loc="lower left")
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"dense_h8_before_after.{suffix}", dpi=210 if suffix == "png" else None)
    plt.close(fig)

    segments = ((0, 8, "1_8"), (8, 16, "9_16"), (16, 32, "17_32"), (32, 64, "33_64"), (64, 128, "65_128"))
    result = {
        "status": "complete",
        "labels": before_labels,
        "evaluation_definition": payload["evaluation_definition"],
        "before_training_graphs": payload["old_training_graph_draws"],
        "after_training_graphs": payload["new_training_graph_draws"],
        "before_unique_training_graphs": payload["old_unique_training_graphs"],
        "after_unique_training_graphs": payload["new_unique_training_graphs"],
        "accuracy_at": {
            str(loop): {
                "before_mean": float(before_mean[loop-1]),
                "after_mean": float(after_mean[loop-1]),
                "delta": float(after_mean[loop-1]-before_mean[loop-1]),
                "before_values": [float(v) for v in before[:,loop-1]],
                "after_values": [float(v) for v in after[:,loop-1]],
            }
            for loop in (1,2,4,5,7,8,9,10,11,12,13,14,15,16,24,32,48,64,96,128)
        },
        "auc": {
            name: {
                "before": _auc(before_mean, start, stop),
                "after": _auc(after_mean, start, stop),
                "delta": _auc(after_mean, start, stop)-_auc(before_mean, start, stop),
            }
            for start, stop, name in segments
        },
        "reliable_prefix": {
            str(threshold): {
                "before": _prefix(before_mean, threshold),
                "after": _prefix(after_mean, threshold),
            }
            for threshold in (0.99,0.95,0.90)
        },
        "files": {"curve_csv":"before_after_curve.csv","figure_png":"dense_h8_before_after.png","figure_pdf":"dense_h8_before_after.pdf"},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--before-label", default="before: sparse horizons")
    parser.add_argument("--after-label", default="after: every K=1..8")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None=None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
