from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _curve(
    rows: list[dict[str, str]],
    *,
    group: str,
    controller: str,
) -> list[float]:
    selected = sorted(
        (
            row
            for row in rows
            if row["group"] == group
            and row["controller"] == controller
        ),
        key=lambda row: int(row["extra_loop"]),
    )
    return [float(row["accuracy"]) for row in selected]


def aggregate(
    *,
    exact_dir: Path,
    learned_dir: Path,
    replica_dir: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_rows = []
    curve_rows = []
    summaries: dict[str, Any] = {}
    exact_summaries = sorted(exact_dir.glob("D*/summary.json"))
    for exact_summary_path in exact_summaries:
        name = exact_summary_path.parent.name
        learned_summary_path = learned_dir / name / "summary.json"
        if not learned_summary_path.exists():
            continue
        exact = json.loads(
            exact_summary_path.read_text(encoding="utf-8")
        )
        learned = json.loads(
            learned_summary_path.read_text(encoding="utf-8")
        )
        exact_rows = _read_csv(
            exact_summary_path.with_name("lifespan_extension_rows.csv")
        )
        minimal = exact["minimal_four_loop_replacement"]
        exact_group = (
            minimal["group"]
            if minimal is not None
            else exact["best_exact_replacement"]["group"]
        )
        exact_curve = _curve(
            exact_rows,
            group=exact_group,
            controller="exact_replace",
        )
        additive_curve = _curve(
            exact_rows,
            group=exact_group,
            controller="additive",
        )
        shuffled_exact_curve = _curve(
            exact_rows,
            group=exact_group,
            controller="shuffled_replace",
        )
        wrong_node_curve = _curve(
            exact_rows,
            group=exact_group,
            controller="wrong_node_replace",
        )
        learned_curve = [
            float(value)
            for value in learned["accuracy_by_extra_loop"]
        ]
        learned_shuffled = [
            float(value)
            for value in learned["shuffled_accuracy_by_extra_loop"]
        ]
        baseline_curve = [
            float(value)
            for value in learned["baseline_accuracy_by_extra_loop"]
        ]
        model_rows.append(
            {
                "model": name,
                "reference_age": exact["candidate"]["reference_age"],
                "reference_path_before": exact["candidate"][
                    "reference_path_before"
                ],
                "programmed_jump": exact["candidate"][
                    "programmed_jump"
                ],
                "young_state_ceiling": exact["candidate"][
                    "selection_accuracy"
                ],
                "exact_group": exact_group,
                "exact_position_count": next(
                    row["position_count"]
                    for row in exact["groups"]
                    if row["group"] == exact_group
                ),
                "exact_mean_accuracy": float(np.mean(exact_curve)),
                "exact_loop8_accuracy": exact_curve[-1],
                "exact_shuffled_mean_accuracy": float(
                    np.mean(shuffled_exact_curve)
                ),
                "learned_group": learned["group"],
                "learned_position_count": learned["position_count"],
                "learned_mean_accuracy": float(
                    np.mean(learned_curve)
                ),
                "learned_loop8_accuracy": learned_curve[-1],
                "learned_shuffled_mean_accuracy": float(
                    np.mean(learned_shuffled)
                ),
                "baseline_mean_accuracy": float(
                    np.mean(baseline_curve)
                ),
                "peak_cuda_reserved_gib": learned[
                    "peak_cuda_reserved_gib"
                ],
            }
        )
        for extra_loop in range(1, len(exact_curve) + 1):
            for condition, curve in (
                ("baseline", baseline_curve),
                ("additive_terminal_delta", additive_curve),
                ("exact_matched_reset", exact_curve),
                ("exact_batch_shuffled", shuffled_exact_curve),
                ("exact_wrong_node", wrong_node_curve),
                ("learned_donor_free", learned_curve),
                ("learned_batch_shuffled", learned_shuffled),
            ):
                curve_rows.append(
                    {
                        "model": name,
                        "condition": condition,
                        "extra_loop": extra_loop,
                        "accuracy": curve[extra_loop - 1],
                    }
                )
        summaries[name] = {
            "candidate": exact["candidate"],
            "exact_group": exact_group,
            "exact_accuracy_by_extra_loop": exact_curve,
            "learned_group": learned["group"],
            "learned_accuracy_by_extra_loop": learned_curve,
            "learned_shuffled_accuracy_by_extra_loop": learned_shuffled,
            "baseline_accuracy_by_extra_loop": baseline_curve,
        }
    _write_csv(out_dir / "model_summary.csv", model_rows)
    _write_csv(out_dir / "lifespan_curves.csv", curve_rows)
    replica_rows = []
    if replica_dir is not None:
        for row in model_rows:
            name = str(row["model"])
            replica_path = replica_dir / name / "summary.json"
            if not replica_path.exists():
                continue
            replica = json.loads(
                replica_path.read_text(encoding="utf-8")
            )
            primary_curve = summaries[name][
                "learned_accuracy_by_extra_loop"
            ]
            replica_curve = [
                float(value)
                for value in replica["accuracy_by_extra_loop"]
            ]
            replica_rows.append(
                {
                    "model": name,
                    "primary_mean_accuracy": float(
                        np.mean(primary_curve)
                    ),
                    "replica_mean_accuracy": float(
                        np.mean(replica_curve)
                    ),
                    "primary_loop8_accuracy": primary_curve[-1],
                    "replica_loop8_accuracy": replica_curve[-1],
                    "mean_absolute_curve_difference": float(
                        np.mean(
                            np.abs(
                                np.asarray(primary_curve)
                                - np.asarray(replica_curve)
                            )
                        )
                    ),
                }
            )
    _write_csv(out_dir / "rejuvenator_replication.csv", replica_rows)
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )

    names = [row["model"] for row in model_rows]
    figure, axes = plt.subplots(
        2,
        len(names),
        figsize=(3.1 * len(names), 6.0),
        sharex=True,
        sharey=True,
    )
    if len(names) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, name in enumerate(names):
        exact = summaries[name]["exact_accuracy_by_extra_loop"]
        learned = summaries[name]["learned_accuracy_by_extra_loop"]
        shuffled = summaries[name][
            "learned_shuffled_accuracy_by_extra_loop"
        ]
        baseline = summaries[name]["baseline_accuracy_by_extra_loop"]
        loops = np.arange(1, len(exact) + 1)
        axes[0, column].plot(
            loops,
            exact,
            marker="o",
            label="matched reset",
        )
        axes[0, column].plot(
            loops,
            shuffled,
            marker="o",
            label="shuffled control",
        )
        axes[0, column].set_title(name)
        axes[1, column].plot(
            loops,
            learned,
            marker="o",
            label="learned rejuvenator",
        )
        axes[1, column].plot(
            loops,
            shuffled,
            marker="o",
            label="shuffled control",
        )
        axes[1, column].plot(
            loops,
            baseline,
            marker="o",
            label="no control",
        )
        for row in range(2):
            axes[row, column].axhline(
                1 / 8,
                color="gray",
                linestyle=":",
                linewidth=1,
            )
            axes[row, column].set_ylim(-0.03, 1.05)
            axes[row, column].grid(alpha=0.25)
            axes[row, column].set_xticks(loops)
        axes[1, column].set_xlabel("extra recurrent loop")
    axes[0, 0].set_ylabel("oracle matched-state accuracy")
    axes[1, 0].set_ylabel("donor-free accuracy")
    axes[0, -1].legend(fontsize=8, loc="lower left")
    axes[1, -1].legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(
        out_dir / "lifespan_extension.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    return {"models": summaries, "model_rows": model_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-dir", type=Path, required=True)
    parser.add_argument("--learned-dir", type=Path, required=True)
    parser.add_argument("--replica-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(
        exact_dir=args.exact_dir,
        learned_dir=args.learned_dir,
        replica_dir=args.replica_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
