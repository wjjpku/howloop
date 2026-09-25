from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results/graph_path_postnorm_tangent_rejuvenator_20260730")
STRICT_ROOT = ROOT / "strict_reeval"
SERIES = (
    ("tangent_terminal_reset_all", "Tangent, one-step pairs", "#94a3b8", "--"),
    ("tangent_rollout4", "Tangent, unroll-4", "#2563eb", "-"),
    ("euclidean_rollout4", "Euclidean, unroll-4", "#dc2626", "-"),
    ("tangent_rollout8", "Tangent, unroll-8", "#059669", "-"),
)


def read_curve(name: str) -> tuple[list[int], list[float]]:
    with (STRICT_ROOT / f"{name}.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [int(row["cycle"]) for row in rows],
        [float(row["strict_novel_target_accuracy"]) for row in rows],
    )


def main() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=220)
    for name, label, color, linestyle in SERIES:
        cycles, accuracy = read_curve(name)
        for axis in axes:
            axis.plot(
                cycles,
                accuracy,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
            )
    for axis in axes:
        axis.axhline(
            1 / 8,
            color="#64748b",
            linewidth=1,
            linestyle=":",
            label="8-way chance",
        )
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Repeated rejuvenation cycle")
        axis.set_ylabel("Strict novel-target accuracy")
        axis.grid(True, alpha=0.25)
    axes[0].set_xlim(1, 20)
    axes[0].axvline(4, color="#7c3aed", alpha=0.45, linestyle=":")
    axes[0].axvline(8, color="#7c3aed", alpha=0.45, linestyle=":")
    axes[0].set_title("Boundary follows the closed-loop training window")
    axes[1].set_xlim(1, 200)
    axes[1].set_title("No tested map extrapolates to 200 cycles")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.94)
    figure.suptitle(
        "Post-Norm tangent-space rejuvenation on D8L8\n"
        "Held-out graphs; targets equal to the original endpoint or previous node excluded",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(ROOT / "strict_rejuvenation_length_curves.png", bbox_inches="tight")
    figure.savefig(ROOT / "strict_rejuvenation_length_curves.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
