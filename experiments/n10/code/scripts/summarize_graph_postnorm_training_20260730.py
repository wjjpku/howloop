from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_dir(root: Path, seed: int, *, postnorm: bool) -> Path:
    family = (
        f"D8_L8_postnorm_seed{seed}" if postnorm else f"D8_L8_seed{seed}"
    )
    return root / family / f"graphpath_N8_D8_d256_B2_L8_seed{seed}"


def summarize_run(path: Path, seed: int, condition: str) -> dict[str, object]:
    summary = json.loads((path / "summary.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    history = json.loads((path / "history.json").read_text())
    final_accuracies = [float(row["loop_accuracy"][-1]) for row in history]

    def first_step(threshold: float) -> int | None:
        return next(
            (
                int(row["step"])
                for row in history
                if float(row["loop_accuracy"][-1]) >= threshold
            ),
            None,
        )

    return {
        "condition": condition,
        "seed": seed,
        "parameter_count": int(summary["parameter_count"]),
        "inner_norm_style": metadata["config"]["inner_norm_style"],
        "warmup_steps": int(metadata["args"]["warmup_steps"]),
        "best_step": int(summary["best_step"]),
        "best_loop8_accuracy": float(summary["best_final_accuracy"]),
        "final_loop8_accuracy": float(
            summary["final_metrics"]["loop_accuracy"][-1]
        ),
        "first_step_loop8_accuracy_ge_0p9": first_step(0.9),
        "first_step_loop8_accuracy_ge_0p99": first_step(0.99),
        "minimum_recorded_loop8_accuracy": min(final_accuracies),
        "maximum_recorded_loop8_accuracy": max(final_accuracies),
        "peak_cuda_memory_allocated_mib": summary.get(
            "peak_cuda_memory_allocated_mib"
        ),
        "peak_cuda_memory_reserved_mib": summary.get(
            "peak_cuda_memory_reserved_mib"
        ),
        "elapsed_sec": float(history[-1]["elapsed_sec"]),
    }


def aggregate(rows: list[dict[str, object]], condition: str) -> dict[str, object]:
    selected = [row for row in rows if row["condition"] == condition]
    best = np.asarray(
        [float(row["best_loop8_accuracy"]) for row in selected]
    )
    final = np.asarray(
        [float(row["final_loop8_accuracy"]) for row in selected]
    )
    return {
        "condition": condition,
        "seed_count": len(selected),
        "successful_best_ge_0p9": int((best >= 0.9).sum()),
        "successful_final_ge_0p9": int((final >= 0.9).sum()),
        "best_accuracy_mean": float(best.mean()),
        "best_accuracy_min": float(best.min()),
        "best_accuracy_max": float(best.max()),
        "final_accuracy_mean": float(final.mean()),
        "final_accuracy_min": float(final.min()),
        "final_accuracy_max": float(final.max()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-root", type=Path, required=True)
    parser.add_argument("--pre-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    histories: dict[tuple[str, int], list[dict[str, object]]] = {}
    for condition, root, postnorm in (
        ("pre_layernorm_warmup500", args.pre_root, False),
        ("post_layernorm_warmup2000", args.post_root, True),
    ):
        for seed in range(6):
            path = run_dir(root, seed, postnorm=postnorm)
            rows.append(summarize_run(path, seed, condition))
            histories[(condition, seed)] = json.loads(
                (path / "history.json").read_text()
            )

    aggregates = [
        aggregate(rows, "pre_layernorm_warmup500"),
        aggregate(rows, "post_layernorm_warmup2000"),
    ]
    write_csv(args.out_dir / "training_rows.csv", rows)
    write_csv(args.out_dir / "training_aggregate.csv", aggregates)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180, sharey=True)
    for axis, condition in zip(
        axes,
        ("pre_layernorm_warmup500", "post_layernorm_warmup2000"),
        strict=True,
    ):
        for seed in range(6):
            history = histories[(condition, seed)]
            axis.plot(
                [row["step"] for row in history],
                [row["loop_accuracy"][-1] for row in history],
                label=f"seed {seed}",
                linewidth=1.5,
            )
        axis.set_title(condition.replace("_", " "))
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("held-out loop-8 accuracy")
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        args.out_dir / "paired_sixseed_training_curves.png",
        bbox_inches="tight",
    )
    plt.close(figure)

    payload = {
        "comparison": (
            "same D8L8 task/model/optimizer/seeds; intended differences are "
            "pre-vs-post LayerNorm and warmup 500-vs-2000"
        ),
        "rows": rows,
        "aggregate": aggregates,
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
