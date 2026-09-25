"""Summarize the fixed-H1, identity-initialized shared/stage rank grid."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


RANK_PATTERN = re.compile(r"/r(?P<shared>\d+)_s(?P<stage>\d+)/")
SPLITS = (
    "single_J",
    "focused_mixture_a",
    "focused_mixture_b",
    "hard_boundary_T05_R5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path, artifact_root: Path | None) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            match = RANK_PATTERN.search(source["artifact"])
            if match is None:
                raise ValueError(f"cannot parse ranks from {source['artifact']}")
            shared = int(match.group("shared"))
            stage = int(match.group("stage"))
            key = (shared, stage)
            row = grouped.setdefault(
                key,
                {
                    "shared_rank": shared,
                    "stage_rank": stage,
                    "artifact": source["artifact"],
                },
            )
            row[source["split"]] = float(source["accuracy_mean"])
    rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        row = grouped[key]
        missing = set(SPLITS) - row.keys()
        if missing:
            raise ValueError(f"missing splits for {key}: {sorted(missing)}")
        artifact = Path(str(row["artifact"]))
        if artifact_root is not None:
            artifact = (
                artifact_root
                / f"r{row['shared_rank']}_s{row['stage_rank']}"
                / "focused5_h1"
                / artifact.name
            )
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        diagonal = state["shared_diagonal_scale"].float()
        shared_update = state["shared_A"].float() @ state["shared_B"].float()
        stage_norms = []
        for age in range(2, 9):
            update = state[f"stage_A.{age}"].float() @ state[f"stage_B.{age}"].float()
            stage_norms.append(float(update.norm()))
        mixture_mean = 0.5 * (
            float(row["focused_mixture_a"]) + float(row["focused_mixture_b"])
        )
        macro_mean = float(np.mean([float(row[split]) for split in SPLITS]))
        row.update(
            parameter_count=int(payload["parameter_count"]),
            mixture_mean=mixture_mean,
            macro_mean=macro_mean,
            diagonal_delta_rms=float((diagonal - 1).square().mean().sqrt()),
            diagonal_delta_max_abs=float((diagonal - 1).abs().max()),
            shared_update_frobenius=float(shared_update.norm()),
            stage_update_frobenius_mean=float(np.mean(stage_norms)),
        )
        rows.append(row)
    return rows


def heatmap(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    key: str,
    title: str,
    *,
    percent: bool = False,
) -> None:
    shared_ranks = sorted({int(row["shared_rank"]) for row in rows})
    stage_ranks = sorted({int(row["stage_rank"]) for row in rows})
    values = np.full((len(stage_ranks), len(shared_ranks)), np.nan)
    for row in rows:
        i = stage_ranks.index(int(row["stage_rank"]))
        j = shared_ranks.index(int(row["shared_rank"]))
        values[i, j] = float(row[key])
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    for i in range(len(stage_ranks)):
        for j in range(len(shared_ranks)):
            value = values[i, j]
            label = f"{100 * value:.1f}%" if percent else f"{value:.3g}"
            ax.text(j, i, label, ha="center", va="center", color="white" if value < np.nanmean(values) else "black")
    ax.set_xticks(range(len(shared_ranks)), shared_ranks)
    ax.set_yticks(range(len(stage_ranks)), stage_ranks)
    ax.set_xlabel("shared rank")
    ax.set_ylabel("stage rank")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.evaluation_csv, args.artifact_root)
    write_csv(args.out_dir / "rank_grid_summary.csv", rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    heatmap(axes[0, 0], rows, "single_J", "Single rollback accuracy", percent=True)
    heatmap(axes[0, 1], rows, "mixture_mean", "Mixed trajectory accuracy", percent=True)
    heatmap(axes[0, 2], rows, "hard_boundary_T05_R5", "Five consecutive rollback accuracy", percent=True)
    heatmap(axes[1, 0], rows, "macro_mean", "Four-split macro accuracy", percent=True)
    heatmap(axes[1, 1], rows, "diagonal_delta_rms", "RMS change of shared diagonal")
    heatmap(axes[1, 2], rows, "parameter_count", "Trainable parameters")
    fig.suptitle("Fixed-H1 identity initialization: shared/stage rank grid", fontsize=16)
    fig.savefig(args.out_dir / "rank_grid_summary.png", dpi=180)
    plt.close(fig)
    best = max(rows, key=lambda row: float(row["macro_mean"]))
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "definition": "fixed H1 start, identity initialization, product composition, final CE only",
                "best_by_macro_accuracy": best,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
