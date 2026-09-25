from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "answer_position": "answer token p28",
    "start_register_position": "start register p26",
    "graph_destination_node0_position": "graph destination p3",
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(args.states)
    loops = data["loops"]
    mask = loops % 8 == 0
    matched_loops = loops[mask]
    metrics: dict[str, Any] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for key, display in LABELS.items():
        values = data[key][:, mask].astype(np.float64)
        mean = values.mean(axis=0)
        first = mean[1] - mean[0]
        denominator = max(float(first @ first), 1e-24)
        progress = (mean - mean[0]) @ first / denominator
        steps = np.diff(mean, axis=0)
        cosine = (steps @ first) / np.maximum(
            np.linalg.norm(steps, axis=-1) * np.linalg.norm(first), 1e-12
        )
        axes[0].plot(matched_loops, progress, marker="o", linewidth=2, label=display)
        axes[1].plot(matched_loops[1:], cosine, marker="o", linewidth=2, label=display)
        metrics[key] = {
            "normalized_progress": [float(v) for v in progress],
            "cosine_to_first_age_step": [float(v) for v in cosine],
        }
    expected = np.arange(len(matched_loops))
    axes[0].plot(matched_loops, expected, color="black", linestyle="--", linewidth=1.2, label="perfectly repeated first drift")
    axes[0].set_xlabel("raw hidden state after loop H")
    axes[0].set_ylabel("distance along the first age-drift direction\n(in units of H8->H16)")
    axes[0].set_title("How far has the state moved in the same age direction?")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("end of each 8-loop interval")
    axes[1].set_ylabel("cosine with the first H8->H16 drift")
    axes[1].set_title("Does each later interval move in the same direction?")
    axes[1].set_ylim(-0.05, 1.05)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Raw F only; graph, content, and token position are matched every 8 loops")
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"raw_age_direction_simple.{suffix}", dpi=210 if suffix == "png" else None)
    plt.close(fig)
    result = {
        "status": "complete",
        "definition": "No PCA: project population-mean H8,H16,... states onto the first H8->H16 drift and compare later 8-loop drift directions in original 256-D space.",
        "matched_loops": [int(v) for v in matched_loops],
        "metrics": metrics,
        "files": {"png": "raw_age_direction_simple.png", "pdf": "raw_age_direction_simple.pdf"},
    }
    (args.out_dir / "raw_age_direction_simple.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
