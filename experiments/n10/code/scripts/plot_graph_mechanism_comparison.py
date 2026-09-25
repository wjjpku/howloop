#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_panel(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("panel must be TITLE=/path/to/summary.json")
    title, path = text.split("=", 1)
    if not title:
        raise ValueError("panel title must not be empty")
    return title, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loop-by-path matrices for several graph-path mechanisms."
    )
    parser.add_argument(
        "--panel",
        action="append",
        required=True,
        help="TITLE=/path/to/summary.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    panels = [parse_panel(spec) for spec in args.panel]
    matrices: list[np.ndarray] = []
    for _, path in panels:
        summary = json.loads(path.read_text(encoding="utf-8"))
        matrices.append(
            np.asarray(summary["final_metrics"]["acc_to_pos"], dtype=np.float64)
        )
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.3 * len(panels), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    image = None
    for axis, (title, _), matrix in zip(axes_array, panels, matrices, strict=True):
        image = axis.imshow(
            matrix,
            vmin=0,
            vmax=1,
            cmap="viridis",
            aspect="equal",
        )
        depth = np.arange(1, matrix.shape[0] + 1)
        axis.set_xticks(depth - 1, depth)
        axis.set_yticks(depth - 1, depth)
        axis.set_xlabel("path position k")
        axis.set_title(title)
    axes_array[0].set_ylabel("readout loop t")
    assert image is not None
    figure.colorbar(
        image,
        ax=axes_array.tolist(),
        label="accuracy",
        shrink=0.82,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
