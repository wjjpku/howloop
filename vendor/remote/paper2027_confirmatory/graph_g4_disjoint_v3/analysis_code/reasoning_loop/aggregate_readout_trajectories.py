from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def extract_readout_trajectory(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    acc_to_pos = summary["final_metrics"]["acc_to_pos"]
    if not acc_to_pos or any(len(row) < len(acc_to_pos) for row in acc_to_pos):
        raise ValueError("acc_to_pos must cover one matched path position per readout")
    return {
        "model": name,
        "family": re.sub(r"_seed\d+$", "", name),
        "final_target_curve": [float(row[-1]) for row in acc_to_pos],
        "matched_step_curve": [float(row[index]) for index, row in enumerate(acc_to_pos)],
    }


def aggregate_trajectories(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    output: dict[str, dict[str, Any]] = {}
    for family, family_rows in sorted(grouped.items()):
        final_curves = np.array([row["final_target_curve"] for row in family_rows])
        matched_curves = np.array([row["matched_step_curve"] for row in family_rows])
        output[family] = {
            "seeds": len(family_rows),
            "final_target_mean": final_curves.mean(axis=0).tolist(),
            "final_target_std": final_curves.std(axis=0).tolist(),
            "matched_step_mean": matched_curves.mean(axis=0).tolist(),
            "matched_step_std": matched_curves.std(axis=0).tolist(),
        }
    return output


def plot_trajectories(families: dict[str, dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = [
        ("final_target", "Accuracy to final target f^6", "When final answer becomes readable"),
        ("matched_step", "Accuracy to f^t", "Whether readout t matches algorithm step t"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        for family, values in families.items():
            mean = np.array(values[f"{metric}_mean"])
            std = np.array(values[f"{metric}_std"])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, marker="o", label=family)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18)
        ax.axhline(0.125, color="black", linestyle=":", linewidth=1, label="random N=8")
        ax.set_xlabel("effective macro-step")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.03)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(2, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare readout trajectories across model families.")
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/summary.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"run spec must be NAME=/path/to/summary.json, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        summary = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        rows.append(extract_readout_trajectory(name, summary))
    families = aggregate_trajectories(rows)
    output = {"models": rows, "families": families}
    (args.out_dir / "readout_trajectory_summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    plot_trajectories(families, args.out_dir / "looped_vs_standard_readout_trajectories.png")


if __name__ == "__main__":
    main()
