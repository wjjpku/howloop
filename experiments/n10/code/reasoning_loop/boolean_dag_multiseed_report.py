from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_NAME = re.compile(r"^(looped|periodic2|standard)_(final|intermediate)_seed(\d+)$")
METRICS = (
    "root_accuracy",
    "root_probability",
    "state_accuracy",
    "exact_state_accuracy",
    "premature_rate",
    "delayed_rate",
    "resolved_value_accuracy",
)
INDEPENDENT_BLOCKS = {
    "looped_intermediate": 1,
    "periodic2_intermediate": 2,
    "standard_intermediate": 4,
}


def collect_evaluation_rows(run_roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_roots:
        for path in sorted(root.glob("*_seed*/eval*/summary.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            match = RUN_NAME.match(path.parent.parent.name)
            if match is None:
                continue
            architecture, loss_mode, seed_text = match.groups()
            summary = json.loads(path.read_text(encoding="utf-8"))
            distribution = summary.get(
                "evaluation_distribution",
                "topology_matched_depth8_master"
                if path.parent.name == "eval_topology_matched"
                else "native_depth_specific",
            )
            for metric in METRICS:
                if metric not in summary:
                    continue
                for depth_index, depth in enumerate(summary["depths"]):
                    for readout_index, readout in enumerate(summary["readouts"]):
                        rows.append(
                            {
                                "architecture": architecture,
                                "loss_mode": loss_mode,
                                "condition": f"{architecture}_{loss_mode}",
                                "seed": int(seed_text),
                                "evaluation_distribution": distribution,
                                "depth": int(depth),
                                "readout": int(readout),
                                "metric": metric,
                                "value": float(summary[metric][depth_index][readout_index]),
                                "source": str(path),
                            }
                        )
    return rows


def aggregate_evaluation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    fields = (
        "architecture",
        "loss_mode",
        "condition",
        "evaluation_distribution",
        "depth",
        "readout",
        "metric",
    )
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row["value"])
    result: list[dict[str, Any]] = []
    for key, all_values in sorted(grouped.items()):
        values = [value for value in all_values if math.isfinite(value)]
        undefined = float("nan")
        result.append(
            {
                **dict(zip(fields, key)),
                "total_seed_count": len(all_values),
                "seed_count": len(values),
                "mean": statistics.mean(values) if values else undefined,
                "std": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                    if values
                    else undefined
                ),
                "min": min(values) if values else undefined,
                "max": max(values) if values else undefined,
            }
        )
    return result


def collect_causal_rows(run_roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_roots:
        for path in sorted(
            root.glob("*_intermediate_seed*/causal_topology_matched/depth_metrics.csv")
        ):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            match = RUN_NAME.match(path.parent.parent.name)
            if match is None:
                continue
            architecture, loss_mode, seed_text = match.groups()
            seed = int(seed_text)
            summary_path = path.parent / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            minimal = summary.get(
                "minimal_95pct_sufficient_and_necessary_subset_by_depth", {}
            )
            with path.open(encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    depth = int(raw["depth"])
                    subset = minimal.get(str(depth))
                    row: dict[str, Any] = {
                        "architecture": architecture,
                        "loss_mode": loss_mode,
                        "condition": f"{architecture}_{loss_mode}",
                        "seed": seed,
                        "depth": depth,
                        "minimal_head_subset": subset,
                        "minimal_head_subset_size": (
                            0 if subset == "none" else len(subset.split("+"))
                        )
                        if subset is not None
                        else None,
                        "source": str(path),
                    }
                    for key, value in raw.items():
                        if key == "depth":
                            continue
                        row[key] = float(value)
                    rows.append(row)
    return rows


def aggregate_causal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = {
        "architecture",
        "loss_mode",
        "condition",
        "seed",
        "depth",
        "minimal_head_subset",
        "source",
    }
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric, value in row.items():
            if metric in metadata or value is None:
                continue
            grouped[(row["condition"], row["depth"], metric)].append(float(value))
    result = []
    for (condition, depth, metric), all_values in sorted(grouped.items()):
        values = [value for value in all_values if math.isfinite(value)]
        result.append(
            {
                "condition": condition,
                "depth": depth,
                "metric": metric,
                "total_seed_count": len(all_values),
                "seed_count": len(values),
                "mean": statistics.mean(values) if values else float("nan"),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values) if values else float("nan"),
                "max": max(values) if values else float("nan"),
            }
        )
    return result


def sharing_continuum_rows(
    rows: list[dict[str, Any]],
    *,
    depth: int,
    metric: str,
) -> list[dict[str, Any]]:
    selected = [
        {**row, "independent_blocks": INDEPENDENT_BLOCKS[row["condition"]]}
        for row in rows
        if row["condition"] in INDEPENDENT_BLOCKS
        and row["depth"] == depth
        and row["metric"] == metric
    ]
    return sorted(selected, key=lambda row: row["independent_blocks"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _matched_series(
    aggregate: list[dict[str, Any]],
    *,
    condition: str,
    distribution: str,
    metric: str,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in aggregate
        if row["condition"] == condition
        and row["evaluation_distribution"] == distribution
        and row["metric"] == metric
    ]
    result = []
    for depth in sorted({row["depth"] for row in candidates}):
        at_depth = [row for row in candidates if row["depth"] == depth]
        desired = depth if any(row["readout"] == depth for row in at_depth) else max(
            row["readout"] for row in at_depth
        )
        result.append(next(row for row in at_depth if row["readout"] == desired))
    return result


def _plot_series(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    if not rows:
        return
    x = np.asarray([row["depth"] for row in rows])
    mean = np.asarray([row["mean"] for row in rows])
    std = np.asarray([row["std"] for row in rows])
    ax.plot(x, mean, marker="o", color=color, linestyle=linestyle, label=label)
    ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.12)


def plot_sharing_continuum(
    *,
    evaluation_aggregate: list[dict[str, Any]],
    causal_aggregate: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    colors = {5: "#0072B2", 6: "#D55E00"}
    for depth in (5, 6):
        causal = sharing_continuum_rows(
            causal_aggregate,
            depth=depth,
            metric="parent_swap_follows_donor_accuracy",
        )
        axes[0].errorbar(
            [row["independent_blocks"] for row in causal],
            [row["mean"] for row in causal],
            yerr=[row["std"] for row in causal],
            marker="o",
            capsize=3,
            color=colors[depth],
            label=f"D{depth}",
        )
        behavior = []
        for condition, independent_blocks in INDEPENDENT_BLOCKS.items():
            matched = _matched_series(
                evaluation_aggregate,
                condition=condition,
                distribution="topology_matched_depth8_master",
                metric="root_accuracy",
            )
            behavior.extend(
                {**row, "independent_blocks": independent_blocks}
                for row in matched
                if row["depth"] == depth
            )
        behavior.sort(key=lambda row: row["independent_blocks"])
        axes[1].errorbar(
            [row["independent_blocks"] for row in behavior],
            [row["mean"] for row in behavior],
            yerr=[row["std"] for row in behavior],
            marker="o",
            capsize=3,
            color=colors[depth],
            label=f"D{depth}",
        )
    for ax, title, ylabel in (
        (
            axes[0],
            "Causal transition reusability",
            "parent swap follows donor",
        ),
        (
            axes[1],
            "Topology-matched depth generalization",
            "root accuracy",
        ),
    ):
        ax.set(
            xlabel="number of independently parameterized blocks",
            ylabel=ylabel,
            ylim=(-0.03, 1.03),
            xticks=[1, 2, 4],
            title=title,
        )
        ax.set_xticklabels(["1\nfully shared", "2\nperiodic", "4\nuntied"])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_multiseed_report(
    *,
    run_roots: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    rows = collect_evaluation_rows(run_roots)
    if not rows:
        raise ValueError("no Boolean DAG evaluation summaries were found")
    aggregate = aggregate_evaluation_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "evaluation_seed_rows.csv", rows)
    _write_csv(out_dir / "evaluation_aggregate.csv", aggregate)
    causal_rows = collect_causal_rows(run_roots)
    causal_aggregate: list[dict[str, Any]] = []
    if causal_rows:
        causal_aggregate = aggregate_causal_rows(causal_rows)
        _write_csv(out_dir / "causal_seed_rows.csv", causal_rows)
        _write_csv(out_dir / "causal_aggregate.csv", causal_aggregate)

    conditions = {
        "looped_intermediate": ("Looped + intermediate", "#0072B2"),
        "periodic2_intermediate": ("Periodic-2 + intermediate", "#E69F00"),
        "standard_intermediate": ("Standard + intermediate", "#D55E00"),
        "looped_final": ("Looped + final", "#009E73"),
        "standard_final": ("Standard + final", "#CC79A7"),
    }
    native = "native_depth_specific"
    topology = "topology_matched_depth8_master"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    for condition, (label, color) in conditions.items():
        _plot_series(
            axes[0, 0],
            _matched_series(
                aggregate,
                condition=condition,
                distribution=native,
                metric="root_accuracy",
            ),
            label=label,
            color=color,
        )
        _plot_series(
            axes[0, 1],
            _matched_series(
                aggregate,
                condition=condition,
                distribution=topology,
                metric="root_accuracy",
            ),
            label=label,
            color=color,
        )
    for ax, title in (
        (axes[0, 0], "Native OOD: root accuracy at matched readout"),
        (axes[0, 1], "Topology-matched OOD: root accuracy at matched readout"),
    ):
        ax.set(xlabel="queried/root depth", ylabel="accuracy", ylim=(-0.03, 1.03), title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    for distribution, label, linestyle in (
        (native, "Looped intermediate: native", "-"),
        (topology, "Looped intermediate: topology-matched", "--"),
    ):
        _plot_series(
            axes[1, 0],
            _matched_series(
                aggregate,
                condition="looped_intermediate",
                distribution=distribution,
                metric="exact_state_accuracy",
            ),
            label=label,
            color="#0072B2",
            linestyle=linestyle,
        )
    axes[1, 0].set(
        xlabel="queried/root depth",
        ylabel="whole-graph exact accuracy",
        ylim=(-0.03, 1.03),
        title="Root-path generalization is stronger than full-graph wavefront",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    for condition, label, color in (
        ("looped_intermediate", "Looped + intermediate", "#0072B2"),
        ("periodic2_intermediate", "Periodic-2 + intermediate", "#E69F00"),
        ("standard_intermediate", "Standard + intermediate", "#D55E00"),
    ):
        selected = [
            row
            for row in aggregate
            if row["condition"] == condition
            and row["evaluation_distribution"] == topology
            and row["metric"] == "premature_rate"
            and row["readout"] == 4
        ]
        _plot_series(axes[1, 1], selected, label=label, color=color)
    axes[1, 1].set(
        xlabel="queried root depth",
        ylabel="premature resolution rate at readout 4",
        ylim=(-0.03, 1.03),
        title="Independent block 4 behaves as a terminal writer",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "multiseed_overview.png", dpi=180, facecolor="white")
    fig.savefig(out_dir / "multiseed_overview.svg", facecolor="white")
    plt.close(fig)

    if causal_aggregate:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        fig.patch.set_facecolor("white")
        causal_seed_counts = {
            condition: len({row["seed"] for row in causal_rows if row["condition"] == condition})
            for condition in {row["condition"] for row in causal_rows}
        }
        causal_panels = (
            (
                "parent_pair_readout_correct_rate",
                "Both parent states are already readable",
                "paired-parent accuracy",
            ),
            (
                "parent_swap_follows_donor_accuracy",
                "Parent-state interchange follows counterfactual",
                "donor-target accuracy",
            ),
            (
                "nonparent_swap_keeps_base_accuracy",
                "Matched non-parent interchange control",
                "base-target accuracy",
            ),
            (
                "parent_rollback_keeps_base_accuracy",
                "Rollback true parents by one state step",
                "base-target accuracy",
            ),
            (
                "minimal_head_subset_size",
                "Minimal 95% sufficient and necessary subset",
                "head count",
            ),
        )
        for ax, (metric, title, ylabel) in zip(axes.flat, causal_panels):
            for condition, label, color in (
                ("looped_intermediate", "Looped + intermediate", "#0072B2"),
                ("periodic2_intermediate", "Periodic-2 + intermediate", "#E69F00"),
                ("standard_intermediate", "Standard + intermediate", "#D55E00"),
            ):
                selected = [
                    row
                    for row in causal_aggregate
                    if row["condition"] == condition and row["metric"] == metric
                ]
                if metric == "minimal_head_subset_size":
                    selected = [
                        row
                        for row in selected
                        if row["seed_count"] == causal_seed_counts.get(condition, 0)
                    ]
                _plot_series(ax, selected, label=label, color=color)
            upper = 4.3 if metric == "minimal_head_subset_size" else 1.03
            ax.set(xlabel="queried root depth", ylabel=ylabel, ylim=(-0.03, upper), title=title)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "multiseed_causal.png", dpi=180, facecolor="white")
        fig.savefig(out_dir / "multiseed_causal.svg", facecolor="white")
        plt.close(fig)
        plot_sharing_continuum(
            evaluation_aggregate=aggregate,
            causal_aggregate=causal_aggregate,
            path=out_dir / "sharing_continuum.png",
        )
        plot_sharing_continuum(
            evaluation_aggregate=aggregate,
            causal_aggregate=causal_aggregate,
            path=out_dir / "sharing_continuum.svg",
        )

    seeds = sorted({row["seed"] for row in rows})
    key_rows = []
    for distribution in (native, topology):
        for condition in conditions:
            for row in _matched_series(
                aggregate,
                condition=condition,
                distribution=distribution,
                metric="root_accuracy",
            ):
                if row["depth"] in (5, 6, 7, 8):
                    key_rows.append(row)
    _write_csv(out_dir / "key_root_metrics.csv", key_rows)
    summary = {
        "seeds": seeds,
        "conditions": sorted({row["condition"] for row in rows}),
        "evaluation_distributions": sorted(
            {row["evaluation_distribution"] for row in rows}
        ),
        "evaluation_summary_count": len(
            {row["source"] for row in rows}
        ),
        "causal_summary_count": len(
            {row["source"] for row in causal_rows}
        ),
        "causal_aggregate": causal_aggregate,
        "key_root_metrics": key_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed Boolean DAG evaluations.")
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_multiseed_report(run_roots=args.run_roots, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
