from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _normalized_restore(
    value: pd.Series | np.ndarray,
    *,
    full: float,
    damaged: float,
) -> pd.Series | np.ndarray:
    denominator = full - damaged
    if abs(denominator) < 1e-9:
        return np.full_like(value, np.nan, dtype=float)
    return (value - damaged) / denominator


def _normalized_corrupt(
    value: pd.Series | np.ndarray,
    *,
    full: float,
    damaged: float,
) -> pd.Series | np.ndarray:
    denominator = full - damaged
    if abs(denominator) < 1e-9:
        return np.full_like(value, np.nan, dtype=float)
    return (full - value) / denominator


def _baseline_map(frame: pd.DataFrame, metric: str) -> dict[tuple[int, str], float]:
    baseline = frame[frame["family"] == "baseline"]
    return {
        (int(row.cycle), str(row.condition)): float(getattr(row, metric))
        for row in baseline.itertuples()
    }


def _add_recovery(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    result = frame.copy()
    baselines = _baseline_map(frame, metric)
    recovered: list[float] = []
    for row in result.itertuples():
        full = baselines[(int(row.cycle), "full")]
        damaged = baselines[(int(row.cycle), "shuffled")]
        value = float(getattr(row, metric))
        if row.mode == "restore":
            score = float(_normalized_restore(np.asarray([value]), full=full, damaged=damaged)[0])
        elif row.mode == "corrupt":
            score = float(_normalized_corrupt(np.asarray([value]), full=full, damaged=damaged)[0])
        elif row.condition == "full":
            score = 1.0
        else:
            score = 0.0
        recovered.append(score)
    result[f"{metric}_normalized_effect"] = recovered
    return result


def _threshold_rows(frame: pd.DataFrame, metric: str, cycle: int) -> list[dict[str, object]]:
    score = f"{metric}_normalized_effect"
    rows: list[dict[str, object]] = []
    selected = frame[(frame.cycle == cycle) & (~frame.family.isin(["baseline", "random"]))]
    for (family, mode), group in selected.groupby(["family", "mode"]):
        ordered = group.sort_values("coordinate_count")
        for threshold in (0.5, 0.8, 0.9):
            passing = ordered[ordered[score] >= threshold]
            rows.append(
                {
                    "cycle": cycle,
                    "metric": metric,
                    "family": family,
                    "mode": mode,
                    "threshold": threshold,
                    "minimum_coordinate_count": (
                        int(passing.coordinate_count.min()) if len(passing) else None
                    ),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    static = pd.read_csv(args.run_dir / "coordinate_static.csv")
    single = pd.read_csv(args.run_dir / "single_coordinate.csv")
    groups = pd.read_csv(args.run_dir / "group_curves.csv")
    positions = pd.read_csv(args.run_dir / "position_curves.csv")
    groups = _add_recovery(groups, "target_margin")
    groups = _add_recovery(groups, "accuracy")
    positions = _add_recovery(positions, "target_margin")
    positions = _add_recovery(positions, "accuracy")

    screen_cycle = int(single.cycle.max())
    full_margin = float(
        single[(single.cycle == screen_cycle) & (single.condition == "full")].target_margin.iloc[0]
    )
    single_effect = single[
        (single.cycle == screen_cycle) & (single.family == "single_coordinate")
    ].copy()
    single_effect["coordinate"] = single_effect.coordinates.astype(int)
    single_effect["single_margin_drop"] = full_margin - single_effect.target_margin
    merged = static.merge(
        single_effect[["coordinate", "single_margin_drop"]], on="coordinate"
    )
    correlation_fields = [
        "abs_D_minus_1",
        "abs_D_minus_shuffled",
        "B2H0_q_input_l2",
        "B2H0_k_input_l2",
        "B2H0_v_input_l2",
        "B2H3_q_input_l2",
        "B2H3_k_input_l2",
        "B2H3_v_input_l2",
        "B2_mlp_input_l2",
        "trajectory_D_signal_answer",
        "trajectory_D_signal_graph",
        "trajectory_D_signal_all",
    ]
    correlation_rows: list[dict[str, object]] = []
    for field in correlation_fields:
        if field not in merged:
            continue
        statistic, pvalue = spearmanr(merged[field], merged.single_margin_drop)
        correlation_rows.append(
            {
                "feature": field,
                "spearman_r": float(statistic),
                "p_value": float(pvalue),
            }
        )
    correlation = pd.DataFrame(correlation_rows).sort_values(
        "spearman_r", ascending=False
    )

    random_summary = (
        groups[groups.family == "random"]
        .groupby(["cycle", "mode", "coordinate_count"], as_index=False)
        .agg(
            target_margin_mean=("target_margin", "mean"),
            target_margin_std=("target_margin", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            target_margin_effect_mean=("target_margin_normalized_effect", "mean"),
            target_margin_effect_std=("target_margin_normalized_effect", "std"),
        )
    )
    thresholds = pd.DataFrame(
        _threshold_rows(groups, "target_margin", args.cycle)
        + _threshold_rows(groups, "accuracy", args.cycle)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "coordinate_effects.csv", index=False)
    correlation.to_csv(args.out_dir / "coordinate_feature_correlations.csv", index=False)
    groups.to_csv(args.out_dir / "group_curves_with_recovery.csv", index=False)
    positions.to_csv(args.out_dir / "position_curves_with_recovery.csv", index=False)
    random_summary.to_csv(args.out_dir / "random_summary.csv", index=False)
    thresholds.to_csv(args.out_dir / "coordinate_thresholds.csv", index=False)

    cycle_groups = groups[groups.cycle == args.cycle]
    cycle_positions = positions[positions.cycle == args.cycle]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    ax = axes[0, 0]
    top = merged.nlargest(32, "single_margin_drop")
    ax.bar(np.arange(len(top)), top.single_margin_drop, color="#4C78A8")
    ax.set_xticks(np.arange(len(top)), top.coordinate.astype(str), rotation=90)
    ax.set(title="Top single-coordinate necessity", ylabel="Cycle-64 margin drop", xlabel="D coordinate")
    ax.grid(axis="y", alpha=0.2)

    for mode, ax in (("restore", axes[0, 1]), ("corrupt", axes[1, 0])):
        plotted = cycle_groups[
            (cycle_groups["mode"] == mode)
            & (~cycle_groups.family.isin(["baseline", "random"]))
        ]
        for family, group in plotted.groupby("family"):
            ordered = group.sort_values("coordinate_count")
            ax.plot(
                ordered.coordinate_count,
                ordered.target_margin_normalized_effect,
                marker="o",
                label=family,
            )
        random = random_summary[
            (random_summary.cycle == args.cycle) & (random_summary["mode"] == mode)
        ].sort_values("coordinate_count")
        if len(random):
            x = random.coordinate_count.to_numpy()
            y = random.target_margin_effect_mean.to_numpy()
            e = random.target_margin_effect_std.fillna(0).to_numpy()
            ax.plot(x, y, color="black", linestyle="--", label="random mean")
            ax.fill_between(x, y - e, y + e, color="black", alpha=0.12)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle=":")
        ax.set(
            title=f"Cumulative D-coordinate {mode}",
            xlabel="Coordinate count",
            ylabel="Normalized margin effect",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    for (family, mode), group in cycle_positions[
        ~cycle_positions.family.eq("baseline")
    ].groupby(["family", "mode"]):
        ordered = group.sort_values("coordinate_count")
        ax.plot(
            ordered.coordinate_count,
            ordered.target_margin_normalized_effect,
            marker="o",
            label=f"{mode}:{family}",
        )
    ax.set(
        title="Where D coordinates act",
        xlabel="Coordinate count",
        ylabel="Normalized margin effect",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "diagonal_coordinate_circuit.png", dpi=200)
    fig.savefig(args.out_dir / "diagonal_coordinate_circuit.pdf")
    plt.close(fig)

    summary: dict[str, object] = {
        "status": "complete",
        "run_dir": str(args.run_dir),
        "cycle": args.cycle,
        "single_coordinate_discovery_examples": int(
            pd.read_csv(args.run_dir / "single_coordinate_per_example.csv")["sample"].nunique()
        ),
        "heldout_validation_examples": int(
            pd.read_csv(args.run_dir / "group_curves_per_example.csv")["sample"].nunique()
        ),
        "best_static_spearman_feature": (
            correlation.iloc[0].to_dict() if len(correlation) else None
        ),
        "threshold_rows": thresholds.replace({np.nan: None}).to_dict("records"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cycle", type=int, default=64)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
