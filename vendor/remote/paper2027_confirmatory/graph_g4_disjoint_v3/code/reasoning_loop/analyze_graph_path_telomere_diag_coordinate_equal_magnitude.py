from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def run(args: argparse.Namespace) -> dict[str, object]:
    raw = pd.read_csv(args.run_dir / "equal_magnitude_coordinate.csv", low_memory=False)
    discovery_raw = pd.read_csv(
        args.discovery_run / "single_coordinate.csv", low_memory=False
    )
    discovery_full = float(
        discovery_raw[
            discovery_raw.cycle.eq(args.cycle) & discovery_raw.condition.eq("full")
        ].target_margin.iloc[0]
    )
    discovery = discovery_raw[
        discovery_raw.cycle.eq(args.cycle)
        & discovery_raw.family.eq("single_coordinate")
    ].copy()
    discovery["coordinate"] = discovery.coordinates.astype(int)
    discovery["discovery_margin_drop"] = discovery_full - discovery.target_margin
    discovery_vector = discovery.set_index("coordinate").discovery_margin_drop
    static = pd.read_csv(args.discovery_run / "coordinate_static.csv", low_memory=False)

    rows: list[pd.DataFrame] = []
    for (data_seed, cycle), group in raw.groupby(["data_seed", "cycle"]):
        full = group[group.condition.eq("full")]
        if len(full) != 1:
            raise ValueError(f"expected one full baseline for seed={data_seed}, cycle={cycle}")
        selected = group[~group.condition.eq("full")].copy()
        selected["coordinate"] = selected.coordinates.astype(int)
        selected["margin_drop"] = float(full.target_margin.iloc[0]) - selected.target_margin
        selected["accuracy_drop"] = float(full.accuracy.iloc[0]) - selected.accuracy
        selected["full_margin"] = float(full.target_margin.iloc[0])
        selected["full_accuracy"] = float(full.accuracy.iloc[0])
        rows.append(selected)
    effects = pd.concat(rows, ignore_index=True)
    summary = (
        effects.groupby(
            ["cycle", "fixed_delta", "signed_delta", "delta_sign", "coordinate"],
            as_index=False,
        )
        .agg(
            margin_drop_mean=("margin_drop", "mean"),
            margin_drop_std=("margin_drop", "std"),
            accuracy_drop_mean=("accuracy_drop", "mean"),
            accuracy_drop_std=("accuracy_drop", "std"),
        )
    )
    summary["effect_percentile"] = summary.groupby(
        ["cycle", "signed_delta"]
    ).margin_drop_mean.rank(method="average", pct=True)
    summary = summary.merge(
        discovery[["coordinate", "discovery_margin_drop"]],
        on="coordinate",
        how="left",
    ).merge(static, on="coordinate", how="left")

    correlation_fields = [
        "D",
        "abs_D_minus_1",
        "bias_abs",
        "AB_output_l2",
        "B1_mlp_input_l2",
        "B2H0_q_input_l2",
        "B2H0_k_input_l2",
        "B2H0_v_input_l2",
        "B2_mlp_input_l2",
        "readout_input_l2",
        "trajectory_D_signal_answer",
        "trajectory_D_signal_graph",
        "trajectory_D_signal_all",
        "discovery_margin_drop",
    ]
    correlation_rows: list[dict[str, object]] = []
    for signed_delta, group in summary[summary.cycle.eq(args.cycle)].groupby(
        "signed_delta"
    ):
        for field in correlation_fields:
            if field not in group:
                continue
            statistic, p_value = spearmanr(group[field], group.margin_drop_mean)
            correlation_rows.append(
                {
                    "signed_delta": float(signed_delta),
                    "feature": field,
                    "spearman_r": float(statistic),
                    "p_value": float(p_value),
                }
            )
    feature_correlations = pd.DataFrame(correlation_rows)

    rank_rows: list[dict[str, object]] = []
    cycle_effects = effects[effects.cycle.eq(args.cycle)]
    for signed_delta, delta_group in cycle_effects.groupby("signed_delta"):
        vectors = {
            f"seed_{seed}": group.set_index("coordinate").margin_drop
            for seed, group in delta_group.groupby("data_seed")
        }
        vectors["seed_mean"] = delta_group.groupby("coordinate").margin_drop.mean()
        for name, vector in vectors.items():
            statistic, p_value = spearmanr(discovery_vector, vector)
            rank_rows.append(
                {
                    "signed_delta": float(signed_delta),
                    "comparison": f"discovery_vs_{name}",
                    "spearman_r": float(statistic),
                    "p_value": float(p_value),
                }
            )
        seeds = sorted(delta_group.data_seed.unique())
        for left_index, left_seed in enumerate(seeds):
            left = vectors[f"seed_{left_seed}"]
            for right_seed in seeds[left_index + 1 :]:
                statistic, p_value = spearmanr(left, vectors[f"seed_{right_seed}"])
                rank_rows.append(
                    {
                        "signed_delta": float(signed_delta),
                        "comparison": f"seed_{left_seed}_vs_seed_{right_seed}",
                        "spearman_r": float(statistic),
                        "p_value": float(p_value),
                    }
                )
    rank_stability = pd.DataFrame(rank_rows)

    top_rows: list[dict[str, object]] = []
    cycle_summary = summary[summary.cycle.eq(args.cycle)]
    for signed_delta, group in cycle_summary.groupby("signed_delta"):
        ordered = group.nlargest(args.top_k, "margin_drop_mean")
        for rank, row in enumerate(ordered.itertuples(), start=1):
            top_rows.append(
                {
                    "signed_delta": float(signed_delta),
                    "rank": rank,
                    "coordinate": int(row.coordinate),
                    "margin_drop_mean": float(row.margin_drop_mean),
                    "margin_drop_std": float(row.margin_drop_std),
                    "accuracy_drop_mean": float(row.accuracy_drop_mean),
                    "effect_percentile": float(row.effect_percentile),
                    "discovery_margin_drop": float(row.discovery_margin_drop),
                }
            )
    top = pd.DataFrame(top_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.out_dir / "equal_magnitude_effects.csv", index=False)
    summary.to_csv(args.out_dir / "equal_magnitude_summary.csv", index=False)
    rank_stability.to_csv(args.out_dir / "rank_stability.csv", index=False)
    top.to_csv(args.out_dir / "top_coordinates_by_delta.csv", index=False)
    feature_correlations.to_csv(args.out_dir / "feature_correlations.csv", index=False)

    discovery_top = list(discovery_vector.nlargest(args.plot_top_k).index)
    pivot = cycle_summary.pivot(
        index="coordinate", columns="signed_delta", values="margin_drop_mean"
    ).loc[discovery_top]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:+.2f}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [str(int(value)) for value in pivot.index])
    ax.set(
        title=f"Equal-magnitude D-coordinate damage, cycle {args.cycle}",
        xlabel="Signed change to one D coordinate",
        ylabel="Discovery-ranked coordinate",
    )
    fig.colorbar(image, ax=ax, label="Mean successor-margin drop")

    ax = axes[1]
    for coordinate in discovery_top[:8]:
        curve = cycle_summary[cycle_summary.coordinate.eq(coordinate)].sort_values(
            "signed_delta"
        )
        ax.errorbar(
            curve.signed_delta,
            curve.margin_drop_mean,
            yerr=curve.margin_drop_std.fillna(0),
            marker="o",
            label=str(int(coordinate)),
        )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(
        title="Coordinate-specific dose response",
        xlabel="Signed change to D coordinate",
        ylabel="Mean successor-margin drop",
    )
    ax.grid(alpha=0.2)
    ax.legend(title="coordinate", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "equal_magnitude_coordinate_stability.png", dpi=200)
    fig.savefig(args.out_dir / "equal_magnitude_coordinate_stability.pdf")
    plt.close(fig)

    primary_rows = cycle_summary[
        cycle_summary.coordinate.isin(discovery_top[:3])
    ][
        [
            "signed_delta",
            "coordinate",
            "margin_drop_mean",
            "margin_drop_std",
            "accuracy_drop_mean",
            "effect_percentile",
        ]
    ].to_dict("records")
    result: dict[str, object] = {
        "status": "complete",
        "cycle": args.cycle,
        "data_seeds": sorted(int(value) for value in effects.data_seed.unique()),
        "examples_per_seed": int(
            pd.read_csv(
                args.run_dir / "equal_magnitude_coordinate_per_example.csv",
                usecols=["data_seed", "sample"],
            )
            .groupby("data_seed")["sample"]
            .nunique()
            .min()
        ),
        "signed_deltas": sorted(float(value) for value in effects.signed_delta.unique()),
        "discovery_top_coordinates": [int(value) for value in discovery_top],
        "primary_coordinate_results": primary_rows,
        "best_positive_feature_correlations": (
            feature_correlations.sort_values(
                ["signed_delta", "spearman_r"], ascending=[True, False]
            )
            .groupby("signed_delta", as_index=False)
            .head(1)
            .to_dict("records")
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--discovery-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cycle", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--plot-top-k", type=int, default=24)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
