from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _coordinate_effects(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seed_column = "data_seed" if "data_seed" in frame else None
    group_columns = ["cycle"] if seed_column is None else [seed_column, "cycle"]
    for keys, group in frame.groupby(group_columns):
        if seed_column is None:
            data_seed = source
            cycle = int(keys[0] if isinstance(keys, tuple) else keys)
        else:
            data_seed, cycle = keys
        full = group[group.condition.eq("full")]
        if len(full) != 1:
            raise ValueError(f"expected one full row for seed={data_seed}, cycle={cycle}")
        singles = group[group.family.eq("single_coordinate")].copy()
        if len(singles) != 256:
            raise ValueError(
                f"expected 256 coordinate rows for seed={data_seed}, cycle={cycle}; "
                f"found {len(singles)}"
            )
        singles["coordinate"] = singles.coordinates.astype(int)
        singles["data_seed"] = data_seed
        singles["source"] = source
        singles["margin_drop"] = float(full.target_margin.iloc[0]) - singles.target_margin
        singles["accuracy_drop"] = float(full.accuracy.iloc[0]) - singles.accuracy
        singles["full_margin"] = float(full.target_margin.iloc[0])
        singles["full_accuracy"] = float(full.accuracy.iloc[0])
        rows.append(
            singles[
                [
                    "source",
                    "data_seed",
                    "cycle",
                    "coordinate",
                    "target_margin",
                    "accuracy",
                    "margin_drop",
                    "accuracy_drop",
                    "full_margin",
                    "full_accuracy",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _rank_stability(
    discovery: pd.DataFrame, validation: pd.DataFrame, *, cycle: int
) -> pd.DataFrame:
    vectors: dict[str, pd.Series] = {
        "discovery": discovery[discovery.cycle.eq(cycle)].set_index("coordinate").margin_drop
    }
    for seed, group in validation[validation.cycle.eq(cycle)].groupby("data_seed"):
        vectors[f"validation_{seed}"] = group.set_index("coordinate").margin_drop
    rows: list[dict[str, object]] = []
    for (left_name, left), (right_name, right) in combinations(vectors.items(), 2):
        joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
        statistic, p_value = spearmanr(joined.left, joined.right)
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "spearman_r": float(statistic),
                "p_value": float(p_value),
                "coordinate_count": int(len(joined)),
            }
        )
    return pd.DataFrame(rows)


def _topk_overlap(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    cycle: int,
    top_ks: Sequence[int],
) -> pd.DataFrame:
    discovery_vector = discovery[discovery.cycle.eq(cycle)].set_index("coordinate").margin_drop
    validation_vectors: dict[str, pd.Series] = {
        f"validation_{seed}": group.set_index("coordinate").margin_drop
        for seed, group in validation[validation.cycle.eq(cycle)].groupby("data_seed")
    }
    validation_vectors["validation_mean"] = (
        validation[validation.cycle.eq(cycle)]
        .groupby("coordinate")
        .margin_drop.mean()
    )
    rows: list[dict[str, object]] = []
    for name, vector in validation_vectors.items():
        for k in top_ks:
            discovery_top = set(discovery_vector.nlargest(k).index)
            validation_top = set(vector.nlargest(k).index)
            overlap = len(discovery_top & validation_top)
            rows.append(
                {
                    "comparison": name,
                    "top_k": int(k),
                    "overlap": overlap,
                    "overlap_fraction": overlap / k,
                    "jaccard": overlap / len(discovery_top | validation_top),
                    "random_expected_overlap": k * k / len(vector),
                }
            )
    return pd.DataFrame(rows)


def _magnitude_matched_targets(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    static: pd.DataFrame,
    *,
    cycle: int,
    target_count: int,
    neighbor_count: int,
) -> pd.DataFrame:
    discovery_vector = discovery[discovery.cycle.eq(cycle)].set_index("coordinate").margin_drop
    validation_mean = (
        validation[validation.cycle.eq(cycle)]
        .groupby("coordinate")
        .agg(
            validation_margin_drop_mean=("margin_drop", "mean"),
            validation_margin_drop_std=("margin_drop", "std"),
            validation_accuracy_drop_mean=("accuracy_drop", "mean"),
            validation_accuracy_drop_std=("accuracy_drop", "std"),
        )
    )
    magnitude = static.set_index("coordinate").abs_D_minus_shuffled.astype(float)
    rows: list[dict[str, object]] = []
    for coordinate in discovery_vector.nlargest(target_count).index:
        distance = (magnitude - magnitude.loc[coordinate]).abs().drop(index=coordinate)
        controls = distance.nsmallest(neighbor_count).index
        target_effect = float(validation_mean.loc[coordinate, "validation_margin_drop_mean"])
        control_effects = validation_mean.loc[controls, "validation_margin_drop_mean"]
        rows.append(
            {
                "coordinate": int(coordinate),
                "discovery_rank": int(
                    discovery_vector.rank(method="min", ascending=False).loc[coordinate]
                ),
                "abs_D_minus_shuffled": float(magnitude.loc[coordinate]),
                "validation_margin_drop_mean": target_effect,
                "validation_margin_drop_std": float(
                    validation_mean.loc[coordinate, "validation_margin_drop_std"]
                ),
                "validation_accuracy_drop_mean": float(
                    validation_mean.loc[coordinate, "validation_accuracy_drop_mean"]
                ),
                "matched_control_count": int(len(controls)),
                "matched_magnitude_min": float(magnitude.loc[controls].min()),
                "matched_magnitude_max": float(magnitude.loc[controls].max()),
                "matched_margin_drop_mean": float(control_effects.mean()),
                "matched_margin_drop_std": float(control_effects.std(ddof=1)),
                "matched_effect_percentile": float(
                    ((control_effects < target_effect).sum() + 0.5 * (control_effects == target_effect).sum())
                    / len(control_effects)
                ),
                "matched_coordinates": " ".join(str(int(value)) for value in controls),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    discovery_raw = pd.read_csv(args.discovery_run / "single_coordinate.csv", low_memory=False)
    validation_raw = pd.read_csv(
        args.validation_run / "single_coordinate_validation.csv", low_memory=False
    )
    static = pd.read_csv(args.discovery_run / "coordinate_static.csv", low_memory=False)
    discovery = _coordinate_effects(discovery_raw, source="discovery")
    validation = _coordinate_effects(validation_raw, source="validation")
    validation_summary = (
        validation.groupby(["cycle", "coordinate"], as_index=False)
        .agg(
            margin_drop_mean=("margin_drop", "mean"),
            margin_drop_std=("margin_drop", "std"),
            accuracy_drop_mean=("accuracy_drop", "mean"),
            accuracy_drop_std=("accuracy_drop", "std"),
        )
        .merge(static, on="coordinate", how="left")
    )
    rank_stability = _rank_stability(discovery, validation, cycle=args.cycle)
    topk = _topk_overlap(
        discovery,
        validation,
        cycle=args.cycle,
        top_ks=args.top_ks,
    )
    matched = _magnitude_matched_targets(
        discovery,
        validation,
        static,
        cycle=args.cycle,
        target_count=args.target_count,
        neighbor_count=args.neighbor_count,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([discovery, validation], ignore_index=True).to_csv(
        args.out_dir / "all_coordinate_effects.csv", index=False
    )
    validation_summary.to_csv(args.out_dir / "validation_coordinate_summary.csv", index=False)
    rank_stability.to_csv(args.out_dir / "rank_stability.csv", index=False)
    topk.to_csv(args.out_dir / "topk_overlap.csv", index=False)
    matched.to_csv(args.out_dir / "magnitude_matched_targets.csv", index=False)

    cycle_discovery = discovery[discovery.cycle.eq(args.cycle)].set_index("coordinate")
    cycle_validation = validation_summary[validation_summary.cycle.eq(args.cycle)].set_index(
        "coordinate"
    )
    top_coordinates = cycle_discovery.margin_drop.nlargest(args.plot_top_k).index
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.scatter(
        cycle_discovery.margin_drop,
        cycle_validation.margin_drop_mean,
        s=18,
        alpha=0.65,
        color="#4C78A8",
    )
    for coordinate in top_coordinates[:8]:
        ax.annotate(
            str(int(coordinate)),
            (cycle_discovery.loc[coordinate, "margin_drop"], cycle_validation.loc[coordinate, "margin_drop_mean"]),
            fontsize=8,
        )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(
        title=f"Single-D damage transfers at cycle {args.cycle}",
        xlabel="Discovery margin drop",
        ylabel="Validation mean margin drop",
    )
    ax.grid(alpha=0.2)

    ax = axes[1]
    x = np.arange(len(top_coordinates))
    ax.bar(
        x - 0.2,
        cycle_discovery.loc[top_coordinates, "margin_drop"],
        width=0.4,
        label="discovery",
        color="#4C78A8",
    )
    ax.bar(
        x + 0.2,
        cycle_validation.loc[top_coordinates, "margin_drop_mean"],
        width=0.4,
        yerr=cycle_validation.loc[top_coordinates, "margin_drop_std"].fillna(0),
        label="validation mean +/- SD",
        color="#F58518",
    )
    ax.set_xticks(x, [str(int(value)) for value in top_coordinates], rotation=90)
    ax.set(
        title="Discovery-ranked coordinates",
        xlabel="D coordinate",
        ylabel="Successor-margin drop",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "coordinate_multiseed_stability.png", dpi=200)
    fig.savefig(args.out_dir / "coordinate_multiseed_stability.pdf")
    plt.close(fig)

    discovery_validation = rank_stability[
        rank_stability.left.eq("discovery")
        | rank_stability.right.eq("discovery")
    ]
    top1 = matched.iloc[0].to_dict() if len(matched) else None
    summary: dict[str, object] = {
        "status": "complete",
        "cycle": args.cycle,
        "discovery_run": str(args.discovery_run),
        "validation_run": str(args.validation_run),
        "validation_data_seeds": sorted(int(value) for value in validation.data_seed.unique()),
        "examples_per_validation_seed": int(
            pd.read_csv(
                args.validation_run / "single_coordinate_validation_per_example.csv",
                usecols=["data_seed", "sample"],
            )
            .groupby("data_seed")
            ["sample"].nunique()
            .min()
        ),
        "discovery_validation_spearman_mean": float(discovery_validation.spearman_r.mean()),
        "discovery_validation_spearman_min": float(discovery_validation.spearman_r.min()),
        "top_discovery_coordinate": int(cycle_discovery.margin_drop.idxmax()),
        "top_discovery_coordinate_validation": top1,
        "magnitude_matching": {
            "definition": "nearest coordinates by abs(D_i - shuffled_D_i), excluding target",
            "neighbors_per_target": args.neighbor_count,
            "target_count": args.target_count,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cycle", type=int, default=64)
    parser.add_argument("--top-ks", type=int, nargs="+", default=(1, 4, 8, 16, 32, 48))
    parser.add_argument("--target-count", type=int, default=16)
    parser.add_argument("--neighbor-count", type=int, default=16)
    parser.add_argument("--plot-top-k", type=int, default=24)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
