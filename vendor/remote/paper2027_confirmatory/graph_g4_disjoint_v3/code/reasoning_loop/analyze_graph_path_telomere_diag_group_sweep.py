from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bootstrap_mean_ci(
    values: pd.DataFrame,
    *,
    columns: Sequence[str],
    seed: int,
    draws: int = 5000,
) -> dict[str, tuple[float, float]]:
    """Stratified example bootstrap, preserving the three evaluation seeds."""
    rng = np.random.default_rng(seed)
    groups = [group for _, group in values.groupby("data_seed")]
    sampled = {column: np.empty(draws) for column in columns}
    for draw in range(draws):
        pieces = [
            group.iloc[rng.integers(0, len(group), size=len(group))]
            for group in groups
        ]
        merged = pd.concat(pieces, ignore_index=True)
        for column in columns:
            sampled[column][draw] = float(merged[column].mean())
    return {
        column: (
            float(np.quantile(sampled[column], 0.025)),
            float(np.quantile(sampled[column], 0.975)),
        )
        for column in columns
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    aggregate = pd.read_csv(args.run_dir / "coarse_D_group_sweep.csv")
    per_example = pd.read_csv(
        args.run_dir / "coarse_D_group_sweep_per_example.csv", low_memory=False
    )

    effect_rows: list[pd.DataFrame] = []
    for (data_seed, cycle), group in aggregate.groupby(["data_seed", "cycle"]):
        full = group[group.condition.eq("full")]
        if len(full) != 1:
            raise ValueError(f"expected one full row for seed={data_seed}, cycle={cycle}")
        selected = group[~group.condition.eq("full")].copy()
        selected["accuracy_drop"] = float(full.accuracy.iloc[0]) - selected.accuracy
        selected["margin_drop"] = (
            float(full.target_margin.iloc[0]) - selected.target_margin
        )
        selected["full_accuracy"] = float(full.accuracy.iloc[0])
        selected["full_margin"] = float(full.target_margin.iloc[0])
        effect_rows.append(selected)
    effects = pd.concat(effect_rows, ignore_index=True)
    summary = (
        effects.groupby(["group", "signed_delta", "cycle"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_drop_mean=("accuracy_drop", "mean"),
            margin_mean=("target_margin", "mean"),
            margin_std=("target_margin", "std"),
            margin_drop_mean=("margin_drop", "mean"),
        )
        .sort_values(["group", "signed_delta", "cycle"])
    )
    baselines = (
        aggregate[aggregate.condition.eq("full")]
        .groupby("cycle", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            margin_mean=("target_margin", "mean"),
            margin_std=("target_margin", "std"),
        )
    )

    key_cycle = args.key_cycle
    key_delta = args.key_delta
    keys = ["data_seed", "cycle", "sample"]
    full_examples = per_example[
        per_example.condition.eq("full") & per_example.cycle.eq(key_cycle)
    ][keys + ["correct", "target_margin"]].rename(
        columns={"correct": "full_correct", "target_margin": "full_margin"}
    )
    key_examples = per_example[
        per_example.cycle.eq(key_cycle)
        & per_example.signed_delta.eq(key_delta)
        & ~per_example.condition.eq("full")
    ]
    high = key_examples[key_examples.group.eq("high_retention")][
        keys + ["correct", "target_margin"]
    ].rename(columns={"correct": "high_correct", "target_margin": "high_margin"})
    paired_base = full_examples.merge(high, on=keys, validate="one_to_one")
    contrast_rows: list[dict[str, object]] = []
    for comparison_group in ("middle", "random", "strongly_damped"):
        comparison = key_examples[key_examples.group.eq(comparison_group)][
            keys + ["correct", "target_margin"]
        ].rename(
            columns={
                "correct": "comparison_correct",
                "target_margin": "comparison_margin",
            }
        )
        paired = paired_base.merge(comparison, on=keys, validate="one_to_one")
        paired["accuracy_difference"] = (
            paired.high_correct - paired.comparison_correct
        )
        paired["margin_difference"] = paired.high_margin - paired.comparison_margin
        ci = _bootstrap_mean_ci(
            paired,
            columns=("accuracy_difference", "margin_difference"),
            seed=args.bootstrap_seed + len(contrast_rows),
        )
        contrast_rows.append(
            {
                "cycle": key_cycle,
                "signed_delta": key_delta,
                "comparison": f"high_retention_vs_{comparison_group}",
                "examples": len(paired),
                "high_retention_accuracy": float(paired.high_correct.mean()),
                "comparison_accuracy": float(paired.comparison_correct.mean()),
                "paired_accuracy_difference": float(
                    paired.accuracy_difference.mean()
                ),
                "accuracy_difference_ci_low": ci["accuracy_difference"][0],
                "accuracy_difference_ci_high": ci["accuracy_difference"][1],
                "paired_margin_difference": float(paired.margin_difference.mean()),
                "margin_difference_ci_low": ci["margin_difference"][0],
                "margin_difference_ci_high": ci["margin_difference"][1],
            }
        )
    contrasts = pd.DataFrame(contrast_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.out_dir / "coarse_group_effects.csv", index=False)
    summary.to_csv(args.out_dir / "coarse_group_summary.csv", index=False)
    baselines.to_csv(args.out_dir / "full_baseline_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "paired_group_contrasts.csv", index=False)

    colors = {
        "high_retention": "#D55E00",
        "middle": "#0072B2",
        "strongly_damped": "#009E73",
        "random": "#999999",
    }
    labels = {
        "high_retention": "high-retention D group",
        "middle": "middle-D control",
        "strongly_damped": "strongly-damped D group",
        "random": "random matched group",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.plot(
        baselines.cycle,
        baselines.accuracy_mean,
        color="black",
        marker="o",
        linewidth=2,
        label="full J",
    )
    selected = summary[summary.signed_delta.eq(key_delta)]
    for group_name, group in selected.groupby("group"):
        group = group.sort_values("cycle")
        ax.plot(
            group.cycle,
            group.accuracy_mean,
            marker="o",
            color=colors[group_name],
            label=labels[group_name],
        )
    ax.set(
        title=f"Same +{key_delta:.3f} gain change on 48 D directions",
        xlabel="controlled continuation loop",
        ylabel="successor accuracy",
        ylim=(-0.02, 1.03),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    late = summary[summary.cycle.eq(key_cycle)]
    for group_name, group in late.groupby("group"):
        group = group.sort_values("signed_delta")
        ax.plot(
            group.signed_delta,
            group.margin_drop_mean,
            marker="o",
            color=colors[group_name],
            label=labels[group_name],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"Dose response at continuation loop {key_cycle}",
        xlabel="signed change applied to the whole D group",
        ylabel="successor-margin loss vs full J",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "coarse_D_functional_groups.png", dpi=220)
    fig.savefig(args.out_dir / "coarse_D_functional_groups.pdf")
    plt.close(fig)

    key = summary[
        summary.cycle.eq(key_cycle) & summary.signed_delta.eq(key_delta)
    ][
        [
            "group",
            "accuracy_mean",
            "accuracy_std",
            "margin_mean",
            "margin_drop_mean",
        ]
    ].sort_values("accuracy_mean")
    result: dict[str, object] = {
        "status": "complete",
        "key_cycle": key_cycle,
        "key_signed_delta": key_delta,
        "data_seeds": sorted(int(value) for value in effects.data_seed.unique()),
        "examples_per_seed": int(
            per_example.groupby("data_seed")["sample"].nunique().min()
        ),
        "key_group_results": key.to_dict("records"),
        "paired_contrasts": contrasts.to_dict("records"),
        "interpretation_boundary": (
            "The D-ranked groups are causal operating-range probes, not semantic "
            "labels for individual residual dimensions."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--key-cycle", type=int, default=64)
    parser.add_argument("--key-delta", type=float, default=0.02)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
