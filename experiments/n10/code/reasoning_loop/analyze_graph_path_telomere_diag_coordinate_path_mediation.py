from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _aggregate_fraction(row: pd.Series, *, metric: str) -> float:
    full = float(row[f"full_{metric}"])
    damaged = float(row[f"damaged_{metric}"])
    patched = float(row[metric])
    denominator = full - damaged
    if abs(denominator) < 1e-9:
        return float("nan")
    if row.direction == "full_into_damaged":
        return (patched - damaged) / denominator
    if row.direction == "damaged_into_full":
        return (full - patched) / denominator
    raise ValueError(f"unknown patch direction: {row.direction}")


def run(args: argparse.Namespace) -> dict[str, object]:
    baseline = pd.read_csv(args.run_dir / "baseline.csv", low_memory=False)
    patch = pd.read_csv(args.run_dir / "path_mediation.csv", low_memory=False)
    full = baseline[baseline.variant.eq("full")][
        [
            "cycle",
            "accuracy",
            "target_margin",
            "B2_lookup_current_destination_attention",
        ]
    ].rename(
        columns={
            "accuracy": "full_accuracy",
            "target_margin": "full_target_margin",
            "B2_lookup_current_destination_attention": "full_lookup_attention",
        }
    )
    damaged = baseline[~baseline.variant.eq("full")][
        [
            "cycle",
            "variant",
            "coordinates",
            "accuracy",
            "target_margin",
            "B2_lookup_current_destination_attention",
        ]
    ].rename(
        columns={
            "variant": "damaged_variant",
            "coordinates": "baseline_damaged_coordinates",
            "accuracy": "damaged_accuracy",
            "target_margin": "damaged_target_margin",
            "B2_lookup_current_destination_attention": "damaged_lookup_attention",
        }
    )
    merged = patch.merge(full, on="cycle", how="left").merge(
        damaged, on=["cycle", "damaged_variant"], how="left"
    )
    merged["aggregate_margin_fraction"] = merged.apply(
        _aggregate_fraction, axis=1, metric="target_margin"
    )
    merged["aggregate_accuracy_fraction"] = merged.apply(
        _aggregate_fraction, axis=1, metric="accuracy"
    )
    merged["lookup_attention_recovery"] = (
        merged.B2_lookup_current_destination_attention - merged.damaged_lookup_attention
    ) / (merged.full_lookup_attention - merged.damaged_lookup_attention).replace(0, np.nan)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "path_mediation_with_baselines.csv", index=False)
    ranked = merged[
        merged.cycle.eq(args.cycle)
        & merged.direction.eq("full_into_damaged")
        & ~merged.negative_control
    ].sort_values(
        ["damaged_variant", "aggregate_margin_fraction"], ascending=[True, False]
    )
    ranked.to_csv(args.out_dir / "repair_ranked.csv", index=False)
    necessity = merged[
        merged.cycle.eq(args.cycle)
        & merged.direction.eq("damaged_into_full")
        & ~merged.negative_control
    ].sort_values(
        ["damaged_variant", "aggregate_margin_fraction"], ascending=[True, False]
    )
    necessity.to_csv(args.out_dir / "necessity_ranked.csv", index=False)

    variants = sorted(ranked.damaged_variant.unique())
    components = list(
        ranked.groupby("component").aggregate_margin_fraction.mean().sort_values(
            ascending=False
        ).index
    )
    repair_matrix = ranked.pivot(
        index="damaged_variant",
        columns="component",
        values="aggregate_margin_fraction",
    ).reindex(index=variants, columns=components)
    damage_matrix = necessity.pivot(
        index="damaged_variant",
        columns="component",
        values="aggregate_margin_fraction",
    ).reindex(index=variants, columns=components)
    repair_matrix.to_csv(args.out_dir / "repair_component_matrix.csv")
    damage_matrix.to_csv(args.out_dir / "damage_component_matrix.csv")

    width = max(13, 0.42 * len(components))
    fig, axes = plt.subplots(2, 1, figsize=(width, 3.4 + 1.0 * len(variants)))
    for ax, matrix, title in (
        (axes[0], repair_matrix, "Full component into D-damaged run: sufficiency"),
        (axes[1], damage_matrix, "D-damaged component into full run: necessity"),
    ):
        image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90)
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
        ax.set(title=title, xlabel="Effective component at current loop", ylabel="D damage")
        fig.colorbar(image, ax=ax, label="Aggregate margin fraction", fraction=0.02)
    fig.tight_layout()
    fig.savefig(args.out_dir / "coordinate_path_mediation.png", dpi=200)
    fig.savefig(args.out_dir / "coordinate_path_mediation.pdf")
    plt.close(fig)

    top_repairs = (
        ranked.groupby("damaged_variant", group_keys=False)
        .head(args.top_k)[
            [
                "damaged_variant",
                "damaged_coordinates",
                "component",
                "stage",
                "accuracy",
                "target_margin",
                "aggregate_margin_fraction",
                "aggregate_accuracy_fraction",
                "lookup_attention_recovery",
            ]
        ]
        .to_dict("records")
    )
    negative = merged[
        merged.cycle.eq(args.cycle)
        & merged.direction.eq("full_into_damaged")
        & merged.negative_control
    ][
        [
            "damaged_variant",
            "component",
            "accuracy",
            "target_margin",
            "aggregate_margin_fraction",
        ]
    ].to_dict("records")
    result: dict[str, object] = {
        "status": "complete",
        "cycle": args.cycle,
        "variants": variants,
        "top_repairs": top_repairs,
        "batch_shuffled_negative_controls": negative,
        "interpretation_boundary": (
            "A component is assigned a coordinate-mediated causal role only when "
            "full-into-damaged repair and damaged-into-full necessity agree and "
            "the batch-shuffled donor fails."
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
    parser.add_argument("--cycle", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
