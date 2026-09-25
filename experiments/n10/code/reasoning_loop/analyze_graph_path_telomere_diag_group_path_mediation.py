from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fraction(
    full: pd.Series, damaged: pd.Series, patched: pd.Series, *, direction: str
) -> pd.Series:
    denominator = full - damaged
    if direction == "full_into_damaged":
        numerator = patched - damaged
    elif direction == "damaged_into_full":
        numerator = full - patched
    else:
        raise ValueError(f"unknown direction: {direction}")
    return numerator / denominator.replace(0, np.nan)


def run(args: argparse.Namespace) -> dict[str, object]:
    baseline = pd.read_csv(args.run_dir / "baseline.csv", low_memory=False)
    patch = pd.read_csv(args.run_dir / "path_mediation.csv", low_memory=False)
    full = baseline[baseline.variant.eq("full")][
        ["cycle", "accuracy", "target_margin"]
    ].rename(columns={"accuracy": "full_accuracy", "target_margin": "full_margin"})
    damaged = baseline[~baseline.variant.eq("full")][
        ["cycle", "variant", "accuracy", "target_margin"]
    ].rename(
        columns={
            "variant": "damaged_variant",
            "accuracy": "damaged_accuracy",
            "target_margin": "damaged_margin",
        }
    )
    merged = patch.merge(full, on="cycle", validate="many_to_one").merge(
        damaged, on=["cycle", "damaged_variant"], validate="many_to_one"
    )
    merged["margin_fraction"] = np.nan
    merged["accuracy_fraction"] = np.nan
    for direction in ("full_into_damaged", "damaged_into_full"):
        mask = merged.direction.eq(direction)
        merged.loc[mask, "margin_fraction"] = _fraction(
            merged.loc[mask, "full_margin"],
            merged.loc[mask, "damaged_margin"],
            merged.loc[mask, "target_margin"],
            direction=direction,
        )
        merged.loc[mask, "accuracy_fraction"] = _fraction(
            merged.loc[mask, "full_accuracy"],
            merged.loc[mask, "damaged_accuracy"],
            merged.loc[mask, "accuracy"],
            direction=direction,
        )

    anchor_components = [
        "B1.input_answer",
        "B1.input_graph",
        "B1.input_answer_plus_graph",
        f"B2.H{args.lookup_head}.q_plus_k",
        f"B2.H{args.lookup_head}.pattern_answer_graph",
        f"B2.H{args.lookup_head}.context_answer",
        "B2.attention_out_answer",
        "B2.residual_mid_answer",
        "B2.mlp_hidden_answer",
        "B2.mlp_out_answer",
    ]
    selected = merged[
        merged.component.isin(anchor_components) & ~merged.negative_control
    ].copy()
    negative = merged[
        merged.negative_control & merged.direction.eq("full_into_damaged")
    ].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "all_group_path_effects.csv", index=False)
    selected.to_csv(args.out_dir / "anchor_component_effects.csv", index=False)
    negative.to_csv(args.out_dir / "batch_shuffled_controls.csv", index=False)

    final = selected[selected.cycle.eq(args.cycle)]
    repair = final[final.direction.eq("full_into_damaged")][
        [
            "component",
            "stage",
            "accuracy",
            "target_margin",
            "margin_fraction",
            "accuracy_fraction",
        ]
    ].rename(
        columns={
            "accuracy": "repair_accuracy",
            "target_margin": "repair_margin",
            "margin_fraction": "repair_margin_fraction",
            "accuracy_fraction": "repair_accuracy_fraction",
        }
    )
    necessity = final[final.direction.eq("damaged_into_full")][
        [
            "component",
            "accuracy",
            "target_margin",
            "margin_fraction",
            "accuracy_fraction",
        ]
    ].rename(
        columns={
            "accuracy": "damaged_into_full_accuracy",
            "target_margin": "damaged_into_full_margin",
            "margin_fraction": "necessity_margin_fraction",
            "accuracy_fraction": "necessity_accuracy_fraction",
        }
    )
    component_table = repair.merge(necessity, on="component", validate="one_to_one")
    component_table["bidirectional_score"] = component_table[
        ["repair_margin_fraction", "necessity_margin_fraction"]
    ].min(axis=1)
    component_table = component_table.sort_values(
        "bidirectional_score", ascending=False
    )
    component_table.to_csv(args.out_dir / "bidirectional_component_table.csv", index=False)

    compact = {
        "B1.input_answer": "answer interface",
        "B1.input_graph": "graph interface",
        "B1.input_answer_plus_graph": "answer + graph interface",
        f"B2.H{args.lookup_head}.q_plus_k": "lookup Q+K",
        f"B2.H{args.lookup_head}.pattern_answer_graph": "lookup pattern",
        f"B2.H{args.lookup_head}.context_answer": "lookup context",
        "B2.attention_out_answer": "attention output",
        "B2.residual_mid_answer": "post-attention residual",
        "B2.mlp_hidden_answer": "MLP hidden writer",
        "B2.mlp_out_answer": "MLP output writer",
    }
    colors = {
        "full": "black",
        "damaged": "#D55E00",
        "repair": "#0072B2",
        "necessity": "#CC79A7",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    ax = axes[0]
    base_full = baseline[baseline.variant.eq("full")].sort_values("cycle")
    base_damaged = baseline[~baseline.variant.eq("full")].sort_values("cycle")
    ax.plot(
        base_full.cycle,
        base_full.accuracy,
        marker="o",
        color=colors["full"],
        linewidth=2,
        label="full J",
    )
    ax.plot(
        base_damaged.cycle,
        base_damaged.accuracy,
        marker="o",
        color=colors["damaged"],
        linewidth=2,
        label="high-retention gain +0.02",
    )
    for component, label in (
        (f"B2.H{args.lookup_head}.q_plus_k", "repair lookup Q+K"),
        (f"B2.H{args.lookup_head}.context_answer", "repair lookup context"),
        ("B2.mlp_hidden_answer", "repair MLP writer"),
    ):
        curve = selected[
            selected.component.eq(component)
            & selected.direction.eq("full_into_damaged")
        ].sort_values("cycle")
        ax.plot(curve.cycle, curve.accuracy, marker="o", label=label)
    ax.set(
        title="Where normal information repairs the D-induced failure",
        xlabel="controlled continuation loop",
        ylabel="successor accuracy",
        ylim=(-0.02, 1.03),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    display = component_table[
        component_table.component.isin(anchor_components)
    ].copy()
    display["label"] = display.component.map(compact)
    display = display.set_index("label").loc[
        [compact[value] for value in anchor_components]
    ].reset_index()
    positions = np.arange(len(display))
    width = 0.36
    ax.barh(
        positions - width / 2,
        display.repair_margin_fraction,
        height=width,
        color=colors["repair"],
        label="normal into damaged (repair)",
    )
    ax.barh(
        positions + width / 2,
        display.necessity_margin_fraction,
        height=width,
        color=colors["necessity"],
        label="damaged into normal (necessity)",
    )
    ax.set_yticks(positions, display.label)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"Bidirectional causal path at continuation loop {args.cycle}",
        xlabel="fraction of full-vs-damaged margin gap",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "coarse_group_circuit_path.png", dpi=220)
    fig.savefig(args.out_dir / "coarse_group_circuit_path.pdf")
    plt.close(fig)

    baseline_final = baseline[baseline.cycle.eq(args.cycle)].copy()
    result: dict[str, object] = {
        "status": "complete",
        "cycle": args.cycle,
        "lookup_head": args.lookup_head,
        "full_accuracy": float(
            baseline_final[baseline_final.variant.eq("full")].accuracy.iloc[0]
        ),
        "damaged_accuracy": float(
            baseline_final[~baseline_final.variant.eq("full")].accuracy.iloc[0]
        ),
        "bidirectional_components": component_table.to_dict("records"),
        "batch_shuffled_controls": negative[
            negative.cycle.eq(args.cycle)
        ][["component", "accuracy", "target_margin", "margin_fraction"]].to_dict(
            "records"
        ),
        "interpretation_boundary": (
            "This establishes a causal route from a D-group operating-point "
            "perturbation to the model-specific lookup and writer components. It "
            "does not assign semantics to individual residual coordinates."
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
    parser.add_argument("--lookup-head", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
