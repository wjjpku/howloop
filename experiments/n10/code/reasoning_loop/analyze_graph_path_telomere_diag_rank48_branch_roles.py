from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROLE_SPECS = (
    ("low_rank_entry", "no_AB", 1),
    ("bias_long_horizon", "no_bias", 64),
)


def _fraction(
    full: pd.Series, damaged: pd.Series, patched: pd.Series, *, direction: str
) -> pd.Series:
    denominator = full - damaged
    if direction in {"full_into_damaged", "batch_shuffled_full_into_damaged"}:
        numerator = patched - damaged
    elif direction == "damaged_into_full":
        numerator = full - patched
    else:
        raise ValueError(f"unknown direction: {direction}")
    return numerator / denominator.replace(0, np.nan)


def _normalize_component(component: str, lookup_head: int) -> str:
    return component.replace(f"H{lookup_head}", "H_LOOKUP")


def _load_run(
    *, run_dir: Path, model: str, lookup_head: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(run_dir / "baseline.csv", low_memory=False)
    patch = pd.read_csv(run_dir / "component_mediation.csv", low_memory=False)
    baseline["model"] = model
    patch["model"] = model
    patch["normalized_component"] = patch.component.map(
        lambda value: _normalize_component(str(value), lookup_head)
    )
    return baseline, patch


def _annotate_role(
    *, baseline: pd.DataFrame, patch: pd.DataFrame, role: str, variant: str, cycle: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    at_cycle = baseline[baseline.cycle.eq(cycle)]
    full = at_cycle[at_cycle.variant.eq("full")].iloc[0]
    damaged = at_cycle[at_cycle.variant.eq(variant)].iloc[0]
    selected = patch[
        patch.cycle.eq(cycle) & patch.damaged_variant.eq(variant)
    ].copy()
    selected["role"] = role
    selected["full_accuracy"] = float(full.accuracy)
    selected["damaged_accuracy"] = float(damaged.accuracy)
    selected["full_margin"] = float(full.target_margin)
    selected["damaged_margin"] = float(damaged.target_margin)
    selected["margin_fraction"] = np.nan
    selected["accuracy_fraction"] = np.nan
    for direction in (
        "full_into_damaged",
        "batch_shuffled_full_into_damaged",
        "damaged_into_full",
    ):
        mask = selected.condition.eq(direction)
        selected.loc[mask, "margin_fraction"] = _fraction(
            selected.loc[mask, "full_margin"],
            selected.loc[mask, "damaged_margin"],
            selected.loc[mask, "target_margin"],
            direction=direction,
        )
        selected.loc[mask, "accuracy_fraction"] = _fraction(
            selected.loc[mask, "full_accuracy"],
            selected.loc[mask, "damaged_accuracy"],
            selected.loc[mask, "accuracy"],
            direction=direction,
        )
    summary = {
        "model": str(full.model),
        "role": role,
        "damaged_variant": variant,
        "cycle": cycle,
        "full_accuracy": float(full.accuracy),
        "damaged_accuracy": float(damaged.accuracy),
        "full_margin": float(full.target_margin),
        "damaged_margin": float(damaged.target_margin),
    }
    return selected, summary


def _path_components() -> list[str]:
    return [
        "B1.block_input_answer",
        "B1.block_input_graph",
        "B1.block_input_answer_plus_graph",
        "B1.residual_mid_answer",
        "B2.H_LOOKUP.q_answer",
        "B2.H_LOOKUP.k_graph",
        "B2.H_LOOKUP.q_plus_k",
        "B2.H_LOOKUP.pattern_answer_graph",
        "B2.H_LOOKUP.context_answer",
        "B2.attention_out_answer",
        "B2.residual_mid_answer",
        "B2.mlp_hidden_answer",
        "B2.mlp_out_answer",
        "B1.residual_mid_plus_B2.H_LOOKUP.context",
    ]


def _path_labels() -> dict[str, str]:
    return {
        "B1.block_input_answer": "loop input: answer",
        "B1.block_input_graph": "loop input: graph",
        "B1.block_input_answer_plus_graph": "loop input: answer + graph",
        "B1.residual_mid_answer": "Block1 post-attention answer",
        "B2.H_LOOKUP.q_answer": "lookup query from answer",
        "B2.H_LOOKUP.k_graph": "lookup keys on graph",
        "B2.H_LOOKUP.q_plus_k": "lookup Q + K",
        "B2.H_LOOKUP.pattern_answer_graph": "lookup routing pattern",
        "B2.H_LOOKUP.context_answer": "retrieved lookup context",
        "B2.attention_out_answer": "Block2 attention output",
        "B2.residual_mid_answer": "Block2 post-attention residual",
        "B2.mlp_hidden_answer": "Block2 MLP hidden",
        "B2.mlp_out_answer": "Block2 MLP output",
        "B1.residual_mid_plus_B2.H_LOOKUP.context": "B1 state + lookup context",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    run_specs = (
        (args.seed0_run, "D8L8_seed0", 0),
        (args.seed3_run, "D8L8_seed3", 3),
    )
    baselines: list[pd.DataFrame] = []
    effects: list[pd.DataFrame] = []
    baseline_summaries: list[dict[str, object]] = []
    for run_dir, model, head in run_specs:
        baseline, patch = _load_run(
            run_dir=run_dir, model=model, lookup_head=head
        )
        baselines.append(baseline)
        for role, variant, cycle in ROLE_SPECS:
            selected, summary = _annotate_role(
                baseline=baseline,
                patch=patch,
                role=role,
                variant=variant,
                cycle=cycle,
            )
            effects.append(selected)
            baseline_summaries.append(summary)

    baseline_all = pd.concat(baselines, ignore_index=True)
    effect_all = pd.concat(effects, ignore_index=True)
    path = _path_components()
    anchor = effect_all[effect_all.normalized_component.isin(path)].copy()

    repair = anchor[anchor.condition.eq("full_into_damaged")][
        [
            "model",
            "role",
            "normalized_component",
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
    necessity = anchor[anchor.condition.eq("damaged_into_full")][
        [
            "model",
            "role",
            "normalized_component",
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
    bidirectional = repair.merge(
        necessity,
        on=["model", "role", "normalized_component"],
        validate="one_to_one",
    )
    bidirectional["bidirectional_score"] = bidirectional[
        ["repair_margin_fraction", "necessity_margin_fraction"]
    ].min(axis=1)
    negative = anchor[
        anchor.condition.eq("batch_shuffled_full_into_damaged")
    ].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_all.to_csv(args.out_dir / "baseline_all.csv", index=False)
    effect_all.to_csv(args.out_dir / "all_branch_component_effects.csv", index=False)
    bidirectional.to_csv(
        args.out_dir / "bidirectional_branch_path.csv", index=False
    )
    negative.to_csv(args.out_dir / "batch_shuffled_controls.csv", index=False)

    labels = _path_labels()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.0), sharex=True)
    for row_index, (role, _variant, cycle) in enumerate(ROLE_SPECS):
        for column_index, (model, head) in enumerate(
            (("D8L8_seed0", 0), ("D8L8_seed3", 3))
        ):
            ax = axes[row_index, column_index]
            table = bidirectional[
                bidirectional.model.eq(model) & bidirectional.role.eq(role)
            ].set_index("normalized_component")
            table = table.reindex(path)
            positions = np.arange(len(path))
            width = 0.36
            ax.barh(
                positions - width / 2,
                table.repair_margin_fraction,
                height=width,
                color="#0072B2",
                label="normal into damaged (repair)",
            )
            ax.barh(
                positions + width / 2,
                table.necessity_margin_fraction,
                height=width,
                color="#CC79A7",
                label="damaged into normal (necessity)",
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(1, color="black", linewidth=0.8, linestyle="--")
            ax.set_yticks(positions, [labels[value] for value in path])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
            title_role = (
                "remove low-rank AB: immediate entry failure"
                if role == "low_rank_entry"
                else "remove bias: accumulated long-horizon failure"
            )
            ax.set_title(f"{model}, lookup H{head}\n{title_role}, cycle {cycle}")
            ax.set_xlabel("fraction of full-vs-damaged margin gap")
            if row_index == 0 and column_index == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "AB_bias_component_paths.png", dpi=220)
    fig.savefig(args.out_dir / "AB_bias_component_paths.pdf")
    plt.close(fig)

    cross_model = (
        bidirectional.groupby(["role", "normalized_component"], as_index=False)
        .agg(
            repair_fraction_mean=("repair_margin_fraction", "mean"),
            repair_fraction_min=("repair_margin_fraction", "min"),
            necessity_fraction_mean=("necessity_margin_fraction", "mean"),
            necessity_fraction_min=("necessity_margin_fraction", "min"),
            bidirectional_score_min=("bidirectional_score", "min"),
        )
        .sort_values(["role", "bidirectional_score_min"], ascending=[True, False])
    )
    cross_model.to_csv(args.out_dir / "cross_model_branch_stability.csv", index=False)
    result: dict[str, object] = {
        "status": "complete",
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": 8,
        "shared_physical_blocks": 2,
        "effective_training_depth": 16,
        "controller": "J(h)=h*D+(hA)B+b",
        "controller_rank": 48,
        "controller_placement": "loop boundary after Block2 FFN",
        "controller_training_loss": (
            "successor CE at every controlled continuation loop; no hidden MSE"
        ),
        "role_tests": baseline_summaries,
        "cross_model_path": cross_model.to_dict("records"),
        "interpretation_boundary": (
            "Cycle 1 no_AB localizes the low-rank branch's immediate orbit-entry "
            "role. Cycle 64 no_bias localizes the bias branch's accumulated "
            "long-horizon role. Bidirectional activation patching establishes "
            "component-level causal mediation, not a unique minimal circuit."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-run", type=Path, required=True)
    parser.add_argument("--seed3-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
