from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(run_dir: Path, model: str) -> dict[str, pd.DataFrame]:
    result = {}
    for key, filename in (
        ("baseline", "baseline.csv"),
        ("stage", "stage_readout.csv"),
        ("intervention", "bypass_and_semantic_edges.csv"),
        ("interchange", "current_interchange.csv"),
        ("geometry", "component_geometry.csv"),
    ):
        frame = pd.read_csv(run_dir / filename, low_memory=False)
        frame["model"] = model
        result[key] = frame
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    runs = (
        _load(args.seed0_run, "D8L8_seed0"),
        _load(args.seed3_run, "D8L8_seed3"),
    )
    data = {
        key: pd.concat([run[key] for run in runs], ignore_index=True)
        for key in runs[0]
    }
    baseline = data["baseline"]
    full = baseline[baseline.run.eq("base_current")][
        ["model", "cycle", "accuracy", "target_margin"]
    ].rename(columns={"accuracy": "full_accuracy", "target_margin": "full_margin"})
    intervention = data["intervention"].merge(
        full, on=["model", "cycle"], validate="many_to_one"
    )
    intervention["accuracy_drop"] = (
        intervention.full_accuracy - intervention.accuracy
    )
    intervention["margin_drop"] = intervention.full_margin - intervention.target_margin
    intervention["margin_retained"] = (
        intervention.target_margin / intervention.full_margin
    )

    head_answer = intervention[
        intervention.component.str.match(r"B1\.H\d+\.context_zero_answer")
    ].copy()
    head_answer["head"] = head_answer.component.str.extract(r"B1\.H(\d+)").astype(int)
    cycle64 = head_answer[head_answer.cycle.eq(64)]
    leading = (
        cycle64.sort_values("margin_drop", ascending=False)
        .groupby("model", as_index=False)
        .first()[["model", "head", "margin_drop", "accuracy_drop"]]
    )

    bypass_labels = [
        f"B1.{update}_update_zero_{group}"
        for update in ("attention", "mlp", "full")
        for group in ("answer", "graph", "metadata", "all")
    ]
    bypass = intervention[intervention.component.isin(bypass_labels)].copy()
    semantic = intervention[intervention.family.eq("semantic_edge")].copy()
    interchange = data["interchange"].copy()
    stages = data["stage"].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stages.to_csv(args.out_dir / "stage_current_successor.csv", index=False)
    bypass.to_csv(args.out_dir / "block1_bypass_effects.csv", index=False)
    head_answer.to_csv(args.out_dir / "block1_head_answer_effects.csv", index=False)
    semantic.to_csv(args.out_dir / "block1_semantic_edge_effects.csv", index=False)
    interchange.to_csv(args.out_dir / "block1_current_interchange.csv", index=False)
    leading.to_csv(args.out_dir / "leading_block1_heads.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    colors = {"D8L8_seed0": "#0072B2", "D8L8_seed3": "#D55E00"}

    ax = axes[0, 0]
    stage_order = [
        "loop_input",
        "B1_post_attention",
        "B1_post_mlp",
        "B2_post_attention",
        "B2_post_mlp",
    ]
    for model in colors:
        table = stages[(stages.model.eq(model)) & stages.cycle.eq(64)].set_index("stage")
        table = table.reindex(stage_order)
        x = np.arange(len(stage_order))
        ax.plot(
            x,
            table.current_accuracy,
            marker="o",
            color=colors[model],
            label=f"{model}: current",
        )
        ax.plot(
            x,
            table.successor_accuracy,
            marker="s",
            linestyle="--",
            color=colors[model],
            label=f"{model}: successor",
        )
    ax.set_xticks(np.arange(len(stage_order)), stage_order, rotation=25, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("What the answer state decodes before and after Block 1")
    ax.set_ylabel("readout accuracy at cycle 64")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    table = bypass[bypass.cycle.eq(64) & bypass.component.str.contains("full_update")]
    labels = [
        "answer",
        "graph",
        "metadata",
        "all",
    ]
    x = np.arange(len(labels))
    width = 0.36
    for index, model in enumerate(colors):
        values = []
        for group in labels:
            row = table[
                table.model.eq(model)
                & table.component.eq(f"B1.full_update_zero_{group}")
            ]
            values.append(float(row.margin_retained.iloc[0]))
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width=width,
            color=colors[model],
            label=model,
        )
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_title("Exact B1-update bypass at cycle 64")
    ax.set_ylabel("successor margin retained")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for model in colors:
        table = head_answer[head_answer.model.eq(model)]
        for head in sorted(table["head"].unique()):
            part = table[table["head"].eq(head)].sort_values("cycle")
            leading_head = int(
                leading[leading.model.eq(model)]["head"].iloc[0]
            )
            ax.plot(
                part.cycle,
                part.margin_drop,
                marker="o",
                color=colors[model],
                alpha=1.0 if int(head) == leading_head else 0.25,
                linewidth=2.5 if int(head) == leading_head else 1.0,
                label=(
                    f"{model} H{head}"
                    if int(head) == leading_head
                    else None
                ),
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Which B1 head affects the answer interface")
    ax.set_xlabel("controlled continuation cycle")
    ax.set_ylabel("successor margin drop when context is zeroed")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    selected_components = [
        "B1.input_answer",
        "B1.post_attention_answer",
        "B1.attention_update_answer",
        "B1.mlp_update_answer",
        "B2.input_answer",
    ]
    table = interchange[
        interchange.cycle.eq(64)
        & interchange.component.isin(selected_components)
    ].copy()
    positions = np.arange(len(selected_components))
    width = 0.18
    offset = 0
    for model in colors:
        for donor, hatch in (("matched_same_graph", ""), ("batch_shuffled", "//")):
            values = [
                float(
                    table[
                        table.model.eq(model)
                        & table.donor.eq(donor)
                        & table.component.eq(component)
                    ].alternate_target_accuracy.iloc[0]
                )
                for component in selected_components
            ]
            ax.bar(
                positions + (offset - 1.5) * width,
                values,
                width=width,
                color=colors[model],
                hatch=hatch,
                alpha=0.85,
                label=f"{model}, {donor}",
            )
            offset += 1
    ax.set_xticks(
        positions,
        [
            "B1 input\nanswer",
            "B1 post-attn\nanswer",
            "B1 attention\nupdate",
            "B1 MLP\nupdate",
            "B2 input\nanswer",
        ],
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Same-graph alternate-current interchange at cycle 64")
    ax.set_ylabel("prediction switches to alternate successor")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "block1_mechanism_summary.png", dpi=220)
    fig.savefig(args.out_dir / "block1_mechanism_summary.pdf")
    plt.close(fig)

    summary: dict[str, object] = {
        "status": "complete",
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": 8,
        "shared_physical_blocks": 2,
        "effective_training_depth": 16,
        "controller_placement": "loop boundary after Block2 FFN",
        "leading_answer_heads_at_cycle64": leading.to_dict("records"),
        "interpretation_boundary": (
            "Current/successor readout, exact update bypass, semantic-edge ablation, "
            "and same-graph alternate-current interchange establish component-level "
            "roles. They do not by themselves establish a unique minimal Block1 circuit."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-run", type=Path, required=True)
    parser.add_argument("--seed3-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
