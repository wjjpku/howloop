from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("D8L8_seed0", "D8L8_seed3")
COLORS = {"D8L8_seed0": "#0072B2", "D8L8_seed3": "#D55E00"}


def _read(run_dir: Path, model: str) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name, filename in (
        ("baseline", "baseline.csv"),
        ("stage", "stage_readout.csv"),
        ("intervention", "bypass_and_semantic_edges.csv"),
        ("interchange", "current_interchange.csv"),
    ):
        table = pd.read_csv(run_dir / filename, low_memory=False)
        table["model"] = model
        result[name] = table
    return result


def _with_effects(run: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline = run["baseline"]
    full = baseline[baseline.run.eq("base_current")][
        ["model", "cycle", "accuracy", "target_margin"]
    ].rename(columns={"accuracy": "full_accuracy", "target_margin": "full_margin"})
    table = run["intervention"].merge(
        full, on=["model", "cycle"], validate="many_to_one"
    )
    table["accuracy_drop"] = table.full_accuracy - table.accuracy
    table["margin_drop"] = table.full_margin - table.target_margin
    return table


def run(args: argparse.Namespace) -> dict[str, object]:
    native_runs = (
        _read(args.native_seed0, MODELS[0]),
        _read(args.native_seed3, MODELS[1]),
    )
    detail_runs = (
        _read(args.detail_seed0, MODELS[0]),
        _read(args.detail_seed3, MODELS[1]),
    )
    continuation_runs = (
        _read(args.continuation_seed0, MODELS[0]),
        _read(args.continuation_seed3, MODELS[1]),
    )
    native = {
        key: pd.concat([item[key] for item in native_runs], ignore_index=True)
        for key in native_runs[0]
    }
    detail_effects = pd.concat(
        [_with_effects(item) for item in detail_runs], ignore_index=True
    )
    native_effects = pd.concat(
        [_with_effects(item) for item in native_runs], ignore_index=True
    )
    continuation_effects = pd.concat(
        [_with_effects(item) for item in continuation_runs], ignore_index=True
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    native["stage"].to_csv(args.out_dir / "native_stage_readout.csv", index=False)
    native_effects.to_csv(args.out_dir / "native_intervention_effects.csv", index=False)
    detail_effects.to_csv(args.out_dir / "native_loop1_position_and_edge_effects.csv", index=False)

    initial_start = detail_effects[
        detail_effects.component.str.match(r"B1\.H\d+\.edge_zero_start")
    ].copy()
    initial_start["head"] = initial_start.component.str.extract(r"B1\.H(\d+)").astype(int)
    late_self = continuation_effects[
        continuation_effects.cycle.eq(64)
        & continuation_effects.component.str.match(
            r"B1\.H\d+\.edge_zero_answer_self"
        )
    ].copy()
    late_self["head"] = late_self.component.str.extract(r"B1\.H(\d+)").astype(int)
    head_roles = pd.concat(
        (
            initial_start.assign(role="native loop 1: read start"),
            late_self.assign(role="controlled cycle 64: answer self"),
        ),
        ignore_index=True,
    )
    head_roles.to_csv(args.out_dir / "block1_head_role_shift.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    stage_order = (
        "loop_input",
        "B1_post_attention",
        "B1_post_mlp",
        "B2_post_attention",
        "B2_post_mlp",
    )
    ax = axes[0, 0]
    for model in MODELS:
        table = native["stage"]
        table = table[table.model.eq(model) & table.cycle.eq(1)].set_index("stage")
        table = table.reindex(stage_order)
        x = np.arange(len(stage_order))
        ax.plot(x, table.current_accuracy, marker="o", color=COLORS[model], label=f"{model}: current")
        ax.plot(x, table.successor_accuracy, marker="s", linestyle="--", color=COLORS[model], label=f"{model}: successor")
    ax.set_xticks(np.arange(len(stage_order)), ("input", "B1 attn", "B1 MLP", "B2 attn", "B2 MLP"))
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("model readout accuracy")
    ax.set_title("Native loop 1: B1 loads current; B2 computes successor")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    conditions = (
        ("full", None),
        ("no answer\nattention", "B1.attention_update_zero_answer"),
        ("no graph\nattention", "B1.attention_update_zero_graph"),
        ("no graph\nMLP", "B1.mlp_update_zero_graph"),
        ("no graph\nupdates", "B1.full_update_zero_graph"),
        ("B1 exact\nbypass", "B1.full_update_zero_all"),
    )
    x = np.arange(len(conditions))
    width = 0.36
    for model_index, model in enumerate(MODELS):
        baseline = native["baseline"]
        baseline = baseline[
            baseline.model.eq(model)
            & baseline.cycle.eq(1)
            & baseline.run.eq("base_current")
        ]
        values: list[float] = []
        for _, component in conditions:
            if component is None:
                values.append(float(baseline.accuracy.iloc[0]))
            else:
                row = detail_effects[
                    detail_effects.model.eq(model)
                    & detail_effects.component.eq(component)
                ]
                values.append(float(row.accuracy.iloc[0]))
        ax.bar(x + (model_index - 0.5) * width, values, width=width, color=COLORS[model], label=model)
    ax.axhline(0.125, color="black", linewidth=0.8, linestyle="--", label="chance")
    ax.set_xticks(x, [label for label, _ in conditions])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("successor accuracy")
    ax.set_title("Native loop 1: answer loading and graph compilation are both required")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for model in MODELS:
        baseline = native["baseline"]
        baseline = baseline[baseline.model.eq(model) & baseline.run.eq("base_current")][["cycle", "accuracy"]]
        bypass = native_effects[
            native_effects.model.eq(model)
            & native_effects.component.eq("B1.full_update_zero_all")
        ][["cycle", "accuracy"]]
        ax.plot(baseline.cycle, baseline.accuracy, marker="o", color=COLORS[model], label=f"{model}: full")
        ax.plot(bypass.cycle, bypass.accuracy, marker="s", linestyle="--", color=COLORS[model], label=f"{model}: bypass B1")
    ax.set_xticks((1, 2, 4, 8))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("native loop")
    ax.set_ylabel("successor accuracy")
    ax.set_title("B1 changes from indispensable initializer to redundant maintainer")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    x = np.arange(4)
    width = 0.18
    offset = 0
    for model in MODELS:
        for role, hatch in (
            ("native loop 1: read start", ""),
            ("controlled cycle 64: answer self", "//"),
        ):
            table = head_roles[head_roles.model.eq(model) & head_roles.role.eq(role)].set_index("head")
            values = [float(table.loc[head, "margin_drop"]) for head in range(4)]
            ax.bar(x + (offset - 1.5) * width, values, width=width, color=COLORS[model], hatch=hatch, alpha=0.85, label=f"{model}, {role}")
            offset += 1
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"H{head}" for head in range(4)])
    ax.set_ylabel("successor margin drop")
    ax.set_title("Shared B1 heads change semantic route with recurrent phase")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out_dir / "block1_native_and_continuation_mechanism.png", dpi=220)
    fig.savefig(args.out_dir / "block1_native_and_continuation_mechanism.pdf")
    plt.close(fig)

    def value(model: str, component: str, column: str) -> float:
        row = detail_effects[
            detail_effects.model.eq(model) & detail_effects.component.eq(component)
        ]
        return float(row[column].iloc[0])

    summary: dict[str, object] = {
        "status": "complete",
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": 8,
        "shared_physical_blocks": 2,
        "effective_training_depth": 16,
        "native_evaluated_loops": [1, 2, 4, 8],
        "controlled_continuation_cycles": [1, 32, 64],
        "examples_per_backbone": 256,
        "data_seed": 20260814,
        "native_loop1_accuracy": {
            model: float(
                native["baseline"][
                    native["baseline"].model.eq(model)
                    & native["baseline"].cycle.eq(1)
                    & native["baseline"].run.eq("base_current")
                ].accuracy.iloc[0]
            )
            for model in MODELS
        },
        "native_loop1_no_answer_attention_accuracy": {
            model: value(model, "B1.attention_update_zero_answer", "accuracy")
            for model in MODELS
        },
        "native_loop1_no_graph_updates_accuracy": {
            model: value(model, "B1.full_update_zero_graph", "accuracy")
            for model in MODELS
        },
        "native_loop1_primary_start_readers": {
            "D8L8_seed0": [1, 2],
            "D8L8_seed3": [0],
        },
        "late_controlled_primary_self_preconditioners": {
            "D8L8_seed0": 1,
            "D8L8_seed3": 2,
        },
        "mechanism": (
            "At native loop 1, B1 attention loads the start node into the answer "
            "interface and B1 attention plus MLP compile raw graph tokens into the "
            "records consumed by B2. From loop 2 onward the current variable is "
            "carried mainly by the answer residual; B1 supplies a phase-conditioned "
            "query correction and MLP margin calibration, while B2 attention computes "
            "the successor."
        ),
        "claim_boundary": (
            "This establishes stage, branch, position, semantic-edge, and same-graph "
            "current-interface causal roles on two backbones, not a unique minimal "
            "neuron-level circuit."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-seed0", type=Path, required=True)
    parser.add_argument("--native-seed3", type=Path, required=True)
    parser.add_argument("--detail-seed0", type=Path, required=True)
    parser.add_argument("--detail-seed3", type=Path, required=True)
    parser.add_argument("--continuation-seed0", type=Path, required=True)
    parser.add_argument("--continuation-seed3", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
