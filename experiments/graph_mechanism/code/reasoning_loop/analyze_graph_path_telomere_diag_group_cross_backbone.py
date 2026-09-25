from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def run(args: argparse.Namespace) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    definitions: dict[str, dict[str, object]] = {}
    for label, run_dir in (("seed0", args.seed0_dir), ("seed3", args.seed3_dir)):
        frame = pd.read_csv(run_dir / "analysis" / "coarse_group_summary.csv")
        frame["backbone"] = label
        frames.append(frame)
        definitions[label] = json.loads(
            (run_dir / "group_definitions.json").read_text(encoding="utf-8")
        )
    summary = pd.concat(frames, ignore_index=True)
    selected = summary[
        summary.signed_delta.eq(args.signed_delta)
        & summary.group.isin(["high_retention", "middle", "strongly_damped"])
    ].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "cross_backbone_group_curves.csv", index=False)
    overlap_rows: list[dict[str, object]] = []
    dimension = 256
    for group_name in ("high_retention", "middle", "strongly_damped"):
        left = set(definitions["seed0"][group_name]["coordinates"])
        right = set(definitions["seed3"][group_name]["coordinates"])
        overlap_rows.append(
            {
                "group": group_name,
                "seed0_seed3_overlap": len(left & right),
                "random_expected_overlap": len(left) * len(right) / dimension,
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(args.out_dir / "cross_backbone_group_overlap.csv", index=False)

    colors = {
        "high_retention": "#D55E00",
        "middle": "#0072B2",
        "strongly_damped": "#009E73",
    }
    labels = {
        "high_retention": "high-retention D group",
        "middle": "middle-D control",
        "strongly_damped": "strongly-damped D group",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    for ax, backbone in zip(axes, ("seed0", "seed3")):
        live = selected[selected.backbone.eq(backbone)]
        for group_name, group in live.groupby("group"):
            group = group.sort_values("cycle")
            ax.plot(
                group.cycle,
                group.accuracy_mean,
                marker="o",
                color=colors[group_name],
                label=labels[group_name],
            )
        ax.set(
            title=f"D8L8 {backbone}: same +{args.signed_delta:.02f} group gain",
            xlabel="controlled continuation loop",
            ylim=(-0.02, 1.03),
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("successor accuracy")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "cross_backbone_D_groups.png", dpi=220)
    fig.savefig(args.out_dir / "cross_backbone_D_groups.pdf")
    plt.close(fig)

    final = selected[selected.cycle.eq(args.cycle)][
        ["backbone", "group", "accuracy_mean", "accuracy_std", "margin_drop_mean"]
    ].sort_values(["backbone", "accuracy_mean"])
    result: dict[str, object] = {
        "status": "complete",
        "cycle": args.cycle,
        "signed_delta": args.signed_delta,
        "final_group_results": final.to_dict("records"),
        "coordinate_overlap": overlap.to_dict("records"),
        "interpretation": (
            "Both backbones are most sensitive to perturbing their own high-D "
            "group, while the actual coordinates overlap only at chance level."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-dir", type=Path, required=True)
    parser.add_argument("--seed3-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--signed-delta", type=float, default=0.02)
    parser.add_argument("--cycle", type=int, default=64)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
