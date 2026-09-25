"""Create a compact paper-facing summary of the J-attention experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anti-dir", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--natural-dir", type=Path, required=True)
    parser.add_argument("--subspace-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _row(frame: pd.DataFrame, **filters) -> pd.Series:
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    anti_patch = pd.read_csv(args.anti_dir / "patch_summary.csv")
    anti_activation = pd.read_csv(args.anti_dir / "activation_rows.csv")
    alignment = pd.read_csv(args.alignment_dir / "operator_effect_summary.csv")
    natural_behavior = pd.read_csv(args.natural_dir / "behavior_summary.csv")
    natural_activation = pd.read_csv(args.natural_dir / "activation_summary.csv")
    coordinates = pd.read_csv(args.natural_dir / "coordinate_summary.csv")
    subspace = pd.read_csv(args.subspace_dir / "behavior_summary.csv")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=190)
    labels = ("baseline", "anti", "random\nstate-match", "H0 pattern\nrescue", "H0 QKV\nrescue")
    for back_count, color, offset in ((8, "#4C78A8", -0.18), (12, "#E45756", 0.18)):
        values = [
            float(_row(anti_patch, back_count=back_count, condition="baseline", patch_label="none", direction="none")["accuracy"]),
            float(_row(anti_patch, back_count=back_count, condition="anti", patch_label="none", direction="none")["accuracy"]),
            float(_row(anti_patch, back_count=back_count, condition="random_state_effect_matched", patch_label="none", direction="none")["accuracy"]),
            float(_row(anti_patch, back_count=back_count, condition="anti", patch_label="B2.H0.pattern", direction="rescue")["accuracy"]),
            float(_row(anti_patch, back_count=back_count, condition="anti", patch_label="B2.H0.qkv", direction="rescue")["accuracy"]),
        ]
        axes[0, 0].bar(np.arange(len(labels)) + offset, values, 0.34, label=f"k={back_count}", color=color)
    axes[0, 0].set_xticks(np.arange(len(labels)), labels)
    axes[0, 0].set(ylim=(0, 1.03), ylabel="final accuracy", title="A. Anti-compression is selective and H0-mediated")
    axes[0, 0].legend()

    routing = anti_activation[
        (anti_activation["block"] == 2) & (anti_activation["head"] == 0)
    ].groupby(["back_count", "condition"], as_index=False).mean(numeric_only=True)
    conditions = ("anti", "random_state_effect_matched")
    for condition, marker in zip(conditions, ("o", "s"), strict=True):
        chosen = routing[routing["condition"] == condition]
        axes[0, 1].plot(
            chosen["back_count"],
            chosen["lookup_destination_mass_receiver"],
            marker=marker,
            label=condition.replace("_", " "),
        )
    clean = routing[routing["condition"] == "anti"]
    axes[0, 1].plot(clean["back_count"], clean["lookup_destination_mass_clean"], "o--", color="black", label="clean J")
    axes[0, 1].set(xlabel="J calls", ylabel="B2H0 destination attention", title="B. Repeated true leakage destroys lookup routing")
    axes[0, 1].legend(fontsize=8)

    for label, style in (
        ("none", "o-"),
        ("B2.H0.pattern", "s--"),
        ("B2.H0.context", "^--"),
        ("B2.H0.qkv", "D--"),
    ):
        chosen = natural_behavior[natural_behavior["patch_label"] == label]
        axes[1, 0].plot(chosen["back_count"], chosen["accuracy"], style, label=label)
    axes[1, 0].set(xlabel="J calls", ylabel="final accuracy", ylim=(0, 1.03), title="C. Natural overthinking failure is only partly H0-mediated")
    axes[1, 0].legend(fontsize=8)

    answer_coordinates = coordinates[coordinates["position_group"] == "answer"]
    for basis, marker in (("bottom_input", "o"), ("bottom_output", "s"), ("top_output", "^")):
        chosen = answer_coordinates[answer_coordinates["basis"] == basis]
        axes[1, 1].plot(chosen["back_count"], chosen["coordinate_rms_ratio"], marker=marker, label=basis)
    full = answer_coordinates[answer_coordinates["basis"] == "bottom_input"]
    axes[1, 1].plot(full["back_count"], full["full_state_rms_ratio"], "D--", color="black", label="full residual RMS")
    axes[1, 1].axhline(1.0, color="gray", linewidth=0.8)
    axes[1, 1].set(xlabel="J calls", ylabel="long / young-matched RMS", title="D. Compressed coordinates stay small while global residual grows")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.out_dir / "j_attention_mechanism_summary.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=190)
    labels = (
        "none",
        "loop_input.bottom_input.rank8",
        "loop_input.bottom_output.rank8",
        "loop_input.top_output.rank8",
        "loop_input.random.rank8",
        "B1.block_input_all",
    )
    width = 0.8 / len(subspace["back_count"].unique())
    for index, back_count in enumerate(sorted(subspace["back_count"].unique())):
        values = [float(_row(subspace, back_count=back_count, patch_label=label)["accuracy"]) for label in labels]
        axis.bar(np.arange(len(labels)) + (index - 1) * width, values, width, label=f"k={back_count}")
    axis.set_xticks(np.arange(len(labels)), [label.replace("loop_input.", "").replace(".rank8", "") for label in labels], rotation=20)
    axis.set(
        ylim=(0, 1.03),
        ylabel="final accuracy",
        title="Coordinate replacement: bottom directions re-inject harmful residue",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.out_dir / "bad_direction_reinjection_summary.png", bbox_inches="tight")
    plt.close(figure)

    result = {
        "status": "complete",
        "anti_dir": str(args.anti_dir),
        "alignment_dir": str(args.alignment_dir),
        "natural_dir": str(args.natural_dir),
        "subspace_dir": str(args.subspace_dir),
        "files": [
            "j_attention_mechanism_summary.png",
            "bad_direction_reinjection_summary.png",
        ],
        "alignment_immediate_effect": alignment.to_dict(orient="records"),
        "natural_b2_activation": natural_activation.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
