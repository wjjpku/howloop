from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SEGMENTS_64 = ((1, 24), (25, 48), (49, 64))
SEGMENTS_128 = (
    (1, 24),
    (25, 48),
    (49, 64),
    (65, 80),
    (81, 96),
    (97, 112),
    (113, 128),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-strict", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _segments(values: list[float], segments) -> list[float]:
    array = np.asarray(values)
    return [
        float(array[start - 1 : end].mean())
        for start, end in segments
    ]


def _task_curve(summary: Path) -> list[float]:
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload["curves"]["task_unit_every"]["accuracy_nonendpoint"][
        "values"
    ]


def _validation_long(root: Path, prefix: str) -> dict[str, Any]:
    curves = [
        _task_curve(path)
        for path in sorted(
            (root / "validation_long_horizon").glob(
                f"{prefix}_eval*/summary.json"
            )
        )
    ]
    values = np.asarray(curves)
    return {
        "evaluation_seeds": len(curves),
        "mean_curve": values.mean(axis=0).tolist(),
        "std_curve": values.std(axis=0).tolist(),
        "segments_mean": _segments(
            values.mean(axis=0).tolist(),
            SEGMENTS_128,
        ),
        "segments_std": [
            float(
                np.asarray(
                    [
                        _segments(curve.tolist(), SEGMENTS_128)[index]
                        for curve in values
                    ]
                ).std()
            )
            for index in range(len(SEGMENTS_128))
        ],
    }


def _strict_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unseen = next(
        item
        for item in payload["results"]
        if item["partition"] == "strictly_unseen_by_J_training"
    )
    curve = unseen["curves"]["learned_J_plus_full_Block2"]
    values = curve["nonendpoint_accuracy_by_cycle"]
    segments = SEGMENTS_64 if len(values) == 64 else SEGMENTS_128[:5]
    return {
        "unique_training_graphs": payload["unique_training_graphs"],
        "strictly_unseen_graphs": payload["strictly_unseen_graphs"],
        "continuation_loops": payload["continuation_loops"],
        "segments": _segments(values, segments),
        "executor_blocked_nonendpoint_accuracy": unseen[
            "one_step_executor_blocked_nonendpoint_accuracy"
        ]["learned_J_Block2_answer_updates_zero"],
    }


def _multiseed(root: Path, name: str) -> dict[str, Any]:
    values = []
    for summary in sorted(
        (root / "multiseed").glob(f"task*/{name}/summary.json")
    ):
        curve = json.loads(summary.read_text(encoding="utf-8"))["curves"][
            "task_unit_every"
        ]["accuracy_nonendpoint"]
        values.append(float(curve["auc_49_64"]))
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
    }


def _spectral(root: Path, name: str, stage: str) -> dict[str, Any]:
    rows = json.loads(
        (root / "analysis_all" / "candidate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(
        item
        for item in rows
        if item["name"] == name and item["stage"] == stage
    )
    keys = (
        "initial_weight_delta_relative",
        "initial_bias_delta_relative",
        "eigen_near_zero_0p1",
        "eigen_near_one_0p1",
        "eigen_abs_max",
        "eigen_angle_abs_p95",
        "logabsdet",
    )
    return {key: row[key] for key in keys}


def _plot_long(metrics: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    labels = {
        "base": "best H64 J",
        "extend_h64_80_96": "extend: H64/80/96",
        "extend_state003_h96": "extend: H96, state=0.03",
    }
    colors = {
        "base": "#777777",
        "extend_h64_80_96": "#4477AA",
        "extend_state003_h96": "#EE6677",
    }
    for name, label in labels.items():
        item = metrics["long_validation"][name]
        mean = np.asarray(item["mean_curve"])
        std = np.asarray(item["std_curve"])
        x = np.arange(1, len(mean) + 1)
        axis.plot(x, mean, label=label, color=colors[name], linewidth=2)
        axis.fill_between(
            x,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=colors[name],
            alpha=0.15,
        )
    axis.axvline(64, linestyle="--", linewidth=1, color="black", alpha=0.5)
    axis.axvline(96, linestyle="--", linewidth=1, color="black", alpha=0.5)
    axis.set_xlabel("Continuation loop")
    axis.set_ylabel("Non-endpoint accuracy")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_multiseed(metrics: dict[str, Any], path: Path) -> None:
    order = (
        "full_lr1e6_h64",
        "lora_r64_h64",
        "lora_r32_mixed",
        "lora_r32_mixed_freeze_bias",
    )
    labels = (
        "full H64",
        "LoRA r64 H64",
        "LoRA r32 mixed",
        "LoRA r32 frozen bias",
    )
    means = [metrics["multiseed"][name]["mean"] for name in order]
    errors = [metrics["multiseed"][name]["std"] for name in order]
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.bar(labels, means, yerr=errors, capsize=5, color="#4477AA")
    axis.set_ylabel("AUC, loops 49–64")
    axis.set_ylim(0.82, 0.96)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_strict(metrics: dict[str, Any], path: Path) -> None:
    names = (
        "original",
        "full64",
        "lora64",
        "extend_h64_80_96",
        "extend_state003_h96",
    )
    labels = (
        "original",
        "full H64",
        "LoRA r64",
        "extend 64/80/96",
        "extend H96 state=.03",
    )
    segment_labels = ("1–24", "25–48", "49–64", "65–80", "81–96")
    x = np.arange(len(segment_labels))
    width = 0.15
    figure, axis = plt.subplots(figsize=(10.0, 5.0))
    for index, (name, label) in enumerate(zip(names, labels, strict=True)):
        values = metrics["strict_unseen"][name]["segments"]
        padded = values + [float("nan")] * (len(segment_labels) - len(values))
        axis.bar(
            x + (index - 2) * width,
            padded,
            width=width,
            label=label,
        )
    axis.set_xticks(x, segment_labels)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("Strict-unseen non-endpoint accuracy")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.root
    metrics: dict[str, Any] = {
        "fixed_model": {
            "loss_placement": (
                "loop8 final CE plus loop1-7 intermediate CE on p_min(2t,D)"
            ),
            "trained_macro_loops": 8,
            "shared_blocks": 2,
            "effective_depth": 16,
            "d_model": 256,
            "J_interface": "Block1 output to Block2 input, once per loop",
        },
        "experiment_counts": {
            "screen_candidates": len(
                json.loads(
                    (root / "screen" / "progress.json").read_text(
                        encoding="utf-8"
                    )
                )
            ),
            "combination_candidates": len(
                json.loads(
                    (root / "combinations" / "progress.json").read_text(
                        encoding="utf-8"
                    )
                )
            ),
            "lowrank_refinement_candidates": len(
                json.loads(
                    (
                        root / "lowrank_refinement" / "progress.json"
                    ).read_text(encoding="utf-8")
                )
            ),
            "long_horizon_candidates": len(
                json.loads(
                    (root / "long_horizon" / "progress.json").read_text(
                        encoding="utf-8"
                    )
                )
            ),
            "task_training_seeds": 3,
        },
        "multiseed": {
            name: _multiseed(root, name)
            for name in (
                "full_lr1e6_h64",
                "lora_r64_h64",
                "lora_r32_mixed",
                "lora_r32_mixed_freeze_bias",
            )
        },
        "long_validation": {
            name: _validation_long(root, name)
            for name in (
                "base",
                "extend_h64_80_96",
                "extend_state003_h96",
            )
        },
        "strict_unseen": {
            "original": _strict_metrics(args.original_strict),
            **{
                name: _strict_metrics(
                    root / "strict_unseen" / name / "summary.json"
                )
                for name in (
                    "full64",
                    "lora64",
                    "extend_h64_80_96",
                    "extend_state003_h96",
                )
            },
        },
        "spectral": {
            "full64": _spectral(
                root,
                "full_lr1e6_h64",
                "combinations",
            ),
            "lora64": _spectral(
                root,
                "lora_r64_lr1e6_h64",
                "lowrank_refinement",
            ),
            "extend_h64_80_96": _spectral(
                root,
                "extend_lr5e7_h64_80_96",
                "long_horizon",
            ),
            "extend_state003_h96": _spectral(
                root,
                "extend_lr5e7_state003_h96",
                "long_horizon",
            ),
        },
    }
    (args.out_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_long(metrics, args.out_dir / "long_horizon_validation.png")
    _plot_multiseed(metrics, args.out_dir / "multiseed_auc.png")
    _plot_strict(metrics, args.out_dir / "strict_unseen_segments.png")
    print(json.dumps(metrics["experiment_counts"], indent=2))


if __name__ == "__main__":
    main()
