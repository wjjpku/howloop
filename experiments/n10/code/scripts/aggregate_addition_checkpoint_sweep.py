from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_STEPS = (10_000, 20_000, 30_000, 40_000, 50_000, 100_000)
EXPECTED_SEEDS = (0, 1, 2)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    selected = [float(value) for value in values]
    return fmean(selected), stdev(selected) if len(selected) > 1 else 0.0


def _horizon(
    curve: dict[int, float], *, threshold: float, start_length: int = 10
) -> int:
    horizon = start_length - 1
    for length in range(start_length, max(curve) + 1):
        if curve.get(length, -math.inf) < threshold:
            break
        horizon = length
    return horizon


def load_summaries(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for step in EXPECTED_STEPS:
            path = root / f"seed{seed}" / f"step_{step:06d}" / "summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                raise ValueError(f"incomplete result: {path}")
            if int(payload["checkpoint_step"]) != step:
                raise ValueError(f"checkpoint step mismatch: {path}")
            if int(payload["backbone_seed"]) != seed:
                raise ValueError(f"backbone seed mismatch: {path}")
            if payload["lengths"] != list(range(10, 41)):
                raise ValueError(f"length grid mismatch: {path}")
            if payload["step_offsets"] != [-2, -1, 0, 1, 2]:
                raise ValueError(f"nearby-loop grid mismatch: {path}")
            summaries.append(payload)
    return summaries


def build_tables(
    summaries: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed_step_length: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    for payload in summaries:
        seed = int(payload["backbone_seed"])
        step = int(payload["checkpoint_step"])
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for raw_row in payload["rows"]:
            row = dict(raw_row)
            row.update({"backbone_seed": seed, "checkpoint_step": step})
            trajectory_rows.append(row)
            by_length[int(row["length"])].append(row)
        for length, rows in by_length.items():
            by_seed_step_length[(seed, step, length)] = rows

    target_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for step in EXPECTED_STEPS:
        per_seed_target_strict: dict[int, dict[int, float]] = defaultdict(dict)
        per_seed_target_actual: dict[int, dict[int, float]] = defaultdict(dict)
        per_seed_nearby_strict: dict[int, dict[int, float]] = defaultdict(dict)
        per_seed_nearby_actual: dict[int, dict[int, float]] = defaultdict(dict)
        for length in range(10, 41):
            per_seed: list[dict[str, Any]] = []
            for seed in EXPECTED_SEEDS:
                rows = by_seed_step_length[(seed, step, length)]
                target = next(row for row in rows if int(row["step_offset"]) == 0)
                nearby = [row for row in rows if -2 <= int(row["step_offset"]) <= 2]
                best_strict = max(
                    nearby, key=lambda row: float(row["strict_exact_match"])
                )
                best_actual = max(
                    nearby, key=lambda row: float(row["actual_answer_exact_match"])
                )
                per_seed_target_strict[seed][length] = float(
                    target["strict_exact_match"]
                )
                per_seed_target_actual[seed][length] = float(
                    target["actual_answer_exact_match"]
                )
                per_seed_nearby_strict[seed][length] = float(
                    best_strict["strict_exact_match"]
                )
                per_seed_nearby_actual[seed][length] = float(
                    best_actual["actual_answer_exact_match"]
                )
                per_seed.append(
                    {
                        "target_strict": float(target["strict_exact_match"]),
                        "target_actual": float(target["actual_answer_exact_match"]),
                        "target_carry": float(target["carry_accuracy"]),
                        "target_token": float(target["actual_answer_token_accuracy"]),
                        "target_margin": float(target["mean_actual_sequence_min_margin"]),
                        "target_frontier": float(target["mean_correct_low_order_digits"]),
                        "nearby_strict": float(best_strict["strict_exact_match"]),
                        "nearby_actual": float(best_actual["actual_answer_exact_match"]),
                        "nearby_strict_offset": int(best_strict["step_offset"]),
                        "nearby_actual_offset": int(best_actual["step_offset"]),
                    }
                )
            output: dict[str, Any] = {
                "checkpoint_step": step,
                "length": length,
                "backbone_seeds": len(per_seed),
                "examples_per_seed": 512,
            }
            for key in (
                "target_strict",
                "target_actual",
                "target_carry",
                "target_token",
                "target_margin",
                "target_frontier",
                "nearby_strict",
                "nearby_actual",
                "nearby_strict_offset",
                "nearby_actual_offset",
            ):
                mean, std = _mean_std(float(item[key]) for item in per_seed)
                output[f"{key}_mean"] = mean
                output[f"{key}_std"] = std
                output[f"{key}_min"] = min(float(item[key]) for item in per_seed)
                output[f"{key}_max"] = max(float(item[key]) for item in per_seed)
            target_rows.append(output)

        for seed in EXPECTED_SEEDS:
            for threshold in (0.90, 0.95, 0.98):
                horizon_rows.append(
                    {
                        "checkpoint_step": step,
                        "backbone_seed": seed,
                        "threshold": threshold,
                        "target_strict_horizon": _horizon(
                            per_seed_target_strict[seed], threshold=threshold
                        ),
                        "target_actual_horizon": _horizon(
                            per_seed_target_actual[seed], threshold=threshold
                        ),
                        "nearby_strict_horizon": _horizon(
                            per_seed_nearby_strict[seed], threshold=threshold
                        ),
                        "nearby_actual_horizon": _horizon(
                            per_seed_nearby_actual[seed], threshold=threshold
                        ),
                    }
                )
    return target_rows, horizon_rows, trajectory_rows


def make_figure(
    *,
    target_rows: Sequence[dict[str, Any]],
    horizon_rows: Sequence[dict[str, Any]],
    trajectory_rows: Sequence[dict[str, Any]],
    out_path: Path,
) -> None:
    target_lookup = {
        (int(row["checkpoint_step"]), int(row["length"])): row
        for row in target_rows
    }
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(EXPECTED_STEPS)))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    steps_k = np.array(EXPECTED_STEPS) / 1000
    for metric, label, marker in (
        ("target_strict", "strict EM (with PAD)", "o"),
        ("target_actual", "sum-digit EM", "s"),
    ):
        means = [target_lookup[(step, 10)][f"{metric}_mean"] for step in EXPECTED_STEPS]
        lows = [target_lookup[(step, 10)][f"{metric}_min"] for step in EXPECTED_STEPS]
        highs = [target_lookup[(step, 10)][f"{metric}_max"] for step in EXPECTED_STEPS]
        ax.plot(steps_k, means, marker=marker, lw=2, label=label)
        ax.fill_between(steps_k, lows, highs, alpha=0.14)
    ax.axhline(0.98, color="0.35", ls=":", lw=1)
    ax.set(title="ID competence at n=10, T=11", xlabel="checkpoint (k updates)", ylabel="EM")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)

    for column, (metric, title) in enumerate(
        (("target_actual", "Registered T(n)=n+1"), ("nearby_actual", "Best in T(n)±2")),
        start=1,
    ):
        ax = axes[0, column]
        for color, step in zip(colors, EXPECTED_STEPS):
            values = [target_lookup[(step, length)][f"{metric}_mean"] for length in range(10, 41)]
            ax.plot(range(10, 41), values, color=color, lw=1.8, label=f"{step//1000}k")
        ax.axvline(10, color="0.35", ls=":", lw=1)
        ax.set(title=title, xlabel="logical length n", ylabel="sum-digit EM")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(ncol=2, fontsize=8, frameon=False)

    ax = axes[1, 0]
    selected_horizons = [
        row for row in horizon_rows if math.isclose(float(row["threshold"]), 0.90)
    ]
    for key, label, marker in (
        ("target_actual_horizon", "registered", "o"),
        ("nearby_actual_horizon", "nearby best", "s"),
    ):
        means, lows, highs = [], [], []
        for step in EXPECTED_STEPS:
            values = [
                float(row[key])
                for row in selected_horizons
                if int(row["checkpoint_step"]) == step
            ]
            means.append(fmean(values))
            lows.append(min(values))
            highs.append(max(values))
        ax.plot(steps_k, means, marker=marker, lw=2, label=label)
        ax.fill_between(steps_k, lows, highs, alpha=0.14)
    ax.set(title="Contiguous 90% horizon from n=10", xlabel="checkpoint (k updates)", ylabel="last passing length")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for color, step in zip(colors, EXPECTED_STEPS):
        values = [target_lookup[(step, length)]["target_carry_mean"] for length in range(10, 41)]
        ax.plot(range(10, 41), values, color=color, lw=1.8, label=f"{step//1000}k")
    ax.set(title="Registered-step carry accuracy", xlabel="logical length n", ylabel="carry accuracy")
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1, 2]
    diagnostic_length = 15
    for color, step in zip(colors, EXPECTED_STEPS):
        selected = [
            row
            for row in trajectory_rows
            if int(row["checkpoint_step"]) == step
            and int(row["length"]) == diagnostic_length
        ]
        by_loop: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            by_loop[int(row["step"])].append(float(row["mean_correct_low_order_digits"]))
        loops = sorted(by_loop)
        means = [fmean(by_loop[loop]) for loop in loops]
        ax.plot(loops, means, color=color, lw=1.8, label=f"{step//1000}k")
    ax.axvline(diagnostic_length + 1, color="0.35", ls=":", lw=1)
    ax.set(title="Digit frontier at n=15", xlabel="readout loop", ylabel="consecutive correct low-order digits")
    ax.legend(ncol=2, fontsize=8, frameon=False)

    fig.suptitle("Addition fixed-n10 baseline: checkpoint sweep (3 backbone seeds)", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_report(
    *,
    target_rows: Sequence[dict[str, Any]],
    horizon_rows: Sequence[dict[str, Any]],
) -> str:
    lookup = {
        (int(row["checkpoint_step"]), int(row["length"])): row
        for row in target_rows
    }
    horizons = [
        row for row in horizon_rows if math.isclose(float(row["threshold"]), 0.90)
    ]
    lines = [
        "# Addition fixed-n10 checkpoint sweep",
        "",
        "All models are 3 shared physical layers, 8 heads, d_model=256, d_mlp=1024, trained with final answer-region CE at T(10)=11. Each cell is the mean across three independently trained backbone seeds; evaluation uses the same 512 examples per length and seed.",
        "",
        "| checkpoint | n10 strict EM | n10 sum-digit EM | n15 target EM | n15 nearby EM | target H@90 | nearby H@90 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in EXPECTED_STEPS:
        target_h = [
            int(row["target_actual_horizon"])
            for row in horizons
            if int(row["checkpoint_step"]) == step
        ]
        nearby_h = [
            int(row["nearby_actual_horizon"])
            for row in horizons
            if int(row["checkpoint_step"]) == step
        ]
        row10 = lookup[(step, 10)]
        row15 = lookup[(step, 15)]
        lines.append(
            f"| {step//1000}k | {row10['target_strict_mean']:.3f} | "
            f"{row10['target_actual_mean']:.3f} | {row15['target_actual_mean']:.3f} | "
            f"{row15['nearby_actual_mean']:.3f} | {fmean(target_h):.1f} | "
            f"{fmean(nearby_h):.1f} |"
        )
    lines.extend(
        [
            "",
            "Strict EM includes the four trailing PAD/EOS targets in the released generator. Sum-digit EM excludes those positions. Nearby EM is the per-seed maximum over T(n)-2 through T(n)+2 and is used only to diagnose clock drift; the registered target remains primary.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = load_summaries(args.root)
    target_rows, horizon_rows, trajectory_rows = build_tables(summaries)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "target_and_nearby.csv", target_rows)
    _write_csv(args.out_dir / "horizons.csv", horizon_rows)
    _write_csv(args.out_dir / "all_trajectories.csv", trajectory_rows)
    make_figure(
        target_rows=target_rows,
        horizon_rows=horizon_rows,
        trajectory_rows=trajectory_rows,
        out_path=args.out_dir / "addition_checkpoint_sweep.png",
    )
    report = make_report(target_rows=target_rows, horizon_rows=horizon_rows)
    (args.out_dir / "REPORT.md").write_text(report)
    summary = {
        "status": "complete",
        "backbone_seeds": list(EXPECTED_SEEDS),
        "checkpoint_steps": list(EXPECTED_STEPS),
        "evaluation_lengths": list(range(10, 41)),
        "examples_per_length_per_seed": 512,
        "target_rows": target_rows,
        "horizon_rows": horizon_rows,
    }
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
