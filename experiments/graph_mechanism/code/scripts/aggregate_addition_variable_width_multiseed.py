from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate variable-width Addition raw/J horizon evaluations."
    )
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_run(run_root: Path) -> tuple[int, list[dict[str, float]]]:
    manifest = json.loads((run_root / "pipeline_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete run: {run_root}")
    seed = int(manifest["backbone_seed"])
    summary_path = run_root / "selected_fullanswer_l1to40_512" / "summary.json"
    summary = json.loads(summary_path.read_text())
    indexed: dict[int, dict[str, dict[str, float]]] = {}
    for row in summary["rows"]:
        indexed.setdefault(int(row["length"]), {})[str(row["variant"])] = row
    rows: list[dict[str, float]] = []
    for length, variants in sorted(indexed.items()):
        raw = variants["raw"]
        full = variants["full"]
        rows.append(
            {
                "seed": seed,
                "length": length,
                "raw_em": float(raw["exact_match"]),
                "j_em": float(full["exact_match"]),
                "delta_em": float(full["exact_match"] - raw["exact_match"]),
                "raw_ce": float(raw["answer_cross_entropy"]),
                "j_ce": float(full["answer_cross_entropy"]),
                "delta_ce": float(
                    full["answer_cross_entropy"] - raw["answer_cross_entropy"]
                ),
                "raw_margin": float(raw["mean_sequence_min_margin"]),
                "j_margin": float(full["mean_sequence_min_margin"]),
            }
        )
    return seed, rows


def first_below(rows: list[dict[str, float]], key: str, threshold: float) -> int | None:
    for row in rows:
        if row[key] < threshold:
            return int(row["length"])
    return None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed: dict[int, list[dict[str, float]]] = {}
    all_rows: list[dict[str, float]] = []
    for run_root in args.run_roots:
        seed, rows = load_run(run_root)
        if seed in per_seed:
            raise ValueError(f"duplicate seed {seed}")
        per_seed[seed] = rows
        all_rows.extend(rows)

    fields = list(all_rows[0])
    with (args.out_dir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    lengths = sorted({int(row["length"]) for row in all_rows})
    aggregate: list[dict[str, float]] = []
    for length in lengths:
        selected = [row for row in all_rows if int(row["length"]) == length]
        record: dict[str, float] = {"length": length, "seeds": len(selected)}
        for key in ("raw_em", "j_em", "delta_em", "raw_ce", "j_ce", "delta_ce"):
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(record)
    with (args.out_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    seed_summaries = []
    for seed, rows in sorted(per_seed.items()):
        seed_summaries.append(
            {
                "seed": seed,
                "raw_first_em_below_0.99": first_below(rows, "raw_em", 0.99),
                "j_first_em_below_0.99": first_below(rows, "j_em", 0.99),
                "raw_first_em_below_0.5": first_below(rows, "raw_em", 0.5),
                "j_first_em_below_0.5": first_below(rows, "j_em", 0.5),
                "mean_delta_em_11_20": float(
                    np.mean([row["delta_em"] for row in rows if 11 <= row["length"] <= 20])
                ),
            }
        )
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "task": "Addition with variable active width m in 1..10",
                "backbone_loss": "final answer-region CE at T(m)=m+1",
                "controller": "identity-initialized diagonal plus rank-48 LoRA and bias",
                "controller_loss": "final answer-region CE at T(m)=m+1 for m in 1..10",
                "selection": "ID-only checkpoint selection",
                "seed_summaries": seed_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    x = np.asarray(lengths)
    raw_mean = np.asarray([row["raw_em_mean"] for row in aggregate])
    raw_std = np.asarray([row["raw_em_std"] for row in aggregate])
    j_mean = np.asarray([row["j_em_mean"] for row in aggregate])
    j_std = np.asarray([row["j_em_std"] for row in aggregate])
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for seed, rows in sorted(per_seed.items()):
        ax.plot(
            [row["length"] for row in rows],
            [row["raw_em"] for row in rows],
            color="#777777",
            alpha=0.22,
            linewidth=1,
        )
        ax.plot(
            [row["length"] for row in rows],
            [row["j_em"] for row in rows],
            color="#2f6fdd",
            alpha=0.22,
            linewidth=1,
        )
    ax.plot(x, raw_mean, color="#333333", linewidth=2.4, label="Raw mean")
    ax.fill_between(x, raw_mean - raw_std, raw_mean + raw_std, color="#777777", alpha=0.15)
    ax.plot(x, j_mean, color="#2f6fdd", linewidth=2.4, label="J mean")
    ax.fill_between(x, j_mean - j_std, j_mean + j_std, color="#2f6fdd", alpha=0.15)
    ax.axvspan(1, 10, color="#f2c14e", alpha=0.12, label="Train widths")
    ax.axvline(10.5, color="#b8860b", linestyle="--", linewidth=1)
    ax.set(xlabel="Logical width / target loops minus 1", ylabel="Exact match", ylim=(-0.03, 1.03), xlim=(1, 40))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(args.out_dir / "variable_width_addition_multiseed.png", dpi=180)
    fig.savefig(args.out_dir / "variable_width_addition_multiseed.pdf")


if __name__ == "__main__":
    main()
