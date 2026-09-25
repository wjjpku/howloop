#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_LENGTHS = (20, 40, 75, 84, 100)
REQUIRED_VARIANTS = (
    "raw",
    "full",
    "no_AB",
    "identity_D",
    "full_executor_off",
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def summary_path(
    run_root: Path, *, seed: int, curriculum: str, controller_seed: int
) -> Path:
    label = (
        f"parity_adaptive_step_released64_seed{seed}_rank48_"
        f"{curriculum}_seed{controller_seed}"
    )
    return run_root / "audits" / label / "summary.json"


def validate_summary(payload: dict[str, Any], *, seed: int, path: Path) -> None:
    checks = {
        "status": payload.get("status") == "complete",
        "paper": payload.get("paper") == "arXiv:2409.15647v5",
        "heads": payload.get("model", {}).get("n_heads") == 64,
        "precision": payload.get("backbone_training_precision") == "fp32",
        "backbone_seed": payload.get("backbone_seed") == seed,
        "supervision": payload.get("backbone_supervision") == "adaptive_step",
        "trained_loop_count": payload.get("trained_loop_count") == 20,
        "examples_per_length": payload.get("examples_per_length", 0) >= 4096,
        "rank": payload.get("controller_rank") == 48,
    }
    curves = payload.get("curves", {})
    checks["variants"] = all(variant in curves for variant in REQUIRED_VARIANTS)
    checks["lengths"] = all(
        str(length) in curves.get("raw", {}) for length in REQUIRED_LENGTHS
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{path}: failed audit invariants: {', '.join(failed)}")


def flatten_summary(payload: dict[str, Any], *, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in REQUIRED_VARIANTS:
        for length in REQUIRED_LENGTHS:
            curve = payload["curves"][variant][str(length)]
            post = curve.get("post_target_exact_match", {})
            rows.append(
                {
                    "backbone_seed": seed,
                    "variant": variant,
                    "length": length,
                    "target_step_exact_match": curve["target_step_exact_match"],
                    "post_target_auc_1_32": curve["post_target_auc_1_32"],
                    "late_auc_17_32": curve["late_auc_17_32"],
                    "post_em_plus_1": post.get("1"),
                    "post_em_plus_2": post.get("2"),
                    "post_em_plus_4": post.get("4"),
                    "post_em_plus_8": post.get("8"),
                    "contiguous_extra_at_0_90": curve[
                        "contiguous_extra_loops_at_or_above"
                    ]["0.90"],
                    "contiguous_extra_at_0_95": curve[
                        "contiguous_extra_loops_at_or_above"
                    ]["0.95"],
                }
            )
    return rows


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["length"])].append(row)
    metrics = (
        "target_step_exact_match",
        "post_target_auc_1_32",
        "late_auc_17_32",
        "post_em_plus_1",
        "post_em_plus_2",
        "post_em_plus_4",
        "post_em_plus_8",
        "contiguous_extra_at_0_90",
        "contiguous_extra_at_0_95",
    )
    result: list[dict[str, Any]] = []
    for (variant, length), members in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "variant": variant,
            "length": length,
            "seeds": len(members),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in members if row[metric] is not None]
            aggregate[f"{metric}_mean"] = statistics.mean(values) if values else None
            aggregate[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        result.append(aggregate)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def fmt(mean: float | None, std: float | None) -> str:
    if mean is None or std is None or math.isnan(mean) or math.isnan(std):
        return "NA"
    return f"{mean:.3f} +/- {std:.3f}"


def markdown_report(aggregate: list[dict[str, Any]], seeds: list[int]) -> str:
    lookup = {(row["variant"], row["length"]): row for row in aggregate}
    lines = [
        "# Released64 computational-telomere study",
        "",
        f"Backbone seeds: {', '.join(map(str, seeds))}.",
        "",
        "All values are mean +/- sample standard deviation across backbone seeds.",
        "",
        "| Length | Raw target EM | Full J target EM | Gain | no_AB | Executor-off |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for length in REQUIRED_LENGTHS:
        raw = lookup[("raw", length)]
        full = lookup[("full", length)]
        no_ab = lookup[("no_AB", length)]
        off = lookup[("full_executor_off", length)]
        raw_mean = raw["target_step_exact_match_mean"]
        full_mean = full["target_step_exact_match_mean"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(length),
                    fmt(raw_mean, raw["target_step_exact_match_std"]),
                    fmt(full_mean, full["target_step_exact_match_std"]),
                    f"{full_mean - raw_mean:+.3f}",
                    fmt(
                        no_ab["target_step_exact_match_mean"],
                        no_ab["target_step_exact_match_std"],
                    ),
                    fmt(
                        off["target_step_exact_match_mean"],
                        off["target_step_exact_match_std"],
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "| Length | Raw post-target AUC | Full J post-target AUC | Gain |",
            "|---:|---:|---:|---:|",
        )
    )
    for length in REQUIRED_LENGTHS:
        raw = lookup[("raw", length)]
        full = lookup[("full", length)]
        raw_mean = raw["post_target_auc_1_32_mean"]
        full_mean = full["post_target_auc_1_32_mean"]
        lines.append(
            f"| {length} | "
            f"{fmt(raw_mean, raw['post_target_auc_1_32_std'])} | "
            f"{fmt(full_mean, full['post_target_auc_1_32_std'])} | "
            f"{full_mean - raw_mean:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--curriculum", default="extension")
    parser.add_argument("--controller-seed", type=int, default=211001)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(args.seeds)
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for seed in seeds:
        path = summary_path(
            args.run_root,
            seed=seed,
            curriculum=args.curriculum,
            controller_seed=args.controller_seed,
        )
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_summary(payload, seed=seed, path=path)
        rows.extend(flatten_summary(payload, seed=seed))
        sources.append(str(path))
    status = "complete" if not missing and len(sources) == len(seeds) else "incomplete"
    if status == "incomplete":
        payload = {"status": status, "missing": missing, "sources": sources}
        atomic_json(args.out_dir / "summary.json", payload)
        if not args.allow_incomplete:
            raise SystemExit("released64 study is incomplete; see summary.json")
        return 0
    aggregate = aggregate_rows(rows)
    write_csv(args.out_dir / "per_seed.csv", rows)
    write_csv(args.out_dir / "aggregate.csv", aggregate)
    atomic_text(args.out_dir / "REPORT.md", markdown_report(aggregate, seeds))
    atomic_json(
        args.out_dir / "summary.json",
        {
            "status": status,
            "paper": "arXiv:2409.15647v5",
            "baseline": "released parity YAML, 64 heads",
            "controller": "J(h)=hD+(hA)B+b, rank 48",
            "controller_curriculum": args.curriculum,
            "controller_loss": "final answer-region task CE only",
            "backbone_seeds": seeds,
            "controller_seed": args.controller_seed,
            "sources": sources,
            "aggregate": aggregate,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
