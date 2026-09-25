from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


CONDITIONS = ("baseline", "exact_young_state", "batch_shuffled", "reverse")


def expected_remaining_loops(
    *,
    donor_age: int,
    endpoint_macro_loop: int = 4,
) -> int:
    if not 0 <= donor_age < endpoint_macro_loop:
        raise ValueError("donor_age must precede the endpoint macro loop")
    return endpoint_macro_loop - donor_age


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    name, checkpoint = text.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def blank_bucket() -> dict[str, float]:
    return {
        "correct": 0.0,
        "probability": 0.0,
        "margin": 0.0,
        "count": 0.0,
    }


def accumulate(
    bucket: dict[str, float],
    metrics: dict[str, float | int],
) -> None:
    count = int(metrics["valid_count"])
    if not count:
        return
    bucket["correct"] += float(metrics["accuracy"]) * count
    bucket["probability"] += float(metrics["probability"]) * count
    bucket["margin"] += float(metrics["margin"]) * count
    bucket["count"] += count


def metrics_excluding_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    excluded_targets: tuple[torch.Tensor, ...],
) -> dict[str, float | int]:
    valid = torch.ones_like(target, dtype=torch.bool)
    for excluded in excluded_targets:
        valid &= target.ne(excluded)
    if not bool(valid.any()):
        return {
            "accuracy": float("nan"),
            "probability": float("nan"),
            "margin": float("nan"),
            "valid_count": 0,
        }
    return _masked_metrics(logits[valid], target[valid])


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("analysis expects D8L8 with two physical blocks")
    endpoint_macro_loop = cfg.max_depth // 2
    donor_ages = tuple(range(endpoint_macro_loop))
    accumulators = {
        (age, condition, extra_loop, target_kind): blank_bucket()
        for age in donor_ages
        for condition in CONDITIONS
        for extra_loop in range(1, extra_loops + 1)
        for target_kind in ("continued_path", "donor_endpoint", "original_endpoint")
    }
    set_seed(seed)
    for _ in range(batches):
        tokens, targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * extra_loops,
        )
        receiver_terminal = cache_states_with_initial(
            model, tokens, loops=cfg.max_loops
        )[-1]
        all_targets = _all_targets(start, targets)
        original_endpoint = all_targets[:, cfg.max_depth]
        for donor_age in donor_ages:
            donor_position = 2 * donor_age
            donor_start = advance_nodes(
                successors,
                start,
                steps=cfg.max_depth - donor_position,
            )
            donor_tokens, _, _, _ = fixed_depth_batch(
                cfg,
                batch_size,
                device,
                path_positions=cfg.max_depth,
                successors=successors,
                start=donor_start,
            )
            donor_state = cache_states_with_initial(
                model,
                donor_tokens,
                loops=max(1, donor_age),
            )[donor_age]
            donor_endpoint_position = (
                cfg.max_depth
                + 2
                * expected_remaining_loops(
                    donor_age=donor_age,
                    endpoint_macro_loop=endpoint_macro_loop,
                )
            )
            donor_endpoint = all_targets[:, donor_endpoint_position]
            remaining_loops = expected_remaining_loops(
                donor_age=donor_age,
                endpoint_macro_loop=endpoint_macro_loop,
            )
            states = {
                "baseline": receiver_terminal.clone(),
                "exact_young_state": donor_state.clone(),
                "batch_shuffled": donor_state.roll(1, dims=0),
                "reverse": 2.0 * receiver_terminal - donor_state,
            }
            for extra_loop in range(1, extra_loops + 1):
                continued_target = all_targets[
                    :, cfg.max_depth + 2 * extra_loop
                ]
                for condition in CONDITIONS:
                    states[condition] = apply_shared_stack(
                        model,
                        states[condition],
                        loop_index=cfg.max_loops + extra_loop - 1,
                    )
                    logits = logits_from_raw_state(
                        model, states[condition]
                    )
                    targets_by_kind = {
                        "continued_path": continued_target,
                        "donor_endpoint": donor_endpoint,
                        "original_endpoint": original_endpoint,
                    }
                    for target_kind, target in targets_by_kind.items():
                        metrics = (
                            metrics_excluding_targets(
                                logits,
                                target,
                                (
                                    (original_endpoint,)
                                    if extra_loop <= remaining_loops
                                    else (
                                        original_endpoint,
                                        donor_endpoint,
                                    )
                                ),
                            )
                            if target_kind == "continued_path"
                            else _masked_metrics(logits, target)
                        )
                        accumulate(
                            accumulators[
                                (
                                    donor_age,
                                    condition,
                                    extra_loop,
                                    target_kind,
                                )
                            ],
                            metrics,
                        )
    rows: list[dict[str, Any]] = []
    for key, bucket in accumulators.items():
        donor_age, condition, extra_loop, target_kind = key
        count = int(bucket["count"])
        rows.append(
            {
                "model": name,
                "donor_age": donor_age,
                "donor_current_path_position": 2 * donor_age,
                "expected_remaining_loops": expected_remaining_loops(
                    donor_age=donor_age,
                    endpoint_macro_loop=endpoint_macro_loop,
                ),
                "condition": condition,
                "extra_loop": extra_loop,
                "target_kind": target_kind,
                "accuracy": (
                    bucket["correct"] / count
                    if count
                    else float("nan")
                ),
                "probability": (
                    bucket["probability"] / count
                    if count
                    else float("nan")
                ),
                "margin": (
                    bucket["margin"] / count
                    if count
                    else float("nan")
                ),
                "valid_count": count,
            }
        )
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "age_reset_staircase.csv", rows)
    exact = [
        row
        for row in rows
        if row["condition"] == "exact_young_state"
        and row["target_kind"] == "continued_path"
    ]
    matrix = np.full((len(donor_ages), extra_loops), np.nan)
    for row in exact:
        matrix[int(row["donor_age"]), int(row["extra_loop"]) - 1] = float(
            row["accuracy"]
        )
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    axis.set_xticks(
        range(extra_loops),
        [f"+{loop}" for loop in range(1, extra_loops + 1)],
    )
    axis.set_yticks(
        range(len(donor_ages)),
        [f"age {age}" for age in donor_ages],
    )
    axis.set_xlabel("extra macro loop after one exact young-state reset")
    axis.set_ylabel("matched donor age")
    axis.set_title(f"{name}: causal telomere lifespan staircase")
    for row in range(matrix.shape[0]):
        boundary = expected_remaining_loops(
            donor_age=row,
            endpoint_macro_loop=endpoint_macro_loop,
        )
        axis.axvline(boundary - 0.5, color="#22d3ee", linewidth=1.2)
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="continued two-hop accuracy")
    figure.tight_layout()
    figure.savefig(run_dir / "age_reset_staircase.png", dpi=200)
    plt.close(figure)
    exact_curves = {
        str(age): [
            float(row["accuracy"])
            for row in sorted(
                [
                    row
                    for row in exact
                    if int(row["donor_age"]) == age
                ],
                key=lambda row: int(row["extra_loop"]),
            )
        ]
        for age in donor_ages
    }
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": asdict(cfg),
        "sample_size": batch_size * batches,
        "intervention": (
            "replace the terminal full recurrent state with a younger state "
            "from the same graph whose decoded current node is matched"
        ),
        "donor_age_to_expected_remaining_loops": {
            str(age): expected_remaining_loops(
                donor_age=age,
                endpoint_macro_loop=endpoint_macro_loop,
            )
            for age in donor_ages
        },
        "exact_young_state_continuation_accuracy": exact_curves,
        "controls": ["baseline", "batch_shuffled", "reverse"],
        "claim_ledger": {
            "lifespan_staircase_supported": all(
                all(value >= 0.80 for value in curve[: 4 - age])
                and (
                    len(curve) <= 4 - age
                    or curve[4 - age] <= 0.30
                )
                for age, curve in (
                    (int(age), values)
                    for age, values in exact_curves.items()
                )
            )
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = {}
    for name, checkpoint in args.run:
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            extra_loops=args.extra_loops,
            seed=args.seed,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
