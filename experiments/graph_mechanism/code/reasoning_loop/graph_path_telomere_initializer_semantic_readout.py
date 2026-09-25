from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def permutation_orbit(
    successors: torch.Tensor,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return f^0(origin)..f^(N-1)(origin) and the origin-cycle length."""

    if successors.ndim != 2 or origin.shape != successors.shape[:1]:
        raise ValueError("expected successors [batch, nodes] and origin [batch]")
    node_count = successors.shape[1]
    current = origin
    orbit = []
    lengths = torch.zeros_like(origin)
    for step in range(node_count):
        orbit.append(current)
        current = successors.gather(1, current[:, None]).squeeze(1)
        returned = current.eq(origin) & lengths.eq(0)
        lengths = torch.where(
            returned,
            torch.full_like(lengths, step + 1),
            lengths,
        )
    if lengths.eq(0).any():
        raise ValueError("successors do not define permutation cycles")
    return torch.stack(orbit, dim=1), lengths


def semantic_offsets(
    predictions: torch.Tensor,
    orbit: torch.Tensor,
    orbit_lengths: torch.Tensor,
) -> torch.Tensor:
    """Map predicted nodes to their minimal forward offset, or -1 off orbit."""

    if predictions.shape != orbit.shape[:1]:
        raise ValueError("predictions and orbit batch dimensions differ")
    offsets = torch.full_like(predictions, -1)
    for offset in range(orbit.shape[1]):
        valid = offset < orbit_lengths
        match = predictions.eq(orbit[:, offset]) & valid & offsets.eq(-1)
        offsets = torch.where(
            match,
            torch.full_like(offsets, offset),
            offsets,
        )
    return offsets


def _load_initializer(
    artifact: Path,
    *,
    device: torch.device,
) -> tuple[VectorAffine, tuple[int, ...]]:
    payload = torch.load(artifact, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_learned_initializer":
        raise ValueError("unexpected initializer artifact kind")
    item = payload["map"]
    weight = item["weight"].to(device=device, dtype=torch.float32)
    initializer = VectorAffine(
        weight=weight,
        bias=item["bias"].to(device=device, dtype=torch.float32),
        update_rank=int(item["rank"]),
        fit_dimension=int(weight.shape[0]),
        retained_fit_energy=float(item["retained_fit_energy"]),
    )
    return initializer, tuple(int(value) for value in payload["positions"])


def _aggregate(
    metric_sums: dict[tuple[str, int], dict[str, float]],
    offset_counts: dict[tuple[str, int, int, int], int],
    offset_mass_sums: dict[tuple[str, int, int, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cycle_rows: list[dict[str, Any]] = []
    for (condition, cycle), values in sorted(metric_sums.items()):
        total = values["count"]
        full = values["full_cycle_count"]
        cycle_rows.append(
            {
                "condition": condition,
                "cycle": cycle,
                "sample_count": int(total),
                "full_cycle_count": int(full),
                "target_accuracy_all": values["target_correct"] / total,
                "target_accuracy_full_cycle": (
                    values["target_correct_full"] / full if full else float("nan")
                ),
                "off_orbit_fraction": values["off_orbit"] / total,
                "top1_agreement_with_exact_h2": (
                    values["agreement_with_exact_h2"] / total
                    if condition != "exact_h2_no_control"
                    else 1.0
                ),
                "mean_logit_cosine_with_exact_h2": (
                    values["logit_cosine_with_exact_h2"] / total
                    if condition != "exact_h2_no_control"
                    else 1.0
                ),
            }
        )

    offset_rows: list[dict[str, Any]] = []
    keys = sorted(offset_counts)
    group_totals: dict[tuple[str, int, int], int] = defaultdict(int)
    for condition, cycle, orbit_length, offset in keys:
        group_totals[(condition, cycle, orbit_length)] += offset_counts[
            (condition, cycle, orbit_length, offset)
        ]
    for condition, cycle, orbit_length, offset in keys:
        key = (condition, cycle, orbit_length, offset)
        count = offset_counts[key]
        total = group_totals[(condition, cycle, orbit_length)]
        offset_rows.append(
            {
                "condition": condition,
                "cycle": cycle,
                "orbit_length": orbit_length,
                "semantic_offset": offset,
                "top1_count": count,
                "top1_fraction": count / total,
                "mean_probability_mass": offset_mass_sums.get(key, 0.0) / total,
            }
        )
    return cycle_rows, offset_rows


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    initializer_artifact: Path,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)

    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("experiment is fixed to the D8L8 two-block model")
    phase_summary = json.loads(phase_summary_path.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_summary["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    initializer, positions = _load_initializer(
        initializer_artifact,
        device=device,
    )
    expected_positions = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if positions != expected_positions:
        raise ValueError("initializer positions do not match expected interface")

    conditions = (
        "raw_h8_no_control",
        "learned_init_only",
        "exact_h2_no_control",
    )
    metric_sums: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    offset_counts: dict[tuple[str, int, int, int], int] = defaultdict(int)
    offset_mass_sums: dict[tuple[str, int, int, int], float] = defaultdict(
        float
    )

    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * extra_loops,
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        orbit, orbit_lengths = permutation_orbit(successors, endpoint)
        full_cycle = orbit_lengths.eq(cfg.node_count)
        h8 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states = {
            "raw_h8_no_control": h8,
            "learned_init_only": h8.clone(),
            "exact_h2_no_control": h2,
        }

        for cycle in range(1, extra_loops + 1):
            target_index = cfg.max_depth + jump * cycle - 1
            target = path_targets[:, target_index]
            steps = {
                condition: run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, initializer)
                        if condition == "learned_init_only" and cycle == 1
                        else None
                    ),
                )
                for condition in conditions
            }
            exact_logits = steps["exact_h2_no_control"].logits.float()
            exact_predictions = exact_logits.argmax(dim=-1)

            for condition, step in steps.items():
                logits = step.logits.float()
                probabilities = logits.softmax(dim=-1)
                predictions = logits.argmax(dim=-1)
                offsets = semantic_offsets(
                    predictions,
                    orbit,
                    orbit_lengths,
                )
                key = (condition, cycle)
                values = metric_sums[key]
                values["count"] += batch_size
                values["full_cycle_count"] += int(full_cycle.sum())
                values["target_correct"] += int(predictions.eq(target).sum())
                values["target_correct_full"] += int(
                    (predictions.eq(target) & full_cycle).sum()
                )
                values["off_orbit"] += int(offsets.eq(-1).sum())
                if condition != "exact_h2_no_control":
                    values["agreement_with_exact_h2"] += int(
                        predictions.eq(exact_predictions).sum()
                    )
                    cosine = F.cosine_similarity(
                        logits,
                        exact_logits,
                        dim=-1,
                    )
                    values["logit_cosine_with_exact_h2"] += float(cosine.sum())

                for orbit_length in range(1, cfg.node_count + 1):
                    group = orbit_lengths.eq(orbit_length)
                    if not group.any():
                        continue
                    group_offsets = offsets[group]
                    for offset in range(-1, orbit_length):
                        count = int(group_offsets.eq(offset).sum())
                        offset_counts[
                            (condition, cycle, orbit_length, offset)
                        ] += count
                    for offset in range(orbit_length):
                        nodes = orbit[group, offset]
                        mass = probabilities[group].gather(
                            1,
                            nodes[:, None],
                        ).sum()
                        offset_mass_sums[
                            (condition, cycle, orbit_length, offset)
                        ] += float(mass)
            states = {
                condition: steps[condition].state for condition in conditions
            }

    cycle_rows, offset_rows = _aggregate(
        metric_sums,
        offset_counts,
        offset_mass_sums,
    )
    full_cycle_modes: dict[str, list[int]] = {}
    for condition in conditions:
        modes = []
        for cycle in range(1, extra_loops + 1):
            parts = [
                row
                for row in offset_rows
                if row["condition"] == condition
                and row["cycle"] == cycle
                and row["orbit_length"] == cfg.node_count
                and row["semantic_offset"] >= 0
            ]
            modes.append(
                int(max(parts, key=lambda row: row["top1_count"])[
                    "semantic_offset"
                ])
                if parts
                else -1
            )
        full_cycle_modes[condition] = modes

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "cycle_metrics.csv", cycle_rows)
    _write_csv(out_dir / "semantic_offset_rows.csv", offset_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "initializer_artifact": str(initializer_artifact),
        "intervention_timing": (
            "initializer applied once, at Block2 input of continuation cycle 1"
        ),
        "loss_placement": "no training; readout-only evaluation",
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": extra_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "graphs": batch_size * batches,
        "seed": seed,
        "jump_per_loop": jump,
        "full_cycle_modes": full_cycle_modes,
        "cycle_metrics": cycle_rows,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "cycle_metrics": "cycle_metrics.csv",
            "semantic_offsets": "semantic_offset_rows.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate initializer-only logits on the graph orbit, with exact-H2 "
            "and raw-H8 controls."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--initializer-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--extra-loops", type=int, default=8)
    parser.add_argument("--seed", type=int, default=131001)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        initializer_artifact=args.initializer_artifact,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        extra_loops=args.extra_loops,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
