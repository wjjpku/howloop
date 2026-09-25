"""Screen natural D8L6 interface states for one-hop and two-hop behavior.

This is an oracle-availability screen, not controller training.  For every
natural age and aligned path position, the same-graph/same-current state is
fed through one frozen shared loop.  Pre-F and shuffled-donor controls reject
answers already written into the donor or graph/content mismatch shortcuts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_jump_controller import JumpMode, _matched_target_state
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_counts(logits: torch.Tensor, target: torch.Tensor, endpoint: torch.Tensor) -> dict[str, int]:
    prediction = logits.argmax(dim=-1)
    strict = target.ne(endpoint)
    return {
        "all_correct": int(prediction.eq(target).sum()),
        "all_count": int(target.numel()),
        "strict_correct": int(prediction[strict].eq(target[strict]).sum()),
        "strict_count": int(strict.sum()),
    }


def add(receiver: dict[str, float], prefix: str, counts: dict[str, int]) -> None:
    for key, value in counts.items():
        receiver[f"{prefix}_{key}"] += value


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def finalize(rows: dict[tuple[int, int], dict[str, float]]) -> list[dict[str, Any]]:
    output = []
    for (age, path_position), counts in sorted(rows.items()):
        row: dict[str, Any] = {"reference_age": age, "reference_path_position": path_position}
        row.update({key: int(value) for key, value in counts.items()})
        for prefix in ("pre_endpoint", "pre_one", "pre_two", "post_endpoint", "post_one", "post_two", "shuffled_endpoint", "shuffled_one", "shuffled_two"):
            row[f"{prefix}_accuracy"] = ratio(counts[f"{prefix}_all_correct"], counts[f"{prefix}_all_count"])
            row[f"{prefix}_strict_accuracy"] = ratio(counts[f"{prefix}_strict_correct"], counts[f"{prefix}_strict_count"])
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_candidate(
    rows: list[dict[str, Any]],
    target: str,
    *,
    source_endpoint_threshold: float,
) -> dict[str, Any]:
    """Select an interface that still denotes the original endpoint pre-F.

    Without this source constraint, a donor that already denotes the one-hop
    target can make an ordinary one-step executor look like a two-hop mode.
    """

    eligible = [
        row
        for row in rows
        if row["pre_endpoint_accuracy"] >= source_endpoint_threshold
    ]
    candidates = eligible if eligible else rows
    selected = max(
        candidates,
        key=lambda row: (
            row[f"post_{target}_strict_accuracy"],
            row["pre_endpoint_accuracy"],
            -row[f"pre_{target}_strict_accuracy"],
        ),
    )
    selected = dict(selected)
    selected["source_endpoint_eligible"] = bool(selected in eligible)
    return selected


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    ages = sorted({int(row["reference_age"]) for row in rows})
    positions = sorted({int(row["reference_path_position"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.75), constrained_layout=True)
    for axis, target, title in zip(axes, ("one", "two"), ("One-hop interface", "Two-hop interface")):
        matrix = np.full((len(ages), len(positions)), np.nan)
        for row in rows:
            matrix[ages.index(int(row["reference_age"])), positions.index(int(row["reference_path_position"]))] = row[f"post_{target}_strict_accuracy"]
        image = axis.imshow(matrix, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(positions)), positions)
        axis.set_yticks(range(len(ages)), ages)
        axis.set_xlabel("Aligned path position")
        axis.set_ylabel("Natural interface age")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, label="Strict post-F accuracy")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026080904)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-path-position", type=int, default=8)
    parser.add_argument("--positive-threshold", type=float, default=0.90)
    parser.add_argument("--control-threshold", type=float, default=0.20)
    parser.add_argument("--source-endpoint-threshold", type=float, default=0.90)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.06)
    return parser.parse_args(argv)


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch-size")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    if cfg.max_depth != 8 or cfg.max_loops != 6 or cfg.n_layers != 2:
        raise ValueError("screen requires D8L6 with two shared physical blocks")
    set_seed(args.seed)
    accumulators = defaultdict(lambda: defaultdict(float))
    natural_correct = torch.zeros(cfg.max_loops + 1, cfg.max_depth + 3, device=device)
    natural_count = 0
    for _ in range(args.examples // args.batch_size):
        tokens, targets, successors, start = fixed_depth_batch(
            cfg, args.batch_size, device, path_positions=cfg.max_depth + 2
        )
        endpoint, one, two = targets[:, 7], targets[:, 8], targets[:, 9]
        all_targets = torch.cat((start[:, None], targets), dim=1)
        natural_states = cache_states_with_initial(model, tokens, loops=cfg.max_loops)
        for age, state in enumerate(natural_states):
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            for position in range(all_targets.shape[1]):
                natural_correct[age, position] += prediction.eq(all_targets[:, position]).sum()
        natural_count += args.batch_size

        for age in range(cfg.max_loops + 1):
            for path_position in range(args.max_path_position + 1):
                mode = JumpMode(
                    name="screen",
                    reference_age=age,
                    reference_path_before=path_position,
                    programmed_jump=0,
                )
                donor = _matched_target_state(
                    model=model, cfg=cfg, successors=successors, start=start, mode=mode
                )
                pre_logits = logits_from_raw_state(model, donor)
                post_logits = run_one_loop(model, donor, loop_index=cfg.max_loops).logits
                shuffled_logits = run_one_loop(
                    model, donor.roll(1, dims=0), loop_index=cfg.max_loops
                ).logits
                bucket = accumulators[(age, path_position)]
                for target_name, target in (("endpoint", endpoint), ("one", one), ("two", two)):
                    add(bucket, f"pre_{target_name}", metric_counts(pre_logits, target, endpoint))
                    add(bucket, f"post_{target_name}", metric_counts(post_logits, target, endpoint))
                    add(bucket, f"shuffled_{target_name}", metric_counts(shuffled_logits, target, endpoint))

    rows = finalize(accumulators)
    one_best = best_candidate(
        rows,
        "one",
        source_endpoint_threshold=args.source_endpoint_threshold,
    )
    two_best = best_candidate(
        rows,
        "two",
        source_endpoint_threshold=args.source_endpoint_threshold,
    )
    decisions = {}
    for target, candidate in (("one", one_best), ("two", two_best)):
        decisions[f"{target}_hop_interface_available"] = bool(
            candidate["source_endpoint_eligible"]
            and candidate["pre_endpoint_accuracy"] >= args.source_endpoint_threshold
            and candidate[f"post_{target}_strict_accuracy"] >= args.positive_threshold
            and candidate[f"pre_{target}_strict_accuracy"] <= args.control_threshold
            and candidate[f"shuffled_{target}_strict_accuracy"] <= args.control_threshold
        )
    natural = []
    accuracy = natural_correct / natural_count
    for age in range(cfg.max_loops + 1):
        best_position = int(accuracy[age].argmax())
        natural.append(
            {
                "age": age,
                "best_path_position": best_position,
                "best_accuracy": float(accuracy[age, best_position]),
                "all_position_accuracies": accuracy[age].detach().cpu().tolist(),
            }
        )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": checkpoint_payload.get("loss_mode", "final_only"),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "examples": args.examples,
        "seed": args.seed,
        "natural_trajectory": natural,
        "best_one_hop_candidate": one_best,
        "best_two_hop_candidate": two_best,
        "decisions": decisions,
        "thresholds": {
            "post_F_positive": args.positive_threshold,
            "pre_F_and_shuffled_max": args.control_threshold,
            "pre_F_endpoint_min": args.source_endpoint_threshold,
        },
        "interpretation_boundary": "Availability of a same-graph oracle interface that still reads as the original endpoint before F is necessary before controller fitting; it is not evidence that a learned controller can reach that interface.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "interface_oracle_grid.csv", rows)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot(rows, args.out_dir / "d8l6_interface_oracle_screen.pdf")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run_experiment(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
