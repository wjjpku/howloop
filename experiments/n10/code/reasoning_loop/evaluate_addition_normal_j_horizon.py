from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F_torch

from reasoning_loop.continue_paper_length_controller import (
    validate_task_payload_compatibility,
)
from reasoning_loop.paper_length_telomere import (
    atomic_json_dump,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
    write_csv,
)
from reasoning_loop.robust_age_path_controller import (
    F,
    alternating_age_path,
    execute_actions,
)


THRESHOLDS = (0.99, 0.95, 0.90, 0.50)


def per_example_exact(logits: torch.Tensor, batch: Any) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    token_correct = predictions.eq(batch.targets) | ~batch.answer_mask
    return token_correct.all(dim=1)


def contiguous_horizon(
    rows: Sequence[dict[str, Any]], *, threshold: float
) -> int:
    """Last consecutive evaluated length whose accuracy meets a threshold."""
    ordered = sorted(rows, key=lambda row: int(row["logical_length"]))
    if not ordered or int(ordered[0]["logical_length"]) != 1:
        raise ValueError("horizon rows must begin at logical length one")
    expected = 1
    horizon = 0
    for row in ordered:
        length = int(row["logical_length"])
        if length != expected:
            raise ValueError("horizon rows must cover consecutive lengths")
        if float(row["exact_match"]) < threshold:
            break
        horizon = length
        expected += 1
    return horizon


def exact_mcnemar_pvalue(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    log_probability = -discordant * math.log(2.0)
    log_terms: list[float] = []
    for value in range(tail + 1):
        log_terms.append(log_probability)
        if value < tail:
            log_probability += math.log(discordant - value) - math.log(value + 1)
    maximum = max(log_terms)
    probability = math.exp(maximum) * math.fsum(
        math.exp(term - maximum) for term in log_terms
    )
    return min(1.0, 2.0 * probability)


def mean_accuracy(rows: Sequence[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["exact_match"]) for row in rows) / len(rows)


def _parse_controller(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("controller must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or label == "raw":
        raise argparse.ArgumentTypeError("controller label must be nonempty and not raw")
    return label, Path(raw_path)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("normal-J horizon evaluation is frozen to Addition")
    model.eval()

    controllers: OrderedDict[str, torch.nn.Module] = OrderedDict()
    controller_metadata: dict[str, dict[str, Any]] = {}
    for label, artifact in args.controller:
        if label in controllers:
            raise ValueError(f"duplicate controller label: {label}")
        controller, payload = load_controller(artifact, device=device)
        validate_task_payload_compatibility(payload["task"], backbone_payload["task"])
        if not payload.get("controller_post_final_j", False):
            raise ValueError(f"{label} does not use the canonical post-final J interface")
        controllers[label] = controller.eval()
        controller_metadata[label] = {
            "artifact": str(artifact),
            "training_seed": payload.get("seed"),
            "total_optimizer_updates": payload.get("training_budget", {}).get(
                "total_optimizer_updates"
            ),
            "curriculum": payload.get("controller_curriculum"),
            "age_path_training": payload.get("age_path_training"),
        }
    if args.reference not in controllers:
        raise ValueError("reference label is not among the controllers")

    labels = ["raw", *controllers.keys()]
    aggregate_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for logical_length in args.lengths:
        forward_steps = logical_length + spec.step_offset
        normal_path = alternating_age_path(forward_steps)
        totals = {
            label: {"correct": 0, "examples": 0, "ce_sum": 0.0, "tokens": 0}
            for label in labels
        }
        paired = {
            label: {"candidate_only": 0, "reference_only": 0, "both": 0, "neither": 0}
            for label in controllers
            if label != args.reference
        }
        for eval_seed in args.seeds:
            seed_totals = {
                label: {"correct": 0, "examples": 0, "ce_sum": 0.0, "tokens": 0}
                for label in labels
            }
            for batch_index in range(args.batches_per_seed):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    eval_seed + 1_000_003 * logical_length + batch_index
                )
                batch = generate_paper_batch(
                    spec,
                    batch_size=args.batch_size,
                    min_length=logical_length,
                    max_length=logical_length,
                    fixed_length=logical_length,
                    generator=generator,
                ).to(device)
                logits_by_label: dict[str, torch.Tensor] = {
                    "raw": execute_actions(
                        model=model,
                        controller=None,
                        batch=batch,
                        actions=(F,) * forward_steps,
                    )
                }
                for label, controller in controllers.items():
                    logits_by_label[label] = execute_actions(
                        model=model,
                        controller=controller,
                        batch=batch,
                        actions=normal_path.actions,
                    )
                exact_by_label: dict[str, torch.Tensor] = {}
                answer_tokens = int(batch.answer_mask.sum())
                for label, logits in logits_by_label.items():
                    exact = per_example_exact(logits, batch)
                    exact_by_label[label] = exact
                    ce_sum = float(
                        F_torch.cross_entropy(
                            logits[batch.answer_mask],
                            batch.targets[batch.answer_mask],
                            reduction="sum",
                        )
                    )
                    for accumulator in (totals[label], seed_totals[label]):
                        accumulator["correct"] += int(exact.sum())
                        accumulator["examples"] += int(exact.numel())
                        accumulator["ce_sum"] += ce_sum
                        accumulator["tokens"] += answer_tokens
                reference_exact = exact_by_label[args.reference]
                for label, counts in paired.items():
                    candidate_exact = exact_by_label[label]
                    counts["candidate_only"] += int(
                        (candidate_exact & ~reference_exact).sum()
                    )
                    counts["reference_only"] += int(
                        (~candidate_exact & reference_exact).sum()
                    )
                    counts["both"] += int(
                        (candidate_exact & reference_exact).sum()
                    )
                    counts["neither"] += int(
                        (~candidate_exact & ~reference_exact).sum()
                    )
            for label, values in seed_totals.items():
                seed_rows.append(
                    {
                        "logical_length": logical_length,
                        "evaluation_seed": eval_seed,
                        "variant": label,
                        "correct": values["correct"],
                        "examples": values["examples"],
                        "exact_match": values["correct"] / values["examples"],
                        "answer_ce": values["ce_sum"] / values["tokens"],
                    }
                )
        for label, values in totals.items():
            relevant_seed_rows = [
                row
                for row in seed_rows
                if row["logical_length"] == logical_length
                and row["variant"] == label
            ]
            seed_accuracies = [float(row["exact_match"]) for row in relevant_seed_rows]
            mean = sum(seed_accuracies) / len(seed_accuracies)
            variance = sum((value - mean) ** 2 for value in seed_accuracies) / max(
                1, len(seed_accuracies) - 1
            )
            aggregate_rows.append(
                {
                    "logical_length": logical_length,
                    "variant": label,
                    "correct": values["correct"],
                    "examples": values["examples"],
                    "exact_match": values["correct"] / values["examples"],
                    "answer_ce": values["ce_sum"] / values["tokens"],
                    "seed_mean_exact_match": mean,
                    "seed_sd_exact_match": math.sqrt(variance),
                    "evaluation_seeds": len(args.seeds),
                }
            )
        for label, counts in paired.items():
            paired_rows.append(
                {
                    "logical_length": logical_length,
                    "candidate": label,
                    "reference": args.reference,
                    **counts,
                    "net_correct_delta": counts["candidate_only"]
                    - counts["reference_only"],
                    "exact_mcnemar_pvalue": exact_mcnemar_pvalue(
                        counts["candidate_only"], counts["reference_only"]
                    ),
                }
            )
        current = {
            row["variant"]: round(float(row["exact_match"]), 6)
            for row in aggregate_rows
            if row["logical_length"] == logical_length
        }
        print(
            json.dumps(
                {"event": "length_complete", "logical_length": logical_length, "accuracy": current},
                sort_keys=True,
            ),
            flush=True,
        )

    metrics: dict[str, Any] = {}
    for label in labels:
        rows = [row for row in aggregate_rows if row["variant"] == label]
        ood_21_40 = [row for row in rows if 21 <= row["logical_length"] <= 40]
        ood_21_50 = [row for row in rows if 21 <= row["logical_length"] <= 50]
        metrics[label] = {
            "contiguous_accuracy_horizons": {
                str(threshold): contiguous_horizon(rows, threshold=threshold)
                for threshold in THRESHOLDS
            },
            "mean_length_accuracy_21_40": mean_accuracy(ood_21_40),
            "mean_length_accuracy_21_50": mean_accuracy(ood_21_50),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "normal_j_by_length.csv", aggregate_rows)
    write_csv(args.out_dir / "normal_j_by_seed.csv", seed_rows)
    write_csv(args.out_dir / "paired_by_length.csv", paired_rows)
    summary = {
        "status": "complete",
        "task": asdict(spec),
        "normal_usage": "canonical (FJ)^T from h0 with T(n)=n+1",
        "checkpoint": str(args.checkpoint),
        "controllers": controller_metadata,
        "reference": args.reference,
        "lengths": list(args.lengths),
        "evaluation_seeds": list(args.seeds),
        "batch_size": args.batch_size,
        "batches_per_seed": args.batches_per_seed,
        "examples_per_length_variant": (
            args.batch_size * args.batches_per_seed * len(args.seeds)
        ),
        "selection_policy": "evaluation only; no checkpoint selected on these OOD results",
        "metrics": metrics,
        "peak_cuda_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_json_dump(summary, args.out_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired multi-seed horizon audit for canonical Addition (FJ)^T use."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--controller", type=_parse_controller, action="append", required=True
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=tuple(range(1, 51)))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=tuple(571_001 + 10_000 * index for index in range(10)),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if sorted(set(args.lengths)) != list(range(1, max(args.lengths) + 1)):
        parser.error("lengths must cover every integer from 1 through the maximum")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("evaluation seeds must be unique")
    if args.batch_size < 1 or args.batches_per_seed < 1:
        parser.error("batch counts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
