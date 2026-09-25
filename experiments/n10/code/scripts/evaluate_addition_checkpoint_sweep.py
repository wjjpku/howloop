from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.paper_length_telomere import (
    PaperBatch,
    PaperLoopedTransformer,
    atomic_json_dump,
    generate_paper_batch,
    load_backbone,
    pick_device,
    write_csv,
)


METRIC_TOTAL_KEYS = (
    "examples",
    "strict_exact_successes",
    "actual_answer_exact_successes",
    "padding_exact_successes",
    "carry_successes",
    "actual_answer_correct_tokens",
    "actual_answer_tokens",
    "strict_answer_ce_sum",
    "actual_answer_ce_sum",
    "strict_answer_tokens",
    "strict_min_margin_sum",
    "actual_min_margin_sum",
    "strict_positive_margin_count",
    "actual_positive_margin_count",
    "low_order_frontier_sum",
)


def _empty_totals() -> dict[str, float]:
    return {key: 0.0 for key in METRIC_TOTAL_KEYS}


def _add_totals(target: dict[str, float], source: dict[str, float]) -> None:
    for key in METRIC_TOTAL_KEYS:
        target[key] += source[key]


@torch.inference_mode()
def addition_batch_metrics(
    logits: torch.Tensor,
    batch: PaperBatch,
    *,
    logical_length: int,
) -> dict[str, float]:
    """Return additive metrics while separating answer digits from PAD/EOS.

    The released task marks every position from the answer start through the
    fixed sequence end as supervised.  The first ``logical_length + 1`` of
    these positions are the binary sum; the remaining four positions are the
    PAD/EOS token.  Both views are useful and must not be conflated.
    """

    predictions = logits.argmax(dim=-1)
    answer_start = 2 * logical_length + 1
    answer_end = answer_start + logical_length + 1
    actual_slice = slice(answer_start, answer_end)
    padding_slice = slice(answer_end, logits.shape[1])

    strict_correct_tokens = predictions.eq(batch.targets) | ~batch.answer_mask
    strict_exact = strict_correct_tokens.all(dim=1)

    actual_correct = predictions[:, actual_slice].eq(batch.targets[:, actual_slice])
    actual_exact = actual_correct.all(dim=1)
    if answer_end < logits.shape[1]:
        padding_exact = predictions[:, padding_slice].eq(
            batch.targets[:, padding_slice]
        ).all(dim=1)
    else:
        padding_exact = torch.ones_like(actual_exact)

    # Binary addition is computed from right to left.  This counts the number
    # of consecutive correct result digits starting at the least-significant
    # (rightmost) result digit and includes the final carry only if every less
    # significant digit before it is already correct.
    low_order_frontier = torch.cumprod(
        actual_correct.flip(dims=(1,)).to(torch.int64), dim=1
    ).sum(dim=1)

    correct_logits = logits.gather(-1, batch.targets.unsqueeze(-1)).squeeze(-1)
    competing = logits.clone()
    competing.scatter_(-1, batch.targets.unsqueeze(-1), -torch.inf)
    token_margins = correct_logits - competing.max(dim=-1).values
    strict_min_margin = token_margins.masked_fill(
        ~batch.answer_mask, torch.inf
    ).min(dim=1).values
    actual_min_margin = token_margins[:, actual_slice].min(dim=1).values

    strict_logits = logits[batch.answer_mask]
    strict_targets = batch.targets[batch.answer_mask]
    actual_logits = logits[:, actual_slice].reshape(-1, logits.shape[-1])
    actual_targets = batch.targets[:, actual_slice].reshape(-1)

    return {
        "examples": float(logits.shape[0]),
        "strict_exact_successes": float(strict_exact.sum()),
        "actual_answer_exact_successes": float(actual_exact.sum()),
        "padding_exact_successes": float(padding_exact.sum()),
        "carry_successes": float(actual_correct[:, 0].sum()),
        "actual_answer_correct_tokens": float(actual_correct.sum()),
        "actual_answer_tokens": float(actual_correct.numel()),
        "strict_answer_ce_sum": float(
            F.cross_entropy(strict_logits, strict_targets, reduction="sum")
        ),
        "actual_answer_ce_sum": float(
            F.cross_entropy(actual_logits, actual_targets, reduction="sum")
        ),
        "strict_answer_tokens": float(strict_targets.numel()),
        "strict_min_margin_sum": float(strict_min_margin.sum()),
        "actual_min_margin_sum": float(actual_min_margin.sum()),
        "strict_positive_margin_count": float((strict_min_margin > 0).sum()),
        "actual_positive_margin_count": float((actual_min_margin > 0).sum()),
        "low_order_frontier_sum": float(low_order_frontier.sum()),
    }


def _row_from_totals(
    *,
    logical_length: int,
    target_step: int,
    step: int,
    totals: dict[str, float],
    batch_size: int,
    elapsed_seconds: float,
    peak_cuda_memory_reserved_gib: float,
) -> dict[str, Any]:
    examples = totals["examples"]
    actual_digit_count = logical_length + 1
    return {
        "length": logical_length,
        "target_step": target_step,
        "step": step,
        "step_offset": step - target_step,
        "examples": int(examples),
        "strict_exact_successes": int(totals["strict_exact_successes"]),
        "strict_exact_match": totals["strict_exact_successes"] / examples,
        "actual_answer_exact_successes": int(
            totals["actual_answer_exact_successes"]
        ),
        "actual_answer_exact_match": (
            totals["actual_answer_exact_successes"] / examples
        ),
        "padding_exact_match": totals["padding_exact_successes"] / examples,
        "carry_accuracy": totals["carry_successes"] / examples,
        "actual_answer_token_accuracy": (
            totals["actual_answer_correct_tokens"]
            / totals["actual_answer_tokens"]
        ),
        "strict_answer_cross_entropy": (
            totals["strict_answer_ce_sum"] / totals["strict_answer_tokens"]
        ),
        "actual_answer_cross_entropy": (
            totals["actual_answer_ce_sum"] / totals["actual_answer_tokens"]
        ),
        "mean_strict_sequence_min_margin": (
            totals["strict_min_margin_sum"] / examples
        ),
        "mean_actual_sequence_min_margin": (
            totals["actual_min_margin_sum"] / examples
        ),
        "strict_positive_margin_fraction": (
            totals["strict_positive_margin_count"] / examples
        ),
        "actual_positive_margin_fraction": (
            totals["actual_positive_margin_count"] / examples
        ),
        "mean_correct_low_order_digits": (
            totals["low_order_frontier_sum"] / examples
        ),
        "mean_correct_low_order_fraction": (
            totals["low_order_frontier_sum"] / (examples * actual_digit_count)
        ),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_reserved_gib": peak_cuda_memory_reserved_gib,
    }


def _selected_steps(
    *,
    target_step: int,
    offsets: Iterable[int],
    keep_full_trajectory: bool,
) -> tuple[int, ...]:
    selected = {
        target_step + int(offset)
        for offset in offsets
        if target_step + int(offset) >= 1
    }
    if keep_full_trajectory:
        selected.update(range(1, max(selected) + 1))
    return tuple(sorted(selected))


@torch.inference_mode()
def evaluate_length(
    *,
    model: PaperLoopedTransformer,
    spec: Any,
    logical_length: int,
    examples: int,
    max_batch_size: int,
    token_budget: int,
    offsets: Sequence[int],
    full_trajectory_lengths: set[int],
    seed: int,
    device: torch.device,
    evaluation_step_offset: int | None = None,
) -> list[dict[str, Any]]:
    target_step = logical_length + int(
        spec.step_offset
        if evaluation_step_offset is None
        else evaluation_step_offset
    )
    selected_steps = _selected_steps(
        target_step=target_step,
        offsets=offsets,
        keep_full_trajectory=logical_length in full_trajectory_lengths,
    )
    maximum_step = max(selected_steps)
    sequence_length = spec.sequence_length(logical_length)
    batch_size = min(
        examples,
        max_batch_size,
        max(1, token_budget // sequence_length),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * logical_length)
    totals = {step: _empty_totals() for step in selected_steps}
    elapsed = defaultdict(float)
    remaining = examples
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while remaining:
        current_batch_size = min(batch_size, remaining)
        batch = generate_paper_batch(
            spec,
            batch_size=current_batch_size,
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=generator,
        ).to(device)
        for step, state in enumerate(
            model.iter_states(batch.inputs, steps=maximum_step), start=1
        ):
            if step not in totals:
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.monotonic()
            logits = model.decode(state).float()
            metrics = addition_batch_metrics(
                logits, batch, logical_length=logical_length
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed[step] += time.monotonic() - started
            _add_totals(totals[step], metrics)
        remaining -= current_batch_size

    peak_gib = (
        float(torch.cuda.max_memory_reserved(device) / 1024**3)
        if device.type == "cuda"
        else 0.0
    )
    return [
        _row_from_totals(
            logical_length=logical_length,
            target_step=target_step,
            step=step,
            totals=totals[step],
            batch_size=batch_size,
            elapsed_seconds=elapsed[step],
            peak_cuda_memory_reserved_gib=peak_gib,
        )
        for step in selected_steps
    ]


def _contiguous_horizon(
    length_to_accuracy: dict[int, float], *, threshold: float
) -> int | None:
    if not length_to_accuracy:
        return None
    start = min(length_to_accuracy)
    horizon = start - 1
    for length in range(start, max(length_to_accuracy) + 1):
        if length_to_accuracy.get(length, -1.0) < threshold:
            break
        horizon = length
    return horizon if horizon >= start else None


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[int(row["length"])].append(row)
    length_summaries: dict[str, Any] = {}
    target_strict: dict[int, float] = {}
    target_actual: dict[int, float] = {}
    nearby_strict: dict[int, float] = {}
    nearby_actual: dict[int, float] = {}
    for length, selected in sorted(by_length.items()):
        target = next(row for row in selected if int(row["step_offset"]) == 0)
        nearby = [row for row in selected if -2 <= int(row["step_offset"]) <= 2]
        best_strict = max(nearby, key=lambda row: float(row["strict_exact_match"]))
        best_actual = max(
            nearby, key=lambda row: float(row["actual_answer_exact_match"])
        )
        target_strict[length] = float(target["strict_exact_match"])
        target_actual[length] = float(target["actual_answer_exact_match"])
        nearby_strict[length] = float(best_strict["strict_exact_match"])
        nearby_actual[length] = float(best_actual["actual_answer_exact_match"])
        length_summaries[str(length)] = {
            "target": target,
            "nearby_window": [-2, 2],
            "nearby_best_strict_step": int(best_strict["step"]),
            "nearby_best_strict_offset": int(best_strict["step_offset"]),
            "nearby_best_strict_exact_match": float(
                best_strict["strict_exact_match"]
            ),
            "nearby_best_actual_step": int(best_actual["step"]),
            "nearby_best_actual_offset": int(best_actual["step_offset"]),
            "nearby_best_actual_exact_match": float(
                best_actual["actual_answer_exact_match"]
            ),
        }
    horizons = {}
    for threshold in (0.90, 0.95, 0.98):
        key = f"{threshold:.2f}"
        horizons[key] = {
            "target_strict": _contiguous_horizon(target_strict, threshold=threshold),
            "target_actual": _contiguous_horizon(target_actual, threshold=threshold),
            "nearby_strict": _contiguous_horizon(nearby_strict, threshold=threshold),
            "nearby_actual": _contiguous_horizon(nearby_actual, threshold=threshold),
        }
    return {"lengths": length_summaries, "contiguous_horizons": horizons}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Addition checkpoint sweep with registered-step, nearby-loop, "
            "margin, carry, and digit-frontier metrics."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--step-offsets", type=int, nargs="+", default=(-2, -1, 0, 1, 2))
    parser.add_argument(
        "--evaluation-step-offset",
        type=int,
        help=(
            "evaluate at T_eval(n)=n+offset instead of the task's registered "
            "T(n)=n+1; useful for fixed-loop baselines trained at n=10"
        ),
    )
    parser.add_argument(
        "--full-trajectory-lengths", type=int, nargs="*", default=(10, 15, 20)
    )
    parser.add_argument("--seed", type=int, default=291001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.examples < 1 or args.max_batch_size < 1 or args.token_budget < 1:
        raise ValueError("evaluation sizes must be positive")
    lengths = tuple(sorted(set(int(length) for length in args.lengths)))
    if not lengths or min(lengths) < 1:
        raise ValueError("logical lengths must be positive")
    if 0 not in args.step_offsets:
        raise ValueError("--step-offsets must include the registered target offset 0")

    device = pick_device(args.device)
    model, spec, payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError(f"expected Addition checkpoint, got {spec.name!r}")
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    full_trajectory_lengths = set(int(value) for value in args.full_trajectory_lengths)
    for length in lengths:
        length_rows = evaluate_length(
            model=model,
            spec=spec,
            logical_length=length,
            examples=args.examples,
            max_batch_size=args.max_batch_size,
            token_budget=args.token_budget,
            offsets=args.step_offsets,
            full_trajectory_lengths=full_trajectory_lengths,
            seed=args.seed,
            device=device,
            evaluation_step_offset=args.evaluation_step_offset,
        )
        rows.extend(length_rows)
        write_csv(args.out_dir / "trajectory.csv", rows)
        print(
            json.dumps(
                {
                    "length": length,
                    "checkpoint_step": int(payload["step"]),
                    "target": next(
                        row for row in length_rows if row["step_offset"] == 0
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    trained_loop_count = int(
        payload.get("training_fixed_loop_count")
        or (
            int(payload.get("training_fixed_logical_length") or 10)
            + int(spec.step_offset)
        )
    )
    summary = {
        "status": "complete",
        "task": "addition",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "backbone_seed": int(payload.get("seed", -1)),
        "model": payload.get("model", payload.get("config")),
        "loss_placement": payload.get("loss_placement"),
        "training_fixed_logical_length": payload.get(
            "training_fixed_logical_length"
        ),
        "trained_loop_count": trained_loop_count,
        "shared_physical_block_layers": model.config.block_layers,
        "effective_training_depth": (
            trained_loop_count * model.config.block_layers
        ),
        "evaluation_seed": args.seed,
        "examples_per_length": args.examples,
        "lengths": list(lengths),
        "step_offsets": list(args.step_offsets),
        "evaluation_step_rule": (
            f"T_eval(n)=n+{int(spec.step_offset)}"
            if args.evaluation_step_offset is None
            else f"T_eval(n)=n{int(args.evaluation_step_offset):+d}"
        ),
        "full_trajectory_lengths": sorted(full_trajectory_lengths),
        "metric_semantics": {
            "strict_exact_match": (
                "paper answer-region EM including four trailing PAD/EOS positions"
            ),
            "actual_answer_exact_match": (
                "EM on only the n+1 binary sum digits, excluding trailing PAD/EOS"
            ),
            "mean_correct_low_order_digits": (
                "consecutive correct result digits from least to most significant"
            ),
        },
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    atomic_json_dump(summary, args.out_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
