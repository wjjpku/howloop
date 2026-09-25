from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.paper_length_telomere import (
    ControllerView,
    DiagonalLowRankController,
    PaperBatch,
    PaperLoopedTransformer,
    atomic_json_dump,
    generate_paper_batch,
    generate_balanced_addition_final_carry_batch,
    load_backbone,
    load_controller,
    pick_device,
    write_csv,
)


@torch.inference_mode()
def endpoint_logits_streaming(
    *,
    model: PaperLoopedTransformer,
    batch: PaperBatch,
    steps: int,
    controller: torch.nn.Module | None,
    anchor_step: int | None,
    post_final_controller: bool,
    executor_off_after_anchor: bool = False,
) -> torch.Tensor:
    """Return final logits without retaining the recurrent trajectory."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if controller is not None and anchor_step is None:
        raise ValueError("controlled evaluation requires an anchor step")
    state = torch.zeros_like(model.read_in(batch.inputs))
    for step_index in range(1, steps + 1):
        embedded = model.input_embeddings(batch.inputs, step_index=step_index)
        controlled = (
            controller is not None
            and anchor_step is not None
            and step_index > anchor_step
        )
        if controlled:
            state = controller(state)
        if not (controlled and executor_off_after_anchor):
            state = model.recurrent_step(state, embedded)
    if controller is not None and post_final_controller:
        state = controller(state)
    return model.decode(state)


@torch.inference_mode()
def _batch_metrics(logits: torch.Tensor, batch: PaperBatch) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    answer_correct_tokens = predictions.eq(batch.targets) & batch.answer_mask
    correct_tokens = predictions.eq(batch.targets) | ~batch.answer_mask
    exact = correct_tokens.all(dim=1)

    correct_logits = logits.gather(-1, batch.targets.unsqueeze(-1)).squeeze(-1)
    competing = logits.clone()
    competing.scatter_(-1, batch.targets.unsqueeze(-1), -torch.inf)
    token_margins = correct_logits - competing.max(dim=-1).values
    sequence_min_margin = token_margins.masked_fill(
        ~batch.answer_mask, torch.inf
    ).min(dim=1).values
    answer_logits = logits[batch.answer_mask]
    answer_targets = batch.targets[batch.answer_mask]
    return {
        "examples": float(batch.inputs.shape[0]),
        "exact_successes": float(exact.sum()),
        "answer_tokens": float(answer_targets.numel()),
        "answer_correct_tokens": float(answer_correct_tokens.sum()),
        "answer_ce_sum": float(
            F.cross_entropy(answer_logits, answer_targets, reduction="sum")
        ),
        "sequence_min_margin_sum": float(sequence_min_margin.sum()),
        "sequence_positive_margin_count": float(
            (sequence_min_margin > 0).sum()
        ),
    }


def _empty_totals() -> dict[str, float]:
    return {
        "examples": 0.0,
        "exact_successes": 0.0,
        "answer_tokens": 0.0,
        "answer_correct_tokens": 0.0,
        "answer_ce_sum": 0.0,
        "sequence_min_margin_sum": 0.0,
        "sequence_positive_margin_count": 0.0,
        "elapsed_seconds": 0.0,
    }


def _tensor_digest_update(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _add_metrics(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] += value


@torch.inference_mode()
def evaluate_length(
    *,
    model: PaperLoopedTransformer,
    controller: torch.nn.Module | None,
    spec: Any,
    length: int,
    examples: int,
    max_batch_size: int,
    token_budget: int,
    anchor_step: int | None,
    post_final_controller: bool,
    seed: int,
    device: torch.device,
    addition_final_carry: bool = False,
    controller_variant: str = "full",
    executor_off_after_anchor: bool = False,
) -> list[dict[str, Any]]:
    if examples < 1 or max_batch_size < 1 or token_budget < 1:
        raise ValueError("evaluation sizes must be positive")
    sequence_length = spec.sequence_length(length)
    batch_size = min(
        examples,
        max_batch_size,
        max(1, token_budget // sequence_length),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * length)
    variants = (
        (("raw", None), (controller_variant, controller))
        if controller is not None
        else (("raw", None),)
    )
    totals = {variant: _empty_totals() for variant, _ in variants}
    input_digest = hashlib.sha256()
    remaining = examples
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target_steps = length + int(spec.step_offset)
    while remaining:
        current_batch_size = min(batch_size, remaining)
        if addition_final_carry:
            batch = generate_balanced_addition_final_carry_batch(
                spec,
                batch_size=current_batch_size,
                logical_length=length,
                generator=generator,
            ).to(device)
        else:
            batch = generate_paper_batch(
                spec,
                batch_size=current_batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
        _tensor_digest_update(input_digest, batch.inputs)
        _tensor_digest_update(input_digest, batch.targets)
        _tensor_digest_update(input_digest, batch.answer_mask)
        for variant, live_controller in variants:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.monotonic()
            logits = endpoint_logits_streaming(
                model=model,
                batch=batch,
                steps=target_steps,
                controller=live_controller,
                anchor_step=(anchor_step if live_controller is not None else None),
                post_final_controller=(
                    post_final_controller if live_controller is not None else False
                ),
                executor_off_after_anchor=(
                    executor_off_after_anchor
                    if live_controller is not None
                    else False
                ),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            totals[variant]["elapsed_seconds"] += time.monotonic() - started
            _add_metrics(totals[variant], _batch_metrics(logits.float(), batch))
        remaining -= current_batch_size

    peak_gib = (
        float(torch.cuda.max_memory_reserved(device) / 1024**3)
        if device.type == "cuda"
        else 0.0
    )
    rows: list[dict[str, Any]] = []
    for variant, _ in variants:
        values = totals[variant]
        rows.append(
            {
                "length": length,
                "target_steps": target_steps,
                "variant": variant,
                "examples": int(values["examples"]),
                "exact_successes": int(values["exact_successes"]),
                "exact_match": values["exact_successes"] / values["examples"],
                "answer_token_accuracy": (
                    values["answer_correct_tokens"] / values["answer_tokens"]
                ),
                "answer_cross_entropy": (
                    values["answer_ce_sum"] / values["answer_tokens"]
                ),
                "mean_sequence_min_margin": (
                    values["sequence_min_margin_sum"] / values["examples"]
                ),
                "positive_sequence_margin_fraction": (
                    values["sequence_positive_margin_count"]
                    / values["examples"]
                ),
                "batch_size": batch_size,
                "elapsed_seconds": values["elapsed_seconds"],
                "peak_cuda_memory_reserved_gib": peak_gib,
                "evaluation_input_sha256": input_digest.hexdigest(),
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming strict-target horizon sweep for registered paper tasks."
    )
    parser.add_argument(
        "--task",
        choices=("parity", "copy4", "addition", "sum_reverse"),
        default="parity",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--controller",
        type=Path,
        help="omit for a raw-backbone-only horizon evaluation",
    )
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=261001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="for Parity, reject checkpoints outside the corrected input-once protocol",
    )
    parser.add_argument(
        "--addition-final-carry",
        action="store_true",
        help="evaluate balanced accuracy on only the carry-chain terminal bit",
    )
    parser.add_argument(
        "--post-final-controller",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--controller-mode",
        choices=(
            "full", "no_AB", "D_only", "identity_D", "AB_only", "mean_D",
            "no_bias", "shuffle_D", "spectrum_matched_random_delta",
        ),
        default="full",
        help="evaluate a component view of the learned controller",
    )
    parser.add_argument(
        "--executor-off-after-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after the anchor, apply J but skip every frozen executor call",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(
        args.checkpoint, device=device, paper_mode=args.paper_mode
    )
    controller: torch.nn.Module | None = None
    controller_payload: dict[str, Any] | None = None
    if args.controller is not None:
        loaded_controller, controller_payload = load_controller(
            args.controller, device=device
        )
        controller = (
            loaded_controller
            if args.controller_mode == "full"
            else ControllerView(
                loaded_controller,
                mode=args.controller_mode,
                shuffle_seed=args.seed + 901,
            ).to(device).eval()
        )
    elif args.controller_mode != "full" or args.executor_off_after_anchor:
        raise ValueError(
            "controller modes and executor-off evaluation require --controller"
        )
    if spec.name != args.task:
        raise ValueError(
            f"checkpoint task {spec.name!r} does not match --task {args.task!r}"
        )
    if controller_payload is not None and (
        controller_payload["checkpoint"] != str(args.checkpoint)
    ):
        raise ValueError("controller belongs to a different backbone")
    post_final_controller = bool(
        controller_payload is not None
        and (
            bool(controller_payload.get("controller_post_final_j", False))
            if args.post_final_controller is None
            else bool(args.post_final_controller)
        )
    )
    anchor_step = (
        int(controller_payload["anchor_step"])
        if controller_payload is not None
        else None
    )
    controller_variant = args.controller_mode + (
        "_executor_off" if args.executor_off_after_anchor else ""
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for length in args.lengths:
        length_rows = evaluate_length(
            model=model,
            controller=controller,
            spec=spec,
            length=length,
            examples=args.examples,
            max_batch_size=args.max_batch_size,
            token_budget=args.token_budget,
            anchor_step=anchor_step,
            post_final_controller=post_final_controller,
            seed=args.seed,
            device=device,
            addition_final_carry=args.addition_final_carry,
            controller_variant=controller_variant,
            executor_off_after_anchor=args.executor_off_after_anchor,
        )
        rows.extend(length_rows)
        write_csv(args.out_dir / "horizon.csv", rows)
        print(json.dumps(length_rows, sort_keys=True), flush=True)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller) if args.controller is not None else None,
        "controller_anchor_step": anchor_step,
        "controller_post_final_j": post_final_controller,
        "controller_mode": args.controller_mode if controller is not None else None,
        "executor_off_after_anchor": args.executor_off_after_anchor,
        "task": spec.name,
        "paper_mode": bool(args.paper_mode),
        "target_loop_rule": (
            "T(n)=n"
            if int(spec.step_offset) == 0
            else f"T(n)=n+{int(spec.step_offset)}"
        ),
        "streaming_recurrence": True,
        "retained_intermediate_states": False,
        "lengths": list(args.lengths),
        "examples_per_length": args.examples,
        "seed": args.seed,
        "supervision_target": (
            "balanced_addition_final_carry"
            if args.addition_final_carry
            else "full_answer"
        ),
        "rows": rows,
    }
    atomic_json_dump(summary, args.out_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
