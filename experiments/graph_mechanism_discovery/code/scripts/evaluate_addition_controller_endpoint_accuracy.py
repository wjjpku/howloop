#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from scripts.compare_addition_readout_order_attention import (
        answer_positions_lsb_to_carry,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from compare_addition_readout_order_attention import (
        answer_positions_lsb_to_carry,
    )
from reasoning_loop.paper_length_telomere import (
    ControllerView,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)


class DenseControllerView(torch.nn.Module):
    """Component views for a row-vector dense affine controller.

    With ``J(h) = h @ W + b``, ``no_offdiag`` keeps only ``diag(W)`` and
    ``identity_D`` replaces the learned diagonal by one while preserving the
    learned off-diagonal residual and bias.
    """

    def __init__(self, controller: torch.nn.Module, *, mode: str) -> None:
        super().__init__()
        self.controller = controller
        self.mode = mode

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        weight = self.controller.weight
        bias = self.controller.bias
        diagonal = torch.diagonal(weight)
        if self.mode == "full":
            return live @ weight + bias
        if self.mode == "no_offdiag":
            return live * diagonal + bias
        if self.mode == "identity_D":
            off_diagonal = weight - torch.diag(diagonal)
            return live + live @ off_diagonal + bias
        raise ValueError(f"unsupported dense controller view: {self.mode}")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def endpoint_logits(
    *,
    model: torch.nn.Module,
    controller: torch.nn.Module | None,
    controller_start_step: int | None,
    executor_off: bool,
    inputs: torch.Tensor,
    target_step: int,
) -> torch.Tensor:
    state = None
    for state in model.iter_states(
        inputs,
        steps=target_step,
        controller=controller,
        controller_start_step=controller_start_step,
        executor_off_after_start=executor_off,
    ):
        pass
    if state is None:
        raise RuntimeError("empty recurrent trajectory")
    return model.decode(state).float()


def update_metrics(
    totals: dict[str, int],
    *,
    logits: torch.Tensor,
    batch: Any,
    semantic_positions: torch.Tensor,
    layout_positions: torch.Tensor,
) -> None:
    predictions = logits.argmax(dim=-1)
    supervised_correct = predictions.eq(batch.targets) | ~batch.answer_mask
    semantic_predictions = predictions.index_select(1, semantic_positions)
    semantic_targets = batch.targets.index_select(1, semantic_positions)
    semantic_correct = semantic_predictions.eq(semantic_targets)
    digit_correct = semantic_correct[:, :-1]
    carry_correct = semantic_correct[:, -1]
    layout_predictions = predictions.index_select(1, layout_positions)
    layout_targets = batch.targets.index_select(1, layout_positions)
    layout_correct = layout_predictions.eq(layout_targets)
    totals["examples"] += int(predictions.shape[0])
    totals["supervised_exact"] += int(supervised_correct.all(dim=1).sum())
    totals["digit_exact"] += int(digit_correct.all(dim=1).sum())
    totals["digit_correct"] += int(digit_correct.sum())
    totals["digit_tokens"] += int(digit_correct.numel())
    totals["carry_correct"] += int(carry_correct.sum())
    totals["semantic_exact"] += int(semantic_correct.all(dim=1).sum())
    totals["semantic_correct"] += int(semantic_correct.sum())
    totals["semantic_tokens"] += int(semantic_correct.numel())
    totals["layout_exact"] += int(layout_correct.all(dim=1).sum())
    totals["layout_correct"] += int(layout_correct.sum())
    totals["layout_tokens"] += int(layout_correct.numel())


@torch.inference_mode()
def evaluate_variant(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module | None,
    controller_start_step: int | None,
    executor_off: bool,
    logical_length: int,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * logical_length)
    target_step = logical_length + int(spec.step_offset)
    width = spec.addition_layout_width(logical_length)
    semantic_positions = torch.tensor(
        answer_positions_lsb_to_carry(spec, logical_length), device=device
    )
    answer_start = 2 * width + 1
    layout_positions = torch.arange(
        answer_start, answer_start + width + 1, device=device
    )
    totals = {
        "examples": 0,
        "supervised_exact": 0,
        "digit_exact": 0,
        "digit_correct": 0,
        "digit_tokens": 0,
        "carry_correct": 0,
        "semantic_exact": 0,
        "semantic_correct": 0,
        "semantic_tokens": 0,
        "layout_exact": 0,
        "layout_correct": 0,
        "layout_tokens": 0,
    }
    for _ in range(batches):
        batch = generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=generator,
        ).to(device)
        logits = endpoint_logits(
            model=model,
            controller=controller,
            controller_start_step=controller_start_step,
            executor_off=executor_off,
            inputs=batch.inputs,
            target_step=target_step,
        )
        update_metrics(
            totals,
            logits=logits,
            batch=batch,
            semantic_positions=semantic_positions,
            layout_positions=layout_positions,
        )
    return {
        "examples": totals["examples"],
        "supervised_answer_exact_match": (
            totals["supervised_exact"] / totals["examples"]
        ),
        "supervised_digit_exact_match": (
            totals["digit_exact"] / totals["examples"]
        ),
        "supervised_digit_bit_accuracy": (
            totals["digit_correct"] / totals["digit_tokens"]
        ),
        "final_carry_accuracy": totals["carry_correct"] / totals["examples"],
        "full_arithmetic_exact_match": (
            totals["semantic_exact"] / totals["examples"]
        ),
        "full_arithmetic_bit_accuracy": (
            totals["semantic_correct"] / totals["semantic_tokens"]
        ),
        # Backward-compatible aliases used by the earlier T(n)=n+1 plots.
        "strict_answer_region_exact_match": (
            totals["supervised_exact"] / totals["examples"]
        ),
        "semantic_sum_exact_match": totals["semantic_exact"] / totals["examples"],
        "semantic_sum_bit_accuracy": totals["semantic_correct"] / totals["semantic_tokens"],
        "layout_sum_exact_match": totals["layout_exact"] / totals["examples"],
        "layout_sum_bit_accuracy": totals["layout_correct"] / totals["layout_tokens"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("endpoint accuracy requires Addition")
    controller_payload: dict[str, Any] | None = None
    modes: dict[str, tuple[torch.nn.Module | None, int | None, bool]] = {
        "raw": (None, None, False)
    }
    if args.controller is not None:
        controller, controller_payload = load_controller(
            args.controller, device=device
        )
        anchor_step = int(controller_payload["anchor_step"])
        parameterization = controller_payload.get(
            "controller_parameterization", "diagonal_low_rank"
        )
        if parameterization == "diagonal_low_rank":
            modes.update({
                "full": (
                    ControllerView(controller, mode="full").to(device).eval(),
                    anchor_step,
                    False,
                ),
                "no_AB": (
                    ControllerView(controller, mode="no_AB").to(device).eval(),
                    anchor_step,
                    False,
                ),
                "identity_D": (
                    ControllerView(controller, mode="identity_D").to(device).eval(),
                    anchor_step,
                    False,
                ),
                "full_executor_off": (
                    ControllerView(controller, mode="full").to(device).eval(),
                    anchor_step,
                    True,
                ),
            })
        elif parameterization == "dense_affine":
            modes.update({
                "full": (
                    DenseControllerView(controller, mode="full").to(device).eval(),
                    anchor_step,
                    False,
                ),
                "no_offdiag": (
                    DenseControllerView(controller, mode="no_offdiag").to(device).eval(),
                    anchor_step,
                    False,
                ),
                "identity_D": (
                    DenseControllerView(controller, mode="identity_D").to(device).eval(),
                    anchor_step,
                    False,
                ),
                "full_executor_off": (
                    DenseControllerView(controller, mode="full").to(device).eval(),
                    anchor_step,
                    True,
                ),
            })
        else:
            raise ValueError(f"unsupported controller parameterization: {parameterization}")
    variants = (
        tuple(modes)
        if args.variants is None
        else tuple(args.variants)
    )
    unavailable = [variant for variant in variants if variant not in modes]
    if unavailable:
        raise ValueError(
            "requested controller variants require --controller: "
            + ", ".join(unavailable)
        )
    modes = {variant: modes[variant] for variant in variants}
    rows: list[dict[str, Any]] = []
    for logical_length in args.lengths:
        for variant, (view, start, executor_off) in modes.items():
            rows.append(
                {
                    "variant": variant,
                    "length": logical_length,
                    "target_step": logical_length + int(spec.step_offset),
                    **evaluate_variant(
                        model=model,
                        spec=spec,
                        controller=view,
                        controller_start_step=start,
                        executor_off=executor_off,
                        logical_length=logical_length,
                        batch_size=args.batch_size,
                        batches=args.batches,
                        seed=args.seed,
                        device=device,
                    ),
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "endpoint_accuracy.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller) if args.controller is not None else None,
        "controller_anchor_step": (
            int(controller_payload["anchor_step"])
            if controller_payload is not None
            else None
        ),
        "target_loop_rule": (
            "T(m)=m"
            if int(spec.step_offset) == 0
            else f"T(m)=m+{int(spec.step_offset)}"
        ),
        "addition_answer_supervision": spec.addition_answer_supervision,
        "loss_placement": {
            "backbone": backbone_payload["supervision"],
            "controller": (
                controller_payload["loss"]
                if controller_payload is not None
                else None
            ),
        },
        "lengths": list(args.lengths),
        "variants": list(variants),
        "evaluation_seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "batches": int(args.batches),
        "examples_per_length": args.batch_size * args.batches,
        "metrics": {
            "supervised_answer_exact_match": "all positions selected by the checkpoint answer mask",
            "supervised_digit_exact_match": "the m numerical sum digits, excluding final carry",
            "supervised_digit_bit_accuracy": "per-bit accuracy on the m numerical sum digits",
            "final_carry_accuracy": "accuracy of the unsupervised final carry bit",
            "full_arithmetic_exact_match": "all m sum digits and final carry are correct",
            "full_arithmetic_bit_accuracy": "per-bit accuracy across m sum digits and final carry",
            "strict_answer_region_exact_match": "backward-compatible alias for supervised_answer_exact_match",
            "semantic_sum_exact_match": "backward-compatible alias for full_arithmetic_exact_match",
            "semantic_sum_bit_accuracy": "backward-compatible alias for full_arithmetic_bit_accuracy",
            "layout_sum_exact_match": "all width+1 sum bits, including required high zero padding",
            "layout_sum_bit_accuracy": "per-bit accuracy on all width+1 layout sum bits",
        },
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "raw",
            "full",
            "no_AB",
            "no_offdiag",
            "identity_D",
            "full_executor_off",
        ),
        default=None,
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=tuple(range(1, 21)))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=584001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
