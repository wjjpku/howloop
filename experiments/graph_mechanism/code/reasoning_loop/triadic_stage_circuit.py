from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.triadic_shortage import (
    TriadicShortageModel,
    all_triples,
    make_visibility_mask,
)
from reasoning_loop.triadic_shortage_circuit import run_intervened
from reasoning_loop.triadic_shortage_diagnostics import load_checkpoint
from reasoning_loop.triadic_shortage_train import pick_device
from reasoning_loop.triadic_stage_composition import stage_targets


def _accuracy(
    logits_by_loop: torch.Tensor,
    target: torch.Tensor,
    *,
    readout_index: int,
) -> float:
    return float(
        logits_by_loop[:, readout_index]
        .argmax(dim=-1)
        .eq(target)
        .float()
        .mean()
        .item()
    )


@torch.no_grad()
def phase_component_table(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    *,
    train_loops: int,
) -> dict[str, Any]:
    if not 2 <= train_loops <= model.cfg.loops:
        raise ValueError("phase analysis requires at least two trained loops")
    device = next(model.parameters()).device
    operands = operands.to(device)
    visibility = make_visibility_mask(
        "full",
        operands.shape[0],
        model.cfg.loops,
        device,
    )
    targets = stage_targets(operands, p=model.cfg.p)
    baseline_logits, _, trace = run_intervened(
        model,
        operands,
        visibility,
        return_trace=True,
    )
    baseline_endpoint = _accuracy(
        baseline_logits,
        targets[:, 1],
        readout_index=train_loops - 1,
    )
    rows: list[dict[str, Any]] = []
    for loop_index in range(train_loops):
        for component in ("attention", "mlp"):
            kwargs = (
                {"zero_attention_loops": {loop_index}}
                if component == "attention"
                else {"zero_mlp_loops": {loop_index}}
            )
            logits, _, _ = run_intervened(
                model,
                operands,
                visibility,
                **kwargs,
            )
            endpoint_accuracy = _accuracy(
                logits,
                targets[:, 1],
                readout_index=train_loops - 1,
            )
            rows.append(
                {
                    "loop": loop_index + 1,
                    "component": component,
                    "endpoint_accuracy": endpoint_accuracy,
                    "endpoint_drop": baseline_endpoint - endpoint_accuracy,
                    "update_norm": float(
                        trace[
                            f"loop{loop_index}."
                            f"{'attention_out' if component == 'attention' else 'mlp_out'}"
                        ]
                        .norm(dim=-1)
                        .mean()
                        .item()
                    ),
                }
            )

    head_rows: list[dict[str, Any]] = []
    for loop_index in range(train_loops):
        for head in range(model.cfg.n_heads):
            logits, _, _ = run_intervened(
                model,
                operands,
                visibility,
                ablate_heads={loop_index: {head}},
            )
            accuracy = _accuracy(
                logits,
                targets[:, 1],
                readout_index=train_loops - 1,
            )
            head_rows.append(
                {
                    "loop": loop_index + 1,
                    "head": head,
                    "endpoint_accuracy": accuracy,
                    "endpoint_drop": baseline_endpoint - accuracy,
                }
            )

    return {
        "examples": operands.shape[0],
        "train_loops": train_loops,
        "baseline_endpoint_accuracy": baseline_endpoint,
        "one_loop_endpoint_accuracy": _accuracy(
            baseline_logits,
            targets[:, 1],
            readout_index=0,
        ),
        "loop1_partial_sum_accuracy": _accuracy(
            baseline_logits,
            targets[:, 0],
            readout_index=0,
        ),
        "extra_loop_endpoint_accuracy": (
            _accuracy(
                baseline_logits,
                targets[:, 1],
                readout_index=train_loops,
            )
            if train_loops < model.cfg.loops
            else None
        ),
        "rows": rows,
        "head_rows": head_rows,
    }


@torch.no_grad()
def hybrid_executor_table(
    model: TriadicShortageModel,
    donor_operands: torch.Tensor,
    receiver_operands: torch.Tensor,
    *,
    train_loops: int,
) -> dict[str, Any]:
    if train_loops < 2:
        raise ValueError("hybrid executor analysis requires at least two loops")
    if donor_operands.shape != receiver_operands.shape:
        raise ValueError("donor and receiver operands must have the same shape")
    device = next(model.parameters()).device
    donor_operands = donor_operands.to(device)
    receiver_operands = receiver_operands.to(device)
    visibility = make_visibility_mask(
        "full",
        donor_operands.shape[0],
        model.cfg.loops,
        device,
    )
    _, _, donor_trace = run_intervened(
        model,
        donor_operands,
        visibility,
        stop_loop=1,
        return_trace=True,
    )
    donor_workspace = donor_trace["loop0.workspace_out"]
    hybrid_target = (
        (donor_operands[:, 0] + donor_operands[:, 1]).remainder(model.cfg.p)
        * receiver_operands[:, 2]
    ).remainder(model.cfg.p)
    receiver_target = stage_targets(receiver_operands, p=model.cfg.p)[:, 1]
    c_only_visibility = torch.zeros_like(visibility)
    c_only_visibility[:, 1:train_loops, 2] = True

    def continue_with(
        *,
        active_visibility: torch.Tensor = visibility,
        active_workspace: torch.Tensor = donor_workspace,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits, _, _ = run_intervened(
            model,
            receiver_operands,
            active_visibility,
            initial_workspace=active_workspace,
            start_loop=1,
            stop_loop=train_loops,
            **kwargs,
        )
        return logits

    baseline_logits = continue_with()
    c_only_logits = continue_with(active_visibility=c_only_visibility)
    shuffled_workspace = donor_workspace.roll(shifts=1, dims=0)
    shuffled_c_only_logits = continue_with(
        active_visibility=c_only_visibility,
        active_workspace=shuffled_workspace,
    )
    reset_c_only_logits = continue_with(
        active_visibility=c_only_visibility,
        active_workspace=model._initial_workspace(donor_operands.shape[0]),
    )
    baseline_hybrid = _accuracy(
        baseline_logits,
        hybrid_target,
        readout_index=-1,
    )
    c_only_hybrid = _accuracy(
        c_only_logits,
        hybrid_target,
        readout_index=-1,
    )
    component_rows: list[dict[str, Any]] = []
    c_only_component_rows: list[dict[str, Any]] = []
    executor_loop = 1
    for component in ("attention", "mlp"):
        kwargs = (
            {"zero_attention_loops": {executor_loop}}
            if component == "attention"
            else {"zero_mlp_loops": {executor_loop}}
        )
        logits = continue_with(**kwargs)
        accuracy = _accuracy(logits, hybrid_target, readout_index=-1)
        component_rows.append(
            {
                "component": component,
                "hybrid_accuracy": accuracy,
                "hybrid_drop": baseline_hybrid - accuracy,
            }
        )
        c_only_ablation_logits = continue_with(
            active_visibility=c_only_visibility,
            **kwargs,
        )
        c_only_accuracy = _accuracy(
            c_only_ablation_logits,
            hybrid_target,
            readout_index=-1,
        )
        c_only_component_rows.append(
            {
                "component": component,
                "hybrid_c_only_accuracy": c_only_accuracy,
                "hybrid_c_only_drop": c_only_hybrid - c_only_accuracy,
            }
        )

    head_rows: list[dict[str, Any]] = []
    for head in range(model.cfg.n_heads):
        logits = continue_with(ablate_heads={executor_loop: {head}})
        accuracy = _accuracy(logits, hybrid_target, readout_index=-1)
        head_rows.append(
            {
                "loop": executor_loop + 1,
                "head": head,
                "hybrid_accuracy": accuracy,
                "hybrid_drop": baseline_hybrid - accuracy,
            }
        )
    return {
        "examples": donor_operands.shape[0],
        "hybrid_accuracy": baseline_hybrid,
        "hybrid_c_only_accuracy": c_only_hybrid,
        "shuffled_workspace_c_only_accuracy": _accuracy(
            shuffled_c_only_logits,
            hybrid_target,
            readout_index=-1,
        ),
        "reset_workspace_c_only_accuracy": _accuracy(
            reset_c_only_logits,
            hybrid_target,
            readout_index=-1,
        ),
        "receiver_target_accuracy": _accuracy(
            baseline_logits,
            receiver_target,
            readout_index=-1,
        ),
        "component_rows": component_rows,
        "c_only_component_rows": c_only_component_rows,
        "head_rows": head_rows,
    }


def _phase_role_scorecard(
    phase: dict[str, Any],
    hybrid: dict[str, Any],
) -> dict[str, Any]:
    rows = phase["rows"]
    per_loop_max_drop = {
        loop: max(
            float(row["endpoint_drop"])
            for row in rows
            if int(row["loop"]) == loop
        )
        for loop in range(1, int(phase["train_loops"]) + 1)
    }
    recurrent_necessity = (
        float(phase["baseline_endpoint_accuracy"])
        - float(phase["one_loop_endpoint_accuracy"])
    )
    extra_loop = phase["extra_loop_endpoint_accuracy"]
    repeat_damage = (
        float(phase["baseline_endpoint_accuracy"]) - float(extra_loop)
        if extra_loop is not None
        else None
    )
    return {
        "behavior_gate": (
            float(phase["baseline_endpoint_accuracy"]) >= 0.99
            and recurrent_necessity >= 0.20
        ),
        "stage_boundary_gate": float(hybrid["hybrid_accuracy"]) >= 0.90,
        "stage_boundary_c_only_gate": (
            float(hybrid["hybrid_c_only_accuracy"]) >= 0.90
            and float(hybrid["shuffled_workspace_c_only_accuracy"]) <= 0.50
            and float(hybrid["reset_workspace_c_only_accuracy"]) <= 0.50
        ),
        "both_loops_causally_necessary": all(
            drop >= 0.10 for drop in per_loop_max_drop.values()
        ),
        "loop1_partial_sum_decodable": (
            float(phase["loop1_partial_sum_accuracy"]) >= 0.90
        ),
        "repeat_damage": repeat_damage,
        "stable_operator_repeat_rejected": (
            repeat_damage is not None and repeat_damage >= 0.20
        ),
        "recurrent_necessity": recurrent_necessity,
        "per_loop_max_component_drop": per_loop_max_drop,
    }


@torch.no_grad()
def analyze_stage_circuit(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    sample_size: int = 1024,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    train_loops = int(payload["train_loops"])
    operands, _ = all_triples(model.cfg.p, device=device)
    heldout_idx = payload["split"]["heldout_idx"].to(device)
    count = min(sample_size, int(heldout_idx.numel()))
    receiver_idx = heldout_idx[:count]
    donor_idx = heldout_idx.roll(shifts=max(1, count // 2))[:count]
    receiver = operands[receiver_idx]
    donor = operands[donor_idx]
    phase = phase_component_table(
        model,
        receiver,
        train_loops=train_loops,
    )
    hybrid = hybrid_executor_table(
        model,
        donor,
        receiver,
        train_loops=train_loops,
    )
    scorecard = _phase_role_scorecard(phase, hybrid)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "architecture": model.cfg.architecture,
        "seed": int(payload["seed"]),
        "step": int(payload["step"]),
        "phase": phase,
        "hybrid": hybrid,
        "scorecard": scorecard,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze phase-specific virtual-depth circuits."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze_stage_circuit(
        args.checkpoint,
        args.out_dir,
        device=pick_device(args.device),
        sample_size=args.sample_size,
    )
    print(json.dumps(result["scorecard"], indent=2), flush=True)


if __name__ == "__main__":
    main()
