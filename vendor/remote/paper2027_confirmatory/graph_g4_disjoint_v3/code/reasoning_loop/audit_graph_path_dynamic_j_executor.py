"""Audit dynamic-J execution semantics against the training-time executor.

This is a diagnostic, not a training script.  It checks four independent
questions:

1. Natural H1 really is the aligned H1 state used by controller training.
2. The long-schedule executor agrees with the canonical mixed-trajectory
   executor on every schedule fragment that stays inside H1..H8.
3. Sequential affine rollback agrees with the correctly ordered one-shot
   affine product under the repository's row-vector convention.
4. A symbolic phase label is supported (or contradicted) by an independently
   fitted natural-state age probe and by continuation to the trained H8
   readout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    AgeTrajectory,
    _run_mixed_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--age-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=827901)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def make_trajectory(actions: Sequence[int]) -> AgeTrajectory:
    age = 1
    ages: list[int] = []
    for action in actions:
        if action not in {-1, 1}:
            raise ValueError("trajectory actions must be +/-1")
        ages.append(age)
        age += action
        if not 1 <= age <= 8:
            raise ValueError("audit trajectory left H1..H8")
    return AgeTrajectory(
        start_age=1,
        end_age=age,
        extra_backs=sum(action == -1 for action in actions),
        actions=tuple(actions),
        ages=tuple(ages),
        mandatory_rollback_source=None,
    )


def tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = left.float() - right.float()
    return {
        "max_abs": float(delta.abs().max()),
        "rms": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / right.float().norm().clamp_min(1e-12)),
    }


def custom_execute(
    *,
    model,
    bank: AgeSpecificJBank,
    state: torch.Tensor,
    current: torch.Tensor,
    successors: torch.Tensor,
    trajectory: AgeTrajectory,
    positions: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Literal implementation used by the long-schedule evaluator."""

    logical_age = trajectory.start_age
    physical_forward_count = 1
    for action in trajectory.actions:
        if action == 1:
            state = model.apply_loop(state, loop_index=physical_forward_count)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
            physical_forward_count += 1
        else:
            state = bank.rollback(
                state,
                source_age=logical_age,
                positions=positions,
            )
            logical_age -= 1
    return state, current, logical_age


def composed_affine(
    bank: AgeSpecificJBank,
    source_ages: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose h -> h W + b in the same order as sequential rollback."""

    device = next(bank.parameters()).device
    weight = torch.eye(bank.dimension, device=device, dtype=torch.float32)
    bias = torch.zeros(bank.dimension, device=device, dtype=torch.float32)
    for source_age in source_ages:
        next_weight, next_bias = bank.affine(source_age)
        bias = bias @ next_weight + next_bias
        weight = weight @ next_weight
    return weight, bias


def exact_phase_metrics(
    *,
    model,
    cfg,
    state: torch.Tensor,
    current: torch.Tensor,
    successors: torch.Tensor,
    logical_age: int,
    phase_positions: list[int],
    probe_weight: torch.Tensor,
    probe_bias: torch.Tensor,
) -> dict[str, float]:
    exact = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=current,
        age=logical_age,
        phase_position=phase_positions[logical_age],
    )
    state_error = tensor_error(state, exact)
    predicted_age = state[:, -1, :].float() @ probe_weight + probe_bias

    continuation = state
    continuation_current = current
    for loop_index in range(logical_age, 8):
        continuation = model.apply_loop(continuation, loop_index=loop_index)
        continuation_current = advance_nodes(successors, continuation_current, steps=1)
    logits = logits_from_raw_state(model, continuation)
    continuation_accuracy = float(
        logits.argmax(-1).eq(continuation_current).float().mean()
    )
    return {
        "symbolic_age": logical_age,
        "probe_age_mean": float(predicted_age.mean()),
        "probe_age_mae": float((predicted_age - logical_age).abs().mean()),
        "probe_rounded_accuracy": float(
            predicted_age.round().clamp(1, 8).eq(logical_age).float().mean()
        ),
        "exact_state_relative_l2": state_error["relative_l2"],
        "exact_state_rms": state_error["rms"],
        "continuation_to_H8_accuracy": continuation_accuracy,
    }


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)

    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    bank_payload = torch.load(args.bank_artifact, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=cfg.d_model,
        rank=int(bank_payload["rank"]),
        stage_rank=int(bank_payload["stage_rank"]),
        map_architecture=str(bank_payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(bank_payload["state_dict"])
    bank = bank.frozen()
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    probe_payload = np.load(args.age_probe)
    probe_weight = torch.as_tensor(
        probe_payload["baseline_weight"], device=device, dtype=torch.float32
    )
    probe_bias = torch.as_tensor(
        probe_payload["baseline_bias"], device=device, dtype=torch.float32
    )
    positions = tuple(range(cfg.seq_len))

    tokens, _, successors, start = fixed_depth_batch(
        cfg,
        args.batch_size,
        device,
        path_positions=cfg.max_depth,
    )
    h0 = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    h1 = model.apply_loop(h0, loop_index=0)
    h1_current = advance_nodes(successors, start, steps=1)
    aligned_h1 = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=h1_current,
        age=1,
        phase_position=phase_positions[1],
    )

    trajectories = {
        "FFJJ_x3": (1, 1, -1, -1) * 3,
        "FFFFJJJJ_x2": (1, 1, 1, 1, -1, -1, -1, -1) * 2,
        "F7J7": (1,) * 7 + (-1,) * 7,
        "F4_then_FJ_x3": (1,) * 4 + (1, -1) * 3,
        "FFJ_x6": (1, 1, -1) * 6,
    }
    parity_rows: list[dict[str, Any]] = []
    for name, actions in trajectories.items():
        trajectory = make_trajectory(actions)
        custom_state, custom_current, custom_age = custom_execute(
            model=model,
            bank=bank,
            state=h1.clone(),
            current=h1_current.clone(),
            successors=successors,
            trajectory=trajectory,
            positions=positions,
        )
        canonical_state, canonical_logits, canonical_current = _run_mixed_trajectory(
            model=model,
            cfg=cfg,
            bank=bank,
            positions=positions,
            successors=successors,
            endpoint=h1_current.clone(),
            initial_state=h1.clone(),
            trajectory=trajectory,
            condition="learned",
            phase_positions=phase_positions,
            rollback_composition="product",
        )
        custom_logits = logits_from_raw_state(model, custom_state)
        parity_rows.append(
            {
                "schedule": name,
                "trajectory": asdict(trajectory),
                "custom_final_age": custom_age,
                "current_exact_match": bool(custom_current.eq(canonical_current).all()),
                "state_error": tensor_error(custom_state, canonical_state),
                "logit_error": tensor_error(custom_logits, canonical_logits),
                "phase_diagnostics": exact_phase_metrics(
                    model=model,
                    cfg=cfg,
                    state=custom_state,
                    current=custom_current,
                    successors=successors,
                    logical_age=custom_age,
                    phase_positions=phase_positions,
                    probe_weight=probe_weight,
                    probe_bias=probe_bias,
                ),
            }
        )

    source_ages = (8, 7, 6, 5)
    product_input = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=h1_current,
        age=8,
        phase_position=phase_positions[8],
    )
    sequential = product_input
    for source_age in source_ages:
        sequential = bank.rollback(
            sequential,
            source_age=source_age,
            positions=positions,
        )
    product_weight, product_bias = composed_affine(bank, source_ages)
    one_shot = product_input.float() @ product_weight + product_bias
    reverse_weight, reverse_bias = composed_affine(bank, tuple(reversed(source_ages)))
    wrong_reverse = product_input.float() @ reverse_weight + reverse_bias

    # Record every state on representative bounded paths, not only endpoints.
    phase_rows: list[dict[str, Any]] = []
    for name in ("F7J7", "FFJ_x6", "FFFFJJJJ_x2"):
        trajectory = make_trajectory(trajectories[name])
        state = h1.clone()
        current = h1_current.clone()
        logical_age = 1
        phase_rows.append(
            {
                "schedule": name,
                "action_index": 0,
                "action": "initial_H1",
                **exact_phase_metrics(
                    model=model,
                    cfg=cfg,
                    state=state,
                    current=current,
                    successors=successors,
                    logical_age=logical_age,
                    phase_positions=phase_positions,
                    probe_weight=probe_weight,
                    probe_bias=probe_bias,
                ),
            }
        )
        for action_index, action in enumerate(trajectory.actions, start=1):
            source_age: int | None = None
            if action == 1:
                state = model.apply_loop(state, loop_index=action_index)
                current = advance_nodes(successors, current, steps=1)
                logical_age += 1
            else:
                source_age = logical_age
                state = bank.rollback(
                    state,
                    source_age=source_age,
                    positions=positions,
                )
                logical_age -= 1
            phase_rows.append(
                {
                    "schedule": name,
                    "action_index": action_index,
                    "action": "F" if action == 1 else "J",
                    "source_age": source_age,
                    **exact_phase_metrics(
                        model=model,
                        cfg=cfg,
                        state=state,
                        current=current,
                        successors=successors,
                        logical_age=logical_age,
                        phase_positions=phase_positions,
                        probe_weight=probe_weight,
                        probe_bias=probe_bias,
                    ),
                }
            )

    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "block_schedule": cfg.block_schedule,
        "shared_block_count": cfg.n_layers,
        "bank_artifact": str(args.bank_artifact),
        "bank_architecture": bank.map_architecture,
        "row_vector_convention": "h' = h @ W + b",
        "natural_H1_alignment_error": tensor_error(h1, aligned_h1),
        "executor_parity": parity_rows,
        "affine_product_audit": {
            "source_ages_in_execution_order": source_ages,
            "correct_formula": (
                "W=W8@W7@W6@W5; "
                "b=(((b8@W7+b7)@W6+b6)@W5+b5)"
            ),
            "sequential_vs_one_shot": tensor_error(sequential, one_shot),
            "sequential_vs_wrong_reversed_order": tensor_error(
                sequential, wrong_reverse
            ),
        },
        "phase_diagnostics": phase_rows,
        "phase_interpretation": (
            "J selection in the evaluated long schedules is open-loop symbolic "
            "bookkeeping, not online classification of the hidden state."
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
