"""Test local closure of adjacent rollback maps against independently trained K maps.

The frozen D8L8 graph-path backbone defines the forward recurrent operator F.
For every interval of length two through five, this module trains an
unrestricted position-shared affine map

    K_(a->b): H_a -> H_b,  a-b in {2, 3, 4, 5},

using only final task cross entropy after K is followed by exactly a-b frozen
loops.  It then compares K against the ordered product of the already-trained
adjacent J maps and against order, stage, shared-only, identity, exact-state,
and batch-shuffle controls.  Hidden-state distances are diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    FullAffineJ,
)


DEFAULT_LENGTHS = (2, 3, 4, 5)


def interval_pairs(
    *,
    max_age: int = 8,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
) -> tuple[tuple[int, int], ...]:
    requested = tuple(sorted(set(int(value) for value in lengths)))
    if not requested or requested[0] < 1:
        raise ValueError("interval lengths must be positive")
    return tuple(
        (source, source - length)
        for length in requested
        for source in range(length + 1, max_age + 1)
    )


def interval_key(source_age: int, target_age: int) -> str:
    if not 1 <= target_age < source_age:
        raise ValueError("an interval must satisfy 1 <= target < source")
    return f"{source_age}_to_{target_age}"


def ordered_source_ages(source_age: int, target_age: int) -> tuple[int, ...]:
    interval_key(source_age, target_age)
    return tuple(range(source_age, target_age, -1))


def cyclic_wrong_source_ages(
    source_age: int,
    target_age: int,
) -> tuple[int, ...]:
    return tuple(2 + ((age - 2 + 1) % 7) for age in ordered_source_ages(source_age, target_age))


def compose_affines(
    affines: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    dimension: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose row-vector affine maps in their sequential application order."""
    weight = torch.eye(dimension, device=device, dtype=torch.float32)
    bias = torch.zeros(dimension, device=device, dtype=torch.float32)
    for next_weight, next_bias in affines:
        if next_weight.shape != (dimension, dimension):
            raise ValueError("affine weight has the wrong shape")
        if next_bias.shape != (dimension,):
            raise ValueError("affine bias has the wrong shape")
        bias = bias @ next_weight.float() + next_bias.float()
        weight = weight @ next_weight.float()
    return weight, bias


def apply_affine_to_positions(
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    positions: tuple[int, ...],
) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError("state must have shape [batch, sequence, dimension]")
    if not positions:
        return state
    index = torch.as_tensor(positions, device=state.device)
    if bool((index < 0).any()) or bool((index >= state.shape[1]).any()):
        raise ValueError("position is out of range")
    result = state.clone()
    result[:, index] = (result[:, index].float() @ weight.float() + bias.float()).to(
        state.dtype
    )
    return result


class DirectIntervalJBank(torch.nn.Module):
    """One independent unrestricted affine map for every registered interval."""

    def __init__(
        self,
        *,
        dimension: int,
        pairs: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.pairs = tuple(interval_pairs() if pairs is None else pairs)
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("direct-map intervals must be unique")
        self.maps = torch.nn.ModuleDict(
            {
                interval_key(source, target): FullAffineJ(dimension=dimension)
                for source, target in self.pairs
            }
        )

    def affine(self, source_age: int, target_age: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = interval_key(source_age, target_age)
        if key not in self.maps:
            raise ValueError(f"unregistered direct interval: {key}")
        module = self.maps[key]
        return module.weight.float(), module.bias.float()

    def rollback(
        self,
        state: torch.Tensor,
        *,
        source_age: int,
        target_age: int,
        positions: tuple[int, ...],
    ) -> torch.Tensor:
        weight, bias = self.affine(source_age, target_age)
        return apply_affine_to_positions(state, weight, bias, positions=positions)

    def frozen(self) -> "DirectIntervalJBank":
        result = copy.deepcopy(self).eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_adjacent_bank(
    path: Path,
    *,
    dimension: int,
    device: torch.device,
) -> tuple[AgeSpecificJBank, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=dimension,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=str(payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    return bank.frozen(), payload


def shared_component_affine(bank: AgeSpecificJBank) -> tuple[torch.Tensor, torch.Tensor]:
    if bank.map_architecture != "shared_diagonal_stage_lora":
        raise TypeError("shared-only control requires shared_diagonal_stage_lora")
    weight = (
        torch.diag(bank.shared_diagonal_scale.float())
        + bank.shared_A.float() @ bank.shared_B.float()
    )
    return weight, bank.shared_bias.float()


def apply_adjacent_sequence(
    state: torch.Tensor,
    *,
    bank: AgeSpecificJBank,
    source_ages: Sequence[int],
    positions: tuple[int, ...],
) -> torch.Tensor:
    result = state
    for age in source_ages:
        result = bank.rollback(result, source_age=int(age), positions=positions)
    return result


def apply_shared_sequence(
    state: torch.Tensor,
    *,
    bank: AgeSpecificJBank,
    steps: int,
    positions: tuple[int, ...],
) -> torch.Tensor:
    weight, bias = shared_component_affine(bank)
    result = state
    for _ in range(steps):
        result = apply_affine_to_positions(result, weight, bias, positions=positions)
    return result


def phase_jump(phase_positions: Sequence[int]) -> int:
    if len(phase_positions) <= 2:
        raise ValueError("phase summary must cover at least H2")
    jumps = {
        int(right) - int(left)
        for left, right in zip(phase_positions[1:-1], phase_positions[2:], strict=True)
    }
    if len(jumps) != 1 or next(iter(jumps)) <= 0:
        raise ValueError("logical ages must have one positive constant phase jump")
    return next(iter(jumps))


def roll_forward(
    *,
    model,
    state: torch.Tensor,
    start_age: int,
    steps: int,
    successors: torch.Tensor,
    current: torch.Tensor,
    jump: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    differentiable_loop = run_one_loop.__wrapped__
    result = state
    for offset in range(steps):
        result = differentiable_loop(
            model,
            result,
            loop_index=start_age + offset,
        ).state
    target = advance_nodes(successors, current, steps=jump * steps)
    return result, logits_from_raw_state(model, result), target


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_direct_artifact(
    path: Path,
    *,
    direct: DirectIntervalJBank,
    args: argparse.Namespace,
    completed_rounds: int,
) -> None:
    payload = {
        "kind": "graph_path_direct_interval_K_bank",
        "checkpoint": str(args.checkpoint),
        "phase_summary": str(args.phase_summary),
        "adjacent_bank": str(args.adjacent_bank),
        "pairs": [list(pair) for pair in direct.pairs],
        "architecture": "one unrestricted position-shared affine map per interval",
        "positions": "all tokens; no token mixing",
        "training_loss": "final task CE only after exact interval-length frozen executor",
        "hidden_state_loss_weight": 0.0,
        "initialization": "identity",
        "completed_rounds": completed_rounds,
        "parameter_count": direct.parameter_count,
        "state_dict": {key: value.detach().cpu() for key, value in direct.state_dict().items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_direct_bank(
    path: Path,
    *,
    dimension: int,
    device: torch.device,
) -> tuple[DirectIntervalJBank, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in payload["pairs"])
    bank = DirectIntervalJBank(dimension=dimension, pairs=pairs).to(device)
    bank.load_state_dict(payload["state_dict"])
    return bank.frozen(), payload


def train_direct_bank(
    *,
    model,
    cfg,
    direct: DirectIntervalJBank,
    phase_positions: Sequence[int],
    positions: tuple[int, ...],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    direct.train()
    optimizer = torch.optim.AdamW(direct.parameters(), lr=args.learning_rate, weight_decay=0.0)
    jump = phase_jump(phase_positions)
    rows: list[dict[str, Any]] = []
    for round_index in range(1, args.train_rounds + 1):
        if round_index > (3 * args.train_rounds) // 4:
            factor = 0.1
        elif round_index > args.train_rounds // 2:
            factor = 1.0 / 3.0
        else:
            factor = 1.0
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * factor
        order_rng = np.random.default_rng(args.seed + 1000 * round_index)
        order = order_rng.permutation(len(direct.pairs))
        round_loss = 0.0
        round_correct = 0
        round_examples = 0
        started = time.monotonic()
        for batch_index, pair_index in enumerate(order):
            source_age, target_age = direct.pairs[int(pair_index)]
            with torch.no_grad():
                _, path_targets, successors, _ = fixed_depth_batch(
                    cfg,
                    args.train_batch_size,
                    device,
                    path_positions=cfg.max_depth,
                )
                current = path_targets[:, cfg.max_depth - 1]
                source = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=source_age,
                    phase_position=phase_positions[source_age],
                )
            optimizer.zero_grad(set_to_none=True)
            rolled = direct.rollback(
                source,
                source_age=source_age,
                target_age=target_age,
                positions=positions,
            )
            _, logits, target = roll_forward(
                model=model,
                state=rolled,
                start_age=target_age,
                steps=source_age - target_age,
                successors=successors,
                current=current,
                jump=jump,
            )
            loss = F.cross_entropy(logits.float(), target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(direct.parameters(), args.grad_clip)
            optimizer.step()
            correct = int(logits.argmax(dim=-1).eq(target).sum())
            round_loss += float(loss.detach())
            round_correct += correct
            round_examples += args.train_batch_size
            rows.append(
                {
                    "round": round_index,
                    "batch": batch_index,
                    "source_age": source_age,
                    "target_age": target_age,
                    "rollback_steps": source_age - target_age,
                    "loss": float(loss.detach()),
                    "accuracy": correct / args.train_batch_size,
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "examples": args.train_batch_size,
                }
            )
        print(
            json.dumps(
                {
                    "round": round_index,
                    "loss": round_loss / len(direct.pairs),
                    "accuracy": round_correct / round_examples,
                    "seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _write_csv(args.out_dir / "direct_training.csv", rows)
        if round_index % args.save_every_rounds == 0 or round_index == args.train_rounds:
            _save_direct_artifact(
                args.out_dir / "direct_interval_k_bank.pt",
                direct=direct,
                args=args,
                completed_rounds=round_index,
            )
    return rows


def _state_metrics(state: torch.Tensor, exact: torch.Tensor) -> tuple[float, float]:
    residual = state.float() - exact.float()
    relative_error = float(residual.norm() / exact.float().norm().clamp_min(1e-12))
    cosine = float(
        F.cosine_similarity(
            state.float().reshape(-1, state.shape[-1]),
            exact.float().reshape(-1, exact.shape[-1]),
            dim=-1,
        ).mean()
    )
    return relative_error, cosine


def _target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    mask = torch.nn.functional.one_hot(target, num_classes=logits.shape[-1]).bool()
    strongest_other = logits.masked_fill(mask, float("-inf")).max(dim=-1).values
    return correct - strongest_other


@torch.no_grad()
def evaluate_closure(
    *,
    model,
    cfg,
    adjacent: AgeSpecificJBank,
    direct: DirectIntervalJBank,
    phase_positions: Sequence[int],
    positions: tuple[int, ...],
    device: torch.device,
    examples: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if examples % batch_size:
        raise ValueError("evaluation examples must be divisible by batch size")
    if batch_size < 2:
        raise ValueError("batch-shuffle controls require batch size >= 2")
    set_seed(seed)
    jump = phase_jump(phase_positions)
    raw_rows: list[dict[str, Any]] = []
    for source_age, target_age in direct.pairs:
        steps = source_age - target_age
        for batch_index in range(examples // batch_size):
            _, path_targets, successors, _ = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            current = path_targets[:, cfg.max_depth - 1]
            source = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=source_age,
                phase_position=phase_positions[source_age],
            )
            exact = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=target_age,
                phase_position=phase_positions[target_age],
            )
            direct_state = direct.rollback(
                source,
                source_age=source_age,
                target_age=target_age,
                positions=positions,
            )
            ordered_state = apply_adjacent_sequence(
                source,
                bank=adjacent,
                source_ages=ordered_source_ages(source_age, target_age),
                positions=positions,
            )
            state_conditions = {
                "direct_K": direct_state,
                "ordered_J_product": ordered_state,
                "reversed_J_order": apply_adjacent_sequence(
                    source,
                    bank=adjacent,
                    source_ages=tuple(reversed(ordered_source_ages(source_age, target_age))),
                    positions=positions,
                ),
                "cyclic_wrong_stage_product": apply_adjacent_sequence(
                    source,
                    bank=adjacent,
                    source_ages=cyclic_wrong_source_ages(source_age, target_age),
                    positions=positions,
                ),
                "shared_component_only": apply_shared_sequence(
                    source,
                    bank=adjacent,
                    steps=steps,
                    positions=positions,
                ),
                "identity": source,
                "exact_H_b_oracle": exact,
                "batch_shuffled_direct_K": direct_state.roll(1, dims=0),
                "batch_shuffled_ordered_product": ordered_state.roll(1, dims=0),
            }
            direct_product_relative = float(
                (direct_state.float() - ordered_state.float()).norm()
                / direct_state.float().norm().clamp_min(1e-12)
            )
            for condition, rolled in state_conditions.items():
                relative_error, cosine = _state_metrics(rolled, exact)
                _, logits, target = roll_forward(
                    model=model,
                    state=rolled,
                    start_age=target_age,
                    steps=steps,
                    successors=successors,
                    current=current,
                    jump=jump,
                )
                probability = logits.softmax(dim=-1).gather(1, target[:, None]).mean()
                raw_rows.append(
                    {
                        "source_age": source_age,
                        "target_age": target_age,
                        "rollback_steps": steps,
                        "batch": batch_index,
                        "condition": condition,
                        "accuracy": float(logits.argmax(dim=-1).eq(target).float().mean()),
                        "target_probability": float(probability),
                        "target_margin": float(_target_margin(logits.float(), target).mean()),
                        "state_relative_error_to_exact_H_b": relative_error,
                        "state_cosine_to_exact_H_b": cosine,
                        "direct_vs_ordered_state_relative_error": direct_product_relative,
                        "examples": batch_size,
                    }
                )
    interval_rows: list[dict[str, Any]] = []
    for source_age, target_age in direct.pairs:
        for condition in sorted({str(row["condition"]) for row in raw_rows}):
            selected = [
                row
                for row in raw_rows
                if row["source_age"] == source_age
                and row["target_age"] == target_age
                and row["condition"] == condition
            ]
            interval_rows.append(
                {
                    "source_age": source_age,
                    "target_age": target_age,
                    "rollback_steps": source_age - target_age,
                    "condition": condition,
                    "accuracy": float(np.mean([row["accuracy"] for row in selected])),
                    "target_probability": float(np.mean([row["target_probability"] for row in selected])),
                    "target_margin": float(np.mean([row["target_margin"] for row in selected])),
                    "state_relative_error_to_exact_H_b": float(
                        np.mean([row["state_relative_error_to_exact_H_b"] for row in selected])
                    ),
                    "state_cosine_to_exact_H_b": float(
                        np.mean([row["state_cosine_to_exact_H_b"] for row in selected])
                    ),
                    "direct_vs_ordered_state_relative_error": float(
                        np.mean([row["direct_vs_ordered_state_relative_error"] for row in selected])
                    ),
                    "examples": examples,
                }
            )
    length_rows: list[dict[str, Any]] = []
    conditions = sorted({str(row["condition"]) for row in interval_rows})
    for length in sorted({source - target for source, target in direct.pairs}):
        row: dict[str, Any] = {"rollback_steps": length}
        for condition in conditions:
            selected = [
                value
                for value in interval_rows
                if value["rollback_steps"] == length and value["condition"] == condition
            ]
            row[condition] = float(np.mean([value["accuracy"] for value in selected]))
        length_rows.append(row)
    return raw_rows, interval_rows, length_rows


def closure_decisions(length_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in length_rows:
        direct = float(row["direct_K"])
        ordered = float(row["ordered_J_product"])
        reversed_accuracy = float(row["reversed_J_order"])
        shuffled = float(row["batch_shuffled_ordered_product"])
        if direct < 0.90:
            decision = "inconclusive"
            reason = "direct K failed its 90% competence prerequisite"
        elif (
            ordered < 0.90
            or abs(direct - ordered) > 0.05
            or ordered - reversed_accuracy < 0.10
            or ordered - shuffled < 0.30
        ):
            decision = "refuted"
            reason = "one or more pre-registered closure/control gates failed"
        else:
            decision = "supported"
            reason = "all pre-registered closure/control gates passed"
        decisions.append(
            {
                **dict(row),
                "direct_product_absolute_gap": abs(direct - ordered),
                "reversed_order_drop": ordered - reversed_accuracy,
                "shuffled_product_drop": ordered - shuffled,
                "decision": decision,
                "reason": reason,
            }
        )
    return decisions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--adjacent-bank", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=826001)
    parser.add_argument("--train-rounds", type=int, default=96)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every-rounds", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--direct-artifact", type=Path)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.train_rounds < 1 and not args.skip_training:
        raise ValueError("train rounds must be positive")
    if args.save_every_rounds < 1:
        raise ValueError("save cadence must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase_payload["trajectory_positions_including_initial"]
    ]
    if len(phase_positions) <= 8:
        raise ValueError("phase summary does not cover H1..H8")
    adjacent, adjacent_payload = load_adjacent_bank(
        args.adjacent_bank, dimension=cfg.d_model, device=device
    )
    positions = tuple(range(cfg.seq_len))
    training_rows: list[dict[str, Any]] = []
    if args.skip_training:
        artifact = args.direct_artifact or (args.out_dir / "direct_interval_k_bank.pt")
        direct, direct_payload = load_direct_bank(
            artifact, dimension=cfg.d_model, device=device
        )
    else:
        direct = DirectIntervalJBank(dimension=cfg.d_model).to(device)
        training_rows = train_direct_bank(
            model=model,
            cfg=cfg,
            direct=direct,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            args=args,
        )
        direct, direct_payload = load_direct_bank(
            args.out_dir / "direct_interval_k_bank.pt",
            dimension=cfg.d_model,
            device=device,
        )
    raw_rows, interval_rows, length_rows = evaluate_closure(
        model=model,
        cfg=cfg,
        adjacent=adjacent,
        direct=direct,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        examples=args.eval_examples,
        batch_size=args.eval_batch_size,
        seed=args.seed + 1,
    )
    decisions = closure_decisions(length_rows)
    _write_csv(args.out_dir / "closure_raw_batches.csv", raw_rows)
    _write_csv(args.out_dir / "closure_by_interval.csv", interval_rows)
    _write_csv(args.out_dir / "closure_by_length.csv", decisions)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "adjacent_bank": str(args.adjacent_bank),
        "adjacent_bank_architecture": adjacent_payload.get("architecture"),
        "adjacent_bank_training_loss": adjacent_payload.get("training_loss"),
        "direct_bank": str(args.direct_artifact or args.out_dir / "direct_interval_k_bank.pt"),
        "direct_initialization": direct_payload.get("initialization"),
        "direct_training_loss": direct_payload.get("training_loss"),
        "direct_hidden_state_loss_weight": direct_payload.get("hidden_state_loss_weight"),
        "direct_parameter_count": direct.parameter_count,
        "training_batch_count": len(training_rows),
        "evaluation_examples_per_interval": args.eval_examples,
        "closure_decisions": decisions,
        "global_closure_decision": (
            "supported"
            if all(row["decision"] == "supported" for row in decisions)
            else (
                "inconclusive"
                if any(row["decision"] == "inconclusive" for row in decisions)
                else "refuted"
            )
        ),
        "claim_boundary": "local task-aware affine closure inside rollback lengths 2..5 on seed0 only",
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
