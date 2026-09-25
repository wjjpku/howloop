from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.triadic_access_diagnostics import (
    evaluate_component_transfer,
    evaluate_reset_map,
    evaluate_state_transplant,
    evaluate_visibility_sweep,
)
from reasoning_loop.triadic_mixed_schedule_train import (
    CONDITION_TO_MODE,
    MIXED_CONDITIONS,
)
from reasoning_loop.triadic_shortage import (
    TriadicShortageModel,
    all_triples,
    make_visibility_mask,
)
from reasoning_loop.triadic_shortage_diagnostics import load_checkpoint


@torch.no_grad()
def evaluate_mode_swap(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    visibility_condition: str,
    correct_mode_id: int,
    batch_size: int,
    seed: int = 161_001,
) -> dict[str, Any]:
    if model.cfg.mode_count < 1:
        raise ValueError("mode swap requires a mode-conditioned model")
    device = next(model.parameters()).device
    operands = operands.to(device)
    labels = labels.to(device)
    indices = indices.to(device)
    rows = []
    model.eval()
    for supplied_mode_id in range(model.cfg.mode_count):
        generator = torch.Generator(device=device).manual_seed(seed)
        correct = 0
        total = 0
        for start in range(0, indices.numel(), batch_size):
            batch_indices = indices[start : start + batch_size]
            visibility = make_visibility_mask(
                visibility_condition,
                batch_indices.numel(),
                model.cfg.loops,
                device,
                generator=generator,
            )
            mode_ids = torch.full(
                (batch_indices.numel(),),
                supplied_mode_id,
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(
                operands[batch_indices],
                visibility,
                mode_ids=mode_ids,
            )
            correct += int(
                logits[:, -1]
                .argmax(dim=-1)
                .eq(labels[batch_indices])
                .sum()
                .item()
            )
            total += batch_indices.numel()
        rows.append(
            {
                "supplied_mode_id": supplied_mode_id,
                "accuracy": correct / total,
                "is_correct_mode": supplied_mode_id == correct_mode_id,
            }
        )
    return {
        "visibility_condition": visibility_condition,
        "correct_mode_id": correct_mode_id,
        "examples": int(indices.numel()),
        "rows": rows,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


@torch.no_grad()
def analyze_mixed_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    sample_size: int = 1024,
    batch_size: int = 1024,
    seed: int = 171_001,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    if model.cfg.mode_count not in {0, len(MIXED_CONDITIONS)}:
        raise ValueError("checkpoint does not contain the expected mixed schedule modes")
    mode_conditioning = model.cfg.mode_count > 0
    operands, labels = all_triples(model.cfg.p, device=device)
    heldout_idx = payload["split"]["heldout_idx"].long().to(device)
    subset_idx = heldout_idx[: min(sample_size, heldout_idx.numel())]
    donor = operands[subset_idx]
    receiver = donor.roll(1, dims=0)
    conditions: dict[str, Any] = {}
    transplant_loops = tuple(range(1, model.cfg.loops))
    for condition_index, condition in enumerate(MIXED_CONDITIONS):
        mode_id = CONDITION_TO_MODE[condition] if mode_conditioning else None
        condition_dir = out_dir / condition
        transplants = [
            evaluate_state_transplant(
                model,
                donor_operands=donor,
                receiver_operands=receiver,
                source_condition=condition,
                target_condition=target_condition,
                transplant_loops=transplant_loops,
                seed=seed + 1000 * condition_index + 100 * target_index,
                source_mode_id=mode_id,
                target_mode_id=(
                    CONDITION_TO_MODE[target_condition]
                    if mode_conditioning
                    else None
                ),
            )
            for target_index, target_condition in enumerate(MIXED_CONDITIONS)
        ]
        result = {
            "mode_id": mode_id,
            "visibility_sweep": evaluate_visibility_sweep(
                model,
                operands,
                labels,
                heldout_idx,
                batch_size=batch_size,
                mode_id=mode_id,
            ),
            "reset_map": evaluate_reset_map(
                model,
                donor,
                labels[subset_idx],
                condition=condition,
                seed=seed + condition_index,
                mode_id=mode_id,
            ),
            "transplants": transplants,
            "components": evaluate_component_transfer(
                model,
                donor_operands=donor,
                receiver_operands=receiver,
                condition=condition,
                seed=seed + 10_000 + condition_index,
                mode_id=mode_id,
            ),
            "mode_swap": (
                evaluate_mode_swap(
                    model,
                    operands,
                    labels,
                    heldout_idx,
                    visibility_condition=condition,
                    correct_mode_id=CONDITION_TO_MODE[condition],
                    batch_size=batch_size,
                    seed=seed + 20_000 + condition_index,
                )
                if mode_conditioning
                else {"not_applicable": True, "rows": []}
            ),
        }
        conditions[condition] = result
        _write_json(condition_dir / "scorecard.json", result)
    scorecard = {
        "checkpoint": str(checkpoint.resolve()),
        "condition": str(payload["condition"]),
        "mode_conditioning": mode_conditioning,
        "seed": int(payload["seed"]),
        "step": int(payload["step"]),
        "examples": int(subset_idx.numel()),
        "conditions": conditions,
    }
    _write_json(out_dir / "scorecard.json", scorecard)
    return scorecard
