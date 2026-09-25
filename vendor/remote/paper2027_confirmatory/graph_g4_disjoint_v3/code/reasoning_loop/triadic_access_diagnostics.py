from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from reasoning_loop.triadic_shortage import (
    TriadicShortageModel,
    all_triples,
    hybrid_sum_target,
    make_visibility_mask,
)
from reasoning_loop.triadic_shortage_circuit import run_intervened
from reasoning_loop.triadic_shortage_diagnostics import load_checkpoint


def build_access_visibility(
    kind: str,
    batch_size: int,
    loops: int,
    device: torch.device,
    *,
    count: int | None = None,
    loop_index: int | None = None,
    order: Sequence[int] | None = None,
) -> torch.Tensor:
    if batch_size < 1 or loops < 1:
        raise ValueError("batch_size and loops must be positive")
    visibility = torch.zeros(
        (batch_size, loops, 3),
        dtype=torch.bool,
        device=device,
    )
    if kind in {"prefix", "suffix"}:
        if count is None or not 1 <= count <= loops:
            raise ValueError("count must lie in [1, loops]")
        if kind == "prefix":
            visibility[:, :count] = True
        else:
            visibility[:, loops - count :] = True
        return visibility
    if kind == "omit":
        if loop_index is None or not 0 <= loop_index < loops:
            raise ValueError("loop_index must index a configured loop")
        visibility.fill_(True)
        visibility[:, loop_index] = False
        return visibility
    if kind == "order":
        if loops < 3:
            raise ValueError("order visibility requires at least three loops")
        if order is None or tuple(sorted(order)) != (0, 1, 2):
            raise ValueError("order must be a permutation of (0, 1, 2)")
        for reveal_loop, operand_index in enumerate(order):
            visibility[:, reveal_loop, operand_index] = True
        return visibility
    raise ValueError(f"unknown access visibility kind: {kind}")


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _final_accuracy(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    visibility_builder: Callable[[int], torch.Tensor],
    *,
    indices: torch.Tensor,
    batch_size: int,
    reset_before: torch.Tensor | None = None,
    mode_id: int | None = None,
) -> float:
    device = _model_device(model)
    operands = operands.to(device)
    labels = labels.to(device)
    indices = indices.to(device)
    correct = 0
    total = 0
    model.eval()
    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start : start + batch_size]
        visibility = visibility_builder(batch_indices.numel())
        mode_ids = (
            torch.full(
                (batch_indices.numel(),),
                mode_id,
                dtype=torch.long,
                device=device,
            )
            if mode_id is not None
            else None
        )
        logits, _ = model(
            operands[batch_indices],
            visibility,
            reset_before=reset_before,
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
    return correct / total


@torch.no_grad()
def evaluate_visibility_sweep(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    mode_id: int | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = _model_device(model)
    rows: list[dict[str, Any]] = []

    def add_row(
        name: str,
        builder: Callable[[int], torch.Tensor],
    ) -> None:
        rows.append(
            {
                "name": name,
                "accuracy": _final_accuracy(
                    model,
                    operands,
                    labels,
                    builder,
                    indices=indices,
                    batch_size=batch_size,
                    mode_id=mode_id,
                ),
                "examples": int(indices.numel()),
            }
        )

    for count in range(1, model.cfg.loops + 1):
        add_row(
            f"prefix_{count}",
            lambda current_batch, count=count: build_access_visibility(
                "prefix",
                current_batch,
                model.cfg.loops,
                device,
                count=count,
            ),
        )
        add_row(
            f"suffix_{count}",
            lambda current_batch, count=count: build_access_visibility(
                "suffix",
                current_batch,
                model.cfg.loops,
                device,
                count=count,
            ),
        )
    for loop_index in range(model.cfg.loops):
        add_row(
            f"omit_{loop_index + 1}",
            lambda current_batch, loop_index=loop_index: build_access_visibility(
                "omit",
                current_batch,
                model.cfg.loops,
                device,
                loop_index=loop_index,
            ),
        )
    for order in itertools.permutations(range(3)):
        add_row(
            "order_" + "".join(str(index) for index in order),
            lambda current_batch, order=order: build_access_visibility(
                "order",
                current_batch,
                model.cfg.loops,
                device,
                order=order,
            ),
        )
    return {"rows": rows}


@torch.no_grad()
def evaluate_reset_map(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    *,
    condition: str,
    seed: int = 121_001,
    mode_id: int | None = None,
) -> dict[str, Any]:
    device = _model_device(model)
    operands = operands.to(device)
    labels = labels.to(device)

    def accuracy(reset_before: torch.Tensor | None) -> float:
        generator = torch.Generator(device=device).manual_seed(seed)
        visibility = make_visibility_mask(
            condition,
            operands.shape[0],
            model.cfg.loops,
            device,
            generator=generator,
        )
        logits, _ = model(
            operands,
            visibility,
            reset_before=reset_before,
            mode_ids=(
                torch.full(
                    (operands.shape[0],),
                    mode_id,
                    dtype=torch.long,
                    device=device,
                )
                if mode_id is not None
                else None
            ),
        )
        return float(
            logits[:, -1]
            .argmax(dim=-1)
            .eq(labels)
            .float()
            .mean()
            .item()
        )

    baseline = accuracy(None)
    individual_rows = []
    prefix_rows = []
    for loop_index in range(model.cfg.loops):
        individual = torch.zeros(
            model.cfg.loops,
            dtype=torch.bool,
            device=device,
        )
        individual[loop_index] = True
        individual_rows.append(
            {
                "loop": loop_index + 1,
                "reset_before": individual.cpu().tolist(),
                "accuracy": accuracy(individual),
            }
        )
        prefix = torch.zeros(
            model.cfg.loops,
            dtype=torch.bool,
            device=device,
        )
        prefix[: loop_index + 1] = True
        prefix_rows.append(
            {
                "prefix_length": loop_index + 1,
                "reset_before": prefix.cpu().tolist(),
                "accuracy": accuracy(prefix),
            }
        )
    return {
        "condition": condition,
        "examples": operands.shape[0],
        "baseline_accuracy": baseline,
        "individual_rows": individual_rows,
        "prefix_rows": prefix_rows,
    }


def _target_accuracies(
    logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float | None]:
    prediction = logits[:, -1].argmax(dim=-1)
    if mask is not None:
        mask = mask.to(device=prediction.device, dtype=torch.bool)
        if mask.shape != prediction.shape:
            raise ValueError("accuracy mask must have shape [batch]")
        if not mask.any():
            return {name: None for name in targets}
        prediction = prediction[mask]
    return {
        name: float(
            prediction.eq(target[mask] if mask is not None else target)
            .float()
            .mean()
            .item()
        )
        for name, target in targets.items()
    }


@torch.no_grad()
def evaluate_state_transplant(
    model: TriadicShortageModel,
    *,
    donor_operands: torch.Tensor,
    receiver_operands: torch.Tensor,
    source_condition: str,
    target_condition: str,
    transplant_loops: Sequence[int],
    seed: int = 131_001,
    source_mode_id: int | None = None,
    target_mode_id: int | None = None,
) -> dict[str, Any]:
    if donor_operands.shape != receiver_operands.shape:
        raise ValueError("donor and receiver operands must have the same shape")
    device = _model_device(model)
    donor_operands = donor_operands.to(device)
    receiver_operands = receiver_operands.to(device)
    source_visibility = make_visibility_mask(
        source_condition,
        donor_operands.shape[0],
        model.cfg.loops,
        device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    target_visibility = make_visibility_mask(
        target_condition,
        receiver_operands.shape[0],
        model.cfg.loops,
        device,
        generator=torch.Generator(device=device).manual_seed(seed + 1),
    )
    donor_mode_ids = (
        torch.full(
            (donor_operands.shape[0],),
            source_mode_id,
            dtype=torch.long,
            device=device,
        )
        if source_mode_id is not None
        else None
    )
    target_mode_ids = (
        torch.full(
            (receiver_operands.shape[0],),
            target_mode_id,
            dtype=torch.long,
            device=device,
        )
        if target_mode_id is not None
        else None
    )
    _, donor_states = model(
        donor_operands,
        source_visibility,
        mode_ids=donor_mode_ids,
    )
    donor_target = donor_operands.sum(dim=1).remainder(model.cfg.p)
    receiver_target = receiver_operands.sum(dim=1).remainder(model.cfg.p)
    rows: list[dict[str, Any]] = []
    permutation = torch.roll(
        torch.arange(donor_operands.shape[0], device=device),
        shifts=1,
    )
    for transplant_loop in transplant_loops:
        if not 1 <= transplant_loop < model.cfg.loops:
            raise ValueError(
                "transplant loops must leave at least one continuation loop"
            )
        visited = source_visibility[:, :transplant_loop].any(dim=1)
        future_receiver = target_visibility[:, transplant_loop:].any(dim=1)
        fully_covered = (visited | future_receiver).all(dim=1)
        overlap = (visited & future_receiver).any(dim=1)
        hybrid_target = hybrid_sum_target(
            donor_operands,
            receiver_operands,
            visited,
            p=model.cfg.p,
        )
        targets = {
            "donor": donor_target,
            "receiver": receiver_target,
            "hybrid": hybrid_target,
        }
        workspace = donor_states[:, transplant_loop - 1]
        transplanted_logits, _ = model.continue_from_workspace(
            receiver_operands,
            target_visibility,
            workspace=workspace,
            start_loop=transplant_loop,
            mode_ids=target_mode_ids,
        )
        random_logits, _ = model.continue_from_workspace(
            receiver_operands,
            target_visibility,
            workspace=workspace[permutation],
            start_loop=transplant_loop,
            mode_ids=target_mode_ids,
        )
        rows.append(
            {
                "transplant_loop": transplant_loop,
                "fully_covered_examples": int(fully_covered.sum().item()),
                "fully_covered_fraction": float(
                    fully_covered.float().mean().item()
                ),
                "overlap_fraction": float(overlap.float().mean().item()),
                "visited_operand_preview": visited[
                    : min(8, visited.shape[0])
                ].cpu().tolist(),
                "accuracy": _target_accuracies(transplanted_logits, targets),
                "covered_accuracy": _target_accuracies(
                    transplanted_logits,
                    targets,
                    mask=fully_covered,
                ),
                "random_workspace_accuracy": _target_accuracies(
                    random_logits,
                    targets,
                ),
                "covered_random_workspace_accuracy": _target_accuracies(
                    random_logits,
                    targets,
                    mask=fully_covered,
                ),
            }
        )
    return {
        "source_condition": source_condition,
        "target_condition": target_condition,
        "examples": donor_operands.shape[0],
        "rows": rows,
    }


@torch.no_grad()
def evaluate_component_transfer(
    model: TriadicShortageModel,
    *,
    donor_operands: torch.Tensor,
    receiver_operands: torch.Tensor,
    condition: str,
    seed: int = 141_001,
    mode_id: int | None = None,
) -> dict[str, Any]:
    if donor_operands.shape != receiver_operands.shape:
        raise ValueError("donor and receiver operands must have the same shape")
    device = _model_device(model)
    donor_operands = donor_operands.to(device)
    receiver_operands = receiver_operands.to(device)
    visibility = make_visibility_mask(
        condition,
        donor_operands.shape[0],
        model.cfg.loops,
        device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    mode_ids = (
        torch.full(
            (donor_operands.shape[0],),
            mode_id,
            dtype=torch.long,
            device=device,
        )
        if mode_id is not None
        else None
    )
    donor_logits, _, donor_trace = run_intervened(
        model,
        donor_operands,
        visibility,
        mode_ids=mode_ids,
        return_trace=True,
    )
    receiver_logits, _, _ = run_intervened(
        model,
        receiver_operands,
        visibility,
        mode_ids=mode_ids,
    )
    donor_target = donor_operands.sum(dim=1).remainder(model.cfg.p)
    receiver_target = receiver_operands.sum(dim=1).remainder(model.cfg.p)
    permutation = torch.roll(
        torch.arange(donor_operands.shape[0], device=device),
        shifts=1,
    )
    rows = []
    for loop_index in range(model.cfg.loops):
        visited = visibility[:, loop_index]
        hybrid_target = hybrid_sum_target(
            donor_operands,
            receiver_operands,
            visited,
            p=model.cfg.p,
        )
        targets = {
            "donor": donor_target,
            "receiver": receiver_target,
            "hybrid": hybrid_target,
        }
        attention_update = donor_trace[f"loop{loop_index}.attention_out"]
        mlp_update = donor_trace[f"loop{loop_index}.mlp_out"]
        attention_logits, _, _ = run_intervened(
            model,
            receiver_operands,
            visibility,
            attention_patch={loop_index: attention_update},
            mode_ids=mode_ids,
        )
        mlp_logits, _, _ = run_intervened(
            model,
            receiver_operands,
            visibility,
            mlp_patch={loop_index: mlp_update},
            mode_ids=mode_ids,
        )
        random_attention_logits, _, _ = run_intervened(
            model,
            receiver_operands,
            visibility,
            attention_patch={loop_index: attention_update[permutation]},
            mode_ids=mode_ids,
        )
        random_mlp_logits, _, _ = run_intervened(
            model,
            receiver_operands,
            visibility,
            mlp_patch={loop_index: mlp_update[permutation]},
            mode_ids=mode_ids,
        )
        rows.append(
            {
                "loop": loop_index + 1,
                "visible_operand_preview": visited[
                    : min(8, visited.shape[0])
                ].cpu().tolist(),
                "attention_patch_accuracy": _target_accuracies(
                    attention_logits,
                    targets,
                ),
                "mlp_patch_accuracy": _target_accuracies(
                    mlp_logits,
                    targets,
                ),
                "random_attention_patch_accuracy": _target_accuracies(
                    random_attention_logits,
                    targets,
                ),
                "random_mlp_patch_accuracy": _target_accuracies(
                    random_mlp_logits,
                    targets,
                ),
            }
        )
    return {
        "condition": condition,
        "examples": donor_operands.shape[0],
        "donor_accuracy": _target_accuracies(
            donor_logits,
            {"donor": donor_target},
        )["donor"],
        "receiver_accuracy": _target_accuracies(
            receiver_logits,
            {"receiver": receiver_target},
        )["receiver"],
        "rows": rows,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


@torch.no_grad()
def analyze_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    sample_size: int = 1024,
    batch_size: int = 1024,
    target_conditions: Sequence[str] = ("full", "full_once", "sequential"),
    seed: int = 151_001,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    condition = str(payload["condition"])
    operands, labels = all_triples(model.cfg.p, device=device)
    heldout_idx = payload["split"]["heldout_idx"].long().to(device)
    subset_idx = heldout_idx[: min(sample_size, heldout_idx.numel())]
    donor = operands[subset_idx]
    receiver = donor.roll(1, dims=0)
    visibility_sweep = evaluate_visibility_sweep(
        model,
        operands,
        labels,
        heldout_idx,
        batch_size=batch_size,
    )
    reset_map = evaluate_reset_map(
        model,
        donor,
        labels[subset_idx],
        condition=condition,
        seed=seed,
    )
    transplant_loops = tuple(range(1, model.cfg.loops))
    transplants = [
        evaluate_state_transplant(
            model,
            donor_operands=donor,
            receiver_operands=receiver,
            source_condition=condition,
            target_condition=target_condition,
            transplant_loops=transplant_loops,
            seed=seed + 100 * index,
        )
        for index, target_condition in enumerate(target_conditions)
    ]
    components = evaluate_component_transfer(
        model,
        donor_operands=donor,
        receiver_operands=receiver,
        condition=condition,
        seed=seed + 10_000,
    )
    scorecard = {
        "checkpoint": str(checkpoint.resolve()),
        "condition": condition,
        "seed": int(payload["seed"]),
        "step": int(payload["step"]),
        "examples": int(subset_idx.numel()),
        "visibility_sweep": visibility_sweep,
        "reset_map": reset_map,
        "transplants": transplants,
        "components": components,
    }
    _write_json(out_dir / "visibility_sweep.json", visibility_sweep)
    _write_json(out_dir / "reset_map.json", reset_map)
    _write_json(out_dir / "state_transplants.json", transplants)
    _write_json(out_dir / "component_transfer.json", components)
    _write_json(out_dir / "scorecard.json", scorecard)
    return scorecard
