from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    TransformerBlock,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    cache_raw_states,
    logits_from_raw_state,
)


PHASE_POSITION = (0, 2, 4, 6, 8, 8, 8, 8, 8)


@dataclass(frozen=True)
class PairBatch:
    source: torch.Tensor
    target: torch.Tensor
    target_post: torch.Tensor
    current: torch.Tensor
    next_target: torch.Tensor
    source_age: torch.Tensor


def _layernorm_chart(
    value: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor,
    minimum_radius: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit-sphere coordinate q plus the source LN radial statistics."""
    if bool((weight.abs() < 1e-7).any()):
        raise ValueError("Post-Norm LayerNorm has a near-zero learned scale")
    standardized = (value - bias) / weight
    center = standardized.mean(dim=-1, keepdim=True)
    centered = standardized - center
    radius = centered.square().mean(dim=-1, keepdim=True).sqrt()
    q = centered / radius.clamp_min(minimum_radius)
    return q, center, radius


def project_layernorm_tangent(
    q: torch.Tensor,
    proposal: torch.Tensor,
) -> torch.Tensor:
    """Project onto vectors orthogonal to both 1 and q for every token."""
    if q.shape != proposal.shape:
        raise ValueError("q and proposal must have the same shape")
    centered = proposal - proposal.mean(dim=-1, keepdim=True)
    return centered - q * (centered * q).mean(dim=-1, keepdim=True)


def retract_unit_layernorm_sphere(
    q: torch.Tensor,
    tangent: torch.Tensor,
    *,
    minimum_radius: float = 1e-6,
) -> torch.Tensor:
    candidate = q + tangent
    candidate = candidate - candidate.mean(dim=-1, keepdim=True)
    radius = candidate.square().mean(dim=-1, keepdim=True).sqrt()
    return candidate / radius.clamp_min(minimum_radius)


class PostNormTangentMap(nn.Module):
    """A global dxd ambient generator followed by tangent projection/retraction."""

    def __init__(self, layernorm: nn.LayerNorm) -> None:
        super().__init__()
        if len(layernorm.normalized_shape) != 1:
            raise ValueError("expected a one-dimensional LayerNorm")
        feature_count = int(layernorm.normalized_shape[0])
        self.matrix = nn.Parameter(torch.zeros(feature_count, feature_count))
        self.register_buffer("weight", layernorm.weight.detach().clone())
        self.register_buffer("bias", layernorm.bias.detach().clone())

    def transform_values(self, value: torch.Tensor) -> torch.Tensor:
        q, center, radius = _layernorm_chart(
            value,
            weight=self.weight,
            bias=self.bias,
        )
        proposal = q @ self.matrix
        tangent = project_layernorm_tangent(q, proposal)
        q_new = retract_unit_layernorm_sphere(q, tangent)
        return (center + radius * q_new) * self.weight + self.bias

    def forward(self, state: torch.Tensor, *, position_mode: str) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError("state must have [batch, position, feature] shape")
        if position_mode == "all":
            return self.transform_values(state)
        if position_mode == "answer":
            result = state.clone()
            result[:, -1] = self.transform_values(state[:, -1])
            return result
        raise ValueError(f"unsupported position mode: {position_mode}")


class EuclideanResidualMap(nn.Module):
    """Matched dxd residual-linear control, initialized to the identity map."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.zeros(feature_count, feature_count))

    def transform_values(self, value: torch.Tensor) -> torch.Tensor:
        return value + value @ self.matrix

    def forward(self, state: torch.Tensor, *, position_mode: str) -> torch.Tensor:
        if position_mode == "all":
            return self.transform_values(state)
        if position_mode == "answer":
            result = state.clone()
            result[:, -1] = self.transform_values(state[:, -1])
            return result
        raise ValueError(f"unsupported position mode: {position_mode}")


Rejuvenator = PostNormTangentMap | EuclideanResidualMap


def _gather_age(states: Sequence[torch.Tensor], ages: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack(tuple(states), dim=1)
    batch_index = torch.arange(ages.shape[0], device=ages.device)
    return stacked[batch_index, ages]


@torch.no_grad()
def collect_pair_batch(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch_size: int,
    device: torch.device,
    pair_mode: str,
) -> PairBatch:
    tokens, _, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth + 2,
    )
    source_states = [
        model.token_embed(tokens) + model.pos_embed.unsqueeze(0),
        *cache_raw_states(model, tokens, max_loop=cfg.max_loops),
    ]
    if pair_mode == "adjacent":
        source_age = torch.randint(
            2,
            cfg.max_loops + 1,
            (batch_size,),
            device=device,
        )
        source_position = torch.tensor(
            PHASE_POSITION,
            device=device,
            dtype=torch.long,
        )[source_age]
        target_age = source_age - 1
        target_position = torch.tensor(
            PHASE_POSITION,
            device=device,
            dtype=torch.long,
        )[target_age]
        start_shift = source_position - target_position
    elif pair_mode in {"direct", "terminal_reset"}:
        source_age = (
            torch.full(
                (batch_size,),
                cfg.max_loops,
                device=device,
                dtype=torch.long,
            )
            if pair_mode == "direct"
            else torch.randint(
                4,
                cfg.max_loops + 1,
                (batch_size,),
                device=device,
            )
        )
        target_age = torch.full(
            (batch_size,),
            3,
            device=device,
            dtype=torch.long,
        )
        source_position = torch.full(
            (batch_size,),
            cfg.max_depth,
            device=device,
            dtype=torch.long,
        )
        start_shift = torch.full(
            (batch_size,),
            cfg.max_depth - PHASE_POSITION[3],
            device=device,
            dtype=torch.long,
        )
    else:
        raise ValueError(f"unsupported pair mode: {pair_mode}")

    shifted_start = start
    for step in range(int(start_shift.max())):
        active = start_shift > step
        advanced = successors.gather(1, shifted_start[:, None]).squeeze(1)
        shifted_start = torch.where(active, advanced, shifted_start)
    target_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth + 2,
        successors=successors,
        start=shifted_start,
    )
    target_states = [
        model.token_embed(target_tokens) + model.pos_embed.unsqueeze(0),
        *cache_raw_states(model, target_tokens, max_loop=cfg.max_loops),
    ]
    source = _gather_age(source_states, source_age)
    target = _gather_age(target_states, target_age)
    target_post = _gather_age(target_states, target_age + 1)
    current = start
    for position in range(cfg.max_depth):
        advanced = successors.gather(1, current[:, None]).squeeze(1)
        current = torch.where(source_position > position, advanced, current)
    phase_table = torch.tensor(
        PHASE_POSITION,
        device=device,
        dtype=torch.long,
    )
    post_position = start_shift + phase_table[target_age + 1]
    next_target = start
    for position in range(int(post_position.max())):
        advanced = successors.gather(1, next_target[:, None]).squeeze(1)
        next_target = torch.where(post_position > position, advanced, next_target)
    return PairBatch(
        source=source,
        target=target,
        target_post=target_post,
        current=current,
        next_target=next_target,
        source_age=source_age,
    )


def _selected_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    position_mode: str,
) -> torch.Tensor:
    if position_mode == "answer":
        prediction = prediction[:, -1:]
        target = target[:, -1:]
        weights = torch.ones(1, device=prediction.device)
    elif position_mode == "all":
        weights = torch.ones(prediction.shape[1], device=prediction.device)
        weights[-4:-1] = 4.0
        weights[-1] = 8.0
    else:
        raise ValueError(f"unsupported position mode: {position_mode}")
    squared = (prediction - target).square().mean(dim=-1)
    denominator = (
        target - target.mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-5)
    return (squared * weights).sum() / weights.sum() / denominator


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


def closed_loop_training_loss(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    rejuvenator: Rejuvenator,
    position_mode: str,
    batch_size: int,
    cycles: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if cycles < 1:
        raise ValueError("closed-loop training requires at least one cycle")
    with torch.no_grad():
        tokens, targets, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * cycles,
        )
        state = cache_raw_states(
            model,
            tokens,
            max_loop=cfg.max_loops,
        )[-1]
    current = targets[:, cfg.max_depth - 1]
    cycle_losses = []
    post_accuracies = []
    pre_accuracies = []
    for cycle in range(1, cycles + 1):
        mapped = rejuvenator(state, position_mode=position_mode)
        pre_logits = logits_from_raw_state(model, mapped)
        pre_loss = F.cross_entropy(pre_logits, current)
        pre_accuracies.append(
            pre_logits.argmax(dim=-1).eq(current).float().mean()
        )
        state = apply_shared_stack(
            model,
            mapped,
            loop_index=cfg.max_loops + cycle - 1,
        )
        target = targets[:, cfg.max_depth + 2 * cycle - 1]
        post_logits = logits_from_raw_state(model, state)
        post_loss = F.cross_entropy(post_logits, target)
        post_accuracies.append(
            post_logits.argmax(dim=-1).eq(target).float().mean()
        )
        cycle_losses.append(0.5 * pre_loss + post_loss)
        current = target
    return torch.stack(cycle_losses).mean(), {
        "rollout_pre_accuracy": torch.stack(pre_accuracies).mean(),
        "rollout_post_accuracy": torch.stack(post_accuracies).mean(),
        "rollout_final_accuracy": post_accuracies[-1],
    }


def _masked_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    previous: torch.Tensor,
) -> tuple[float, int]:
    valid = target.ne(previous)
    if not bool(valid.any()):
        return float("nan"), 0
    return (
        float(logits[valid].argmax(dim=-1).eq(target[valid]).float().mean()),
        int(valid.sum()),
    )


@torch.no_grad()
def evaluate_pairs(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    rejuvenator: Rejuvenator,
    position_mode: str,
    pair_mode: str,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    set_seed(seed)
    buckets: dict[int, dict[str, float]] = {}
    for _ in range(batches):
        pair = collect_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=batch_size,
            device=device,
            pair_mode=pair_mode,
        )
        mapped = rejuvenator(pair.source, position_mode=position_mode)
        post = apply_shared_stack(model, mapped, loop_index=cfg.max_loops)
        for age in pair.source_age.unique():
            age_int = int(age)
            mask = pair.source_age.eq(age)
            bucket = buckets.setdefault(
                age_int,
                {
                    "count": 0.0,
                    "relative_mse": 0.0,
                    "pre_current": 0.0,
                    "post_next": 0.0,
                },
            )
            count = int(mask.sum())
            numerator = (mapped[mask] - pair.target[mask]).square().mean()
            denominator = (
                pair.target[mask]
                - pair.target[mask].mean(dim=0, keepdim=True)
            ).square().mean().clamp_min(1e-5)
            bucket["count"] += count
            bucket["relative_mse"] += float(numerator / denominator) * count
            bucket["pre_current"] += (
                _accuracy(
                    logits_from_raw_state(model, mapped[mask]),
                    pair.current[mask],
                )
                * count
            )
            bucket["post_next"] += (
                _accuracy(
                    logits_from_raw_state(model, post[mask]),
                    pair.next_target[mask],
                )
                * count
            )
    return [
        {
            "source_age": age,
            "count": int(bucket["count"]),
            "relative_mse": bucket["relative_mse"] / bucket["count"],
            "pre_current_accuracy": bucket["pre_current"] / bucket["count"],
            "post_next_accuracy": bucket["post_next"] / bucket["count"],
        }
        for age, bucket in sorted(buckets.items())
    ]


@torch.no_grad()
def evaluate_repeated_rejuvenation(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    rejuvenator: Rejuvenator,
    position_mode: str,
    pair_mode: str,
    batch_size: int,
    batches: int,
    cycles: int,
    device: torch.device,
    seed: int,
    conditions: Sequence[str] = ("learned", "identity", "reverse"),
) -> list[dict[str, Any]]:
    set_seed(seed)
    allowed_conditions = {"learned", "identity", "reverse"}
    if not conditions or not set(conditions).issubset(allowed_conditions):
        raise ValueError("unsupported repeated-evaluation conditions")
    conditions = tuple(conditions)
    totals = {
        (condition, cycle): {
            "count": 0,
            "correct": 0.0,
            "motion_count": 0,
            "motion_correct": 0.0,
            "strict_count": 0,
            "strict_correct": 0.0,
            "pre_correct": 0.0,
        }
        for condition in conditions
        for cycle in range(1, cycles + 1)
    }
    for _ in range(batches):
        tokens, targets, _, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * cycles,
        )
        terminal = cache_raw_states(
            model,
            tokens,
            max_loop=cfg.max_loops,
        )[-1]
        states = {condition: terminal.clone() for condition in conditions}
        original_endpoint = targets[:, cfg.max_depth - 1]
        previous = original_endpoint
        for cycle in range(1, cycles + 1):
            target = targets[:, cfg.max_depth + 2 * cycle - 1]
            applications = (
                cfg.max_loops - 3
                if pair_mode == "adjacent" and cycle == 1
                else 1
            )
            for condition in conditions:
                state_before = states[condition]
                if condition != "identity":
                    for _application in range(applications):
                        mapped = rejuvenator(
                            state_before,
                            position_mode=position_mode,
                        )
                        state_before = (
                            mapped
                            if condition == "learned"
                            else 2.0 * state_before - mapped
                        )
                totals[(condition, cycle)]["pre_correct"] += (
                    logits_from_raw_state(model, state_before)
                    .argmax(dim=-1)
                    .eq(previous)
                    .float()
                    .sum()
                    .item()
                )
                states[condition] = apply_shared_stack(
                    model,
                    state_before,
                    loop_index=cfg.max_loops + cycle - 1,
                )
                logits = logits_from_raw_state(model, states[condition])
                total = totals[(condition, cycle)]
                total["count"] += batch_size
                total["correct"] += (
                    logits.argmax(dim=-1).eq(target).float().sum().item()
                )
                motion_accuracy, motion_count = _masked_accuracy(
                    logits,
                    target,
                    previous,
                )
                total["motion_count"] += motion_count
                if motion_count:
                    total["motion_correct"] += motion_accuracy * motion_count
                strict_valid = target.ne(previous) & target.ne(original_endpoint)
                strict_count = int(strict_valid.sum())
                total["strict_count"] += strict_count
                if strict_count:
                    total["strict_correct"] += (
                        logits[strict_valid]
                        .argmax(dim=-1)
                        .eq(target[strict_valid])
                        .float()
                        .sum()
                        .item()
                    )
            previous = target
    rows = []
    for condition in conditions:
        for cycle in range(1, cycles + 1):
            total = totals[(condition, cycle)]
            rows.append(
                {
                    "condition": condition,
                    "cycle": cycle,
                    "path_position": cfg.max_depth + 2 * cycle,
                    "pre_current_accuracy": total["pre_correct"] / total["count"],
                    "post_target_accuracy": total["correct"] / total["count"],
                    "motion_controlled_accuracy": (
                        total["motion_correct"] / total["motion_count"]
                        if total["motion_count"]
                        else float("nan")
                    ),
                    "valid_motion_count": total["motion_count"],
                    "strict_novel_target_accuracy": (
                        total["strict_correct"] / total["strict_count"]
                        if total["strict_count"]
                        else float("nan")
                    ),
                    "valid_strict_novel_target_count": total["strict_count"],
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_rejuvenator(
    model: LoopedGraphPathTransformer,
    *,
    transform: str,
) -> Rejuvenator:
    if transform == "tangent":
        block = model.blocks[-1]
        if not isinstance(block, TransformerBlock):
            raise TypeError("tangent experiment requires a legacy TransformerBlock")
        if block.inner_norm_style != "post_layernorm":
            raise ValueError("tangent experiment requires a Post-Norm checkpoint")
        if not isinstance(block.ln_2, nn.LayerNorm):
            raise TypeError("last Post-Norm module must be LayerNorm")
        return PostNormTangentMap(block.ln_2)
    if transform == "euclidean":
        return EuclideanResidualMap(model.cfg.d_model)
    raise ValueError(f"unsupported transform: {transform}")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    if (
        cfg.node_count != 8
        or cfg.max_depth != 8
        or cfg.max_loops != 8
        or cfg.n_layers != 2
        or cfg.inner_norm_style != "post_layernorm"
    ):
        raise ValueError("expected the Post-Norm N8 D8 L8 two-block checkpoint")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rejuvenator = _make_rejuvenator(model, transform=args.transform).to(device)
    optimizer = torch.optim.AdamW(
        rejuvenator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    history = []
    for step in range(1, args.steps + 1):
        pair = collect_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=args.batch_size,
            device=device,
            pair_mode=args.pair_mode,
        )
        mapped = rejuvenator(pair.source, position_mode=args.position_mode)
        post = apply_shared_stack(model, mapped, loop_index=cfg.max_loops)
        state_loss = _selected_state_loss(
            mapped,
            pair.target,
            position_mode=args.position_mode,
        )
        post_state_loss = _selected_state_loss(
            post,
            pair.target_post,
            position_mode=args.position_mode,
        )
        current_ce = F.cross_entropy(
            logits_from_raw_state(model, mapped),
            pair.current,
        )
        next_ce = F.cross_entropy(
            logits_from_raw_state(model, post),
            pair.next_target,
        )
        rollout_loss = torch.zeros((), device=device)
        rollout_metrics = {
            "rollout_pre_accuracy": torch.ones((), device=device),
            "rollout_post_accuracy": torch.ones((), device=device),
            "rollout_final_accuracy": torch.ones((), device=device),
        }
        if args.rollout_train_cycles:
            rollout_loss, rollout_metrics = closed_loop_training_loss(
                model=model,
                cfg=cfg,
                rejuvenator=rejuvenator,
                position_mode=args.position_mode,
                batch_size=args.batch_size,
                cycles=args.rollout_train_cycles,
                device=device,
            )
        loss = (
            args.state_weight * state_loss
            + args.post_state_weight * post_state_loss
            + args.current_ce_weight * current_ce
            + args.next_ce_weight * next_ce
            + args.rollout_weight * rollout_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            rejuvenator.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        if step == 1 or step % args.print_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "state_loss": float(state_loss.detach()),
                "post_state_loss": float(post_state_loss.detach()),
                "current_ce": float(current_ce.detach()),
                "next_ce": float(next_ce.detach()),
                "rollout_loss": float(rollout_loss.detach()),
                "rollout_pre_accuracy": float(
                    rollout_metrics["rollout_pre_accuracy"].detach()
                ),
                "rollout_post_accuracy": float(
                    rollout_metrics["rollout_post_accuracy"].detach()
                ),
                "rollout_final_accuracy": float(
                    rollout_metrics["rollout_final_accuracy"].detach()
                ),
                "train_pre_current_accuracy": _accuracy(
                    logits_from_raw_state(model, mapped),
                    pair.current,
                ),
                "train_post_next_accuracy": _accuracy(
                    logits_from_raw_state(model, post),
                    pair.next_target,
                ),
                "matrix_frobenius_norm": float(
                    rejuvenator.matrix.detach().norm()
                ),
                "gradient_norm": float(gradient_norm),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    pair_rows = evaluate_pairs(
        model=model,
        cfg=cfg,
        rejuvenator=rejuvenator,
        position_mode=args.position_mode,
        pair_mode=args.pair_mode,
        batch_size=args.eval_batch_size,
        batches=args.eval_batches,
        device=device,
        seed=args.eval_seed,
    )
    repeated_rows = evaluate_repeated_rejuvenation(
        model=model,
        cfg=cfg,
        rejuvenator=rejuvenator,
        position_mode=args.position_mode,
        pair_mode=args.pair_mode,
        batch_size=args.eval_batch_size,
        batches=args.eval_batches,
        cycles=args.eval_cycles,
        device=device,
        seed=args.eval_seed + 1,
    )
    learned_rows = [
        row for row in repeated_rows if row["condition"] == "learned"
    ]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": asdict(cfg),
        "loss_placement": (
            "frozen base model; global rejuvenator trained with aligned hidden-state "
            "loss plus current and one-loop functional CE"
        ),
        "trained_base_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "trained_effective_depth": cfg.max_loops * cfg.n_layers,
        "transform": args.transform,
        "position_mode": args.position_mode,
        "pair_mode": args.pair_mode,
        "train_source_ages": (
            list(range(2, 9))
            if args.pair_mode == "adjacent"
            else (list(range(4, 9)) if args.pair_mode == "terminal_reset" else [8])
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in rejuvenator.parameters()
        ),
        "matrix_frobenius_norm": float(rejuvenator.matrix.detach().norm()),
        "pair_rows": pair_rows,
        "learned_repeated_min_accuracy": min(
            row["post_target_accuracy"] for row in learned_rows
        ),
        "learned_repeated_final_accuracy": learned_rows[-1][
            "post_target_accuracy"
        ],
        "learned_repeated_rows": learned_rows,
        "device": str(device),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "training_history.csv", history)
    _write_csv(args.out_dir / "pair_evaluation.csv", pair_rows)
    _write_csv(args.out_dir / "repeated_rejuvenation.csv", repeated_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "summary": summary,
            "matrix": rejuvenator.matrix.detach().cpu(),
            "state_dict": rejuvenator.state_dict(),
        },
        args.out_dir / "rejuvenator.pt",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a global Post-Norm tangent-space rejuvenator."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--transform",
        choices=("tangent", "euclidean"),
        default="tangent",
    )
    parser.add_argument(
        "--position-mode",
        choices=("answer", "all"),
        default="answer",
    )
    parser.add_argument(
        "--pair-mode",
        choices=("adjacent", "direct", "terminal_reset"),
        default="adjacent",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--post-state-weight", type=float, default=0.25)
    parser.add_argument("--current-ce-weight", type=float, default=0.5)
    parser.add_argument("--next-ce-weight", type=float, default=1.0)
    parser.add_argument("--rollout-train-cycles", type=int, default=0)
    parser.add_argument("--rollout-weight", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-cycles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--eval-seed", type=int, default=9731)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_experiment(parse_args(argv))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
