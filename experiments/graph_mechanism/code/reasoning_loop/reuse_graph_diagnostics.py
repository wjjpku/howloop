from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.graph_path_loop import (
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_stepwise import (
    StepwiseGraphPathConfig,
    make_stepwise_batch,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    cache_raw_states,
    logits_from_raw_state,
    rollout_targets,
)


def replace_context_state(receiver: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    if receiver.shape != donor.shape or receiver.ndim != 3:
        raise ValueError("receiver and donor must have identical [batch, seq, d_model] shapes")
    patched = receiver.clone()
    patched[:, :-1, :] = donor[:, :-1, :]
    return patched


def add_answer_noise(
    state: torch.Tensor,
    *,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError("state must have shape [batch, seq, d_model]")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    noisy = state.clone()
    noise = torch.randn(
        noisy[:, -1, :].shape,
        dtype=noisy.dtype,
        device=noisy.device,
        generator=generator,
    )
    noisy[:, -1, :] += sigma * noise
    return noisy


def _accuracy_probability(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    accuracy = logits.argmax(dim=-1).eq(target).float().mean()
    probability = logits.softmax(dim=-1).gather(1, target[:, None]).squeeze(1).mean()
    return accuracy, probability


@torch.no_grad()
def context_dependence_batch_metrics(
    *,
    model: LoopedGraphPathTransformer,
    clean_tokens: torch.Tensor,
    clean_targets: torch.Tensor,
    clean_successors: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    corrupt_successors: torch.Tensor,
    state_loops: list[int],
    max_delta: int,
    final_target_position: int,
) -> dict[str, Any]:
    if not state_loops or min(state_loops) < 1:
        raise ValueError("state_loops must contain positive 1-indexed loops")
    if max_delta < 0:
        raise ValueError("max_delta must be nonnegative")
    if final_target_position < 1 or final_target_position > clean_targets.shape[1]:
        raise ValueError("final_target_position is outside clean_targets")
    max_loop = max(state_loops)
    clean_states = cache_raw_states(model, clean_tokens, max_loop=max_loop)
    corrupt_states = cache_raw_states(model, corrupt_tokens, max_loop=max_loop)
    condition_names = [
        "clean",
        "corrupt_context",
        "zero_context",
        "shuffled_context",
    ]
    shape = (len(condition_names), len(state_loops), max_delta + 1)
    result: dict[str, Any] = {
        "condition_names": condition_names,
        "state_loops": state_loops,
        "clean_endpoint_accuracy": torch.zeros(shape, device=clean_tokens.device),
        "clean_endpoint_probability": torch.zeros(shape, device=clean_tokens.device),
        "clean_path_accuracy": torch.zeros(shape, device=clean_tokens.device),
        "clean_path_probability": torch.zeros(shape, device=clean_tokens.device),
        "receiver_path_accuracy": torch.zeros(shape, device=clean_tokens.device),
        "receiver_path_probability": torch.zeros(shape, device=clean_tokens.device),
    }
    endpoint = clean_targets[:, final_target_position - 1]
    for loop_index, state_loop in enumerate(state_loops):
        clean_state = clean_states[state_loop - 1]
        corrupt_state = corrupt_states[state_loop - 1]
        conditions = [
            clean_state.clone(),
            replace_context_state(clean_state, corrupt_state),
            replace_context_state(clean_state, torch.zeros_like(clean_state)),
            replace_context_state(clean_state, clean_state.roll(1, dims=0)),
        ]
        current = clean_targets[:, state_loop - 1]
        clean_path = rollout_targets(
            clean_successors,
            current,
            steps=max_delta,
        )
        receiver_path = rollout_targets(
            corrupt_successors,
            current,
            steps=max_delta,
        )
        for condition_index, state in enumerate(conditions):
            for delta in range(max_delta + 1):
                logits = logits_from_raw_state(model, state)
                for prefix, target in (
                    ("clean_endpoint", endpoint),
                    ("clean_path", clean_path[:, delta]),
                    ("receiver_path", receiver_path[:, delta]),
                ):
                    accuracy, probability = _accuracy_probability(logits, target)
                    result[f"{prefix}_accuracy"][condition_index, loop_index, delta] = accuracy
                    result[f"{prefix}_probability"][condition_index, loop_index, delta] = probability
                if delta < max_delta:
                    state = apply_shared_stack(model, state)
    return result


@torch.no_grad()
def noise_recovery_batch_metrics(
    *,
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    state_loops: list[int],
    sigmas: list[float],
    max_delta: int,
    final_target_position: int,
    seed: int,
) -> dict[str, Any]:
    if not state_loops or min(state_loops) < 1:
        raise ValueError("state_loops must contain positive 1-indexed loops")
    if not sigmas or min(sigmas) <= 0:
        raise ValueError("sigmas must be positive")
    if max_delta < 0:
        raise ValueError("max_delta must be nonnegative")
    if final_target_position < 1 or final_target_position > targets.shape[1]:
        raise ValueError("final_target_position is outside targets")
    states = cache_raw_states(model, tokens, max_loop=max(state_loops))
    shape = (len(sigmas), len(state_loops), max_delta + 1)
    result: dict[str, Any] = {
        "sigmas": sigmas,
        "state_loops": state_loops,
        "distance": torch.zeros(shape, device=tokens.device),
        "recovery": torch.zeros(shape, device=tokens.device),
        "endpoint_accuracy": torch.zeros(shape, device=tokens.device),
        "endpoint_probability": torch.zeros(shape, device=tokens.device),
    }
    endpoint = targets[:, final_target_position - 1]
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    for sigma_index, sigma in enumerate(sigmas):
        for loop_index, state_loop in enumerate(state_loops):
            clean_state = states[state_loop - 1].clone()
            noisy_state = add_answer_noise(
                clean_state,
                sigma=sigma,
                generator=generator,
            )
            distance_before = (
                noisy_state[:, -1, :] - clean_state[:, -1, :]
            ).norm(dim=-1).mean()
            for delta in range(max_delta + 1):
                distance = (
                    noisy_state[:, -1, :] - clean_state[:, -1, :]
                ).norm(dim=-1).mean()
                result["distance"][sigma_index, loop_index, delta] = distance
                result["recovery"][sigma_index, loop_index, delta] = (
                    1.0 - distance / distance_before.clamp_min(1e-8)
                )
                logits = logits_from_raw_state(model, noisy_state)
                accuracy, probability = _accuracy_probability(logits, endpoint)
                result["endpoint_accuracy"][sigma_index, loop_index, delta] = accuracy
                result["endpoint_probability"][sigma_index, loop_index, delta] = probability
                if delta < max_delta:
                    clean_state = apply_shared_stack(model, clean_state)
                    noisy_state = apply_shared_stack(model, noisy_state)
    return result


def _accumulate_metrics(
    total: dict[str, Any] | None,
    batch: dict[str, Any],
) -> dict[str, Any]:
    if total is None:
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    for key, value in batch.items():
        if torch.is_tensor(value):
            total[key] += value
    return total


def _average_metrics(total: dict[str, Any], batches: int) -> dict[str, Any]:
    return {
        key: value / batches if torch.is_tensor(value) else value
        for key, value in total.items()
    }


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


@torch.no_grad()
def analyze_checkpoint(
    *,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    state_loops: list[int],
    batch_size: int,
    batches: int,
    max_delta: int,
    sigmas: list[float],
    seed: int,
) -> dict[str, Any]:
    if batches < 1 or batch_size < 1:
        raise ValueError("batch_size and batches must be positive")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = StepwiseGraphPathConfig(**payload["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    mode = str(payload.get("mode", payload.get("loss_mode", "unknown")))
    checkpoint_seed = int(payload.get("seed", -1))
    set_seed(seed)
    context_total: dict[str, Any] | None = None
    noise_total: dict[str, Any] | None = None
    path_positions = max(cfg.max_depth, max(state_loops) + max_delta)
    for batch_index in range(batches):
        clean_tokens, clean_targets, clean_successors, _ = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        corrupt_tokens, _, corrupt_successors, _ = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        context_total = _accumulate_metrics(
            context_total,
            context_dependence_batch_metrics(
                model=model,
                clean_tokens=clean_tokens,
                clean_targets=clean_targets,
                clean_successors=clean_successors,
                corrupt_tokens=corrupt_tokens,
                corrupt_successors=corrupt_successors,
                state_loops=state_loops,
                max_delta=max_delta,
                final_target_position=cfg.max_depth,
            ),
        )
        noise_total = _accumulate_metrics(
            noise_total,
            noise_recovery_batch_metrics(
                model=model,
                tokens=clean_tokens,
                targets=clean_targets,
                state_loops=state_loops,
                sigmas=sigmas,
                max_delta=max_delta,
                final_target_position=cfg.max_depth,
                seed=seed + batch_index,
            ),
        )
    assert context_total is not None and noise_total is not None
    context = _average_metrics(context_total, batches)
    noise = _average_metrics(noise_total, batches)
    condition_index = {
        name: index for index, name in enumerate(context["condition_names"])
    }
    diagnostic_delta = min(1, max_delta)
    corrupt_index = condition_index["corrupt_context"]
    clean_index = condition_index["clean"]
    context_clean_drop = (
        context["clean_path_probability"][clean_index, :, diagnostic_delta]
        - context["clean_path_probability"][corrupt_index, :, diagnostic_delta]
    ).mean()
    context_receiver_gain = (
        context["receiver_path_probability"][corrupt_index, :, diagnostic_delta]
        - context["receiver_path_probability"][clean_index, :, diagnostic_delta]
    ).mean()
    noise_recovery = noise["recovery"][:, :, -1].mean()
    scorecard = {
        "checkpoint": str(checkpoint.resolve()),
        "mode": mode,
        "seed": checkpoint_seed,
        "information_context_dependence": float(context_clean_drop.cpu()),
        "receiver_rule_context_gain": float(context_receiver_gain.cpu()),
        "reusable_transition": None,
        "attractor_refinement": float(noise_recovery.cpu()),
        "classification_is_provisional": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "context_dependence.json", context)
    _write_json(out_dir / "noise_recovery.json", noise)
    _write_json(out_dir / "scorecard.json", scorecard)
    return scorecard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure context dependence and state-noise recovery for a graph checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--state-loops", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--max-delta", type=int, default=3)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = analyze_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=pick_device(args.device),
        state_loops=args.state_loops,
        batch_size=args.batch_size,
        batches=args.batches,
        max_delta=args.max_delta,
        sigmas=args.sigmas,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
