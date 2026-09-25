from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.sharded_evidence import (
    ShardedEvidenceConfig,
    ShardedEvidenceModel,
    hybrid_majority_target,
    make_batch,
    make_visibility_indices,
)


def _accuracy_probability(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    accuracy = logits.argmax(dim=-1).eq(target).float().mean()
    probability = logits.softmax(dim=-1).gather(1, target[:, None]).mean()
    return accuracy, probability


@torch.no_grad()
def workspace_reset_batch_metrics(
    *,
    model: ShardedEvidenceModel,
    bits: torch.Tensor,
    labels: torch.Tensor,
    visibility_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    persistent, _ = model(bits, visibility_indices, reset_between_loops=False)
    reset, _ = model(bits, visibility_indices, reset_between_loops=True)
    persistent_accuracy = persistent[:, -1].argmax(dim=-1).eq(labels).float().mean()
    reset_accuracy = reset[:, -1].argmax(dim=-1).eq(labels).float().mean()
    return {
        "persistent_accuracy": persistent_accuracy,
        "reset_accuracy": reset_accuracy,
        "reset_effect": persistent_accuracy - reset_accuracy,
    }


@torch.no_grad()
def workspace_patch_batch_metrics(
    *,
    model: ShardedEvidenceModel,
    donor_bits: torch.Tensor,
    receiver_bits: torch.Tensor,
    visibility_indices: torch.Tensor,
    patch_after_loops: list[int],
) -> dict[str, Any]:
    if not patch_after_loops or min(patch_after_loops) < 1:
        raise ValueError("patch_after_loops must contain positive loop counts")
    if max(patch_after_loops) >= model.cfg.loops:
        raise ValueError("patches must leave at least one receiver loop")
    _, donor_states = model(donor_bits, visibility_indices)
    accuracy = torch.zeros(len(patch_after_loops), device=donor_bits.device)
    probability = torch.zeros_like(accuracy)
    for index, patch_after in enumerate(patch_after_loops):
        logits, _ = model.continue_from_workspace(
            receiver_bits,
            visibility_indices,
            workspace=donor_states[:, patch_after - 1],
            start_loop=patch_after,
        )
        visited = visibility_indices[:, :patch_after, :].reshape(donor_bits.shape[0], -1)
        target = hybrid_majority_target(donor_bits, receiver_bits, visited)
        accuracy[index], probability[index] = _accuracy_probability(logits[:, -1], target)
    return {
        "patch_after_loops": patch_after_loops,
        "hybrid_accuracy": accuracy,
        "hybrid_probability": probability,
    }


@torch.no_grad()
def future_corruption_batch_metrics(
    *,
    model: ShardedEvidenceModel,
    bits: torch.Tensor,
    visibility_indices: torch.Tensor,
    future_loops: list[int],
) -> dict[str, Any]:
    if visibility_indices.shape[2] != 1:
        raise ValueError("future-shard corruption requires one visible shard per loop")
    if not future_loops or min(future_loops) < 1:
        raise ValueError("future_loops must leave a nonempty pre-visit prefix")
    if max(future_loops) >= model.cfg.loops:
        raise ValueError("future_loops contains an invalid loop index")
    clean_logits, _ = model(bits, visibility_indices)
    prefix_max_abs = torch.zeros(len(future_loops), device=bits.device)
    at_visit_mean_abs = torch.zeros_like(prefix_max_abs)
    final_mean_abs = torch.zeros_like(prefix_max_abs)
    for index, future_loop in enumerate(future_loops):
        corrupt = bits.clone()
        evidence_index = visibility_indices[:, future_loop, 0:1]
        value = corrupt.gather(1, evidence_index)
        corrupt.scatter_(1, evidence_index, 1 - value)
        corrupt_logits, _ = model(corrupt, visibility_indices)
        prefix_max_abs[index] = (
            clean_logits[:, :future_loop] - corrupt_logits[:, :future_loop]
        ).abs().max()
        at_visit_mean_abs[index] = (
            clean_logits[:, future_loop] - corrupt_logits[:, future_loop]
        ).abs().mean()
        final_mean_abs[index] = (
            clean_logits[:, -1] - corrupt_logits[:, -1]
        ).abs().mean()
    return {
        "future_loops": future_loops,
        "prefix_max_abs": prefix_max_abs,
        "at_visit_mean_abs": at_visit_mean_abs,
        "final_mean_abs": final_mean_abs,
    }


def _accumulate(total: dict[str, Any] | None, batch: dict[str, Any]) -> dict[str, Any]:
    if total is None:
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    for key, value in batch.items():
        if torch.is_tensor(value):
            total[key] += value
    return total


def _average(total: dict[str, Any], batches: int) -> dict[str, Any]:
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
    batch_size: int,
    batches: int,
    seed: int,
    analysis_condition: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ShardedEvidenceConfig(**payload["config"])
    training_condition = payload.get("mode", payload.get("condition"))
    condition = analysis_condition or training_condition
    if condition not in {"shard", "shard_reset", "shard_shuffled"}:
        raise ValueError("workspace diagnostics require a one-shard condition")
    model = ShardedEvidenceModel(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    reset_total: dict[str, Any] | None = None
    patch_total: dict[str, Any] | None = None
    future_total: dict[str, Any] | None = None
    patch_after = list(range(1, cfg.loops))
    future_loops = list(range(1, cfg.loops))
    for _ in range(batches):
        donor_bits, donor_labels = make_batch(
            batch_size=batch_size,
            evidence_count=cfg.evidence_count,
            device=device,
            generator=generator,
        )
        receiver_bits, _ = make_batch(
            batch_size=batch_size,
            evidence_count=cfg.evidence_count,
            device=device,
            generator=generator,
        )
        visibility = make_visibility_indices(
            condition,
            batch_size=batch_size,
            cfg=cfg,
            device=device,
            generator=generator,
        )
        reset_total = _accumulate(
            reset_total,
            workspace_reset_batch_metrics(
                model=model,
                bits=donor_bits,
                labels=donor_labels,
                visibility_indices=visibility,
            ),
        )
        patch_total = _accumulate(
            patch_total,
            workspace_patch_batch_metrics(
                model=model,
                donor_bits=donor_bits,
                receiver_bits=receiver_bits,
                visibility_indices=visibility,
                patch_after_loops=patch_after,
            ),
        )
        future_total = _accumulate(
            future_total,
            future_corruption_batch_metrics(
                model=model,
                bits=donor_bits,
                visibility_indices=visibility,
                future_loops=future_loops,
            ),
        )
    if reset_total is None or patch_total is None or future_total is None:
        raise ValueError("batches must be positive")
    reset = _average(reset_total, batches)
    patch = _average(patch_total, batches)
    future = _average(future_total, batches)
    reset_effect = float(reset["reset_effect"].cpu())
    patch_mean = float(patch["hybrid_accuracy"].mean().cpu())
    prefix_max = float(future["prefix_max_abs"].max().cpu())
    reset_component = max(0.0, min(1.0, reset_effect / 0.25))
    patch_component = max(0.0, min(1.0, (patch_mean - 0.6875) / 0.3125))
    invariance_component = 1.0 if prefix_max <= 1e-8 else 0.0
    scorecard = {
        "checkpoint": str(checkpoint.resolve()),
        "condition": condition,
        "training_condition": training_condition,
        "analysis_condition": condition,
        "seed": int(payload.get("seed", -1)),
        "step": int(payload.get("step", -1)),
        "information_workspace_score": (
            reset_component + patch_component + invariance_component
        )
        / 3.0,
        "reset_effect": reset_effect,
        "hybrid_patch_accuracy_mean": patch_mean,
        "unseen_prefix_max_abs": prefix_max,
        "classification_is_provisional": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "workspace_reset.json", reset)
    _write_json(out_dir / "workspace_patch.json", patch)
    _write_json(out_dir / "future_corruption.json", future)
    _write_json(out_dir / "scorecard.json", scorecard)
    return scorecard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sharded-evidence workspace reuse.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument(
        "--analysis-condition",
        choices=["shard", "shard_reset", "shard_shuffled"],
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        analysis_condition=args.analysis_condition,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
