from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_fj_halfstep_multi4 import (
    RATE_SPECS,
    build_all_start_age_dataset,
)
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.graph_path_twohop_reprogram_j import (
    DenseAffineJ,
    _permutation_digest,
    _sha256,
    strict_unseen_permutations,
    validate_d8l8_onehop_config,
    write_summary_atomic,
)


def diagnostic_from_counts(
    *,
    examples: int,
    moving_examples: int,
    raw_current: int,
    mapped_current: int,
    raw_boundary: int,
    mapped_boundary: int,
    margin: float = 0.05,
) -> dict[str, Any]:
    if examples < 1 or moving_examples < 1:
        raise ValueError("nonzero example and moving-example counts are required")
    raw_current_accuracy = raw_current / examples
    mapped_current_accuracy = mapped_current / examples
    raw_boundary_accuracy = raw_boundary / moving_examples
    mapped_boundary_accuracy = mapped_boundary / moving_examples
    return {
        "examples": examples,
        "moving_examples": moving_examples,
        "raw_current_accuracy": raw_current_accuracy,
        "mapped_current_accuracy": mapped_current_accuracy,
        "raw_moving_boundary_accuracy": raw_boundary_accuracy,
        "mapped_moving_boundary_accuracy": mapped_boundary_accuracy,
        "current_accuracy_change": mapped_current_accuracy - raw_current_accuracy,
        "moving_boundary_accuracy_change": mapped_boundary_accuracy - raw_boundary_accuracy,
        "absolute_no_prewrite_gate": (
            mapped_current_accuracy >= 0.80 and mapped_boundary_accuracy <= 0.20
        ),
        "no_incremental_prewrite_vs_raw": (
            mapped_current_accuracy >= raw_current_accuracy - margin
            and mapped_boundary_accuracy <= raw_boundary_accuracy + margin
        ),
        "relative_margin": margin,
    }


def load_selected_controller(path: Path, *, dimension: int, device: torch.device) -> tuple[str, DenseAffineJ]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    selected = str(payload["selected"])
    state_dict = payload["candidates"][selected]["state_dict"]
    controller = DenseAffineJ(dimension).to(device)
    controller.load_state_dict(state_dict)
    controller.eval()
    return selected, controller


@torch.no_grad()
def evaluate_controller(
    *,
    model,
    controller: DenseAffineJ,
    dataset,
    batch_size: int,
) -> dict[str, Any]:
    counts = {
        "examples": 0,
        "moving_examples": 0,
        "raw_current": 0,
        "mapped_current": 0,
        "raw_boundary": 0,
        "mapped_boundary": 0,
    }
    for offset in range(0, dataset.source.shape[0], batch_size):
        stop = min(offset + batch_size, dataset.source.shape[0])
        source = dataset.source[offset:stop]
        current = dataset.current_node[offset:stop]
        boundary = dataset.targets[offset:stop, 0]
        moving = current.ne(boundary)
        raw_prediction = logits_from_raw_state(model, source).argmax(dim=-1)
        mapped_prediction = logits_from_raw_state(model, controller(source)).argmax(dim=-1)
        counts["examples"] += int(source.shape[0])
        counts["moving_examples"] += int(moving.sum())
        counts["raw_current"] += int(raw_prediction.eq(current).sum())
        counts["mapped_current"] += int(mapped_prediction.eq(current).sum())
        counts["raw_boundary"] += int(raw_prediction[moving].eq(boundary[moving]).sum())
        counts["mapped_boundary"] += int(mapped_prediction[moving].eq(boundary[moving]).sum())
    return diagnostic_from_counts(**counts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--controller-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-permutations", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--collection-batch-size", type=int, default=256)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.15)
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    model.eval().requires_grad_(False)
    formal = strict_unseen_permutations(
        cfg.node_count, set(), count=args.eval_permutations, seed=args.eval_seed
    )
    datasets = {
        rate_name: build_all_start_age_dataset(
            model=model,
            cfg=cfg,
            permutations=formal,
            logical_steps=8,
            semantic_stride=spec.semantic_stride,
            device=device,
            collection_batch_size=args.collection_batch_size,
        )
        for rate_name, spec in RATE_SPECS.items()
        if rate_name != "zero"
    }
    rows = []
    for controller_seed in args.controller_seeds:
        for rate_name in ("half", "one", "two"):
            artifact = args.controller_root / f"seed{controller_seed}" / rate_name / "controller.pt"
            selected, controller = load_selected_controller(
                artifact, dimension=cfg.d_model, device=device
            )
            rows.append(
                {
                    "controller_seed": controller_seed,
                    "rate_name": rate_name,
                    "selected_candidate": selected,
                    "controller_artifact": str(artifact),
                    **evaluate_controller(
                        model=model,
                        controller=controller,
                        dataset=datasets[rate_name],
                        batch_size=args.batch_size,
                    ),
                }
            )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "source_code_sha256": _sha256(Path(__file__)),
        "controller_seeds": list(args.controller_seeds),
        "strict_eval_permutations": args.eval_permutations,
        "strict_eval_examples": int(next(iter(datasets.values())).source.shape[0]),
        "strict_eval_permutation_sha256": _permutation_digest(formal),
        "interpretation": (
            "The absolute gate is the preregistered current>=0.80 and moving-target<=0.20 test. "
            "The relative diagnostic asks whether J itself increases target prewrite versus the raw receiver state; it does not replace the absolute gate."
        ),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
