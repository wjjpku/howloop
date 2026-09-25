from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_fj_onehop_translation import _subset
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_twohop_reprogram_j import (
    DenseAffineJ,
    _permutation_digest,
    _sha256,
    build_all_start_age_dataset,
    strict_unseen_permutations,
    validate_d8l8_onehop_config,
    write_summary_atomic,
)


class MetricAccumulator:
    def __init__(self) -> None:
        self.examples = 0
        self.moving_examples = 0
        self.target_correct = 0
        self.moving_target_correct = 0
        self.target_ce_sum = 0.0
        self.moving_target_ce_sum = 0.0

    def update(self, logits: torch.Tensor, *, current: torch.Tensor, target: torch.Tensor) -> None:
        moving = current.ne(target)
        prediction = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits.float(), target, reduction="none")
        self.examples += int(target.shape[0])
        self.moving_examples += int(moving.sum())
        self.target_correct += int(prediction.eq(target).sum())
        self.moving_target_correct += int(prediction[moving].eq(target[moving]).sum())
        self.target_ce_sum += float(loss.sum())
        self.moving_target_ce_sum += float(loss[moving].sum())

    def finalize(self) -> dict[str, float | int]:
        return {
            "examples": self.examples,
            "moving_examples": self.moving_examples,
            "target_accuracy": self.target_correct / max(self.examples, 1),
            "moving_target_accuracy": self.moving_target_correct / max(self.moving_examples, 1),
            "target_cross_entropy": self.target_ce_sum / max(self.examples, 1),
            "moving_cross_entropy": self.moving_target_ce_sum / max(self.moving_examples, 1),
        }


def _controller_from_artifact(path: Path, *, dimension: int, device: torch.device) -> DenseAffineJ:
    payload = torch.load(path, map_location=device, weights_only=False)
    selected = str(payload["selected"])
    state_dict = payload["candidates"][selected]["state_dict"]
    controller = DenseAffineJ(dimension).to(device)
    controller.load_state_dict({key: value.to(device) for key, value in state_dict.items()})
    controller.eval()
    return controller


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_audit(
    *,
    checkpoint: Path,
    controller_path: Path,
    out_dir: Path,
    device_name: str,
    controller_seed: int,
    eval_permutations: int,
    eval_seed: int,
    examples: int,
    batch_size: int,
) -> dict[str, Any]:
    if examples < 1 or batch_size < 1:
        raise ValueError("examples and batch_size must be positive")
    device = pick_device(device_name)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    if int(checkpoint_payload.get("step", -1)) != 20_000:
        raise ValueError("expected D8L8 seed3 final checkpoint at step 20000")
    model.requires_grad_(False)
    formal = strict_unseen_permutations(cfg.node_count, set(), count=eval_permutations, seed=eval_seed)
    dataset = build_all_start_age_dataset(
        model=model,
        cfg=cfg,
        permutations=formal,
        device=device,
        collection_batch_size=batch_size,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(eval_seed + 90_001)
    index = torch.randperm(dataset.source.shape[0], device=device, generator=generator)[: min(examples, dataset.source.shape[0])]
    dataset = _subset(dataset, index)
    controller = _controller_from_artifact(controller_path, dimension=cfg.d_model, device=device)
    answer = explicit_depth_position_groups(cfg.node_count)["answer"]
    variants: dict[str, tuple[FunctionalIntervention, ...]] = {
        "full": (),
        "B2_H0_answer_context_zero": (
            FunctionalIntervention(site=1, component="head_context", mode="zero", positions=answer, heads=(0,)),
        ),
        "B2_H1_answer_context_zero": (
            FunctionalIntervention(site=1, component="head_context", mode="zero", positions=answer, heads=(1,)),
        ),
        "B2_H0_answer_context_patch": (
            FunctionalIntervention(site=1, component="head_context", mode="patch", positions=answer, heads=(0,)),
        ),
        "B2_H1_answer_context_patch": (
            FunctionalIntervention(site=1, component="head_context", mode="patch", positions=answer, heads=(1,)),
        ),
    }
    accumulators = {
        f"{trajectory}_{variant}": MetricAccumulator()
        for trajectory in ("raw_F", "FJ_one")
        for variant in variants
    }
    for offset in range(0, dataset.source.shape[0], batch_size):
        batch_index = torch.arange(
            offset,
            min(offset + batch_size, dataset.source.shape[0]),
            device=device,
        )
        batch = _subset(dataset, batch_index)
        raw_state = batch.source
        fj_state = controller(raw_state).to(dtype=raw_state.dtype)
        raw_logits, raw_trace = run_instrumented_state(model, raw_state, loop_indices=(cfg.max_loops,))
        fj_logits, fj_trace = run_instrumented_state(model, fj_state, loop_indices=(cfg.max_loops,))
        accumulators["raw_F_full"].update(raw_logits, current=batch.current_node, target=batch.one_node)
        accumulators["FJ_one_full"].update(fj_logits, current=batch.current_node, target=batch.one_node)
        for trajectory, state, donor in (
            ("raw_F", raw_state, fj_trace),
            ("FJ_one", fj_state, raw_trace),
        ):
            for name, interventions in variants.items():
                if name == "full":
                    continue
                logits, _ = run_instrumented_state(
                    model,
                    state,
                    loop_indices=(cfg.max_loops,),
                    interventions=interventions,
                    donor_trace=donor if name.endswith("patch") else None,
                )
                accumulators[f"{trajectory}_{name}"].update(
                    logits, current=batch.current_node, target=batch.one_node
                )
    rows = [
        {"trajectory": key.rsplit("_", 1)[0] if key.endswith("_full") else key.split("_B2")[0], "condition": "full" if key.endswith("_full") else key[len(key.split("_B2")[0]) + 1 :], **accumulator.finalize()}
        for key, accumulator in accumulators.items()
    ]
    row_lookup = {f"{row['trajectory']}_{row['condition']}": row for row in rows}
    causal_summary: dict[str, Any] = {}
    for trajectory in ("raw_F", "FJ_one"):
        full = row_lookup[f"{trajectory}_full"]
        h0 = row_lookup[f"{trajectory}_B2_H0_answer_context_zero"]
        h1 = row_lookup[f"{trajectory}_B2_H1_answer_context_zero"]
        causal_summary[trajectory] = {
            "full_moving_target_accuracy": full["moving_target_accuracy"],
            "H0_zero_accuracy_drop": full["moving_target_accuracy"] - h0["moving_target_accuracy"],
            "H1_zero_accuracy_drop": full["moving_target_accuracy"] - h1["moving_target_accuracy"],
            "H0_zero_CE_increase": h0["moving_cross_entropy"] - full["moving_cross_entropy"],
            "H1_zero_CE_increase": h1["moving_cross_entropy"] - full["moving_cross_entropy"],
        }
    shared_h0_dependency = (
        causal_summary["raw_F"]["H0_zero_accuracy_drop"] >= 0.30
        and causal_summary["FJ_one"]["H0_zero_accuracy_drop"] >= 0.30
        and causal_summary["raw_F"]["H0_zero_accuracy_drop"] > causal_summary["raw_F"]["H1_zero_accuracy_drop"]
        and causal_summary["FJ_one"]["H0_zero_accuracy_drop"] > causal_summary["FJ_one"]["H1_zero_accuracy_drop"]
    )
    summary: dict[str, Any] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "controller_path": str(controller_path),
        "controller_seed": controller_seed,
        "source_code_sha256": _sha256(Path(__file__)),
        "formal_permutation_sha256": _permutation_digest(formal),
        "sample_examples": int(dataset.source.shape[0]),
        "intervention": "Block-2 head-context at answer position; H0 zero/patch with H1 position-matched control",
        "rows": rows,
        "causal_summary": causal_summary,
        "shared_H0_dependency": shared_h0_dependency,
        "claim_boundary": "shared mediator dependency is narrower than an original or unique whole-circuit claim",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rows.csv", rows)
    write_summary_atomic(out_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--controller-seed", type=int, required=True)
    parser.add_argument("--eval-permutations", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_audit(
        checkpoint=args.checkpoint,
        controller_path=args.controller_path,
        out_dir=args.out_dir,
        device_name=args.device,
        controller_seed=args.controller_seed,
        eval_permutations=args.eval_permutations,
        eval_seed=args.eval_seed,
        examples=args.examples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
