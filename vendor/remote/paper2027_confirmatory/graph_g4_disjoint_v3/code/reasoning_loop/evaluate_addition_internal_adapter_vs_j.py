from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.addition_internal_weight_adapter import (
    SharedInternalWeightAdapter,
    forward_fixed_length,
)
from reasoning_loop.continue_paper_length_controller import (
    validate_task_payload_compatibility,
)
from reasoning_loop.evaluate_addition_normal_j_horizon import (
    contiguous_horizon,
    exact_mcnemar_pvalue,
    per_example_exact,
)
from reasoning_loop.paper_length_telomere import (
    PaperBatch,
    PaperLoopedTransformer,
    PaperTaskSpec,
    atomic_json_dump,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
    write_csv,
)
from reasoning_loop.robust_age_path_controller import (
    alternating_age_path,
    execute_actions,
)


THRESHOLDS = (0.99, 0.95, 0.90, 0.50)


def _parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("adapter must be LABEL=PATH")
    label, path = raw.split("=", 1)
    if not label or label in {"raw", "inter_loop_j"} or not path:
        raise argparse.ArgumentTypeError("adapter label/path is invalid")
    return label, Path(path)


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("integer list must be positive and nonempty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("integer list contains duplicates")
    return values


def paired_outcome_counts(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, int]:
    if candidate.shape != reference.shape:
        raise ValueError("paired outcome tensors must have the same shape")
    candidate = candidate.bool()
    reference = reference.bool()
    return {
        "candidate_only": int((candidate & ~reference).sum()),
        "reference_only": int((~candidate & reference).sum()),
        "both": int((candidate & reference).sum()),
        "neither": int((~candidate & ~reference).sum()),
    }


@torch.inference_mode()
def paired_variant_logits(
    *,
    model: PaperLoopedTransformer,
    batch: PaperBatch,
    target_step: int,
    adapters: Mapping[str, SharedInternalWeightAdapter],
    j_controller: torch.nn.Module,
) -> OrderedDict[str, torch.Tensor]:
    if target_step < 1:
        raise ValueError("target step must be positive")
    state = forward_fixed_length(
        model=model,
        inputs=batch.inputs,
        steps=target_step,
        gradient_checkpointing=False,
    )
    logits: OrderedDict[str, torch.Tensor] = OrderedDict(
        [("raw", model.decode(state).float())]
    )
    for label, adapter in adapters.items():
        with adapter.applied():
            adapted_state = forward_fixed_length(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                gradient_checkpointing=False,
            )
            logits[label] = model.decode(adapted_state).float()
    logits["inter_loop_j"] = execute_actions(
        model=model,
        controller=j_controller,
        batch=batch,
        actions=alternating_age_path(target_step).actions,
    )
    return logits


def _load_adapter(
    *,
    path: Path,
    checkpoint: Path,
    model: PaperLoopedTransformer,
    spec: PaperTaskSpec,
    device: torch.device,
) -> tuple[SharedInternalWeightAdapter, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") == "addition_internal_weight_adapter_candidate":
        run_summary_path = path.parent.parent / "summary.json"
        if not run_summary_path.is_file():
            raise ValueError(
                f"candidate adapter lacks its completed run summary: {path}"
            )
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary.get("status") != "complete":
            raise ValueError(f"candidate adapter run is incomplete: {path}")
        baseline = run_summary["baseline"]
        payload = {
            **payload,
            "source_kind": "addition_internal_weight_adapter_candidate",
            "kind": "addition_internal_weight_adapter",
            "seed": run_summary.get("training", {}).get("seed", 0),
            "selected_update": int(payload["update"]),
            "selected_cumulative_update": int(payload["cumulative_update"]),
            "backbone_checkpoint": baseline["checkpoint"],
            "task": baseline["task"],
            "model": baseline["model"],
        }
    if payload.get("kind") != "addition_internal_weight_adapter":
        raise ValueError(f"not an internal-weight adapter: {path}")
    if payload.get("site") != "k":
        raise ValueError(f"comparison is restricted to K adapters: {path}")
    source_backbone = Path(payload["backbone_checkpoint"]).expanduser().resolve()
    if source_backbone != checkpoint:
        raise ValueError(
            f"adapter backbone mismatch: {source_backbone} != {checkpoint}"
        )
    validate_task_payload_compatibility(payload["task"], asdict(spec))
    delta = payload["delta"].detach().float()
    expected_shape = (model.config.d_model, model.config.d_model)
    if tuple(delta.shape) != expected_shape:
        raise ValueError(f"adapter has shape {tuple(delta.shape)}, expected {expected_shape}")
    adapter = SharedInternalWeightAdapter(
        model=model, site="k", init_std=0.0, seed=int(payload.get("seed", 0))
    ).to(device)
    adapter.delta.data.copy_(delta.to(device))
    adapter.eval()
    return adapter, payload


def _accumulate_metrics(
    totals: dict[str, float | int], logits: torch.Tensor, batch: PaperBatch
) -> torch.Tensor:
    exact = per_example_exact(logits, batch)
    predictions = logits.argmax(dim=-1)
    answer_correct = predictions.eq(batch.targets)[batch.answer_mask]
    totals["correct"] += int(exact.sum())
    totals["examples"] += int(exact.numel())
    totals["answer_correct"] += int(answer_correct.sum())
    totals["answer_tokens"] += int(answer_correct.numel())
    totals["answer_nll_sum"] += float(
        F.cross_entropy(
            logits[batch.answer_mask],
            batch.targets[batch.answer_mask],
            reduction="sum",
        )
    )
    return exact


def _band_mean(
    rows: Sequence[dict[str, Any]], low: int, high: int
) -> float | None:
    selected = [
        float(row["exact_match"])
        for row in rows
        if low <= int(row["logical_length"]) <= high
    ]
    return sum(selected) / len(selected) if selected else None


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    checkpoint = args.checkpoint.expanduser().resolve()
    controller_path = args.controller.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    lengths = (
        list(range(1, args.max_length + 1))
        if args.lengths is None
        else _parse_csv_ints(args.lengths)
    )
    evaluation_seeds = _parse_csv_ints(args.evaluation_seeds)
    if len(lengths) != len(set(lengths)):
        raise ValueError("evaluation lengths contain duplicates")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, spec, backbone_payload = load_backbone(checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("paired K/J evaluation requires Addition")
    model.eval()

    adapters: OrderedDict[str, SharedInternalWeightAdapter] = OrderedDict()
    adapter_payloads: dict[str, dict[str, Any]] = {}
    for label, raw_path in args.adapter:
        if label in adapters:
            raise ValueError(f"duplicate adapter label: {label}")
        path = raw_path.expanduser().resolve()
        adapter, payload = _load_adapter(
            path=path,
            checkpoint=checkpoint,
            model=model,
            spec=spec,
            device=device,
        )
        adapters[label] = adapter
        adapter_payloads[label] = {**payload, "artifact": str(path)}

    j_controller, controller_payload = load_controller(
        controller_path, device=device
    )
    validate_task_payload_compatibility(controller_payload["task"], backbone_payload["task"])
    if not controller_payload.get("controller_post_final_j", False):
        raise ValueError("comparison requires canonical post-final inter-loop J")

    labels = ["raw", *adapters, "inter_loop_j"]
    by_length_rows: list[dict[str, Any]] = []
    by_seed_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for logical_length in lengths:
        totals: dict[str, dict[str, float | int]] = {
            label: {
                "correct": 0,
                "examples": 0,
                "answer_correct": 0,
                "answer_tokens": 0,
                "answer_nll_sum": 0.0,
            }
            for label in labels
        }
        paired_totals = {
            label: {
                "candidate_only": 0,
                "reference_only": 0,
                "both": 0,
                "neither": 0,
            }
            for label in labels
            if label != "inter_loop_j"
        }
        for evaluation_seed in evaluation_seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(evaluation_seed + 1_000_003 * logical_length)
            batch = generate_paper_batch(
                spec,
                batch_size=args.batch_size,
                min_length=logical_length,
                max_length=logical_length,
                fixed_length=logical_length,
                generator=generator,
            ).to(device)
            logits = paired_variant_logits(
                model=model,
                batch=batch,
                target_step=logical_length + spec.step_offset,
                adapters=adapters,
                j_controller=j_controller,
            )
            exact: dict[str, torch.Tensor] = {}
            for label in labels:
                seed_totals = {
                    "correct": 0,
                    "examples": 0,
                    "answer_correct": 0,
                    "answer_tokens": 0,
                    "answer_nll_sum": 0.0,
                }
                exact[label] = _accumulate_metrics(seed_totals, logits[label], batch)
                _accumulate_metrics(totals[label], logits[label], batch)
                by_seed_rows.append(
                    {
                        "logical_length": logical_length,
                        "evaluation_seed": evaluation_seed,
                        "variant": label,
                        "correct": seed_totals["correct"],
                        "examples": seed_totals["examples"],
                        "exact_match": seed_totals["correct"]
                        / seed_totals["examples"],
                        "answer_token_accuracy": seed_totals["answer_correct"]
                        / seed_totals["answer_tokens"],
                        "answer_nll": seed_totals["answer_nll_sum"]
                        / seed_totals["answer_tokens"],
                    }
                )
            reference = exact["inter_loop_j"]
            for label in paired_totals:
                counts = paired_outcome_counts(exact[label], reference)
                for key, value in counts.items():
                    paired_totals[label][key] += value

        for label, values in totals.items():
            by_length_rows.append(
                {
                    "logical_length": logical_length,
                    "target_step": logical_length + spec.step_offset,
                    "variant": label,
                    "correct": values["correct"],
                    "examples": values["examples"],
                    "exact_match": values["correct"] / values["examples"],
                    "answer_token_accuracy": values["answer_correct"]
                    / values["answer_tokens"],
                    "answer_nll": values["answer_nll_sum"]
                    / values["answer_tokens"],
                }
            )
        for label, counts in paired_totals.items():
            paired_rows.append(
                {
                    "logical_length": logical_length,
                    "candidate": label,
                    "reference": "inter_loop_j",
                    **counts,
                    "net_correct_delta": counts["candidate_only"]
                    - counts["reference_only"],
                    "exact_mcnemar_pvalue": exact_mcnemar_pvalue(
                        counts["candidate_only"], counts["reference_only"]
                    ),
                }
            )
        progress = {
            row["variant"]: round(float(row["exact_match"]), 6)
            for row in by_length_rows
            if row["logical_length"] == logical_length
        }
        print(
            json.dumps(
                {
                    "event": "length_complete",
                    "logical_length": logical_length,
                    "exact_match": progress,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dense_consecutive = lengths == list(range(1, max(lengths) + 1))
    metrics: dict[str, Any] = {}
    for label in labels:
        rows = [row for row in by_length_rows if row["variant"] == label]
        metrics[label] = {
            "contiguous_accuracy_horizons": (
                {
                    str(threshold): contiguous_horizon(rows, threshold=threshold)
                    for threshold in THRESHOLDS
                }
                if dense_consecutive
                else None
            ),
            "band_mean_exact_match": {
                "id_1_19": _band_mean(rows, 1, 19),
                "repair_20_40": _band_mean(rows, 20, 40),
                "unseen_41_60": _band_mean(rows, 41, 60),
                "far_61_80": _band_mean(rows, 61, 80),
                "extreme_81_100": _band_mean(rows, 81, 100),
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "by_length.csv", by_length_rows)
    write_csv(out_dir / "by_seed.csv", by_seed_rows)
    write_csv(out_dir / "paired_vs_j.csv", paired_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "task": asdict(spec),
        "target_loop_rule": f"T(m)=m+{spec.step_offset}",
        "normal_usage": {
            "raw": "F^T",
            "internal_k": "F_K^T; one shared additive K matrix in every physical layer and loop",
            "inter_loop_j": "(FJ)^T including post-final J",
        },
        "lengths": lengths,
        "evaluation_seeds": evaluation_seeds,
        "batch_size": args.batch_size,
        "examples_per_length_variant": args.batch_size * len(evaluation_seeds),
        "selection_policy": "evaluation only; no checkpoint selected on these results",
        "variants": {
            "raw": {"trainable_parameters": 0},
            **{
                label: {
                    "artifact": payload["artifact"],
                    "site": payload["site"],
                    "trainable_parameters": int(payload["delta"].numel()),
                    "selected_update": payload.get("selected_update"),
                    "selected_cumulative_update": payload.get(
                        "selected_cumulative_update", payload.get("selected_update")
                    ),
                }
                for label, payload in adapter_payloads.items()
            },
            "inter_loop_j": {
                "artifact": str(controller_path),
                "rank": int(controller_payload["rank"]),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in j_controller.parameters()
                ),
                "training_seed": controller_payload.get("seed"),
                "training_budget": controller_payload.get("training_budget"),
                "controller_post_final_j": True,
            },
        },
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_json_dump(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired raw/internal-K/inter-loop-J Addition horizon evaluation."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--adapter", type=_parse_named_path, action="append", required=True
    )
    parser.add_argument("--controller", type=Path, required=True)
    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument("--max-length", type=int, default=100)
    length_group.add_argument("--lengths")
    parser.add_argument(
        "--evaluation-seeds",
        default="571001,581001,591001,601001",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
