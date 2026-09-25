from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.paper_length_telomere import (
    DenseAffineController,
    DiagonalLowRankController,
    _controlled_ce_batch,
    _controller_learning_rate,
    _logical_controller_plan,
    atomic_json_dump,
    atomic_torch_save,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
    set_seed,
    write_csv,
)


@dataclass(frozen=True)
class LengthBand:
    name: str
    minimum: int
    maximum: int
    probability: float


@dataclass(frozen=True)
class ContinuationSchedule:
    start_total_update: int
    target_total_update: int
    additional_updates: int
    warmup_updates: int
    stable_updates: int
    decay_updates: int
    final_learning_rate_ratio: float


CONTINUATION_BANDS = (
    LengthBand("retention", 1, 19, 0.20),
    LengthBand("transition", 20, 32, 0.30),
    LengthBand("boundary", 33, 40, 0.50),
)


TASK_COMPATIBILITY_FIELDS = (
    "name",
    "block_layers",
    "train_max_length",
    "step_offset",
    "vocab_size",
)

MODEL_COMPATIBILITY_FIELDS = (
    "vocab_size",
    "d_model",
    "n_heads",
    "d_mlp",
    "block_layers",
    "layer_norm_epsilon",
)


def validate_model_payload_compatibility(
    source_model: Any, backbone_model: Any
) -> None:
    """Allow metadata additions while rejecting architectural mismatches."""
    if not isinstance(source_model, dict) or not isinstance(backbone_model, dict):
        if source_model != backbone_model:
            raise ValueError("source controller and backbone model do not match")
        return
    for field in MODEL_COMPATIBILITY_FIELDS:
        if source_model.get(field) != backbone_model.get(field):
            raise ValueError(
                "source controller and backbone model do not match "
                f"at {field!r}: {source_model.get(field)!r} != "
                f"{backbone_model.get(field)!r}"
            )


def validate_task_payload_compatibility(
    source_task: Any, backbone_task: Any
) -> None:
    """Accept schema additions while rejecting task-semantic mismatches."""
    if not isinstance(source_task, dict) or not isinstance(backbone_task, dict):
        if source_task != backbone_task:
            raise ValueError("source controller and backbone task do not match")
        return
    for field in TASK_COMPATIBILITY_FIELDS:
        if source_task.get(field) != backbone_task.get(field):
            raise ValueError(
                "source controller and backbone task do not match "
                f"at {field!r}: {source_task.get(field)!r} != "
                f"{backbone_task.get(field)!r}"
            )
    source_symbols = tuple(source_task.get("copy_symbols", (0, 1)))
    backbone_symbols = tuple(backbone_task.get("copy_symbols", (0, 1)))
    if source_symbols != backbone_symbols:
        raise ValueError(
            "source controller and backbone task do not match at "
            f"'copy_symbols': {source_symbols!r} != {backbone_symbols!r}"
        )


def validate_length_mix(
    bands: Sequence[LengthBand] = CONTINUATION_BANDS,
) -> tuple[LengthBand, ...]:
    checked = tuple(bands)
    if not checked:
        raise ValueError("length mix must contain at least one band")
    if not math.isclose(
        sum(band.probability for band in checked), 1.0, abs_tol=1e-12
    ):
        raise ValueError("length mix probabilities must sum to one")
    if checked[0].minimum < 1:
        raise ValueError("length mix must start at a positive logical length")
    previous_maximum = checked[0].minimum - 1
    for band in checked:
        if band.minimum != previous_maximum + 1:
            raise ValueError("length mix bands must be contiguous")
        if band.maximum < band.minimum:
            raise ValueError("length mix band maximum precedes its minimum")
        if band.probability <= 0:
            raise ValueError("length mix probabilities must be positive")
        previous_maximum = band.maximum
    return checked


def make_length_bands(
    *,
    minimum_length: int = 1,
    retention_maximum: int,
    transition_maximum: int,
    boundary_maximum: int,
    retention_probability: float,
    transition_probability: float,
    boundary_probability: float,
) -> tuple[LengthBand, ...]:
    return validate_length_mix(
        (
            LengthBand(
                "retention",
                minimum_length,
                retention_maximum,
                retention_probability,
            ),
            LengthBand(
                "transition",
                retention_maximum + 1,
                transition_maximum,
                transition_probability,
            ),
            LengthBand(
                "boundary",
                transition_maximum + 1,
                boundary_maximum,
                boundary_probability,
            ),
        )
    )


def sample_mixed_logical_length(
    generator: torch.Generator,
    bands: Sequence[LengthBand] = CONTINUATION_BANDS,
) -> int:
    checked = validate_length_mix(bands)
    probabilities = torch.tensor(
        [band.probability for band in checked], dtype=torch.float64
    )
    band_index = int(
        torch.multinomial(probabilities, 1, generator=generator).item()
    )
    selected = checked[band_index]
    return int(
        torch.randint(
            selected.minimum,
            selected.maximum + 1,
            (1,),
            generator=generator,
        ).item()
    )


def restore_generator_state(
    generator: torch.Generator, state: torch.Tensor
) -> None:
    generator.set_state(state.detach().cpu())


def validate_continuation_schedule(
    *,
    start_total_update: int,
    target_total_update: int,
    warmup_updates: int,
    stable_updates: int,
    final_learning_rate_ratio: float = 0.1,
) -> ContinuationSchedule:
    if start_total_update < 0:
        raise ValueError("starting update must be non-negative")
    if target_total_update <= start_total_update:
        raise ValueError("target update must exceed starting update")
    additional_updates = target_total_update - start_total_update
    if warmup_updates < 0 or stable_updates < 0:
        raise ValueError("WSD phase lengths must be non-negative")
    if warmup_updates + stable_updates > additional_updates:
        raise ValueError("WSD warmup and stable phases exceed continuation")
    if not 0 < final_learning_rate_ratio <= 1:
        raise ValueError("final learning-rate ratio must lie in (0, 1]")
    return ContinuationSchedule(
        start_total_update=start_total_update,
        target_total_update=target_total_update,
        additional_updates=additional_updates,
        warmup_updates=warmup_updates,
        stable_updates=stable_updates,
        decay_updates=additional_updates - warmup_updates - stable_updates,
        final_learning_rate_ratio=final_learning_rate_ratio,
    )


def continuation_learning_rate(
    schedule: ContinuationSchedule,
    *,
    local_update: int,
    peak: float,
) -> float:
    return _controller_learning_rate(
        peak=peak,
        update=local_update,
        total_updates=schedule.additional_updates,
        warmup_updates=schedule.warmup_updates,
        final_ratio=schedule.final_learning_rate_ratio,
        schedule="wsd",
        stable_updates=schedule.stable_updates,
    )


def _band_name(logical_length: int, bands: Sequence[LengthBand]) -> str:
    for band in bands:
        if band.minimum <= logical_length <= band.maximum:
            return band.name
    raise ValueError("sampled logical length is outside the declared mix")


def _make_optimizer(
    controller: torch.nn.Module,
    *,
    peak_learning_rate: float,
    diagonal_learning_rate_multiplier: float,
) -> torch.optim.Optimizer:
    if isinstance(controller, DiagonalLowRankController):
        parameter_groups: Any = [
            {"params": [controller.A, controller.B, controller.bias]},
            {
                "params": [controller.diagonal],
                "lr": peak_learning_rate * diagonal_learning_rate_multiplier,
            },
        ]
    elif isinstance(controller, DenseAffineController):
        parameter_groups = controller.parameters()
    else:
        raise TypeError(f"unsupported controller type: {type(controller).__name__}")
    return torch.optim.AdamW(
        parameter_groups, lr=peak_learning_rate, weight_decay=0.0
    )


def _controller_optimizer_step(
    *,
    loss: torch.Tensor,
    controller: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> tuple[float, bool]:
    """Take a J update, tolerating lengths where the inter-loop J is unused.

    For example, Addition with T(1)=1 and an inter-loop controller has no
    controller application at m=1.  Its final loss is still evaluated and
    logged, but it cannot provide a gradient to J without changing the model
    interface semantics.
    """
    if not loss.requires_grad:
        return 0.0, False
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
    optimizer.step()
    return float(grad_norm), True


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu() for key, value in module.state_dict().items()
    }


def _rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def evaluate_controller(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module,
    anchor_step: int,
    post_final_controller: bool,
    lengths: Sequence[int],
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    controller.eval()
    results: dict[int, dict[str, float]] = {}
    for logical_length in lengths:
        plan = _logical_controller_plan(
            spec,
            logical_length,
            controller_anchor_step=anchor_step,
        )
        loss_total = 0.0
        em_total = 0.0
        for batch_index in range(batches):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 10_000 * logical_length + batch_index)
            batch = generate_paper_batch(
                spec,
                batch_size=batch_size,
                min_length=logical_length,
                max_length=logical_length,
                fixed_length=logical_length,
                generator=generator,
            ).to(device)
            loss, em = _controlled_ce_batch(
                model=model,
                controller=controller,
                batch=batch,
                anchor_step=anchor_step,
                controlled_steps=plan.controlled_steps,
                post_final_controller=post_final_controller,
            )
            loss_total += float(loss)
            em_total += em
        results[logical_length] = {
            "answer_ce": loss_total / batches,
            "exact_match": em_total / batches,
        }
    controller.train()
    return results


def _standard_controller_artifact(
    *,
    source_payload: dict[str, Any],
    controller: torch.nn.Module,
    source_controller: Path,
    schedule: ContinuationSchedule,
    bands: Sequence[LengthBand],
    total_update: int,
    batch_size: int,
    length_counts: dict[int, int],
    peak_learning_rate: float,
    diagonal_learning_rate_multiplier: float,
    seed: int,
) -> dict[str, Any]:
    artifact = copy.deepcopy(source_payload)
    old_budget = source_payload["training_budget"]
    parameterization = source_payload.get(
        "controller_parameterization", "diagonal_low_rank"
    )
    if parameterization not in {"diagonal_low_rank", "dense_affine"}:
        raise ValueError(f"unsupported parameterization: {parameterization}")
    new_examples = (total_update - schedule.start_total_update) * batch_size
    old_counts = {
        int(length): int(count)
        for length, count in (
            source_payload.get("controller_logical_length_example_counts") or {}
        ).items()
    }
    combined_counts = dict(old_counts)
    for length, count in length_counts.items():
        combined_counts[length] = combined_counts.get(length, 0) + count
    dense_updates = int(old_budget.get("dense_optimizer_updates", 0))
    low_rank_updates = int(old_budget.get("low_rank_optimizer_updates", 0))
    dense_examples = int(old_budget.get("dense_training_examples", 0))
    low_rank_examples = int(old_budget.get("low_rank_training_examples", 0))
    if parameterization == "dense_affine":
        dense_updates = total_update
        dense_examples += new_examples
    else:
        low_rank_updates = total_update
        low_rank_examples += new_examples
    artifact.update(
        {
            "controller_state_dict": _state_dict_cpu(controller),
            "snapshot_update": total_update,
            "snapshot_total_updates": schedule.target_total_update,
            "controller_sampled_logical_lengths": sorted(combined_counts),
            "controller_logical_length_example_counts": dict(
                sorted(combined_counts.items())
            ),
            "controller_logical_min_length": min(combined_counts),
            "controller_logical_max_length": max(combined_counts),
            "training_budget": {
                **old_budget,
                "dense_optimizer_updates": dense_updates,
                "low_rank_optimizer_updates": low_rank_updates,
                "total_optimizer_updates": total_update,
                "dense_training_examples": dense_examples,
                "low_rank_training_examples": low_rank_examples,
                "total_training_examples": dense_examples + low_rank_examples,
            },
            "continuation": {
                "source_controller": str(source_controller),
                "optimizer_resume_mode": "fresh_adamw_from_source_weights",
                "start_total_update": schedule.start_total_update,
                "target_total_update": schedule.target_total_update,
                "current_total_update": total_update,
                "additional_wsd": asdict(schedule),
                "peak_learning_rate": peak_learning_rate,
                "diagonal_learning_rate_multiplier": (
                    diagonal_learning_rate_multiplier
                ),
                "batch_size": batch_size,
                "length_mix": [asdict(band) for band in bands],
                "continuation_seed": seed,
                "loss": "final registered T(n) answer-region task CE only",
            },
        }
    )
    return artifact


def _checkpoint_payload(
    *,
    source_controller: Path,
    source_payload: dict[str, Any],
    controller: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    configuration: dict[str, Any],
    local_update: int,
    total_update: int,
    length_counts: dict[int, int],
    band_counts: dict[str, int],
    history: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    best_score: float,
    best_total_update: int,
    no_gradient_updates: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "kind": "paper_length_controller_continuation_checkpoint",
        "source_controller": str(source_controller),
        "source_controller_seed": source_payload["seed"],
        "configuration": configuration,
        "local_update": local_update,
        "total_update": total_update,
        "controller_state_dict": _state_dict_cpu(controller),
        "optimizer_state_dict": optimizer.state_dict(),
        "data_generator_state": data_generator.get_state(),
        "rng_state": _rng_state(),
        "length_counts": dict(length_counts),
        "band_counts": dict(band_counts),
        "history": history,
        "evaluations": evaluations,
        "best_score": best_score,
        "best_total_update": best_total_update,
        "no_gradient_updates": no_gradient_updates,
        "elapsed_seconds": elapsed_seconds,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != args.task:
        raise ValueError(
            f"checkpoint task {spec.name!r} does not match --task {args.task!r}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.initialize_dense_identity:
        if args.source_controller is not None:
            raise ValueError(
                "--initialize-dense-identity and --source-controller are mutually exclusive"
            )
        controller = DenseAffineController(model.config.d_model).to(device)
        anchor_step = args.anchor_step
        if anchor_step is None:
            raise ValueError("dense identity initialization requires --anchor-step")
        initial_counts = {
            int(length): 0
            for length in range(args.minimum_length, args.boundary_maximum + 1)
        }
        source_payload = {
            "kind": "paper_length_telomere_controller",
            "checkpoint": str(args.checkpoint),
            "backbone_step": backbone_payload["step"],
            "task": asdict(spec),
            "model": asdict(model.config),
            "controller": "J(h)=hW+b",
            "controller_parameterization": "dense_affine",
            "rank": None,
            "seed": args.seed,
            "anchor_step": anchor_step,
            "loss": "final registered T(n) answer-region task CE only",
            "state_loss_weight": 0.0,
            "controller_curriculum": "logical_range",
            "controller_training_profile": "mixed_length_wsd",
            "controller_initialization": "identity",
            "controller_logical_min_length": args.minimum_length,
            "controller_logical_max_length": args.boundary_maximum,
            "controller_sampled_logical_lengths": sorted(initial_counts),
            "controller_logical_length_example_counts": initial_counts,
            "controller_post_final_j": False,
            "snapshot_update": 0,
            "snapshot_total_updates": args.target_total_update,
            "training_budget": {
                "dense_optimizer_updates": 0,
                "low_rank_optimizer_updates": 0,
                "total_optimizer_updates": 0,
                "dense_training_examples": 0,
                "low_rank_training_examples": 0,
                "total_training_examples": 0,
            },
            "controller_state_dict": _state_dict_cpu(controller),
        }
        source_controller = args.out_dir / "identity_initial_controller.pt"
        atomic_torch_save(source_payload, source_controller)
    else:
        if args.source_controller is None:
            raise ValueError(
                "provide --source-controller or --initialize-dense-identity"
            )
        source_controller = args.source_controller
        controller, source_payload = load_controller(
            source_controller, device=device
        )
    validate_model_payload_compatibility(
        source_payload["model"], backbone_payload["model"]
    )
    validate_task_payload_compatibility(
        source_payload["task"], backbone_payload["task"]
    )
    source_total_update = int(
        source_payload["training_budget"]["total_optimizer_updates"]
    )
    if args.start_total_update is not None and (
        args.start_total_update != source_total_update
    ):
        raise ValueError("declared start update does not match source artifact")
    schedule = validate_continuation_schedule(
        start_total_update=source_total_update,
        target_total_update=args.target_total_update,
        warmup_updates=args.warmup_updates,
        stable_updates=args.stable_updates,
        final_learning_rate_ratio=args.final_learning_rate_ratio,
    )
    bands = make_length_bands(
        minimum_length=args.minimum_length,
        retention_maximum=args.retention_maximum,
        transition_maximum=args.transition_maximum,
        boundary_maximum=args.boundary_maximum,
        retention_probability=args.retention_probability,
        transition_probability=args.transition_probability,
        boundary_probability=args.boundary_probability,
    )
    anchor_step = int(source_payload["anchor_step"])
    if args.anchor_step is not None and args.anchor_step != anchor_step:
        raise ValueError("declared anchor step does not match source artifact")
    post_final_controller = bool(source_payload["controller_post_final_j"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    controller.train()
    optimizer = _make_optimizer(
        controller,
        peak_learning_rate=args.peak_learning_rate,
        diagonal_learning_rate_multiplier=args.diagonal_lr_multiplier,
    )
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(args.seed + 17)
    set_seed(args.seed)

    configuration = {
        "checkpoint": str(args.checkpoint),
        "source_controller": str(source_controller),
        "schedule": asdict(schedule),
        "length_mix": [asdict(band) for band in bands],
        "batch_size": args.batch_size,
        "peak_learning_rate": args.peak_learning_rate,
        "diagonal_lr_multiplier": args.diagonal_lr_multiplier,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "anchor_step": anchor_step,
        "post_final_controller": post_final_controller,
        "eval_lengths": list(args.eval_lengths),
        "selection_lengths": list(args.selection_lengths),
        "selection_mode": args.selection_mode,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
    }
    local_update = 0
    length_counts: dict[int, int] = {}
    band_counts = {band.name: 0 for band in bands}
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    best_score = math.inf
    best_total_update = source_total_update
    no_gradient_updates = 0
    elapsed_before_resume = 0.0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("kind") != "paper_length_controller_continuation_checkpoint":
            raise ValueError("resume file is not a continuation checkpoint")
        if resume["configuration"] != configuration:
            raise ValueError("continuation resume configuration does not match")
        controller.load_state_dict(resume["controller_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        restore_generator_state(data_generator, resume["data_generator_state"])
        _restore_rng_state(resume["rng_state"])
        local_update = int(resume["local_update"])
        if int(resume["total_update"]) != source_total_update + local_update:
            raise ValueError("resume local and total update counters disagree")
        length_counts = {
            int(length): int(count)
            for length, count in resume["length_counts"].items()
        }
        band_counts = {
            str(name): int(count) for name, count in resume["band_counts"].items()
        }
        history = list(resume["history"])
        evaluations = list(resume["evaluations"])
        best_score = float(resume["best_score"])
        best_total_update = int(resume["best_total_update"])
        no_gradient_updates = int(resume.get("no_gradient_updates", 0))
        elapsed_before_resume = float(resume["elapsed_seconds"])

    out_dir = args.out_dir
    started_at = time.monotonic()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + time.monotonic() - started_at

    def save_standard(path: Path, total_update: int) -> None:
        atomic_torch_save(
            _standard_controller_artifact(
                source_payload=source_payload,
                controller=controller,
                source_controller=source_controller,
                schedule=schedule,
                bands=bands,
                total_update=total_update,
                batch_size=args.batch_size,
                length_counts=length_counts,
                peak_learning_rate=args.peak_learning_rate,
                diagonal_learning_rate_multiplier=args.diagonal_lr_multiplier,
                seed=args.seed,
            ),
            path,
        )

    def run_evaluation(total_update: int) -> None:
        nonlocal best_score, best_total_update
        metrics = evaluate_controller(
            model=model,
            spec=spec,
            controller=controller,
            anchor_step=anchor_step,
            post_final_controller=post_final_controller,
            lengths=args.eval_lengths,
            batch_size=args.eval_batch_size,
            batches=args.eval_batches,
            seed=args.eval_seed,
            device=device,
        )
        selection_lengths = tuple(args.selection_lengths)
        missing = sorted(set(selection_lengths) - set(metrics))
        if missing:
            raise ValueError(
                f"selection lengths are missing from evaluation: {missing}"
            )
        selection_values = [
            metrics[length]["answer_ce"] for length in selection_lengths
        ]
        if args.selection_mode == "mean_ce":
            score = sum(selection_values) / len(selection_values)
        else:
            score = max(selection_values)
        row: dict[str, Any] = {
            "total_update": total_update,
            "local_update": total_update - source_total_update,
            "selection_boundary_ce": score,
        }
        for length, values in metrics.items():
            row[f"L{length}_ce"] = values["answer_ce"]
            row[f"L{length}_em"] = values["exact_match"]
        evaluations.append(row)
        print(json.dumps({"event": "evaluation", **row}, sort_keys=True), flush=True)
        snapshot_directory = out_dir / "controller_snapshots" / "checkpoints"
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        save_standard(
            snapshot_directory / f"controller_{total_update:06d}.pt",
            total_update,
        )
        if score < best_score:
            best_score = score
            best_total_update = total_update
            save_standard(out_dir / "best.pt", total_update)

    if local_update == 0 and not evaluations:
        run_evaluation(source_total_update)

    rolling_loss = 0.0
    rolling_em = 0.0
    rolling_grad = 0.0
    rolling_updates = 0
    while local_update < schedule.additional_updates:
        local_update += 1
        total_update = source_total_update + local_update
        logical_length = sample_mixed_logical_length(data_generator, bands)
        band_name = _band_name(logical_length, bands)
        batch = generate_paper_batch(
            spec,
            batch_size=args.batch_size,
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=data_generator,
        ).to(device)
        plan = _logical_controller_plan(
            spec,
            logical_length,
            controller_anchor_step=anchor_step,
        )
        learning_rate = continuation_learning_rate(
            schedule,
            local_update=local_update,
            peak=args.peak_learning_rate,
        )
        optimizer.param_groups[0]["lr"] = learning_rate
        if isinstance(controller, DiagonalLowRankController):
            optimizer.param_groups[1]["lr"] = (
                learning_rate * args.diagonal_lr_multiplier
            )
        optimizer.zero_grad(set_to_none=True)
        loss, em = _controlled_ce_batch(
            model=model,
            controller=controller,
            batch=batch,
            anchor_step=anchor_step,
            controlled_steps=plan.controlled_steps,
            post_final_controller=post_final_controller,
        )
        grad_norm, applied_gradient = _controller_optimizer_step(
            loss=loss,
            controller=controller,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        if not applied_gradient:
            no_gradient_updates += 1
        length_counts[logical_length] = (
            length_counts.get(logical_length, 0) + args.batch_size
        )
        band_counts[band_name] += args.batch_size
        rolling_loss += float(loss.detach())
        rolling_em += em
        rolling_grad += grad_norm
        rolling_updates += 1

        should_log = local_update % args.log_every == 0
        if should_log or local_update == schedule.additional_updates:
            row = {
                "total_update": total_update,
                "local_update": local_update,
                "answer_ce": rolling_loss / rolling_updates,
                "exact_match": rolling_em / rolling_updates,
                "preclip_gradient_norm": rolling_grad / rolling_updates,
                "learning_rate": learning_rate,
                "last_logical_length": logical_length,
                "last_band": band_name,
                "elapsed_seconds": elapsed_seconds(),
            }
            history.append(row)
            print(json.dumps({"event": "training", **row}, sort_keys=True), flush=True)
            rolling_loss = 0.0
            rolling_em = 0.0
            rolling_grad = 0.0
            rolling_updates = 0

        should_evaluate = (
            local_update % args.eval_every == 0
            or local_update == schedule.additional_updates
        )
        if should_evaluate:
            run_evaluation(total_update)
            write_csv(out_dir / "training.csv", history)
            write_csv(out_dir / "evaluations.csv", evaluations)

        should_checkpoint = (
            local_update % args.checkpoint_every == 0
            or local_update == schedule.additional_updates
        )
        if should_checkpoint or should_evaluate:
            checkpoint = _checkpoint_payload(
                source_controller=source_controller,
                source_payload=source_payload,
                controller=controller,
                optimizer=optimizer,
                data_generator=data_generator,
                configuration=configuration,
                local_update=local_update,
                total_update=total_update,
                length_counts=length_counts,
                band_counts=band_counts,
                history=history,
                evaluations=evaluations,
                best_score=best_score,
                best_total_update=best_total_update,
                no_gradient_updates=no_gradient_updates,
                elapsed_seconds=elapsed_seconds(),
            )
            atomic_torch_save(checkpoint, out_dir / "latest.pt")
            if should_checkpoint:
                atomic_torch_save(
                    checkpoint, out_dir / f"checkpoint_{total_update:06d}.pt"
                )

    final_total_update = source_total_update + local_update
    save_standard(out_dir / "controller.pt", final_total_update)
    summary = {
        "status": "complete",
        "source_controller": str(source_controller),
        "controller_parameterization": source_payload.get(
            "controller_parameterization", "diagonal_low_rank"
        ),
        "source_total_update": source_total_update,
        "final_total_update": final_total_update,
        "schedule": asdict(schedule),
        "length_mix": [asdict(band) for band in bands],
        "continuation_length_example_counts": dict(sorted(length_counts.items())),
        "continuation_band_example_counts": band_counts,
        "best_total_update": best_total_update,
        "best_selection_boundary_ce": best_score,
        "no_gradient_updates": no_gradient_updates,
        "gradient_updates": schedule.additional_updates - no_gradient_updates,
        "elapsed_seconds": elapsed_seconds(),
        "peak_cuda_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_json_dump(summary, out_dir / "summary.json")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or continue a paper-task J to a fixed update budget."
    )
    parser.add_argument(
        "--task", choices=("copy", "addition", "sum_reverse"), default="copy"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-controller", type=Path)
    parser.add_argument(
        "--initialize-dense-identity",
        action="store_true",
        help="start a full dense affine J exactly at identity at update zero",
    )
    parser.add_argument(
        "--anchor-step",
        type=int,
        help="required for dense identity initialization; otherwise validates the source",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--start-total-update", type=int)
    parser.add_argument("--target-total-update", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-length", type=int, default=1)
    parser.add_argument("--retention-probability", type=float, default=0.20)
    parser.add_argument("--transition-probability", type=float, default=0.30)
    parser.add_argument("--boundary-probability", type=float, default=0.50)
    parser.add_argument("--retention-maximum", type=int, default=19)
    parser.add_argument("--transition-maximum", type=int, default=32)
    parser.add_argument("--boundary-maximum", type=int, default=40)
    parser.add_argument("--peak-learning-rate", type=float, default=3e-4)
    parser.add_argument("--diagonal-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--warmup-updates", type=int, default=5_000)
    parser.add_argument("--stable-updates", type=int, default=35_000)
    parser.add_argument("--final-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=311_001)
    parser.add_argument("--eval-seed", type=int, default=361_001)
    parser.add_argument(
        "--eval-lengths", type=int, nargs="+", default=(1, 19, 25, 32, 35, 38, 40, 50)
    )
    parser.add_argument(
        "--selection-lengths",
        type=int,
        nargs="+",
        default=(35, 38, 40),
        help="Evaluation lengths whose mean CE selects best.pt.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("mean_ce", "max_ce"),
        default="mean_ce",
        help="Select best.pt by mean or worst-length CE.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in (
        "batch_size",
        "eval_batch_size",
        "eval_batches",
        "eval_every",
        "log_every",
        "checkpoint_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.peak_learning_rate <= 0:
        parser.error("--peak-learning-rate must be positive")
    if args.diagonal_lr_multiplier <= 0:
        parser.error("--diagonal-lr-multiplier must be positive")
    if args.grad_clip <= 0:
        parser.error("--grad-clip must be positive")
    if args.initialize_dense_identity == (args.source_controller is not None):
        parser.error(
            "choose exactly one of --source-controller and --initialize-dense-identity"
        )
    if args.initialize_dense_identity and args.anchor_step is None:
        parser.error("--initialize-dense-identity requires --anchor-step")
    if args.anchor_step is not None and args.anchor_step < 0:
        parser.error("--anchor-step must be non-negative")
    if not (
        1 <= args.minimum_length <= args.retention_maximum
        < args.transition_maximum
        < args.boundary_maximum
    ):
        parser.error(
            "minimum length and length-band maxima must be positive and increasing"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    summary = train(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
