from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.continue_paper_length_controller import (
    _make_optimizer,
    _restore_rng_state,
    _rng_state,
    _standard_controller_artifact,
    continuation_learning_rate,
    make_length_bands,
    restore_generator_state,
    sample_mixed_logical_length,
    validate_continuation_schedule,
    validate_task_payload_compatibility,
)
from reasoning_loop.paper_length_telomere import (
    DiagonalLowRankController,
    atomic_json_dump,
    atomic_torch_save,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    masked_cross_entropy,
    pick_device,
    set_seed,
    write_csv,
)


F = "F"
J = "J"


@dataclass(frozen=True)
class AgePath:
    actions: tuple[str, ...]
    family: str
    forward_steps: int
    maximum_age: int
    maximum_j_run: int


def analyze_age_path(actions: Sequence[str], *, forward_steps: int) -> AgePath:
    """Validate a balanced age path and return its realized statistics.

    F advances the frozen task executor and increases interface age by one.
    J leaves task progress fixed and decreases interface age by one.  A legal
    path starts at age zero, never becomes negative, and finishes at age zero
    after exactly ``forward_steps`` calls of each operator.
    """
    if forward_steps < 1:
        raise ValueError("forward_steps must be positive")
    age = 0
    f_count = 0
    j_count = 0
    maximum_age = 0
    current_j_run = 0
    maximum_j_run = 0
    checked = tuple(actions)
    for action in checked:
        if action == F:
            f_count += 1
            age += 1
            current_j_run = 0
            maximum_age = max(maximum_age, age)
        elif action == J:
            j_count += 1
            age -= 1
            current_j_run += 1
            maximum_j_run = max(maximum_j_run, current_j_run)
            if age < 0:
                raise ValueError("age path rejuvenates below age zero")
        else:
            raise ValueError(f"unsupported age-path action: {action!r}")
    if f_count != forward_steps or j_count != forward_steps:
        raise ValueError("age path must contain equal declared F and J counts")
    if age != 0:
        raise ValueError("age path must finish at age zero")
    return AgePath(
        actions=checked,
        family="validated",
        forward_steps=forward_steps,
        maximum_age=maximum_age,
        maximum_j_run=maximum_j_run,
    )


def _path(actions: Sequence[str], *, family: str, forward_steps: int) -> AgePath:
    stats = analyze_age_path(actions, forward_steps=forward_steps)
    return AgePath(
        actions=stats.actions,
        family=family,
        forward_steps=stats.forward_steps,
        maximum_age=stats.maximum_age,
        maximum_j_run=stats.maximum_j_run,
    )


def alternating_age_path(forward_steps: int) -> AgePath:
    return _path(
        (F, J) * forward_steps,
        family="alternating",
        forward_steps=forward_steps,
    )


def burst_age_path(forward_steps: int, burst_size: int) -> AgePath:
    if burst_size < 1:
        raise ValueError("burst size must be positive")
    actions: list[str] = []
    remaining = forward_steps
    while remaining:
        live = min(remaining, burst_size)
        actions.extend((F,) * live)
        actions.extend((J,) * live)
        remaining -= live
    return _path(
        actions,
        family=f"burst_{burst_size}",
        forward_steps=forward_steps,
    )


def full_burst_age_path(forward_steps: int) -> AgePath:
    return _path(
        (F,) * forward_steps + (J,) * forward_steps,
        family="full_burst_heldout",
        forward_steps=forward_steps,
    )


def random_dyck_age_path(
    forward_steps: int,
    *,
    maximum_age: int,
    generator: torch.Generator,
    family: str = "random_dyck",
    forbid_full_burst: bool = True,
) -> AgePath:
    if maximum_age < 1:
        raise ValueError("maximum age must be positive")
    age_cap = min(forward_steps, maximum_age)
    for _ in range(64):
        actions: list[str] = []
        f_count = 0
        j_count = 0
        age = 0
        while f_count < forward_steps or j_count < forward_steps:
            can_f = f_count < forward_steps and age < age_cap
            can_j = j_count < forward_steps and age > 0
            if not can_f and not can_j:
                raise RuntimeError("Dyck sampler reached an invalid dead end")
            if can_f and can_j:
                # A mild forward bias exposes the controller to sustained aging
                # without degenerating into the held-out all-F/all-J path.
                choose_f = bool(
                    torch.rand((), generator=generator).item() < 0.58
                )
            else:
                choose_f = can_f
            if choose_f:
                actions.append(F)
                f_count += 1
                age += 1
            else:
                actions.append(J)
                j_count += 1
                age -= 1
        if not forbid_full_burst or tuple(actions) != (
            (F,) * forward_steps + (J,) * forward_steps
        ):
            return _path(
                actions,
                family=family,
                forward_steps=forward_steps,
            )
    # This occurs only for tiny path spaces.  The canonical path is valid and
    # preserves the strict no-full-burst training contract.
    return alternating_age_path(forward_steps)


def curriculum_maximum_age(
    *, local_update: int, total_updates: int, forward_steps: int
) -> int:
    if not 1 <= local_update <= total_updates:
        raise ValueError("curriculum update is outside the training run")
    progress = local_update / total_updates
    if progress <= 0.10:
        cap = 1
    elif progress <= 0.25:
        cap = 2
    elif progress <= 0.45:
        cap = 4
    elif progress <= 0.70:
        cap = 8
    else:
        cap = 20
    return min(forward_steps, cap)


def sample_training_age_path(
    forward_steps: int,
    *,
    maximum_age: int,
    generator: torch.Generator,
) -> AgePath:
    if maximum_age <= 1 or forward_steps <= 2:
        return alternating_age_path(forward_steps)
    family_index = int(
        torch.multinomial(
            torch.tensor((0.20, 0.30, 0.50), dtype=torch.float64),
            1,
            generator=generator,
        ).item()
    )
    if family_index == 0:
        return alternating_age_path(forward_steps)
    if family_index == 1:
        # Never use one all-F/all-J block in training.  It remains a genuinely
        # held-out path family at audit time.
        largest = min(maximum_age, forward_steps - 1)
        if largest < 2:
            return alternating_age_path(forward_steps)
        burst_size = int(
            torch.randint(2, largest + 1, (1,), generator=generator).item()
        )
        return burst_age_path(forward_steps, burst_size)
    return random_dyck_age_path(
        forward_steps,
        maximum_age=maximum_age,
        generator=generator,
        forbid_full_burst=True,
    )


def execute_actions(
    *,
    model: torch.nn.Module,
    controller: DiagonalLowRankController | None,
    batch: Any,
    actions: Sequence[str],
) -> torch.Tensor:
    embedded = model.input_embeddings(batch.inputs)
    state = torch.zeros_like(embedded)
    for action in actions:
        if action == F:
            state = model.recurrent_step(state, embedded)
        elif action == J:
            if controller is None:
                raise ValueError("J action requires a controller")
            state = controller(state)
        else:
            raise ValueError(f"unsupported action: {action!r}")
    return model.decode(state).float()


def controlled_ce_age_path(
    *,
    model: torch.nn.Module,
    controller: DiagonalLowRankController,
    batch: Any,
    path: AgePath,
) -> tuple[torch.Tensor, float]:
    logits = execute_actions(
        model=model,
        controller=controller,
        batch=batch,
        actions=path.actions,
    )
    return masked_cross_entropy(logits, batch), exact_match(logits.detach(), batch)


def _evaluation_paths(
    forward_steps: int, *, generator: torch.Generator
) -> tuple[AgePath, ...]:
    paths = [
        alternating_age_path(forward_steps),
        burst_age_path(forward_steps, 2),
        burst_age_path(forward_steps, 4),
        burst_age_path(forward_steps, 8),
        random_dyck_age_path(
            forward_steps,
            maximum_age=min(8, forward_steps),
            generator=generator,
            family="random_dyck_heldout_seed",
            forbid_full_burst=True,
        ),
        full_burst_age_path(forward_steps),
    ]
    unique: dict[tuple[str, ...], AgePath] = {}
    for path in paths:
        unique.setdefault(path.actions, path)
    return tuple(unique.values())


@torch.no_grad()
def evaluate_id_path_suite(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: DiagonalLowRankController,
    lengths: Sequence[int],
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
    include_full_burst: bool,
) -> tuple[list[dict[str, Any]], float]:
    controller.eval()
    rows: list[dict[str, Any]] = []
    selection_losses: list[float] = []
    for logical_length in lengths:
        forward_steps = logical_length + spec.step_offset
        path_generator = torch.Generator(device="cpu")
        path_generator.manual_seed(seed + 1_000_003 * logical_length)
        paths = _evaluation_paths(forward_steps, generator=path_generator)
        if not include_full_burst:
            paths = tuple(
                path for path in paths if path.family != "full_burst_heldout"
            )
        accumulators = {
            path.family: {"ce": 0.0, "em": 0.0} for path in paths
        }
        for batch_index in range(batches):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                seed + 10_000 * logical_length + batch_index
            )
            batch = generate_paper_batch(
                spec,
                batch_size=batch_size,
                min_length=logical_length,
                max_length=logical_length,
                fixed_length=logical_length,
                generator=generator,
            ).to(device)
            for path in paths:
                loss, em = controlled_ce_age_path(
                    model=model,
                    controller=controller,
                    batch=batch,
                    path=path,
                )
                accumulators[path.family]["ce"] += float(loss)
                accumulators[path.family]["em"] += em
        for path in paths:
            row = {
                "logical_length": logical_length,
                "path_family": path.family,
                "answer_ce": accumulators[path.family]["ce"] / batches,
                "exact_match": accumulators[path.family]["em"] / batches,
                "maximum_age": path.maximum_age,
                "maximum_j_run": path.maximum_j_run,
                "actions": len(path.actions),
                "path_sha256": hashlib.sha256(
                    "".join(path.actions).encode("ascii")
                ).hexdigest(),
            }
            rows.append(row)
            if path.family != "full_burst_heldout":
                selection_losses.append(float(row["answer_ce"]))
    controller.train()
    if not selection_losses:
        raise ValueError("ID path evaluation produced no selection cases")
    return rows, sum(selection_losses) / len(selection_losses)


def _save_controller_artifact(
    *,
    source_payload: dict[str, Any],
    controller: DiagonalLowRankController,
    source_controller: Path,
    schedule: Any,
    bands: Sequence[Any],
    total_update: int,
    batch_size: int,
    length_counts: dict[int, int],
    peak_learning_rate: float,
    diagonal_learning_rate_multiplier: float,
    seed: int,
    path_counts: Counter[str],
    maximum_age_counts: Counter[int],
    path_protocol: dict[str, Any],
    path: Path,
) -> None:
    artifact = _standard_controller_artifact(
        source_payload=source_payload,
        controller=controller,
        source_controller=source_controller,
        schedule=schedule,
        bands=bands,
        total_update=total_update,
        batch_size=batch_size,
        length_counts=length_counts,
        peak_learning_rate=peak_learning_rate,
        diagonal_learning_rate_multiplier=diagonal_learning_rate_multiplier,
        seed=seed,
    )
    artifact.update(
        {
            "controller_curriculum": "logical_range_age_path",
            "loss": (
                "final registered T(n) answer-region task CE only over "
                "balanced F/J age paths; no hidden or intermediate loss"
            ),
            "age_path_training": {
                **copy.deepcopy(path_protocol),
                "path_family_examples": dict(sorted(path_counts.items())),
                "maximum_age_examples": {
                    str(key): value
                    for key, value in sorted(maximum_age_counts.items())
                },
            },
        }
    )
    atomic_torch_save(artifact, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    controller, source_payload = load_controller(
        args.source_controller, device=device
    )
    if spec.name != "addition":
        raise ValueError("robust age-path training is frozen to Addition")
    validate_task_payload_compatibility(
        source_payload["task"], backbone_payload["task"]
    )
    if not source_payload.get("controller_post_final_j", False):
        raise ValueError("source controller must implement canonical (FJ)^T")
    source_total_update = int(
        source_payload["training_budget"]["total_optimizer_updates"]
    )
    schedule = validate_continuation_schedule(
        start_total_update=source_total_update,
        target_total_update=args.target_total_update,
        warmup_updates=args.warmup_updates,
        stable_updates=args.stable_updates,
        final_learning_rate_ratio=args.final_learning_rate_ratio,
    )
    bands = make_length_bands(
        retention_maximum=args.retention_maximum,
        transition_maximum=args.transition_maximum,
        boundary_maximum=args.boundary_maximum,
        retention_probability=args.retention_probability,
        transition_probability=args.transition_probability,
        boundary_probability=args.boundary_probability,
    )
    if bands[0].minimum != 1 or bands[-1].maximum != 20:
        raise ValueError("robust Addition training must use exactly lengths 1--20")

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

    path_protocol = {
        "semantics": {
            "F": "advance Addition executor and increase interface age by one",
            "J": "leave task progress fixed and decrease interface age by one",
        },
        "constraints": {
            "training_logical_lengths": [1, 20],
            "equal_F_and_J_counts": True,
            "nonnegative_prefix_age": True,
            "final_age": 0,
            "heldout_family": "all F followed by all J",
        },
        "family_probabilities_after_warmup": {
            "alternating": 0.20,
            "bounded_burst": 0.30,
            "random_dyck": 0.50,
        },
        "maximum_age_curriculum": [
            [0.00, 0.10, 1],
            [0.10, 0.25, 2],
            [0.25, 0.45, 4],
            [0.45, 0.70, 8],
            [0.70, 1.00, 20],
        ],
        "supervision": "final answer-region CE only",
        "hidden_state_targets": False,
        "intermediate_losses": False,
    }
    configuration = {
        "task": "addition",
        "checkpoint": str(args.checkpoint),
        "source_controller": str(args.source_controller),
        "schedule": asdict(schedule),
        "length_mix": [asdict(band) for band in bands],
        "batch_size": args.batch_size,
        "peak_learning_rate": args.peak_learning_rate,
        "diagonal_lr_multiplier": args.diagonal_lr_multiplier,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "path_protocol": path_protocol,
        "eval_lengths": list(args.eval_lengths),
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
    }
    local_update = 0
    length_counts: dict[int, int] = {}
    path_counts: Counter[str] = Counter()
    maximum_age_counts: Counter[int] = Counter()
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    best_score = math.inf
    best_total_update = source_total_update
    elapsed_before_resume = 0.0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("kind") != "robust_addition_age_path_checkpoint":
            raise ValueError("resume file is not a robust age-path checkpoint")
        if resume["configuration"] != configuration:
            raise ValueError("resume configuration does not match")
        controller.load_state_dict(resume["controller_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        restore_generator_state(data_generator, resume["data_generator_state"])
        _restore_rng_state(resume["rng_state"])
        local_update = int(resume["local_update"])
        length_counts = {
            int(key): int(value) for key, value in resume["length_counts"].items()
        }
        path_counts = Counter(resume["path_counts"])
        maximum_age_counts = Counter(
            {int(key): int(value) for key, value in resume["maximum_age_counts"].items()}
        )
        history = list(resume["history"])
        evaluations = list(resume["evaluations"])
        best_score = float(resume["best_score"])
        best_total_update = int(resume["best_total_update"])
        elapsed_before_resume = float(resume["elapsed_seconds"])

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + time.monotonic() - started_at

    def save_standard(path: Path, total_update: int) -> None:
        _save_controller_artifact(
            source_payload=source_payload,
            controller=controller,
            source_controller=args.source_controller,
            schedule=schedule,
            bands=bands,
            total_update=total_update,
            batch_size=args.batch_size,
            length_counts=length_counts,
            peak_learning_rate=args.peak_learning_rate,
            diagonal_learning_rate_multiplier=args.diagonal_lr_multiplier,
            seed=args.seed,
            path_counts=path_counts,
            maximum_age_counts=maximum_age_counts,
            path_protocol=path_protocol,
            path=path,
        )

    def save_checkpoint(path: Path, total_update: int) -> None:
        atomic_torch_save(
            {
                "kind": "robust_addition_age_path_checkpoint",
                "configuration": configuration,
                "local_update": local_update,
                "total_update": total_update,
                "controller_state_dict": {
                    key: value.detach().cpu()
                    for key, value in controller.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "data_generator_state": data_generator.get_state(),
                "rng_state": _rng_state(),
                "length_counts": length_counts,
                "path_counts": dict(path_counts),
                "maximum_age_counts": dict(maximum_age_counts),
                "history": history,
                "evaluations": evaluations,
                "best_score": best_score,
                "best_total_update": best_total_update,
                "elapsed_seconds": elapsed_seconds(),
            },
            path,
        )

    def run_evaluation(total_update: int) -> None:
        nonlocal best_score, best_total_update
        rows, score = evaluate_id_path_suite(
            model=model,
            spec=spec,
            controller=controller,
            lengths=args.eval_lengths,
            batch_size=args.eval_batch_size,
            batches=args.eval_batches,
            seed=args.eval_seed,
            device=device,
            include_full_burst=False,
        )
        row: dict[str, Any] = {
            "total_update": total_update,
            "local_update": total_update - source_total_update,
            "id_path_selection_ce": score,
        }
        for value in rows:
            key = f"L{value['logical_length']}_{value['path_family']}"
            row[f"{key}_ce"] = value["answer_ce"]
            row[f"{key}_em"] = value["exact_match"]
        evaluations.append(row)
        print(json.dumps({"event": "evaluation", **row}, sort_keys=True), flush=True)
        if score < best_score:
            best_score = score
            best_total_update = total_update
            save_standard(out_dir / "best_id.pt", total_update)

    if local_update == 0 and not evaluations:
        run_evaluation(source_total_update)

    rolling = {"loss": 0.0, "em": 0.0, "grad": 0.0, "updates": 0}
    while local_update < schedule.additional_updates:
        local_update += 1
        total_update = source_total_update + local_update
        logical_length = sample_mixed_logical_length(data_generator, bands)
        forward_steps = logical_length + spec.step_offset
        maximum_age = curriculum_maximum_age(
            local_update=local_update,
            total_updates=schedule.additional_updates,
            forward_steps=forward_steps,
        )
        path = sample_training_age_path(
            forward_steps,
            maximum_age=maximum_age,
            generator=data_generator,
        )
        batch = generate_paper_batch(
            spec,
            batch_size=args.batch_size,
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=data_generator,
        ).to(device)
        learning_rate = continuation_learning_rate(
            schedule,
            local_update=local_update,
            peak=args.peak_learning_rate,
        )
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.param_groups[1]["lr"] = (
            learning_rate * args.diagonal_lr_multiplier
        )
        optimizer.zero_grad(set_to_none=True)
        loss, em = controlled_ce_age_path(
            model=model,
            controller=controller,
            batch=batch,
            path=path,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            controller.parameters(), args.grad_clip
        )
        optimizer.step()

        length_counts[logical_length] = (
            length_counts.get(logical_length, 0) + args.batch_size
        )
        path_counts[path.family] += args.batch_size
        maximum_age_counts[path.maximum_age] += args.batch_size
        rolling["loss"] += float(loss.detach())
        rolling["em"] += em
        rolling["grad"] += float(grad_norm)
        rolling["updates"] += 1

        if local_update % args.log_every == 0 or local_update == schedule.additional_updates:
            denominator = rolling["updates"]
            row = {
                "total_update": total_update,
                "local_update": local_update,
                "answer_ce": rolling["loss"] / denominator,
                "exact_match": rolling["em"] / denominator,
                "preclip_gradient_norm": rolling["grad"] / denominator,
                "learning_rate": learning_rate,
                "last_logical_length": logical_length,
                "last_path_family": path.family,
                "last_maximum_age": path.maximum_age,
                "last_maximum_j_run": path.maximum_j_run,
                "elapsed_seconds": elapsed_seconds(),
            }
            history.append(row)
            print(json.dumps({"event": "training", **row}, sort_keys=True), flush=True)
            rolling = {"loss": 0.0, "em": 0.0, "grad": 0.0, "updates": 0}

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
        if should_evaluate or should_checkpoint:
            save_checkpoint(out_dir / "latest.pt", total_update)
            if should_checkpoint:
                save_checkpoint(
                    out_dir / f"checkpoint_{total_update:06d}.pt",
                    total_update,
                )

    final_total_update = source_total_update + local_update
    save_standard(out_dir / "controller.pt", final_total_update)
    summary = {
        "status": "complete",
        "source_controller": str(args.source_controller),
        "source_total_update": source_total_update,
        "final_total_update": final_total_update,
        "strict_training_logical_range": [1, 20],
        "schedule": asdict(schedule),
        "length_mix": [asdict(band) for band in bands],
        "continuation_length_example_counts": dict(sorted(length_counts.items())),
        "path_family_example_counts": dict(sorted(path_counts.items())),
        "maximum_age_example_counts": {
            str(key): value for key, value in sorted(maximum_age_counts.items())
        },
        "best_total_update_selected_on_id_paths_only": best_total_update,
        "best_id_path_selection_ce": best_score,
        "path_protocol": path_protocol,
        "elapsed_seconds": elapsed_seconds(),
        "peak_cuda_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_json_dump(summary, out_dir / "summary.json")
    return summary


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    controller, payload = load_controller(args.controller, device=device)
    validate_task_payload_compatibility(payload["task"], backbone_payload["task"])
    model.eval()
    controller.eval()
    rows: list[dict[str, Any]] = []
    for logical_length in args.lengths:
        forward_steps = logical_length + spec.step_offset
        path_generator = torch.Generator(device="cpu")
        path_generator.manual_seed(args.path_seed + 1_000_003 * logical_length)
        paths = _evaluation_paths(forward_steps, generator=path_generator)
        totals: dict[str, dict[str, float]] = {
            "raw": {"ce": 0.0, "em": 0.0}
        }
        totals.update(
            {path.family: {"ce": 0.0, "em": 0.0} for path in paths}
        )
        for batch_index in range(args.batches):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                args.seed + 10_000 * logical_length + batch_index
            )
            batch = generate_paper_batch(
                spec,
                batch_size=args.batch_size,
                min_length=logical_length,
                max_length=logical_length,
                fixed_length=logical_length,
                generator=generator,
            ).to(device)
            raw_logits = execute_actions(
                model=model,
                controller=None,
                batch=batch,
                actions=(F,) * forward_steps,
            )
            totals["raw"]["ce"] += float(masked_cross_entropy(raw_logits, batch))
            totals["raw"]["em"] += exact_match(raw_logits, batch)
            for path in paths:
                loss, em = controlled_ce_age_path(
                    model=model,
                    controller=controller,
                    batch=batch,
                    path=path,
                )
                totals[path.family]["ce"] += float(loss)
                totals[path.family]["em"] += em
        rows.append(
            {
                "logical_length": logical_length,
                "variant": "raw",
                "answer_ce": totals["raw"]["ce"] / args.batches,
                "exact_match": totals["raw"]["em"] / args.batches,
                "examples": args.batch_size * args.batches,
                "forward_steps": forward_steps,
                "j_steps": 0,
                "maximum_age": forward_steps,
                "maximum_j_run": 0,
                "path_sha256": hashlib.sha256(
                    (F * forward_steps).encode("ascii")
                ).hexdigest(),
            }
        )
        for path in paths:
            rows.append(
                {
                    "logical_length": logical_length,
                    "variant": path.family,
                    "answer_ce": totals[path.family]["ce"] / args.batches,
                    "exact_match": totals[path.family]["em"] / args.batches,
                    "examples": args.batch_size * args.batches,
                    "forward_steps": forward_steps,
                    "j_steps": forward_steps,
                    "maximum_age": path.maximum_age,
                    "maximum_j_run": path.maximum_j_run,
                    "path_sha256": hashlib.sha256(
                        "".join(path.actions).encode("ascii")
                    ).hexdigest(),
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "path_audit.csv", rows)
    summary = {
        "status": "complete",
        "task": asdict(spec),
        "backbone": str(args.checkpoint),
        "controller": str(args.controller),
        "controller_age_path_training": payload.get("age_path_training"),
        "lengths": list(args.lengths),
        "examples_per_length_path": args.batch_size * args.batches,
        "evaluation_seed": args.seed,
        "heldout_path_seed": args.path_seed,
        "selection_policy": "no OOD length or full-burst family used for training/checkpoint selection",
        "rows": rows,
    }
    atomic_json_dump(summary, args.out_dir / "summary.json")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or audit an Addition J on balanced F/J age paths."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--source-controller", type=Path, required=True)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--target-total-update", type=int, default=50_000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--retention-maximum", type=int, default=10)
    train_parser.add_argument("--transition-maximum", type=int, default=15)
    train_parser.add_argument("--boundary-maximum", type=int, default=20)
    train_parser.add_argument("--retention-probability", type=float, default=0.20)
    train_parser.add_argument("--transition-probability", type=float, default=0.30)
    train_parser.add_argument("--boundary-probability", type=float, default=0.50)
    train_parser.add_argument("--peak-learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--diagonal-lr-multiplier", type=float, default=0.1)
    train_parser.add_argument("--warmup-updates", type=int, default=5_000)
    train_parser.add_argument("--stable-updates", type=int, default=35_000)
    train_parser.add_argument("--final-learning-rate-ratio", type=float, default=0.1)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=411_001)
    train_parser.add_argument("--eval-seed", type=int, default=461_001)
    train_parser.add_argument(
        "--eval-lengths", type=int, nargs="+", default=(10, 15, 19, 20)
    )
    train_parser.add_argument("--eval-batch-size", type=int, default=64)
    train_parser.add_argument("--eval-batches", type=int, default=2)
    train_parser.add_argument("--eval-every", type=int, default=1_000)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--checkpoint-every", type=int, default=5_000)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--out-dir", type=Path, required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--checkpoint", type=Path, required=True)
    audit_parser.add_argument("--controller", type=Path, required=True)
    audit_parser.add_argument("--lengths", type=int, nargs="+", required=True)
    audit_parser.add_argument("--batch-size", type=int, default=64)
    audit_parser.add_argument("--batches", type=int, default=4)
    audit_parser.add_argument("--seed", type=int, default=471_001)
    audit_parser.add_argument("--path-seed", type=int, default=481_001)
    audit_parser.add_argument("--device", default="auto")
    audit_parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "train":
        if not 1 <= args.retention_maximum < args.transition_maximum < args.boundary_maximum:
            parser.error("length-band maxima must be strictly increasing")
        if args.boundary_maximum != 20:
            parser.error("strict Addition training boundary must equal 20")
        if any(length > 20 for length in args.eval_lengths):
            parser.error("checkpoint-selection eval lengths must stay within 1--20")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = train(args) if args.action == "train" else audit(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
