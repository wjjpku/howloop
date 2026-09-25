from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_graph_blind_mlp_j import (
    GraphBlindResidualMLP,
)
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_unit_j import (
    _training_case,
    exact_interfaces,
    load_unit_j_map,
    unit_j_dose_schedule,
)


@dataclass(frozen=True)
class TaskMLPStage:
    name: str
    rounds: int
    batch_size: int
    batches_per_round: int
    horizons: tuple[int, ...]
    learning_rate: float
    data_seed: int

    @property
    def graphs(self) -> int:
        return self.rounds * self.batch_size * self.batches_per_round


# This is the completed affine Unit-J task-training recipe.  The nonlinear
# experiment starts from the same post-DAgger affine map and changes only the
# controller class used during these four task-aware stages.
GOOD_TASK_STAGES = (
    TaskMLPStage(
        name="task_h24",
        rounds=32,
        batch_size=64,
        batches_per_round=8,
        horizons=(4, 8, 16, 24),
        learning_rate=1e-5,
        data_seed=175003,
    ),
    TaskMLPStage(
        name="task_h32",
        rounds=16,
        batch_size=32,
        batches_per_round=9,
        horizons=(16, 24, 32),
        learning_rate=3e-6,
        data_seed=177003,
    ),
    TaskMLPStage(
        name="task_h48",
        rounds=16,
        batch_size=32,
        batches_per_round=9,
        horizons=(24, 32, 48),
        learning_rate=3e-6,
        data_seed=178003,
    ),
    TaskMLPStage(
        name="task_h64",
        rounds=16,
        batch_size=32,
        batches_per_round=9,
        horizons=(32, 48, 64),
        learning_rate=3e-6,
        data_seed=179003,
    ),
)


def _weighted_losses(
    losses: Sequence[tuple[torch.Tensor, int, int]],
) -> torch.Tensor:
    if not losses:
        raise ValueError("cannot aggregate an empty loss sequence")
    return torch.stack([loss for loss, _, _ in losses]).mean()


def _controlled_loop(
    *,
    loop_runner,
    model,
    state: torch.Tensor,
    loop_index: int,
    positions: tuple[int, ...],
    operator,
    placement: str,
    captures: list[torch.Tensor] | None = None,
):
    """Apply one controller dose at an explicitly named recurrent site.

    ``pre_block2`` is the historical Unit-J site: after Block1's FFN and
    before Block2's attention. ``loop_boundary`` acts on the recurrent state
    before Block1, which is exactly the state produced after Block2's FFN in
    the preceding loop.  The boundary intervention therefore cannot directly
    edit the preceding readout; it must prepare the next complete loop.
    """

    if placement == "pre_block2":
        def transform(value: torch.Tensor) -> torch.Tensor:
            live = operator(value)
            if captures is not None:
                captures.append(live)
            return live

        return loop_runner(
            model,
            state,
            loop_index=loop_index,
            block2_position_transform=(positions, transform),
        )
    if placement == "loop_boundary":
        index = list(positions)
        controlled_state = state.clone()
        live = operator(controlled_state[:, index])
        if captures is not None:
            captures.append(live)
        controlled_state[:, index] = live.to(dtype=controlled_state.dtype)
        return loop_runner(
            model,
            controlled_state,
            loop_index=loop_index,
        )
    raise ValueError(f"unsupported controller placement: {placement}")


def _weighted_interface_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    answer_relative_position: int,
) -> torch.Tensor:
    """Relative MSE used by the completed affine Unit-J recipe."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("interface tensors must share [batch, position, feature]")
    weights = torch.ones(
        prediction.shape[1],
        device=prediction.device,
        dtype=torch.float32,
    )
    weights[answer_relative_position] = 1.0
    weights = weights.view(1, -1, 1)
    denominator = prediction.shape[0] * weights.sum() * prediction.shape[2]
    target_float = target.float()
    target_mean = (target_float * weights).sum(dim=(0, 1), keepdim=True) / (
        prediction.shape[0] * weights.sum()
    )
    target_scale = (
        (target_float - target_mean).square() * weights
    ).sum() / denominator
    mse = ((prediction.float() - target_float).square() * weights).sum() / (
        denominator
    )
    return mse / target_scale.clamp_min(1e-6)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _train_stage(
    *,
    module: GraphBlindResidualMLP,
    stage: TaskMLPStage,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    device: torch.device,
    state_loss_weight: float,
    grad_clip: float,
    placement: str,
    optimizer_parameter_groups: list[dict[str, Any]] | None = None,
    closed_loop_age: int = 8,
    learning_rate_schedule: Callable[[int, int], float] | None = None,
) -> list[dict[str, Any]]:
    """Train one stage with the exact completed affine Unit-J objective."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    module.train()
    optimizer = torch.optim.AdamW(
        (
            optimizer_parameter_groups
            if optimizer_parameter_groups is not None
            else module.parameters()
        ),
        lr=stage.learning_rate,
        weight_decay=0.0,
    )
    group_learning_rate_multipliers = [
        float(group["lr"]) / stage.learning_rate
        for group in optimizer.param_groups
    ]
    total_optimizer_updates = stage.rounds * stage.batches_per_round
    differentiable_loop = run_one_loop.__wrapped__
    jump = phase_positions[3] - phase_positions[2]
    answer_relative_position = positions.index(cfg.seq_len - 1)
    rows: list[dict[str, Any]] = []
    for round_index in range(1, stage.rounds + 1):
        round_seed = stage.data_seed + 1000 * round_index
        set_seed(round_seed)
        totals: dict[str, float] = defaultdict(float)
        started = time.monotonic()
        for batch_index in range(stage.batches_per_round):
            optimizer_update = (
                (round_index - 1) * stage.batches_per_round + batch_index + 1
            )
            scheduled_learning_rate = (
                learning_rate_schedule(
                    optimizer_update,
                    total_optimizer_updates,
                )
                if learning_rate_schedule is not None
                else stage.learning_rate
            )
            if scheduled_learning_rate <= 0:
                raise ValueError("scheduled learning rate must be positive")
            for group, multiplier in zip(
                optimizer.param_groups,
                group_learning_rate_multipliers,
                strict=True,
            ):
                group["lr"] = scheduled_learning_rate * multiplier
            totals["learning_rate"] += scheduled_learning_rate
            if batch_index == 0:
                totals["learning_rate_min"] = scheduled_learning_rate
                totals["learning_rate_max"] = scheduled_learning_rate
            else:
                totals["learning_rate_min"] = min(
                    totals["learning_rate_min"], scheduled_learning_rate
                )
                totals["learning_rate_max"] = max(
                    totals["learning_rate_max"], scheduled_learning_rate
                )
            policy, start_age, horizon = _training_case(
                batch_index=batch_index,
                rollout_horizons=stage.horizons,
                policies=("unit_every",),
                start_ages=(closed_loop_age,),
                seed=round_seed,
            )
            doses, ages = unit_j_dose_schedule(
                name=policy,
                horizon=horizon,
                start_age=start_age,
                seed=round_seed * 1009 + batch_index,
            )
            with torch.no_grad():
                _, path_targets, successors, _ = fixed_depth_batch(
                    cfg,
                    stage.batch_size,
                    device,
                    path_positions=cfg.max_depth,
                )
                endpoint = path_targets[:, cfg.max_depth - 1]
                state = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=endpoint,
                    age=start_age,
                    phase_position=phase_positions[start_age],
                )

            optimizer.zero_grad(set_to_none=True)
            task_losses: list[tuple[torch.Tensor, int, int]] = []
            state_losses: list[tuple[torch.Tensor, int, int]] = []
            for cycle, (dose, age) in enumerate(
                zip(doses, ages, strict=True),
                start=1,
            ):
                if dose != 1 or age != closed_loop_age:
                    raise RuntimeError(
                        "the good recipe requires one J application from "
                        f"H{closed_loop_age}"
                    )
                current = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * (cycle - 1),
                )
                target = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * cycle,
                )
                captures: list[torch.Tensor] = []

                step = _controlled_loop(
                    loop_runner=differentiable_loop,
                    model=model,
                    state=state,
                    loop_index=cfg.max_loops + cycle - 1,
                    positions=positions,
                    operator=module,
                    placement=placement,
                    captures=captures,
                )
                task_loss = F.cross_entropy(step.logits.float(), target)
                task_losses.append((task_loss, cycle, horizon))
                totals["supervised_steps"] += 1
                totals["examples"] += stage.batch_size
                totals["correct"] += int(
                    step.logits.argmax(dim=-1).eq(target).sum()
                )
                totals["task_loss"] += float(task_loss.detach())

                if state_loss_weight > 0.0:
                    with torch.no_grad():
                        if placement == "pre_block2":
                            oracle = exact_interfaces(
                                model=model,
                                cfg=cfg,
                                phase_positions=phase_positions,
                                positions=positions,
                                successors=successors,
                                current=current,
                                ages=(7,),
                                loop_index=cfg.max_loops + cycle - 1,
                            )[7]
                        else:
                            oracle = _aligned_state_at_age(
                                model=model,
                                cfg=cfg,
                                successors=successors,
                                current=current,
                                age=7,
                                phase_position=phase_positions[7],
                            )[:, list(positions)]
                    state_loss = _weighted_interface_loss(
                        captures[0],
                        oracle,
                        answer_relative_position=answer_relative_position,
                    )
                    state_losses.append((state_loss, cycle, horizon))
                    totals["state_loss"] += float(state_loss.detach())
                totals["J_applications"] += 1
                state = step.state

            loss = _weighted_losses(task_losses)
            if state_loss_weight > 0.0:
                loss = loss + state_loss_weight * _weighted_losses(
                    state_losses
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(),
                max_norm=grad_clip,
            )
            optimizer.step()
            totals["optimizer_steps"] += 1
            totals["grad_norm"] += float(grad_norm)

        rows.append(
            {
                "stage": stage.name,
                "round": round_index,
                "round_seed": round_seed,
                "graphs": stage.batch_size * stage.batches_per_round,
                "horizons": "/".join(map(str, stage.horizons)),
                "optimizer_steps": int(totals["optimizer_steps"]),
                "J_applications": int(totals["J_applications"]),
                "supervised_composition_steps": int(
                    totals["supervised_steps"]
                ),
                "composition_accuracy": (
                    totals["correct"] / totals["examples"]
                ),
                "task_ce": (
                    totals["task_loss"] / totals["supervised_steps"]
                ),
                "H7_relative_mse": (
                    totals["state_loss"] / totals["J_applications"]
                    if state_loss_weight > 0.0
                    else None
                ),
                "mean_preclip_gradient_norm": (
                    totals["grad_norm"] / totals["optimizer_steps"]
                ),
                "learning_rate": (
                    totals["learning_rate"] / totals["optimizer_steps"]
                ),
                "learning_rate_min": totals["learning_rate_min"],
                "learning_rate_max": totals["learning_rate_max"],
                "state_loss_weight": state_loss_weight,
                "placement": placement,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    module.eval()
    return rows


@torch.no_grad()
def evaluate_random_unit_every(
    *,
    maps: dict[str, Any],
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    seed: int,
    placement: str,
) -> list[dict[str, Any]]:
    set_seed(seed)
    jump = phase_positions[3] - phase_positions[2]
    correct = {
        label: [0 for _ in range(continuation_loops)] for label in maps
    }
    total = 0
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        states = {label: initial.clone() for label in maps}
        for cycle in range(1, continuation_loops + 1):
            target = advance_nodes(successors, endpoint, steps=jump * cycle)
            for label, operator in maps.items():
                step = _controlled_loop(
                    loop_runner=run_one_loop,
                    model=model,
                    state=states[label],
                    loop_index=cfg.max_loops + cycle - 1,
                    positions=positions,
                    operator=operator,
                    placement=placement,
                )
                correct[label][cycle - 1] += int(
                    step.logits.argmax(dim=-1).eq(target).sum()
                )
                states[label] = step.state
        total += batch_size
    return [
        {
            "variant": label,
            "cycle": cycle,
            "accuracy": values[cycle - 1] / total,
            "examples": total,
        }
        for label, values in correct.items()
        for cycle in range(1, continuation_loops + 1)
    ]


def load_task_mlp_modules(
    artifact: Path,
    *,
    device: torch.device,
) -> tuple[str, tuple[int, ...], dict[str, GraphBlindResidualMLP], dict]:
    payload = torch.load(artifact, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_task_mlp_j":
        raise ValueError("unexpected task-aware MLP-J artifact kind")
    modules: dict[str, GraphBlindResidualMLP] = {}
    for label, item in payload["modules"].items():
        initial = VectorAffine(
            weight=item["initial_affine_weight"].to(
                device=device,
                dtype=torch.float32,
            ),
            bias=item["initial_affine_bias"].to(
                device=device,
                dtype=torch.float32,
            ),
            update_rank=int(item["dimension"]),
            fit_dimension=int(item["dimension"]),
            retained_fit_energy=1.0,
        )
        module = GraphBlindResidualMLP(
            initial=initial,
            hidden_width=int(item["hidden_width"]),
        ).to(device)
        module.load_state_dict(
            {key: value.to(device) for key, value in item["state_dict"].items()}
        )
        modules[label] = module.frozen()
    return (
        str(payload["checkpoint"]),
        tuple(int(value) for value in payload["positions"]),
        modules,
        payload,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage_round_limit is not None and args.stage_round_limit <= 0:
        raise ValueError("stage_round_limit must be positive")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if (
        cfg.node_count != 8
        or cfg.max_depth != 8
        or cfg.max_loops != 8
        or cfg.n_layers != 2
        or cfg.d_model != 256
    ):
        raise ValueError("experiment requires the frozen D8L8 N8 d256 model")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    initial_affine, initial_checkpoint = load_unit_j_map(
        args.post_dagger_affine_artifact,
        label=args.post_dagger_affine_label,
        device=device,
    )
    if initial_checkpoint != str(args.checkpoint):
        raise ValueError("post-DAgger affine J belongs to another checkpoint")
    reference_affine, reference_checkpoint = load_unit_j_map(
        args.reference_affine_artifact,
        label=args.reference_affine_label,
        device=device,
    )
    if reference_checkpoint != str(args.checkpoint):
        raise ValueError("reference affine J belongs to another checkpoint")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modules: dict[str, GraphBlindResidualMLP] = {}
    module_payloads: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    initial_probe = torch.randn(
        4,
        3,
        cfg.d_model,
        device=device,
        dtype=torch.float32,
    )
    stages = tuple(
        TaskMLPStage(
            name=stage.name,
            rounds=(
                min(stage.rounds, args.stage_round_limit)
                if args.stage_round_limit is not None
                else stage.rounds
            ),
            batch_size=stage.batch_size,
            batches_per_round=stage.batches_per_round,
            horizons=stage.horizons,
            learning_rate=stage.learning_rate,
            data_seed=stage.data_seed,
        )
        for stage in GOOD_TASK_STAGES
    )
    for hidden_width in args.hidden_widths:
        for initialization_seed in args.initialization_seeds:
            set_seed(initialization_seed)
            label = f"task_mlp_w{hidden_width}_seed{initialization_seed}"
            module = GraphBlindResidualMLP(
                initial=initial_affine,
                hidden_width=hidden_width,
            ).to(device)
            with torch.no_grad():
                initialization_error = float(
                    (
                        module(initial_probe)
                        - initial_affine(initial_probe)
                    )
                    .abs()
                    .max()
                    .item()
                )
            if initialization_error != 0.0:
                raise RuntimeError("MLP does not start at the affine baseline")
            variant_rows: list[dict[str, Any]] = []
            snapshots: dict[str, dict[str, torch.Tensor]] = {}
            for stage in stages:
                rows = _train_stage(
                    module=module,
                    stage=stage,
                    model=model,
                    cfg=cfg,
                    phase_positions=phase_positions,
                    positions=positions,
                    device=device,
                    state_loss_weight=args.state_loss_weight,
                    grad_clip=args.grad_clip,
                    placement=args.placement,
                )
                for row in rows:
                    row.update(
                        {
                            "variant": label,
                            "hidden_width": hidden_width,
                            "initialization_seed": initialization_seed,
                        }
                    )
                variant_rows.extend(rows)
                snapshots[stage.name] = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            modules[label] = module.frozen()
            training_rows.extend(variant_rows)
            module_payloads[label] = {
                "hidden_width": hidden_width,
                "initialization_seed": initialization_seed,
                "dimension": cfg.d_model,
                "parameter_count": module.parameter_count,
                "initialization_max_abs_error": initialization_error,
                "initial_affine_weight": initial_affine.weight.detach().cpu(),
                "initial_affine_bias": initial_affine.bias.detach().cpu(),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in module.state_dict().items()
                },
                "stage_snapshots": snapshots,
            }
            artifact_tmp = out_dir / "task_mlp_j.pt.tmp"
            artifact = out_dir / "task_mlp_j.pt"
            torch.save(
                {
                    "kind": "graph_path_telomere_task_mlp_j",
                    "checkpoint": str(args.checkpoint),
                    "positions": positions,
                    "post_dagger_affine_artifact": str(
                        args.post_dagger_affine_artifact
                    ),
                    "post_dagger_affine_label": args.post_dagger_affine_label,
                    "reference_affine_artifact": str(
                        args.reference_affine_artifact
                    ),
                    "reference_affine_label": args.reference_affine_label,
                    "state_loss_weight": args.state_loss_weight,
                    "placement": args.placement,
                    "stages": [asdict(stage) for stage in stages],
                    "modules": module_payloads,
                },
                artifact_tmp,
            )
            os.replace(artifact_tmp, artifact)
            _write_csv(out_dir / "training_rounds.csv", training_rows)

    evaluation_maps: dict[str, Any] = {"reference_affine": reference_affine}
    evaluation_maps.update(modules)
    random_rows = evaluate_random_unit_every(
        maps=evaluation_maps,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.evaluation_batch_size,
        batches=args.evaluation_batches,
        continuation_loops=args.evaluation_loops,
        seed=args.evaluation_seed,
        placement=args.placement,
    )
    _write_csv(out_dir / "random_graph_closed_loop.csv", random_rows)

    curves: dict[str, dict[str, Any]] = {}
    for label in evaluation_maps:
        curve = [
            float(row["accuracy"])
            for row in random_rows
            if row["variant"] == label
        ]
        curves[label] = {
            "accuracy_by_cycle": curve,
            "auc_1_24": sum(curve[:24]) / min(24, len(curve)),
            "auc_25_48": (
                sum(curve[24:48]) / min(24, len(curve) - 24)
                if len(curve) > 24
                else None
            ),
            "auc_49_64": (
                sum(curve[48:64]) / min(16, len(curve) - 48)
                if len(curve) > 48
                else None
            ),
        }
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            "frozen backbone final-only CE at loop 8; task-aware MLP-J "
            "uses successor CE only at every controlled loop"
            if args.state_loss_weight == 0.0
            else (
                "frozen backbone final-only CE at loop 8; task-aware MLP-J "
                f"uses successor CE plus {args.state_loss_weight:g} "
                "exact-H7 state loss at every controlled loop"
            )
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "controller": (
            "position-shared residual two-layer MLP: affine(z) + "
            "up(GELU(down(z))); zero up initialization"
        ),
        "post_dagger_affine_artifact": str(args.post_dagger_affine_artifact),
        "post_dagger_affine_label": args.post_dagger_affine_label,
        "reference_affine_artifact": str(args.reference_affine_artifact),
        "reference_affine_label": args.reference_affine_label,
        "hidden_widths": list(args.hidden_widths),
        "initialization_seeds": list(args.initialization_seeds),
        "state_loss_weight": args.state_loss_weight,
        "controller_placement": args.placement,
        "controller_placement_definition": (
            "after Block1 FFN and before Block2 attention"
            if args.placement == "pre_block2"
            else (
                "on the recurrent boundary state produced after Block2 FFN; "
                "applied before the next loop's Block1, with readout only "
                "after that complete loop"
            )
        ),
        "stages": [asdict(stage) for stage in stages],
        "task_training_graph_draws": sum(stage.graphs for stage in stages),
        "full_recipe_graph_draws_including_ridge_and_dagger": (
            1024 + 8 * 1024 + sum(stage.graphs for stage in stages)
        ),
        "variants": {
            label: {
                "hidden_width": module.hidden_width,
                "parameter_count": module.parameter_count,
                "initialization_seed": int(label.rsplit("seed", 1)[1]),
            }
            for label, module in modules.items()
        },
        "random_graph_evaluation": {
            "examples": args.evaluation_batch_size * args.evaluation_batches,
            "loops": args.evaluation_loops,
            "seed": args.evaluation_seed,
            "curves": curves,
        },
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "prelaunch_used_mib": args.prelaunch_used_mib,
            "prelaunch_free_mib": args.prelaunch_free_mib,
            "declared_peak_gib": args.declared_peak_gib,
            "reserve_gib": args.reserve_gib,
            "shared": args.shared_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "files": {
            "artifact": "task_mlp_j.pt",
            "training": "training_rounds.csv",
            "random_evaluation": "random_graph_closed_loop.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the successful task-aware affine Unit-J with a nested "
            "two-layer residual MLP while keeping its training recipe fixed."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument(
        "--post-dagger-affine-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--post-dagger-affine-label", default="reg")
    parser.add_argument("--reference-affine-artifact", type=Path, required=True)
    parser.add_argument("--reference-affine-label", default="task")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-widths", type=int, nargs="+", default=(64,))
    parser.add_argument(
        "--initialization-seeds",
        type=int,
        nargs="+",
        default=(211001, 311001, 411001),
    )
    parser.add_argument("--state-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--placement",
        choices=("pre_block2", "loop_boundary"),
        default="pre_block2",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--stage-round-limit",
        type=int,
        help="cap each stage's rounds for a smoke test",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--evaluation-loops", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int, default=212004)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.04)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=3.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
