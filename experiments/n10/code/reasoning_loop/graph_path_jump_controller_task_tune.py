from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_jump_controller import (
    ONE_MODE,
    JumpMode,
    _aggregate_rows,
    _flatten_positions,
    _validate_modes,
    _write_csv,
    apply_vector_map,
    behavior_metrics,
    collect_jump_pair_batch,
    component_similarity,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    fit_reduced_rank_update_family,
    run_one_loop,
)
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


@dataclass(frozen=True)
class ControllerDataset:
    source: torch.Tensor
    one_target_state: torch.Tensor
    two_target_state: torch.Tensor
    endpoint_node: torch.Tensor
    two_target_node: torch.Tensor


class TrainableSharedAffine(nn.Module):
    """One row-vector affine map shared over all sequence positions."""

    def __init__(self, initial: VectorAffine) -> None:
        super().__init__()
        self.weight = nn.Parameter(initial.weight.detach().clone())
        self.bias = nn.Parameter(initial.bias.detach().clone())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state.float() @ self.weight + self.bias

    def frozen(self) -> VectorAffine:
        return VectorAffine(
            weight=self.weight.detach().clone(),
            bias=self.bias.detach().clone(),
            update_rank=self.weight.shape[0],
            fit_dimension=self.weight.shape[0],
            retained_fit_energy=1.0,
        )


def truncate_residual_affine(
    controller: VectorAffine,
    *,
    rank: int,
) -> VectorAffine:
    """SVD-truncate the learned residual update while retaining its bias."""

    dimension = controller.weight.shape[0]
    if not 0 <= rank <= dimension:
        raise ValueError("rank must lie between zero and map dimension")
    identity = torch.eye(
        dimension,
        device=controller.weight.device,
        dtype=controller.weight.dtype,
    )
    update = controller.weight - identity
    if rank == 0:
        truncated = torch.zeros_like(update)
        retained = 0.0
    else:
        left, singular_values, right = torch.linalg.svd(
            update,
            full_matrices=False,
        )
        truncated = (
            left[:, :rank]
            * singular_values[:rank].unsqueeze(0)
        ) @ right[:rank]
        energy = singular_values.square()
        retained = float(
            energy[:rank].sum()
            / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
        )
    return VectorAffine(
        weight=identity + truncated,
        bias=controller.bias.detach().clone(),
        update_rank=rank,
        fit_dimension=dimension,
        retained_fit_energy=retained,
    )


@torch.no_grad()
def collect_dataset(
    *,
    model,
    cfg,
    two_mode: JumpMode,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
) -> ControllerDataset:
    set_seed(seed)
    parts: dict[str, list[torch.Tensor]] = {
        "source": [],
        "one_target_state": [],
        "two_target_state": [],
        "endpoint_node": [],
        "two_target_node": [],
    }
    for _ in range(batches):
        pair = collect_jump_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=batch_size,
            device=device,
            two_mode=two_mode,
        )
        parts["source"].append(pair.terminal)
        parts["one_target_state"].append(pair.one_target_state)
        parts["two_target_state"].append(pair.two_target_state)
        parts["endpoint_node"].append(
            pair.all_targets[:, cfg.max_depth]
        )
        parts["two_target_node"].append(
            pair.all_targets[:, cfg.max_depth + 2]
        )
    return ControllerDataset(
        source=torch.cat(parts["source"]),
        one_target_state=torch.cat(parts["one_target_state"]),
        two_target_state=torch.cat(parts["two_target_state"]),
        endpoint_node=torch.cat(parts["endpoint_node"]),
        two_target_node=torch.cat(parts["two_target_node"]),
    )


def train_task_controller(
    *,
    model,
    cfg,
    dataset: ControllerDataset,
    initial: VectorAffine,
    state_regularization: float,
    preloop_regularization: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[VectorAffine, list[dict[str, Any]]]:
    if state_regularization < 0:
        raise ValueError("state_regularization must be nonnegative")
    if preloop_regularization < 0:
        raise ValueError("preloop_regularization must be nonnegative")
    controller = TrainableSharedAffine(initial).to(dataset.source.device)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=dataset.source.device)
    generator.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    example_count = dataset.source.shape[0]
    for step in range(1, steps + 1):
        index = torch.randint(
            0,
            example_count,
            (batch_size,),
            generator=generator,
            device=dataset.source.device,
        )
        source = dataset.source[index]
        target_state = dataset.two_target_state[index]
        endpoint_node = dataset.endpoint_node[index]
        target_node = dataset.two_target_node[index]
        mapped = controller(source)
        preloop_logits = logits_from_raw_state(model, mapped)
        output = model.apply_loop(mapped, loop_index=cfg.max_loops)
        logits = logits_from_raw_state(model, output)
        task_loss = F.cross_entropy(logits, target_node)
        preloop_loss = F.cross_entropy(preloop_logits, endpoint_node)
        centered_target = target_state - target_state.mean(
            dim=0,
            keepdim=True,
        )
        state_loss = (
            (mapped - target_state).square().mean()
            / centered_target.square().mean().clamp_min(1e-12)
        )
        loss = (
            task_loss
            + state_regularization * state_loss
            + preloop_regularization * preloop_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            controller.parameters(),
            max_norm=5.0,
        )
        optimizer.step()
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            rows.append(
                {
                    "step": step,
                    "state_regularization": state_regularization,
                    "preloop_regularization": preloop_regularization,
                    "loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "preloop_endpoint_loss": float(preloop_loss.detach()),
                    "state_relative_mse": float(state_loss.detach()),
                    "train_accuracy": float(
                        logits.argmax(dim=-1)
                        .eq(target_node)
                        .float()
                        .mean()
                    ),
                    "preloop_endpoint_accuracy": float(
                        preloop_logits.argmax(dim=-1)
                        .eq(endpoint_node)
                        .float()
                        .mean()
                    ),
                    "gradient_norm": float(gradient_norm),
                }
            )
    return controller.frozen(), rows


def _map_payload(controller: VectorAffine) -> dict[str, Any]:
    return {
        "weight": controller.weight.detach().cpu(),
        "bias": controller.bias.detach().cpu(),
    }


def run_task_tuning(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    eval_batch_size: int,
    eval_batches: int,
    controller_seeds: tuple[int, ...],
    state_regularizations: tuple[float, ...],
    preloop_regularizations: tuple[float, ...],
    steps: int,
    train_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    ridge: float,
    calibration_seed: int,
    eval_seed: int,
    compression_ranks: tuple[int, ...],
    two_initialization: str,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("JUMP_CONTROLLER_CUDA_MEMORY_FRACTION", "0.06")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    two_mode = JumpMode(
        name="two",
        reference_age=1,
        reference_path_before=2,
        programmed_jump=2,
    )
    _validate_modes(cfg, two_mode=two_mode)
    if (
        not compression_ranks
        or min(compression_ranks) < 0
        or max(compression_ranks) > cfg.d_model
    ):
        raise ValueError("compression ranks must lie in [0, d_model]")
    model.requires_grad_(False)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = tuple(range(cfg.seq_len))

    controllers: dict[str, VectorAffine] = {}
    training_rows: list[dict[str, Any]] = []
    for controller_seed in controller_seeds:
        dataset = collect_dataset(
            model=model,
            cfg=cfg,
            two_mode=two_mode,
            batch_size=calibration_batch_size,
            batches=calibration_batches,
            device=device,
            seed=calibration_seed + 1009 * controller_seed,
        )
        source_flat = _flatten_positions(dataset.source, positions)
        one_flat = _flatten_positions(
            dataset.one_target_state,
            positions,
        )
        two_flat = _flatten_positions(
            dataset.two_target_state,
            positions,
        )
        one_map = fit_reduced_rank_update_family(
            source_flat,
            one_flat,
            ridge=ridge,
        ).map_for_rank(8)
        if two_initialization == "hidden_mse":
            two_initial = fit_reduced_rank_update_family(
                source_flat,
                two_flat,
                ridge=ridge,
            ).map_for_rank(cfg.d_model)
        elif two_initialization == "identity":
            two_initial = VectorAffine(
                weight=torch.eye(
                    cfg.d_model,
                    device=device,
                    dtype=source_flat.dtype,
                ),
                bias=torch.zeros(
                    cfg.d_model,
                    device=device,
                    dtype=source_flat.dtype,
                ),
                update_rank=0,
                fit_dimension=cfg.d_model,
                retained_fit_energy=0.0,
            )
        else:
            raise ValueError(f"unknown two-hop initialization: {two_initialization}")
        controllers[f"seed{controller_seed}_J_one_rank8"] = one_map
        if two_initialization == "hidden_mse":
            controllers[f"seed{controller_seed}_J_two_hidden_mse"] = two_initial
        for regularization in state_regularizations:
            for preloop_regularization in preloop_regularizations:
                tuned, rows = train_task_controller(
                    model=model,
                    cfg=cfg,
                    dataset=dataset,
                    initial=two_initial,
                    state_regularization=regularization,
                    preloop_regularization=preloop_regularization,
                    steps=steps,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=17_003 + controller_seed,
                )
                name = (
                    f"seed{controller_seed}_J_two_task_"
                    f"lambda{regularization:g}"
                )
                if preloop_regularization:
                    name += f"_pre{preloop_regularization:g}"
                controllers[name] = tuned
                for rank in compression_ranks:
                    controllers[f"{name}_rank{rank}"] = (
                        truncate_residual_affine(tuned, rank=rank)
                    )
                controllers[f"{name}_no_bias"] = VectorAffine(
                    weight=tuned.weight,
                    bias=torch.zeros_like(tuned.bias),
                    update_rank=cfg.d_model,
                    fit_dimension=cfg.d_model,
                    retained_fit_energy=1.0,
                )
                for row in rows:
                    row["controller_seed"] = controller_seed
                    row["condition"] = name
                    training_rows.append(row)

    torch.save(
        {
            name: _map_payload(controller)
            for name, controller in controllers.items()
        },
        out_dir / "task_tuned_controllers.pt",
    )

    set_seed(eval_seed)
    evaluation_rows: list[dict[str, Any]] = []
    for batch_index in range(eval_batches):
        pair = collect_jump_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=eval_batch_size,
            device=device,
            two_mode=two_mode,
        )
        one_oracle = run_one_loop(
            model,
            pair.one_target_state,
            loop_index=cfg.max_loops,
        )
        two_oracle = run_one_loop(
            model,
            pair.two_target_state,
            loop_index=cfg.max_loops,
        )
        states: list[tuple[str, torch.Tensor]] = [
            ("identity", pair.terminal),
            ("exact_one", pair.one_target_state),
            ("exact_two", pair.two_target_state),
        ]
        states.extend(
            (
                name,
                apply_vector_map(
                    pair.terminal,
                    positions=positions,
                    controller=controller,
                ),
            )
            for name, controller in controllers.items()
        )
        trained_names = {
            name
            for name in controllers
            if "_J_two_task_" in name
            and "_rank" not in name
            and not name.endswith("_no_bias")
        }
        states.extend(
            (f"{name}_shuffled", torch.roll(state, shifts=1, dims=0))
            for name, state in list(states)
            if name in trained_names
        )
        for name, state in states:
            preloop_metrics = behavior_metrics(
                logits_from_raw_state(model, state),
                pair.all_targets,
                endpoint_position=cfg.max_depth,
            )
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops,
            )
            row: dict[str, Any] = {
                "batch": batch_index,
                "condition": name,
            }
            row.update(
                {f"pre_{key}": value for key, value in preloop_metrics.items()}
            )
            row.update(
                behavior_metrics(
                    step.logits,
                    pair.all_targets,
                    endpoint_position=cfg.max_depth,
                )
            )
            row.update(
                component_similarity(
                    step,
                    one_oracle,
                    cfg=cfg,
                    prefix="to_one_oracle",
                )
            )
            row.update(
                component_similarity(
                    step,
                    two_oracle,
                    cfg=cfg,
                    prefix="to_two_oracle",
                )
            )
            evaluation_rows.append(row)

    condition_summary = _aggregate_rows(
        evaluation_rows,
        key="condition",
    )
    lookup = {str(row["condition"]): row for row in condition_summary}
    exact_two_accuracy = float(lookup["exact_two"]["two_accuracy"])
    replicate_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    for controller_seed in controller_seeds:
        for regularization in state_regularizations:
            for preloop_regularization in preloop_regularizations:
                name = (
                    f"seed{controller_seed}_J_two_task_"
                    f"lambda{regularization:g}"
                )
                if preloop_regularization:
                    name += f"_pre{preloop_regularization:g}"
                row = lookup[name]
                shuffled_row = lookup[f"{name}_shuffled"]
                replicate_rows.append(
                    {
                        "controller_seed": controller_seed,
                        "state_regularization": regularization,
                        "preloop_regularization": preloop_regularization,
                        "two_accuracy": row["two_accuracy"],
                        "one_accuracy": row["one_accuracy"],
                        "distinct_two_accuracy": row[
                            "distinct_two_accuracy"
                        ],
                        "distinct_one_accuracy": row[
                            "distinct_one_accuracy"
                        ],
                        "pre_endpoint_accuracy": row["pre_endpoint_accuracy"],
                        "pre_distinct_two_accuracy": row[
                            "pre_distinct_two_accuracy"
                        ],
                        "shuffled_distinct_two_accuracy": shuffled_row[
                            "distinct_two_accuracy"
                        ],
                        "two_oracle_fraction": (
                            float(row["two_accuracy"])
                            / max(exact_two_accuracy, 1e-12)
                        ),
                        "selects_two": bool(
                            float(row["distinct_two_accuracy"])
                            > float(row["distinct_one_accuracy"])
                        ),
                        "to_two_mlp_answer_cosine": row[
                            "to_two_oracle_mlp_answer_cosine"
                        ],
                        "to_one_mlp_answer_cosine": row[
                            "to_one_oracle_mlp_answer_cosine"
                        ],
                    }
                )
                for rank in compression_ranks:
                    rank_name = f"{name}_rank{rank}"
                    rank_row = lookup[rank_name]
                    rank_rows.append(
                        {
                            "controller_seed": controller_seed,
                            "state_regularization": regularization,
                            "preloop_regularization": (
                                preloop_regularization
                            ),
                            "rank": rank,
                            "parameter_count": (
                                cfg.d_model
                                if rank == 0
                                else 2 * cfg.d_model * rank + cfg.d_model
                            ),
                            "two_accuracy": rank_row["two_accuracy"],
                            "one_accuracy": rank_row["one_accuracy"],
                            "distinct_two_accuracy": rank_row[
                                "distinct_two_accuracy"
                            ],
                            "distinct_one_accuracy": rank_row[
                                "distinct_one_accuracy"
                            ],
                            "retained_update_energy": controllers[
                                rank_name
                            ].retained_fit_energy,
                        }
                    )
                no_bias_row = lookup[f"{name}_no_bias"]
                bias_rows.append(
                    {
                        "controller_seed": controller_seed,
                        "state_regularization": regularization,
                        "preloop_regularization": (
                            preloop_regularization
                        ),
                        "with_bias_two_accuracy": row["two_accuracy"],
                        "without_bias_two_accuracy": no_bias_row[
                            "two_accuracy"
                        ],
                        "with_bias_distinct_two_accuracy": row[
                            "distinct_two_accuracy"
                        ],
                        "without_bias_distinct_two_accuracy": no_bias_row[
                            "distinct_two_accuracy"
                        ],
                    }
                )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "config": asdict(cfg),
        "loss_placement": "final_only",
        "trained_loop_count": cfg.max_loops,
        "evaluated_receiver_age": cfg.max_loops,
        "evaluated_extra_loops": 1,
        "shared_physical_blocks": cfg.n_layers,
        "effective_depth_at_training_horizon": cfg.n_layers * cfg.max_loops,
        "two_mode": asdict(two_mode),
        "controller_training_objective": (
            "frozen-backbone two-hop cross entropy plus optional matched-state "
            "relative MSE; pure CE exactly when both regularization weights are zero"
        ),
        "two_initialization": two_initialization,
        "calibration_examples_per_controller_seed": (
            calibration_batch_size * calibration_batches
        ),
        "evaluation_examples": eval_batch_size * eval_batches,
        "controller_seeds": list(controller_seeds),
        "state_regularizations": list(state_regularizations),
        "preloop_regularizations": list(preloop_regularizations),
        "steps": steps,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "ridge": ridge,
        "compression_ranks": list(compression_ranks),
        "exact_one_accuracy": lookup["exact_one"]["one_accuracy"],
        "exact_two_accuracy": exact_two_accuracy,
        "replicate_rows": replicate_rows,
        "rank_rows": rank_rows,
        "bias_rows": bias_rows,
        "condition_summary": condition_summary,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    _write_csv(out_dir / "training_rows.csv", training_rows)
    _write_csv(out_dir / "evaluation_per_batch.csv", evaluation_rows)
    _write_csv(out_dir / "condition_summary.csv", condition_summary)
    _write_csv(out_dir / "replicate_summary.csv", replicate_rows)
    _write_csv(out_dir / "rank_summary.csv", rank_rows)
    _write_csv(out_dir / "bias_summary.csv", bias_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task-tune a shared affine two-hop controller on a frozen D8L6 "
            "backbone, with matched-state regularization controls."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--controller-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--state-regularizations",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.1],
    )
    parser.add_argument(
        "--preloop-regularizations",
        type=float,
        nargs="+",
        default=[0.0],
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--two-initialization",
        choices=("hidden_mse", "identity"),
        default="hidden_mse",
        help=(
            "Initialization for J_two. Use identity with zero state/preloop "
            "regularization for training driven only by post-executor CE."
        ),
    )
    parser.add_argument("--calibration-seed", type=int, default=7301)
    parser.add_argument("--eval-seed", type=int, default=7401)
    parser.add_argument(
        "--compression-ranks",
        type=int,
        nargs="+",
        default=[0, 8, 16, 32, 64, 128, 256],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_task_tuning(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        controller_seeds=tuple(args.controller_seeds),
        state_regularizations=tuple(args.state_regularizations),
        preloop_regularizations=tuple(args.preloop_regularizations),
        steps=args.steps,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        eval_seed=args.eval_seed,
        compression_ranks=tuple(sorted(set(args.compression_ranks))),
        two_initialization=args.two_initialization,
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
