from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_graph_blind_j import (
    calibrate,
    collect_onpolicy_pairs,
    curve_windows,
    hidden_loss,
    sample_graph_current_batch,
    stratified_permutation_split,
    write_csv,
)
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    audit_partition,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)


class GraphBlindResidualMLP(torch.nn.Module):
    """A nested nonlinear extension of the full affine age map.

    The map starts exactly at the calibrated affine baseline and learns a
    positionwise nonlinear correction shared by every token position:

        J(z) = z @ W + b + up(gelu(down(z))).

    Zero-initializing ``up`` makes the initial function exactly equal to the
    affine map, so any gain can be attributed to added nonlinear capacity
    rather than a worse initialization.
    """

    def __init__(self, *, initial: VectorAffine, hidden_width: int) -> None:
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        dimension = initial.weight.shape[0]
        self.dimension = dimension
        self.hidden_width = hidden_width
        self.affine_weight = torch.nn.Parameter(initial.weight.detach().clone())
        self.affine_bias = torch.nn.Parameter(initial.bias.detach().clone())
        self.down = torch.nn.Linear(dimension, hidden_width)
        self.up = torch.nn.Linear(hidden_width, dimension)
        torch.nn.init.xavier_uniform_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        affine = live @ self.affine_weight + self.affine_bias
        correction = self.up(F.gelu(self.down(live)))
        return affine + correction

    def nonlinear_correction(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        return self.up(F.gelu(self.down(live)))

    def frozen(self) -> GraphBlindResidualMLP:
        result = copy.deepcopy(self).eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def train_mlp_variant(
    *,
    label: str,
    module: GraphBlindResidualMLP,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    train_pool: torch.Tensor,
    device: torch.device,
    rounds: int,
    batch_size: int,
    horizons: Sequence[int],
    epochs_per_round: int,
    mini_batch_size: int,
    learning_rate: float,
    target_scale: torch.Tensor,
    seed: int,
) -> tuple[GraphBlindResidualMLP, list[dict[str, Any]]]:
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        round_seed = seed + 1000 * round_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(round_seed)
        successors, current = sample_graph_current_batch(
            train_pool,
            batch_size=batch_size,
            node_count=cfg.node_count,
            generator=generator,
            device=device,
        )
        horizon = int(horizons[(round_index - 1) % len(horizons)])
        source, target = collect_onpolicy_pairs(
            module=module,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=current,
            horizon=horizon,
        )
        optimizer_generator = torch.Generator(device=device)
        optimizer_generator.manual_seed(round_seed + 17)
        loss_sum = 0.0
        gradient_sum = 0.0
        optimizer_steps = 0
        for _ in range(epochs_per_round):
            order = torch.randperm(
                source.shape[0],
                generator=optimizer_generator,
                device=device,
            )
            for start in range(0, source.shape[0], mini_batch_size):
                index = order[start : start + mini_batch_size]
                prediction = module(source[index])
                loss = hidden_loss(
                    prediction,
                    target[index],
                    target_scale=target_scale,
                    position_feature_weights=None,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                loss_sum += float(loss.detach())
                gradient_sum += float(gradient)
                optimizer_steps += 1
        with torch.no_grad():
            prediction = module(source)
            final_loss = hidden_loss(
                prediction,
                target,
                target_scale=target_scale,
                position_feature_weights=None,
            )
            correction_rms = float(
                module.nonlinear_correction(source).square().mean().sqrt()
            )
            total_update_rms = float(
                (prediction.float() - source.float()).square().mean().sqrt()
            )
        rows.append(
            {
                "variant": label,
                "round": round_index,
                "graph_seed": round_seed,
                "graphs": batch_size,
                "horizon": horizon,
                "state_pairs": int(source.shape[0]),
                "optimizer_steps": optimizer_steps,
                "mean_training_loss": loss_sum / optimizer_steps,
                "final_onpolicy_loss": float(final_loss),
                "mean_preclip_gradient_norm": gradient_sum / optimizer_steps,
                "nonlinear_correction_rms": correction_rms,
                "total_update_rms": total_update_rms,
                "learning_rate": learning_rate,
            }
        )
    return module.frozen(), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=190001)
    parser.add_argument("--calibration-seed", type=int, default=190002)
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=(190003, 290003, 390003),
    )
    parser.add_argument("--evaluation-seed", type=int, default=190004)
    parser.add_argument("--hidden-widths", nargs="+", type=int, default=(64, 256))
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=(8, 16, 24, 32),
    )
    parser.add_argument("--epochs-per-round", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--variance-floor", type=float, default=0.1)
    parser.add_argument("--evaluation-train-graphs", type=int, default=64)
    parser.add_argument("--evaluation-heldout-graphs", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--continuation-loops", type=int, default=24)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=4.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            0.05,
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
    ):
        raise ValueError("experiment requires the frozen D8L8 N8 two-block model")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    train_permutations, heldout_permutations, split_distribution = (
        stratified_permutation_split(
            node_count=cfg.node_count,
            seed=args.split_seed,
        )
    )
    if set(train_permutations) & set(heldout_permutations):
        raise RuntimeError("train and held-out permutation pools overlap")
    train_pool = torch.tensor(train_permutations, dtype=torch.long)
    calibration = calibrate(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        train_pool=train_pool,
        device=device,
        batch_size=args.calibration_batch_size,
        batches=args.calibration_batches,
        seed=args.calibration_seed,
        ridge=args.ridge,
        variance_floor=args.variance_floor,
    )

    sampled_train, sampled_heldout, matched_distribution = stratified_samples(
        train_permutations,
        heldout_permutations,
        count=max(
            args.evaluation_train_graphs,
            args.evaluation_heldout_graphs,
        ),
        seed=args.evaluation_seed,
    )
    trained_modules: dict[str, GraphBlindResidualMLP] = {}
    training_rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for hidden_width in args.hidden_widths:
        for training_seed in args.training_seeds:
            set_seed(training_seed)
            label = f"mlp_w{hidden_width}_seed{training_seed}"
            module = GraphBlindResidualMLP(
                initial=calibration.initial_map,
                hidden_width=hidden_width,
            ).to(device)
            trained, rows = train_mlp_variant(
                label=label,
                module=module,
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                train_pool=train_pool,
                device=device,
                rounds=args.rounds,
                batch_size=args.training_batch_size,
                horizons=args.horizons,
                epochs_per_round=args.epochs_per_round,
                mini_batch_size=args.mini_batch_size,
                learning_rate=args.learning_rate,
                target_scale=calibration.target_scale,
                seed=training_seed,
            )
            trained_modules[label] = trained
            training_rows.extend(rows)
            for partition, sample in (
                (
                    "train_pool",
                    sampled_train[: args.evaluation_train_graphs],
                ),
                (
                    "strict_heldout_pool",
                    sampled_heldout[: args.evaluation_heldout_graphs],
                ),
            ):
                result = audit_partition(
                    label=f"{label}:{partition}",
                    permutations=sample,
                    model=model,
                    cfg=cfg,
                    phase_positions=phase_positions,
                    positions=positions,
                    age_map=trained,
                    device=device,
                    batch_size=args.evaluation_batch_size,
                    continuation_loops=args.continuation_loops,
                )
                learned_accuracy = result["curves"][
                    "learned_J_plus_full_Block2"
                ]["accuracy_by_cycle"]
                result["variant"] = label
                result["hidden_width"] = hidden_width
                result["training_seed"] = training_seed
                result["primary_windows"] = curve_windows(learned_accuracy)
                evaluations.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "graph_path_telomere_graph_blind_mlp_j",
            "checkpoint": str(args.checkpoint),
            "positions": positions,
            "modules": {
                label: {
                    "hidden_width": module.hidden_width,
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in module.state_dict().items()
                    },
                }
                for label, module in trained_modules.items()
            },
        },
        args.out_dir / "graph_blind_mlp_j.pt",
    )
    write_csv(args.out_dir / "training_rounds.csv", training_rows)
    evaluation_rows = [
        {
            "variant": result["variant"],
            "partition": result["partition"],
            "hidden_width": result["hidden_width"],
            "training_seed": result["training_seed"],
            **result["primary_windows"],
            "block2_off_accuracy": result[
                "one_step_executor_blocked_accuracy"
            ][
                "learned_J_Block2_answer_updates_zero"
            ],
            **result["interface_alignment"],
        }
        for result in evaluations
    ]
    write_csv(args.out_dir / "evaluation_summary.csv", evaluation_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            "frozen backbone final-only; nonlinear J uses hidden-state age"
            " loss only; no successor labels, logits, or graph CE"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": args.continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "positions": positions,
        "J_definition": (
            "shared positionwise affine-plus-two-linear-layer GELU correction"
            " at every pre-Block2 residual"
        ),
        "split": {
            "seed": args.split_seed,
            "train_permutations": len(train_permutations),
            "heldout_permutations": len(heldout_permutations),
            "overlap": 0,
            "cycle_type_distribution": split_distribution,
        },
        "calibration": {
            "graphs": calibration.graphs,
            "seed": args.calibration_seed,
            "ridge": args.ridge,
            "natural_H8_to_H7_relative_mse": (
                calibration.source_target_relative_mse
            ),
            "target_scale": float(calibration.target_scale),
        },
        "training": {
            "successor_labels_used": False,
            "logits_used": False,
            "graph_CE_used": False,
            "hidden_widths": list(args.hidden_widths),
            "training_seeds": list(args.training_seeds),
            "rounds": args.rounds,
            "graphs_per_round": args.training_batch_size,
            "horizons": list(args.horizons),
            "epochs_per_round": args.epochs_per_round,
            "mini_batch_size": args.mini_batch_size,
            "learning_rate": args.learning_rate,
        },
        "variants": {
            label: {
                "hidden_width": module.hidden_width,
                "parameter_count": module.parameter_count,
            }
            for label, module in trained_modules.items()
        },
        "evaluation": {
            "seed": args.evaluation_seed,
            "matched_cycle_type_distribution": matched_distribution,
            "train_graphs_per_variant": args.evaluation_train_graphs,
            "heldout_graphs_per_variant": args.evaluation_heldout_graphs,
            "all_currents_per_graph": cfg.node_count,
            "results": evaluations,
        },
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "prelaunch_used_mib": args.prelaunch_used_mib,
            "prelaunch_free_mib": args.prelaunch_free_mib,
            "declared_peak_gib": args.declared_peak_gib,
            "reserve_gib": args.reserve_gib,
            "shared_gpu": args.shared_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "artifacts": {
            "modules": "graph_blind_mlp_j.pt",
            "training_rounds": "training_rounds.csv",
            "evaluation_summary": "evaluation_summary.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "variants": summary["variants"],
                "evaluation": evaluation_rows,
                "gpu_runtime": summary["gpu_runtime"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    set_seed(0)
    main()
