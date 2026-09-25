from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_learned_initializer import (
    _curve_summary,
    _write_csv,
)
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    _relative_mse,
    fit_reduced_rank_update_family,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import _all_targets
from reasoning_loop.graph_path_telomere_periodic_booster import (
    evaluate_periodic_booster,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    fit_weighted_full_affine,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


ROLE_NAMES = (
    "edge_marker",
    "source",
    "destination",
    "query_metadata",
    "answer",
)


@dataclass(frozen=True)
class RoleConditionedAffine:
    interface: tuple[int, ...]
    role_positions: dict[str, tuple[int, ...]]
    maps: dict[str, VectorAffine]

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2] != len(self.interface):
            raise ValueError("role-conditioned input has wrong position count")
        result = value.clone()
        global_to_local = {
            position: index
            for index, position in enumerate(self.interface)
        }
        for role, positions in self.role_positions.items():
            local = [global_to_local[position] for position in positions]
            result[:, local] = self.maps[role](value[:, local]).to(
                dtype=value.dtype
            )
        return result


@torch.no_grad()
def collect_role_pairs(
    *,
    model,
    cfg,
    phase_positions: list[int],
    role_positions: dict[str, tuple[int, ...]],
    feedback: VectorAffine,
    switch_cycle: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    jump = phase_positions[3] - phase_positions[2]
    sources = {role: [] for role in role_positions}
    targets_out = {role: [] for role in role_positions}
    set_seed(seed)
    interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    for _ in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * switch_cycle,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        for cycle in range(1, switch_cycle):
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    None
                    if cycle == 1
                    else (interface, feedback)
                ),
            )
            state = step.state
        current = all_targets[
            :, cfg.max_depth + jump * (switch_cycle - 1)
        ]
        source = run_one_loop(
            model,
            state,
            loop_index=cfg.max_loops + switch_cycle - 1,
        ).block2_hidden_pre_intervention
        oracle_input = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        target = run_one_loop(
            model,
            oracle_input,
            loop_index=cfg.max_loops + switch_cycle - 1,
        ).block2_hidden_in
        for role, positions in role_positions.items():
            index = list(positions)
            sources[role].append(
                source[:, index].reshape(-1, cfg.d_model).float()
            )
            targets_out[role].append(
                target[:, index].reshape(-1, cfg.d_model).float()
            )
    return {
        role: (
            torch.cat(sources[role]),
            torch.cat(targets_out[role]),
        )
        for role in role_positions
    }


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    feedback_artifact: Path,
    feedback_label: str,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    heldout_batch_size: int,
    heldout_batches: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    switch_cycle: int,
    period: int,
    ridge: float,
    role_rank: int,
    calibration_seed: int,
    heldout_seed: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("experiment is fixed to the D8L8 two-block model")
    if role_rank < 0 or role_rank > cfg.d_model:
        raise ValueError("role_rank must be between 0 and d_model")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    groups = explicit_depth_position_groups(cfg.node_count)
    role_positions = {role: groups[role] for role in ROLE_NAMES}
    interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if tuple(
        sorted(
            position
            for positions in role_positions.values()
            for position in positions
        )
    ) != interface:
        raise ValueError("role partition does not cover the interface")
    answer_position = groups["answer"][0]
    feedback, feedback_positions = _load_map(
        feedback_artifact,
        label=feedback_label,
        device=device,
    )
    if feedback_positions != interface:
        raise ValueError("feedback positions do not match the interface")
    training = collect_role_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        role_positions=role_positions,
        feedback=feedback,
        switch_cycle=switch_cycle,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed,
    )
    maps = {
        role: fit_reduced_rank_update_family(
            source,
            target,
            ridge=ridge,
        ).map_for_rank(role_rank)
        for role, (source, target) in training.items()
    }
    shared_source = torch.cat(
        [training[role][0] for role in ROLE_NAMES]
    )
    shared_target = torch.cat(
        [training[role][1] for role in ROLE_NAMES]
    )
    shared_weights = torch.cat(
        [
            torch.full(
                (training[role][0].shape[0],),
                28.0 if role == "answer" else 1.0,
                device=device,
            )
            for role in ROLE_NAMES
        ]
    )
    matched_shared = fit_weighted_full_affine(
        shared_source,
        shared_target,
        shared_weights,
        ridge=ridge,
    )
    role_booster = RoleConditionedAffine(
        interface=interface,
        role_positions=role_positions,
        maps=maps,
    )
    heldout_pairs = collect_role_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        role_positions=role_positions,
        feedback=feedback,
        switch_cycle=switch_cycle,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=heldout_seed,
    )
    heldout = {
        role: {
            "relative_mse": _relative_mse(
                maps[role](source),
                target,
            ),
            "sample_pairs": int(source.shape[0]),
        }
        for role, (source, target) in heldout_pairs.items()
    }
    closed_rows = evaluate_periodic_booster(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=interface,
        answer_position=answer_position,
        feedback=feedback,
        booster=role_booster,
        switch_cycle=switch_cycle,
        period=period,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=evaluation_seed,
        executor_head=2,
    )
    shared_rows = evaluate_periodic_booster(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=interface,
        answer_position=answer_position,
        feedback=feedback,
        booster=matched_shared,
        switch_cycle=switch_cycle,
        period=period,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=evaluation_seed,
        executor_head=2,
    )
    for row in shared_rows:
        if row["condition"] in (
            "learned_booster_once",
            "learned_booster_periodic",
        ):
            row = dict(row)
            row["condition"] = row["condition"].replace(
                "learned_booster",
                "matched_shared_booster",
            )
            closed_rows.append(row)
    curves = _curve_summary(closed_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "graph_path_telomere_role_conditioned_booster",
            "positions": interface,
            "role_positions": role_positions,
            "maps": {
                role: {
                    "weight": age_map.weight.cpu(),
                    "bias": age_map.bias.cpu(),
                    "rank": age_map.update_rank,
                    "retained_fit_energy": age_map.retained_fit_energy,
                }
                for role, age_map in maps.items()
            },
            "matched_shared_map": {
                "weight": matched_shared.weight.cpu(),
                "bias": matched_shared.bias.cpu(),
                "rank": matched_shared.update_rank,
                "retained_fit_energy": matched_shared.retained_fit_energy,
                "answer_weight": 28,
            },
        },
        out_dir / "role_conditioned_booster.pt",
    )
    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "feedback_artifact": str(feedback_artifact),
        "feedback_label": feedback_label,
        "booster": {
            "form": (
                "five role-conditioned shared-within-role "
                "256x256+b affine maps"
            ),
            "roles": list(ROLE_NAMES),
            "role_positions": {
                role: list(positions)
                for role, positions in role_positions.items()
            },
            "parameter_count": len(ROLE_NAMES)
            * (
                role_rank * (2 * cfg.d_model - role_rank)
                + cfg.d_model
            ),
            "role_update_rank": role_rank,
            "matched_shared_parameter_count": (
                cfg.d_model**2 + cfg.d_model
            ),
            "training_relation": (
                f"feedback pre-map state at cycle {switch_cycle} "
                "-> exact H2 Block2 input"
            ),
            "training_loss": "rolewise state MSE only",
            "ridge": ridge,
            "graphs": calibration_batch_size * calibration_batches,
            "switch_cycle": switch_cycle,
            "period": period,
            "excluded": [
                "task CE",
                "closed-loop task loss",
                "scheduled-state DAgger",
            ],
        },
        "heldout": heldout,
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "extra_loops": extra_loops,
        "closed_loop": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "booster": "role_conditioned_booster.pt",
            "closed_loop": "closed_loop_rows.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether forcing answer, graph roles, and metadata through "
            "one shared affine causes the late-renewal failure."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--feedback-artifact", type=Path, required=True)
    parser.add_argument("--feedback-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--heldout-batch-size", type=int, default=128)
    parser.add_argument("--heldout-batches", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=128)
    parser.add_argument("--switch-cycle", type=int, default=24)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--role-rank", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=140101)
    parser.add_argument("--heldout-seed", type=int, default=140102)
    parser.add_argument("--evaluation-seed", type=int, default=140103)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        feedback_artifact=args.feedback_artifact,
        feedback_label=args.feedback_label,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        heldout_batch_size=args.heldout_batch_size,
        heldout_batches=args.heldout_batches,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        extra_loops=args.extra_loops,
        switch_cycle=args.switch_cycle,
        period=args.period,
        ridge=args.ridge,
        role_rank=args.role_rank,
        calibration_seed=args.calibration_seed,
        heldout_seed=args.heldout_seed,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
