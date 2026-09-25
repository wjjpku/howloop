"""Executable invariants for age-specific J data generation and final-CE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    AgeTrajectory,
    MAX_AGE,
    MIN_AGE,
    ROLLBACK_SOURCE_AGES,
    _age_path,
    _run_mixed_trajectory,
    sample_bounded_bridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=941001)
    parser.add_argument("--bridge-samples", type=int, default=5000)
    parser.add_argument("--numeric-trajectories", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _trajectory(start: int, actions: tuple[int, ...]) -> AgeTrajectory:
    ages = _age_path(start, actions)
    return AgeTrajectory(
        start_age=start,
        end_age=ages[-1],
        extra_backs=actions.count(-1),
        actions=actions,
        ages=ages[:-1],
        mandatory_rollback_source=None,
    )


def audit_bridge_sampler(samples: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    checked = 0
    boundary_checked = 0
    curriculum = (
        (2, 1), (2, 2), (3, 2), (3, 3),
        (4, 3), (4, 4), (5, 4), (5, 5),
    )
    for maximum_total, maximum_run in curriculum:
        for index in range(samples):
            mandatory = ROLLBACK_SOURCE_AGES[index % len(ROLLBACK_SOURCE_AGES)]
            trajectory = sample_bounded_bridge(
                rng=rng,
                max_total_backs=maximum_total,
                mandatory_rollback_source=mandatory,
                max_consecutive_backs=maximum_run,
            )
            full_path = _age_path(trajectory.start_age, trajectory.actions)
            assert full_path[:-1] == trajectory.ages
            assert full_path[-1] == trajectory.end_age
            assert min(full_path) >= MIN_AGE and max(full_path) <= MAX_AGE
            assert trajectory.forward_count - trajectory.back_count == (
                trajectory.end_age - trajectory.start_age
            )
            assert trajectory.back_count == (
                max(0, trajectory.start_age - trajectory.end_age)
                + trajectory.extra_backs
            )
            assert mandatory in trajectory.rollback_sources
            assert trajectory.back_count <= maximum_total
            assert trajectory.max_rollback_run <= maximum_run
            for age, action in zip(
                trajectory.ages, trajectory.actions, strict=True
            ):
                if action == -1:
                    assert age in ROLLBACK_SOURCE_AGES
            checked += 1
        for _ in range(samples):
            trajectory = sample_bounded_bridge(
                rng=rng,
                max_total_backs=maximum_total,
                minimum_total_backs=maximum_total,
                mandatory_rollback_source=None,
                max_consecutive_backs=maximum_run,
                required_consecutive_backs=maximum_run,
            )
            full_path = _age_path(trajectory.start_age, trajectory.actions)
            assert min(full_path) >= MIN_AGE and max(full_path) <= MAX_AGE
            assert trajectory.back_count == maximum_total
            assert trajectory.max_rollback_run == maximum_run
            boundary_checked += 1
    return {
        "bridge_trajectories_checked": checked,
        "hard_boundary_trajectories_checked": boundary_checked,
    }


def audit_numeric_execution(args: argparse.Namespace) -> dict[str, float | int]:
    device = pick_device(args.device)
    set_seed(args.seed)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text())
    phase_positions = tuple(
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    )
    assert phase_positions[:9] == tuple(range(9))
    positions = tuple(range(cfg.seq_len))
    bank = AgeSpecificJBank(
        dimension=cfg.d_model,
        rank=cfg.d_model,
        map_architecture="full_affine",
    ).to(device)
    assert bank.parameter_count == 7 * cfg.d_model * (cfg.d_model + 1)

    rng = np.random.default_rng(args.seed + 1)
    example_actions = (1, -1, -1, 1, 1, 1, 1, -1, 1, 1)
    trajectories = [_trajectory(3, example_actions)]
    trajectories.extend(
        sample_bounded_bridge(
            rng=rng,
            max_total_backs=5,
            mandatory_rollback_source=ROLLBACK_SOURCE_AGES[index % 7],
            max_consecutive_backs=5,
        )
        for index in range(args.numeric_trajectories - 1)
    )
    maximum_exact_error = 0.0
    maximum_target_mismatch = 0
    for trajectory in trajectories:
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            args.batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        initial_state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=trajectory.start_age,
            phase_position=phase_positions[trajectory.start_age],
        )
        final_state, _, target = _run_mixed_trajectory(
            model=model,
            cfg=cfg,
            bank=bank,
            positions=positions,
            successors=successors,
            endpoint=endpoint,
            initial_state=initial_state,
            trajectory=trajectory,
            condition="exact",
            phase_positions=list(phase_positions),
        )
        expected_target = advance_nodes(
            successors, endpoint, steps=trajectory.forward_count
        )
        maximum_target_mismatch = max(
            maximum_target_mismatch,
            int(target.ne(expected_target).sum()),
        )
        expected_state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=expected_target,
            age=trajectory.end_age,
            phase_position=phase_positions[trajectory.end_age],
        )
        maximum_exact_error = max(
            maximum_exact_error,
            float((final_state - expected_state).abs().max()),
        )

    gradient_trajectory = _trajectory(
        8,
        (-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1),
    )
    _, path_targets, successors, _ = fixed_depth_batch(
        cfg, args.batch_size, device, path_positions=cfg.max_depth
    )
    endpoint = path_targets[:, cfg.max_depth - 1]
    initial_state = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=endpoint,
        age=gradient_trajectory.start_age,
        phase_position=phase_positions[gradient_trajectory.start_age],
    )
    bank.zero_grad(set_to_none=True)
    _, logits, target = _run_mixed_trajectory(
        model=model,
        cfg=cfg,
        bank=bank,
        positions=positions,
        successors=successors,
        endpoint=endpoint,
        initial_state=initial_state,
        trajectory=gradient_trajectory,
        condition="learned",
        phase_positions=list(phase_positions),
    )
    final_ce = F.cross_entropy(logits.float(), target)
    final_ce.backward()
    map_gradient_norms = {
        age: sum(
            float(parameter.grad.float().square().sum())
            for parameter in bank.maps[str(age)].parameters()
            if parameter.grad is not None
        ) ** 0.5
        for age in ROLLBACK_SOURCE_AGES
    }
    assert all(value > 0 for value in map_gradient_norms.values())
    assert all(parameter.grad is None for parameter in model.parameters())
    return {
        "numeric_trajectories_checked": len(trajectories),
        "maximum_exact_state_absolute_error": maximum_exact_error,
        "maximum_graph_target_mismatch_count": maximum_target_mismatch,
        "final_ce_for_gradient_test": float(final_ce.detach()),
        "minimum_used_J_gradient_norm": min(map_gradient_norms.values()),
        "backbone_parameters_with_gradients": sum(
            parameter.grad is not None for parameter in model.parameters()
        ),
        "full_affine_parameter_count": bank.parameter_count,
    }


def main(args: argparse.Namespace) -> None:
    result = {
        "status": "pass",
        "loss_definition": "one final cross entropy after the complete trajectory",
        "hidden_state_loss": 0.0,
        **audit_bridge_sampler(args.bridge_samples, args.seed),
        **audit_numeric_execution(args),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
