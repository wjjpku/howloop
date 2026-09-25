"""Train seven age-specific loop-boundary rollback maps with final CE only.

The frozen D8L8 backbone supplies the forward operator F.  Seven distinct
diagonal-plus-low-rank affine maps implement the legal adjacent rollbacks

    J_a: H_a -> H_{a-1},  a in {2, ..., 8}.

Every optimizer batch samples a random bounded bridge in age space.  A +1
transition runs one complete recurrent loop F; a -1 transition applies only
the J associated with the current source age.  The sole optimized objective is
task cross entropy at the final state of the complete mixed trajectory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.analyze_graph_path_j_transition_matrices import (
    affine_metrics,
    fit_affine,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine


MIN_AGE = 1
MAX_AGE = 8
ROLLBACK_SOURCE_AGES = tuple(range(2, 9))


@dataclass(frozen=True)
class MixedStage:
    name: str
    rounds: int
    batch_size: int
    batches_per_round: int
    max_total_backs: int
    max_consecutive_backs: int
    learning_rate: float
    data_seed: int


DEFAULT_STAGES = (
    MixedStage("backs_1", 24, 64, 14, 1, 7, 3e-4, 820101),
    MixedStage("backs_4", 20, 48, 14, 4, 7, 1e-4, 820201),
    MixedStage("backs_8", 20, 32, 14, 8, 7, 5e-5, 820301),
    MixedStage("backs_16", 16, 24, 14, 16, 7, 2e-5, 820401),
    MixedStage("backs_28", 12, 16, 14, 28, 7, 1e-5, 820501),
)


TWO_AXIS_STAGES = (
    MixedStage("T04_R1", 6, 48, 14, 4, 1, 3e-4, 821101),
    MixedStage("T08_R1", 6, 44, 14, 8, 1, 2.5e-4, 821201),
    MixedStage("T08_R2", 6, 40, 14, 8, 2, 2e-4, 821301),
    MixedStage("T12_R2", 6, 36, 14, 12, 2, 1.5e-4, 821401),
    MixedStage("T12_R3", 6, 32, 14, 12, 3, 1e-4, 821501),
    MixedStage("T16_R3", 6, 30, 14, 16, 3, 8e-5, 821601),
    MixedStage("T16_R4", 6, 28, 14, 16, 4, 7e-5, 821701),
    MixedStage("T20_R4", 6, 26, 14, 20, 4, 6e-5, 821801),
    MixedStage("T20_R5", 6, 24, 14, 20, 5, 5e-5, 821901),
    MixedStage("T24_R5", 6, 22, 14, 24, 5, 4e-5, 822001),
    MixedStage("T24_R6", 6, 20, 14, 24, 6, 3e-5, 822101),
    MixedStage("T28_R6", 6, 18, 14, 28, 6, 2.5e-5, 822201),
    MixedStage("T28_R7", 12, 16, 14, 28, 7, 2e-5, 822301),
)


SINGLE_BACK_STAGES = (
    MixedStage("T01_R1", 48, 64, 14, 1, 1, 3e-4, 823101),
)


# Continue from the fully trained one-rollback checkpoint.  This curriculum
# spends capacity inside the requested compact support instead of increasing
# trajectory length: total J calls <= 5 and consecutive J calls <= 5.
FOCUSED5_STAGES = (
    MixedStage("T02_R1", 48, 64, 14, 2, 1, 3e-4, 824101),
    MixedStage("T02_R2", 48, 64, 14, 2, 2, 2e-4, 824201),
    MixedStage("T03_R2", 48, 64, 14, 3, 2, 1.5e-4, 824301),
    MixedStage("T03_R3", 48, 64, 14, 3, 3, 1e-4, 824401),
    MixedStage("T04_R3", 48, 64, 14, 4, 3, 8e-5, 824501),
    MixedStage("T04_R4", 48, 64, 14, 4, 4, 6e-5, 824601),
    MixedStage("T05_R4", 48, 64, 14, 5, 4, 4e-5, 824701),
    MixedStage("T05_R5", 72, 64, 14, 5, 5, 3e-5, 824801),
)


# Fine-tune the focused5 result without spending half of the batches on the
# already-saturated (5, 5) boundary.  Each round covers every non-(5, 5)
# (total J count, maximum J run) stratum once, then retains five (5, 5)
# batches to prevent catastrophic forgetting: 14/19 vs 5/19.
BALANCED5_MIXED_STRATA = tuple(
    (total, run)
    for total in range(1, 6)
    for run in range(1, total + 1)
    if (total, run) != (5, 5)
)


BALANCED5_STAGES = (
    MixedStage("B74_26_lr1e6", 64, 64, 19, 5, 5, 1e-5, 825101),
    MixedStage("B74_26_lr3e7", 64, 64, 19, 5, 5, 3e-6, 825201),
)


# A low-LR continuation with exactly 10,000 online optimizer batches.  Total
# J reuse counts k=1..5 are stratified with probability proportional to 1/k.
INVERSE_REUSE_STAGES = (
    MixedStage("inverse_reuse_10k", 100, 64, 100, 5, 5, 1e-5, 826101),
)


@dataclass(frozen=True)
class AgeTrajectory:
    start_age: int
    end_age: int
    extra_backs: int
    actions: tuple[int, ...]
    ages: tuple[int, ...]
    mandatory_rollback_source: int | None

    @property
    def forward_count(self) -> int:
        return self.actions.count(1)

    @property
    def back_count(self) -> int:
        return self.actions.count(-1)

    @property
    def rollback_sources(self) -> tuple[int, ...]:
        return tuple(
            age for age, action in zip(self.ages, self.actions, strict=True)
            if action == -1
        )

    @property
    def max_rollback_run(self) -> int:
        longest = current = 0
        for action in self.actions:
            if action == -1:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest


class FullAffineJ(torch.nn.Module):
    """A position-shared unrestricted affine map, using row-vector convention."""

    def __init__(self, *, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.weight = torch.nn.Parameter(torch.eye(dimension, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(dimension, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() @ self.weight + self.bias


class AgeSpecificJBank(torch.nn.Module):
    """Seven independent position-shared affine rollback maps."""

    def __init__(
        self,
        *,
        dimension: int,
        rank: int,
        stage_rank: int | None = None,
        map_architecture: str = "diagonal_lora",
    ) -> None:
        super().__init__()
        if map_architecture not in {
            "diagonal_lora", "full_affine", "shared_diagonal_stage_lora"
        }:
            raise ValueError(f"unknown map architecture: {map_architecture}")
        self.dimension = dimension
        self.rank = rank
        self.stage_rank = rank if stage_rank is None else stage_rank
        self.map_architecture = map_architecture
        if not 1 <= self.stage_rank <= dimension:
            raise ValueError("stage rank must lie in [1, dimension]")
        if map_architecture == "shared_diagonal_stage_lora":
            self.maps = torch.nn.ModuleDict()
            self.shared_diagonal_scale = torch.nn.Parameter(torch.ones(dimension))
            self.shared_A = torch.nn.Parameter(torch.empty(dimension, rank))
            self.shared_B = torch.nn.Parameter(torch.zeros(rank, dimension))
            self.shared_bias = torch.nn.Parameter(torch.zeros(dimension))
            self.stage_A = torch.nn.ParameterDict(
                {
                    str(age): torch.nn.Parameter(
                        torch.empty(dimension, self.stage_rank)
                    )
                    for age in ROLLBACK_SOURCE_AGES
                }
            )
            self.stage_B = torch.nn.ParameterDict(
                {
                    str(age): torch.nn.Parameter(
                        torch.zeros(self.stage_rank, dimension)
                    )
                    for age in ROLLBACK_SOURCE_AGES
                }
            )
            torch.nn.init.normal_(
                self.shared_A, mean=0.0, std=1.0 / math.sqrt(dimension)
            )
            for value in self.stage_A.values():
                torch.nn.init.normal_(
                    value, mean=0.0, std=1.0 / math.sqrt(dimension)
                )
        else:
            self.maps = torch.nn.ModuleDict(
                {
                    str(age): (
                        DiagonalIdentityLoRAJ(dimension=dimension, rank=rank)
                        if map_architecture == "diagonal_lora"
                        else FullAffineJ(dimension=dimension)
                    )
                    for age in ROLLBACK_SOURCE_AGES
                }
            )

    def affine(self, source_age: int) -> tuple[torch.Tensor, torch.Tensor]:
        if source_age not in ROLLBACK_SOURCE_AGES:
            raise ValueError(f"no legal rollback from H{source_age}")
        if self.map_architecture == "shared_diagonal_stage_lora":
            weight = (
                torch.diag(self.shared_diagonal_scale.float())
                + self.shared_A.float() @ self.shared_B.float()
                + self.stage_A[str(source_age)].float()
                @ self.stage_B[str(source_age)].float()
            )
            return weight, self.shared_bias.float()
        return _affine(self.maps[str(source_age)])

    def rollback(
        self,
        state: torch.Tensor,
        *,
        source_age: int,
        positions: tuple[int, ...],
    ) -> torch.Tensor:
        if source_age not in ROLLBACK_SOURCE_AGES:
            raise ValueError(f"no legal rollback from H{source_age}")
        index = list(positions)
        result = state.clone()
        if self.map_architecture == "shared_diagonal_stage_lora":
            weight, bias = self.affine(source_age)
            updated = result[:, index].float() @ weight + bias
        else:
            updated = self.maps[str(source_age)](result[:, index])
        result[:, index] = updated.to(dtype=state.dtype)
        return result

    def rollback_composed(
        self,
        state: torch.Tensor,
        *,
        source_ages: Sequence[int],
        positions: tuple[int, ...],
        composition: str,
    ) -> torch.Tensor:
        """Apply one consecutive rollback run under a selected composition law."""
        if not source_ages:
            return state
        if composition == "product":
            for source_age in source_ages:
                state = self.rollback(
                    state, source_age=source_age, positions=positions
                )
            return state
        if composition == "pure_residual":
            identity = torch.eye(
                self.dimension, device=state.device, dtype=torch.float32
            )
            affines = [self.affine(age) for age in source_ages]
            weight = identity + sum(
                (value[0] - identity for value in affines),
                start=torch.zeros_like(identity),
            )
            bias = sum(
                (value[1] for value in affines),
                start=torch.zeros(
                    self.dimension, device=state.device, dtype=torch.float32
                ),
            )
            indices = list(positions)
            result = state.clone()
            result[:, indices] = (
                result[:, indices].float() @ weight + bias
            ).to(state.dtype)
            return result
        if composition in {"diag_product_u_sum", "diag_left_gated_residual"}:
            if self.map_architecture != "shared_diagonal_stage_lora":
                raise TypeError(
                    f"{composition} requires shared_diagonal_stage_lora"
                )
            diagonal = torch.diag(self.shared_diagonal_scale.float())
            updates = [self.affine(age)[0] - diagonal for age in source_ages]
            identity = torch.eye(
                self.dimension, device=state.device, dtype=torch.float32
            )
            diagonal_prefix = identity
            residual = torch.zeros_like(identity)
            for update in updates:
                if composition == "diag_product_u_sum":
                    residual = residual + update
                else:
                    residual = residual + diagonal_prefix @ update
                diagonal_prefix = diagonal_prefix @ diagonal
            weight = diagonal_prefix + residual
            bias = len(source_ages) * self.shared_bias.float()
            indices = list(positions)
            result = state.clone()
            result[:, indices] = (
                result[:, indices].float() @ weight + bias
            ).to(state.dtype)
            return result
        if composition != "diag_gated_residual":
            raise ValueError(f"unknown rollback composition: {composition}")
        if self.map_architecture != "shared_diagonal_stage_lora":
            raise TypeError(
                "diag_gated_residual requires shared_diagonal_stage_lora"
            )
        diagonal = torch.diag(self.shared_diagonal_scale.float())
        updates = [self.affine(age)[0] - diagonal for age in source_ages]
        identity = torch.eye(
            self.dimension, device=state.device, dtype=torch.float32
        )
        prefixes = [identity]
        for _ in source_ages:
            prefixes.append(prefixes[-1] @ diagonal)
        suffixes = [identity for _ in range(len(source_ages) + 1)]
        for index in range(len(source_ages) - 1, -1, -1):
            suffixes[index] = diagonal @ suffixes[index + 1]
        weight = prefixes[-1]
        for index, update in enumerate(updates):
            weight = weight + prefixes[index] @ update @ suffixes[index + 1]
        bias = torch.zeros(
            self.dimension, device=state.device, dtype=torch.float32
        )
        for index in range(len(source_ages)):
            bias = bias + self.shared_bias.float() @ suffixes[index + 1]
        indices = list(positions)
        result = state.clone()
        result[:, indices] = (
            result[:, indices].float() @ weight + bias
        ).to(state.dtype)
        return result

    def frozen(self) -> "AgeSpecificJBank":
        result = copy.deepcopy(self).eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument(
        "--stage-rank",
        type=int,
        default=16,
        help="Per-stage LoRA rank for shared_diagonal_stage_lora.",
    )
    parser.add_argument(
        "--map-architecture",
        choices=("diagonal_lora", "full_affine", "shared_diagonal_stage_lora"),
        default="diagonal_lora",
    )
    parser.add_argument("--seed", type=int, default=820001)
    parser.add_argument(
        "--curriculum",
        choices=(
            "legacy_extra",
            "single_back",
            "two_axis",
            "focused5",
            "balanced5",
            "inverse_reuse",
        ),
        default="legacy_extra",
    )
    parser.add_argument(
        "--initialization",
        choices=(
            "identity",
            "adjacent_regression",
            "canonical_shared",
            "age_specific_bank",
        ),
        default="adjacent_regression",
    )
    parser.add_argument("--canonical-artifact", type=Path)
    parser.add_argument("--canonical-label")
    parser.add_argument("--bank-init-artifact", type=Path)
    parser.add_argument(
        "--allow-canonical-checkpoint-path-mismatch",
        action="store_true",
        help="Only for a byte-copied local checkpoint whose absolute path changed.",
    )
    parser.add_argument("--calibration-examples", type=int, default=1024)
    parser.add_argument("--calibration-heldout-examples", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    parser.add_argument("--calibration-ridge", type=float, default=1e-3)
    parser.add_argument("--stage-round-limit", type=int)
    parser.add_argument(
        "--stage-round-multiplier",
        type=int,
        default=1,
        help="Multiply every curriculum stage's round count after applying the optional cap.",
    )
    parser.add_argument("--max-stages", type=int)
    parser.add_argument("--batches-per-round", type=int)
    parser.add_argument("--batch-size-cap", type=int)
    parser.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.0,
        help="Linear LR warmup fraction applied independently within every stage.",
    )
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument(
        "--lr-schedule",
        choices=("warmup_constant", "wsd"),
        default="warmup_constant",
        help="Per-stage warmup-constant or warmup-stable-decay schedule.",
    )
    parser.add_argument(
        "--decay-fraction",
        type=float,
        default=0.2,
        help="Final fraction of each stage used for WSD linear decay.",
    )
    parser.add_argument(
        "--decay-end-factor",
        type=float,
        default=0.1,
        help="Final LR divided by target LR at the end of WSD decay.",
    )
    parser.add_argument("--replay-max-backs", type=int)
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    parser.add_argument(
        "--fixed-start-age",
        type=int,
        choices=range(MIN_AGE, MAX_AGE + 1),
        help=(
            "Fix every training and primary evaluation trajectory to this "
            "natural starting age instead of sampling H1..H8."
        ),
    )
    parser.add_argument("--diagonal-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--rollback-composition",
        choices=(
            "product",
            "diag_gated_residual",
            "pure_residual",
            "diag_product_u_sum",
            "diag_left_gated_residual",
        ),
        default="product",
        help="How each consecutive run of J calls is composed.",
    )
    parser.add_argument("--eval-trajectories", type=int, default=56)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-train-max-backs", type=int, default=28)
    parser.add_argument("--eval-unseen-max-backs", type=int, default=40)
    parser.add_argument("--composition-examples", type=int, default=256)
    parser.add_argument("--composition-batch-size", type=int, default=64)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.06)
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _age_path(start_age: int, actions: Iterable[int]) -> tuple[int, ...]:
    age = start_age
    ages = [age]
    for action in actions:
        if action not in (-1, 1):
            raise ValueError("age actions must be -1 or +1")
        age += action
        if not MIN_AGE <= age <= MAX_AGE:
            raise ValueError("trajectory leaves H1..H8")
        ages.append(age)
    return tuple(ages)


def sample_bounded_bridge(
    *,
    rng: np.random.Generator,
    max_extra_backs: int | None = None,
    max_total_backs: int | None = None,
    minimum_total_backs: int | None = None,
    mandatory_rollback_source: int | None = None,
    max_consecutive_backs: int | None = None,
    required_consecutive_backs: int = 0,
    fixed_start_age: int | None = None,
) -> AgeTrajectory:
    """Sample start/end first, then a random valid bridge between them."""

    if (max_extra_backs is None) == (max_total_backs is None):
        raise ValueError(
            "provide exactly one of max_extra_backs and max_total_backs"
        )
    if max_extra_backs is not None and max_extra_backs < 1:
        raise ValueError("max_extra_backs must be positive")
    if max_total_backs is not None and max_total_backs < 1:
        raise ValueError("max_total_backs must be positive")
    if minimum_total_backs is not None:
        if max_total_backs is None:
            raise ValueError("minimum_total_backs requires max_total_backs")
        if not 1 <= minimum_total_backs <= max_total_backs:
            raise ValueError("minimum_total_backs must lie in [1, max_total_backs]")
    if max_consecutive_backs is None:
        max_consecutive_backs = MAX_AGE - MIN_AGE
    if not 1 <= max_consecutive_backs <= MAX_AGE - MIN_AGE:
        raise ValueError("max_consecutive_backs must lie in [1, 7]")
    if not 0 <= required_consecutive_backs <= max_consecutive_backs:
        raise ValueError(
            "required_consecutive_backs must lie in [0, max_consecutive_backs]"
        )
    if (
        mandatory_rollback_source is not None
        and mandatory_rollback_source not in ROLLBACK_SOURCE_AGES
    ):
        raise ValueError("mandatory rollback source must lie in H2..H8")
    if fixed_start_age is not None and not MIN_AGE <= fixed_start_age <= MAX_AGE:
        raise ValueError("fixed start age must lie in H1..H8")

    for _ in range(256):
        start = (
            fixed_start_age
            if fixed_start_age is not None
            else int(rng.integers(MIN_AGE, MAX_AGE + 1))
        )
        end = int(rng.integers(MIN_AGE, MAX_AGE + 1))
        minimum_down = max(0, start - end)
        if max_total_backs is not None:
            lower_down = max(1, minimum_down, minimum_total_backs or 1)
            if lower_down > max_total_backs:
                continue
            down = int(rng.integers(lower_down, max_total_backs + 1))
            up = down + end - start
            extra = down - minimum_down
        else:
            assert max_extra_backs is not None
            extra = int(rng.integers(1, max_extra_backs + 1))
            down = minimum_down + extra
            up = max(0, end - start) + extra

        @lru_cache(maxsize=None)
        def count(
            age: int,
            ups: int,
            downs: int,
            seen: bool,
            down_run: int,
            longest_down_run: int,
        ) -> int:
            if ups == 0 and downs == 0:
                return int(
                    age == end
                    and (mandatory_rollback_source is None or seen)
                    and longest_down_run >= required_consecutive_backs
                )
            total = 0
            if ups and age < MAX_AGE:
                total += count(
                    age + 1, ups - 1, downs, seen, 0, longest_down_run
                )
            if (
                downs
                and age > MIN_AGE
                and down_run < max_consecutive_backs
            ):
                total += count(
                    age - 1,
                    ups,
                    downs - 1,
                    seen or age == mandatory_rollback_source,
                    down_run + 1,
                    max(longest_down_run, down_run + 1),
                )
            return total

        if count(start, up, down, False, 0, 0) == 0:
            continue
        age, ups, downs, seen, down_run, longest_down_run = (
            start, up, down, False, 0, 0
        )
        actions: list[int] = []
        while ups or downs:
            choices: list[tuple[int, int, bool, int, int, int]] = []
            if ups and age < MAX_AGE:
                ways = count(
                    age + 1, ups - 1, downs, seen, 0, longest_down_run
                )
                if ways:
                    choices.append(
                        (1, ways, seen, age + 1, 0, longest_down_run)
                    )
            if (
                downs
                and age > MIN_AGE
                and down_run < max_consecutive_backs
            ):
                next_seen = seen or age == mandatory_rollback_source
                next_run = down_run + 1
                next_longest = max(longest_down_run, next_run)
                ways = count(
                    age - 1,
                    ups,
                    downs - 1,
                    next_seen,
                    next_run,
                    next_longest,
                )
                if ways:
                    choices.append(
                        (-1, ways, next_seen, age - 1, next_run, next_longest)
                    )
            total = sum(choice[1] for choice in choices)
            draw = int(rng.integers(0, total)) if total < 2**63 else int(
                rng.random() * total
            )
            cumulative = 0
            selected = choices[-1]
            for choice in choices:
                cumulative += choice[1]
                if draw < cumulative:
                    selected = choice
                    break
            action, _, seen, next_age, down_run, longest_down_run = selected
            actions.append(action)
            age = next_age
            if action == 1:
                ups -= 1
            else:
                downs -= 1
        ages = _age_path(start, actions)
        result = AgeTrajectory(
            start_age=start,
            end_age=end,
            extra_backs=extra,
            actions=tuple(actions),
            ages=ages[:-1],
            mandatory_rollback_source=mandatory_rollback_source,
        )
        if ages[-1] != end:
            raise RuntimeError("bridge endpoint mismatch")
        if mandatory_rollback_source is not None and (
            mandatory_rollback_source not in result.rollback_sources
        ):
            raise RuntimeError("mandatory rollback was not sampled")
        if result.max_rollback_run > max_consecutive_backs:
            raise RuntimeError("rollback composition exceeds the requested cap")
        if (
            minimum_total_backs is not None
            and result.back_count < minimum_total_backs
        ):
            raise RuntimeError("rollback count is below the requested minimum")
        if result.max_rollback_run < required_consecutive_backs:
            raise RuntimeError("rollback composition misses the required run")
        return result
    raise RuntimeError("could not sample a legal bounded bridge")


def _trajectory_text(trajectory: AgeTrajectory) -> str:
    final_age = trajectory.end_age
    return ",".join(str(age) for age in (*trajectory.ages, final_age))


def _optimizer(bank: AgeSpecificJBank, stage: MixedStage, diagonal_scale: float):
    scale_parameters = []
    other_parameters = []
    for name, parameter in bank.named_parameters():
        if name.endswith("diagonal_scale"):
            scale_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups = [{"params": other_parameters}]
    if scale_parameters:
        groups.append(
            {
                "params": scale_parameters,
                "lr": stage.learning_rate * diagonal_scale,
            }
        )
    return torch.optim.AdamW(
        groups,
        lr=stage.learning_rate,
        weight_decay=0.0,
    )


def _learning_rate_scale(
    *,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_fraction: float,
    warmup_start_factor: float,
    decay_fraction: float,
    decay_end_factor: float,
) -> float:
    """Return the scalar LR schedule for a one-indexed optimizer step."""
    if not 1 <= step <= total_steps:
        raise ValueError("step must be inside the stage")
    warmup_steps = int(math.ceil(total_steps * warmup_fraction))
    if warmup_steps and step <= warmup_steps:
        return warmup_start_factor + (1.0 - warmup_start_factor) * (
            step / warmup_steps
        )
    if schedule == "warmup_constant":
        return 1.0
    if schedule != "wsd":
        raise ValueError(f"unknown LR schedule: {schedule}")
    decay_steps = int(math.ceil(total_steps * decay_fraction))
    if not decay_steps or step <= total_steps - decay_steps:
        return 1.0
    decay_progress = (step - (total_steps - decay_steps)) / decay_steps
    return 1.0 + (decay_end_factor - 1.0) * decay_progress


def _inverse_reuse_schedule(total_steps: int, seed: int) -> tuple[int, ...]:
    """Return a shuffled exact quota with counts proportional to 1/k, k=1..5."""
    if total_steps < 5:
        raise ValueError("inverse-reuse schedule needs at least five steps")
    weights = 1.0 / np.arange(1, 6, dtype=np.float64)
    expected = total_steps * weights / weights.sum()
    counts = np.floor(expected).astype(np.int64)
    remainder = total_steps - int(counts.sum())
    fractional_order = np.argsort(-(expected - counts), kind="stable")
    counts[fractional_order[:remainder]] += 1
    schedule = np.concatenate(
        [np.full(int(count), reuse, dtype=np.int64) for reuse, count in enumerate(counts, 1)]
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(schedule)
    return tuple(int(value) for value in schedule)


def _sample_inverse_reuse_balanced_trajectory(
    *,
    rng: np.random.Generator,
    total_backs: int,
    rollback_run: int,
    cumulative_calls: dict[int, int],
    fixed_start_age: int,
) -> AgeTrajectory:
    """Sample a legal path while greedily equalizing total calls to J2..J8.

    A single mandatory rollback only makes the *first* selected source uniform:
    extra rollbacks still bias training toward low ages when every rollout starts
    at H1.  We therefore draw one legal candidate for each mandatory source and
    keep the candidate whose complete rollback multiset best equalizes the
    aggregate J-call counts after this batch.
    """
    if total_backs < 1 or rollback_run < 1 or rollback_run > total_backs:
        raise ValueError("invalid inverse-reuse stratum")
    total_after = sum(cumulative_calls.values()) + total_backs
    target = total_after / len(ROLLBACK_SOURCE_AGES)
    candidates: list[tuple[float, int, AgeTrajectory]] = []
    mandatory_order = rng.permutation(ROLLBACK_SOURCE_AGES)
    for mandatory in mandatory_order:
        child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        trajectory = sample_bounded_bridge(
            rng=child_rng,
            max_total_backs=total_backs,
            minimum_total_backs=total_backs,
            mandatory_rollback_source=int(mandatory),
            max_consecutive_backs=rollback_run,
            required_consecutive_backs=rollback_run,
            fixed_start_age=fixed_start_age,
        )
        after = dict(cumulative_calls)
        for age in trajectory.rollback_sources:
            after[age] += 1
        squared_error = sum((after[age] - target) ** 2 for age in ROLLBACK_SOURCE_AGES)
        spread = max(after.values()) - min(after.values())
        candidates.append((squared_error, spread, trajectory))
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


@torch.no_grad()
def _collect_aligned_age_states(
    *,
    model,
    cfg,
    phase_positions: list[int],
    device: torch.device,
    examples: int,
    batch_size: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    if examples % batch_size:
        raise ValueError("calibration examples must be divisible by batch size")
    set_seed(seed)
    chunks = {age: [] for age in range(MIN_AGE, MAX_AGE + 1)}
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        current = path_targets[:, cfg.max_depth - 1]
        for age in chunks:
            chunks[age].append(
                _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=age,
                    phase_position=phase_positions[age],
                ).float()
            )
    return {age: torch.cat(values) for age, values in chunks.items()}


@torch.no_grad()
def _factorize_low_rank(
    value: torch.Tensor, *, rank: int, gauge_seed: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    left, singular, right_t = torch.linalg.svd(value.float(), full_matrices=False)
    selected = singular[:rank]
    root = selected.sqrt()
    factor_a = left[:, :rank] * root.unsqueeze(0)
    factor_b = root.unsqueeze(1) * right_t[:rank]
    generator = torch.Generator(device=value.device).manual_seed(gauge_seed)
    random = torch.randn(
        rank, rank, generator=generator, device=value.device, dtype=torch.float32
    )
    gauge, _ = torch.linalg.qr(random)
    retained = float(
        selected.square().sum() / singular.square().sum().clamp_min(1e-12)
    )
    return factor_a @ gauge, gauge.T @ factor_b, retained


@torch.no_grad()
def _initialize_shared_stage_bank_from_affines(
    bank: AgeSpecificJBank,
    affines: dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    if bank.map_architecture != "shared_diagonal_stage_lora":
        raise TypeError("joint initializer requires shared_diagonal_stage_lora")
    mean_weight = torch.stack([affines[age][0].float() for age in ROLLBACK_SOURCE_AGES]).mean(0)
    mean_bias = torch.stack([affines[age][1].float() for age in ROLLBACK_SOURCE_AGES]).mean(0)
    diagonal = torch.diagonal(mean_weight)
    common_off_diagonal = mean_weight - torch.diag(diagonal)
    shared_a, shared_b, shared_retained = _factorize_low_rank(
        common_off_diagonal, rank=bank.rank, gauge_seed=seed + 11
    )
    bank.shared_diagonal_scale.copy_(diagonal)
    bank.shared_A.copy_(shared_a)
    bank.shared_B.copy_(shared_b)
    bank.shared_bias.copy_(mean_bias)
    shared_weight = torch.diag(diagonal) + shared_a @ shared_b
    rows = []
    for age in ROLLBACK_SOURCE_AGES:
        source_weight, source_bias = affines[age]
        stage_target = source_weight.float() - shared_weight
        stage_a, stage_b, stage_retained = _factorize_low_rank(
            stage_target,
            rank=bank.stage_rank,
            gauge_seed=seed + 1000 + age,
        )
        bank.stage_A[str(age)].copy_(stage_a)
        bank.stage_B[str(age)].copy_(stage_b)
        fitted_weight, fitted_bias = bank.affine(age)
        rows.append(
            {
                "source_age": age,
                "target_age": age - 1,
                "initialization": label,
                "shared_rank": bank.rank,
                "stage_rank": bank.stage_rank,
                "shared_off_diagonal_retained_energy": shared_retained,
                "stage_residual_retained_energy": stage_retained,
                "source_weight_relative_error": float(
                    (fitted_weight - source_weight.float()).norm()
                    / source_weight.float().norm().clamp_min(1e-12)
                ),
                "source_bias_relative_error": float(
                    (fitted_bias - source_bias.float()).norm()
                    / source_bias.float().norm().clamp_min(1e-12)
                ),
            }
        )
    return rows


@torch.no_grad()
def initialize_bank(
    *,
    bank: AgeSpecificJBank,
    model,
    cfg,
    phase_positions: list[int],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.initialization == "identity":
        return [
            {
                "source_age": age,
                "target_age": age - 1,
                "initialization": "identity",
                "retained_fit_energy": 0.0,
            }
            for age in ROLLBACK_SOURCE_AGES
        ]
    if args.initialization == "canonical_shared":
        if args.canonical_artifact is None or args.canonical_label is None:
            raise ValueError(
                "canonical_shared initialization requires --canonical-artifact "
                "and --canonical-label"
            )
        artifact_checkpoint, positions, modules, _ = load_task_lora_modules(
            args.canonical_artifact, device=device
        )
        if (
            artifact_checkpoint != str(args.checkpoint)
            and not args.allow_canonical_checkpoint_path_mismatch
        ):
            raise ValueError("canonical J and backbone checkpoint differ")
        if positions != tuple(range(cfg.seq_len)):
            raise ValueError("canonical J must act at every token position")
        canonical = modules[args.canonical_label]
        if not isinstance(canonical, DiagonalIdentityLoRAJ):
            raise TypeError("canonical initializer must be diagonal + LoRA + bias")
        if bank.map_architecture == "diagonal_lora" and canonical.rank != bank.rank:
            raise ValueError("canonical initializer rank differs from J-bank rank")
        canonical_weight, canonical_bias = _affine(canonical)
        if bank.map_architecture == "shared_diagonal_stage_lora":
            _initialize_shared_stage_bank_from_affines(
                bank,
                {
                    age: (canonical_weight, canonical_bias)
                    for age in ROLLBACK_SOURCE_AGES
                },
                seed=args.seed,
                label="compressed canonical shared CE J",
            )
        else:
            for age in ROLLBACK_SOURCE_AGES:
                module = bank.maps[str(age)]
                if isinstance(module, FullAffineJ):
                    module.weight.copy_(canonical_weight)
                    module.bias.copy_(canonical_bias)
                else:
                    module.load_state_dict(canonical.state_dict())
        heldout = _collect_aligned_age_states(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            device=device,
            examples=args.calibration_heldout_examples,
            batch_size=args.calibration_batch_size,
            seed=args.seed + 101,
        )
        rows = []
        for age in ROLLBACK_SOURCE_AGES:
            fitted_weight, fitted_bias = bank.affine(age)
            singular_values = torch.linalg.svdvals(fitted_weight)
            rows.append(
                {
                    "source_age": age,
                    "target_age": age - 1,
                    "initialization": "copied canonical long-rollout CE J",
                    "canonical_artifact": str(args.canonical_artifact),
                    "canonical_label": args.canonical_label,
                    "heldout_examples": args.calibration_heldout_examples,
                    **affine_metrics(
                        heldout[age], heldout[age - 1], fitted_weight, fitted_bias
                    ),
                    "spectral_norm": float(singular_values.max()),
                    "minimum_singular_value": float(singular_values.min()),
                    "spectral_radius": float(
                        torch.linalg.eigvals(fitted_weight).abs().max()
                    ),
                }
            )
        return rows
    if args.initialization == "age_specific_bank":
        if args.bank_init_artifact is None:
            raise ValueError(
                "age_specific_bank initialization requires --bank-init-artifact"
            )
        payload = torch.load(
            args.bank_init_artifact,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("checkpoint") != str(args.checkpoint):
            raise ValueError("source J bank and backbone checkpoint differ")
        source = AgeSpecificJBank(
            dimension=cfg.d_model,
            rank=int(payload["rank"]),
            stage_rank=int(payload.get("stage_rank", payload["rank"])),
            map_architecture=payload.get("map_architecture", "diagonal_lora"),
        )
        source.load_state_dict(payload["state_dict"])
        source = source.to(device).frozen()
        source_affines = {age: source.affine(age) for age in ROLLBACK_SOURCE_AGES}
        if (
            bank.map_architecture == source.map_architecture
            == "shared_diagonal_stage_lora"
            and bank.rank == source.rank
            and bank.stage_rank == source.stage_rank
        ):
            bank.load_state_dict(source.state_dict())
            initialization_label = "exact copy of matching final-CE J bank"
        elif bank.map_architecture == "shared_diagonal_stage_lora":
            _initialize_shared_stage_bank_from_affines(
                bank,
                source_affines,
                seed=args.seed,
                label="joint SVD compression of final-CE age-specific J",
            )
            initialization_label = "joint SVD compression of final-CE age-specific J"
        else:
            for age in ROLLBACK_SOURCE_AGES:
                source_weight, source_bias = source_affines[age]
                target = bank.maps[str(age)]
                if isinstance(target, FullAffineJ):
                    target.weight.copy_(source_weight)
                    target.bias.copy_(source_bias)
                else:
                    target.initialize_from_affine_svd(
                        VectorAffine(
                            weight=source_weight,
                            bias=source_bias,
                            update_rank=cfg.d_model,
                            fit_dimension=cfg.d_model,
                            retained_fit_energy=1.0,
                        ),
                        gauge_seed=args.seed + 1000 + age,
                        diagonal_scale_init="diagonal",
                    )
            initialization_label = "per-stage expansion/compression of final-CE J bank"
        heldout = _collect_aligned_age_states(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            device=device,
            examples=args.calibration_heldout_examples,
            batch_size=args.calibration_batch_size,
            seed=args.seed + 101,
        )
        rows = []
        for age in ROLLBACK_SOURCE_AGES:
            copied_weight, copied_bias = bank.affine(age)
            source_weight, source_bias = source.affine(age)
            rows.append(
                {
                    "source_age": age,
                    "target_age": age - 1,
                    "initialization": initialization_label,
                    "source_bank_artifact": str(args.bank_init_artifact),
                    "source_vs_expanded_weight_relative_error": float(
                        (copied_weight - source_weight).norm()
                        / source_weight.norm().clamp_min(1e-12)
                    ),
                    "source_vs_expanded_bias_relative_error": float(
                        (copied_bias - source_bias).norm()
                        / source_bias.norm().clamp_min(1e-12)
                    ),
                    "heldout_examples": args.calibration_heldout_examples,
                    **affine_metrics(
                        heldout[age], heldout[age - 1], copied_weight, copied_bias
                    ),
                }
            )
        return rows
    calibration = _collect_aligned_age_states(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        examples=args.calibration_examples,
        batch_size=args.calibration_batch_size,
        seed=args.seed + 100,
    )
    heldout = _collect_aligned_age_states(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        examples=args.calibration_heldout_examples,
        batch_size=args.calibration_batch_size,
        seed=args.seed + 101,
    )
    regression_affines = {
        age: fit_affine(
            calibration[age], calibration[age - 1], args.calibration_ridge
        )
        for age in ROLLBACK_SOURCE_AGES
    }
    if bank.map_architecture == "shared_diagonal_stage_lora":
        rows = _initialize_shared_stage_bank_from_affines(
            bank,
            regression_affines,
            seed=args.seed,
            label="joint SVD of same-current adjacent regressions",
        )
        for row in rows:
            age = int(row["source_age"])
            fitted_weight, fitted_bias = bank.affine(age)
            singular_values = torch.linalg.svdvals(fitted_weight)
            row.update(
                calibration_examples=args.calibration_examples,
                heldout_examples=args.calibration_heldout_examples,
                ridge=args.calibration_ridge,
                **affine_metrics(
                    heldout[age], heldout[age - 1], fitted_weight, fitted_bias
                ),
                spectral_norm=float(singular_values.max()),
                minimum_singular_value=float(singular_values.min()),
                spectral_radius=float(torch.linalg.eigvals(fitted_weight).abs().max()),
            )
        return rows
    rows: list[dict[str, Any]] = []
    for age in ROLLBACK_SOURCE_AGES:
        weight, bias = regression_affines[age]
        module = bank.maps[str(age)]
        if isinstance(module, FullAffineJ):
            module.weight.copy_(weight)
            module.bias.copy_(bias)
            retained = 1.0
        else:
            retained = module.initialize_from_affine_svd(
                VectorAffine(
                    weight=weight,
                    bias=bias,
                    update_rank=cfg.d_model,
                    fit_dimension=cfg.d_model,
                    retained_fit_energy=1.0,
                ),
                gauge_seed=args.seed + 1000 + age,
                diagonal_scale_init="diagonal",
            )
        fitted_weight, fitted_bias = _affine(module)
        singular_values = torch.linalg.svdvals(fitted_weight)
        eigen_radius = torch.linalg.eigvals(fitted_weight).abs().max()
        rows.append(
            {
                "source_age": age,
                "target_age": age - 1,
                "initialization": "same-current adjacent regression",
                "calibration_examples": args.calibration_examples,
                "heldout_examples": args.calibration_heldout_examples,
                "ridge": args.calibration_ridge,
                "retained_fit_energy": retained,
                **affine_metrics(
                    heldout[age], heldout[age - 1], fitted_weight, fitted_bias
                ),
                "spectral_norm": float(singular_values.max()),
                "minimum_singular_value": float(singular_values.min()),
                "spectral_radius": float(eigen_radius),
            }
        )
    return rows


def _run_mixed_trajectory(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    positions: tuple[int, ...],
    successors: torch.Tensor,
    endpoint: torch.Tensor,
    initial_state: torch.Tensor,
    trajectory: AgeTrajectory,
    condition: str,
    phase_positions: list[int],
    rollback_composition: str = "product",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return final state, logits, and the graph node represented at the end."""

    if condition not in {
        "learned",
        "exact",
        "identity",
        "wrong_stage",
        "reverse_stage",
        "shared_J8",
    }:
        raise ValueError(f"unknown condition: {condition}")
    differentiable_loop = run_one_loop.__wrapped__
    state = initial_state
    current = endpoint
    logical_age = trajectory.start_age
    jump = phase_positions[2] - phase_positions[1]
    if jump <= 0:
        raise ValueError("phase positions must advance with logical age")
    pending_rollback_ages: list[int] = []
    for action in trajectory.actions:
        if action == 1:
            if pending_rollback_ages:
                state = bank.rollback_composed(
                    state,
                    source_ages=pending_rollback_ages,
                    positions=positions,
                    composition=rollback_composition,
                )
                pending_rollback_ages = []
            step = differentiable_loop(
                model,
                state,
                loop_index=logical_age,
            )
            state = step.state
            current = advance_nodes(successors, current, steps=jump)
            logical_age += 1
        else:
            if condition == "learned":
                pending_rollback_ages.append(logical_age)
            elif condition == "wrong_stage":
                wrong_age = 2 + ((logical_age - 2 + 1) % 7)
                pending_rollback_ages.append(wrong_age)
            elif condition == "reverse_stage":
                wrong_age = 10 - logical_age
                pending_rollback_ages.append(wrong_age)
            elif condition == "shared_J8":
                pending_rollback_ages.append(8)
            elif condition == "exact":
                state = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=logical_age - 1,
                    phase_position=phase_positions[logical_age - 1],
                )
            logical_age -= 1
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("executor left H1..H8")
    if pending_rollback_ages:
        state = bank.rollback_composed(
            state,
            source_ages=pending_rollback_ages,
            positions=positions,
            composition=rollback_composition,
        )
    if logical_age != trajectory.end_age:
        raise RuntimeError("executor endpoint age mismatch")
    return state, logits_from_raw_state(model, state), current


def _save_artifact(
    path: Path,
    *,
    args: argparse.Namespace,
    cfg: Any,
    positions: tuple[int, ...],
    bank: AgeSpecificJBank,
    stages: Sequence[MixedStage],
    completed_stages: int,
    initialization_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "kind": "graph_path_age_specific_J_bank",
        "checkpoint": str(args.checkpoint),
        "phase_summary": str(args.phase_summary),
        "age_range": [MIN_AGE, MAX_AGE],
        "rollback_roles": {str(age): f"H{age}->H{age-1}" for age in ROLLBACK_SOURCE_AGES},
        "architecture": (
            "position-shared diagonal + rank-r LoRA + bias"
            if bank.map_architecture == "diagonal_lora"
            else (
                "shared diagonal + shared rank-r LoRA + per-stage rank-s LoRA + shared bias"
                if bank.map_architecture == "shared_diagonal_stage_lora"
                else "position-shared full affine (256x256 weight + bias)"
            )
        ),
        "map_architecture": bank.map_architecture,
        "rank": args.rank,
        "stage_rank": bank.stage_rank,
        "positions": positions,
        "training_loss": "one final task CE per complete mixed trajectory",
        "training_start_age": args.fixed_start_age,
        "rollback_composition": args.rollback_composition,
        "hidden_state_loss_weight": 0.0,
        "initialization": args.initialization,
        "curriculum": args.curriculum,
        "canonical_artifact": (
            str(args.canonical_artifact) if args.canonical_artifact is not None else None
        ),
        "canonical_label": args.canonical_label,
        "bank_init_artifact": (
            str(args.bank_init_artifact) if args.bank_init_artifact is not None else None
        ),
        "initialization_rows": initialization_rows,
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "parameter_count": bank.parameter_count,
        "stages": [asdict(stage) for stage in stages],
        "completed_stages": completed_stages,
        "state_dict": {key: value.detach().cpu() for key, value in bank.state_dict().items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_bank(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    phase_positions: list[int],
    positions: tuple[int, ...],
    stages: Sequence[MixedStage],
    device: torch.device,
    args: argparse.Namespace,
    initialization_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    bank.train()
    rows: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(stages, start=1):
        optimizer = _optimizer(bank, stage, args.diagonal_lr_multiplier)
        target_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        total_stage_steps = stage.rounds * stage.batches_per_round
        inverse_reuse_schedule = (
            _inverse_reuse_schedule(total_stage_steps, stage.data_seed)
            if args.curriculum == "inverse_reuse"
            else None
        )
        inverse_global_call_counts = {age: 0 for age in ROLLBACK_SOURCE_AGES}
        stage_step = 0
        for round_index in range(1, stage.rounds + 1):
            rng = np.random.default_rng(stage.data_seed + 1000 * round_index)
            round_loss = 0.0
            round_correct = 0
            round_examples = 0
            call_counts = {age: 0 for age in ROLLBACK_SOURCE_AGES}
            started = time.monotonic()
            for batch_index in range(stage.batches_per_round):
                stage_step += 1
                lr_scale = _learning_rate_scale(
                    step=stage_step,
                    total_steps=total_stage_steps,
                    schedule=args.lr_schedule,
                    warmup_fraction=args.warmup_fraction,
                    warmup_start_factor=args.warmup_start_factor,
                    decay_fraction=args.decay_fraction,
                    decay_end_factor=args.decay_end_factor,
                )
                for group, target_lr in zip(
                    optimizer.param_groups, target_lrs, strict=True
                ):
                    group["lr"] = target_lr * lr_scale
                sampling_stratum: tuple[int, int] | None = None
                if args.curriculum == "inverse_reuse":
                    assert inverse_reuse_schedule is not None
                    sampled_total_cap = inverse_reuse_schedule[stage_step - 1]
                    sampled_run_cap = int(rng.integers(1, sampled_total_cap + 1))
                    sampling_stratum = (sampled_total_cap, sampled_run_cap)
                    hard_boundary = sampled_run_cap == sampled_total_cap
                    mandatory = None
                    sampled_minimum_total = sampled_total_cap
                    sampled_required_run = sampled_run_cap
                elif args.curriculum == "balanced5":
                    if batch_index < len(BALANCED5_MIXED_STRATA):
                        sampling_stratum = BALANCED5_MIXED_STRATA[batch_index]
                        hard_boundary = False
                        mandatory = ROLLBACK_SOURCE_AGES[batch_index % 7]
                    else:
                        sampling_stratum = (5, 5)
                        hard_boundary = True
                        mandatory = None
                    sampled_total_cap, sampled_run_cap = sampling_stratum
                    sampled_minimum_total = sampled_total_cap
                    sampled_required_run = sampled_run_cap
                else:
                    hard_boundary = (
                        args.curriculum == "focused5" and batch_index % 14 >= 7
                    )
                    mandatory = (
                        None
                        if hard_boundary
                        else ROLLBACK_SOURCE_AGES[batch_index % 7]
                    )
                    sampled_total_cap = stage.max_total_backs
                    sampled_run_cap = stage.max_consecutive_backs
                    sampled_minimum_total = (
                        stage.max_total_backs if hard_boundary else None
                    )
                    sampled_required_run = (
                        stage.max_consecutive_backs if hard_boundary else 0
                    )
                use_replay = (
                    args.replay_max_backs is not None
                    and rng.random() < args.replay_fraction
                )
                sampled_max_total_backs = (
                    args.replay_max_backs if use_replay else sampled_total_cap
                )
                bridge_budget = (
                    {"max_extra_backs": sampled_max_total_backs}
                    if args.curriculum == "legacy_extra"
                    else {"max_total_backs": sampled_max_total_backs}
                )
                if args.curriculum == "inverse_reuse":
                    trajectory = _sample_inverse_reuse_balanced_trajectory(
                        rng=rng,
                        total_backs=sampled_total_cap,
                        rollback_run=sampled_run_cap,
                        cumulative_calls=inverse_global_call_counts,
                        fixed_start_age=args.fixed_start_age,
                    )
                else:
                    trajectory = sample_bounded_bridge(
                        rng=rng,
                        mandatory_rollback_source=mandatory,
                        max_consecutive_backs=sampled_run_cap,
                        minimum_total_backs=sampled_minimum_total,
                        required_consecutive_backs=sampled_required_run,
                        fixed_start_age=args.fixed_start_age,
                        **bridge_budget,
                    )
                for age in trajectory.rollback_sources:
                    call_counts[age] += 1
                    if args.curriculum == "inverse_reuse":
                        inverse_global_call_counts[age] += 1
                with torch.no_grad():
                    _, path_targets, successors, _ = fixed_depth_batch(
                        cfg,
                        stage.batch_size,
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
                optimizer.zero_grad(set_to_none=True)
                state, logits, target = _run_mixed_trajectory(
                    model=model,
                    cfg=cfg,
                    bank=bank,
                    positions=positions,
                    successors=successors,
                    endpoint=endpoint,
                    initial_state=initial_state,
                    trajectory=trajectory,
                    condition="learned",
                    phase_positions=phase_positions,
                    rollback_composition=args.rollback_composition,
                )
                del state
                loss = F.cross_entropy(logits.float(), target)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    bank.parameters(), args.grad_clip
                )
                optimizer.step()
                correct = int(logits.argmax(dim=-1).eq(target).sum())
                round_loss += float(loss.detach())
                round_correct += correct
                round_examples += stage.batch_size
                rows.append(
                    {
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "round": round_index,
                        "batch": batch_index,
                        "loss": float(loss.detach()),
                        "accuracy": correct / stage.batch_size,
                        "grad_norm": float(grad_norm),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "warmup_scale": lr_scale,
                        "lr_scale": lr_scale,
                        "lr_schedule": args.lr_schedule,
                        "replay_trajectory": use_replay,
                        "sampled_max_total_backs": sampled_max_total_backs,
                        "back_budget_kind": (
                            "extra_backs"
                            if args.curriculum == "legacy_extra"
                            else "total_backs"
                        ),
                        "start_age": trajectory.start_age,
                        "end_age": trajectory.end_age,
                        "extra_backs": trajectory.extra_backs,
                        "forward_count": trajectory.forward_count,
                        "back_count": trajectory.back_count,
                        "max_rollback_run": trajectory.max_rollback_run,
                        "max_rollback_run_cap": sampled_run_cap,
                        "hard_boundary": hard_boundary,
                        "minimum_total_backs": sampled_minimum_total,
                        "required_consecutive_backs": sampled_required_run,
                        "sampling_stratum": (
                            f"T{sampling_stratum[0]}_R{sampling_stratum[1]}"
                            if sampling_stratum is not None
                            else None
                        ),
                        "mandatory_J": mandatory,
                        "rollback_sources": ",".join(map(str, trajectory.rollback_sources)),
                        "age_path": _trajectory_text(trajectory),
                        "examples": stage.batch_size,
                    }
                )
            coverage = {age: count for age, count in call_counts.items() if count > 0}
            if len(coverage) != 7:
                raise RuntimeError(f"round did not cover every J: {coverage}")
            print(
                json.dumps(
                    {
                        "stage": stage.name,
                        "round": round_index,
                        "loss": round_loss / stage.batches_per_round,
                        "accuracy": round_correct / round_examples,
                        "J_calls": call_counts,
                        "seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _write_csv(args.out_dir / "training_trajectories.csv", rows)
        _save_artifact(
            args.out_dir / "age_specific_j_bank.pt",
            args=args,
            cfg=cfg,
            positions=positions,
            bank=bank,
            stages=stages,
            completed_stages=stage_index,
            initialization_rows=initialization_rows,
        )
        _save_artifact(
            args.out_dir / f"age_specific_j_bank_after_{stage.name}.pt",
            args=args,
            cfg=cfg,
            positions=positions,
            bank=bank,
            stages=stages,
            completed_stages=stage_index,
            initialization_rows=initialization_rows,
        )
    return rows


def _state_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = predicted.float().reshape(-1, predicted.shape[-1])
    t = target.float().reshape(-1, target.shape[-1])
    residual = p - t
    centered = t - t.mean(dim=0, keepdim=True)
    return {
        "relative_error": float(residual.norm() / t.norm().clamp_min(1e-12)),
        "r2": float(1.0 - residual.square().sum() / centered.square().sum().clamp_min(1e-12)),
        "mean_cosine": float(F.cosine_similarity(p, t, dim=-1).mean()),
    }


@torch.no_grad()
def evaluate_random_trajectories(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    phase_positions: list[int],
    positions: tuple[int, ...],
    device: torch.device,
    trajectories: int,
    batch_size: int,
    max_extra_backs: int | None,
    max_total_backs: int | None = None,
    max_consecutive_backs: int | None = None,
    minimum_total_backs: int | None = None,
    required_consecutive_backs: int = 0,
    schedule_mandatory_j: bool = True,
    seed: int,
    split: str,
    conditions: tuple[str, ...] = (
        "learned",
        "exact",
        "identity",
        "wrong_stage",
        "reverse_stage",
        "shared_J8",
    ),
    rollback_composition: str = "product",
    fixed_start_age: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Make both the sampled age paths and the random graph batches identical
    # when two independently saved J banks are compared on the same split.
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for trajectory_index in range(trajectories):
        trajectory = sample_bounded_bridge(
            rng=rng,
            max_extra_backs=max_extra_backs,
            max_total_backs=max_total_backs,
            minimum_total_backs=minimum_total_backs,
            mandatory_rollback_source=(
                ROLLBACK_SOURCE_AGES[trajectory_index % 7]
                if schedule_mandatory_j
                else None
            ),
            max_consecutive_backs=max_consecutive_backs,
            required_consecutive_backs=required_consecutive_backs,
            fixed_start_age=fixed_start_age,
        )
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
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
        for condition in conditions:
            _, logits, target = _run_mixed_trajectory(
                model=model,
                cfg=cfg,
                bank=bank,
                positions=positions,
                successors=successors,
                endpoint=endpoint,
                initial_state=initial_state,
                trajectory=trajectory,
                condition=condition,
                phase_positions=phase_positions,
                rollback_composition=rollback_composition,
            )
            rows.append(
                {
                    "split": split,
                    "trajectory": trajectory_index,
                    "condition": condition,
                    "start_age": trajectory.start_age,
                    "end_age": trajectory.end_age,
                    "extra_backs": trajectory.extra_backs,
                    "forward_count": trajectory.forward_count,
                    "back_count": trajectory.back_count,
                    "path_length": len(trajectory.actions),
                    "accuracy": float(logits.argmax(dim=-1).eq(target).float().mean()),
                    "rollback_sources": ",".join(map(str, trajectory.rollback_sources)),
                    "age_path": _trajectory_text(trajectory),
                    "examples": batch_size,
                }
            )
    summary: list[dict[str, Any]] = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summary.append(
            {
                "split": split,
                "condition": condition,
                "trajectories": len(selected),
                "examples_per_trajectory": batch_size,
                "accuracy_mean": float(np.mean([row["accuracy"] for row in selected])),
                "accuracy_min": float(np.min([row["accuracy"] for row in selected])),
            }
        )
    return rows, summary


def _affine(
    module: DiagonalIdentityLoRAJ | FullAffineJ,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(module, FullAffineJ):
        return module.weight.float(), module.bias.float()
    return (
        torch.diag(module.diagonal_scale.float()) + module.A.float() @ module.B.float(),
        module.bias.float(),
    )


def _compose_affines(
    affines: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    dimension: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.eye(dimension, device=device, dtype=torch.float32)
    bias = torch.zeros(dimension, device=device, dtype=torch.float32)
    for next_weight, next_bias in affines:
        bias = bias @ next_weight + next_bias
        weight = weight @ next_weight
    return weight, bias


@torch.no_grad()
def evaluate_products(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    phase_positions: list[int],
    positions: tuple[int, ...],
    device: torch.device,
    examples: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("composition examples must be divisible by batch size")
    set_seed(seed)
    storage: dict[tuple[int, int], dict[str, list[torch.Tensor] | int]] = {}
    for source_age in range(2, 9):
        for target_age in range(1, source_age):
            storage[(source_age, target_age)] = {
                "sequential": [], "one_shot": [], "target": [], "current": [],
            }
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        current = path_targets[:, cfg.max_depth - 1]
        for source_age in range(2, 9):
            source = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=source_age,
                phase_position=phase_positions[source_age],
            )
            sequential = source
            affines: list[tuple[torch.Tensor, torch.Tensor]] = []
            for target_age in range(source_age - 1, 0, -1):
                affines.append(bank.affine(target_age + 1))
                sequential = bank.rollback(
                    sequential,
                    source_age=target_age + 1,
                    positions=positions,
                )
                weight, bias = _compose_affines(
                    affines, dimension=cfg.d_model, device=device
                )
                one_shot = source.float() @ weight + bias
                exact = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=target_age,
                    phase_position=phase_positions[target_age],
                )
                slot = storage[(source_age, target_age)]
                slot["sequential"].append(sequential.detach().cpu())
                slot["one_shot"].append(one_shot.detach().cpu())
                slot["target"].append(exact.detach().cpu())
                slot["current"].append(current.detach().cpu())
    rows: list[dict[str, Any]] = []
    model_cpu = copy.deepcopy(model).cpu().eval()
    for (source_age, target_age), values in sorted(storage.items()):
        sequential = torch.cat(values["sequential"])
        one_shot = torch.cat(values["one_shot"])
        exact = torch.cat(values["target"])
        current = torch.cat(values["current"])
        metrics = _state_metrics(sequential, exact)
        seq_logits = logits_from_raw_state(model_cpu, sequential)
        exact_logits = logits_from_raw_state(model_cpu, exact)
        rows.append(
            {
                "source_age": source_age,
                "target_age": target_age,
                "rollback_steps": source_age - target_age,
                **metrics,
                "sequential_vs_product_relative_error": float(
                    (sequential - one_shot).norm() / sequential.norm().clamp_min(1e-12)
                ),
                "product_readout_accuracy": float(
                    seq_logits.argmax(dim=-1).eq(current).float().mean()
                ),
                "exact_target_readout_accuracy": float(
                    exact_logits.argmax(dim=-1).eq(current).float().mean()
                ),
                "examples": examples,
            }
        )
    return rows


def main(args: argparse.Namespace) -> None:
    if not 0.0 <= args.warmup_fraction <= 1.0:
        raise ValueError("warmup fraction must lie in [0, 1]")
    if not 0.0 <= args.warmup_start_factor <= 1.0:
        raise ValueError("warmup start factor must lie in [0, 1]")
    if args.stage_round_multiplier < 1:
        raise ValueError("stage round multiplier must be positive")
    if not 0.0 <= args.decay_fraction <= 1.0:
        raise ValueError("decay fraction must lie in [0, 1]")
    if args.warmup_fraction + args.decay_fraction > 1.0:
        raise ValueError("warmup and decay fractions cannot overlap")
    if not 0.0 <= args.decay_end_factor <= 1.0:
        raise ValueError("decay end factor must lie in [0, 1]")
    if not 0.0 <= args.replay_fraction <= 1.0:
        raise ValueError("replay fraction must lie in [0, 1]")
    if args.replay_fraction and (
        args.replay_max_backs is None or args.replay_max_backs < 1
    ):
        raise ValueError("positive --replay-max-backs is required with replay")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase_payload["trajectory_positions_including_initial"]
    ]
    if len(phase_positions) <= MAX_AGE:
        raise ValueError("phase summary does not cover H1..H8")
    positions = tuple(range(cfg.seq_len))
    bank = AgeSpecificJBank(
        dimension=cfg.d_model,
        rank=args.rank,
        stage_rank=args.stage_rank,
        map_architecture=args.map_architecture,
    ).to(device)
    initialization_rows = initialize_bank(
        bank=bank,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        args=args,
    )
    _write_csv(args.out_dir / "initialization_metrics.csv", initialization_rows)
    curriculum_stages = {
        "legacy_extra": DEFAULT_STAGES,
        "single_back": SINGLE_BACK_STAGES,
        "two_axis": TWO_AXIS_STAGES,
        "focused5": FOCUSED5_STAGES,
        "balanced5": BALANCED5_STAGES,
        "inverse_reuse": INVERSE_REUSE_STAGES,
    }
    stages = list(curriculum_stages[args.curriculum])
    if args.max_stages is not None:
        stages = stages[: args.max_stages]
    stages = [
        replace(
            stage,
            rounds=(
                min(stage.rounds, args.stage_round_limit)
                if args.stage_round_limit is not None
                else stage.rounds
            ) * args.stage_round_multiplier,
            batch_size=(
                min(stage.batch_size, args.batch_size_cap)
                if args.batch_size_cap is not None
                else stage.batch_size
            ),
            batches_per_round=(
                args.batches_per_round
                if args.batches_per_round is not None
                else stage.batches_per_round
            ),
            learning_rate=stage.learning_rate * args.learning_rate_multiplier,
        )
        for stage in stages
    ]
    if any(stage.batches_per_round < 7 for stage in stages):
        raise ValueError("at least seven batches per round are required for J coverage")
    if args.curriculum in {"focused5", "balanced5", "inverse_reuse"}:
        if args.replay_max_backs is not None or args.replay_fraction:
            raise ValueError("focused5 forbids replay outside its compact support")
        if any(
            stage.max_total_backs > 5 or stage.max_consecutive_backs > 5
            for stage in stages
        ):
            raise ValueError("focused5 must keep total and consecutive J counts <= 5")
        if args.curriculum == "focused5" and any(
            stage.batches_per_round % 14 for stage in stages
        ):
            raise ValueError(
                "focused5 needs multiples of 14 batches for balanced coverage/boundary data"
            )
        if args.curriculum == "balanced5" and any(
            stage.batches_per_round != 19 for stage in stages
        ):
            raise ValueError(
                "balanced5 needs exactly 19 batches per round: 14 strata + 5 boundary"
            )
        if args.curriculum == "inverse_reuse" and sum(
            stage.rounds * stage.batches_per_round for stage in stages
        ) != 10_000:
            raise ValueError("inverse_reuse curriculum must contain exactly 10,000 steps")
    training_rows = train_bank(
        model=model,
        cfg=cfg,
        bank=bank,
        phase_positions=phase_positions,
        positions=positions,
        stages=stages,
        device=device,
        args=args,
        initialization_rows=initialization_rows,
    )
    bank = bank.frozen()
    evaluation_rows: list[dict[str, Any]] = []
    evaluation_summary: list[dict[str, Any]] = []
    if args.curriculum == "single_back":
        evaluation_specs = (
            dict(split="single_back_heldout", max_extra=None, max_total=1,
                 max_run=1, minimum_total=1, required_run=1,
                 schedule_mandatory=True, seed_offset=1),
            dict(split="composed_unseen", max_extra=args.eval_unseen_max_backs,
                 max_total=None, max_run=None, minimum_total=None,
                 required_run=0, schedule_mandatory=True, seed_offset=2),
        )
    elif args.curriculum in {"focused5", "balanced5", "inverse_reuse"}:
        evaluation_specs = (
            dict(split="single_J", max_extra=None, max_total=1, max_run=1,
                 minimum_total=1, required_run=1, schedule_mandatory=True,
                 seed_offset=1),
            dict(split="focused_mixture_a", max_extra=None, max_total=5,
                 max_run=5, minimum_total=None, required_run=0,
                 schedule_mandatory=True, seed_offset=2),
            dict(split="focused_mixture_b", max_extra=None, max_total=5,
                 max_run=5, minimum_total=None, required_run=0,
                 schedule_mandatory=True, seed_offset=3),
            dict(split="hard_boundary_T05_R5", max_extra=None, max_total=5,
                 max_run=5, minimum_total=5, required_run=5,
                 schedule_mandatory=False, seed_offset=4),
        )
    else:
        evaluation_specs = (
            dict(split="train_like", max_extra=args.eval_train_max_backs,
                 max_total=None, max_run=None, minimum_total=None,
                 required_run=0, schedule_mandatory=True, seed_offset=1),
            dict(split="longer_unseen", max_extra=args.eval_unseen_max_backs,
                 max_total=None, max_run=None, minimum_total=None,
                 required_run=0, schedule_mandatory=True, seed_offset=2),
        )
    for spec in evaluation_specs:
        rows, summary = evaluate_random_trajectories(
            model=model,
            cfg=cfg,
            bank=bank,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            trajectories=args.eval_trajectories,
            batch_size=args.eval_batch_size,
            max_extra_backs=spec["max_extra"],
            max_total_backs=spec["max_total"],
            max_consecutive_backs=spec["max_run"],
            minimum_total_backs=spec["minimum_total"],
            required_consecutive_backs=spec["required_run"],
            schedule_mandatory_j=spec["schedule_mandatory"],
            seed=args.seed + spec["seed_offset"],
            split=spec["split"],
            rollback_composition=args.rollback_composition,
            fixed_start_age=args.fixed_start_age,
        )
        evaluation_rows.extend(rows)
        evaluation_summary.extend(summary)
    product_rows = evaluate_products(
        model=model,
        cfg=cfg,
        bank=bank,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        examples=args.composition_examples,
        batch_size=args.composition_batch_size,
        seed=args.seed + 3,
    )
    _write_csv(args.out_dir / "random_trajectory_evaluation.csv", evaluation_rows)
    _write_csv(args.out_dir / "random_trajectory_summary.csv", evaluation_summary)
    _write_csv(args.out_dir / "matrix_product_evaluation.csv", product_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "J_bank": {
            "count": 7,
            "roles": {str(age): f"H{age}->H{age-1}" for age in ROLLBACK_SOURCE_AGES},
            "rank": args.rank,
            "stage_rank": bank.stage_rank,
            "architecture": args.map_architecture,
            "positions": "all tokens, position-shared, no token mixing",
            "parameter_count": bank.parameter_count,
            "initialization": args.initialization,
        },
        "controller_training_loss": "final task CE only; hidden-state loss 0",
        "training_start_age": args.fixed_start_age,
        "rollback_composition": args.rollback_composition,
        "curriculum": args.curriculum,
        "lr_schedule": {
            "name": args.lr_schedule,
            "warmup_fraction": args.warmup_fraction,
            "warmup_start_factor": args.warmup_start_factor,
            "decay_fraction": args.decay_fraction,
            "decay_end_factor": args.decay_end_factor,
            "diagonal_lr_multiplier": args.diagonal_lr_multiplier,
            "stage_round_multiplier": args.stage_round_multiplier,
        },
        "training_trajectory_count": len(training_rows),
        "initialization_metrics": initialization_rows,
        "stages": [asdict(stage) for stage in stages],
        "random_trajectory_summary": evaluation_summary,
        "product_evaluation": product_rows,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
