from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from reasoning_loop.boolean_dag_data import UNKNOWN, BooleanDAGBatch, BooleanDAGConfig
from reasoning_loop.boolean_dag_model import (
    BooleanDAGBlock,
    BooleanDAGModelConfig,
    _BooleanDAGTransformerBase,
)


HELDOUT_PROGRAMS: tuple[tuple[int, ...], ...] = (
    (1, 4, 2),
    (2, 1, 4),
    (3, 4, 1),
    (4, 2, 3),
    (1, 3, 2, 4),
    (2, 4, 1, 3),
    (3, 1, 4, 2),
    (4, 3, 2, 1),
)


@dataclass(frozen=True)
class MacroStepProgram:
    increments: torch.Tensor
    active_mask: torch.Tensor
    cumulative_depths: torch.Tensor
    lengths: torch.Tensor
    total_depths: torch.Tensor

    def to(self, device: torch.device | str) -> MacroStepProgram:
        return MacroStepProgram(
            increments=self.increments.to(device),
            active_mask=self.active_mask.to(device),
            cumulative_depths=self.cumulative_depths.to(device),
            lengths=self.lengths.to(device),
            total_depths=self.total_depths.to(device),
        )


def depth_increment_bits(
    increments: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if increments.numel() == 0 or increments.min() < 1 or increments.max() > 4:
        raise ValueError("depth increments must lie in 1 through 4")
    zero_based = increments - 1
    return torch.stack((zero_based // 2, zero_based % 2), dim=-1).to(dtype=dtype)


def _heldout_rows(increments: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    heldout = torch.zeros(len(lengths), dtype=torch.bool, device=increments.device)
    for program in HELDOUT_PROGRAMS:
        if len(program) > increments.shape[1]:
            continue
        expected = torch.tensor(program, device=increments.device)
        heldout |= lengths.ge(len(program)) & increments[:, : len(program)].eq(
            expected
        ).all(dim=1)
    return heldout


def sample_macrostep_programs(
    *,
    batch_size: int,
    max_loops: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    exclude_holdout: bool = True,
) -> MacroStepProgram:
    if batch_size < 1 or max_loops < 1:
        raise ValueError("batch_size and max_loops must be positive")
    lengths = torch.randint(
        1,
        max_loops + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )
    increments = torch.randint(
        1,
        5,
        (batch_size, max_loops),
        device=device,
        generator=generator,
    )
    if exclude_holdout:
        rejected = _heldout_rows(increments, lengths)
        attempts = 0
        while rejected.any():
            attempts += 1
            if attempts > 128:
                raise RuntimeError("could not sample non-heldout macro-step programs")
            increments[rejected] = torch.randint(
                1,
                5,
                (int(rejected.sum()), max_loops),
                device=device,
                generator=generator,
            )
            rejected = _heldout_rows(increments, lengths)
    active_mask = torch.arange(max_loops, device=device)[None] < lengths[:, None]
    increments = torch.where(active_mask, increments, torch.ones_like(increments))
    cumulative_depths = (increments * active_mask.long()).cumsum(dim=1)
    total_depths = cumulative_depths.gather(1, (lengths - 1)[:, None]).squeeze(1)
    return MacroStepProgram(
        increments=increments,
        active_mask=active_mask,
        cumulative_depths=cumulative_depths,
        lengths=lengths,
        total_depths=total_depths,
    )


def _phase_balanced_increments(
    total_depths: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_loops: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample bounded compositions while keeping paired total depths equal."""
    batch_size = total_depths.shape[0]
    device = total_depths.device
    increments = torch.ones(batch_size, max_loops, dtype=torch.long, device=device)
    remaining = total_depths - lengths
    for position in range(max_loops):
        active = position < lengths
        slots_left = (lengths - position - 1).clamp_min(0)
        lower = (remaining - 3 * slots_left).clamp_min(0)
        upper = remaining.clamp_max(3)
        span = (upper - lower + 1).clamp_min(1)
        random_unit = torch.rand(batch_size, device=device, generator=generator)
        extra = lower + (random_unit * span).long().clamp_max(span - 1)
        extra = torch.where(active, extra, torch.zeros_like(extra))
        increments[:, position] = 1 + extra
        remaining = remaining - extra
    if remaining.ne(0).any():
        raise RuntimeError("phase-balanced composition failed to close")
    return increments


def sample_phase_balanced_macrostep_programs(
    *,
    batch_size: int,
    max_loops: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    exclude_holdout: bool = True,
) -> MacroStepProgram:
    """Pair equal semantic depths with different numbers of recurrent steps."""
    if batch_size < 2 or batch_size % 2 or max_loops < 1:
        raise ValueError("batch_size must be even and max_loops must be positive")
    pair_count = batch_size // 2
    max_total = min(12, 4 * max_loops)
    if max_total < 4:
        raise ValueError("max_loops must support a total depth of at least 4")

    pair_totals = torch.randint(
        4,
        max_total + 1,
        (pair_count,),
        device=device,
        generator=generator,
    )
    lower = (pair_totals + 3) // 4
    upper = pair_totals.clamp_max(max_loops)
    spans = upper - lower + 1
    first_lengths = lower + torch.floor(
        torch.rand(pair_count, device=device, generator=generator) * spans
    ).long().clamp_max(spans - 1)
    second_lengths = lower + (first_lengths - lower + 1).remainder(spans)
    total_depths = pair_totals.repeat_interleave(2)
    lengths = torch.stack((first_lengths, second_lengths), dim=1).reshape(-1)
    increments = _phase_balanced_increments(
        total_depths,
        lengths,
        max_loops=max_loops,
        generator=generator,
    )
    if exclude_holdout:
        for _ in range(128):
            rejected = _heldout_rows(increments, lengths)
            if not rejected.any():
                break
            increments[rejected] = _phase_balanced_increments(
                total_depths[rejected],
                lengths[rejected],
                max_loops=max_loops,
                generator=generator,
            )
        else:
            raise RuntimeError("could not sample non-heldout phase-balanced programs")
    active_mask = torch.arange(max_loops, device=device)[None] < lengths[:, None]
    increments = torch.where(active_mask, increments, torch.ones_like(increments))
    cumulative_depths = (increments * active_mask.long()).cumsum(dim=1)
    return MacroStepProgram(
        increments=increments,
        active_mask=active_mask,
        cumulative_depths=cumulative_depths,
        lengths=lengths,
        total_depths=total_depths,
    )


def macrostep_targets(
    batch: BooleanDAGBatch,
    cumulative_depths: torch.Tensor,
) -> torch.Tensor:
    if cumulative_depths.ndim != 2 or cumulative_depths.shape[0] != batch.batch_size:
        raise ValueError("cumulative_depths must have shape [batch, loop]")
    resolved = batch.levels[:, None, :] <= cumulative_depths[:, :, None]
    values = (batch.values + 1)[:, None, :].expand_as(resolved)
    return torch.where(resolved, values, torch.full_like(values, UNKNOWN))


def _stratified_group_mean(
    losses: torch.Tensor,
    kinds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    cells = []
    for kind in range(4):
        for target in range(3):
            selected = mask & kinds.eq(kind) & targets.eq(target)
            if selected.any():
                cells.append(losses[selected].mean())
    if not cells:
        raise ValueError("stratified group has no active cells")
    return torch.stack(cells).mean()


def macrostep_intermediate_loss(
    logits_by_loop: torch.Tensor,
    batch: BooleanDAGBatch,
    program: MacroStepProgram,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected_shape = (
        batch.batch_size,
        program.increments.shape[1],
        batch.node_count,
        3,
    )
    if logits_by_loop.shape != expected_shape:
        raise ValueError(f"logits_by_loop must have shape {expected_shape}")
    targets = macrostep_targets(batch, program.cumulative_depths)
    losses = F.cross_entropy(
        logits_by_loop.reshape(-1, 3),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    previous = torch.cat(
        (
            torch.zeros(
                batch.batch_size,
                1,
                dtype=program.cumulative_depths.dtype,
                device=program.cumulative_depths.device,
            ),
            program.cumulative_depths[:, :-1],
        ),
        dim=1,
    )
    step_losses = []
    group_losses: dict[str, list[torch.Tensor]] = {
        "frontier_ce": [],
        "persistence_ce": [],
        "unknown_ce": [],
    }
    for loop in range(program.increments.shape[1]):
        active = program.active_mask[:, loop, None]
        if not active.any():
            continue
        level = batch.levels
        frontier = (
            active
            & level.gt(previous[:, loop, None])
            & level.le(program.cumulative_depths[:, loop, None])
        )
        persistence = active & level.le(previous[:, loop, None])
        unknown = active & level.gt(program.cumulative_depths[:, loop, None])
        groups = []
        for name, mask in (
            ("frontier_ce", frontier),
            ("persistence_ce", persistence),
            ("unknown_ce", unknown),
        ):
            if mask.any():
                value = _stratified_group_mean(
                    losses[:, loop],
                    batch.kinds,
                    targets[:, loop],
                    mask,
                )
                groups.append(value)
                group_losses[name].append(value)
        step_losses.append(torch.stack(groups).mean())
    if not step_losses:
        raise ValueError("program contains no active loops")
    zero = logits_by_loop.new_zeros(())
    return torch.stack(step_losses).mean(), {
        name: torch.stack(values).mean() if values else zero
        for name, values in group_losses.items()
    }


class DepthConditionedBooleanDAGTransformer(_BooleanDAGTransformerBase):
    def __init__(
        self,
        data_cfg: BooleanDAGConfig,
        model_cfg: BooleanDAGModelConfig,
        *,
        use_instruction: bool,
    ) -> None:
        if model_cfg.d_model < 3:
            raise ValueError("d_model must leave at least one non-instruction dimension")
        super().__init__(data_cfg, model_cfg)
        self.use_instruction = use_instruction
        self.block = BooleanDAGBlock(model_cfg)

    def inject_depth_increment(
        self,
        state: torch.Tensor,
        increments: torch.Tensor,
    ) -> torch.Tensor:
        if increments.shape != (state.shape[0],):
            raise ValueError("increments must have shape [batch]")
        bits = depth_increment_bits(increments, dtype=state.dtype)
        if not self.use_instruction:
            bits = torch.zeros_like(bits)
        expanded = bits[:, None, :].expand(-1, state.shape[1], -1)
        return torch.cat((state[:, :, :-2], expanded), dim=-1)

    def apply_macro_step(
        self,
        state: torch.Tensor,
        increments: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(self.inject_depth_increment(state, increments))

    def forward_all(
        self,
        batch: BooleanDAGBatch,
        *,
        depth_increments: torch.Tensor,
        return_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        if depth_increments.ndim != 2 or depth_increments.shape[0] != batch.batch_size:
            raise ValueError("depth_increments must have shape [batch, loop]")
        state = self.encode(batch)
        logits_by_loop = []
        states_by_loop = []
        for loop in range(depth_increments.shape[1]):
            state = self.apply_macro_step(state, depth_increments[:, loop])
            logits, readout_state = self._readout(state)
            logits_by_loop.append(logits)
            if return_states:
                states_by_loop.append(readout_state)
        output = {"logits_by_loop": torch.stack(logits_by_loop, dim=1)}
        if return_states:
            output["states_by_loop"] = torch.stack(states_by_loop, dim=1)
        return output

    def forward(
        self,
        batch: BooleanDAGBatch,
        depth_increments: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_all(
            batch,
            depth_increments=depth_increments,
        )["logits_by_loop"][:, -1]
