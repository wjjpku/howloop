from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


VISIBILITY_CONDITIONS = (
    "full",
    "full_once",
    "sequential",
    "sequential_shuffled",
)


@dataclass(frozen=True)
class TriadicShortageConfig:
    p: int = 17
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 128
    loops: int = 6
    architecture: str = "looped"
    dropout: float = 0.0
    mode_count: int = 0

    def __post_init__(self) -> None:
        if self.p < 2:
            raise ValueError("p must be at least 2")
        if self.d_model < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_mlp < 1:
            raise ValueError("d_mlp must be positive")
        if self.loops < 1:
            raise ValueError("loops must be positive")
        if self.architecture not in {"looped", "unshared"}:
            raise ValueError("architecture must be 'looped' or 'unshared'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mode_count < 0:
            raise ValueError("mode_count must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TriadicShortageConfig":
        return cls(**data)


def all_triples(
    p: int = 17,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if p < 2:
        raise ValueError("p must be at least 2")
    values = torch.arange(p, dtype=torch.long)
    operands = torch.cartesian_prod(values, values, values)
    labels = operands.sum(dim=1).remainder(p)
    if device is not None:
        operands = operands.to(device)
        labels = labels.to(device)
    return operands, labels


def split_indices(
    n_examples: int,
    train_fraction: float,
    *,
    seed: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_examples < 2:
        raise ValueError("n_examples must be at least 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(n_examples, generator=generator)
    train_size = int(round(n_examples * train_fraction))
    train_size = max(1, min(train_size, n_examples - 1))
    train = permutation[:train_size]
    heldout = permutation[train_size:]
    if device is not None:
        train = train.to(device)
        heldout = heldout.to(device)
    return train, heldout


def make_visibility_mask(
    condition: str,
    batch_size: int,
    loops: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if condition not in VISIBILITY_CONDITIONS:
        raise ValueError(f"unknown visibility condition: {condition}")
    if batch_size < 1 or loops < 1:
        raise ValueError("batch_size and loops must be positive")
    mask = torch.zeros((batch_size, loops, 3), dtype=torch.bool, device=device)
    if condition == "full":
        mask.fill_(True)
        return mask
    if condition == "full_once":
        mask[:, 0, :] = True
        return mask
    if loops < 3:
        raise ValueError(f"{condition} requires at least three loops")
    if condition == "sequential":
        operand_index = torch.arange(3, device=device)
        mask[:, operand_index, operand_index] = True
        return mask
    priorities = torch.rand(
        (batch_size, 3),
        device=device,
        generator=generator,
    )
    order = priorities.argsort(dim=1)
    batch_index = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, 3)
    loop_index = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, -1)
    mask[batch_index, loop_index, order] = True
    return mask


def hybrid_sum_target(
    donor_operands: torch.Tensor,
    receiver_operands: torch.Tensor,
    visited_mask: torch.Tensor,
    p: int = 17,
) -> torch.Tensor:
    if donor_operands.shape != receiver_operands.shape:
        raise ValueError("donor and receiver operands must have the same shape")
    if donor_operands.ndim != 2 or donor_operands.shape[1] != 3:
        raise ValueError("operands must have shape [batch, 3]")
    if visited_mask.shape != donor_operands.shape or visited_mask.dtype != torch.bool:
        raise ValueError("visited_mask must be boolean with shape [batch, 3]")
    hybrid = torch.where(visited_mask, donor_operands, receiver_operands)
    return hybrid.sum(dim=1).remainder(p)


class StaticOperandWorkspaceCell(nn.Module):
    def __init__(self, cfg: TriadicShortageConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.norm_attention = nn.LayerNorm(cfg.d_model)
        self.attention = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )

    def forward(
        self,
        workspace: torch.Tensor,
        static_operands: torch.Tensor,
        visible_operands: torch.Tensor,
        *,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if workspace.ndim != 2:
            raise ValueError("workspace must have shape [batch, d_model]")
        if static_operands.shape != (workspace.shape[0], 3, workspace.shape[1]):
            raise ValueError("static_operands has the wrong shape")
        if visible_operands.shape != (workspace.shape[0], 3):
            raise ValueError("visible_operands must have shape [batch, 3]")
        query = self.norm_attention(workspace).unsqueeze(1)
        operand_values = self.norm_attention(static_operands)
        key_value = torch.cat((query, operand_values), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    (workspace.shape[0], 1),
                    dtype=torch.bool,
                    device=workspace.device,
                ),
                ~visible_operands,
            ),
            dim=1,
        )
        attention_update, attention_weights = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=padding_mask,
            need_weights=return_cache,
            average_attn_weights=False,
        )
        attention_update = attention_update.squeeze(1)
        residual_mid = workspace + attention_update
        mlp_update = self.mlp(self.norm_mlp(residual_mid))
        output = residual_mid + mlp_update
        cache: dict[str, torch.Tensor] = {}
        if return_cache:
            cache = {
                "workspace_in": workspace.detach(),
                "attention_out": attention_update.detach(),
                "attention_weights": attention_weights.detach(),
                "residual_mid": residual_mid.detach(),
                "mlp_out": mlp_update.detach(),
                "workspace_out": output.detach(),
            }
        return output, cache


class TriadicShortageModel(nn.Module):
    def __init__(self, cfg: TriadicShortageConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.value_embedding = nn.Embedding(cfg.p, cfg.d_model)
        self.operand_position_embedding = nn.Parameter(torch.empty(3, cfg.d_model))
        self.initial_workspace = nn.Parameter(torch.empty(cfg.d_model))
        self.schedule_mode_embedding = (
            nn.Embedding(cfg.mode_count, cfg.d_model)
            if cfg.mode_count > 0
            else None
        )
        if cfg.architecture == "looped":
            self.shared_cell = StaticOperandWorkspaceCell(cfg)
            self.cells = None
        else:
            self.shared_cell = None
            self.cells = nn.ModuleList(
                StaticOperandWorkspaceCell(cfg) for _ in range(cfg.loops)
            )
        self.readout_norm = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, cfg.p, bias=False)
        nn.init.normal_(self.operand_position_embedding, std=0.02)
        nn.init.normal_(self.initial_workspace, std=0.02)
        if self.schedule_mode_embedding is not None:
            nn.init.normal_(self.schedule_mode_embedding.weight, std=0.02)

    def encode_operands(self, operands: torch.Tensor) -> torch.Tensor:
        if operands.ndim != 2 or operands.shape[1] != 3:
            raise ValueError("operands must have shape [batch, 3]")
        if operands.numel() and (operands.min() < 0 or operands.max() >= self.cfg.p):
            raise ValueError("operand values must lie in [0, p)")
        return self.value_embedding(operands.long()) + self.operand_position_embedding.unsqueeze(0)

    def cell_for_loop(self, loop_index: int) -> StaticOperandWorkspaceCell:
        if not 0 <= loop_index < self.cfg.loops:
            raise ValueError("loop_index is out of range")
        if self.shared_cell is not None:
            return self.shared_cell
        if self.cells is None:
            raise RuntimeError("unshared cells are unavailable")
        return self.cells[loop_index]

    def _initial_workspace(
        self,
        batch_size: int,
        mode_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        workspace = self.initial_workspace.unsqueeze(0).expand(batch_size, -1)
        if self.schedule_mode_embedding is None:
            if mode_ids is not None:
                raise ValueError("mode_ids require a mode-conditioned model")
            return workspace
        if mode_ids is None:
            raise ValueError("mode_ids are required for a mode-conditioned model")
        mode_ids = mode_ids.to(device=workspace.device, dtype=torch.long)
        if mode_ids.shape != (batch_size,):
            raise ValueError("mode_ids must have shape [batch]")
        if mode_ids.numel() and (
            mode_ids.min() < 0 or mode_ids.max() >= self.cfg.mode_count
        ):
            raise ValueError("mode_ids must lie in the configured range")
        return workspace + self.schedule_mode_embedding(mode_ids)

    def _reset_mask(
        self,
        reset_before: torch.Tensor | None,
        loop_index: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if reset_before is None:
            return None
        reset_before = reset_before.to(device=device, dtype=torch.bool)
        if reset_before.ndim == 1:
            if reset_before.shape[0] < self.cfg.loops:
                raise ValueError("reset_before is shorter than configured loops")
            return reset_before[loop_index].expand(batch_size)
        if reset_before.ndim == 2 and reset_before.shape == (batch_size, self.cfg.loops):
            return reset_before[:, loop_index]
        raise ValueError("reset_before must have shape [loops] or [batch, loops]")

    def _run(
        self,
        operands: torch.Tensor,
        visibility: torch.Tensor,
        *,
        workspace: torch.Tensor,
        reset_workspace: torch.Tensor,
        start_loop: int,
        stop_loop: int,
        reset_before: torch.Tensor | None,
        return_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        static_operands = self.encode_operands(operands)
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}
        for loop_index in range(start_loop, stop_loop):
            reset_mask = self._reset_mask(
                reset_before,
                loop_index,
                operands.shape[0],
                operands.device,
            )
            if reset_mask is not None and reset_mask.any():
                workspace = torch.where(
                    reset_mask.unsqueeze(1),
                    reset_workspace,
                    workspace,
                )
            workspace, loop_cache = self.cell_for_loop(loop_index)(
                workspace,
                static_operands,
                visibility[:, loop_index],
                return_cache=return_cache,
            )
            states.append(workspace)
            logits.append(self.readout(self.readout_norm(workspace)))
            if return_cache:
                for key, value in loop_cache.items():
                    cache[f"loop{loop_index}.{key}"] = value
        return torch.stack(logits, dim=1), torch.stack(states, dim=1), cache

    def forward(
        self,
        operands: torch.Tensor,
        visibility: torch.Tensor,
        *,
        reset_before: torch.Tensor | None = None,
        initial_workspace: torch.Tensor | None = None,
        mode_ids: torch.Tensor | None = None,
        active_loops: int | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size = operands.shape[0]
        stop_loop = self.cfg.loops if active_loops is None else active_loops
        if not 1 <= stop_loop <= self.cfg.loops:
            raise ValueError("active_loops must be in [1, configured loops]")
        if visibility.shape != (batch_size, self.cfg.loops, 3):
            raise ValueError("visibility must have shape [batch, configured loops, 3]")
        reset_workspace = self._initial_workspace(batch_size, mode_ids=mode_ids)
        if initial_workspace is None:
            workspace = reset_workspace
        else:
            if initial_workspace.shape != (batch_size, self.cfg.d_model):
                raise ValueError("initial_workspace has the wrong shape")
            workspace = initial_workspace
        logits, states, cache = self._run(
            operands,
            visibility,
            workspace=workspace,
            reset_workspace=reset_workspace,
            start_loop=0,
            stop_loop=stop_loop,
            reset_before=reset_before,
            return_cache=return_cache,
        )
        if return_cache:
            return logits, states, cache
        return logits, states

    def continue_from_workspace(
        self,
        operands: torch.Tensor,
        visibility: torch.Tensor,
        *,
        workspace: torch.Tensor,
        start_loop: int,
        stop_loop: int | None = None,
        reset_before: torch.Tensor | None = None,
        mode_ids: torch.Tensor | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if not 0 <= start_loop < self.cfg.loops:
            raise ValueError("start_loop must index a configured loop")
        final_loop = self.cfg.loops if stop_loop is None else stop_loop
        if not start_loop < final_loop <= self.cfg.loops:
            raise ValueError("stop_loop must be after start_loop and within configured loops")
        if workspace.shape != (operands.shape[0], self.cfg.d_model):
            raise ValueError("workspace has the wrong shape")
        if visibility.shape != (operands.shape[0], self.cfg.loops, 3):
            raise ValueError("visibility must have shape [batch, configured loops, 3]")
        reset_workspace = self._initial_workspace(
            operands.shape[0],
            mode_ids=mode_ids,
        )
        logits, states, cache = self._run(
            operands,
            visibility,
            workspace=workspace,
            reset_workspace=reset_workspace,
            start_loop=start_loop,
            stop_loop=final_loop,
            reset_before=reset_before,
            return_cache=return_cache,
        )
        if return_cache:
            return logits, states, cache
        return logits, states
