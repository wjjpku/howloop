from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.boolean_dag_data import BooleanDAGBatch, BooleanDAGConfig


@dataclass(frozen=True)
class BooleanDAGModelConfig:
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    steps: int = 4
    dropout: float = 0.0
    outer_norm_groups: int = 0
    inner_norm_groups: int = 0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        for name, groups in (
            ("outer_norm_groups", self.outer_norm_groups),
            ("inner_norm_groups", self.inner_norm_groups),
        ):
            if groups < 0 or (groups and self.d_model % groups):
                raise ValueError(f"{name} must be zero or divide d_model")
        if self.rms_norm_eps < 0:
            raise ValueError("rms_norm_eps must be nonnegative")


class GroupRMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        groups: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model < 1 or groups < 1 or groups > d_model or d_model % groups:
            raise ValueError("groups must be positive and divide d_model")
        if eps < 0:
            raise ValueError("eps must be nonnegative")
        self.d_model = d_model
        self.groups = groups
        self.group_size = d_model // groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.shape[-1]}")
        grouped = x.float().reshape(*x.shape[:-1], self.groups, self.group_size)
        inverse_rms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (grouped * inverse_rms).reshape_as(x).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=normalized.dtype)


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, cfg: BooleanDAGModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, node_count, d_model = x.shape
        qkv = self.qkv(x).view(batch, node_count, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, node_count, d_model)
        return self.out_proj(attended)


class BooleanDAGBlock(nn.Module):
    def __init__(self, cfg: BooleanDAGModelConfig) -> None:
        super().__init__()
        norm = lambda: (
            GroupRMSNorm(
                cfg.d_model,
                groups=cfg.inner_norm_groups,
                eps=cfg.rms_norm_eps,
            )
            if cfg.inner_norm_groups
            else nn.LayerNorm(cfg.d_model)
        )
        self.ln_1 = norm()
        self.attn = BidirectionalSelfAttention(cfg)
        self.ln_2 = norm()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _BooleanDAGTransformerBase(nn.Module):
    def __init__(
        self,
        data_cfg: BooleanDAGConfig,
        model_cfg: BooleanDAGModelConfig,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.id_embed = nn.Embedding(data_cfg.node_count + 1, model_cfg.d_model)
        self.self_id_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.left_parent_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.right_parent_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.kind_embed = nn.Embedding(4, model_cfg.d_model)
        self.state_embed = nn.Embedding(3, model_cfg.d_model)
        self.root_embed = nn.Embedding(2, model_cfg.d_model)
        self.outer_norm = (
            GroupRMSNorm(
                model_cfg.d_model,
                groups=model_cfg.outer_norm_groups,
                eps=model_cfg.rms_norm_eps,
            )
            if model_cfg.outer_norm_groups
            else nn.Identity()
        )
        self.ln_final = nn.LayerNorm(model_cfg.d_model)
        self.state_readout = nn.Linear(model_cfg.d_model, 3, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.id_embed,
            self.kind_embed,
            self.state_embed,
            self.root_embed,
        ):
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.normal_(self.state_readout.weight, std=0.02)

    def encode(self, batch: BooleanDAGBatch) -> torch.Tensor:
        self_id = self.self_id_proj(self.id_embed(batch.self_ids))
        left_parent = self.left_parent_proj(self.id_embed(batch.parent_ids[:, :, 0]))
        right_parent = self.right_parent_proj(self.id_embed(batch.parent_ids[:, :, 1]))
        return (
            self_id
            + left_parent
            + right_parent
            + self.kind_embed(batch.kinds)
            + self.state_embed(batch.initial_states)
            + self.root_embed(batch.root_mask.long())
        )

    def _readout(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.ln_final(x)
        return self.state_readout(state), state


class LoopedBooleanDAGTransformer(_BooleanDAGTransformerBase):
    def __init__(
        self,
        data_cfg: BooleanDAGConfig,
        model_cfg: BooleanDAGModelConfig,
    ) -> None:
        super().__init__(data_cfg, model_cfg)
        self.block = BooleanDAGBlock(model_cfg)

    def forward_all(
        self,
        batch: BooleanDAGBatch,
        *,
        max_steps: int | None = None,
        return_states: bool = False,
        return_dynamics: bool = False,
    ) -> dict[str, torch.Tensor]:
        steps = self.model_cfg.steps if max_steps is None else max_steps
        if steps < 1:
            raise ValueError("max_steps must be >= 1")
        x = self.encode(batch)
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        incoming_states: list[torch.Tensor] = []
        pre_outer_states: list[torch.Tensor] = []
        recurrent_states: list[torch.Tensor] = []
        for _ in range(steps):
            if return_dynamics:
                incoming_states.append(x)
            pre_outer = self.block(x)
            x = self.outer_norm(pre_outer)
            if return_dynamics:
                pre_outer_states.append(pre_outer)
                recurrent_states.append(x)
            step_logits, step_state = self._readout(x)
            logits.append(step_logits)
            if return_states:
                states.append(step_state)
        output = {"logits_by_step": torch.stack(logits, dim=1)}
        if return_states:
            output["states_by_step"] = torch.stack(states, dim=1)
        if return_dynamics:
            output["incoming_by_step"] = torch.stack(incoming_states, dim=1)
            output["pre_outer_by_step"] = torch.stack(pre_outer_states, dim=1)
            output["recurrent_states_by_step"] = torch.stack(recurrent_states, dim=1)
        return output

    def forward(self, batch: BooleanDAGBatch) -> torch.Tensor:
        return self.forward_all(batch)["logits_by_step"][:, -1]


class StandardBooleanDAGTransformer(_BooleanDAGTransformerBase):
    def __init__(
        self,
        data_cfg: BooleanDAGConfig,
        model_cfg: BooleanDAGModelConfig,
    ) -> None:
        super().__init__(data_cfg, model_cfg)
        self.blocks = nn.ModuleList(
            [BooleanDAGBlock(model_cfg) for _ in range(model_cfg.steps)]
        )

    def forward_all(
        self,
        batch: BooleanDAGBatch,
        *,
        max_steps: int | None = None,
        return_states: bool = False,
        return_dynamics: bool = False,
    ) -> dict[str, torch.Tensor]:
        steps = self.model_cfg.steps if max_steps is None else max_steps
        if not 1 <= steps <= len(self.blocks):
            raise ValueError(f"max_steps cannot exceed {len(self.blocks)} for a standard model")
        x = self.encode(batch)
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        incoming_states: list[torch.Tensor] = []
        pre_outer_states: list[torch.Tensor] = []
        recurrent_states: list[torch.Tensor] = []
        for block in self.blocks[:steps]:
            if return_dynamics:
                incoming_states.append(x)
            pre_outer = block(x)
            x = self.outer_norm(pre_outer)
            if return_dynamics:
                pre_outer_states.append(pre_outer)
                recurrent_states.append(x)
            step_logits, step_state = self._readout(x)
            logits.append(step_logits)
            if return_states:
                states.append(step_state)
        output = {"logits_by_step": torch.stack(logits, dim=1)}
        if return_states:
            output["states_by_step"] = torch.stack(states, dim=1)
        if return_dynamics:
            output["incoming_by_step"] = torch.stack(incoming_states, dim=1)
            output["pre_outer_by_step"] = torch.stack(pre_outer_states, dim=1)
            output["recurrent_states_by_step"] = torch.stack(recurrent_states, dim=1)
        return output

    def forward(self, batch: BooleanDAGBatch) -> torch.Tensor:
        return self.forward_all(batch)["logits_by_step"][:, -1]


class PeriodicBooleanDAGTransformer(_BooleanDAGTransformerBase):
    def __init__(
        self,
        data_cfg: BooleanDAGConfig,
        model_cfg: BooleanDAGModelConfig,
        *,
        period: int = 2,
    ) -> None:
        super().__init__(data_cfg, model_cfg)
        if not 1 <= period <= model_cfg.steps:
            raise ValueError("period must lie within 1 through the trained step count")
        self.period = period
        self.blocks = nn.ModuleList([BooleanDAGBlock(model_cfg) for _ in range(period)])

    def forward_all(
        self,
        batch: BooleanDAGBatch,
        *,
        max_steps: int | None = None,
        return_states: bool = False,
        return_dynamics: bool = False,
    ) -> dict[str, torch.Tensor]:
        steps = self.model_cfg.steps if max_steps is None else max_steps
        if steps < 1:
            raise ValueError("max_steps must be >= 1")
        x = self.encode(batch)
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        incoming_states: list[torch.Tensor] = []
        pre_outer_states: list[torch.Tensor] = []
        recurrent_states: list[torch.Tensor] = []
        for step in range(steps):
            if return_dynamics:
                incoming_states.append(x)
            pre_outer = self.blocks[step % self.period](x)
            x = self.outer_norm(pre_outer)
            if return_dynamics:
                pre_outer_states.append(pre_outer)
                recurrent_states.append(x)
            step_logits, step_state = self._readout(x)
            logits.append(step_logits)
            if return_states:
                states.append(step_state)
        output = {"logits_by_step": torch.stack(logits, dim=1)}
        if return_states:
            output["states_by_step"] = torch.stack(states, dim=1)
        if return_dynamics:
            output["incoming_by_step"] = torch.stack(incoming_states, dim=1)
            output["pre_outer_by_step"] = torch.stack(pre_outer_states, dim=1)
            output["recurrent_states_by_step"] = torch.stack(recurrent_states, dim=1)
        return output

    def forward(self, batch: BooleanDAGBatch) -> torch.Tensor:
        return self.forward_all(batch)["logits_by_step"][:, -1]


def build_boolean_dag_model(
    *,
    architecture: str,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
) -> _BooleanDAGTransformerBase:
    if architecture == "looped":
        return LoopedBooleanDAGTransformer(data_cfg, model_cfg)
    if architecture == "standard":
        return StandardBooleanDAGTransformer(data_cfg, model_cfg)
    if architecture == "periodic2":
        return PeriodicBooleanDAGTransformer(data_cfg, model_cfg, period=2)
    raise ValueError(f"unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
