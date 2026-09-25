from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.boolean_dag_data import BooleanDAGBatch, BooleanDAGConfig
from reasoning_loop.boolean_dag_model import GroupRMSNorm


@dataclass(frozen=True)
class OuroBooleanDAGConfig:
    d_model: int = 512
    n_heads: int = 4
    d_mlp: int = 1456
    n_layers: int = 12
    steps: int = 4
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_exit_gate: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_mlp < 1 or self.n_layers < 1 or self.steps < 1:
            raise ValueError("d_mlp, n_layers, and steps must be positive")

    @classmethod
    def forty_million(cls) -> "OuroBooleanDAGConfig":
        return cls()

    @classmethod
    def shallow_forty_million(cls) -> "OuroBooleanDAGConfig":
        return cls(d_model=896, n_heads=7, d_mlp=2384, n_layers=4)

    @classmethod
    def hundred_million(cls) -> "OuroBooleanDAGConfig":
        return cls(d_model=768, n_heads=6, d_mlp=2048, n_layers=14)

    @classmethod
    def shallow_hundred_million(cls) -> "OuroBooleanDAGConfig":
        return cls(d_model=1408, n_heads=11, d_mlp=3744, n_layers=4)


class OuroBooleanDAGBlock(nn.Module):
    def __init__(self, cfg: OuroBooleanDAGConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.attention_norms = nn.ModuleList(
            [GroupRMSNorm(cfg.d_model, groups=1, eps=cfg.rms_norm_eps) for _ in range(2)]
        )
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp_norms = nn.ModuleList(
            [GroupRMSNorm(cfg.d_model, groups=1, eps=cfg.rms_norm_eps) for _ in range(2)]
        )
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.down_proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, d_model = x.shape
        shape = (batch, nodes, self.n_heads, self.d_head)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, nodes, d_model)
        return self.o_proj(attended)

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attention_norms[0](hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.attention_norms[1](hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norms[0](hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_norms[1](hidden_states)
        return residual + hidden_states


class OuroBooleanDAG(nn.Module):
    def __init__(
        self,
        data_cfg: BooleanDAGConfig,
        model_cfg: OuroBooleanDAGConfig,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.id_embed = nn.Embedding(data_cfg.node_count + 1, model_cfg.d_model)
        self.self_id_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.left_parent_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.right_parent_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.kind_embed = nn.Embedding(
            5 if data_cfg.balanced_logic else 4,
            model_cfg.d_model,
        )
        self.state_embed = nn.Embedding(3, model_cfg.d_model)
        self.root_embed = nn.Embedding(2, model_cfg.d_model)
        self.blocks = nn.ModuleList(
            [OuroBooleanDAGBlock(model_cfg) for _ in range(model_cfg.n_layers)]
        )
        self.outer_norm = GroupRMSNorm(
            model_cfg.d_model,
            groups=1,
            eps=model_cfg.rms_norm_eps,
        )
        self.state_readout = nn.Linear(model_cfg.d_model, 3, bias=False)
        self.exit_gate = (
            nn.Linear(model_cfg.d_model, 1, bias=True)
            if model_cfg.use_exit_gate
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=self.model_cfg.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, batch: BooleanDAGBatch) -> torch.Tensor:
        return (
            self.self_id_proj(self.id_embed(batch.self_ids))
            + self.left_parent_proj(self.id_embed(batch.parent_ids[:, :, 0]))
            + self.right_parent_proj(self.id_embed(batch.parent_ids[:, :, 1]))
            + self.kind_embed(batch.kinds)
            + self.state_embed(batch.initial_states)
            + self.root_embed(batch.root_mask.long())
        )

    def apply_stack(self, state: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            state = block(state)
        return state

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
            raise ValueError("max_steps must be positive")
        state = self.encode(batch)
        logits: list[torch.Tensor] = []
        readout_states: list[torch.Tensor] = []
        incoming: list[torch.Tensor] = []
        pre_outer: list[torch.Tensor] = []
        recurrent: list[torch.Tensor] = []
        exit_lambdas: list[torch.Tensor] = []
        for _ in range(steps):
            if return_dynamics:
                incoming.append(state)
            before_norm = self.apply_stack(state)
            state = self.outer_norm(before_norm)
            logits.append(self.state_readout(state))
            if self.exit_gate is not None:
                exit_lambdas.append(torch.sigmoid(self.exit_gate(state)).squeeze(-1))
            if return_states:
                readout_states.append(state)
            if return_dynamics:
                pre_outer.append(before_norm)
                recurrent.append(state)
        output = {"logits_by_step": torch.stack(logits, dim=1)}
        if exit_lambdas:
            lambdas = torch.stack(exit_lambdas, dim=1)
            lambdas = torch.cat((lambdas[:, :-1], torch.ones_like(lambdas[:, -1:])), dim=1)
            survival = torch.cumprod(
                torch.cat((torch.ones_like(lambdas[:, :1]), 1.0 - lambdas[:, :-1]), dim=1),
                dim=1,
            )
            output["exit_lambdas"] = lambdas
            output["exit_probs"] = lambdas * survival
        if return_states:
            output["states_by_step"] = torch.stack(readout_states, dim=1)
        if return_dynamics:
            output["incoming_by_step"] = torch.stack(incoming, dim=1)
            output["pre_outer_by_step"] = torch.stack(pre_outer, dim=1)
            output["recurrent_states_by_step"] = torch.stack(recurrent, dim=1)
        return output

    def forward(self, batch: BooleanDAGBatch) -> torch.Tensor:
        return self.forward_all(batch)["logits_by_step"][:, -1]
