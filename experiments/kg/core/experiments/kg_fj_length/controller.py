"""Shared residual interface controller and strict alternating execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import LoopedCompositionTransformer


@dataclass(frozen=True)
class ControllerConfig:
    d_model: int = 256
    hidden_width: int = 1024
    initial_scale: float = 1.0e-2
    architecture: str = "mlp"
    attention_heads: int = 8

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.hidden_width <= 0:
            raise ValueError("controller dimensions must be positive")
        if self.initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        if self.architecture not in {
            "affine",
            "mlp",
            "gated_rms_mlp",
            "attention",
        }:
            raise ValueError(
                "architecture must be affine, mlp, gated_rms_mlp, or attention"
            )
        if self.attention_heads <= 0 or self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")


class ResidualMLPJ(nn.Module):
    """Identity-near two-layer token-wise J with nonzero gradients at init."""

    def __init__(self, config: ControllerConfig, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.norm = nn.LayerNorm(config.d_model)
        self.input = nn.Linear(config.d_model, config.hidden_width)
        self.output = nn.Linear(config.hidden_width, config.d_model)
        self.activation = nn.GELU()
        if config.architecture == "gated_rms_mlp":
            self.residual_scale = nn.Parameter(torch.tensor(float(config.initial_scale)))
            self.output_norm: nn.Module = nn.RMSNorm(config.d_model)
        else:
            self.register_buffer("residual_scale", torch.tensor(float(config.initial_scale)))
            self.output_norm = nn.Identity()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.normal_(self.input.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.input.bias)
            nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        delta = self.output(self.activation(self.input(self.norm(state))))
        updated = state + self.residual_scale.to(dtype=state.dtype) * delta
        return self.output_norm(updated)


class AffineJ(nn.Module):
    """Exactly identity-initialized trainable affine interface."""

    def __init__(self, config: ControllerConfig, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.affine = nn.Linear(config.d_model, config.d_model)
        with torch.no_grad():
            self.affine.weight.copy_(torch.eye(config.d_model))
            self.affine.bias.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.affine(state)


class CausalAttentionJ(nn.Module):
    """Identity-near NoPE causal attention controller with shared parameters."""

    def __init__(self, config: ControllerConfig, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = seed
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, config.d_model),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(config.initial_scale)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.attention._reset_parameters()
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        length = state.shape[1]
        mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=state.device),
            diagonal=1,
        )
        normalized = self.attention_norm(state)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        delta = attended + self.mlp(self.mlp_norm(state + attended))
        return state + self.residual_scale.to(dtype=state.dtype) * delta


def build_controller(config: ControllerConfig, seed: int) -> nn.Module:
    if config.architecture == "affine":
        return AffineJ(config, seed)
    if config.architecture == "attention":
        return CausalAttentionJ(config, seed)
    return ResidualMLPJ(config, seed)


@dataclass(frozen=True)
class AlternatingTrace:
    logits: torch.Tensor
    post_f_states: tuple[torch.Tensor, ...]
    post_j_states: tuple[torch.Tensor, ...]


def run_alternating(
    model: LoopedCompositionTransformer,
    controller: nn.Module,
    tokens: torch.Tensor,
    calls: int,
    return_states: bool = False,
) -> torch.Tensor | AlternatingTrace:
    """Execute F once and then the same J,F pair exactly calls-1 times."""
    if calls <= 0:
        raise ValueError("calls must be positive")
    state, input_embedding = model.prepare_recurrence(tokens)
    post_f: list[torch.Tensor] = []
    post_j: list[torch.Tensor] = []
    for call_index in range(calls):
        recurrent_input = (
            state + input_embedding if input_embedding is not None else state
        )
        state = model.apply_f(recurrent_input)
        if return_states:
            post_f.append(state)
        if call_index < calls - 1:
            state = controller(state)
            if return_states:
                post_j.append(state)
    logits = model.readout(state)
    if return_states:
        return AlternatingTrace(logits, tuple(post_f), tuple(post_j))
    return logits
