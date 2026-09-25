"""Causal input-once executor for variable-length KG composition."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as functional

from .data import KGLengthConfig


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    d_mlp: int = 1024
    physical_blocks: int = 2
    dropout: float = 0.0
    position_encoding: str = "sinusoidal"
    input_injection: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_mlp <= 0 or self.physical_blocks <= 0:
            raise ValueError("model dimensions and physical_blocks must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0,1]")
        if self.position_encoding not in {"sinusoidal", "none"}:
            raise ValueError("position_encoding must be sinusoidal or none")


def _sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    even_width = (width + 1) // 2
    frequencies = torch.exp(
        torch.arange(even_width, dtype=torch.float32) * (-math.log(10_000.0) / max(even_width - 1, 1))
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros((length, width), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles[:, : encoding[:, 0::2].shape[1]])
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding


def _position_buffer(config: ModelConfig, length: int) -> torch.Tensor:
    if config.position_encoding == "none":
        return torch.zeros((length, config.d_model), dtype=torch.float32)
    return _sinusoidal_positions(length, config.d_model)


class _CausalPreLNBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.GELU(),
            nn.Linear(config.d_mlp, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, state: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(state)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        state = state + attended
        return state + self.mlp(self.mlp_norm(state))


class LoopedCompositionTransformer(nn.Module):
    """Embed a variable-length causal query once and reuse the same F call."""

    def __init__(self, kg_config: KGLengthConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.kg_config = kg_config
        self.model_config = model_config
        self.token_embedding = nn.Embedding(kg_config.vocabulary_size, model_config.d_model)
        self.blocks = nn.ModuleList(
            _CausalPreLNBlock(model_config) for _ in range(model_config.physical_blocks)
        )
        self.final_norm = nn.LayerNorm(model_config.d_model)
        self.register_buffer(
            "position_encoding",
            _position_buffer(model_config, kg_config.max_length + 2),
            persistent=True,
        )

    def embed_once(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must be rank two")
        if not 3 <= tokens.shape[1] <= self.kg_config.max_length + 2:
            raise ValueError("token length is outside the registered composition range")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must use torch.long dtype")
        embedded = self.token_embedding(tokens)
        if self.model_config.position_encoding == "none":
            return embedded
        positions = self.position_encoding[: tokens.shape[1]].to(
            device=tokens.device, dtype=embedded.dtype
        )
        return embedded + positions.unsqueeze(0)

    def prepare_recurrence(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedded = self.embed_once(tokens)
        if self.model_config.input_injection:
            return torch.zeros_like(embedded), embedded
        return embedded, None

    def apply_f(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[-1] != self.model_config.d_model:
            raise ValueError("state has the wrong shape")
        length = state.shape[1]
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=state.device), diagonal=1
        )
        for block in self.blocks:
            state = block(state, causal_mask)
        return state

    def run_raw(
        self,
        state: torch.Tensor,
        calls: int,
        return_states: bool = False,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if calls <= 0:
            raise ValueError("calls must be positive")
        if self.model_config.input_injection and input_embedding is None:
            raise ValueError("input-injection model requires the original embedding")
        if not self.model_config.input_injection and input_embedding is not None:
            raise ValueError("input embedding supplied to an input-once model")
        states: list[torch.Tensor] = []
        for _ in range(calls):
            recurrent_input = (
                state + input_embedding
                if input_embedding is not None
                else state
            )
            state = self.apply_f(recurrent_input)
            if return_states:
                states.append(state)
        if return_states:
            return state, tuple(states)
        return state

    def readout(self, state: torch.Tensor, answer_index: int | None = None) -> torch.Tensor:
        if answer_index is None:
            answer_index = state.shape[1] - 1
        if not 0 <= answer_index < state.shape[1]:
            raise ValueError("answer_index is outside the sequence")
        normalized = self.final_norm(state[:, answer_index, :])
        entity_weight = self.token_embedding.weight[
            self.kg_config.entity_offset : self.kg_config.entity_offset
            + self.kg_config.entity_count
        ]
        return functional.linear(normalized, entity_weight)

    def forward(self, tokens: torch.Tensor, calls: int) -> torch.Tensor:
        state, input_embedding = self.prepare_recurrence(tokens)
        final = self.run_raw(state, calls, input_embedding=input_embedding)
        assert isinstance(final, torch.Tensor)
        return self.readout(final)
