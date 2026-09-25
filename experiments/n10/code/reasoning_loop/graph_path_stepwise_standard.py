from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from reasoning_loop.graph_path_loop import TransformerBlock


class MacroStepConfig(Protocol):
    node_count: int
    d_model: int
    n_layers: int
    max_loops: int
    vocab_size: int
    seq_len: int


class StandardMacroStepGraphPathTransformer(nn.Module):
    """Independent blocks grouped to match one shared-stack loop per readout."""

    def __init__(self, cfg: MacroStepConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks_per_macro_step = cfg.n_layers
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers * cfg.max_loops)]
        )
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward_all(
        self,
        tokens: torch.Tensor,
        *,
        max_loops: int | None = None,
        return_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        macro_steps = self.cfg.max_loops if max_loops is None else max_loops
        if not 1 <= macro_steps <= self.cfg.max_loops:
            raise ValueError(f"max_loops must be in [1, {self.cfg.max_loops}]")
        x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        logits_by_macro_step: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for macro_step in range(macro_steps):
            first_block = macro_step * self.blocks_per_macro_step
            last_block = first_block + self.blocks_per_macro_step
            for block in self.blocks[first_block:last_block]:
                x = block(x)
            answer_state = self.ln_final(x[:, -1, :])
            logits_by_macro_step.append(
                self.unembed(answer_state)[:, : self.cfg.node_count]
            )
            if return_states:
                states.append(answer_state)
        output = {"logits_by_loop": torch.stack(logits_by_macro_step, dim=1)}
        if return_states:
            output["states_by_loop"] = torch.stack(states, dim=1)
        return output

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_all(tokens)["logits_by_loop"][:, -1, :]
