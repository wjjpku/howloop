from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ShardedEvidenceConfig:
    evidence_count: int = 5
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    loops: int = 5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.evidence_count < 1 or self.evidence_count % 2 == 0:
            raise ValueError("evidence_count must be positive and odd")
        if self.loops < 1 or self.loops > self.evidence_count:
            raise ValueError("loops must be between 1 and evidence_count")
        if self.d_model < 1 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_mlp < 1:
            raise ValueError("d_mlp must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def majority_labels(bits: torch.Tensor) -> torch.Tensor:
    if bits.ndim != 2 or bits.shape[1] % 2 == 0:
        raise ValueError("bits must have shape [batch, odd_evidence_count]")
    if bits.dtype == torch.bool:
        count = bits.long().sum(dim=1)
    else:
        if not torch.all((bits == 0) | (bits == 1)):
            raise ValueError("bits must be binary")
        count = bits.sum(dim=1)
    return (count > bits.shape[1] // 2).long()


def make_batch(
    *,
    batch_size: int,
    evidence_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if evidence_count < 1 or evidence_count % 2 == 0:
        raise ValueError("evidence_count must be positive and odd")
    bits = torch.randint(
        0,
        2,
        (batch_size, evidence_count),
        device=device,
        generator=generator,
    )
    return bits, majority_labels(bits)


def make_visibility_indices(
    condition: str,
    *,
    batch_size: int,
    cfg: ShardedEvidenceConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if condition == "full":
        indices = torch.arange(cfg.evidence_count, device=device)
        return indices.view(1, 1, -1).expand(batch_size, cfg.loops, -1).clone()
    if condition in {"shard", "shard_reset"}:
        indices = torch.arange(cfg.loops, device=device)
        return indices.view(1, cfg.loops, 1).expand(batch_size, -1, -1).clone()
    if condition == "shard_shuffled":
        priorities = torch.rand(
            (batch_size, cfg.evidence_count),
            device=device,
            generator=generator,
        )
        order = priorities.argsort(dim=1)[:, : cfg.loops]
        return order.unsqueeze(-1)
    raise ValueError(f"unknown visibility condition: {condition}")


def hybrid_majority_target(
    donor_bits: torch.Tensor,
    receiver_bits: torch.Tensor,
    visited_indices: torch.Tensor,
) -> torch.Tensor:
    if donor_bits.shape != receiver_bits.shape or donor_bits.ndim != 2:
        raise ValueError("donor_bits and receiver_bits must have the same 2D shape")
    if visited_indices.ndim != 2 or visited_indices.shape[0] != donor_bits.shape[0]:
        raise ValueError("visited_indices must have shape [batch, visited_count]")
    hybrid = receiver_bits.clone()
    donor_values = donor_bits.gather(1, visited_indices)
    hybrid.scatter_(1, visited_indices, donor_values)
    return majority_labels(hybrid)


class SharedWorkspaceCell(nn.Module):
    def __init__(self, cfg: ShardedEvidenceConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )
        self.norm_mlp = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        workspace: torch.Tensor,
        visible_evidence: torch.Tensor,
    ) -> torch.Tensor:
        query = workspace.unsqueeze(1)
        key_value = torch.cat((query, visible_evidence), dim=1)
        update, _ = self.attention(query, key_value, key_value, need_weights=False)
        workspace = self.norm_attention(workspace + update.squeeze(1))
        return self.norm_mlp(workspace + self.mlp(workspace))


class ShardedEvidenceModel(nn.Module):
    def __init__(self, cfg: ShardedEvidenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.bit_embedding = nn.Embedding(2, cfg.d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(cfg.evidence_count, cfg.d_model)
        )
        self.initial_workspace = nn.Parameter(torch.empty(cfg.d_model))
        self.cell = SharedWorkspaceCell(cfg)
        self.readout_norm = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, 2)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.initial_workspace, std=0.02)

    def encode_evidence(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != self.cfg.evidence_count:
            raise ValueError(
                f"bits must have shape [batch, {self.cfg.evidence_count}]"
            )
        return self.bit_embedding(bits.long()) + self.position_embedding.unsqueeze(0)

    def forward(
        self,
        bits: torch.Tensor,
        visibility_indices: torch.Tensor,
        *,
        reset_between_loops: bool = False,
        initial_workspace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = bits.shape[0]
        if visibility_indices.ndim != 3 or visibility_indices.shape[:2] != (
            batch_size,
            self.cfg.loops,
        ):
            raise ValueError(
                "visibility_indices must have shape [batch, loops, visible_count]"
            )
        evidence = self.encode_evidence(bits)
        if initial_workspace is None:
            workspace = self.initial_workspace.unsqueeze(0).expand(batch_size, -1)
        else:
            if initial_workspace.shape != (batch_size, self.cfg.d_model):
                raise ValueError("initial_workspace has the wrong shape")
            workspace = initial_workspace
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for loop_index in range(self.cfg.loops):
            if reset_between_loops and loop_index > 0:
                workspace = self.initial_workspace.unsqueeze(0).expand(batch_size, -1)
            indices = visibility_indices[:, loop_index, :]
            gather_indices = indices.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
            visible = evidence.gather(1, gather_indices)
            workspace = self.cell(workspace, visible)
            states.append(workspace)
            logits.append(self.readout(self.readout_norm(workspace)))
        return torch.stack(logits, dim=1), torch.stack(states, dim=1)

    def continue_from_workspace(
        self,
        bits: torch.Tensor,
        visibility_indices: torch.Tensor,
        *,
        workspace: torch.Tensor,
        start_loop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Continue a visibility schedule from a causally supplied workspace."""

        batch_size = bits.shape[0]
        if not 0 <= start_loop < self.cfg.loops:
            raise ValueError("start_loop must index a remaining loop")
        if workspace.shape != (batch_size, self.cfg.d_model):
            raise ValueError("workspace has the wrong shape")
        if visibility_indices.ndim != 3 or visibility_indices.shape[:2] != (
            batch_size,
            self.cfg.loops,
        ):
            raise ValueError(
                "visibility_indices must have shape [batch, loops, visible_count]"
            )
        evidence = self.encode_evidence(bits)
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for loop_index in range(start_loop, self.cfg.loops):
            indices = visibility_indices[:, loop_index, :]
            gather_indices = indices.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
            visible = evidence.gather(1, gather_indices)
            workspace = self.cell(workspace, visible)
            states.append(workspace)
            logits.append(self.readout(self.readout_norm(workspace)))
        return torch.stack(logits, dim=1), torch.stack(states, dim=1)
