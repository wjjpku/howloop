from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TypedRelationConfig:
    node_count: int = 16
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    loops: int = 3
    architecture: str = "looped"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ValueError("node_count must be at least 2")
        if self.d_model < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_mlp < 1:
            raise ValueError("d_mlp must be positive")
        if self.loops < 1:
            raise ValueError("loops must be positive")
        if self.architecture not in {"looped", "unshared"}:
            raise ValueError("architecture must be looped or unshared")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedRelationConfig":
        return cls(**data)


@dataclass(frozen=True)
class RelationBatch:
    f: torch.Tensor
    g: torch.Tensor
    query: torch.Tensor
    targets: torch.Tensor
    composition_order: str = "f_then_g"

    @property
    def batch_size(self) -> int:
        return int(self.query.shape[0])


def _random_permutations(
    batch_size: int,
    node_count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    priorities = torch.rand(
        (batch_size, node_count),
        device=device,
        generator=generator,
    )
    return priorities.argsort(dim=1)


def make_relation_batch(
    *,
    batch_size: int,
    node_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    composition_order: str = "f_then_g",
) -> RelationBatch:
    if batch_size < 1 or node_count < 2:
        raise ValueError("batch_size must be positive and node_count at least 2")
    if composition_order not in {"f_then_g", "g_then_f"}:
        raise ValueError("composition_order must be f_then_g or g_then_f")
    f = _random_permutations(
        batch_size,
        node_count,
        device=device,
        generator=generator,
    )
    g = _random_permutations(
        batch_size,
        node_count,
        device=device,
        generator=generator,
    )
    query = torch.randint(
        0,
        node_count,
        (batch_size,),
        device=device,
        generator=generator,
    )
    first_mapping, second_mapping = (
        (f, g) if composition_order == "f_then_g" else (g, f)
    )
    first = first_mapping.gather(1, query[:, None]).squeeze(1)
    endpoint = second_mapping.gather(1, first[:, None]).squeeze(1)
    return RelationBatch(
        f=f,
        g=g,
        query=query,
        targets=torch.stack((first, endpoint), dim=1),
        composition_order=composition_order,
    )


def relation_visibility(
    condition: str,
    *,
    batch_size: int,
    loops: int,
    node_count: int,
    device: torch.device,
    composition_order: str = "f_then_g",
) -> torch.Tensor:
    if condition not in {"full", "aligned", "swapped", "f_only", "g_only"}:
        raise ValueError(f"unknown relation visibility condition: {condition}")
    if batch_size < 1 or loops < 1 or node_count < 2:
        raise ValueError("batch, loop, and node sizes must be positive")
    if composition_order not in {"f_then_g", "g_then_f"}:
        raise ValueError("composition_order must be f_then_g or g_then_f")
    mask = torch.zeros(
        (batch_size, loops, 2 * node_count),
        dtype=torch.bool,
        device=device,
    )
    if condition == "full":
        mask.fill_(True)
    elif condition == "f_only":
        mask[:, :, :node_count] = True
    elif condition == "g_only":
        mask[:, :, node_count:] = True
    else:
        aligned = condition == "aligned"
        first_is_f = composition_order == "f_then_g"
        expose_f_first = first_is_f == aligned
        first_slice = slice(0, node_count) if expose_f_first else slice(node_count, None)
        second_slice = slice(node_count, None) if expose_f_first else slice(0, node_count)
        mask[:, 0, first_slice] = True
        if loops >= 2:
            mask[:, 1, second_slice] = True
    return mask


def stage_accuracy_matrix(
    logits_by_loop: torch.Tensor,
    targets_by_stage: torch.Tensor,
) -> torch.Tensor:
    if logits_by_loop.ndim != 3:
        raise ValueError("logits_by_loop must have shape [batch, loops, nodes]")
    if targets_by_stage.ndim != 2 or targets_by_stage.shape[1] != 2:
        raise ValueError("targets_by_stage must have shape [batch, 2]")
    if logits_by_loop.shape[0] != targets_by_stage.shape[0]:
        raise ValueError("logits and targets must share the batch axis")
    predictions = logits_by_loop.argmax(dim=-1)
    return (
        predictions.unsqueeze(2)
        .eq(targets_by_stage.unsqueeze(1))
        .float()
        .mean(dim=0)
    )


class TypedRelationWorkspaceCell(nn.Module):
    def __init__(self, cfg: TypedRelationConfig) -> None:
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
        static_edges: torch.Tensor,
        visible_edges: torch.Tensor,
        *,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if workspace.ndim != 2:
            raise ValueError("workspace must have shape [batch, d_model]")
        if (
            static_edges.ndim != 3
            or static_edges.shape[0] != workspace.shape[0]
            or static_edges.shape[2] != workspace.shape[1]
        ):
            raise ValueError("static_edges has the wrong shape")
        if visible_edges.shape != static_edges.shape[:2]:
            raise ValueError("visible_edges has the wrong shape")
        query = self.norm_attention(workspace).unsqueeze(1)
        edge_values = self.norm_attention(static_edges)
        key_value = torch.cat((query, edge_values), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    (workspace.shape[0], 1),
                    dtype=torch.bool,
                    device=workspace.device,
                ),
                ~visible_edges,
            ),
            dim=1,
        )
        attention_update, weights = self.attention(
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
                "attention_weights": weights.detach(),
                "residual_mid": residual_mid.detach(),
                "mlp_out": mlp_update.detach(),
                "workspace_out": output.detach(),
            }
        return output, cache


class TypedRelationModel(nn.Module):
    def __init__(self, cfg: TypedRelationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.source_embedding = nn.Embedding(cfg.node_count, cfg.d_model)
        self.destination_embedding = nn.Embedding(cfg.node_count, cfg.d_model)
        self.query_embedding = nn.Embedding(cfg.node_count, cfg.d_model)
        self.relation_embedding = nn.Parameter(torch.empty(2, cfg.d_model))
        self.initial_workspace = nn.Parameter(torch.empty(cfg.d_model))
        if cfg.architecture == "looped":
            self.shared_cell = TypedRelationWorkspaceCell(cfg)
            self.cells = None
        else:
            self.shared_cell = None
            self.cells = nn.ModuleList(
                TypedRelationWorkspaceCell(cfg) for _ in range(cfg.loops)
            )
        self.readout_norm = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, cfg.node_count, bias=False)
        nn.init.normal_(self.relation_embedding, std=0.02)
        nn.init.normal_(self.initial_workspace, std=0.02)

    def cell_for_loop(self, loop_index: int) -> TypedRelationWorkspaceCell:
        if not 0 <= loop_index < self.cfg.loops:
            raise ValueError("loop_index is out of range")
        if self.shared_cell is not None:
            return self.shared_cell
        if self.cells is None:
            raise RuntimeError("unshared cells are unavailable")
        return self.cells[loop_index]

    def encode_edges(self, batch: RelationBatch) -> torch.Tensor:
        if batch.f.shape != (batch.batch_size, self.cfg.node_count):
            raise ValueError("f has the wrong shape")
        if batch.g.shape != batch.f.shape:
            raise ValueError("g has the wrong shape")
        source = torch.arange(
            self.cfg.node_count,
            device=batch.query.device,
        ).unsqueeze(0).expand(batch.batch_size, -1)
        source_part = self.source_embedding(source)
        f_edges = (
            source_part
            + self.destination_embedding(batch.f)
            + self.relation_embedding[0]
        )
        g_edges = (
            source_part
            + self.destination_embedding(batch.g)
            + self.relation_embedding[1]
        )
        return torch.cat((f_edges, g_edges), dim=1)

    def _initial_workspace(self, query: torch.Tensor) -> torch.Tensor:
        if query.shape != (query.shape[0],):
            raise ValueError("query must be one-dimensional")
        return self.initial_workspace.unsqueeze(0) + self.query_embedding(query)

    def _run(
        self,
        batch: RelationBatch,
        visibility: torch.Tensor,
        *,
        workspace: torch.Tensor,
        start_loop: int,
        stop_loop: int,
        return_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        static_edges = self.encode_edges(batch)
        logits: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}
        for loop_index in range(start_loop, stop_loop):
            workspace, loop_cache = self.cell_for_loop(loop_index)(
                workspace,
                static_edges,
                visibility[:, loop_index],
                return_cache=return_cache,
            )
            states.append(workspace)
            logits.append(self.readout(self.readout_norm(workspace)))
            if return_cache:
                for name, value in loop_cache.items():
                    cache[f"loop{loop_index}.{name}"] = value
        return torch.stack(logits, dim=1), torch.stack(states, dim=1), cache

    def forward(
        self,
        batch: RelationBatch,
        visibility: torch.Tensor,
        *,
        active_loops: int | None = None,
        return_cache: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        stop_loop = self.cfg.loops if active_loops is None else active_loops
        if not 1 <= stop_loop <= self.cfg.loops:
            raise ValueError("active_loops must be within configured loops")
        expected = (batch.batch_size, self.cfg.loops, 2 * self.cfg.node_count)
        if visibility.shape != expected:
            raise ValueError("visibility has the wrong shape")
        result = self._run(
            batch,
            visibility,
            workspace=self._initial_workspace(batch.query),
            start_loop=0,
            stop_loop=stop_loop,
            return_cache=return_cache,
        )
        return result if return_cache else result[:2]

    def continue_from_workspace(
        self,
        batch: RelationBatch,
        visibility: torch.Tensor,
        *,
        workspace: torch.Tensor,
        start_loop: int,
        stop_loop: int,
        return_cache: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        if not 0 < start_loop < stop_loop <= self.cfg.loops:
            raise ValueError("continuation must cover a later nonempty loop range")
        if workspace.shape != (batch.batch_size, self.cfg.d_model):
            raise ValueError("workspace has the wrong shape")
        result = self._run(
            batch,
            visibility,
            workspace=workspace,
            start_loop=start_loop,
            stop_loop=stop_loop,
            return_cache=return_cache,
        )
        return result if return_cache else result[:2]
