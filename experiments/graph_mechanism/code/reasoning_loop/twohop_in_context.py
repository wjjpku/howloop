from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TwoHopConfig:
    entity_count: int = 64
    chain_count: int = 5
    total_depth: int = 6
    architecture: str = "standard"
    period: int = 6
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.entity_count < 3 * self.chain_count:
            raise ValueError("entity_count must cover three distinct entities per chain")
        if self.chain_count < 2:
            raise ValueError("chain_count must be at least two")
        if self.total_depth < 1:
            raise ValueError("total_depth must be positive")
        if self.architecture not in {"standard", "periodic"}:
            raise ValueError("architecture must be standard or periodic")
        if self.architecture == "standard" and self.period != self.total_depth:
            raise ValueError("standard architecture expects period == total_depth")
        if self.architecture == "periodic":
            if not 1 <= self.period < self.total_depth:
                raise ValueError("periodic architecture expects 1 <= period < total_depth")
            if self.total_depth % self.period:
                raise ValueError("period must divide total_depth")
        if self.d_model < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_mlp < 0:
            raise ValueError("d_mlp must be nonnegative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def bos_token(self) -> int:
        return self.entity_count

    @property
    def vocab_size(self) -> int:
        return self.entity_count + 1

    @property
    def seq_len(self) -> int:
        return 2 + 4 * self.chain_count

    @property
    def unique_block_count(self) -> int:
        return self.total_depth if self.architecture == "standard" else self.period

    def parameter_index(self, effective_depth: int) -> int:
        if effective_depth < 0:
            raise ValueError("effective_depth must be nonnegative")
        if self.architecture == "standard":
            if effective_depth >= self.total_depth:
                raise ValueError("standard model has no block beyond trained depth")
            return effective_depth
        return effective_depth % self.period

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TwoHopConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class TwoHopBatch:
    tokens: torch.Tensor
    labels: torch.Tensor
    chains: torch.Tensor
    target_indices: torch.Tensor
    first_parent_positions: torch.Tensor
    first_child_positions: torch.Tensor
    second_parent_positions: torch.Tensor
    second_child_positions: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def chain_count(self) -> int:
        return int(self.chains.shape[1])

    @property
    def query_position(self) -> int:
        return int(self.tokens.shape[1] - 1)

    def target_positions(self, field: str) -> torch.Tensor:
        values = getattr(self, field)
        return values.gather(1, self.target_indices[:, None]).squeeze(1)


@dataclass(frozen=True)
class PairedTwoHopBatch:
    clean: TwoHopBatch
    corrupt: TwoHopBatch


def _premise_order(
    *,
    batch_size: int,
    chain_count: int,
    device: torch.device,
    generator: torch.Generator,
    mode: str,
) -> torch.Tensor:
    flat_indices = torch.arange(
        2 * chain_count,
        device=device,
    ).reshape(1, chain_count, 2).expand(batch_size, -1, -1)
    if mode in {"topological", "reversed"}:
        keys = torch.rand(
            (batch_size, chain_count, 2),
            device=device,
            generator=generator,
        ).sort(dim=2).values
        if mode == "reversed":
            keys = keys.flip(dims=(2,))
        return flat_indices.reshape(batch_size, -1).gather(
            1,
            keys.reshape(batch_size, -1).argsort(dim=1),
        )
    chain_priority = torch.rand(
        (batch_size, chain_count),
        device=device,
        generator=generator,
    ).argsort(dim=1)
    if mode == "adjacent":
        return torch.stack(
            (2 * chain_priority, 2 * chain_priority + 1),
            dim=2,
        ).reshape(batch_size, -1)
    if mode == "grouped":
        second_priority = torch.rand(
            (batch_size, chain_count),
            device=device,
            generator=generator,
        ).argsort(dim=1)
        return torch.cat((2 * chain_priority, 2 * second_priority + 1), dim=1)
    raise ValueError(f"unknown premise order mode: {mode}")


def _retarget(batch: TwoHopBatch, target_indices: torch.Tensor) -> TwoHopBatch:
    if target_indices.shape != batch.target_indices.shape:
        raise ValueError("target_indices has the wrong shape")
    batch_indices = torch.arange(batch.batch_size, device=batch.tokens.device)
    query = batch.chains[batch_indices, target_indices, 0]
    labels = batch.chains[batch_indices, target_indices, 2]
    tokens = batch.tokens.clone()
    tokens[:, -1] = query
    return TwoHopBatch(
        tokens=tokens,
        labels=labels,
        chains=batch.chains,
        target_indices=target_indices,
        first_parent_positions=batch.first_parent_positions,
        first_child_positions=batch.first_child_positions,
        second_parent_positions=batch.second_parent_positions,
        second_child_positions=batch.second_child_positions,
    )


def make_twohop_batch(
    cfg: TwoHopConfig,
    batch_size: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    order_mode: str = "topological",
    target_indices: torch.Tensor | None = None,
) -> TwoHopBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    priorities = torch.rand(
        (batch_size, cfg.entity_count),
        device=device,
        generator=generator,
    )
    selected = priorities.argsort(dim=1)[:, : 3 * cfg.chain_count]
    chains = selected.reshape(batch_size, cfg.chain_count, 3)
    first_hops = chains[:, :, :2]
    second_hops = chains[:, :, 1:]
    premises = torch.stack((first_hops, second_hops), dim=2)
    flat_premises = premises.reshape(batch_size, 2 * cfg.chain_count, 2)
    order = _premise_order(
        batch_size=batch_size,
        chain_count=cfg.chain_count,
        device=device,
        generator=generator,
        mode=order_mode,
    )
    ordered_premises = flat_premises.gather(
        1,
        order[:, :, None].expand(-1, -1, 2),
    )

    bos = torch.full(
        (batch_size, 1),
        cfg.bos_token,
        dtype=torch.long,
        device=device,
    )
    if target_indices is None:
        target_indices = torch.randint(
            0,
            cfg.chain_count,
            (batch_size,),
            device=device,
            generator=generator,
        )
    if target_indices.shape != (batch_size,):
        raise ValueError("target_indices has the wrong shape")
    batch_indices = torch.arange(batch_size, device=device)
    query = chains[batch_indices, target_indices, 0:1]
    labels = chains[batch_indices, target_indices, 2]
    tokens = torch.cat(
        (bos, ordered_premises.reshape(batch_size, -1), query),
        dim=1,
    )

    base_chain_ids = torch.arange(cfg.chain_count, device=device).repeat_interleave(2)
    base_hop_ids = torch.arange(2, device=device).repeat(cfg.chain_count)
    ordered_chain_ids = base_chain_ids[order]
    ordered_hop_ids = base_hop_ids[order]
    first_parent = torch.empty(
        (batch_size, cfg.chain_count),
        dtype=torch.long,
        device=device,
    )
    first_child = torch.empty_like(first_parent)
    second_parent = torch.empty_like(first_parent)
    second_child = torch.empty_like(first_parent)
    for slot in range(2 * cfg.chain_count):
        parent_position = 1 + 2 * slot
        child_position = parent_position + 1
        chain_ids = ordered_chain_ids[:, slot]
        hop_ids = ordered_hop_ids[:, slot]
        first_mask = hop_ids.eq(0)
        second_mask = ~first_mask
        first_rows = batch_indices[first_mask]
        second_rows = batch_indices[second_mask]
        first_parent[first_rows, chain_ids[first_mask]] = parent_position
        first_child[first_rows, chain_ids[first_mask]] = child_position
        second_parent[second_rows, chain_ids[second_mask]] = parent_position
        second_child[second_rows, chain_ids[second_mask]] = child_position

    return TwoHopBatch(
        tokens=tokens,
        labels=labels,
        chains=chains,
        target_indices=target_indices,
        first_parent_positions=first_parent,
        first_child_positions=first_child,
        second_parent_positions=second_parent,
        second_child_positions=second_child,
    )


def make_paired_twohop_batch(
    cfg: TwoHopConfig,
    batch_size: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    order_mode: str = "topological",
    corruption_mode: str = "query",
) -> PairedTwoHopBatch:
    clean = make_twohop_batch(
        cfg,
        batch_size,
        device=device,
        generator=generator,
        order_mode=order_mode,
    )
    offset = torch.randint(
        1,
        cfg.chain_count,
        (batch_size,),
        device=device,
        generator=generator,
    )
    paired_indices = (clean.target_indices + offset) % cfg.chain_count
    if corruption_mode == "query":
        corrupt = _retarget(clean, paired_indices)
    elif corruption_mode == "bridge_swap":
        rows = torch.arange(batch_size, device=device)
        corrupt_chains = clean.chains.clone()
        target_tail = corrupt_chains[
            rows,
            clean.target_indices,
            1:,
        ].clone()
        paired_tail = corrupt_chains[
            rows,
            paired_indices,
            1:,
        ].clone()
        corrupt_chains[rows, clean.target_indices, 1:] = paired_tail
        corrupt_chains[rows, paired_indices, 1:] = target_tail

        corrupt_tokens = clean.tokens.clone()
        target_first_child = clean.first_child_positions[
            rows,
            clean.target_indices,
        ]
        paired_first_child = clean.first_child_positions[
            rows,
            paired_indices,
        ]
        target_bridge = clean.chains[
            rows,
            clean.target_indices,
            1,
        ]
        paired_bridge = clean.chains[
            rows,
            paired_indices,
            1,
        ]
        corrupt_tokens[rows, target_first_child] = paired_bridge
        corrupt_tokens[rows, paired_first_child] = target_bridge

        second_parent = clean.second_parent_positions.clone()
        second_child = clean.second_child_positions.clone()
        target_second_parent = second_parent[
            rows,
            clean.target_indices,
        ].clone()
        paired_second_parent = second_parent[
            rows,
            paired_indices,
        ].clone()
        target_second_child = second_child[
            rows,
            clean.target_indices,
        ].clone()
        paired_second_child = second_child[
            rows,
            paired_indices,
        ].clone()
        second_parent[rows, clean.target_indices] = paired_second_parent
        second_parent[rows, paired_indices] = target_second_parent
        second_child[rows, clean.target_indices] = paired_second_child
        second_child[rows, paired_indices] = target_second_child
        corrupt = TwoHopBatch(
            tokens=corrupt_tokens,
            labels=corrupt_chains[
                rows,
                clean.target_indices,
                2,
            ],
            chains=corrupt_chains,
            target_indices=clean.target_indices,
            first_parent_positions=clean.first_parent_positions,
            first_child_positions=clean.first_child_positions,
            second_parent_positions=second_parent,
            second_child_positions=second_child,
        )
    else:
        raise ValueError(f"unknown corruption mode: {corruption_mode}")
    return PairedTwoHopBatch(clean=clean, corrupt=corrupt)


Intervention = Mapping[str, torch.Tensor]
Interventions = Mapping[int, Intervention]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TwoHopConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        values: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        intervention: Intervention | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, _ = values.shape
        qkv = self.qkv(values).reshape(
            batch,
            seq_len,
            3,
            self.n_heads,
            self.d_head,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.d_head**0.5
        scores = scores.masked_fill(
            ~causal_mask,
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = self.dropout(attention)
        z = torch.matmul(attention, value).transpose(1, 2)
        if intervention is not None:
            if "z_patch" in intervention:
                patch = intervention["z_patch"].to(device=z.device, dtype=z.dtype)
                mask = intervention.get("z_mask")
                if mask is None:
                    z = patch
                else:
                    z = torch.where(mask.to(device=z.device, dtype=torch.bool), patch, z)
            if "z_zero_mask" in intervention:
                zero_mask = intervention["z_zero_mask"].to(
                    device=z.device,
                    dtype=torch.bool,
                )
                z = torch.where(zero_mask, torch.zeros_like(z), z)
        output = self.out_proj(z.reshape(batch, seq_len, self.d_model))
        cache: dict[str, torch.Tensor] = {}
        if return_cache:
            cache = {
                "q": query.detach(),
                "k": key.detach(),
                "v": value.detach(),
                "attention": attention.detach(),
                "z": z.detach(),
                "attention_out": output.detach(),
            }
        return output, cache


class TwoHopBlock(nn.Module):
    def __init__(self, cfg: TwoHopConfig) -> None:
        super().__init__()
        self.ln_attention = nn.LayerNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg)
        self.ln_mlp: nn.LayerNorm | None = None
        self.mlp: nn.Module | None = None
        if cfg.d_mlp > 0:
            self.ln_mlp = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_mlp),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_mlp, cfg.d_model),
            )

    def forward(
        self,
        values: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        intervention: Intervention | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attention_input = self.ln_attention(values)
        attention_out, attention_cache = self.attention(
            attention_input,
            causal_mask,
            intervention=intervention,
            return_cache=return_cache,
        )
        residual_mid = values + attention_out
        if self.mlp is None or self.ln_mlp is None:
            mlp_out = torch.zeros_like(residual_mid)
        else:
            mlp_out = self.mlp(self.ln_mlp(residual_mid))
        output = residual_mid + mlp_out
        if intervention is not None and "resid_patch" in intervention:
            patch = intervention["resid_patch"].to(
                device=output.device,
                dtype=output.dtype,
            )
            mask = intervention.get("resid_mask")
            if mask is None:
                output = patch
            else:
                output = torch.where(
                    mask.to(device=output.device, dtype=torch.bool),
                    patch,
                    output,
                )
        cache: dict[str, torch.Tensor] = {}
        if return_cache:
            cache = {
                "input": values.detach(),
                "attention_input": attention_input.detach(),
                "attention_out": attention_out.detach(),
                "residual_mid": residual_mid.detach(),
                "mlp_out": mlp_out.detach(),
                "resid": output.detach(),
                **attention_cache,
            }
        return output, cache


class TwoHopTransformer(nn.Module):
    def __init__(self, cfg: TwoHopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Common modules are instantiated before the blocks. Re-seeding before
        # each paired model therefore gives exactly matched common tensors and
        # matched initial values for the first `period` blocks.
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(cfg.seq_len, cfg.d_model)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, cfg.entity_count, bias=False)
        self.blocks = nn.ModuleList(
            TwoHopBlock(cfg) for _ in range(cfg.unique_block_count)
        )
        mask = torch.tril(
            torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            mask.reshape(1, 1, cfg.seq_len, cfg.seq_len),
            persistent=False,
        )

    def parameter_index(self, effective_depth: int) -> int:
        return self.cfg.parameter_index(effective_depth)

    def forward_all(
        self,
        tokens: torch.Tensor,
        *,
        active_depth: int | None = None,
        interventions: Interventions | None = None,
        return_cache: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if tokens.ndim != 2 or tokens.shape[1] != self.cfg.seq_len:
            raise ValueError(
                f"tokens must have shape [batch, {self.cfg.seq_len}]"
            )
        depth = self.cfg.total_depth if active_depth is None else active_depth
        if depth < 1:
            raise ValueError("active_depth must be positive")
        if self.cfg.architecture == "standard" and depth > self.cfg.total_depth:
            raise ValueError("standard model cannot run beyond its configured depth")
        values = (
            self.token_embedding(tokens)
            + self.position_embedding.unsqueeze(0)
        )
        logits_by_depth: list[torch.Tensor] = []
        states_by_depth: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}
        if return_cache:
            cache["embed"] = values.detach()
        for effective_depth in range(depth):
            parameter_index = self.parameter_index(effective_depth)
            values, block_cache = self.blocks[parameter_index](
                values,
                self.causal_mask,
                intervention=(
                    interventions.get(effective_depth)
                    if interventions is not None
                    else None
                ),
                return_cache=return_cache,
            )
            states_by_depth.append(values)
            logits_by_depth.append(
                self.readout(self.final_norm(values[:, -1]))
            )
            if return_cache:
                for name, tensor in block_cache.items():
                    cache[f"depth{effective_depth}.{name}"] = tensor
        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "logits_by_depth": torch.stack(logits_by_depth, dim=1),
            "states_by_depth": torch.stack(states_by_depth, dim=1),
        }
        if return_cache:
            result["cache"] = cache
        return result

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        active_depth: int | None = None,
        interventions: Interventions | None = None,
    ) -> torch.Tensor:
        output = self.forward_all(
            tokens,
            active_depth=active_depth,
            interventions=interventions,
        )
        logits = output["logits_by_depth"]
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("logits_by_depth is not a tensor")
        return logits[:, -1]


def build_twohop_model(
    cfg: TwoHopConfig,
    *,
    seed: int,
    device: torch.device,
) -> TwoHopTransformer:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return TwoHopTransformer(cfg).to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def accuracy_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    predictions = logits.argmax(dim=-1)
    accuracy = float(predictions.eq(labels).float().mean().item())
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], torch.finfo(logits.dtype).min)
    margin = float((correct - competitors.max(dim=1).values).mean().item())
    return accuracy, margin


def cross_entropy_by_depth(
    logits_by_depth: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if logits_by_depth.ndim != 3:
        raise ValueError("logits_by_depth must have [batch, depth, entity] axes")
    targets = labels[:, None].expand(-1, logits_by_depth.shape[1])
    return F.cross_entropy(
        logits_by_depth.flatten(0, 1),
        targets.reshape(-1),
        reduction="none",
    ).reshape(logits_by_depth.shape[:2])
