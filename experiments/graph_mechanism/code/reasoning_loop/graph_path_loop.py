from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GraphPathConfig:
    node_count: int = 32
    max_depth: int = 6
    d_model: int = 512
    n_heads: int = 8
    d_mlp: int = 2048
    n_layers: int = 4
    max_loops: int = 6
    dropout: float = 0.0
    outer_norm_groups: int = 0
    residual_projection_groups: int = 0
    rms_norm_eps: float = 1e-6
    inner_norm_style: str = "pre_layernorm"
    readout_norm_style: str = "layernorm"
    block_style: str = "legacy"
    rope_theta: float = 1_000_000.0
    initializer_range: float = 0.02
    block_schedule: str = "all_blocks"

    @property
    def edge_token(self) -> int:
        return self.node_count

    @property
    def query_token(self) -> int:
        return self.node_count + 1

    @property
    def answer_token(self) -> int:
        return self.node_count + 2

    @property
    def bos_token(self) -> int:
        return self.node_count + 3

    @property
    def depth_token_base(self) -> int:
        return self.node_count + 4

    @property
    def vocab_size(self) -> int:
        return self.node_count + 4 + self.max_depth

    @property
    def seq_len(self) -> int:
        return 1 + 3 * self.node_count + 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train 10M+ looped transformers on an in-context graph-path "
            "reasoning task."
        )
    )
    parser.add_argument("--node-count", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-mlp", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help=(
            "Optional cosine schedule horizon when intentionally stopping before "
            "--steps would otherwise finish. Must be at least --steps."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-lr-scale", type=float, default=1.0)
    parser.add_argument("--attention-lr-scale", type=float, default=1.0)
    parser.add_argument("--mlp-lr-scale", type=float, default=1.0)
    parser.add_argument("--norm-lr-scale", type=float, default=1.0)
    parser.add_argument("--readout-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--block-lr-scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional multiplier for each physical block. The number of values "
            "must equal --n-layers; multipliers compose with component scales."
        ),
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initialization-seed",
        type=int,
        default=None,
        help=(
            "Optional explicit parameter-initialization seed. By default the "
            "legacy seed + 1009 * loops rule is preserved."
        ),
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "Optional seed reset after model initialization. Use the same "
            "value across loop horizons for a matched training-batch stream."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional model-state checkpoint used before optimization. Parameter "
            "shapes must match; the current CLI architecture remains authoritative."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--inner-norm-style",
        choices=("pre_layernorm", "post_layernorm"),
        default="pre_layernorm",
        help=(
            "Transformer block normalization placement. post_layernorm uses "
            "LN(x + Attention(x)) followed by LN(x + MLP(x))."
        ),
    )
    parser.add_argument("--aux-loss", type=float, default=0.0)
    parser.add_argument(
        "--trajectory-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary CE weight for a reusable graph-walk trajectory. "
            "Loop t is supervised to f^min(t * jump, queried_depth)(start). "
            "The default 0 preserves final-only training."
        ),
    )
    parser.add_argument(
        "--trajectory-aux-jump",
        type=int,
        default=2,
        help="Graph steps represented by one recurrent loop in the trajectory target.",
    )
    parser.add_argument(
        "--trajectory-aux-active-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply trajectory CE only through the first loop that reaches the "
            "queried depth, rather than repeatedly rewarding later halt slots."
        ),
    )
    parser.add_argument(
        "--trajectory-aux-hold-steps",
        type=int,
        default=0,
        help="Keep the full trajectory auxiliary weight through this train step.",
    )
    parser.add_argument(
        "--trajectory-aux-end-step",
        type=int,
        default=0,
        help=(
            "Linearly anneal the trajectory auxiliary weight to zero at this "
            "step. Values <= 0 keep the weight constant."
        ),
    )
    parser.add_argument(
        "--norm-transition-start-step",
        type=int,
        default=0,
        help=(
            "For post_layernorm only, optionally train through a Pre-to-Post "
            "homotopy. The computation is exactly Pre-Norm through this step."
        ),
    )
    parser.add_argument(
        "--norm-transition-end-step",
        type=int,
        default=0,
        help=(
            "End the optional Pre-to-Post homotopy at this step. Values <= 0 "
            "disable it; at and after the end step the forward is exactly "
            "canonical Post-Norm."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-eval-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When saving checkpoints, also save a checkpoint at every evaluation step.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/graph_path_loop_reasoning"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_seeds(
    args: argparse.Namespace,
    *,
    max_loops: int,
) -> tuple[int, int | None]:
    """Resolve backward-compatible initialization and optional data seeds."""
    initialization_seed = (
        args.seed + 1009 * max_loops
        if args.initialization_seed is None
        else args.initialization_seed
    )
    return initialization_seed, args.data_seed


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint completely before replacing its final path."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(y)


class OuroRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, *, theta: float = 1_000_000.0) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if theta <= 0:
            raise ValueError("theta must be positive")
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse_frequency = self.inverse_frequency[None, :, None].float()
        positions = position_ids[:, None, :].float()
        frequencies = (inverse_frequency @ positions).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embedding.cos().to(device=x.device, dtype=x.dtype).unsqueeze(1),
            embedding.sin().to(device=x.device, dtype=x.dtype).unsqueeze(1),
        )

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        rotated_half = torch.cat((-second, first), dim=-1)
        return x * cos + rotated_half * sin


class OuroSelfAttention(nn.Module):
    def __init__(self, cfg: GraphPathConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary_emb = OuroRotaryEmbedding(
            self.head_dim,
            theta=getattr(cfg, "rope_theta", 1_000_000.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        def split_heads(projection: nn.Linear) -> torch.Tensor:
            return projection(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.q_proj)
        key = split_heads(self.k_proj)
        value = split_heads(self.v_proj)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(query, position_ids)
        query = self.rotary_emb.apply(query, cos, sin)
        key = self.rotary_emb.apply(key, cos, sin)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.o_proj(attended)


class OuroSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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


class GroupOrthogonalResidual(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        groups: int,
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

    def forward(
        self,
        hidden: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape != update.shape or hidden.shape[-1] != self.d_model:
            raise ValueError(
                "hidden and update must have matching shapes ending in d_model"
            )
        grouped_hidden = hidden.float().reshape(
            *hidden.shape[:-1], self.groups, self.group_size
        )
        grouped_update = update.float().reshape_as(grouped_hidden)
        coefficient = (grouped_hidden * grouped_update).sum(dim=-1, keepdim=True)
        coefficient = coefficient / (
            grouped_hidden.square().sum(dim=-1, keepdim=True) + self.eps
        )
        projected = grouped_update - coefficient * grouped_hidden
        return projected.reshape_as(update).to(dtype=update.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GraphPathConfig) -> None:
        super().__init__()
        self.inner_norm_style = getattr(cfg, "inner_norm_style", "pre_layernorm")
        self.norm_transition_alpha: float | None = None
        if self.inner_norm_style in ("pre_layernorm", "post_layernorm"):
            self.ln_1 = nn.LayerNorm(cfg.d_model)
            self.ln_2 = nn.LayerNorm(cfg.d_model)
        elif self.inner_norm_style in ("pre_rmsnorm", "ouro_sandwich_rms"):
            self.ln_1 = GroupRMSNorm(cfg.d_model, groups=1, eps=cfg.rms_norm_eps)
            self.ln_2 = GroupRMSNorm(cfg.d_model, groups=1, eps=cfg.rms_norm_eps)
            if self.inner_norm_style == "ouro_sandwich_rms":
                self.attn_out_norm = GroupRMSNorm(
                    cfg.d_model, groups=1, eps=cfg.rms_norm_eps
                )
                self.mlp_out_norm = GroupRMSNorm(
                    cfg.d_model, groups=1, eps=cfg.rms_norm_eps
                )
        else:
            raise ValueError(f"unsupported inner_norm_style: {self.inner_norm_style}")
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        projection_groups = getattr(cfg, "residual_projection_groups", 0)
        self.residual_projector = (
            GroupOrthogonalResidual(
                cfg.d_model,
                groups=projection_groups,
                eps=cfg.rms_norm_eps,
            )
            if projection_groups
            else None
        )
        if (
            self.inner_norm_style == "post_layernorm"
            and self.residual_projector is not None
        ):
            raise ValueError(
                "post_layernorm is not defined with residual_projection_groups"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inner_norm_style == "post_layernorm":
            transition = self.norm_transition_alpha
            if transition is None or transition >= 1.0:
                x = self.ln_1(x + self.attn(x))
                return self.ln_2(x + self.mlp(x))
            pre_attention = x + self.attn(self.ln_1(x))
            pre_output = pre_attention + self.mlp(self.ln_2(pre_attention))
            if transition <= 0.0:
                return pre_output
            post_attention = self.ln_1(x + self.attn(x))
            post_output = self.ln_2(
                post_attention + self.mlp(post_attention)
            )
            return torch.lerp(pre_output, post_output, transition)
        block_input = x
        attention_update = self.attn(self.ln_1(x))
        if self.inner_norm_style == "ouro_sandwich_rms":
            attention_update = self.attn_out_norm(attention_update)
        x = x + attention_update
        mlp_update = self.mlp(self.ln_2(x))
        if self.inner_norm_style == "ouro_sandwich_rms":
            mlp_update = self.mlp_out_norm(mlp_update)
        x = x + mlp_update
        if self.residual_projector is None:
            return x
        return block_input + self.residual_projector(block_input, x - block_input)


class TinyOuroBlock(nn.Module):
    def __init__(self, cfg: GraphPathConfig) -> None:
        super().__init__()
        self.inner_norm_style = getattr(cfg, "inner_norm_style", "ouro_sandwich_rms")
        if self.inner_norm_style not in ("pre_rmsnorm", "ouro_sandwich_rms"):
            raise ValueError(
                "tiny_ouro supports inner_norm_style pre_rmsnorm or ouro_sandwich_rms"
            )
        self.input_layernorm = GroupRMSNorm(
            cfg.d_model, groups=1, eps=cfg.rms_norm_eps
        )
        self.post_attention_layernorm = GroupRMSNorm(
            cfg.d_model, groups=1, eps=cfg.rms_norm_eps
        )
        if self.inner_norm_style == "ouro_sandwich_rms":
            self.input_layernorm_2 = GroupRMSNorm(
                cfg.d_model, groups=1, eps=cfg.rms_norm_eps
            )
            self.post_attention_layernorm_2 = GroupRMSNorm(
                cfg.d_model, groups=1, eps=cfg.rms_norm_eps
            )
        self.attn = OuroSelfAttention(cfg)
        self.mlp = OuroSwiGLU(cfg.d_model, cfg.d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attention_update = self.attn(self.input_layernorm(x))
        if self.inner_norm_style == "ouro_sandwich_rms":
            attention_update = self.input_layernorm_2(attention_update)
        x = residual + attention_update

        residual = x
        mlp_update = self.mlp(self.post_attention_layernorm(x))
        if self.inner_norm_style == "ouro_sandwich_rms":
            mlp_update = self.post_attention_layernorm_2(mlp_update)
        return residual + mlp_update


class LoopedGraphPathTransformer(nn.Module):
    def __init__(self, cfg: GraphPathConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_style = getattr(cfg, "block_style", "legacy")
        self.block_schedule = getattr(cfg, "block_schedule", "all_blocks")
        if self.block_schedule not in {"all_blocks", "first_block_once"}:
            raise ValueError(f"unsupported block_schedule: {self.block_schedule}")
        if self.block_schedule == "first_block_once" and cfg.n_layers < 2:
            raise ValueError("first_block_once requires at least two blocks")
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if self.block_style == "legacy":
            self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
            block_type: type[nn.Module] = TransformerBlock
        elif self.block_style == "tiny_ouro":
            block_type = TinyOuroBlock
        else:
            raise ValueError(f"unsupported block_style: {self.block_style}")
        self.blocks = nn.ModuleList([block_type(cfg) for _ in range(cfg.n_layers)])
        outer_groups = getattr(cfg, "outer_norm_groups", 0)
        self.outer_norm = (
            GroupRMSNorm(
                cfg.d_model,
                groups=outer_groups,
                eps=getattr(cfg, "rms_norm_eps", 1e-6),
            )
            if outer_groups
            else None
        )
        readout_norm_style = getattr(cfg, "readout_norm_style", "layernorm")
        if readout_norm_style == "layernorm":
            self.ln_final = nn.LayerNorm(cfg.d_model)
        elif readout_norm_style == "rmsnorm":
            self.ln_final = GroupRMSNorm(
                cfg.d_model,
                groups=1,
                eps=getattr(cfg, "rms_norm_eps", 1e-6),
            )
        elif readout_norm_style == "identity":
            self.ln_final = nn.Identity()
        else:
            raise ValueError(f"unsupported readout_norm_style: {readout_norm_style}")
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.block_style == "legacy":
            nn.init.normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.token_embed.weight, std=0.02)
            nn.init.normal_(self.unembed.weight, std=0.02)
            return
        initializer_range = getattr(self.cfg, "initializer_range", 0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def active_block_indices(self, loop_index: int) -> tuple[int, ...]:
        if loop_index < 0:
            raise ValueError("loop_index must be nonnegative")
        if self.block_schedule == "first_block_once" and loop_index > 0:
            return tuple(range(1, len(self.blocks)))
        return tuple(range(len(self.blocks)))

    def apply_blocks_for_loop(
        self,
        x: torch.Tensor,
        *,
        loop_index: int,
    ) -> torch.Tensor:
        for block_index in self.active_block_indices(loop_index):
            x = self.blocks[block_index](x)
        return x

    def apply_loop(self, x: torch.Tensor, *, loop_index: int) -> torch.Tensor:
        x = self.apply_blocks_for_loop(x, loop_index=loop_index)
        if self.outer_norm is not None:
            x = self.outer_norm(x)
        return x

    def set_norm_transition(self, alpha: float | None) -> None:
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("norm transition alpha must lie in [0, 1]")
        if alpha is not None and any(
            getattr(block, "inner_norm_style", None) != "post_layernorm"
            for block in self.blocks
        ):
            raise ValueError("norm transition requires post_layernorm blocks")
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                block.norm_transition_alpha = alpha

    def forward_all(
        self,
        tokens: torch.Tensor,
        *,
        max_loops: int | None = None,
        return_states: bool = False,
        return_dynamics: bool = False,
    ) -> dict[str, torch.Tensor]:
        loops = self.cfg.max_loops if max_loops is None else max_loops
        x = self.token_embed(tokens)
        if self.block_style == "legacy":
            x = x + self.pos_embed.unsqueeze(0)
        logits_by_loop: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        incoming_answer_states: list[torch.Tensor] = []
        pre_outer_answer_states: list[torch.Tensor] = []
        recurrent_answer_states: list[torch.Tensor] = []
        for loop_index in range(loops):
            if return_dynamics:
                incoming_answer_states.append(x[:, -1, :])
            x = self.apply_blocks_for_loop(x, loop_index=loop_index)
            if return_dynamics:
                pre_outer_answer_states.append(x[:, -1, :])
            if self.outer_norm is not None:
                x = self.outer_norm(x)
            if return_dynamics:
                recurrent_answer_states.append(x[:, -1, :])
            final_state = self.ln_final(x[:, -1, :])
            logits_by_loop.append(self.unembed(final_state)[:, : self.cfg.node_count])
            if return_states:
                states.append(final_state)
        out = {"logits_by_loop": torch.stack(logits_by_loop, dim=1)}
        if return_states:
            out["states_by_loop"] = torch.stack(states, dim=1)
        if return_dynamics:
            out["incoming_answer_states_by_loop"] = torch.stack(
                incoming_answer_states, dim=1
            )
            out["pre_outer_answer_states_by_loop"] = torch.stack(
                pre_outer_answer_states, dim=1
            )
            out["recurrent_answer_states_by_loop"] = torch.stack(
                recurrent_answer_states, dim=1
            )
        return out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_all(tokens)["logits_by_loop"][:, -1, :]


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _component_for_parameter(name: str) -> str:
    if name.startswith(("token_embed.", "pos_embed")):
        return "embedding"
    if name.startswith(("ln_final.", "unembed.")):
        return "readout"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if any(
        marker in name
        for marker in (
            ".ln_",
            "layernorm",
            "_norm.",
            "outer_norm.",
        )
    ):
        return "norm"
    return "other"


def build_optimizer_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    component_lr_scales: dict[str, float],
    block_lr_scales: Sequence[float] | None,
) -> list[dict[str, Any]]:
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    for component, scale in component_lr_scales.items():
        if scale <= 0:
            raise ValueError(f"{component} lr scale must be positive")
    if block_lr_scales is not None and any(scale <= 0 for scale in block_lr_scales):
        raise ValueError("block lr scales must be positive")

    grouped: dict[tuple[str, int | None, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = _component_for_parameter(name)
        component_scale = component_lr_scales.get(component, 1.0)
        match = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
        block_index = int(match.group(1)) if match is not None else None
        block_scale = (
            float(block_lr_scales[block_index])
            if block_index is not None and block_lr_scales is not None
            else 1.0
        )
        scale = float(component_scale * block_scale)
        grouped.setdefault((component, block_index, scale), []).append(parameter)

    return [
        {
            "params": parameters,
            "lr": base_lr * scale,
            "lr_scale": scale,
            "group_name": (
                component
                if block_index is None
                else f"block{block_index}.{component}"
            ),
        }
        for (component, block_index, scale), parameters in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][1] is not None,
                -1 if item[0][1] is None else item[0][1],
                item[0][0],
            ),
        )
    ]


def set_optimizer_base_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * float(group.get("lr_scale", 1.0))


def make_graph_path_batch(
    cfg: GraphPathConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.rand(batch_size, cfg.node_count, device=device)
    successors = noise.argsort(dim=-1)
    src = torch.arange(cfg.node_count, device=device).view(1, cfg.node_count).expand(batch_size, -1)

    edge_triplets = torch.empty(batch_size, cfg.node_count, 3, dtype=torch.long, device=device)
    edge_triplets[:, :, 0] = cfg.edge_token
    edge_triplets[:, :, 1] = src
    edge_triplets[:, :, 2] = successors

    depth = torch.randint(1, cfg.max_depth + 1, (batch_size,), dtype=torch.long, device=device)
    start = torch.randint(0, cfg.node_count, (batch_size,), dtype=torch.long, device=device)
    targets_by_depth = torch.empty(batch_size, cfg.max_depth, dtype=torch.long, device=device)
    current = start
    for step in range(cfg.max_depth):
        current = successors.gather(1, current.view(-1, 1)).squeeze(1)
        targets_by_depth[:, step] = current
    target = targets_by_depth.gather(1, (depth - 1).view(-1, 1)).squeeze(1)

    tokens = torch.empty(batch_size, cfg.seq_len, dtype=torch.long, device=device)
    tokens[:, 0] = cfg.bos_token
    tokens[:, 1 : 1 + 3 * cfg.node_count] = edge_triplets.reshape(batch_size, 3 * cfg.node_count)
    tokens[:, -4] = cfg.query_token
    tokens[:, -3] = start
    tokens[:, -2] = cfg.depth_token_base + depth - 1
    tokens[:, -1] = cfg.answer_token
    return tokens, target, depth, targets_by_depth


def ce_by_loop(logits_by_loop: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch, loops, node_count = logits_by_loop.shape
    expanded_target = target[:, None].expand(batch, loops).reshape(batch * loops)
    return F.cross_entropy(
        logits_by_loop.reshape(batch * loops, node_count),
        expanded_target,
        reduction="none",
    ).view(batch, loops)


def graph_path_trajectory_targets(
    depth: torch.Tensor,
    targets_by_depth: torch.Tensor,
    *,
    max_loops: int,
    jump: int,
) -> torch.Tensor:
    """Return the desired node after each reusable graph-walk macro-step."""
    if jump < 1:
        raise ValueError("trajectory jump must be positive")
    if max_loops < 1:
        raise ValueError("max_loops must be positive")
    if depth.ndim != 1:
        raise ValueError("depth must have shape [batch]")
    if targets_by_depth.ndim != 2 or targets_by_depth.shape[0] != depth.shape[0]:
        raise ValueError("targets_by_depth must have shape [batch, max_depth]")

    loop_steps = (
        torch.arange(1, max_loops + 1, device=depth.device, dtype=depth.dtype)
        * jump
    )
    reached_depth = torch.minimum(depth[:, None], loop_steps[None, :])
    return targets_by_depth.gather(1, reached_depth - 1)


def graph_path_trajectory_active_mask(
    depth: torch.Tensor,
    *,
    max_loops: int,
    jump: int,
) -> torch.Tensor:
    """Mark progress loops through the first loop that reaches query depth."""
    if jump < 1:
        raise ValueError("trajectory jump must be positive")
    if max_loops < 1:
        raise ValueError("max_loops must be positive")
    if depth.ndim != 1:
        raise ValueError("depth must have shape [batch]")
    progress_before_loop = (
        torch.arange(max_loops, device=depth.device, dtype=depth.dtype) * jump
    )
    return progress_before_loop[None, :] < depth[:, None]


def trajectory_aux_weight_at_step(
    step: int,
    *,
    initial_weight: float,
    hold_steps: int,
    end_step: int,
) -> float:
    """Piecewise-linear auxiliary schedule with an optional final-only tail."""
    if initial_weight <= 0.0:
        return 0.0
    if hold_steps < 0:
        raise ValueError("trajectory auxiliary hold steps must be non-negative")
    if end_step <= 0:
        return float(initial_weight)
    if end_step <= hold_steps:
        raise ValueError("trajectory auxiliary end step must exceed hold steps")
    if step <= hold_steps:
        return float(initial_weight)
    if step >= end_step:
        return 0.0
    fraction_remaining = (end_step - step) / float(end_step - hold_steps)
    return float(initial_weight) * fraction_remaining


def norm_transition_alpha_at_step(
    step: int,
    *,
    start_step: int,
    end_step: int,
) -> float | None:
    """Return None when disabled, otherwise a linear Pre-to-Post mixture."""
    if start_step < 0:
        raise ValueError("norm transition start step must be non-negative")
    if end_step <= 0:
        return None
    if end_step <= start_step:
        raise ValueError("norm transition end step must exceed start step")
    if step <= start_step:
        return 0.0
    if step >= end_step:
        return 1.0
    return (step - start_step) / float(end_step - start_step)


def ce_by_trajectory(
    logits_by_loop: torch.Tensor,
    trajectory_targets: torch.Tensor,
) -> torch.Tensor:
    batch, loops, node_count = logits_by_loop.shape
    if trajectory_targets.shape != (batch, loops):
        raise ValueError("trajectory_targets must have shape [batch, loops]")
    return F.cross_entropy(
        logits_by_loop.reshape(batch * loops, node_count),
        trajectory_targets.reshape(batch * loops),
        reduction="none",
    ).view(batch, loops)


@torch.no_grad()
def evaluate(
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    max_loops: int,
    amp_enabled: bool,
    trajectory_jump: int = 2,
) -> dict[str, Any]:
    model.eval()
    correct_by_loop = torch.zeros(max_loops, device=device)
    count_by_loop = torch.zeros(max_loops, device=device)
    correct_by_depth_loop = torch.zeros(cfg.max_depth, max_loops, device=device)
    count_by_depth_loop = torch.zeros(cfg.max_depth, max_loops, device=device)
    loss_sum_by_loop = torch.zeros(max_loops, device=device)
    trajectory_correct_by_loop = torch.zeros(max_loops, device=device)
    trajectory_correct_by_depth_loop = torch.zeros(
        cfg.max_depth, max_loops, device=device
    )
    earliest_correct_sum = torch.zeros((), device=device)
    earliest_correct_count = torch.zeros((), device=device)

    autocast_device = "cuda" if device.type == "cuda" else device.type
    for _ in range(batches):
        tokens, target, depth, targets_by_depth = make_graph_path_batch(
            cfg, batch_size, device
        )
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]
            losses = ce_by_loop(logits_by_loop, target)
        pred = logits_by_loop.argmax(dim=-1)
        correct = pred.eq(target[:, None])
        trajectory_targets = graph_path_trajectory_targets(
            depth,
            targets_by_depth,
            max_loops=max_loops,
            jump=trajectory_jump,
        )
        trajectory_correct_by_loop += (
            pred.eq(trajectory_targets).float().sum(dim=0)
        )
        correct_by_loop += correct.float().sum(dim=0)
        count_by_loop += correct.shape[0]
        loss_sum_by_loop += losses.float().sum(dim=0)
        any_correct = correct.any(dim=1)
        first_correct = correct.float().argmax(dim=1).float() + 1.0
        earliest_correct_sum += first_correct[any_correct].sum()
        earliest_correct_count += any_correct.float().sum()
        for depth_idx in range(cfg.max_depth):
            mask = depth.eq(depth_idx + 1)
            if mask.any():
                correct_by_depth_loop[depth_idx] += correct[mask].float().sum(dim=0)
                trajectory_correct_by_depth_loop[depth_idx] += (
                    pred[mask].eq(trajectory_targets[mask]).float().sum(dim=0)
                )
                count_by_depth_loop[depth_idx] += mask.float().sum()

    loop_acc = correct_by_loop / count_by_loop.clamp_min(1)
    loop_loss = loss_sum_by_loop / count_by_loop.clamp_min(1)
    trajectory_loop_acc = trajectory_correct_by_loop / count_by_loop.clamp_min(1)
    depth_loop_acc = correct_by_depth_loop / count_by_depth_loop.clamp_min(1)
    trajectory_depth_loop_acc = (
        trajectory_correct_by_depth_loop / count_by_depth_loop.clamp_min(1)
    )
    earliest = earliest_correct_sum / earliest_correct_count.clamp_min(1)
    return {
        "loop_accuracy": [float(x) for x in loop_acc.detach().cpu()],
        "loop_loss": [float(x) for x in loop_loss.detach().cpu()],
        "trajectory_accuracy_by_loop": [
            float(x) for x in trajectory_loop_acc.detach().cpu()
        ],
        "trajectory_depth_loop_accuracy": (
            trajectory_depth_loop_acc.detach().cpu().tolist()
        ),
        "depth_loop_accuracy": depth_loop_acc.detach().cpu().tolist(),
        "earliest_correct_loop": float(earliest.detach().cpu()),
        "any_correct_fraction": float((earliest_correct_count / count_by_loop[0].clamp_min(1)).detach().cpu()),
    }


def cosine_lr(step: int, *, base_lr: float, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_one_loop_count(
    cfg: GraphPathConfig,
    args: argparse.Namespace,
    *,
    max_loops: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    run_dir = out_dir / f"graphpath_N{cfg.node_count}_D{cfg.max_depth}_d{cfg.d_model}_B{cfg.n_layers}_L{max_loops}_seed{args.seed}"
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"{run_dir} exists. Pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = GraphPathConfig(**{**asdict(cfg), "max_loops": max_loops})
    initialization_seed, data_seed = resolve_training_seeds(
        args, max_loops=max_loops
    )
    set_seed(initialization_seed)
    model = LoopedGraphPathTransformer(run_cfg).to(device)
    initialization_checkpoint: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        payload = torch.load(
            args.init_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model"], strict=True)
        initialization_checkpoint = {
            "path": str(args.init_checkpoint),
            "step": payload.get("step"),
            "config": payload.get("config"),
        }
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    if data_seed is not None:
        set_seed(data_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    block_lr_scales = args.block_lr_scales
    if block_lr_scales is not None and len(block_lr_scales) != run_cfg.n_layers:
        raise ValueError(
            "--block-lr-scales must provide exactly one value per physical block"
        )
    component_lr_scales = {
        "embedding": args.embedding_lr_scale,
        "attention": args.attention_lr_scale,
        "mlp": args.mlp_lr_scale,
        "norm": args.norm_lr_scale,
        "readout": args.readout_lr_scale,
    }
    optimizer_groups = build_optimizer_param_groups(
        model,
        base_lr=args.lr,
        component_lr_scales=component_lr_scales,
        block_lr_scales=block_lr_scales,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    param_count = count_parameters(model)
    history: list[dict[str, Any]] = []
    best_acc = -1.0
    best_step = 0
    start_time = time.time()
    autocast_device = "cuda" if device.type == "cuda" else device.type
    lr_decay_steps = (
        args.steps if args.lr_decay_steps is None else args.lr_decay_steps
    )
    if lr_decay_steps < args.steps:
        raise ValueError("--lr-decay-steps must be at least --steps")

    metadata = {
        "config": asdict(run_cfg),
        "args": vars(args),
        "parameter_count": param_count,
        "device": str(device),
        "run_dir": str(run_dir),
        "initialization_seed": initialization_seed,
        "data_seed": data_seed,
        "initialization_checkpoint": initialization_checkpoint,
        "optimizer_groups": [
            {
                "group_name": str(group["group_name"]),
                "lr_scale": float(group["lr_scale"]),
                "parameter_count": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in optimizer_groups
        ],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    for step in range(1, args.steps + 1):
        model.train()
        norm_transition_alpha = norm_transition_alpha_at_step(
            step,
            start_step=args.norm_transition_start_step,
            end_step=args.norm_transition_end_step,
        )
        model.set_norm_transition(norm_transition_alpha)
        lr = cosine_lr(
            step - 1,
            base_lr=args.lr,
            total_steps=lr_decay_steps,
            warmup_steps=args.warmup_steps,
        )
        set_optimizer_base_lr(optimizer, lr)
        tokens, target, depth, targets_by_depth = make_graph_path_batch(
            run_cfg, args.batch_size, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]
            per_loop_loss = ce_by_loop(logits_by_loop, target)
            final_loss = per_loop_loss[:, -1].mean()
            aux_loss = per_loop_loss[:, :-1].mean() if max_loops > 1 else final_loss
            trajectory_targets = graph_path_trajectory_targets(
                depth,
                targets_by_depth,
                max_loops=max_loops,
                jump=args.trajectory_aux_jump,
            )
            trajectory_loss_by_loop = ce_by_trajectory(
                logits_by_loop, trajectory_targets
            )
            if max_loops > 1:
                trajectory_loss_values = trajectory_loss_by_loop[:, :-1]
                if args.trajectory_aux_active_only:
                    active_mask = graph_path_trajectory_active_mask(
                        depth,
                        max_loops=max_loops,
                        jump=args.trajectory_aux_jump,
                    )[:, :-1]
                    trajectory_loss = (
                        trajectory_loss_values * active_mask
                    ).sum() / active_mask.sum().clamp_min(1)
                else:
                    trajectory_loss = trajectory_loss_values.mean()
            else:
                trajectory_loss = final_loss.new_zeros(())
            trajectory_aux_weight = trajectory_aux_weight_at_step(
                step,
                initial_weight=args.trajectory_aux_weight,
                hold_steps=args.trajectory_aux_hold_steps,
                end_step=args.trajectory_aux_end_step,
            )
            loss = (
                final_loss
                + args.aux_loss * aux_loss
                + trajectory_aux_weight * trajectory_loss
            )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            metrics = evaluate(
                model,
                run_cfg,
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                max_loops=max_loops,
                amp_enabled=args.amp,
                trajectory_jump=args.trajectory_aux_jump,
            )
            final_acc = metrics["loop_accuracy"][-1]
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.detach().cpu()),
                "train_final_loss": float(final_loss.detach().cpu()),
                "train_aux_loss": float(aux_loss.detach().cpu()),
                "train_trajectory_loss": float(trajectory_loss.detach().cpu()),
                "trajectory_aux_weight": trajectory_aux_weight,
                "norm_transition_alpha": norm_transition_alpha,
                "elapsed_sec": time.time() - start_time,
                **metrics,
            }
            history.append(row)
            write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if args.save_checkpoints and args.save_eval_checkpoints:
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(run_cfg),
                        "step": step,
                        "metrics": metrics,
                        "parameter_count": param_count,
                        "initialization_seed": initialization_seed,
                        "data_seed": data_seed,
                        "norm_transition_alpha": norm_transition_alpha,
                    },
                    run_dir / f"checkpoint_step_{step:05d}.pt",
                )
            if final_acc > best_acc:
                best_acc = final_acc
                best_step = step
                if args.save_checkpoints:
                    atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "config": asdict(run_cfg),
                            "step": step,
                            "metrics": metrics,
                            "parameter_count": param_count,
                            "initialization_seed": initialization_seed,
                            "data_seed": data_seed,
                            "norm_transition_alpha": norm_transition_alpha,
                        },
                        run_dir / "best.pt",
                    )
            if step % args.print_every == 0 or step == 1 or step == args.steps:
                loop_acc = " ".join(f"L{i+1}:{acc:.3f}" for i, acc in enumerate(metrics["loop_accuracy"]))
                depth_max_trajectory = " ".join(
                    f"L{i+1}:{acc:.3f}"
                    for i, acc in enumerate(
                        metrics["trajectory_depth_loop_accuracy"][-1]
                    )
                )
                print(
                    f"[L={max_loops} step={step:05d}] loss={float(loss.detach().cpu()):.4f} "
                    f"final_acc={final_acc:.3f} {loop_acc} "
                    f"trajectory_D{run_cfg.max_depth}=({depth_max_trajectory})",
                    flush=True,
                )

    final_metrics = evaluate(
        model,
        run_cfg,
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 32),
        max_loops=max_loops,
        amp_enabled=args.amp,
        trajectory_jump=args.trajectory_aux_jump,
    )
    summary = {
        "max_loops": max_loops,
        "initialization_seed": initialization_seed,
        "data_seed": data_seed,
        "parameter_count": param_count,
        "best_step": best_step,
        "best_final_accuracy": best_acc,
        "final_metrics": final_metrics,
        "final_norm_transition_alpha": norm_transition_alpha_at_step(
            args.steps,
            start_step=args.norm_transition_start_step,
            end_step=args.norm_transition_end_step,
        ),
        "run_dir": str(run_dir),
        "peak_cuda_memory_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else None
        ),
        "peak_cuda_memory_reserved_mib": (
            float(torch.cuda.max_memory_reserved(device) / 2**20)
            if device.type == "cuda"
            else None
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_checkpoints:
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "config": asdict(run_cfg),
                "step": args.steps,
                "metrics": final_metrics,
                "parameter_count": param_count,
                "initialization_seed": initialization_seed,
                "data_seed": data_seed,
                "norm_transition_alpha": norm_transition_alpha_at_step(
                    args.steps,
                    start_step=args.norm_transition_start_step,
                    end_step=args.norm_transition_end_step,
                ),
            },
            run_dir / "final.pt",
        )
    plot_run(run_dir, history, summary)
    return summary


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    scalar_keys = [
        "step",
        "lr",
        "train_loss",
        "train_final_loss",
        "train_aux_loss",
        "train_trajectory_loss",
        "trajectory_aux_weight",
        "norm_transition_alpha",
        "elapsed_sec",
        "earliest_correct_loop",
        "any_correct_fraction",
    ]
    max_loop = len(history[-1]["loop_accuracy"])
    for idx in range(max_loop):
        scalar_keys.append(f"eval_acc_loop_{idx + 1}")
        scalar_keys.append(f"eval_loss_loop_{idx + 1}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in history:
            flat = {key: row.get(key, "") for key in scalar_keys}
            for idx, value in enumerate(row["loop_accuracy"]):
                flat[f"eval_acc_loop_{idx + 1}"] = value
            for idx, value in enumerate(row["loop_loss"]):
                flat[f"eval_loss_loop_{idx + 1}"] = value
            writer.writerow(flat)


def plot_run(run_dir: Path, history: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if not history:
        return
    steps = [row["step"] for row in history]
    max_loop = len(history[-1]["loop_accuracy"])
    plt.figure(figsize=(9, 5))
    for idx in range(max_loop):
        plt.plot(steps, [row["loop_accuracy"][idx] for row in history], label=f"loop {idx + 1}")
    plt.xlabel("train step")
    plt.ylabel("eval accuracy")
    plt.ylim(0, 1.02)
    plt.title(f"Graph-path accuracy by internal loop, max_loops={max_loop}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_by_loop_over_training.png", dpi=180)
    plt.close()

    depth_loop = np.array(summary["final_metrics"]["depth_loop_accuracy"], dtype=np.float32)
    plt.figure(figsize=(1.1 * max_loop + 3, 5))
    im = plt.imshow(depth_loop, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="accuracy")
    plt.xticks(range(max_loop), [str(i + 1) for i in range(max_loop)])
    plt.yticks(range(depth_loop.shape[0]), [str(i + 1) for i in range(depth_loop.shape[0])])
    plt.xlabel("internal loop used for readout")
    plt.ylabel("queried path depth")
    plt.title("Final depth x loop accuracy")
    for y in range(depth_loop.shape[0]):
        for x in range(depth_loop.shape[1]):
            plt.text(x, y, f"{depth_loop[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "final_depth_loop_accuracy_heatmap.png", dpi=180)
    plt.close()


def plot_cross_run(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    plt.figure(figsize=(8, 5))
    for summary in summaries:
        run_dir = Path(summary["run_dir"])
        hist_path = run_dir / "history.json"
        if not hist_path.exists():
            continue
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        plt.plot(
            [row["step"] for row in history],
            [row["loop_accuracy"][-1] for row in history],
            label=f"max loops {summary['max_loops']}",
        )
    plt.xlabel("train step")
    plt.ylabel("final-readout eval accuracy")
    plt.ylim(0, 1.02)
    plt.title("Graph-path reasoning: loop count comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loop_count_final_accuracy_comparison.png", dpi=180)
    plt.close()

    max_depth = len(summaries[0]["final_metrics"]["depth_loop_accuracy"])
    rows = []
    for summary in summaries:
        loop_count = summary["max_loops"]
        final_acc = np.array(summary["final_metrics"]["depth_loop_accuracy"], dtype=np.float32)[:, -1]
        for depth_idx in range(max_depth):
            rows.append((loop_count, depth_idx + 1, float(final_acc[depth_idx])))
    with (out_dir / "depth_accuracy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["max_loops", "query_depth", "final_accuracy"])
        writer.writerows(rows)


def write_readme(out_dir: Path, cfg: GraphPathConfig, args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Graph-Path Loop Reasoning Pilot",
        "",
        "This experiment trains shared-weight looped transformers on an in-context graph traversal task.",
        "",
        "Each sample contains a fresh random permutation graph as an edge table, then asks for the node reached after 1-6 successor steps. Because the graph changes every sample, the model cannot solve the task by memorizing a fixed transition table.",
        "",
        "## Config",
        "",
        f"- node_count: {cfg.node_count}",
        f"- max_depth: {cfg.max_depth}",
        f"- d_model: {cfg.d_model}",
        f"- n_layers per recurrent step: {cfg.n_layers}",
        f"- n_heads: {cfg.n_heads}",
        f"- d_mlp: {cfg.d_mlp}",
        f"- train steps per loop count: {args.steps}",
        f"- loop counts: {args.loops}",
        "",
        "## Final Summary",
        "",
        "| max loops | params | best final acc | final loop acc | earliest correct loop |",
        "|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        final_metrics = summary["final_metrics"]
        lines.append(
            f"| {summary['max_loops']} | {summary['parameter_count']:,} | "
            f"{summary['best_final_accuracy']:.4f} | {final_metrics['loop_accuracy'][-1]:.4f} | "
            f"{final_metrics['earliest_correct_loop']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `loop_count_final_accuracy_comparison.png`: final-readout accuracy over training.",
            "- `depth_accuracy_summary.csv`: final-readout accuracy by queried path depth and loop count.",
            "- Per-run folders contain `accuracy_by_loop_over_training.png` and `final_depth_loop_accuracy_heatmap.png`.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    cfg = GraphPathConfig(
        node_count=args.node_count,
        max_depth=args.max_depth,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        max_loops=max(args.loops),
        dropout=args.dropout,
        inner_norm_style=args.inner_norm_style,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"device={device} cfg={cfg}", flush=True)

    summaries = []
    for loop_count in args.loops:
        summary = train_one_loop_count(cfg, args, max_loops=loop_count, device=device, out_dir=args.out_dir)
        summaries.append(summary)
        (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        plot_cross_run(args.out_dir, summaries)
        write_readme(args.out_dir, cfg, args, summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
