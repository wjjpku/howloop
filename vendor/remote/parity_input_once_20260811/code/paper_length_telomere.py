from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F


OFFICIAL_SOURCE_COMMIT = "33650c3c0cec3dd1d466b32489aaf30df4bda796"


@dataclass(frozen=True)
class PaperTaskSpec:
    name: str
    block_layers: int
    train_max_length: int
    curriculum_interval: int
    test_lengths: tuple[int, ...]
    step_offset: int = 0
    vocab_size: int = 6
    copy_symbols: tuple[int, ...] = (0, 1)
    addition_lsb_first: bool = False
    addition_minimum_width: int | None = None
    addition_answer_supervision: str = "full_layout"

    def target_steps(self, lengths: torch.Tensor) -> torch.Tensor:
        return lengths + self.step_offset

    def sequence_length(self, maximum_length: int) -> int:
        if self.name == "parity":
            return maximum_length + 2
        if self.name in {"copy", "copy4"}:
            return 2 * (maximum_length + 2)
        if self.name == "addition":
            return 3 * (self.addition_layout_width(maximum_length) + 2)
        if self.name == "sum_reverse":
            # The released generator is called with max_len=n+1 and allocates
            # max_len+7 positions.
            return maximum_length + 8
        raise ValueError(f"unsupported task: {self.name}")

    def addition_layout_width(self, maximum_length: int) -> int:
        if self.name != "addition":
            raise ValueError("addition layout width is defined only for Addition")
        minimum_width = self.addition_minimum_width or maximum_length
        return max(maximum_length, minimum_width)


PAPER_TASKS: dict[str, PaperTaskSpec] = {
    "parity": PaperTaskSpec(
        name="parity",
        block_layers=1,
        train_max_length=20,
        curriculum_interval=500,
        test_lengths=(10, 20, 30, 40, 50, 75, 100),
    ),
    "copy": PaperTaskSpec(
        name="copy",
        block_layers=2,
        # Released config ends curriculum.n_points at 20; randint excludes 20.
        train_max_length=19,
        curriculum_interval=1000,
        test_lengths=(10, 20, 25, 30, 40, 50),
    ),
    "copy4": PaperTaskSpec(
        name="copy4",
        block_layers=2,
        # Matched Copy control: retain the six-token model vocabulary and use
        # the two token IDs that are idle in binary Copy as additional data
        # symbols.  Separator=2 and PAD/EOS=3 remain unchanged.
        train_max_length=19,
        curriculum_interval=1000,
        test_lengths=(10, 20, 25, 30, 40, 50),
        copy_symbols=(0, 1, 4, 5),
    ),
    "addition": PaperTaskSpec(
        name="addition",
        block_layers=3,
        # Released config ends curriculum.n_points at 20; randint excludes 20.
        train_max_length=19,
        curriculum_interval=1600,
        test_lengths=(10, 20, 25, 30, 40, 50),
        step_offset=1,
    ),
    "sum_reverse": PaperTaskSpec(
        name="sum_reverse",
        block_layers=2,
        # Released config ends curriculum.n_points at 20, while randint's
        # upper bound is exclusive, so the largest sampled logical length is 19.
        train_max_length=19,
        curriculum_interval=500,
        test_lengths=(10, 19, 24, 30, 40, 50, 75, 100),
    ),
}


@dataclass(frozen=True)
class PaperBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    answer_mask: torch.Tensor
    lengths: torch.Tensor
    target_steps: torch.Tensor

    def to(self, device: torch.device) -> PaperBatch:
        return PaperBatch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            answer_mask=self.answer_mask.to(device),
            lengths=self.lengths.to(device),
            target_steps=self.target_steps.to(device),
        )


def _binary_addition(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    n = first.numel()
    result = torch.zeros(n + 1, dtype=torch.long)
    carry = 0
    for offset in range(n - 1, -1, -1):
        total = int(first[offset]) + int(second[offset]) + carry
        result[offset + 1] = total % 2
        carry = total // 2
    result[0] = carry
    return result


def generate_paper_batch(
    spec: PaperTaskSpec,
    *,
    batch_size: int,
    min_length: int,
    max_length: int,
    generator: torch.Generator,
    fixed_length: int | None = None,
) -> PaperBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 1 <= min_length <= max_length:
        raise ValueError("invalid length range")
    if fixed_length is not None and not min_length <= fixed_length <= max_length:
        raise ValueError("fixed_length is outside the requested range")
    if fixed_length is None:
        lengths = torch.randint(
            min_length,
            max_length + 1,
            (batch_size,),
            generator=generator,
            dtype=torch.long,
        )
    else:
        lengths = torch.full((batch_size,), fixed_length, dtype=torch.long)

    seq_len = spec.sequence_length(max_length)
    token_ids = torch.full((batch_size, seq_len), 3, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), 3, dtype=torch.long)
    answer_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for row, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        if spec.name == "parity":
            bits = torch.randint(0, 2, (length,), generator=generator)
            token_ids[row, :length] = bits
            token_ids[row, length] = 2
            targets[row, :length] = 5
            targets[row, length] = int(bits.sum()) % 2
            answer_mask[row, length:] = True
        elif spec.name in {"copy", "copy4"}:
            copy_symbols = torch.tensor(spec.copy_symbols, dtype=torch.long)
            symbol_indices = torch.randint(
                0,
                copy_symbols.numel(),
                (length,),
                generator=generator,
            )
            symbols = copy_symbols[symbol_indices]
            token_ids[row, :length] = symbols
            token_ids[row, length] = 2
            targets[row, :length] = 4
            targets[row, length : 2 * length] = symbols
            answer_mask[row, length:] = True
        elif spec.name == "addition":
            if spec.addition_answer_supervision not in {
                "full_layout",
                "logical_digits",
                "progressive_carry",
            }:
                raise ValueError("unsupported Addition answer supervision")
            width = spec.addition_layout_width(max_length)
            first = torch.randint(0, 2, (length,), generator=generator)
            second = torch.randint(0, 2, (length,), generator=generator)
            token_ids[row, :width] = 0
            token_ids[row, width + 1 : 2 * width + 1] = 0
            if spec.addition_lsb_first:
                token_ids[row, :length] = first.flip(0)
                token_ids[row, width + 1 : width + 1 + length] = second.flip(0)
            else:
                token_ids[row, width - length : width] = first
                token_ids[row, 2 * width + 1 - length : 2 * width + 1] = second
            token_ids[row, width] = 2
            answer_start = 2 * width + 1
            token_ids[row, answer_start] = 5
            targets[row, :answer_start] = 4
            answer = _binary_addition(first, second)
            targets[row, answer_start : answer_start + width + 1] = 0
            if spec.addition_lsb_first:
                targets[row, answer_start : answer_start + length + 1] = (
                    answer.flip(0)
                )
            else:
                answer_offset = width - length
                targets[
                    row,
                    answer_start + answer_offset : answer_start + width + 1,
                ] = answer
            if spec.addition_answer_supervision == "full_layout":
                answer_mask[row, answer_start:] = True
            elif spec.addition_answer_supervision == "progressive_carry":
                if not spec.addition_lsb_first or spec.step_offset != 1:
                    raise ValueError(
                        "progressive_carry requires LSB-first Addition with "
                        "T(n)=n+1"
                    )
                answer_mask[
                    row, answer_start : answer_start + length + 1
                ] = True
            elif spec.addition_lsb_first:
                # The first ``length`` answer slots are exactly the logical
                # sum digits from LSB upward.  The next slot is final carry;
                # high layout slots are only zero padding.  Excluding both
                # makes T(m)=m an exact one-digit-per-loop supervision rule.
                answer_mask[row, answer_start : answer_start + length] = True
            else:
                answer_offset = width - length
                answer_mask[
                    row,
                    answer_start + answer_offset + 1 : answer_start + width + 1,
                ] = True
        elif spec.name == "sum_reverse":
            bits = torch.randint(0, 2, (length,), generator=generator)
            token_ids[row, :length] = bits
            token_ids[row, length] = 5
            targets[row, :length] = 4
            total = int(bits.sum())
            if total:
                encoded = torch.tensor(
                    [int(bit) for bit in reversed(f"{total:b}")],
                    dtype=torch.long,
                )
                targets[row, length : length + encoded.numel()] = encoded
            answer_mask[row, length:] = True
        else:
            raise ValueError(f"unsupported task: {spec.name}")

    inputs = F.one_hot(token_ids, num_classes=spec.vocab_size).float()
    return PaperBatch(
        inputs=inputs,
        targets=targets,
        answer_mask=answer_mask,
        lengths=lengths,
        target_steps=spec.target_steps(lengths),
    )


def generate_balanced_addition_final_carry_batch(
    spec: PaperTaskSpec,
    *,
    batch_size: int,
    logical_length: int,
    generator: torch.Generator,
) -> PaperBatch:
    """Generate k-bit additions supervised only on the final carry-chain bit."""
    if spec.name != "addition":
        raise ValueError("final-carry supervision is defined only for Addition")
    if logical_length < 1 or batch_size < 1:
        raise ValueError("balanced final-carry batches require positive k and batch")
    required = {0: (batch_size + 1) // 2, 1: batch_size // 2}
    selected: list[PaperBatch] = []
    counts = {0: 0, 1: 0}
    width = spec.addition_layout_width(logical_length)
    answer_start = 2 * width + 1
    carry_position = answer_start + (
        logical_length if spec.addition_lsb_first else width - logical_length
    )
    while counts != required:
        remaining = sum(required[value] - counts[value] for value in (0, 1))
        candidate = generate_paper_batch(
            spec,
            batch_size=max(32, 4 * remaining),
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=generator,
        )
        carries = candidate.targets[:, carry_position]
        for value in (0, 1):
            take = required[value] - counts[value]
            if take <= 0:
                continue
            indices = torch.nonzero(carries == value, as_tuple=False).flatten()[:take]
            if not indices.numel():
                continue
            mask = torch.zeros_like(candidate.answer_mask[indices])
            mask[:, carry_position] = True
            selected.append(
                PaperBatch(
                    inputs=candidate.inputs[indices],
                    targets=candidate.targets[indices],
                    answer_mask=mask,
                    lengths=candidate.lengths[indices],
                    target_steps=candidate.target_steps[indices],
                )
            )
            counts[value] += int(indices.numel())
    combined = PaperBatch(
        inputs=torch.cat([part.inputs for part in selected]),
        targets=torch.cat([part.targets for part in selected]),
        answer_mask=torch.cat([part.answer_mask for part in selected]),
        lengths=torch.cat([part.lengths for part in selected]),
        target_steps=torch.cat([part.target_steps for part in selected]),
    )
    permutation = torch.randperm(batch_size, generator=generator)
    return PaperBatch(
        inputs=combined.inputs[permutation],
        targets=combined.targets[permutation],
        answer_mask=combined.answer_mask[permutation],
        lengths=combined.lengths[permutation],
        target_steps=combined.target_steps[permutation],
    )


@dataclass(frozen=True)
class PaperModelConfig:
    vocab_size: int = 6
    d_model: int = 256
    n_heads: int = 8
    d_mlp: int = 1024
    block_layers: int = 1
    layer_norm_epsilon: float = 1e-5
    attention_mode: str = "causal"
    token_embedding_injection: str = "every_loop"
    position_embedding: str = "none"
    position_injection: str = "initial_only"
    max_positions: int = 4096


def official_model_config(spec: PaperTaskSpec) -> PaperModelConfig:
    released_heads = {
        "parity": 64,
        "copy": 8,
        "copy4": 8,
        "addition": 8,
        "sum_reverse": 16,
    }
    if spec.name not in released_heads:
        raise ValueError(f"no released model configuration for {spec.name}")
    return PaperModelConfig(
        vocab_size=spec.vocab_size,
        d_model=256,
        n_heads=released_heads[spec.name],
        d_mlp=1024,
        block_layers=spec.block_layers,
    )


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: PaperModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if config.attention_mode not in {"causal", "full"}:
            raise ValueError("unsupported attention mode")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.is_causal = config.attention_mode == "causal"
        self.qkv = torch.nn.Linear(config.d_model, 3 * config.d_model)
        self.output = torch.nn.Linear(config.d_model, config.d_model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = value.shape
        qkv = self.qkv(value).view(
            batch, length, 3, self.n_heads, self.head_dim
        )
        query, key, val = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            val,
            dropout_p=0.0,
            is_causal=self.is_causal,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        return self.output(attended)


class PaperTransformerLayer(torch.nn.Module):
    def __init__(self, config: PaperModelConfig) -> None:
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = torch.nn.LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.d_mlp),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.attention_norm(value))
        return value + self.mlp(self.mlp_norm(value))


class PaperLoopedTransformer(torch.nn.Module):
    def __init__(self, config: PaperModelConfig) -> None:
        super().__init__()
        if config.token_embedding_injection not in {
            "initial_only",
            "every_loop",
        }:
            raise ValueError("unsupported token embedding injection")
        if config.position_embedding not in {"none", "learned_absolute"}:
            raise ValueError("unsupported position embedding")
        if config.position_injection not in {"initial_only", "every_loop"}:
            raise ValueError("unsupported position injection")
        if config.max_positions < 1:
            raise ValueError("max_positions must be positive")
        self.config = config
        self.read_in = torch.nn.Linear(config.vocab_size, config.d_model)
        self.position_embedding = (
            torch.nn.Embedding(config.max_positions, config.d_model)
            if config.position_embedding == "learned_absolute"
            else None
        )
        self.layers = torch.nn.ModuleList(
            [PaperTransformerLayer(config) for _ in range(config.block_layers)]
        )
        # Hugging Face GPT2Model applies ln_f after its physical layer stack.
        # The paper's released looped_forward calls the whole GPT2Model on
        # every recurrence, so this normalization belongs inside the shared
        # recurrent block rather than only before the final readout.
        self.final_norm = torch.nn.LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.read_out = torch.nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._initialize)
        # Match GPT-2's residual-path initialization used by the released
        # baseline: the attention and MLP output projections are scaled by
        # 1/sqrt(2 * n_layer).  Conv1D in the vendored implementation stores
        # the transpose of nn.Linear, but the distribution is identical.
        residual_std = 0.02 / math.sqrt(2 * config.block_layers)
        with torch.no_grad():
            for layer in self.layers:
                torch.nn.init.normal_(
                    layer.attention.output.weight,
                    mean=0.0,
                    std=residual_std,
                )
                torch.nn.init.normal_(
                    layer.mlp[-1].weight,
                    mean=0.0,
                    std=residual_std,
                )

    @staticmethod
    def _initialize(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _position_signal(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if self.position_embedding is None:
            return None
        if inputs.shape[1] > self.config.max_positions:
            raise ValueError("sequence exceeds learned position table")
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        return self.position_embedding(positions).unsqueeze(0)

    def input_embeddings(
        self, inputs: torch.Tensor, *, step_index: int = 1
    ) -> torch.Tensor:
        if step_index < 1:
            raise ValueError("step index must be positive")
        token_signal = self.read_in(inputs)
        embedded = (
            token_signal
            if (
                self.config.token_embedding_injection == "every_loop"
                or step_index == 1
            )
            else torch.zeros_like(token_signal)
        )
        position_signal = self._position_signal(inputs)
        if position_signal is None:
            return embedded
        if (
            self.config.position_injection == "every_loop"
            or step_index == 1
        ):
            return embedded + position_signal
        return embedded

    def recurrent_step(
        self, state: torch.Tensor, input_embeddings: torch.Tensor
    ) -> torch.Tensor:
        state = state + input_embeddings
        for layer in self.layers:
            state = layer(state)
        return self.final_norm(state)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.read_out(state)

    def states(
        self,
        inputs: torch.Tensor,
        *,
        steps: int,
        controller: torch.nn.Module | None = None,
        controller_start_step: int | None = None,
        executor_off_after_start: bool = False,
    ) -> list[torch.Tensor]:
        return list(
            self.iter_states(
                inputs,
                steps=steps,
                controller=controller,
                controller_start_step=controller_start_step,
                executor_off_after_start=executor_off_after_start,
            )
        )

    def iter_states(
        self,
        inputs: torch.Tensor,
        *,
        steps: int,
        controller: torch.nn.Module | None = None,
        controller_start_step: int | None = None,
        executor_off_after_start: bool = False,
    ) -> Iterator[torch.Tensor]:
        if steps < 1:
            raise ValueError("steps must be positive")
        token_embeddings = self.read_in(inputs)
        position_signal = self._position_signal(inputs)
        state = torch.zeros_like(token_embeddings)
        for step_index in range(1, steps + 1):
            embedded = (
                token_embeddings
                if (
                    self.config.token_embedding_injection == "every_loop"
                    or step_index == 1
                )
                else torch.zeros_like(token_embeddings)
            )
            if position_signal is not None and (
                self.config.position_injection == "every_loop"
                or step_index == 1
            ):
                embedded = embedded + position_signal
            controlled = (
                controller is not None
                and controller_start_step is not None
                and step_index > controller_start_step
            )
            if controlled:
                state = controller(state)
            if not (controlled and executor_off_after_start):
                state = self.recurrent_step(state, embedded)
            yield state


class DenseAffineController(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.weight = torch.nn.Parameter(torch.eye(dimension))
        self.bias = torch.nn.Parameter(torch.zeros(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() @ self.weight + self.bias

    @torch.no_grad()
    def initialize_iid_identity(
        self,
        *,
        residual_rms: float,
        bias_rms: float,
        seed: int,
    ) -> dict[str, float]:
        """Initialize a variance-controlled dense perturbation of identity.

        For a unit-variance row-state with independent coordinates, the
        off-diagonal matrix perturbation contributes ``residual_rms**2``
        expected variance to each output coordinate.  The diagonal remains
        exactly one, and the independent bias has RMS ``bias_rms``.
        """
        if self.dimension < 2:
            raise ValueError("iid identity initialization requires dimension >= 2")
        if residual_rms <= 0 or bias_rms <= 0:
            raise ValueError("iid initialization RMS values must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        off_diagonal_std = residual_rms / math.sqrt(self.dimension - 1)
        noise = torch.randn(
            self.dimension,
            self.dimension,
            generator=generator,
            dtype=torch.float32,
        ) * off_diagonal_std
        noise.fill_diagonal_(0.0)
        bias = torch.randn(
            self.dimension,
            generator=generator,
            dtype=torch.float32,
        ) * bias_rms
        identity = torch.eye(self.dimension, dtype=torch.float32)
        self.weight.copy_((identity + noise).to(self.weight.device))
        self.bias.copy_(bias.to(self.bias.device))
        return {
            "requested_residual_rms": residual_rms,
            "requested_bias_rms": bias_rms,
            "off_diagonal_entry_std": off_diagonal_std,
            "realized_off_diagonal_rms_per_output": float(
                noise.square().sum(dim=0).mean().sqrt()
            ),
            "realized_bias_rms": float(bias.square().mean().sqrt()),
            "maximum_diagonal_error": float(
                (self.weight.detach().diagonal().cpu() - 1.0).abs().max()
            ),
        }


class DiagonalLowRankController(torch.nn.Module):
    """The current D + low-rank LoRA recurrent boundary controller."""

    def __init__(self, dimension: int, rank: int) -> None:
        super().__init__()
        if not 1 <= rank <= dimension:
            raise ValueError("rank must lie in [1, dimension]")
        self.dimension = dimension
        self.rank = rank
        self.diagonal = torch.nn.Parameter(torch.ones(dimension))
        self.A = torch.nn.Parameter(torch.empty(dimension, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, dimension))
        self.bias = torch.nn.Parameter(torch.zeros(dimension))
        torch.nn.init.normal_(self.A, mean=0.0, std=1.0 / math.sqrt(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        return live * self.diagonal + (live @ self.A) @ self.B + self.bias

    def matrix(self) -> torch.Tensor:
        return torch.diag(self.diagonal) + self.A @ self.B

    @torch.no_grad()
    def initialize_from_dense(
        self,
        dense: DenseAffineController,
        *,
        gauge_seed: int,
    ) -> float:
        # The factorization is a small, one-off initializer.  Compute it on
        # CPU so CUDA and MPS produce the same deterministic gauge and so MPS
        # does not depend on unsupported QR/SVD kernels.
        target_device = self.diagonal.device
        dense_weight = dense.weight.detach().float().cpu()
        diagonal = torch.diagonal(dense_weight)
        residual = dense_weight - torch.diag(diagonal)
        left, singular, right_t = torch.linalg.svd(residual, full_matrices=False)
        selected = singular[: self.rank]
        root = selected.sqrt()
        base_A = left[:, : self.rank] * root.unsqueeze(0)
        base_B = root.unsqueeze(1) * right_t[: self.rank]
        generator = torch.Generator(device=residual.device)
        generator.manual_seed(gauge_seed)
        random_matrix = torch.randn(
            self.rank,
            self.rank,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        gauge, _ = torch.linalg.qr(random_matrix)
        self.diagonal.copy_(diagonal.to(target_device))
        self.A.copy_((base_A @ gauge).to(target_device))
        self.B.copy_((gauge.transpose(0, 1) @ base_B).to(target_device))
        self.bias.copy_(dense.bias.detach().float().to(target_device))
        return float(
            selected.square().sum()
            / singular.square().sum().clamp_min(1e-12)
        )

    def frozen(self) -> DiagonalLowRankController:
        result = copy.deepcopy(self).eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result


def _initialize_diagonal_low_rank_controller(
    *,
    dimension: int,
    rank: int,
    initialization: str,
    dense: DenseAffineController | None,
    gauge_seed: int,
    device: torch.device,
) -> tuple[DiagonalLowRankController, dict[str, float | None]]:
    controller = DiagonalLowRankController(dimension, rank).to(device)
    if initialization == "identity":
        return controller, {
            "retained_dense_residual_energy": None,
            "dense_to_factor_max_abs_error": 0.0,
        }
    if initialization != "dense_svd":
        raise ValueError(f"unsupported controller initialization: {initialization}")
    if dense is None:
        raise ValueError("dense_svd initialization requires a dense controller")
    retained_energy = controller.initialize_from_dense(
        dense, gauge_seed=gauge_seed
    )
    probe = torch.randn(4, 3, dimension, device=device)
    with torch.no_grad():
        dense_output = dense(probe)
        factor_output = controller(probe)
        initialization_error = float(
            (dense_output - factor_output).abs().max()
        )
    return controller, {
        "retained_dense_residual_energy": retained_energy,
        "dense_to_factor_max_abs_error": initialization_error,
    }


class ControllerView(torch.nn.Module):
    def __init__(
        self,
        controller: DiagonalLowRankController,
        *,
        mode: str,
        shuffle_seed: int = 0,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.mode = mode
        if mode == "shuffle_D":
            generator = torch.Generator(device=controller.diagonal.device)
            generator.manual_seed(shuffle_seed)
            permutation = torch.randperm(
                controller.dimension,
                generator=generator,
                device=controller.diagonal.device,
            )
        else:
            permutation = torch.arange(
                controller.dimension, device=controller.diagonal.device
            )
        self.register_buffer("permutation", permutation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        diagonal = self.controller.diagonal
        update = (live @ self.controller.A) @ self.controller.B
        bias = self.controller.bias
        if self.mode == "full":
            return live * diagonal + update + bias
        if self.mode == "no_AB":
            return live * diagonal + bias
        if self.mode == "identity_D":
            return live + update + bias
        if self.mode == "mean_D":
            return live * diagonal.mean() + update + bias
        if self.mode == "no_bias":
            return live * diagonal + update
        if self.mode == "shuffle_D":
            return live * diagonal[self.permutation] + update + bias
        raise ValueError(f"unsupported controller view: {self.mode}")


def masked_cross_entropy(logits: torch.Tensor, batch: PaperBatch) -> torch.Tensor:
    return F.cross_entropy(logits[batch.answer_mask], batch.targets[batch.answer_mask])


def _progressive_addition_prefix_mask(batch: PaperBatch) -> torch.Tensor:
    """Drop the final-carry slot from each sample's full arithmetic mask."""
    prefix_mask = batch.answer_mask.clone()
    for row, logical_length_tensor in enumerate(batch.lengths):
        active = torch.nonzero(prefix_mask[row], as_tuple=False).flatten()
        expected = int(logical_length_tensor) + 1
        if int(active.numel()) != expected:
            raise ValueError(
                "progressive_carry requires exactly n+1 final answer slots"
            )
        prefix_mask[row, active[-1]] = False
    return prefix_mask


def _adaptive_readout_loss(
    *,
    model: PaperLoopedTransformer,
    trajectory: Sequence[torch.Tensor],
    batch: PaperBatch,
    spec: PaperTaskSpec,
    selected_steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    endpoint_state = selected_state(trajectory, selected_steps)
    endpoint_logits = model.decode(endpoint_state).float()
    if spec.addition_answer_supervision != "progressive_carry":
        return masked_cross_entropy(endpoint_logits, batch), endpoint_logits
    if not torch.equal(selected_steps, batch.target_steps):
        raise ValueError(
            "progressive_carry requires sample-dependent T(n)=n+1 readout"
        )
    prefix_mask = _progressive_addition_prefix_mask(batch)
    prefix_state = selected_state(trajectory, selected_steps - 1)
    prefix_logits = model.decode(prefix_state).float()
    pooled_logits = torch.cat(
        (
            prefix_logits[prefix_mask],
            endpoint_logits[batch.answer_mask],
        ),
        dim=0,
    )
    pooled_targets = torch.cat(
        (
            batch.targets[prefix_mask],
            batch.targets[batch.answer_mask],
        ),
        dim=0,
    )
    return F.cross_entropy(pooled_logits, pooled_targets), endpoint_logits


@torch.no_grad()
def answer_cross_entropy(logits: torch.Tensor, batch: PaperBatch) -> float:
    return float(masked_cross_entropy(logits, batch))


@torch.no_grad()
def answer_predictive_entropy(logits: torch.Tensor, batch: PaperBatch) -> float:
    answer_logits = logits[batch.answer_mask]
    log_probabilities = F.log_softmax(answer_logits, dim=-1)
    probabilities = log_probabilities.exp()
    return float(-(probabilities * log_probabilities).sum(dim=-1).mean())


@torch.no_grad()
def exact_match(logits: torch.Tensor, batch: PaperBatch) -> float:
    predictions = logits.argmax(dim=-1)
    correct_tokens = predictions.eq(batch.targets) | ~batch.answer_mask
    return float(correct_tokens.all(dim=1).float().mean())


@torch.no_grad()
def predicted_confidence_loss(logits: torch.Tensor, batch: PaperBatch) -> float:
    predictions = logits.argmax(dim=-1)
    return float(F.cross_entropy(logits[batch.answer_mask], predictions[batch.answer_mask]))


def selected_state(
    states: Sequence[torch.Tensor], target_steps: torch.Tensor
) -> torch.Tensor:
    stacked = torch.stack(list(states), dim=1)
    indices = target_steps.to(stacked.device) - 1
    if int(indices.min()) < 0 or int(indices.max()) >= stacked.shape[1]:
        raise ValueError("target step is outside the generated trajectory")
    batch_index = torch.arange(stacked.shape[0], device=stacked.device)
    return stacked[batch_index, indices]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _apply_cuda_memory_fraction(device)
            return device
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    _apply_cuda_memory_fraction(device)
    return device


def _apply_cuda_memory_fraction(device: torch.device) -> None:
    raw_fraction = os.environ.get("PAPER_CUDA_MEMORY_FRACTION")
    if device.type != "cuda" or raw_fraction is None:
        return
    fraction = float(raw_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("PAPER_CUDA_MEMORY_FRACTION must lie in (0, 1]")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _model_payload(
    *,
    model: PaperLoopedTransformer,
    spec: PaperTaskSpec,
    supervision: str,
    seed: int,
    step: int,
    metrics: dict[str, Any],
    amp_enabled: bool,
    training_fixed_logical_length: int | None = None,
    training_fixed_loop_count: int | None = None,
    training_batch_shared_logical_length: bool = False,
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": "paper_length_telomere_backbone",
        "paper": "arXiv:2409.15647v5",
        "official_model_config": model.config == official_model_config(spec),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "initialization": "gpt2_scaled_residual_projection",
        "task": asdict(spec),
        "model": asdict(model.config),
        "supervision": supervision,
        "training_precision": (
            "bfloat16_autocast" if amp_enabled else "fp32"
        ),
        "training_fixed_logical_length": training_fixed_logical_length,
        "training_fixed_loop_count": training_fixed_loop_count,
        "training_batch_shared_logical_length": (
            training_batch_shared_logical_length
        ),
        "seed": seed,
        "step": step,
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }
    if training_state is not None:
        payload["training_state"] = training_state
    return payload


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _curriculum_max(spec: PaperTaskSpec, step: int) -> int:
    return min(1 + step // spec.curriculum_interval, spec.train_max_length)


def _backbone_training_length_range(
    spec: PaperTaskSpec,
    *,
    step: int,
    fixed_logical_length: int | None,
) -> tuple[int, int]:
    if fixed_logical_length is not None:
        return fixed_logical_length, fixed_logical_length
    return 1, _curriculum_max(spec, step)


def _learning_rate(
    *,
    base: float,
    step: int,
    total_steps: int,
    scheduler_anchor_zero_index: int,
) -> float:
    """Match the released post-optimizer CosineAnnealingLR timing exactly.

    The official loop is zero-indexed and calls ``scheduler.step()`` only
    after optimizer steps whose index is greater than ``end * interval``.
    Consequently, the first optimizer step that sees a decayed learning rate
    is two indices after that anchor.
    """
    if not 1 <= step <= total_steps:
        raise ValueError("step must be within the configured training run")
    zero_index = step - 1
    if zero_index <= scheduler_anchor_zero_index + 1:
        return base
    t_max = max(1, total_steps - scheduler_anchor_zero_index)
    t_cur = zero_index - (scheduler_anchor_zero_index + 1)
    progress = min(1.0, t_cur / t_max)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate_backbone_endpoint(
    *,
    model: PaperLoopedTransformer,
    spec: PaperTaskSpec,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
    logical_length: int | None = None,
    fixed_loop_count: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loss_total = 0.0
    accuracy_total = 0.0
    evaluation_length = (
        spec.train_max_length if logical_length is None else logical_length
    )
    horizon = (
        evaluation_length + spec.step_offset
        if fixed_loop_count is None
        else fixed_loop_count
    )
    for _ in range(batches):
        batch = generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=evaluation_length,
            max_length=evaluation_length,
            fixed_length=evaluation_length,
            generator=generator,
        ).to(device)
        state = model.states(batch.inputs, steps=horizon)[-1]
        logits = model.decode(state).float()
        loss_total += float(masked_cross_entropy(logits, batch))
        accuracy_total += exact_match(logits, batch)
    model.train(was_training)
    return {
        "validation_loss": loss_total / batches,
        "validation_exact_match": accuracy_total / batches,
    }


def load_backbone(
    checkpoint: Path, *, device: torch.device
) -> tuple[PaperLoopedTransformer, PaperTaskSpec, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper_length_telomere_backbone":
        raise ValueError("checkpoint is not a paper-style telomere backbone")
    task_payload = payload["task"]
    task_payload["test_lengths"] = tuple(task_payload["test_lengths"])
    spec = PaperTaskSpec(**task_payload)
    model = PaperLoopedTransformer(PaperModelConfig(**payload["model"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, spec, payload


def _backbone_model_config(
    spec: PaperTaskSpec, args: argparse.Namespace
) -> PaperModelConfig:
    if args.official_model_config:
        base = official_model_config(spec)
    else:
        base = PaperModelConfig(
            vocab_size=spec.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            block_layers=(
                spec.block_layers
                if args.block_layers is None
                else args.block_layers
            ),
        )
    return PaperModelConfig(
        **{
            **asdict(base),
            "attention_mode": args.attention_mode,
            "token_embedding_injection": args.token_embedding_injection,
            "position_embedding": args.position_embedding,
            "position_injection": args.position_injection,
            "max_positions": args.max_positions,
        }
    )


def train_backbone(args: argparse.Namespace) -> dict[str, Any]:
    if args.supervision not in {"adaptive_step", "fixed_horizon"}:
        raise ValueError("unsupported supervision")
    spec = PAPER_TASKS[args.task]
    if (
        args.train_max_length is not None
        or args.curriculum_interval is not None
        or args.task_step_offset is not None
        or args.addition_lsb_first
        or args.addition_minimum_width is not None
        or args.addition_answer_supervision is not None
    ):
        if (
            args.task != "addition"
            and (
                args.addition_lsb_first
                or args.addition_minimum_width is not None
                or args.addition_answer_supervision is not None
            )
        ):
            raise ValueError("Addition layout options require --task addition")
        if (
            args.addition_minimum_width is not None
            and args.addition_minimum_width < 1
        ):
            raise ValueError("Addition minimum width must be positive")
        if args.task_step_offset is not None and args.task_step_offset < 0:
            raise ValueError("task step offset must be non-negative")
        spec = PaperTaskSpec(
            **{
                **asdict(spec),
                "train_max_length": (
                    args.train_max_length
                    if args.train_max_length is not None
                    else spec.train_max_length
                ),
                "curriculum_interval": (
                    args.curriculum_interval
                    if args.curriculum_interval is not None
                    else spec.curriculum_interval
                ),
                "step_offset": (
                    args.task_step_offset
                    if args.task_step_offset is not None
                    else spec.step_offset
                ),
                "test_lengths": tuple(spec.test_lengths),
                "addition_lsb_first": args.addition_lsb_first,
                "addition_minimum_width": args.addition_minimum_width,
                "addition_answer_supervision": (
                    args.addition_answer_supervision
                    if args.addition_answer_supervision is not None
                    else spec.addition_answer_supervision
                ),
            }
        )
    if (
        spec.addition_answer_supervision == "progressive_carry"
        and args.supervision != "adaptive_step"
    ):
        raise ValueError(
            "progressive_carry requires adaptive_step backbone supervision"
        )
    fixed_logical_length = args.train_fixed_logical_length
    if fixed_logical_length is not None:
        if not 1 <= fixed_logical_length <= spec.train_max_length:
            raise ValueError(
                "fixed training logical length must lie within the task range"
            )
    fixed_loop_count = args.train_fixed_loop_count
    if fixed_loop_count is not None:
        if fixed_logical_length is None:
            raise ValueError(
                "fixed training loop count requires a fixed logical length"
            )
        if fixed_loop_count < 1:
            raise ValueError("fixed training loop count must be positive")
        if args.supervision != "fixed_horizon":
            raise ValueError(
                "fixed training loop count requires fixed_horizon supervision"
            )
    if args.batch_shared_logical_length and fixed_logical_length is not None:
        raise ValueError(
            "batch-shared logical length cannot be combined with one fixed "
            "training logical length"
        )
    if args.block_layers is not None and args.block_layers < 1:
        raise ValueError("block layer count must be positive")
    if args.official_model_config and args.block_layers is not None:
        raise ValueError(
            "--block-layers cannot be combined with --official-model-config"
        )
    device = pick_device(args.device)
    set_seed(args.seed)
    config = _backbone_model_config(spec, args)
    model = PaperLoopedTransformer(config).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = bool(args.amp and device.type == "cuda")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 17)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Released train.py uses ``points.end * points.interval`` as the
    # zero-indexed scheduler anchor.  ``points.end`` is one greater than the
    # maximum actual sampled length because numpy randint excludes its upper
    # bound.
    scheduler_anchor_zero_index = (
        spec.train_max_length + 1
    ) * spec.curriculum_interval
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_accuracy = -1.0
    best_step = 0
    starting_step = 0
    elapsed_before_resume = 0.0
    source_path = args.resume or args.initialize_from
    source_payload: dict[str, Any] | None = None
    if source_path is not None:
        source_payload = torch.load(
            source_path, map_location="cpu", weights_only=False
        )
        source_mode = "resume" if args.resume is not None else "initialize"
        if source_payload.get("kind") != "paper_length_telomere_backbone":
            raise ValueError(
                f"{source_mode} file is not a paper-style backbone checkpoint"
            )
        source_task = dict(source_payload["task"])
        source_task["test_lengths"] = tuple(source_task["test_lengths"])
        if PaperTaskSpec(**source_task) != spec:
            raise ValueError(f"{source_mode} task configuration does not match")
        if source_payload["model"] != asdict(config):
            raise ValueError(f"{source_mode} model configuration does not match")
        if source_payload["supervision"] != args.supervision:
            raise ValueError(f"{source_mode} supervision does not match")
        if (
            source_payload.get("training_fixed_logical_length")
            != fixed_logical_length
        ):
            raise ValueError(
                f"{source_mode} fixed training logical length does not match"
            )
        if source_payload.get("training_fixed_loop_count") != fixed_loop_count:
            raise ValueError(
                f"{source_mode} fixed training loop count does not match"
            )
        if bool(
            source_payload.get("training_batch_shared_logical_length", False)
        ) != bool(args.batch_shared_logical_length):
            raise ValueError(
                f"{source_mode} batch logical-length mode does not match"
            )
        expected_precision = "bfloat16_autocast" if use_amp else "fp32"
        if source_payload.get("training_precision") != expected_precision:
            raise ValueError(f"{source_mode} training precision does not match")
        if args.initialize_from is not None and source_payload["seed"] != args.seed:
            raise ValueError("initialize seed does not match")
        starting_step = int(source_payload["step"])
        if not 0 < starting_step < args.steps:
            raise ValueError(
                f"{source_mode} step must be before the requested final step"
            )
        model.load_state_dict(source_payload["state_dict"])
        if args.resume is not None:
            training_state = source_payload.get("training_state")
            if training_state is None:
                raise ValueError(
                    "resume checkpoint has no optimizer/RNG training state"
                )
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            _optimizer_to_device(optimizer, device)
            generator.set_state(training_state["data_generator_state"])
            torch.set_rng_state(training_state["torch_rng_state"])
            if device.type == "cuda" and training_state.get("cuda_rng_state_all"):
                torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
            history = list(training_state["history"])
            best_loss = float(training_state["best_loss"])
            best_accuracy = float(training_state["best_accuracy"])
            best_step = int(training_state["best_step"])
            elapsed_before_resume = float(training_state["elapsed_seconds"])
        else:
            warm_restart_seed = (
                args.seed + args.initialize_rng_seed_offset + starting_step
            )
            generator.manual_seed(warm_restart_seed + 17)
            torch.manual_seed(warm_restart_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(warm_restart_seed)
    started = time.monotonic() - elapsed_before_resume

    def current_training_state() -> dict[str, Any]:
        return {
            "optimizer_state_dict": optimizer.state_dict(),
            "data_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            ),
            "history": history,
            "best_loss": best_loss,
            "best_accuracy": best_accuracy,
            "best_step": best_step,
            "elapsed_seconds": time.monotonic() - started,
        }

    for step in range(starting_step + 1, args.steps + 1):
        current_min, current_max = _backbone_training_length_range(
            spec,
            step=step - 1,
            fixed_logical_length=fixed_logical_length,
        )
        if args.batch_shared_logical_length:
            batch_logical_length = int(
                torch.randint(
                    current_min,
                    current_max + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            batch_min = batch_logical_length
            batch_max = batch_logical_length
            batch_fixed_length = batch_logical_length
        else:
            batch_min = current_min
            batch_max = current_max
            batch_fixed_length = fixed_logical_length
        batch = generate_paper_batch(
            spec,
            batch_size=args.batch_size,
            min_length=batch_min,
            max_length=batch_max,
            fixed_length=batch_fixed_length,
            generator=generator,
        ).to(device)
        if args.supervision == "adaptive_step":
            selected_steps = batch.target_steps
            horizon = int(selected_steps.max())
        else:
            # A fixed-horizon control must keep the recurrent readout depth
            # fixed throughout the length curriculum.  Moving this horizon
            # with the curriculum gives the same short example incompatible
            # depth targets at different training times.
            horizon = (
                fixed_loop_count
                if fixed_loop_count is not None
                else (
                    (
                        fixed_logical_length
                        if fixed_logical_length is not None
                        else spec.train_max_length
                    )
                    + spec.step_offset
                )
            )
            selected_steps = torch.full_like(batch.target_steps, horizon)
        learning_rate = _learning_rate(
            base=args.learning_rate,
            step=step,
            total_steps=(args.schedule_total_steps or args.steps),
            scheduler_anchor_zero_index=scheduler_anchor_zero_index,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            trajectory = model.states(batch.inputs, steps=horizon)
            loss, logits = _adaptive_readout_loss(
                model=model,
                trajectory=trajectory,
                batch=batch,
                spec=spec,
                selected_steps=selected_steps,
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        optimizer.step()

        validation: dict[str, float] = {}
        if (
            (
                fixed_logical_length is not None
                or current_max == spec.train_max_length
            )
            and (step % args.eval_every == 0 or step == args.steps)
        ):
            validation = validate_backbone_endpoint(
                model=model,
                spec=spec,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                seed=args.seed + 900_001,
                device=device,
                logical_length=fixed_logical_length,
                fixed_loop_count=fixed_loop_count,
            )
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "exact_match": exact_match(logits.detach(), batch),
                "gradient_norm": float(grad_norm),
                "learning_rate": learning_rate,
                "curriculum_min_length": current_min,
                "curriculum_max_length": current_max,
                "batch_min_length": int(batch.lengths.min()),
                "batch_max_length": int(batch.lengths.max()),
                "maximum_selected_step": horizon,
                "elapsed_seconds": time.monotonic() - started,
                **validation,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            write_csv(out_dir / "history.csv", history)
            validation_accuracy = row.get("validation_exact_match")
            validation_loss = row.get("validation_loss")
            improved = bool(
                validation_accuracy is not None
                and (
                    validation_accuracy > best_accuracy
                    or (
                        validation_accuracy == best_accuracy
                        and validation_loss is not None
                        and validation_loss < best_loss
                    )
                )
            )
            if improved:
                best_accuracy = float(validation_accuracy)
                best_loss = float(validation_loss)
                best_step = step
                atomic_torch_save(
                    _model_payload(
                        model=model,
                        spec=spec,
                        supervision=args.supervision,
                        seed=args.seed,
                        step=step,
                        metrics=row,
                        amp_enabled=use_amp,
                        training_fixed_logical_length=fixed_logical_length,
                        training_fixed_loop_count=fixed_loop_count,
                        training_batch_shared_logical_length=(
                            args.batch_shared_logical_length
                        ),
                    ),
                    out_dir / "best.pt",
                )

        if step % args.checkpoint_every == 0:
            atomic_torch_save(
                _model_payload(
                    model=model,
                    spec=spec,
                    supervision=args.supervision,
                    seed=args.seed,
                    step=step,
                    metrics=history[-1] if history else {},
                    amp_enabled=use_amp,
                    training_fixed_logical_length=fixed_logical_length,
                    training_fixed_loop_count=fixed_loop_count,
                    training_batch_shared_logical_length=(
                        args.batch_shared_logical_length
                    ),
                    training_state=current_training_state(),
                ),
                out_dir / f"checkpoint_{step:06d}.pt",
            )

    final_metrics = history[-1]
    atomic_torch_save(
        _model_payload(
            model=model,
            spec=spec,
            supervision=args.supervision,
            seed=args.seed,
            step=args.steps,
            metrics=final_metrics,
            amp_enabled=use_amp,
            training_fixed_logical_length=fixed_logical_length,
            training_fixed_loop_count=fixed_loop_count,
            training_batch_shared_logical_length=(
                args.batch_shared_logical_length
            ),
            training_state=current_training_state(),
        ),
        out_dir / "final.pt",
    )
    if best_step == 0:
        atomic_torch_save(
            _model_payload(
                model=model,
                spec=spec,
                supervision=args.supervision,
                seed=args.seed,
                step=args.steps,
                metrics=final_metrics,
                amp_enabled=use_amp,
                training_fixed_logical_length=fixed_logical_length,
                training_fixed_loop_count=fixed_loop_count,
                training_batch_shared_logical_length=(
                    args.batch_shared_logical_length
                ),
            ),
            out_dir / "best.pt",
        )
        best_step = args.steps
        best_loss = float(final_metrics.get("validation_loss", final_metrics["loss"]))
        best_accuracy = float(final_metrics.get("validation_exact_match", -1.0))
    summary = {
        "status": "complete",
        "paper": "arXiv:2409.15647v5",
        "official_model_config": config == official_model_config(spec),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "task": asdict(spec),
        "model": asdict(config),
        "supervision": args.supervision,
        "training_precision": (
            "bfloat16_autocast" if use_amp else "fp32"
        ),
        "training_fixed_logical_length": fixed_logical_length,
        "training_fixed_loop_count": fixed_loop_count,
        "training_batch_shared_logical_length": bool(
            args.batch_shared_logical_length
        ),
        "training_batch_length_mode": (
            "one uniformly sampled logical length per optimizer batch"
            if args.batch_shared_logical_length
            else "independent logical length per example"
        ),
        "training_logical_length_support": (
            [fixed_logical_length, fixed_logical_length]
            if fixed_logical_length is not None
            else [1, spec.train_max_length]
        ),
        "loss_placement": (
            (
                "token-pooled answer CE at loop n over n sum digits and at "
                "loop n+1 over all n+1 arithmetic digits"
            )
            if spec.addition_answer_supervision == "progressive_carry"
            else "answer-region CE at each sample's T(n) only"
            if args.supervision == "adaptive_step"
            else (
                f"answer-region CE at one global fixed loop count "
                f"{fixed_loop_count}"
                if fixed_loop_count is not None
                else (
                    "answer-region CE at one global fixed horizon throughout "
                    "the length curriculum"
                )
            )
        ),
        "trained_loop_counts": (
            (
                f"paired loop n and loop n+1 readouts for logical lengths "
                f"1..{spec.train_max_length}"
            )
            if spec.addition_answer_supervision == "progressive_carry"
            else f"globally fixed at {fixed_loop_count} for logical length "
            f"{fixed_logical_length}"
            if fixed_loop_count is not None
            else (
                f"globally fixed at T({fixed_logical_length})="
                f"{fixed_logical_length + spec.step_offset}"
                if fixed_logical_length is not None
                else (
                    f"1..{spec.train_max_length + spec.step_offset} sample-dependent"
                    if args.supervision == "adaptive_step"
                    else f"globally fixed at {spec.train_max_length + spec.step_offset}"
                )
            )
        ),
        "shared_physical_block_layers": config.block_layers,
        "maximum_effective_training_depth": (
            config.block_layers
            * (
                fixed_loop_count
                if fixed_loop_count is not None
                else (
                    (fixed_logical_length + spec.step_offset)
                    if fixed_logical_length is not None
                    else (spec.train_max_length + spec.step_offset)
                )
            )
        ),
        "steps": args.steps,
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "initialized_from": (
            str(args.initialize_from)
            if args.initialize_from is not None
            else None
        ),
        "optimizer_resume_mode": (
            "full_state"
            if args.resume is not None
            else (
                "fresh_optimizer_from_checkpoint_weights"
                if args.initialize_from is not None
                else "new_training_run"
            )
        ),
        "warm_restart_rng_seed": (
            args.seed + args.initialize_rng_seed_offset + starting_step
            if args.initialize_from is not None
            else None
        ),
        "starting_step": starting_step,
        "learning_rate_schedule_total_steps": (
            args.schedule_total_steps or args.steps
        ),
        "released_scheduler_anchor_zero_index": scheduler_anchor_zero_index,
        "seed": args.seed,
        "best_step": best_step,
        "best_validation_loss_at_length_max": best_loss,
        "best_validation_exact_match_at_length_max": best_accuracy,
        "validation_logical_length": (
            fixed_logical_length
            if fixed_logical_length is not None
            else spec.train_max_length
        ),
        "peak_cuda_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_json_dump(summary, out_dir / "summary.json")
    return summary


@torch.no_grad()
def evaluate_trajectory(
    *,
    model: PaperLoopedTransformer,
    spec: PaperTaskSpec,
    controller: torch.nn.Module | None,
    variant: str,
    lengths: Iterable[int],
    batch_size: int,
    batches: int,
    maximum_step: int,
    controller_start_step: int | None,
    post_final_controller: bool = False,
    executor_off: bool,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    totals: dict[tuple[int, int], dict[str, float]] = {}
    for length in lengths:
        for _ in range(batches):
            batch = generate_paper_batch(
                spec,
                batch_size=batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            trajectory = model.iter_states(
                batch.inputs,
                steps=maximum_step,
                controller=controller,
                controller_start_step=controller_start_step,
                executor_off_after_start=executor_off,
            )
            for step_index, state in enumerate(trajectory, start=1):
                readout_state = (
                    controller(state)
                    if post_final_controller and controller is not None
                    else state
                )
                logits = model.decode(readout_state).float()
                key = (length, step_index)
                values = totals.setdefault(
                    key,
                    {
                        "correct": 0.0,
                        "confidence_loss": 0.0,
                        "answer_nll": 0.0,
                        "answer_entropy": 0.0,
                        "batches": 0.0,
                    },
                )
                values["correct"] += exact_match(logits, batch) * batch_size
                values["confidence_loss"] += predicted_confidence_loss(
                    logits, batch
                )
                values["answer_nll"] += answer_cross_entropy(logits, batch)
                values["answer_entropy"] += answer_predictive_entropy(
                    logits, batch
                )
                values["batches"] += 1
    rows: list[dict[str, Any]] = []
    for (length, step), values in sorted(totals.items()):
        rows.append(
            {
                "variant": variant,
                "length": length,
                "target_step": length + spec.step_offset,
                "step": step,
                "extra_steps_after_target": step - length - spec.step_offset,
                "exact_match": values["correct"] / (batch_size * batches),
                "predicted_confidence_ce": (
                    values["confidence_loss"] / values["batches"]
                ),
                "answer_nll": values["answer_nll"] / values["batches"],
                "answer_predictive_entropy": (
                    values["answer_entropy"] / values["batches"]
                ),
                "examples": batch_size * batches,
                "controller_start_step": controller_start_step,
                "executor_off": executor_off,
            }
        )
    return rows


def _curve_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    variants = sorted({str(row["variant"]) for row in rows})
    lengths = sorted({int(row["length"]) for row in rows})
    for variant in variants:
        summary[variant] = {}
        for length in lengths:
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["length"] == length
            ]
            if not selected:
                continue
            target_step = int(selected[0]["target_step"])
            at_target = next(
                (row for row in selected if row["step"] == target_step), None
            )
            post = [
                float(row["exact_match"])
                for row in selected
                if target_step < int(row["step"]) <= target_step + 32
            ]
            late = post[16:32]
            confidence_choice = min(
                selected, key=lambda row: float(row["predicted_confidence_ce"])
            )
            best = max(selected, key=lambda row: float(row["exact_match"]))
            local_window = [
                row
                for row in selected
                if target_step - 2 <= int(row["step"]) <= target_step + 8
            ]
            local_best = max(
                local_window, key=lambda row: float(row["exact_match"])
            )
            post_by_offset = {
                int(row["step"]) - target_step: float(row["exact_match"])
                for row in selected
                if target_step < int(row["step"]) <= target_step + 32
            }
            contiguous_survival: dict[str, int] = {}
            for threshold in (0.95, 0.90):
                survived = 0
                for offset in range(1, 33):
                    if post_by_offset.get(offset, 0.0) >= threshold:
                        survived = offset
                    else:
                        break
                contiguous_survival[f"{threshold:.2f}"] = survived
            summary[variant][str(length)] = {
                "target_step_exact_match": (
                    float(at_target["exact_match"]) if at_target else None
                ),
                "target_step_answer_nll": (
                    float(at_target["answer_nll"]) if at_target else None
                ),
                "target_step_predictive_entropy": (
                    float(at_target["answer_predictive_entropy"])
                    if at_target
                    else None
                ),
                "post_target_auc_1_32": sum(post) / len(post) if post else None,
                "late_auc_17_32": sum(late) / len(late) if late else None,
                "maximum_confidence_step": int(confidence_choice["step"]),
                "maximum_confidence_exact_match": float(
                    confidence_choice["exact_match"]
                ),
                "best_step": int(best["step"]),
                "best_exact_match": float(best["exact_match"]),
                "local_window": [target_step - 2, target_step + 8],
                "local_window_best_step": int(local_best["step"]),
                "local_window_best_offset": int(local_best["step"])
                - target_step,
                "local_window_best_exact_match": float(
                    local_best["exact_match"]
                ),
                "post_target_exact_match": {
                    str(offset): post_by_offset.get(offset)
                    for offset in (1, 2, 4, 8, 16, 32)
                },
                "contiguous_extra_loops_at_or_above": contiguous_survival,
            }
    return summary


def _telomere_disease_gate(
    *,
    anchor_target_accuracy: float | None,
    extension_target_accuracy: float | None,
    extension_local_best_accuracy: float | None,
    minimum_endpoint_accuracy: float,
    minimum_decline: float,
) -> dict[str, bool]:
    endpoint_is_healthy = bool(
        anchor_target_accuracy is not None
        and anchor_target_accuracy >= minimum_endpoint_accuracy
    )
    target_telomere_failure = bool(
        endpoint_is_healthy
        and extension_target_accuracy is not None
        and extension_target_accuracy
        <= anchor_target_accuracy - minimum_decline
    )
    nearby_executor_exhaustion = bool(
        endpoint_is_healthy
        and extension_local_best_accuracy is not None
        and extension_local_best_accuracy
        <= anchor_target_accuracy - minimum_decline
    )
    return {
        "passed": target_telomere_failure,
        "target_telomere_failure": target_telomere_failure,
        "nearby_executor_exhaustion": nearby_executor_exhaustion,
        "clock_drift_detected": bool(
            target_telomere_failure
            and extension_local_best_accuracy is not None
            and not nearby_executor_exhaustion
        ),
    }


def diagnose_backbone(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, payload = load_backbone(args.checkpoint, device=device)
    lengths = tuple(args.lengths or spec.test_lengths)
    maximum_step = args.maximum_step or max(
        max(lengths) + spec.step_offset + 32,
        spec.train_max_length + spec.step_offset + 32,
    )
    rows = evaluate_trajectory(
        model=model,
        spec=spec,
        controller=None,
        variant="raw",
        lengths=lengths,
        batch_size=args.batch_size,
        batches=args.batches,
        maximum_step=maximum_step,
        controller_start_step=None,
        executor_off=False,
        seed=args.seed,
        device=device,
    )
    curves = _curve_summary(rows)
    gate_length = spec.train_max_length
    gate_metrics = curves["raw"].get(str(gate_length), {})
    target_accuracy = gate_metrics.get("target_step_exact_match")
    late_auc = gate_metrics.get("late_auc_17_32")
    extension_length = args.extension_gate_length or min(
        2 * spec.train_max_length,
        max(lengths),
    )
    extension_metrics = curves["raw"].get(str(extension_length), {})
    extension_accuracy = extension_metrics.get("target_step_exact_match")
    extension_local_best = extension_metrics.get(
        "local_window_best_exact_match"
    )
    gate = _telomere_disease_gate(
        anchor_target_accuracy=target_accuracy,
        extension_target_accuracy=extension_accuracy,
        extension_local_best_accuracy=extension_local_best,
        minimum_endpoint_accuracy=args.minimum_endpoint_accuracy,
        minimum_decline=args.minimum_decline,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "trajectory.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": payload["step"],
        "task": asdict(spec),
        "model": asdict(model.config),
        "supervision": payload["supervision"],
        "loss_placement": payload["supervision"],
        "shared_physical_block_layers": spec.block_layers,
        "maximum_evaluated_step": maximum_step,
        "maximum_effective_evaluated_depth": maximum_step * spec.block_layers,
        "curves": curves,
        "disease_gate": {
            "anchor_length": gate_length,
            "extension_length": extension_length,
            "minimum_endpoint_accuracy": args.minimum_endpoint_accuracy,
            "minimum_extension_decline": args.minimum_decline,
            "anchor_target_step_exact_match": target_accuracy,
            "extension_target_step_exact_match": extension_accuracy,
            "extension_local_window": [
                extension_length + spec.step_offset - 2,
                extension_length + spec.step_offset + 8,
            ],
            "extension_local_window_best_exact_match": extension_local_best,
            "extension_local_window_best_step": extension_metrics.get(
                "local_window_best_step"
            ),
            "clock_drift_boundary": (
                "failure at registered T(n) with high local-window accuracy "
                "is timing/halting drift, not an exhausted executor"
            ),
            "post_answer_late_auc_17_32": late_auc,
            "post_answer_role": (
                "halting/stability diagnostic only; not sufficient evidence "
                "of computational aging"
            ),
            **gate,
        },
    }
    atomic_json_dump(summary, out_dir / "summary.json")
    return summary


@dataclass(frozen=True)
class ControllerStage:
    name: str
    rounds: int
    batches_per_round: int
    batch_size: int
    extra_steps: tuple[int, ...]
    learning_rate: float
    data_seed: int


@dataclass(frozen=True)
class ControllerTrainingCase:
    logical_extra_steps: int
    overloop_steps: int

    @property
    def controlled_steps(self) -> int:
        return self.logical_extra_steps + self.overloop_steps


@dataclass(frozen=True)
class LogicalControllerPlan:
    logical_length: int
    target_step: int
    anchor_step: int
    controlled_steps: int


CONTROLLER_STAGES = (
    ControllerStage("ce_extend_8", 8, 4, 64, (2, 4, 8), 3e-4, 240001),
    ControllerStage("ce_extend_16", 12, 4, 48, (4, 8, 16), 1.5e-4, 241001),
    ControllerStage("ce_extend_32", 12, 4, 32, (8, 16, 32), 9e-5, 242001),
    ControllerStage("ce_extend_64", 12, 4, 24, (16, 32, 64), 9e-5, 243001),
)


def _identity_long_warmup_stages(
    maximum_logical_length: int,
    *,
    minimum_logical_length: int | None = None,
) -> tuple[ControllerStage, ...]:
    if maximum_logical_length < 3:
        raise ValueError("identity_long_warmup requires logical max >= 3")
    if minimum_logical_length is None:
        minimum_logical_length = math.ceil(maximum_logical_length / 2)
    if not 1 <= minimum_logical_length <= maximum_logical_length:
        raise ValueError("invalid identity warmup logical interval")
    first_stage_maximum = max(
        minimum_logical_length,
        math.ceil(3 * maximum_logical_length / 5),
    )
    second_stage_maximum = max(
        minimum_logical_length,
        math.ceil(4 * maximum_logical_length / 5),
    )
    peak_learning_rate = 2e-5
    return (
        ControllerStage(
            (
                f"identity_warmup_{minimum_logical_length}_"
                f"{first_stage_maximum}"
            ),
            64,
            4,
            64,
            tuple(
                range(minimum_logical_length, first_stage_maximum + 1)
            ),
            peak_learning_rate,
            250001,
        ),
        ControllerStage(
            (
                f"identity_warmup_{minimum_logical_length}_"
                f"{second_stage_maximum}"
            ),
            128,
            4,
            48,
            tuple(
                range(minimum_logical_length, second_stage_maximum + 1)
            ),
            peak_learning_rate,
            251001,
        ),
        ControllerStage(
            (
                f"identity_warmup_{minimum_logical_length}_"
                f"{maximum_logical_length}"
            ),
            256,
            4,
            32,
            tuple(
                range(minimum_logical_length, maximum_logical_length + 1)
            ),
            peak_learning_rate,
            252001,
        ),
    )


def _identity_tail_warmup_stages(
    maximum_logical_length: int,
) -> tuple[ControllerStage, ...]:
    """Keep the identity schedule but concentrate CE on the in-range boundary."""
    if maximum_logical_length < 3:
        raise ValueError("identity_tail_warmup requires logical max >= 3")
    stage_minima = (
        math.ceil(7 * maximum_logical_length / 8),
        math.ceil(3 * maximum_logical_length / 4),
        math.ceil(maximum_logical_length / 2),
    )
    peak_learning_rate = 2e-5
    templates = (
        (64, 4, 64, 250001),
        (128, 4, 48, 251001),
        (256, 4, 32, 252001),
    )
    return tuple(
        ControllerStage(
            name=(
                f"identity_tail_warmup_{minimum_logical_length}_"
                f"{maximum_logical_length}"
            ),
            rounds=rounds,
            batches_per_round=batches_per_round,
            batch_size=batch_size,
            extra_steps=tuple(
                range(minimum_logical_length, maximum_logical_length + 1)
            ),
            learning_rate=peak_learning_rate,
            data_seed=data_seed,
        )
        for minimum_logical_length, (
            rounds,
            batches_per_round,
            batch_size,
            data_seed,
        ) in zip(stage_minima, templates, strict=True)
    )


def _controller_training_budget(
    stages: Sequence[ControllerStage],
    *,
    dense_stage_count: int,
    final_parameterization: str = "diagonal_low_rank",
) -> dict[str, int]:
    if dense_stage_count < 0:
        raise ValueError("dense_stage_count must be non-negative")
    if final_parameterization not in {"diagonal_low_rank", "dense_affine"}:
        raise ValueError("unsupported final controller parameterization")
    if final_parameterization == "dense_affine" and dense_stage_count != 0:
        raise ValueError("dense affine training has no separate dense warm start")
    dense_stages = stages[:dense_stage_count]
    if final_parameterization == "dense_affine":
        dense_updates = sum(
            stage.rounds * stage.batches_per_round for stage in stages
        )
        low_rank_updates = 0
        dense_examples = sum(
            stage.rounds * stage.batches_per_round * stage.batch_size
            for stage in stages
        )
        low_rank_examples = 0
    else:
        dense_updates = sum(
            stage.rounds * stage.batches_per_round for stage in dense_stages
        )
        low_rank_updates = sum(
            stage.rounds * stage.batches_per_round for stage in stages
        )
        dense_examples = sum(
            stage.rounds * stage.batches_per_round * stage.batch_size
            for stage in dense_stages
        )
        low_rank_examples = sum(
            stage.rounds * stage.batches_per_round * stage.batch_size
            for stage in stages
        )
    return {
        "dense_optimizer_updates": dense_updates,
        "low_rank_optimizer_updates": low_rank_updates,
        "total_optimizer_updates": dense_updates + low_rank_updates,
        "dense_training_examples": dense_examples,
        "low_rank_training_examples": low_rank_examples,
        "total_training_examples": dense_examples + low_rank_examples,
    }


def _backbone_training_metadata(
    spec: PaperTaskSpec, *, supervision: str
) -> dict[str, Any]:
    maximum_target_step = spec.train_max_length + spec.step_offset
    if spec.addition_answer_supervision == "progressive_carry":
        trained_counts = (
            "paired loop n and loop n+1 readouts for logical lengths "
            f"1..{spec.train_max_length}"
        )
    else:
        trained_counts = (
            f"1..{maximum_target_step} sample-dependent"
            if supervision == "adaptive_step"
            else f"globally fixed at {maximum_target_step}"
        )
    return {
        "backbone_trained_loop_counts": trained_counts,
        "maximum_backbone_effective_training_depth": (
            spec.block_layers * maximum_target_step
        ),
    }


def _loaded_backbone_training_metadata(
    spec: PaperTaskSpec,
    *,
    config: PaperModelConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Describe the actual checkpoint protocol, including custom controls."""
    supervision = str(payload["supervision"])
    fixed_length = payload.get("training_fixed_logical_length")
    fixed_loop_count = payload.get("training_fixed_loop_count")
    if spec.addition_answer_supervision == "progressive_carry":
        maximum_target_step = (
            int(fixed_length) + 1
            if fixed_length is not None
            else spec.train_max_length + 1
        )
        length_description = (
            str(int(fixed_length))
            if fixed_length is not None
            else f"1..{spec.train_max_length}"
        )
        trained_counts = (
            "paired loop n and loop n+1 readouts for logical lengths "
            f"{length_description}"
        )
        loss_placement = (
            "token-pooled answer CE at loop n over n sum digits and at loop "
            "n+1 over all n+1 arithmetic digits"
        )
    elif fixed_loop_count is not None:
        maximum_target_step = int(fixed_loop_count)
        trained_counts = f"globally fixed at {maximum_target_step}"
        loss_placement = (
            "answer-region CE at one global fixed loop count "
            f"{maximum_target_step}"
        )
    elif fixed_length is not None:
        maximum_target_step = int(fixed_length) + spec.step_offset
        trained_counts = f"globally fixed at {maximum_target_step}"
        loss_placement = (
            "answer-region CE at the registered endpoint "
            f"T({int(fixed_length)})={maximum_target_step}"
        )
    else:
        maximum_target_step = spec.train_max_length + spec.step_offset
        trained_counts = (
            f"1..{maximum_target_step} sample-dependent"
            if supervision == "adaptive_step"
            else f"globally fixed at {maximum_target_step}"
        )
        loss_placement = (
            "answer-region CE at each sample's T(n) only"
            if supervision == "adaptive_step"
            else "answer-region CE at one global fixed horizon"
        )
    return {
        "backbone_training_fixed_logical_length": fixed_length,
        "backbone_training_fixed_loop_count": fixed_loop_count,
        "backbone_trained_loop_counts": trained_counts,
        "backbone_loss_placement": loss_placement,
        "shared_physical_block_layers": config.block_layers,
        "maximum_backbone_effective_training_depth": (
            config.block_layers * maximum_target_step
        ),
    }


def _logical_controller_stages(
    maximum_logical_length: int,
    *,
    minimum_logical_length: int = 2,
) -> tuple[ControllerStage, ...]:
    if not 1 <= minimum_logical_length <= maximum_logical_length:
        raise ValueError("invalid logical-range J training interval")
    stage_maxima = (
        max(minimum_logical_length, math.ceil(maximum_logical_length / 8)),
        max(minimum_logical_length, math.ceil(maximum_logical_length / 4)),
        max(minimum_logical_length, math.ceil(maximum_logical_length / 2)),
        maximum_logical_length,
    )
    return tuple(
        ControllerStage(
            name=(
                f"ce_logical_{minimum_logical_length}_{stage_maximum}"
            ),
            rounds=template.rounds,
            batches_per_round=template.batches_per_round,
            batch_size=template.batch_size,
            extra_steps=tuple(
                range(minimum_logical_length, stage_maximum + 1)
            ),
            learning_rate=template.learning_rate,
            data_seed=template.data_seed,
        )
        for template, stage_maximum in zip(
            CONTROLLER_STAGES, stage_maxima, strict=True
        )
    )


def _controller_anchor_step(
    spec: PaperTaskSpec,
    curriculum_mode: str,
    controller_anchor_step: int | None = None,
) -> int:
    if controller_anchor_step is not None:
        if curriculum_mode != "logical_range":
            raise ValueError("custom controller anchor requires logical_range")
        if controller_anchor_step < 0:
            raise ValueError("controller anchor must be non-negative")
        return controller_anchor_step
    if curriculum_mode == "logical_range":
        # J is an inter-loop map: loop 1 is the unchanged backbone, and J is
        # applied before every executor call beginning with loop 2.
        return 1
    return spec.train_max_length + spec.step_offset


def _controller_loss_description(
    curriculum_mode: str,
    *,
    spec: PaperTaskSpec | None = None,
) -> str:
    if (
        spec is not None
        and spec.addition_answer_supervision == "progressive_carry"
    ):
        return (
            "token-pooled task CE at loop n over n sum digits and at loop "
            "n+1 over all n+1 arithmetic digits; no state loss"
        )
    if curriculum_mode == "logical_range":
        return (
            "answer-region task CE only at each sampled logical length's "
            "registered final T(n); no intermediate or state loss"
        )
    return (
        "answer-region task CE only at each sampled case's final controlled "
        "depth; no intermediate or state loss"
    )


def _logical_controller_plan(
    spec: PaperTaskSpec,
    logical_length: int,
    *,
    controller_anchor_step: int | None = None,
) -> LogicalControllerPlan:
    if logical_length < 1:
        raise ValueError("logical length must be positive")
    anchor_step = _controller_anchor_step(
        spec,
        "logical_range",
        controller_anchor_step,
    )
    target_step = logical_length + spec.step_offset
    controlled_steps = target_step - anchor_step
    if controlled_steps < 0:
        raise ValueError("registered target step precedes the J anchor")
    return LogicalControllerPlan(
        logical_length=logical_length,
        target_step=target_step,
        anchor_step=anchor_step,
        controlled_steps=controlled_steps,
    )


def _task_logical_controller_stages(
    spec: PaperTaskSpec,
    maximum_logical_length: int,
    *,
    controller_anchor_step: int | None = None,
) -> tuple[ControllerStage, ...]:
    sampled_lengths = tuple(
        logical_length
        for logical_length in range(1, maximum_logical_length + 1)
        if _logical_controller_plan(
            spec,
            logical_length,
            controller_anchor_step=controller_anchor_step,
        ).controlled_steps
        > 0
    )
    if not sampled_lengths:
        raise ValueError("logical-range curriculum never applies J")
    if sampled_lengths != tuple(
        range(sampled_lengths[0], maximum_logical_length + 1)
    ):
        raise ValueError("gradient-bearing logical lengths must be contiguous")
    return _logical_controller_stages(
        maximum_logical_length,
        minimum_logical_length=sampled_lengths[0],
    )


def _sampled_logical_lengths(
    stages: Sequence[ControllerStage],
) -> tuple[int, ...]:
    """Recover the exact logical lengths visited by the staged update order."""
    return tuple(sorted(_logical_length_example_counts(stages)))


def _logical_length_example_counts(
    stages: Sequence[ControllerStage],
) -> dict[int, int]:
    """Count task-CE examples seen at each logical length."""
    sampled: dict[int, int] = {}
    for stage in stages:
        if not stage.extra_steps:
            raise ValueError("logical-range stage has no candidate lengths")
        updates = stage.rounds * stage.batches_per_round
        for index in range(updates):
            logical_length = int(
                stage.extra_steps[index % len(stage.extra_steps)]
            )
            sampled[logical_length] = (
                sampled.get(logical_length, 0) + stage.batch_size
            )
    return dict(sorted(sampled.items()))


def _controller_learning_rate(
    *,
    peak: float,
    update: int,
    total_updates: int,
    warmup_updates: int,
    final_ratio: float,
    schedule: str = "cosine",
    stable_updates: int = 0,
) -> float:
    if peak <= 0:
        raise ValueError("controller peak learning rate must be positive")
    if not 1 <= update <= total_updates:
        raise ValueError("controller update must lie within the training run")
    if not 0 <= warmup_updates <= total_updates:
        raise ValueError("invalid controller warmup length")
    if not 0 <= stable_updates <= total_updates - warmup_updates:
        raise ValueError("invalid controller stable length")
    if not 0 < final_ratio <= 1:
        raise ValueError("controller final LR ratio must lie in (0, 1]")
    if schedule not in {"cosine", "wsd"}:
        raise ValueError(f"unsupported controller LR schedule: {schedule}")
    if schedule == "cosine" and stable_updates:
        raise ValueError("cosine controller schedule has no stable phase")
    if warmup_updates and update <= warmup_updates:
        return peak * update / warmup_updates
    if schedule == "wsd" and update <= warmup_updates + stable_updates:
        return peak
    decay_updates = total_updates - warmup_updates - stable_updates
    if decay_updates == 0:
        return peak
    decay_index = update - warmup_updates - stable_updates
    progress = min(1.0, max(0.0, decay_index / decay_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (final_ratio + (1.0 - final_ratio) * cosine)


def _controlled_ce_batch(
    *,
    model: PaperLoopedTransformer,
    controller: torch.nn.Module,
    batch: PaperBatch,
    anchor_step: int,
    controlled_steps: int,
    ce_temperature: float = 1.0,
    post_final_controller: bool = False,
    progressive_supervision: bool = False,
) -> tuple[torch.Tensor, float]:
    if ce_temperature <= 0:
        raise ValueError("controller CE temperature must be positive")
    with torch.no_grad():
        state = torch.zeros(
            batch.inputs.shape[0],
            batch.inputs.shape[1],
            model.config.d_model,
            device=batch.inputs.device,
        )
        for step_index in range(1, anchor_step + 1):
            embedded = model.input_embeddings(
                batch.inputs, step_index=step_index
            )
            state = model.recurrent_step(state, embedded)
    final_step = anchor_step + controlled_steps
    prefix_state = (
        state if progressive_supervision and final_step - 1 == anchor_step else None
    )
    for step_index in range(
        anchor_step + 1, anchor_step + controlled_steps + 1
    ):
        embedded = model.input_embeddings(batch.inputs, step_index=step_index)
        state = controller(state)
        state = model.recurrent_step(state, embedded)
        if progressive_supervision and step_index == final_step - 1:
            prefix_state = state
    if post_final_controller:
        state = controller(state)
    logits = model.decode(state).float()
    if progressive_supervision:
        if prefix_state is None:
            raise ValueError(
                "progressive controller supervision needs a preceding loop"
            )
        if not torch.equal(
            batch.target_steps,
            torch.full_like(batch.target_steps, final_step),
        ):
            raise ValueError(
                "progressive controller supervision must stop at T(n)=n+1"
            )
        prefix_logits = model.decode(prefix_state).float()
        prefix_mask = _progressive_addition_prefix_mask(batch)
        loss = F.cross_entropy(
            torch.cat(
                (
                    prefix_logits[prefix_mask],
                    logits[batch.answer_mask],
                ),
                dim=0,
            )
            / ce_temperature,
            torch.cat(
                (
                    batch.targets[prefix_mask],
                    batch.targets[batch.answer_mask],
                ),
                dim=0,
            ),
        )
    else:
        loss = masked_cross_entropy(logits / ce_temperature, batch)
    return (
        loss,
        exact_match(logits.detach(), batch),
    )


def _concatenate_paper_batches(
    first: PaperBatch, second: PaperBatch
) -> PaperBatch:
    return PaperBatch(
        inputs=torch.cat((first.inputs, second.inputs), dim=0),
        targets=torch.cat((first.targets, second.targets), dim=0),
        answer_mask=torch.cat((first.answer_mask, second.answer_mask), dim=0),
        lengths=torch.cat((first.lengths, second.lengths), dim=0),
        target_steps=torch.cat((first.target_steps, second.target_steps), dim=0),
    )


def _controller_training_batch(
    *,
    spec: PaperTaskSpec,
    batch_size: int,
    extra_steps: int,
    curriculum_mode: str,
    generator: torch.Generator,
) -> PaperBatch:
    target_length = spec.train_max_length + extra_steps
    if curriculum_mode == "extension":
        return generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=target_length,
            max_length=target_length,
            fixed_length=target_length,
            generator=generator,
        )
    if curriculum_mode == "overloop":
        return generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=spec.train_max_length,
            max_length=target_length,
            fixed_length=spec.train_max_length,
            generator=generator,
        )
    if curriculum_mode != "mixed":
        raise ValueError(f"unsupported controller curriculum: {curriculum_mode}")
    extension_size = (batch_size + 1) // 2
    overloop_size = batch_size - extension_size
    if overloop_size == 0:
        raise ValueError("mixed controller curriculum requires batch_size >= 2")
    extension = generate_paper_batch(
        spec,
        batch_size=extension_size,
        min_length=target_length,
        max_length=target_length,
        fixed_length=target_length,
        generator=generator,
    )
    overloop = generate_paper_batch(
        spec,
        batch_size=overloop_size,
        min_length=spec.train_max_length,
        max_length=target_length,
        fixed_length=spec.train_max_length,
        generator=generator,
    )
    return _concatenate_paper_batches(extension, overloop)


def _grid_controller_training_case(
    extra_steps: Sequence[int], batch_index: int
) -> ControllerTrainingCase:
    """Pair logical length extension with post-solution overloop depth.

    Every generated sequence has its true logical length, so an overloop case
    changes only recurrent depth and never appends synthetic padding tokens.
    The four cases jointly cover the training boundary, continued computation,
    and stability after solving at two extended lengths.
    """
    if len(extra_steps) != 3:
        raise ValueError("grid curriculum requires three stage extra-step values")
    short, medium, long = map(int, extra_steps)
    cases = (
        ControllerTrainingCase(0, short),
        ControllerTrainingCase(short, 0),
        ControllerTrainingCase(medium, short),
        ControllerTrainingCase(long, medium),
    )
    return cases[batch_index % len(cases)]


def _train_controller_stages(
    *,
    model: PaperLoopedTransformer,
    spec: PaperTaskSpec,
    controller: torch.nn.Module,
    stages: Sequence[ControllerStage],
    device: torch.device,
    anchor_step: int,
    grad_clip: float,
    diagonal_learning_rate_multiplier: float,
    curriculum_mode: str,
    ce_temperature: float = 1.0,
    warmup_updates: int = 0,
    final_learning_rate_ratio: float = 1.0,
    learning_rate_schedule: str = "cosine",
    stable_updates: int = 0,
    post_final_controller: bool = False,
    controller_supervision: str = "full_answer",
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, torch.nn.Module], None] | None = None,
) -> list[dict[str, Any]]:
    if grad_clip <= 0:
        raise ValueError("controller gradient clip must be positive")
    if diagonal_learning_rate_multiplier <= 0:
        raise ValueError("diagonal learning-rate multiplier must be positive")
    if ce_temperature <= 0:
        raise ValueError("controller CE temperature must be positive")
    if checkpoint_every < 0:
        raise ValueError("controller checkpoint interval cannot be negative")
    if checkpoint_every > 0 and checkpoint_callback is None:
        raise ValueError("controller checkpoint interval requires a callback")
    if checkpoint_callback is not None and checkpoint_every == 0:
        raise ValueError("controller checkpoint callback requires an interval")
    total_updates = sum(
        stage.rounds * stage.batches_per_round for stage in stages
    )
    if total_updates < 1:
        raise ValueError("controller training requires at least one update")
    use_global_schedule = (
        warmup_updates > 0
        or final_learning_rate_ratio != 1.0
        or learning_rate_schedule != "cosine"
        or stable_updates > 0
    )
    if use_global_schedule:
        peak_learning_rates = {stage.learning_rate for stage in stages}
        if len(peak_learning_rates) != 1:
            raise ValueError(
                "scheduled controller stages must share one peak learning rate"
            )
        peak_learning_rate = next(iter(peak_learning_rates))
        # Validate the whole schedule before allocating optimizer state.
        _controller_learning_rate(
            peak=peak_learning_rate,
            update=1,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            final_ratio=final_learning_rate_ratio,
            schedule=learning_rate_schedule,
            stable_updates=stable_updates,
        )
    else:
        peak_learning_rate = stages[0].learning_rate

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    controller.train()
    rows: list[dict[str, Any]] = []

    def make_optimizer(learning_rate: float) -> torch.optim.Optimizer:
        parameter_groups: list[dict[str, Any]] | Iterable[torch.nn.Parameter]
        if isinstance(controller, DiagonalLowRankController):
            parameter_groups = [
                {"params": [controller.A, controller.B, controller.bias]},
                {
                    "params": [controller.diagonal],
                    "lr": learning_rate * diagonal_learning_rate_multiplier,
                },
            ]
        else:
            parameter_groups = controller.parameters()
        return torch.optim.AdamW(
            parameter_groups, lr=learning_rate, weight_decay=0.0
        )

    shared_optimizer = (
        make_optimizer(peak_learning_rate) if use_global_schedule else None
    )
    global_update = 0
    for stage in stages:
        optimizer = shared_optimizer or make_optimizer(stage.learning_rate)
        for round_index in range(1, stage.rounds + 1):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stage.data_seed + 1000 * round_index)
            loss_total = 0.0
            accuracy_total = 0.0
            grad_total = 0.0
            for batch_index in range(stage.batches_per_round):
                if curriculum_mode == "logical_range":
                    value_index = (
                        (round_index - 1) * stage.batches_per_round
                        + batch_index
                    ) % len(stage.extra_steps)
                    logical_length = stage.extra_steps[value_index]
                    if controller_supervision == "addition_final_carry":
                        batch = generate_balanced_addition_final_carry_batch(
                            spec,
                            batch_size=stage.batch_size,
                            logical_length=logical_length,
                            generator=generator,
                        ).to(device)
                    else:
                        batch = generate_paper_batch(
                            spec,
                            batch_size=stage.batch_size,
                            min_length=logical_length,
                            max_length=logical_length,
                            fixed_length=logical_length,
                            generator=generator,
                        ).to(device)
                    plan = _logical_controller_plan(
                        spec,
                        logical_length,
                        controller_anchor_step=anchor_step,
                    )
                    if plan.anchor_step != anchor_step:
                        raise ValueError("logical-range J anchor is inconsistent")
                    controlled_steps = plan.controlled_steps
                    training_case = ControllerTrainingCase(
                        logical_extra_steps=0,
                        overloop_steps=0,
                    )
                else:
                    extra_steps = stage.extra_steps[
                        batch_index % len(stage.extra_steps)
                    ]
                    logical_length = spec.train_max_length + extra_steps
                if curriculum_mode == "grid":
                    training_case = _grid_controller_training_case(
                        stage.extra_steps, batch_index
                    )
                    target_length = (
                        spec.train_max_length
                        + training_case.logical_extra_steps
                    )
                    batch = generate_paper_batch(
                        spec,
                        batch_size=stage.batch_size,
                        min_length=target_length,
                        max_length=target_length,
                        fixed_length=target_length,
                        generator=generator,
                    ).to(device)
                    controlled_steps = training_case.controlled_steps
                elif curriculum_mode != "logical_range":
                    training_case = ControllerTrainingCase(
                        logical_extra_steps=(
                            extra_steps if curriculum_mode == "extension" else 0
                        ),
                        overloop_steps=(
                            extra_steps if curriculum_mode == "overloop" else 0
                        ),
                    )
                    batch = _controller_training_batch(
                        spec=spec,
                        batch_size=stage.batch_size,
                        extra_steps=extra_steps,
                        curriculum_mode=curriculum_mode,
                        generator=generator,
                    ).to(device)
                    controlled_steps = extra_steps
                global_update += 1
                if use_global_schedule:
                    live_learning_rate = _controller_learning_rate(
                        peak=peak_learning_rate,
                        update=global_update,
                        total_updates=total_updates,
                        warmup_updates=warmup_updates,
                        final_ratio=final_learning_rate_ratio,
                        schedule=learning_rate_schedule,
                        stable_updates=stable_updates,
                    )
                    optimizer.param_groups[0]["lr"] = live_learning_rate
                    if isinstance(controller, DiagonalLowRankController):
                        optimizer.param_groups[1]["lr"] = (
                            live_learning_rate
                            * diagonal_learning_rate_multiplier
                        )
                else:
                    live_learning_rate = stage.learning_rate
                optimizer.zero_grad(set_to_none=True)
                loss, accuracy = _controlled_ce_batch(
                    model=model,
                    controller=controller,
                    batch=batch,
                    anchor_step=anchor_step,
                    controlled_steps=controlled_steps,
                    ce_temperature=ce_temperature,
                    post_final_controller=post_final_controller,
                    progressive_supervision=(
                        spec.addition_answer_supervision
                        == "progressive_carry"
                    ),
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    controller.parameters(), grad_clip
                )
                optimizer.step()
                if checkpoint_callback is not None and (
                    global_update % checkpoint_every == 0
                    or global_update == total_updates
                ):
                    checkpoint_callback(global_update, controller)
                loss_total += float(loss.detach())
                accuracy_total += accuracy
                grad_total += float(grad_norm)
            row = {
                "stage": stage.name,
                "round": round_index,
                "task_ce": loss_total / stage.batches_per_round,
                "mean_final_exact_match": (
                    accuracy_total / stage.batches_per_round
                ),
                "mean_preclip_gradient_norm": grad_total / stage.batches_per_round,
                "extra_steps": "/".join(map(str, stage.extra_steps)),
                "maximum_training_length": int(batch.lengths.max()),
                "logical_training_length": int(batch.lengths.max()),
                "training_axis": (
                    "logical_length"
                    if curriculum_mode == "logical_range"
                    else "post_anchor_extension"
                ),
                "logical_extra_steps": training_case.logical_extra_steps,
                "overloop_steps": training_case.overloop_steps,
                "controlled_steps": controlled_steps,
                "effective_supervision_depth": anchor_step + controlled_steps,
                "learning_rate": live_learning_rate,
                "global_update": global_update,
                "state_loss_weight": 0.0,
                "hidden_state_targets": False,
                "controller_curriculum": curriculum_mode,
                "controller_ce_temperature": ce_temperature,
                "controller_lr_schedule": learning_rate_schedule,
                "controller_stable_updates": stable_updates,
                "post_final_controller": post_final_controller,
                "controller_supervision": controller_supervision,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    controller.eval()
    return rows


def train_controller(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.addition_answer_supervision == "progressive_carry" and (
        args.controller_curriculum != "logical_range"
        or args.controller_supervision != "full_answer"
    ):
        raise ValueError(
            "progressive_carry J training requires logical_range with its "
            "paired full-answer supervision"
        )
    if args.controller_supervision == "addition_final_carry" and (
        spec.name != "addition" or args.controller_curriculum != "logical_range"
    ):
        raise ValueError(
            "addition_final_carry requires Addition with logical_range curriculum"
        )
    if args.diagnosis is not None and not args.force:
        diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
        if not diagnosis["disease_gate"]["passed"]:
            raise RuntimeError("backbone did not pass the preregistered disease gate")
    set_seed(args.seed)
    if args.controller_curriculum == "logical_range":
        if args.controller_logical_max_length is None:
            raise ValueError(
                "logical_range requires --controller-logical-max-length"
            )
        if args.controller_training_profile in {
            "identity_long_warmup",
            "identity_tail_warmup",
        }:
            warmup_initializations = {"identity"}
            if args.controller_parameterization == "dense_affine":
                warmup_initializations.add("dense_iid")
            if args.controller_initialization not in warmup_initializations:
                raise ValueError(
                    "identity warmup profiles require a near-identity "
                    "initialization"
                )
            if args.controller_training_profile == "identity_long_warmup":
                stage_templates = _identity_long_warmup_stages(
                    args.controller_logical_max_length,
                    minimum_logical_length=(
                        args.controller_logical_min_length
                    ),
                )
            else:
                if args.controller_logical_min_length is not None:
                    raise ValueError(
                        "explicit logical minimum is supported only by "
                        "identity_long_warmup"
                    )
                stage_templates = _identity_tail_warmup_stages(
                    args.controller_logical_max_length
                )
        else:
            if args.controller_logical_min_length is not None:
                raise ValueError(
                    "explicit logical minimum requires identity_long_warmup"
                )
            stage_templates = _task_logical_controller_stages(
                spec,
                args.controller_logical_max_length,
                controller_anchor_step=args.controller_anchor_step,
            )
    else:
        if args.controller_logical_max_length is not None:
            raise ValueError(
                "--controller-logical-max-length is only valid for logical_range"
            )
        if args.controller_logical_min_length is not None:
            raise ValueError(
                "--controller-logical-min-length is only valid for logical_range"
            )
        stage_templates = CONTROLLER_STAGES
        if args.controller_training_profile != "legacy":
            raise ValueError(
                "identity warmup profiles require the logical_range curriculum"
            )
    if args.stage_round_multiplier < 1:
        raise ValueError("stage round multiplier must be at least one")
    stages = tuple(
        ControllerStage(
            name=stage.name,
            rounds=args.stage_round_multiplier
            * (
                min(stage.rounds, args.stage_round_limit)
                if args.stage_round_limit is not None
                else stage.rounds
            ),
            batches_per_round=stage.batches_per_round,
            batch_size=stage.batch_size,
            extra_steps=stage.extra_steps,
            learning_rate=stage.learning_rate * args.learning_rate_multiplier,
            data_seed=stage.data_seed,
        )
        for stage in stage_templates
    )
    if any(stage.rounds < 1 for stage in stages):
        raise ValueError("controller stages must contain at least one round")
    sampled_logical_lengths = (
        _sampled_logical_lengths(stages)
        if args.controller_curriculum == "logical_range"
        else None
    )
    logical_length_example_counts = (
        _logical_length_example_counts(stages)
        if args.controller_curriculum == "logical_range"
        else None
    )
    anchor_step = _controller_anchor_step(
        spec,
        args.controller_curriculum,
        args.controller_anchor_step,
    )
    for logical_length in sampled_logical_lengths or ():
        _logical_controller_plan(
            spec,
            logical_length,
            controller_anchor_step=anchor_step,
        )

    dense: DenseAffineController | None = None
    dense_stages: tuple[ControllerStage, ...] = ()
    dense_rows: list[dict[str, Any]] = []
    if args.controller_parameterization == "dense_affine":
        if args.controller_initialization not in {"identity", "dense_iid"}:
            raise ValueError(
                "dense affine controllers require identity or dense_iid "
                "initialization"
            )
        if args.dense_stage_count != 0:
            raise ValueError(
                "dense affine training requires --dense-stage-count 0"
            )
        controller: torch.nn.Module = DenseAffineController(
            model.config.d_model
        ).to(device)
        if args.controller_initialization == "dense_iid":
            residual_rms = (
                args.dense_iid_residual_rms
                if args.dense_iid_off_diagonal_std is None
                else args.dense_iid_off_diagonal_std
                * math.sqrt(model.config.d_model - 1)
            )
            iid_metadata = controller.initialize_iid_identity(
                residual_rms=residual_rms,
                bias_rms=args.dense_iid_bias_rms,
                seed=args.seed,
            )
        else:
            iid_metadata = {
                "requested_residual_rms": 0.0,
                "requested_bias_rms": 0.0,
                "off_diagonal_entry_std": 0.0,
                "realized_off_diagonal_rms_per_output": 0.0,
                "realized_bias_rms": 0.0,
                "maximum_diagonal_error": 0.0,
            }
        initialization_metadata: dict[str, float | None] = {
            "retained_dense_residual_energy": None,
            "dense_to_factor_max_abs_error": None,
            **iid_metadata,
        }
    else:
        if args.controller_initialization == "dense_iid":
            raise ValueError(
                "dense_iid initialization requires dense_affine parameterization"
            )
        if args.controller_initialization == "identity":
            if args.dense_stage_count != 0:
                raise ValueError(
                    "identity initialization requires --dense-stage-count 0"
                )
        else:
            if args.dense_stage_count < 1:
                raise ValueError(
                    "dense_svd initialization requires a positive dense stage count"
                )
            dense = DenseAffineController(model.config.d_model).to(device)
            dense_stages = stages[: args.dense_stage_count]
        dense_rows = (
            _train_controller_stages(
                model=model,
                spec=spec,
                controller=dense,
                stages=dense_stages,
                device=device,
                anchor_step=anchor_step,
                grad_clip=args.grad_clip,
                diagonal_learning_rate_multiplier=1.0,
                curriculum_mode=args.controller_curriculum,
                ce_temperature=args.controller_ce_temperature,
                post_final_controller=args.controller_post_final_j,
                controller_supervision=args.controller_supervision,
            )
            if dense is not None
            else []
        )
        controller, initialization_metadata = (
            _initialize_diagonal_low_rank_controller(
                dimension=model.config.d_model,
                rank=args.rank,
                initialization=args.controller_initialization,
                dense=dense,
                gauge_seed=args.seed,
                device=device,
            )
        )
    dense_logical_length_example_counts = (
        _logical_length_example_counts(dense_stages)
        if args.controller_curriculum == "logical_range" and dense_stages
        else None
    )
    training_budget = _controller_training_budget(
        stages,
        dense_stage_count=args.dense_stage_count,
        final_parameterization=args.controller_parameterization,
    )
    retained_energy = initialization_metadata[
        "retained_dense_residual_energy"
    ]
    initialization_error = initialization_metadata[
        "dense_to_factor_max_abs_error"
    ]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def controller_artifact(
        current_controller: torch.nn.Module,
        *,
        snapshot_update: int | None,
    ) -> dict[str, Any]:
        return {
            "kind": "paper_length_telomere_controller",
            "checkpoint": str(args.checkpoint),
            "backbone_step": backbone_payload["step"],
            "task": asdict(spec),
            "model": asdict(model.config),
            "controller": (
                "J(h)=hW+b"
                if args.controller_parameterization == "dense_affine"
                else "J(h)=hD+(hA)B+b"
            ),
            "controller_parameterization": args.controller_parameterization,
            "rank": (
                args.rank
                if args.controller_parameterization == "diagonal_low_rank"
                else None
            ),
            "seed": args.seed,
            "anchor_step": anchor_step,
            "loss": _controller_loss_description(
                args.controller_curriculum, spec=spec
            ),
            "state_loss_weight": 0.0,
            "controller_curriculum": args.controller_curriculum,
            "controller_training_profile": args.controller_training_profile,
            "controller_initialization": args.controller_initialization,
            "dense_iid_initialization": {
                key: initialization_metadata.get(key)
                for key in (
                    "requested_residual_rms",
                    "requested_bias_rms",
                    "off_diagonal_entry_std",
                    "realized_off_diagonal_rms_per_output",
                    "realized_bias_rms",
                    "maximum_diagonal_error",
                )
            },
            "controller_logical_max_length": args.controller_logical_max_length,
            "controller_logical_min_length": args.controller_logical_min_length,
            "controller_warmup_updates": args.controller_warmup_updates,
            "controller_final_lr_ratio": args.controller_final_lr_ratio,
            "controller_ce_temperature": args.controller_ce_temperature,
            "controller_lr_schedule": args.controller_lr_schedule,
            "controller_stable_updates": args.controller_stable_updates,
            "controller_post_final_j": args.controller_post_final_j,
            "controller_supervision": args.controller_supervision,
            "controller_checkpoint_every": args.controller_checkpoint_every,
            "snapshot_update": snapshot_update,
            "snapshot_total_updates": (
                training_budget["dense_optimizer_updates"]
                if args.controller_parameterization == "dense_affine"
                else training_budget["low_rank_optimizer_updates"]
            ),
            "learning_rate_multiplier": args.learning_rate_multiplier,
            "diagonal_lr_multiplier": args.diagonal_lr_multiplier,
            "grad_clip": args.grad_clip,
            "stage_round_multiplier": args.stage_round_multiplier,
            "controller_sampled_logical_lengths": (
                list(sampled_logical_lengths)
                if sampled_logical_lengths is not None
                else None
            ),
            "controller_logical_length_example_counts": (
                logical_length_example_counts
                if logical_length_example_counts is not None
                else None
            ),
            "dense_logical_length_example_counts": (
                dense_logical_length_example_counts
                if dense_logical_length_example_counts is not None
                else None
            ),
            "training_budget": training_budget,
            "retained_dense_residual_energy": retained_energy,
            "dense_to_factor_max_abs_error": initialization_error,
            "dense_state_dict": (
                {
                    key: value.detach().cpu()
                    for key, value in dense.state_dict().items()
                }
                if dense is not None
                else None
            ),
            "controller_state_dict": {
                key: value.detach().cpu()
                for key, value in current_controller.state_dict().items()
            },
            "stages": [asdict(stage) for stage in stages],
        }

    snapshot_directory = out_dir / "checkpoints"

    def save_controller_snapshot(
        update: int, current_controller: torch.nn.Module
    ) -> None:
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            controller_artifact(
                current_controller,
                snapshot_update=update,
            ),
            snapshot_directory / f"controller_{update:06d}.pt",
        )

    if args.controller_checkpoint_every > 0:
        save_controller_snapshot(0, controller)
    controller_rows = _train_controller_stages(
        model=model,
        spec=spec,
        controller=controller,
        stages=stages,
        device=device,
        anchor_step=anchor_step,
        grad_clip=args.grad_clip,
        diagonal_learning_rate_multiplier=args.diagonal_lr_multiplier,
        curriculum_mode=args.controller_curriculum,
        ce_temperature=args.controller_ce_temperature,
        warmup_updates=args.controller_warmup_updates,
        final_learning_rate_ratio=args.controller_final_lr_ratio,
        learning_rate_schedule=args.controller_lr_schedule,
        stable_updates=args.controller_stable_updates,
        post_final_controller=args.controller_post_final_j,
        controller_supervision=args.controller_supervision,
        checkpoint_every=args.controller_checkpoint_every,
        checkpoint_callback=(
            save_controller_snapshot
            if args.controller_checkpoint_every > 0
            else None
        ),
    )
    all_rows = [
        {**row, "phase": "dense_ce_warm_start"} for row in dense_rows
    ] + [
        {
            **row,
            "phase": (
                "dense_affine_ce"
                if args.controller_parameterization == "dense_affine"
                else "diagonal_low_rank_ce"
            ),
        }
        for row in controller_rows
    ]
    write_csv(out_dir / "training.csv", all_rows)
    artifact = controller_artifact(controller, snapshot_update=None)
    atomic_torch_save(artifact, out_dir / "controller.pt")
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "backbone_supervision": backbone_payload["supervision"],
        **_loaded_backbone_training_metadata(
            spec,
            config=model.config,
            payload=backbone_payload,
        ),
        "controller": (
            "full-rank dense affine matrix and bias"
            if args.controller_parameterization == "dense_affine"
            else "diagonal base plus rank-constrained LoRA and bias"
        ),
        "controller_parameterization": args.controller_parameterization,
        "controller_curriculum": args.controller_curriculum,
        "controller_training_profile": args.controller_training_profile,
        "controller_initialization": args.controller_initialization,
        "dense_iid_initialization": {
            key: initialization_metadata.get(key)
            for key in (
                "requested_residual_rms",
                "requested_bias_rms",
                "off_diagonal_entry_std",
                "realized_off_diagonal_rms_per_output",
                "realized_bias_rms",
                "maximum_diagonal_error",
            )
        },
        "controller_logical_max_length": args.controller_logical_max_length,
        "controller_logical_min_length": args.controller_logical_min_length,
        "controller_warmup_updates": args.controller_warmup_updates,
        "controller_final_lr_ratio": args.controller_final_lr_ratio,
        "controller_ce_temperature": args.controller_ce_temperature,
        "controller_lr_schedule": args.controller_lr_schedule,
        "controller_stable_updates": args.controller_stable_updates,
        "controller_post_final_j": args.controller_post_final_j,
        "controller_supervision": args.controller_supervision,
        "controller_checkpoint_every": args.controller_checkpoint_every,
        "learning_rate_multiplier": args.learning_rate_multiplier,
        "diagonal_lr_multiplier": args.diagonal_lr_multiplier,
        "grad_clip": args.grad_clip,
        "stage_round_multiplier": args.stage_round_multiplier,
        "controller_sampled_logical_lengths": (
            list(sampled_logical_lengths)
            if sampled_logical_lengths is not None
            else None
        ),
        "controller_logical_length_example_counts": (
            logical_length_example_counts
            if logical_length_example_counts is not None
            else None
        ),
        "dense_logical_length_example_counts": (
            dense_logical_length_example_counts
            if dense_logical_length_example_counts is not None
            else None
        ),
        "training_budget": training_budget,
        "rank": (
            args.rank
            if args.controller_parameterization == "diagonal_low_rank"
            else None
        ),
        "parameter_count": sum(p.numel() for p in controller.parameters()),
        "loss": _controller_loss_description(
            args.controller_curriculum, spec=spec
        ),
        "anchor_step": anchor_step,
        "retained_dense_residual_energy": retained_energy,
        "dense_to_factor_max_abs_error": initialization_error,
        "seed": args.seed,
        "peak_cuda_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_json_dump(summary, out_dir / "summary.json")
    return summary


def load_controller(
    artifact: Path, *, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper_length_telomere_controller":
        raise ValueError("artifact is not a paper-style telomere controller")
    parameterization = payload.get(
        "controller_parameterization", "diagonal_low_rank"
    )
    if parameterization == "dense_affine":
        controller: torch.nn.Module = DenseAffineController(
            payload["model"]["d_model"]
        )
    elif parameterization == "diagonal_low_rank":
        controller = DiagonalLowRankController(
            payload["model"]["d_model"], payload["rank"]
        )
    else:
        raise ValueError(
            f"unsupported controller parameterization: {parameterization}"
        )
    controller.load_state_dict(payload["controller_state_dict"])
    controller.to(device).eval()
    return controller, payload


def _controller_audit_training_metadata(
    controller_payload: dict[str, Any],
) -> dict[str, Any]:
    curriculum = str(controller_payload["controller_curriculum"])
    logical_maximum = controller_payload.get("controller_logical_max_length")
    if curriculum == "logical_range":
        if logical_maximum is None:
            raise ValueError("logical_range controller lacks its logical maximum")
        logical_range: list[int] | None = [1, int(logical_maximum)]
        raw_sampled = controller_payload.get("controller_sampled_logical_lengths")
        if not isinstance(raw_sampled, list) or not raw_sampled:
            raise ValueError(
                "logical_range controller lacks its sampled logical lengths"
            )
        sampled_logical_lengths: list[int] | None = sorted(
            {int(value) for value in raw_sampled}
        )
        if sampled_logical_lengths[0] < 1 or sampled_logical_lengths[-1] > int(
            logical_maximum
        ):
            raise ValueError("sampled logical lengths fall outside named range")
        raw_counts = controller_payload.get(
            "controller_logical_length_example_counts"
        )
        if not isinstance(raw_counts, dict) or not raw_counts:
            raise ValueError(
                "logical_range controller lacks logical length example counts"
            )
        logical_length_example_counts: dict[int, int] | None = {
            int(length): int(count) for length, count in raw_counts.items()
        }
        if sorted(logical_length_example_counts) != sampled_logical_lengths:
            raise ValueError(
                "logical length example counts do not match sampled lengths"
            )
        sampled_logical_range: list[int] | None = [
            sampled_logical_lengths[0], sampled_logical_lengths[-1]
        ]
    else:
        logical_range = None
        sampled_logical_range = None
        sampled_logical_lengths = None
        logical_length_example_counts = None
    return {
        "controller_curriculum": curriculum,
        "controller_logical_max_length": logical_maximum,
        "controller_start_step": int(controller_payload["anchor_step"]),
        "controller_trained_logical_length_range": logical_range,
        "controller_sampled_logical_length_range": sampled_logical_range,
        "controller_sampled_logical_lengths": sampled_logical_lengths,
        "controller_logical_length_example_counts": (
            logical_length_example_counts
        ),
    }


def _audit_protocol_metadata(
    spec: PaperTaskSpec, *, evaluation_seed: int
) -> dict[str, Any]:
    minimum_target_step = 1 + spec.step_offset
    maximum_target_step = spec.train_max_length + spec.step_offset
    target_rule = (
        "T(n)=n"
        if spec.step_offset == 0
        else f"T(n)=n+{spec.step_offset}"
    )
    return {
        "backbone_trained_logical_length_range": [1, spec.train_max_length],
        "backbone_trained_target_loop_range": [
            minimum_target_step,
            maximum_target_step,
        ],
        "maximum_backbone_effective_training_depth": (
            spec.block_layers * maximum_target_step
        ),
        "target_loop_rule": target_rule,
        "evaluation_seed": evaluation_seed,
    }


def audit_controller(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(
        args.controller, device=device
    )
    if controller_payload["checkpoint"] != str(args.checkpoint):
        raise ValueError("controller belongs to a different backbone checkpoint")
    lengths = tuple(args.lengths or spec.test_lengths)
    maximum_step = args.maximum_step or max(
        max(lengths) + spec.step_offset + 32,
        controller_payload["anchor_step"] + 64,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(
        evaluate_trajectory(
            model=model,
            spec=spec,
            controller=None,
            variant="raw",
            lengths=lengths,
            batch_size=args.batch_size,
            batches=args.batches,
            maximum_step=maximum_step,
            controller_start_step=None,
            post_final_controller=False,
            executor_off=False,
            seed=args.seed,
            device=device,
        )
    )
    modes = tuple(args.modes)
    for mode in modes:
        view = ControllerView(
            controller, mode=mode, shuffle_seed=args.seed + 901
        ).to(device).eval()
        rows.extend(
            evaluate_trajectory(
                model=model,
                spec=spec,
                controller=view,
                variant=mode,
                lengths=lengths,
                batch_size=args.batch_size,
                batches=args.batches,
                maximum_step=maximum_step,
                controller_start_step=controller_payload["anchor_step"],
                post_final_controller=args.post_final_j,
                executor_off=False,
                seed=args.seed,
                device=device,
            )
        )
    rows.extend(
        evaluate_trajectory(
            model=model,
            spec=spec,
            controller=ControllerView(controller, mode="full").to(device),
            variant="full_executor_off",
            lengths=lengths,
            batch_size=args.batch_size,
            batches=args.batches,
            maximum_step=maximum_step,
            controller_start_step=controller_payload["anchor_step"],
            post_final_controller=args.post_final_j,
            executor_off=True,
            seed=args.seed,
            device=device,
        )
    )
    curves = _curve_summary(rows)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "audit_trajectory.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": backbone_payload["step"],
        "paper": backbone_payload["paper"],
        "task": asdict(spec),
        "model": asdict(model.config),
        "backbone_seed": backbone_payload["seed"],
        "backbone_initialization": backbone_payload.get("initialization"),
        "backbone_official_model_config": backbone_payload.get(
            "official_model_config"
        ),
        "backbone_official_source_commit": backbone_payload.get(
            "official_source_commit"
        ),
        "backbone_training_precision": backbone_payload.get(
            "training_precision"
        ),
        "backbone_supervision": backbone_payload["supervision"],
        **_audit_protocol_metadata(spec, evaluation_seed=args.seed),
        **_loaded_backbone_training_metadata(
            spec,
            config=model.config,
            payload=backbone_payload,
        ),
        "controller": str(args.controller),
        "controller_rank": controller.rank,
        "controller_seed": controller_payload["seed"],
        **_controller_audit_training_metadata(controller_payload),
        "controller_parameter_count": sum(
            parameter.numel() for parameter in controller.parameters()
        ),
        "controller_training_budget": controller_payload.get("training_budget"),
        "controller_loss": controller_payload["loss"],
        "loss_placement": {
            "backbone": backbone_payload["supervision"],
            "controller": controller_payload["loss"],
        },
        "trained_loop_count": controller_payload["anchor_step"],
        "shared_physical_block_layers": model.config.block_layers,
        "maximum_effective_evaluated_depth": (
            maximum_step * model.config.block_layers
        ),
        "lengths": list(lengths),
        "examples_per_length": args.batch_size * args.batches,
        "audited_controller_modes": list(modes),
        "post_final_controller_at_readout": args.post_final_j,
        "curves": curves,
        "claim_boundary": (
            "behavioral restoration requires full controller gains plus failure "
            "of executor-off and destructive component controls"
        ),
    }
    atomic_json_dump(summary, out_dir / "summary.json")
    return summary


def _add_common_backbone_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=tuple(PAPER_TASKS), default="parity")
    parser.add_argument(
        "--supervision",
        choices=("adaptive_step", "fixed_horizon"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=100_001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--schedule-total-steps",
        type=int,
        help="use the formal cosine schedule during a shorter pilot",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument(
        "--block-layers",
        type=int,
        help="override the number of physical Transformer layers per shared loop",
    )
    parser.add_argument(
        "--official-model-config",
        action="store_true",
        help="use the task-specific d_model/layer/head values in the released YAML",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("causal", "full"),
        default="causal",
        help="causal decoder attention or unrestricted full self-attention",
    )
    parser.add_argument(
        "--token-embedding-injection",
        choices=("initial_only", "every_loop"),
        default="every_loop",
        help=(
            "inject immutable token embeddings only before recurrent call 1 "
            "or before every shared-block call"
        ),
    )
    parser.add_argument(
        "--position-embedding",
        choices=("none", "learned_absolute"),
        default="none",
        help=(
            "position signal injected with the original input before every "
            "shared-block call"
        ),
    )
    parser.add_argument(
        "--position-injection",
        choices=("initial_only", "every_loop"),
        default="initial_only",
        help="inject learned positions only at loop 1 or before every loop",
    )
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument(
        "--addition-lsb-first",
        action="store_true",
        help="encode both Addition operands and its answer least-significant bit first",
    )
    parser.add_argument(
        "--addition-minimum-width",
        type=int,
        help=(
            "reserve at least this many operand slots and zero-pad the high-bit "
            "side of shorter Addition examples"
        ),
    )
    parser.add_argument(
        "--addition-answer-supervision",
        choices=("full_layout", "logical_digits", "progressive_carry"),
        help=(
            "full_layout supervises every answer/padding slot; logical_digits "
            "supervises only the sample's m numerical sum digits and excludes "
            "both final carry and high zero padding; progressive_carry uses "
            "the m+1 arithmetic positions at T(m)=m+1, with an additional "
            "m-digit readout at loop m during training"
        ),
    )
    parser.add_argument("--train-max-length", type=int)
    parser.add_argument(
        "--task-step-offset",
        type=int,
        help=(
            "override the registered recurrent target rule from T(n)=n+offset; "
            "for example, Addition uses 1 by default and --task-step-offset 0 "
            "registers T(n)=n in the saved task protocol"
        ),
    )
    parser.add_argument(
        "--train-fixed-logical-length",
        type=int,
        help=(
            "sample only this logical length during backbone training while "
            "retaining the registered task and learning-rate schedule"
        ),
    )
    parser.add_argument(
        "--batch-shared-logical-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "sample one logical length uniformly from the live curriculum "
            "range for the whole optimizer batch, so sequence layout changes "
            "between batches without padding every sample to the global bound"
        ),
    )
    parser.add_argument(
        "--train-fixed-loop-count",
        type=int,
        help=(
            "with fixed_horizon supervision and one fixed logical length, "
            "supervise exactly this recurrent loop instead of T(n)"
        ),
    )
    parser.add_argument(
        "--curriculum-interval",
        type=int,
        help="override only for smoke/benchmark runs; formal runs use the paper schedule",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument(
        "--resume",
        type=Path,
        help="resume model, optimizer, and RNG state from a training checkpoint",
    )
    restart.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "continue the checkpoint's weights and global step with a fresh "
            "optimizer and fresh RNG state"
        ),
    )
    parser.add_argument(
        "--initialize-rng-seed-offset",
        type=int,
        default=1_000_003,
        help=(
            "deterministic offset used to avoid replaying the source run's "
            "data and dropout stream during weight-only continuation"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-style FAP length-generalization experiment for the "
            "computational telomere hypothesis."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    backbone = subparsers.add_parser("backbone")
    _add_common_backbone_arguments(backbone)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--checkpoint", type=Path, required=True)
    diagnose.add_argument("--lengths", type=int, nargs="+")
    diagnose.add_argument("--batch-size", type=int, default=128)
    diagnose.add_argument("--batches", type=int, default=32)
    diagnose.add_argument("--maximum-step", type=int)
    diagnose.add_argument("--seed", type=int, default=260001)
    diagnose.add_argument("--device", default="auto")
    diagnose.add_argument("--minimum-endpoint-accuracy", type=float, default=0.95)
    diagnose.add_argument("--extension-gate-length", type=int)
    diagnose.add_argument("--minimum-decline", type=float, default=0.20)
    diagnose.add_argument("--out-dir", type=Path, required=True)

    controller = subparsers.add_parser("controller")
    controller.add_argument("--checkpoint", type=Path, required=True)
    controller.add_argument("--diagnosis", type=Path)
    controller.add_argument(
        "--controller-parameterization",
        choices=("diagonal_low_rank", "dense_affine"),
        default="diagonal_low_rank",
    )
    controller.add_argument("--rank", type=int, default=48)
    controller.add_argument("--seed", type=int, default=211001)
    controller.add_argument("--device", default="auto")
    controller.add_argument("--grad-clip", type=float, default=1.0)
    controller.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    controller.add_argument("--diagonal-lr-multiplier", type=float, default=0.1)
    controller.add_argument("--dense-stage-count", type=int, default=2)
    controller.add_argument(
        "--controller-training-profile",
        choices=(
            "legacy",
            "identity_long_warmup",
            "identity_tail_warmup",
        ),
        default="legacy",
    )
    controller.add_argument(
        "--controller-initialization",
        choices=("dense_svd", "identity", "dense_iid"),
        default="dense_svd",
    )
    controller.add_argument(
        "--dense-iid-residual-rms",
        type=float,
        default=0.01,
        help=(
            "expected per-output RMS of the iid off-diagonal perturbation "
            "for unit-RMS hidden states"
        ),
    )
    controller.add_argument(
        "--dense-iid-off-diagonal-std",
        type=float,
        help=(
            "direct standard deviation for each iid off-diagonal entry; "
            "when set, overrides --dense-iid-residual-rms"
        ),
    )
    controller.add_argument(
        "--dense-iid-bias-rms",
        type=float,
        default=0.01,
    )
    controller.add_argument(
        "--controller-curriculum",
        choices=("grid", "mixed", "extension", "overloop", "logical_range"),
        default="grid",
        help=(
            "grid samples logical length n and additional overloop u, then "
            "applies task CE only at final depth n+u; mixed is the earlier "
            "half-extension/half-boundary-overloop curriculum; logical_range "
            "trains the inter-loop J directly on logical lengths 1..N"
        ),
    )
    controller.add_argument("--controller-logical-max-length", type=int)
    controller.add_argument(
        "--controller-logical-min-length",
        type=int,
        help=(
            "override the first sampled logical length for the "
            "identity_long_warmup logical-range curriculum"
        ),
    )
    controller.add_argument(
        "--controller-anchor-step",
        type=int,
        help=(
            "leave loops 1..K raw, then apply J before loop K+1 onward; "
            "K=0 applies the shared J before the first and every later loop"
        ),
    )
    controller.add_argument(
        "--controller-warmup-updates", type=int, default=0
    )
    controller.add_argument(
        "--controller-final-lr-ratio", type=float, default=1.0
    )
    controller.add_argument(
        "--controller-lr-schedule",
        choices=("cosine", "wsd"),
        default="cosine",
    )
    controller.add_argument(
        "--controller-stable-updates", type=int, default=0
    )
    controller.add_argument(
        "--controller-post-final-j",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply one additional J after the final recurrent loop before readout",
    )
    controller.add_argument(
        "--controller-ce-temperature",
        type=float,
        default=1.0,
        help=(
            "divide final answer logits by this positive temperature during "
            "controller-only CE training; evaluation logits are unchanged"
        ),
    )
    controller.add_argument(
        "--controller-supervision",
        choices=("full_answer", "addition_final_carry"),
        default="full_answer",
        help="select one mutually exclusive controller target circuit",
    )
    controller.add_argument("--stage-round-limit", type=int)
    controller.add_argument("--stage-round-multiplier", type=int, default=1)
    controller.add_argument(
        "--controller-checkpoint-every",
        type=int,
        default=0,
        help=(
            "save loadable controller artifacts every N optimizer updates; "
            "zero disables intermediate snapshots"
        ),
    )
    controller.add_argument("--force", action="store_true")
    controller.add_argument("--out-dir", type=Path, required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--controller", type=Path, required=True)
    audit.add_argument("--lengths", type=int, nargs="+")
    audit.add_argument("--batch-size", type=int, default=128)
    audit.add_argument("--batches", type=int, default=32)
    audit.add_argument("--maximum-step", type=int)
    audit.add_argument(
        "--modes",
        nargs="+",
        choices=("full", "no_AB", "identity_D", "mean_D", "no_bias", "shuffle_D"),
        default=("full", "no_AB", "identity_D", "mean_D", "no_bias", "shuffle_D"),
    )
    audit.add_argument("--seed", type=int, default=261001)
    audit.add_argument("--device", default="auto")
    audit.add_argument(
        "--post-final-j",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply J once to each stopped state for readout only",
    )
    audit.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.action == "backbone":
        summary = train_backbone(args)
    elif args.action == "diagnose":
        summary = diagnose_backbone(args)
    elif args.action == "controller":
        summary = train_controller(args)
    elif args.action == "audit":
        summary = audit_controller(args)
    else:
        raise RuntimeError(f"unhandled action: {args.action}")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
