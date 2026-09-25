from __future__ import annotations

import copy
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial


def _reft_rank(method: str) -> int | None:
    suffix = method.removeprefix("reft")
    if suffix == method or not suffix.isdigit():
        return None
    rank = int(suffix)
    return rank if rank > 0 else None


def adapter_parameter_count(method: str, *, d_model: int, blocks: int) -> int:
    if d_model < 1 or blocks < 1:
        raise ValueError("d_model and blocks must be positive")
    if method == "j":
        return d_model * d_model + d_model
    rank = _reft_rank(method)
    if rank is not None:
        if rank > d_model:
            raise ValueError("ReFT rank cannot exceed d_model")
        return 2 * d_model * rank + d_model
    if method == "lora_qkv32":
        return blocks * 32 * (d_model + 3 * d_model)
    if method == "prefix64":
        return blocks * (2 * 64 * d_model + 1)
    if method == "steering":
        return d_model
    if method == "phase_steering":
        return 4 * d_model
    raise ValueError(f"unknown adapter method: {method}")


def future_targets(
    path_targets: Tensor, ages: Tensor, *, horizon: int
) -> tuple[Tensor, Tensor]:
    if path_targets.ndim != 2 or ages.ndim != 1 or path_targets.shape[0] != ages.shape[0]:
        raise ValueError("path_targets must be [batch,path] and ages must be [batch]")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    current_index = ages.to(dtype=torch.long) - 1
    if bool((current_index < 0).any()) or bool((current_index + horizon >= path_targets.shape[1]).any()):
        raise ValueError("ages do not leave enough future targets")
    batch_index = torch.arange(path_targets.shape[0], device=path_targets.device)
    target_index = current_index[:, None] + torch.arange(
        1, horizon + 1, device=path_targets.device
    )
    return path_targets[batch_index, current_index], path_targets.gather(1, target_index)


def permutation_partitions(
    *, node_count: int, validation_count: int, formal_count: int, seed: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[tuple[int, ...]]]:
    if node_count < 1 or validation_count < 1 or formal_count < 1:
        raise ValueError("node_count, validation_count, and formal_count must be positive")
    candidates = list(itertools.permutations(range(node_count)))
    if validation_count + formal_count >= len(candidates):
        raise ValueError("not enough permutations for train, validation, and formal partitions")
    generator = random.Random(seed)
    generator.shuffle(candidates)
    formal = candidates[:formal_count]
    validation = candidates[formal_count : formal_count + validation_count]
    train = candidates[formal_count + validation_count :]
    return train, validation, formal


def closed_loop_loss(
    state: Tensor,
    targets: Tensor,
    *,
    step: Callable[[Tensor, int], Tensor],
    readout: Callable[[Tensor], Tensor],
) -> float:
    if targets.ndim != 2 or targets.shape[0] != state.shape[0]:
        raise ValueError("targets must be [batch,horizon]")
    losses: list[Tensor] = []
    current = state
    for loop_index in range(targets.shape[1]):
        current = step(current, loop_index)
        losses.append(F.cross_entropy(readout(current), targets[:, loop_index]))
    return float(torch.stack(losses).mean().detach())


def prefix_causal_mask(*, seq_len: int, prefix_len: int, device: torch.device) -> Tensor:
    if seq_len < 1 or prefix_len < 1:
        raise ValueError("seq_len and prefix_len must be positive")
    return torch.arange(seq_len, device=device)[:, None] >= (
        torch.arange(seq_len + prefix_len, device=device)[None, :] - prefix_len
    )


def interface_decision(
    *,
    controlled_q: list[float],
    pre_executor_q1: float,
    random_executor_q1: float,
    shuffled_q1: float,
) -> dict[str, bool]:
    if len(controlled_q) != 4:
        raise ValueError("controlled_q must contain q1..q4")
    closure = all(value >= 0.90 for value in controlled_q)
    executor_dependent = pre_executor_q1 <= 0.20 and random_executor_q1 <= 0.20
    input_specific = shuffled_q1 <= 0.20
    return {
        "four_step_closure": closure,
        "no_prewrite": pre_executor_q1 <= 0.20,
        "random_executor_off": random_executor_q1 <= 0.20,
        "shuffled_input_off": input_specific,
        "interface_dependency_positive": closure and executor_dependent and input_specific,
    }


def swap_cell_labels() -> tuple[str, str, str, str]:
    return ("J_A_to_F_A", "J_A_to_F_B", "J_B_to_F_A", "J_B_to_F_B")


class DenseInterfaceJ(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, state: Tensor) -> Tensor:
        return state.float() @ self.weight + self.bias


class AffineReFT(nn.Module):
    def __init__(self, dimension: int, rank: int = 128) -> None:
        super().__init__()
        if not 1 <= rank <= dimension:
            raise ValueError("ReFT rank must lie in [1, dimension]")
        self.left = nn.Parameter(torch.randn(dimension, rank) / dimension**0.5)
        self.right = nn.Parameter(torch.zeros(rank, dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, state: Tensor) -> Tensor:
        state = state.float()
        return state + (state @ self.left) @ self.right + self.bias


class StaticSteering(nn.Module):
    def __init__(self, dimension: int, *, phases: int = 1) -> None:
        super().__init__()
        if phases < 1:
            raise ValueError("phases must be positive")
        self.delta = nn.Parameter(torch.zeros(phases, dimension))

    def forward(self, state: Tensor, *, phase: int) -> Tensor:
        return state.float() + self.delta[phase % self.delta.shape[0]]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.left = nn.Parameter(torch.randn(rank, base.in_features) / base.in_features**0.5)
        self.right = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, value: Tensor) -> Tensor:
        return self.base(value) + F.linear(F.linear(value.float(), self.left), self.right)


class PrefixAttention(nn.Module):
    """A standard learned KV prefix around the legacy packed-QKV attention."""

    def __init__(self, base: nn.Module, *, prefix_len: int) -> None:
        super().__init__()
        if prefix_len < 1:
            raise ValueError("prefix_len must be positive")
        if not hasattr(base, "qkv") or not hasattr(base, "out_proj"):
            raise TypeError("prefix tuning requires legacy packed-QKV attention")
        self.base = base
        self.base.requires_grad_(False)
        self.prefix_len = prefix_len
        self.n_heads = int(base.n_heads)
        self.d_head = int(base.d_head)
        self.prefix_key = nn.Parameter(torch.zeros(prefix_len, self.n_heads, self.d_head))
        self.prefix_value = nn.Parameter(torch.zeros(prefix_len, self.n_heads, self.d_head))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, value: Tensor) -> Tensor:
        base_output = self.base(value)
        batch, seq_len, d_model = value.shape
        qkv = self.base.qkv(value).view(batch, seq_len, 3, self.n_heads, self.d_head)
        query, key, val = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)
        prefix_key = self.prefix_key.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1)
        prefix_value = self.prefix_value.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1, -1)
        key = torch.cat((prefix_key.to(dtype=key.dtype), key), dim=2)
        val = torch.cat((prefix_value.to(dtype=val.dtype), val), dim=2)
        allowed = prefix_causal_mask(
            seq_len=seq_len, prefix_len=self.prefix_len, device=value.device
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            val,
            attn_mask=allowed,
            dropout_p=self.base.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        prefix_output = self.base.out_proj(attended)
        return base_output + torch.tanh(self.gate) * (prefix_output - base_output)


class Adapter(nn.Module):
    method: str
    changes_executor: bool

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def frozen_backbone_parameters(self) -> list[nn.Parameter]:
        trainable = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        return [parameter for parameter in self.parameters() if id(parameter) not in trainable]

    def pre_state(self, state: Tensor, *, phase: int) -> Tensor | None:
        return None

    def step(self, state: Tensor, *, loop_index: int, phase: int) -> Tensor:
        raise NotImplementedError


class InterfaceStateAdapter(Adapter):
    changes_executor = False

    def __init__(self, model: LoopedGraphPathTransformer, transform: nn.Module, method: str) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.transform = transform
        self.method = method

    def pre_state(self, state: Tensor, *, phase: int) -> Tensor:
        if isinstance(self.transform, StaticSteering):
            return self.transform(state, phase=phase)
        return self.transform(state)

    def step(self, state: Tensor, *, loop_index: int, phase: int) -> Tensor:
        mapped = self.pre_state(state, phase=phase)
        return self.model.apply_loop(mapped.to(dtype=state.dtype), loop_index=loop_index)


class ExecutorAdapter(Adapter):
    changes_executor = True

    def __init__(self, model: LoopedGraphPathTransformer, method: str) -> None:
        super().__init__()
        self.model = model
        self.method = method

    def step(self, state: Tensor, *, loop_index: int, phase: int) -> Tensor:
        del phase
        return self.model.apply_loop(state, loop_index=loop_index)


class ConjugatedExecutor(nn.Module):
    """A behavior-identical executor with a deliberately different residual ABI."""

    def __init__(self, base: LoopedGraphPathTransformer, permutation: Tensor) -> None:
        super().__init__()
        if permutation.ndim != 1 or permutation.numel() != base.cfg.d_model:
            raise ValueError("permutation must cover exactly d_model coordinates")
        if not torch.equal(torch.sort(permutation.cpu()).values, torch.arange(base.cfg.d_model)):
            raise ValueError("permutation must contain each channel exactly once")
        self.base = base
        self.base.requires_grad_(False)
        self.cfg = base.cfg
        self.register_buffer("permutation", permutation.to(dtype=torch.long))
        inverse = torch.empty_like(self.permutation)
        inverse[self.permutation] = torch.arange(self.permutation.numel(), device=self.permutation.device)
        self.register_buffer("inverse_permutation", inverse)

    def from_base_state(self, state: Tensor) -> Tensor:
        return state.index_select(-1, self.permutation)

    def to_base_state(self, state: Tensor) -> Tensor:
        return state.index_select(-1, self.inverse_permutation)

    def apply_loop(self, state: Tensor, *, loop_index: int) -> Tensor:
        return self.from_base_state(
            self.base.apply_loop(self.to_base_state(state), loop_index=loop_index)
        )

    def readout_logits(self, state: Tensor) -> Tensor:
        base_state = self.to_base_state(state)
        return self.base.unembed(self.base.ln_final(base_state[:, -1, :]))[:, : self.cfg.node_count]


def build_adapter(method: str, model: LoopedGraphPathTransformer) -> Adapter:
    model.requires_grad_(False)
    if method == "j":
        return InterfaceStateAdapter(model, DenseInterfaceJ(model.cfg.d_model), method)
    rank = _reft_rank(method)
    if rank is not None:
        return InterfaceStateAdapter(model, AffineReFT(model.cfg.d_model, rank=rank), method)
    if method == "steering":
        return InterfaceStateAdapter(model, StaticSteering(model.cfg.d_model), method)
    if method == "phase_steering":
        return InterfaceStateAdapter(model, StaticSteering(model.cfg.d_model, phases=4), method)
    adapted = copy.deepcopy(model)
    adapted.requires_grad_(False)
    if method == "lora_qkv32":
        for block in adapted.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=32)
        return ExecutorAdapter(adapted, method)
    if method == "prefix64":
        for block in adapted.blocks:
            block.attn = PrefixAttention(block.attn, prefix_len=64)
        return ExecutorAdapter(adapted, method)
    raise ValueError(f"unknown adapter method: {method}")


def readout_logits(model: LoopedGraphPathTransformer | ConjugatedExecutor, state: Tensor) -> Tensor:
    if isinstance(model, ConjugatedExecutor):
        return model.readout_logits(state)
    return model.unembed(model.ln_final(state[:, -1, :]))[:, : model.cfg.node_count]


def closed_loop_objective(
    adapter: Adapter,
    state: Tensor,
    targets: Tensor,
    *,
    loop_index: int,
) -> Tensor:
    if targets.ndim != 2 or targets.shape[0] != state.shape[0]:
        raise ValueError("targets must be [batch,horizon]")
    current = state
    losses: list[Tensor] = []
    for phase in range(targets.shape[1]):
        current = adapter.step(current, loop_index=loop_index + phase, phase=phase)
        losses.append(F.cross_entropy(readout_logits(adapter.model, current), targets[:, phase]))
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class InterfaceBatch:
    source: Tensor
    current: Tensor
    targets: Tensor
    age: Tensor
    example_digest: str


def conjugate_interface_batch(
    batch: InterfaceBatch, executor: ConjugatedExecutor
) -> InterfaceBatch:
    """Express an otherwise identical interface batch in executor B coordinates."""
    permutation_digest = hashlib.sha256(executor.permutation.detach().cpu().numpy().tobytes()).hexdigest()
    return InterfaceBatch(
        source=executor.from_base_state(batch.source),
        current=batch.current,
        targets=batch.targets,
        age=batch.age,
        example_digest=hashlib.sha256(
            f"{batch.example_digest}:residual-permutation:{permutation_digest}".encode()
        ).hexdigest(),
    )


@dataclass(frozen=True)
class CandidateResult:
    learning_rate: float
    trainable_state: dict[str, Tensor]
    validation: dict[str, float]
    history: list[dict[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _permutation_digest(permutations: Sequence[tuple[int, ...]]) -> str:
    return hashlib.sha256(json.dumps([list(row) for row in permutations]).encode()).hexdigest()


def _example_digest(
    permutations: Sequence[tuple[int, ...]], starts: Sequence[int], ages: Sequence[int]
) -> str:
    return hashlib.sha256(
        json.dumps([[list(p), int(s), int(a)] for p, s, a in zip(permutations, starts, ages, strict=True)]).encode()
    ).hexdigest()


def _subset(batch: InterfaceBatch, index: Tensor) -> InterfaceBatch:
    return InterfaceBatch(
        source=batch.source[index],
        current=batch.current[index],
        targets=batch.targets[index],
        age=batch.age[index],
        example_digest=batch.example_digest,
    )


def select_cached_age_states(states: Sequence[Tensor], ages: Tensor) -> Tensor:
    if not states:
        raise ValueError("cached states must be nonempty")
    if ages.ndim != 1 or states[0].shape[0] != ages.shape[0]:
        raise ValueError("ages must index one cached state per batch example")
    if bool((ages < 0).any()) or bool((ages >= len(states)).any()):
        raise ValueError("age index is outside the cached state range")
    stacked = torch.stack(list(states), dim=0)
    batch_index = torch.arange(ages.shape[0], device=ages.device)
    return stacked[ages.to(dtype=torch.long), batch_index]


@torch.no_grad()
def build_interface_dataset(
    *,
    source_model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    permutations: Sequence[tuple[int, ...]],
    starts: Sequence[int],
    ages: Sequence[int],
    horizon: int,
    device: torch.device,
    collection_batch_size: int,
) -> InterfaceBatch:
    if not permutations or not (len(permutations) == len(starts) == len(ages)):
        raise ValueError("permutations, starts, and ages must be nonempty and aligned")
    if horizon < 1 or collection_batch_size < 1:
        raise ValueError("horizon and collection_batch_size must be positive")
    if min(ages) < 1 or max(ages) > cfg.max_loops - 2:
        raise ValueError("ages must lie in [1, max_loops-2]")
    source_parts: list[Tensor] = []
    current_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    age_parts: list[Tensor] = []
    max_path_positions = max(ages) + horizon
    for offset in range(0, len(permutations), collection_batch_size):
        stop = min(offset + collection_batch_size, len(permutations))
        successors = torch.tensor(permutations[offset:stop], dtype=torch.long, device=device)
        start = torch.tensor(starts[offset:stop], dtype=torch.long, device=device)
        age = torch.tensor(ages[offset:stop], dtype=torch.long, device=device)
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            stop - offset,
            device,
            path_positions=max_path_positions,
            successors=successors,
            start=start,
        )
        states = cache_states_with_initial(source_model, tokens, loops=max(ages))
        batch_index = torch.arange(stop - offset, device=device)
        current, targets = future_targets(path_targets, age, horizon=horizon)
        source_parts.append(select_cached_age_states(states, age).detach())
        current_parts.append(current)
        target_parts.append(targets)
        age_parts.append(age)
    return InterfaceBatch(
        source=torch.cat(source_parts),
        current=torch.cat(current_parts),
        targets=torch.cat(target_parts),
        age=torch.cat(age_parts),
        example_digest=_example_digest(permutations, starts, ages),
    )


def sample_interface_examples(
    *,
    train_permutations: Sequence[tuple[int, ...]],
    examples: int,
    source_ages: Sequence[int],
    seed: int,
) -> tuple[list[tuple[int, ...]], list[int], list[int]]:
    if examples < 1 or not source_ages or min(source_ages) < 1:
        raise ValueError("examples and source_ages must be positive")
    generator = random.Random(seed)
    return (
        generator.choices(list(train_permutations), k=examples),
        [generator.randrange(8) for _ in range(examples)],
        [generator.choice(list(source_ages)) for _ in range(examples)],
    )


def interface_training_seed(controller_seed: int) -> int:
    """The controller-data sample seed shared by every comparable runner mode."""
    return controller_seed + 41_701


def interface_candidate_seed(controller_seed: int, candidate_index: int) -> int:
    return controller_seed * 1009 + candidate_index + 17


def all_interface_examples(
    *, permutations: Sequence[tuple[int, ...]], source_ages: Sequence[int]
) -> tuple[list[tuple[int, ...]], list[int], list[int]]:
    if not source_ages or min(source_ages) < 1:
        raise ValueError("source_ages must be nonempty and positive")
    rows = [
        (permutation, start, age)
        for permutation in permutations
        for start in range(8)
        for age in source_ages
    ]
    return [row[0] for row in rows], [row[1] for row in rows], [row[2] for row in rows]


def trainable_state(adapter: Adapter) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(adapter: Adapter, state: dict[str, Tensor]) -> None:
    parameters = dict(adapter.named_parameters())
    if set(state) != {name for name, parameter in parameters.items() if parameter.requires_grad}:
        raise ValueError("saved trainable state does not match adapter parameters")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


@torch.no_grad()
def evaluate_curve(
    *,
    adapter: Adapter,
    dataset: InterfaceBatch,
    horizon: int,
    batch_size: int,
    loop_index: int,
) -> list[dict[str, Any]]:
    if horizon > dataset.targets.shape[1]:
        raise ValueError("dataset lacks requested target horizon")
    totals = [
        {"examples": 0, "moving_examples": 0, "correct": 0, "moving_correct": 0}
        for _ in range(horizon)
    ]
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        state = batch.source
        for phase in range(horizon):
            state = adapter.step(state, loop_index=loop_index + phase, phase=phase)
            target = batch.targets[:, phase]
            moving = target.ne(batch.current)
            prediction = readout_logits(adapter.model, state).argmax(dim=-1)
            row = totals[phase]
            row["examples"] += int(target.shape[0])
            row["moving_examples"] += int(moving.sum())
            row["correct"] += int(prediction.eq(target).sum())
            row["moving_correct"] += int(prediction[moving].eq(target[moving]).sum())
    return [
        {
            "q": phase + 1,
            "examples": row["examples"],
            "moving_examples": row["moving_examples"],
            "target_accuracy": row["correct"] / max(row["examples"], 1),
            "moving_target_accuracy": row["moving_correct"] / max(row["moving_examples"], 1),
        }
        for phase, row in enumerate(totals)
    ]


@torch.no_grad()
def evaluate_pre_executor_q1(
    *, adapter: Adapter, dataset: InterfaceBatch, batch_size: int
) -> dict[str, float | str]:
    if adapter.pre_state(dataset.source[:1], phase=0) is None:
        return {"applicable": "false", "moving_target_accuracy": float("nan"), "current_accuracy": float("nan")}
    total = moving_total = target_correct = current_correct = 0
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        state = adapter.pre_state(batch.source, phase=0)
        assert state is not None
        prediction = readout_logits(adapter.model, state).argmax(dim=-1)
        moving = batch.targets[:, 0].ne(batch.current)
        total += int(prediction.shape[0])
        moving_total += int(moving.sum())
        target_correct += int(prediction[moving].eq(batch.targets[:, 0][moving]).sum())
        current_correct += int(prediction.eq(batch.current).sum())
    return {
        "applicable": "true",
        "moving_target_accuracy": target_correct / max(moving_total, 1),
        "current_accuracy": current_correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_first_step_override(
    *,
    adapter: Adapter,
    dataset: InterfaceBatch,
    batch_size: int,
    transform: Callable[[Tensor, int], Tensor],
) -> float:
    moving_total = correct = 0
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        state = transform(batch.source, offset)
        output = adapter.step(state, loop_index=adapter.model.cfg.max_loops, phase=0)
        prediction = readout_logits(adapter.model, output).argmax(dim=-1)
        moving = batch.targets[:, 0].ne(batch.current)
        moving_total += int(moving.sum())
        correct += int(prediction[moving].eq(batch.targets[:, 0][moving]).sum())
    return correct / max(moving_total, 1)


def _matched_random(value: Tensor, generator: torch.Generator) -> Tensor:
    noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
    mean = value.float().mean(dim=(1, 2), keepdim=True)
    centered = value.float() - mean
    scale = torch.sqrt(centered.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12))
    normalized = (noise.float() - noise.float().mean(dim=(1, 2), keepdim=True))
    normalized = normalized / torch.sqrt(normalized.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12))
    return (normalized * scale + mean).to(dtype=value.dtype)


@torch.no_grad()
def evaluate_j_dependency_controls(
    *,
    adapter: Adapter,
    dataset: InterfaceBatch,
    batch_size: int,
    random_seed: int,
) -> dict[str, float | str]:
    if adapter.method != "j":
        return {"applicable": "false"}
    generator = torch.Generator(device=dataset.source.device)
    generator.manual_seed(random_seed)
    shuffled = evaluate_first_step_override(
        adapter=adapter,
        dataset=dataset,
        batch_size=batch_size,
        transform=lambda state, _offset: state[
            torch.randperm(state.shape[0], device=state.device, generator=generator)
        ],
    )
    random_input = evaluate_first_step_override(
        adapter=adapter,
        dataset=dataset,
        batch_size=batch_size,
        transform=lambda state, _offset: _matched_random(state, generator),
    )
    random_executor = LoopedGraphPathTransformer(adapter.model.cfg).to(dataset.source.device)
    random_executor.requires_grad_(False)
    moving_total = correct = 0
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        mapped = adapter.pre_state(batch.source, phase=0)
        assert mapped is not None
        output = random_executor.apply_loop(mapped.to(dtype=batch.source.dtype), loop_index=random_executor.cfg.max_loops)
        prediction = readout_logits(random_executor, output).argmax(dim=-1)
        moving = batch.targets[:, 0].ne(batch.current)
        moving_total += int(moving.sum())
        correct += int(prediction[moving].eq(batch.targets[:, 0][moving]).sum())
    return {
        "applicable": "true",
        "shuffled_input_moving_target_accuracy": shuffled,
        "random_input_moving_target_accuracy": random_input,
        "random_executor_moving_target_accuracy": correct / max(moving_total, 1),
    }


@torch.no_grad()
def evaluate_j_region_curve(
    *, adapter: Adapter, dataset: InterfaceBatch, batch_size: int, random_seed: int
) -> list[dict[str, Any]]:
    if adapter.method != "j":
        return []
    generator = torch.Generator(device=dataset.source.device)
    generator.manual_seed(random_seed)
    rows: list[dict[str, Any]] = []
    for condition, alpha in (("along_J_delta", 0.0), ("along_J_delta", 0.25), ("along_J_delta", 0.5), ("along_J_delta", 0.75), ("along_J_delta", 1.0), ("along_J_delta", 1.25), ("norm_matched_random", 1.0)):
        moving_total = correct = 0
        for offset in range(0, dataset.source.shape[0], batch_size):
            index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
            batch = _subset(dataset, index)
            mapped = adapter.pre_state(batch.source, phase=0)
            assert mapped is not None
            delta = mapped - batch.source
            if condition == "along_J_delta":
                candidate = batch.source + alpha * delta
            else:
                noise = torch.randn(delta.shape, device=delta.device, dtype=delta.dtype, generator=generator)
                scale = delta.float().norm(dim=(1, 2), keepdim=True) / noise.float().norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                candidate = batch.source + noise * scale.to(dtype=noise.dtype)
            output = adapter.model.apply_loop(candidate.to(dtype=batch.source.dtype), loop_index=adapter.model.cfg.max_loops)
            prediction = readout_logits(adapter.model, output).argmax(dim=-1)
            moving = batch.targets[:, 0].ne(batch.current)
            moving_total += int(moving.sum())
            correct += int(prediction[moving].eq(batch.targets[:, 0][moving]).sum())
        rows.append(
            {
                "condition": condition,
                "alpha": alpha,
                "moving_target_accuracy": correct / max(moving_total, 1),
            }
        )
    return rows


def _validation_score(rows: list[dict[str, Any]]) -> float:
    return float(sum(row["moving_target_accuracy"] for row in rows[:4]) / 4)


def train_candidate(
    *,
    method: str,
    base_model: LoopedGraphPathTransformer,
    train_dataset: InterfaceBatch,
    validation_dataset: InterfaceBatch,
    learning_rate: float,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_every: int,
    seed: int,
) -> CandidateResult:
    adapter = build_adapter(method, base_model).to(train_dataset.source.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad], lr=learning_rate
    )
    generator = torch.Generator(device=train_dataset.source.device)
    generator.manual_seed(seed)
    best_score = -math.inf
    best_state: dict[str, Tensor] | None = None
    best_validation: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        index = torch.randint(0, train_dataset.source.shape[0], (batch_size,), device=train_dataset.source.device, generator=generator)
        batch = _subset(train_dataset, index)
        loss = closed_loop_objective(adapter, batch.source, batch.targets[:, :4], loop_index=base_model.cfg.max_loops)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in adapter.frozen_backbone_parameters()):
            raise RuntimeError("frozen backbone received gradients")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            validation_rows = evaluate_curve(
                adapter=adapter,
                dataset=validation_dataset,
                horizon=4,
                batch_size=eval_batch_size,
                loop_index=base_model.cfg.max_loops,
            )
            validation = {f"q{row['q']}_moving_target_accuracy": row["moving_target_accuracy"] for row in validation_rows}
            validation["mean_q1_q4_moving_target_accuracy"] = _validation_score(validation_rows)
            row = {
                "method": method,
                "learning_rate": learning_rate,
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **validation,
            }
            history.append(row)
            print(json.dumps({"event": "training", **row}), flush=True)
            if validation["mean_q1_q4_moving_target_accuracy"] > best_score:
                best_score = validation["mean_q1_q4_moving_target_accuracy"]
                best_state = trainable_state(adapter)
                best_validation = validation
    if best_state is None or best_validation is None:
        raise RuntimeError("candidate did not produce a validation checkpoint")
    return CandidateResult(learning_rate, best_state, best_validation, history)


def select_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(candidates, key=lambda item: item.validation["mean_q1_q4_moving_target_accuracy"])


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default_learning_rates(method: str) -> tuple[float, ...]:
    if method in {"lora_qkv32", "prefix64"}:
        return (3e-5, 1e-4, 3e-4)
    return (1e-5, 3e-5, 1e-4)


def _report_method(summary: dict[str, Any]) -> str:
    q = summary["controlled_q"]
    pre = summary["pre_executor"]
    lines = [
        f"# D8L8 interface-specialness: {summary['method']}",
        "",
        f"- backbone seed/checkpoint: {summary['backbone_seed']} / `{summary['checkpoint_sha256']}`; controller-data seed: {summary['controller_seed']}.",
        f"- runtime: {summary['runtime_order']}; trainable parameters: {summary['trainable_parameters']}; executor changed: {summary['changes_executor']}.",
        f"- strict unseen graphs: {summary['strict_eval_examples']} examples, digest `{summary['strict_eval_permutation_sha256']}`.",
        "",
        "## Closed-loop continuation",
        "",
        "| q | moving target accuracy |",
        "|---:|---:|",
        *[f"| {row['q']} | {row['moving_target_accuracy']:.4f} |" for row in q],
        "",
        "## Executor-dependence controls",
        "",
        f"- pre-executor future q1: {pre['moving_target_accuracy'] if pre['applicable'] == 'true' else 'N/A'}; current: {pre['current_accuracy'] if pre['applicable'] == 'true' else 'N/A'}.",
        f"- decision: {summary['decision']}.",
    ]
    if summary["dependency_controls"].get("applicable") == "true":
        controls = summary["dependency_controls"]
        lines.extend(
            [
                f"- shuffled J input q1: {controls['shuffled_input_moving_target_accuracy']:.4f}; matched-random input q1: {controls['random_input_moving_target_accuracy']:.4f}; random executor q1: {controls['random_executor_moving_target_accuracy']:.4f}.",
                "- A positive J result is evidence for frozen-executor interface dependence, not proof of a unique original circuit.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_method_experiment(
    *,
    checkpoint: Path,
    out_dir: Path,
    method: str,
    device_name: str,
    controller_seed: int,
    train_examples: int,
    validation_permutations: int,
    formal_permutations: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    collection_batch_size: int,
    eval_every: int,
    learning_rates: Sequence[float] | None,
    partition_seed: int,
    source_ages: Sequence[int] = (1,),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "checkpoint": str(checkpoint),
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if not (cfg.node_count == 8 and cfg.max_depth == 8 and cfg.d_model == 256 and cfg.n_layers == 2 and cfg.max_loops == 8):
        raise ValueError("interface specialness requires D8L8 N8 d256 B2")
    model.requires_grad_(False)
    train_perms, validation_perms, formal_perms = permutation_partitions(
        node_count=cfg.node_count,
        validation_count=validation_permutations,
        formal_count=formal_permutations,
        seed=partition_seed,
    )
    train_examples_spec = sample_interface_examples(
        train_permutations=train_perms,
        examples=train_examples,
        source_ages=source_ages,
        seed=interface_training_seed(controller_seed),
    )
    validation_examples = all_interface_examples(permutations=validation_perms, source_ages=source_ages)
    formal_examples = all_interface_examples(permutations=formal_perms, source_ages=source_ages)
    train_dataset = build_interface_dataset(
        source_model=model,
        cfg=cfg,
        permutations=train_examples_spec[0],
        starts=train_examples_spec[1],
        ages=train_examples_spec[2],
        horizon=8,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    validation_dataset = build_interface_dataset(
        source_model=model,
        cfg=cfg,
        permutations=validation_examples[0],
        starts=validation_examples[1],
        ages=validation_examples[2],
        horizon=8,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    formal_dataset = build_interface_dataset(
        source_model=model,
        cfg=cfg,
        permutations=formal_examples[0],
        starts=formal_examples[1],
        ages=formal_examples[2],
        horizon=8,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    candidates: list[CandidateResult] = []
    history: list[dict[str, Any]] = []
    for candidate_index, learning_rate in enumerate(learning_rates or _default_learning_rates(method)):
        result = train_candidate(
            method=method,
            base_model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            learning_rate=float(learning_rate),
            steps=steps,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            seed=interface_candidate_seed(controller_seed, candidate_index),
        )
        candidates.append(result)
        history.extend(result.history)
    selected = select_candidate(candidates)
    adapter = build_adapter(method, model).to(device)
    load_trainable_state(adapter, selected.trainable_state)
    adapter.eval()
    controlled_q = evaluate_curve(
        adapter=adapter,
        dataset=formal_dataset,
        horizon=8,
        batch_size=eval_batch_size,
        loop_index=cfg.max_loops,
    )
    raw_adapter = ExecutorAdapter(model, "raw_F")
    raw_q = evaluate_curve(
        adapter=raw_adapter,
        dataset=formal_dataset,
        horizon=8,
        batch_size=eval_batch_size,
        loop_index=cfg.max_loops,
    )
    pre = evaluate_pre_executor_q1(adapter=adapter, dataset=formal_dataset, batch_size=eval_batch_size)
    dependency = evaluate_j_dependency_controls(
        adapter=adapter,
        dataset=formal_dataset,
        batch_size=eval_batch_size,
        random_seed=partition_seed + controller_seed,
    )
    decision = interface_decision(
        controlled_q=[float(row["moving_target_accuracy"]) for row in controlled_q[:4]],
        pre_executor_q1=float(pre["moving_target_accuracy"]) if pre["applicable"] == "true" else float("inf"),
        random_executor_q1=float(dependency.get("random_executor_moving_target_accuracy", float("inf"))),
        shuffled_q1=float(dependency.get("shuffled_input_moving_target_accuracy", float("inf"))),
    ) if method == "j" else {"four_step_closure": all(float(row["moving_target_accuracy"]) >= 0.90 for row in controlled_q[:4])}
    region_curve = evaluate_j_region_curve(
        adapter=adapter,
        dataset=formal_dataset,
        batch_size=eval_batch_size,
        random_seed=partition_seed + 99,
    )
    candidate_rows = [
        {"learning_rate": candidate.learning_rate, **candidate.validation}
        for candidate in candidates
    ]
    summary: dict[str, Any] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "backbone_seed": int(Path(checkpoint).parent.name.rsplit("seed", 1)[-1]),
        "source_code_sha256": _sha256_file(Path(__file__)),
        "loss_placement": "mean_final_CE_after_each_closed_loop_q1_to_q4",
        "runtime_order": "adapter_then_F_each_call" if not adapter.changes_executor else "F_with_internal_adapter_each_call",
        "changes_executor": adapter.changes_executor,
        "config": asdict(cfg),
        "controller_seed": controller_seed,
        "trainable_parameters": adapter.trainable_parameter_count,
        "train_examples": train_examples,
        "validation_permutations": validation_permutations,
        "strict_eval_permutations": formal_permutations,
        "strict_eval_examples": formal_dataset.source.shape[0],
        "strict_eval_permutation_sha256": _permutation_digest(formal_perms),
        "source_ages": list(source_ages),
        "training_horizon": 4,
        "formal_horizon": 8,
        "learning_rates": list(learning_rates or _default_learning_rates(method)),
        "steps_per_candidate": steps,
        "selected_learning_rate": selected.learning_rate,
        "candidate_summary": candidate_rows,
        "controlled_q": controlled_q,
        "raw_f_q": raw_q,
        "pre_executor": pre,
        "dependency_controls": dependency,
        "region_curve": region_curve,
        "decision": decision,
        "peak_cuda_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0,
    }
    torch.save({"method": method, "selected_learning_rate": selected.learning_rate, "state": selected.trainable_state}, out_dir / "adapter.pt")
    _write_csv(out_dir / "candidate_rows.csv", candidate_rows)
    _write_csv(out_dir / "training_rows.csv", history)
    _write_csv(out_dir / "controlled_q.csv", controlled_q)
    _write_csv(out_dir / "raw_f_q.csv", raw_q)
    _write_csv(out_dir / "region_curve.csv", region_curve)
    _write_json_atomic(out_dir / "summary.json", summary)
    (out_dir / "REPORT_CN.md").write_text(_report_method(summary), encoding="utf-8")
    manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
    _write_json_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "method": method, "decision": decision}), flush=True)
    return summary


@torch.no_grad()
def repeated_interface_steps(
    *,
    transform: Callable[[Tensor], Tensor],
    receiver: Any,
    state: Tensor,
    horizon: int,
) -> list[Tensor]:
    """Run the same boundary controller before every frozen-executor step."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    current = state
    states: list[Tensor] = []
    for phase in range(horizon):
        current = receiver.apply_loop(
            transform(current).to(dtype=state.dtype),
            loop_index=receiver.cfg.max_loops + phase,
        )
        states.append(current)
    return states


@torch.no_grad()
def evaluate_transform_with_receiver(
    *,
    transform: DenseInterfaceJ,
    receiver: LoopedGraphPathTransformer | ConjugatedExecutor,
    dataset: InterfaceBatch,
    horizon: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    totals = [{"moving_examples": 0, "moving_correct": 0} for _ in range(horizon)]
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        states = repeated_interface_steps(
            transform=transform,
            receiver=receiver,
            state=batch.source,
            horizon=horizon,
        )
        for phase, state in enumerate(states):
            moving = batch.targets[:, phase].ne(batch.current)
            prediction = readout_logits(receiver, state).argmax(dim=-1)
            totals[phase]["moving_examples"] += int(moving.sum())
            totals[phase]["moving_correct"] += int(prediction[moving].eq(batch.targets[:, phase][moving]).sum())
    return [
        {"q": phase + 1, "moving_target_accuracy": row["moving_correct"] / max(row["moving_examples"], 1)}
        for phase, row in enumerate(totals)
    ]


def _fit_receiver_j(
    *,
    receiver: LoopedGraphPathTransformer | ConjugatedExecutor,
    train_dataset: InterfaceBatch,
    validation_dataset: InterfaceBatch,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_every: int,
    learning_rates: Sequence[float],
    seed: int,
) -> tuple[DenseInterfaceJ, CandidateResult]:
    candidates: list[CandidateResult] = []
    for index, learning_rate in enumerate(learning_rates):
        result = train_candidate(
            method="j",
            base_model=receiver,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            learning_rate=float(learning_rate),
            steps=steps,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            seed=seed + index,
        )
        candidates.append(result)
    selected = select_candidate(candidates)
    transform = DenseInterfaceJ(receiver.cfg.d_model).to(train_dataset.source.device)
    state = {name.removeprefix("transform."): value for name, value in selected.trainable_state.items()}
    transform.load_state_dict({name: value.to(train_dataset.source.device) for name, value in state.items()})
    transform.eval()
    return transform, selected


def run_swap_experiment(
    *,
    donor_checkpoint: Path,
    receiver_a_checkpoint: Path,
    receiver_b_checkpoint: Path,
    out_dir: Path,
    device_name: str,
    controller_seed: int,
    train_examples: int,
    validation_permutations: int,
    formal_permutations: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    collection_batch_size: int,
    eval_every: int,
    learning_rates: Sequence[float],
    partition_seed: int,
    source_ages: Sequence[int] = (1,),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "swap",
        "donor_checkpoint": str(donor_checkpoint),
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    donor, cfg, donor_payload = load_checkpoint(donor_checkpoint, device)
    receiver_a, cfg_a, payload_a = load_checkpoint(receiver_a_checkpoint, device)
    receiver_b, cfg_b, payload_b = load_checkpoint(receiver_b_checkpoint, device)
    if asdict(cfg) != asdict(cfg_a) or asdict(cfg) != asdict(cfg_b):
        raise ValueError("donor and receivers must share the identical D8L8 architecture")
    donor.requires_grad_(False)
    receiver_a.requires_grad_(False)
    receiver_b.requires_grad_(False)
    train_perms, validation_perms, formal_perms = permutation_partitions(
        node_count=cfg.node_count, validation_count=validation_permutations, formal_count=formal_permutations, seed=partition_seed
    )
    train_spec = sample_interface_examples(
        train_permutations=train_perms, examples=train_examples, source_ages=source_ages, seed=interface_training_seed(controller_seed)
    )
    validation_spec = all_interface_examples(permutations=validation_perms, source_ages=source_ages)
    formal_spec = all_interface_examples(permutations=formal_perms, source_ages=source_ages)
    build_kwargs = {"source_model": donor, "cfg": cfg, "horizon": 4, "device": device, "collection_batch_size": collection_batch_size}
    train_dataset = build_interface_dataset(permutations=train_spec[0], starts=train_spec[1], ages=train_spec[2], **build_kwargs)
    validation_dataset = build_interface_dataset(permutations=validation_spec[0], starts=validation_spec[1], ages=validation_spec[2], **build_kwargs)
    formal_dataset = build_interface_dataset(permutations=formal_spec[0], starts=formal_spec[1], ages=formal_spec[2], **build_kwargs)
    transform_a, selected_a = _fit_receiver_j(
        receiver=receiver_a, train_dataset=train_dataset, validation_dataset=validation_dataset, steps=steps, batch_size=batch_size, eval_batch_size=eval_batch_size, eval_every=eval_every, learning_rates=learning_rates, seed=interface_candidate_seed(controller_seed, 0)
    )
    transform_b, selected_b = _fit_receiver_j(
        receiver=receiver_b, train_dataset=train_dataset, validation_dataset=validation_dataset, steps=steps, batch_size=batch_size, eval_batch_size=eval_batch_size, eval_every=eval_every, learning_rates=learning_rates, seed=interface_candidate_seed(controller_seed, 0)
    )
    cells = {
        "J_A_to_F_A": evaluate_transform_with_receiver(transform=transform_a, receiver=receiver_a, dataset=formal_dataset, horizon=4, batch_size=eval_batch_size),
        "J_A_to_F_B": evaluate_transform_with_receiver(transform=transform_a, receiver=receiver_b, dataset=formal_dataset, horizon=4, batch_size=eval_batch_size),
        "J_B_to_F_A": evaluate_transform_with_receiver(transform=transform_b, receiver=receiver_a, dataset=formal_dataset, horizon=4, batch_size=eval_batch_size),
        "J_B_to_F_B": evaluate_transform_with_receiver(transform=transform_b, receiver=receiver_b, dataset=formal_dataset, horizon=4, batch_size=eval_batch_size),
    }
    diagonal = [cells["J_A_to_F_A"], cells["J_B_to_F_B"]]
    off_diagonal = [cells["J_A_to_F_B"], cells["J_B_to_F_A"]]
    summary = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "donor_checkpoint": str(donor_checkpoint),
        "donor_checkpoint_sha256": _sha256_file(donor_checkpoint),
        "receiver_a_checkpoint": str(receiver_a_checkpoint),
        "receiver_a_checkpoint_sha256": _sha256_file(receiver_a_checkpoint),
        "receiver_b_checkpoint": str(receiver_b_checkpoint),
        "receiver_b_checkpoint_sha256": _sha256_file(receiver_b_checkpoint),
        "checkpoint_steps": {"donor": donor_payload.get("step"), "A": payload_a.get("step"), "B": payload_b.get("step")},
        "source_code_sha256": _sha256_file(Path(__file__)),
        "source_state": "common donor states at the recorded source_ages for all four cells",
        "source_ages": list(source_ages),
        "donor_state_digest": formal_dataset.example_digest,
        "strict_eval_permutation_sha256": _permutation_digest(formal_perms),
        "strict_eval_examples": formal_dataset.source.shape[0],
        "controller_seed": controller_seed,
        "selected_learning_rates": {"J_A": selected_a.learning_rate, "J_B": selected_b.learning_rate},
        "cells": cells,
        "diagonal_mean_q1": sum(row[0]["moving_target_accuracy"] for row in diagonal) / 2,
        "off_diagonal_mean_q1": sum(row[0]["moving_target_accuracy"] for row in off_diagonal) / 2,
    }
    summary["diagonal_minus_offdiagonal_q1"] = summary["diagonal_mean_q1"] - summary["off_diagonal_mean_q1"]
    summary["decision"] = {
        "diagonal_learned": all(all(row["moving_target_accuracy"] >= 0.90 for row in curve) for curve in diagonal),
        "executor_specific_swap": summary["diagonal_mean_q1"] >= 0.90 and summary["diagonal_minus_offdiagonal_q1"] >= 0.40,
    }
    torch.save({"J_A": transform_a.state_dict(), "J_B": transform_b.state_dict()}, out_dir / "swap_controllers.pt")
    _write_json_atomic(out_dir / "summary.json", summary)
    rows = [
        {"cell": cell, **row}
        for cell, curve in cells.items()
        for row in curve
    ]
    _write_csv(out_dir / "swap_q.csv", rows)
    (out_dir / "REPORT_CN.md").write_text(
        "# D8L8 common-donor executor swap\n\n"
        f"- common donor formal digest: `{formal_dataset.example_digest}`.\n"
        f"- diagonal q1 mean: {summary['diagonal_mean_q1']:.4f}; off-diagonal q1 mean: {summary['off_diagonal_mean_q1']:.4f}; gap: {summary['diagonal_minus_offdiagonal_q1']:.4f}.\n"
        f"- decision: {summary['decision']}.\n",
        encoding="utf-8",
    )
    manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
    _write_json_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "swap_complete", "decision": summary["decision"]}), flush=True)
    return summary


def run_conjugated_swap_experiment(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    controller_seed: int,
    conjugation_seed: int,
    train_examples: int,
    validation_permutations: int,
    formal_permutations: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    collection_batch_size: int,
    eval_every: int,
    learning_rates: Sequence[float],
    partition_seed: int,
    source_ages: Sequence[int] = (1,),
) -> dict[str, Any]:
    """Calibrate executor-specificity with a behavior-equivalent residual ABI swap.

    The B executor represents the same input/output function as A after a fixed
    channel permutation, but requires a different hidden-state coordinate system.
    This is deliberately a controlled ABI test, rather than a claim that two
    independently trained seeds are necessarily swappable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "swap-conjugate",
        "checkpoint": str(checkpoint),
        "conjugation_seed": conjugation_seed,
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    donor, cfg, payload = load_checkpoint(checkpoint, device)
    donor.requires_grad_(False)
    receiver_a = donor
    permutation_generator = torch.Generator(device="cpu").manual_seed(conjugation_seed)
    permutation = torch.randperm(cfg.d_model, generator=permutation_generator)
    receiver_b = ConjugatedExecutor(copy.deepcopy(donor), permutation.to(device)).to(device).eval()
    train_perms, validation_perms, formal_perms = permutation_partitions(
        node_count=cfg.node_count,
        validation_count=validation_permutations,
        formal_count=formal_permutations,
        seed=partition_seed,
    )
    train_spec = sample_interface_examples(
        train_permutations=train_perms,
        examples=train_examples,
        source_ages=source_ages,
        seed=interface_training_seed(controller_seed),
    )
    validation_spec = all_interface_examples(permutations=validation_perms, source_ages=source_ages)
    formal_spec = all_interface_examples(permutations=formal_perms, source_ages=source_ages)
    build_kwargs = {
        "source_model": donor,
        "cfg": cfg,
        "horizon": 4,
        "device": device,
        "collection_batch_size": collection_batch_size,
    }
    train_dataset = build_interface_dataset(
        permutations=train_spec[0], starts=train_spec[1], ages=train_spec[2], **build_kwargs
    )
    validation_dataset = build_interface_dataset(
        permutations=validation_spec[0], starts=validation_spec[1], ages=validation_spec[2], **build_kwargs
    )
    formal_dataset = build_interface_dataset(
        permutations=formal_spec[0], starts=formal_spec[1], ages=formal_spec[2], **build_kwargs
    )
    train_dataset_b = conjugate_interface_batch(train_dataset, receiver_b)
    validation_dataset_b = conjugate_interface_batch(validation_dataset, receiver_b)
    formal_dataset_b = conjugate_interface_batch(formal_dataset, receiver_b)
    transform_a, selected_a = _fit_receiver_j(
        receiver=receiver_a,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        steps=steps,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        eval_every=eval_every,
        learning_rates=learning_rates,
        seed=interface_candidate_seed(controller_seed, 0),
    )
    transform_b, selected_b = _fit_receiver_j(
        receiver=receiver_b,
        train_dataset=train_dataset_b,
        validation_dataset=validation_dataset_b,
        steps=steps,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        eval_every=eval_every,
        learning_rates=learning_rates,
        seed=interface_candidate_seed(controller_seed, 0),
    )
    cells = {
        "J_A_to_F_A": evaluate_transform_with_receiver(
            transform=transform_a, receiver=receiver_a, dataset=formal_dataset, horizon=4, batch_size=eval_batch_size
        ),
        "J_A_to_F_B": evaluate_transform_with_receiver(
            transform=transform_a, receiver=receiver_b, dataset=formal_dataset, horizon=4, batch_size=eval_batch_size
        ),
        "J_B_to_F_A": evaluate_transform_with_receiver(
            transform=transform_b, receiver=receiver_a, dataset=formal_dataset_b, horizon=4, batch_size=eval_batch_size
        ),
        "J_B_to_F_B": evaluate_transform_with_receiver(
            transform=transform_b, receiver=receiver_b, dataset=formal_dataset_b, horizon=4, batch_size=eval_batch_size
        ),
    }
    diagonal = [cells["J_A_to_F_A"], cells["J_B_to_F_B"]]
    off_diagonal = [cells["J_A_to_F_B"], cells["J_B_to_F_A"]]
    permutation_sha256 = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
    summary = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "controlled_behavior_equivalent_residual_ABI_swap",
        "interpretation_boundary": "calibration of executor-coordinate binding; not an independent-seed swap result",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": payload.get("step"),
        "source_code_sha256": _sha256_file(Path(__file__)),
        "source_state": "same graph/path/age examples in each receiver ABI; B rows use exactly P times the A source state",
        "source_ages": list(source_ages),
        "donor_state_digest": formal_dataset.example_digest,
        "receiver_B_coordinate_matched_donor_state_digest": formal_dataset_b.example_digest,
        "strict_eval_permutation_sha256": _permutation_digest(formal_perms),
        "strict_eval_examples": formal_dataset.source.shape[0],
        "controller_seed": controller_seed,
        "conjugation_seed": conjugation_seed,
        "residual_channel_permutation_sha256": permutation_sha256,
        "receiver_A": "base executor F_A",
        "receiver_B": "F_B = P F_A P^-1 in the residual channel coordinates",
        "selected_learning_rates": {"J_A": selected_a.learning_rate, "J_B": selected_b.learning_rate},
        "cells": cells,
        "diagonal_mean_q1": sum(row[0]["moving_target_accuracy"] for row in diagonal) / 2,
        "off_diagonal_mean_q1": sum(row[0]["moving_target_accuracy"] for row in off_diagonal) / 2,
    }
    summary["diagonal_minus_offdiagonal_q1"] = summary["diagonal_mean_q1"] - summary["off_diagonal_mean_q1"]
    summary["decision"] = {
        "diagonal_learned": all(all(row["moving_target_accuracy"] >= 0.90 for row in curve) for curve in diagonal),
        "executor_specific_abi_swap": summary["diagonal_mean_q1"] >= 0.90 and summary["diagonal_minus_offdiagonal_q1"] >= 0.40,
    }
    torch.save({"J_A": transform_a.state_dict(), "J_B": transform_b.state_dict()}, out_dir / "swap_controllers.pt")
    _write_json_atomic(out_dir / "summary.json", summary)
    _write_csv(out_dir / "swap_q.csv", [{"cell": cell, **row} for cell, curve in cells.items() for row in curve])
    (out_dir / "REPORT_CN.md").write_text(
        "# D8L8 controlled residual-ABI executor swap\\n\\n"
        "F_B is behavior-equivalent to F_A after the recorded channel permutation; this is a controlled ABI calibration, not an independent-seed swap.\\n\\n"
        f"- base-coordinate donor formal digest: `{formal_dataset.example_digest}`; B-coordinate digest: `{formal_dataset_b.example_digest}`.\\n"
        f"- diagonal q1 mean: {summary['diagonal_mean_q1']:.4f}; off-diagonal q1 mean: {summary['off_diagonal_mean_q1']:.4f}; gap: {summary['diagonal_minus_offdiagonal_q1']:.4f}.\\n"
        f"- decision: {summary['decision']}.\\n",
        encoding="utf-8",
    )
    manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
    _write_json_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "conjugated_swap_complete", "decision": summary["decision"]}), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D8L8 interface-controller specialness campaign")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    method = subparsers.add_parser("method")
    method.add_argument("--checkpoint", type=Path, required=True)
    method.add_argument("--out-dir", type=Path, required=True)
    method.add_argument("--method", choices=("j", "reft32", "reft64", "reft128", "steering", "phase_steering", "lora_qkv32", "prefix64"), required=True)
    swap = subparsers.add_parser("swap")
    swap.add_argument("--donor-checkpoint", type=Path, required=True)
    swap.add_argument("--receiver-a-checkpoint", type=Path, required=True)
    swap.add_argument("--receiver-b-checkpoint", type=Path, required=True)
    swap.add_argument("--out-dir", type=Path, required=True)
    conjugated_swap = subparsers.add_parser("swap-conjugate")
    conjugated_swap.add_argument("--checkpoint", type=Path, required=True)
    conjugated_swap.add_argument("--out-dir", type=Path, required=True)
    conjugated_swap.add_argument("--conjugation-seed", type=int, default=47_091)
    for child in (method, swap, conjugated_swap):
        child.add_argument("--device", default="auto")
        child.add_argument("--controller-seed", type=int, default=0)
        child.add_argument("--train-examples", type=int, default=12_288)
        child.add_argument("--validation-permutations", type=int, default=128)
        child.add_argument("--formal-permutations", type=int, default=512)
        child.add_argument("--steps", type=int, default=3_000)
        child.add_argument("--batch-size", type=int, default=128)
        child.add_argument("--eval-batch-size", type=int, default=256)
        child.add_argument("--collection-batch-size", type=int, default=256)
        child.add_argument("--eval-every", type=int, default=250)
        child.add_argument("--learning-rates", type=float, nargs="+", default=None)
        child.add_argument("--partition-seed", type=int, default=8_609_204)
        child.add_argument("--source-ages", type=int, nargs="+", default=(1,))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "method":
        run_method_experiment(
            checkpoint=args.checkpoint, out_dir=args.out_dir, method=args.method, device_name=args.device,
            controller_seed=args.controller_seed, train_examples=args.train_examples,
            validation_permutations=args.validation_permutations, formal_permutations=args.formal_permutations,
            steps=args.steps, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            collection_batch_size=args.collection_batch_size, eval_every=args.eval_every,
            learning_rates=None if args.learning_rates is None else tuple(args.learning_rates), partition_seed=args.partition_seed,
            source_ages=tuple(args.source_ages),
        )
        return
    if args.mode == "swap":
        run_swap_experiment(
        donor_checkpoint=args.donor_checkpoint, receiver_a_checkpoint=args.receiver_a_checkpoint,
        receiver_b_checkpoint=args.receiver_b_checkpoint, out_dir=args.out_dir, device_name=args.device,
        controller_seed=args.controller_seed, train_examples=args.train_examples,
        validation_permutations=args.validation_permutations, formal_permutations=args.formal_permutations,
        steps=args.steps, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        collection_batch_size=args.collection_batch_size, eval_every=args.eval_every,
        learning_rates=tuple(args.learning_rates or _default_learning_rates("j")), partition_seed=args.partition_seed,
        source_ages=tuple(args.source_ages),
        )
        return
    run_conjugated_swap_experiment(
        checkpoint=args.checkpoint, out_dir=args.out_dir, device_name=args.device,
        controller_seed=args.controller_seed, conjugation_seed=args.conjugation_seed,
        train_examples=args.train_examples, validation_permutations=args.validation_permutations,
        formal_permutations=args.formal_permutations, steps=args.steps, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, collection_batch_size=args.collection_batch_size,
        eval_every=args.eval_every, learning_rates=tuple(args.learning_rates or _default_learning_rates("j")),
        partition_seed=args.partition_seed, source_ages=tuple(args.source_ages),
    )


if __name__ == "__main__":
    main()
