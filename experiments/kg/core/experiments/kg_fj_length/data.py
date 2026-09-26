"""Deterministic permutation worlds and variable-length KG composition data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


CompositionKey = tuple[int, ...]


@dataclass(frozen=True)
class KGLengthConfig:
    entity_count: int = 128
    relation_count: int = 16
    max_length: int = 6

    def __post_init__(self) -> None:
        if self.entity_count <= 1:
            raise ValueError("entity_count must exceed one")
        if self.relation_count <= 1:
            raise ValueError("relation_count must exceed one")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    @property
    def bos_token(self) -> int:
        return 0

    @property
    def entity_offset(self) -> int:
        return 1

    @property
    def relation_offset(self) -> int:
        return 1 + self.entity_count

    @property
    def vocabulary_size(self) -> int:
        return 1 + self.entity_count + self.relation_count


@dataclass(frozen=True)
class PermutationWorld:
    config: KGLengthConfig
    permutations: torch.Tensor
    seed: int

    @classmethod
    def create(cls, config: KGLengthConfig, seed: int) -> "PermutationWorld":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows = [torch.randperm(config.entity_count, generator=generator) for _ in range(config.relation_count)]
        return cls(config=config, permutations=torch.stack(rows), seed=seed)

    def __post_init__(self) -> None:
        expected_shape = (self.config.relation_count, self.config.entity_count)
        if self.permutations.shape != expected_shape or self.permutations.dtype != torch.long:
            raise ValueError("permutations have the wrong shape or dtype")
        expected = torch.arange(self.config.entity_count)
        if any(not torch.equal(torch.sort(row.cpu()).values, expected) for row in self.permutations):
            raise ValueError("every relation must be an entity permutation")

    def apply(self, start: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        if start.ndim != 1 or relations.ndim != 2 or relations.shape[0] != start.shape[0]:
            raise ValueError("start and relations must have shapes [batch] and [batch,length]")
        current = start.to(dtype=torch.long, device="cpu")
        relation_rows = relations.to(dtype=torch.long, device="cpu")
        for column in range(relation_rows.shape[1]):
            current = self.permutations[relation_rows[:, column], current]
        return current


@dataclass(frozen=True)
class CompositionBatch:
    tokens: torch.Tensor
    start: torch.Tensor
    relations: torch.Tensor
    target: torch.Tensor
    length: int

    def to(self, device: torch.device | str) -> "CompositionBatch":
        return CompositionBatch(
            tokens=self.tokens.to(device),
            start=self.start.to(device),
            relations=self.relations.to(device),
            target=self.target.to(device),
            length=self.length,
        )


def composition_key(start: int, relations: Iterable[int]) -> CompositionKey:
    return (int(start), *(int(relation) for relation in relations))


def _validate_length(config: KGLengthConfig, length: int) -> None:
    if not 1 <= length <= config.max_length:
        raise ValueError(f"length must be in [1, {config.max_length}]")


def _encode(
    config: KGLengthConfig,
    world: PermutationWorld,
    start: torch.Tensor,
    relations: torch.Tensor,
) -> CompositionBatch:
    batch_size, length = relations.shape
    tokens = torch.empty((batch_size, length + 2), dtype=torch.long)
    tokens[:, 0] = config.bos_token
    tokens[:, 1] = config.entity_offset + start
    tokens[:, 2:] = config.relation_offset + relations
    return CompositionBatch(
        tokens=tokens,
        start=start,
        relations=relations,
        target=world.apply(start, relations),
        length=length,
    )


def sample_batch(
    config: KGLengthConfig,
    world: PermutationWorld,
    batch_size: int,
    length: int,
    generator: torch.Generator,
    forbidden: frozenset[CompositionKey],
) -> CompositionBatch:
    """Draw iid relation compositions and reject exact held-out semantic keys."""
    _validate_length(config, length)
    if world.config != config:
        raise ValueError("world/config mismatch")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    universe_size = config.entity_count * config.relation_count**length
    if len(forbidden) >= universe_size:
        raise ValueError("forbidden keys exhaust the composition universe")

    start = torch.randint(config.entity_count, (batch_size,), generator=generator)
    relations = torch.randint(config.relation_count, (batch_size, length), generator=generator)
    for _ in range(10_000):
        rejected = torch.tensor(
            [
                composition_key(int(row_start), row_relations.tolist()) in forbidden
                for row_start, row_relations in zip(start, relations, strict=True)
            ],
            dtype=torch.bool,
        )
        rejected_count = int(rejected.sum())
        if rejected_count == 0:
            return _encode(config, world, start, relations)
        start[rejected] = torch.randint(config.entity_count, (rejected_count,), generator=generator)
        relations[rejected] = torch.randint(
            config.relation_count, (rejected_count, length), generator=generator
        )
    raise RuntimeError("failed to sample a non-forbidden composition batch")


def make_fixed_split(
    config: KGLengthConfig,
    world: PermutationWorld,
    length: int,
    count: int,
    seed: int,
    forbidden: frozenset[CompositionKey] = frozenset(),
) -> CompositionBatch:
    """Create an ordered deterministic split with globally unique semantic keys."""
    _validate_length(config, length)
    if count <= 0:
        raise ValueError("count must be positive")
    universe_size = config.entity_count * config.relation_count**length
    if count + len(forbidden) > universe_size:
        raise ValueError("requested unique split exceeds the available composition universe")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    seen = set(forbidden)
    starts: list[int] = []
    relation_rows: list[tuple[int, ...]] = []
    while len(starts) < count:
        start = int(torch.randint(config.entity_count, (), generator=generator))
        relations = tuple(
            int(value)
            for value in torch.randint(config.relation_count, (length,), generator=generator).tolist()
        )
        key = composition_key(start, relations)
        if key in seen:
            continue
        seen.add(key)
        starts.append(start)
        relation_rows.append(relations)
    return _encode(
        config,
        world,
        torch.tensor(starts, dtype=torch.long),
        torch.tensor(relation_rows, dtype=torch.long),
    )
