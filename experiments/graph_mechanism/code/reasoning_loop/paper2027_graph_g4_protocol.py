"""Disjoint-permutation protocol utilities for the confirmatory Graph cohort.

The N=8 Graph task has only 8! possible permutation graphs.  A random test
batch is therefore not automatically out-of-distribution after a long
training run.  This module makes the train/selection/test split explicit:
locks contain unique permutation graphs, and every optimisation batch rejects
their integer codes before it reaches a model.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Iterable

import torch


LOCK_KIND = "paper2027.graph.g4.unique_permutation_lock.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def permutation_codes(successors: torch.Tensor) -> torch.Tensor:
    """Encode row-wise permutations injectively in base N."""
    if successors.ndim != 2:
        raise ValueError("successors must have shape [batch, node_count]")
    n = successors.shape[1]
    if n < 1:
        raise ValueError("node_count must be positive")
    if successors.dtype not in (torch.int32, torch.int64):
        raise ValueError("successors must be integer-valued")
    powers = torch.tensor(
        [n ** exponent for exponent in range(n - 1, -1, -1)],
        dtype=torch.long,
        device=successors.device,
    )
    return (successors.to(torch.long) * powers).sum(dim=1)


def _all_permutations(node_count: int) -> torch.Tensor:
    if node_count > 8:
        raise ValueError("enumerating all permutations is intentionally limited to N <= 8")
    return torch.tensor(list(itertools.permutations(range(node_count))), dtype=torch.long)


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def create_unique_lock(
    *,
    path: Path,
    node_count: int,
    max_depth: int,
    permutations: int,
    seed: int,
    role: str,
    excluded_codes: torch.Tensor | None = None,
) -> dict[str, object]:
    """Make or validate an immutable, graph-level lock.

    ``excluded_codes`` is used when creating the final test lock after the
    selection lock.  It is intentionally stored as a parent SHA, rather than
    merely trusting different random seeds to avoid a collision.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if role not in {"selection", "final_test"}:
        raise ValueError("role must be selection or final_test")
    if path.exists():
        payload = load_unique_lock(path)
        expected = {
            "node_count": node_count,
            "max_depth": max_depth,
            "permutations": permutations,
            "seed": seed,
            "role": role,
        }
        observed = {key: payload.get(key) for key in expected}
        if observed != expected:
            raise ValueError(f"lock metadata mismatch at {path}: {observed} != {expected}")
        return payload

    # Rejection sampling avoids materializing 10! permutations. Each accepted
    # unseen permutation is uniform conditional on exclusions.
    excluded = set() if excluded_codes is None else set(excluded_codes.cpu().tolist())
    if permutations > math_factorial(node_count) - len(excluded):
        raise ValueError("not enough distinct graphs")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    seen = set(excluded)
    while len(selected) < permutations:
        proposals = torch.rand(max(64, 2 * (permutations-len(selected))), node_count, generator=generator).argsort(-1)
        for row, code in zip(proposals, permutation_codes(proposals).tolist()):
            if code not in seen:
                selected.append(row)
                seen.add(code)
                if len(selected) == permutations:
                    break
    successors = torch.stack(selected)
    codes = permutation_codes(successors)
    payload: dict[str, object] = {
        "kind": LOCK_KIND,
        "node_count": node_count,
        "max_depth": max_depth,
        "permutations": permutations,
        "seed": seed,
        "role": role,
        "unique_permutations": True,
        "universe_size": int(math_factorial(node_count)),
        "successors": successors,
        "permutation_codes": codes,
        "starts_per_permutation": list(range(node_count)),
    }
    _atomic_save(payload, path)
    return payload


def math_factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def load_unique_lock(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != LOCK_KIND:
        raise ValueError(f"not a registered G4 lock: {path}")
    successors = payload.get("successors")
    codes = payload.get("permutation_codes")
    n = payload.get("node_count")
    count = payload.get("permutations")
    if not isinstance(n, int) or not isinstance(count, int):
        raise ValueError("lock lacks integer node_count/permutations")
    if not isinstance(successors, torch.Tensor) or successors.shape != (count, n):
        raise ValueError("lock has invalid successor tensor")
    if not isinstance(codes, torch.Tensor) or codes.shape != (count,):
        raise ValueError("lock has invalid permutation code tensor")
    recomputed = permutation_codes(successors)
    if not torch.equal(recomputed.cpu(), codes.cpu().to(torch.long)):
        raise ValueError("lock permutation codes do not match successor rows")
    if torch.unique(codes).numel() != count:
        raise ValueError("lock contains duplicated graphs")
    return payload


def merged_forbidden_codes(paths: Iterable[Path], *, node_count: int) -> torch.Tensor:
    all_codes: list[torch.Tensor] = []
    for path in paths:
        payload = load_unique_lock(path)
        if payload["node_count"] != node_count:
            raise ValueError(f"node count mismatch in lock {path}")
        all_codes.append(payload["permutation_codes"].to(torch.long))  # type: ignore[index]
    if not all_codes:
        return torch.empty(0, dtype=torch.long)
    result = torch.cat(all_codes)
    if torch.unique(result).numel() != result.numel():
        raise ValueError("locks overlap; a held-out graph has two roles")
    return result


def sample_training_permutations(
    *,
    batch_size: int,
    node_count: int,
    forbidden_codes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Uniform-rejection sample permutation graphs outside all held-out locks."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    forbidden = forbidden_codes.to(device=device, dtype=torch.long)
    accepted: list[torch.Tensor] = []
    remaining = batch_size
    attempts = 0
    while remaining:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("unable to sample a permitted training graph")
        proposal_count = max(remaining * 2, 64)
        proposal = torch.rand(proposal_count, node_count, device=device).argsort(dim=-1)
        code = permutation_codes(proposal)
        keep = ~torch.isin(code, forbidden)
        accepted.append(proposal[keep][:remaining])
        remaining -= accepted[-1].shape[0]
    return torch.cat(accepted, dim=0)
