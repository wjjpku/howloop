"""Training and evaluation for the variable-length alternating F/J experiment."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import torch
from torch import nn

from .controller import (
    AlternatingTrace,
    ControllerConfig,
    build_controller,
    run_alternating,
)
from .data import (
    CompositionBatch,
    CompositionKey,
    KGLengthConfig,
    PermutationWorld,
    composition_key,
    make_fixed_split,
)
from .model import LoopedCompositionTransformer, ModelConfig, _position_buffer


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TrainConfig:
    train_lengths: tuple[int, ...]
    seed: int
    steps: int = 80_000
    batch_size: int = 512
    learning_rate: float = 3.0e-4
    warmup_steps: int = 2_000
    stable_steps: int = 60_000
    decay_steps: int = 18_000
    eval_interval: int = 2_000
    selection_count: int = 512
    test_count: int = 4_096
    gradient_clip: float = 1.0
    source_commit: str = "unknown"
    supervision_mode: str = "final_only"
    controller_loss_mode: str = "final_only"
    frontier_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.train_lengths or len(set(self.train_lengths)) != len(self.train_lengths):
            raise ValueError("train_lengths must be non-empty and unique")
        if any(not 1 <= length <= 6 for length in self.train_lengths):
            raise ValueError("train lengths must be in [1,6]")
        if self.steps <= 0 or self.batch_size <= 0 or self.eval_interval <= 0:
            raise ValueError("steps, batch_size, and eval_interval must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.warmup_steps < 0 or self.stable_steps < 0 or self.decay_steps < 0:
            raise ValueError("WSD phase lengths must be nonnegative")
        if self.warmup_steps + self.stable_steps + self.decay_steps != self.steps:
            raise ValueError("WSD phase lengths must sum to steps")
        if self.selection_count <= 0 or self.test_count <= 0:
            raise ValueError("selection_count and test_count must be positive")
        if self.supervision_mode not in {"final_only", "aligned_intermediate"}:
            raise ValueError(
                "supervision_mode must be final_only or aligned_intermediate"
            )
        if self.controller_loss_mode not in {
            "final_only",
            "local_successor",
            "truncated_unroll",
        }:
            raise ValueError(
                "controller_loss_mode must be final_only, local_successor, "
                "or truncated_unroll"
            )
        if self.frontier_weight < 0:
            raise ValueError("frontier_weight must be nonnegative")


def wsd_multiplier(step: int, warmup: int, stable: int, decay: int) -> float:
    total = warmup + stable + decay
    if not 1 <= step <= total or total <= 0:
        raise ValueError("step must be inside a nonempty WSD schedule")
    if step <= warmup:
        return step / max(warmup, 1)
    if step <= warmup + stable:
        return 1.0
    decay_step = step - warmup - stable
    return max(0.0, 1.0 - decay_step / max(decay, 1))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _torch_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _prepare_output(output_dir: Path | str, stage: str, config: dict[str, Any]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _json_write(output / "manifest.json", {"schema": "kg-fj-manifest-v1", "stage": stage, "status": "running", "config": config})
    return output


def _validate_sha256(value: str, label: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _load_trusted_torch(path: Path | str, expected_sha256: str, label: str) -> dict[str, Any]:
    _validate_sha256(expected_sha256, f"expected {label} sha256")
    candidate = Path(path).read_bytes()
    actual = hashlib.sha256(candidate).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} file sha256 mismatch with trusted root")
    payload = torch.load(io.BytesIO(candidate), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload must be a dictionary")
    return payload


def _model_from_payload(payload: dict[str, Any], device: torch.device | str) -> LoopedCompositionTransformer:
    if payload.get("schema") != "kg-fj-executor-checkpoint-v1":
        raise ValueError("executor checkpoint schema mismatch")
    kg_config = KGLengthConfig(**payload["kg_config"])
    model_config = ModelConfig(**payload["model_config"])
    model = LoopedCompositionTransformer(kg_config, model_config)
    model.load_state_dict(payload["model_state"])
    return model.to(device)


def load_executor_checkpoint(
    path: Path | str, expected_sha256: str, device: torch.device | str
) -> tuple[LoopedCompositionTransformer, dict[str, Any]]:
    payload = _load_trusted_torch(path, expected_sha256, "executor checkpoint")
    return _model_from_payload(payload, device), payload


def load_controller_checkpoint(
    path: Path | str,
    expected_sha256: str,
    expected_backbone_sha256: str,
    device: torch.device | str,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = _load_trusted_torch(path, expected_sha256, "controller checkpoint")
    if payload.get("schema") != "kg-fj-controller-checkpoint-v1":
        raise ValueError("controller checkpoint schema mismatch")
    if payload.get("backbone_file_sha256") != expected_backbone_sha256:
        raise ValueError("controller checkpoint backbone root mismatch")
    controller = build_controller(
        ControllerConfig(**payload["controller_config"]), seed=int(payload["seed"])
    )
    controller.load_state_dict(payload["controller_state"])
    return controller.to(device), payload


def extend_executor_max_length(
    model: LoopedCompositionTransformer, max_length: int
) -> LoopedCompositionTransformer:
    """Extend the deterministic PE buffer (all-zero for a NoPE model)."""
    current = model.kg_config.max_length
    if max_length < current:
        raise ValueError("max_length extension cannot shrink the registered range")
    if max_length == current:
        return model
    encoding = _position_buffer(model.model_config, max_length + 2).to(
        device=model.position_encoding.device, dtype=model.position_encoding.dtype
    )
    if not torch.equal(encoding[: current + 2], model.position_encoding):
        raise RuntimeError("extended positional encoding changed the registered prefix")
    model.kg_config = KGLengthConfig(
        entity_count=model.kg_config.entity_count,
        relation_count=model.kg_config.relation_count,
        max_length=max_length,
    )
    model.position_encoding = encoding
    return model


def _semantic_codes(start: torch.Tensor, relations: torch.Tensor, relation_count: int) -> torch.Tensor:
    """Return the exact unbounded-length composition tuple as int64 columns.

    Packing a whole relation string into one or two scalar limbs eventually
    overflows even when every individual entity/relation id is tiny. Keeping
    ``(start, r1, ..., rm)`` columnwise is exact for every practical sequence
    length. :func:`_membership` builds only a short overflow-checked prefix
    index for fast lookup and always verifies the complete candidate row.
    """
    if start.ndim != 1 or relations.ndim != 2 or start.shape[0] != relations.shape[0]:
        raise ValueError("start and relations must be aligned rank-1/rank-2 tensors")
    if relation_count <= 0:
        raise ValueError("relation_count must be positive")
    return torch.cat(
        (start.to(dtype=torch.long).unsqueeze(1), relations.to(dtype=torch.long)),
        dim=1,
    )


def _forbidden_codes(keys: Iterable[CompositionKey], length: int, relation_count: int, device: torch.device) -> torch.Tensor:
    values: list[tuple[int, ...]] = []
    for key in keys:
        if len(key) != length + 1:
            continue
        if key[0] < 0 or any(not 0 <= value < relation_count for value in key[1:]):
            raise ValueError("composition key contains an out-of-range value")
        values.append(tuple(int(value) for value in key))
    if not values:
        return torch.empty((0, length + 1), dtype=torch.long, device=device)
    return torch.tensor(sorted(set(values)), dtype=torch.long, device=device)


def _membership(codes: torch.Tensor, sorted_forbidden: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2 or sorted_forbidden.ndim != 2:
        raise ValueError("semantic codes must be rank-two")
    if codes.shape[1] != sorted_forbidden.shape[1]:
        raise ValueError("candidate and forbidden semantic widths differ")
    if sorted_forbidden.numel() == 0:
        return torch.zeros(codes.shape[0], dtype=torch.bool, device=codes.device)

    # Pack an int64-safe lexicographic prefix as an acceleration index. Every
    # hit is checked against the complete row below, so shared prefixes cannot
    # create false positives and no collision-prone hash enters the protocol.
    limit = torch.iinfo(torch.long).max
    upper_bound = max(
        int(codes[:, 0].max()), int(sorted_forbidden[:, 0].max())
    )
    radices: list[int] = []
    for column in range(1, min(codes.shape[1], 9)):
        radix = max(
            int(codes[:, column].max()),
            int(sorted_forbidden[:, column].max()),
        ) + 1
        radix = max(radix, 1)
        if upper_bound > (limit - (radix - 1)) // radix:
            break
        radices.append(radix)
        upper_bound = upper_bound * radix + radix - 1

    forbidden_heads = sorted_forbidden[:, 0].contiguous()
    candidate_heads = codes[:, 0].contiguous()
    for column, radix in enumerate(radices, start=1):
        forbidden_heads = forbidden_heads * radix + sorted_forbidden[:, column]
        candidate_heads = candidate_heads * radix + codes[:, column]
    left = torch.searchsorted(forbidden_heads, candidate_heads, right=False)
    right = torch.searchsorted(forbidden_heads, candidate_heads, right=True)
    spans = right - left
    found = torch.zeros(codes.shape[0], dtype=torch.bool, device=codes.device)
    max_span = int(spans.max())
    if max_span == 0:
        return found
    offsets = torch.arange(max_span, device=codes.device).unsqueeze(0)
    valid = offsets < spans.unsqueeze(1)
    candidate_indices = (left.unsqueeze(1) + offsets).clamp_max(
        sorted_forbidden.shape[0] - 1
    )
    candidate_rows = sorted_forbidden[candidate_indices]
    exact = torch.all(candidate_rows == codes.unsqueeze(1), dim=2)
    return torch.any(valid & exact, dim=1)


def _sample_device_batch(
    kg_config: KGLengthConfig,
    world_permutations: torch.Tensor,
    batch_size: int,
    length: int,
    generator: torch.Generator,
    forbidden_codes: torch.Tensor,
) -> CompositionBatch:
    device = world_permutations.device
    start = torch.randint(kg_config.entity_count, (batch_size,), generator=generator, device=device)
    relations = torch.randint(
        kg_config.relation_count, (batch_size, length), generator=generator, device=device
    )
    for _ in range(10_000):
        rejected = _membership(
            _semantic_codes(start, relations, kg_config.relation_count), forbidden_codes
        )
        rejected_count = int(rejected.sum())
        if rejected_count == 0:
            break
        start[rejected] = torch.randint(
            kg_config.entity_count, (rejected_count,), generator=generator, device=device
        )
        relations[rejected] = torch.randint(
            kg_config.relation_count,
            (rejected_count, length),
            generator=generator,
            device=device,
        )
    else:
        raise RuntimeError("device sampler could not avoid held-out keys")
    current = start
    for column in range(length):
        current = world_permutations[relations[:, column], current]
    tokens = torch.empty((batch_size, length + 2), dtype=torch.long, device=device)
    tokens[:, 0] = kg_config.bos_token
    tokens[:, 1] = kg_config.entity_offset + start
    tokens[:, 2:] = kg_config.relation_offset + relations
    return CompositionBatch(tokens, start, relations, current, length)


def _sample_device_batch_prefix_safe(
    kg_config: KGLengthConfig,
    world_permutations: torch.Tensor,
    batch_size: int,
    length: int,
    generator: torch.Generator,
    forbidden_by_length: dict[int, torch.Tensor],
) -> CompositionBatch:
    """Draw a fixed-container batch whose registered prefixes are all held out-safe."""
    device = world_permutations.device
    start = torch.randint(
        kg_config.entity_count, (batch_size,), generator=generator, device=device
    )
    relations = torch.randint(
        kg_config.relation_count,
        (batch_size, length),
        generator=generator,
        device=device,
    )
    relevant = sorted(value for value in forbidden_by_length if value <= length)
    for _ in range(10_000):
        rejected = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for prefix_length in relevant:
            rejected |= _membership(
                _semantic_codes(
                    start,
                    relations[:, :prefix_length],
                    kg_config.relation_count,
                ),
                forbidden_by_length[prefix_length],
            )
        rejected_count = int(rejected.sum())
        if rejected_count == 0:
            break
        start[rejected] = torch.randint(
            kg_config.entity_count,
            (rejected_count,),
            generator=generator,
            device=device,
        )
        relations[rejected] = torch.randint(
            kg_config.relation_count,
            (rejected_count, length),
            generator=generator,
            device=device,
        )
    else:
        raise RuntimeError("prefix-safe sampler could not avoid held-out keys")
    current = start
    for column in range(length):
        current = world_permutations[relations[:, column], current]
    tokens = torch.empty((batch_size, length + 2), dtype=torch.long, device=device)
    tokens[:, 0] = kg_config.bos_token
    tokens[:, 1] = kg_config.entity_offset + start
    tokens[:, 2:] = kg_config.relation_offset + relations
    return CompositionBatch(tokens, start, relations, current, length)


def _fixed_splits(
    kg_config: KGLengthConfig,
    world: PermutationWorld,
    lengths: tuple[int, ...],
    selection_count: int,
    test_count: int,
    seed: int,
) -> tuple[dict[int, CompositionBatch], dict[int, CompositionBatch], dict[int, frozenset[CompositionKey]]]:
    selection: dict[int, CompositionBatch] = {}
    test: dict[int, CompositionBatch] = {}
    forbidden_by_length: dict[int, frozenset[CompositionKey]] = {}
    for length in lengths:
        universe = kg_config.entity_count * kg_config.relation_count**length
        heldout_budget = max(2, universe // 2)
        selection_n = min(selection_count, max(1, heldout_budget // 2))
        test_n = min(test_count, heldout_budget - selection_n)
        selection_batch = make_fixed_split(
            kg_config, world, length, selection_n, seed + 100 * length
        )
        selection_keys = frozenset(
            composition_key(int(start), relations.tolist())
            for start, relations in zip(
                selection_batch.start, selection_batch.relations, strict=True
            )
        )
        test_batch = make_fixed_split(
            kg_config,
            world,
            length,
            test_n,
            seed + 100 * length + 1,
            forbidden=selection_keys,
        )
        test_keys = frozenset(
            composition_key(int(start), relations.tolist())
            for start, relations in zip(test_batch.start, test_batch.relations, strict=True)
        )
        selection[length] = selection_batch
        test[length] = test_batch
        # Random permutation edges are arbitrary atomic facts, so every
        # (entity, relation) pair must remain trainable.  Generalization is
        # defined over unseen multi-edge compositions, not unseen facts.
        forbidden_by_length[length] = (
            frozenset() if length == 1 else selection_keys | test_keys
        )
    return selection, test, forbidden_by_length


@torch.no_grad()
def _accuracy(
    model: LoopedCompositionTransformer,
    batch: CompositionBatch,
    device: torch.device,
    controller: nn.Module | None = None,
) -> float:
    moved = batch.to(device)
    if controller is None:
        logits = model(moved.tokens, calls=moved.length)
    else:
        output = run_alternating(model, controller, moved.tokens, calls=moved.length)
        assert isinstance(output, torch.Tensor)
        logits = output
    return float((logits.argmax(dim=-1) == moved.target).float().mean())


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _executor_training_loss(
    model: LoopedCompositionTransformer,
    batch: CompositionBatch,
    world_permutations: torch.Tensor,
    supervision_mode: str,
    frontier_head: nn.Module | None = None,
    frontier_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute either the registered endpoint loss or call/position-aligned prefix CE.

    For a query ``[BOS, e0, r1, ..., rm]``, aligned supervision reads relation
    position ``r_t`` (token index ``t + 1``) after exactly ``t`` F calls and
    supervises the composed entity ``e_t``.  Causal attention prevents that
    position from observing later relation tokens.
    """
    if frontier_head is None and supervision_mode == "final_only":
        return nn.functional.cross_entropy(
            model(batch.tokens, calls=batch.length), batch.target
        ), {
            "supervised_calls": [batch.length],
            "answer_indices": [batch.length + 1],
        }
    if supervision_mode not in {"final_only", "aligned_intermediate"}:
        raise ValueError("unknown executor supervision mode")
    state, input_embedding = model.prepare_recurrence(batch.tokens)
    _, states = model.run_raw(
        state,
        batch.length,
        return_states=True,
        input_embedding=input_embedding,
    )
    current = batch.start
    entity_losses: list[torch.Tensor] = []
    frontier_losses: list[torch.Tensor] = []
    for step, state in enumerate(states, start=1):
        current = world_permutations[batch.relations[:, step - 1], current]
        if supervision_mode == "aligned_intermediate" or step == batch.length:
            entity_losses.append(
                nn.functional.cross_entropy(
                    model.readout(state, answer_index=step + 1), current
                )
            )
        if frontier_head is not None:
            relation_state = state[:, 2 : batch.length + 2]
            frontier_logits = frontier_head(relation_state).squeeze(-1)
            target = (
                torch.arange(batch.length, device=state.device).unsqueeze(0) < step
            ).expand_as(frontier_logits)
            frontier_losses.append(
                nn.functional.binary_cross_entropy_with_logits(
                    frontier_logits, target.to(dtype=frontier_logits.dtype)
                )
            )
    entity_loss = torch.stack(entity_losses).mean()
    frontier_loss = (
        torch.stack(frontier_losses).mean()
        if frontier_losses
        else entity_loss.new_zeros(())
    )
    supervised_calls = (
        list(range(1, batch.length + 1))
        if supervision_mode == "aligned_intermediate"
        else [batch.length]
    )
    answer_indices = (
        list(range(2, batch.length + 2))
        if supervision_mode == "aligned_intermediate"
        else [batch.length + 1]
    )
    return entity_loss + frontier_weight * frontier_loss, {
        "supervised_calls": supervised_calls,
        "answer_indices": answer_indices,
        "entity_loss": float(entity_loss.detach()),
        "frontier_loss": float(frontier_loss.detach()),
        "frontier_weight": frontier_weight,
    }


def _train_executor(
    output_dir: Path | str,
    kind: str,
    kg_config: KGLengthConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    world_seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    expected = (1, 2, 3) if kind == "backbone" else (1, 2, 3, 4, 5, 6)
    if train_config.train_lengths != expected:
        raise ValueError(f"{kind} train_lengths must equal {expected}")
    output = _prepare_output(
        output_dir,
        kind,
        {"kg_config": asdict(kg_config), "model_config": asdict(model_config), "train_config": asdict(train_config), "world_seed": world_seed},
    )
    world = PermutationWorld.create(kg_config, world_seed)
    selection, test, forbidden = _fixed_splits(
        kg_config,
        world,
        train_config.train_lengths,
        train_config.selection_count,
        train_config.test_count,
        train_config.seed + 50_000,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(train_config.seed)
        model = LoopedCompositionTransformer(kg_config, model_config)
        frontier_head = (
            nn.Linear(model_config.d_model, 1)
            if train_config.frontier_weight > 0
            else None
        )
    target_device = torch.device(device)
    model.to(target_device)
    model.train()
    if frontier_head is not None:
        frontier_head.to(target_device).train()
    optimized_parameters = list(model.parameters())
    if frontier_head is not None:
        optimized_parameters.extend(frontier_head.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    generator_device = target_device.type if target_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(train_config.seed + 1)
    world_permutations = world.permutations.to(target_device)
    forbidden_tensors = {
        length: _forbidden_codes(keys, length, kg_config.relation_count, target_device)
        for length, keys in forbidden.items()
    }
    best_key: tuple[float, ...] | None = None
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_frontier_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    curves: list[dict[str, Any]] = []
    last_loss = float("nan")
    for step in range(1, train_config.steps + 1):
        length = train_config.train_lengths[(step - 1) % len(train_config.train_lengths)]
        batch = _sample_device_batch(
            kg_config,
            world_permutations,
            train_config.batch_size,
            length,
            generator,
            forbidden_tensors[length],
        )
        multiplier = wsd_multiplier(
            step,
            train_config.warmup_steps,
            train_config.stable_steps,
            train_config.decay_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = train_config.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        with _autocast(target_device):
            loss, loss_details = _executor_training_loss(
                model,
                batch,
                world_permutations,
                train_config.supervision_mode,
                frontier_head=frontier_head,
                frontier_weight=train_config.frontier_weight,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            optimized_parameters, train_config.gradient_clip
        )
        optimizer.step()
        last_loss = float(loss.detach())
        if step % train_config.eval_interval == 0 or step == train_config.steps:
            model.eval()
            if frontier_head is not None:
                frontier_head.eval()
            metrics = {
                str(value): _accuracy(model, selection[value], target_device)
                for value in train_config.train_lengths
            }
            macro = sum(metrics.values()) / len(metrics)
            key = (macro, *(metrics[str(value)] for value in train_config.train_lengths))
            progress = {"event": "training_eval", "stage": kind, "step": step, "loss": last_loss, "lr": optimizer.param_groups[0]["lr"], "macro_accuracy": macro, "per_length_accuracy": metrics, "supervision_mode": train_config.supervision_mode, "last_batch_supervision": loss_details}
            curves.append(progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            if best_key is None or key > best_key:
                best_key = key
                best_step = step
                best_metrics = metrics
                best_state = _cpu_state_dict(model)
                best_frontier_state = (
                    _cpu_state_dict(frontier_head)
                    if frontier_head is not None
                    else None
                )
            model.train()
            if frontier_head is not None:
                frontier_head.train()
    assert best_state is not None
    checkpoint = {
        "schema": "kg-fj-executor-checkpoint-v1",
        "kind": kind,
        "kg_config": asdict(kg_config),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "world_seed": world_seed,
        "seed": train_config.seed,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "model_state": best_state,
        "frontier_head_state": best_frontier_state,
        "frontier_head_config": (
            {"d_model": model_config.d_model, "output": 1}
            if best_frontier_state is not None
            else None
        ),
    }
    _torch_write(output / "best.pt", checkpoint)
    summary = {
        "schema": "kg-fj-training-summary-v1",
        "stage": kind,
        "trained_lengths": list(train_config.train_lengths),
        "supervision_mode": train_config.supervision_mode,
        "frontier_weight": train_config.frontier_weight,
        "best_step": best_step,
        "selection_accuracy": best_metrics,
        "test_accuracy": {},
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "curves": curves,
    }
    selected_model = _model_from_payload(checkpoint, target_device).eval()
    summary["test_accuracy"] = {
        str(length): _accuracy(selected_model, test[length], target_device)
        for length in train_config.train_lengths
    }
    _json_write(output / "summary.json", summary)
    _json_write(output / "manifest.json", {"schema": "kg-fj-manifest-v1", "stage": kind, "status": "complete", "outputs": {"best.pt": summary["best_checkpoint_sha256"], "summary.json": sha256_file(output / "summary.json")}})
    return summary


def train_backbone(
    output_dir: Path | str,
    kg_config: KGLengthConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    world_seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    return _train_executor(output_dir, "backbone", kg_config, model_config, train_config, world_seed, device)


def train_oracle(
    output_dir: Path | str,
    kg_config: KGLengthConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    world_seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    return _train_executor(output_dir, "oracle", kg_config, model_config, train_config, world_seed, device)


def _controller_training_loss(
    model: LoopedCompositionTransformer,
    controller: nn.Module,
    batch: CompositionBatch,
    world_permutations: torch.Tensor,
    loss_mode: str,
    calls: int,
    frontier_head: nn.Module | None = None,
    frontier_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not 1 <= calls <= batch.length:
        raise ValueError("controller calls must lie inside the sampled composition")
    if loss_mode == "final_only":
        logits = run_alternating(model, controller, batch.tokens, calls=calls)
        assert isinstance(logits, torch.Tensor)
        target = batch.start
        for column in range(calls):
            target = world_permutations[batch.relations[:, column], target]
        loss = nn.functional.cross_entropy(logits, target)
        return loss, {
            "loss_mode": loss_mode,
            "sampled_length": batch.length,
            "executed_calls": calls,
            "supervised_calls": [calls],
            "entity_loss": float(loss.detach()),
            "frontier_loss": 0.0,
        }
    if loss_mode not in {"local_successor", "truncated_unroll"}:
        raise ValueError("unknown controller loss mode")
    trace = run_alternating(
        model, controller, batch.tokens, calls=calls, return_states=True
    )
    assert isinstance(trace, AlternatingTrace)
    current = batch.start
    entity_losses: list[torch.Tensor] = []
    frontier_losses: list[torch.Tensor] = []
    supervised_calls: list[int] = []
    for step, state in enumerate(trace.post_f_states, start=1):
        current = world_permutations[batch.relations[:, step - 1], current]
        # Call 1 precedes every J and therefore carries no controller gradient.
        if step >= 2:
            entity_losses.append(
                nn.functional.cross_entropy(
                    model.readout(state, answer_index=step + 1), current
                )
            )
            supervised_calls.append(step)
        if frontier_head is not None:
            relation_state = state[:, 2 : batch.length + 2]
            frontier_logits = frontier_head(relation_state).squeeze(-1)
            target = (
                torch.arange(batch.length, device=state.device).unsqueeze(0) < step
            ).expand_as(frontier_logits)
            frontier_losses.append(
                nn.functional.binary_cross_entropy_with_logits(
                    frontier_logits, target.to(dtype=frontier_logits.dtype)
                )
            )
    entity_loss = torch.stack(entity_losses).mean()
    frontier_loss = (
        torch.stack(frontier_losses).mean()
        if frontier_losses
        else entity_loss.new_zeros(())
    )
    return entity_loss + frontier_weight * frontier_loss, {
        "loss_mode": loss_mode,
        "sampled_length": batch.length,
        "executed_calls": calls,
        "supervised_calls": supervised_calls,
        "entity_loss": float(entity_loss.detach()),
        "frontier_loss": float(frontier_loss.detach()),
        "frontier_weight": frontier_weight,
    }


def train_controller(
    output_dir: Path | str,
    backbone_checkpoint_path: Path | str,
    expected_backbone_sha256: str,
    controller_config: ControllerConfig,
    train_config: TrainConfig,
    device: torch.device | str,
) -> dict[str, Any]:
    if train_config.train_lengths != (4, 5, 6):
        raise ValueError("controller train_lengths must equal (4,5,6)")
    target_device = torch.device(device)
    model, backbone_payload = load_executor_checkpoint(
        backbone_checkpoint_path, expected_backbone_sha256, target_device
    )
    if controller_config.d_model != model.model_config.d_model:
        raise ValueError("controller/backbone width mismatch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    output = _prepare_output(
        output_dir,
        "controller",
        {"controller_config": asdict(controller_config), "train_config": asdict(train_config), "backbone_file_sha256": expected_backbone_sha256},
    )
    world = PermutationWorld.create(model.kg_config, int(backbone_payload["world_seed"]))
    selection, test, forbidden = _fixed_splits(
        model.kg_config,
        world,
        train_config.train_lengths,
        train_config.selection_count,
        train_config.test_count,
        train_config.seed + 70_000,
    )
    frontier_head: nn.Module | None = None
    if train_config.frontier_weight > 0:
        frontier_state = backbone_payload.get("frontier_head_state")
        if not isinstance(frontier_state, dict):
            raise ValueError(
                "frontier-supervised controller requires a frontier backbone checkpoint"
            )
        frontier_head = nn.Linear(model.model_config.d_model, 1).to(target_device)
        frontier_head.load_state_dict(frontier_state)
        frontier_head.eval()
        for parameter in frontier_head.parameters():
            parameter.requires_grad_(False)
    controller = build_controller(
        controller_config, seed=train_config.seed
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    generator_device = target_device.type if target_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(train_config.seed + 1)
    world_permutations = world.permutations.to(target_device)
    forbidden_tensors = {
        length: _forbidden_codes(keys, length, model.kg_config.relation_count, target_device)
        for length, keys in forbidden.items()
    }
    best_key: tuple[float, ...] | None = None
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    curves: list[dict[str, Any]] = []
    for step in range(1, train_config.steps + 1):
        if train_config.controller_loss_mode == "truncated_unroll":
            length = 6
            active_calls = int(
                torch.randint(4, 7, (), generator=generator, device=target_device)
            )
            batch = _sample_device_batch_prefix_safe(
                model.kg_config,
                world_permutations,
                train_config.batch_size,
                length,
                generator,
                forbidden_tensors,
            )
        else:
            length = train_config.train_lengths[(step - 1) % 3]
            active_calls = length
            batch = _sample_device_batch(
                model.kg_config,
                world_permutations,
                train_config.batch_size,
                length,
                generator,
                forbidden_tensors[length],
            )
        multiplier = wsd_multiplier(step, train_config.warmup_steps, train_config.stable_steps, train_config.decay_steps)
        for group in optimizer.param_groups:
            group["lr"] = train_config.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        with _autocast(target_device):
            loss, loss_details = _controller_training_loss(
                model,
                controller,
                batch,
                world_permutations,
                train_config.controller_loss_mode,
                active_calls,
                frontier_head=frontier_head,
                frontier_weight=train_config.frontier_weight,
            )
        loss.backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen backbone received a gradient")
        torch.nn.utils.clip_grad_norm_(controller.parameters(), train_config.gradient_clip)
        optimizer.step()
        if step % train_config.eval_interval == 0 or step == train_config.steps:
            controller.eval()
            metrics = {
                str(value): _accuracy(model, selection[value], target_device, controller)
                for value in train_config.train_lengths
            }
            macro = sum(metrics.values()) / 3
            key = (macro, *(metrics[str(value)] for value in train_config.train_lengths))
            progress = {"event": "training_eval", "stage": "controller", "step": step, "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"], "macro_accuracy": macro, "per_length_accuracy": metrics, "controller_architecture": controller_config.architecture, "controller_loss_mode": train_config.controller_loss_mode, "last_batch_supervision": loss_details}
            curves.append(progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            if best_key is None or key > best_key:
                best_key = key
                best_step = step
                best_metrics = metrics
                best_state = _cpu_state_dict(controller)
            controller.train()
    assert best_state is not None
    checkpoint = {
        "schema": "kg-fj-controller-checkpoint-v1",
        "controller_config": asdict(controller_config),
        "train_config": asdict(train_config),
        "seed": train_config.seed,
        "world_seed": int(backbone_payload["world_seed"]),
        "backbone_file_sha256": expected_backbone_sha256,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "controller_state": best_state,
    }
    _torch_write(output / "best.pt", checkpoint)
    selected = build_controller(controller_config, seed=train_config.seed).to(
        target_device
    )
    selected.load_state_dict(best_state)
    selected.eval()
    summary = {
        "schema": "kg-fj-training-summary-v1",
        "stage": "controller",
        "trained_lengths": [4, 5, 6],
        "execution": "F(JF)^(m-1)",
        "controller_architecture": controller_config.architecture,
        "controller_loss_mode": train_config.controller_loss_mode,
        "frontier_weight": train_config.frontier_weight,
        "best_step": best_step,
        "selection_accuracy": best_metrics,
        "test_accuracy": {str(length): _accuracy(model, test[length], target_device, selected) for length in train_config.train_lengths},
        "backbone_file_sha256": expected_backbone_sha256,
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "curves": curves,
    }
    _json_write(output / "summary.json", summary)
    _json_write(output / "manifest.json", {"schema": "kg-fj-manifest-v1", "stage": "controller", "status": "complete", "inputs": {"backbone_file_sha256": expected_backbone_sha256}, "outputs": {"best.pt": summary["best_checkpoint_sha256"], "summary.json": sha256_file(output / "summary.json")}})
    return summary


@torch.no_grad()
def evaluate_length_grid(
    model: LoopedCompositionTransformer,
    world: PermutationWorld,
    controller: nn.Module | None,
    count: int,
    seed: int,
    device: torch.device | str,
    lengths: tuple[int, ...] | None = None,
    adaptive_extra_calls: int = 0,
) -> dict[str, dict[str, Any]]:
    target_device = torch.device(device)
    model = model.to(target_device).eval()
    if controller is not None:
        controller = controller.to(target_device).eval()
    selected_lengths = lengths or tuple(range(1, model.kg_config.max_length + 1))
    if not selected_lengths or len(set(selected_lengths)) != len(selected_lengths):
        raise ValueError("evaluation lengths must be nonempty and unique")
    if any(not 1 <= length <= model.kg_config.max_length for length in selected_lengths):
        raise ValueError("evaluation length is outside the model range")
    if adaptive_extra_calls < 0:
        raise ValueError("adaptive_extra_calls must be nonnegative")
    rows: dict[str, dict[str, Any]] = {}
    for length in selected_lengths:
        universe = model.kg_config.entity_count * model.kg_config.relation_count**length
        fixed = make_fixed_split(model.kg_config, world, length, min(count, universe), seed + length).to(target_device)
        initial, input_embedding = model.prepare_recurrence(fixed.tokens)
        raw_final, raw_states = model.run_raw(
            initial,
            length,
            return_states=True,
            input_embedding=input_embedding,
        )
        raw_logits = model.readout(raw_final)
        raw_accuracy = float((raw_logits.argmax(-1) == fixed.target).float().mean())
        per_call = [float((model.readout(state).argmax(-1) == fixed.target).float().mean()) for state in raw_states]
        current = fixed.start
        aligned_prefix_accuracy: list[float] = []
        world_permutations = world.permutations.to(target_device)
        for step, state in enumerate(raw_states, start=1):
            current = world_permutations[fixed.relations[:, step - 1], current]
            aligned_prefix_accuracy.append(
                float(
                    (
                        model.readout(state, answer_index=step + 1).argmax(-1)
                        == current
                    )
                    .float()
                    .mean()
                )
            )
        row: dict[str, Any] = {
            "count": int(fixed.target.numel()),
            "raw_accuracy": raw_accuracy,
            "raw_per_call_accuracy": per_call,
            "aligned_prefix_accuracy": aligned_prefix_accuracy,
            "learned_j_accuracy": None,
            "pre_final_j_accuracy": None,
        }
        if controller is not None:
            maximum_calls = length + adaptive_extra_calls
            trace = run_alternating(
                model,
                controller,
                fixed.tokens,
                maximum_calls,
                return_states=True,
            )
            assert isinstance(trace, AlternatingTrace)
            registered_logits = model.readout(trace.post_f_states[length - 1])
            row["learned_j_accuracy"] = float(
                (registered_logits.argmax(-1) == fixed.target).float().mean()
            )
            if length > 1:
                registered_pre_f = trace.post_j_states[length - 2]
                row["pre_final_j_accuracy"] = float(
                    (model.readout(registered_pre_f).argmax(-1) == fixed.target)
                    .float()
                    .mean()
                )
            if adaptive_extra_calls > 0:
                call_logits = torch.stack(
                    [model.readout(state) for state in trace.post_f_states], dim=1
                )
                predictions = call_logits.argmax(-1)
                confidences = call_logits.softmax(-1).amax(-1)
                selected_call = confidences.argmax(-1)
                selected_prediction = predictions.gather(
                    1, selected_call.unsqueeze(1)
                ).squeeze(1)
                per_call_accuracy = (
                    predictions.eq(fixed.target.unsqueeze(1)).float().mean(0)
                )
                best_fixed_index = int(per_call_accuracy.argmax())
                histogram = torch.bincount(
                    selected_call.cpu(), minlength=maximum_calls
                )
                row.update(
                    {
                        "adaptive_extra_calls": adaptive_extra_calls,
                        "controller_per_call_accuracy": [
                            float(value) for value in per_call_accuracy
                        ],
                        "adaptive_instance_accuracy": float(
                            selected_prediction.eq(fixed.target).float().mean()
                        ),
                        "adaptive_selected_call_histogram": {
                            str(index + 1): int(value)
                            for index, value in enumerate(histogram.tolist())
                        },
                        "oracle_any_call_accuracy": float(
                            predictions.eq(fixed.target.unsqueeze(1))
                            .any(1)
                            .float()
                            .mean()
                        ),
                        "best_fixed_call": best_fixed_index + 1,
                        "best_fixed_call_accuracy": float(
                            per_call_accuracy[best_fixed_index]
                        ),
                    }
                )
        rows[str(length)] = row
    return rows
