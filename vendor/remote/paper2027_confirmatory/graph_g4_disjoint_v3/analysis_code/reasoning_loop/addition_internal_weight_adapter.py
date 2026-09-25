from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from reasoning_loop.paper_length_telomere import (
    PaperBatch,
    PaperLoopedTransformer,
    PaperTaskSpec,
    atomic_json_dump,
    atomic_torch_save,
    generate_paper_batch,
    load_backbone,
    write_csv,
)


ADAPTER_SITES = ("q", "k", "v", "mlp_pre")


class SharedInternalWeightAdapter(torch.nn.Module):
    """One additive d_model x d_model matrix shared across physical layers."""

    def __init__(
        self,
        *,
        model: PaperLoopedTransformer,
        site: str,
        init_std: float,
        seed: int,
    ) -> None:
        super().__init__()
        if site not in ADAPTER_SITES:
            raise ValueError(f"unsupported adapter site: {site}")
        if init_std < 0:
            raise ValueError("adapter initialization std must be non-negative")
        self.site = site
        self.dimension = model.config.d_model
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        initial = torch.randn(
            self.dimension,
            self.dimension,
            generator=generator,
            dtype=torch.float32,
        ) * init_std
        self.delta = torch.nn.Parameter(initial)
        # Keep the frozen backbone outside this module's state_dict.  The
        # adapter artifact must contain exactly one trainable matrix.
        object.__setattr__(self, "_model", model)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _qkv_hook(
        self,
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        update = F.linear(inputs[0], self.delta)
        chunks = list(output.chunk(3, dim=-1))
        chunks[("q", "k", "v").index(self.site)] = (
            chunks[("q", "k", "v").index(self.site)] + update
        )
        return torch.cat(chunks, dim=-1)

    def _mlp_pre_hook(
        self,
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        value = inputs[0]
        return (value + F.linear(value, self.delta), *inputs[1:])

    def attach(self) -> None:
        if self._handles:
            raise RuntimeError("adapter is already attached")
        for layer in self._model.layers:
            if self.site in {"q", "k", "v"}:
                handle = layer.attention.qkv.register_forward_hook(
                    self._qkv_hook
                )
            else:
                handle = layer.mlp.register_forward_pre_hook(
                    self._mlp_pre_hook
                )
            self._handles.append(handle)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def applied(self) -> Iterator[SharedInternalWeightAdapter]:
        self.attach()
        try:
            yield self
        finally:
            self.detach()


def freeze_backbone_for_adapter(
    model: PaperLoopedTransformer,
    adapter: SharedInternalWeightAdapter,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter.delta.requires_grad_(True)
    model.eval()
    adapter.train()


def count_trainable_parameters(
    model: PaperLoopedTransformer,
    adapter: SharedInternalWeightAdapter,
) -> int:
    return sum(
        parameter.numel()
        for parameter in [*model.parameters(), *adapter.parameters()]
        if parameter.requires_grad
    )


def sample_training_length(
    *,
    update: int,
    id_min: int,
    id_max: int,
    repair_min: int,
    repair_max: int,
    generator: torch.Generator,
) -> int:
    if update < 1:
        raise ValueError("update must be positive")
    if not 1 <= id_min <= id_max < repair_min <= repair_max:
        raise ValueError("ID and repair bands must be positive and disjoint")
    low, high = (
        (id_min, id_max) if update % 2 else (repair_min, repair_max)
    )
    return int(torch.randint(low, high + 1, (), generator=generator))


def advance_training_generator(
    *,
    spec: PaperTaskSpec,
    generator: torch.Generator,
    prior_updates: int,
    batch_size: int,
    id_min: int,
    id_max: int,
    repair_min: int,
    repair_max: int,
) -> None:
    """Replay data draws so continuation does not repeat earlier examples."""
    if prior_updates < 0:
        raise ValueError("prior updates must be non-negative")
    for update in range(1, prior_updates + 1):
        length = sample_training_length(
            update=update,
            id_min=id_min,
            id_max=id_max,
            repair_min=repair_min,
            repair_max=repair_max,
            generator=generator,
        )
        generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=length,
            max_length=length,
            fixed_length=length,
            generator=generator,
        )


def forward_fixed_length(
    *,
    model: PaperLoopedTransformer,
    inputs: torch.Tensor,
    steps: int,
    gradient_checkpointing: bool,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    token_embeddings = model.read_in(inputs)
    position_signal = model._position_signal(inputs)
    state = torch.zeros_like(token_embeddings)
    for step_index in range(1, steps + 1):
        embedded = token_embeddings
        if position_signal is not None and (
            model.config.position_injection == "every_loop" or step_index == 1
        ):
            embedded = token_embeddings + position_signal
        if gradient_checkpointing:
            state = checkpoint(
                model.recurrent_step,
                state,
                embedded,
                use_reentrant=False,
            )
        else:
            state = model.recurrent_step(state, embedded)
    return state


def wsd_learning_rate(
    *,
    update: int,
    total_updates: int,
    peak: float,
    warmup_updates: int,
    stable_updates: int,
    final_ratio: float,
) -> float:
    if not 1 <= update <= total_updates:
        raise ValueError("update lies outside the schedule")
    if peak <= 0 or warmup_updates < 0 or stable_updates < 0:
        raise ValueError("invalid WSD schedule")
    if not 0.0 < final_ratio <= 1.0:
        raise ValueError("final ratio must lie in (0, 1]")
    decay_updates = total_updates - warmup_updates - stable_updates
    if decay_updates < 1:
        raise ValueError("WSD schedule requires at least one decay update")
    if warmup_updates and update <= warmup_updates:
        return peak * update / warmup_updates
    if update <= warmup_updates + stable_updates:
        return peak
    decay_index = update - warmup_updates - stable_updates
    progress = decay_index / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (final_ratio + (1.0 - final_ratio) * cosine)


@torch.no_grad()
def target_metrics(logits: torch.Tensor, batch: PaperBatch) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    answer_correct = predictions.eq(batch.targets)[batch.answer_mask]
    per_example_correct = predictions.eq(batch.targets) | ~batch.answer_mask
    return {
        "strict_exact_match": float(
            per_example_correct.all(dim=1).float().mean()
        ),
        "answer_token_accuracy": float(answer_correct.float().mean()),
        "answer_nll": float(
            F.cross_entropy(
                logits[batch.answer_mask], batch.targets[batch.answer_mask]
            )
        ),
    }


def select_best_snapshot(
    candidates: Sequence[dict[str, Any]], *, id_gate: float
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("snapshot selection requires candidates")
    passing = [
        candidate
        for candidate in candidates
        if float(candidate["id_mean_em"]) >= id_gate
    ]
    if passing:
        return max(
            passing,
            key=lambda candidate: (
                float(candidate["seen_ood_mean_em"]),
                float(candidate["id_mean_em"]),
                int(candidate["update"]),
            ),
        )
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["id_mean_em"]),
            float(candidate["seen_ood_mean_em"]),
            int(candidate["update"]),
        ),
    )


@torch.no_grad()
def evaluate_target_lengths(
    *,
    model: PaperLoopedTransformer,
    spec: PaperTaskSpec,
    lengths: Sequence[int],
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if batch_size < 1 or batches < 1:
        raise ValueError("evaluation batch size and batch count must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    for length in lengths:
        totals = {
            "strict_exact_match": 0.0,
            "answer_token_accuracy": 0.0,
            "answer_nll": 0.0,
        }
        for _ in range(batches):
            batch = generate_paper_batch(
                spec,
                batch_size=batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            state = forward_fixed_length(
                model=model,
                inputs=batch.inputs,
                steps=length + spec.step_offset,
                gradient_checkpointing=False,
            )
            metrics = target_metrics(model.decode(state).float(), batch)
            for key in totals:
                totals[key] += metrics[key]
        rows.append(
            {
                "length": int(length),
                "target_step": int(length + spec.step_offset),
                **{key: value / batches for key, value in totals.items()},
                "examples": int(batch_size * batches),
            }
        )
    return rows


def band_means(
    rows: Sequence[dict[str, Any]],
    *,
    id_lengths: Sequence[int],
    seen_ood_lengths: Sequence[int],
) -> dict[str, float]:
    by_length = {int(row["length"]): row for row in rows}

    def mean_for(lengths: Sequence[int]) -> float:
        if not lengths or any(int(length) not in by_length for length in lengths):
            raise ValueError("band mean requested missing lengths")
        return sum(
            float(by_length[int(length)]["strict_exact_match"])
            for length in lengths
        ) / len(lengths)

    return {
        "id_mean_em": mean_for(id_lengths),
        "seen_ood_mean_em": mean_for(seen_ood_lengths),
    }


def _parse_lengths(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("length lists must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError("length lists must not contain duplicates")
    return values


def _band_summary(
    rows: Sequence[dict[str, Any]],
    *,
    id_min: int,
    id_max: int,
    repair_min: int,
    repair_max: int,
) -> dict[str, dict[str, float | int | None]]:
    bands = {
        "id": (id_min, id_max),
        "seen_ood": (repair_min, repair_max),
        "unseen_ood": (repair_max + 1, math.inf),
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for name, (low, high) in bands.items():
        selected = [
            row for row in rows if low <= int(row["length"]) <= high
        ]
        result[name] = {
            "evaluated_lengths": len(selected),
            "mean_strict_exact_match": (
                sum(float(row["strict_exact_match"]) for row in selected)
                / len(selected)
                if selected
                else None
            ),
            "mean_answer_token_accuracy": (
                sum(float(row["answer_token_accuracy"]) for row in selected)
                / len(selected)
                if selected
                else None
            ),
        }
    return result


def _annotate_regimes(
    rows: Sequence[dict[str, Any]],
    *,
    id_max: int,
    repair_max: int,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        length = int(row["length"])
        regime = (
            "id"
            if length <= id_max
            else "seen_ood"
            if length <= repair_max
            else "unseen_ood"
        )
        annotated.append({**row, "regime": regime})
    return annotated


def _adapter_norms(
    model: PaperLoopedTransformer,
    adapter: SharedInternalWeightAdapter,
) -> dict[str, Any]:
    delta_norm = float(adapter.delta.detach().float().norm().cpu())
    if adapter.site in {"q", "k", "v"}:
        index = ("q", "k", "v").index(adapter.site)
        dimension = model.config.d_model
        base_norms = [
            float(
                layer.attention.qkv.weight.detach()
                .float()[index * dimension : (index + 1) * dimension]
                .norm()
                .cpu()
            )
            for layer in model.layers
        ]
        reference = sum(base_norms) / len(base_norms)
        reference_name = "mean_selected_qkv_base_frobenius"
    else:
        base_norms = []
        reference = math.sqrt(model.config.d_model)
        reference_name = "identity_frobenius"
    return {
        "delta_frobenius": delta_norm,
        reference_name: reference,
        "delta_to_reference_ratio": delta_norm / max(reference, 1e-12),
        "per_physical_layer_base_frobenius": base_norms,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one shared internal Addition weight adapter."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--prior-updates", type=int, default=0)
    parser.add_argument("--site", choices=ADAPTER_SITES, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-min", type=int, required=True)
    parser.add_argument("--id-max", type=int, required=True)
    parser.add_argument("--repair-min", type=int, required=True)
    parser.add_argument("--repair-max", type=int, required=True)
    parser.add_argument("--updates", type=int, default=5376)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--init-std", type=float, default=1e-4)
    parser.add_argument("--warmup-updates", type=int, default=2048)
    parser.add_argument("--stable-updates", type=int, default=2816)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval-every", type=int, default=256)
    parser.add_argument("--selection-id-lengths", required=True)
    parser.add_argument("--selection-seen-lengths", required=True)
    parser.add_argument("--final-lengths", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--selection-eval-batches", type=int, default=2)
    parser.add_argument("--final-eval-batches", type=int, default=8)
    parser.add_argument("--id-gate", type=float, default=0.99)
    parser.add_argument("--force", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.id_min <= args.id_max < args.repair_min <= args.repair_max:
        raise ValueError("ID and repair bands must be positive and disjoint")
    if args.updates < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("training counts must be positive")
    if args.prior_updates < 0:
        raise ValueError("prior updates must be non-negative")
    if args.prior_updates and args.adapter_checkpoint is None:
        raise ValueError("prior updates require an adapter checkpoint")
    if args.adapter_checkpoint is not None and args.prior_updates < 1:
        raise ValueError("adapter continuation requires positive prior updates")
    if args.eval_batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if args.selection_eval_batches < 1 or args.final_eval_batches < 1:
        raise ValueError("evaluation batch counts must be positive")
    if args.learning_rate <= 0 or args.init_std < 0 or args.grad_clip <= 0:
        raise ValueError("invalid optimizer or initialization setting")
    if not 0.0 <= args.id_gate <= 1.0:
        raise ValueError("ID gate must lie in [0, 1]")
    # Validate the full schedule now, rather than after a long run begins.
    wsd_learning_rate(
        update=1,
        total_updates=args.updates,
        peak=args.learning_rate,
        warmup_updates=args.warmup_updates,
        stable_updates=args.stable_updates,
        final_ratio=args.final_lr_ratio,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    selection_id_lengths = _parse_lengths(args.selection_id_lengths)
    selection_seen_lengths = _parse_lengths(args.selection_seen_lengths)
    final_lengths = _parse_lengths(args.final_lengths)
    if any(not args.id_min <= value <= args.id_max for value in selection_id_lengths):
        raise ValueError("selection ID lengths lie outside the ID band")
    if any(
        not args.repair_min <= value <= args.repair_max
        for value in selection_seen_lengths
    ):
        raise ValueError("selection seen-OOD lengths lie outside the repair band")

    out_dir = args.out_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    adapter_checkpoint_path = (
        args.adapter_checkpoint.expanduser().resolve()
        if args.adapter_checkpoint is not None
        else None
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if adapter_checkpoint_path is not None and not adapter_checkpoint_path.is_file():
        raise FileNotFoundError(adapter_checkpoint_path)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"output directory is non-empty; pass --force to overwrite: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.amp and device.type != "cuda":
        raise ValueError("bfloat16 autocast is supported here only on CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    manifest: dict[str, Any] = {
        "status": "running",
        "checkpoint": str(checkpoint_path),
        "adapter_checkpoint": (
            str(adapter_checkpoint_path) if adapter_checkpoint_path else None
        ),
        "prior_updates": args.prior_updates,
        "site": args.site,
        "seed": args.seed,
        "started_unix": started_at,
        "device": str(device),
    }
    atomic_json_dump(manifest, out_dir / "manifest.json")

    try:
        model, spec, backbone_payload = load_backbone(
            checkpoint_path, device=device
        )
        if spec.name != "addition":
            raise ValueError("internal weight adapter experiment requires Addition")
        adapter = SharedInternalWeightAdapter(
            model=model,
            site=args.site,
            init_std=args.init_std,
            seed=args.seed,
        ).to(device)
        if adapter_checkpoint_path is not None:
            adapter_payload = torch.load(
                adapter_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if adapter_payload.get("kind") != "addition_internal_weight_adapter":
                raise ValueError("checkpoint is not an internal weight adapter")
            if adapter_payload.get("site") != args.site:
                raise ValueError("adapter checkpoint site does not match --site")
            source_backbone = Path(
                adapter_payload["backbone_checkpoint"]
            ).expanduser().resolve()
            if source_backbone != checkpoint_path:
                raise ValueError("adapter checkpoint backbone does not match")
            loaded_delta = adapter_payload["delta"].detach().float()
            if loaded_delta.shape != adapter.delta.shape:
                raise ValueError("adapter checkpoint matrix shape does not match")
            adapter.delta.data.copy_(loaded_delta.to(device))
            adapter_initialization = {
                "source": "checkpoint",
                "checkpoint": str(adapter_checkpoint_path),
                "source_selected_update": int(
                    adapter_payload.get("selected_update", -1)
                ),
                "loaded_delta_frobenius": float(loaded_delta.norm()),
            }
        else:
            adapter_initialization = {
                "source": "random_normal",
                "standard_deviation": args.init_std,
                "loaded_delta_frobenius": float(
                    adapter.delta.detach().float().norm().cpu()
                ),
            }
        freeze_backbone_for_adapter(model, adapter)
        trainable = count_trainable_parameters(model, adapter)
        expected_trainable = model.config.d_model**2
        if trainable != expected_trainable:
            raise RuntimeError(
                f"expected {expected_trainable} trainable parameters, got {trainable}"
            )

        optimizer = torch.optim.AdamW(
            [adapter.delta], lr=args.learning_rate, weight_decay=0.0
        )
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(args.seed + 10_000)
        advance_training_generator(
            spec=spec,
            generator=train_generator,
            prior_updates=args.prior_updates,
            batch_size=args.batch_size,
            id_min=args.id_min,
            id_max=args.id_max,
            repair_min=args.repair_min,
            repair_max=args.repair_max,
        )
        history: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        candidate_deltas: dict[int, torch.Tensor] = {}
        log_path = out_dir / "training.jsonl"

        with adapter.applied():
            for update in range(1, args.updates + 1):
                cumulative_update = args.prior_updates + update
                length = sample_training_length(
                    update=cumulative_update,
                    id_min=args.id_min,
                    id_max=args.id_max,
                    repair_min=args.repair_min,
                    repair_max=args.repair_max,
                    generator=train_generator,
                )
                batch = generate_paper_batch(
                    spec,
                    batch_size=args.batch_size,
                    min_length=length,
                    max_length=length,
                    fixed_length=length,
                    generator=train_generator,
                ).to(device)
                learning_rate = wsd_learning_rate(
                    update=update,
                    total_updates=args.updates,
                    peak=args.learning_rate,
                    warmup_updates=args.warmup_updates,
                    stable_updates=args.stable_updates,
                    final_ratio=args.final_lr_ratio,
                )
                optimizer.param_groups[0]["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.amp,
                ):
                    state = forward_fixed_length(
                        model=model,
                        inputs=batch.inputs,
                        steps=length + spec.step_offset,
                        gradient_checkpointing=args.gradient_checkpointing,
                    )
                    logits = model.decode(state).float()
                    loss = F.cross_entropy(
                        logits[batch.answer_mask], batch.targets[batch.answer_mask]
                    )
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_([adapter.delta], args.grad_clip)
                )
                optimizer.step()
                row = {
                    "update": update,
                    "cumulative_update": cumulative_update,
                    "band": "id" if cumulative_update % 2 else "seen_ood",
                    "length": length,
                    "target_step": length + spec.step_offset,
                    "loss": float(loss.detach()),
                    "learning_rate": learning_rate,
                    "gradient_norm_before_clip": grad_norm,
                    "delta_frobenius": float(adapter.delta.detach().float().norm()),
                    "elapsed_seconds": time.time() - started_at,
                }
                history.append(row)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

                if update % args.eval_every == 0 or update == args.updates:
                    selection_rows = evaluate_target_lengths(
                        model=model,
                        spec=spec,
                        lengths=selection_id_lengths + selection_seen_lengths,
                        batch_size=args.eval_batch_size,
                        batches=args.selection_eval_batches,
                        seed=args.seed + 20_000,
                        device=device,
                    )
                    means = band_means(
                        selection_rows,
                        id_lengths=selection_id_lengths,
                        seen_ood_lengths=selection_seen_lengths,
                    )
                    candidate = {
                        "update": update,
                        "cumulative_update": cumulative_update,
                        **means,
                        "metrics": selection_rows,
                    }
                    candidates.append(candidate)
                    candidate_deltas[update] = adapter.delta.detach().cpu().clone()
                    atomic_torch_save(
                        {
                            "kind": "addition_internal_weight_adapter_candidate",
                            "site": args.site,
                            "update": update,
                            "cumulative_update": cumulative_update,
                            "delta": candidate_deltas[update],
                            "selection": candidate,
                        },
                        out_dir / "checkpoints" / f"adapter_{update:06d}.pt",
                    )
                    atomic_json_dump(candidates, out_dir / "selection.json")

            selected = select_best_snapshot(candidates, id_gate=args.id_gate)
            adapter.delta.data.copy_(
                candidate_deltas[int(selected["update"])].to(device)
            )
            adapted_metrics = evaluate_target_lengths(
                model=model,
                spec=spec,
                lengths=final_lengths,
                batch_size=args.eval_batch_size,
                batches=args.final_eval_batches,
                seed=args.seed + 30_000,
                device=device,
            )

        raw_metrics = evaluate_target_lengths(
            model=model,
            spec=spec,
            lengths=final_lengths,
            batch_size=args.eval_batch_size,
            batches=args.final_eval_batches,
            seed=args.seed + 30_000,
            device=device,
        )
        adapted_metrics = _annotate_regimes(
            adapted_metrics, id_max=args.id_max, repair_max=args.repair_max
        )
        raw_metrics = _annotate_regimes(
            raw_metrics, id_max=args.id_max, repair_max=args.repair_max
        )
        finished_at = time.time()
        cuda_memory = (
            {
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            if device.type == "cuda"
            else None
        )
        adapter_artifact = {
            "kind": "addition_internal_weight_adapter",
            "site": args.site,
            "seed": args.seed,
            "delta": adapter.delta.detach().cpu(),
            "selected_update": int(selected["update"]),
            "selected_cumulative_update": int(selected["cumulative_update"]),
            "prior_updates": args.prior_updates,
            "backbone_checkpoint": str(checkpoint_path),
            "task": asdict(spec),
            "model": asdict(model.config),
        }
        atomic_torch_save(adapter_artifact, out_dir / "adapter_best.pt")
        write_csv(out_dir / "training.csv", history)
        write_csv(
            out_dir / "selection.csv",
            [
                {
                    "update": candidate["update"],
                    "cumulative_update": candidate["cumulative_update"],
                    "id_mean_em": candidate["id_mean_em"],
                    "seen_ood_mean_em": candidate["seen_ood_mean_em"],
                }
                for candidate in candidates
            ],
        )
        write_csv(
            out_dir / "final_metrics.csv",
            [
                {"condition": condition, **row}
                for condition, rows in (
                    ("raw", raw_metrics),
                    ("adapted", adapted_metrics),
                )
                for row in rows
            ],
        )
        summary = {
            "status": "complete",
            "baseline": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": int(backbone_payload.get("step", -1)),
                "task": asdict(spec),
                "model": asdict(model.config),
            },
            "adapter": {
                "site": args.site,
                "sharing": "one matrix across physical layers and recurrent loops",
                "trainable_parameters": trainable,
                "initialization_std": args.init_std,
                "initialization": adapter_initialization,
                "norms": _adapter_norms(model, adapter),
            },
            "training": {
                "seed": args.seed,
                "id_band": [args.id_min, args.id_max],
                "seen_ood_repair_band": [args.repair_min, args.repair_max],
                "sampling": "strict 50/50 alternating updates",
                "updates": args.updates,
                "prior_updates": args.prior_updates,
                "continuation_updates": args.updates,
                "cumulative_updates": args.prior_updates + args.updates,
                "optimizer_state_restored": False,
                "data_stream_prior_updates_replayed": args.prior_updates,
                "batch_size": args.batch_size,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "peak_learning_rate": args.learning_rate,
                "warmup_updates": args.warmup_updates,
                "stable_updates": args.stable_updates,
                "final_lr_ratio": args.final_lr_ratio,
                "gradient_clip_norm": args.grad_clip,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "precision": "bfloat16_autocast" if args.amp else "fp32",
                "loss": "final-only answer-mask cross entropy at canonical T(m)",
            },
            "selection": {
                "id_gate": args.id_gate,
                "id_lengths": selection_id_lengths,
                "seen_ood_lengths": selection_seen_lengths,
                "selected": selected,
                "candidate_count": len(candidates),
            },
            "raw_metrics": raw_metrics,
            "adapted_metrics": adapted_metrics,
            "raw_band_summary": _band_summary(
                raw_metrics,
                id_min=args.id_min,
                id_max=args.id_max,
                repair_min=args.repair_min,
                repair_max=args.repair_max,
            ),
            "adapted_band_summary": _band_summary(
                adapted_metrics,
                id_min=args.id_min,
                id_max=args.id_max,
                repair_min=args.repair_min,
                repair_max=args.repair_max,
            ),
            "evaluation": {
                "batch_size": args.eval_batch_size,
                "selection_batches_per_length": args.selection_eval_batches,
                "batches_per_length": args.final_eval_batches,
                "same_examples_for_raw_and_adapted": True,
            },
            "cuda_memory": cuda_memory,
            "started_unix": started_at,
            "finished_unix": finished_at,
            "elapsed_seconds": finished_at - started_at,
        }
        atomic_json_dump(summary, out_dir / "summary.json")
        atomic_json_dump(
            {**manifest, "status": "complete", "finished_unix": finished_at},
            out_dir / "manifest.json",
        )
        return summary
    except Exception as error:
        atomic_json_dump(
            {
                **manifest,
                "status": "failed",
                "failed_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
            },
            out_dir / "manifest.json",
        )
        raise


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
