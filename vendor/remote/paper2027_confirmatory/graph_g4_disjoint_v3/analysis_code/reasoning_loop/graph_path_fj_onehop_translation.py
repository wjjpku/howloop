from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.graph_path_twohop_reprogram_j import (
    ActiveAgeBatch,
    DenseAffineJ,
    _permutation_digest,
    _sha256,
    _write_csv,
    build_all_start_age_dataset,
    build_training_dataset,
    strict_unseen_permutations,
    validate_d8l8_onehop_config,
    write_summary_atomic,
)


@dataclass(frozen=True)
class OneHopLoss:
    total: Tensor
    target: Tensor
    mapped: Tensor
    output: Tensor
    logits: Tensor
    calls: int = 1


@dataclass(frozen=True)
class CandidateSpec:
    learning_rate: float

    @property
    def name(self) -> str:
        return f"onehop_lr{self.learning_rate:g}"


@dataclass(frozen=True)
class CandidateResult:
    spec: CandidateSpec
    state_dict: dict[str, Tensor]
    validation: dict[str, float]
    history: list[dict[str, Any]]


def moving_edge_mask(current: Tensor, one: Tensor) -> Tensor:
    if current.ndim != 1 or current.shape != one.shape:
        raise ValueError("current and one must be matching one-dimensional tensors")
    return current.ne(one)


def closure_target_steps(calls: tuple[int, ...]) -> dict[int, int]:
    if not calls or any(call < 1 for call in calls):
        raise ValueError("closure calls must be positive")
    return {call: call for call in calls}


def matched_random_state(state: Tensor, generator: torch.Generator) -> Tensor:
    """Return a state with the input's global mean and second moment."""

    reference = state.float()
    mean = reference.mean()
    centered = reference - mean
    variance = centered.square().mean()
    noise = torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    noise = noise - noise.mean()
    noise = noise * torch.sqrt(variance / noise.square().mean().clamp_min(1e-12))
    return noise + mean


def controlled_call(
    model: nn.Module,
    controller: nn.Module,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    mapped = controller(state)
    output = model.apply_loop(mapped.to(dtype=state.dtype), loop_index=model.cfg.max_loops)
    return mapped, output


def onehop_loss(
    model: nn.Module,
    controller: nn.Module,
    batch: ActiveAgeBatch,
) -> OneHopLoss:
    mapped, output = controlled_call(model, controller, batch.source)
    logits = logits_from_raw_state(model, output)
    return OneHopLoss(
        total=F.cross_entropy(logits, batch.one_node),
        target=batch.one_node,
        mapped=mapped,
        output=output,
        logits=logits,
    )


def _condition_lookup(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {str(row["condition"]): row for row in rows}


def onehop_decision(
    condition_rows: list[dict[str, float]],
    closure_rows: list[dict[str, float]],
) -> dict[str, bool]:
    rows = _condition_lookup(condition_rows)
    endpoint = rows["onehop_after_F"]["moving_one_accuracy"]
    endpoint_accuracy_gate = endpoint >= 0.90
    improves_natural = (
        endpoint - rows["raw_F"]["moving_one_accuracy"] >= 0.05
        and endpoint - rows["identity_J_then_F"]["moving_one_accuracy"] >= 0.05
    )
    controls_causal = (
        endpoint - rows["onehop_shuffle_J_before_F"]["moving_one_accuracy"] >= 0.60
        and endpoint - rows["onehop_random_before_F"]["moving_one_accuracy"] >= 0.60
    )
    no_prewrite = (
        rows["onehop_J_pre"]["moving_current_accuracy"] >= 0.80
        and rows["onehop_J_pre"]["moving_one_accuracy"] <= 0.20
    )
    endpoint_positive = (
        endpoint_accuracy_gate and improves_natural and controls_causal and no_prewrite
    )
    closure = {int(row["calls"]): row["moving_target_accuracy"] for row in closure_rows}
    repeat_positive = endpoint_positive and all(closure[call] >= 0.70 for call in (2, 4, 6))
    return {
        "endpoint_accuracy_gate": endpoint_accuracy_gate,
        "improves_natural_F": improves_natural,
        "controls_causal": controls_causal,
        "no_prewrite": no_prewrite,
        "endpoint_positive": endpoint_positive,
        "repeat_positive": repeat_positive,
    }


def _subset(dataset: ActiveAgeBatch, index: Tensor) -> ActiveAgeBatch:
    return ActiveAgeBatch(
        source=dataset.source[index],
        current_node=dataset.current_node[index],
        one_node=dataset.one_node[index],
        two_node=dataset.two_node[index],
        age=dataset.age[index],
    )


class _MetricTable:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], dict[str, int]] = {}

    def update(self, condition: str, batch: ActiveAgeBatch, prediction: Tensor) -> None:
        moving = moving_edge_mask(batch.current_node, batch.one_node)
        for age in torch.unique(batch.age).tolist():
            mask = batch.age.eq(int(age))
            moving_mask = mask & moving
            row = self.rows.setdefault(
                (condition, int(age)),
                {
                    "examples": 0,
                    "moving_examples": 0,
                    "current_correct": 0,
                    "one_correct": 0,
                    "two_correct": 0,
                    "moving_current_correct": 0,
                    "moving_one_correct": 0,
                },
            )
            row["examples"] += int(mask.sum())
            row["moving_examples"] += int(moving_mask.sum())
            row["current_correct"] += int(prediction[mask].eq(batch.current_node[mask]).sum())
            row["one_correct"] += int(prediction[mask].eq(batch.one_node[mask]).sum())
            row["two_correct"] += int(prediction[mask].eq(batch.two_node[mask]).sum())
            row["moving_current_correct"] += int(
                prediction[moving_mask].eq(batch.current_node[moving_mask]).sum()
            )
            row["moving_one_correct"] += int(
                prediction[moving_mask].eq(batch.one_node[moving_mask]).sum()
            )

    def finalized(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for (condition, age), values in sorted(self.rows.items()):
            total = max(values["examples"], 1)
            moving = max(values["moving_examples"], 1)
            result.append(
                {
                    "condition": condition,
                    "age": age,
                    "examples": values["examples"],
                    "moving_examples": values["moving_examples"],
                    "current_accuracy": values["current_correct"] / total,
                    "one_accuracy": values["one_correct"] / total,
                    "two_accuracy": values["two_correct"] / total,
                    "moving_current_accuracy": values["moving_current_correct"] / moving,
                    "moving_one_accuracy": values["moving_one_correct"] / moving,
                }
            )
        return result


def aggregate_condition_rows(per_age_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = {}
    for row in per_age_rows:
        condition = str(row["condition"])
        values = totals.setdefault(
            condition,
            {
                "examples": 0.0,
                "moving_examples": 0.0,
                "current_accuracy": 0.0,
                "one_accuracy": 0.0,
                "two_accuracy": 0.0,
                "moving_current_accuracy": 0.0,
                "moving_one_accuracy": 0.0,
            },
        )
        examples = float(row["examples"])
        moving = float(row["moving_examples"])
        values["examples"] += examples
        values["moving_examples"] += moving
        for key in ("current_accuracy", "one_accuracy", "two_accuracy"):
            values[key] += examples * float(row[key])
        for key in ("moving_current_accuracy", "moving_one_accuracy"):
            values[key] += moving * float(row[key])
    result: list[dict[str, Any]] = []
    for condition, values in sorted(totals.items()):
        examples = max(values["examples"], 1.0)
        moving = max(values["moving_examples"], 1.0)
        result.append(
            {
                "condition": condition,
                "examples": int(values["examples"]),
                "moving_examples": int(values["moving_examples"]),
                "current_accuracy": values["current_accuracy"] / examples,
                "one_accuracy": values["one_accuracy"] / examples,
                "two_accuracy": values["two_accuracy"] / examples,
                "moving_current_accuracy": values["moving_current_accuracy"] / moving,
                "moving_one_accuracy": values["moving_one_accuracy"] / moving,
            }
        )
    return result


@torch.no_grad()
def evaluate_primary(
    *, model: nn.Module, controller: nn.Module, dataset: ActiveAgeBatch, batch_size: int
) -> dict[str, float]:
    total = 0
    moving_total = 0
    current_correct = 0
    one_correct = 0
    moving_one_correct = 0
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(
            offset,
            min(offset + batch_size, dataset.source.shape[0]),
            device=dataset.source.device,
        )
        batch = _subset(dataset, index)
        _, output = controlled_call(model, controller, batch.source)
        prediction = logits_from_raw_state(model, output).argmax(dim=-1)
        moving = moving_edge_mask(batch.current_node, batch.one_node)
        total += batch.source.shape[0]
        moving_total += int(moving.sum())
        current_correct += int(prediction.eq(batch.current_node).sum())
        one_correct += int(prediction.eq(batch.one_node).sum())
        moving_one_correct += int(prediction[moving].eq(batch.one_node[moving]).sum())
    return {
        "examples": float(total),
        "moving_examples": float(moving_total),
        "current_accuracy": current_correct / max(total, 1),
        "one_accuracy": one_correct / max(total, 1),
        "moving_target_accuracy": moving_one_correct / max(moving_total, 1),
    }


def train_candidate(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    train_dataset: ActiveAgeBatch,
    validation_dataset: ActiveAgeBatch,
    spec: CandidateSpec,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_every: int,
    controller_seed: int,
) -> CandidateResult:
    if steps < 1 or batch_size < 1 or eval_every < 1:
        raise ValueError("steps, batch_size, and eval_every must be positive")
    controller = DenseAffineJ(cfg.d_model).to(train_dataset.source.device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=spec.learning_rate)
    generator = torch.Generator(device=train_dataset.source.device)
    generator.manual_seed(controller_seed)
    best_score = -math.inf
    best_state: dict[str, Tensor] | None = None
    best_validation: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        index = torch.randint(
            0,
            train_dataset.source.shape[0],
            (batch_size,),
            device=train_dataset.source.device,
            generator=generator,
        )
        batch = _subset(train_dataset, index)
        loss = onehop_loss(model, controller, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen backbone unexpectedly received gradients")
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            validation = evaluate_primary(
                model=model,
                controller=controller,
                dataset=validation_dataset,
                batch_size=eval_batch_size,
            )
            row = {
                "candidate": spec.name,
                "learning_rate": spec.learning_rate,
                "step": step,
                "loss": float(loss.total.detach()),
                "gradient_norm": float(gradient_norm),
                **validation,
            }
            history.append(row)
            print(json.dumps({"event": "training", **row}), flush=True)
            if validation["moving_target_accuracy"] > best_score:
                best_score = validation["moving_target_accuracy"]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in controller.state_dict().items()
                }
                best_validation = dict(validation)
    if best_state is None or best_validation is None:
        raise RuntimeError("candidate training produced no checkpoint")
    return CandidateResult(spec, best_state, best_validation, history)


def select_best_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(candidates, key=lambda candidate: candidate.validation["moving_target_accuracy"])


def controller_from_result(
    result: CandidateResult, *, dimension: int, device: torch.device
) -> DenseAffineJ:
    controller = DenseAffineJ(dimension).to(device)
    controller.load_state_dict({name: value.to(device) for name, value in result.state_dict.items()})
    controller.eval()
    return controller


@torch.no_grad()
def evaluate_conditions(
    *,
    model: nn.Module,
    controller: nn.Module,
    dataset: ActiveAgeBatch,
    batch_size: int,
    shuffle_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = _MetricTable()
    generator = torch.Generator(device=dataset.source.device)
    generator.manual_seed(shuffle_seed)
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(
            offset,
            min(offset + batch_size, dataset.source.shape[0]),
            device=dataset.source.device,
        )
        batch = _subset(dataset, index)
        raw = model.apply_loop(batch.source, loop_index=model.cfg.max_loops)
        mapped, output = controlled_call(model, controller, batch.source)
        shuffle = torch.randperm(
            batch.source.shape[0], generator=generator, device=batch.source.device
        )
        shuffled_output = model.apply_loop(
            mapped[shuffle].to(dtype=batch.source.dtype), loop_index=model.cfg.max_loops
        )
        random_output = model.apply_loop(
            matched_random_state(mapped, generator).to(dtype=batch.source.dtype),
            loop_index=model.cfg.max_loops,
        )
        for condition, state in (
            ("source", batch.source),
            ("raw_F", raw),
            ("identity_J_then_F", raw),
            ("onehop_J_pre", mapped),
            ("onehop_after_F", output),
            ("onehop_shuffle_J_before_F", shuffled_output),
            ("onehop_random_before_F", random_output),
        ):
            table.update(condition, batch, logits_from_raw_state(model, state).argmax(dim=-1))
    per_age = table.finalized()
    return per_age, aggregate_condition_rows(per_age)


@torch.no_grad()
def evaluate_closure(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    controller: nn.Module,
    permutations: list[tuple[int, ...]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    requested = (1, 2, 4, 6)
    target_steps = closure_target_steps(requested)
    counts = {
        call: {"examples": 0, "moving_examples": 0, "correct": 0, "moving_correct": 0}
        for call in requested
    }
    examples = [(permutation, start) for permutation in permutations for start in range(cfg.node_count)]
    for offset in range(0, len(examples), batch_size):
        chunk = examples[offset : offset + batch_size]
        successors = torch.tensor([row[0] for row in chunk], dtype=torch.long, device=device)
        starts = torch.tensor([row[1] for row in chunk], dtype=torch.long, device=device)
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            len(chunk),
            device,
            path_positions=max(requested) + 1,
            successors=successors,
            start=starts,
        )
        state = cache_states_with_initial(model, tokens, loops=1)[1]
        current = path_targets[:, 0]
        for call in range(1, max(requested) + 1):
            _, state = controlled_call(model, controller, state)
            if call not in counts:
                continue
            target = path_targets[:, target_steps[call]]
            moving = current.ne(target)
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            values = counts[call]
            values["examples"] += len(chunk)
            values["moving_examples"] += int(moving.sum())
            values["correct"] += int(prediction.eq(target).sum())
            values["moving_correct"] += int(prediction[moving].eq(target[moving]).sum())
    return [
        {
            "calls": call,
            "target_step": target_steps[call],
            "examples": values["examples"],
            "moving_examples": values["moving_examples"],
            "target_accuracy": values["correct"] / max(values["examples"], 1),
            "moving_target_accuracy": values["moving_correct"] / max(values["moving_examples"], 1),
        }
        for call, values in sorted(counts.items())
    ]


def _candidate_payload(result: CandidateResult) -> dict[str, Any]:
    return {
        "spec": asdict(result.spec),
        "validation": result.validation,
        "state_dict": result.state_dict,
    }


def _report_lines(summary: dict[str, Any]) -> list[str]:
    rows = {row["condition"]: row for row in summary["condition_summary"]}
    decision = summary["decision"]
    return [
        "# D8L8 seed3：FJ 一步接口翻译实验",
        "",
        f"- controller-data seed: {summary['controller_seed']}; checkpoint step: {summary['checkpoint_step']};",
        "- loss: CE(readout(J_one→F), one-hop)，J_one 与此前 pair/ID 控制器独立；",
        "",
        "## 严格正式集",
        "",
        f"- raw F moving one-hop: {rows['raw_F']['moving_one_accuracy']:.4f};",
        f"- J_one→F moving one-hop: {rows['onehop_after_F']['moving_one_accuracy']:.4f};",
        f"- J_one pre-F moving current / one-hop: {rows['onehop_J_pre']['moving_current_accuracy']:.4f} / {rows['onehop_J_pre']['moving_one_accuracy']:.4f};",
        f"- shuffled / matched-random J→F moving one-hop: {rows['onehop_shuffle_J_before_F']['moving_one_accuracy']:.4f} / {rows['onehop_random_before_F']['moving_one_accuracy']:.4f};",
        "",
        "## 判定",
        "",
        f"- endpoint translation gate: {decision['endpoint_positive']}; repeat gate: {decision['repeat_positive']};",
        "- endpoint success supports a frozen-F interface translation, not by itself a unique original-circuit claim.",
    ]


def run_experiment(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    controller_seed: int,
    train_examples: int,
    val_permutations: int,
    eval_permutations: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    collection_batch_size: int,
    eval_every: int,
    learning_rates: tuple[float, ...],
    eval_seed: int,
) -> dict[str, Any]:
    if not learning_rates:
        raise ValueError("at least one learning rate is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "initializing",
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "controller_seed": controller_seed,
    }
    write_summary_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("FJ_ONEHOP_CUDA_MEMORY_FRACTION", "0.30")),
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    if int(checkpoint_payload.get("step", -1)) != 20_000:
        raise ValueError("expected D8L8 seed3 final checkpoint at step 20000")
    model.requires_grad_(False)
    set_seed(controller_seed)
    formal = strict_unseen_permutations(cfg.node_count, set(), count=eval_permutations, seed=eval_seed)
    validation = strict_unseen_permutations(
        cfg.node_count, set(formal), count=val_permutations, seed=eval_seed + 1
    )
    train_dataset, train_seen = build_training_dataset(
        model=model,
        cfg=cfg,
        excluded_permutations=set(formal) | set(validation),
        examples=train_examples,
        seed=controller_seed + 43_007,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    validation_dataset = build_all_start_age_dataset(
        model=model,
        cfg=cfg,
        permutations=validation,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    if train_seen & (set(formal) | set(validation)):
        raise RuntimeError("reserved permutation leaked into training")
    print(json.dumps({"event": "dataset_ready", "controller_seed": controller_seed, "train_examples": train_dataset.source.shape[0], "validation_examples": validation_dataset.source.shape[0], "unique_train_permutations": len(train_seen)}), flush=True)
    candidates: list[CandidateResult] = []
    history: list[dict[str, Any]] = []
    for candidate_index, learning_rate in enumerate(learning_rates):
        spec = CandidateSpec(learning_rate)
        print(json.dumps({"event": "candidate_start", "candidate_index": candidate_index, "learning_rate": learning_rate}), flush=True)
        result = train_candidate(
            model=model,
            cfg=cfg,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            spec=spec,
            steps=steps,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            controller_seed=controller_seed * 1009 + candidate_index + 29,
        )
        candidates.append(result)
        history.extend(result.history)
    selected = select_best_candidate(candidates)
    controller = controller_from_result(selected, dimension=cfg.d_model, device=device)
    torch.save(
        {
            "selected": selected.spec.name,
            "candidates": {candidate.spec.name: _candidate_payload(candidate) for candidate in candidates},
        },
        out_dir / "controller.pt",
    )
    strict_dataset = build_all_start_age_dataset(
        model=model,
        cfg=cfg,
        permutations=formal,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    per_age, condition_summary = evaluate_conditions(
        model=model,
        controller=controller,
        dataset=strict_dataset,
        batch_size=eval_batch_size,
        shuffle_seed=eval_seed + controller_seed,
    )
    closure = evaluate_closure(
        model=model,
        cfg=cfg,
        controller=controller,
        permutations=formal,
        device=device,
        batch_size=eval_batch_size,
    )
    decision = onehop_decision(condition_summary, closure)
    summary: dict[str, Any] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "source_code_sha256": _sha256(Path(__file__)),
        "loss_placement": "final_CE_after_J_one_F_to_one_hop_only",
        "runtime_order": "J_one_then_F",
        "config": asdict(cfg),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "natural_semantic_step_per_loop": 1,
        "controller": "separate tokenwise full-rank affine J_one shared over 29 positions",
        "controller_parameters": cfg.d_model * cfg.d_model + cfg.d_model,
        "controller_seed": controller_seed,
        "train_examples": train_examples,
        "unique_train_permutations": len(train_seen),
        "validation_permutations": val_permutations,
        "strict_eval_permutations": eval_permutations,
        "strict_eval_examples": strict_dataset.source.shape[0],
        "strict_eval_permutation_sha256": _permutation_digest(formal),
        "source_ages": list(range(1, 7)),
        "learning_rates": list(learning_rates),
        "steps_per_candidate": steps,
        "batch_size": batch_size,
        "selected_candidate": selected.spec.name,
        "candidate_summary": [
            {"name": candidate.spec.name, "learning_rate": candidate.spec.learning_rate, **candidate.validation}
            for candidate in candidates
        ],
        "per_age_rows": per_age,
        "condition_summary": condition_summary,
        "closure_rows": closure,
        "decision": decision,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0
        ),
    }
    _write_csv(out_dir / "candidate_rows.csv", summary["candidate_summary"])
    _write_csv(out_dir / "training_rows.csv", history)
    _write_csv(out_dir / "per_age_rows.csv", per_age)
    _write_csv(out_dir / "condition_summary.csv", condition_summary)
    _write_csv(out_dir / "closure_rows.csv", closure)
    write_summary_atomic(out_dir / "summary.json", summary)
    (out_dir / "REPORT_CN.md").write_text("\n".join(_report_lines(summary)) + "\n", encoding="utf-8")
    manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "summary": str(out_dir / "summary.json")})
    write_summary_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "selected": selected.spec.name, "decision": decision}), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=12_288)
    parser.add_argument("--val-permutations", type=int, default=128)
    parser.add_argument("--eval-permutations", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--collection-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-5, 3e-5, 1e-4))
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device_name=args.device,
        controller_seed=args.controller_seed,
        train_examples=args.train_examples,
        val_permutations=args.val_permutations,
        eval_permutations=args.eval_permutations,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        collection_batch_size=args.collection_batch_size,
        eval_every=args.eval_every,
        learning_rates=tuple(args.learning_rates),
        eval_seed=args.eval_seed,
    )


if __name__ == "__main__":
    main()
