"""Confirmatory pure-CE D+AB+b controller for final-only N10 D8L8 Graph.

This deliberately does not consume any historical controller or oracle state:
the frozen backbone, controller-training graphs, selection graphs, and locked
G1 test permutations are separate.  It is intentionally a narrow executor
for G3 rather than a compatibility wrapper around exploratory controller code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import checkpoint_loss_mode, fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device, set_seed
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.paper2027_graph_g4_protocol import (
    load_unique_lock,
    merged_forbidden_codes,
    sample_training_permutations,
    sha256 as lock_sha256,
)
from scripts.evaluate_paper2027_graph_g1 import (
    _counts,
    make_or_load_locked_test,
    summarize_clusters,
    write_csv,
)


class DiagonalLowRankGraphController(torch.nn.Module):
    def __init__(self, dimension: int, rank: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.rank = rank
        self.diagonal = torch.nn.Parameter(torch.ones(dimension))
        # One factor is random and the other zero.  This keeps AB exactly zero
        # (so J starts as I), while avoiding the A=B=0 dead-gradient bug.
        self.A = torch.nn.Parameter(torch.empty(dimension, rank))
        torch.nn.init.normal_(self.A, mean=0.0, std=1.0 / math.sqrt(dimension))
        self.B = torch.nn.Parameter(torch.zeros(rank, dimension))
        self.bias = torch.nn.Parameter(torch.zeros(dimension))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state.float() * self.diagonal + (state.float() @ self.A) @ self.B + self.bias


def _verify_config(cfg: GraphPathConfig, checkpoint: Path) -> None:
    expected = {"node_count": 10, "max_depth": 8, "d_model": 256, "n_heads": 4, "d_mlp": 1024, "n_layers": 2, "max_loops": 8, "block_schedule": "all_blocks", "inner_norm_style": "pre_layernorm"}
    mismatch = {key: (value, getattr(cfg, key)) for key, value in expected.items() if getattr(cfg, key) != value}
    if mismatch:
        raise ValueError(f"G3 requires registered final-only N10 D8L8 Graph: {mismatch}")
    if checkpoint_loss_mode(checkpoint) != "final_only":
        raise ValueError("G3 requires final-only backbone supervision")


def _initial(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model.token_embed(tokens) + model.pos_embed.unsqueeze(0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _payload(*, controller: DiagonalLowRankGraphController, checkpoint: Path, update: int, validation: dict[str, float], seed: int, max_train_call: int, protocol_id: str, excluded_locks: dict[str, str] | None) -> dict[str, Any]:
    return {"kind": "paper2027_graph_g3_controller", "protocol_id": protocol_id, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "update": update, "seed": seed, "rank": controller.rank, "dimension": controller.dimension, "placement": "before_each_frozen_executor_call", "objective": "on_policy_successor_CE", "controlled_training_calls": [1, max_train_call], "validation": validation, "excluded_locks": excluded_locks, "controller_state_dict": {key: value.detach().cpu() for key, value in controller.state_dict().items()}}


@torch.no_grad()
def _validate(*, model: torch.nn.Module, cfg: GraphPathConfig, controller: DiagonalLowRankGraphController, device: torch.device, seed: int, batches: int, batch_size: int, max_call: int) -> dict[str, float]:
    if max_call < 8:
        raise ValueError("G3 validation must include the trained call 8")
    # Validation is a locked selection stream.  Restoring the RNG afterwards is
    # essential: otherwise each validation would silently reseed the following
    # training batch and contaminate the training/selection split.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        set_seed(seed)
        loss_sum = 0.0
        correct = {call: 0.0 for call in range(1, max_call + 1)}
        count = 0
        for _ in range(batches):
            tokens, targets, _, _ = fixed_depth_batch(cfg, batch_size, device, path_positions=max_call)
            state = _initial(model, tokens)
            for call in range(1, max_call + 1):
                state = model.apply_loop(controller(state), loop_index=call - 1)
                logits = logits_from_raw_state(model, state).float()
                target = targets[:, call - 1]
                loss_sum += float(F.cross_entropy(logits, target))
                correct[call] += float(logits.argmax(dim=-1).eq(target).sum())
            count += batch_size
        return {"mean_successor_ce": loss_sum / (batches * max_call), "call8_successor_accuracy": correct[8] / count, "call16_successor_accuracy": correct[max_call] / count}
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)


@torch.no_grad()
def _validate_lock(*, model: torch.nn.Module, cfg: GraphPathConfig, controller: DiagonalLowRankGraphController, device: torch.device, lock_path: Path, batch_size: int, max_call: int) -> dict[str, float]:
    """Validate over every start of every graph in the immutable selection lock."""
    lock = load_unique_lock(lock_path)
    if lock["role"] != "selection" or lock["node_count"] != cfg.node_count:
        raise ValueError("controller selection lock does not match the N10 protocol")
    if batch_size % cfg.node_count:
        raise ValueError("validation batch must preserve graph clusters")
    successors_all = lock["successors"]
    if not isinstance(successors_all, torch.Tensor):
        raise ValueError("selection lock lacks successors")
    loss_sum, count = 0.0, 0
    correct = {call: 0.0 for call in range(1, max_call + 1)}
    graphs_per_batch = batch_size // cfg.node_count
    for first in range(0, successors_all.shape[0], graphs_per_batch):
        last = min(successors_all.shape[0], first + graphs_per_batch)
        successors = successors_all[first:last].repeat_interleave(cfg.node_count, dim=0).to(device)
        starts = torch.arange(cfg.node_count, device=device).repeat(last - first)
        tokens, targets, _, _ = fixed_depth_batch(cfg, successors.shape[0], device, path_positions=max_call, successors=successors, start=starts)
        state = _initial(model, tokens)
        for call in range(1, max_call + 1):
            state = model.apply_loop(controller(state), loop_index=call - 1)
            logits = logits_from_raw_state(model, state).float()
            target = targets[:, call - 1]
            loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
            correct[call] += float(logits.argmax(dim=-1).eq(target).sum())
        count += targets.shape[0]
    return {"mean_successor_ce": loss_sum / (count * max_call), "call8_successor_accuracy": correct[8] / count, "call16_successor_accuracy": correct[max_call] / count, "selection_lock_graphs": float(successors_all.shape[0])}


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    _verify_config(cfg, args.checkpoint)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    set_seed(args.seed)  # Seed controller initialization, not just training batches.
    controller = DiagonalLowRankGraphController(cfg.d_model, args.rank).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate, weight_decay=0.0)
    if args.updates < 1 or args.validation_every < 1:
        raise ValueError("updates and validation_every must be positive")
    if args.max_train_call < cfg.max_loops:
        raise ValueError("G3 successor supervision must include the trained call 8")
    if bool(args.train_exclude_locks) != bool(args.validation_lock):
        raise ValueError("use both --train-exclude-locks and --validation-lock for graph-disjoint training")
    excluded_locks: dict[str, str] | None = None
    forbidden_codes: torch.Tensor | None = None
    if args.train_exclude_locks:
        if args.validation_lock not in args.train_exclude_locks:
            raise ValueError("the selection lock must also be excluded from controller training")
        forbidden_codes = merged_forbidden_codes(args.train_exclude_locks, node_count=cfg.node_count)
        selection = load_unique_lock(args.validation_lock)
        if selection["role"] != "selection":
            raise ValueError("--validation-lock must have selection role")
        excluded_locks = {str(path): lock_sha256(path) for path in args.train_exclude_locks}
    set_seed(args.seed)
    best = math.inf
    rows: list[dict[str, Any]] = []
    for update in range(1, args.updates + 1):
        successors = (
            sample_training_permutations(batch_size=args.batch_size, node_count=cfg.node_count, forbidden_codes=forbidden_codes, device=device)
            if forbidden_codes is not None else None
        )
        tokens, targets, _, _ = fixed_depth_batch(cfg, args.batch_size, device, path_positions=args.max_train_call, successors=successors)
        state = _initial(model, tokens)
        losses: list[torch.Tensor] = []
        for call in range(1, args.max_train_call + 1):
            state = model.apply_loop(controller(state), loop_index=call - 1)
            losses.append(F.cross_entropy(logits_from_raw_state(model, state).float(), targets[:, call - 1]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip); optimizer.step()
        if update % args.validation_every == 0 or update == args.updates:
            validation = (
                _validate_lock(model=model, cfg=cfg, controller=controller, device=device, lock_path=args.validation_lock, batch_size=args.validation_batch_size, max_call=args.max_train_call)
                if args.validation_lock is not None else
                _validate(model=model, cfg=cfg, controller=controller, device=device, seed=args.validation_seed, batches=args.validation_batches, batch_size=args.validation_batch_size, max_call=args.max_train_call)
            )
            row = {"update": update, "train_successor_ce": float(loss.detach()), **validation}
            rows.append(row)
            print(json.dumps({"event":"validation", **row}), flush=True)
            if validation["mean_successor_ce"] < best:
                best = validation["mean_successor_ce"]
                _save(args.out_dir / "best_controller.pt", _payload(controller=controller, checkpoint=args.checkpoint, update=update, validation=validation, seed=args.seed, max_train_call=args.max_train_call, protocol_id=args.protocol_id, excluded_locks=excluded_locks))
    _save(args.out_dir / "final_controller.pt", _payload(controller=controller, checkpoint=args.checkpoint, update=args.updates, validation=validation, seed=args.seed, max_train_call=args.max_train_call, protocol_id=args.protocol_id, excluded_locks=excluded_locks))
    _write_csv(args.out_dir / "training.csv", rows)
    result = {"status": "complete", "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_sec": time.time()-started, "peak_memory_gib": torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0, "protocol_id": args.protocol_id, "checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint), "controller_seed": args.seed, "rank": args.rank, "updates": args.updates, "max_train_call": args.max_train_call, "training_seed": args.seed, "validation_seed": args.validation_seed, "validation": validation, "excluded_locks": excluded_locks}
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _load_controller(
    *, path: Path, checkpoint: Path, device: torch.device
) -> tuple[DiagonalLowRankGraphController, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper2027_graph_g3_controller":
        raise ValueError(f"not a registered G3 controller: {path}")
    if payload.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("controller checkpoint hash does not match evaluated backbone")
    dimension, rank = int(payload["dimension"]), int(payload["rank"])
    controller = DiagonalLowRankGraphController(dimension, rank).to(device)
    controller.load_state_dict(payload["controller_state_dict"], strict=True)
    controller.eval()
    return controller, payload


def _controller_transform(
    *,
    controller: DiagonalLowRankGraphController,
    mode: str,
    device: torch.device,
    random_seed: int,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]]:
    """Return a row-vector controller ablation with a recorded construction."""
    d = controller.dimension
    diagonal = controller.diagonal.detach()
    A, B, bias = controller.A.detach(), controller.B.detach(), controller.bias.detach()
    identity = torch.eye(d, device=device, dtype=diagonal.dtype)
    full_matrix = torch.diag(diagonal) + A @ B
    if mode == "raw":
        return (lambda state: state), {"mode": mode, "placement": "no_controller"}
    if mode == "batch_shuffle":
        # The actual graph-instance permutation is constructed inside
        # ``evaluate`` because it must respect the locked all-starts clusters.
        # Keep the learned J itself intact: this control asks whether its
        # *output is paired with the correct graph instance*, not whether its
        # coordinate system happens to be useful.
        return controller, {
            "mode": mode,
            "placement": "D_plus_AB_plus_b_then_cross_graph_batch_shuffle",
        }
    if mode == "executor_off":
        return controller, {"mode": mode, "placement": "controller_then_readout_without_executor"}
    if mode == "full":
        return controller, {"mode": mode, "placement": "D_plus_AB_plus_b"}
    if mode == "D_only":
        return (lambda state: state.float() * diagonal), {"mode": mode, "placement": "D"}
    if mode == "no_AB":
        return (
            lambda state: state.float() * diagonal + bias,
            {"mode": mode, "placement": "D_plus_b"},
        )
    if mode == "identity_D":
        return (
            lambda state: state.float() + (state.float() @ A) @ B + bias,
            {"mode": mode, "placement": "I_plus_AB_plus_b"},
        )
    if mode == "AB_only":
        return (
            lambda state: state.float() + (state.float() @ A) @ B,
            {"mode": mode, "placement": "I_plus_AB"},
        )
    if mode == "no_bias":
        return (
            lambda state: state.float() @ full_matrix,
            {"mode": mode, "placement": "D_plus_AB"},
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    if mode == "mean_D":
        mean_d = diagonal.mean()
        matrix = identity * mean_d + A @ B
        return (
            lambda state: state.float() @ matrix + bias,
            {"mode": mode, "placement": "mean_D_plus_AB_plus_b", "mean_diagonal": float(mean_d)},
        )
    if mode == "shuffle_D":
        shuffled = diagonal[torch.randperm(d, device=device, generator=generator)]
        matrix = torch.diag(shuffled) + A @ B
        return (
            lambda state: state.float() @ matrix + bias,
            {"mode": mode, "placement": "shuffle_D_plus_AB_plus_b", "random_seed": random_seed},
        )
    if mode == "spectrum_matched_random_delta":
        delta = full_matrix - identity
        singular = torch.linalg.svdvals(delta)
        left = torch.linalg.qr(torch.randn(d, d, device=device, generator=generator)).Q
        right = torch.linalg.qr(torch.randn(d, d, device=device, generator=generator)).Q
        random_delta = (left * singular) @ right.T
        matrix = identity + random_delta
        return (
            lambda state: state.float() @ matrix + bias,
            {
                "mode": mode,
                "placement": "I_plus_random_delta_plus_b",
                "random_seed": random_seed,
                "delta_singular_values": [float(value) for value in singular.cpu()],
            },
        )
    raise ValueError(f"unsupported controller mode: {mode}")


def _plot_evaluation(summary: list[dict[str, Any]], path: Path, *, mode: str) -> None:
    calls = [int(row["call"]) for row in summary]
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for key, label in (
        ("endpoint_hold_accuracy", "hold f^8(start)"),
        ("strict_successor_accuracy", "strict f^t(start)"),
        ("strict_other_rate", "strict other"),
    ):
        axis.plot(calls, [float(row[key]) for row in summary], label=label)
    axis.axvline(8, color="black", linestyle="--", linewidth=1, label="trained call")
    axis.set(title=f"G3 controller mode: {mode}", xlabel="recurrent call t", ylabel="rate", ylim=(-0.03, 1.03))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    _verify_config(cfg, args.checkpoint)
    if args.max_call < cfg.max_loops:
        raise ValueError("max_call must include the trained call 8")
    if args.batch_size % cfg.node_count:
        raise ValueError("batch_size must be divisible by node_count")
    controller, controller_payload = _load_controller(
        path=args.controller, checkpoint=args.checkpoint, device=device
    )
    if args.require_final_test_lock_excluded:
        backbone_excluded = checkpoint_payload.get("excluded_locks", {})
        if not isinstance(backbone_excluded, dict) or backbone_excluded.get("final_test_sha256") != lock_sha256(args.locked_test):
            raise ValueError("the evaluated final lock was not excluded from backbone training")
        excluded = controller_payload.get("excluded_locks")
        if not isinstance(excluded, dict) or lock_sha256(args.locked_test) not in set(excluded.values()):
            raise ValueError("the evaluated final lock was not excluded from controller training")
    transform, control_metadata = _controller_transform(
        controller=controller, mode=args.mode, device=device, random_seed=args.random_seed
    )
    locked = make_or_load_locked_test(
        cfg=cfg,
        path=args.locked_test,
        permutations=args.permutations,
        seed=args.test_seed,
    )
    successors_all = locked["successors"]
    rows: list[dict[str, Any]] = []
    cluster_batch = args.batch_size // cfg.node_count
    for first in range(0, args.permutations, cluster_batch):
        last = min(args.permutations, first + cluster_batch)
        successors = successors_all[first:last].repeat_interleave(cfg.node_count, dim=0).to(device)
        starts = torch.arange(cfg.node_count, device=device).repeat(last - first)
        tokens, targets, _, _ = fixed_depth_batch(
            cfg, successors.shape[0], device, path_positions=args.max_call, successors=successors, start=starts
        )
        endpoint = targets[:, cfg.max_depth - 1]
        state = _initial(model, tokens)
        for call in range(1, args.max_call + 1):
            # This is a trajectory-matched executor-off control: it applies
            # the learned J at every boundary but never calls F.  A single
            # readout of J(h_0) would not test whether the controller itself
            # carries the continuation dynamics.
            prepared = transform(state)
            if args.mode == "batch_shuffle":
                # ``fixed_depth_batch`` is ordered as one contiguous block of
                # all start nodes per graph permutation.  Rotating these
                # blocks preserves every state statistic but severs the
                # controller-output/graph pairing before the frozen executor.
                prepared = prepared.reshape(
                    last - first, cfg.node_count, *prepared.shape[1:]
                ).roll(1, dims=0).reshape_as(prepared)
            state = (
                prepared
                if args.mode == "executor_off"
                else model.apply_loop(prepared, loop_index=call - 1)
            )
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            successor = targets[:, call - 1]
            for local_graph in range(last - first):
                begin, end = local_graph * cfg.node_count, (local_graph + 1) * cfg.node_count
                rows.append({"permutation": first + local_graph, "call": call, **_counts(prediction=prediction[begin:end], endpoint=endpoint[begin:end], successor=successor[begin:end])})
    summary = summarize_clusters(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "permutation_clusters.csv", rows)
    write_csv(args.out_dir / "aggregate_calls.csv", summary)
    _plot_evaluation(summary, args.out_dir / "continuation_curve.png", mode=args.mode)
    result = {
        "status": "complete",
        "protocol_id": args.protocol_id,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "controller_training": {key: controller_payload.get(key) for key in ("seed", "rank", "objective", "controlled_training_calls", "validation", "excluded_locks")},
        "mode": args.mode,
        "control_metadata": control_metadata,
        "locked_test": str(args.locked_test),
        "locked_test_sha256": _sha256(args.locked_test),
        "test_seed": args.test_seed,
        "permutations": args.permutations,
        "starts_per_permutation": cfg.node_count,
        "top_level_cluster": "graph permutation",
        "max_call": args.max_call,
        "aggregate_rows": summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--checkpoint", type=Path, required=True); train_parser.add_argument("--out-dir", type=Path, required=True)
    train_parser.add_argument("--rank", type=int, default=48); train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--updates", type=int, default=8_000); train_parser.add_argument("--max-train-call", type=int, default=16)
    train_parser.add_argument("--batch-size", type=int, default=128); train_parser.add_argument("--learning-rate", type=float, default=1e-4); train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--validation-every", type=int, default=400); train_parser.add_argument("--validation-seed", type=int, required=True); train_parser.add_argument("--validation-batches", type=int, default=4); train_parser.add_argument("--validation-batch-size", type=int, default=128); train_parser.add_argument("--train-exclude-locks", type=Path, nargs="+"); train_parser.add_argument("--validation-lock", type=Path); train_parser.add_argument("--protocol-id", default="paper2027.graph.g3.controller.v1"); train_parser.add_argument("--device", default="cuda")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--controller", type=Path, required=True)
    eval_parser.add_argument("--locked-test", type=Path, required=True)
    eval_parser.add_argument("--out-dir", type=Path, required=True)
    eval_parser.add_argument("--mode", choices=("raw", "full", "D_only", "no_AB", "identity_D", "AB_only", "no_bias", "mean_D", "shuffle_D", "batch_shuffle", "spectrum_matched_random_delta", "executor_off"), required=True)
    eval_parser.add_argument("--permutations", type=int, default=512)
    eval_parser.add_argument("--test-seed", type=int, default=2026093001)
    eval_parser.add_argument("--random-seed", type=int, default=2026093002)
    eval_parser.add_argument("--max-call", type=int, default=128)
    eval_parser.add_argument("--batch-size", type=int, default=256)
    eval_parser.add_argument("--require-final-test-lock-excluded", action="store_true"); eval_parser.add_argument("--protocol-id", default="paper2027.graph.g3.locked_evaluation.v1"); eval_parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "train":
        print(json.dumps(train(args), indent=2, sort_keys=True))
    elif args.action == "evaluate":
        print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
