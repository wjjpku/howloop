"""Command-line stages for the variable-length alternating F/J experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .controller import ControllerConfig
from .data import KGLengthConfig, PermutationWorld
from .model import ModelConfig
from .training import (
    TrainConfig,
    evaluate_length_grid,
    extend_executor_max_length,
    load_controller_checkpoint,
    load_executor_checkpoint,
    sha256_file,
    train_backbone,
    train_controller,
    train_oracle,
)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entities", type=int, default=128)
    parser.add_argument("--relations", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument(
        "--position-encoding",
        choices=("sinusoidal", "none"),
        default="sinusoidal",
    )
    parser.add_argument("--input-injection", action="store_true")


def _add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    steps: int,
    lr: float,
    warmup: int,
    stable: int,
    decay: int,
    eval_interval: int,
) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--world-seed", type=int, default=7)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=lr)
    parser.add_argument("--warmup-steps", type=int, default=warmup)
    parser.add_argument("--stable-steps", type=int, default=stable)
    parser.add_argument("--decay-steps", type=int, default=decay)
    parser.add_argument("--eval-interval", type=int, default=eval_interval)
    parser.add_argument("--selection-count", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=4096)
    parser.add_argument("--frontier-weight", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backbone = subparsers.add_parser("backbone")
    _add_model_arguments(backbone)
    _add_training_arguments(
        backbone,
        steps=80_000,
        lr=3.0e-4,
        warmup=2_000,
        stable=60_000,
        decay=18_000,
        eval_interval=2_000,
    )
    backbone.add_argument(
        "--supervision-mode",
        choices=("final_only", "aligned_intermediate"),
        default="final_only",
    )

    oracle = subparsers.add_parser("oracle")
    _add_model_arguments(oracle)
    _add_training_arguments(
        oracle,
        steps=120_000,
        lr=3.0e-4,
        warmup=2_000,
        stable=90_000,
        decay=28_000,
        eval_interval=2_000,
    )

    controller = subparsers.add_parser("controller")
    _add_training_arguments(
        controller,
        steps=50_000,
        lr=1.0e-4,
        warmup=1_000,
        stable=39_000,
        decay=10_000,
        eval_interval=1_000,
    )
    controller.add_argument("--backbone-checkpoint", type=Path, required=True)
    controller.add_argument("--backbone-sha256", required=True)
    controller.add_argument("--hidden-width", type=int, default=1024)
    controller.add_argument("--initial-scale", type=float, default=1.0e-2)
    controller.add_argument(
        "--controller-architecture",
        choices=("affine", "mlp", "gated_rms_mlp", "attention"),
        default="mlp",
    )
    controller.add_argument("--attention-heads", type=int, default=8)
    controller.add_argument(
        "--controller-loss-mode",
        choices=("final_only", "local_successor", "truncated_unroll"),
        default="final_only",
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--seed", type=int, default=20260814)
    evaluate.add_argument("--count", type=int, default=4096)
    evaluate.add_argument("--min-length", type=int, default=1)
    evaluate.add_argument("--max-length", type=int, default=6)
    evaluate.add_argument("--source-commit", default="unknown")
    evaluate.add_argument("--backbone-checkpoint", type=Path, required=True)
    evaluate.add_argument("--backbone-sha256", required=True)
    evaluate.add_argument("--controller-checkpoint", type=Path, required=True)
    evaluate.add_argument("--controller-sha256", required=True)
    evaluate.add_argument("--oracle-checkpoint", type=Path)
    evaluate.add_argument("--oracle-sha256")
    evaluate.add_argument("--adaptive-extra-calls", type=int, default=0)
    return parser


def _kg_config(args: argparse.Namespace) -> KGLengthConfig:
    return KGLengthConfig(entity_count=args.entities, relation_count=args.relations, max_length=6)


def _model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        d_model=args.d_model,
        n_heads=args.heads,
        d_mlp=args.d_mlp,
        physical_blocks=args.blocks,
        dropout=0.0,
        position_encoding=args.position_encoding,
        input_injection=args.input_injection,
    )


def _train_config(args: argparse.Namespace, lengths: tuple[int, ...]) -> TrainConfig:
    return TrainConfig(
        train_lengths=lengths,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        decay_steps=args.decay_steps,
        eval_interval=args.eval_interval,
        selection_count=args.selection_count,
        test_count=args.test_count,
        source_commit=args.source_commit,
        supervision_mode=getattr(args, "supervision_mode", "final_only"),
        controller_loss_mode=getattr(args, "controller_loss_mode", "final_only"),
        frontier_weight=args.frontier_weight,
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("evaluation length range must satisfy 1 <= min <= max")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    backbone, backbone_payload = load_executor_checkpoint(
        args.backbone_checkpoint, args.backbone_sha256, args.device
    )
    if (args.oracle_checkpoint is None) != (args.oracle_sha256 is None):
        raise ValueError("oracle checkpoint and sha256 must be supplied together")
    oracle = oracle_payload = None
    if args.oracle_checkpoint is not None:
        oracle, oracle_payload = load_executor_checkpoint(
            args.oracle_checkpoint, args.oracle_sha256, args.device
        )
    controller, controller_payload = load_controller_checkpoint(
        args.controller_checkpoint,
        args.controller_sha256,
        expected_backbone_sha256=args.backbone_sha256,
        device=args.device,
    )
    if oracle_payload is not None and backbone_payload["kg_config"] != oracle_payload["kg_config"]:
        raise ValueError("backbone/oracle KG configuration mismatch")
    if oracle_payload is not None and backbone_payload["world_seed"] != oracle_payload["world_seed"]:
        raise ValueError("backbone/oracle world seed mismatch")
    if controller_payload["world_seed"] != backbone_payload["world_seed"]:
        raise ValueError("controller/backbone world seed mismatch")
    checkpoint_max_length = backbone.kg_config.max_length
    if oracle is not None and oracle.kg_config.max_length != checkpoint_max_length:
        raise ValueError("backbone/oracle maximum length mismatch")
    if args.max_length > checkpoint_max_length:
        extend_executor_max_length(backbone, args.max_length)
        if oracle is not None:
            extend_executor_max_length(oracle, args.max_length)
    world = PermutationWorld.create(
        backbone.kg_config, seed=int(backbone_payload["world_seed"])
    )
    lengths = tuple(range(args.min_length, args.max_length + 1))
    primary = evaluate_length_grid(
        backbone,
        world,
        controller=controller,
        count=args.count,
        seed=args.seed,
        device=args.device,
        lengths=lengths,
        adaptive_extra_calls=args.adaptive_extra_calls,
    )
    oracle_grid = None
    if oracle is not None:
        oracle_grid = evaluate_length_grid(
            oracle,
            world,
            controller=None,
            count=args.count,
            seed=args.seed,
            device=args.device,
            lengths=lengths,
            adaptive_extra_calls=0,
        )
    for length, row in primary.items():
        row["oracle_accuracy"] = (
            None if oracle_grid is None else oracle_grid[length]["raw_accuracy"]
        )
        learned = row["learned_j_accuracy"]
        row["j_minus_raw"] = None if learned is None else learned - row["raw_accuracy"]
    summary: dict[str, object] = {
        "schema": "kg-fj-evaluation-v1",
        "source_commit": args.source_commit,
        "execution": "F(JF)^(m-1)",
        "evaluation_lengths": list(lengths),
        "position_encoding": {
            "type": backbone.model_config.position_encoding,
            "checkpoint_max_length": checkpoint_max_length,
            "evaluated_max_length": args.max_length,
            "extension": (
                "all_zero_nope"
                if backbone.model_config.position_encoding == "none"
                else "deterministic_sinusoidal_prefix_preserving"
            ),
        },
        "input_injection": backbone.model_config.input_injection,
        "adaptive_extra_calls": args.adaptive_extra_calls,
        "inputs": {
            "backbone_sha256": args.backbone_sha256,
            "controller_sha256": args.controller_sha256,
            "oracle_sha256": args.oracle_sha256,
        },
        "per_length": primary,
    }
    _write_json_atomic(output / "summary.json", summary)
    _write_json_atomic(
        output / "manifest.json",
        {
            "schema": "kg-fj-manifest-v1",
            "stage": "evaluate",
            "status": "complete",
            "outputs": {"summary.json": sha256_file(output / "summary.json")},
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "backbone":
        train_backbone(
            args.output_dir,
            _kg_config(args),
            _model_config(args),
            _train_config(args, (1, 2, 3)),
            args.world_seed,
            args.device,
        )
    elif args.command == "oracle":
        train_oracle(
            args.output_dir,
            _kg_config(args),
            _model_config(args),
            _train_config(args, (1, 2, 3, 4, 5, 6)),
            args.world_seed,
            args.device,
        )
    elif args.command == "controller":
        model, _ = load_executor_checkpoint(
            args.backbone_checkpoint, args.backbone_sha256, device="cpu"
        )
        train_controller(
            args.output_dir,
            args.backbone_checkpoint,
            args.backbone_sha256,
            ControllerConfig(
                d_model=model.model_config.d_model,
                hidden_width=args.hidden_width,
                initial_scale=args.initial_scale,
                architecture=args.controller_architecture,
                attention_heads=args.attention_heads,
            ),
            _train_config(args, (4, 5, 6)),
            args.device,
        )
    elif args.command == "evaluate":
        _evaluate(args)
    else:
        raise AssertionError(f"unhandled command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
