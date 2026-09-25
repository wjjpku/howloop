"""Matched m=64 curriculum for a causal NoPE residual-FFN controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import torch

from experiments.kg_fj_length.controller import ControllerConfig
from experiments.kg_fj_length.data import PermutationWorld
from experiments.kg_fj_length.training import (
    _accuracy,
    _autocast,
    _controller_training_loss,
    _cpu_state_dict,
    _fixed_splits,
    _forbidden_codes,
    _json_write,
    _sample_device_batch_prefix_safe,
    _torch_write,
    extend_executor_max_length,
    load_controller_checkpoint,
    load_executor_checkpoint,
    sha256_file,
    wsd_multiplier,
)


@dataclass(frozen=True)
class CurriculumConfig:
    stages: tuple[int, ...] = (7, 8, 9, 10)
    seed: int = 401
    batch_size: int = 512
    max_steps_per_stage: int = 12_000
    min_steps_per_stage: int = 4_000
    eval_interval: int = 1_000
    learning_rate: float = 5.0e-5
    warmup_steps: int = 500
    stable_steps: int = 9_000
    decay_steps: int = 2_500
    current_length_probability: float = 0.5
    stage_accuracy_threshold: float = 0.98
    retention_accuracy_threshold: float = 0.98
    selection_count: int = 512
    test_count: int = 4_096
    gradient_clip: float = 1.0
    source_commit: str = "unknown"

    def __post_init__(self) -> None:
        if not self.stages or tuple(sorted(set(self.stages))) != self.stages:
            raise ValueError("curriculum stages must be nonempty, unique, and increasing")
        if any(stage < 4 for stage in self.stages):
            raise ValueError("curriculum stages must be at least four")
        if self.batch_size <= 0 or self.max_steps_per_stage <= 0:
            raise ValueError("batch size and stage steps must be positive")
        if not 0 < self.min_steps_per_stage <= self.max_steps_per_stage:
            raise ValueError("min stage steps must lie inside max stage steps")
        if self.eval_interval <= 0 or self.max_steps_per_stage % self.eval_interval:
            raise ValueError("eval interval must divide max stage steps")
        if (
            self.warmup_steps + self.stable_steps + self.decay_steps
            != self.max_steps_per_stage
        ):
            raise ValueError("per-stage WSD phases must sum to max stage steps")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if not 0 <= self.current_length_probability <= 1:
            raise ValueError("current length probability must be in [0,1]")
        for value in (
            self.stage_accuracy_threshold,
            self.retention_accuracy_threshold,
        ):
            if not 0 <= value <= 1:
                raise ValueError("accuracy thresholds must be in [0,1]")
        if self.selection_count <= 0 or self.test_count <= 0:
            raise ValueError("selection and test counts must be positive")


def _choose_active_calls(
    stage: int,
    probability: float,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    if stage < 4:
        raise ValueError("stage must be at least four")
    use_current = bool(
        torch.rand((), generator=generator, device=device) < probability
    )
    if use_current or stage == 4:
        return stage
    return int(torch.randint(4, stage + 1, (), generator=generator, device=device))


def _prepare_output(
    output_dir: Path | str,
    config: CurriculumConfig,
    backbone_sha256: str,
    initial_controller_sha256: str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _json_write(
        output / "manifest.json",
        {
            "schema": "kg-fj-curriculum-manifest-v1",
            "stage": "controller_curriculum",
            "status": "running",
            "config": asdict(config),
            "inputs": {
                "backbone_file_sha256": backbone_sha256,
                "initial_controller_file_sha256": initial_controller_sha256,
            },
        },
    )
    return output


def train_residual_ffn_curriculum(
    output_dir: Path | str,
    backbone_checkpoint_path: Path | str,
    expected_backbone_sha256: str,
    initial_controller_checkpoint_path: Path | str,
    expected_initial_controller_sha256: str,
    config: CurriculumConfig,
    device: torch.device | str,
) -> dict[str, object]:
    target_device = torch.device(device)
    model, backbone_payload = load_executor_checkpoint(
        backbone_checkpoint_path, expected_backbone_sha256, target_device
    )
    controller, controller_payload = load_controller_checkpoint(
        initial_controller_checkpoint_path,
        expected_initial_controller_sha256,
        expected_backbone_sha256,
        target_device,
    )
    controller_config = ControllerConfig(**controller_payload["controller_config"])
    if controller_config.architecture != "mlp":
        raise ValueError("curriculum continuation requires a residual-FFN controller")
    if model.model_config.position_encoding != "none":
        raise ValueError("curriculum continuation requires NoPE")
    if model.model_config.input_injection:
        raise ValueError("curriculum continuation requires input-once execution")
    if int(controller_payload["world_seed"]) != int(backbone_payload["world_seed"]):
        raise ValueError("controller/backbone world seed mismatch")

    output = _prepare_output(
        output_dir,
        config,
        expected_backbone_sha256,
        expected_initial_controller_sha256,
    )
    maximum_stage = max(config.stages)
    if maximum_stage > model.kg_config.max_length:
        extend_executor_max_length(model, maximum_stage)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    controller.train()

    world = PermutationWorld.create(
        model.kg_config, int(backbone_payload["world_seed"])
    )
    registered_lengths = tuple(range(4, maximum_stage + 1))
    selection, test, forbidden = _fixed_splits(
        model.kg_config,
        world,
        registered_lengths,
        config.selection_count,
        config.test_count,
        config.seed + 70_000,
    )
    world_permutations = world.permutations.to(target_device)
    forbidden_tensors = {
        length: _forbidden_codes(
            keys, length, model.kg_config.relation_count, target_device
        )
        for length, keys in forbidden.items()
    }
    generator_device = target_device.type if target_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(config.seed + 1)

    stage_summaries: list[dict[str, object]] = []
    total_updates = 0
    final_state: dict[str, torch.Tensor] | None = None
    final_metrics: dict[str, float] = {}
    for stage in config.stages:
        stage_dir = output / f"stage_m{stage}"
        stage_dir.mkdir()
        optimizer = torch.optim.AdamW(
            controller.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        best_key: tuple[float, ...] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: dict[str, float] = {}
        best_step = 0
        passed = False
        curves: list[dict[str, object]] = []
        last_details: dict[str, object] = {}
        for stage_step in range(1, config.max_steps_per_stage + 1):
            active_calls = _choose_active_calls(
                stage,
                config.current_length_probability,
                generator,
                target_device,
            )
            batch = _sample_device_batch_prefix_safe(
                model.kg_config,
                world_permutations,
                config.batch_size,
                stage,
                generator,
                {
                    length: forbidden_tensors[length]
                    for length in range(4, stage + 1)
                },
            )
            multiplier = wsd_multiplier(
                stage_step,
                config.warmup_steps,
                config.stable_steps,
                config.decay_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * multiplier
            optimizer.zero_grad(set_to_none=True)
            with _autocast(target_device):
                loss, last_details = _controller_training_loss(
                    model,
                    controller,
                    batch,
                    world_permutations,
                    "truncated_unroll",
                    active_calls,
                )
            loss.backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("frozen backbone received a gradient")
            torch.nn.utils.clip_grad_norm_(
                controller.parameters(), config.gradient_clip
            )
            optimizer.step()
            total_updates += 1

            if stage_step % config.eval_interval == 0:
                controller.eval()
                metrics = {
                    str(length): _accuracy(
                        model, selection[length], target_device, controller
                    )
                    for length in range(4, stage + 1)
                }
                minimum = min(metrics.values())
                macro = sum(metrics.values()) / len(metrics)
                key = (
                    minimum,
                    metrics[str(stage)],
                    macro,
                    *(metrics[str(length)] for length in range(4, stage + 1)),
                )
                progress: dict[str, object] = {
                    "event": "curriculum_eval",
                    "stage_length": stage,
                    "stage_step": stage_step,
                    "total_updates": total_updates,
                    "loss": float(loss.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "minimum_accuracy": minimum,
                    "macro_accuracy": macro,
                    "per_length_accuracy": metrics,
                    "last_batch_supervision": last_details,
                }
                curves.append(progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
                if best_key is None or key > best_key:
                    best_key = key
                    best_state = _cpu_state_dict(controller)
                    best_metrics = metrics
                    best_step = stage_step
                passed = (
                    stage_step >= config.min_steps_per_stage
                    and metrics[str(stage)] >= config.stage_accuracy_threshold
                    and minimum >= config.retention_accuracy_threshold
                )
                controller.train()
                if passed:
                    break

        assert best_state is not None
        controller.load_state_dict(best_state)
        controller.train()
        checkpoint = {
            "schema": "kg-fj-controller-checkpoint-v1",
            "controller_config": asdict(controller_config),
            "train_config": {
                "kind": "length_curriculum",
                "curriculum_config": asdict(config),
                "completed_stage": stage,
            },
            "seed": config.seed,
            "world_seed": int(backbone_payload["world_seed"]),
            "backbone_file_sha256": expected_backbone_sha256,
            "initial_controller_file_sha256": expected_initial_controller_sha256,
            "best_step": best_step,
            "best_metrics": best_metrics,
            "controller_state": best_state,
        }
        _torch_write(stage_dir / "best.pt", checkpoint)
        stage_summary: dict[str, object] = {
            "stage_length": stage,
            "passed": passed,
            "best_step": best_step,
            "best_metrics": best_metrics,
            "checkpoint_sha256": sha256_file(stage_dir / "best.pt"),
            "curves": curves,
        }
        _json_write(stage_dir / "summary.json", stage_summary)
        stage_summaries.append(stage_summary)
        final_state = best_state
        final_metrics = best_metrics

    assert final_state is not None
    final_checkpoint = {
        "schema": "kg-fj-controller-checkpoint-v1",
        "controller_config": asdict(controller_config),
        "train_config": {
            "kind": "length_curriculum",
            "curriculum_config": asdict(config),
            "completed_stages": list(config.stages),
        },
        "seed": config.seed,
        "world_seed": int(backbone_payload["world_seed"]),
        "backbone_file_sha256": expected_backbone_sha256,
        "initial_controller_file_sha256": expected_initial_controller_sha256,
        "best_step": stage_summaries[-1]["best_step"],
        "best_metrics": final_metrics,
        "controller_state": final_state,
    }
    _torch_write(output / "best.pt", final_checkpoint)
    controller.load_state_dict(final_state)
    controller.eval()
    test_metrics = {
        str(length): _accuracy(model, test[length], target_device, controller)
        for length in registered_lengths
    }
    summary: dict[str, object] = {
        "schema": "kg-fj-curriculum-summary-v1",
        "source_commit": config.source_commit,
        "architecture": "causal_nope_residual_ffn_j",
        "execution": "F(JF)^(m-1)",
        "trained_lengths": list(range(4, maximum_stage + 1)),
        "curriculum_stages": list(config.stages),
        "initial_controller_file_sha256": expected_initial_controller_sha256,
        "backbone_file_sha256": expected_backbone_sha256,
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "test_accuracy": test_metrics,
        "stages": stage_summaries,
    }
    _json_write(output / "summary.json", summary)
    _json_write(
        output / "manifest.json",
        {
            "schema": "kg-fj-curriculum-manifest-v1",
            "stage": "controller_curriculum",
            "status": "complete",
            "inputs": {
                "backbone_file_sha256": expected_backbone_sha256,
                "initial_controller_file_sha256": expected_initial_controller_sha256,
            },
            "outputs": {
                "best.pt": summary["best_checkpoint_sha256"],
                "summary.json": sha256_file(output / "summary.json"),
            },
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--initial-controller-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-controller-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--stages", default="7,8,9,10")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps-per-stage", type=int, default=12_000)
    parser.add_argument("--min-steps-per-stage", type=int, default=4_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--stable-steps", type=int, default=9_000)
    parser.add_argument("--decay-steps", type=int, default=2_500)
    parser.add_argument("--current-length-probability", type=float, default=0.5)
    parser.add_argument("--stage-accuracy-threshold", type=float, default=0.98)
    parser.add_argument("--retention-accuracy-threshold", type=float, default=0.98)
    parser.add_argument("--selection-count", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=4_096)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = tuple(int(value) for value in args.stages.split(","))
    config = CurriculumConfig(
        stages=stages,
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps_per_stage=args.max_steps_per_stage,
        min_steps_per_stage=args.min_steps_per_stage,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        decay_steps=args.decay_steps,
        current_length_probability=args.current_length_probability,
        stage_accuracy_threshold=args.stage_accuracy_threshold,
        retention_accuracy_threshold=args.retention_accuracy_threshold,
        selection_count=args.selection_count,
        test_count=args.test_count,
        gradient_clip=args.gradient_clip,
        source_commit=args.source_commit,
    )
    train_residual_ffn_curriculum(
        args.output_dir,
        args.backbone_checkpoint,
        args.backbone_sha256,
        args.initial_controller_checkpoint,
        args.initial_controller_sha256,
        config,
        args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
