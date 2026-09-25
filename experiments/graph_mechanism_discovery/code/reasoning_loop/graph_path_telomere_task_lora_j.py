from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import (
    GOOD_TASK_STAGES,
    TaskMLPStage,
    _train_stage,
    evaluate_random_unit_every,
)
from reasoning_loop.graph_path_telomere_unit_j import load_unit_j_map


CANONICAL_TASK_STAGES = GOOD_TASK_STAGES + (
    TaskMLPStage(
        name="task_h96",
        rounds=16,
        batch_size=32,
        batches_per_round=9,
        horizons=(48, 64, 96),
        learning_rate=3e-6,
        data_seed=180003,
    ),
)


# A deliberately closed training distribution for measuring length
# generalization.  Every unrolled example contains at most eight repeated
# J->F compositions.  This is kept separate from CANONICAL_TASK_STAGES so
# reproducing the historical h24/h32/h48/h64 runs is unchanged.
STRICT_H8_TASK_STAGES = (
    TaskMLPStage(
        name="strict_h1",
        rounds=16,
        batch_size=64,
        batches_per_round=8,
        horizons=(1,),
        learning_rate=1e-5,
        data_seed=181003,
    ),
    TaskMLPStage(
        name="strict_h2",
        rounds=16,
        batch_size=64,
        batches_per_round=8,
        horizons=(1, 2),
        learning_rate=1e-5,
        data_seed=182003,
    ),
    TaskMLPStage(
        name="strict_h4",
        rounds=32,
        batch_size=64,
        batches_per_round=8,
        horizons=(1, 2, 4),
        learning_rate=6e-6,
        data_seed=184003,
    ),
    TaskMLPStage(
        name="strict_h8",
        rounds=64,
        batch_size=64,
        batches_per_round=8,
        horizons=(1, 2, 4, 6, 8),
        learning_rate=3e-6,
        data_seed=188003,
    ),
)


def _selected_task_stages(args: argparse.Namespace) -> tuple[TaskMLPStage, ...]:
    if args.curriculum == "strict_h8":
        if args.max_training_horizon != 8:
            raise ValueError("strict_h8 curriculum requires max_training_horizon=8")
        source = STRICT_H8_TASK_STAGES
    else:
        if args.max_training_horizon < 24:
            raise ValueError("canonical curriculum starts at max_training_horizon=24")
        source = CANONICAL_TASK_STAGES
    stages = tuple(
        TaskMLPStage(
            name=stage.name,
            rounds=(
                min(stage.rounds, args.stage_round_limit)
                if args.stage_round_limit is not None
                else stage.rounds
            ),
            batch_size=stage.batch_size,
            batches_per_round=stage.batches_per_round,
            horizons=stage.horizons,
            learning_rate=stage.learning_rate * args.learning_rate_multiplier,
            data_seed=stage.data_seed,
        )
        for stage in source
        if max(stage.horizons) <= args.max_training_horizon
    )
    if not stages:
        raise ValueError("selected curriculum has no stages")
    if any(max(stage.horizons) > args.max_training_horizon for stage in stages):
        raise RuntimeError("training stage exceeds the declared horizon")
    return stages


def _validate_affine_placement(
    artifact: Path,
    *,
    expected: str,
    require_label: bool,
) -> None:
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    recorded = payload.get("placement")
    if recorded is None:
        if require_label:
            raise ValueError(f"affine artifact does not record placement: {artifact}")
        return
    if recorded != expected:
        raise ValueError(
            f"affine artifact placement {recorded!r} does not match "
            f"controller placement {expected!r}: {artifact}"
        )


class ResidualLoRAJ(torch.nn.Module):
    """A position-shared low-rank recurrent rejuvenator.

    J(h) = h + scale * (h A) B + bias.

    ``B`` and ``bias`` start at zero, so every rank starts at the exact same
    identity function.  The only learned capacity is the rank-constrained
    residual update and one shared bias vector; there is no full affine path.
    """

    def __init__(
        self,
        *,
        dimension: int,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not 1 <= rank <= dimension:
            raise ValueError("rank must lie in [1, dimension]")
        self.dimension = dimension
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scale = self.alpha / rank
        self.A = torch.nn.Parameter(torch.empty(dimension, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, dimension))
        self.bias = torch.nn.Parameter(torch.zeros(dimension))
        torch.nn.init.normal_(self.A, mean=0.0, std=1.0 / math.sqrt(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        update = (live @ self.A) @ self.B
        return live + self.scale * update + self.bias

    def frozen(self) -> ResidualLoRAJ:
        result = copy.deepcopy(self).eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def initialize_from_affine_svd(
        self,
        affine: VectorAffine,
        *,
        gauge_seed: int,
    ) -> float:
        """Initialize from the best rank-r approximation to affine-I.

        A random orthogonal gauge changes the factor coordinates without
        changing A@B, giving optimizer-seed replication with an identical
        initial function.
        """

        identity = torch.eye(
            self.dimension,
            device=affine.weight.device,
            dtype=torch.float32,
        )
        delta = affine.weight.float() - identity
        left, singular_values, right_t = torch.linalg.svd(
            delta,
            full_matrices=False,
        )
        selected = singular_values[: self.rank]
        root = selected.sqrt()
        base_A = left[:, : self.rank] * root.unsqueeze(0)
        base_B = root.unsqueeze(1) * right_t[: self.rank]
        generator = torch.Generator(device=delta.device)
        generator.manual_seed(gauge_seed)
        random = torch.randn(
            self.rank,
            self.rank,
            generator=generator,
            device=delta.device,
            dtype=torch.float32,
        )
        gauge, _ = torch.linalg.qr(random)
        self.A.copy_(base_A @ gauge)
        self.B.copy_(gauge.transpose(0, 1) @ base_B)
        self.bias.copy_(affine.bias.float())
        return float(
            selected.square().sum()
            / singular_values.square().sum().clamp_min(1e-12)
        )


class ScaledIdentityLoRAJ(ResidualLoRAJ):
    """A learnable scalar identity path plus a low-rank correction.

    J(h) = identity_scale * h + (h A) B + bias.

    The scalar path can capture a global contraction/expansion while the
    task-specific deviation from that scalar map is rank constrained.
    """

    def __init__(
        self,
        *,
        dimension: int,
        rank: int,
        identity_scale: float = 1.0,
    ) -> None:
        # alpha=rank makes the inherited LoRA scale exactly one.
        super().__init__(dimension=dimension, rank=rank, alpha=float(rank))
        self.identity_scale = torch.nn.Parameter(
            torch.tensor(float(identity_scale), dtype=torch.float32)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        update = (live @ self.A) @ self.B
        return self.identity_scale * live + update + self.bias

    @torch.no_grad()
    def initialize_from_affine_svd(
        self,
        affine: VectorAffine,
        *,
        gauge_seed: int,
        identity_scale_init: str = "trace",
    ) -> float:
        if identity_scale_init == "trace":
            scalar = torch.trace(affine.weight.float()) / self.dimension
        elif identity_scale_init == "one":
            scalar = torch.ones(
                (), device=affine.weight.device, dtype=torch.float32
            )
        else:
            raise ValueError(
                f"unknown identity_scale_init: {identity_scale_init}"
            )
        self.identity_scale.copy_(scalar)
        identity = torch.eye(
            self.dimension,
            device=affine.weight.device,
            dtype=torch.float32,
        )
        delta = affine.weight.float() - scalar * identity
        left, singular_values, right_t = torch.linalg.svd(
            delta,
            full_matrices=False,
        )
        selected = singular_values[: self.rank]
        root = selected.sqrt()
        base_A = left[:, : self.rank] * root.unsqueeze(0)
        base_B = root.unsqueeze(1) * right_t[: self.rank]
        generator = torch.Generator(device=delta.device)
        generator.manual_seed(gauge_seed)
        random = torch.randn(
            self.rank,
            self.rank,
            generator=generator,
            device=delta.device,
            dtype=torch.float32,
        )
        gauge, _ = torch.linalg.qr(random)
        self.A.copy_(base_A @ gauge)
        self.B.copy_(gauge.transpose(0, 1) @ base_B)
        self.bias.copy_(affine.bias.float())
        return float(
            selected.square().sum()
            / singular_values.square().sum().clamp_min(1e-12)
        )


class DiagonalIdentityLoRAJ(ResidualLoRAJ):
    """A learnable diagonal path plus a low-rank correction.

    J(h) = h * diagonal_scale + (h A) B + bias.
    """

    def __init__(self, *, dimension: int, rank: int) -> None:
        super().__init__(dimension=dimension, rank=rank, alpha=float(rank))
        self.diagonal_scale = torch.nn.Parameter(torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        live = value.float()
        update = (live @ self.A) @ self.B
        return live * self.diagonal_scale + update + self.bias

    @torch.no_grad()
    def initialize_from_affine_svd(
        self,
        affine: VectorAffine,
        *,
        gauge_seed: int,
        diagonal_scale_init: str = "diagonal",
    ) -> float:
        if diagonal_scale_init == "diagonal":
            diagonal = torch.diagonal(affine.weight.float())
        elif diagonal_scale_init == "one":
            diagonal = torch.ones(
                self.dimension,
                device=affine.weight.device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"unknown diagonal_scale_init: {diagonal_scale_init}"
            )
        self.diagonal_scale.copy_(diagonal)
        delta = affine.weight.float() - torch.diag(diagonal)
        left, singular_values, right_t = torch.linalg.svd(
            delta,
            full_matrices=False,
        )
        selected = singular_values[: self.rank]
        root = selected.sqrt()
        base_A = left[:, : self.rank] * root.unsqueeze(0)
        base_B = root.unsqueeze(1) * right_t[: self.rank]
        generator = torch.Generator(device=delta.device)
        generator.manual_seed(gauge_seed)
        random = torch.randn(
            self.rank,
            self.rank,
            generator=generator,
            device=delta.device,
            dtype=torch.float32,
        )
        gauge, _ = torch.linalg.qr(random)
        self.A.copy_(base_A @ gauge)
        self.B.copy_(gauge.transpose(0, 1) @ base_B)
        self.bias.copy_(affine.bias.float())
        return float(
            selected.square().sum()
            / singular_values.square().sum().clamp_min(1e-12)
        )


def _scale_statistics(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


class IdentityJ(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _curve_summary(values: list[float]) -> dict[str, Any]:
    def segment(start: int, stop: int) -> float | None:
        selected = values[start:stop]
        return sum(selected) / len(selected) if selected else None

    return {
        "accuracy_by_cycle": values,
        "auc_1_24": segment(0, 24),
        "auc_25_48": segment(24, 48),
        "auc_49_64": segment(48, 64),
    }


def load_task_lora_modules(
    artifact: Path,
    *,
    device: torch.device,
) -> tuple[str, tuple[int, ...], dict[str, torch.nn.Module], dict[str, Any]]:
    payload = torch.load(artifact, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_task_lora_j":
        raise ValueError("unexpected task-aware LoRA-J artifact kind")
    modules: dict[str, torch.nn.Module] = {}
    for label, item in payload["modules"].items():
        if item.get("parameterization") == "scalar_low_rank":
            module = ScaledIdentityLoRAJ(
                dimension=int(item["dimension"]),
                rank=int(item["rank"]),
                identity_scale=float(item["initial_identity_scale"]),
            ).to(device)
        elif item.get("parameterization") == "diagonal_low_rank":
            module = DiagonalIdentityLoRAJ(
                dimension=int(item["dimension"]),
                rank=int(item["rank"]),
            ).to(device)
        else:
            module = ResidualLoRAJ(
                dimension=int(item["dimension"]),
                rank=int(item["rank"]),
                alpha=float(item["alpha"]),
            ).to(device)
        module.load_state_dict(
            {key: value.to(device) for key, value in item["state_dict"].items()}
        )
        modules[label] = module.frozen()
    return (
        str(payload["checkpoint"]),
        tuple(int(value) for value in payload["positions"]),
        modules,
        payload,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage_round_limit is not None and args.stage_round_limit <= 0:
        raise ValueError("stage_round_limit must be positive")
    if args.learning_rate_multiplier <= 0:
        raise ValueError("learning_rate_multiplier must be positive")
    if args.scale_learning_rate_multiplier <= 0:
        raise ValueError("scale_learning_rate_multiplier must be positive")
    if args.state_loss_weight < 0:
        raise ValueError("state_loss_weight must be non-negative")
    if len(set(args.ranks)) != len(args.ranks):
        raise ValueError("ranks must be unique")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if (
        cfg.node_count != 8
        or cfg.max_depth != 8
        or cfg.max_loops != 8
        or cfg.n_layers != 2
        or cfg.d_model != 256
    ):
        raise ValueError("experiment requires the frozen D8L8 N8 d256 model")
    if any(rank < 1 or rank > cfg.d_model for rank in args.ranks):
        raise ValueError("each rank must lie in [1, d_model]")

    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    task_step_size = phase_positions[3] - phase_positions[2]
    if task_step_size <= 0:
        raise ValueError("phase summary must define a positive recurrent step")
    positions = intervention_groups(cfg.node_count)["all"]
    _validate_affine_placement(
        args.reference_affine_artifact,
        expected=args.placement,
        require_label=args.require_affine_placement,
    )
    reference_affine, reference_checkpoint = load_unit_j_map(
        args.reference_affine_artifact,
        label=args.reference_affine_label,
        device=device,
    )
    if reference_checkpoint != str(args.checkpoint):
        raise ValueError("reference affine J belongs to another checkpoint")
    initial_affine: VectorAffine | None = None
    if args.initialization == "affine_svd":
        if args.initial_affine_artifact is None:
            raise ValueError("affine_svd initialization requires an artifact")
        _validate_affine_placement(
            args.initial_affine_artifact,
            expected=args.placement,
            require_label=args.require_affine_placement,
        )
        initial_affine, initial_checkpoint = load_unit_j_map(
            args.initial_affine_artifact,
            label=args.initial_affine_label,
            device=device,
        )
        if initial_checkpoint != str(args.checkpoint):
            raise ValueError("initial affine J belongs to another checkpoint")

    stages = _selected_task_stages(args)
    backbone_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modules: dict[str, torch.nn.Module] = {}
    module_payloads: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    artifact = out_dir / "task_lora_j.pt"
    if args.resume and artifact.exists():
        (
            resume_checkpoint,
            resume_positions,
            modules,
            resume_payload,
        ) = load_task_lora_modules(artifact, device=device)
        if resume_checkpoint != str(args.checkpoint):
            raise ValueError("resume artifact belongs to another checkpoint")
        if tuple(positions) != resume_positions:
            raise ValueError("resume artifact uses different positions")
        if resume_payload.get("placement") != args.placement:
            raise ValueError("resume artifact uses a different placement")
        if resume_payload.get("stages") != [asdict(stage) for stage in stages]:
            raise ValueError("resume artifact uses a different curriculum")
        if float(resume_payload.get("state_loss_weight", 0.0)) != float(
            args.state_loss_weight
        ):
            raise ValueError("resume artifact uses a different state loss weight")
        if int(resume_payload.get("task_step_size", task_step_size)) != int(
            task_step_size
        ):
            raise ValueError("resume artifact uses a different task step size")
        module_payloads = dict(resume_payload["modules"])
        for item in module_payloads.values():
            item.setdefault("parameterization", "fixed_identity")
            item.setdefault("identity_scale_init", args.identity_scale_init)
            item.setdefault("diagonal_scale_init", args.diagonal_scale_init)
            item.setdefault("initial_identity_scale", 1.0)
            item.setdefault("final_identity_scale", 1.0)
            item.setdefault(
                "initial_scale_statistics",
                {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
            )
            item.setdefault(
                "final_scale_statistics",
                {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
            )
        training_rows = _read_csv(out_dir / "training_rounds.csv")
    probe = torch.randn(4, 3, cfg.d_model, device=device)

    for rank in args.ranks:
        for initialization_seed in args.initialization_seeds:
            set_seed(initialization_seed)
            label_prefix = (
                "task_scalar_lora"
                if args.parameterization == "scalar_low_rank"
                else (
                    "task_diagonal_lora"
                    if args.parameterization == "diagonal_low_rank"
                    else "task_lora"
                )
            )
            label = f"{label_prefix}_r{rank}_seed{initialization_seed}"
            if label in module_payloads:
                continue
            if args.parameterization == "scalar_low_rank":
                module = ScaledIdentityLoRAJ(
                    dimension=cfg.d_model,
                    rank=rank,
                ).to(device)
            elif args.parameterization == "diagonal_low_rank":
                module = DiagonalIdentityLoRAJ(
                    dimension=cfg.d_model,
                    rank=rank,
                ).to(device)
            else:
                module = ResidualLoRAJ(
                    dimension=cfg.d_model,
                    rank=rank,
                    alpha=float(rank),
                ).to(device)
            initial_identity_scale = (
                float(module.identity_scale.item())
                if isinstance(module, ScaledIdentityLoRAJ)
                else 1.0
            )
            initial_scale_statistics = (
                _scale_statistics(module.diagonal_scale)
                if isinstance(module, DiagonalIdentityLoRAJ)
                else _scale_statistics(
                    torch.tensor([initial_identity_scale], device=device)
                )
            )
            retained_initial_update_energy: float | None = None
            if initial_affine is not None:
                if isinstance(module, ScaledIdentityLoRAJ):
                    retained_initial_update_energy = (
                        module.initialize_from_affine_svd(
                            initial_affine,
                            gauge_seed=initialization_seed,
                            identity_scale_init=args.identity_scale_init,
                        )
                    )
                    initial_identity_scale = float(
                        module.identity_scale.item()
                    )
                    initial_scale_statistics = _scale_statistics(
                        module.identity_scale.reshape(1)
                    )
                elif isinstance(module, DiagonalIdentityLoRAJ):
                    retained_initial_update_energy = (
                        module.initialize_from_affine_svd(
                            initial_affine,
                            gauge_seed=initialization_seed,
                            diagonal_scale_init=args.diagonal_scale_init,
                        )
                    )
                    initial_scale_statistics = _scale_statistics(
                        module.diagonal_scale
                    )
                else:
                    retained_initial_update_energy = module.initialize_from_affine_svd(
                        initial_affine,
                        gauge_seed=initialization_seed,
                    )
            with torch.no_grad():
                if initial_affine is None:
                    initialization_error = float(
                        (module(probe) - probe).abs().max().item()
                    )
                else:
                    identity = torch.eye(
                        cfg.d_model,
                        device=device,
                        dtype=torch.float32,
                    )
                    if isinstance(module, ScaledIdentityLoRAJ):
                        truncated_weight = (
                            module.identity_scale * identity
                            + module.A @ module.B
                        )
                    elif isinstance(module, DiagonalIdentityLoRAJ):
                        truncated_weight = (
                            torch.diag(module.diagonal_scale)
                            + module.A @ module.B
                        )
                    else:
                        truncated_weight = identity + module.scale * (
                            module.A @ module.B
                        )
                    expected = (
                        probe.float() @ truncated_weight
                        + initial_affine.bias.float()
                    )
                    initialization_error = float(
                        (module(probe) - expected).abs().max().item()
                    )
            if initialization_error > 1e-5:
                raise RuntimeError("LoRA-J initialization equivalence failed")

            variant_rows: list[dict[str, Any]] = []
            snapshots: dict[str, dict[str, torch.Tensor]] = {}
            for stage in stages:
                optimizer_parameter_groups = None
                if isinstance(module, ScaledIdentityLoRAJ):
                    optimizer_parameter_groups = [
                        {"params": [module.A, module.B, module.bias]},
                        {
                            "params": [module.identity_scale],
                            "lr": (
                                stage.learning_rate
                                * args.scale_learning_rate_multiplier
                            ),
                        },
                    ]
                elif isinstance(module, DiagonalIdentityLoRAJ):
                    optimizer_parameter_groups = [
                        {"params": [module.A, module.B, module.bias]},
                        {
                            "params": [module.diagonal_scale],
                            "lr": (
                                stage.learning_rate
                                * args.scale_learning_rate_multiplier
                            ),
                        },
                    ]
                rows = _train_stage(
                    module=module,
                    stage=stage,
                    model=model,
                    cfg=cfg,
                    phase_positions=phase_positions,
                    positions=positions,
                    device=device,
                    state_loss_weight=args.state_loss_weight,
                    grad_clip=args.grad_clip,
                    placement=args.placement,
                    optimizer_parameter_groups=optimizer_parameter_groups,
                )
                for row in rows:
                    row.update(
                        {
                            "variant": label,
                            "rank": rank,
                            "initialization_seed": initialization_seed,
                            "controller_parameter_count": (
                                module.parameter_count
                            ),
                            "controller_to_backbone_ratio": (
                                module.parameter_count
                                / backbone_parameter_count
                            ),
                        }
                    )
                variant_rows.extend(rows)
                snapshots[stage.name] = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }

            modules[label] = module.frozen()
            training_rows.extend(variant_rows)
            module_payloads[label] = {
                "rank": rank,
                "alpha": module.alpha,
                "parameterization": args.parameterization,
                "identity_scale_init": args.identity_scale_init,
                "diagonal_scale_init": args.diagonal_scale_init,
                "scale_learning_rate_multiplier": (
                    args.scale_learning_rate_multiplier
                ),
                "initial_identity_scale": initial_identity_scale,
                "final_identity_scale": (
                    float(module.identity_scale.detach().item())
                    if isinstance(module, ScaledIdentityLoRAJ)
                    else 1.0
                ),
                "initial_scale_statistics": initial_scale_statistics,
                "final_scale_statistics": _scale_statistics(
                    module.diagonal_scale
                    if isinstance(module, DiagonalIdentityLoRAJ)
                    else (
                        module.identity_scale.reshape(1)
                        if isinstance(module, ScaledIdentityLoRAJ)
                        else torch.ones(1, device=device)
                    )
                ),
                "dimension": cfg.d_model,
                "initialization_seed": initialization_seed,
                "parameter_count": module.parameter_count,
                "controller_to_backbone_ratio": (
                    module.parameter_count / backbone_parameter_count
                ),
                "initialization_max_abs_error": initialization_error,
                "retained_initial_update_energy": (
                    retained_initial_update_energy
                ),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in module.state_dict().items()
                },
                "stage_snapshots": snapshots,
            }
            artifact_tmp = out_dir / "task_lora_j.pt.tmp"
            torch.save(
                {
                    "kind": "graph_path_telomere_task_lora_j",
                    "checkpoint": str(args.checkpoint),
                    "positions": positions,
                    "placement": args.placement,
                    "reference_affine_artifact": str(
                        args.reference_affine_artifact
                    ),
                    "reference_affine_label": args.reference_affine_label,
                    "initialization_mode": args.initialization,
                    "initial_affine_artifact": (
                        str(args.initial_affine_artifact)
                        if args.initial_affine_artifact is not None
                        else None
                    ),
                    "initial_affine_label": args.initial_affine_label,
                    "require_affine_placement": args.require_affine_placement,
                    "state_loss_weight": args.state_loss_weight,
                    "curriculum": args.curriculum,
                    "max_training_horizon": args.max_training_horizon,
                    "backbone_loss_description": args.backbone_loss_description,
                    "task_step_size": task_step_size,
                    "stages": [asdict(stage) for stage in stages],
                    "modules": module_payloads,
                },
                artifact_tmp,
            )
            os.replace(artifact_tmp, artifact)
            _write_csv(out_dir / "training_rounds.csv", training_rows)

    evaluation_maps: dict[str, Any] = {
        "identity_no_J": IdentityJ().to(device).eval(),
        "reference_full_affine": reference_affine,
    }
    evaluation_maps.update(modules)
    random_rows = evaluate_random_unit_every(
        maps=evaluation_maps,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.evaluation_batch_size,
        batches=args.evaluation_batches,
        continuation_loops=args.evaluation_loops,
        seed=args.evaluation_seed,
        placement=args.placement,
    )
    _write_csv(out_dir / "random_graph_closed_loop.csv", random_rows)
    curves = {
        label: _curve_summary(
            [
                float(row["accuracy"])
                for row in random_rows
                if row["variant"] == label
            ]
        )
        for label in evaluation_maps
    }

    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            f"frozen backbone: {args.backbone_loss_description}; LoRA-J "
            f"trained with phase-matched f^{task_step_size} CE at every "
            "controlled continuation loop"
            + (
                "; no hidden-state loss"
                if args.state_loss_weight == 0.0
                else (
                    "; normalized hidden-state MSE weight "
                    f"{args.state_loss_weight:g}"
                )
            )
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "backbone_parameter_count": backbone_parameter_count,
        "controller": (
            "J(h)=alpha*h+(hA)B+b; alpha learned; A/B rank constrained; "
            "no full affine path"
            if args.parameterization == "scalar_low_rank"
            else (
                "J(h)=h*diag(alpha)+(hA)B+b; diagonal learned; A/B rank "
                "constrained; no full affine path"
                if args.parameterization == "diagonal_low_rank"
                else "J(h)=h+(hA)B+b; A/B rank constrained; no full affine path"
            )
        ),
        "parameterization": args.parameterization,
        "identity_scale_init": args.identity_scale_init,
        "diagonal_scale_init": args.diagonal_scale_init,
        "scale_learning_rate_multiplier": args.scale_learning_rate_multiplier,
        "controller_placement": args.placement,
        "controller_placement_definition": (
            "after Block1 FFN and before Block2 attention"
            if args.placement == "pre_block2"
            else (
                "on the recurrent state produced after Block2 FFN; applied "
                "before the next complete loop, whose output is read out"
            )
        ),
        "ranks": list(args.ranks),
        "initialization_seeds": list(args.initialization_seeds),
        "initialization": (
            "exact identity via zero B and zero bias"
            if args.initialization == "identity"
            else (
                "rank-truncated SVD of the explicitly named affine artifact "
                "at its recorded placement, with identical-function random "
                "orthogonal gauges"
            )
        ),
        "initialization_mode": args.initialization,
        "initial_affine_artifact": (
            str(args.initial_affine_artifact)
            if args.initial_affine_artifact is not None
            else None
        ),
        "initial_affine_label": args.initial_affine_label,
        "require_affine_placement": args.require_affine_placement,
        "curriculum": args.curriculum,
        "max_training_horizon": args.max_training_horizon,
        "task_step_size": task_step_size,
        "learning_rate_multiplier": args.learning_rate_multiplier,
        "state_loss_weight": args.state_loss_weight,
        "stages": [asdict(stage) for stage in stages],
        "task_training_graph_draws": sum(stage.graphs for stage in stages),
        "variants": {
            label: {
                "rank": module.rank,
                "parameter_count": module.parameter_count,
                "initial_identity_scale": module_payloads[label][
                    "initial_identity_scale"
                ],
                "final_identity_scale": module_payloads[label][
                    "final_identity_scale"
                ],
                "initial_scale_statistics": module_payloads[label][
                    "initial_scale_statistics"
                ],
                "final_scale_statistics": module_payloads[label][
                    "final_scale_statistics"
                ],
                "controller_to_backbone_ratio": (
                    module.parameter_count / backbone_parameter_count
                ),
                "initialization_seed": int(label.rsplit("seed", 1)[1]),
            }
            for label, module in modules.items()
        },
        "random_graph_evaluation": {
            "examples": args.evaluation_batch_size * args.evaluation_batches,
            "loops": args.evaluation_loops,
            "seed": args.evaluation_seed,
            "curves": curves,
        },
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "prelaunch_used_mib": args.prelaunch_used_mib,
            "prelaunch_free_mib": args.prelaunch_free_mib,
            "declared_peak_gib": args.declared_peak_gib,
            "reserve_gib": args.reserve_gib,
            "shared": args.shared_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "files": {
            "artifact": "task_lora_j.pt",
            "training": "training_rounds.csv",
            "random_evaluation": "random_graph_closed_loop.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a low-rank residual LoRA rejuvenator at a recurrent or "
            "internal interface while keeping the D8L8 backbone frozen."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--reference-affine-artifact", type=Path, required=True)
    parser.add_argument("--reference-affine-label", default="task")
    parser.add_argument(
        "--initialization",
        choices=("identity", "affine_svd"),
        default="identity",
    )
    parser.add_argument("--initial-affine-artifact", type=Path)
    parser.add_argument("--initial-affine-label", default="reg")
    parser.add_argument(
        "--require-affine-placement",
        action="store_true",
        help="reject affine artifacts that do not record the controller placement",
    )
    parser.add_argument(
        "--parameterization",
        choices=("fixed_identity", "scalar_low_rank", "diagonal_low_rank"),
        default="fixed_identity",
    )
    parser.add_argument(
        "--identity-scale-init",
        choices=("trace", "one"),
        default="trace",
    )
    parser.add_argument(
        "--diagonal-scale-init",
        choices=("diagonal", "one"),
        default="diagonal",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--placement",
        choices=("pre_block2", "loop_boundary"),
        default="loop_boundary",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    parser.add_argument(
        "--initialization-seeds",
        type=int,
        nargs="+",
        default=(211001, 311001, 411001),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--curriculum",
        choices=("canonical", "strict_h8"),
        default="canonical",
        help=(
            "strict_h8 uses only 1/2/4/6/8-step unrolls; canonical preserves "
            "the historical h24/h32/h48/h64 recipe"
        ),
    )
    parser.add_argument(
        "--max-training-horizon",
        type=int,
        choices=(8, 24, 32, 48, 64, 96),
        default=64,
    )
    parser.add_argument(
        "--backbone-loss-description",
        default="final-only CE at loop 8",
    )
    parser.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--state-loss-weight",
        type=float,
        default=0.0,
        help=(
            "weight of normalized hidden-state MSE added to per-step successor "
            "CE; canonical training uses 0"
        ),
    )
    parser.add_argument(
        "--scale-learning-rate-multiplier",
        type=float,
        default=1.0,
        help="learning-rate multiplier for scalar or diagonal scale parameters",
    )
    parser.add_argument("--stage-round-limit", type=int)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--evaluation-loops", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int, default=212004)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.04)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=3.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed rank/seed variants from out-dir/task_lora_j.pt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
