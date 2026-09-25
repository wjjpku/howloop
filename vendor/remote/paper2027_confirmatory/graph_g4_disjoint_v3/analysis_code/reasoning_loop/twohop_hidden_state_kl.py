from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.twohop_in_context import (
    TwoHopBatch,
    TwoHopTransformer,
    build_twohop_model,
    make_twohop_batch,
)


ROLE_NAMES = (
    "BOS",
    "distractor_first_parent",
    "distractor_first_child",
    "distractor_second_parent",
    "distractor_second_child",
    "target_first_parent",
    "target_first_child",
    "target_second_parent",
    "target_second_child",
    "query",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hidden-state distributions across effective depths and "
            "positions in trained standard and periodic two-hop Transformers."
        )
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["S6", "P2x3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--examples", type=int, default=8192)
    parser.add_argument("--pca-examples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--cov-jitter", type=float, default=1e-6)
    parser.add_argument("--question-seed-base", type=int, default=61000)
    parser.add_argument("--sample-question-count", type=int, default=16)
    parser.add_argument(
        "--active-depth",
        type=int,
        default=None,
        help="Effective depth to analyze; defaults to the trained depth.",
    )
    parser.add_argument(
        "--cycle-standard-beyond-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For a standard model evaluated beyond its trained depth, repeat "
            "the complete trained block stack as an explicit cycle control."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def load_model(
    suite_dir: Path,
    run_summary: Mapping[str, Any],
    device: torch.device,
) -> TwoHopTransformer:
    checkpoint_path = Path(str(run_summary["checkpoint"]))
    if not checkpoint_path.exists():
        checkpoint_path = (
            suite_dir
            / "runs"
            / str(run_summary["run_name"])
            / "best.pt"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    from reasoning_loop.twohop_in_context import TwoHopConfig

    cfg = TwoHopConfig.from_dict(checkpoint["config"])
    model = build_twohop_model(
        cfg,
        seed=int(checkpoint.get("init_seed", checkpoint["seed"])),
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def all_hidden_states(
    model: TwoHopTransformer,
    tokens: torch.Tensor,
    *,
    active_depth: int | None = None,
    cycle_standard_beyond_depth: bool = False,
) -> torch.Tensor:
    depth = model.cfg.total_depth if active_depth is None else active_depth
    if depth < 1:
        raise ValueError("active_depth must be positive")
    embedding = (
        model.token_embedding(tokens)
        + model.position_embedding.unsqueeze(0)
    )
    if (
        model.cfg.architecture == "standard"
        and depth > model.cfg.total_depth
    ):
        if not cycle_standard_beyond_depth:
            raise ValueError(
                "standard model requires cycle_standard_beyond_depth=True "
                "when active_depth exceeds the trained depth"
            )
        values = embedding
        state_list: list[torch.Tensor] = []
        for effective_depth in range(depth):
            parameter_index = effective_depth % model.cfg.total_depth
            values, _ = model.blocks[parameter_index](
                values,
                model.causal_mask,
                return_cache=False,
            )
            state_list.append(values)
        states = torch.stack(state_list, dim=1)
    else:
        result = model.forward_all(tokens, active_depth=depth)
        states = result["states_by_depth"]
        if not isinstance(states, torch.Tensor):
            raise RuntimeError("states_by_depth is not a tensor")
    return torch.cat((embedding[:, None], states), dim=1).float()


def semantic_role_ids(batch: TwoHopBatch) -> torch.Tensor:
    device = batch.tokens.device
    batch_size = batch.batch_size
    chain_count = batch.chain_count
    roles = torch.full_like(batch.tokens, -1)
    roles[:, 0] = 0
    roles[:, -1] = 9
    rows = torch.arange(batch_size, device=device)[:, None].expand(
        -1,
        chain_count,
    )
    chains = torch.arange(chain_count, device=device)[None].expand(
        batch_size,
        -1,
    )
    is_target = chains.eq(batch.target_indices[:, None])
    fields = (
        ("first_parent_positions", 1, 5),
        ("first_child_positions", 2, 6),
        ("second_parent_positions", 3, 7),
        ("second_child_positions", 4, 8),
    )
    for field, distractor_role, target_role in fields:
        positions = getattr(batch, field)
        roles[rows[~is_target], positions[~is_target]] = distractor_role
        roles[rows[is_target], positions[is_target]] = target_role
    if not roles.ge(0).all():
        raise RuntimeError("semantic role assignment did not cover every token")
    return roles


@dataclass
class MomentAccumulator:
    count: torch.Tensor
    total: torch.Tensor
    cross: torch.Tensor

    @classmethod
    def create(
        cls,
        group_shape: tuple[int, ...],
        dim: int,
        device: torch.device,
    ) -> "MomentAccumulator":
        return cls(
            count=torch.zeros(group_shape, dtype=torch.float64, device=device),
            total=torch.zeros(
                (*group_shape, dim),
                dtype=torch.float64,
                device=device,
            ),
            cross=torch.zeros(
                (*group_shape, dim, dim),
                dtype=torch.float64,
                device=device,
            ),
        )

    def update_dense(self, values: torch.Tensor) -> None:
        # values: [batch, *group_shape, dim]
        expected_group_shape = tuple(self.count.shape)
        if tuple(values.shape[1:-1]) != expected_group_shape:
            raise ValueError("dense values have the wrong group shape")
        self.count += values.shape[0]
        self.total += values.sum(dim=0, dtype=torch.float64)
        self.cross += torch.einsum(
            "b...i,b...j->...ij",
            values,
            values,
        ).to(torch.float64)

    def update_role_values(
        self,
        values: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> None:
        # values: [batch, depth_or_step, position, dim]
        if self.count.ndim != 2:
            raise ValueError("role accumulator must have [time, role] groups")
        _, time_count, position_count, dim = values.shape
        if role_ids.shape != (values.shape[0], position_count):
            raise ValueError("role_ids has the wrong shape")
        flat = values.permute(1, 0, 2, 3).reshape(
            time_count,
            -1,
            dim,
        )
        flat_roles = role_ids.reshape(-1)
        for role in range(self.count.shape[1]):
            selected = flat[:, flat_roles.eq(role), :]
            sample_count = selected.shape[1]
            if sample_count == 0:
                raise RuntimeError(f"semantic role {role} has no samples")
            self.count[:, role] += sample_count
            self.total[:, role] += selected.sum(dim=1, dtype=torch.float64)
            self.cross[:, role] += torch.einsum(
                "tni,tnj->tij",
                selected,
                selected,
            ).to(torch.float64)

    def numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.count.cpu().numpy(),
            self.total.cpu().numpy(),
            self.cross.cpu().numpy(),
        )


def finalize_gaussians(
    accumulator: MomentAccumulator,
    *,
    shrinkage: float,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, total, cross = accumulator.numpy()
    if np.any(count < 2):
        raise RuntimeError("at least two samples are required per distribution")
    mean = total / count[..., None]
    covariance = (
        cross
        - count[..., None, None]
        * np.einsum("...i,...j->...ij", mean, mean)
    ) / (count[..., None, None] - 1.0)
    dim = mean.shape[-1]
    identity = np.eye(dim, dtype=np.float64)
    flat_covariance = covariance.reshape(-1, dim, dim)
    for index in range(flat_covariance.shape[0]):
        cov = 0.5 * (
            flat_covariance[index] + flat_covariance[index].T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        scale = max(
            float(np.abs(eigenvalues).mean()),
            float(eigenvalues.max(initial=0.0) / dim),
            jitter,
        )
        eigenvalues = np.clip(eigenvalues, jitter * scale, None)
        cov = (eigenvectors * eigenvalues) @ eigenvectors.T
        flat_covariance[index] = (
            (1.0 - shrinkage) * cov
            + shrinkage * scale * identity
            + jitter * scale * identity
        )
    covariance = flat_covariance.reshape(covariance.shape)
    return count, mean, covariance


def gaussian_kl(
    mean_left: np.ndarray,
    cov_left: np.ndarray,
    mean_right: np.ndarray,
    cov_right: np.ndarray,
) -> float:
    dim = mean_left.shape[0]
    difference = mean_right - mean_left
    sign_left, logdet_left = np.linalg.slogdet(cov_left)
    sign_right, logdet_right = np.linalg.slogdet(cov_right)
    if sign_left <= 0 or sign_right <= 0:
        raise np.linalg.LinAlgError("covariance must be positive definite")
    trace_term = np.trace(np.linalg.solve(cov_right, cov_left))
    mean_term = float(
        difference @ np.linalg.solve(cov_right, difference)
    )
    value = 0.5 * (
        trace_term
        + mean_term
        - dim
        + logdet_right
        - logdet_left
    )
    return max(float(value), 0.0)


def symmetric_gaussian_kl(
    mean_left: np.ndarray,
    cov_left: np.ndarray,
    mean_right: np.ndarray,
    cov_right: np.ndarray,
) -> float:
    return 0.5 * (
        gaussian_kl(mean_left, cov_left, mean_right, cov_right)
        + gaussian_kl(mean_right, cov_right, mean_left, cov_left)
    )


def pairwise_kl(
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    item_count = means.shape[0]
    matrix = np.zeros((item_count, item_count), dtype=np.float64)
    for left in range(item_count):
        for right in range(left + 1, item_count):
            value = symmetric_gaussian_kl(
                means[left],
                covariances[left],
                means[right],
                covariances[right],
            )
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def compute_kl_metrics(
    state_absolute: tuple[np.ndarray, np.ndarray, np.ndarray],
    state_role: tuple[np.ndarray, np.ndarray, np.ndarray],
    delta_role: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    _, absolute_mean, absolute_cov = state_absolute
    _, role_mean, role_cov = state_role
    _, delta_mean, delta_cov = delta_role
    depth_count, position_count = absolute_mean.shape[:2]
    role_count = role_mean.shape[1]
    step_count = delta_mean.shape[0]

    absolute_transition = np.zeros(
        (depth_count - 1, position_count),
        dtype=np.float64,
    )
    role_transition = np.zeros(
        (depth_count - 1, role_count),
        dtype=np.float64,
    )
    for depth in range(depth_count - 1):
        for position in range(position_count):
            absolute_transition[depth, position] = symmetric_gaussian_kl(
                absolute_mean[depth, position],
                absolute_cov[depth, position],
                absolute_mean[depth + 1, position],
                absolute_cov[depth + 1, position],
            )
        for role in range(role_count):
            role_transition[depth, role] = symmetric_gaussian_kl(
                role_mean[depth, role],
                role_cov[depth, role],
                role_mean[depth + 1, role],
                role_cov[depth + 1, role],
            )

    role_depth_pairwise = np.stack(
        [
            pairwise_kl(role_mean[:, role], role_cov[:, role])
            for role in range(role_count)
        ]
    )
    position_pairwise_by_depth = np.stack(
        [
            pairwise_kl(role_mean[depth], role_cov[depth])
            for depth in range(depth_count)
        ]
    )
    delta_step_pairwise = np.stack(
        [
            pairwise_kl(delta_mean[:, role], delta_cov[:, role])
            for role in range(role_count)
        ]
    )
    if delta_step_pairwise.shape[1:] != (step_count, step_count):
        raise RuntimeError("delta pairwise matrix has an unexpected shape")
    return {
        "absolute_transition": absolute_transition,
        "role_transition": role_transition,
        "role_depth_pairwise": role_depth_pairwise,
        "position_pairwise_by_depth": position_pairwise_by_depth,
        "delta_step_pairwise": delta_step_pairwise,
    }


def _global_moment_update(
    count: int,
    total: torch.Tensor,
    cross: torch.Tensor,
    values: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    flat = values.reshape(-1, values.shape[-1])
    count += flat.shape[0]
    total += flat.sum(dim=0, dtype=torch.float64)
    cross += (flat.T @ flat).to(torch.float64)
    return count, total, cross


@torch.inference_mode()
def fit_shared_pca(
    models: Mapping[str, TwoHopTransformer],
    *,
    examples: int,
    batch_size: int,
    seed: int,
    pca_dim: int,
    device: torch.device,
    active_depth: int | None = None,
    cycle_standard_beyond_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    first_model = next(iter(models.values()))
    d_model = first_model.cfg.d_model
    count = 0
    total = torch.zeros(d_model, dtype=torch.float64, device=device)
    cross = torch.zeros(
        (d_model, d_model),
        dtype=torch.float64,
        device=device,
    )
    generator = generator_for(device, seed)
    for start in range(0, examples, batch_size):
        current = min(batch_size, examples - start)
        batch = make_twohop_batch(
            first_model.cfg,
            current,
            device=device,
            generator=generator,
            order_mode="topological",
        )
        for model in models.values():
            states = all_hidden_states(
                model,
                batch.tokens,
                active_depth=active_depth,
                cycle_standard_beyond_depth=(
                    cycle_standard_beyond_depth
                ),
            )
            count, total, cross = _global_moment_update(
                count,
                total,
                cross,
                states,
            )
    mean = total / count
    covariance = (
        cross - count * torch.outer(mean, mean)
    ) / (count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    retained = min(pca_dim, d_model)
    basis = eigenvectors[:, :retained]
    explained = (
        eigenvalues[:retained].sum()
        / eigenvalues.sum().clamp_min(1e-12)
    )
    spectrum = eigenvalues.cpu().numpy()
    return mean.float(), basis.float(), np.array(float(explained.item()))


def _decode_question(batch: TwoHopBatch, index: int) -> dict[str, Any]:
    token_ids = batch.tokens[index].cpu().tolist()
    target = int(batch.target_indices[index].item())
    chain_ids = batch.chains[index].cpu().tolist()
    names = lambda values: [f"E{int(value):02d}" for value in values]
    premise_tokens = token_ids[1:-1]
    facts = [
        names(premise_tokens[offset : offset + 2])
        for offset in range(0, len(premise_tokens), 2)
    ]
    return {
        "tokens": ["BOS", *names(token_ids[1:])],
        "facts_in_input_order": facts,
        "chains": [names(chain) for chain in chain_ids],
        "target_chain_index": target,
        "query": f"E{token_ids[-1]:02d}",
        "answer": f"E{int(batch.labels[index].item()):02d}",
    }


@torch.inference_mode()
def analyze_models_for_seed(
    models: Mapping[str, TwoHopTransformer],
    *,
    examples: int,
    batch_size: int,
    seed: int,
    pca_mean: torch.Tensor,
    pca_basis: torch.Tensor,
    shrinkage: float,
    jitter: float,
    device: torch.device,
    sample_question_count: int,
    active_depth: int | None = None,
    cycle_standard_beyond_depth: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    first_model = next(iter(models.values()))
    cfg = first_model.cfg
    effective_depth = (
        cfg.total_depth if active_depth is None else active_depth
    )
    if effective_depth < 1:
        raise ValueError("active_depth must be positive")
    depth_count = effective_depth + 1
    dim = pca_basis.shape[1]
    accumulators: dict[str, dict[str, MomentAccumulator]] = {}
    for name in models:
        accumulators[name] = {
            "state_absolute": MomentAccumulator.create(
                (depth_count, cfg.seq_len),
                dim,
                device,
            ),
            "state_role": MomentAccumulator.create(
                (depth_count, len(ROLE_NAMES)),
                dim,
                device,
            ),
            "delta_role": MomentAccumulator.create(
                (effective_depth, len(ROLE_NAMES)),
                dim,
                device,
            ),
        }

    generator = generator_for(device, seed)
    sampled_questions: list[dict[str, Any]] = []
    for start in range(0, examples, batch_size):
        current = min(batch_size, examples - start)
        batch = make_twohop_batch(
            cfg,
            current,
            device=device,
            generator=generator,
            order_mode="topological",
        )
        if len(sampled_questions) < sample_question_count:
            keep = min(
                current,
                sample_question_count - len(sampled_questions),
            )
            sampled_questions.extend(
                _decode_question(batch, index)
                for index in range(keep)
            )
        role_ids = semantic_role_ids(batch)
        for name, model in models.items():
            states = all_hidden_states(
                model,
                batch.tokens,
                active_depth=effective_depth,
                cycle_standard_beyond_depth=(
                    cycle_standard_beyond_depth
                ),
            )
            projected = (states - pca_mean) @ pca_basis
            deltas = projected[:, 1:] - projected[:, :-1]
            accumulators[name]["state_absolute"].update_dense(projected)
            accumulators[name]["state_role"].update_role_values(
                projected,
                role_ids,
            )
            accumulators[name]["delta_role"].update_role_values(
                deltas,
                role_ids,
            )

    metrics: dict[str, dict[str, np.ndarray]] = {}
    for name, model_accumulators in accumulators.items():
        finalized = {
            key: finalize_gaussians(
                accumulator,
                shrinkage=shrinkage,
                jitter=jitter,
            )
            for key, accumulator in model_accumulators.items()
        }
        model_metrics = compute_kl_metrics(
            finalized["state_absolute"],
            finalized["state_role"],
            finalized["delta_role"],
        )
        model_metrics.update(
            {
                "state_role_count": finalized["state_role"][0],
                "state_role_mean": finalized["state_role"][1],
                "state_role_covariance": finalized["state_role"][2],
                "delta_role_count": finalized["delta_role"][0],
                "delta_role_mean": finalized["delta_role"][1],
                "delta_role_covariance": finalized["delta_role"][2],
            }
        )
        metrics[name] = model_metrics
    return metrics, sampled_questions


@torch.inference_mode()
def evaluate_models_by_depth(
    models: Mapping[str, TwoHopTransformer],
    *,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    active_depth: int | None = None,
    cycle_standard_beyond_depth: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    first_model = next(iter(models.values()))
    depth = (
        first_model.cfg.total_depth
        if active_depth is None
        else active_depth
    )
    if examples < 1 or batch_size < 1 or depth < 1:
        raise ValueError("examples, batch_size, and active_depth must be positive")
    totals = {
        name: {
            "correct": torch.zeros(depth, dtype=torch.float64),
            "margin": torch.zeros(depth, dtype=torch.float64),
            "loss": torch.zeros(depth, dtype=torch.float64),
        }
        for name in models
    }
    generator = generator_for(device, seed)
    seen = 0
    for start in range(0, examples, batch_size):
        current = min(batch_size, examples - start)
        batch = make_twohop_batch(
            first_model.cfg,
            current,
            device=device,
            generator=generator,
            order_mode="topological",
        )
        for name, model in models.items():
            states = all_hidden_states(
                model,
                batch.tokens,
                active_depth=depth,
                cycle_standard_beyond_depth=(
                    cycle_standard_beyond_depth
                ),
            )
            query_states = states[:, 1:, -1]
            logits = model.readout(model.final_norm(query_states)).float()
            predictions = logits.argmax(dim=-1)
            totals[name]["correct"] += (
                predictions.eq(batch.labels[:, None])
                .sum(dim=0)
                .double()
                .cpu()
            )
            correct_logits = logits.gather(
                2,
                batch.labels[:, None, None].expand(-1, depth, 1),
            ).squeeze(2)
            competitors = logits.clone()
            competitors.scatter_(
                2,
                batch.labels[:, None, None].expand(-1, depth, 1),
                -torch.inf,
            )
            totals[name]["margin"] += (
                correct_logits - competitors.max(dim=2).values
            ).sum(dim=0).double().cpu()
            repeated_labels = batch.labels[:, None].expand(-1, depth)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                repeated_labels.reshape(-1),
                reduction="none",
            ).reshape(current, depth)
            totals[name]["loss"] += losses.sum(dim=0).double().cpu()
        seen += current
    return {
        name: {
            "depth_accuracy": values["correct"].numpy() / seen,
            "depth_margin": values["margin"].numpy() / seen,
            "depth_loss": values["loss"].numpy() / seen,
        }
        for name, values in totals.items()
    }


def aggregate_seed_metrics(
    all_seed_metrics: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    model_names: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    aggregate: dict[str, dict[str, np.ndarray]] = {}
    metric_names = (
        "absolute_transition",
        "role_transition",
        "role_depth_pairwise",
        "position_pairwise_by_depth",
        "delta_step_pairwise",
    )
    for model_name in model_names:
        aggregate[model_name] = {}
        for metric_name in metric_names:
            values = np.stack(
                [
                    all_seed_metrics[seed][model_name][metric_name]
                    for seed in sorted(all_seed_metrics)
                ]
            )
            aggregate[model_name][f"{metric_name}_mean"] = values.mean(axis=0)
            aggregate[model_name][f"{metric_name}_std"] = values.std(axis=0)
    return aggregate


def _operator_signature(
    delta_pairwise: np.ndarray,
    schedule: Sequence[int],
) -> tuple[float, float]:
    same: list[float] = []
    different: list[float] = []
    for left in range(len(schedule)):
        for right in range(left + 1, len(schedule)):
            target = same if schedule[left] == schedule[right] else different
            target.append(float(delta_pairwise[left, right]))
    return (
        float(np.mean(same)) if same else float("nan"),
        float(np.mean(different)) if different else float("nan"),
    )


def _write_long_csv(
    out_dir: Path,
    aggregate: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    rows: list[dict[str, Any]] = []
    for model_name, metrics in aggregate.items():
        role_transition = metrics["role_transition_mean"]
        role_transition_std = metrics["role_transition_std"]
        for depth in range(role_transition.shape[0]):
            for role, role_name in enumerate(ROLE_NAMES):
                rows.append(
                    {
                        "model": model_name,
                        "metric": "role_transition",
                        "left": f"D{depth}",
                        "right": f"D{depth + 1}",
                        "position_or_role": role_name,
                        "mean_symmetric_kl": role_transition[depth, role],
                        "std_symmetric_kl": role_transition_std[depth, role],
                    }
                )
    path = out_dir / "semantic_transition_kl.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(
    axis: Any,
    raw: np.ndarray,
    *,
    title: str,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    vmax: float,
    annotate: bool,
) -> None:
    transformed = np.log1p(raw)
    image = axis.imshow(
        transformed,
        aspect="auto",
        cmap="coolwarm_r",
        vmin=0.0,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_xticks(range(len(xlabels)))
    axis.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(ylabels)))
    axis.set_yticklabels(ylabels, fontsize=8)
    if annotate:
        for row in range(raw.shape[0]):
            for column in range(raw.shape[1]):
                value = raw[row, column]
                text = f"{value:.1f}" if value < 100 else f"{value:.0f}"
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=(
                        "white"
                        if (
                            transformed[row, column] < 0.18 * vmax
                            or transformed[row, column] > 0.82 * vmax
                        )
                        else "black"
                    ),
                )
    return image


def make_plots(
    out_dir: Path,
    aggregate: Mapping[str, Mapping[str, np.ndarray]],
    model_names: Sequence[str],
    model_schedules: Mapping[str, Sequence[int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def shared_vmax(
        metric_name: str,
        *,
        exclude_bos_axis: str | None = None,
    ) -> float:
        selected: list[np.ndarray] = []
        for name in model_names:
            values = aggregate[name][metric_name]
            if exclude_bos_axis == "column":
                values = values[..., 1:]
            elif exclude_bos_axis == "matrix":
                values = values[..., 1:, 1:]
            selected.append(np.log1p(values).ravel())
        values = np.concatenate(selected)
        return max(float(np.quantile(values, 0.98)), 1e-8)

    reference_name = model_names[0]
    step_count = aggregate[reference_name]["role_transition_mean"].shape[0]
    position_count = aggregate[reference_name]["absolute_transition_mean"].shape[1]
    transition_labels = [
        f"D{depth}→D{depth + 1}" for depth in range(step_count)
    ]
    absolute_labels = [
        "BOS",
        *[str(index) for index in range(1, position_count - 1)],
        "query",
    ]

    for metric_name, filename, xlabels, ylabels, annotate in (
        (
            "role_transition_mean",
            "semantic_transition_kl.png",
            ROLE_NAMES,
            transition_labels,
            True,
        ),
        (
            "absolute_transition_mean",
            "absolute_transition_kl.png",
            absolute_labels,
            transition_labels,
            False,
        ),
    ):
        fig, axes = plt.subplots(
            1,
            len(model_names),
            figsize=(8 * len(model_names), 5),
            constrained_layout=True,
        )
        if len(model_names) == 1:
            axes = [axes]
        vmax = shared_vmax(
            metric_name,
            exclude_bos_axis="column",
        )
        image = None
        for axis, name in zip(axes, model_names):
            image = _plot_heatmap(
                axis,
                aggregate[name][metric_name],
                title=name,
                xlabels=xlabels,
                ylabels=ylabels,
                vmax=vmax,
                annotate=annotate,
            )
        fig.colorbar(
            image,
            ax=axes,
            label="log(1 + symmetric Gaussian KL); red = more similar",
            shrink=0.82,
        )
        fig.suptitle(filename.replace("_", " ").replace(".png", ""))
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    query_role = ROLE_NAMES.index("query")
    for metric_name, filename, labels in (
        (
            "role_depth_pairwise_mean",
            "query_depth_pairwise_kl.png",
            [f"D{depth}" for depth in range(step_count + 1)],
        ),
        (
            "delta_step_pairwise_mean",
            "query_delta_step_pairwise_kl.png",
            transition_labels,
        ),
    ):
        matrices = {
            name: aggregate[name][metric_name][query_role]
            for name in model_names
        }
        values = np.concatenate(
            [np.log1p(matrix).ravel() for matrix in matrices.values()]
        )
        vmax = max(float(np.quantile(values, 0.98)), 1e-8)
        fig, axes = plt.subplots(
            1,
            len(model_names),
            figsize=(6 * len(model_names), 5),
            constrained_layout=True,
        )
        if len(model_names) == 1:
            axes = [axes]
        image = None
        for axis, name in zip(axes, model_names):
            model_labels = labels
            if metric_name == "delta_step_pairwise_mean":
                model_labels = [
                    f"D{depth}→D{depth + 1}\nP{parameter}"
                    for depth, parameter in enumerate(model_schedules[name])
                ]
            image = _plot_heatmap(
                axis,
                matrices[name],
                title=name,
                xlabels=model_labels,
                ylabels=model_labels,
                vmax=vmax,
                annotate=True,
            )
        fig.colorbar(
            image,
            ax=axes,
            label="log(1 + symmetric Gaussian KL); red = more similar",
            shrink=0.82,
        )
        fig.suptitle(filename.replace("_", " ").replace(".png", ""))
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    postblock_matrices = {
        name: aggregate[name]["role_depth_pairwise_mean"][query_role][1:, 1:]
        for name in model_names
    }
    values = np.concatenate(
        [np.log1p(matrix).ravel() for matrix in postblock_matrices.values()]
    )
    vmax = max(float(np.quantile(values, 0.98)), 1e-8)
    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(6 * len(model_names), 5),
        constrained_layout=True,
    )
    if len(model_names) == 1:
        axes = [axes]
    image = None
    postblock_labels = [f"D{depth}" for depth in range(1, step_count + 1)]
    for axis, name in zip(axes, model_names):
        image = _plot_heatmap(
            axis,
            postblock_matrices[name],
            title=name,
            xlabels=postblock_labels,
            ylabels=postblock_labels,
            vmax=vmax,
            annotate=True,
        )
    fig.colorbar(
        image,
        ax=axes,
        label="log(1 + symmetric Gaussian KL); red = more similar",
        shrink=0.82,
    )
    fig.suptitle("query depth pairwise KL, post-block states only")
    fig.savefig(out_dir / "query_depth_pairwise_kl_postblock.png", dpi=200)
    plt.close(fig)

    vmax = shared_vmax(
        "position_pairwise_by_depth_mean",
        exclude_bos_axis="matrix",
    )
    for name in model_names:
        matrices = aggregate[name]["position_pairwise_by_depth_mean"]
        column_count = 4
        row_count = math.ceil(matrices.shape[0] / column_count)
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(18, 4.2 * row_count),
            constrained_layout=True,
            squeeze=False,
        )
        image = None
        for depth, axis in enumerate(axes.flat):
            if depth >= matrices.shape[0]:
                axis.axis("off")
                continue
            image = _plot_heatmap(
                axis,
                matrices[depth],
                title=f"{name} D{depth}",
                xlabels=ROLE_NAMES,
                ylabels=ROLE_NAMES,
                vmax=vmax,
                annotate=False,
            )
        fig.colorbar(
            image,
            ax=axes,
            label="log(1 + symmetric Gaussian KL); red = more similar",
            shrink=0.82,
        )
        fig.suptitle(f"{name}: pairwise semantic-position KL by depth")
        fig.savefig(
            out_dir / f"{name}_semantic_position_pairwise_kl.png",
            dpi=200,
        )
        plt.close(fig)


def write_report(
    args: argparse.Namespace,
    aggregate: Mapping[str, Mapping[str, np.ndarray]],
    model_schedules: Mapping[str, Sequence[int]],
    pca_explained: Mapping[int, float],
) -> None:
    query_role = ROLE_NAMES.index("query")
    step_count = aggregate[args.models[0]]["role_transition_mean"].shape[0]
    transition_labels = [
        f"D{depth}→D{depth + 1}" for depth in range(step_count)
    ]
    lines = [
        "# Hidden-state KL analysis: standard vs periodic two-hop Transformer",
        "",
        "The reported divergence is the symmetric KL between shrinkage-Gaussian "
        "approximations in a per-seed PCA space shared by both models. Heatmaps "
        "display `log(1 + KL)` for dynamic range; CSV/NPZ files contain raw KL. "
        "Red means lower KL / greater distributional similarity, while blue "
        "means higher KL / lower similarity.",
        "",
        f"- examples per model/seed: {args.examples}",
        f"- seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- PCA dimension: {args.pca_dim}",
        f"- covariance shrinkage: {args.cov_shrinkage}",
        f"- analyzed effective depth: {step_count}",
        f"- mean retained PCA variance: {np.mean(list(pca_explained.values())):.4f}",
        "",
        "Low KL means two hidden-state distributions are similar. It is a "
        "distributional similarity measure, not a causal influence score.",
        "",
        "## Consecutive query-state transitions",
        "",
        "| model | " + " | ".join(transition_labels) + " |",
        "|---|" + "---:|" * step_count,
    ]
    for name in args.models:
        values = aggregate[name]["role_transition_mean"][:, query_role]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Repeated-operator signature on query updates",
            "",
            "| model | same parameter block | different parameter blocks | ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in args.models:
        matrix = aggregate[name]["delta_step_pairwise_mean"][query_role]
        same, different = _operator_signature(
            matrix,
            model_schedules[name],
        )
        ratio = same / different if different > 0 and math.isfinite(same) else float("nan")
        lines.append(
            f"| {name} | {same:.3f} | {different:.3f} | {ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "Lower within-parameter than across-parameter delta KL indicates "
            "an operator-specific update signature; high within-parameter KL "
            "indicates strong state conditioning. If a standard stack is "
            "explicitly cycled beyond its trained depth, repeated IDs refer "
            "to repeating the complete trained stack as a control, not native "
            "weight sharing during training.",
            "",
            "## Files",
            "",
            "- `semantic_transition_kl.png`: consecutive-depth KL by semantic role.",
            "- `absolute_transition_kl.png`: consecutive-depth KL by absolute token position.",
            "- `query_depth_pairwise_kl.png`: query-state phase distances.",
            "- `query_depth_pairwise_kl_postblock.png`: the exact post-block "
            "state matrix, excluding the D0 embedding state.",
            "- `query_delta_step_pairwise_kl.png`: query update-distribution distances.",
            "- `*_semantic_position_pairwise_kl.png`: pairwise role distributions at every depth.",
            "- `aggregate_metrics.npz`: raw aggregate means and standard deviations.",
            "- `per_seed/`: per-seed PCA and raw KL matrices.",
            "",
        ]
    )
    (args.out_dir / "REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.models) != 2:
        raise ValueError("this paired analysis expects exactly two models")
    if args.examples < 2 or args.pca_examples < 2:
        raise ValueError("examples and pca-examples must be at least two")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 0.0 <= args.cov_shrinkage < 1.0:
        raise ValueError("cov-shrinkage must be in [0, 1)")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    if args.out_dir is None:
        args.out_dir = args.suite_dir / "hidden_state_kl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = args.out_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    suite_summary = json.loads(
        (args.suite_dir / "summary.json").read_text(encoding="utf-8")
    )
    run_lookup = {
        (str(run["model"]), int(run["seed"])): run
        for run in suite_summary["runs"]
    }
    all_seed_metrics: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    pca_explained: dict[int, float] = {}
    sampled_questions: dict[str, list[dict[str, Any]]] = {}
    model_schedules: dict[str, list[int]] = {}
    analysis_depth: int | None = None

    for seed in args.seeds:
        print(f"=== hidden-state KL seed={seed} ===", flush=True)
        models = {
            name: load_model(
                args.suite_dir,
                run_lookup[(name, seed)],
                device,
            )
            for name in args.models
        }
        configs = [model.cfg for model in models.values()]
        reference = configs[0]
        for cfg in configs[1:]:
            for field in (
                "entity_count",
                "chain_count",
                "total_depth",
                "d_model",
                "n_heads",
                "d_mlp",
            ):
                if getattr(cfg, field) != getattr(reference, field):
                    raise ValueError(f"paired configs disagree on {field}")
        if args.pca_dim > reference.d_model:
            raise ValueError("pca-dim cannot exceed d_model")
        current_depth = (
            reference.total_depth
            if args.active_depth is None
            else args.active_depth
        )
        if current_depth < 1:
            raise ValueError("active-depth must be positive")
        if analysis_depth is None:
            analysis_depth = current_depth
        elif analysis_depth != current_depth:
            raise RuntimeError("analysis depth changed across seeds")
        for name, model in models.items():
            if (
                model.cfg.architecture == "standard"
                and current_depth > model.cfg.total_depth
            ):
                if not args.cycle_standard_beyond_depth:
                    raise ValueError(
                        "active-depth exceeds the standard model's trained "
                        "depth; pass --cycle-standard-beyond-depth to use the "
                        "explicit complete-stack cycle control"
                    )
                model_schedules[name] = [
                    depth % model.cfg.total_depth
                    for depth in range(current_depth)
                ]
            else:
                model_schedules[name] = [
                    model.parameter_index(depth)
                    for depth in range(current_depth)
                ]

        pca_seed = args.question_seed_base + 1000 * seed
        pca_mean, pca_basis, explained = fit_shared_pca(
            models,
            examples=args.pca_examples,
            batch_size=args.batch_size,
            seed=pca_seed,
            pca_dim=args.pca_dim,
            device=device,
            active_depth=current_depth,
            cycle_standard_beyond_depth=(
                args.cycle_standard_beyond_depth
            ),
        )
        pca_explained[seed] = float(explained)
        metrics, questions = analyze_models_for_seed(
            models,
            examples=args.examples,
            batch_size=args.batch_size,
            seed=pca_seed + 1,
            pca_mean=pca_mean,
            pca_basis=pca_basis,
            shrinkage=args.cov_shrinkage,
            jitter=args.cov_jitter,
            device=device,
            sample_question_count=args.sample_question_count,
            active_depth=current_depth,
            cycle_standard_beyond_depth=(
                args.cycle_standard_beyond_depth
            ),
        )
        all_seed_metrics[seed] = metrics
        sampled_questions[str(seed)] = questions
        seed_payload: dict[str, np.ndarray] = {
            "pca_mean": pca_mean.cpu().numpy(),
            "pca_basis": pca_basis.cpu().numpy(),
            "pca_explained_variance": np.array(explained),
        }
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                seed_payload[f"{model_name}__{metric_name}"] = value
        np.savez_compressed(
            per_seed_dir / f"seed_{seed}.npz",
            **seed_payload,
        )
        del models, metrics
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_seed_metrics(all_seed_metrics, args.models)
    aggregate_payload = {
        f"{model_name}__{metric_name}": value
        for model_name, metrics in aggregate.items()
        for metric_name, value in metrics.items()
    }
    np.savez_compressed(
        args.out_dir / "aggregate_metrics.npz",
        **aggregate_payload,
    )
    (args.out_dir / "sampled_questions.json").write_text(
        json.dumps(sampled_questions, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "method": (
            "per-seed shared PCA followed by shrinkage-Gaussian symmetric KL"
        ),
        "heatmap_transform": "log1p",
        "heatmap_colormap": (
            "coolwarm_r: red=lower KL/more similar; "
            "blue=higher KL/less similar"
        ),
        "analyzed_effective_depth": analysis_depth,
        "role_names": ROLE_NAMES,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "pca_explained_variance_by_seed": pca_explained,
        "parameter_schedule_by_model": model_schedules,
        "caveat": (
            "KL is distributional similarity, not positional causal influence; "
            "Gaussian and PCA approximations discard higher-order and omitted "
            "subspace structure."
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    _write_long_csv(args.out_dir, aggregate)
    make_plots(
        args.out_dir,
        aggregate,
        args.models,
        model_schedules,
    )
    write_report(
        args,
        aggregate,
        model_schedules,
        pca_explained,
    )
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        manifest["observed_peak_allocated_mib"] = peak_mib
        (args.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"observed_peak_allocated_mib={peak_mib:.1f}", flush=True)
    print(f"wrote hidden-state KL outputs under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
