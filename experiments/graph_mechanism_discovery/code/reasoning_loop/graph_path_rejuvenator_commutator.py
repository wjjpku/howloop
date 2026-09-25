from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as torch_f

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_rejuvenation_circuit import behavior_metrics
from reasoning_loop.graph_path_seed1_lowrank_telomere import LowRankRejuvenator
from reasoning_loop.graph_path_telomere_overloop import (
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


AnswerMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CommutatorBatch:
    h2: torch.Tensor
    h3: torch.Tensor
    h8: torch.Tensor
    oracle_h2_next: torch.Tensor
    oracle_h3_next: torch.Tensor
    oracle_h8_next: torch.Tensor
    current_h2: torch.Tensor
    current_h3: torch.Tensor
    current_h8: torch.Tensor
    target_h2: torch.Tensor
    target_h3: torch.Tensor
    target_h8: torch.Tensor
    future_h2: torch.Tensor
    future_h3: torch.Tensor
    future_h8: torch.Tensor


@dataclass(frozen=True)
class AffineAnswerMap:
    update_matrix: torch.Tensor
    bias: torch.Tensor

    def __call__(self, answer: torch.Tensor) -> torch.Tensor:
        return answer + answer @ self.update_matrix + self.bias


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rejuvenator_affine_parts(
    rejuvenator: LowRankRejuvenator,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_count = rejuvenator.feature_count
    if rejuvenator.rank:
        assert rejuvenator.left is not None and rejuvenator.right is not None
        update = rejuvenator.left @ rejuvenator.right
    else:
        update = torch.zeros(
            feature_count,
            feature_count,
            device=rejuvenator.bias.device,
            dtype=rejuvenator.bias.dtype,
        )
    return update, rejuvenator.bias


def load_rejuvenator(
    artifact: Path,
    *,
    model_name: str,
    device: torch.device,
) -> tuple[LowRankRejuvenator, dict[str, Any]]:
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    model_payload = payload["models"][model_name]
    rejuvenator = LowRankRejuvenator(
        int(model_payload["feature_count"]),
        int(model_payload["rank"]),
        use_bias=bool(model_payload["use_bias"]),
    ).to(device)
    rejuvenator.load_state_dict(model_payload["state_dict"])
    rejuvenator.eval()
    return rejuvenator, payload


def random_orientation_control(
    learned: AffineAnswerMap,
    *,
    seed: int,
) -> AffineAnswerMap:
    """Rotate every learned update by one Haar-like orthogonal matrix.

    For every input answer h, the control update has exactly the same norm as
    the learned update:

        ||(h Delta + b) Q|| = ||h Delta + b||.
    """

    device = learned.update_matrix.device
    dtype = learned.update_matrix.dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_matrix = torch.randn(
        learned.update_matrix.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    q, r = torch.linalg.qr(random_matrix)
    signs = torch.where(
        torch.diagonal(r) >= 0,
        torch.ones(r.shape[0], dtype=q.dtype),
        -torch.ones(r.shape[0], dtype=q.dtype),
    )
    q = (q * signs.unsqueeze(0)).to(device=device, dtype=dtype)
    return AffineAnswerMap(
        update_matrix=learned.update_matrix @ q,
        bias=learned.bias @ q,
    )


def apply_answer_map(state: torch.Tensor, answer_map: AnswerMap) -> torch.Tensor:
    result = state.clone()
    result[:, -1] = answer_map(result[:, -1])
    return result


def apply_shuffled_answer_map(
    state: torch.Tensor,
    answer_map: AnswerMap,
    permutation: torch.Tensor,
) -> torch.Tensor:
    mapped = answer_map(state[:, -1])
    result = state.clone()
    result[:, -1] = mapped[permutation]
    return result


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.float() - target.float()).square().mean()
    denominator = (
        target.float() - target.float().mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-12)
    return float(numerator / denominator)


def _relative_pair_rms(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = (first.float() - second.float()).square().mean().sqrt()
    scale = 0.5 * (
        first.float().square().mean().sqrt()
        + second.float().square().mean().sqrt()
    )
    return float(difference / scale.clamp_min(1e-12))


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        torch_f.cosine_similarity(
            first.float().flatten(1),
            second.float().flatten(1),
            dim=1,
        ).mean()
    )


@torch.no_grad()
def collect_commutator_batch(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> CommutatorBatch:
    reference_age = int(candidate["reference_age"])
    reference_path_before = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    if reference_age != 2:
        raise ValueError("this audit expects the D8L8 age-2 executor")
    set_seed(seed)
    path_positions = cfg.max_depth + 3 * jump
    tokens, targets, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
    )
    base_states = cache_states_with_initial(
        model,
        tokens,
        loops=cfg.max_loops,
    )
    reference_start = advance_nodes(
        successors,
        start,
        steps=cfg.max_depth - reference_path_before,
    )
    next_reference_start = advance_nodes(
        successors,
        reference_start,
        steps=jump,
    )
    shifted_start = advance_nodes(successors, start, steps=jump)
    reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        successors=successors,
        start=reference_start,
    )
    next_reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        successors=successors,
        start=next_reference_start,
    )
    shifted_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        successors=successors,
        start=shifted_start,
    )
    reference_states = cache_states_with_initial(
        model,
        reference_tokens,
        loops=reference_age + 1,
    )
    next_reference_states = cache_states_with_initial(
        model,
        next_reference_tokens,
        loops=reference_age + 1,
    )
    shifted_states = cache_states_with_initial(
        model,
        shifted_tokens,
        loops=cfg.max_loops,
    )
    endpoint_index = cfg.max_depth - 1
    h2_target_index = cfg.max_depth + jump - 1
    h3_target_index = cfg.max_depth + 2 * jump - 1
    future_h3_index = cfg.max_depth + 3 * jump - 1
    return CommutatorBatch(
        h2=reference_states[reference_age],
        h3=reference_states[reference_age + 1],
        h8=base_states[-1],
        oracle_h2_next=next_reference_states[reference_age],
        oracle_h3_next=next_reference_states[reference_age + 1],
        oracle_h8_next=shifted_states[-1],
        current_h2=targets[:, endpoint_index],
        current_h3=targets[:, h2_target_index],
        current_h8=targets[:, endpoint_index],
        target_h2=targets[:, h2_target_index],
        target_h3=targets[:, h3_target_index],
        target_h8=targets[:, h2_target_index],
        future_h2=targets[:, h3_target_index],
        future_h3=targets[:, future_h3_index],
        future_h8=targets[:, h3_target_index],
    )


def _source_spec(
    batch: CommutatorBatch,
    source_age: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if source_age == "h2":
        return (
            batch.h2,
            batch.oracle_h2_next,
            batch.current_h2,
            batch.target_h2,
            batch.future_h2,
        )
    if source_age == "h3":
        return (
            batch.h3,
            batch.oracle_h3_next,
            batch.current_h3,
            batch.target_h3,
            batch.future_h3,
        )
    if source_age == "h8":
        return (
            batch.h8,
            batch.oracle_h8_next,
            batch.current_h8,
            batch.target_h8,
            batch.future_h8,
        )
    raise ValueError(f"unknown source age: {source_age}")


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _plain_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


@torch.no_grad()
def evaluate_condition(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: CommutatorBatch,
    source_age: str,
    condition: str,
    answer_map: AnswerMap,
    shuffle_seed: int | None = None,
) -> dict[str, Any]:
    source, oracle, current, target, future = _source_spec(batch, source_age)
    if shuffle_seed is None:
        apply_j = lambda value: apply_answer_map(value, answer_map)
    else:
        generator = torch.Generator(device=source.device).manual_seed(shuffle_seed)
        permutation = torch.randperm(
            source.shape[0],
            generator=generator,
            device=source.device,
        )
        apply_j = lambda value: apply_shuffled_answer_map(
            value,
            answer_map,
            permutation,
        )
    after_f = apply_shared_stack(model, source, loop_index=cfg.max_loops)
    jf = apply_j(after_f)
    j_source = apply_j(source)
    fj = apply_shared_stack(model, j_source, loop_index=cfg.max_loops)
    jf_logits = logits_from_raw_state(model, jf)
    fj_logits = logits_from_raw_state(model, fj)
    oracle_logits = logits_from_raw_state(model, oracle)
    jf_future = apply_shared_stack(model, jf, loop_index=cfg.max_loops + 1)
    fj_future = apply_shared_stack(model, fj, loop_index=cfg.max_loops + 1)
    oracle_future = apply_shared_stack(
        model,
        oracle,
        loop_index=cfg.max_loops + 1,
    )
    jf_future_logits = logits_from_raw_state(model, jf_future)
    fj_future_logits = logits_from_raw_state(model, fj_future)
    oracle_future_logits = logits_from_raw_state(model, oracle_future)
    jf_answer = jf[:, -1]
    fj_answer = fj[:, -1]
    oracle_answer = oracle[:, -1]
    return {
        "source_age": source_age,
        "condition": condition,
        "sample_count": source.shape[0],
        "answer_commutator_relative_rms": _relative_pair_rms(
            jf_answer,
            fj_answer,
        ),
        "full_commutator_relative_rms": _relative_pair_rms(jf, fj),
        "answer_commutator_cosine": _cosine(jf_answer, fj_answer),
        "full_commutator_cosine": _cosine(jf, fj),
        "jf_answer_relative_mse_to_oracle": _relative_mse(
            jf_answer,
            oracle_answer,
        ),
        "fj_answer_relative_mse_to_oracle": _relative_mse(
            fj_answer,
            oracle_answer,
        ),
        "jf_full_relative_mse_to_oracle": _relative_mse(jf, oracle),
        "fj_full_relative_mse_to_oracle": _relative_mse(fj, oracle),
        "jf_answer_cosine_to_oracle": _cosine(jf_answer, oracle_answer),
        "fj_answer_cosine_to_oracle": _cosine(fj_answer, oracle_answer),
        **_prefixed(
            "jf_direct",
            behavior_metrics(jf_logits, target=target, endpoint=current),
        ),
        **_prefixed(
            "fj_direct",
            behavior_metrics(fj_logits, target=target, endpoint=current),
        ),
        **_prefixed(
            "oracle_direct",
            behavior_metrics(oracle_logits, target=target, endpoint=current),
        ),
        **_prefixed(
            "jf_future",
            behavior_metrics(jf_future_logits, target=future, endpoint=target),
        ),
        **_prefixed(
            "fj_future",
            behavior_metrics(fj_future_logits, target=future, endpoint=target),
        ),
        **_prefixed(
            "oracle_future",
            behavior_metrics(
                oracle_future_logits,
                target=future,
                endpoint=target,
            ),
        ),
    }


@torch.no_grad()
def evaluate_h8_six_step_reset(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: CommutatorBatch,
    condition: str,
    answer_map: AnswerMap,
    shuffle_seed: int | None = None,
) -> dict[str, Any]:
    state = batch.h8
    if shuffle_seed is None:
        apply_j = lambda value: apply_answer_map(value, answer_map)
    else:
        generator = torch.Generator(device=state.device).manual_seed(shuffle_seed)
        permutation = torch.randperm(
            state.shape[0],
            generator=generator,
            device=state.device,
        )
        apply_j = lambda value: apply_shuffled_answer_map(
            value,
            answer_map,
            permutation,
        )
    reset = state
    for _ in range(6):
        reset = apply_j(reset)
    reset_logits = logits_from_raw_state(model, reset)
    post = apply_shared_stack(model, reset, loop_index=cfg.max_loops)
    post_logits = logits_from_raw_state(model, post)
    oracle = batch.h2
    oracle_post = apply_shared_stack(model, oracle, loop_index=cfg.max_loops)
    oracle_post_logits = logits_from_raw_state(model, oracle_post)
    return {
        "condition": condition,
        "sample_count": state.shape[0],
        "reset_answer_relative_mse_to_h2": _relative_mse(
            reset[:, -1],
            oracle[:, -1],
        ),
        "reset_answer_cosine_to_h2": _cosine(
            reset[:, -1],
            oracle[:, -1],
        ),
        "reset_pre_current_accuracy": _plain_accuracy(
            reset_logits,
            batch.current_h8,
        ),
        "reset_pre_next_accuracy": behavior_metrics(
            reset_logits,
            target=batch.target_h8,
            endpoint=batch.current_h8,
        )["accuracy"],
        "reset_post_next_accuracy": behavior_metrics(
            post_logits,
            target=batch.target_h8,
            endpoint=batch.current_h8,
        )["accuracy"],
        "oracle_post_next_accuracy": behavior_metrics(
            oracle_post_logits,
            target=batch.target_h8,
            endpoint=batch.current_h8,
        )["accuracy"],
    }


def spectral_metrics(answer_map: AffineAnswerMap) -> dict[str, Any]:
    update = answer_map.update_matrix.detach().cpu().double()
    bias = answer_map.bias.detach().cpu().double()
    feature_count = update.shape[0]
    linear = torch.eye(feature_count, dtype=torch.double) + update
    eigenvalues = torch.linalg.eigvals(linear)
    singular_values = torch.linalg.svdvals(linear)
    nontrivial_eigenvalues = eigenvalues[(eigenvalues - 1).abs() >= 1e-6]
    nontrivial_eigenvalues = nontrivial_eigenvalues[
        torch.argsort(nontrivial_eigenvalues.abs(), descending=True)
    ]
    determinant_sign, log_absolute_determinant = torch.linalg.slogdet(linear)
    nonnormality = torch.linalg.matrix_norm(
        linear.T @ linear - linear @ linear.T
    ) / torch.linalg.matrix_norm(linear).square().clamp_min(1e-12)
    augmented = torch.eye(feature_count + 1, dtype=torch.double)
    augmented[:feature_count, :feature_count] = linear.T
    augmented[:feature_count, feature_count] = bias
    augmented_eigenvalues = torch.linalg.eigvals(augmented)
    return {
        "feature_count": feature_count,
        "update_rank_numerical_1e-6": int(
            (torch.linalg.svdvals(update) > 1e-6).sum()
        ),
        "update_frobenius_norm": float(torch.linalg.matrix_norm(update)),
        "bias_norm": float(bias.norm()),
        "spectral_radius": float(eigenvalues.abs().max()),
        "minimum_eigenvalue_modulus": float(eigenvalues.abs().min()),
        "eigenvalues_near_one_1e-6": int(
            ((eigenvalues - 1).abs() < 1e-6).sum()
        ),
        "eigenvalues_modulus_gt_1p01": int(
            (eigenvalues.abs() > 1.01).sum()
        ),
        "eigenvalues_modulus_lt_0p99": int(
            (eigenvalues.abs() < 0.99).sum()
        ),
        "complex_eigenvalues_imag_gt_1e-6": int(
            (eigenvalues.imag.abs() > 1e-6).sum()
        ),
        "nontrivial_eigenvalue_moduli": [
            float(value) for value in nontrivial_eigenvalues.abs()
        ],
        "determinant_sign": float(determinant_sign),
        "log_absolute_determinant": float(log_absolute_determinant),
        "maximum_singular_value": float(singular_values.max()),
        "minimum_singular_value": float(singular_values.min()),
        "condition_number": float(
            singular_values.max() / singular_values.min().clamp_min(1e-15)
        ),
        "relative_nonnormality": float(nonnormality),
        "augmented_spectral_radius": float(
            augmented_eigenvalues.abs().max()
        ),
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["source_age"]), str(row["condition"])),
            [],
        ).append(row)
    fields = [
        "answer_commutator_relative_rms",
        "full_commutator_relative_rms",
        "answer_commutator_cosine",
        "jf_answer_relative_mse_to_oracle",
        "fj_answer_relative_mse_to_oracle",
        "jf_answer_cosine_to_oracle",
        "fj_answer_cosine_to_oracle",
        "jf_direct_accuracy",
        "fj_direct_accuracy",
        "oracle_direct_accuracy",
        "jf_future_accuracy",
        "fj_future_accuracy",
        "oracle_future_accuracy",
    ]
    aggregates = []
    for (source_age, condition), selected in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "source_age": source_age,
            "condition": condition,
            "replications": len(selected),
        }
        for field in fields:
            values = torch.tensor(
                [float(row[field]) for row in selected],
                dtype=torch.double,
            )
            aggregate[f"{field}_mean"] = float(values.mean())
            aggregate[f"{field}_min"] = float(values.min())
            aggregate[f"{field}_max"] = float(values.max())
        aggregates.append(aggregate)
    return aggregates


def run_experiment(
    *,
    checkpoint: Path,
    rejuvenator_artifact: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    data_seeds: Sequence[int],
    control_seed: int,
) -> dict[str, Any]:
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("expected the D8L8 two-block checkpoint")
    if any(
        model.active_block_indices(index) != tuple(range(cfg.n_layers))
        for index in range(cfg.max_loops + 2)
    ):
        raise ValueError("F changes its active physical blocks across loops")
    bias_model, artifact_payload = load_rejuvenator(
        rejuvenator_artifact,
        model_name="reusable_bias_rank15",
        device=device,
    )
    no_bias_model, second_payload = load_rejuvenator(
        rejuvenator_artifact,
        model_name="reusable_no_bias_rank15",
        device=device,
    )
    if artifact_payload["checkpoint"] != second_payload["checkpoint"]:
        raise ValueError("rejuvenator payloads disagree about their checkpoint")
    candidate = artifact_payload["candidate"]
    bias_update, bias = rejuvenator_affine_parts(bias_model)
    no_bias_update, no_bias = rejuvenator_affine_parts(no_bias_model)
    bias_affine = AffineAnswerMap(bias_update, bias)
    no_bias_affine = AffineAnswerMap(no_bias_update, no_bias)
    random_bias = random_orientation_control(
        bias_affine,
        seed=control_seed,
    )
    random_no_bias = random_orientation_control(
        no_bias_affine,
        seed=control_seed + 1,
    )
    identity = AffineAnswerMap(
        torch.zeros_like(bias_update),
        torch.zeros_like(bias),
    )
    conditions: list[tuple[str, AffineAnswerMap, bool]] = [
        ("learned_bias_rank15", bias_affine, False),
        ("learned_no_bias_rank15", no_bias_affine, False),
        ("random_orientation_bias_rank15", random_bias, False),
        ("random_orientation_no_bias_rank15", random_no_bias, False),
        ("identity", identity, False),
        ("batch_shuffled_learned_bias_rank15", bias_affine, True),
    ]
    rows = []
    reset_rows = []
    for replication, data_seed in enumerate(data_seeds):
        batch = collect_commutator_batch(
            model=model,
            cfg=cfg,
            candidate=candidate,
            batch_size=batch_size,
            device=device,
            seed=int(data_seed),
        )
        for source_age in ("h2", "h3", "h8"):
            for condition, answer_map, shuffled in conditions:
                row = evaluate_condition(
                    model=model,
                    cfg=cfg,
                    batch=batch,
                    source_age=source_age,
                    condition=condition,
                    answer_map=answer_map,
                    shuffle_seed=(
                        int(data_seed) + 7000 if shuffled else None
                    ),
                )
                rows.append(
                    {
                        "replication": replication,
                        "data_seed": int(data_seed),
                        **row,
                    }
                )
        for condition, answer_map, shuffled in conditions:
            reset_rows.append(
                {
                    "replication": replication,
                    "data_seed": int(data_seed),
                    **evaluate_h8_six_step_reset(
                        model=model,
                        cfg=cfg,
                        batch=batch,
                        condition=condition,
                        answer_map=answer_map,
                        shuffle_seed=(
                            int(data_seed) + 7000 if shuffled else None
                        ),
                    ),
                }
            )
        print(f"completed replication {replication}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_rows(rows)
    _write_csv(out_dir / "commutator_rows.csv", rows)
    _write_csv(out_dir / "commutator_aggregate.csv", aggregate)
    _write_csv(out_dir / "h8_six_step_reset_rows.csv", reset_rows)
    spectra = {
        "learned_bias_rank15": spectral_metrics(bias_affine),
        "learned_no_bias_rank15": spectral_metrics(no_bias_affine),
        "random_orientation_bias_rank15": spectral_metrics(random_bias),
        "random_orientation_no_bias_rank15": spectral_metrics(random_no_bias),
    }
    (out_dir / "spectral_summary.json").write_text(
        json.dumps(spectra, indent=2),
        encoding="utf-8",
    )
    summary = {
        "model": "D8L8-seed1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": asdict(cfg),
        "loss_placement": "final CE at recurrent loop 8 only",
        "shared_unit": "two physical transformer blocks per recurrent cycle",
        "trained_recurrent_cycles": cfg.max_loops,
        "trained_effective_block_applications": cfg.max_loops * cfg.n_layers,
        "F_definition": "one recurrent cycle containing both shared blocks",
        "J_definition": (
            "rank-15 residual affine answer-token map "
            "R(h)=h+(hU)V+b; both bias and no-bias variants"
        ),
        "rejuvenator_artifact": str(rejuvenator_artifact.resolve()),
        "rejuvenator_artifact_sha256": _sha256(rejuvenator_artifact),
        "candidate": candidate,
        "device": str(device),
        "batch_size_per_replication": batch_size,
        "data_seeds": [int(value) for value in data_seeds],
        "control_seed": control_seed,
        "conditions": [name for name, _, _ in conditions],
        "source_ages": ["h2", "h3", "h8"],
        "primary_metrics": [
            "answer_commutator_relative_rms",
            "JF/FJ answer relative MSE to same-content natural-age oracle",
            "JF/FJ direct and one-further-cycle task accuracy",
        ],
        "aggregate": aggregate,
        "h8_six_step_reset": reset_rows,
        "spectra": spectra,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the learned D8L8 answer-state rejuvenator commutes "
            "with one recurrent cycle F."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rejuvenator-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--data-seeds",
        type=int,
        nargs="+",
        default=(2026076301, 2026076401),
    )
    parser.add_argument("--control-seed", type=int, default=2026076501)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        rejuvenator_artifact=args.rejuvenator_artifact,
        out_dir=args.out_dir,
        device=pick_device(args.device),
        batch_size=args.batch_size,
        data_seeds=args.data_seeds,
        control_seed=args.control_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
