"""Probe and causally intervene on the shared compressed directions of seven J maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import (
    MAX_AGE,
    MIN_AGE,
    sample_equivalent_word_pair,
)


AGES = tuple(range(2, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=834001)
    parser.add_argument("--probe-examples", type=int, default=1024)
    parser.add_argument("--intervention-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-probe-draws", type=int, default=8)
    parser.add_argument("--random-intervention-draws", type=int, default=3)
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--primary-rank", type=int, default=16)
    parser.add_argument(
        "--intervention-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4)
    )
    parser.add_argument("--intervention-seed-base", type=int, default=835001)
    parser.add_argument("--word-seed", type=int, default=835701)
    parser.add_argument("--intervention-back-counts", type=int, nargs="+", default=(8, 12, 16))
    parser.add_argument("--path-pairs-per-k", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def orthonormal_random(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
    return basis[:, :rank]


def consensus_basis(
    weights: dict[int, np.ndarray],
    rank: int,
    *,
    bottom: bool,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    if side not in {"input", "output"}:
        raise ValueError("side must be input or output")
    projectors = []
    for age in AGES:
        left, _, right_t = np.linalg.svd(weights[age])
        singular_basis = left if side == "input" else right_t.T
        basis = singular_basis[:, -rank:] if bottom else singular_basis[:, :rank]
        projectors.append(basis @ basis.T)
    eigenvalues, eigenvectors = np.linalg.eigh(np.mean(projectors, axis=0))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[:rank]], eigenvalues[order]


def load_bank_and_bases(
    path: Path,
    device: torch.device,
    dimension: int,
    ranks: Sequence[int],
    *,
    expected_checkpoint: Path,
    allow_relocated_checkpoint: bool,
) -> tuple[
    AgeSpecificJBank,
    dict[int, np.ndarray],
    dict[str, dict[int, np.ndarray]],
    dict[str, dict[int, np.ndarray]],
    dict[str, Any],
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    recorded = str(payload.get("checkpoint"))
    if recorded != str(expected_checkpoint) and not allow_relocated_checkpoint:
        raise ValueError("J bank/backbone mismatch")
    bank = AgeSpecificJBank(
        dimension=dimension,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=str(payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    bank = bank.frozen()
    state = payload["state_dict"]
    diagonal = state["shared_diagonal_scale"].double().numpy()
    shared = state["shared_A"].double().numpy() @ state["shared_B"].double().numpy()
    weights = {
        age: np.diag(diagonal)
        + shared
        + state[f"stage_A.{age}"].double().numpy()
        @ state[f"stage_B.{age}"].double().numpy()
        for age in AGES
    }
    bases: dict[str, dict[int, np.ndarray]] = {
        family: {}
        for family in ("bottom_input", "bottom_output", "top_input", "top_output")
    }
    consensus: dict[str, dict[int, np.ndarray]] = {
        family: {} for family in bases
    }
    for rank in ranks:
        for side in ("input", "output"):
            for band, is_bottom in (("bottom", True), ("top", False)):
                family = f"{band}_{side}"
                bases[family][rank], consensus[family][rank] = consensus_basis(
                    weights, rank, bottom=is_bottom, side=side
                )
    return bank, weights, bases, consensus, payload


def fit_ridge_regression(
    features: np.ndarray, target: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(0)
    centered = features - mean
    gram = centered.T @ centered
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    weight = np.linalg.solve(
        gram + np.eye(gram.shape[0]) * ridge * max(scale, 1e-12),
        centered.T @ target,
    )
    bias = np.asarray(target.mean(0) - mean @ weight)
    return weight, bias


def age_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    residual = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "r2": float(1 - np.square(residual).sum() / max(denominator, 1e-30)),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "rounded_accuracy": float(
            np.mean(np.clip(np.rint(prediction), 1, 8).astype(int) == target.astype(int))
        ),
    }


def classifier_accuracy(logits: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(logits.argmax(-1) == target.astype(int)))


def project_features(
    features: np.ndarray,
    *,
    basis: np.ndarray | None,
    complement: bool = False,
) -> np.ndarray:
    if basis is None:
        return features
    coordinates = features @ basis
    if complement:
        return features - coordinates @ basis.T
    return coordinates


@torch.no_grad()
def collect_probe_states(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    examples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    if examples % batch_size:
        raise ValueError("probe examples must divide by batch size")
    rng = np.random.default_rng(seed + 17)
    schedules = []
    for back_count in (2, 5, 8, 12):
        for source_age in (2, 4, 6, 8):
            left, _ = sample_equivalent_word_pair(
                rng=rng,
                back_count=back_count,
                mandatory_source_age=source_age,
            )
            schedules.append(left)
    stores: dict[str, dict[str, list[np.ndarray]]] = {
        domain: {key: [] for key in ("features", "age", "current", "split")}
        for domain in ("natural", "post_J")
    }
    positions = tuple(range(cfg.seq_len))
    for batch_index in range(examples // batch_size):
        tokens, _, successors, start = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        state = raw
        current = start
        for age in range(1, 9):
            state = model.apply_loop(state, loop_index=age - 1)
            current = advance_nodes(successors, current, steps=1)
            stores["natural"]["features"].append(state[:, -1].float().cpu().numpy())
            stores["natural"]["age"].append(np.full(batch_size, age, dtype=np.float64))
            stores["natural"]["current"].append(current.cpu().numpy())
            stores["natural"]["split"].append(
                np.full(batch_size, batch_index % 2, dtype=np.int64)
            )
        h1 = model.apply_loop(raw, loop_index=0)
        h1_current = advance_nodes(successors, start, steps=1)
        for actions in schedules:
            state = h1.clone()
            current = h1_current.clone()
            logical_age = MIN_AGE
            for action in actions:
                if action == 1:
                    state = model.apply_loop(state, loop_index=logical_age)
                    current = advance_nodes(successors, current, steps=1)
                    logical_age += 1
                else:
                    state = bank.rollback(
                        state, source_age=logical_age, positions=positions
                    )
                    logical_age -= 1
                    stores["post_J"]["features"].append(
                        state[:, -1].float().cpu().numpy()
                    )
                    stores["post_J"]["age"].append(
                        np.full(batch_size, logical_age, dtype=np.float64)
                    )
                    stores["post_J"]["current"].append(current.cpu().numpy())
                    stores["post_J"]["split"].append(
                        np.full(batch_size, batch_index % 2, dtype=np.int64)
                    )
    return {
        domain: {key: np.concatenate(values) for key, values in store.items()}
        for domain, store in stores.items()
    }


def ambient_weight(weight: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
    return weight if basis is None else basis @ weight


def energy_in_subspace(weight: np.ndarray, basis: np.ndarray) -> float:
    return float(np.square(basis.T @ weight).sum() / max(float(np.square(weight).sum()), 1e-30))


def centroid_drift_energy(
    features: np.ndarray, ages: np.ndarray, basis: np.ndarray
) -> float:
    observed_ages = np.unique(ages)
    if observed_ages.size < 2:
        raise ValueError("centroid drift requires at least two observed ages")
    centroids = np.stack(
        [features[ages == age].mean(0) for age in observed_ages]
    )
    differences = np.diff(centroids, axis=0)
    return float(
        np.square(differences @ basis).sum()
        / max(float(np.square(differences).sum()), 1e-30)
    )


def run_probes(
    *,
    data: dict[str, dict[str, np.ndarray]],
    bases: dict[str, dict[int, np.ndarray]],
    random_bases: dict[int, list[np.ndarray]],
    ridge: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {}
    full_weights: dict[tuple[str, str], np.ndarray] = {}
    for domain, values in data.items():
        features = values["features"]
        age = values["age"]
        current = values["current"]
        split = values["split"]
        feature_specs: list[tuple[str, int, int, np.ndarray | None, bool]] = [
            ("full", features.shape[1], -1, None, False)
        ]
        for family, family_bases in bases.items():
            for rank, basis in family_bases.items():
                feature_specs.append((family, rank, -1, basis, False))
                if family.startswith("bottom_"):
                    feature_specs.append(
                        (f"complement_{family}", features.shape[1] - rank, -1, basis, True)
                    )
        for rank in sorted(random_bases):
            feature_specs.extend(
                ("random", rank, draw, random_basis, False)
                for draw, random_basis in enumerate(random_bases[rank])
            )
        for family, rank, draw, basis, complement in feature_specs:
            transformed = project_features(features, basis=basis, complement=complement)
            train = split == 0
            test = split == 1
            age_weight, age_bias = fit_ridge_regression(
                transformed[train], age[train], ridge
            )
            current_target = np.eye(8, dtype=np.float64)[current]
            current_weight, current_bias = fit_ridge_regression(
                transformed[train], current_target[train], ridge
            )
            age_prediction = transformed[test] @ age_weight + age_bias
            current_logits = transformed[test] @ current_weight + current_bias
            common = {
                "domain": domain,
                "feature_family": family,
                "rank": rank,
                "random_draw": draw,
                "train_observations": int(train.sum()),
                "test_observations": int(test.sum()),
            }
            rows.append(
                {
                    **common,
                    "target": "logical_age",
                    **age_metrics(age_prediction, age[test]),
                }
            )
            rows.append(
                {
                    **common,
                    "target": "graph_current",
                    "classification_accuracy": classifier_accuracy(
                        current_logits, current[test]
                    ),
                }
            )
            if family == "full":
                full_weights[(domain, "logical_age")] = age_weight
                full_weights[(domain, "graph_current")] = current_weight
                saved[f"{domain}_full_age_weight"] = age_weight
                saved[f"{domain}_full_age_bias"] = np.asarray(age_bias)
                saved[f"{domain}_full_current_weight"] = current_weight
                saved[f"{domain}_full_current_bias"] = np.asarray(current_bias)
        for target in ("logical_age", "graph_current"):
            weight = full_weights[(domain, target)]
            for family, family_bases in bases.items():
                for rank, basis in family_bases.items():
                    alignments.append(
                        {
                            "domain": domain,
                            "target": target,
                            "subspace_family": family,
                            "rank": rank,
                            "random_draw": -1,
                            "probe_weight_energy_fraction": energy_in_subspace(
                                weight, basis
                            ),
                            "age_centroid_drift_energy_fraction": (
                                centroid_drift_energy(features, age, basis)
                                if target == "logical_age"
                                else ""
                            ),
                        }
                    )
            for rank in sorted(random_bases):
                for draw, basis in enumerate(random_bases[rank]):
                    alignments.append(
                        {
                            "domain": domain,
                            "target": target,
                            "subspace_family": "random",
                            "rank": rank,
                            "random_draw": draw,
                            "probe_weight_energy_fraction": energy_in_subspace(
                                weight, basis
                            ),
                            "age_centroid_drift_energy_fraction": (
                                centroid_drift_energy(features, age, basis)
                                if target == "logical_age"
                                else ""
                            ),
                        }
                    )
    return rows, alignments, saved


def intervention_conditions(
    *,
    bases: dict[str, dict[int, np.ndarray]],
    random_bases: dict[int, list[np.ndarray]],
    primary_rank: int,
    random_draws: int,
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [
        {"name": "baseline", "family": "baseline", "locus": "none", "rank": 0, "draw": -1, "scale": 1.0, "norm_match": False, "basis": None}
    ]
    for locus, side in (("pre_J", "input"), ("post_J", "output")):
        bottom_family = f"bottom_{side}"
        top_family = f"top_{side}"
        for rank, basis in bases[bottom_family].items():
            conditions.append(
                {"name": f"{locus}_bottom{rank}_delete", "family": bottom_family, "locus": locus, "rank": rank, "draw": -1, "scale": 0.0, "norm_match": False, "basis": basis}
            )
        for scale in (0.5, 1.5, 2.0):
            conditions.append(
                {"name": f"{locus}_bottom{primary_rank}_scale{scale:g}", "family": bottom_family, "locus": locus, "rank": primary_rank, "draw": -1, "scale": scale, "norm_match": False, "basis": bases[bottom_family][primary_rank]}
            )
        conditions.append(
            {"name": f"{locus}_top{primary_rank}_delete", "family": top_family, "locus": locus, "rank": primary_rank, "draw": -1, "scale": 0.0, "norm_match": False, "basis": bases[top_family][primary_rank]}
        )
        for rank, rank_bases in random_bases.items():
            draws = min(random_draws if rank == primary_rank else 1, len(rank_bases))
            for draw, basis in enumerate(rank_bases[:draws]):
                conditions.append(
                    {"name": f"{locus}_random{rank}_d{draw}_delete", "family": "random", "locus": locus, "rank": rank, "draw": draw, "scale": 0.0, "norm_match": False, "basis": basis}
                )
        for family, basis in (
            (bottom_family, bases[bottom_family][primary_rank]),
            ("random", random_bases[primary_rank][0]),
        ):
            for scale in (0.0, 2.0):
                short_family = "bottom" if family == bottom_family else "random"
                conditions.append(
                    {"name": f"{locus}_{short_family}{primary_rank}_scale{scale:g}_normmatch", "family": family, "locus": locus, "rank": primary_rank, "draw": 0 if family == "random" else -1, "scale": scale, "norm_match": True, "basis": basis}
                )
    return conditions


def apply_projection_intervention(
    state: torch.Tensor,
    basis: torch.Tensor,
    *,
    scale: float,
    norm_match: bool,
) -> torch.Tensor:
    before_norm = state.float().norm(dim=-1, keepdim=True)
    component = torch.matmul(torch.matmul(state, basis), basis.T)
    result = state + (scale - 1.0) * component
    if norm_match:
        after_norm = result.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result = result * (before_norm / after_norm).to(result.dtype)
    return result


@torch.no_grad()
def execute_intervened_word(
    *,
    model,
    bank: AgeSpecificJBank,
    h1: torch.Tensor,
    h1_current: torch.Tensor,
    successors: torch.Tensor,
    actions: Sequence[int],
    positions: tuple[int, ...],
    basis: torch.Tensor | None,
    locus: str,
    scale: float,
    norm_match: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = h1
    current = h1_current
    logical_age = MIN_AGE
    for action in actions:
        if action == 1:
            state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
        else:
            if basis is not None and locus == "pre_J":
                state = apply_projection_intervention(
                    state, basis, scale=scale, norm_match=norm_match
                )
            state = bank.rollback(state, source_age=logical_age, positions=positions)
            logical_age -= 1
            if basis is not None and locus == "post_J":
                state = apply_projection_intervention(
                    state, basis, scale=scale, norm_match=norm_match
                )
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("intervention executor left H1..H8")
    if logical_age != MAX_AGE:
        raise RuntimeError("intervention word did not end at H8")
    return state, current


@torch.no_grad()
def run_interventions(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    conditions: list[dict[str, Any]],
    examples: int,
    batch_size: int,
    device: torch.device,
    seed_offsets: Sequence[int],
    seed_base: int,
    word_seed: int,
    back_counts: Sequence[int],
    path_pairs_per_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if examples % batch_size:
        raise ValueError("intervention examples must divide by batch size")
    rng = np.random.default_rng(word_seed)
    words: list[tuple[int, str, tuple[int, ...]]] = []
    for back_count in back_counts:
        for pair_index in range(path_pairs_per_k):
            source_age = AGES[pair_index % len(AGES)]
            left, right = sample_equivalent_word_pair(
                rng=rng,
                back_count=back_count,
                mandatory_source_age=source_age,
            )
            words.append((back_count, f"k{back_count}_p{pair_index}_left", left))
            words.append((back_count, f"k{back_count}_p{pair_index}_right", right))
    positions = tuple(range(cfg.seq_len))
    per_path: list[dict[str, Any]] = []
    tensor_conditions = []
    for condition in conditions:
        tensor_conditions.append(
            {
                **condition,
                "basis": (
                    None
                    if condition["basis"] is None
                    else torch.as_tensor(condition["basis"], dtype=torch.float32, device=device)
                ),
            }
        )
    for seed_offset in seed_offsets:
        graph_seed = seed_base + int(seed_offset)
        set_seed(graph_seed)
        print(
            json.dumps(
                {
                    "event": "intervention_seed_start",
                    "graph_seed": graph_seed,
                    "conditions": len(tensor_conditions),
                    "words": len(words),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        accumulators: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"correct": 0.0, "count": 0.0, "ce": 0.0, "margin": 0.0, "rms": 0.0}
        )
        total_batches = examples // batch_size
        for batch_index in range(total_batches):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            h1 = model.apply_loop(raw, loop_index=0)
            h1_current = advance_nodes(successors, start, steps=1)
            for back_count, word_name, actions in words:
                for condition in tensor_conditions:
                    state, target = execute_intervened_word(
                        model=model,
                        bank=bank,
                        h1=h1,
                        h1_current=h1_current,
                        successors=successors,
                        actions=actions,
                        positions=positions,
                        basis=condition["basis"],
                        locus=str(condition["locus"]),
                        scale=float(condition["scale"]),
                        norm_match=bool(condition["norm_match"]),
                    )
                    logits = logits_from_raw_state(model, state).float()
                    predictions = logits.argmax(-1)
                    correct_logits = logits.gather(1, target[:, None]).squeeze(1)
                    distractor = logits.clone()
                    distractor.scatter_(1, target[:, None], -torch.inf)
                    margin = correct_logits - distractor.max(-1).values
                    slot = accumulators[(word_name, condition["name"])]
                    slot["correct"] += float(predictions.eq(target).sum())
                    slot["count"] += batch_size
                    slot["ce"] += float(
                        torch_functional.cross_entropy(logits, target, reduction="sum")
                    )
                    slot["margin"] += float(margin.sum())
                    slot["rms"] += float(state[:, -1].float().square().mean(-1).sqrt().sum())
            if (batch_index + 1) % max(1, total_batches // 4) == 0:
                print(
                    json.dumps(
                        {
                            "event": "intervention_seed_progress",
                            "graph_seed": graph_seed,
                            "batch": batch_index + 1,
                            "total_batches": total_batches,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        condition_lookup = {condition["name"]: condition for condition in conditions}
        word_k = {name: back_count for back_count, name, _ in words}
        for (word_name, condition_name), slot in accumulators.items():
            condition = condition_lookup[condition_name]
            count = slot["count"]
            per_path.append(
                {
                    "graph_seed": graph_seed,
                    "back_count": word_k[word_name],
                    "word": word_name,
                    "condition": condition_name,
                    "family": condition["family"],
                    "locus": condition["locus"],
                    "rank": condition["rank"],
                    "random_draw": condition["draw"],
                    "scale": condition["scale"],
                    "norm_match": condition["norm_match"],
                    "accuracy": slot["correct"] / count,
                    "ce": slot["ce"] / count,
                    "correct_margin": slot["margin"] / count,
                    "answer_rms": slot["rms"] / count,
                    "examples": int(count),
                }
            )
    baseline = {
        (int(row["graph_seed"]), int(row["back_count"]), row["word"]): row
        for row in per_path
        if row["condition"] == "baseline"
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_path:
        grouped[(row["condition"], int(row["back_count"]))].append(row)
    aggregate: list[dict[str, Any]] = []
    condition_lookup = {condition["name"]: condition for condition in conditions}
    for (condition_name, back_count), rows in sorted(grouped.items()):
        condition = condition_lookup[condition_name]
        accuracies = np.asarray([float(row["accuracy"]) for row in rows])
        margins = np.asarray([float(row["correct_margin"]) for row in rows])
        rms = np.asarray([float(row["answer_rms"]) for row in rows])
        baseline_rows = [
            baseline[(int(row["graph_seed"]), back_count, row["word"])]
            for row in rows
        ]
        accuracy_delta = accuracies - np.asarray(
            [float(row["accuracy"]) for row in baseline_rows]
        )
        margin_delta = margins - np.asarray(
            [float(row["correct_margin"]) for row in baseline_rows]
        )
        aggregate.append(
            {
                "condition": condition_name,
                "family": condition["family"],
                "locus": condition["locus"],
                "rank": condition["rank"],
                "random_draw": condition["draw"],
                "scale": condition["scale"],
                "norm_match": condition["norm_match"],
                "back_count": back_count,
                "path_seed_sides": len(rows),
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_min": float(accuracies.min()),
                "accuracy_sem": float(accuracies.std(ddof=1) / np.sqrt(len(rows))),
                "paired_accuracy_delta_mean": float(accuracy_delta.mean()),
                "paired_accuracy_delta_sem": float(
                    accuracy_delta.std(ddof=1) / np.sqrt(len(rows))
                ),
                "correct_margin_mean": float(margins.mean()),
                "paired_margin_delta_mean": float(margin_delta.mean()),
                "paired_margin_delta_sem": float(
                    margin_delta.std(ddof=1) / np.sqrt(len(rows))
                ),
                "answer_rms_mean": float(rms.mean()),
            }
        )
    return per_path, aggregate


def plot_results(
    *,
    probe_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    primary_rank: int,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=180)
    probe_specs = (
        ("natural", "full", "natural full"),
        ("natural", "bottom_input", "natural input-bottom"),
        ("natural", "complement_bottom_input", "natural input-complement"),
        ("post_J", "full", "post-J full"),
        ("post_J", "bottom_output", "post-J output-bottom"),
        ("post_J", "complement_bottom_output", "post-J output-complement"),
    )
    probe_values = []
    for domain, family, _ in probe_specs:
        row = next(
            value
            for value in probe_rows
            if value["domain"] == domain
            and value["target"] == "logical_age"
            and value["feature_family"] == family
            and (family == "full" or int(value["rank"]) in {primary_rank, 256 - primary_rank})
        )
        probe_values.append(float(row["rounded_accuracy"]))
    axes[0, 0].bar(np.arange(len(probe_specs)), probe_values)
    axes[0, 0].set_xticks(
        np.arange(len(probe_specs)),
        [label for _, _, label in probe_specs],
        rotation=25,
        ha="right",
    )
    axes[0, 0].set(title=f"Logical-age probe at rank {primary_rank}", ylabel="rounded age accuracy", ylim=(0, 1.03))

    for domain, family, label in (
        ("natural", "bottom_input", "natural vs input-bottom"),
        ("post_J", "bottom_output", "post-J vs output-bottom"),
    ):
        selected = [
            row
            for row in alignment_rows
            if row["domain"] == domain
            and row["target"] == "logical_age"
            and row["subspace_family"] == family
        ]
        axes[0, 1].plot(
            [int(row["rank"]) for row in selected],
            [float(row["age_centroid_drift_energy_fraction"]) for row in selected],
            marker="o",
            label=label,
        )
    axes[0, 1].plot((4, 8, 16, 32), np.asarray((4, 8, 16, 32)) / 256, color="black", linestyle=":", label="random expectation")
    axes[0, 1].set(title="Age-centroid drift in matched compressed subspace", xlabel="bottom-k rank", ylabel="energy fraction", ylim=(0, 1.03))
    axes[0, 1].legend(fontsize=8)

    selected_conditions = [
        "baseline",
        f"pre_J_bottom{primary_rank}_delete",
        f"pre_J_random{primary_rank}_d0_delete",
        f"post_J_bottom{primary_rank}_delete",
        f"post_J_random{primary_rank}_d0_delete",
        f"post_J_bottom{primary_rank}_scale0.5",
        f"post_J_bottom{primary_rank}_scale1.5",
        f"post_J_bottom{primary_rank}_scale2",
    ]
    for back_count in sorted({int(row["back_count"]) for row in intervention_rows}):
        lookup = {
            row["condition"]: row
            for row in intervention_rows
            if int(row["back_count"]) == back_count
        }
        names = [name for name in selected_conditions if name in lookup]
        axes[1, 0].plot(
            np.arange(len(names)),
            [float(lookup[name]["accuracy_mean"]) for name in names],
            marker="o",
            label=f"k={back_count}",
        )
    axes[1, 0].set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    axes[1, 0].set(title="Causal projection intervention after every J", ylabel="final accuracy", ylim=(0, 1.03))
    axes[1, 0].legend()

    rank_rows = [
        row
        for row in intervention_rows
        if row["family"] in {"bottom_input", "bottom_output", "random"}
        and float(row["scale"]) == 0.0
        and not bool(row["norm_match"])
        and int(row["back_count"]) == max(int(value["back_count"]) for value in intervention_rows)
        and int(row["random_draw"]) in {-1, 0}
    ]
    for locus, family, label in (
        ("pre_J", "bottom_input", "pre-J input-bottom"),
        ("pre_J", "random", "pre-J random"),
        ("post_J", "bottom_output", "post-J output-bottom"),
        ("post_J", "random", "post-J random"),
    ):
        selected = sorted(
            [
                row
                for row in rank_rows
                if row["family"] == family and row["locus"] == locus
            ],
            key=lambda row: int(row["rank"]),
        )
        axes[1, 1].plot(
            [int(row["rank"]) for row in selected],
            [float(row["paired_accuracy_delta_mean"]) for row in selected],
            marker="o",
            label=label,
        )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set(title="Deletion rank sweep at longest k", xlabel="removed rank", ylabel="paired accuracy change")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.probe_examples % args.batch_size or args.intervention_examples % args.batch_size:
        raise ValueError("example counts must divide by batch size")
    if args.probe_examples // args.batch_size < 2:
        raise ValueError(
            "probe requires at least two batches so the even/odd train-test split is non-empty"
        )
    if args.primary_rank not in args.ranks:
        raise ValueError("primary rank must be included in ranks")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    atomic_json(
        manifest_path,
        {
            "status": "running",
            "pid": os.getpid(),
            "started_unix_time": time.time(),
            "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "out_dir": str(args.out_dir),
        },
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank, weights, bases, consensus, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        tuple(args.ranks),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    rng = np.random.default_rng(args.seed + 101)
    random_bases = {
        rank: [
            orthonormal_random(cfg.d_model, rank, rng)
            for _ in range(args.random_probe_draws)
        ]
        for rank in args.ranks
    }
    data = collect_probe_states(
        model=model,
        cfg=cfg,
        bank=bank,
        examples=args.probe_examples,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    probe_rows, alignment_rows, saved = run_probes(
        data=data,
        bases=bases,
        random_bases=random_bases,
        ridge=args.ridge,
    )
    print(
        json.dumps(
            {
                "event": "probe_complete",
                "probe_rows": len(probe_rows),
                "alignment_rows": len(alignment_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    conditions = intervention_conditions(
        bases=bases,
        random_bases=random_bases,
        primary_rank=args.primary_rank,
        random_draws=args.random_intervention_draws,
    )
    intervention_per_path, intervention_aggregate = run_interventions(
        model=model,
        cfg=cfg,
        bank=bank,
        conditions=conditions,
        examples=args.intervention_examples,
        batch_size=args.batch_size,
        device=device,
        seed_offsets=tuple(args.intervention_seeds),
        seed_base=args.intervention_seed_base,
        word_seed=args.word_seed,
        back_counts=tuple(args.intervention_back_counts),
        path_pairs_per_k=args.path_pairs_per_k,
    )
    write_csv(args.out_dir / "probe_metrics.csv", probe_rows)
    write_csv(args.out_dir / "probe_alignment.csv", alignment_rows)
    write_csv(args.out_dir / "intervention_per_path.csv", intervention_per_path)
    write_csv(args.out_dir / "intervention_aggregate.csv", intervention_aggregate)
    bias = bank_payload["state_dict"]["shared_bias"].double().numpy()
    relationship_rows = []
    for rank in args.ranks:
        input_basis = bases["bottom_input"][rank]
        output_basis = bases["bottom_output"][rank]
        relationship_rows.append(
            {
                "rank": rank,
                "input_output_subspace_overlap": float(
                    np.square(input_basis.T @ output_basis).sum() / rank
                ),
                "bias_energy_in_input_bottom": energy_in_subspace(bias, input_basis),
                "bias_energy_in_output_bottom": energy_in_subspace(bias, output_basis),
                "input_consensus_mean": float(consensus["bottom_input"][rank][:rank].mean()),
                "output_consensus_mean": float(consensus["bottom_output"][rank][:rank].mean()),
                "mean_stage_bottom_singular_value": float(
                    np.mean(
                        [np.linalg.svd(weights[age], compute_uv=False)[-rank:] for age in AGES]
                    )
                ),
            }
        )
    write_csv(args.out_dir / "basis_relationships.csv", relationship_rows)
    arrays = {**saved}
    for rank in args.ranks:
        for family in bases:
            arrays[f"{family}_consensus_rank{rank}"] = bases[family][rank]
            arrays[f"{family}_consensus_eigenvalues_rank{rank}"] = consensus[family][rank]
        for draw, basis in enumerate(random_bases[rank]):
            arrays[f"random_rank{rank}_draw{draw}"] = basis
    np.savez_compressed(args.out_dir / "probe_weights_and_bases.npz", **arrays)
    plot_results(
        probe_rows=probe_rows,
        alignment_rows=alignment_rows,
        intervention_rows=intervention_aggregate,
        primary_rank=args.primary_rank,
        path=args.out_dir / "compressed_subspace_probe_intervention.png",
    )
    primary = args.primary_rank
    def find_probe(domain: str, family: str, target: str) -> dict[str, Any]:
        return next(
            row
            for row in probe_rows
            if row["domain"] == domain
            and row["feature_family"] == family
            and row["target"] == target
            and (family == "full" or int(row["rank"]) == primary)
            and int(row["random_draw"]) in {-1, 0}
        )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_recorded_checkpoint": bank_payload.get("checkpoint"),
        "ranks": list(args.ranks),
        "primary_rank": primary,
        "probe_examples": args.probe_examples,
        "intervention_examples_per_seed": args.intervention_examples,
        "intervention_graph_seeds": [args.intervention_seed_base + int(value) for value in args.intervention_seeds],
        "probe_age_histograms": {
            domain: {
                str(int(age)): int(np.sum(values["age"] == age))
                for age in np.unique(values["age"])
            }
            for domain, values in data.items()
        },
        "probe_highlights": {
            "natural_full_age_accuracy": find_probe("natural", "full", "logical_age")["rounded_accuracy"],
            "natural_input_bottom_age_accuracy": find_probe("natural", "bottom_input", "logical_age")["rounded_accuracy"],
            "natural_output_bottom_age_accuracy": find_probe("natural", "bottom_output", "logical_age")["rounded_accuracy"],
            "post_J_full_age_accuracy": find_probe("post_J", "full", "logical_age")["rounded_accuracy"],
            "post_J_output_bottom_age_accuracy": find_probe("post_J", "bottom_output", "logical_age")["rounded_accuracy"],
            "post_J_input_bottom_age_accuracy": find_probe("post_J", "bottom_input", "logical_age")["rounded_accuracy"],
            "natural_input_bottom_current_accuracy": find_probe("natural", "bottom_input", "graph_current")["classification_accuracy"],
            "post_J_output_bottom_current_accuracy": find_probe("post_J", "bottom_output", "graph_current")["classification_accuracy"],
        },
        "claim_boundary": (
            "Probe alignment is localization evidence. Selective matched-control projection effects support a causal role under this intervention family, "
            "but do not establish a unique or complete circuit; projection scaling can be off-manifold."
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    atomic_json(args.out_dir / "summary.json", result)
    atomic_json(
        manifest_path,
        {
            "status": "complete",
            "pid": os.getpid(),
            "completed_unix_time": time.time(),
            "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "out_dir": str(args.out_dir),
            "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
