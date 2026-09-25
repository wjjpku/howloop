#!/usr/bin/env python3
"""Distinguish prefix, count-reduction, and distributed Parity mechanisms."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_parity_four_phase import continue_state, natural_states
from reasoning_loop.paper_length_telomere import generate_paper_batch, load_backbone, pick_device
from reasoning_loop.parity_computation import (
    COUNTERFACTUAL_NAMES,
    RidgeProbe,
    fit_ridge,
    make_prefix_counterfactuals,
    matched_random_patch,
    orthogonalized_variable_bases,
    patch_site_subspace,
    permutation_baseline,
    predict_ridge,
    score_probe,
    serializable_probe,
)


DEFAULT_DISCOVERY_LENGTHS = (12, 16, 20, 24, 32, 40)
DEFAULT_CAUSAL_LENGTHS = (10, 14, 18, 22)
TOKEN_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.9)
CALL_FRACTIONS = (0.25, 0.5, 0.75, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--discovery-lengths", nargs="+", type=int, default=DEFAULT_DISCOVERY_LENGTHS)
    parser.add_argument("--causal-lengths", nargs="+", type=int, default=DEFAULT_CAUSAL_LENGTHS)
    parser.add_argument("--discovery-batch-size", type=int, default=512)
    parser.add_argument("--discovery-batches", type=int, default=2)
    parser.add_argument("--causal-batch-size", type=int, default=256)
    parser.add_argument("--discovery-seed", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--causal-seed", type=int, required=True)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--continuation-calls", type=int, default=2)
    parser.add_argument("--backbone-seed", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, gzip_output: bool = False) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    kwargs = {"newline": "", "encoding": "utf-8"}
    with opener(path, "wt" if gzip_output else "w", **kwargs) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_protocol(model: Any, *, allow_tiny_test_model: bool = False) -> None:
    config = model.config
    checks = {
        "one shared layer": len(model.layers) == 1,
        "causal attention": config.attention_mode == "causal",
        "token input-once": config.token_embedding_injection == "initial_only",
        "NoPE": config.position_embedding == "none",
        "position input-once": config.position_injection == "initial_only",
    }
    if not allow_tiny_test_model:
        checks.update(
            {
                "d_model=256": config.d_model == 256,
                "64 heads": config.n_heads == 64,
                "MLP width 1024": config.d_mlp == 1024,
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("input-once parity protocol mismatch: " + ", ".join(failed))


def validate_splits(
    discovery_lengths: Sequence[int],
    causal_lengths: Sequence[int],
    *,
    discovery_seed: int,
    causal_seed: int,
) -> None:
    if not discovery_lengths or not causal_lengths:
        raise ValueError("discovery and causal lengths must be nonempty")
    overlap = sorted(set(discovery_lengths) & set(causal_lengths))
    if overlap:
        raise ValueError(f"discovery/causal length overlap: {overlap}")
    if discovery_seed == causal_seed:
        raise ValueError("discovery and causal seeds must differ")


def _fixed_batch(spec: Any, *, length: int, batch_size: int, seed: int, device: torch.device) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return generate_paper_batch(
        spec,
        batch_size=batch_size,
        min_length=length,
        max_length=length,
        fixed_length=length,
        generator=generator,
    ).to(device)


def _bits_from_inputs(inputs: torch.Tensor, length: int) -> torch.Tensor:
    return inputs[:, :length, :2].argmax(dim=-1).long()


def _inputs_with_bits(inputs: torch.Tensor, bits: torch.Tensor, length: int) -> torch.Tensor:
    output = inputs.clone()
    output[:, :length, :2] = 0.0
    rows = torch.arange(bits.shape[0], device=inputs.device)[:, None]
    columns = torch.arange(length, device=inputs.device)[None, :]
    output[rows, columns, bits.to(inputs.device)] = 1.0
    return output


def _map_fraction(fraction: float, maximum_index: int) -> int:
    return int(round(fraction * maximum_index))


def _site_name(role: str, token_fraction: float, call_fraction: float) -> str:
    return f"{role}__p{token_fraction:.2f}__t{call_fraction:.2f}"


def _site_metadata(role: str, token_fraction: float, call_fraction: float) -> dict[str, Any]:
    return {
        "site": _site_name(role, token_fraction, call_fraction),
        "role": role,
        "token_fraction": token_fraction,
        "call_fraction": call_fraction,
    }


def _targets(bits: torch.Tensor, token_position: int) -> dict[str, torch.Tensor]:
    prefix = bits[:, : token_position + 1]
    prefix_count = prefix.sum(dim=1)
    total_count = bits.sum(dim=1)
    return {
        "current_bit": bits[:, token_position],
        "prefix_parity": prefix_count % 2,
        "prefix_count": prefix_count.float() / float(token_position + 1),
        "prefix_count_mod4": prefix_count % 4,
        "total_parity": total_count % 2,
        "total_count": total_count.float() / float(bits.shape[1]),
        "total_count_mod4": total_count % 4,
    }


def _probe_task(variable: str) -> str:
    if variable.endswith("count"):
        return "regression"
    if variable.endswith("mod4"):
        return "multiclass"
    return "binary"


def _atlas_candidates(smoke: bool) -> list[dict[str, Any]]:
    token_fractions = (0.25, 0.75) if smoke else TOKEN_FRACTIONS
    call_fractions = (0.5, 0.9) if smoke else CALL_FRACTIONS
    candidates = [
        _site_metadata("prefix", token_fraction, call_fraction)
        for token_fraction in token_fractions
        for call_fraction in call_fractions
    ]
    candidates.extend(
        _site_metadata("answer", 1.0, call_fraction)
        for call_fraction in call_fractions
    )
    return candidates


@torch.inference_mode()
def build_atlas(
    model: Any,
    spec: Any,
    *,
    lengths: Sequence[int],
    batch_size: int,
    batches: int,
    seed: int,
    smoke: bool,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, RidgeProbe]], dict[str, dict[str, torch.Tensor]]]:
    candidates = _atlas_candidates(smoke)
    collected: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    train_lengths = set(lengths[::2])
    validation_lengths = set(lengths[1::2])
    if not validation_lengths:
        validation_lengths = {lengths[-1]}
        train_lengths.discard(lengths[-1])
    if not train_lengths:
        raise ValueError("atlas requires at least one train and one validation length")
    for length_index, length in enumerate(lengths):
        for batch_index in range(batches):
            batch = _fixed_batch(
                spec,
                length=length,
                batch_size=batch_size,
                seed=seed + 100_003 * length_index + batch_index,
                device=device,
            )
            bits = _bits_from_inputs(batch.inputs, length)
            states = natural_states(
                model, batch.inputs, steps=length, controller=None, controller_anchor=None
            )
            split = "train" if length in train_lengths else "validation"
            for site in candidates:
                call = max(1, min(length, _map_fraction(site["call_fraction"], length)))
                token = length if site["role"] == "answer" else _map_fraction(site["token_fraction"], length - 1)
                variables = (
                    ("total_parity", "total_count", "total_count_mod4")
                    if site["role"] == "answer"
                    else ("current_bit", "prefix_parity", "prefix_count", "prefix_count_mod4")
                )
                labels = _targets(bits, min(token, length - 1))
                collected[site["site"]][split]["features"].append(states[call - 1][:, token].float().cpu())
                for variable in variables:
                    collected[site["site"]][split][variable].append(labels[variable].cpu())

    atlas_rows: list[dict[str, Any]] = []
    probes: dict[str, dict[str, RidgeProbe]] = {}
    bases: dict[str, dict[str, torch.Tensor]] = {}
    candidates_by_name = {site["site"]: site for site in candidates}
    for name, splits in collected.items():
        probes[name] = {}
        train_x = torch.cat(splits["train"]["features"])
        validation_x = torch.cat(splits["validation"]["features"])
        variables = [key for key in splits["train"] if key != "features"]
        decoder_rows: dict[str, torch.Tensor] = {}
        for variable in variables:
            train_y = torch.cat(splits["train"][variable])
            validation_y = torch.cat(splits["validation"][variable])
            task = _probe_task(variable)
            probe = fit_ridge(train_x, train_y, validation_x, validation_y, task=task)
            probes[name][variable] = probe
            score = score_probe(validation_y, predict_ridge(probe, validation_x), task=task)
            baseline = permutation_baseline(
                train_x,
                train_y,
                validation_x,
                validation_y,
                task=task,
                seeds=(seed + 7001, seed + 9001),
            )
            atlas_rows.append(
                {
                    **candidates_by_name[name],
                    "variable": variable,
                    "probe_task": task,
                    "validation_score": score,
                    "permutation_baseline": baseline,
                    "score_above_permutation": score - baseline,
                    "penalty": probe.penalty,
                    "train_lengths": ";".join(map(str, sorted(train_lengths))),
                    "validation_lengths": ";".join(map(str, sorted(validation_lengths))),
                    "train_examples": train_x.shape[0],
                    "validation_examples": validation_x.shape[0],
                }
            )
            decoder_rows[variable.replace("prefix_", "").replace("total_", "")] = probe.decoder_rows()
        bases[name] = orthogonalized_variable_bases(decoder_rows)
    return atlas_rows, probes, bases


def select_sites(
    atlas_rows: Sequence[Mapping[str, Any]], *, smoke: bool
) -> dict[str, Any]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {
        (str(row["site"]), str(row["variable"])): row for row in atlas_rows
    }
    prefix_candidates = []
    answer_candidates = []
    for site in sorted({str(row["site"]) for row in atlas_rows}):
        role = str(next(row["role"] for row in atlas_rows if row["site"] == site))
        if role == "prefix":
            parity = lookup[(site, "prefix_parity")]
            count = lookup[(site, "prefix_count")]
            score = float(parity["validation_score"]) - max(
                float(parity["permutation_baseline"]),
                max(0.0, float(count["validation_score"])),
            )
            prefix_candidates.append((score, float(parity["validation_score"]), site))
        else:
            count = lookup[(site, "total_count")]
            mod4 = lookup[(site, "total_count_mod4")]
            score = max(
                float(count["score_above_permutation"]),
                float(mod4["score_above_permutation"]),
            )
            answer_candidates.append((score, float(mod4["validation_score"]), site))
    limit = 1 if smoke else 3
    prefix_sites = [site for _, _, site in sorted(prefix_candidates, reverse=True)[:limit]]
    answer_sites = [site for _, _, site in sorted(answer_candidates, reverse=True)[:limit]]
    return {
        "prefix_sites": prefix_sites,
        "answer_sites": answer_sites,
        "prefix_ranking": [list(item) for item in sorted(prefix_candidates, reverse=True)],
        "answer_ranking": [list(item) for item in sorted(answer_candidates, reverse=True)],
        "selection_uses_causal_data": False,
    }


def _oriented_margins(model: Any, state: torch.Tensor, labels: torch.Tensor, answer: int) -> torch.Tensor:
    logits = model.decode(state)[:, answer]
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, labels] - logits[rows, 1 - labels]


def _prediction(model: Any, state: torch.Tensor, answer: int) -> torch.Tensor:
    return model.decode(state)[:, answer].argmax(dim=-1)


def _trajectory_after_patch(
    model: Any,
    patched: torch.Tensor,
    inputs: torch.Tensor,
    *,
    start_call: int,
    requested_calls: Sequence[int],
) -> dict[int, torch.Tensor]:
    requested = sorted(set(int(value) for value in requested_calls))
    if requested[0] < start_call:
        raise ValueError("requested evaluation precedes intervention")
    state = patched
    output: dict[int, torch.Tensor] = {}
    if start_call in requested:
        output[start_call] = state
    for call in range(start_call + 1, requested[-1] + 1):
        state = continue_state(
            model,
            state,
            inputs,
            start_step=call - 1,
            calls=1,
            controller=None,
            controller_anchor=None,
        )
        if call in requested:
            output[call] = state
    return output


def _site_from_name(name: str) -> dict[str, float | str]:
    role, position, call = name.split("__")
    return {
        "role": role,
        "token_fraction": float(position[1:]),
        "call_fraction": float(call[1:]),
    }


def _condition_states(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    bases: Mapping[str, torch.Tensor],
    *,
    variable: str,
    token: int,
    random_controls: int,
    seed: int,
) -> list[tuple[str, torch.Tensor]]:
    basis = bases[variable]
    protected_names = [name for name in ("current_bit", "count", "parity") if name != variable]
    protected_parts = [bases[name] for name in protected_names if name in bases and bases[name].numel()]
    protected = torch.cat(protected_parts, dim=1) if protected_parts else None
    # Tiny random models can exhaust their eight-dimensional residual space
    # after orthogonalizing all diagnostic variables. Formal d=256 runs keep
    # the full protected alternative-variable span.
    if receiver.shape[-1] < 32:
        protected = None
    conditions = [(f"{variable}_subspace", patch_site_subspace(receiver, donor, basis, token_position=token))]
    for index in range(random_controls):
        conditions.append(
            (
                f"{variable}_random_{index}",
                matched_random_patch(
                    receiver,
                    donor,
                    basis,
                    protected_basis=protected,
                    token_position=token,
                    seed=seed + 104729 * (index + 1),
                ),
            )
        )
    full = receiver.clone()
    full[:, token] = donor[:, token]
    conditions.append(("full_token", full))
    return conditions


@torch.inference_mode()
def run_causal_interchange(
    model: Any,
    spec: Any,
    *,
    lengths: Sequence[int],
    selection: Mapping[str, Any],
    bases: Mapping[str, Mapping[str, torch.Tensor]],
    batch_size: int,
    seed: int,
    random_controls: int,
    continuation_calls: int,
    device: torch.device,
    backbone_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    sites = [(site, "prefix") for site in selection["prefix_sites"]] + [
        (site, "answer") for site in selection["answer_sites"]
    ]
    for length_index, length in enumerate(lengths):
        batch = _fixed_batch(
            spec,
            length=length,
            batch_size=batch_size,
            seed=seed + 1_000_003 * length_index,
            device=device,
        )
        receiver_bits = _bits_from_inputs(batch.inputs, length)
        receiver_labels = receiver_bits.sum(dim=1) % 2
        receiver_states = natural_states(
            model, batch.inputs, steps=length, controller=None, controller_anchor=None
        )
        for site_index, (site_name, role) in enumerate(sites):
            metadata = _site_from_name(site_name)
            call = max(1, min(length - 1, _map_fraction(float(metadata["call_fraction"]), length)))
            token = length if role == "answer" else _map_fraction(float(metadata["token_fraction"]), length - 1)
            counterfactuals = make_prefix_counterfactuals(
                receiver_bits,
                prefix_end=max(1, token if role == "prefix" else length - 1),
                seed=seed + 1009 * site_index + length,
            )
            receiver_state = receiver_states[call - 1]
            for family_index, family in enumerate(COUNTERFACTUAL_NAMES):
                valid = counterfactuals.valid[:, family_index]
                if not bool(valid.any()):
                    continue
                donor_bits_all = counterfactuals.variant(family)
                donor_inputs_all = _inputs_with_bits(batch.inputs, donor_bits_all, length)
                donor_states_all = natural_states(
                    model, donor_inputs_all, steps=length, controller=None, controller_anchor=None
                )
                indices = torch.nonzero(valid, as_tuple=False).flatten()
                donor_bits = donor_bits_all[indices]
                receiver_inputs = batch.inputs[indices]
                receiver_selected = receiver_state[indices]
                donor_selected = donor_states_all[call - 1][indices]
                receiver_label = receiver_labels[indices]
                donor_label = donor_bits.sum(dim=1) % 2
                prefix_stop = token + 1 if role == "prefix" else length
                receiver_prefix = receiver_bits[indices, :prefix_stop].sum(dim=1) % 2
                donor_prefix = donor_bits[:, :prefix_stop].sum(dim=1) % 2
                receiver_suffix = receiver_bits[indices, prefix_stop:].sum(dim=1) % 2
                predicted_label = donor_prefix ^ receiver_suffix
                for local_index, original_index in enumerate(indices.tolist()):
                    pair_rows.append(
                        {
                            "backbone_seed": backbone_seed,
                            "data_seed": seed,
                            "length": length,
                            "site": site_name,
                            "role": role,
                            "call": call,
                            "token_position": token,
                            "example": original_index,
                            "family": family,
                            "receiver_prefix_parity": int(receiver_prefix[local_index].cpu()),
                            "donor_prefix_parity": int(donor_prefix[local_index].cpu()),
                            "receiver_suffix_parity": int(receiver_suffix[local_index].cpu()),
                            "receiver_total_count": int(receiver_bits[original_index].sum().cpu()),
                            "donor_total_count": int(donor_bits[local_index].sum().cpu()),
                            "predicted_final_label": int(predicted_label[local_index].cpu()),
                            "changed_index_0": int(counterfactuals.changed_indices[original_index, family_index, 0].cpu()),
                            "changed_index_1": int(counterfactuals.changed_indices[original_index, family_index, 1].cpu()),
                        }
                    )
                variables = ("parity", "count_given_parity")
                for variable in variables:
                    conditions = _condition_states(
                        receiver_selected,
                        donor_selected,
                        bases[site_name],
                        variable=variable,
                        token=token,
                        random_controls=random_controls,
                        seed=seed + 10_007 * site_index + 101 * family_index,
                    )
                    conditions.append(("untouched_receiver", receiver_selected))
                    conditions.append(("natural_donor", donor_selected))
                    evaluation_calls = sorted(
                        {
                            call,
                            min(length, call + 1),
                            min(length, call + continuation_calls),
                            length,
                        }
                    )
                    for condition, patched in conditions:
                        continuation_inputs = (
                            donor_inputs_all[indices]
                            if condition == "natural_donor"
                            else receiver_inputs
                        )
                        trajectory = _trajectory_after_patch(
                            model,
                            patched,
                            continuation_inputs,
                            start_call=call,
                            requested_calls=evaluation_calls,
                        )
                        for evaluation_call, state in trajectory.items():
                            prediction = _prediction(model, state, length)
                            predicted_margin = _oriented_margins(
                                model, state, predicted_label, length
                            )
                            receiver_margin = _oriented_margins(
                                model, state, receiver_label, length
                            )
                            for local_index, original_index in enumerate(indices.tolist()):
                                rows.append(
                                    {
                                        "backbone_seed": backbone_seed,
                                        "data_seed": seed,
                                        "length": length,
                                        "site": site_name,
                                        "role": role,
                                        "call": call,
                                        "token_position": token,
                                        "state_site": "post_call_final_normalized",
                                        "family": family,
                                        "variable": variable,
                                        "condition": condition,
                                        "example": original_index,
                                        "evaluation_call": evaluation_call,
                                        "continuation_calls": evaluation_call - call,
                                        "receiver_label": int(receiver_label[local_index].cpu()),
                                        "donor_label": int(donor_label[local_index].cpu()),
                                        "predicted_final_label": int(predicted_label[local_index].cpu()),
                                        "prediction": int(prediction[local_index].cpu()),
                                        "predicted_label_match": float(
                                            prediction[local_index] == predicted_label[local_index]
                                        ),
                                        "receiver_label_match": float(
                                            prediction[local_index] == receiver_label[local_index]
                                        ),
                                        "predicted_oriented_margin": float(predicted_margin[local_index].cpu()),
                                        "receiver_oriented_margin": float(receiver_margin[local_index].cpu()),
                                    }
                                )
    return rows, pair_rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "backbone_seed",
        "data_seed",
        "length",
        "site",
        "role",
        "family",
        "variable",
        "condition",
        "evaluation_call",
        "continuation_calls",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        summary = dict(zip(keys, key))
        summary.update(
            {
                "examples": len(selected),
                "predicted_label_match": float(np.mean([float(row["predicted_label_match"]) for row in selected])),
                "receiver_label_match": float(np.mean([float(row["receiver_label_match"]) for row in selected])),
                "predicted_oriented_margin": float(np.mean([float(row["predicted_oriented_margin"]) for row in selected])),
                "receiver_oriented_margin": float(np.mean([float(row["receiver_oriented_margin"]) for row in selected])),
                "normalized_recovery": float(
                    np.nanmean([float(row.get("normalized_recovery", float("nan"))) for row in selected])
                ),
            }
        )
        output.append(summary)
    return output


def _attach_normalized_recovery(rows: list[dict[str, Any]]) -> None:
    """Attach donor-target recovery within exact example/intervention cells."""

    reference_keys = (
        "backbone_seed",
        "data_seed",
        "length",
        "site",
        "role",
        "family",
        "variable",
        "example",
        "evaluation_call",
    )
    references: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in {"untouched_receiver", "natural_donor"}:
            key = tuple(row[name] for name in reference_keys)
            references[key][str(row["condition"])] = float(row["predicted_oriented_margin"])
    for row in rows:
        key = tuple(row[name] for name in reference_keys)
        reference = references.get(key, {})
        receiver = reference.get("untouched_receiver")
        target = reference.get("natural_donor")
        if receiver is None or target is None or abs(target - receiver) < 1e-8:
            row["normalized_recovery"] = float("nan")
        else:
            row["normalized_recovery"] = (
                float(row["predicted_oriented_margin"]) - receiver
            ) / (target - receiver)


@torch.inference_mode()
def select_restoration_order(
    model: Any,
    spec: Any,
    *,
    length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    """Freeze a token ordering on a discovery length and independent seed."""

    batch = _fixed_batch(spec, length=length, batch_size=batch_size, seed=seed, device=device)
    clean_bits = _bits_from_inputs(batch.inputs, length)
    corrupt_bits = clean_bits.clone()
    corrupt_bits[:, 0] = 1 - corrupt_bits[:, 0]
    corrupt_inputs = _inputs_with_bits(batch.inputs, corrupt_bits, length)
    clean_labels = clean_bits.sum(dim=1) % 2
    call = max(1, length // 2)
    clean_states = natural_states(model, batch.inputs, steps=length, controller=None, controller_anchor=None)
    corrupt_states = natural_states(model, corrupt_inputs, steps=length, controller=None, controller_anchor=None)
    clean_margin = _oriented_margins(model, clean_states[-1], clean_labels, length)
    corrupt_margin = _oriented_margins(model, corrupt_states[-1], clean_labels, length)
    scores = []
    for token in range(length + 1):
        patched = corrupt_states[call - 1].clone()
        patched[:, token] = clean_states[call - 1][:, token]
        final = continue_state(
            model,
            patched,
            corrupt_inputs,
            start_step=call,
            calls=length - call,
            controller=None,
            controller_anchor=None,
        )
        patched_margin = _oriented_margins(model, final, clean_labels, length)
        denominator = clean_margin - corrupt_margin
        recovery = (patched_margin - corrupt_margin) / denominator.where(
            denominator.abs() > 1e-8,
            torch.full_like(denominator, float("nan")),
        )
        scores.append((float(torch.nanmean(recovery).cpu()), token / length))
    return [fraction for _, fraction in sorted(scores, reverse=True)[:5]]


@torch.inference_mode()
def restoration_scan(
    model: Any,
    spec: Any,
    *,
    lengths: Sequence[int],
    batch_size: int,
    seed: int,
    smoke: bool,
    device: torch.device,
    backbone_seed: int,
    ordered_token_fractions: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    first_length = lengths[0]
    for length_index, length in enumerate(lengths):
        batch = _fixed_batch(
            spec,
            length=length,
            batch_size=min(batch_size, 32 if smoke else batch_size),
            seed=seed + 300_007 * length_index,
            device=device,
        )
        clean_bits = _bits_from_inputs(batch.inputs, length)
        corrupt_bits = clean_bits.clone()
        corrupt_bits[:, 0] = 1 - corrupt_bits[:, 0]
        corrupt_inputs = _inputs_with_bits(batch.inputs, corrupt_bits, length)
        clean_labels = clean_bits.sum(dim=1) % 2
        call_fractions = (0.5,) if smoke else CALL_FRACTIONS
        clean_states = natural_states(model, batch.inputs, steps=length, controller=None, controller_anchor=None)
        corrupt_states = natural_states(model, corrupt_inputs, steps=length, controller=None, controller_anchor=None)
        for call_fraction in call_fractions:
            call = max(1, min(length - 1, _map_fraction(call_fraction, length)))
            directions = (
                (
                    "corrupt_to_clean",
                    corrupt_states,
                    clean_states,
                    corrupt_inputs,
                    clean_labels,
                ),
                (
                    "clean_to_corrupt",
                    clean_states,
                    corrupt_states,
                    batch.inputs,
                    1 - clean_labels,
                ),
            )
            for direction, receiver_states, donor_states, receiver_inputs, target_labels in directions:
                target_final = _oriented_margins(model, donor_states[-1], target_labels, length)
                receiver_final = _oriented_margins(model, receiver_states[-1], target_labels, length)
                for token in range(length + 1):
                    patched = receiver_states[call - 1].clone()
                    patched[:, token] = donor_states[call - 1][:, token]
                    final = continue_state(
                        model,
                        patched,
                        receiver_inputs,
                        start_step=call,
                        calls=length - call,
                        controller=None,
                        controller_anchor=None,
                    )
                    patched_margin = _oriented_margins(model, final, target_labels, length)
                    denominator = target_final - receiver_final
                    recovery = (patched_margin - receiver_final) / denominator.where(
                        denominator.abs() > 1e-8,
                        torch.full_like(denominator, float("nan")),
                    )
                    rows.append(
                        {
                            "backbone_seed": backbone_seed,
                            "data_seed": seed,
                            "length": length,
                            "direction": direction,
                            "flipped_bit_position": 0,
                            "call": call,
                            "call_fraction": call_fraction,
                            "token_position": token,
                            "token_fraction": token / max(1, length),
                            "token_role": "source" if token == 0 else "answer" if token == length else "intermediate",
                            "mean_normalized_recovery": float(torch.nanmean(recovery).cpu()),
                            "mean_target_margin": float(target_final.mean().cpu()),
                            "mean_receiver_margin": float(receiver_final.mean().cpu()),
                            "examples": batch.inputs.shape[0],
                        }
                    )
    joint_rows: list[dict[str, Any]] = []
    # The cumulative experiment is executed on the first held-out length, using
    # the separately generated causal batch and discovery-frozen token ordering.
    length = first_length
    batch = _fixed_batch(spec, length=length, batch_size=min(batch_size, 32 if smoke else batch_size), seed=seed + 910_001, device=device)
    clean_bits = _bits_from_inputs(batch.inputs, length)
    corrupt_bits = clean_bits.clone()
    corrupt_bits[:, 0] = 1 - corrupt_bits[:, 0]
    corrupt_inputs = _inputs_with_bits(batch.inputs, corrupt_bits, length)
    clean_labels = clean_bits.sum(dim=1) % 2
    call = max(1, length // 2)
    clean_states = natural_states(model, batch.inputs, steps=length, controller=None, controller_anchor=None)
    corrupt_states = natural_states(model, corrupt_inputs, steps=length, controller=None, controller_anchor=None)
    clean_margin = _oriented_margins(model, clean_states[-1], clean_labels, length)
    corrupt_margin = _oriented_margins(model, corrupt_states[-1], clean_labels, length)
    ordered = [
        max(0, min(length, int(round(fraction * length))))
        for fraction in ordered_token_fractions
    ]
    for set_size in range(1, len(ordered) + 1):
        selected = ordered[:set_size]
        controls = {
            "ordered": selected,
            "position_reversed": [length - token for token in selected],
        }
        generator = torch.Generator(device="cpu").manual_seed(seed + set_size)
        controls["random_tokens"] = torch.randperm(length + 1, generator=generator)[:set_size].tolist()
        for condition, tokens in controls.items():
            patched = corrupt_states[call - 1].clone()
            for token in sorted(set(max(0, min(length, int(value))) for value in tokens)):
                patched[:, token] = clean_states[call - 1][:, token]
            final = continue_state(model, patched, corrupt_inputs, start_step=call, calls=length - call, controller=None, controller_anchor=None)
            patched_margin = _oriented_margins(model, final, clean_labels, length)
            recovery = (patched_margin - corrupt_margin) / (clean_margin - corrupt_margin).where(
                (clean_margin - corrupt_margin).abs() > 1e-8,
                torch.full_like(clean_margin, float("nan")),
            )
            joint_rows.append(
                {
                    "backbone_seed": backbone_seed,
                    "data_seed": seed,
                    "length": length,
                    "call": call,
                    "condition": condition,
                    "set_size": set_size,
                    "tokens": ";".join(map(str, tokens)),
                    "mean_normalized_recovery": float(torch.nanmean(recovery).cpu()),
                    "examples": batch.inputs.shape[0],
                }
            )
    return rows, joint_rows


def _plot_artifacts(
    out_dir: Path,
    atlas_rows: Sequence[Mapping[str, Any]],
    causal_summary: Sequence[Mapping[str, Any]],
    restoration_rows: Sequence[Mapping[str, Any]],
) -> None:
    prefix = [row for row in atlas_rows if row["variable"] == "prefix_parity"]
    answer = [row for row in atlas_rows if row["variable"] in {"total_count", "total_count_mod4"}]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(len(prefix)), [float(row["validation_score"]) for row in prefix])
    axes[0].set_title("Prefix parity probe")
    axes[0].set_ylim(-0.1, 1.05)
    axes[1].bar(range(len(answer)), [float(row["validation_score"]) for row in answer])
    axes[1].set_title("Answer count probes")
    fig.tight_layout()
    fig.savefig(out_dir / "variable_atlas.png", dpi=180)
    plt.close(fig)

    selected = [
        row
        for row in causal_summary
        if row["condition"] in {"parity_subspace", "count_given_parity_subspace", "full_token"}
        and int(row["evaluation_call"]) >= int(row["length"])
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{row['role']}:{row['variable']}:{row['family']}" for row in selected]
    values = [float(row["predicted_label_match"]) for row in selected]
    ax.bar(range(len(values)), values)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(values)), labels, rotation=90, fontsize=6)
    ax.set_ylabel("counterfactual label match")
    fig.tight_layout()
    fig.savefig(out_dir / "causal_route_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    by_call: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in restoration_rows:
        by_call[int(row["call"])].append(row)
    for call, rows in sorted(by_call.items()):
        ax.plot(
            [float(row["token_fraction"]) for row in rows],
            [float(row["mean_normalized_recovery"]) for row in rows],
            marker="o",
            label=f"call {call}",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("token position / length")
    ax.set_ylabel("clean restoration")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "restoration_path.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    allow_tiny = bool(getattr(args, "allow_tiny_test_model", False))
    validate_splits(
        args.discovery_lengths,
        args.causal_lengths,
        discovery_seed=args.discovery_seed,
        causal_seed=args.causal_seed,
    )
    model, spec, payload = load_backbone(
        Path(args.checkpoint), device=device, paper_mode=not allow_tiny
    )
    validate_protocol(model, allow_tiny_test_model=allow_tiny)
    if spec.name != "parity":
        raise ValueError("Stage B runner accepts only parity checkpoints")
    source_paths = [Path(__file__), Path(__file__).with_name("parity_computation.py")]
    manifest = {
        "status": "running",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "backbone_seed": int(getattr(args, "backbone_seed", payload.get("seed", -1))),
        "discovery_lengths": list(args.discovery_lengths),
        "causal_lengths": list(args.causal_lengths),
        "discovery_seed": args.discovery_seed,
        "selection_seed": args.selection_seed,
        "causal_seed": args.causal_seed,
        "state_site": "post_call_final_normalized",
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "device": str(device),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    try:
        atlas_rows, probes, bases = build_atlas(
            model,
            spec,
            lengths=args.discovery_lengths,
            batch_size=args.discovery_batch_size,
            batches=args.discovery_batches,
            seed=args.discovery_seed,
            smoke=args.smoke,
            device=device,
        )
        selection = select_sites(atlas_rows, smoke=args.smoke)
        selection.update(
            {
                "selection_seed": args.selection_seed,
                "discovery_lengths": list(args.discovery_lengths),
                "causal_lengths_read": False,
            }
        )
        selection["ordered_restoration_token_fractions"] = select_restoration_order(
            model,
            spec,
            length=args.discovery_lengths[-1],
            batch_size=min(args.discovery_batch_size, 32 if args.smoke else 128),
            seed=args.selection_seed,
            device=device,
        )
        write_csv(out_dir / "atlas.csv", atlas_rows)
        write_csv(out_dir / "probe_summary.csv", atlas_rows)
        torch.save(
            {
                site: {variable: serializable_probe(probe) for variable, probe in site_probes.items()}
                for site, site_probes in probes.items()
            },
            out_dir / "probe_models.pt",
        )
        write_json(out_dir / "selection.json", selection)
        causal_rows, pair_rows = run_causal_interchange(
            model,
            spec,
            lengths=args.causal_lengths,
            selection=selection,
            bases=bases,
            batch_size=args.causal_batch_size,
            seed=args.causal_seed,
            random_controls=args.random_controls,
            continuation_calls=args.continuation_calls,
            device=device,
            backbone_seed=manifest["backbone_seed"],
        )
        _attach_normalized_recovery(causal_rows)
        causal_summary = _summarize_rows(causal_rows)
        restoration_rows, joint_rows = restoration_scan(
            model,
            spec,
            lengths=args.causal_lengths,
            batch_size=args.causal_batch_size,
            seed=args.causal_seed + 5_000_003,
            smoke=args.smoke,
            device=device,
            backbone_seed=manifest["backbone_seed"],
            ordered_token_fractions=selection[
                "ordered_restoration_token_fractions"
            ],
        )
        write_csv(out_dir / "counterfactual_pairs.csv.gz", pair_rows, gzip_output=True)
        write_csv(out_dir / "causal_per_example.csv.gz", causal_rows, gzip_output=True)
        write_csv(out_dir / "causal_summary.csv", causal_summary)
        write_csv(out_dir / "restoration_scan.csv", restoration_rows)
        write_csv(out_dir / "joint_patch_summary.csv", joint_rows)
        _plot_artifacts(out_dir, atlas_rows, causal_summary, restoration_rows)
        summary = {
            "status": "complete",
            "backbone_seed": manifest["backbone_seed"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "atlas_rows": len(atlas_rows),
            "causal_per_example_rows": len(causal_rows),
            "causal_summary_rows": len(causal_summary),
            "restoration_rows": len(restoration_rows),
            "joint_rows": len(joint_rows),
            "prefix_sites": selection["prefix_sites"],
            "answer_sites": selection["answer_sites"],
            "elapsed_seconds": time.time() - start,
            "scientific_decision": "pending_multiseed_aggregation",
        }
        write_json(out_dir / "summary.json", summary)
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": summary["elapsed_seconds"],
                "artifacts": sorted(path.name for path in out_dir.iterdir()),
            }
        )
        write_json(out_dir / "run_manifest.json", manifest)
        return summary
    except Exception:
        manifest.update({"status": "failed", "traceback": traceback.format_exc()})
        write_json(out_dir / "run_manifest.json", manifest)
        raise


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
