from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_global_rejuvenator import (
    PairDataset,
    _masked_accuracy,
    _plain_accuracy,
    age_path_position,
    collect_evaluation_batch,
    collect_pair_dataset,
    predict_pair_dataset,
    select_and_refit_map,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
)
from reasoning_loop.graph_path_rejuvenator_commutator import (
    _cosine,
    _relative_mse,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


TRAIN_SOURCE_AGES = tuple(range(3, 9))
NORMALIZATION_NAMES = ("unit_layernorm", "block0_ln1", "ln_final")


@dataclass(frozen=True)
class NormalizationSpec:
    name: str
    eps: float
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None

    def to(self, device: torch.device) -> "NormalizationSpec":
        return NormalizationSpec(
            name=self.name,
            eps=self.eps,
            weight=None if self.weight is None else self.weight.to(device),
            bias=None if self.bias is None else self.bias.to(device),
        )

    def coordinates(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=-1, keepdim=True)
        variance = (value - mean).square().mean(dim=-1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight
        if self.bias is not None:
            normalized = normalized + self.bias
        return normalized

    def invert_with_reference_stats(
        self,
        coordinates: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        normalized = coordinates
        if self.bias is not None:
            normalized = normalized - self.bias
        if self.weight is not None:
            if bool((self.weight.abs() < 1e-7).any()):
                raise ValueError(f"{self.name} has a near-zero LayerNorm weight")
            normalized = normalized / self.weight
        mean = reference.mean(dim=-1, keepdim=True)
        variance = (reference - mean).square().mean(dim=-1, keepdim=True)
        return normalized * torch.sqrt(variance + self.eps) + mean


def normalization_specs(
    model: LoopedGraphPathTransformer,
) -> dict[str, NormalizationSpec]:
    d_model = model.cfg.d_model

    def learned(name: str, module: torch.nn.Module) -> NormalizationSpec:
        if not isinstance(module, torch.nn.LayerNorm):
            raise TypeError(f"{name} is not torch.nn.LayerNorm")
        if tuple(module.normalized_shape) != (d_model,):
            raise ValueError(f"unexpected {name} normalized shape")
        return NormalizationSpec(
            name=name,
            eps=float(module.eps),
            weight=module.weight.detach().float().cpu().clone(),
            bias=module.bias.detach().float().cpu().clone(),
        )

    return {
        "unit_layernorm": NormalizationSpec(
            name="unit_layernorm",
            eps=float(model.ln_final.eps),
        ),
        "block0_ln1": learned("block0_ln1", model.blocks[0].ln_1),
        "ln_final": learned("ln_final", model.ln_final),
    }


def transform_pair_dataset(
    data: PairDataset,
    spec: NormalizationSpec,
) -> PairDataset:
    return PairDataset(
        source=spec.coordinates(data.source),
        target=spec.coordinates(data.target),
        source_age=data.source_age,
    )


def spectral_summary(matrix: torch.Tensor) -> dict[str, Any]:
    matrix64 = matrix.detach().cpu().double()
    eigenvalues = torch.linalg.eigvals(matrix64)
    real = eigenvalues.real.numpy()
    imaginary = eigenvalues.imag.numpy()
    modulus = eigenvalues.abs().numpy()
    sign, logabsdet = torch.linalg.slogdet(matrix64)
    singular_values = torch.linalg.svdvals(matrix64)

    limit = max(1.05, float(modulus.max()) * 1.05)
    grid = np.linspace(-limit, limit, 241)
    xx, yy = np.meshgrid(grid, grid)
    try:
        kde = gaussian_kde(np.vstack([real, imaginary]))
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        peak_index = np.unravel_index(int(np.argmax(density)), density.shape)
        peak_real = float(xx[peak_index])
        peak_imag = float(yy[peak_index])
    except np.linalg.LinAlgError:
        peak_real = float(real.mean())
        peak_imag = float(imaginary.mean())
    peak = peak_real + 1j * peak_imag

    return {
        "dimension": int(matrix.shape[0]),
        "spectral_radius": float(modulus.max()),
        "minimum_eigenvalue_modulus": float(modulus.min()),
        "log_absolute_determinant": float(logabsdet),
        "determinant_sign": float(sign),
        "log_absolute_determinant_per_dimension": float(
            logabsdet / matrix.shape[0]
        ),
        "geometric_mean_eigenvalue_modulus": float(
            torch.exp(logabsdet / matrix.shape[0])
        ),
        "eigenvalue_centroid_real": float(real.mean()),
        "eigenvalue_centroid_imag": float(imaginary.mean()),
        "median_eigenvalue_real": float(np.median(real)),
        "median_eigenvalue_modulus": float(np.median(modulus)),
        "kde_peak_real": peak_real,
        "kde_peak_imag": peak_imag,
        "kde_peak_distance_to_one": float(abs(peak - 1.0)),
        "fraction_eigenvalues_within_0p1_of_one": float(
            np.mean(np.abs(eigenvalues.numpy() - 1.0) < 0.1)
        ),
        "fraction_eigenvalues_within_0p2_of_one": float(
            np.mean(np.abs(eigenvalues.numpy() - 1.0) < 0.2)
        ),
        "fraction_eigenvalue_modulus_within_0p1_of_one": float(
            np.mean(np.abs(modulus - 1.0) < 0.1)
        ),
        "maximum_singular_value": float(singular_values.max()),
        "minimum_singular_value": float(singular_values.min()),
        "condition_number": float(
            singular_values.max() / singular_values.min().clamp_min(1e-15)
        ),
        "eigenvalues_real": [float(value) for value in real],
        "eigenvalues_imag": [float(value) for value in imaginary],
    }


def plot_spectra(
    spectra: dict[str, dict[str, Any]],
    *,
    output_png: Path,
    output_svg: Path,
) -> None:
    display_names = {
        "raw_residual": "Raw residual",
        "unit_layernorm": "Unit LayerNorm",
        "block0_ln1": "Next-cycle block0.ln_1",
        "ln_final": "Readout ln_final",
    }
    colors = {
        "raw_residual": "#6b7280",
        "unit_layernorm": "#2563eb",
        "block0_ln1": "#059669",
        "ln_final": "#dc2626",
    }
    global_limit = max(
        1.05,
        max(float(item["spectral_radius"]) for item in spectra.values()) * 1.07,
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 1000)
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 12.0), dpi=220)
    for axis, name in zip(axes.ravel(), spectra, strict=True):
        item = spectra[name]
        real = np.asarray(item["eigenvalues_real"])
        imaginary = np.asarray(item["eigenvalues_imag"])
        modulus = np.hypot(real, imaginary)
        axis.plot(
            np.cos(angle),
            np.sin(angle),
            color="#9ca3af",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        axis.scatter(
            real,
            imaginary,
            c=modulus,
            cmap="viridis",
            vmin=0.0,
            vmax=max(global_limit, 1.0),
            s=20,
            alpha=0.76,
            edgecolors="white",
            linewidths=0.2,
            zorder=3,
        )
        axis.scatter(
            [item["kde_peak_real"]],
            [item["kde_peak_imag"]],
            marker="X",
            s=105,
            color=colors[name],
            edgecolor="white",
            linewidth=0.7,
            label=(
                "KDE peak "
                f"({item['kde_peak_real']:.2f}, {item['kde_peak_imag']:.2f})"
            ),
            zorder=5,
        )
        axis.scatter(
            [1.0],
            [0.0],
            marker="*",
            s=135,
            color="#f59e0b",
            edgecolor="#78350f",
            linewidth=0.45,
            label=r"Identity eigenvalue $\lambda=1$",
            zorder=6,
        )
        axis.axhline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
        axis.axvline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
        axis.set_xlim(-global_limit, global_limit)
        axis.set_ylim(-global_limit, global_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#e5e7eb", linewidth=0.6, alpha=0.8)
        axis.set_title(
            f"{display_names[name]}\n"
            rf"$\rho={item['spectral_radius']:.3f}$, "
            rf"median $|\lambda|={item['median_eigenvalue_modulus']:.3f}$, "
            rf"$\log|\det J|/d={item['log_absolute_determinant_per_dimension']:.3f}$",
            fontsize=11,
        )
        axis.legend(loc="lower right", fontsize=8, framealpha=0.92)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Imaginary part, $\operatorname{Im}(\lambda)$")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Real part, $\operatorname{Re}(\lambda)$")
    figure.suptitle(
        "Global linear rejuvenator J before and after per-example normalization\n"
        "Same D8L8 checkpoint, aligned age pairs, fitting seeds, and ridge selection",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _oracle_rollback_answers(batch: Any) -> list[torch.Tensor]:
    return [
        batch.base_states[7][:, -1],
        batch.base_states[6][:, -1],
        batch.base_states[5][:, -1],
        batch.base_states[4][:, -1],
        batch.states_by_start_shift[2][3][:, -1],
        batch.reset_oracles[8][:, -1],
    ]


def _apply_matrix(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return value @ matrix


@torch.no_grad()
def normalized_reset_answer(
    source_answer: torch.Tensor,
    *,
    matrix: torch.Tensor,
    spec: NormalizationSpec,
    mode: str,
    oracle_target_answers: Sequence[torch.Tensor],
) -> torch.Tensor:
    if mode == "direct_normalized_as_raw":
        coordinates = spec.coordinates(source_answer)
        for _ in oracle_target_answers:
            coordinates = _apply_matrix(coordinates, matrix)
        return coordinates
    if mode == "composed_source_stats":
        coordinates = spec.coordinates(source_answer)
        for _ in oracle_target_answers:
            coordinates = _apply_matrix(coordinates, matrix)
        return spec.invert_with_reference_stats(coordinates, source_answer)
    if mode == "composed_oracle_stats":
        coordinates = spec.coordinates(source_answer)
        for _ in oracle_target_answers:
            coordinates = _apply_matrix(coordinates, matrix)
        return spec.invert_with_reference_stats(
            coordinates,
            oracle_target_answers[-1],
        )
    if mode not in {"source_stats", "oracle_target_stats"}:
        raise ValueError(f"unsupported normalized reset mode: {mode}")
    answer = source_answer
    for oracle_target in oracle_target_answers:
        coordinates = spec.coordinates(answer)
        mapped = _apply_matrix(coordinates, matrix)
        reference = answer if mode == "source_stats" else oracle_target
        answer = spec.invert_with_reference_stats(mapped, reference)
    return answer


@torch.no_grad()
def evaluate_reset_condition(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch: Any,
    condition: str,
    reset_answer: torch.Tensor,
    coordinate_answer: torch.Tensor,
    coordinate_oracle: torch.Tensor,
) -> dict[str, Any]:
    source = batch.base_states[8]
    oracle = batch.reset_oracles[8]
    reset = source.clone()
    reset[:, -1] = reset_answer
    post = apply_shared_stack(model, reset, loop_index=cfg.max_loops)
    reset_mean = reset_answer.mean(dim=-1, keepdim=True)
    reset_std = (
        (reset_answer - reset_mean).square().mean(dim=-1).sqrt().mean()
    )
    source_position = age_path_position(cfg, candidate, 8)
    h2_position = age_path_position(cfg, candidate, 2)
    h3_position = age_path_position(cfg, candidate, 3)
    total_shift = source_position - h2_position
    post_position = total_shift + h3_position
    current = advance_nodes(batch.successors, batch.start, steps=source_position)
    post_target = advance_nodes(batch.successors, batch.start, steps=post_position)
    return {
        "condition": condition,
        "sample_count": int(source.shape[0]),
        "reset_raw_relative_mse_to_oracle_h2": _relative_mse(
            reset_answer, oracle[:, -1]
        ),
        "reset_raw_cosine_to_oracle_h2": _cosine(
            reset_answer, oracle[:, -1]
        ),
        "reset_mean_feature_std": float(reset_std),
        "reset_mean_l2_norm": float(reset_answer.norm(dim=-1).mean()),
        "reset_coordinate_relative_mse_to_oracle_h2": _relative_mse(
            coordinate_answer, coordinate_oracle
        ),
        "reset_coordinate_cosine_to_oracle_h2": _cosine(
            coordinate_answer, coordinate_oracle
        ),
        "reset_pre_current_accuracy": _plain_accuracy(
            logits_from_raw_state(model, reset), current
        ),
        "reset_pre_post_target_accuracy": _masked_accuracy(
            logits_from_raw_state(model, reset),
            post_target,
            endpoint=current,
        ),
        "reset_post_target_accuracy": _masked_accuracy(
            logits_from_raw_state(model, post),
            post_target,
            endpoint=current,
        ),
    }


def _aggregate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    key: str,
    ignored_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    result = []
    ignored = {key, "replication", "data_seed", *ignored_fields}
    for name, selected in groups.items():
        aggregate: dict[str, Any] = {key: name, "replications": len(selected)}
        for field in selected[0]:
            if field in ignored or not isinstance(selected[0][field], (int, float)):
                continue
            values = torch.tensor(
                [float(row[field]) for row in selected],
                dtype=torch.float64,
            )
            aggregate[f"{field}_mean"] = float(values.mean())
            aggregate[f"{field}_min"] = float(values.min())
            aggregate[f"{field}_max"] = float(values.max())
        result.append(aggregate)
    return result


def _radial_row(
    value: torch.Tensor,
    *,
    state: str,
    replication: int,
    data_seed: int,
) -> dict[str, Any]:
    value = value.float()
    per_row_mean = value.mean(dim=-1)
    per_row_std = (value - per_row_mean[:, None]).square().mean(dim=-1).sqrt()
    return {
        "replication": replication,
        "data_seed": data_seed,
        "state": state,
        "sample_count": int(value.shape[0]),
        "mean_feature_std": float(per_row_std.mean()),
        "mean_l2_norm": float(value.norm(dim=-1).mean()),
        "mean_absolute_feature_mean": float(per_row_mean.abs().mean()),
    }


def run_experiment(
    *,
    checkpoint: Path,
    raw_artifact: Path,
    out_dir: Path,
    device: torch.device,
    calibration_samples: int,
    validation_samples: int,
    collection_batch_size: int,
    evaluation_batch_size: int,
    calibration_seed: int,
    validation_seed: int,
    evaluation_seeds: Sequence[int],
    ridge_grid: Sequence[float],
) -> dict[str, Any]:
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("expected the D8L8 two-block checkpoint")
    raw_payload = torch.load(raw_artifact, map_location="cpu", weights_only=False)
    candidate = raw_payload["candidate"]
    raw_matrix = raw_payload["linear"]["matrix"].detach().float().cpu()
    specs = normalization_specs(model)

    calibration_raw = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=candidate,
        sample_count=calibration_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=calibration_seed,
    )
    print("collected calibration pairs", flush=True)
    validation_raw = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=candidate,
        sample_count=validation_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=validation_seed,
    )
    print("collected validation pairs", flush=True)

    matrices = {"raw_residual": raw_matrix}
    selected_ridges = {
        "raw_residual": float(raw_payload["linear"]["selected_ridge_multiplier"])
    }
    fit_rows: list[dict[str, Any]] = []
    for name in NORMALIZATION_NAMES:
        calibration = transform_pair_dataset(calibration_raw, specs[name])
        validation = transform_pair_dataset(validation_raw, specs[name])
        matrix, bias, rows, selected_ridge = select_and_refit_map(
            calibration=calibration,
            validation=validation,
            use_bias=False,
            ridge_grid=ridge_grid,
        )
        if not bool((bias == 0).all()):
            raise AssertionError("pure linear fit unexpectedly returned a bias")
        matrices[name] = matrix
        selected_ridges[name] = selected_ridge
        fit_rows.extend({"coordinate": name, **row} for row in rows)
        print(f"fit {name} map", flush=True)

    spectra = {name: spectral_summary(matrix) for name, matrix in matrices.items()}
    plot_spectra(
        spectra,
        output_png=out_dir / "normalized_J_eigenvalues_complex_plane.png",
        output_svg=out_dir / "normalized_J_eigenvalues_complex_plane.svg",
    )

    adjacent_rows: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    for replication, seed in enumerate(evaluation_seeds):
        eval_pairs_raw = collect_pair_dataset(
            model=model,
            cfg=cfg,
            candidate=candidate,
            sample_count=evaluation_batch_size,
            batch_size=evaluation_batch_size,
            device=device,
            seed=int(seed),
        )
        for name, matrix in matrices.items():
            data = (
                eval_pairs_raw
                if name == "raw_residual"
                else transform_pair_dataset(eval_pairs_raw, specs[name])
            )
            prediction = predict_pair_dataset(
                data,
                matrix,
                torch.zeros(cfg.d_model),
            )
            for age in TRAIN_SOURCE_AGES:
                mask = data.source_age == age
                adjacent_rows.append(
                    {
                        "replication": replication,
                        "data_seed": int(seed),
                        "coordinate": name,
                        "source_age": age,
                        "relative_mse": _relative_mse(
                            prediction[mask], data.target[mask]
                        ),
                        "cosine": _cosine(prediction[mask], data.target[mask]),
                    }
                )

        batch = collect_evaluation_batch(
            model=model,
            cfg=cfg,
            candidate=candidate,
            batch_size=evaluation_batch_size,
            device=device,
            seed=int(seed),
            maximum_commutator_age=8,
            reset_source_ages=(8,),
        )
        source = batch.base_states[8]
        oracle = batch.reset_oracles[8]
        oracle_targets = _oracle_rollback_answers(batch)
        for age in range(2, 9):
            radial_rows.append(
                _radial_row(
                    batch.base_states[age][:, -1],
                    state=f"base_h{age}",
                    replication=replication,
                    data_seed=int(seed),
                )
            )
        radial_rows.append(
            _radial_row(
                oracle[:, -1],
                state="aligned_oracle_h2_for_h8_reset",
                replication=replication,
                data_seed=int(seed),
            )
        )

        raw_matrix_device = raw_matrix.to(device)
        raw_reset_answer = source[:, -1]
        for _ in oracle_targets:
            raw_reset_answer = _apply_matrix(raw_reset_answer, raw_matrix_device)
        reset_rows.append(
            {
                "replication": replication,
                "data_seed": int(seed),
                **evaluate_reset_condition(
                    model=model,
                    cfg=cfg,
                    candidate=candidate,
                    batch=batch,
                    condition="raw_residual/composed",
                    reset_answer=raw_reset_answer,
                    coordinate_answer=raw_reset_answer,
                    coordinate_oracle=oracle[:, -1],
                ),
            }
        )

        for name in NORMALIZATION_NAMES:
            spec = specs[name].to(device)
            matrix = matrices[name].to(device)
            oracle_coordinate = spec.coordinates(oracle[:, -1])
            composed_coordinate = spec.coordinates(source[:, -1])
            for _ in oracle_targets:
                composed_coordinate = _apply_matrix(composed_coordinate, matrix)
            for mode in (
                "composed_source_stats",
                "composed_oracle_stats",
                "source_stats",
                "oracle_target_stats",
                "direct_normalized_as_raw",
            ):
                reset_answer = normalized_reset_answer(
                    source[:, -1],
                    matrix=matrix,
                    spec=spec,
                    mode=mode,
                    oracle_target_answers=oracle_targets,
                )
                reset_rows.append(
                    {
                        "replication": replication,
                        "data_seed": int(seed),
                        **evaluate_reset_condition(
                            model=model,
                            cfg=cfg,
                            candidate=candidate,
                            batch=batch,
                            condition=f"{name}/{mode}",
                            reset_answer=reset_answer,
                            coordinate_answer=composed_coordinate,
                            coordinate_oracle=oracle_coordinate,
                        ),
                    }
                )
        print(f"completed evaluation replication {replication}", flush=True)

    spectrum_rows = []
    excluded_spectrum_fields = {"eigenvalues_real", "eigenvalues_imag"}
    for name, item in spectra.items():
        spectrum_rows.append(
            {
                "coordinate": name,
                **{
                    key: value
                    for key, value in item.items()
                    if key not in excluded_spectrum_fields
                },
                "selected_ridge_multiplier": selected_ridges[name],
            }
        )
    adjacent_aggregate = _aggregate_rows(
        adjacent_rows,
        key="coordinate",
        ignored_fields=("source_age",),
    )
    reset_aggregate = _aggregate_rows(reset_rows, key="condition")
    radial_aggregate = _aggregate_rows(radial_rows, key="state")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "spectral_comparison.csv", spectrum_rows)
    _write_csv(out_dir / "fit_selection_rows.csv", fit_rows)
    _write_csv(out_dir / "adjacent_mapping_rows.csv", adjacent_rows)
    _write_csv(out_dir / "adjacent_mapping_aggregate.csv", adjacent_aggregate)
    _write_csv(out_dir / "repeated_reset_rows.csv", reset_rows)
    _write_csv(out_dir / "repeated_reset_aggregate.csv", reset_aggregate)
    _write_csv(out_dir / "raw_state_radial_profile_rows.csv", radial_rows)
    _write_csv(out_dir / "raw_state_radial_profile_aggregate.csv", radial_aggregate)
    artifact = {
        "checkpoint": str(checkpoint.resolve()),
        "candidate": candidate,
        "train_source_ages": list(TRAIN_SOURCE_AGES),
        "normalizations": {
            name: {
                "eps": spec.eps,
                "weight": spec.weight,
                "bias": spec.bias,
            }
            for name, spec in specs.items()
        },
        "maps": {
            name: {
                "matrix": matrix,
                "selected_ridge_multiplier": selected_ridges[name],
            }
            for name, matrix in matrices.items()
        },
    }
    torch.save(artifact, out_dir / "normalized_rejuvenators.pt")
    summary = {
        "model": "D8L8-seed1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": asdict(cfg),
        "loss_placement": "final CE at recurrent loop 8 only",
        "shared_unit": "two physical transformer blocks per recurrent cycle",
        "trained_recurrent_cycles": cfg.max_loops,
        "trained_effective_block_applications": cfg.max_loops * cfg.n_layers,
        "J_definition": (
            "pure linear 256x256 map jointly fit on every position-aligned "
            "adjacent age pair h_k->h_{k-1}, k=3,...,8"
        ),
        "normalization_note": (
            "pre-LayerNorm model: unit LayerNorm, next-cycle block0 ln_1, and "
            "readout ln_final are tested as distinct coordinate systems"
        ),
        "calibration_samples_per_age": calibration_samples,
        "validation_samples_per_age": validation_samples,
        "evaluation_batch_size": evaluation_batch_size,
        "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
        "selected_ridge_multipliers": selected_ridges,
        "spectra": spectra,
        "adjacent_mapping_aggregate": adjacent_aggregate,
        "repeated_reset_aggregate": reset_aggregate,
        "raw_state_radial_profile_aggregate": radial_aggregate,
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit global D8L8 rejuvenators after several per-example "
            "normalizations and compare their spectra and reset behavior."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--collection-batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--calibration-seed", type=int, default=2026077001)
    parser.add_argument("--validation-seed", type=int, default=2026077101)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=(2026077201, 2026077301),
    )
    parser.add_argument(
        "--ridge-grid",
        type=float,
        nargs="+",
        default=(
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        raw_artifact=args.raw_artifact,
        out_dir=args.out_dir,
        device=pick_device(args.device),
        calibration_samples=args.calibration_samples,
        validation_samples=args.validation_samples,
        collection_batch_size=args.collection_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        calibration_seed=args.calibration_seed,
        validation_seed=args.validation_seed,
        evaluation_seeds=args.evaluation_seeds,
        ridge_grid=args.ridge_grid,
    )
    compact = {
        name: {
            "kde_peak_real": item["kde_peak_real"],
            "kde_peak_imag": item["kde_peak_imag"],
            "spectral_radius": item["spectral_radius"],
            "median_eigenvalue_modulus": item["median_eigenvalue_modulus"],
            "log_absolute_determinant_per_dimension": item[
                "log_absolute_determinant_per_dimension"
            ],
        }
        for name, item in summary["spectra"].items()
    }
    print(json.dumps({"spectral_headline": compact}, indent=2))


if __name__ == "__main__":
    main()
