"""Measure whether shared J-compressed directions grow along natural H1..H8 age."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    atomic_json,
    load_bank_and_bases,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed


AGES = tuple(range(1, 9))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--calibration-examples", type=int, default=1024)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=878001)
    parser.add_argument(
        "--graph-seeds",
        type=int,
        nargs="+",
        default=(878101, 878102, 878103, 878104, 878105),
    )
    parser.add_argument("--random-draws", type=int, default=2)
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def named_bases(
    *,
    bases: dict[str, dict[int, np.ndarray]],
    dimension: int,
    rank: int,
    random_draws: int,
) -> dict[str, np.ndarray | None]:
    result: dict[str, np.ndarray | None] = {
        "full": None,
        "bottom_input": bases["bottom_input"][rank],
        "bottom_output": bases["bottom_output"][rank],
        "top_input": bases["top_input"][rank],
    }
    rng = np.random.default_rng(878701)
    for draw in range(random_draws):
        value, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
        result[f"random{draw}"] = value[:, :rank]
    return result


def site_coordinates(
    state: torch.Tensor,
    basis: torch.Tensor | None,
    site: str,
) -> np.ndarray:
    selected = state[:, -1:] if site == "answer" else state
    if basis is not None:
        selected = selected.float() @ basis
    return selected.float().flatten(1).cpu().numpy().astype(np.float64, copy=False)


@torch.no_grad()
def collect_calibration_centroids(
    *,
    model,
    cfg,
    bases: dict[str, torch.Tensor | None],
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[tuple[str, str, int], np.ndarray]:
    if examples % batch_size:
        raise ValueError("calibration examples must divide by batch size")
    set_seed(seed)
    sums: dict[tuple[str, str, int], np.ndarray] = {}
    counts: defaultdict[tuple[str, str, int], int] = defaultdict(int)
    for _ in range(examples // batch_size):
        tokens, _, _, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        for age in AGES:
            state = model.apply_loop(state, loop_index=age - 1)
            for site in ("answer", "all_tokens"):
                for family, basis in bases.items():
                    value = site_coordinates(state, basis, site)
                    key = (site, family, age)
                    sums[key] = sums.get(key, np.zeros(value.shape[1])) + value.sum(0)
                    counts[key] += value.shape[0]
    return {key: value / counts[key] for key, value in sums.items()}


def age_axes(
    centroids: dict[tuple[str, str, int], np.ndarray]
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    axes: dict[tuple[str, str], np.ndarray] = {}
    spans: dict[tuple[str, str], float] = {}
    for site, family, _ in centroids:
        key = (site, family)
        if key in axes:
            continue
        delta = centroids[(site, family, 8)] - centroids[(site, family, 1)]
        span = float(np.linalg.norm(delta))
        axes[key] = delta / max(span, 1e-12)
        spans[key] = span
    return axes, spans


@torch.no_grad()
def evaluate_seed(
    *,
    model,
    cfg,
    bases: dict[str, torch.Tensor | None],
    centroids: dict[tuple[str, str, int], np.ndarray],
    axes: dict[tuple[str, str], np.ndarray],
    spans: dict[tuple[str, str], float],
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("examples must divide by batch size")
    set_seed(seed)
    values: defaultdict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    for _ in range(examples // batch_size):
        tokens, _, _, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        for age in AGES:
            state = model.apply_loop(state, loop_index=age - 1)
            for site in ("answer", "all_tokens"):
                for family, basis in bases.items():
                    values[(site, family, age)].append(
                        site_coordinates(state, basis, site)
                    )
    rows: list[dict[str, Any]] = []
    for (site, family, age), parts in sorted(values.items()):
        value = np.concatenate(parts)
        center1 = centroids[(site, family, 1)]
        signed = (value - center1) @ axes[(site, family)]
        full = np.concatenate(values[(site, "full", age)])
        rows.append(
            {
                "graph_seed": seed,
                "site": site,
                "family": family,
                "age": age,
                "signed_projection_mean": float(signed.mean()),
                "signed_projection_sample_sem": float(
                    signed.std(ddof=1) / np.sqrt(signed.size)
                ),
                "normalized_age_progress": float(
                    signed.mean() / max(spans[(site, family)], 1e-12)
                ),
                "coordinate_rms": float(np.sqrt(np.square(value).mean())),
                "h1_centered_coordinate_rms": float(
                    np.sqrt(np.square(value - center1).mean())
                ),
                "ambient_projection_energy_fraction": (
                    1.0
                    if family == "full"
                    else float(np.square(value).sum() / np.square(full).sum())
                ),
                "calibration_H1_H8_span": spans[(site, family)],
                "examples": examples,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = (
        "signed_projection_mean",
        "normalized_age_progress",
        "coordinate_rms",
        "h1_centered_coordinate_rms",
        "ambient_projection_energy_fraction",
    )
    keys = sorted({(row["site"], row["family"], row["age"]) for row in rows})
    for site, family, age in keys:
        selected = [
            row
            for row in rows
            if row["site"] == site
            and row["family"] == family
            and row["age"] == age
        ]
        result: dict[str, Any] = {
            "site": site,
            "family": family,
            "age": age,
            "graph_seeds": len(selected),
        }
        for metric in metrics:
            vector = np.asarray([row[metric] for row in selected])
            result[f"{metric}_mean"] = float(vector.mean())
            result[f"{metric}_seed_sem"] = float(
                vector.std(ddof=1) / np.sqrt(vector.size)
            )
        output.append(result)
    return output


def trend_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for site, family in sorted({(row["site"], row["family"]) for row in rows}):
        seed_correlations = []
        seed_monotone = []
        seed_rms_correlations = []
        for seed in sorted({row["graph_seed"] for row in rows}):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["site"] == site
                    and row["family"] == family
                    and row["graph_seed"] == seed
                ],
                key=lambda row: row["age"],
            )
            progress = np.asarray([row["normalized_age_progress"] for row in selected])
            rms = np.asarray([row["coordinate_rms"] for row in selected])
            ages = np.arange(1, 9)
            seed_correlations.append(float(np.corrcoef(ages, progress)[0, 1]))
            seed_rms_correlations.append(float(np.corrcoef(ages, rms)[0, 1]))
            seed_monotone.append(int(np.sum(np.diff(progress) > 0)))
        for metric, vector in (
            ("age_signed_projection_pearson", seed_correlations),
            ("age_coordinate_rms_pearson", seed_rms_correlations),
            ("increasing_steps_out_of_7", seed_monotone),
        ):
            value = np.asarray(vector, dtype=np.float64)
            output.append(
                {
                    "site": site,
                    "family": family,
                    "metric": metric,
                    "mean": float(value.mean()),
                    "seed_sem": float(value.std(ddof=1) / np.sqrt(value.size)),
                    "graph_seeds": value.size,
                }
            )
    return output


def plot_summary(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    families = ("bottom_input", "bottom_output", "top_input", "random0")
    labels = {
        "bottom_input": "bottom-8 input",
        "bottom_output": "bottom-8 output",
        "top_input": "top-8 input control",
        "random0": "random-8 control",
    }
    for column, site in enumerate(("answer", "all_tokens")):
        for family in families:
            selected = sorted(
                [row for row in rows if row["site"] == site and row["family"] == family],
                key=lambda row: row["age"],
            )
            ages = [row["age"] for row in selected]
            progress = [row["normalized_age_progress_mean"] for row in selected]
            sem = [row["normalized_age_progress_seed_sem"] for row in selected]
            axes[0, column].errorbar(
                ages, progress, yerr=sem, marker="o", capsize=2, label=labels[family]
            )
            energy = [row["ambient_projection_energy_fraction_mean"] for row in selected]
            energy_sem = [row["ambient_projection_energy_fraction_seed_sem"] for row in selected]
            axes[1, column].errorbar(
                ages, energy, yerr=energy_sem, marker="o", capsize=2, label=labels[family]
            )
        axes[0, column].set(
            title=f"{site}: signed H1-to-H8 projection",
            xlabel="natural hidden-state age",
            ylabel="normalized projection (H1=0, H8=1)",
            xticks=AGES,
        )
        axes[1, column].set(
            title=f"{site}: raw subspace energy",
            xlabel="natural hidden-state age",
            ylabel="fraction of full hidden energy",
            xticks=AGES,
        )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    _, _, common_bases, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (args.rank,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    basis_arrays = named_bases(
        bases=common_bases,
        dimension=cfg.d_model,
        rank=args.rank,
        random_draws=args.random_draws,
    )
    basis_tensors = {
        name: (
            None
            if value is None
            else torch.as_tensor(value, dtype=torch.float32, device=device)
        )
        for name, value in basis_arrays.items()
    }
    centroids = collect_calibration_centroids(
        model=model,
        cfg=cfg,
        bases=basis_tensors,
        examples=args.calibration_examples,
        batch_size=args.batch_size,
        seed=args.calibration_seed,
        device=device,
    )
    axes, spans = age_axes(centroids)
    rows: list[dict[str, Any]] = []
    for seed in args.graph_seeds:
        rows.extend(
            evaluate_seed(
                model=model,
                cfg=cfg,
                bases=basis_tensors,
                centroids=centroids,
                axes=axes,
                spans=spans,
                examples=args.examples,
                batch_size=args.batch_size,
                seed=seed,
                device=device,
            )
        )
    summary_rows = summarize(rows)
    trend_rows = trend_summary(rows)
    write_csv(args.out_dir / "age_projection_per_seed.csv", rows)
    write_csv(args.out_dir / "age_projection_summary.csv", summary_rows)
    write_csv(args.out_dir / "age_projection_trends.csv", trend_rows)
    plot_summary(summary_rows, args.out_dir / "natural_age_projection.png")
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "rank": args.rank,
        "calibration_examples": args.calibration_examples,
        "calibration_seed": args.calibration_seed,
        "examples_per_graph_seed": args.examples,
        "graph_seeds": list(args.graph_seeds),
        "sites": ["answer", "all_tokens"],
        "age_axis_definition": (
            "H1-to-H8 centroid direction fit on an independent calibration graph seed; "
            "evaluation is on held-out graph seeds"
        ),
        "claim_boundary": (
            "Growth of a held-out signed projection is localization evidence for an age-aligned "
            "representation. Raw norm growth alone is not a signed age signal and neither result "
            "establishes causal use."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
