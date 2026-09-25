#!/usr/bin/env python3
"""Plot the component-supervised direct-H3 affine-J experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_formal(results: Path) -> None:
    summary = load_json(
        results / "remote_artifacts" / "formal_eval" / "summary.json"
    )
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    for condition, label, color, width in (
        ("task_unit_every", "learned direct-H3 J", "#0072B2", 2.5),
        ("exact_operating_every", "exact H3 interface", "#009E73", 2.0),
        (
            "shuffled_final_unit_every",
            "shuffled learned output",
            "#CC79A7",
            1.5,
        ),
        ("no_control", "no J", "#555555", 1.5),
    ):
        values = summary["curves"][condition]["accuracy_nonendpoint"][
            "values"
        ]
        axis.plot(
            np.arange(1, len(values) + 1),
            values,
            label=label,
            color=color,
            linewidth=width,
        )
    axis.axhline(0.9, color="black", linestyle=":", linewidth=1.1)
    axis.axhline(0.125, color="#888888", linestyle="--", linewidth=1.0)
    axis.set_xlim(1, 96)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("additional two-hop continuation loops")
    axis.set_ylabel("strict nonendpoint accuracy")
    axis.set_title("2,048 fresh graphs: direct-H3 affine J")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results / "formal_nonendpoint_curves.png", dpi=180)
    plt.close(figure)


def plot_curriculum(results: Path) -> None:
    stages = ("h24", "h32", "h48", "h64")
    windows = ("auc_1_24", "auc_25_48", "auc_49_64")
    values = []
    for stage in stages:
        summary = load_json(
            results / "remote_artifacts" / stage / "summary.json"
        )
        metric = summary["curves"]["task_unit_every"][
            "accuracy_nonendpoint"
        ]
        values.append(
            [
                np.nan if metric[window] is None else metric[window]
                for window in windows
            ]
        )
    values_array = np.asarray(values)
    x = np.arange(len(stages))
    width = 0.23
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    colors = ("#0072B2", "#E69F00", "#009E73")
    for index, (window, color) in enumerate(zip(windows, colors, strict=True)):
        axis.bar(
            x + (index - 1) * width,
            values_array[:, index],
            width,
            label=window.replace("auc_", "loops "),
            color=color,
        )
    axis.axhline(0.9, color="black", linestyle=":", linewidth=1.1)
    axis.set_xticks(x, [stage.upper() for stage in stages])
    axis.set_ylim(0.4, 1.02)
    axis.set_ylabel("strict nonendpoint AUC")
    axis.set_title("Curriculum progressively extends closed-loop lifespan")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results / "curriculum_window_progress.png", dpi=180)
    plt.close(figure)


def plot_strict_unseen(results: Path) -> None:
    audit = load_json(
        results
        / "remote_artifacts"
        / "strict_unseen_audit_nonendpoint"
        / "summary.json"
    )
    partitions = {item["partition"]: item for item in audit["results"]}
    seen = partitions["seen_during_J_training"]["curves"]
    unseen = partitions["strictly_unseen_by_J_training"]["curves"]

    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    for curves, condition, label, color, style in (
        (
            unseen,
            "learned_J_plus_full_Block2",
            "learned J: strictly unseen",
            "#0072B2",
            "-",
        ),
        (
            seen,
            "learned_J_plus_full_Block2",
            "learned J: seen",
            "#56B4E9",
            "--",
        ),
        (
            unseen,
            "exact_H3_plus_full_Block2",
            "exact H3: unseen",
            "#009E73",
            "-",
        ),
        (
            unseen,
            "no_J_plus_full_Block2",
            "no J: unseen",
            "#555555",
            "-",
        ),
    ):
        values = curves[condition]["nonendpoint_accuracy_by_cycle"]
        axis.plot(
            np.arange(1, len(values) + 1),
            values,
            label=label,
            color=color,
            linestyle=style,
            linewidth=2.0,
        )
    axis.axhline(0.9, color="black", linestyle=":", linewidth=1.1)
    axis.set_xlim(1, 64)
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("additional two-hop continuation loops")
    axis.set_ylabel("strict nonendpoint accuracy")
    axis.set_title("Cycle-type matched J-training seen vs strictly unseen graphs")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results / "strict_unseen_nonendpoint_curves.png", dpi=180)
    plt.close(figure)


def plot_spectrum(results: Path) -> None:
    artifact = torch.load(
        results
        / "remote_artifacts"
        / "h64"
        / "unit_j_maps.pt",
        map_location="cpu",
        weights_only=False,
    )
    weight_tensor = artifact["maps"]["task"]["weight"].double()
    eigenvalues = np.linalg.eigvals(weight_tensor.numpy())
    _, log_abs_det_tensor = torch.linalg.slogdet(weight_tensor)
    log_abs_det = float(log_abs_det_tensor)
    bandwidth = 0.035
    density = np.asarray(
        [
            np.exp(
                -np.abs(eigenvalues - candidate) ** 2
                / (2.0 * bandwidth**2)
            ).sum()
            for candidate in eigenvalues
        ]
    )
    peak = eigenvalues[int(density.argmax())]
    median_modulus = float(np.median(np.abs(eigenvalues)))
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    figure, axis = plt.subplots(figsize=(6.4, 6.1))
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    axis.plot(np.cos(theta), np.sin(theta), "--", color="#9AA5B1")
    axis.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        s=19,
        alpha=0.68,
        color="#2F6FDB",
    )
    axis.scatter([peak.real], [peak.imag], marker="x", s=90, color="red")
    axis.axhline(0.0, color="#C7CDD4", linewidth=0.8)
    axis.axvline(0.0, color="#C7CDD4", linewidth=0.8)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xlabel(r"Re($\lambda$)")
    axis.set_ylabel(r"Im($\lambda$)")
    axis.set_title(
        "Direct-H3 affine J (linear part)\n"
        f"peak≈{peak.real:.3f}{peak.imag:+.3f}i, "
        f"median |λ|={median_modulus:.3f}, "
        f"ρ={spectral_radius:.3f}, log|det|={log_abs_det:.1f}"
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(results / "direct_H3_J_complex_spectrum.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "results/"
            "graph_path_prenorm_component_unit_j_direct_H3_"
            "curriculum64_20260731"
        ),
    )
    args = parser.parse_args()
    plot_formal(args.results)
    plot_curriculum(args.results)
    plot_strict_unseen(args.results)
    plot_spectrum(args.results)


if __name__ == "__main__":
    main()
