from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATE_SITES = (
    "loop_input",
    "b1_output_preJ",
    "b2_input_postJ",
    "loop_output",
)
REPRESENTATIONS = ("raw", "centered_unit_direction")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _centered_unit_direction(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    return centered / np.maximum(norm, 1e-12)


def _coordinate_energy_entropy(values: np.ndarray) -> float:
    centered = values - values.mean(axis=-1, keepdims=True)
    energy = np.square(centered)
    probability = energy / np.maximum(
        energy.sum(axis=-1, keepdims=True),
        1e-30,
    )
    entropy = -(
        probability * np.log(np.maximum(probability, 1e-30))
    ).sum(axis=-1)
    return float((entropy / np.log(values.shape[-1])).mean())


def _spectral_stats(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64, copy=False)
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )
    eigenvalues = np.square(singular_values) / max(1, len(values) - 1)
    trace = float(eigenvalues.sum())
    if trace <= 1e-30:
        return {
            "normalized_spectral_entropy": float("nan"),
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "covariance_trace": 0.0,
            "pc1_fraction": float("nan"),
        }
    probability = eigenvalues / trace
    nonzero = probability > 0
    entropy = float(
        -(probability[nonzero] * np.log(probability[nonzero])).sum()
    )
    return {
        "normalized_spectral_entropy": entropy / np.log(values.shape[-1]),
        "effective_rank": float(np.exp(entropy)),
        "participation_ratio": float(
            trace * trace / np.square(eigenvalues).sum()
        ),
        "covariance_trace": trace,
        "pc1_fraction": float(probability[0]),
    }


def _entropy_rows(
    *,
    replica: int | str,
    cycles: np.ndarray,
    site: str,
    values: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for age_index, cycle in enumerate(cycles):
        state = values[:, age_index].astype(np.float64, copy=False)
        coordinate_entropy = _coordinate_energy_entropy(state)
        for representation in REPRESENTATIONS:
            transformed = (
                state
                if representation == "raw"
                else _centered_unit_direction(state)
            )
            state_rows.append(
                {
                    "replica": replica,
                    "cycle": int(cycle),
                    "site": site,
                    "representation": representation,
                    "examples": len(state),
                    "coordinate_energy_entropy": coordinate_entropy,
                    **_spectral_stats(transformed),
                }
            )
        if age_index == 0:
            continue
        drift = state - values[:, 0].astype(np.float64, copy=False)
        for representation in REPRESENTATIONS:
            transformed = (
                drift
                if representation == "raw"
                else _centered_unit_direction(drift)
            )
            drift_rows.append(
                {
                    "replica": replica,
                    "cycle": int(cycle),
                    "reference_cycle": int(cycles[0]),
                    "site": site,
                    "representation": representation,
                    "examples": len(drift),
                    "coordinate_energy_entropy": (
                        _coordinate_energy_entropy(drift)
                    ),
                    **_spectral_stats(transformed),
                }
            )
    return state_rows, drift_rows


def _read_accuracy(path: Path, cycles: np.ndarray) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_cycle = {int(row["cycle"]): float(row["accuracy"]) for row in rows}
    missing = [int(cycle) for cycle in cycles if int(cycle) not in by_cycle]
    if missing:
        raise ValueError(f"accuracy file is missing cycles: {missing}")
    return {int(cycle): by_cycle[int(cycle)] for cycle in cycles}


def _pooled_lookup(
    rows: list[dict[str, Any]],
    *,
    site: str,
    representation: str,
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["replica"] == "pooled"
            and row["site"] == site
            and row["representation"] == representation
        ),
        key=lambda row: int(row["cycle"]),
    )


def _plot_state_entropy(
    *,
    out_dir: Path,
    cycles: np.ndarray,
    rows: list[dict[str, Any]],
    accuracy: dict[int, float],
) -> str:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        constrained_layout=True,
    )
    panels = (
        ("raw", "normalized_spectral_entropy", "Raw spectral entropy"),
        (
            "centered_unit_direction",
            "normalized_spectral_entropy",
            "Norm-free direction spectral entropy",
        ),
        (
            "centered_unit_direction",
            "effective_rank",
            "Norm-free direction effective rank",
        ),
        ("raw", "covariance_trace", "Raw covariance trace"),
    )
    for axis, (representation, metric, title) in zip(
        axes.ravel(),
        panels,
        strict=True,
    ):
        for site in STATE_SITES:
            selected = _pooled_lookup(
                rows,
                site=site,
                representation=representation,
            )
            axis.plot(
                cycles,
                [float(row[metric]) for row in selected],
                marker="o",
                label=site,
            )
        axis.set(title=title, xlabel="matched continuation loop")
        axis.grid(alpha=0.18)
        if metric == "covariance_trace":
            axis.set_yscale("log")
        twin = axis.twinx()
        twin.plot(
            cycles,
            [accuracy[int(cycle)] for cycle in cycles],
            color="black",
            linestyle="--",
            marker="s",
            alpha=0.65,
            label="accuracy",
        )
        twin.set_ylim(0, 1.05)
        twin.set_ylabel("accuracy")
    axes[0, 0].legend(fontsize=8)
    path = out_dir / "hidden_state_entropy_and_accuracy.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_drift_entropy(
    *,
    out_dir: Path,
    cycles: np.ndarray,
    rows: list[dict[str, Any]],
    accuracy: dict[int, float],
) -> str:
    drift_cycles = cycles[1:]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        constrained_layout=True,
    )
    panels = (
        ("raw", "normalized_spectral_entropy", "Drift spectral entropy"),
        ("raw", "effective_rank", "Drift effective rank"),
        ("raw", "pc1_fraction", "Drift PC1 variance fraction"),
        ("raw", "covariance_trace", "Drift covariance trace"),
    )
    for axis, (representation, metric, title) in zip(
        axes.ravel(),
        panels,
        strict=True,
    ):
        for site in STATE_SITES:
            selected = _pooled_lookup(
                rows,
                site=site,
                representation=representation,
            )
            axis.plot(
                drift_cycles,
                [float(row[metric]) for row in selected],
                marker="o",
                label=site,
            )
        axis.set(title=title, xlabel="matched continuation loop")
        axis.grid(alpha=0.18)
        if metric == "covariance_trace":
            axis.set_yscale("log")
        twin = axis.twinx()
        twin.plot(
            drift_cycles,
            [accuracy[int(cycle)] for cycle in drift_cycles],
            color="black",
            linestyle="--",
            marker="s",
            alpha=0.65,
        )
        twin.set_ylim(0, 1.05)
        twin.set_ylabel("accuracy")
    axes[0, 0].legend(fontsize=8)
    path = out_dir / "hidden_drift_entropy_and_accuracy.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_coordinate_entropy(
    *,
    out_dir: Path,
    cycles: np.ndarray,
    rows: list[dict[str, Any]],
    accuracy: dict[int, float],
) -> str:
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for site in STATE_SITES:
        selected = _pooled_lookup(
            rows,
            site=site,
            representation="raw",
        )
        axis.plot(
            cycles,
            [float(row["coordinate_energy_entropy"]) for row in selected],
            marker="o",
            label=site,
        )
    axis.set(
        title=(
            "Per-sample coordinate-energy entropy "
            "(basis-dependent diagnostic)"
        ),
        xlabel="matched continuation loop",
        ylabel="normalized entropy",
    )
    axis.grid(alpha=0.18)
    axis.legend(fontsize=8)
    twin = axis.twinx()
    twin.plot(
        cycles,
        [accuracy[int(cycle)] for cycle in cycles],
        color="black",
        linestyle="--",
        marker="s",
        alpha=0.65,
        label="accuracy",
    )
    twin.set_ylim(0, 1.05)
    twin.set_ylabel("accuracy")
    path = out_dir / "hidden_coordinate_energy_entropy.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _replica_stability(
    rows: list[dict[str, Any]],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    replica_ids = sorted(
        {int(row["replica"]) for row in rows if row["replica"] != "pooled"}
    )
    if len(replica_ids) != 2:
        raise ValueError("stability audit expects exactly two replicas")
    output = []
    for site in STATE_SITES:
        for representation in REPRESENTATIONS:
            curves = []
            for replica in replica_ids:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["replica"] == replica
                        and row["site"] == site
                        and row["representation"] == representation
                    ),
                    key=lambda row: int(row["cycle"]),
                )
                curves.append(np.array([float(row[metric]) for row in selected]))
            output.append(
                {
                    "site": site,
                    "representation": representation,
                    "metric": metric,
                    "replica_0": replica_ids[0],
                    "replica_1": replica_ids[1],
                    "pearson": float(np.corrcoef(curves[0], curves[1])[0, 1]),
                    "max_absolute_difference": float(
                        np.max(np.abs(curves[0] - curves[1]))
                    ),
                    "mean_absolute_difference": float(
                        np.mean(np.abs(curves[0] - curves[1]))
                    ),
                }
            )
    return output


def run_experiment(
    *,
    activation_dir: Path,
    accuracy_csv: Path,
    out_dir: Path,
) -> dict[str, Any]:
    replica_paths = sorted(
        activation_dir.glob("replica_*/fixed_content_answer_states.npz")
    )
    if len(replica_paths) != 2:
        raise ValueError(
            f"expected two activation replicas, found {len(replica_paths)}"
        )
    payloads = [np.load(path) for path in replica_paths]
    cycles = payloads[0]["matched_cycles"].astype(np.int64)
    if any(not np.array_equal(item["matched_cycles"], cycles) for item in payloads):
        raise ValueError("replicas use different matched cycles")
    accuracy = _read_accuracy(accuracy_csv, cycles)

    state_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for replica, item in enumerate(payloads):
        for site in STATE_SITES:
            new_state, new_drift = _entropy_rows(
                replica=replica,
                cycles=cycles,
                site=site,
                values=item[site],
            )
            state_rows.extend(new_state)
            drift_rows.extend(new_drift)
    for site in STATE_SITES:
        pooled = np.concatenate([item[site] for item in payloads], axis=0)
        new_state, new_drift = _entropy_rows(
            replica="pooled",
            cycles=cycles,
            site=site,
            values=pooled,
        )
        state_rows.extend(new_state)
        drift_rows.extend(new_drift)

    stability_rows = _replica_stability(
        state_rows,
        metric="normalized_spectral_entropy",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "hidden_state_entropy.csv", state_rows)
    _write_csv(out_dir / "hidden_drift_entropy.csv", drift_rows)
    _write_csv(out_dir / "entropy_replica_stability.csv", stability_rows)
    figures = {
        "state": _plot_state_entropy(
            out_dir=out_dir,
            cycles=cycles,
            rows=state_rows,
            accuracy=accuracy,
        ),
        "drift": _plot_drift_entropy(
            out_dir=out_dir,
            cycles=cycles,
            rows=drift_rows,
            accuracy=accuracy,
        ),
        "coordinate": _plot_coordinate_entropy(
            out_dir=out_dir,
            cycles=cycles,
            rows=state_rows,
            accuracy=accuracy,
        ),
    }

    selected_cycles = [8, 48, 64, 72, 80, 96]
    selected_rows = [
        row
        for row in state_rows
        if row["replica"] == "pooled"
        and int(row["cycle"]) in selected_cycles
    ]
    selected_drift_rows = [
        row
        for row in drift_rows
        if row["replica"] == "pooled"
        and int(row["cycle"]) in selected_cycles
    ]
    summary = {
        "status": "complete",
        "source_activation_dir": str(activation_dir),
        "source_accuracy_csv": str(accuracy_csv),
        "loss_placement": "frozen final-only D8L8 seed0; no new inference",
        "trained_loop_count": 8,
        "shared_block_count": 2,
        "effective_training_depth": 16,
        "evaluated_matched_cycles": cycles.tolist(),
        "examples": int(sum(len(item["starts"]) for item in payloads)),
        "d_model": int(payloads[0]["loop_output"].shape[-1]),
        "entropy_definitions": {
            "normalized_spectral_entropy": (
                "H(lambda/sum(lambda))/log(d_model), where lambda are "
                "sample-covariance eigenvalues; rotation invariant"
            ),
            "effective_rank": "exp(H(lambda/sum(lambda)))",
            "participation_ratio": "sum(lambda)^2/sum(lambda^2)",
            "centered_unit_direction": (
                "subtract each vector's channel mean and normalize its L2 "
                "norm before population covariance; removes radial scale"
            ),
            "coordinate_energy_entropy": (
                "mean H((h-mean_channel(h))^2 / energy)/log(d_model); "
                "basis dependent, auxiliary only"
            ),
        },
        "selected_hidden_state_entropy": selected_rows,
        "selected_hidden_drift_entropy": selected_drift_rows,
        "replica_stability": stability_rows,
        "claim_ledger": [
            {
                "claim": "overloop failure is monotonic entropy growth",
                "status": "tested descriptively, not assumed",
                "evidence": "hidden_state_entropy.csv",
            },
            {
                "claim": "late drift becomes concentrated in fewer modes",
                "status": "localization if drift entropy falls and PC1 rises",
                "evidence": "hidden_drift_entropy.csv",
            },
            {
                "claim": "hidden entropy alone causally explains failure",
                "status": "not established",
                "evidence_needed": (
                    "entropy-matched state interventions that selectively "
                    "change task behavior"
                ),
            },
        ],
        "files": {
            "state_csv": "hidden_state_entropy.csv",
            "drift_csv": "hidden_drift_entropy.csv",
            "stability_csv": "entropy_replica_stability.csv",
            "figures": figures,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in payloads:
        item.close()
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entropy audit of fixed-content learned-J hidden states."
    )
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--accuracy-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        activation_dir=args.activation_dir,
        accuracy_csv=args.accuracy_csv,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
