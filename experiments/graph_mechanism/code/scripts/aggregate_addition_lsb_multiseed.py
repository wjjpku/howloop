from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TRANSITION_LENGTHS = tuple(range(12, 18))
CONTROL_LENGTHS = tuple(range(10, 18))
CONTROL_MODES = ("raw", "full", "no_AB", "identity_D", "full_executor_off")
COLORS = {
    "raw": "#5f6b73",
    "full": "#d62728",
    "no_AB": "#ff7f0e",
    "identity_D": "#2ca02c",
    "full_executor_off": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete artifact: {path}")
    return payload


def row_index(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["length"]), str(row["variant"])): row
        for row in payload["rows"]
    }


def metric(row: dict[str, Any], name: str = "supervised_digit_exact_match") -> float:
    return float(row[name])


def frontier(rows: dict[tuple[int, str], dict[str, Any]], mode: str, threshold: float) -> int | None:
    passing = [
        length
        for (length, variant), row in rows.items()
        if variant == mode and metric(row) >= threshold
    ]
    return max(passing) if passing else None


def load_matrix(path: Path, *, label: str | None) -> dict[str, Any]:
    payload = load_json(path)
    controllers = payload.get("controllers")
    if not isinstance(controllers, list) or not controllers:
        raise ValueError(f"expected controller rows in {path}")
    if label is not None:
        matches = [row for row in controllers if row.get("label") == label]
        if len(matches) != 1:
            raise ValueError(f"expected matrix label {label!r} exactly once in {path}")
        return matches[0]
    if len(controllers) != 1:
        raise ValueError(f"multiple controllers in {path}; manifest must set matrix_label")
    return controllers[0]


def validate_run(
    run: dict[str, Any],
    *,
    base: Path,
) -> dict[str, Any]:
    seed = int(run["backbone_seed"])
    transition_path = resolve(base, run["transition_summary"])
    controls_path = resolve(base, run["controls_summary"])
    wide_path = resolve(base, run["wide_summary"])
    id_path = resolve(base, run["id_summary"])
    matrix_path = resolve(base, run["matrix_analysis"])
    transition = load_json(transition_path)
    controls = load_json(controls_path)
    wide = load_json(wide_path)
    identity = load_json(id_path)
    matrix = load_matrix(matrix_path, label=run.get("matrix_label"))
    for name, payload in (
        ("transition", transition),
        ("controls", controls),
        ("wide", wide),
        ("id", identity),
    ):
        if int(payload["checkpoint_step"]) != int(run["selected_step"]):
            raise ValueError(f"seed {seed}: {name} checkpoint step mismatch")
        if payload.get("target_loop_rule") != "T(m)=m":
            raise ValueError(f"seed {seed}: {name} is not T(m)=m")
        if payload.get("addition_answer_supervision") != "logical_digits":
            raise ValueError(f"seed {seed}: {name} uses the wrong supervision mask")

    if tuple(transition["lengths"]) != TRANSITION_LENGTHS:
        raise ValueError(f"seed {seed}: transition lengths mismatch")
    if int(transition["examples_per_length"]) != 8192:
        raise ValueError(f"seed {seed}: transition sample count mismatch")
    if set(transition["variants"]) != {"raw", "full"}:
        raise ValueError(f"seed {seed}: transition variants mismatch")

    if not set(CONTROL_LENGTHS).issubset(set(map(int, controls["lengths"]))):
        raise ValueError(f"seed {seed}: controls do not cover 10..17")
    if not set(CONTROL_MODES).issubset(set(controls["variants"])):
        raise ValueError(f"seed {seed}: component controls are incomplete")
    if int(controls["examples_per_length"]) < 512:
        raise ValueError(f"seed {seed}: controls have fewer than 512 examples")

    if not set(range(1, 31)).issubset(set(map(int, wide["lengths"]))):
        raise ValueError(f"seed {seed}: wide sweep does not cover 1..30")
    if int(wide["examples_per_length"]) < 512:
        raise ValueError(f"seed {seed}: wide sweep has fewer than 512 examples")
    if not set(range(1, 11)).issubset(set(map(int, identity["lengths"]))):
        raise ValueError(f"seed {seed}: ID evaluation does not cover 1..10")
    if int(identity["examples_per_length"]) < 1024:
        raise ValueError(f"seed {seed}: ID evaluation has fewer than 1024 examples")

    return {
        "seed": seed,
        "selected_step": int(run["selected_step"]),
        "transition": row_index(transition),
        "controls": row_index(controls),
        "wide": row_index(wide),
        "id": row_index(identity),
        "matrix": matrix,
        "paths": {
            "transition": str(transition_path),
            "controls": str(controls_path),
            "wide": str(wide_path),
            "id": str(id_path),
            "matrix": str(matrix_path),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.parent
    runs = [validate_run(run, base=base) for run in manifest["runs"]]
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("duplicate backbone seeds in manifest")
    runs.sort(key=lambda run: run["seed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    transition_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    seed_summary: list[dict[str, Any]] = []
    for run in runs:
        seed = run["seed"]
        for length in TRANSITION_LENGTHS:
            for mode in ("raw", "full"):
                source = run["transition"][(length, mode)]
                transition_rows.append(
                    {
                        "backbone_seed": seed,
                        "length": length,
                        "mode": mode,
                        "examples": int(source["examples"]),
                        "supervised_digit_exact_match": metric(source),
                        "supervised_digit_bit_accuracy": metric(
                            source, "supervised_digit_bit_accuracy"
                        ),
                        "final_carry_accuracy": metric(source, "final_carry_accuracy"),
                    }
                )
        for length in CONTROL_LENGTHS:
            for mode in CONTROL_MODES:
                source = run["controls"][(length, mode)]
                control_rows.append(
                    {
                        "backbone_seed": seed,
                        "length": length,
                        "mode": mode,
                        "examples": int(source["examples"]),
                        "supervised_digit_exact_match": metric(source),
                        "supervised_digit_bit_accuracy": metric(
                            source, "supervised_digit_bit_accuracy"
                        ),
                    }
                )
        raw_transition = np.mean(
            [metric(run["transition"][(length, "raw")]) for length in TRANSITION_LENGTHS]
        )
        full_transition = np.mean(
            [metric(run["transition"][(length, "full")]) for length in TRANSITION_LENGTHS]
        )
        raw_id = np.mean(
            [metric(run["id"][(length, "raw")]) for length in range(1, 11)]
        )
        full_id = np.mean(
            [metric(run["id"][(length, "full")]) for length in range(1, 11)]
        )
        matrix = run["matrix"]
        seed_summary.append(
            {
                "backbone_seed": seed,
                "selected_step": run["selected_step"],
                "raw_id_mean_em": float(raw_id),
                "full_id_mean_em": float(full_id),
                "raw_transition_mean_em": float(raw_transition),
                "full_transition_mean_em": float(full_transition),
                "transition_delta_em": float(full_transition - raw_transition),
                "raw_q90_frontier": frontier(run["wide"], "raw", 0.90),
                "full_q90_frontier": frontier(run["wide"], "full", 0.90),
                "raw_q50_frontier": frontier(run["wide"], "raw", 0.50),
                "full_q50_frontier": frontier(run["wide"], "full", 0.50),
                "AB_frobenius_norm": float(matrix["AB_frobenius_norm"]),
                "AB_operator_norm": float(matrix["AB_operator_norm"]),
                "diagonal_delta_frobenius_norm": float(
                    matrix["diagonal_delta_frobenius_norm"]
                ),
                "J_condition_number": float(matrix["J_condition_number"]),
                "J_spectral_radius": float(matrix["J_spectral_radius"]),
            }
        )

    write_csv(args.out_dir / "transition_rows.csv", transition_rows)
    write_csv(args.out_dir / "control_rows.csv", control_rows)
    write_csv(args.out_dir / "seed_summary.csv", seed_summary)

    transition_index = {
        (int(row["backbone_seed"]), int(row["length"]), str(row["mode"])): row
        for row in transition_rows
    }
    control_index = {
        (int(row["backbone_seed"]), int(row["length"]), str(row["mode"])): row
        for row in control_rows
    }
    seeds = [run["seed"] for run in runs]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(
        "Addition LSB-causal-NoPE: frozen-backbone J multiseed replication",
        fontsize=16,
    )

    axis = axes[0, 0]
    for mode in ("raw", "full"):
        matrix = np.asarray(
            [
                [
                    transition_index[(seed, length, mode)]["supervised_digit_exact_match"]
                    for length in TRANSITION_LENGTHS
                ]
                for seed in seeds
            ],
            dtype=float,
        )
        axis.plot(
            TRANSITION_LENGTHS,
            matrix.mean(axis=0),
            marker="o",
            color=COLORS[mode],
            label=mode,
        )
        axis.fill_between(
            TRANSITION_LENGTHS,
            matrix.min(axis=0),
            matrix.max(axis=0),
            color=COLORS[mode],
            alpha=0.15,
        )
    axis.set(title="Transition band; mean and seed range", xlabel="m", ylabel="supervised-digit EM")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    for seed in seeds:
        delta = [
            transition_index[(seed, length, "full")]["supervised_digit_exact_match"]
            - transition_index[(seed, length, "raw")]["supervised_digit_exact_match"]
            for length in TRANSITION_LENGTHS
        ]
        axis.plot(TRANSITION_LENGTHS, delta, marker="o", label=f"seed {seed}")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(title="Per-backbone J effect", xlabel="m", ylabel="full EM - raw EM")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[1, 0]
    for mode in CONTROL_MODES:
        values = [
            np.mean(
                [
                    control_index[(seed, length, mode)]["supervised_digit_exact_match"]
                    for seed in seeds
                ]
            )
            for length in CONTROL_LENGTHS
        ]
        axis.plot(
            CONTROL_LENGTHS,
            values,
            marker="o",
            color=COLORS[mode],
            label=mode,
        )
    axis.set(title="Matched component controls", xlabel="m", ylabel="mean EM")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[1, 1]
    x = np.arange(len(seeds))
    width = 0.24
    axis.bar(
        x - width,
        [row["AB_frobenius_norm"] for row in seed_summary],
        width,
        label="||AB||F",
    )
    axis.bar(
        x,
        [row["AB_operator_norm"] for row in seed_summary],
        width,
        label="||AB||2",
    )
    axis.bar(
        x + width,
        [row["diagonal_delta_frobenius_norm"] for row in seed_summary],
        width,
        label="||D-I||F",
    )
    axis.set_xticks(x, [f"seed {seed}" for seed in seeds])
    axis.set(title="Learned controller scale", ylabel="norm")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    figure.savefig(args.out_dir / "addition_lsb_multiseed.png", dpi=180)
    figure.savefig(args.out_dir / "addition_lsb_multiseed.pdf")
    plt.close(figure)

    raw_transition_population = float(
        np.mean([row["raw_transition_mean_em"] for row in seed_summary])
    )
    full_transition_population = float(
        np.mean([row["full_transition_mean_em"] for row in seed_summary])
    )
    summary = {
        "status": "complete",
        "backbone_seeds": seeds,
        "transition_lengths": list(TRANSITION_LENGTHS),
        "transition_examples_per_seed_length": 8192,
        "population_raw_transition_mean_em": raw_transition_population,
        "population_full_transition_mean_em": full_transition_population,
        "population_transition_delta_em": (
            full_transition_population - raw_transition_population
        ),
        "seed_summary": seed_summary,
        "evidence_paths": {str(run["seed"]): run["paths"] for run in runs},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Addition LSB-causal-NoPE multiseed report",
        "",
        "All runs use native variable length, T(m)=m, ordinary-digit final-only CE, causal attention, no positional embedding, frozen backbones, and the same rank-48 identity-initialized 5,376-update J protocol.",
        "",
        "| seed | checkpoint | raw ID | J ID | raw transition | J transition | delta | q90 raw/J | q50 raw/J |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summary:
        lines.append(
            f"| {row['backbone_seed']} | {row['selected_step']} | {row['raw_id_mean_em']:.4f} | "
            f"{row['full_id_mean_em']:.4f} | {row['raw_transition_mean_em']:.4f} | "
            f"{row['full_transition_mean_em']:.4f} | {row['transition_delta_em']:+.4f} | "
            f"{row['raw_q90_frontier']}/{row['full_q90_frontier']} | "
            f"{row['raw_q50_frontier']}/{row['full_q50_frontier']} |"
        )
    lines.extend(
        [
            "",
            f"Population transition mean: raw {raw_transition_population:.5f}, J {full_transition_population:.5f}, delta {full_transition_population - raw_transition_population:+.5f}.",
            "",
            "![multiseed replication](addition_lsb_multiseed.png)",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
