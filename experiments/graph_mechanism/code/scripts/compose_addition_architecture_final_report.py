from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compose(experiment_root: Path, checkpoint_root: Path, out_path: Path) -> None:
    checkpoint = json.loads((checkpoint_root / "aggregate.json").read_text())
    architecture = json.loads((experiment_root / "aggregate/aggregate.json").read_text())
    carry_gate = json.loads(
        (experiment_root / "balanced_carry_screen/aggregate/aggregate.json").read_text()
    )
    j_aggregate = json.loads(
        (experiment_root / "carry_j_round/aggregate/aggregate.json").read_text()
    )
    spectra = json.loads(
        (experiment_root / "carry_j_round/aggregate_spectra/aggregate.json").read_text()
    )
    controls = _csv(experiment_root / "carry_j_round/aggregate/control_summary.csv")
    aggregate_curves = _csv(
        experiment_root / "carry_j_round/aggregate/aggregate_curves.csv"
    )

    checkpoint_n10 = {
        int(row["checkpoint_step"]): row
        for row in checkpoint["target_rows"]
        if int(row["length"]) == 10
    }
    earliest_solved = next(
        step
        for step in sorted(checkpoint_n10)
        if float(checkpoint_n10[step]["target_actual_min"]) >= 0.98
    )
    full_n10: dict[str, dict[str, float]] = {}
    for row in aggregate_curves:
        if (
            row["dataset"] == "full_answer"
            and int(row["length"]) == 10
            and row["variant"] in {"raw", "full"}
        ):
            full_n10.setdefault(row["group"], {})[row["variant"]] = float(
                row["mean_exact_match"]
            )

    lines = [
        "# Fixed-n10 Addition baseline and recurrent-interface J study",
        "",
        "## Protocol",
        "",
        "Backbones are frozen shared-block looped Transformers. New architecture controls were trained only at logical length n=10 with final-only full answer-region CE. The registered paper endpoint is T(n)=n+1; T8/T6 controls were additionally evaluated at an endpoint-aligned rule. J uses the row-vector convention `h_out = h diag(D) + (h A)B + b = hJ+b`, rank 48, identity initialization, anchor 1, the same J before loops 2 onward, no post-final J, and balanced final-carry CE over k=1..10.",
        "",
        "## 1. Training amount",
        "",
        f"The first checkpoint where every seed reaches n=10 full sum-digit EM >=0.98 is {earliest_solved // 1000}k updates.",
        "",
        "| updates | n10 full EM | n10 token acc | n10 carry acc | n15 full EM |",
        "|---:|---:|---:|---:|---:|",
    ]
    for step in sorted(checkpoint_n10):
        n15 = next(
            row
            for row in checkpoint["target_rows"]
            if int(row["checkpoint_step"]) == step and int(row["length"]) == 15
        )
        row = checkpoint_n10[step]
        lines.append(
            f"| {step // 1000}k | {row['target_actual_mean']:.3f} | "
            f"{row['target_token_mean']:.3f} | {row['target_carry_mean']:.3f} "
            f"| {n15['target_actual_mean']:.3f} |"
        )
    ten_k = checkpoint_n10[10000]
    independent_product = float(ten_k["target_token_mean"]) ** 11
    lines.extend(
        [
            "",
            f"At 10k, token accuracy is {ten_k['target_token_mean']:.3f}; its independent-error product over 11 sum digits is {independent_product:.3f}, close to the observed EM {ten_k['target_actual_mean']:.3f}. Thus low EM is primarily compounded per-digit error, not evidence that every learned component is random. Carry is already {ten_k['target_carry_mean']:.3f}.",
            "",
            "## 2. Architecture, heads, and trained loop count at 10k",
            "",
            "| config | params | effective depth | full EM | token acc | carry acc | full-task eligible |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in architecture["configs"]:
        lines.append(
            f"| {row['label']} | {int(row['parameter_count']):,} | {row['effective_depth']} "
            f"| {row['id_aligned_actual_em_mean']:.3f} "
            f"| {row['id_aligned_token_accuracy_mean']:.3f} "
            f"| {row['id_aligned_carry_accuracy_mean']:.3f} "
            f"| {'yes' if row['eligible_all_seeds_id98'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "No 10k architecture is admitted as a full-Addition J backbone unless all three independently trained seeds pass the 0.98 full-answer gate. Carry-only eligibility is kept separate.",
            "",
            "## 3. Balanced carry gate",
            "",
            carry_gate["eligibility_rule"] + ".",
            "",
            "| group | mean n10 carry | minimum seed | eligible |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in carry_gate["per_group"]:
        if not row["complete_three_seed_group"]:
            continue
        lines.append(
            f"| {row['group']} | {row['mean_balanced_carry_accuracy_n10']:.3f} "
            f"| {row['minimum_balanced_carry_accuracy_n10']:.3f} "
            f"| {'yes' if row['eligible_for_carry_j'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## 4. J outcomes",
            "",
            "| group | seeds | raw ID1-10 | J ID1-10 | raw OOD11-50 | J OOD11-50 | raw H90 | J H90 | full n10 raw | full n10 J |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in j_aggregate["per_group"]:
        group = row["group"]
        lines.append(
            f"| {group} | {row['seed_count']} | {row['raw_id1to10_mean']:.3f} "
            f"| {row['j_id1to10_mean']:.3f} | {row['raw_ood11to50_mean']:.3f} "
            f"| {row['j_ood11to50_mean']:.3f} | {row['raw_prefix_horizon_90']:.1f} "
            f"| {row['j_prefix_horizon_90']:.1f} | {full_n10[group]['raw']:.3f} "
            f"| {full_n10[group]['full']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 5. Matched controls at n=20",
            "",
            "| group | raw | full J | no AB | D=I | no bias | executor off |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in controls:
        if int(row["length"]) != 20:
            continue
        lines.append(
            f"| {row['group']} | {float(row['raw_mean']):.3f} "
            f"| {float(row['full_mean']):.3f} | {float(row['no_AB_mean']):.3f} "
            f"| {float(row['identity_D_mean']):.3f} | {float(row['no_bias_mean']):.3f} "
            f"| {float(row['executor_off_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## 6. Matrix diagnostics",
            "",
            "| group | ||D-I||2 | ||AB||2 | stable-rank(J-I) | spectral-radius(J) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in spectra["groups"]:
        lines.append(
            f"| {row['group']} | {row['D_delta_l2_mean']:.4f} "
            f"| {row['AB_operator_norm_mean']:.3f} "
            f"| {row['delta_J_stable_rank_mean']:.2f} "
            f"| {row['J_spectral_radius_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Balanced final-carry accuracy identifies a carry subcircuit, not the full Addition algorithm. A J-induced carry gain is a behavioral component result only when it is seed-stable and survives matched controls; executor-off failure establishes continued dependence on the frozen executor. J-only spectra are diagnostics and do not replace the local trajectory Jacobian of F composed with J.",
            "",
            "Figures: `aggregate/architecture_round.png`, `carry_j_round/aggregate/carry_length_curves.png`, `carry_j_round/aggregate/carry_horizon90.png`, `carry_j_round/aggregate/carry_controls.png`, and `carry_j_round/aggregate_spectra/spectrum_comparison.png`.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    compose(args.experiment_root, args.checkpoint_root, args.out_path)


if __name__ == "__main__":
    main()
