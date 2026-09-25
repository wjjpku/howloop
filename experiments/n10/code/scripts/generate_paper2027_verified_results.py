#!/usr/bin/env python3
"""Generate TeX literals only from completed confirmatory aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-aggregate", type=Path, required=True)
    parser.add_argument("--parity-p1-aggregate", type=Path, required=True)
    parser.add_argument("--parity-p2-aggregate", type=Path)
    parser.add_argument("--parity-p2-audit", type=Path)
    parser.add_argument("--parity-p4-dir", type=Path)
    parser.add_argument("--parity-p3-dir", type=Path)
    parser.add_argument("--parity-j1-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph = read_json(args.graph_aggregate / "summary.json")
    parity = read_json(args.parity_p1_aggregate / "summary.json")
    if graph.get("status") != "complete" or parity.get("status") != "complete":
        raise ValueError("only complete confirmatory aggregates may generate paper literals")
    raw = graph["raw_strict_successor_auc_9_128"]
    full = graph["full_strict_successor_auc_9_128"]
    hold = graph["full_endpoint_hold_auc_9_128"]
    statistics = parity["statistics"]
    p100 = statistics["raw_exact_match_length_100"]
    auc = statistics["raw_exact_match_auc_lengths_24_to_500"]
    phase_rows = read_csv(args.parity_p1_aggregate / "phase_deep_seed_table.csv")
    macros = [
        macro("GraphGFourBackbones", str(graph["completed_backbone_count"])),
        macro("GraphGFourFinalGraphs", str(raw["graphs_per_replica"])),
        macro("GraphGFourRawSuccessorAUC", fmt(raw["point_estimate"])),
        macro("GraphGFourRawSuccessorAUCci", f"[{fmt(raw['ci95_low'])}, {fmt(raw['ci95_high'])}]"),
        macro("GraphGFourControllerSuccessorAUC", fmt(full["point_estimate"])),
        macro("GraphGFourControllerSuccessorAUCci", f"[{fmt(full['ci95_low'])}, {fmt(full['ci95_high'])}]"),
        macro("GraphGFourControllerHoldAUC", fmt(hold["point_estimate"])),
        macro("ParityPoneBackbones", str(len(parity["backbone_seeds"]))),
        macro("ParityPoneLengthHundred", fmt(p100["estimate"])),
        macro("ParityPoneLengthHundredCI", f"[{fmt(p100['ci_low'])}, {fmt(p100['ci_high'])}]"),
        macro("ParityPoneAUC", fmt(auc["estimate"])),
        macro("ParityPoneAUCci", f"[{fmt(auc['ci_low'])}, {fmt(auc['ci_high'])}]"),
        macro("ParityDeepSeeds", ", ".join(row["seed"] for row in phase_rows)),
        macro("ParityDeepPeriods", ", ".join(fmt(float(row["period_calls"]), 2) for row in phase_rows)),
        macro("ParityDeepPlaneEnergy", ", ".join(fmt(float(row["shared_phase_plane_energy_fraction"])) for row in phase_rows)),
    ]
    if args.parity_p2_aggregate is not None:
        if args.parity_p2_audit is None or args.parity_p4_dir is None:
            raise ValueError("P2 paper literals require its audit attestation and P4 directory")
        p2 = read_json(args.parity_p2_aggregate / "summary.json")
        if p2.get("status") != "complete" or not p2.get("final_audit_attestation_sha256"):
            raise ValueError("P2 aggregate is incomplete")
        gate = p2.get("gate_rows", [])
        positive = [row for row in gate if row.get("raw_boundary_status") == "disease_detected"]
        if len(positive) != len(p2.get("disease_positive_backbones", [])):
            raise ValueError("P2 gate accounting and positive-backbone list disagree")
        if len(positive) != 1:
            raise ValueError("this paper revision expects exactly one prospectively positive P2 backbone")
        positive_row = positive[0]
        if positive_row.get("training_min") is None or positive_row.get("training_max") is None:
            raise ValueError("P2 positive backbone lacks a fixed training range")
        audit = read_json(args.parity_p2_audit)
        if audit.get("status") != "PASS":
            raise ValueError("P2 paper literals require a PASS audit attestation")
        if p2.get("final_audit_attestation_sha256") != sha256(args.parity_p2_audit):
            raise ValueError("P2 aggregate is not bound to the supplied audit attestation")
        primary_rows = read_csv(args.parity_p2_aggregate / "primary_horizon.csv")
        capacity_rows = read_csv(args.parity_p2_aggregate / "capacity_horizon.csv")
        disease_length = int(positive_row["first_diseased_length"])
        rank48 = [
            float(row["exact_match"]) for row in primary_rows
            if row["controller"].startswith("rank48") and int(row["length"]) == disease_length
        ]
        dense = [
            float(row["exact_match"]) for row in capacity_rows
            if row["controller"].startswith("dense") and int(row["length"]) == disease_length
        ]
        if len(rank48) != 2 or len(dense) != 2:
            raise ValueError("P2 capacity table is incomplete at the disease gate")
        p4_phase = args.parity_p4_dir / "four_phase_summary.json"
        p4_svd = args.parity_p4_dir / "svd_summary.json"
        p4_hidden = args.parity_p4_dir / "hidden_effect_summary.json"
        for path in (p4_phase, p4_svd, p4_hidden):
            if not path.is_file():
                raise FileNotFoundError(path)
        expected_hashes = audit.get("checked_artifact_sha256", {})
        remote_map = {
            p4_phase: "p4_mechanism_parallel_v2/seed5/four_phase/summary.json",
            p4_svd: "p4_mechanism_parallel_v2/seed5/svd/summary.json",
            p4_hidden: "p4_mechanism_parallel_v2/seed5/hidden_effect/summary.json",
        }
        for path, remote in remote_map.items():
            if expected_hashes.get(remote) != sha256(path):
                raise ValueError(f"P4 artifact is not bound to the P2 audit: {path.name}")
        p4 = read_json(p4_phase)
        dynamics = p4["dynamics"]["controlled"]["heldout_evaluation"]
        macros.extend([
            macro("ParityPtwoPositiveBackbones", str(len(p2["disease_positive_backbones"]))),
            macro("ParityPtwoPositiveSeeds", ", ".join(map(str, p2["disease_positive_backbones"])) or "none"),
            macro("ParityPtwoPositiveSeed", str(positive_row["seed"])),
            macro("ParityPtwoDiseaseLength", str(positive_row["first_diseased_length"])),
            macro("ParityPtwoTrainRange", f"{positive_row['training_min']}--{positive_row['training_max']}"),
            macro("ParityPtwoGateBackbones", str(len(gate))),
            macro("ParityPtwoRankFortyEightAtDisease", fmt(max(rank48))),
            macro("ParityPtwoDenseMaximumAtDisease", fmt(max(dense))),
            macro("ParityPfourPeriod", fmt(dynamics["polar_rotation_period_calls"], 2)),
            macro("ParityPfourPlaneEnergy", fmt(dynamics["shared_phase_plane_energy_fraction"])),
            macro("ParityPfourTransitionRtwo", fmt(dynamics["transition_r2"])),
            macro("ParityPfourFourStepError", fmt(dynamics["scaled_four_step_relative_error"])),
        ])
    if args.parity_p3_dir is not None:
        p3_summary_path = args.parity_p3_dir / "summary.json"
        p3_attestation_path = args.parity_p3_dir / "source_attestation.json"
        if not p3_summary_path.is_file() or not p3_attestation_path.is_file():
            raise FileNotFoundError("P3 summary or source attestation is missing")
        p3 = read_json(p3_summary_path)
        attestation = read_json(p3_attestation_path)
        if p3.get("status") != "complete_registered_three_seed_null":
            raise ValueError("P3 must be a completed registered three-seed null result")
        if p3.get("backbone_seeds") != [3, 4, 5] or attestation.get("status") != "PASS":
            raise ValueError("P3 backbone cohort or source attestation is invalid")
        source_hashes = attestation.get("checked_source_sha256", {})
        if source_hashes != p3.get("source_sha256") or len(source_hashes) != 6:
            raise ValueError("P3 summary is not bound to all six source files")
        for raw, expected_hash in source_hashes.items():
            source = Path(raw)
            if not source.is_file() or sha256(source) != expected_hash:
                raise ValueError(f"P3 source hash mismatch: {source}")
        conditions = p3["means_by_condition"]
        rotation = conditions["phase_rotate_pi_answer"]
        random_rotation = conditions["random2d_rotation_pi_answer"]
        skips = p3["skip_means"]
        macros.extend([
            macro("ParityPthreeBackbones", str(len(p3["backbone_seeds"]))),
            macro("ParityPthreePhaseRotationRecovery", fmt(rotation["phase_translation_recovery"])),
            macro("ParityPthreeRandomRotationRecovery", fmt(random_rotation["phase_translation_recovery"])),
            macro("ParityPthreeMarginRotationRecovery", fmt(rotation["margin_translation_recovery"])),
            macro("ParityPthreeMarginRandomRecovery", fmt(random_rotation["margin_translation_recovery"])),
            macro("ParityPthreeMlpLag", fmt(skips["mlp_skip_fraction_closer_to_previous_phase"])),
            macro("ParityPthreeAttentionLag", fmt(skips["attention_skip_fraction_closer_to_previous_phase"])),
        ])
    if args.parity_j1_dir is not None:
        j1_summary_path = args.parity_j1_dir / "summary.json"
        j1_audit_path = args.parity_j1_dir / "audit_attestation.json"
        j1_remote_path = args.parity_j1_dir / "remote_attestation.json"
        for path in (j1_summary_path, j1_audit_path, j1_remote_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        j1 = read_json(j1_summary_path)
        j1_audit = read_json(j1_audit_path)
        j1_remote = read_json(j1_remote_path)
        if j1.get("status") != "complete_audited_three_backbone_boundary_j":
            raise ValueError("J1 aggregate is incomplete")
        if j1.get("backbone_seeds") != [0, 1, 2] or j1_audit.get("status") != "PASS":
            raise ValueError("J1 cohort or local audit attestation is invalid")
        remote_hash = sha256(j1_remote_path)
        if (
            j1.get("remote_attestation_sha256") != remote_hash
            or j1_audit.get("remote_attestation_sha256") != remote_hash
            or j1_remote.get("status") != "PASS_WITH_RECORDED_LIMITATIONS"
        ):
            raise ValueError("J1 aggregate is not bound to the remote attestation")
        per_seed = j1.get("per_seed", [])
        if [row.get("backbone_seed") for row in per_seed] != [0, 1, 2]:
            raise ValueError("J1 per-backbone records are incomplete or reordered")
        macros.extend([
            macro("ParityJoneBackbones", str(len(per_seed))),
            macro("ParityJoneSeedZeroRaw", fmt(per_seed[0]["first_doubling_raw_exact_match"])),
            macro("ParityJoneSeedZeroControlled", fmt(per_seed[0]["first_doubling_j_exact_match"])),
            macro("ParityJoneSeedOneRaw", fmt(per_seed[1]["first_doubling_raw_exact_match"])),
            macro("ParityJoneSeedOneControlled", fmt(per_seed[1]["first_doubling_j_exact_match"])),
            macro("ParityJoneSeedTwoRaw", fmt(per_seed[2]["first_doubling_raw_exact_match"])),
            macro("ParityJoneSeedTwoControlled", fmt(per_seed[2]["first_doubling_j_exact_match"])),
        ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("% Generated by generate_paper2027_verified_results.py; do not edit.\n" + "\n".join(macros) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
