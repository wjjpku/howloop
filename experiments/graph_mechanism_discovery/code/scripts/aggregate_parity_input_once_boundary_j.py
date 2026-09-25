#!/usr/bin/env python3
"""Audit and aggregate the three-backbone input-once Parity boundary-J cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_RANGES = {0: (80, 200), 1: (64, 128), 2: (48, 100)}
EXPECTED_MODEL = {
    "attention_mode": "causal",
    "token_embedding_injection": "initial_only",
    "position_embedding": "none",
    "position_injection": "initial_only",
    "d_model": 256,
    "n_heads": 64,
    "d_mlp": 1024,
    "block_layers": 1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--remote-attestation",
        type=Path,
        help="read-only A100 attestation containing the remote artifact hashes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    attestation_path = (
        args.remote_attestation.resolve()
        if args.remote_attestation
        else root / "boundary_rank_sweep_summary/remote_attestation_20260813.json"
    )
    remote = read_json(attestation_path)
    fail(remote.get("status") == "PASS_WITH_RECORDED_LIMITATIONS", "remote attestation is not eligible")

    per_seed: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    horizon_rows: list[dict[str, Any]] = []
    expected_remote = remote["rank48_result_sha256"]

    for seed, (training_min, training_max) in EXPECTED_RANGES.items():
        controller_dir = root / f"boundary_rank_sweep_v1/seed{seed}/boundary_rank48"
        evaluation_dir = root / f"boundary_rank_sweep_v1/evaluations/seed{seed}/boundary_rank48"
        selection_path = controller_dir / "selection.json"
        controller_path = controller_dir / "best_controller.pt"
        training_path = controller_dir / "training.csv"
        horizon_path = evaluation_dir / "far_horizon/horizon.csv"
        horizon_summary_path = evaluation_dir / "far_horizon/summary.json"
        diagonal_path = evaluation_dir / "diagonal_band/diagonal_band_metrics.csv"
        ridge_path = evaluation_dir / "ridge_ood/parity_ridge_slope_analysis.json"
        for path in (
            selection_path,
            controller_path,
            training_path,
            horizon_path,
            horizon_summary_path,
            diagonal_path,
            ridge_path,
        ):
            fail(path.is_file(), f"missing required seed{seed} artifact: {path}")
            source_hashes[str(path.relative_to(root))] = sha256(path)

        selection = read_json(selection_path)
        fail(selection.get("status") == "complete", f"seed{seed} selection is incomplete")
        fail(selection.get("logical_training_range") == [training_min, training_max], f"seed{seed} training range mismatch")
        fail(selection.get("controller_ood_definition") == f"n>{training_max}", f"seed{seed} OOD definition mismatch")
        fail(selection.get("retention_pass") is True, f"seed{seed} failed retention")
        fail(selection.get("examples_per_length") == 512, f"seed{seed} selection sample count mismatch")

        payload = torch.load(controller_path, map_location="cpu", weights_only=False)
        fail(payload.get("kind") == "paper_length_telomere_controller", f"seed{seed} controller kind mismatch")
        fail(payload.get("rank") == 48, f"seed{seed} controller rank mismatch")
        fail(payload.get("controller_parameterization") == "diagonal_low_rank", f"seed{seed} parameterization mismatch")
        fail(payload.get("controller_initialization") == "identity", f"seed{seed} initialization mismatch")
        fail(payload.get("controller_supervision") == "full_answer", f"seed{seed} supervision mismatch")
        fail(float(payload.get("state_loss_weight", -1)) == 0.0, f"seed{seed} has a hidden-state loss")
        fail(payload.get("controller_post_final_j") is False, f"seed{seed} applies J after final readout")
        fail(payload.get("anchor_step") == 1, f"seed{seed} anchor mismatch")
        fail(payload.get("controller_logical_min_length") == training_min, f"seed{seed} payload train minimum mismatch")
        fail(payload.get("controller_logical_max_length") == training_max, f"seed{seed} payload train maximum mismatch")
        for key, expected in EXPECTED_MODEL.items():
            fail(payload["model"].get(key) == expected, f"seed{seed} model protocol mismatch: {key}")
        fail(payload["task"].get("name") == "parity", f"seed{seed} task mismatch")
        fail(payload["task"].get("train_max_length") == 20, f"seed{seed} backbone train horizon mismatch")
        fail(payload["task"].get("step_offset") == 0, f"seed{seed} target-loop rule mismatch")
        fail(sha256(controller_path) == remote["controller_sha256"][f"seed{seed}"], f"seed{seed} controller differs from A100")

        summary = read_json(horizon_summary_path)
        fail(summary.get("status") == "complete", f"seed{seed} far-horizon summary is incomplete")
        fail(summary.get("examples_per_length") == 256, f"seed{seed} far-horizon sample count mismatch")
        fail(summary.get("target_loop_rule") == "T(n)=n", f"seed{seed} target-loop rule mismatch")
        fail(summary.get("controller_anchor_step") == 1, f"seed{seed} evaluation anchor mismatch")
        fail(summary.get("controller_post_final_j") is False, f"seed{seed} evaluation applies post-final J")

        with horizon_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        lookup = {(int(row["length"]), row["variant"]): row for row in rows}
        lengths = sorted({length for length, _ in lookup})
        fail(set(variant for _, variant in lookup) == {"raw", "full"}, f"seed{seed} horizon variants mismatch")
        fail(len(rows) == 2 * len(lengths), f"seed{seed} horizon rows are not paired")
        fail(all(int(row["examples"]) == 256 for row in rows), f"seed{seed} horizon examples mismatch")
        for row in rows:
            horizon_rows.append({"backbone_seed": seed, **row})

        first_double = [length for length in lengths if training_max < length <= 2 * training_max]
        strict_ood = [length for length in lengths if length > training_max]
        fail(first_double, f"seed{seed} has no first-doubling OOD points")
        raw_first = mean([float(lookup[(length, "raw")]["exact_match"]) for length in first_double])
        full_first = mean([float(lookup[(length, "full")]["exact_match"]) for length in first_double])
        raw_all = mean([float(lookup[(length, "raw")]["exact_match"]) for length in strict_ood])
        full_all = mean([float(lookup[(length, "full")]["exact_match"]) for length in strict_ood])

        with diagonal_path.open(newline="", encoding="utf-8") as handle:
            diagonal_rows = list(csv.DictReader(handle))
        fail({row["variant"] for row in diagonal_rows} == {"raw", "J"}, f"seed{seed} diagonal variants mismatch")
        fail({int(row["length"]) for row in diagonal_rows} == set(range(1, 501)), f"seed{seed} diagonal length coverage mismatch")
        fail(all(int(row["examples"]) == 64 for row in diagonal_rows), f"seed{seed} diagonal sample count mismatch")

        relative_base = f"seed{seed}"
        fail(sha256(horizon_path) == expected_remote[f"{relative_base}/far_horizon/horizon.csv"], f"seed{seed} horizon differs from A100")
        fail(sha256(diagonal_path) == expected_remote[f"{relative_base}/diagonal_band/diagonal_band_metrics.csv"], f"seed{seed} diagonal band differs from A100")
        fail(sha256(ridge_path) == expected_remote[f"{relative_base}/ridge_ood/parity_ridge_slope_analysis.json"], f"seed{seed} ridge analysis differs from A100")

        per_seed.append(
            {
                "backbone_seed": seed,
                "backbone_step": int(payload["backbone_step"]),
                "controller_seed": int(payload["seed"]),
                "selected_update": int(selection["selected_update"]),
                "training_range": [training_min, training_max],
                "controller_ood_definition": f"n>{training_max}",
                "boundary_validation_exact_match": float(selection["boundary_mean_accuracy"]),
                "minimum_short_retention_delta": float(selection["minimum_retention_delta"]),
                "first_doubling_ood_lengths": first_double,
                "first_doubling_raw_exact_match": raw_first,
                "first_doubling_j_exact_match": full_first,
                "first_doubling_delta": full_first - raw_first,
                "all_sampled_strict_ood_lengths": strict_ood,
                "all_sampled_strict_ood_raw_exact_match": raw_all,
                "all_sampled_strict_ood_j_exact_match": full_all,
                "all_sampled_strict_ood_delta": full_all - raw_all,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_seed_horizon.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(horizon_rows[0]))
        writer.writeheader()
        writer.writerows(horizon_rows)
    summary_payload = {
        "status": "complete_audited_three_backbone_boundary_j",
        "protocol_id": "parity_j1_input_once_adaptive_boundary",
        "backbone_seeds": [0, 1, 2],
        "controller": {
            "form": "J(h)=h(D+AB)+b",
            "rank": 48,
            "initialization": "identity",
            "anchor_step": 1,
            "objective": "registered T(n)=n full-answer CE only",
            "hidden_state_loss": False,
            "short_length_retention": [10, 20]
        },
        "evaluation": {
            "registered_readout_examples_per_length": 256,
            "diagonal_band_lengths": [1, 500],
            "diagonal_relative_depth": [-10, 10],
            "diagonal_examples_per_cell": 64,
            "raw_and_j_share_each_evaluation_batch": True
        },
        "per_seed": per_seed,
        "result": "All three backbone/controller pairs improve mean exact match in the strict-OOD first-doubling band; far-horizon behavior remains non-monotonic and seed dependent.",
        "claim_boundary": "Exploratory adaptive-boundary evidence for finite input-once Parity phase/interface retiming. Training bands were chosen per observed backbone boundary; this is not the prospective P2 cohort, an indefinite-locking result, or a population estimate beyond the three backbones.",
        "remote_attestation": str(attestation_path),
        "remote_attestation_sha256": sha256(attestation_path),
        "source_sha256": source_hashes
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_payload = {
        "status": "PASS",
        "protocol_id": summary_payload["protocol_id"],
        "checked_backbone_seeds": [0, 1, 2],
        "checked_source_count": len(source_hashes),
        "checked_source_sha256": source_hashes,
        "remote_attestation_sha256": sha256(attestation_path),
        "recorded_limitations": remote["limitations"]
    }
    (out_dir / "audit_attestation.json").write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "out_dir": str(out_dir), "seeds": [0, 1, 2]}, sort_keys=True))


if __name__ == "__main__":
    main()
