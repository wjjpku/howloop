#!/usr/bin/env python3
"""Fail-closed audit for the isolated prospective Parity P2/P4 campaign.

This script intentionally accepts no partially completed label set.  It makes
the later appendix disposition reproducible: two pre-registered no-disease
seeds remain visible; the positive seed must contain all capacity fits, the
registered controls, and the P4 diagnostics linked to the selected rank-48
controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LABELS = (
    "rank48_seed1", "rank48_seed2", "rank128_seed1", "rank128_seed2",
    "dense_seed1", "dense_seed2",
)
COMPONENTS = (
    "no_AB", "D_only", "identity_D", "AB_only", "mean_D", "no_bias",
    "shuffle_D", "spectrum_matched_random_delta", "full_executor_off",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_complete(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing manifest: {path}")
        return None
    value = read(path)
    if value.get("status") != "complete":
        errors.append(f"manifest not complete: {path.name}={value.get('status')!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--out", type=Path,
        help="write an immutable machine-readable audit attestation; refuses to overwrite",
    )
    args = parser.parse_args()
    root = args.root
    namespace = args.namespace
    seed = args.seed
    errors: list[str] = []
    manifests = root / "manifests"

    # Gate accounting is part of the result rather than a pre-filter.
    for no_disease_seed in (3, 4):
        boundary_path = root / "evaluation" / f"seed{no_disease_seed}" / "prospective_boundary.json"
        if not boundary_path.is_file():
            errors.append(f"missing prospective boundary: {boundary_path}")
            continue
        boundary = read(boundary_path)
        if boundary.get("status") != "no_disease_detected":
            errors.append(f"seed {no_disease_seed} no-disease gate changed")
        for kind in ("controller", "evaluation"):
            path = manifests / f"p2_{kind}_seed{no_disease_seed}.json"
            if not path.is_file() or read(path).get("status") != "no_disease_skip":
                errors.append(f"seed {no_disease_seed} missing explicit no-disease {kind} skip")

    boundary_path = root / "evaluation" / f"seed{seed}" / "prospective_boundary.json"
    boundary = read(boundary_path) if boundary_path.is_file() else {}
    if not boundary_path.is_file():
        errors.append(f"missing prospective boundary: {boundary_path}")
    elif boundary.get("status") != "disease_detected":
        errors.append(f"seed {seed} is not a prospective disease-positive case")
    expected_range = boundary.get("training_range")
    checkpoint = root / "backbones" / f"parity_input_once_seed{seed}" / "final.pt"
    if not checkpoint.is_file():
        errors.append("positive-seed frozen checkpoint is absent")
    checkpoint_hash = sha256(checkpoint) if checkpoint.is_file() else None
    boundary_hash = sha256(boundary_path) if boundary_path.is_file() else None

    for kind in ("controller", "evaluation"):
        campaign = require_complete(manifests / f"p2_{kind}_{namespace}_seed{seed}.json", errors)
        if campaign is not None:
            labels = tuple(campaign.get("labels", ()))
            # The campaign records its physical parallel scheduling order; the
            # scientific preregistration is the duplicate-free label set.
            if len(labels) != len(LABELS) or set(labels) != set(LABELS):
                errors.append(f"{kind} campaign labels do not equal the registered label set")
        for label in LABELS:
            worker = require_complete(manifests / f"p2_{kind}_{namespace}_seed{seed}_{label}.json", errors)
            if worker is not None:
                if worker.get("checkpoint_sha256") != checkpoint_hash:
                    errors.append(f"{kind}/{label} checkpoint hash mismatch")
                if worker.get("raw_boundary_sha256") != boundary_hash:
                    errors.append(f"{kind}/{label} prospective boundary hash mismatch")

    controller_root = root / f"p2_controllers_{namespace}" / f"seed{seed}"
    evaluation_root = root / f"p2_evaluation_{namespace}" / f"seed{seed}"
    selected: dict[str, Path] = {}
    for label in LABELS:
        selection_path = controller_root / label / "selection.json"
        best = controller_root / label / "best_controller.pt"
        if not selection_path.is_file() or not best.is_file():
            errors.append(f"missing selection result for {label}")
            continue
        selection = read(selection_path)
        if selection.get("status") != "complete" or selection.get("checkpoint") != str(checkpoint):
            errors.append(f"invalid selection provenance for {label}")
        if selection.get("logical_training_range") != expected_range:
            errors.append(f"selection training range differs from prospective gate for {label}")
        selected[label] = best
        summary_path = evaluation_root / label / "summary.json"
        if not summary_path.is_file():
            errors.append(f"missing horizon summary for {label}")
            continue
        summary = read(summary_path)
        if summary.get("status") != "complete" or summary.get("paper_mode") is not True:
            errors.append(f"invalid horizon summary for {label}")
        if summary.get("checkpoint") != str(checkpoint) or summary.get("controller") != str(best):
            errors.append(f"horizon provenance mismatch for {label}")
        variants = {row.get("variant") for row in summary.get("rows", [])}
        if not {"raw", "full"} <= variants:
            errors.append(f"horizon lacks raw/full matched outputs for {label}")

    primary = selected.get("rank48_seed1")
    if primary is not None:
        for mode in COMPONENTS:
            directory = (
                evaluation_root / "rank48_seed1_executor_off"
                if mode == "full_executor_off"
                else evaluation_root / "rank48_seed1_components" / mode
            )
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                errors.append(f"missing component/executor control: {mode}")
                continue
            summary = read(summary_path)
            expected_variant = mode
            if summary.get("controller") != str(primary) or expected_variant not in {row.get("variant") for row in summary.get("rows", [])}:
                errors.append(f"invalid component/executor control provenance: {mode}")

    p4 = require_complete(manifests / f"p4_mechanism_{namespace}_seed{seed}.json", errors)
    if p4 is not None and primary is not None:
        if p4.get("checkpoint_sha256") != checkpoint_hash or p4.get("controller_sha256") != sha256(primary):
            errors.append("P4 checkpoint/controller hash mismatch")
        p4_root = root / f"p4_mechanism_{namespace}" / f"seed{seed}"
        for relative in ("four_phase/summary.json", "svd/summary.json", "hidden_effect/summary.json"):
            if not (p4_root / relative).is_file():
                errors.append(f"missing P4 artifact: {relative}")

    # Hash the exact artifacts that make the final P2 claims inspectable.  A
    # PASS in a log is not sufficient provenance: downstream paper figures
    # bind to this attestation and to their aggregate inputs.
    checked_paths: list[Path] = [boundary_path, checkpoint]
    for label in LABELS:
        checked_paths.extend([
            controller_root / label / "selection.json",
            controller_root / label / "best_controller.pt",
            evaluation_root / label / "summary.json",
            evaluation_root / label / "horizon.csv",
        ])
    for mode in COMPONENTS:
        directory = (
            evaluation_root / "rank48_seed1_executor_off"
            if mode == "full_executor_off"
            else evaluation_root / "rank48_seed1_components" / mode
        )
        checked_paths.extend([directory / "summary.json", directory / "horizon.csv"])
    p4_root = root / f"p4_mechanism_{namespace}" / f"seed{seed}"
    checked_paths.extend([
        p4_root / "four_phase" / "summary.json",
        p4_root / "svd" / "summary.json",
        p4_root / "hidden_effect" / "summary.json",
    ])
    artifact_hashes = {
        str(path.relative_to(root)): sha256(path)
        for path in checked_paths if path.is_file()
    }
    payload = {
        "protocol_id": "paper2027.parity.p2.p4.final_audit.v1",
        "namespace": namespace,
        "seed": seed,
        "label_count": len(LABELS),
        "checkpoint_sha256": checkpoint_hash,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checked_artifact_sha256": artifact_hashes,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite audit attestation: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
