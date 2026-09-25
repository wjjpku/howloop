#!/usr/bin/env python3
"""Fail-closed summary of preregistered Parity P2 controller outcomes.

P2 has a gate: each of the three prespecified deep backbones is either a
recorded no-disease control or contributes its two rank-48 controller fits.
Controller fits never count as independent backbone replications.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEEDS = (3, 4, 5)
LABELS = (
    "rank48_seed1", "rank48_seed2", "rank128_seed1", "rank128_seed2",
    "dense_seed1", "dense_seed2",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _status(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(read_json(path).get("status"))


def _horizon(path: Path, *, expected_variant: str) -> list[dict[str, str]]:
    rows = [row for row in read_csv(path) if row["variant"] == expected_variant]
    if not rows:
        raise ValueError(f"missing {expected_variant!r} rows in {path}")
    return rows


def aggregate(
    root: Path,
    seeds: tuple[int, ...],
    *,
    namespace: str = "",
    audit_attestation: Path | None = None,
) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    eligible: list[int] = []
    audit: dict[str, Any] | None = None
    if namespace:
        if audit_attestation is None or not audit_attestation.is_file():
            raise ValueError("isolated P2 aggregation requires a completed final audit attestation")
        audit = read_json(audit_attestation)
        if (audit.get("status") != "PASS" or audit.get("namespace") != namespace
                or audit.get("protocol_id") != "paper2027.parity.p2.p4.final_audit.v1"):
            raise ValueError("P2 final audit attestation is not a PASS for this namespace")
    for seed in seeds:
        boundary = read_json(root / "evaluation" / f"seed{seed}" / "prospective_boundary.json")
        suffix = f"_{namespace}" if namespace else ""
        if boundary.get("status") == "no_disease_detected":
            p2_status = _status(root / "manifests" / f"p2_controller_seed{seed}.json")
            evaluation_status = _status(root / "manifests" / f"p2_evaluation_seed{seed}.json")
        else:
            p2_status = _status(root / "manifests" / f"p2_controller_{namespace}_seed{seed}.json")
            evaluation_status = _status(root / "manifests" / f"p2_evaluation_{namespace}_seed{seed}.json")
        gate_rows.append(
            {
                "seed": seed,
                "raw_boundary_status": boundary.get("status"),
                "first_diseased_length": boundary.get("first_diseased_length"),
                "training_min": (boundary.get("training_range") or [None, None])[0],
                "training_max": (boundary.get("training_range") or [None, None])[1],
                "p2_controller_status": p2_status,
                "p2_evaluation_status": evaluation_status,
            }
        )
        if boundary.get("status") == "no_disease_detected":
            if (p2_status, evaluation_status) != ("no_disease_skip", "no_disease_skip"):
                raise ValueError(f"no-disease seed {seed} was not retained as explicit skips")
            continue
        if boundary.get("status") != "disease_detected":
            raise ValueError(f"unknown P2 boundary status for seed {seed}")
        if not namespace:
            raise ValueError("disease-positive P2 aggregation requires an isolated namespace")
        if (p2_status, evaluation_status) != ("complete", "complete"):
            raise ValueError(f"disease-positive P2 seed {seed} is incomplete")
        for kind in ("controller", "evaluation"):
            campaign = read_json(root / "manifests" / f"p2_{kind}_{namespace}_seed{seed}.json")
            observed_labels = tuple(campaign.get("labels", ()))
            if (campaign.get("status") != "complete" or len(observed_labels) != len(LABELS)
                    or set(observed_labels) != set(LABELS)):
                raise ValueError(f"P2 {kind} campaign manifest is incomplete or has an unexpected label set")
            for label in LABELS:
                worker = root / "manifests" / f"p2_{kind}_{namespace}_seed{seed}_{label}.json"
                if _status(worker) != "complete":
                    raise ValueError(f"P2 {kind} worker is incomplete: {label}")
        eligible.append(seed)
        controller_root = root / f"p2_controllers{suffix}" / f"seed{seed}"
        base = root / f"p2_evaluation{suffix}" / f"seed{seed}"
        raw_reference: dict[int, float] | None = None
        for label in LABELS:
            rows = read_csv(base / label / "horizon.csv")
            full = [row for row in rows if row["variant"] == "full"]
            raw = [row for row in rows if row["variant"] == "raw"]
            if not full or not raw:
                raise ValueError(f"missing raw/full P2 horizon rows for {seed}/{label}")
            current_raw = {int(row["length"]): float(row["exact_match"]) for row in raw}
            if audit is not None:
                expected = audit.get("checked_artifact_sha256", {})
                for path in (
                    controller_root / label / "selection.json",
                    controller_root / label / "best_controller.pt",
                    base / label / "summary.json",
                    base / label / "horizon.csv",
                ):
                    relative = str(path.relative_to(root))
                    if expected.get(relative) != sha256(path):
                        raise ValueError(f"P2 audit-attested artifact changed or was not attested: {relative}")
            if raw_reference is None:
                raw_reference = current_raw
            elif raw_reference != current_raw:
                raise ValueError(f"P2 raw matched samples disagree across controller fits for seed {seed}")
            target = primary_rows if label.startswith("rank48") else capacity_rows
            for row in full:
                target.append({"seed": seed, "controller": label, "variant": "full", **row})
        assert raw_reference is not None
        for length, exact in sorted(raw_reference.items()):
            primary_rows.append({"seed": seed, "controller": "raw", "variant": "raw", "length": length, "exact_match": exact})
        for mode in ("no_AB", "D_only", "identity_D", "AB_only", "mean_D", "no_bias", "shuffle_D", "spectrum_matched_random_delta", "full_executor_off"):
            directory = "rank48_seed1_executor_off" if mode == "full_executor_off" else f"rank48_seed1_components/{mode}"
            expected = "full_executor_off" if mode == "full_executor_off" else mode
            component_summary = base / directory / "summary.json"
            component_horizon = base / directory / "horizon.csv"
            if audit is not None:
                expected_hashes = audit.get("checked_artifact_sha256", {})
                for path in (component_summary, component_horizon):
                    relative = str(path.relative_to(root))
                    if expected_hashes.get(relative) != sha256(path):
                        raise ValueError(f"P2 audit-attested artifact changed or was not attested: {relative}")
            for row in _horizon(component_horizon, expected_variant=expected):
                component_rows.append({"seed": seed, "mode": mode, **row})
    if len(gate_rows) != len(seeds):
        raise AssertionError("P2 gate accounting is incomplete")
    return {
        "status": "complete",
        "protocol_id": "paper2027.parity.p2.prospective_boundary.v1",
        "namespace": namespace or "default",
        "final_audit_attestation": str(audit_attestation) if audit_attestation else None,
        "final_audit_attestation_sha256": sha256(audit_attestation) if audit_attestation else None,
        "deep_backbone_seeds": list(seeds),
        "disease_positive_backbones": eligible,
        "gate_rows": gate_rows,
        "evidence_boundary": "P2 controller replicas are nested optimization repeats. No-disease seeds remain in the gate table; controller efficacy is reported per eligible backbone rather than pooled as a backbone population estimate.",
        "tables": {
            "gate": "gate_table.csv",
            "primary": "primary_horizon.csv",
            "capacity": "capacity_horizon.csv",
            "components": "component_horizon.csv",
        },
        "_rows": {
            "gate": gate_rows,
            "primary": primary_rows,
            "capacity": capacity_rows,
            "components": component_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--namespace", default="", help="isolated completed P2 campaign namespace")
    parser.add_argument("--audit-attestation", type=Path)
    args = parser.parse_args()
    result = aggregate(
        args.root, tuple(args.seeds), namespace=args.namespace,
        audit_attestation=args.audit_attestation,
    )
    rows = result.pop("_rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / result["tables"]["gate"], rows["gate"])
    write_csv(args.out_dir / result["tables"]["primary"], rows["primary"])
    write_csv(args.out_dir / result["tables"]["capacity"], rows["capacity"])
    write_csv(args.out_dir / result["tables"]["components"], rows["components"])
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
