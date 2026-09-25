"""Create trace-preserving and component-ablation controls for a J bank."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(830001, 830002, 830003, 830004, 830005),
    )
    return parser.parse_args()


def orthogonal_matrix(dimension: int, seed: int) -> torch.Tensor:
    """Return a deterministic float64 Haar-style orthogonal matrix."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(
        dimension,
        dimension,
        dtype=torch.float64,
        generator=generator,
    )
    q, r = torch.linalg.qr(sample)
    signs = torch.where(
        torch.diag(r) < 0,
        -torch.ones(dimension, dtype=torch.float64),
        torch.ones(dimension, dtype=torch.float64),
    )
    return q * signs.unsqueeze(0)


def shared_ab(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return state["shared_A"].double() @ state["shared_B"].double()


def component_metrics(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    diagonal = state["shared_diagonal_scale"].double()
    ab = shared_ab(state)
    singular = torch.linalg.svdvals(ab)
    return {
        "dimension": int(diagonal.numel()),
        "d_trace": float(diagonal.sum()),
        "d_update_trace": float((diagonal - 1).sum()),
        "d_update_frobenius": float(torch.linalg.vector_norm(diagonal - 1)),
        "d_min": float(diagonal.min()),
        "d_max": float(diagonal.max()),
        "ab_trace": float(torch.trace(ab)),
        "ab_frobenius": float(torch.linalg.matrix_norm(ab)),
        "ab_singular_values": [float(value) for value in singular],
    }


def invariant_delta(
    original: dict[str, Any],
    intervened: dict[str, Any],
) -> dict[str, float]:
    original_singular = torch.tensor(original["ab_singular_values"])
    intervened_singular = torch.tensor(intervened["ab_singular_values"])
    return {
        "d_trace_abs": abs(intervened["d_trace"] - original["d_trace"]),
        "d_update_frobenius_abs": abs(
            intervened["d_update_frobenius"]
            - original["d_update_frobenius"]
        ),
        "ab_trace_abs": abs(intervened["ab_trace"] - original["ab_trace"]),
        "ab_frobenius_abs": abs(
            intervened["ab_frobenius"] - original["ab_frobenius"]
        ),
        "ab_singular_max_abs": float(
            torch.max(torch.abs(intervened_singular - original_singular))
        ),
    }


def make_payload(
    source_payload: dict[str, Any],
    *,
    source_artifact: Path,
    name: str,
    seed: int | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(source_payload)
    payload["intervention"] = {
        "name": name,
        "seed": seed,
        "source_artifact": str(source_artifact),
        "claim_strength": "component causal-role control",
    }
    return payload


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def generate_interventions(
    bank_artifact: Path,
    out_dir: Path,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    source_payload = torch.load(
        bank_artifact,
        map_location="cpu",
        weights_only=False,
    )
    if source_payload.get("map_architecture") != "shared_diagonal_stage_lora":
        raise ValueError("trace interventions require shared_diagonal_stage_lora")
    source_state = source_payload["state_dict"]
    dimension = int(source_state["shared_diagonal_scale"].numel())
    source_metrics = component_metrics(source_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    def save(payload: dict[str, Any], filename: str) -> None:
        metrics = component_metrics(payload["state_dict"])
        deltas = invariant_delta(source_metrics, metrics)
        payload["intervention"]["source_metrics"] = source_metrics
        payload["intervention"]["intervened_metrics"] = metrics
        payload["intervention"]["invariant_deltas"] = deltas
        path = out_dir / filename
        atomic_torch_save(payload, path)
        manifest_rows.append(
            {
                "name": payload["intervention"]["name"],
                "seed": payload["intervention"]["seed"],
                "artifact": str(path),
                "metrics": metrics,
                "invariant_deltas": deltas,
            }
        )

    payload = make_payload(
        source_payload,
        source_artifact=bank_artifact,
        name="d_mean",
        seed=None,
    )
    diagonal = payload["state_dict"]["shared_diagonal_scale"]
    payload["state_dict"]["shared_diagonal_scale"] = torch.full_like(
        diagonal,
        diagonal.mean(),
    )
    save(payload, "d_mean.pt")

    payload = make_payload(
        source_payload,
        source_artifact=bank_artifact,
        name="d_identity",
        seed=None,
    )
    payload["state_dict"]["shared_diagonal_scale"] = torch.ones_like(
        payload["state_dict"]["shared_diagonal_scale"]
    )
    save(payload, "d_identity.pt")

    payload = make_payload(
        source_payload,
        source_artifact=bank_artifact,
        name="ab_zero",
        seed=None,
    )
    payload["state_dict"]["shared_B"] = torch.zeros_like(
        payload["state_dict"]["shared_B"]
    )
    save(payload, "ab_zero.pt")

    original_diagonal = source_state["shared_diagonal_scale"]
    original_a = source_state["shared_A"]
    original_b = source_state["shared_B"]
    original_ab = original_a.double() @ original_b.double()
    for seed in seeds:
        payload = make_payload(
            source_payload,
            source_artifact=bank_artifact,
            name="d_shuffle",
            seed=seed,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(dimension, generator=generator)
        payload["state_dict"]["shared_diagonal_scale"] = (
            original_diagonal[permutation].clone()
        )
        payload["intervention"]["permutation"] = permutation.tolist()
        save(payload, f"d_shuffle_seed{seed}.pt")

        payload = make_payload(
            source_payload,
            source_artifact=bank_artifact,
            name="ab_rotate",
            seed=seed,
        )
        q = orthogonal_matrix(dimension, seed)
        rotated_a = q.T @ original_a.double()
        rotated_b = original_b.double() @ q
        payload["state_dict"]["shared_A"] = rotated_a.to(original_a.dtype)
        payload["state_dict"]["shared_B"] = rotated_b.to(original_b.dtype)
        rotated_ab = shared_ab(payload["state_dict"])
        expected_ab = q.T @ original_ab @ q
        payload["intervention"]["orthogonality_max_abs"] = float(
            torch.max(torch.abs(q.T @ q - torch.eye(dimension, dtype=q.dtype)))
        )
        payload["intervention"]["similarity_relative_error"] = float(
            torch.linalg.matrix_norm(rotated_ab - expected_ab)
            / torch.linalg.matrix_norm(expected_ab)
        )
        save(payload, f"ab_rotate_seed{seed}.pt")

    manifest = {
        "status": "complete",
        "source_artifact": str(bank_artifact),
        "source_metrics": source_metrics,
        "seeds": list(seeds),
        "artifacts": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_rows


def main() -> None:
    args = parse_args()
    rows = generate_interventions(
        args.bank_artifact,
        args.out_dir,
        tuple(args.seeds),
    )
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
