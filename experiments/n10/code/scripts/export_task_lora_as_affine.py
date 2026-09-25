from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse a trained residual LoRA-J into explicit W,b maps."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if payload.get("kind") != "graph_path_telomere_task_lora_j":
        raise ValueError("unexpected source artifact kind")
    maps = {}
    summaries = {}
    for label, item in payload["modules"].items():
        state = item["state_dict"]
        left = state["A"].double()
        right = state["B"].double()
        bias = state["bias"].double()
        rank = int(item["rank"])
        dimension = int(item["dimension"])
        identity = torch.eye(dimension, dtype=torch.float64)
        parameterization = item.get("parameterization", "fixed_identity")
        diagonal_scale_statistics = None
        if parameterization == "scalar_low_rank":
            identity_scale = float(state["identity_scale"].item())
            correction = left @ right
            weight = identity_scale * identity + correction
        elif parameterization == "diagonal_low_rank":
            diagonal_scale = state["diagonal_scale"].double()
            identity_scale = None
            correction = left @ right
            weight = torch.diag(diagonal_scale) + correction
            diagonal_scale_statistics = {
                "mean": float(diagonal_scale.mean()),
                "std": float(diagonal_scale.std(unbiased=False)),
                "min": float(diagonal_scale.min()),
                "max": float(diagonal_scale.max()),
            }
        else:
            identity_scale = 1.0
            scale = float(item["alpha"]) / rank
            correction = scale * left @ right
            weight = identity + correction
        correction_singular = torch.linalg.svdvals(correction)
        identity_delta_singular = torch.linalg.svdvals(weight - identity)
        eigenvalues = torch.linalg.eigvals(weight)
        sign, logabsdet = torch.linalg.slogdet(weight)
        threshold = max(float(correction_singular[0]) * 1e-8, 1e-10)
        numerical_update_rank = int(correction_singular.gt(threshold).sum())
        maps[label] = {
            "weight": weight.float(),
            "bias": bias.float(),
            "rank": dimension,
            "source_factor_rank": rank,
            "source_initialization_seed": int(item["initialization_seed"]),
        }
        summaries[label] = {
            "source_factor_rank": rank,
            "source_initialization_seed": int(item["initialization_seed"]),
            "parameterization": parameterization,
            "identity_scale": identity_scale,
            "diagonal_scale_statistics": diagonal_scale_statistics,
            "collapsed_parameter_count": dimension * dimension + dimension,
            "numerical_correction_rank": numerical_update_rank,
            "correction_spectral_norm": float(correction_singular[0]),
            "correction_frobenius_norm": float(
                correction_singular.square().sum().sqrt()
            ),
            "identity_delta_spectral_norm": float(identity_delta_singular[0]),
            # Backwards-compatible aliases refer to the low-rank correction.
            "numerical_update_rank": numerical_update_rank,
            "delta_spectral_norm": float(correction_singular[0]),
            "delta_frobenius_norm": float(
                correction_singular.square().sum().sqrt()
            ),
            "bias_norm": float(bias.norm()),
            "weight_slogdet_sign": float(sign),
            "weight_logabsdet": float(logabsdet),
            "eigenvalue_abs_mean": float(eigenvalues.abs().mean()),
            "eigenvalue_abs_median": float(eigenvalues.abs().median()),
            "eigenvalues_within_0p1_of_zero": int(eigenvalues.abs().lt(0.1).sum()),
            "eigenvalues_within_0p1_of_one": int(
                (eigenvalues - 1).abs().lt(0.1).sum()
            ),
            "eigenvalue_abs_angle_median": float(eigenvalues.angle().abs().median()),
            "eigenvalue_abs_angle_max": float(eigenvalues.angle().abs().max()),
        }
    destination = {
        "kind": "graph_path_telomere_collapsed_affine_j",
        "checkpoint": payload["checkpoint"],
        "positions": payload["positions"],
        "placement": payload["placement"],
        "source_artifact": str(args.artifact),
        "loss_placement": (
            "frozen backbone final-only CE at loop 8; J successor CE only "
            "at every controlled continuation loop"
        ),
        "maps": maps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(destination, temporary)
    os.replace(temporary, args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "status": "complete",
                "source_artifact": str(args.artifact),
                "output_artifact": str(args.out),
                "checkpoint": payload["checkpoint"],
                "placement": payload["placement"],
                "maps": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
