"""Align J stage detector/writer subspaces with frozen attention Q/K/V weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _energy(weight: torch.Tensor, basis: torch.Tensor) -> float:
    return float((weight @ basis).square().sum() / weight.square().sum().clamp_min(1e-30))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, torch.device("cpu"))
    payload = torch.load(args.bank_artifact, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    rows: list[dict[str, Any]] = []
    bottom_rows: list[dict[str, Any]] = []
    diagonal = np.diag(state["shared_diagonal_scale"].double().numpy())
    shared = (
        state["shared_A"].double().numpy()
        @ state["shared_B"].double().numpy()
    )
    for source_age in range(2, 9):
        stage = (
            state[f"stage_A.{source_age}"].double()
            @ state[f"stage_B.{source_age}"].double()
        ).numpy()
        left, _, right_t = np.linalg.svd(stage, full_matrices=True)
        named_bases = {
            "stage_input_detector": torch.from_numpy(left[:, :16]),
            "stage_output_writer": torch.from_numpy(right_t.T[:, :16]),
        }
        full = diagonal + shared + stage
        full_left, _, full_right_t = np.linalg.svd(full, full_matrices=True)
        bottom_bases = {
            rank: {
                "J_bottom_input": torch.from_numpy(full_left[:, -rank:]),
                "J_bottom_output": torch.from_numpy(full_right_t.T[:, -rank:]),
            }
            for rank in (8, 16)
        }
        for block_index, block in enumerate(model.blocks):
            qkv = block.attn.qkv.weight.detach().double().reshape(
                3, cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.d_model
            )
            for component_index, component in enumerate(("q", "k", "v")):
                for head in range(cfg.n_heads):
                    for basis_name, basis in named_bases.items():
                        fraction = _energy(qkv[component_index, head], basis)
                        rows.append(
                            {
                                "source_age": source_age,
                                "user_J": source_age - 1,
                                "block": block_index + 1,
                                "head": head,
                                "component": component,
                                "basis": basis_name,
                                "rank": 16,
                                "weight_energy_fraction": fraction,
                                "enrichment_over_rank_fraction": fraction / (16 / cfg.d_model),
                            }
                        )
                    for rank, rank_bases in bottom_bases.items():
                        for basis_name, basis in rank_bases.items():
                            fraction = _energy(qkv[component_index, head], basis)
                            bottom_rows.append(
                                {
                                    "source_age": source_age,
                                    "user_J": source_age - 1,
                                    "block": block_index + 1,
                                    "head": head,
                                    "component": component,
                                    "basis": basis_name,
                                    "rank": rank,
                                    "weight_energy_fraction": fraction,
                                    "enrichment_over_rank_fraction": fraction
                                    / (rank / cfg.d_model),
                                }
                            )
    _write_csv(args.out_dir / "stage_subspace_qkv_alignment.csv", rows)
    _write_csv(args.out_dir / "matrix_bottom_qkv_alignment.csv", bottom_rows)
    selected = [
        row
        for row in rows
        if row["block"] == 2 and row["head"] == 0
    ]
    highlights: dict[str, float] = {}
    for basis in ("stage_input_detector", "stage_output_writer"):
        for component in ("q", "k", "v"):
            values = [
                float(row["enrichment_over_rank_fraction"])
                for row in selected
                if row["basis"] == basis and row["component"] == component
            ]
            highlights[f"B2_H0_{basis}_{component}_enrichment_mean"] = float(
                np.mean(values)
            )
            highlights[f"B2_H0_{basis}_{component}_enrichment_max"] = float(
                np.max(values)
            )
    selected_bottom = [
        row
        for row in bottom_rows
        if row["block"] == 2 and row["head"] == 0
    ]
    for rank in (8, 16):
        for basis in ("J_bottom_input", "J_bottom_output"):
            for component in ("q", "k", "v"):
                values = [
                    float(row["enrichment_over_rank_fraction"])
                    for row in selected_bottom
                    if row["rank"] == rank
                    and row["basis"] == basis
                    and row["component"] == component
                ]
                highlights[
                    f"B2_H0_{basis}_rank{rank}_{component}_enrichment_mean"
                ] = float(np.mean(values))
                highlights[
                    f"B2_H0_{basis}_rank{rank}_{component}_enrichment_max"
                ] = float(np.max(values))
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "highlights": highlights,
        "claim_boundary": (
            "Weight-space enrichment shows a compatible write/read interface, not causal "
            "mediation. Causal claims require the separate matched attention patching results."
        ),
    }
    (args.out_dir / "qkv_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
