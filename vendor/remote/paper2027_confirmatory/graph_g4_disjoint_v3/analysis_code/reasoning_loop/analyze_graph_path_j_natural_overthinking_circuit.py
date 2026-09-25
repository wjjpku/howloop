"""Test whether natural long-J failure is mediated by the same attention circuit.

For every F immediately following J, each example is matched to its earliest
previous visit with the same permutation graph, original query, current node,
and logical age.  The long trajectory is compared with, and optionally patched
from, this younger visit.  No graph token, answer, or task variable changes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_attention_circuit import (
    ForwardRecord,
    PatchCase,
    activation_rows_for_pair,
    aggregate_numeric,
    behavior_row,
    build_patch_cases,
    build_words,
    clean_rollout,
    currents_by_forward,
    final_state_from_trace,
    receiver_rollout,
    relative_rms,
    write_csv,
)
from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    atomic_json,
    load_bank_and_bases,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalSiteTrace,
    FunctionalTrace,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=(848001, 848002))
    parser.add_argument(
        "--back-counts", type=int, nargs="+", default=(8, 16, 24, 32, 48)
    )
    parser.add_argument("--path-pairs-per-k", type=int, default=1)
    parser.add_argument("--word-seed", type=int, default=848701)
    parser.add_argument(
        "--patch-labels",
        nargs="+",
        default=(
            "B2.H0.qkv",
            "B2.H0.pattern",
            "B2.H0.context",
            "B2.attention_out",
            "B2.mlp_out",
        ),
    )
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _gather_examples(
    values: Sequence[torch.Tensor], source_indices: torch.Tensor
) -> torch.Tensor:
    stacked = torch.stack(tuple(values), dim=0)
    batch = torch.arange(source_indices.numel(), device=source_indices.device)
    return stacked[source_indices, batch]


def _gather_trace(
    records: Sequence[ForwardRecord], source_indices: torch.Tensor
) -> FunctionalTrace:
    sites: list[FunctionalSiteTrace] = []
    tensor_fields = (
        "hidden_in",
        "q",
        "k",
        "v",
        "attention_pattern",
        "head_context",
        "attention_out",
        "residual_mid",
        "mlp_hidden",
        "mlp_out",
        "hidden_out",
    )
    for site_index in range(len(records[0].trace.sites)):
        template = records[0].trace.sites[site_index]
        gathered = {
            field: _gather_examples(
                [getattr(record.trace.sites[site_index], field) for record in records],
                source_indices,
            )
            for field in tensor_fields
        }
        sites.append(
            FunctionalSiteTrace(
                loop_index=template.loop_index,
                block_index=template.block_index,
                **gathered,
            )
        )
    return FunctionalTrace(
        sites=sites,
        logits_by_loop=_gather_examples(
            [record.trace.logits_by_loop for record in records], source_indices
        ),
    )


@torch.no_grad()
def build_matched_reference_records(
    *,
    model,
    input_currents: Sequence[torch.Tensor],
    output_currents: Sequence[torch.Tensor],
    trajectory_records: Sequence[ForwardRecord],
) -> tuple[list[ForwardRecord], list[dict[str, float]]]:
    if not (
        len(input_currents) == len(output_currents) == len(trajectory_records)
    ):
        raise ValueError("reference inputs and trajectory records must align")
    reference_records: list[ForwardRecord] = []
    step_rows: list[dict[str, float]] = []
    for forward_index, trajectory in enumerate(trajectory_records):
        age = trajectory.logical_age_before
        batch_size = input_currents[forward_index].shape[0]
        source_indices = torch.full(
            (batch_size,), forward_index, dtype=torch.long,
            device=input_currents[forward_index].device,
        )
        for earlier_index in range(forward_index):
            if trajectory_records[earlier_index].logical_age_before != age:
                continue
            matches = input_currents[earlier_index].eq(
                input_currents[forward_index]
            )
            still_unmatched = source_indices.eq(forward_index)
            source_indices[matches & still_unmatched] = earlier_index
        reference_trace = _gather_trace(trajectory_records, source_indices)
        reference_state = _gather_examples(
            [record.state for record in trajectory_records], source_indices
        )
        reference_logits = logits_from_raw_state(model, reference_state)
        trajectory_logits = logits_from_raw_state(model, trajectory.state)
        target = output_currents[forward_index]
        matched = source_indices.ne(forward_index)
        reference_records.append(
            ForwardRecord(
                trace=reference_trace,
                state=reference_state,
                logical_age_before=age,
                follows_j=trajectory.follows_j,
                rollback_count=trajectory.rollback_count,
            )
        )
        if trajectory.follows_j:
            step_rows.append(
                {
                    "forward_index": float(forward_index),
                    "rollback_index": float(trajectory.rollback_count),
                    "logical_age_before_F": float(age),
                    "matched_fraction": float(matched.float().mean()),
                    "reference_accuracy": float(
                        reference_logits.argmax(-1).eq(target).float().mean()
                    ),
                    "trajectory_accuracy": float(
                        trajectory_logits.argmax(-1).eq(target).float().mean()
                    ),
                }
            )
    return reference_records, step_rows


def plot_horizon(rows: Sequence[dict[str, Any]], path: Path) -> None:
    baselines = [
        row
        for row in rows
        if row["condition"] == "natural_J"
        and row["patch_label"] == "none"
    ]
    patches = [row for row in rows if row["patch_label"] != "none"]
    if not baselines:
        return
    figure, axis = plt.subplots(figsize=(9, 5), dpi=180)
    x = [int(row["back_count"]) for row in baselines]
    axis.plot(x, [float(row["accuracy"]) for row in baselines], "o-", label="natural J")
    for label in sorted({str(row["patch_label"]) for row in patches}):
        selected = [row for row in patches if row["patch_label"] == label]
        axis.plot(
            [int(row["back_count"]) for row in selected],
            [float(row["accuracy"]) for row in selected],
            "o--",
            label=f"young-match patch {label}",
        )
    axis.set(
        xlabel="number of J calls",
        ylabel="final accuracy",
        ylim=(0, 1.03),
        title="Natural overthinking and age-matched circuit rescue",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def coordinate_rows_for_pair(
    *,
    reference_records: Sequence[ForwardRecord],
    trajectory_records: Sequence[ForwardRecord],
    named_bases: dict[str, torch.Tensor],
    graph_seed: int,
    batch_index: int,
    word_label: str,
    back_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for forward_index, (reference, trajectory) in enumerate(
        zip(reference_records, trajectory_records, strict=True)
    ):
        if not trajectory.follows_j:
            continue
        reference_state = reference.trace.sites[0].hidden_in.float()
        trajectory_state = trajectory.trace.sites[0].hidden_in.float()
        for position_group, positions in (
            ("answer", (-1,)),
            ("all", tuple(range(reference_state.shape[1]))),
        ):
            ref_selected = reference_state[:, list(positions)]
            traj_selected = trajectory_state[:, list(positions)]
            for basis_name, basis in named_bases.items():
                ref_coordinates = ref_selected @ basis
                traj_coordinates = traj_selected @ basis
                ref_rms = float(ref_coordinates.square().mean().sqrt())
                traj_rms = float(traj_coordinates.square().mean().sqrt())
                rows.append(
                    {
                        "graph_seed": graph_seed,
                        "batch": batch_index,
                        "word": word_label,
                        "back_count": back_count,
                        "forward_index": forward_index,
                        "rollback_index": trajectory.rollback_count,
                        "logical_age_before_F": trajectory.logical_age_before,
                        "position_group": position_group,
                        "basis": basis_name,
                        "reference_coordinate_rms": ref_rms,
                        "trajectory_coordinate_rms": traj_rms,
                        "coordinate_rms_ratio": traj_rms / max(ref_rms, 1e-12),
                        "coordinate_relative_rms": relative_rms(
                            traj_coordinates, ref_coordinates
                        ),
                        "full_state_rms_ratio": float(
                            traj_selected.square().mean().sqrt()
                            / ref_selected.square().mean().sqrt().clamp_min(1e-12)
                        ),
                    }
                )
    return rows


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    bank,
    words: Sequence[tuple[int, str, tuple[int, ...]]],
    patch_cases: Sequence[PatchCase],
    named_bases: dict[str, torch.Tensor],
    subspace_patch_bases: dict[str, torch.Tensor],
    graph_seeds: Sequence[int],
    examples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch_size")
    positions = tuple(range(cfg.seq_len))
    activation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    for graph_seed in graph_seeds:
        set_seed(int(graph_seed))
        for batch_index in range(examples // batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            h1 = model.apply_loop(raw, loop_index=0)
            h1_current = advance_nodes(successors, start, steps=1)
            for back_count, word_label, actions in words:
                trajectory_records, trajectory_state = clean_rollout(
                    model=model,
                    bank=bank,
                    h1=h1,
                    actions=actions,
                    positions=positions,
                )
                input_currents, output_currents, final_target = currents_by_forward(
                    successors=successors,
                    h1_current=h1_current,
                    actions=actions,
                )
                reference_records, local_steps = build_matched_reference_records(
                    model=model,
                    input_currents=input_currents,
                    output_currents=output_currents,
                    trajectory_records=trajectory_records,
                )
                for row in local_steps:
                    step_rows.append(
                        {
                            "graph_seed": int(graph_seed),
                            "batch": batch_index,
                            "word": word_label,
                            "back_count": back_count,
                            **row,
                        }
                    )
                activation_rows.extend(
                    activation_rows_for_pair(
                        clean_records=reference_records,
                        receiver_records=trajectory_records,
                        condition="natural_long_vs_young_match",
                        cfg=cfg,
                        input_current_by_forward=input_currents,
                        output_current_by_forward=output_currents,
                        graph_seed=int(graph_seed),
                        word_label=word_label,
                        back_count=back_count,
                        batch_index=batch_index,
                    )
                )
                coordinate_rows.extend(
                    coordinate_rows_for_pair(
                        reference_records=reference_records,
                        trajectory_records=trajectory_records,
                        named_bases=named_bases,
                        graph_seed=int(graph_seed),
                        batch_index=batch_index,
                        word_label=word_label,
                        back_count=back_count,
                    )
                )
                behavior_rows.append(
                    behavior_row(
                        state=trajectory_state,
                        model=model,
                        target=final_target,
                        graph_seed=int(graph_seed),
                        batch_index=batch_index,
                        word=word_label,
                        back_count=back_count,
                        condition="natural_J",
                        patch_label="none",
                        direction="none",
                    )
                )
                for patch_case in patch_cases:
                    patched_state, _, _ = receiver_rollout(
                        model=model,
                        bank=bank,
                        h1=h1,
                        actions=actions,
                        positions=positions,
                        deltas=None,
                        donor_records=reference_records,
                        patch_case=patch_case,
                        patch_scope="post_j",
                    )
                    behavior_rows.append(
                        behavior_row(
                            state=patched_state,
                            model=model,
                            target=final_target,
                            graph_seed=int(graph_seed),
                            batch_index=batch_index,
                            word=word_label,
                            back_count=back_count,
                            condition="natural_J",
                            patch_label=patch_case.label,
                            direction="young_match_rescue",
                        )
                    )
                for basis_name, basis in subspace_patch_bases.items():
                    patched_state, _, _ = receiver_rollout(
                        model=model,
                        bank=bank,
                        h1=h1,
                        actions=actions,
                        positions=positions,
                        deltas=None,
                        donor_records=reference_records,
                        input_subspace_basis=basis,
                        patch_scope="post_j",
                    )
                    behavior_rows.append(
                        behavior_row(
                            state=patched_state,
                            model=model,
                            target=final_target,
                            graph_seed=int(graph_seed),
                            batch_index=batch_index,
                            word=word_label,
                            back_count=back_count,
                            condition="natural_J",
                            patch_label=f"loop_input.{basis_name}.rank8",
                            direction="young_match_subspace_rescue",
                        )
                    )
    return activation_rows, behavior_rows, step_rows, coordinate_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {
            "status": "running",
            "checkpoint": str(args.checkpoint),
            "bank_artifact": str(args.bank_artifact),
            "pid": os.getpid(),
        },
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        fraction = float(os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.12"))
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    bank, _, common_bases, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (8,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    cases = {case.label: case for case in build_patch_cases(cfg)}
    missing = sorted(set(args.patch_labels) - set(cases))
    if missing:
        raise ValueError(f"unknown patch labels: {missing}")
    patch_cases = [cases[label] for label in args.patch_labels]
    words = build_words(
        back_counts=args.back_counts,
        path_pairs_per_k=args.path_pairs_per_k,
        seed=args.word_seed,
    )
    named_bases = {
        name: torch.as_tensor(
            common_bases[name][8], dtype=torch.float32, device=device
        )
        for name in ("bottom_input", "bottom_output", "top_output")
    }
    random_array, _ = np.linalg.qr(
        np.random.default_rng(848901).standard_normal((cfg.d_model, 8))
    )
    subspace_patch_bases = {
        **named_bases,
        "random": torch.as_tensor(
            random_array[:, :8], dtype=torch.float32, device=device
        ),
    }
    full_input_patch = PatchCase(
        label="B1.block_input_all",
        interventions=(
            FunctionalIntervention(
                site=0,
                component="block_input",
                mode="patch",
                positions=tuple(range(cfg.seq_len)),
            ),
        ),
    )
    patch_cases.append(full_input_patch)
    activation_rows, behavior_rows, step_rows, coordinate_rows = evaluate(
        model=model,
        cfg=cfg,
        bank=bank,
        words=words,
        patch_cases=patch_cases,
        named_bases=named_bases,
        subspace_patch_bases=subspace_patch_bases,
        graph_seeds=args.graph_seeds,
        examples=args.examples,
        batch_size=args.batch_size,
        device=device,
    )
    activation_summary = aggregate_numeric(
        activation_rows, ("back_count", "condition", "block", "head")
    )
    behavior_summary = aggregate_numeric(
        behavior_rows,
        ("back_count", "condition", "patch_label", "direction"),
    )
    step_summary = aggregate_numeric(
        step_rows, ("back_count", "rollback_index", "logical_age_before_F")
    )
    coordinate_summary = aggregate_numeric(
        coordinate_rows, ("back_count", "position_group", "basis")
    )
    write_csv(args.out_dir / "activation_rows.csv", activation_rows)
    write_csv(args.out_dir / "activation_summary.csv", activation_summary)
    write_csv(args.out_dir / "behavior_rows.csv", behavior_rows)
    write_csv(args.out_dir / "behavior_summary.csv", behavior_summary)
    write_csv(args.out_dir / "step_rows.csv", step_rows)
    write_csv(args.out_dir / "step_summary.csv", step_summary)
    write_csv(args.out_dir / "coordinate_rows.csv", coordinate_rows)
    write_csv(args.out_dir / "coordinate_summary.csv", coordinate_summary)
    plot_horizon(behavior_summary, args.out_dir / "natural_overthinking_rescue.png")
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "back_counts": list(args.back_counts),
        "path_pairs_per_k": args.path_pairs_per_k,
        "patch_cases": [case.label for case in patch_cases],
        "reference_definition": (
            "per-example earliest previous visit with identical graph, original "
            "query tokens, current node, and logical age"
        ),
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "claim_boundary": (
            "Unmatched examples are patched from themselves and therefore contribute no "
            "intervention. Rescue is interpretable only together with matched_fraction; "
            "the reference is younger in trajectory history, not independently generated."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
