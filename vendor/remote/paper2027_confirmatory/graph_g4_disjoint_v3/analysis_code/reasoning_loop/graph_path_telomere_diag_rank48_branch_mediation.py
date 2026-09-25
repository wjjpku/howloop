from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalTrace,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_diag_rank48_circuit import (
    _component_specs,
    _joint_component_specs,
    _metrics,
    _recovery_metrics,
    _rolled_trace,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop


@dataclass(frozen=True)
class Variant:
    name: str
    operator: Callable[[torch.Tensor], torch.Tensor]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _variants(
    operator: DiagonalIdentityLoRAJ,
    *,
    shuffle_seed: int,
) -> dict[str, Variant]:
    diagonal = operator.diagonal_scale.detach().float()
    bias = operator.bias.detach().float()
    left = operator.A.detach().float()
    right = operator.B.detach().float()
    generator = torch.Generator(device=diagonal.device).manual_seed(shuffle_seed)
    shuffled = diagonal[
        torch.randperm(diagonal.numel(), generator=generator, device=diagonal.device)
    ]

    def affine(
        scale: torch.Tensor,
        include_low_rank: bool,
        offset: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def apply(value: torch.Tensor) -> torch.Tensor:
            live = value.float()
            update = (live @ left) @ right if include_low_rank else 0.0
            return live * scale + update + offset

        return apply

    zero = torch.zeros_like(bias)
    one = torch.ones_like(diagonal)
    mean = torch.full_like(diagonal, float(diagonal.mean()))
    return {
        "full": Variant("full", affine(diagonal, True, bias)),
        "no_bias": Variant("no_bias", affine(diagonal, True, zero)),
        "no_AB": Variant("no_AB", affine(diagonal, False, bias)),
        "identity_D": Variant("identity_D", affine(one, True, bias)),
        "mean_D": Variant("mean_D", affine(mean, True, bias)),
        "shuffled_D": Variant("shuffled_D", affine(shuffled, True, bias)),
    }


def _run_patch(
    *,
    model,
    state: torch.Tensor,
    loop_index: int,
    interventions: Sequence[FunctionalIntervention],
    donor: FunctionalTrace,
) -> torch.Tensor:
    logits, _ = run_instrumented_state(
        model,
        state,
        loop_indices=(loop_index,),
        interventions=tuple(interventions),
        donor_trace=donor,
    )
    return logits


def _patch_row(
    *,
    cycle: int,
    damaged: str,
    component: str,
    condition: str,
    patched: torch.Tensor,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    target: torch.Tensor,
    joint: bool,
) -> dict[str, Any]:
    return {
        "cycle": cycle,
        "effective_loop": cycle + 8,
        "damaged_variant": damaged,
        "component": component,
        "condition": condition,
        "joint": joint,
        **_metrics(patched, target),
        **_recovery_metrics(
            clean_logits=clean,
            corrupt_logits=corrupt,
            patched_logits=patched,
            target=target,
        ),
    }


def _fixed_neuron_intervention(
    *, cfg, neurons: Sequence[int]
) -> FunctionalIntervention:
    return FunctionalIntervention(
        site=1,
        component="mlp_hidden",
        mode="patch",
        positions=(cfg.seq_len - 1,),
        neurons=tuple(int(value) for value in neurons),
    )


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    manifest = {
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "checkpoint": str(args.checkpoint),
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "physical_gpu": args.physical_gpu,
        "prelaunch_used_mib": args.prelaunch_used_mib,
        "prelaunch_free_mib": args.prelaunch_free_mib,
        "declared_peak_gib": args.declared_peak_gib,
        "reserve_gib": args.reserve_gib,
        "shared_gpu": args.shared_gpu,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.cycles or min(args.cycles) < 1:
        raise ValueError("cycles must be positive")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, loaded, payload = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if payload.get("placement") != "loop_boundary":
        raise ValueError("branch mediation requires loop-boundary J")
    operator = loaded[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ) or operator.rank != 48:
        raise ValueError("branch mediation requires diagonal rank-48 J")
    variants = _variants(operator, shuffle_seed=args.shuffle_seed)
    unknown = set(args.mediated_variants) - set(variants)
    if unknown:
        raise ValueError(f"unknown mediated variants: {sorted(unknown)}")

    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    set_seed(args.seed)
    _, path_targets, successors, start = fixed_depth_batch(
        cfg, args.batch_size, device, path_positions=cfg.max_depth
    )
    endpoint = path_targets[:, cfg.max_depth - 1]
    initial = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=endpoint,
        age=8,
        phase_position=phase_positions[8],
    )
    trajectory = {name: initial.clone() for name in variants}
    requested = set(args.cycles)
    baseline_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []

    component_specs = _component_specs(cfg)
    joint_specs = _joint_component_specs(cfg, lookup_head=args.lookup_head)
    for cycle in range(1, max(requested) + 1):
        current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
        target = advance_nodes(successors, endpoint, steps=jump * cycle)
        loop_index = cfg.max_loops + cycle - 1
        if cycle in requested:
            controlled_states: dict[str, torch.Tensor] = {}
            logits: dict[str, torch.Tensor] = {}
            traces: dict[str, FunctionalTrace] = {}
            for name, variant in variants.items():
                controlled = variant.operator(trajectory[name])
                output, trace = run_instrumented_state(
                    model, controlled, loop_indices=(loop_index,)
                )
                controlled_states[name] = controlled
                logits[name] = output
                traces[name] = trace
                baseline_rows.append(
                    {
                        "cycle": cycle,
                        "effective_loop": cfg.max_loops + cycle,
                        "variant": name,
                        **_metrics(output, target),
                    }
                )
                prediction = output.argmax(dim=-1)
                margin = target_margin(output.float(), target)
                sample_index = torch.arange(target.shape[0], device=device)
                destination_position = 3 + 3 * current
                destination_positions = 3 + 3 * torch.arange(
                    cfg.node_count, device=device
                )
                b2_pattern = trace.sites[1].attention_pattern
                b2_head_destination = torch.stack(
                    [
                        b2_pattern[
                            sample_index, head, -1, destination_position
                        ]
                        for head in range(cfg.n_heads)
                    ],
                    dim=1,
                )
                b2_head_top_destination = torch.stack(
                    [
                        b2_pattern[:, head, -1, destination_positions].argmax(
                            dim=-1
                        )
                        for head in range(cfg.n_heads)
                    ],
                    dim=1,
                )
                b2_head_query_norm = trace.sites[1].q[:, :, -1].float().norm(
                    dim=-1
                )
                graph_token_norm = controlled[:, :-1].float().norm(dim=-1).mean(
                    dim=-1
                )
                mlp_positive_fraction = trace.sites[1].mlp_hidden[
                    :, -1
                ].gt(0).float().mean(dim=-1)
                for sample in range(target.shape[0]):
                    per_example_rows.append(
                        {
                            "cycle": cycle,
                            "sample": sample,
                            "variant": name,
                            "successors": " ".join(
                                str(int(value))
                                for value in successors[sample].tolist()
                            ),
                            "start": int(start[sample]),
                            "current": int(current[sample]),
                            "target": int(target[sample]),
                            "prediction": int(prediction[sample]),
                            "correct": int(prediction[sample].eq(target[sample])),
                            "target_margin": float(margin[sample]),
                            **{
                                f"B2H{head}_current_destination_attention": float(
                                    b2_head_destination[sample, head]
                                )
                                for head in range(cfg.n_heads)
                            },
                            **{
                                f"B2H{head}_top_destination_is_current": int(
                                    b2_head_top_destination[sample, head].eq(
                                        current[sample]
                                    )
                                )
                                for head in range(cfg.n_heads)
                            },
                            **{
                                f"B2H{head}_query_norm": float(
                                    b2_head_query_norm[sample, head]
                                )
                                for head in range(cfg.n_heads)
                            },
                            "loop_input_answer_norm": float(
                                controlled[sample, -1].float().norm()
                            ),
                            "loop_input_mean_nonanswer_norm": float(
                                graph_token_norm[sample]
                            ),
                            "B1_output_answer_norm": float(
                                trace.sites[0].hidden_out[sample, -1].float().norm()
                            ),
                            "B2_post_attention_answer_norm": float(
                                trace.sites[1].residual_mid[sample, -1].float().norm()
                            ),
                            "B2_MLP_hidden_norm": float(
                                trace.sites[1].mlp_hidden[sample, -1].float().norm()
                            ),
                            "B2_MLP_positive_fraction": float(
                                mlp_positive_fraction[sample]
                            ),
                        }
                    )

            shuffled_full = _rolled_trace(traces["full"])
            for damaged in args.mediated_variants:
                for spec in component_specs:
                    repaired = _run_patch(
                        model=model,
                        state=controlled_states[damaged],
                        loop_index=loop_index,
                        interventions=(spec.intervention,),
                        donor=traces["full"],
                    )
                    patch_rows.append(
                        _patch_row(
                            cycle=cycle,
                            damaged=damaged,
                            component=spec.label,
                            condition="full_into_damaged",
                            patched=repaired,
                            clean=logits["full"],
                            corrupt=logits[damaged],
                            target=target,
                            joint=False,
                        )
                    )
                    shuffled = _run_patch(
                        model=model,
                        state=controlled_states[damaged],
                        loop_index=loop_index,
                        interventions=(spec.intervention,),
                        donor=shuffled_full,
                    )
                    patch_rows.append(
                        _patch_row(
                            cycle=cycle,
                            damaged=damaged,
                            component=spec.label,
                            condition="batch_shuffled_full_into_damaged",
                            patched=shuffled,
                            clean=logits["full"],
                            corrupt=logits[damaged],
                            target=target,
                            joint=False,
                        )
                    )
                    damaged_into_full = _run_patch(
                        model=model,
                        state=controlled_states["full"],
                        loop_index=loop_index,
                        interventions=(spec.intervention,),
                        donor=traces[damaged],
                    )
                    patch_rows.append(
                        _patch_row(
                            cycle=cycle,
                            damaged=damaged,
                            component=spec.label,
                            condition="damaged_into_full",
                            patched=damaged_into_full,
                            clean=logits[damaged],
                            corrupt=logits["full"],
                            target=target,
                            joint=False,
                        )
                    )
                for spec in joint_specs:
                    repaired = _run_patch(
                        model=model,
                        state=controlled_states[damaged],
                        loop_index=loop_index,
                        interventions=spec.interventions,
                        donor=traces["full"],
                    )
                    patch_rows.append(
                        _patch_row(
                            cycle=cycle,
                            damaged=damaged,
                            component=spec.label,
                            condition="full_into_damaged",
                            patched=repaired,
                            clean=logits["full"],
                            corrupt=logits[damaged],
                            target=target,
                            joint=True,
                        )
                    )
                    damaged_into_full = _run_patch(
                        model=model,
                        state=controlled_states["full"],
                        loop_index=loop_index,
                        interventions=spec.interventions,
                        donor=traces[damaged],
                    )
                    patch_rows.append(
                        _patch_row(
                            cycle=cycle,
                            damaged=damaged,
                            component=spec.label,
                            condition="damaged_into_full",
                            patched=damaged_into_full,
                            clean=logits[damaged],
                            corrupt=logits["full"],
                            target=target,
                            joint=True,
                        )
                    )
                if args.fixed_mlp_neurons:
                    fixed = _fixed_neuron_intervention(
                        cfg=cfg, neurons=args.fixed_mlp_neurons
                    )
                    repaired = _run_patch(
                        model=model,
                        state=controlled_states[damaged],
                        loop_index=loop_index,
                        interventions=(fixed,),
                        donor=traces["full"],
                    )
                    patch_rows.append(
                        _patch_row(
                            cycle=cycle,
                            damaged=damaged,
                            component="B2.fixed64_mlp_hidden_answer",
                            condition="full_into_damaged",
                            patched=repaired,
                            clean=logits["full"],
                            corrupt=logits[damaged],
                            target=target,
                            joint=False,
                        )
                    )

        for name, variant in variants.items():
            step = _controlled_loop(
                loop_runner=run_one_loop,
                model=model,
                state=trajectory[name],
                loop_index=loop_index,
                positions=positions,
                operator=variant.operator,
                placement="loop_boundary",
            )
            trajectory[name] = step.state

    _write(args.out_dir / "baseline.csv", baseline_rows)
    _write(args.out_dir / "component_mediation.csv", patch_rows)
    _write(args.out_dir / "per_example.csv", per_example_rows)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller": "J(h)=h*D+(hA)B+b",
        "controller_rank": operator.rank,
        "controller_parameters": operator.parameter_count,
        "controller_placement": "loop boundary after Block2 FFN",
        "controller_training_loss": (
            "successor CE at every controlled continuation loop; no hidden MSE"
        ),
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "examples": args.batch_size,
        "cycles": sorted(requested),
        "trajectory_variants": list(variants),
        "mediated_variants": list(args.mediated_variants),
        "shuffle_seed": args.shuffle_seed,
        "fixed_mlp_neurons": list(args.fixed_mlp_neurons),
        "data_seed": args.seed,
        "lookup_head": args.lookup_head,
        "gpu_peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu_peak_allocated_gib": result["gpu_peak_allocated_gib"],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream mediation of diagonal and bias J branches."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument(
        "--mediated-variants", nargs="*", default=("no_bias", "shuffled_D")
    )
    parser.add_argument("--fixed-mlp-neurons", type=int, nargs="*", default=())
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--shuffle-seed", type=int, default=20260801)
    parser.add_argument("--lookup-head", type=int, default=0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.05)
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--prelaunch-used-mib", type=int, default=None)
    parser.add_argument("--prelaunch-free-mib", type=int, default=None)
    parser.add_argument("--declared-peak-gib", type=float, default=None)
    parser.add_argument("--reserve-gib", type=float, default=None)
    parser.add_argument(
        "--shared-gpu", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run_experiment(parse_args()), indent=2, sort_keys=True))
