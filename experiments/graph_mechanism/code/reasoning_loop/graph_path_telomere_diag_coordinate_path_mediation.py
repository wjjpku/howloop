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
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_diag_rank48_circuit import (
    _metrics,
    _rolled_trace,
)
from reasoning_loop.graph_path_telomere_diag_group_sweep import coarse_diagonal_groups
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop


@dataclass(frozen=True)
class Variant:
    name: str
    coordinates: tuple[int, ...]
    operator: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PathSpec:
    label: str
    interventions: tuple[FunctionalIntervention, ...]
    stage: str


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_coordinate_group(value: str) -> tuple[int, ...]:
    coordinates = tuple(int(item) for item in value.split(",") if item.strip())
    if not coordinates:
        raise argparse.ArgumentTypeError("coordinate group cannot be empty")
    if len(set(coordinates)) != len(coordinates):
        raise argparse.ArgumentTypeError("coordinate group contains duplicates")
    return coordinates


def _parse_coarse_group_variant(value: str) -> tuple[str, float]:
    try:
        group_name, delta_text = value.rsplit(":", 1)
        delta = float(delta_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "coarse group variant must have the form high_retention:+0.02"
        ) from exc
    if group_name not in {
        "strongly_damped",
        "high_retention",
        "middle",
        "random",
    }:
        raise argparse.ArgumentTypeError(f"unknown coarse D group: {group_name}")
    if delta == 0:
        raise argparse.ArgumentTypeError("coarse group delta cannot be zero")
    return group_name, delta


def _variants(
    operator: DiagonalIdentityLoRAJ,
    *,
    coordinate_groups: Sequence[tuple[int, ...]],
    shuffle_seed: int,
) -> dict[str, Variant]:
    full = operator.diagonal_scale.detach().float()
    generator = torch.Generator(device=full.device).manual_seed(shuffle_seed)
    shuffled = full[
        torch.randperm(full.numel(), generator=generator, device=full.device)
    ]
    left = operator.A.detach().float()
    right = operator.B.detach().float()
    bias = operator.bias.detach().float()

    def affine(scale: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        def apply(value: torch.Tensor) -> torch.Tensor:
            live = value.float()
            return live * scale + (live @ left) @ right + bias

        return apply

    result = {"full": Variant("full", tuple(range(full.numel())), affine(full))}
    for coordinates in coordinate_groups:
        if min(coordinates) < 0 or max(coordinates) >= full.numel():
            raise ValueError(f"coordinate group is out of range: {coordinates}")
        scale = full.clone()
        index = torch.as_tensor(coordinates, device=full.device)
        scale[index] = shuffled[index]
        name = "damage_D_" + "_".join(str(value) for value in coordinates)
        result[name] = Variant(name, coordinates, affine(scale))
    return result


def _coarse_variants(
    operator: DiagonalIdentityLoRAJ,
    *,
    group_variants: Sequence[tuple[str, float]],
    group_size: int,
    group_seed: int,
) -> dict[str, Variant]:
    full = operator.diagonal_scale.detach().float()
    groups = coarse_diagonal_groups(
        full, group_size=group_size, random_seed=group_seed
    )
    left = operator.A.detach().float()
    right = operator.B.detach().float()
    bias = operator.bias.detach().float()

    def affine(scale: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        def apply(value: torch.Tensor) -> torch.Tensor:
            live = value.float()
            return live * scale + (live @ left) @ right + bias

        return apply

    result = {"full": Variant("full", tuple(range(full.numel())), affine(full))}
    for group_name, delta in group_variants:
        coordinates = groups[group_name]
        scale = full.clone()
        scale[coordinates] += delta
        sign = "plus" if delta > 0 else "minus"
        name = f"damage_D_{group_name}.{sign}.{abs(delta):.6f}"
        if name in result:
            raise ValueError(f"duplicate coarse D variant: {name}")
        result[name] = Variant(
            name,
            tuple(int(value) for value in coordinates.tolist()),
            affine(scale),
        )
    return result


def _path_specs(cfg, *, lookup_head: int, current: torch.Tensor) -> list[PathSpec]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    main_head = (lookup_head,)

    def intervention(
        site: int, component: str, **fields: Any
    ) -> FunctionalIntervention:
        return FunctionalIntervention(
            site=site,
            component=component,  # type: ignore[arg-type]
            mode="patch",
            **fields,
        )

    b1_input_answer = intervention(0, "block_input", positions=answer)
    b1_input_graph = intervention(0, "block_input", positions=graph)
    b1_attn = intervention(0, "attention_out", positions=answer)
    b1_mid = intervention(0, "residual_mid", positions=answer)
    b1_mlp_hidden = intervention(0, "mlp_hidden", positions=answer)
    b1_mlp_out = intervention(0, "mlp_out", positions=answer)
    b2_input_answer = intervention(1, "block_input", positions=answer)
    b2_input_graph = intervention(1, "block_input", positions=graph)
    q = intervention(1, "q", positions=answer, heads=main_head)
    k = intervention(1, "k", positions=graph, heads=main_head)
    v = intervention(1, "v", positions=graph, heads=main_head)
    pattern = intervention(
        1,
        "attention_pattern",
        positions=answer,
        source_positions=graph,
        heads=main_head,
        renormalize=True,
    )
    current_destination = 3 + 3 * current
    current_edge = intervention(
        1,
        "attention_pattern",
        positions=answer,
        dynamic_source_positions=current_destination[:, None],
        heads=main_head,
        renormalize=True,
    )
    context = intervention(1, "head_context", positions=answer, heads=main_head)
    attention_out = intervention(1, "attention_out", positions=answer)
    residual_mid = intervention(1, "residual_mid", positions=answer)
    mlp_hidden = intervention(1, "mlp_hidden", positions=answer)
    mlp_out = intervention(1, "mlp_out", positions=answer)
    return [
        PathSpec("B1.input_answer", (b1_input_answer,), "boundary_to_B1"),
        PathSpec("B1.input_graph", (b1_input_graph,), "boundary_to_B1"),
        PathSpec(
            "B1.input_answer_plus_graph",
            (b1_input_answer, b1_input_graph),
            "boundary_to_B1",
        ),
        PathSpec("B1.attention_out_answer", (b1_attn,), "B1_attention"),
        PathSpec("B1.residual_mid_answer", (b1_mid,), "B1_attention"),
        PathSpec("B1.mlp_hidden_answer", (b1_mlp_hidden,), "B1_MLP"),
        PathSpec("B1.mlp_out_answer", (b1_mlp_out,), "B1_MLP"),
        PathSpec("B2.input_answer", (b2_input_answer,), "B2_input"),
        PathSpec("B2.input_graph", (b2_input_graph,), "B2_input"),
        PathSpec(
            "B2.input_answer_plus_graph",
            (b2_input_answer, b2_input_graph),
            "B2_input",
        ),
        PathSpec(f"B2.H{lookup_head}.q_answer", (q,), "lookup_QK"),
        PathSpec(f"B2.H{lookup_head}.k_graph", (k,), "lookup_QK"),
        PathSpec(f"B2.H{lookup_head}.q_plus_k", (q, k), "lookup_QK"),
        PathSpec(f"B2.H{lookup_head}.v_graph", (v,), "lookup_V"),
        PathSpec(
            f"B2.H{lookup_head}.q_plus_k_plus_v", (q, k, v), "lookup_QKV"
        ),
        PathSpec(
            f"B2.H{lookup_head}.pattern_answer_graph", (pattern,), "lookup_pattern"
        ),
        PathSpec(
            f"B2.H{lookup_head}.pattern_current_destination",
            (current_edge,),
            "lookup_pattern",
        ),
        PathSpec(
            f"B2.H{lookup_head}.pattern_plus_v", (pattern, v), "lookup_context"
        ),
        PathSpec(
            f"B2.H{lookup_head}.context_answer", (context,), "lookup_context"
        ),
        PathSpec("B2.attention_out_answer", (attention_out,), "B2_attention"),
        PathSpec("B2.residual_mid_answer", (residual_mid,), "B2_attention"),
        PathSpec("B2.mlp_hidden_answer", (mlp_hidden,), "B2_MLP"),
        PathSpec("B2.mlp_out_answer", (mlp_out,), "B2_MLP"),
        PathSpec(
            f"B2.H{lookup_head}.context_plus_mlp_hidden",
            (context, mlp_hidden),
            "lookup_to_MLP",
        ),
        PathSpec(
            "B2.attention_out_plus_mlp_hidden",
            (attention_out, mlp_hidden),
            "lookup_to_MLP",
        ),
    ]


def _attention_diagnostic(
    trace: FunctionalTrace,
    *,
    lookup_head: int,
    current: torch.Tensor,
) -> torch.Tensor:
    batch = torch.arange(current.shape[0], device=current.device)
    destination_position = 3 + 3 * current
    return trace.sites[1].attention_pattern[
        batch, lookup_head, -1, destination_position
    ]


def _effect_fraction(
    *,
    full_margin: torch.Tensor,
    damaged_margin: torch.Tensor,
    patched_margin: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    denominator = full_margin - damaged_margin
    if direction == "full_into_damaged":
        numerator = patched_margin - damaged_margin
    elif direction == "damaged_into_full":
        numerator = full_margin - patched_margin
    else:
        raise ValueError(f"unknown direction: {direction}")
    result = torch.full_like(denominator, float("nan"))
    finite = denominator.abs().gt(1e-6)
    result[finite] = numerator[finite] / denominator[finite]
    return result


def _patch_rows(
    *,
    model,
    state: torch.Tensor,
    loop_index: int,
    target: torch.Tensor,
    current: torch.Tensor,
    successors: torch.Tensor,
    start: torch.Tensor,
    full_logits: torch.Tensor,
    damaged_logits: torch.Tensor,
    full_trace: FunctionalTrace,
    damaged_trace: FunctionalTrace,
    donor_trace: FunctionalTrace,
    variant: Variant,
    spec: PathSpec,
    direction: str,
    cycle: int,
    lookup_head: int,
    negative_control: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    patched_logits, patched_trace = run_instrumented_state(
        model,
        state,
        loop_indices=(loop_index,),
        interventions=spec.interventions,
        donor_trace=donor_trace,
    )
    full_margin = target_margin(full_logits.float(), target)
    damaged_margin = target_margin(damaged_logits.float(), target)
    patched_margin = target_margin(patched_logits.float(), target)
    fraction = _effect_fraction(
        full_margin=full_margin,
        damaged_margin=damaged_margin,
        patched_margin=patched_margin,
        direction=direction,
    )
    attention = _attention_diagnostic(
        patched_trace, lookup_head=lookup_head, current=current
    )
    finite = fraction.isfinite()
    aggregate = {
        "cycle": cycle,
        "effective_loop": cycle + 8,
        "damaged_variant": variant.name,
        "damaged_coordinates": " ".join(str(value) for value in variant.coordinates),
        "component": spec.label,
        "stage": spec.stage,
        "direction": direction,
        "negative_control": negative_control,
        **_metrics(patched_logits, target),
        "causal_fraction_mean": float(fraction[finite].mean()) if bool(finite.any()) else float("nan"),
        "causal_fraction_median": float(fraction[finite].median()) if bool(finite.any()) else float("nan"),
        "B2_lookup_current_destination_attention": float(attention.mean()),
    }
    prediction = patched_logits.argmax(dim=-1)
    examples = [
        {
            "cycle": cycle,
            "effective_loop": cycle + 8,
            "sample": sample,
            "damaged_variant": variant.name,
            "damaged_coordinates": " ".join(str(value) for value in variant.coordinates),
            "component": spec.label,
            "stage": spec.stage,
            "direction": direction,
            "negative_control": negative_control,
            "successors": " ".join(
                str(int(value)) for value in successors[sample].tolist()
            ),
            "start": int(start[sample]),
            "current": int(current[sample]),
            "target": int(target[sample]),
            "prediction": int(prediction[sample]),
            "correct": int(prediction[sample].eq(target[sample])),
            "target_margin": float(patched_margin[sample]),
            "causal_fraction": float(fraction[sample]),
            "B2_lookup_current_destination_attention": float(attention[sample]),
        }
        for sample in range(target.shape[0])
    ]
    return aggregate, examples


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
        raise ValueError("path mediation requires loop-boundary J")
    if tuple(positions) != tuple(range(cfg.seq_len)):
        raise ValueError("path mediation currently requires all-position J")
    operator = loaded[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ) or operator.rank != 48:
        raise ValueError("path mediation requires diagonal rank-48 J")
    if not 0 <= args.lookup_head < cfg.n_heads:
        raise ValueError("lookup head is out of range")
    if args.coarse_group_variants:
        variants = _coarse_variants(
            operator,
            group_variants=args.coarse_group_variants,
            group_size=args.coarse_group_size,
            group_seed=args.coarse_group_seed,
        )
    else:
        variants = _variants(
            operator,
            coordinate_groups=args.coordinate_groups,
            shuffle_seed=args.shuffle_seed,
        )
    negative_control_components = set(args.negative_control_components)
    if not negative_control_components:
        negative_control_components = {
            "B1.input_answer_plus_graph",
            "B2.input_answer_plus_graph",
            f"B2.H{args.lookup_head}.context_answer",
            "B2.mlp_hidden_answer",
        }
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(value) for value in phase["trajectory_positions_including_initial"]]
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
    trajectories = {name: initial.clone() for name in variants}
    requested = set(args.cycles)
    baseline_rows: list[dict[str, Any]] = []
    baseline_examples: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    patch_examples: list[dict[str, Any]] = []

    for cycle in range(1, max(requested) + 1):
        current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
        target = advance_nodes(successors, endpoint, steps=jump * cycle)
        loop_index = cfg.max_loops + cycle - 1
        if cycle in requested:
            controlled: dict[str, torch.Tensor] = {}
            logits: dict[str, torch.Tensor] = {}
            traces: dict[str, FunctionalTrace] = {}
            for name, variant in variants.items():
                state = variant.operator(trajectories[name])
                output, trace = run_instrumented_state(
                    model, state, loop_indices=(loop_index,)
                )
                controlled[name] = state
                logits[name] = output
                traces[name] = trace
                attention = _attention_diagnostic(
                    trace, lookup_head=args.lookup_head, current=current
                )
                baseline_rows.append(
                    {
                        "cycle": cycle,
                        "effective_loop": cycle + 8,
                        "variant": name,
                        "coordinates": " ".join(
                            str(value) for value in variant.coordinates
                        ),
                        **_metrics(output, target),
                        "B2_lookup_current_destination_attention": float(
                            attention.mean()
                        ),
                    }
                )
                prediction = output.argmax(dim=-1)
                margin = target_margin(output.float(), target)
                for sample in range(target.shape[0]):
                    baseline_examples.append(
                        {
                            "cycle": cycle,
                            "effective_loop": cycle + 8,
                            "sample": sample,
                            "variant": name,
                            "coordinates": " ".join(
                                str(value) for value in variant.coordinates
                            ),
                            "successors": " ".join(
                                str(int(value)) for value in successors[sample].tolist()
                            ),
                            "start": int(start[sample]),
                            "current": int(current[sample]),
                            "target": int(target[sample]),
                            "prediction": int(prediction[sample]),
                            "correct": int(prediction[sample].eq(target[sample])),
                            "target_margin": float(margin[sample]),
                            "B2_lookup_current_destination_attention": float(
                                attention[sample]
                            ),
                        }
                    )

            specs = _path_specs(cfg, lookup_head=args.lookup_head, current=current)
            shuffled_full = _rolled_trace(traces["full"])
            for name, variant in variants.items():
                if name == "full":
                    continue
                for spec in specs:
                    for direction, state, donor in (
                        (
                            "full_into_damaged",
                            controlled[name],
                            traces["full"],
                        ),
                        (
                            "damaged_into_full",
                            controlled["full"],
                            traces[name],
                        ),
                    ):
                        aggregate, examples = _patch_rows(
                            model=model,
                            state=state,
                            loop_index=loop_index,
                            target=target,
                            current=current,
                            successors=successors,
                            start=start,
                            full_logits=logits["full"],
                            damaged_logits=logits[name],
                            full_trace=traces["full"],
                            damaged_trace=traces[name],
                            donor_trace=donor,
                            variant=variant,
                            spec=spec,
                            direction=direction,
                            cycle=cycle,
                            lookup_head=args.lookup_head,
                            negative_control=False,
                        )
                        patch_rows.append(aggregate)
                        patch_examples.extend(examples)
                    if spec.label in negative_control_components:
                        aggregate, examples = _patch_rows(
                            model=model,
                            state=controlled[name],
                            loop_index=loop_index,
                            target=target,
                            current=current,
                            successors=successors,
                            start=start,
                            full_logits=logits["full"],
                            damaged_logits=logits[name],
                            full_trace=traces["full"],
                            damaged_trace=traces[name],
                            donor_trace=shuffled_full,
                            variant=variant,
                            spec=spec,
                            direction="full_into_damaged",
                            cycle=cycle,
                            lookup_head=args.lookup_head,
                            negative_control=True,
                        )
                        patch_rows.append(aggregate)
                        patch_examples.extend(examples)

        for name, variant in variants.items():
            step = _controlled_loop(
                loop_runner=run_one_loop,
                model=model,
                state=trajectories[name],
                loop_index=loop_index,
                positions=positions,
                operator=variant.operator,
                placement="loop_boundary",
            )
            trajectories[name] = step.state

    _write(args.out_dir / "baseline.csv", baseline_rows)
    _write(args.out_dir / "baseline_per_example.csv", baseline_examples)
    _write(args.out_dir / "path_mediation.csv", patch_rows)
    _write(args.out_dir / "path_mediation_per_example.csv", patch_examples)
    peak = (
        float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else None
    )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_continuation_cycles": sorted(requested),
        "controller": "J(h)=h*D+(hA)B+b",
        "controller_rank": operator.rank,
        "controller_placement": "loop boundary after Block2 FFN",
        "controller_training_loss": "successor CE at every controlled continuation loop; no hidden MSE",
        "coordinate_groups": [list(group) for group in args.coordinate_groups],
        "coarse_group_variants": [
            {"group": group, "signed_delta": delta}
            for group, delta in args.coarse_group_variants
        ],
        "coarse_group_size": args.coarse_group_size,
        "coarse_group_seed": args.coarse_group_seed,
        "lookup_head": args.lookup_head,
        "examples": args.batch_size,
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "path_components": [spec.label for spec in _path_specs(cfg, lookup_head=args.lookup_head, current=torch.zeros(args.batch_size, device=device, dtype=torch.long))],
        "gpu_peak_allocated_gib": peak,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu_peak_allocated_gib": peak,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cycles", type=int, nargs="+", default=(16, 32, 48, 64))
    parser.add_argument(
        "--coordinate-groups",
        type=_parse_coordinate_group,
        nargs="+",
        default=((176,), (243,), (85,), (103,), (176, 243, 85)),
    )
    parser.add_argument(
        "--coarse-group-variants",
        type=_parse_coarse_group_variant,
        nargs="+",
        default=(),
        help=(
            "Use D-ranked functional groups instead of coordinate shuffle damage; "
            "for example high_retention:+0.02"
        ),
    )
    parser.add_argument("--coarse-group-size", type=int, default=48)
    parser.add_argument("--coarse-group-seed", type=int, default=20260816)
    parser.add_argument("--lookup-head", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--shuffle-seed", type=int, default=20260801)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.025)
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--prelaunch-used-mib", type=int, default=None)
    parser.add_argument("--prelaunch-free-mib", type=int, default=None)
    parser.add_argument("--declared-peak-gib", type=float, default=None)
    parser.add_argument("--reserve-gib", type=float, default=None)
    parser.add_argument(
        "--shared-gpu", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--negative-control-components",
        nargs="*",
        default=(),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run_experiment(parse_args()), indent=2, sort_keys=True))
