from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_jump_controller import (
    JumpMode,
    _aggregate_rows,
    _validate_modes,
    _write_csv,
    apply_vector_map,
    collect_jump_pair_batch,
    controller_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


@dataclass(frozen=True)
class InterventionCase:
    name: str
    family: str
    site: int
    component: str
    head_subset: tuple[int, ...]
    position_group: str
    interventions: tuple[FunctionalIntervention, ...]


def _load_controller(
    *,
    path: Path,
    name: str,
    device: torch.device,
) -> VectorAffine:
    payload = torch.load(path, map_location=device)
    if name not in payload:
        raise KeyError(f"controller is missing: {name}")
    item = payload[name]
    weight = item["weight"].to(device)
    return VectorAffine(
        weight=weight,
        bias=item["bias"].to(device),
        update_rank=int(item.get("update_rank", weight.shape[0])),
        fit_dimension=int(item.get("fit_dimension", weight.shape[0])),
        retained_fit_energy=float(item.get("retained_fit_energy", 1.0)),
    )


def nonempty_head_subsets(n_heads: int) -> tuple[tuple[int, ...], ...]:
    if n_heads < 1:
        raise ValueError("n_heads must be positive")
    return tuple(
        subset
        for size in range(1, n_heads + 1)
        for subset in itertools.combinations(range(n_heads), size)
    )


def _head_label(heads: tuple[int, ...]) -> str:
    return "".join(str(head) for head in heads)


def build_intervention_cases(cfg) -> tuple[InterventionCase, ...]:
    groups = explicit_depth_position_groups(cfg.node_count)
    controller_groups = controller_position_groups(cfg)
    answer = groups["answer"]
    all_positions = controller_groups["all"]
    graph = groups["graph"]
    named_positions = {
        "answer": answer,
        "registers": controller_groups["registers"],
        "query_work": controller_groups["query_work"],
        "graph": graph,
        "all": all_positions,
    }
    all_heads = tuple(range(cfg.n_heads))
    cases: list[InterventionCase] = []

    for site in range(cfg.n_layers):
        block = site + 1
        for group_name, positions in named_positions.items():
            cases.append(
                InterventionCase(
                    name=f"B{block}.block_input.{group_name}",
                    family="block_input",
                    site=site,
                    component="block_input",
                    head_subset=(),
                    position_group=group_name,
                    interventions=(
                        FunctionalIntervention(
                            site=site,
                            component="block_input",
                            mode="patch",
                            positions=positions,
                        ),
                    ),
                )
            )

        for component in ("q", "k", "v", "attention_pattern", "head_context"):
            for heads in nonempty_head_subsets(cfg.n_heads):
                if component == "q":
                    kwargs: dict[str, Any] = {"positions": answer}
                elif component in {"k", "v"}:
                    kwargs = {"positions": all_positions}
                elif component == "attention_pattern":
                    kwargs = {
                        "positions": answer,
                        "renormalize": False,
                    }
                else:
                    kwargs = {"positions": answer}
                cases.append(
                    InterventionCase(
                        name=(
                            f"B{block}.{component}.heads_{_head_label(heads)}"
                        ),
                        family="head_subset",
                        site=site,
                        component=component,
                        head_subset=heads,
                        position_group=(
                            "answer" if component not in {"k", "v"} else "all"
                        ),
                        interventions=(
                            FunctionalIntervention(
                                site=site,
                                component=component,
                                mode="patch",
                                heads=heads,
                                **kwargs,
                            ),
                        ),
                    )
                )

        q_intervention = FunctionalIntervention(
            site=site,
            component="q",
            mode="patch",
            positions=answer,
            heads=all_heads,
        )
        k_intervention = FunctionalIntervention(
            site=site,
            component="k",
            mode="patch",
            positions=all_positions,
            heads=all_heads,
        )
        v_intervention = FunctionalIntervention(
            site=site,
            component="v",
            mode="patch",
            positions=all_positions,
            heads=all_heads,
        )
        for name, interventions in (
            ("qk", (q_intervention, k_intervention)),
            ("qv", (q_intervention, v_intervention)),
            ("kv", (k_intervention, v_intervention)),
            ("qkv", (q_intervention, k_intervention, v_intervention)),
        ):
            cases.append(
                InterventionCase(
                    name=f"B{block}.{name}.all_heads",
                    family="qkv_combination",
                    site=site,
                    component=name,
                    head_subset=all_heads,
                    position_group="answer_q_all_kv",
                    interventions=interventions,
                )
            )

        token_components = (
            "attention_out",
            "residual_mid",
            "mlp_hidden",
            "mlp_out",
        )
        for component in token_components:
            position_variants = (
                named_positions
                if site == 0
                else {"answer": answer}
            )
            for group_name, positions in position_variants.items():
                cases.append(
                    InterventionCase(
                        name=f"B{block}.{component}.{group_name}",
                        family="token_component",
                        site=site,
                        component=component,
                        head_subset=(),
                        position_group=group_name,
                        interventions=(
                            FunctionalIntervention(
                                site=site,
                                component=component,
                                mode="patch",
                                positions=positions,
                            ),
                        ),
                    )
                )
    return tuple(cases)


def switch_metrics(
    logits: torch.Tensor,
    *,
    receiver_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    all_targets: torch.Tensor,
    endpoint_position: int,
    direction: str,
) -> dict[str, float | int]:
    endpoint = all_targets[:, endpoint_position]
    one = all_targets[:, endpoint_position + 1]
    two = all_targets[:, endpoint_position + 2]
    distinct = endpoint.ne(one) & endpoint.ne(two) & one.ne(two)
    if direction == "one_to_two":
        desired, undesired = two, one
    elif direction == "two_to_one":
        desired, undesired = one, two
    else:
        raise ValueError(f"unsupported direction: {direction}")

    receiver_prediction = receiver_logits.argmax(dim=-1)
    donor_prediction = donor_logits.argmax(dim=-1)
    paired = (
        distinct
        & receiver_prediction.eq(undesired)
        & donor_prediction.eq(desired)
    )
    prediction = logits.argmax(dim=-1)
    probability = logits.softmax(dim=-1)
    desired_logits = logits.gather(1, desired[:, None]).squeeze(1)
    undesired_logits = logits.gather(1, undesired[:, None]).squeeze(1)
    margin = desired_logits - undesired_logits

    receiver_margin = (
        receiver_logits.gather(1, desired[:, None]).squeeze(1)
        - receiver_logits.gather(1, undesired[:, None]).squeeze(1)
    )
    donor_margin = (
        donor_logits.gather(1, desired[:, None]).squeeze(1)
        - donor_logits.gather(1, undesired[:, None]).squeeze(1)
    )
    denominator = donor_margin - receiver_margin
    recovery_valid = paired & denominator.abs().gt(1e-6)
    recovery = (margin - receiver_margin) / denominator

    def selected_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
        return float(value[mask].float().mean()) if bool(mask.any()) else float("nan")

    return {
        "distinct_count": int(distinct.sum()),
        "paired_valid_count": int(paired.sum()),
        "desired_accuracy": selected_mean(prediction.eq(desired), distinct),
        "undesired_accuracy": selected_mean(
            prediction.eq(undesired), distinct
        ),
        "endpoint_accuracy": selected_mean(
            prediction.eq(endpoint), distinct
        ),
        "desired_probability": selected_mean(
            probability.gather(1, desired[:, None]).squeeze(1), distinct
        ),
        "undesired_probability": selected_mean(
            probability.gather(1, undesired[:, None]).squeeze(1), distinct
        ),
        "desired_minus_undesired_logit": selected_mean(margin, distinct),
        "paired_desired_accuracy": selected_mean(
            prediction.eq(desired), paired
        ),
        "paired_undesired_accuracy": selected_mean(
            prediction.eq(undesired), paired
        ),
        "recovery_valid_count": int(recovery_valid.sum()),
        "normalized_margin_recovery": selected_mean(
            recovery, recovery_valid
        ),
    }


def _evaluate_condition(
    *,
    logits: torch.Tensor,
    receiver_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    all_targets: torch.Tensor,
    endpoint_position: int,
    direction: str,
    batch_index: int,
    controller_seed: int,
    case: InterventionCase | None,
    condition: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "batch": batch_index,
        "controller_seed": controller_seed,
        "direction": direction,
        "condition": condition,
        "run_condition": (
            f"seed{controller_seed}.{direction}.{condition}"
        ),
        "family": "baseline" if case is None else case.family,
        "site": -1 if case is None else case.site,
        "block": 0 if case is None else case.site + 1,
        "component": "baseline" if case is None else case.component,
        "head_subset": (
            "" if case is None else "+".join(map(str, case.head_subset))
        ),
        "head_count": 0 if case is None else len(case.head_subset),
        "position_group": "" if case is None else case.position_group,
    }
    row.update(
        switch_metrics(
            logits,
            receiver_logits=receiver_logits,
            donor_logits=donor_logits,
            all_targets=all_targets,
            endpoint_position=endpoint_position,
            direction=direction,
        )
    )
    return row


@torch.no_grad()
def run_causal_switch(
    *,
    checkpoint: Path,
    controller_path: Path,
    out_dir: Path,
    device_name: str,
    controller_seeds: tuple[int, ...],
    eval_batch_size: int,
    eval_batches: int,
    eval_seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("JUMP_CONTROLLER_CUDA_MEMORY_FRACTION", "0.06")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)

    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    two_mode = JumpMode(
        name="two",
        reference_age=1,
        reference_path_before=2,
        programmed_jump=2,
    )
    _validate_modes(cfg, two_mode=two_mode)
    cases = build_intervention_cases(cfg)
    positions = controller_position_groups(cfg)["all"]
    loop_indices = (cfg.max_loops,)
    out_dir.mkdir(parents=True, exist_ok=True)

    controllers: dict[int, tuple[VectorAffine, VectorAffine]] = {}
    for seed in controller_seeds:
        controllers[seed] = (
            _load_controller(
                path=controller_path,
                name=f"seed{seed}_J_one_rank8",
                device=device,
            ),
            _load_controller(
                path=controller_path,
                name=f"seed{seed}_J_two_task_lambda10_pre1",
                device=device,
            ),
        )

    set_seed(eval_seed)
    rows: list[dict[str, Any]] = []
    preloop_rows: list[dict[str, Any]] = []
    for batch_index in range(eval_batches):
        pair = collect_jump_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=eval_batch_size,
            device=device,
            two_mode=two_mode,
        )
        for controller_seed in controller_seeds:
            one_controller, two_controller = controllers[controller_seed]
            one_state = apply_vector_map(
                pair.terminal,
                positions=positions,
                controller=one_controller,
            )
            two_state = apply_vector_map(
                pair.terminal,
                positions=positions,
                controller=two_controller,
            )
            one_preloop = logits_from_raw_state(model, one_state)
            two_preloop = logits_from_raw_state(model, two_state)
            one_logits, one_trace = run_instrumented_state(
                model,
                one_state,
                loop_indices=loop_indices,
            )
            two_logits, two_trace = run_instrumented_state(
                model,
                two_state,
                loop_indices=loop_indices,
            )
            traces: dict[str, tuple[torch.Tensor, FunctionalTrace]] = {
                "one": (one_logits, one_trace),
                "two": (two_logits, two_trace),
            }

            for mode, preloop_logits in (
                ("one", one_preloop),
                ("two", two_preloop),
            ):
                endpoint = pair.all_targets[:, cfg.max_depth]
                one = pair.all_targets[:, cfg.max_depth + 1]
                two = pair.all_targets[:, cfg.max_depth + 2]
                prediction = preloop_logits.argmax(dim=-1)
                distinct = endpoint.ne(one) & endpoint.ne(two) & one.ne(two)
                preloop_rows.append(
                    {
                        "batch": batch_index,
                        "controller_seed": controller_seed,
                        "mode": mode,
                        "seed_mode": f"seed{controller_seed}.{mode}",
                        "distinct_count": int(distinct.sum()),
                        "endpoint_accuracy": float(
                            prediction[distinct].eq(endpoint[distinct]).float().mean()
                        ),
                        "one_accuracy": float(
                            prediction[distinct].eq(one[distinct]).float().mean()
                        ),
                        "two_accuracy": float(
                            prediction[distinct].eq(two[distinct]).float().mean()
                        ),
                    }
                )

            for direction, receiver_name, donor_name in (
                ("one_to_two", "one", "two"),
                ("two_to_one", "two", "one"),
            ):
                receiver_logits, _ = traces[receiver_name]
                donor_logits, donor_trace = traces[donor_name]
                rows.append(
                    _evaluate_condition(
                        logits=receiver_logits,
                        receiver_logits=receiver_logits,
                        donor_logits=donor_logits,
                        all_targets=pair.all_targets,
                        endpoint_position=cfg.max_depth,
                        direction=direction,
                        batch_index=batch_index,
                        controller_seed=controller_seed,
                        case=None,
                        condition="receiver",
                    )
                )
                rows.append(
                    _evaluate_condition(
                        logits=donor_logits,
                        receiver_logits=receiver_logits,
                        donor_logits=donor_logits,
                        all_targets=pair.all_targets,
                        endpoint_position=cfg.max_depth,
                        direction=direction,
                        batch_index=batch_index,
                        controller_seed=controller_seed,
                        case=None,
                        condition="donor",
                    )
                )
                receiver_state = (
                    one_state if receiver_name == "one" else two_state
                )
                for case in cases:
                    patched_logits, _ = run_instrumented_state(
                        model,
                        receiver_state,
                        loop_indices=loop_indices,
                        interventions=case.interventions,
                        donor_trace=donor_trace,
                    )
                    rows.append(
                        _evaluate_condition(
                            logits=patched_logits,
                            receiver_logits=receiver_logits,
                            donor_logits=donor_logits,
                            all_targets=pair.all_targets,
                            endpoint_position=cfg.max_depth,
                            direction=direction,
                            batch_index=batch_index,
                            controller_seed=controller_seed,
                            case=case,
                            condition=case.name,
                        )
                    )

    summary_rows = _aggregate_rows(
        rows,
        key="run_condition",
    )
    preloop_summary = _aggregate_rows(preloop_rows, key="seed_mode")
    _write_csv(out_dir / "intervention_rows.csv", rows)
    _write_csv(out_dir / "condition_summary.csv", summary_rows)
    _write_csv(out_dir / "preloop_rows.csv", preloop_rows)
    _write_csv(out_dir / "preloop_summary.csv", preloop_summary)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "controller_path": str(controller_path),
        "controller_seeds": list(controller_seeds),
        "config": asdict(cfg),
        "loss_placement": "final_only",
        "trained_loop_count": cfg.max_loops,
        "evaluated_receiver_age": cfg.max_loops,
        "evaluated_extra_loops": 1,
        "shared_physical_blocks": cfg.n_layers,
        "effective_depth_at_training_horizon": cfg.n_layers * cfg.max_loops,
        "evaluation_examples_per_controller": eval_batch_size * eval_batches,
        "evaluation_seed": eval_seed,
        "two_mode": asdict(two_mode),
        "intervention_case_count": len(cases),
        "preloop_summary": preloop_summary,
        "condition_summary": summary_rows,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bidirectionally patch the one-hop and two-hop controller "
            "trajectories to identify the circuit that selects jump size."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--controller-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=15401)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_causal_switch(
        checkpoint=args.checkpoint,
        controller_path=args.controller_path,
        out_dir=args.out_dir,
        device_name=args.device,
        controller_seeds=tuple(args.controller_seeds),
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        eval_seed=args.eval_seed,
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
