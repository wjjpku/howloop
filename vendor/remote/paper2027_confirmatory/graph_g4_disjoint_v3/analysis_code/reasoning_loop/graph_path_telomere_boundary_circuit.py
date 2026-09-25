from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    logit_difference,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_rejuvenation_circuit import (
    MatchedAgeBatch,
    attention_function_rows,
    relative_state_mse,
)
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import load_task_lora_modules
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


def _target_probability(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        logits.float()
        .softmax(dim=-1)
        .gather(1, target[:, None])
        .mean()
    )


def _cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("cosine tensors must have identical shapes")
    return float(
        F.cosine_similarity(
            left.float().reshape(-1, left.shape[-1]),
            right.float().reshape(-1, right.shape[-1]),
            dim=-1,
        ).mean()
    )


def _intervention_specs(
    *,
    cfg,
    site: int,
    head: int | None,
) -> list[tuple[str, FunctionalIntervention]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    if head is None:
        return [
            (
                "block_input_all",
                FunctionalIntervention(
                    site=site,
                    component="block_input",
                    mode="patch",
                ),
            ),
            (
                "attention_out_answer",
                FunctionalIntervention(
                    site=site,
                    component="attention_out",
                    mode="patch",
                    positions=answer,
                ),
            ),
            (
                "mlp_out_answer",
                FunctionalIntervention(
                    site=site,
                    component="mlp_out",
                    mode="patch",
                    positions=answer,
                ),
            ),
        ]
    heads = (head,)
    return [
        (
            "q_answer",
            FunctionalIntervention(
                site=site,
                component="q",
                mode="patch",
                positions=answer,
                heads=heads,
            ),
        ),
        (
            "k_graph",
            FunctionalIntervention(
                site=site,
                component="k",
                mode="patch",
                positions=graph,
                heads=heads,
            ),
        ),
        (
            "v_graph",
            FunctionalIntervention(
                site=site,
                component="v",
                mode="patch",
                positions=graph,
                heads=heads,
            ),
        ),
        (
            "pattern_answer_graph",
            FunctionalIntervention(
                site=site,
                component="attention_pattern",
                mode="patch",
                positions=answer,
                source_positions=graph,
                heads=heads,
                renormalize=True,
            ),
        ),
        (
            "context_answer",
            FunctionalIntervention(
                site=site,
                component="head_context",
                mode="patch",
                positions=answer,
                heads=heads,
            ),
        ),
    ]


def _causal_gate_specs(
    cfg,
    *,
    candidate_head: int = 0,
) -> dict[str, tuple[FunctionalIntervention, ...]]:
    """Candidate answer-path gates for necessity and circuit-only tests.

    The graph-token carrier remains intact in every condition.  The
    circuit-only condition retains only the selected Block2 head at the answer position
    plus the Block2 answer-position MLP among answer-token updates.
    """

    if not 0 <= candidate_head < cfg.n_heads:
        raise ValueError("candidate_head must index a Block2 attention head")

    answer = explicit_depth_position_groups(cfg.node_count)["answer"]
    zero_b1_answer = (
        FunctionalIntervention(
            site=0,
            component="attention_out",
            mode="zero",
            positions=answer,
        ),
        FunctionalIntervention(
            site=0,
            component="mlp_out",
            mode="zero",
            positions=answer,
        ),
    )
    zero_b2_other_heads = FunctionalIntervention(
        site=1,
        component="head_context",
        mode="zero",
        positions=answer,
        heads=tuple(head for head in range(cfg.n_heads) if head != candidate_head),
    )
    zero_b2_candidate = FunctionalIntervention(
        site=1,
        component="head_context",
        mode="zero",
        positions=answer,
        heads=(candidate_head,),
    )
    zero_b2_mlp = FunctionalIntervention(
        site=1,
        component="mlp_out",
        mode="zero",
        positions=answer,
    )
    return {
        "clean_full_frozen_loop": (),
        f"necessity_zero_B2_H{candidate_head}_context": (zero_b2_candidate,),
        "necessity_zero_B2_MLP_answer": (zero_b2_mlp,),
        f"necessity_zero_B2_H{candidate_head}_and_MLP": (
            zero_b2_candidate,
            zero_b2_mlp,
        ),
        "candidate_answer_circuit_only": zero_b1_answer
        + (zero_b2_other_heads,),
        "candidate_answer_complement_only": (zero_b2_candidate, zero_b2_mlp),
    }


@torch.no_grad()
def analyze_checkpoint(
    *,
    model,
    cfg,
    state: torch.Tensor,
    operator,
    successors: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    endpoint: torch.Tensor,
    phase_positions: list[int],
    cycle: int,
    candidate_head: int = 0,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    loop_index = cfg.max_loops + cycle - 1
    learned_boundary = operator(state)
    exact_boundary = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=current,
        age=7,
        phase_position=phase_positions[7],
    )
    old_logits, old_trace = run_instrumented_state(
        model,
        state,
        loop_indices=(loop_index,),
    )
    learned_logits, learned_trace = run_instrumented_state(
        model,
        learned_boundary,
        loop_indices=(loop_index,),
    )
    exact_logits, exact_trace = run_instrumented_state(
        model,
        exact_boundary,
        loop_indices=(loop_index,),
    )

    clean_accuracy = _accuracy(learned_logits, target)
    clean_probability = _target_probability(learned_logits, target)
    rows: list[dict[str, Any]] = []
    for site in range(cfg.n_layers):
        for role, intervention in _intervention_specs(
            cfg=cfg,
            site=site,
            head=None,
        ):
            patched_logits, _ = run_instrumented_state(
                model,
                learned_boundary,
                loop_indices=(loop_index,),
                interventions=(intervention,),
                donor_trace=old_trace,
            )
            patched_accuracy = _accuracy(patched_logits, target)
            patched_probability = _target_probability(patched_logits, target)
            rows.append(
                {
                    "cycle": cycle,
                    "block": site + 1,
                    "head": None,
                    "role": role,
                    "clean_accuracy": clean_accuracy,
                    "patched_accuracy": patched_accuracy,
                    "accuracy_drop": clean_accuracy - patched_accuracy,
                    "clean_target_probability": clean_probability,
                    "patched_target_probability": patched_probability,
                    "probability_drop": clean_probability - patched_probability,
                    "patched_target_vs_current_margin": float(
                        logit_difference(patched_logits, target, current).mean()
                    ),
                }
            )
        for head in range(cfg.n_heads):
            for role, intervention in _intervention_specs(
                cfg=cfg,
                site=site,
                head=head,
            ):
                patched_logits, _ = run_instrumented_state(
                    model,
                    learned_boundary,
                    loop_indices=(loop_index,),
                    interventions=(intervention,),
                    donor_trace=old_trace,
                )
                patched_accuracy = _accuracy(patched_logits, target)
                patched_probability = _target_probability(patched_logits, target)
                rows.append(
                    {
                        "cycle": cycle,
                        "block": site + 1,
                        "head": head,
                        "role": role,
                        "clean_accuracy": clean_accuracy,
                        "patched_accuracy": patched_accuracy,
                        "accuracy_drop": clean_accuracy - patched_accuracy,
                        "clean_target_probability": clean_probability,
                        "patched_target_probability": patched_probability,
                        "probability_drop": clean_probability - patched_probability,
                        "patched_target_vs_current_margin": float(
                            logit_difference(
                                patched_logits,
                                target,
                                current,
                            ).mean()
                        ),
                    }
                )

    similarity_rows: list[dict[str, Any]] = [
        {
            "cycle": cycle,
            "stage": "loop_boundary_input",
            "learned_vs_exact_relative_mse": relative_state_mse(
                learned_boundary,
                exact_boundary,
            ),
            "learned_vs_exact_answer_cosine": _cosine_mean(
                learned_boundary[:, -1],
                exact_boundary[:, -1],
            ),
            "old_vs_exact_relative_mse": relative_state_mse(
                state,
                exact_boundary,
            ),
        }
    ]
    for site in range(cfg.n_layers):
        learned_site = learned_trace.sites[site]
        exact_site = exact_trace.sites[site]
        old_site = old_trace.sites[site]
        for stage, learned_value, exact_value, old_value in (
            (
                "block_input",
                learned_site.hidden_in,
                exact_site.hidden_in,
                old_site.hidden_in,
            ),
            (
                "post_attention",
                learned_site.residual_mid,
                exact_site.residual_mid,
                old_site.residual_mid,
            ),
            (
                "post_mlp",
                learned_site.hidden_out,
                exact_site.hidden_out,
                old_site.hidden_out,
            ),
        ):
            similarity_rows.append(
                {
                    "cycle": cycle,
                    "stage": f"B{site + 1}_{stage}",
                    "learned_vs_exact_relative_mse": relative_state_mse(
                        learned_value,
                        exact_value,
                    ),
                    "learned_vs_exact_answer_cosine": _cosine_mean(
                        learned_value[:, -1],
                        exact_value[:, -1],
                    ),
                    "old_vs_exact_relative_mse": relative_state_mse(
                        old_value,
                        exact_value,
                    ),
                }
            )
        similarity_rows.append(
            {
                "cycle": cycle,
                "stage": f"B{site + 1}_answer_query",
                "learned_vs_exact_relative_mse": relative_state_mse(
                    learned_site.q[:, :, -1],
                    exact_site.q[:, :, -1],
                ),
                "learned_vs_exact_answer_cosine": _cosine_mean(
                    learned_site.q[:, :, -1],
                    exact_site.q[:, :, -1],
                ),
                "old_vs_exact_relative_mse": relative_state_mse(
                    old_site.q[:, :, -1],
                    exact_site.q[:, :, -1],
                ),
            }
        )

    matched = MatchedAgeBatch(
        terminal=state,
        young=exact_boundary,
        successors=successors,
        current=current,
        target=target,
        endpoint=endpoint,
    )
    attention_rows = attention_function_rows(
        cfg=cfg,
        batch=matched,
        traces={
            "old_no_J": old_trace,
            "learned_boundary_J": learned_trace,
            "exact_H7_boundary": exact_trace,
        },
    )
    for row in attention_rows:
        row["cycle"] = cycle
    gate_rows: list[dict[str, Any]] = []
    for condition, interventions in _causal_gate_specs(
        cfg,
        candidate_head=candidate_head,
    ).items():
        gated_logits, _ = run_instrumented_state(
            model,
            learned_boundary,
            loop_indices=(loop_index,),
            interventions=interventions,
        )
        gate_rows.append(
            {
                "cycle": cycle,
                "condition": condition,
                "accuracy": _accuracy(gated_logits, target),
                "target_probability": _target_probability(gated_logits, target),
                "target_vs_current_margin": float(
                    logit_difference(gated_logits, target, current).mean()
                ),
            }
        )
    gate_rows.append(
        {
            "cycle": cycle,
            "condition": "no_J_full_frozen_loop",
            "accuracy": _accuracy(old_logits, target),
            "target_probability": _target_probability(old_logits, target),
            "target_vs_current_margin": float(
                logit_difference(old_logits, target, current).mean()
            ),
        }
    )
    similarity_rows[0].update(
        {
            "old_accuracy": _accuracy(old_logits, target),
            "learned_accuracy": clean_accuracy,
            "exact_accuracy": _accuracy(exact_logits, target),
        }
    )
    return rows, similarity_rows, attention_rows, gate_rows


def _aggregate_mediation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["block"]), row["head"], str(row["role"]))
        grouped[key].append(row)
    result = []
    for (block, head, role), parts in grouped.items():
        result.append(
            {
                "block": block,
                "head": head,
                "role": role,
                "mean_accuracy_drop": sum(
                    float(part["accuracy_drop"]) for part in parts
                )
                / len(parts),
                "mean_probability_drop": sum(
                    float(part["probability_drop"]) for part in parts
                )
                / len(parts),
                "cycles": [int(part["cycle"]) for part in parts],
            }
        )
    return sorted(
        result,
        key=lambda row: float(row["mean_probability_drop"]),
        reverse=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal path audit of a learned loop-boundary affine J."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument("--candidate-head", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.cycles or min(args.cycles) < 1:
        raise ValueError("cycles must be positive")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, operators, artifact_payload = (
        load_task_lora_modules(args.operator_artifact, device=device)
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if artifact_payload.get("placement") != "loop_boundary":
        raise ValueError("circuit audit requires a loop-boundary operator")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("circuit audit requires an all-position operator")
    if args.operator_label not in operators:
        raise ValueError(f"operator label not found: {args.operator_label}")
    operator = operators[args.operator_label]
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    set_seed(args.seed)
    _, path_targets, successors, _ = fixed_depth_batch(
        cfg,
        args.batch_size,
        device,
        path_positions=cfg.max_depth,
    )
    endpoint = path_targets[:, cfg.max_depth - 1]
    state = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=endpoint,
        age=8,
        phase_position=phase_positions[8],
    )
    requested = set(int(value) for value in args.cycles)
    mediation_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for cycle in range(1, max(requested) + 1):
        current = advance_nodes(
            successors,
            endpoint,
            steps=jump * (cycle - 1),
        )
        target = advance_nodes(
            successors,
            endpoint,
            steps=jump * cycle,
        )
        if cycle in requested:
            mediation, similarity, attention, gates = analyze_checkpoint(
                model=model,
                cfg=cfg,
                state=state,
                operator=operator,
                successors=successors,
                current=current,
                target=target,
                endpoint=endpoint,
                phase_positions=phase_positions,
                cycle=cycle,
                candidate_head=args.candidate_head,
            )
            mediation_rows.extend(mediation)
            similarity_rows.extend(similarity)
            attention_rows.extend(attention)
            gate_rows.extend(gates)
        step = _controlled_loop(
            loop_runner=run_one_loop,
            model=model,
            state=state,
            loop_index=cfg.max_loops + cycle - 1,
            positions=positions,
            operator=operator,
            placement="loop_boundary",
        )
        state = step.state

    aggregate = _aggregate_mediation(mediation_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "causal_mediation.csv", mediation_rows)
    _write_csv(args.out_dir / "state_similarity.csv", similarity_rows)
    _write_csv(args.out_dir / "attention_function.csv", attention_rows)
    _write_csv(args.out_dir / "causal_gates.csv", gate_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "frozen_model_loss_placement": artifact_payload.get(
            "backbone_loss_description",
            "not recorded",
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "controller_placement": "loop_boundary",
        "candidate_answer_head": args.candidate_head,
        "cycles": sorted(requested),
        "examples": args.batch_size,
        "causal_test": (
            "patch no-J component values into the learned boundary-J run; "
            "positive drops identify native Block1/Block2 paths mediated by J; "
            "zero-ablation and candidate circuit-only/complement gates test "
            "necessity and partial sufficiency of the answer path"
        ),
        "top_mediators_by_probability_drop": aggregate[:20],
        "files": {
            "causal_mediation": "causal_mediation.csv",
            "state_similarity": "state_similarity.csv",
            "attention_function": "attention_function.csv",
            "causal_gates": "causal_gates.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
