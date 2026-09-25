from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    FunctionalSiteTrace,
    FunctionalTrace,
    explicit_depth_position_groups,
    run_instrumented_state,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_rejuvenation_circuit import (
    MatchedAgeBatch,
    attention_function_rows,
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
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


@dataclass(frozen=True)
class ComponentSpec:
    label: str
    intervention: FunctionalIntervention


@dataclass(frozen=True)
class JointComponentSpec:
    label: str
    interventions: tuple[FunctionalIntervention, ...]


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


def _metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    probability = logits.float().softmax(dim=-1).gather(
        1, target[:, None]
    ).squeeze(1)
    margin = target_margin(logits.float(), target)
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(target).float().mean()),
        "target_probability": float(probability.mean()),
        "target_margin": float(margin.mean()),
    }


def _recovery_metrics(
    *,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    clean_margin = target_margin(clean_logits.float(), target)
    corrupt_margin = target_margin(corrupt_logits.float(), target)
    patched_margin = target_margin(patched_logits.float(), target)
    denominator = clean_margin - corrupt_margin
    finite = denominator.abs().gt(1e-6)
    recovery = (patched_margin - corrupt_margin) / denominator
    controlled = (
        finite
        & clean_logits.argmax(dim=-1).eq(target)
        & corrupt_logits.argmax(dim=-1).ne(target)
    )
    return {
        "recovery_all": (
            float(recovery[finite].mean()) if bool(finite.any()) else float("nan")
        ),
        "recovery_controlled": (
            float(recovery[controlled].mean())
            if bool(controlled.any())
            else float("nan")
        ),
        "controlled_count": int(controlled.sum()),
    }


def _rolled_trace(trace: FunctionalTrace) -> FunctionalTrace:
    def roll(site: FunctionalSiteTrace) -> FunctionalSiteTrace:
        return FunctionalSiteTrace(
            loop_index=site.loop_index,
            block_index=site.block_index,
            hidden_in=site.hidden_in.roll(1, dims=0),
            q=site.q.roll(1, dims=0),
            k=site.k.roll(1, dims=0),
            v=site.v.roll(1, dims=0),
            attention_pattern=site.attention_pattern.roll(1, dims=0),
            head_context=site.head_context.roll(1, dims=0),
            attention_out=site.attention_out.roll(1, dims=0),
            residual_mid=site.residual_mid.roll(1, dims=0),
            mlp_hidden=site.mlp_hidden.roll(1, dims=0),
            mlp_out=site.mlp_out.roll(1, dims=0),
            hidden_out=site.hidden_out.roll(1, dims=0),
        )

    return FunctionalTrace(
        sites=[roll(site) for site in trace.sites],
        logits_by_loop=trace.logits_by_loop.roll(1, dims=0),
    )


def _component_specs(cfg) -> list[ComponentSpec]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    result: list[ComponentSpec] = []
    for site in range(cfg.n_layers):
        block = site + 1
        result.extend(
            (
                ComponentSpec(
                    f"B{block}.block_input_all",
                    FunctionalIntervention(
                        site=site, component="block_input", mode="patch"
                    ),
                ),
                ComponentSpec(
                    f"B{block}.block_input_answer",
                    FunctionalIntervention(
                        site=site,
                        component="block_input",
                        mode="patch",
                        positions=answer,
                    ),
                ),
                ComponentSpec(
                    f"B{block}.block_input_graph",
                    FunctionalIntervention(
                        site=site,
                        component="block_input",
                        mode="patch",
                        positions=graph,
                    ),
                ),
                ComponentSpec(
                    f"B{block}.block_input_answer_plus_graph",
                    FunctionalIntervention(
                        site=site,
                        component="block_input",
                        mode="patch",
                        positions=answer + graph,
                    ),
                ),
                ComponentSpec(
                    f"B{block}.attention_out_answer",
                    FunctionalIntervention(
                        site=site,
                        component="attention_out",
                        mode="patch",
                        positions=answer,
                    ),
                ),
                ComponentSpec(
                    f"B{block}.residual_mid_answer",
                    FunctionalIntervention(
                        site=site,
                        component="residual_mid",
                        mode="patch",
                        positions=answer,
                    ),
                ),
                ComponentSpec(
                    f"B{block}.mlp_hidden_answer",
                    FunctionalIntervention(
                        site=site,
                        component="mlp_hidden",
                        mode="patch",
                        positions=answer,
                    ),
                ),
                ComponentSpec(
                    f"B{block}.mlp_out_answer",
                    FunctionalIntervention(
                        site=site,
                        component="mlp_out",
                        mode="patch",
                        positions=answer,
                    ),
                ),
            )
        )
        for head in range(cfg.n_heads):
            heads = (head,)
            prefix = f"B{block}.H{head}"
            result.extend(
                (
                    ComponentSpec(
                        f"{prefix}.q_answer",
                        FunctionalIntervention(
                            site=site,
                            component="q",
                            mode="patch",
                            positions=answer,
                            heads=heads,
                        ),
                    ),
                    ComponentSpec(
                        f"{prefix}.k_graph",
                        FunctionalIntervention(
                            site=site,
                            component="k",
                            mode="patch",
                            positions=graph,
                            heads=heads,
                        ),
                    ),
                    ComponentSpec(
                        f"{prefix}.v_graph",
                        FunctionalIntervention(
                            site=site,
                            component="v",
                            mode="patch",
                            positions=graph,
                            heads=heads,
                        ),
                    ),
                    ComponentSpec(
                        f"{prefix}.pattern_answer_graph",
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
                    ComponentSpec(
                        f"{prefix}.context_answer",
                        FunctionalIntervention(
                            site=site,
                            component="head_context",
                            mode="patch",
                            positions=answer,
                            heads=heads,
                        ),
                    ),
                )
            )
    return result


def _patched_run(
    *,
    model,
    initial_state: torch.Tensor,
    loop_index: int,
    spec: ComponentSpec,
    donor_trace: FunctionalTrace,
) -> torch.Tensor:
    logits, _ = run_instrumented_state(
        model,
        initial_state,
        loop_indices=(loop_index,),
        interventions=(spec.intervention,),
        donor_trace=donor_trace,
    )
    return logits


def _joint_component_specs(cfg, *, lookup_head: int = 0) -> list[JointComponentSpec]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    site = 1
    if not 0 <= lookup_head < cfg.n_heads:
        raise ValueError("lookup head is out of range")
    head = (lookup_head,)
    head_name = f"H{lookup_head}"

    def intervention(component: str, **fields: Any) -> FunctionalIntervention:
        return FunctionalIntervention(
            site=site,
            component=component,  # type: ignore[arg-type]
            mode="patch",
            **fields,
        )

    q = intervention("q", positions=answer, heads=head)
    k = intervention("k", positions=graph, heads=head)
    v = intervention("v", positions=graph, heads=head)
    pattern = intervention(
        "attention_pattern",
        positions=answer,
        source_positions=graph,
        heads=head,
        renormalize=True,
    )
    context = intervention("head_context", positions=answer, heads=head)
    mlp_hidden = intervention("mlp_hidden", positions=answer)
    attention_out = intervention("attention_out", positions=answer)
    all_contexts = intervention(
        "head_context",
        positions=answer,
        heads=tuple(range(cfg.n_heads)),
    )
    all_values = intervention(
        "v",
        positions=graph,
        heads=tuple(range(cfg.n_heads)),
    )
    b1_residual = FunctionalIntervention(
        site=0,
        component="residual_mid",
        mode="patch",
        positions=answer,
    )
    return [
        JointComponentSpec(f"B2.{head_name}.q_plus_k", (q, k)),
        JointComponentSpec(f"B2.{head_name}.q_plus_k_plus_v", (q, k, v)),
        JointComponentSpec(f"B2.{head_name}.pattern_plus_v", (pattern, v)),
        JointComponentSpec(
            f"B2.{head_name}.context_plus_MLP_hidden", (context, mlp_hidden)
        ),
        JointComponentSpec(
            "B2.attention_out_plus_MLP_hidden", (attention_out, mlp_hidden)
        ),
        JointComponentSpec("B2.all_head_contexts", (all_contexts,)),
        JointComponentSpec("B2.all_graph_values", (all_values,)),
        JointComponentSpec(
            f"B1.residual_mid_plus_B2.{head_name}.value", (b1_residual, v)
        ),
        JointComponentSpec(
            f"B1.residual_mid_plus_B2.{head_name}.context", (b1_residual, context)
        ),
    ]


def _joint_hybrid_rows(
    *,
    model,
    cfg,
    loop_index: int,
    cycle: int,
    target: torch.Tensor,
    states: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
) -> list[dict[str, Any]]:
    pairs = (
        ("J_into_no_J", "no_J", "J", "J", "no_J"),
        ("no_J_into_J", "J", "no_J", "J", "no_J"),
        ("exact_H7_into_J", "J", "exact_H7", "exact_H7", "J"),
        ("J_into_exact_H7", "exact_H7", "J", "J", "exact_H7"),
        ("shuffled_J_into_no_J", "no_J", "shuffled_J", "J", "no_J"),
        ("shuffled_J_into_J", "J", "shuffled_J", "J", "no_J"),
    )
    donor_traces = {**traces, "shuffled_J": _rolled_trace(traces["J"])}
    rows: list[dict[str, Any]] = []
    for spec in _joint_component_specs(cfg):
        for condition, base, donor, clean, corrupt in pairs:
            patched, _ = run_instrumented_state(
                model,
                states[base],
                loop_indices=(loop_index,),
                interventions=spec.interventions,
                donor_trace=donor_traces[donor],
            )
            rows.append(
                {
                    "cycle": cycle,
                    "effective_loop": cfg.max_loops + cycle,
                    "component": spec.label,
                    "condition": condition,
                    "joint": True,
                    **_metrics(patched, target),
                    **_recovery_metrics(
                        clean_logits=logits[clean],
                        corrupt_logits=logits[corrupt],
                        patched_logits=patched,
                        target=target,
                    ),
                }
            )
    return rows


def _mlp_neuron_hybrid_rows(
    *,
    model,
    cfg,
    loop_index: int,
    cycle: int,
    target: torch.Tensor,
    states: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
    top_ks: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    pairs = (
        ("J_into_no_J", "no_J", "J", "J", "no_J"),
        ("no_J_into_J", "J", "no_J", "J", "no_J"),
        ("exact_H7_into_J", "J", "exact_H7", "exact_H7", "J"),
    )
    generator = torch.Generator(device=target.device).manual_seed(seed + cycle)
    rows: list[dict[str, Any]] = []
    for site in range(cfg.n_layers):
        neuron_count = traces["J"].sites[site].mlp_hidden.shape[-1]
        all_neurons = torch.arange(neuron_count, device=target.device)
        for condition, base, donor, clean, corrupt in pairs:
            delta = (
                traces[donor].sites[site].mlp_hidden[:, -1]
                - traces[base].sites[site].mlp_hidden[:, -1]
            )
            ranked = delta.float().abs().mean(dim=0).argsort(descending=True)
            for requested in top_ks:
                count = min(int(requested), neuron_count)
                top = ranked[:count]
                remaining = all_neurons[~torch.isin(all_neurons, top)]
                random = remaining[
                    torch.randperm(
                        remaining.numel(), device=remaining.device, generator=generator
                    )[:count]
                ]
                for selection, neurons in (("top", top), ("random", random)):
                    intervention = FunctionalIntervention(
                        site=site,
                        component="mlp_hidden",
                        mode="patch",
                        positions=(cfg.seq_len - 1,),
                        neurons=tuple(int(value) for value in neurons),
                    )
                    patched, _ = run_instrumented_state(
                        model,
                        states[base],
                        loop_indices=(loop_index,),
                        interventions=(intervention,),
                        donor_trace=traces[donor],
                    )
                    rows.append(
                        {
                            "cycle": cycle,
                            "block": site + 1,
                            "condition": condition,
                            "selection": selection,
                            "neuron_count": count,
                            "neuron_indices": " ".join(
                                str(int(value)) for value in neurons
                            ),
                            **_metrics(patched, target),
                            **_recovery_metrics(
                                clean_logits=logits[clean],
                                corrupt_logits=logits[corrupt],
                                patched_logits=patched,
                                target=target,
                            ),
                        }
                    )
    return rows


def _fixed_neuron_rows(
    *,
    model,
    cfg,
    loop_index: int,
    cycle: int,
    target: torch.Tensor,
    states: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
    neurons: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    if not neurons:
        return []
    neuron_count = traces["J"].sites[1].mlp_hidden.shape[-1]
    fixed = torch.as_tensor(neurons, device=target.device, dtype=torch.long)
    if fixed.unique().numel() != fixed.numel():
        raise ValueError("fixed MLP neuron indices must be unique")
    if bool((fixed < 0).any()) or bool((fixed >= neuron_count).any()):
        raise ValueError("fixed MLP neuron index is out of range")
    all_neurons = torch.arange(neuron_count, device=target.device)
    remaining = all_neurons[~torch.isin(all_neurons, fixed)]
    generator = torch.Generator(device=target.device).manual_seed(seed + cycle + 9000)
    random = remaining[
        torch.randperm(
            remaining.numel(), device=remaining.device, generator=generator
        )[: fixed.numel()]
    ]
    rows: list[dict[str, Any]] = []
    for selection, selected in (("fixed_seed211", fixed), ("random", random)):
        intervention = FunctionalIntervention(
            site=1,
            component="mlp_hidden",
            mode="patch",
            positions=(cfg.seq_len - 1,),
            neurons=tuple(int(value) for value in selected),
        )
        patched, _ = run_instrumented_state(
            model,
            states["J"],
            loop_indices=(loop_index,),
            interventions=(intervention,),
            donor_trace=traces["exact_H7"],
        )
        rows.append(
            {
                "cycle": cycle,
                "block": 2,
                "condition": "exact_H7_into_J",
                "selection": selection,
                "neuron_count": int(selected.numel()),
                "neuron_indices": " ".join(str(int(value)) for value in selected),
                **_metrics(patched, target),
                **_recovery_metrics(
                    clean_logits=logits["exact_H7"],
                    corrupt_logits=logits["J"],
                    patched_logits=patched,
                    target=target,
                ),
            }
        )
    return rows


def _candidate_circuit_rows(
    *,
    model,
    cfg,
    loop_index: int,
    cycle: int,
    target: torch.Tensor,
    states: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
    neurons: Sequence[int],
    seed: int,
    random_draws: int,
) -> list[dict[str, Any]]:
    """Test a preregistered B2 H0 + MLP-neuron candidate circuit.

    The neuron set must be fixed outside the evaluated controller seed.  This
    function records circuit-only, complement-only, leave-one-part-out, and
    size-matched random repair, plus native zero/shuffle necessity controls.
    """
    if not neurons:
        return []
    answer = (cfg.seq_len - 1,)
    site = 1
    neuron_count = traces["J"].sites[site].mlp_hidden.shape[-1]
    fixed = torch.as_tensor(neurons, device=target.device, dtype=torch.long)
    if fixed.unique().numel() != fixed.numel():
        raise ValueError("candidate MLP neuron indices must be unique")
    if bool((fixed < 0).any()) or bool((fixed >= neuron_count).any()):
        raise ValueError("candidate MLP neuron index is out of range")
    all_neurons = torch.arange(neuron_count, device=target.device)
    complement = all_neurons[~torch.isin(all_neurons, fixed)]

    def component(
        name: str,
        *,
        mode: str = "patch",
        heads: Sequence[int] | None = None,
        selected_neurons: torch.Tensor | None = None,
    ) -> FunctionalIntervention:
        return FunctionalIntervention(
            site=site,
            component=name,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            positions=answer,
            heads=None if heads is None else tuple(int(v) for v in heads),
            neurons=(
                None
                if selected_neurons is None
                else tuple(int(v) for v in selected_neurons)
            ),
        )

    h0 = component("head_context", heads=(0,))
    other_heads = component(
        "head_context", heads=tuple(range(1, cfg.n_heads))
    )
    all_heads = component("head_context", heads=tuple(range(cfg.n_heads)))
    top_mlp = component("mlp_hidden", selected_neurons=fixed)
    other_mlp = component("mlp_hidden", selected_neurons=complement)
    all_mlp = component("mlp_hidden", selected_neurons=all_neurons)

    rows: list[dict[str, Any]] = []

    def evaluate(
        label: str,
        interventions: Sequence[FunctionalIntervention],
        *,
        donor: FunctionalTrace | None,
        condition: str,
        random_draw: int | None = None,
    ) -> None:
        head_indices = sorted(
            {
                int(value)
                for item in interventions
                if item.component == "head_context"
                for value in (item.heads or ())
            }
        )
        neuron_indices = sorted(
            {
                int(value)
                for item in interventions
                if item.component == "mlp_hidden"
                for value in (item.neurons or ())
            }
        )
        patched, _ = run_instrumented_state(
            model,
            states["J"],
            loop_indices=(loop_index,),
            interventions=tuple(interventions),
            donor_trace=donor,
        )
        row: dict[str, Any] = {
            "cycle": cycle,
            "effective_loop": cfg.max_loops + cycle,
            "candidate": label,
            "condition": condition,
            "head_count": sum(
                len(item.heads or ())
                for item in interventions
                if item.component == "head_context"
            ),
            "mlp_neuron_count": sum(
                len(item.neurons or ())
                for item in interventions
                if item.component == "mlp_hidden"
            ),
            "head_indices": " ".join(str(value) for value in head_indices),
            "mlp_neuron_indices": (
                " ".join(str(value) for value in neuron_indices)
                if len(neuron_indices) <= 128
                else "recorded_by_candidate_definition"
            ),
            **_metrics(patched, target),
            **_recovery_metrics(
                clean_logits=logits["exact_H7"],
                corrupt_logits=logits["J"],
                patched_logits=patched,
                target=target,
            ),
        }
        if random_draw is not None:
            row["random_draw"] = random_draw
        rows.append(row)

    exact = traces["exact_H7"]
    evaluate(
        "selected_H0_plus_fixed_MLP",
        (h0, top_mlp),
        donor=exact,
        condition="exact_H7_into_J",
    )
    evaluate(
        "selected_H0_only",
        (h0,),
        donor=exact,
        condition="exact_H7_into_J_leave_MLP_out",
    )
    evaluate(
        "selected_fixed_MLP_only",
        (top_mlp,),
        donor=exact,
        condition="exact_H7_into_J_leave_H0_out",
    )
    evaluate(
        "complement_H1_H3_plus_nonfixed_MLP",
        (other_heads, other_mlp),
        donor=exact,
        condition="exact_H7_into_J_complement_only",
    )
    evaluate(
        "all_heads_plus_all_MLP",
        (all_heads, all_mlp),
        donor=exact,
        condition="exact_H7_into_J_positive_control",
    )

    shuffled = _rolled_trace(traces["J"])
    evaluate(
        "selected_H0_plus_fixed_MLP",
        (h0, top_mlp),
        donor=shuffled,
        condition="shuffled_J_into_J",
    )
    evaluate(
        "selected_H0_plus_fixed_MLP",
        (
            replace(h0, mode="zero"),
            replace(top_mlp, mode="zero"),
        ),
        donor=None,
        condition="zero_in_J",
    )

    generator = torch.Generator(device=target.device).manual_seed(
        seed + cycle + 20000
    )
    for draw in range(random_draws):
        random_head = int(
            torch.randint(
                1,
                cfg.n_heads,
                (1,),
                generator=generator,
                device=target.device,
            )
        )
        random_neurons = complement[
            torch.randperm(
                complement.numel(),
                generator=generator,
                device=target.device,
            )[: fixed.numel()]
        ]
        random_head_intervention = component(
            "head_context", heads=(random_head,)
        )
        random_mlp_intervention = component(
            "mlp_hidden", selected_neurons=random_neurons
        )
        evaluate(
            f"random_H{random_head}_plus_random_MLP",
            (random_head_intervention, random_mlp_intervention),
            donor=exact,
            condition="exact_H7_into_J_size_matched_random",
            random_draw=draw,
        )
    return rows


def _hybrid_rows(
    *,
    model,
    cfg,
    loop_index: int,
    cycle: int,
    target: torch.Tensor,
    states: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
    conditions: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    pairs = (
        ("J_into_no_J", "no_J", "J", "J", "no_J"),
        ("no_J_into_J", "J", "no_J", "J", "no_J"),
        ("exact_H7_into_J", "J", "exact_H7", "exact_H7", "J"),
        ("J_into_exact_H7", "exact_H7", "J", "J", "exact_H7"),
        ("shuffled_J_into_no_J", "no_J", "shuffled_J", "J", "no_J"),
        ("shuffled_J_into_J", "J", "shuffled_J", "J", "no_J"),
    )
    shuffled = _rolled_trace(traces["J"])
    donor_traces = {**traces, "shuffled_J": shuffled}
    requested = set(conditions) if conditions is not None else None
    rows: list[dict[str, Any]] = []
    for spec in _component_specs(cfg):
        for condition, base, donor, clean, corrupt in pairs:
            if requested is not None and condition not in requested:
                continue
            patched = _patched_run(
                model=model,
                initial_state=states[base],
                loop_index=loop_index,
                spec=spec,
                donor_trace=donor_traces[donor],
            )
            rows.append(
                {
                    "cycle": cycle,
                    "effective_loop": cfg.max_loops + cycle,
                    "component": spec.label,
                    "condition": condition,
                    **_metrics(patched, target),
                    **_recovery_metrics(
                        clean_logits=logits[clean],
                        corrupt_logits=logits[corrupt],
                        patched_logits=patched,
                        target=target,
                    ),
                }
            )
        if requested is None or "zero_in_J" in requested:
            zero_intervention = replace(spec.intervention, mode="zero")
            zero_logits, _ = run_instrumented_state(
                model,
                states["J"],
                loop_indices=(loop_index,),
                interventions=(zero_intervention,),
            )
            zero_metrics = _metrics(zero_logits, target)
            j_metrics = _metrics(logits["J"], target)
            rows.append(
                {
                    "cycle": cycle,
                    "effective_loop": cfg.max_loops + cycle,
                    "component": spec.label,
                    "condition": "zero_in_J",
                    **zero_metrics,
                    "accuracy_drop_from_J": (
                        j_metrics["accuracy"] - zero_metrics["accuracy"]
                    ),
                    "margin_drop_from_J": (
                        j_metrics["target_margin"] - zero_metrics["target_margin"]
                    ),
                }
            )
    return rows


def _stage_rows(
    *,
    model,
    cycle: int,
    target: torch.Tensor,
    states: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, trace in traces.items():
        stages = (
            ("loop_input", states[run]),
            ("B1_post_attention", trace.sites[0].residual_mid),
            ("B1_post_mlp", trace.sites[0].hidden_out),
            ("B2_post_attention", trace.sites[1].residual_mid),
            ("B2_post_mlp", trace.sites[1].hidden_out),
        )
        for stage, state in stages:
            rows.append(
                {
                    "cycle": cycle,
                    "run": run,
                    "stage": stage,
                    **_metrics(logits_from_raw_state(model, state), target),
                }
            )
    return rows


def _subset_masks(
    logits: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    correct = logits.argmax(dim=-1).eq(target)
    return {
        "all": torch.ones_like(correct, dtype=torch.bool),
        "correct": correct,
        "incorrect": ~correct,
    }


def _moment_row(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    selected = value[mask].float()
    if selected.numel() == 0:
        return {
            "examples": 0,
            "feature_mean": float("nan"),
            "feature_std": float("nan"),
            "feature_rms": float("nan"),
            "vector_norm": float("nan"),
        }
    return {
        "examples": int(mask.sum()),
        "feature_mean": float(selected.mean()),
        "feature_std": float(selected.std(unbiased=False)),
        "feature_rms": float(selected.square().mean().sqrt()),
        "vector_norm": float(selected.norm(dim=-1).mean()),
    }


def _norm_and_mlp_rows(
    *,
    model,
    cycle: int,
    target: torch.Tensor,
    logits: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    norm_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    masks = _subset_masks(logits["J"], target)
    for run, trace in traces.items():
        for site, site_trace in enumerate(trace.sites):
            block = model.blocks[site_trace.block_index]
            for stage, raw, normalized in (
                (
                    "pre_attention",
                    site_trace.hidden_in[:, -1],
                    block.ln_1(site_trace.hidden_in)[:, -1],
                ),
                (
                    "pre_mlp",
                    site_trace.residual_mid[:, -1],
                    block.ln_2(site_trace.residual_mid)[:, -1],
                ),
            ):
                for subset, mask in masks.items():
                    norm_rows.append(
                        {
                            "cycle": cycle,
                            "run": run,
                            "block": site + 1,
                            "stage": stage,
                            "subset_by_J_outcome": subset,
                            **{f"raw_{k}": v for k, v in _moment_row(raw, mask).items()},
                            **{
                                f"normalized_{k}": v
                                for k, v in _moment_row(normalized, mask).items()
                            },
                        }
                    )
            for subset, mask in masks.items():
                hidden = site_trace.mlp_hidden[:, -1]
                row = _moment_row(hidden, mask)
                selected = hidden[mask].float()
                row["positive_fraction"] = (
                    float(selected.gt(0).float().mean())
                    if selected.numel()
                    else float("nan")
                )
                mlp_rows.append(
                    {
                        "cycle": cycle,
                        "run": run,
                        "block": site + 1,
                        "subset_by_J_outcome": subset,
                        **row,
                    }
                )
    return norm_rows, mlp_rows


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().reshape(-1, left.shape[-1])
    right_flat = right.float().reshape(-1, right.shape[-1])
    return float(F.cosine_similarity(left_flat, right_flat, dim=-1).mean())


def _j_contribution_rows(
    *,
    operator: DiagonalIdentityLoRAJ,
    state: torch.Tensor,
    exact: torch.Tensor,
    cycle: int,
    cfg,
) -> list[dict[str, Any]]:
    diagonal = state.float() * (operator.diagonal_scale - 1.0)
    low_rank = (state.float() @ operator.A) @ operator.B
    bias = operator.bias.view(1, 1, -1).expand_as(state)
    total = diagonal + low_rank + bias
    exact_delta = exact.float() - state.float()
    groups = explicit_depth_position_groups(cfg.node_count)
    selected_groups = {
        "answer": groups["answer"],
        "graph": groups["graph"],
        "query_metadata": groups["query_metadata"],
        "all": tuple(range(cfg.seq_len)),
    }
    rows: list[dict[str, Any]] = []
    for group, positions in selected_groups.items():
        index = torch.as_tensor(positions, device=state.device)
        target = exact_delta[:, index]
        full = total[:, index]
        selected_diagonal = diagonal[:, index]
        selected_low_rank = low_rank[:, index]
        selected_bias = bias[:, index]
        for name, value in (
            ("D_minus_I", selected_diagonal),
            ("AB", selected_low_rank),
            ("bias", selected_bias),
            ("total_J_minus_I", full),
        ):
            rows.append(
                {
                    "cycle": cycle,
                    "position_group": group,
                    "J_component": name,
                    "mean_vector_norm": float(value.norm(dim=-1).mean()),
                    "cosine_with_total_J_correction": _cosine(value, full),
                    "cosine_with_exact_H7_delta": _cosine(value, target),
                    "relative_MSE_to_exact_H7_delta": float(
                        (value - target).square().mean()
                        / target.square().mean().clamp_min(1e-12)
                    ),
                    "cosine_D_with_AB": _cosine(
                        selected_diagonal, selected_low_rank
                    ),
                    "cosine_D_with_bias": _cosine(
                        selected_diagonal, selected_bias
                    ),
                    "cosine_AB_with_bias": _cosine(
                        selected_low_rank, selected_bias
                    ),
                }
            )
    return rows


def _relative_change(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left.float() - right.float()).square().mean().sqrt()
        / right.float().square().mean().sqrt().clamp_min(1e-12)
    )


def _dynamic_current_destination_attention(
    *,
    trace: FunctionalTrace,
    successors: torch.Tensor,
    current: torch.Tensor,
    site: int,
    head: int,
) -> float:
    destination_position = 3 + 3 * current
    batch = torch.arange(current.shape[0], device=current.device)
    return float(
        trace.sites[site].attention_pattern[
            batch, head, -1, destination_position
        ].mean()
    )


def _permutation_cycle_length(
    successors: torch.Tensor, current: torch.Tensor
) -> torch.Tensor:
    batch = torch.arange(successors.shape[0], device=successors.device)
    probe = current.clone()
    length = torch.zeros_like(current)
    for step in range(1, successors.shape[1] + 1):
        probe = successors[batch, probe]
        newly_closed = probe.eq(current) & length.eq(0)
        length[newly_closed] = step
    if bool(length.eq(0).any()):
        raise ValueError("successor rows must be permutations")
    return length


def _per_example_rows(
    *,
    cycle: int,
    start: torch.Tensor,
    endpoint: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    successors: torch.Tensor,
    states: dict[str, torch.Tensor],
    logits: dict[str, torch.Tensor],
    traces: dict[str, FunctionalTrace],
) -> list[dict[str, Any]]:
    cycle_length = _permutation_cycle_length(successors, current)
    batch = torch.arange(target.shape[0], device=target.device)
    rows: list[dict[str, Any]] = []
    successors_list = successors.detach().cpu().tolist()
    start_list = start.detach().cpu().tolist()
    endpoint_list = endpoint.detach().cpu().tolist()
    current_list = current.detach().cpu().tolist()
    target_list = target.detach().cpu().tolist()
    cycle_length_list = cycle_length.detach().cpu().tolist()
    for run, run_logits in logits.items():
        probability = run_logits.float().softmax(dim=-1)[batch, target]
        margin = target_margin(run_logits.float(), target)
        prediction = run_logits.argmax(dim=-1)
        trace = traces[run]
        b2_pattern = trace.sites[1].attention_pattern
        current_destination_position = 3 + 3 * current
        current_source_position = 2 + 3 * current
        head_current_destination_attention = torch.stack(
            [
                b2_pattern[batch, head, -1, current_destination_position]
                for head in range(b2_pattern.shape[1])
            ],
            dim=1,
        )
        head0_current_source_attention = b2_pattern[
            batch, 0, -1, current_source_position
        ]
        head0_top_position = b2_pattern[:, 0, -1].argmax(dim=-1)
        destination_positions = 3 + 3 * torch.arange(
            successors.shape[1], device=successors.device
        )
        head0_top_destination_node = b2_pattern[
            :, 0, -1, destination_positions
        ].argmax(dim=-1)
        input_norm = states[run][:, -1].float().norm(dim=-1)
        b1_norm = trace.sites[0].hidden_out[:, -1].float().norm(dim=-1)
        b2_attention_norm = trace.sites[1].residual_mid[:, -1].float().norm(dim=-1)
        b2_output_norm = trace.sites[1].hidden_out[:, -1].float().norm(dim=-1)
        mlp_hidden_norm = trace.sites[1].mlp_hidden[:, -1].float().norm(dim=-1)
        prediction_list = prediction.detach().cpu().tolist()
        probability_list = probability.detach().cpu().tolist()
        margin_list = margin.detach().cpu().tolist()
        head_current_destination_attention_list = (
            head_current_destination_attention.detach().cpu().tolist()
        )
        head0_current_source_attention_list = (
            head0_current_source_attention.detach().cpu().tolist()
        )
        head0_top_position_list = head0_top_position.detach().cpu().tolist()
        head0_top_destination_node_list = (
            head0_top_destination_node.detach().cpu().tolist()
        )
        input_norm_list = input_norm.detach().cpu().tolist()
        b1_norm_list = b1_norm.detach().cpu().tolist()
        b2_attention_norm_list = b2_attention_norm.detach().cpu().tolist()
        b2_output_norm_list = b2_output_norm.detach().cpu().tolist()
        mlp_hidden_norm_list = mlp_hidden_norm.detach().cpu().tolist()
        for index in range(target.shape[0]):
            rows.append(
                {
                    "cycle": cycle,
                    "sample": index,
                    "run": run,
                    "successors": " ".join(
                        str(int(value)) for value in successors_list[index]
                    ),
                    "start": int(start_list[index]),
                    "endpoint": int(endpoint_list[index]),
                    "current": int(current_list[index]),
                    "target": int(target_list[index]),
                    "current_cycle_length": int(cycle_length_list[index]),
                    "prediction": int(prediction_list[index]),
                    "correct": int(prediction_list[index] == target_list[index]),
                    "target_probability": float(probability_list[index]),
                    "target_margin": float(margin_list[index]),
                    "B2H0_current_source_attention": float(
                        head0_current_source_attention_list[index]
                    ),
                    "B2H0_top_attended_position": int(
                        head0_top_position_list[index]
                    ),
                    "B2H0_top_destination_node": int(
                        head0_top_destination_node_list[index]
                    ),
                    **{
                        f"B2H{head}_current_destination_attention": float(
                            head_current_destination_attention_list[index][head]
                        )
                        for head in range(b2_pattern.shape[1])
                    },
                    "loop_input_answer_norm": float(input_norm_list[index]),
                    "B1_output_answer_norm": float(b1_norm_list[index]),
                    "B2_post_attention_answer_norm": float(
                        b2_attention_norm_list[index]
                    ),
                    "B2_output_answer_norm": float(b2_output_norm_list[index]),
                    "B2_MLP_hidden_norm": float(mlp_hidden_norm_list[index]),
                }
            )
    return rows


def _j_mode_per_example_rows(
    *,
    operator: DiagonalIdentityLoRAJ,
    state: torch.Tensor,
    target: torch.Tensor,
    successors: torch.Tensor,
    current: torch.Tensor,
    logits: torch.Tensor,
    trace: FunctionalTrace,
    cycle: int,
    modes: Sequence[int],
) -> list[dict[str, Any]]:
    correction = operator.A.float() @ operator.B.float()
    left, singular_values, _ = torch.linalg.svd(
        correction, full_matrices=False
    )
    requested = [int(mode) for mode in modes]
    if any(mode < 0 or mode >= operator.rank for mode in requested):
        raise ValueError("requested J mode is outside the learned rank")
    answer = state[:, -1].float()
    coefficients = answer @ left[:, requested]
    contribution_scalars = coefficients * singular_values[requested]
    batch = torch.arange(target.shape[0], device=target.device)
    margin = target_margin(logits.float(), target)
    correct = logits.argmax(dim=-1).eq(target)
    cycle_length = _permutation_cycle_length(successors, current)
    destination_position = 3 + 3 * current
    head0_attention = trace.sites[1].attention_pattern[
        batch, 0, -1, destination_position
    ]
    rows: list[dict[str, Any]] = []
    for column, mode in enumerate(requested):
        for sample in range(target.shape[0]):
            rows.append(
                {
                    "cycle": cycle,
                    "sample": sample,
                    "mode": mode,
                    "singular_value": float(singular_values[mode]),
                    "left_coordinate": float(coefficients[sample, column]),
                    "signed_AB_contribution": float(
                        contribution_scalars[sample, column]
                    ),
                    "absolute_AB_contribution": float(
                        contribution_scalars[sample, column].abs()
                    ),
                    "current_cycle_length": int(cycle_length[sample]),
                    "correct": int(correct[sample]),
                    "target_margin": float(margin[sample]),
                    "B2H0_current_destination_attention": float(
                        head0_attention[sample]
                    ),
                }
            )
    return rows


def _random_projection_per_example_rows(
    *,
    state: torch.Tensor,
    target: torch.Tensor,
    successors: torch.Tensor,
    current: torch.Tensor,
    logits: torch.Tensor,
    trace: FunctionalTrace,
    cycle: int,
    dimension: int,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    if draws <= 0:
        return []
    if dimension < 1 or dimension > state.shape[-1]:
        raise ValueError("random projection dimension is invalid")
    answer = state[:, -1].float()
    batch = torch.arange(target.shape[0], device=target.device)
    margin = target_margin(logits.float(), target)
    correct = logits.argmax(dim=-1).eq(target)
    cycle_length = _permutation_cycle_length(successors, current)
    destination_position = 3 + 3 * current
    head0_attention = trace.sites[1].attention_pattern[
        batch, 0, -1, destination_position
    ]
    rows: list[dict[str, Any]] = []
    for draw in range(draws):
        generator = torch.Generator(device=state.device).manual_seed(
            seed + cycle * 1000 + draw
        )
        random_matrix = torch.randn(
            state.shape[-1],
            dimension,
            generator=generator,
            device=state.device,
            dtype=torch.float32,
        )
        basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
        coordinates = answer @ basis
        for coordinate in range(dimension):
            for sample in range(target.shape[0]):
                rows.append(
                    {
                        "cycle": cycle,
                        "sample": sample,
                        "projection_draw": draw,
                        "coordinate": coordinate,
                        "projection_value": float(
                            coordinates[sample, coordinate]
                        ),
                        "current_cycle_length": int(cycle_length[sample]),
                        "correct": int(correct[sample]),
                        "target_margin": float(margin[sample]),
                        "B2H0_current_destination_attention": float(
                            head0_attention[sample]
                        ),
                    }
                )
    return rows


@torch.no_grad()
def _mode_rows(
    *,
    model,
    cfg,
    operator: DiagonalIdentityLoRAJ,
    state: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    successors: torch.Tensor,
    loop_index: int,
    cycle: int,
    full_logits: torch.Tensor,
    full_trace: FunctionalTrace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    correction = operator.A.float() @ operator.B.float()
    left, singular_values, right_t = torch.linalg.svd(
        correction, full_matrices=False
    )
    rank = operator.rank
    behavior_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    full_metrics = _metrics(full_logits, target)
    base = (
        state.float() * operator.diagonal_scale
        + operator.bias.view(1, 1, -1)
    )
    for mode in range(rank):
        mode_update = (
            (state.float() @ left[:, mode : mode + 1])
            * singular_values[mode]
        ) @ right_t[mode : mode + 1]
        boundary = operator(state) - mode_update
        ablated_logits, ablated_trace = run_instrumented_state(
            model,
            boundary,
            loop_indices=(loop_index,),
        )
        metrics = _metrics(ablated_logits, target)
        behavior_rows.append(
            {
                "cycle": cycle,
                "mode": mode,
                "singular_value": float(singular_values[mode]),
                "mean_mode_update_norm": float(mode_update.norm(dim=-1).mean()),
                **metrics,
                "accuracy_drop": full_metrics["accuracy"] - metrics["accuracy"],
                "probability_drop": (
                    full_metrics["target_probability"]
                    - metrics["target_probability"]
                ),
                "margin_drop": full_metrics["target_margin"] - metrics["target_margin"],
            }
        )
        for site in range(cfg.n_layers):
            for head in range(cfg.n_heads):
                component_rows.append(
                    {
                        "cycle": cycle,
                        "mode": mode,
                        "singular_value": float(singular_values[mode]),
                        "block": site + 1,
                        "head": head,
                        "q_answer_relative_change": _relative_change(
                            ablated_trace.sites[site].q[:, head, -1],
                            full_trace.sites[site].q[:, head, -1],
                        ),
                        "context_answer_relative_change": _relative_change(
                            ablated_trace.sites[site].head_context[:, head, -1],
                            full_trace.sites[site].head_context[:, head, -1],
                        ),
                        "current_destination_attention_full": (
                            _dynamic_current_destination_attention(
                                trace=full_trace,
                                successors=successors,
                                current=current,
                                site=site,
                                head=head,
                            )
                        ),
                        "current_destination_attention_mode_removed": (
                            _dynamic_current_destination_attention(
                                trace=ablated_trace,
                                successors=successors,
                                current=current,
                                site=site,
                                head=head,
                            )
                        ),
                    }
                )
    for count in (1, 2, 4, 8, 16, 24, 32, 48):
        selected_left = left[:, :count]
        selected = (
            (state.float() @ selected_left) * singular_values[:count]
        ) @ right_t[:count]
        for condition, boundary in (
            ("top_modes_removed", operator(state) - selected),
            ("top_modes_only", base + selected),
        ):
            group_logits, _ = run_instrumented_state(
                model, boundary, loop_indices=(loop_index,)
            )
            metrics = _metrics(group_logits, target)
            behavior_rows.append(
                {
                    "cycle": cycle,
                    "mode": f"top{count}",
                    "condition": condition,
                    "singular_value": float(singular_values[:count].sum()),
                    "mean_mode_update_norm": float(selected.norm(dim=-1).mean()),
                    **metrics,
                    "accuracy_drop": full_metrics["accuracy"] - metrics["accuracy"],
                    "probability_drop": (
                        full_metrics["target_probability"]
                        - metrics["target_probability"]
                    ),
                    "margin_drop": full_metrics["target_margin"] - metrics["target_margin"],
                }
            )
    return behavior_rows, component_rows


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
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
    artifact_checkpoint, positions, operators, artifact_payload = (
        load_task_lora_modules(args.operator_artifact, device=device)
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if artifact_payload.get("placement") != "loop_boundary":
        raise ValueError("analysis requires a loop-boundary J")
    if args.operator_label not in operators:
        raise ValueError(f"missing operator label: {args.operator_label}")
    operator = operators[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ):
        raise TypeError("analysis requires DiagonalIdentityLoRAJ")
    if operator.rank != 48:
        raise ValueError("analysis is preregistered for rank 48")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("analysis requires J at every token position")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    set_seed(args.seed)
    _, path_targets, successors, start = fixed_depth_batch(
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

    requested = set(args.cycles)
    mode_cycles = requested & set(args.mode_cycles)
    baseline_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    mode_component_rows: list[dict[str, Any]] = []
    mlp_neuron_rows: list[dict[str, Any]] = []
    candidate_circuit_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    j_mode_per_example_rows: list[dict[str, Any]] = []
    random_projection_rows: list[dict[str, Any]] = []

    for cycle in range(1, max(requested) + 1):
        current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
        target = advance_nodes(successors, endpoint, steps=jump * cycle)
        if cycle in requested:
            loop_index = cfg.max_loops + cycle - 1
            states = {
                "no_J": state,
                "J": operator(state),
                "exact_H7": _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=7,
                    phase_position=phase_positions[7],
                ),
            }
            logits: dict[str, torch.Tensor] = {}
            traces: dict[str, FunctionalTrace] = {}
            for run, initial in states.items():
                run_logits, run_trace = run_instrumented_state(
                    model, initial, loop_indices=(loop_index,)
                )
                logits[run] = run_logits
                traces[run] = run_trace
                baseline_rows.append(
                    {
                        "cycle": cycle,
                        "effective_loop": cfg.max_loops + cycle,
                        "run": run,
                        **_metrics(run_logits, target),
                    }
                )
            hybrid_rows.extend(
                _hybrid_rows(
                    model=model,
                    cfg=cfg,
                    loop_index=loop_index,
                    cycle=cycle,
                    target=target,
                    states=states,
                    logits=logits,
                    traces=traces,
                    conditions=args.hybrid_conditions,
                )
            )
            if not args.skip_joint_hybrids:
                hybrid_rows.extend(
                    _joint_hybrid_rows(
                        model=model,
                        cfg=cfg,
                        loop_index=loop_index,
                        cycle=cycle,
                        target=target,
                        states=states,
                        logits=logits,
                        traces=traces,
                    )
                )
            mlp_neuron_rows.extend(
                _mlp_neuron_hybrid_rows(
                    model=model,
                    cfg=cfg,
                    loop_index=loop_index,
                    cycle=cycle,
                    target=target,
                    states=states,
                    logits=logits,
                    traces=traces,
                    top_ks=args.mlp_top_ks,
                    seed=args.seed,
                )
            )
            mlp_neuron_rows.extend(
                _fixed_neuron_rows(
                    model=model,
                    cfg=cfg,
                    loop_index=loop_index,
                    cycle=cycle,
                    target=target,
                    states=states,
                    logits=logits,
                    traces=traces,
                    neurons=args.fixed_mlp_neurons,
                    seed=args.seed,
                )
            )
            candidate_circuit_rows.extend(
                _candidate_circuit_rows(
                    model=model,
                    cfg=cfg,
                    loop_index=loop_index,
                    cycle=cycle,
                    target=target,
                    states=states,
                    logits=logits,
                    traces=traces,
                    neurons=args.fixed_mlp_neurons,
                    seed=args.seed,
                    random_draws=args.candidate_random_draws,
                )
            )
            stage_rows.extend(
                _stage_rows(
                    model=model,
                    cycle=cycle,
                    target=target,
                    states=states,
                    traces=traces,
                )
            )
            matched = MatchedAgeBatch(
                terminal=state,
                young=states["exact_H7"],
                successors=successors,
                current=current,
                target=target,
                endpoint=endpoint,
            )
            current_attention = attention_function_rows(
                cfg=cfg, batch=matched, traces=traces
            )
            for row in current_attention:
                row["cycle"] = cycle
            attention_rows.extend(current_attention)
            current_norm, current_mlp = _norm_and_mlp_rows(
                model=model,
                cycle=cycle,
                target=target,
                logits=logits,
                traces=traces,
            )
            norm_rows.extend(current_norm)
            mlp_rows.extend(current_mlp)
            contribution_rows.extend(
                _j_contribution_rows(
                    operator=operator,
                    state=state,
                    exact=states["exact_H7"],
                    cycle=cycle,
                    cfg=cfg,
                )
            )
            per_example_rows.extend(
                _per_example_rows(
                    cycle=cycle,
                    start=start,
                    endpoint=endpoint,
                    current=current,
                    target=target,
                    successors=successors,
                    states=states,
                    logits=logits,
                    traces=traces,
                )
            )
            j_mode_per_example_rows.extend(
                _j_mode_per_example_rows(
                    operator=operator,
                    state=state,
                    target=target,
                    successors=successors,
                    current=current,
                    logits=logits["J"],
                    trace=traces["J"],
                    cycle=cycle,
                    modes=args.per_example_modes,
                )
            )
            random_projection_rows.extend(
                _random_projection_per_example_rows(
                    state=state,
                    target=target,
                    successors=successors,
                    current=current,
                    logits=logits["J"],
                    trace=traces["J"],
                    cycle=cycle,
                    dimension=operator.rank,
                    draws=args.random_projection_draws,
                    seed=args.random_projection_seed,
                )
            )
            if cycle in mode_cycles:
                current_modes, current_mode_components = _mode_rows(
                    model=model,
                    cfg=cfg,
                    operator=operator,
                    state=state,
                    target=target,
                    current=current,
                    successors=successors,
                    loop_index=loop_index,
                    cycle=cycle,
                    full_logits=logits["J"],
                    full_trace=traces["J"],
                )
                mode_rows.extend(current_modes)
                mode_component_rows.extend(current_mode_components)

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "baseline": "baseline.csv",
        "component_hybrids": "component_hybrids.csv",
        "stage_readout": "stage_readout.csv",
        "attention_semantics": "attention_semantics.csv",
        "layernorm_statistics": "layernorm_statistics.csv",
        "mlp_statistics": "mlp_statistics.csv",
        "J_contributions": "J_contributions.csv",
        "J_mode_ablation": "J_mode_ablation.csv",
        "J_mode_component_effects": "J_mode_component_effects.csv",
        "MLP_neuron_hybrids": "MLP_neuron_hybrids.csv",
        "candidate_circuit": "candidate_circuit.csv",
        "per_example": "per_example.csv",
        "J_mode_per_example": "J_mode_per_example.csv",
        "random_projection_per_example": "random_projection_per_example.csv",
    }
    for filename, rows in (
        (files["baseline"], baseline_rows),
        (files["component_hybrids"], hybrid_rows),
        (files["stage_readout"], stage_rows),
        (files["attention_semantics"], attention_rows),
        (files["layernorm_statistics"], norm_rows),
        (files["mlp_statistics"], mlp_rows),
        (files["J_contributions"], contribution_rows),
        (files["J_mode_ablation"], mode_rows),
        (files["J_mode_component_effects"], mode_component_rows),
        (files["MLP_neuron_hybrids"], mlp_neuron_rows),
        (files["candidate_circuit"], candidate_circuit_rows),
        (files["per_example"], per_example_rows),
        (files["J_mode_per_example"], j_mode_per_example_rows),
        (files["random_projection_per_example"], random_projection_rows),
    ):
        _write_csv(args.out_dir / filename, rows)
    summary = {
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
        "mode_cycles": sorted(mode_cycles),
        "hybrid_conditions": list(args.hybrid_conditions),
        "joint_hybrids_skipped": bool(args.skip_joint_hybrids),
        "files": files,
        "gpu_peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank-48 diagonal-low-rank J circuit audit."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--cycles", type=int, nargs="+", default=(1, 2, 8, 16, 32, 48, 64)
    )
    parser.add_argument("--mode-cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument(
        "--mlp-top-ks", type=int, nargs="+", default=(16, 64, 256)
    )
    parser.add_argument("--fixed-mlp-neurons", type=int, nargs="*", default=())
    parser.add_argument("--candidate-random-draws", type=int, default=0)
    parser.add_argument(
        "--per-example-modes", type=int, nargs="*", default=(0, 2, 17)
    )
    parser.add_argument("--random-projection-draws", type=int, default=0)
    parser.add_argument("--random-projection-seed", type=int, default=20260801)
    parser.add_argument(
        "--hybrid-conditions",
        nargs="+",
        default=(
            "J_into_no_J",
            "no_J_into_J",
            "exact_H7_into_J",
            "J_into_exact_H7",
            "shuffled_J_into_no_J",
            "shuffled_J_into_J",
            "zero_in_J",
        ),
    )
    parser.add_argument("--skip-joint-hybrids", action="store_true")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.06)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(run_experiment(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
