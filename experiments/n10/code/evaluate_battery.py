"""Locked multi-backbone routing/content transplant and J-mediation battery."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def advance(graph: torch.Tensor, current: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(steps):
        current = graph.gather(1, current[:, None])[:, 0]
    return current


def select(trace: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[index] for key, value in trace.items()}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--looplus", type=Path, default=Path("/data/paperexperiment/LooPlus"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--controllers", type=Path, required=True)
    ap.add_argument("--controller-sha256")
    ap.add_argument("--rank", type=int, default=48)
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--backbone-name", required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--panel", choices=("smoke", "discovery", "confirmation"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.looplus))
    from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
    from reasoning_loop.graph_path_functional_circuit import FunctionalIntervention, run_instrumented_state
    from reasoning_loop.graph_path_jump_controller import apply_vector_map
    from reasoning_loop.graph_path_jump_controller_causal_switch import _load_controller

    assert 0 <= args.head < 4
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    model, cfg, payload = load_checkpoint(args.checkpoint, device)
    model.eval().requires_grad_(False)
    versions = [parameter._version for parameter in model.parameters()]
    assert sha256(args.checkpoint) == args.checkpoint_sha256
    assert payload["step"] == args.checkpoint_step
    assert cfg.max_loops == 6 and cfg.node_count == 10 and cfg.d_model == 256
    if args.controller_sha256:
        assert sha256(args.controllers) == args.controller_sha256

    maps = {
        seed: {
            "J_one": _load_controller(
                path=args.controllers,
                name=f"seed{seed}_J_one_beh_rank{args.rank}",
                device=device,
            ),
            "J_two": _load_controller(
                path=args.controllers,
                name=f"seed{seed}_J_two_beh_rank{args.rank}",
                device=device,
            ),
        }
        for seed in args.seeds
    }

    def read(answer_state: torch.Tensor) -> torch.Tensor:
        return model.unembed(model.ln_final(answer_state))[:, :cfg.node_count]

    def initial_state(graph: torch.Tensor, current: torch.Tensor, depth: int) -> torch.Tensor:
        start = advance(graph.argsort(-1), current, depth)
        tokens, _, _, _ = fixed_depth_batch(
            cfg,
            len(current),
            device,
            path_positions=10,
            successors=graph,
            start=start,
        )
        return model.token_embed(tokens) + model.pos_embed[None]

    def state(graph: torch.Tensor, current: torch.Tensor, depth: int, age: int) -> torch.Tensor:
        hidden = initial_state(graph, current, depth)
        for loop in range(age):
            hidden = model.apply_loop(hidden, loop_index=loop)
        return hidden

    def trace(hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, instrumented = run_instrumented_state(model, hidden, loop_indices=(6,))
        site = instrumented.sites[1]
        return {
            "res": site.hidden_in[:, -1].clone(),
            "q": site.q[:, :, -1].clone(),
            "v": site.v.clone(),
            "k": site.k.clone(),
            "pat": site.attention_pattern[:, :, -1].clone(),
            "ctx": site.head_context[:, :, -1].clone(),
            "mlp": site.mlp_out[:, -1].clone(),
            "pre": read(hidden[:, -1]).argmax(-1),
            "logits": logits,
            "pred": logits.argmax(-1),
        }

    def fast(
        base: dict[str, torch.Tensor],
        q: torch.Tensor | None = None,
        pat: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        mlp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ctx is None:
            if pat is None:
                pat = base["pat"] if q is None else (
                    (q[:, :, None, :] @ base["k"].transpose(-2, -1)) / 8
                ).softmax(-1).squeeze(2)
            values = base["v"] if v is None else v
            ctx = (pat[:, :, None, :] @ values).squeeze(2)
        block = model.blocks[1]
        residual = base["res"] + block.attn.out_proj(ctx.reshape(len(ctx), 256))
        result = residual + (block.mlp(block.ln_2(residual)) if mlp is None else mlp)
        return read(result)

    def bundle(graph: torch.Tensor, current: torch.Tensor, seed: int) -> dict[str, dict[str, torch.Tensor]]:
        h6 = state(graph, current, 8, 6)
        output = {
            "native_one": trace(state(graph, current, 7, 5)),
            "native_two": trace(state(graph, current, 2, 1)),
            "identity": trace(h6),
        }
        for name, controller in maps[seed].items():
            output[name] = trace(
                apply_vector_map(h6, positions=tuple(range(cfg.seq_len)), controller=controller)
            )
        return output

    datasets = json.loads(args.datasets.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out / "datasets.json"
    if dataset_path.exists():
        assert json.loads(dataset_path.read_text()) == datasets
    else:
        save_json(dataset_path, datasets)

    panel_dir = args.out / args.panel
    panel_dir.mkdir(exist_ok=True)
    event_path = panel_dir / "events.jsonl.gz"
    checks: dict[str, float] = {}
    started = time.monotonic()
    event_count = 0

    with gzip.open(event_path, "wt") as event_file:
        def emit(
            kind: str,
            receiver: str,
            seed: int,
            condition: str,
            prediction: torch.Tensor,
            target: torch.Tensor,
            original: torch.Tensor,
            eligible: torch.Tensor,
            graph_ids: torch.Tensor,
            donor_prediction: torch.Tensor | None = None,
            donor_target: torch.Tensor | None = None,
            extra: dict[str, object] | None = None,
        ) -> None:
            nonlocal event_count
            row: dict[str, object] = {
                "kind": kind,
                "receiver": receiver,
                "controller_seed": seed,
                "condition": condition,
                "graph_ids": graph_ids.cpu().tolist(),
                "prediction": prediction.cpu().tolist(),
                "target": target.cpu().tolist(),
                "original": original.cpu().tolist(),
                "eligible": eligible.cpu().tolist(),
            }
            if donor_prediction is not None:
                row["donor_prediction"] = donor_prediction.cpu().tolist()
            if donor_target is not None:
                row["donor_target"] = donor_target.cpu().tolist()
            if extra:
                row.update(extra)
            event_file.write(json.dumps(row) + "\n")
            event_count += 1

        # Three implementation checks precede all outcomes.
        graph = torch.tensor(datasets[args.panel][:1], device=device).repeat_interleave(cfg.node_count, 0)
        current = torch.arange(cfg.node_count, device=device)
        hidden = state(graph, current, 7, 5)
        base_trace = trace(hidden)
        direct = read(model.apply_loop(hidden, loop_index=5)[:, -1])
        quick = fast(base_trace)
        checks["unmodified_logits_max_error"] = float((quick - direct).abs().max())
        assert torch.equal(quick.argmax(-1), direct.argmax(-1))

        donor_index = (torch.arange(cfg.node_count, device=device) + 3) % cfg.node_count
        donor_trace = select(base_trace, donor_index)
        patched_pattern = base_trace["pat"].clone()
        patched_pattern[:, args.head] = donor_trace["pat"][:, args.head]
        _, full_donor = run_instrumented_state(model, hidden[donor_index], loop_indices=(6,))
        reference, _ = run_instrumented_state(
            model,
            hidden,
            loop_indices=(6,),
            interventions=(
                FunctionalIntervention(
                    site=1,
                    component="attention_pattern",
                    mode="patch",
                    heads=(args.head,),
                    positions=(cfg.seq_len - 1,),
                ),
            ),
            donor_trace=full_donor,
        )
        manual = fast(base_trace, pat=patched_pattern)
        checks["pattern_patch_logits_max_error"] = float((manual - reference).abs().max())
        assert torch.equal(manual.argmax(-1), reference.argmax(-1))

        def context_hook(_module: object, hook_args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor]:
            value = hook_args[0].clone().reshape(cfg.node_count, cfg.seq_len, cfg.n_heads, cfg.d_model // cfg.n_heads)
            value[:, -1, args.head] = donor_trace["ctx"][:, args.head]
            return (value.reshape(cfg.node_count, cfg.seq_len, cfg.d_model),)

        handle = model.blocks[1].attn.out_proj.register_forward_pre_hook(context_hook)
        direct_context = read(model.apply_loop(hidden, loop_index=5)[:, -1])
        handle.remove()
        patched_context = base_trace["ctx"].clone()
        patched_context[:, args.head] = donor_trace["ctx"][:, args.head]
        manual_context = fast(base_trace, ctx=patched_context)
        checks["independent_context_hook_error"] = float((manual_context - direct_context).abs().max())
        assert torch.equal(manual_context.argmax(-1), direct_context.argmax(-1))
        checks["self_patch_error"] = float((fast(base_trace, pat=base_trace["pat"]) - quick).abs().max())
        save_json(panel_dir / "validation.json", checks)
        print(json.dumps({"event": "validation", **checks}), flush=True)

        for first in range(0, len(datasets[args.panel]), 4):
            graph_list = datasets[args.panel][first:first + 4]
            graph = torch.tensor(graph_list, device=device).repeat_interleave(cfg.node_count, 0)
            current = torch.arange(cfg.node_count, device=device).repeat(len(graph_list))
            row_index = torch.arange(len(graph), device=device)
            graph_ids = row_index // cfg.node_count + first
            shifted_index = row_index // cfg.node_count * cfg.node_count + (current + 3) % cfg.node_count
            wrong_current = (current + 3) % cfg.node_count

            for seed in args.seeds:
                traces = bundle(graph, current, seed)
                for name, run in traces.items():
                    jump = 2 if name in ("J_two", "native_two") else 1
                    target = advance(graph, current, jump)
                    emit(
                        "baseline", name, seed, "clean", run["pred"], target, run["pred"],
                        torch.ones(len(graph), dtype=torch.bool, device=device), graph_ids,
                        extra={"current_correct": run["pre"].eq(current).cpu().tolist(), "current": current.cpu().tolist(), "one_target": advance(graph,current,1).cpu().tolist(), "two_target": advance(graph,current,2).cpu().tolist(), "pre_prediction": run["pre"].cpu().tolist()},
                    )

                # Unconditional matched-input swaps: do not filter by J success.
                raw = traces["identity"]
                for label in ("J_one", "J_two"):
                    steered = traces[label]
                    target = advance(graph, current, 2 if label == "J_two" else 1)
                    for receiving, source, direction in ((raw,steered,"rescue"),(steered,raw,"damage")):
                        for heads in ((args.head,), (0,1,2,3)):
                            pattern = receiving["pat"].clone()
                            pattern[:,heads] = source["pat"][:,heads]
                            emit("unconditional_swap",label,seed,direction+"_H"+"".join(map(str,heads)),
                                 fast(receiving,pat=pattern).argmax(-1),target,receiving["pred"],
                                 torch.ones_like(current,dtype=torch.bool),graph_ids,
                                 extra={"current":current.cpu().tolist(),"pre_prediction":steered["pre"].cpu().tolist()})

                # Related-graph donors change the selected address while preserving receiver values.
                donors: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
                for kind, donor_address in (
                    ("one", wrong_current),
                    ("two", advance(graph, wrong_current, 1)),
                ):
                    donor_graph = graph.clone()
                    replacement = (donor_address + 1) % cfg.node_count
                    if kind == "two":
                        replacement = torch.where(replacement == wrong_current, (replacement + 1) % cfg.node_count, replacement)
                    temporary = donor_graph[row_index, donor_address].clone()
                    donor_graph[row_index, donor_address] = donor_graph[row_index, replacement]
                    donor_graph[row_index, replacement] = temporary
                    donor_bundle = bundle(donor_graph, wrong_current, seed)
                    for name in (("native_one", "J_one") if kind == "one" else ("J_two",)):
                        donors[name] = (donor_graph, donor_bundle[name])

                for name in ("native_one", "native_two", "J_one", "J_two"):
                    if seed != args.seeds[0] and name.startswith("native"):
                        continue
                    run = traces[name]
                    jump = 2 if name in ("J_two", "native_two") else 1
                    target = advance(graph, current, jump)
                    address = advance(graph, current, jump - 1)
                    correct = run["pre"].eq(current) & run["pred"].eq(target)
                    wrong_run = select(run, shifted_index)
                    wrong_target = advance(graph, wrong_current, jump)

                    # Same-graph current change, retained as a donor-copy-ambiguous diagnostic.
                    for component in ("q", "pat", "ctx"):
                        patched = run[component].clone()
                        patched[:, args.head] = wrong_run[component][:, args.head]
                        prediction = fast(run, **{component: patched}).argmax(-1)
                        emit(
                            "same_graph_current", name, seed, f"{component}_H{args.head}",
                            prediction, wrong_target, run["pred"],
                            correct & wrong_run["pre"].eq(wrong_current)
                            & wrong_run["pred"].eq(wrong_target) & wrong_target.ne(target),
                            graph_ids, wrong_run["pred"], wrong_target,
                        )

                    mass = run["pat"][:, args.head].gather(1, (3 + 3 * address)[:, None])[:, 0]
                    emit(
                        "routing", name, seed, f"H{args.head}_destination",
                        run["pat"][:, args.head].argmax(-1), 3 + 3 * address, run["pred"], correct,
                        graph_ids, extra={"mass": mass.cpu().tolist()},
                    )

                    for donor_name, (donor_graph, donor_run) in donors.items():
                        donor_jump = 2 if donor_name == "J_two" else 1
                        donor_address = advance(donor_graph, wrong_current, donor_jump - 1)
                        counterfactual = graph[row_index, donor_address]
                        donor_target = advance(donor_graph, wrong_current, donor_jump)
                        valid = (
                            correct & donor_run["pre"].eq(wrong_current)
                            & donor_run["pred"].eq(donor_target)
                            & counterfactual.ne(target)
                            & counterfactual.ne(donor_run["pred"])
                            & counterfactual.ne(donor_target)
                            & target.ne(donor_target)
                        )
                        for component in ("q", "pat", "ctx"):
                            patched = run[component].clone()
                            patched[:, args.head] = donor_run[component][:, args.head]
                            prediction = fast(run, **{component: patched}).argmax(-1)
                            emit(
                                "cross_graph_address", name, seed,
                                f"{donor_name}_{component}_H{args.head}", prediction,
                                counterfactual, run["pred"], valid, graph_ids,
                                donor_run["pred"], donor_target,
                                extra={"base_pre_correct":run["pre"].eq(current).cpu().tolist(),
                                       "base_post_correct":run["pred"].eq(target).cpu().tolist(),
                                       "donor_pre_correct":donor_run["pre"].eq(wrong_current).cpu().tolist(),
                                       "donor_post_correct":donor_run["pred"].eq(donor_target).cpu().tolist(),
                                       "labels_distinct":(counterfactual.ne(target)&counterfactual.ne(donor_target)&target.ne(donor_target)).cpu().tolist()},
                            )
                        if donor_name == "native_one":
                            onehot = run["pat"].clone()
                            onehot[:, args.head] = 0
                            onehot[row_index, args.head, 3 + 3 * donor_address] = 1
                            emit(
                                "address_onehot", name, seed, "native_one_address",
                                fast(run, pat=onehot).argmax(-1), counterfactual, run["pred"],
                                valid, graph_ids, donor_run["pred"], donor_target,
                            )

                    # Replace graph value vectors at selected and irrelevant addresses.
                    donor_graph = graph.clone()
                    replacement = (address + 1) % cfg.node_count
                    temporary = donor_graph[row_index, address].clone()
                    donor_graph[row_index, address] = donor_graph[row_index, replacement]
                    donor_graph[row_index, replacement] = temporary
                    donor_run = bundle(donor_graph, wrong_current, seed)[name]
                    donor_target = advance(donor_graph, wrong_current, jump)
                    counterfactual = donor_graph[row_index, address]
                    valid = (
                        correct & donor_run["pre"].eq(wrong_current)
                        & donor_run["pred"].eq(donor_target)
                        & counterfactual.ne(target)
                        & counterfactual.ne(donor_run["pred"])
                        & counterfactual.ne(donor_target)
                    )
                    for head in range(4):
                        values = run["v"].clone()
                        values[row_index, head, 3 + 3 * address] = donor_run["v"][row_index, head, 3 + 3 * address]
                        emit(
                            "edge_value", name, seed, f"destination_V_H{head}",
                            fast(run, v=values).argmax(-1), counterfactual, run["pred"],
                            valid, graph_ids, donor_run["pred"], donor_target,
                        )
                    values = run["v"].clone()
                    irrelevant = (address + 2) % cfg.node_count
                    values[row_index, args.head, 3 + 3 * irrelevant] = donor_run["v"][
                        row_index, args.head, 3 + 3 * irrelevant
                    ]
                    emit(
                        "edge_value", name, seed, f"irrelevant_V_H{args.head}",
                        fast(run, v=values).argmax(-1), counterfactual, run["pred"],
                        valid, graph_ids, donor_run["pred"], donor_target,
                    )

                    # Corrupt all answer queries with a wrong-current run and restore contexts.
                    broken_pattern = (
                        (wrong_run["q"][:, :, None, :] @ run["k"].transpose(-2, -1)) / 8
                    ).softmax(-1).squeeze(2)
                    broken_context = (broken_pattern[:, :, None, :] @ run["v"]).squeeze(2)
                    broken = fast(run, ctx=broken_context).argmax(-1)
                    mediation_ok = (
                        correct & wrong_run["pre"].eq(wrong_current)
                        & wrong_run["pred"].eq(wrong_target) & wrong_target.ne(target)
                    )
                    emit(
                        "block_error", name, seed, "wrong_current_all_Q", broken,
                        wrong_target, run["pred"], mediation_ok, graph_ids,
                        wrong_run["pred"], wrong_target,
                    )
                    emit(
                        "rescue", name, seed, "blocked", broken, target, run["pred"],
                        mediation_ok, graph_ids, extra={"broken": broken.ne(target).cpu().tolist()},
                    )
                    for heads in ((0,), (1,), (2,), (3,), (0, 1, 2, 3)):
                        contexts = broken_context.clone()
                        contexts[:, heads] = run["ctx"][:, heads]
                        emit(
                            "rescue", name, seed, "clean_context_H" + "".join(map(str, heads)),
                            fast(run, ctx=contexts).argmax(-1), target, run["pred"],
                            mediation_ok, graph_ids,
                            extra={"broken": broken.ne(target).cpu().tolist()},
                        )

                    native_address = select(traces["native_one"], row_index // cfg.node_count * cfg.node_count + address)
                    for component in ("pat", "ctx"):
                        patched = broken_pattern.clone() if component == "pat" else broken_context.clone()
                        patched[:, args.head] = native_address[component][:, args.head]
                        emit(
                            "rescue", name, seed, f"native_address_{component}_H{args.head}",
                            fast(run, **{component: patched}).argmax(-1), target, run["pred"],
                            mediation_ok & native_address["pred"].eq(target)
                            & native_address["pre"].eq(address),
                            graph_ids, extra={"broken": broken.ne(target).cpu().tolist()},
                        )

                    if name.startswith("J_"):
                        raw = traces["identity"]
                        raw_prediction = raw["pred"]
                        j_ok = correct & raw["pre"].eq(current)
                        emit(
                            "J_mediation", name, seed, "remove_J", raw_prediction,
                            target, run["pred"], j_ok, graph_ids,
                            extra={"broken": raw_prediction.ne(target).cpu().tolist()},
                        )
                        emit(
                            "J_mediation", name, seed, "executor_off", run["pre"],
                            target, run["pred"], j_ok, graph_ids,
                            extra={"broken": raw_prediction.ne(target).cpu().tolist()},
                        )
                        emit(
                            "J_mediation", name, seed, "full_J", run["pred"],
                            target, run["pred"], j_ok, graph_ids,
                            extra={"broken": raw_prediction.ne(target).cpu().tolist()},
                        )
                        for component in ("q", "pat", "v", "ctx"):
                            for heads in ((args.head,), (0, 1, 2, 3)):
                                patched = raw[component].clone()
                                patched[:, heads] = run[component][:, heads]
                                label = "candidate" if heads == (args.head,) else "all"
                                prediction = fast(raw, **{component: patched}).argmax(-1)
                                emit(
                                    "J_mediation", name, seed, f"J_{component}_{label}",
                                    prediction, target, run["pred"], j_ok, graph_ids,
                                    extra={"broken": raw_prediction.ne(target).cpu().tolist()},
                                )
                        for head in range(4):
                            contexts = raw["ctx"].clone()
                            contexts[:, head] = run["ctx"][:, head]
                            emit(
                                "J_mediation", name, seed, f"J_ctx_H{head}",
                                fast(raw, ctx=contexts).argmax(-1), target, run["pred"],
                                j_ok, graph_ids,
                                extra={"broken": raw_prediction.ne(target).cpu().tolist()},
                            )

            event_file.flush()
            print(json.dumps({
                "event": "progress",
                "panel": args.panel,
                "graphs": min(first + 4, len(datasets[args.panel])),
                "total": len(datasets[args.panel]),
                "seconds": round(time.monotonic() - started, 1),
            }), flush=True)

    assert versions == [parameter._version for parameter in model.parameters()]
    manifest = {
        "status": "complete",
        "family": args.family,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "backbone_name": args.backbone_name,
        "head": args.head,
        "panel": args.panel,
        "graphs": len(datasets[args.panel]),
        "seeds": args.seeds,
        "events": event_count,
        "seconds": time.monotonic() - started,
        "backbone_path": str(args.checkpoint),
        "backbone_sha256": sha256(args.checkpoint),
        "backbone_step": payload["step"],
        "controller_path": str(args.controllers),
        "controller_sha256": sha256(args.controllers),
        "controller_rank": args.rank,
        "dataset_sha256": sha256(dataset_path),
        "code_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "predictions_sha256": sha256(event_path),
        "parameters_unchanged": True,
    }
    save_json(panel_dir / "manifest.json", manifest)
    print(json.dumps({"event": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
