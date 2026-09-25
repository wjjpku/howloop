"""Train symmetric rank-128 J_one/J_two maps for a specified D8L6 backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--looplus", type=Path, default=Path("/data/wujiaju/LooPlus"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--cal-batches", type=int, default=32)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--device",default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(args.looplus))
    from reasoning_loop.graph_path_depth_circuit import load_checkpoint
    from reasoning_loop.graph_path_jump_controller import collect_jump_pair_batch, JumpMode, _flatten_positions
    from reasoning_loop.graph_path_telomere_localized_query import fit_reduced_rank_update_family
    from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
    from reasoning_loop.graph_path_loop import set_seed

    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, cfg, payload = load_checkpoint(args.checkpoint, device)
    model.eval().requires_grad_(False)
    assert sha256(args.checkpoint) == args.checkpoint_sha256
    assert payload["step"] == args.checkpoint_step
    assert cfg.max_loops == 6 and cfg.node_count == 10
    versions = [p._version for p in model.parameters()]
    two_mode = JumpMode(name="two", reference_age=1, reference_path_before=2, programmed_jump=2)

    class TrainableAffine(nn.Module):
        def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight.clone())
            self.bias = nn.Parameter(bias.clone())

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return state.float() @ self.weight + self.bias

    from reasoning_loop.paper2027_graph_g4_protocol import merged_forbidden_codes, sample_training_permutations
    root = Path(__file__).resolve().parents[1]
    forbidden = merged_forbidden_codes(list((root/'locks').glob('*.pt')),node_count=10)

    def collect(batches: int, seed: int) -> dict[str, torch.Tensor]:
        set_seed(seed)
        parts = {k: [] for k in ("source", "one_state", "two_state", "y8", "y9", "y10")}
        for _ in range(batches):
            graphs = sample_training_permutations(batch_size=args.batch_size,node_count=10,forbidden_codes=forbidden,device=device)
            pair = collect_jump_pair_batch(
                model=model,
                cfg=cfg,
                batch_size=args.batch_size,
                device=device,
                two_mode=two_mode,
                successors=graphs,
            )
            parts["source"].append(pair.terminal)
            parts["one_state"].append(pair.one_target_state)
            parts["two_state"].append(pair.two_target_state)
            parts["y8"].append(pair.all_targets[:, 8])
            parts["y9"].append(pair.all_targets[:, 9])
            parts["y10"].append(pair.all_targets[:, 10])
        return {k: torch.cat(v) for k, v in parts.items()}

    @torch.no_grad()
    def evaluate(ds: dict[str, torch.Tensor], weight: torch.Tensor, bias: torch.Tensor) -> dict[str, float | int]:
        pred, pre = [], []
        for lo in range(0, len(ds["source"]), 256):
            mapped = ds["source"][lo:lo + 256].float() @ weight + bias
            pre.append(logits_from_raw_state(model, mapped).argmax(-1))
            out = model.apply_loop(mapped, loop_index=cfg.max_loops)
            pred.append(logits_from_raw_state(model, out).argmax(-1))
        pred, pre = torch.cat(pred), torch.cat(pre)
        coll = (ds["y8"] != ds["y9"]) & (ds["y8"] != ds["y10"]) & (ds["y9"] != ds["y10"])
        result: dict[str, float | int] = {
            "n": int(len(pred)),
            "one": float(pred.eq(ds["y9"]).float().mean()),
            "two": float(pred.eq(ds["y10"]).float().mean()),
            "pre8": float(pre.eq(ds["y8"]).float().mean()),
            "n_collision_free": int(coll.sum()),
        }
        if coll.any():
            result.update({
                "one_collision_free": float(pred[coll].eq(ds["y9"][coll]).float().mean()),
                "two_collision_free": float(pred[coll].eq(ds["y10"][coll]).float().mean()),
                "pre8_collision_free": float(pre[coll].eq(ds["y8"][coll]).float().mean()),
            })
        return result

    eval_ds = collect(args.eval_batches, 20260921)
    saved: dict[str, dict[str, torch.Tensor]] = {}
    results: dict[str, object] = {}
    histories: dict[str, object] = {}
    started = time.time()
    for seed in args.seeds:
        ds = collect(args.cal_batches, 916921 + 1009 * seed)
        source_flat = _flatten_positions(ds["source"], tuple(range(cfg.seq_len)))
        init_by_kind = {
            "one": fit_reduced_rank_update_family(
                source_flat,
                _flatten_positions(ds["one_state"], tuple(range(cfg.seq_len))),
                ridge=1e-3,
            ).map_for_rank(cfg.d_model),
            "two": fit_reduced_rank_update_family(
                source_flat,
                _flatten_positions(ds["two_state"], tuple(range(cfg.seq_len))),
                ridge=1e-3,
            ).map_for_rank(cfg.d_model),
        }
        for kind, target_key, state_key in (("one", "y9", "one_state"), ("two", "y10", "two_state")):
            init = init_by_kind[kind]
            torch.manual_seed(17003 + seed)
            controller = TrainableAffine(init.weight, init.bias)
            optimizer = torch.optim.AdamW(controller.parameters(), lr=3e-4, weight_decay=1e-4)
            generator = torch.Generator().manual_seed(17003 + seed)
            rows = []
            for step in range(1, args.steps + 1):
                ix = torch.randint(0, len(ds["source"]), (args.batch_size,), generator=generator).to(device)
                source = ds["source"][ix]
                reference = ds[state_key][ix]
                mapped = controller(source)
                pre_logits = logits_from_raw_state(model, mapped)
                out = model.apply_loop(mapped, loop_index=cfg.max_loops)
                logits = logits_from_raw_state(model, out)
                task_loss = F.cross_entropy(logits, ds[target_key][ix])
                pre_loss = F.cross_entropy(pre_logits, ds["y8"][ix])
                centered = reference - reference.mean(0, keepdim=True)
                state_loss = (mapped - reference).square().mean() / centered.square().mean().clamp_min(1e-12)
                loss = task_loss + 10.0 * state_loss + pre_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
                optimizer.step()
                if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
                    row = {
                        "step": step,
                        "loss": float(loss),
                        "task_loss": float(task_loss),
                        "state_loss": float(state_loss),
                        "pre_loss": float(pre_loss),
                        "batch_accuracy": float(logits.argmax(-1).eq(ds[target_key][ix]).float().mean()),
                    }
                    rows.append(row)
                    print(json.dumps({"seed": seed, "kind": kind, **row}), flush=True)
            weight = controller.weight.detach()
            bias = controller.bias.detach()
            identity = torch.eye(weight.shape[0], device=device)
            u, singular, vh = torch.linalg.svd(weight - identity, full_matrices=False)
            rank_weight = identity + (u[:, :args.rank] * singular[:args.rank].unsqueeze(0)) @ vh[:args.rank]
            kept = float(singular[:args.rank].square().sum() / singular.square().sum().clamp_min(1e-12))
            name = f"seed{seed}_J_{kind}_beh_rank{args.rank}"
            saved[name] = {"weight": rank_weight.cpu(), "bias": bias.cpu()}
            results[name] = {"metrics": evaluate(eval_ds, rank_weight, bias), "residual_energy_kept": kept}
            histories[name] = rows

    args.out.mkdir(parents=True, exist_ok=True)
    controller_path = args.out / "controllers.pt"
    torch.save(saved, controller_path)
    manifest = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": payload["step"],
        "controller_path": str(controller_path),
        "controller_sha256": sha256(controller_path),
        "code_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "seeds": args.seeds,
        "steps": args.steps,
        "cal_batches": args.cal_batches,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "results": results,
        "diagnostic_eval_population": "random training-domain graphs; use separate locked mechanism confirmation for paper",
        "histories": histories,
        "elapsed_seconds": time.time() - started,
        "backbone_parameters_unchanged": versions == [p._version for p in model.parameters()],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    assert manifest["backbone_parameters_unchanged"]
    print(json.dumps({"event": "complete", "controller_sha256": manifest["controller_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
