from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.graph_path_loop import (
    TransformerBlock,
    cosine_lr,
    count_parameters,
    pick_device,
    set_seed,
)

TaskName = Literal["A", "B"]


@dataclass
class MultiTaskGraphConfig:
    node_count: int = 8
    max_depth: int = 6
    program_length: int = 12
    d_model: int = 256
    n_heads: int = 4
    d_mlp: int = 1024
    n_layers: int = 2
    max_loops: int = 6
    dropout: float = 0.0
    overlap_mode: str = "disjoint"
    loss_mode: str = "final_only"
    task_b_kind: str = "program"
    query_mode: str = "fixed_depth"

    @property
    def seq_len(self) -> int:
        # BOS TASK, two relation-map slots, QUERY [DEPTH] START, program slots, ANSWER.
        depth_slot = 1 if self.query_mode == "input_depth" else 0
        return 2 + 4 * 2 * self.node_count + 2 + depth_slot + self.program_length + 1

    @property
    def vocab_size(self) -> int:
        return make_token_scheme(self).vocab_size


@dataclass(frozen=True)
class TokenScheme:
    node_a: tuple[int, ...]
    node_b: tuple[int, ...]
    rel_a0: int
    rel_b0: int
    rel_b1: int
    map_a: int
    map_b: int
    query_a: int
    query_b: int
    answer_a: int
    answer_b: int
    task_a: int
    task_b: int
    bos_a: int
    bos_b: int
    pad_a: int
    pad_b: int
    depth_tokens: tuple[int, ...]
    vocab_size: int

    def nodes(self, task: TaskName) -> tuple[int, ...]:
        return self.node_a if task == "A" else self.node_b

    def bos(self, task: TaskName) -> int:
        return self.bos_a if task == "A" else self.bos_b

    def task_token(self, task: TaskName) -> int:
        return self.task_a if task == "A" else self.task_b

    def map_token(self, task: TaskName) -> int:
        return self.map_a if task == "A" else self.map_b

    def query_token(self, task: TaskName) -> int:
        return self.query_a if task == "A" else self.query_b

    def answer_token(self, task: TaskName) -> int:
        return self.answer_a if task == "A" else self.answer_b

    def pad_token(self, task: TaskName) -> int:
        return self.pad_a if task == "A" else self.pad_b

    def rel_tokens(self, task: TaskName) -> tuple[int, ...]:
        return (self.rel_a0,) if task == "A" else (self.rel_b0, self.rel_b1)

    def depth_token(self, depth: int) -> int:
        if not self.depth_tokens:
            raise ValueError("depth tokens are only available when query_mode='input_depth'")
        return self.depth_tokens[depth - 1]

    def namespace_sets(self) -> dict[str, set[int]]:
        a_nodes = set(self.node_a)
        b_nodes = set(self.node_b)
        shared = a_nodes & b_nodes
        a_only = a_nodes - b_nodes
        b_only = b_nodes - a_nodes
        special = set(range(self.vocab_size)) - (a_nodes | b_nodes)
        return {
            "shared_nodes": shared,
            "a_only_nodes": a_only,
            "b_only_nodes": b_only,
            "special": special,
        }


def make_token_scheme(cfg: MultiTaskGraphConfig) -> TokenScheme:
    n = cfg.node_count
    if cfg.overlap_mode == "disjoint":
        node_a = tuple(range(0, n))
        node_b = tuple(range(n, 2 * n))
        rel_a0 = 2 * n
        rel_b0 = rel_a0 + 1
        rel_b1 = rel_a0 + 2
        map_a = rel_a0 + 3
        map_b = rel_a0 + 4
        query_a = rel_a0 + 5
        query_b = rel_a0 + 6
        answer_a = rel_a0 + 7
        answer_b = rel_a0 + 8
        task_a = rel_a0 + 9
        task_b = rel_a0 + 10
        bos_a = rel_a0 + 11
        bos_b = rel_a0 + 12
        pad_a = rel_a0 + 13
        pad_b = rel_a0 + 14
        vocab_size = rel_a0 + 15
    elif cfg.overlap_mode == "partial":
        shared = n // 2
        node_a = tuple(range(0, n))
        node_b = tuple(range(0, shared)) + tuple(range(n, n + (n - shared)))
        next_id = n + (n - shared)
        rel_a0 = next_id
        rel_b0 = next_id + 1
        rel_b1 = next_id + 2
        bos_a = bos_b = next_id + 3
        query_a = query_b = next_id + 4
        answer_a = answer_b = next_id + 5
        map_a = next_id + 6
        map_b = next_id + 7
        task_a = next_id + 8
        task_b = next_id + 9
        pad_a = pad_b = next_id + 10
        vocab_size = next_id + 11
    elif cfg.overlap_mode == "full":
        node_a = node_b = tuple(range(0, n))
        rel_a0 = rel_b0 = n
        rel_b1 = n + 1
        map_a = map_b = n + 2
        query_a = query_b = n + 3
        answer_a = answer_b = n + 4
        task_a = n + 5
        task_b = n + 6
        bos_a = bos_b = n + 7
        pad_a = pad_b = n + 8
        vocab_size = n + 9
    else:
        raise ValueError(f"unknown overlap_mode={cfg.overlap_mode!r}")
    if cfg.query_mode == "input_depth":
        depth_tokens = tuple(range(vocab_size, vocab_size + cfg.max_depth))
        vocab_size += cfg.max_depth
    elif cfg.query_mode == "fixed_depth":
        depth_tokens = ()
    else:
        raise ValueError(f"unknown query_mode={cfg.query_mode!r}")
    return TokenScheme(
        node_a=node_a,
        node_b=node_b,
        rel_a0=rel_a0,
        rel_b0=rel_b0,
        rel_b1=rel_b1,
        map_a=map_a,
        map_b=map_b,
        query_a=query_a,
        query_b=query_b,
        answer_a=answer_a,
        answer_b=answer_b,
        task_a=task_a,
        task_b=task_b,
        bos_a=bos_a,
        bos_b=bos_b,
        pad_a=pad_a,
        pad_b=pad_b,
        depth_tokens=depth_tokens,
        vocab_size=vocab_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multitask looped graph-path models.")
    parser.add_argument("--overlap-mode", choices=["disjoint", "partial", "full"], required=True)
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--program-length", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--loops", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--loss-mode", choices=["final_only", "intermediate"], default="final_only")
    parser.add_argument("--task-b-kind", choices=["program", "alternating"], default="program")
    parser.add_argument("--query-mode", choices=["fixed_depth", "input_depth"], default="fixed_depth")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--task-b-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/multitask_graph_path"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class LoopedMultiTaskTransformer(nn.Module):
    def __init__(self, cfg: MultiTaskGraphConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward_all(self, tokens: torch.Tensor, *, max_loops: int | None = None) -> dict[str, torch.Tensor]:
        loops = self.cfg.max_loops if max_loops is None else max_loops
        x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        logits_by_loop: list[torch.Tensor] = []
        for _ in range(loops):
            for block in self.blocks:
                x = block(x)
            final_state = self.ln_final(x[:, -1, :])
            logits_by_loop.append(self.unembed(final_state))
        return {"logits_by_loop": torch.stack(logits_by_loop, dim=1)}


def make_task_batch(
    cfg: MultiTaskGraphConfig,
    scheme: TokenScheme,
    batch_size: int,
    task: TaskName,
    device: torch.device,
    *,
    path_positions: int | None = None,
    query_depth_override: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = cfg.max_depth if path_positions is None else path_positions
    if positions > cfg.program_length:
        raise ValueError("path_positions must be <= program_length")
    if positions < cfg.max_depth:
        raise ValueError("path_positions must be >= max_depth")

    nodes = torch.tensor(scheme.nodes(task), dtype=torch.long, device=device)
    src_idx = torch.arange(cfg.node_count, device=device).view(1, cfg.node_count).expand(batch_size, -1)
    src_tokens = nodes[src_idx]
    start_idx = torch.randint(0, cfg.node_count, (batch_size,), dtype=torch.long, device=device)
    start_tokens = nodes[start_idx]

    tokens = torch.full(
        (batch_size, cfg.seq_len),
        scheme.pad_token(task),
        dtype=torch.long,
        device=device,
    )
    tokens[:, 0] = scheme.bos(task)
    tokens[:, 1] = scheme.task_token(task)

    map_start = 2
    query_start = 2 + 4 * 2 * cfg.node_count
    depth_slot = 1 if cfg.query_mode == "input_depth" else 0
    program_start = query_start + 2 + depth_slot
    tokens[:, query_start] = scheme.query_token(task)
    if cfg.query_mode == "input_depth":
        if query_depth_override is None:
            query_depth = torch.randint(1, cfg.max_depth + 1, (batch_size,), dtype=torch.long, device=device)
        else:
            if not 1 <= query_depth_override <= cfg.max_depth:
                raise ValueError("query_depth_override must be in [1, max_depth]")
            query_depth = torch.full((batch_size,), query_depth_override, dtype=torch.long, device=device)
        depth_token_tensor = torch.tensor(scheme.depth_tokens, dtype=torch.long, device=device)
        tokens[:, query_start + 1] = depth_token_tensor[query_depth - 1]
    elif cfg.query_mode == "fixed_depth":
        query_depth = torch.full((batch_size,), cfg.max_depth, dtype=torch.long, device=device)
    else:
        raise ValueError(f"unknown query_mode={cfg.query_mode!r}")
    tokens[:, query_start + 1 + depth_slot] = start_tokens
    tokens[:, -1] = scheme.answer_token(task)

    targets_by_pos = torch.empty(batch_size, positions, dtype=torch.long, device=device)
    if task == "A":
        noise = torch.rand(batch_size, cfg.node_count, device=device)
        successors = noise.argsort(dim=-1)
        dst_tokens = nodes[successors]
        quads = torch.empty(batch_size, cfg.node_count, 4, dtype=torch.long, device=device)
        quads[:, :, 0] = scheme.map_token(task)
        quads[:, :, 1] = scheme.rel_a0
        quads[:, :, 2] = src_tokens
        quads[:, :, 3] = dst_tokens
        tokens[:, map_start : map_start + 4 * cfg.node_count] = quads.reshape(batch_size, 4 * cfg.node_count)
        tokens[:, program_start : program_start + cfg.program_length] = scheme.rel_a0
        current = start_idx
        for step in range(positions):
            current = successors.gather(1, current.view(-1, 1)).squeeze(1)
            targets_by_pos[:, step] = nodes[current]
    else:
        noise = torch.rand(batch_size, 2, cfg.node_count, device=device)
        successors = noise.argsort(dim=-1)
        rel_tokens = torch.tensor((scheme.rel_b0, scheme.rel_b1), dtype=torch.long, device=device)
        rel_idx = torch.arange(2, device=device).view(1, 2, 1).expand(batch_size, -1, cfg.node_count)
        src_idx_3 = torch.arange(cfg.node_count, device=device).view(1, 1, cfg.node_count).expand(batch_size, 2, -1)
        quads = torch.empty(batch_size, 2, cfg.node_count, 4, dtype=torch.long, device=device)
        quads[:, :, :, 0] = scheme.map_token(task)
        quads[:, :, :, 1] = rel_tokens[rel_idx]
        quads[:, :, :, 2] = nodes[src_idx_3]
        quads[:, :, :, 3] = nodes[successors]
        tokens[:, map_start : map_start + 8 * cfg.node_count] = quads.reshape(batch_size, 8 * cfg.node_count)
        if cfg.task_b_kind == "program":
            program = torch.randint(0, 2, (batch_size, cfg.program_length), dtype=torch.long, device=device)
        elif cfg.task_b_kind == "alternating":
            pattern = torch.arange(cfg.program_length, dtype=torch.long, device=device) % 2
            program = pattern.view(1, cfg.program_length).expand(batch_size, -1)
        else:
            raise ValueError(f"unknown task_b_kind={cfg.task_b_kind!r}")
        tokens[:, program_start : program_start + cfg.program_length] = rel_tokens[program]
        current = start_idx
        batch_idx = torch.arange(batch_size, device=device)
        for step in range(positions):
            step_rel = program[:, step]
            selected_successors = successors[batch_idx, step_rel, :]
            current = selected_successors.gather(1, current.view(-1, 1)).squeeze(1)
            targets_by_pos[:, step] = nodes[current]
    return tokens, targets_by_pos, query_depth


def make_mixed_batch(
    cfg: MultiTaskGraphConfig,
    scheme: TokenScheme,
    batch_size: int,
    device: torch.device,
    *,
    task_b_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= task_b_fraction <= 1.0:
        raise ValueError("task_b_fraction must be in [0, 1]")
    n_b = int(round(batch_size * task_b_fraction))
    n_a = batch_size - n_b
    tokens_a, targets_a, query_depth_a = make_task_batch(cfg, scheme, n_a, "A", device, path_positions=cfg.max_depth)
    tokens_b, targets_b, query_depth_b = make_task_batch(cfg, scheme, n_b, "B", device, path_positions=cfg.max_depth)
    tokens = torch.cat([tokens_a, tokens_b], dim=0)
    targets = torch.cat([targets_a, targets_b], dim=0)
    query_depth = torch.cat([query_depth_a, query_depth_b], dim=0)
    perm = torch.randperm(batch_size, device=device)
    return tokens[perm], targets[perm], query_depth[perm]


def ce_to_positions(logits_by_loop: torch.Tensor, targets_by_pos: torch.Tensor) -> torch.Tensor:
    batch, loops, vocab = logits_by_loop.shape
    target = targets_by_pos[:, :loops].reshape(batch * loops)
    return F.cross_entropy(
        logits_by_loop.reshape(batch * loops, vocab),
        target,
        reduction="none",
    ).view(batch, loops)


def ce_final_only(
    logits_by_loop: torch.Tensor,
    targets_by_pos: torch.Tensor,
    query_depth: torch.Tensor,
) -> torch.Tensor:
    final_logits = logits_by_loop[:, -1, :]
    final_target = targets_by_pos.gather(1, (query_depth - 1).view(-1, 1)).squeeze(1)
    return F.cross_entropy(final_logits, final_target)


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    cfg: MultiTaskGraphConfig,
    scheme: TokenScheme,
    task: TaskName,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    max_loops: int,
    path_positions: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    correct_to_pos = torch.zeros(max_loops, path_positions, device=device)
    prob_to_pos = torch.zeros(max_loops, path_positions, device=device)
    correct_to_query = torch.zeros(max_loops, device=device)
    prob_to_query = torch.zeros(max_loops, device=device)
    entropy_sum = torch.zeros(max_loops, device=device)
    namespace_sets = scheme.namespace_sets()
    namespace_mass = {
        key: torch.zeros(max_loops, device=device)
        for key in ["shared_nodes", "a_only_nodes", "b_only_nodes", "special"]
    }
    count = 0
    autocast_device = "cuda" if device.type == "cuda" else device.type
    for _ in range(batches):
        tokens, targets_by_pos, query_depth = make_task_batch(
            cfg, scheme, batch_size, task, device, path_positions=path_positions
        )
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]
        probs_by_loop = logits_by_loop.softmax(dim=-1)
        pred = logits_by_loop.argmax(dim=-1)
        query_target = targets_by_pos.gather(1, (query_depth - 1).view(-1, 1)).squeeze(1)
        count += batch_size
        for loop_idx in range(max_loops):
            probs = probs_by_loop[:, loop_idx, :]
            entropy_sum[loop_idx] += (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).sum()
            correct_to_query[loop_idx] += pred[:, loop_idx].eq(query_target).float().sum()
            prob_to_query[loop_idx] += probs.gather(1, query_target[:, None]).squeeze(1).sum()
            for key, token_set in namespace_sets.items():
                if token_set:
                    idx = torch.tensor(sorted(token_set), dtype=torch.long, device=device)
                    namespace_mass[key][loop_idx] += probs[:, idx].sum(dim=-1).sum()
            for pos_idx in range(path_positions):
                target = targets_by_pos[:, pos_idx]
                correct_to_pos[loop_idx, pos_idx] += pred[:, loop_idx].eq(target).float().sum()
                prob_to_pos[loop_idx, pos_idx] += probs.gather(1, target[:, None]).squeeze(1).sum()
    acc_to_pos = correct_to_pos / count
    mean_prob_to_pos = prob_to_pos / count
    best_acc, best_pos = acc_to_pos.max(dim=1)
    rolling_idx = torch.arange(max_loops, device=device).clamp(max=path_positions - 1)
    rolling_acc = acc_to_pos[torch.arange(max_loops, device=device), rolling_idx]
    train_positions = torch.arange(min(max_loops, cfg.max_depth), device=device)
    train_acc = acc_to_pos[train_positions, train_positions].mean()
    query_acc = correct_to_query / count
    query_prob = prob_to_query / count
    final_target_acc = query_acc[cfg.max_loops - 1]
    metrics = {
        "acc_to_pos": acc_to_pos.detach().cpu().tolist(),
        "mean_prob_to_pos": mean_prob_to_pos.detach().cpu().tolist(),
        "rolling_acc_by_loop": [float(x) for x in rolling_acc.detach().cpu()],
        "query_acc_by_loop": [float(x) for x in query_acc.detach().cpu()],
        "query_prob_by_loop": [float(x) for x in query_prob.detach().cpu()],
        "best_position_by_loop": [int(x) + 1 for x in best_pos.detach().cpu().tolist()],
        "best_acc_by_loop": [float(x) for x in best_acc.detach().cpu()],
        "trained_step_mean_acc": float(train_acc.detach().cpu()),
        "final_target_acc": float(final_target_acc.detach().cpu()),
        "mean_entropy_by_loop": [float(x) for x in (entropy_sum / count).detach().cpu()],
        "namespace_mass_by_loop": {
            key: [float(x) for x in (value / count).detach().cpu()]
            for key, value in namespace_mass.items()
        },
    }
    if cfg.query_mode == "input_depth":
        depth_acc = torch.zeros(cfg.max_depth, max_loops, device=device)
        depth_prob = torch.zeros(cfg.max_depth, max_loops, device=device)
        for depth in range(1, cfg.max_depth + 1):
            depth_correct = torch.zeros(max_loops, device=device)
            depth_prob_sum = torch.zeros(max_loops, device=device)
            depth_count = 0
            for _ in range(batches):
                tokens, targets_by_pos, query_depth = make_task_batch(
                    cfg,
                    scheme,
                    batch_size,
                    task,
                    device,
                    path_positions=path_positions,
                    query_depth_override=depth,
                )
                with torch.autocast(
                    device_type=autocast_device,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled and device.type == "cuda",
                ):
                    logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]
                probs_by_loop = logits_by_loop.softmax(dim=-1)
                pred = logits_by_loop.argmax(dim=-1)
                target = targets_by_pos[:, depth - 1]
                depth_count += batch_size
                for loop_idx in range(max_loops):
                    probs = probs_by_loop[:, loop_idx, :]
                    depth_correct[loop_idx] += pred[:, loop_idx].eq(target).float().sum()
                    depth_prob_sum[loop_idx] += probs.gather(1, target[:, None]).squeeze(1).sum()
            depth_acc[depth - 1] = depth_correct / depth_count
            depth_prob[depth - 1] = depth_prob_sum / depth_count
        metrics["query_depth_acc"] = depth_acc.detach().cpu().tolist()
        metrics["query_depth_prob"] = depth_prob.detach().cpu().tolist()
    return metrics


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fields = [
        "step",
        "lr",
        "train_loss",
        "elapsed_sec",
        "task_a_step_acc",
        "task_b_step_acc",
        "task_a_final_acc",
        "task_b_final_acc",
    ]
    for idx in range(len(history[-1]["task_a_rolling"])):
        fields.append(f"task_a_rolling_loop_{idx + 1}")
        fields.append(f"task_b_rolling_loop_{idx + 1}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            flat = {key: row.get(key, "") for key in fields}
            for idx, value in enumerate(row["task_a_rolling"]):
                flat[f"task_a_rolling_loop_{idx + 1}"] = value
            for idx, value in enumerate(row["task_b_rolling"]):
                flat[f"task_b_rolling_loop_{idx + 1}"] = value
            writer.writerow(flat)


def save_heatmap(arr: np.ndarray, *, path: Path, title: str, label: str) -> None:
    loops, positions = arr.shape
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(positions), [str(i) for i in range(1, positions + 1)])
    ax.set_yticks(range(loops), [str(i) for i in range(1, loops + 1)])
    ax.set_xlabel("path position k")
    ax.set_ylabel("readout loop t")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_run(run_dir: Path, history: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if history:
        steps = [row["step"] for row in history]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(steps, [row["task_a_step_acc"] for row in history], marker="o", label="task A step mean")
        ax.plot(steps, [row["task_b_step_acc"] for row in history], marker="o", label="task B step mean")
        ax.plot(steps, [row["task_a_final_acc"] for row in history], marker="o", linestyle="--", label="task A final")
        ax.plot(steps, [row["task_b_final_acc"] for row in history], marker="o", linestyle="--", label="task B final")
        ax.set_xlabel("train step")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "task_step_accuracy_over_training.png", dpi=180)
        plt.close(fig)
    for task_key in ["task_a", "task_b"]:
        save_heatmap(
            np.array(summary["final_metrics"][task_key]["acc_to_pos"], dtype=np.float32),
            path=run_dir / f"{task_key}_loop_by_path_accuracy_heatmap.png",
            title=f"{task_key}: loop x path-position accuracy",
            label="accuracy",
        )
        if "query_depth_acc" in summary["final_metrics"][task_key]:
            save_heatmap(
                np.array(summary["final_metrics"][task_key]["query_depth_acc"], dtype=np.float32).T,
                path=run_dir / f"{task_key}_loop_by_query_depth_accuracy_heatmap.png",
                title=f"{task_key}: loop x input-query-depth accuracy",
                label="accuracy",
            )


def main() -> None:
    args = parse_args()
    cfg = MultiTaskGraphConfig(
        node_count=args.node_count,
        max_depth=args.max_depth,
        program_length=args.program_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        max_loops=args.loops,
        dropout=args.dropout,
        overlap_mode=args.overlap_mode,
        loss_mode=args.loss_mode,
        task_b_kind=args.task_b_kind,
        query_mode=args.query_mode,
    )
    if cfg.max_depth > cfg.program_length:
        raise ValueError("program_length must be >= max_depth")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / (
        f"multitask_{cfg.loss_mode}_{cfg.task_b_kind}_{cfg.query_mode}_{cfg.overlap_mode}_"
        f"N{cfg.node_count}_D{cfg.max_depth}_"
        f"P{cfg.program_length}_d{cfg.d_model}_B{cfg.n_layers}_L{cfg.max_loops}_seed{args.seed}"
    )
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"{run_dir} exists. Pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed + {"disjoint": 11, "partial": 23, "full": 37}[cfg.overlap_mode])
    device = pick_device(args.device)
    scheme = make_token_scheme(cfg)
    print(f"device={device} cfg={cfg} scheme={scheme}", flush=True)
    model: nn.Module = LoopedMultiTaskTransformer(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    param_count = count_parameters(model)
    metadata = {
        "config": asdict(cfg),
        "args": vars(args),
        "token_scheme": asdict(scheme),
        "parameter_count": param_count,
        "task": "multitask_graph_path",
        "run_dir": str(run_dir),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_step = 0
    start_time = time.time()
    autocast_device = "cuda" if device.type == "cuda" else device.type
    for step in range(1, args.steps + 1):
        model.train()
        lr = cosine_lr(step - 1, base_lr=args.lr, total_steps=args.steps, warmup_steps=args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tokens, targets_by_pos, query_depth = make_mixed_batch(
            cfg,
            scheme,
            args.batch_size,
            device,
            task_b_fraction=args.task_b_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=cfg.max_loops)["logits_by_loop"]  # type: ignore[union-attr]
            if cfg.loss_mode == "final_only":
                loss = ce_final_only(logits_by_loop, targets_by_pos, query_depth)
            else:
                loss = ce_to_positions(logits_by_loop, targets_by_pos).mean()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            task_a = evaluate_task(
                model,
                cfg,
                scheme,
                "A",
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                max_loops=cfg.max_loops,
                path_positions=cfg.max_depth,
                amp_enabled=args.amp,
            )
            task_b = evaluate_task(
                model,
                cfg,
                scheme,
                "B",
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                max_loops=cfg.max_loops,
                path_positions=cfg.max_depth,
                amp_enabled=args.amp,
            )
            if cfg.loss_mode == "final_only":
                score = 0.5 * (task_a["final_target_acc"] + task_b["final_target_acc"])
            else:
                score = 0.5 * (task_a["trained_step_mean_acc"] + task_b["trained_step_mean_acc"])
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.detach().cpu()),
                "elapsed_sec": time.time() - start_time,
                "task_a_step_acc": task_a["trained_step_mean_acc"],
                "task_b_step_acc": task_b["trained_step_mean_acc"],
                "task_a_final_acc": task_a["final_target_acc"],
                "task_b_final_acc": task_b["final_target_acc"],
                "task_a_rolling": task_a["rolling_acc_by_loop"],
                "task_b_rolling": task_b["rolling_acc_by_loop"],
            }
            history.append(row)
            write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if args.save_checkpoints and score > best_score:
                best_score = score
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "token_scheme": asdict(scheme),
                        "step": step,
                        "parameter_count": param_count,
                        "task": "multitask_graph_path",
                    },
                    run_dir / "best.pt",
                )
            if step % args.print_every == 0 or step == 1 or step == args.steps:
                print(
                    f"[{cfg.overlap_mode} step={step:05d}] loss={float(loss.detach().cpu()):.4f} "
                    f"A_step={task_a['trained_step_mean_acc']:.3f} "
                    f"B_step={task_b['trained_step_mean_acc']:.3f} "
                    f"A_final={task_a['final_target_acc']:.3f} "
                    f"B_final={task_b['final_target_acc']:.3f} "
                    f"A_roll={' '.join(f'{x:.2f}' for x in task_a['rolling_acc_by_loop'])} "
                    f"B_roll={' '.join(f'{x:.2f}' for x in task_b['rolling_acc_by_loop'])}",
                    flush=True,
                )

    final_a = evaluate_task(
        model,
        cfg,
        scheme,
        "A",
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 64),
        max_loops=cfg.max_loops,
        path_positions=cfg.max_depth,
        amp_enabled=args.amp,
    )
    final_b = evaluate_task(
        model,
        cfg,
        scheme,
        "B",
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 64),
        max_loops=cfg.max_loops,
        path_positions=cfg.max_depth,
        amp_enabled=args.amp,
    )
    summary = {
        "overlap_mode": cfg.overlap_mode,
        "parameter_count": param_count,
        "best_step": best_step,
        "best_mean_step_accuracy": best_score,
        "final_metrics": {"task_a": final_a, "task_b": final_b},
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_checkpoints:
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "token_scheme": asdict(scheme),
                "step": args.steps,
                "parameter_count": param_count,
                "task": "multitask_graph_path",
            },
            run_dir / "final.pt",
        )
    plot_run(run_dir, history, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
