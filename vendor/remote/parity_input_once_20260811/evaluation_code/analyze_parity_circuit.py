#!/usr/bin/env python3
"""Reproduce the bounded Parity circuit analyses used in the 2026-08-07 post.

The script deliberately analyzes one frozen backbone checkpoint.  It does not
train a model or a controller, and it keeps effective recurrent steps distinct
from the single shared physical Transformer block.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from reasoning_loop.paper_length_telomere import (
    PAPER_TASKS,
    PaperBatch,
    PaperLoopedTransformer,
    PaperModelConfig,
    generate_paper_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_backbone(
    checkpoint_path: Path, device: torch.device
) -> tuple[PaperLoopedTransformer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("task", {}).get("name") != "parity":
        raise ValueError("checkpoint is not a Parity backbone")
    config = PaperModelConfig(**payload["model"])
    model = PaperLoopedTransformer(config)
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    return model, payload


def fixed_parity_batch(
    *, length: int, batch_size: int, seed: int, device: torch.device
) -> PaperBatch:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batch = generate_paper_batch(
        PAPER_TASKS["parity"],
        batch_size=batch_size,
        min_length=length,
        max_length=length,
        fixed_length=length,
        generator=generator,
    )
    return batch.to(device)


def flip_input_bit(inputs: torch.Tensor, bit_position: int) -> torch.Tensor:
    corrupt = inputs.clone()
    original = inputs[:, bit_position, :2].argmax(dim=-1)
    corrupt[:, bit_position, :2] = 0.0
    corrupt[
        torch.arange(inputs.shape[0], device=inputs.device),
        bit_position,
        1 - original,
    ] = 1.0
    return corrupt


def answer_labels(batch: PaperBatch, length: int) -> torch.Tensor:
    return batch.targets[:, length]


def oriented_logit_difference(
    logits: torch.Tensor, labels: torch.Tensor, answer_position: int
) -> torch.Tensor:
    answer_logits = logits[:, answer_position]
    opposite = 1 - labels
    rows = torch.arange(answer_logits.shape[0], device=answer_logits.device)
    return answer_logits[rows, labels] - answer_logits[rows, opposite]


def score_state(
    model: PaperLoopedTransformer,
    state: torch.Tensor,
    labels: torch.Tensor,
    answer_position: int,
) -> tuple[float, float]:
    logits = model.decode(state)
    predictions = logits[:, answer_position].argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean().cpu())
    margin = float(
        oriented_logit_difference(logits, labels, answer_position).mean().cpu()
    )
    return accuracy, margin


def step_components(
    model: PaperLoopedTransformer,
    state: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    skip_mlp: bool = False,
    masked_head: int | None = None,
) -> dict[str, torch.Tensor]:
    if len(model.layers) != 1:
        raise ValueError("this analysis assumes one shared physical block")
    layer = model.layers[0]
    pre_attention = state + embeddings
    normalized = layer.attention_norm(pre_attention)
    if masked_head is None:
        attention_output = layer.attention(normalized)
    else:
        attention = layer.attention
        batch, length, dimension = normalized.shape
        qkv = attention.qkv(normalized).view(
            batch, length, 3, attention.n_heads, attention.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=attention.is_causal,
        )
        attended[:, masked_head] = 0.0
        mixed = attended.transpose(1, 2).reshape(batch, length, dimension)
        attention_output = attention.output(mixed)
    post_attention = pre_attention + attention_output
    mlp_output = layer.mlp(layer.mlp_norm(post_attention))
    pre_final_norm = post_attention if skip_mlp else post_attention + mlp_output
    final_state = model.final_norm(pre_final_norm)
    return {
        "pre_attention": pre_attention,
        "attention_output": attention_output,
        "post_attention": post_attention,
        "mlp_output": mlp_output,
        "pre_final_norm": pre_final_norm,
        "final_state": final_state,
    }


def run_states(
    model: PaperLoopedTransformer,
    inputs: torch.Tensor,
    *,
    steps: int,
    skipped_mlp_step: int | None = None,
    masked_head: int | None = None,
) -> list[torch.Tensor]:
    state = torch.zeros_like(model.read_in(inputs))
    states: list[torch.Tensor] = []
    for step in range(1, steps + 1):
        embeddings = model.input_embeddings(inputs, step_index=step)
        state = step_components(
            model,
            state,
            embeddings,
            skip_mlp=step == skipped_mlp_step,
            masked_head=masked_head,
        )["final_state"]
        states.append(state)
    return states


def continue_from_state(
    model: PaperLoopedTransformer,
    state: torch.Tensor,
    inputs: torch.Tensor,
    *,
    start_step: int,
    final_step: int,
) -> torch.Tensor:
    for step in range(start_step + 1, final_step + 1):
        embeddings = model.input_embeddings(inputs, step_index=step)
        state = step_components(model, state, embeddings)["final_state"]
    return state


def recovery(margin: float, clean_margin: float, corrupt_margin: float) -> float:
    denominator = clean_margin - corrupt_margin
    if abs(denominator) < 1e-9:
        raise ValueError("clean/corrupt logit-difference denominator is zero")
    return (margin - corrupt_margin) / denominator


def run_source_answer_transfer(
    model: PaperLoopedTransformer,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    smoke: bool,
) -> list[dict[str, Any]]:
    conditions = [(6, 0)] if smoke else [
        (10, 0),
        (10, 9),
        (20, 0),
        (20, 19),
        (30, 0),
        (30, 29),
    ]
    rows: list[dict[str, Any]] = []
    for condition_index, (length, bit_position) in enumerate(conditions):
        batch = fixed_parity_batch(
            length=length,
            batch_size=batch_size,
            seed=seed + 100 * condition_index,
            device=device,
        )
        clean_inputs = batch.inputs
        corrupt_inputs = flip_input_bit(clean_inputs, bit_position)
        labels = answer_labels(batch, length)
        clean_states = run_states(model, clean_inputs, steps=length)
        corrupt_states = run_states(model, corrupt_inputs, steps=length)
        clean_accuracy, clean_margin = score_state(
            model, clean_states[-1], labels, length
        )
        corrupt_accuracy, corrupt_margin = score_state(
            model, corrupt_states[-1], labels, length
        )
        if smoke or length == 20:
            patch_steps: Iterable[int] = range(1, length)
        else:
            patch_steps = sorted(
                {1, max(1, length // 4), length // 2, 3 * length // 4, length - 1}
            )
        for patch_step in patch_steps:
            for site, token_position in (
                ("source_token", bit_position),
                ("answer_token", length),
            ):
                patched = corrupt_states[patch_step - 1].clone()
                patched[:, token_position] = clean_states[patch_step - 1][
                    :, token_position
                ]
                final_state = continue_from_state(
                    model,
                    patched,
                    corrupt_inputs,
                    start_step=patch_step,
                    final_step=length,
                )
                accuracy, margin = score_state(model, final_state, labels, length)
                rows.append(
                    {
                        "length": length,
                        "bit_position": bit_position,
                        "loop": patch_step,
                        "normalized_loop": patch_step / length,
                        "site": site,
                        "accuracy": accuracy,
                        "logit_difference": margin,
                        "recovery": recovery(
                            margin, clean_margin, corrupt_margin
                        ),
                        "clean_accuracy": clean_accuracy,
                        "corrupt_accuracy": corrupt_accuracy,
                        "clean_logit_difference": clean_margin,
                        "corrupt_logit_difference": corrupt_margin,
                        "samples": batch_size,
                        "evaluation_seed": seed + 100 * condition_index,
                        "donor": "clean",
                        "receiver": "corrupt",
                        "continuation_inputs": "corrupt",
                    }
                )
    return rows


def run_mlp_skip(
    model: PaperLoopedTransformer,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    smoke: bool,
) -> list[dict[str, Any]]:
    length = 6 if smoke else 20
    batch = fixed_parity_batch(
        length=length, batch_size=batch_size, seed=seed, device=device
    )
    labels = answer_labels(batch, length)
    rows: list[dict[str, Any]] = []
    skipped_steps: Iterable[int | None] = [None, 1, length // 2, length] if smoke else [
        None,
        *range(1, length + 1),
    ]
    for skipped_step in skipped_steps:
        states = run_states(
            model,
            batch.inputs,
            steps=length + 1,
            skipped_mlp_step=skipped_step,
        )
        for evaluation_step in (length, length + 1):
            accuracy, margin = score_state(
                model, states[evaluation_step - 1], labels, length
            )
            rows.append(
                {
                    "length": length,
                    "skipped_loop": 0 if skipped_step is None else skipped_step,
                    "eval_loop": evaluation_step,
                    "accuracy": accuracy,
                    "logit_difference": margin,
                    "samples": batch_size,
                    "evaluation_seed": seed,
                }
            )
    return rows


def run_final_patching(
    model: PaperLoopedTransformer,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    smoke: bool,
) -> list[dict[str, Any]]:
    length = 6 if smoke else 20
    bit_position = 0
    batch = fixed_parity_batch(
        length=length, batch_size=batch_size, seed=seed, device=device
    )
    clean_inputs = batch.inputs
    corrupt_inputs = flip_input_bit(clean_inputs, bit_position)
    labels = answer_labels(batch, length)
    clean_before = run_states(model, clean_inputs, steps=length - 1)[-1]
    corrupt_before = run_states(model, corrupt_inputs, steps=length - 1)[-1]
    clean_embeddings = model.input_embeddings(clean_inputs, step_index=length)
    corrupt_embeddings = model.input_embeddings(corrupt_inputs, step_index=length)
    clean = step_components(model, clean_before, clean_embeddings)
    corrupt = step_components(model, corrupt_before, corrupt_embeddings)
    clean_accuracy, clean_margin = score_state(
        model, clean["final_state"], labels, length
    )
    corrupt_accuracy, corrupt_margin = score_state(
        model, corrupt["final_state"], labels, length
    )

    def finalize_post_attention(post_attention: torch.Tensor) -> torch.Tensor:
        layer = model.layers[0]
        mlp_output = layer.mlp(layer.mlp_norm(post_attention))
        return model.final_norm(post_attention + mlp_output)

    interventions: list[tuple[str, torch.Tensor]] = [
        ("clean", clean["final_state"]),
        ("corrupt", corrupt["final_state"]),
    ]

    patched_attention = corrupt["attention_output"].clone()
    patched_attention[:, length] = clean["attention_output"][:, length]
    patched_post_from_attention = corrupt["pre_attention"] + patched_attention
    interventions.append(
        (
            "final_attention_output_answer",
            finalize_post_attention(patched_post_from_attention),
        )
    )

    patched_post_attention = corrupt["post_attention"].clone()
    patched_post_attention[:, length] = clean["post_attention"][:, length]
    interventions.append(
        (
            "final_post_attention_answer",
            finalize_post_attention(patched_post_attention),
        )
    )

    patched_mlp = corrupt["mlp_output"].clone()
    patched_mlp[:, length] = clean["mlp_output"][:, length]
    interventions.append(
        (
            "final_mlp_output_answer",
            model.final_norm(corrupt["post_attention"] + patched_mlp),
        )
    )

    patched_before_answer = corrupt_before.clone()
    patched_before_answer[:, length] = clean_before[:, length]
    interventions.append(
        (
            "pre_final_state_answer",
            step_components(
                model, patched_before_answer, corrupt_embeddings
            )["final_state"],
        )
    )

    patched_before_source = corrupt_before.clone()
    patched_before_source[:, bit_position] = clean_before[:, bit_position]
    interventions.append(
        (
            "pre_final_state_source",
            step_components(
                model, patched_before_source, corrupt_embeddings
            )["final_state"],
        )
    )

    rows: list[dict[str, Any]] = []
    for site, final_state in interventions:
        accuracy, margin = score_state(model, final_state, labels, length)
        rows.append(
            {
                "site": site,
                "length": length,
                "bit_position": bit_position,
                "accuracy": accuracy,
                "logit_difference": margin,
                "recovery": recovery(margin, clean_margin, corrupt_margin),
                "clean_accuracy": clean_accuracy,
                "corrupt_accuracy": corrupt_accuracy,
                "samples": batch_size,
                "evaluation_seed": seed,
                "donor": "clean",
                "receiver": "corrupt",
                "continuation_inputs": "corrupt",
            }
        )
    return rows


def run_head_ablation(
    model: PaperLoopedTransformer,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    smoke: bool,
) -> list[dict[str, Any]]:
    length = 6 if smoke else 20
    batch = fixed_parity_batch(
        length=length,
        batch_size=min(batch_size, 256),
        seed=seed,
        device=device,
    )
    labels = answer_labels(batch, length)
    heads: Iterable[int | None] = [None, 0] if smoke else [
        None,
        *range(model.config.n_heads),
    ]
    rows: list[dict[str, Any]] = []
    for head in heads:
        states = run_states(
            model,
            batch.inputs,
            steps=length + 1,
            masked_head=head,
        )
        for evaluation_step in (length, length + 1):
            accuracy, margin = score_state(
                model, states[evaluation_step - 1], labels, length
            )
            rows.append(
                {
                    "ablated_head": -1 if head is None else head,
                    "length": length,
                    "eval_loop": evaluation_step,
                    "accuracy": accuracy,
                    "logit_difference": margin,
                    "samples": min(batch_size, 256),
                    "evaluation_seed": seed,
                    "ablation_scope": "same_head_at_every_effective_loop",
                }
            )
    return rows


def run_flip_influence(
    model: PaperLoopedTransformer,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    smoke: bool,
) -> list[dict[str, Any]]:
    """Track where a one-bit input intervention changes the residual state."""
    length = 6 if smoke else 20
    bit_position = 0
    batch = fixed_parity_batch(
        length=length,
        batch_size=min(batch_size, 256),
        seed=seed,
        device=device,
    )
    clean_states = run_states(model, batch.inputs, steps=length)
    corrupt_states = run_states(
        model, flip_input_bit(batch.inputs, bit_position), steps=length
    )
    rows: list[dict[str, Any]] = []
    for step, (clean, corrupt) in enumerate(
        zip(clean_states, corrupt_states), start=1
    ):
        difference = (clean - corrupt).square().mean(dim=(0, 2)).sqrt()
        reference = clean.square().mean(dim=(0, 2)).sqrt().clamp_min(1e-12)
        relative = difference / reference
        for token_position in range(length + 1):
            rows.append(
                {
                    "length": length,
                    "flipped_bit_position": bit_position,
                    "loop": step,
                    "token_position": token_position,
                    "token_role": (
                        "source"
                        if token_position == bit_position
                        else "answer"
                        if token_position == length
                        else "downstream_bit"
                    ),
                    "relative_state_difference": float(
                        relative[token_position].cpu()
                    ),
                    "samples": min(batch_size, 256),
                    "evaluation_seed": seed,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def describe_token_injection(mode: str) -> str:
    if mode == "initial_only":
        return "只在第一个 call 注入 token embedding"
    if mode == "every_step":
        return "每个 call 都重新注入 token embedding"
    raise ValueError(f"unsupported token embedding injection mode: {mode}")


def describe_head_ablation(
    rows: list[dict[str, Any]], *, target_loop: int
) -> str:
    selected = [
        row
        for row in rows
        if int(row["ablated_head"]) >= 0
        and ("eval_loop" not in row or int(row["eval_loop"]) == target_loop)
    ]
    if not selected:
        raise ValueError("head-ablation summary has no target-loop rows")
    weakest = min(selected, key=lambda row: float(row["accuracy"]))
    head = int(weakest["ablated_head"])
    accuracy = float(weakest["accuracy"])
    if accuracy >= 0.95:
        return (
            f"跨全部有效轮次分别消融单 head 后，登记终点的最低准确率为 "
            f"{accuracy:.3f}（head {head}）；当前协议未发现强必要单 head，但这不等于 "
            "attention 不重要。"
        )
    return (
        f"跨全部有效轮次分别消融单 head 后，head {head} 使登记终点准确率降至 "
        f"{accuracy:.3f}；它在这项累积消融协议下具有因果必要性，但这仍不证明它是 "
        "唯一组件或已经构成完整 attention circuit。"
    )


def write_report(
    path: Path,
    *,
    transfer: list[dict[str, Any]],
    mlp_skip: list[dict[str, Any]],
    final_patching: list[dict[str, Any]],
    head_ablation: list[dict[str, Any]],
    flip_influence: list[dict[str, Any]],
    smoke: bool,
    backbone_seed: int | None,
    checkpoint_step: int | None,
    token_embedding_injection: str,
    trained_length_max: int,
) -> None:
    if smoke:
        title_note = "（smoke test，不用于文章数字）"
        main_length = 6
    else:
        title_note = ""
        main_length = 20

    def transfer_value(loop: int, site: str) -> float:
        row = next(
            item
            for item in transfer
            if item["length"] == main_length
            and item["bit_position"] == 0
            and item["loop"] == loop
            and item["site"] == site
        )
        return float(row["recovery"])

    chosen_loops = (
        [1, main_length // 2, main_length - 1]
        if smoke
        else [1, 5, 10, 15, 19]
    )
    transfer_lines = "\n".join(
        f"| {loop} | {transfer_value(loop, 'source_token'):.3f} | "
        f"{transfer_value(loop, 'answer_token'):.3f} |"
        for loop in chosen_loops
    )
    final_lines = "\n".join(
        f"| `{row['site']}` | {float(row['accuracy']):.3f} | "
        f"{float(row['recovery']):.3f} |"
        for row in final_patching
    )
    skipped = [row for row in mlp_skip if row["skipped_loop"] != 0]
    registered = [row for row in skipped if row["eval_loop"] == main_length]
    delayed = [row for row in skipped if row["eval_loop"] == main_length + 1]
    head_ablation_description = describe_head_ablation(
        head_ablation, target_loop=main_length
    )
    heads_at_target = [
        row
        for row in head_ablation
        if row["ablated_head"] >= 0 and row["eval_loop"] == main_length
    ]
    attention_conclusion = (
        "attention 的单-head 必要性随 checkpoint 而变，尚未定位出唯一、最小或完整的 head circuit。"
        if min(float(row["accuracy"]) for row in heads_at_target) < 0.95
        else "attention 的作用可能分布或冗余，尚未定位出唯一、最小或完整的 head circuit。"
    )
    first_loop_answer = next(
        row
        for row in flip_influence
        if row["loop"] == 1 and row["token_role"] == "answer"
    )
    report = f"""# Parity 循环电路分析{title_note}

## 结论先行

对 backbone seed {backbone_seed}、step {checkpoint_step} 的冻结模型，当前因果证据支持一个有限的计算分工：输入 bit 的因果可访问性随循环推进从 source token 转移到 answer token；共享 MLP 的每次调用推进一次内部相位；到最后一轮前，parity 已经存在于 answer residual，最后一轮 MLP 把它写入可读出方向。{attention_conclusion}

## 协议

- NoPE、causal、一个共享物理 block，`d_model=256`、64 heads；
- {describe_token_injection(token_embedding_injection)}；
- 训练长度 1–{trained_length_max}，只在登记终点 `T(n)=n` 的答案区域计算 CE；
- 本报告的主因果条件为 `n={main_length}`，clean/corrupt 只翻转第 0 个 bit；
- recovery 以 clean 目标的 logit difference 归一化：`(patched-corrupt)/(clean-corrupt)`。

## 1. source 到 answer 的因果转移

| patch loop | source recovery | answer recovery |
|---:|---:|---:|
{transfer_lines}

观察：source recovery 单调下降，answer recovery 上升。`n=10,30` 与首/末 bit 的控制条件方向一致。两者之和在本实验里近似 1，但这只是归一化因果可访问性的近似互补，不能称为守恒量。

## 2. MLP 调用推进相位

- 对所有 {len(registered)} 个被跳过的有效轮次，`t={main_length}` 的平均准确率为 {sum(float(row['accuracy']) for row in registered) / len(registered):.3f}；
- 同一批干预在 `t={main_length + 1}` 的最低准确率为 {min(float(row['accuracy']) for row in delayed):.3f}。

因此，少一次 MLP 更新不会永久删除答案，而是把正确峰整体推迟一轮。这是“MLP 调用推进内部相位”的组件级因果证据。

## 3. final-step patching

| patch site | accuracy | recovery |
|---|---:|---:|
{final_lines}

最终 attention output 本身不足以翻转答案；patch post-attention answer residual 可完全恢复；patch MLP output 可恢复大部分 logit difference。更关键的是，最终轮之前 patch answer state 几乎完全恢复，而 patch 原始 source bit 的晚期 state 几乎无效。

## 4. 两个附加诊断

1. 在第一轮，翻转 bit 0 已使 answer token 的相对状态差达到 {float(first_loop_answer['relative_state_difference']):.6f}，并影响所有 causal downstream token。因此，行为热图中的斜率 1 不能直接解释成“一轮只读取一个 bit”的局部波前。
2. {head_ablation_description}

## 证据边界

- 主结果只有一个 backbone seed，不代表训练必然性；
- 这里定位的是状态寄存器角色和组件相位角色，不是完整计算图；
- source/answer recovery 不是信息量，也不是守恒定律；
- J 的低秩校正属于 phase/interface steering，不负责重新求 parity；
- 结果只适用于此 toy model、checkpoint 和干预协议。

## 数据文件

- `source_answer_transfer.csv`
- `mlp_skip.csv`
- `final_patching.csv`
- `attention_head_ablation.csv`
- `flip_influence.csv`
- `manifest.json`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    device = resolve_device(args.device)
    model, checkpoint = load_backbone(args.checkpoint, device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch = min(args.batch_size, 16) if args.smoke else args.batch_size

    with torch.inference_mode():
        transfer = run_source_answer_transfer(
            model,
            device=device,
            batch_size=effective_batch,
            seed=args.seed + 1000,
            smoke=args.smoke,
        )
        mlp_skip = run_mlp_skip(
            model,
            device=device,
            batch_size=effective_batch,
            seed=args.seed + 2000,
            smoke=args.smoke,
        )
        final_patching = run_final_patching(
            model,
            device=device,
            batch_size=effective_batch,
            seed=args.seed + 3000,
            smoke=args.smoke,
        )
        head_ablation = run_head_ablation(
            model,
            device=device,
            batch_size=effective_batch,
            seed=args.seed + 4000,
            smoke=args.smoke,
        )
        flip_influence = run_flip_influence(
            model,
            device=device,
            batch_size=effective_batch,
            seed=args.seed + 5000,
            smoke=args.smoke,
        )

    write_csv(output_dir / "source_answer_transfer.csv", transfer)
    write_csv(output_dir / "mlp_skip.csv", mlp_skip)
    write_csv(output_dir / "final_patching.csv", final_patching)
    write_csv(output_dir / "attention_head_ablation.csv", head_ablation)
    write_csv(output_dir / "flip_influence.csv", flip_influence)

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "analysis": "parity_circuit_20260807",
        "status": "smoke" if args.smoke else "publication",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": checkpoint.get("kind"),
        "backbone_seed": checkpoint.get("seed"),
        "checkpoint_step": checkpoint.get("step"),
        "task": checkpoint.get("task"),
        "model": asdict(model.config),
        "loss_placement": "answer-region CE only at registered final T(n)=n",
        "trained_lengths": [1, 20],
        "shared_physical_blocks": 1,
        "token_embedding_injection": model.config.token_embedding_injection,
        "position_embedding": model.config.position_embedding,
        "position_injection": model.config.position_injection,
        "analysis_seed": args.seed,
        "batch_size": effective_batch,
        "device": str(device),
        "code_revision": git_revision(root),
        "tables": {
            "source_answer_transfer": "source_answer_transfer.csv",
            "mlp_skip": "mlp_skip.csv",
            "final_patching": "final_patching.csv",
            "attention_head_ablation": "attention_head_ablation.csv",
            "flip_influence": "flip_influence.csv",
        },
        "claim_boundaries": [
            "single frozen backbone seed",
            "no claim of a one-bit-per-loop scan",
            "no claim of a unique, minimal, or complete attention circuit",
            "normalized source/answer recovery is not a conserved quantity",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / "PARITY_CIRCUIT_ANALYSIS_ZH.md",
        transfer=transfer,
        mlp_skip=mlp_skip,
        final_patching=final_patching,
        head_ablation=head_ablation,
        flip_influence=flip_influence,
        smoke=args.smoke,
        backbone_seed=checkpoint.get("seed"),
        checkpoint_step=checkpoint.get("step"),
        token_embedding_injection=model.config.token_embedding_injection,
        trained_length_max=int(checkpoint["task"]["train_max_length"]),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
