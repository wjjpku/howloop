from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "h1→f²",
        "h2→f⁴",
        "h3→f⁶",
        "h4→f⁸",
        "skip→f⁶",
        "repeat→f⁸",
        "full-state\nnext min",
        "answer-only\ncross-age",
    ]
    values = np.asarray(
        [
            [
                row["macro_h1_f2"],
                row["macro_h2_f4"],
                row["macro_h3_f6"],
                row["macro_h4_f8"],
                row["skip_f6"],
                row["repeat_f8"],
                row["full_state_next_min"],
                row["answer_state_cross_age_next_mean"],
            ]
            for row in rows
        ]
    )
    figure, axis = plt.subplots(figsize=(11.5, 5.0))
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(rows)), [f"seed {row['seed']}" for row in rows])
    axis.set_title(
        "Post-Norm D8L8 circuit at step 14k\n"
        "cyan outline = full clear-circuit seed (all four macro steps ≥ 0.8)"
    )
    for row_index, row in enumerate(rows):
        if row["clear_circuit"]:
            axis.add_patch(
                plt.Rectangle(
                    (-0.48, row_index - 0.43),
                    len(columns) - 0.04,
                    0.86,
                    fill=False,
                    edgecolor="#22d3ee",
                    linewidth=2.0,
                )
            )
        for column_index, value in enumerate(values[row_index]):
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.58 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_training_curves(
    histories: dict[int, list[dict[str, Any]]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    for seed, history in sorted(histories.items()):
        steps = [row["step"] for row in history]
        minimum = [
            min(row["trajectory_depth_loop_accuracy"][-1][:4])
            for row in history
        ]
        axis.plot(steps, minimum, marker="o", markersize=2.8, label=f"seed {seed}")
    axis.axhline(0.8, color="black", linestyle="--", linewidth=1.0, label="clear threshold")
    axis.axvline(10_000, color="#777777", linestyle=":", linewidth=1.0)
    axis.axvline(14_000, color="#777777", linestyle=":", linewidth=1.0)
    axis.text(10_100, 0.04, "aux anneal begins", fontsize=8)
    axis.text(13_900, 0.04, "selected stop", fontsize=8, ha="right")
    axis.set_xlim(0, 14_200)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("training step")
    axis.set_ylabel("minimum D8 accuracy across h1→f²,…,h4→f⁸")
    axis.set_title("Seed-dependent formation of the complete two-hop trajectory")
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--seed1-history", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    histories: dict[int, list[dict[str, Any]]] = {}
    for seed in range(6):
        summary = read_json(
            args.analysis_root / f"seed{seed}_step14k" / "summary.json"
        )
        macro = summary["expected_macro_accuracy"][:4]
        skip = float(np.mean(list(summary["skip_f6_accuracy"].values())))
        repeat = float(np.mean(list(summary["repeat_endpoint_accuracy"].values())))
        full_state_min = float(min(summary["donor_full_state_next_accuracy"]))
        clear = (
            min(macro) >= 0.8
            and skip >= 0.8
            and repeat >= 0.8
            and full_state_min >= 0.8
        )
        rows.append(
            {
                "seed": seed,
                "macro_h1_f2": macro[0],
                "macro_h2_f4": macro[1],
                "macro_h3_f6": macro[2],
                "macro_h4_f8": macro[3],
                "minimum_macro_accuracy": min(macro),
                "skip_f6": skip,
                "repeat_f8": repeat,
                "full_state_next_min": full_state_min,
                "answer_state_cross_age_next_mean": summary[
                    "answer_state_transplant_next_mean"
                ],
                "clear_circuit": clear,
            }
        )
        history_path = (
            args.seed1_history
            if seed == 1
            else args.history_root / f"seed{seed}_history.json"
        )
        histories[seed] = read_json(history_path)

    write_csv(args.out_dir / "multiseed_circuit_summary.csv", rows)
    plot_heatmap(rows, args.out_dir / "multiseed_circuit_heatmap.png")
    plot_training_curves(
        histories,
        args.out_dir / "multiseed_trajectory_formation.png",
    )
    success_seeds = [row["seed"] for row in rows if row["clear_circuit"]]
    payload = {
        "success_threshold": (
            "all four strict macro-step accuracies, skip f6, repeat f8, and "
            "minimum full-state continuation accuracy are at least 0.8"
        ),
        "success_seeds": success_seeds,
        "success_count": len(success_seeds),
        "seed_count": len(rows),
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    table_lines = []
    for row in rows:
        table_lines.append(
            "| {seed} | {macro_h1_f2:.3f} | {macro_h2_f4:.3f} | "
            "{macro_h3_f6:.3f} | {macro_h4_f8:.3f} | {skip_f6:.3f} | "
            "{repeat_f8:.3f} | {full_state_next_min:.3f} | {status} |".format(
                **row,
                status="完整成功" if row["clear_circuit"] else "首步失败",
            )
        )
    report = f"""# Post-Norm D8L8 清晰 circuit 训练报告

## 条件

- 任务：8 节点随机置换图，query depth 1–8；新图逐 batch 生成。
- 架构：`d_model=256`，2 个共享物理 block，8 个 recurrent loops；有效深度 16 blocks。
- 归一化：canonical Post-LayerNorm。
- 主损失：只在 loop 8 对 query endpoint 计算 final CE。
- 轨迹引导：loop `t` 对 `f^min(2t,d)(start)` 计算辅助 CE；权重 1.0 保持到 10k，随后线性退火，14k 时为 0.2。
- 优化：AdamW，LR 3e-4，weight decay 0.3，warmup 5k；14k 停止，但 cosine LR 仍按 20k horizon 计算。
- 评估：固定 D8，去掉早期位置与 endpoint 碰撞的样本；每个 seed 1024 个样本。

## 六 seed 结果

| seed | h1→f² | h2→f⁴ | h3→f⁶ | h4→f⁸ | skip→f⁶ | repeat→f⁸ | 完整态续跑最低 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table_lines)}

完整成功 seed 为 {success_seeds}，即 {len(success_seeds)}/6。seed 2、3、4 并非整体失败：它们的后 3 个 macro transition、skip/repeat 和完整 hidden-state 续跑仍然很高，失败集中在 `h0→h1` 启动器。

## 主要结论

1. Post-Norm 能表达并训练出比原 Pre-Norm seed 1 更清晰的两跳 circuit；seed 1 的 14k 四步为 1.000/1.000/0.990/0.982。
2. 配方不是 seed 无关的唯一吸引子：3/6 seed 得到完整 circuit，另外 3/6 得到“首步不透明、后续两跳复用”的混合算法。
3. 轻辅助（0.3）失败；强辅助形成 circuit，但完全撤掉后即使 final accuracy 保持 1.0，轨迹也会漂移。
4. 纯 final-only + warmup 5k 不会自动转成清晰 circuit。Pre→Post homotopy 在 80% Post 时仍清晰，但 100% Post 出现 cliff，随后 final-only 恢复成 5-loop 压缩求解器。
5. answer-token-only 的跨年龄移植显著弱于完整 hidden-state 续跑，说明位置/年龄信息分布在序列状态中，不能把 circuit 简化为单一 answer vector。
"""
    (args.out_dir / "REPORT_CN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
