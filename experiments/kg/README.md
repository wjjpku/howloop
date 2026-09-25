# §9.2 Controller 容量×课程阶段×测试长度热力图

## 实验设计

四种 controller 在同一 backbone、同一课程协议下从 m=4 训到 m=32（步长 4），每阶段结束时在全部 8 个测试长度上评测，产出 8×8 准确率热力图。

| Controller | 架构 | 参数量 | 关键区别 |
|---|---|---|---|
| lora_r48 | h + (h@A)@B + bias | 25K | 低秩线性 |
| dense | h + h@W + bias | 66K | 满秩线性 |
| mlp_256 | h + gelu(h@W₁+b₁)@W₂ + b₂ | 132K | 非线性（无 LayerNorm） |
| attention | LayerNorm + MHA + residual | ~200K | 跨 token 路由 |

## 训练协议

- Backbone: KG 关系复合（N=128实体，R=16关系），NoPE causal input-once，d256/8h/1024mlp/2blk，SHA256 `4d6bb56b...`
- 执行架构: F (JF)^{m-1}（第一次 F 前无 J，用已有 `run_alternating`）
- 损失: truncated_unroll（逐调用监督，移动 answer_index=step+1，从 step 2 起）
- 课程: stages [4, 8, 12, 16, 20, 24, 28, 32]，max 5000 步/阶段，早停 ≥99%
- 回放: 50% 当前长度，50% 随机更短长度（4 到当前阶段）
- lr=1e-4, batch=128, AdamW, clip=1.0

## 关键实现细节（踩坑记录）

1. **J 位置**: 第一次 F 调用前不施加 J（`run_alternating` 的设计）。在所有边界施加 J（含第一次）会导致训练完全失败。
2. **模型加载**: 必须用原始 max_length=6 + strict=True 加载，然后原位改 kg_config.max_length。用扩展 max_length + strict=False 加载会微妙地破坏学习。
3. **LayerNorm 杀死 MLP**: 前置 LayerNorm 的 MLP（已有 ResidualMLPJ）完全无法学习此任务（m=4 均失败）。去掉 LayerNorm 后 MLP 成为最强 controller。原因：LayerNorm 剥夺了 MLP 对状态幅度的感知。
4. **端点 CE vs truncated_unroll**: 端点 CE（只看最后一步）在 m=4 可收敛但 m≥8 灾难遗忘。truncated_unroll（移动位置的逐调用监督）配合随机长度回放是必需的。

## 最终结果（5k 步/阶段，m=32 训完后各测试长度准确率）

| 测试长度 | lora_r48 (25K) | dense (66K) | mlp_256 (132K) | attention (~200K) |
|---|---|---|---|---|
| m=4 | 100% | 100% | 100% | 100% |
| m=8 | 79% | 98% | 98% | 99% |
| m=12 | 48% | 92% | 98% | 99% |
| m=16 | 2% | 79% | 93% | 95% |
| m=20 | 0% | 59% | 90% | 94% |
| m=24 | 1% | 33% | 75% | 84% |
| m=28 | 0% | 10% | 58% | 56% |
| m=32 | 1% | 3% | 36% | 24% |

## 主要发现

1. **affine 的长度不泛化确实是表达力瓶颈**: dense (66K) 只能推到 m=16–20，同一位置放 mlp_256 或 attention 就能推到 m=24–28。
2. **非线性优势随长度增大**: m=12 处 dense 和 mlp 差 6pp，m=24 处拉大到 42pp。
3. **rank 是真实瓶颈**: lora_r48 只能推到 m=12。
4. **attention 与 mlp 各有所长**: attention 在 m=16–24 中间段最强（95/94/84），m=28+ 被 mlp 反超（58 vs 56, 36 vs 24）。
5. **所有 controller 在 m=28+ 都崩**: 即使训练就在那个长度（5000 步/阶段），这是训练预算限制而非表达力限制。

## 文件清单

```
├── README.md               ← 本文件
├── train_5k.py             ← 训练脚本（最终正确版，含 run_alternating + truncated_unroll）
├── results_5k/             ← 5k步/阶段 最终结果
│   ├── dense/results.json
│   ├── lora_r48/results.json
│   ├── mlp_256/results.json
│   └── attention/results.json
└── results_3k/             ← 3k步/阶段 对照（相同协议，仅步数不同）
    ├── ...
```

## Backbone checkpoint（不在本仓库）

路径: `A100-80G-34200:/data/wujiaju/kg-fj-nope-20260815-v1/nope_only/backbone/best.pt`
SHA256: `4d6bb56b501ddeb0b1056945fd41f6a5f7ce1b79651732eaef077b007b2641c5`

## 共享代码依赖

训练脚本 `import` 以下模块（来自 `kg-fj-affine-resffn-m64-code-5f49178`，已拷贝至 `_shared/` 或需从 A100 取）：
- `experiments.kg_fj_length.data`（KGLengthConfig, PermutationWorld, sample_batch）
- `experiments.kg_fj_length.model`（LoopedCompositionTransformer, ModelConfig）
- `experiments.kg_fj_length.controller`（run_alternating, AffineJ, CausalAttentionJ, ControllerConfig）
