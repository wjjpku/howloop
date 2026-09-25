# Ouro 反义词实验回查（2026-09-24）

## 固定 4 loops 的行为对比

远程来源：
- `/data/wujiaju/ouro26_antonym_full_20260915/run/shared_j_lora128_k18`
- `/data/wujiaju/ouro26_stepwise_pair_control_20260915/shared_j_lora128_k18`

重新逐行汇总 `eval_outputs.jsonl`，template=0，step=0/500。四组 (final/stepwise × step0/500) 的 512 个 `(k,sequence,expected)` 完全相同；每个 k 64 例。step0 为精确恒等初始化的 J。以下均为 deleted-pair 输出准确率。

|请求删除次序 k|Final-only raw|Final-only + J|Stepwise raw|Stepwise + J|
|---|---:|---:|---:|---:|
|5|0/64|64/64 (100%)|0/64|6/64 (9.375%)|
|6|0/64|62/64 (96.875%)|0/64|6/64 (9.375%)|
|7|0/64|62/64 (96.875%)|0/64|6/64 (9.375%)|
|8|0/64|63/64 (98.4375%)|0/64|2/64 (3.125%)|
|合计|0/256|251/256 (98.047%)|0/256|20/256 (7.813%)|

J 两侧配置一致：共享 rank128 affine，冻结 backbone，4 次应用（每个 loop RMSNorm 后，包括最后一次），500 optimizer steps，lr=1e-4，训练 k1–8，0.8 task CE + 0.2 general CE。同一测试集；不是多个 backbone seed 的统计。

重要区别：final-only backbone 的训练 prompt 请求 k1–4；stepwise backbone 的训练 prompt 固定请求4，逐 loop t=1..4 监督。因此这是已有两个训练方案的可控性对比，不能称为仅改变监督位置的严格单变量消融。J 训练包含 k5–8，属于固定计算预算下的监督适配，不是 J 对未见任务深度的外推。失败不能证明机制不存在。

## 提前读出

本地来源：`/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/sec08_supervision/p2b_ouro_antonym_delta_deps/ouro26_antonym_four_j_20260915/baseline_readout12` 与 `baseline_readout34`。

无 J 的 final-only backbone，在 k1–4 共256例上，各 loop 的 parsed-pair 正确数：81、112、255、256。支持第3 loop 已能读出几乎全部目标，不支持第1 loop 已完成全部任务。部分输出未产生 EOS，因此应称 pair readout，不能混同完整输出序列正确率。此面板与前面的 J 对比不是同一个评估配置，不能拼接 raw 数字。

## L8.H5 机制干预确实做过

远程来源：`/data/wujiaju/ouro_l8h5_localization_20260916`、`/data/wujiaju/ouro_l8h5_combinations_20260916`。本目录保存 confirmation 原始 JSONL。

组合定位实验的独立 confirmation n=48，L8.H5 输出在 o_proj 前置零：
- native k4：47/48 → 32/48。
- J k4：47/48 → 35/48。
- J k8：48/48 → 25/48。

以上三组均执行4 loops；j8 指请求 k8，不是8 loops。多个候选干预有相同聚合结果，具体 loop/phase 条件见 JSONL，不将不同条件当作重复实验。另一个局部定位 confirmation 面板效果较小：native 48→43，J k4 45→41，J k8 48→42，均分母48。

机制实验与行为对比使用相同 final-only backbone SHA256 f29324252ee3da7eb73de700b1af1a0679d2a1d93c559033ca0906ca080bd3ae；但机制实验用的是 dense J，来自 `/data/wujiaju/ouro26_antonym_four_j_20260915/fit/checkpoint.pt`，已核验 SHA256 为 203712ced24e6dd3be01bc275a2fc5983a3372db9bde380d08b7db4069972e0f。不能写成同一 rank128 J 的机制分析。

该证据支持同一个已有 attention head 对 native 和受控任务都具有因果贡献；尚不能单独证明该 head 在每 loop 内执行了多次删除。native_signals_v2 的报告也明确未建立 semantic substitution 或 native rescue。

## 论文呈现建议

将 Fig7 的 k1–4 面板换为固定4 loops、k5–8的 paired grouped bars；正文以行为差异为主，配提前读出和共享 head 干预作机制支持。明确 backbone 训练 prompt 差异、J 已见 k5–8、两个 J 配置不同。此次仅回查整理，未修改论文。
