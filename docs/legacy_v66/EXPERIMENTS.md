# v66 实验清单与协议身份

以 `paper/main.tex` 为范围；这份清单不把已移出论文的 Qwen、旧 N8 机制结果或其他任务当作当前论文证据。

## E1：原生轨迹与 E2：目标控制

- 图任务 N=10，D=8；共享块含两层，hidden=256，heads=4，MLP=1024，final-only CE，请求深度均匀 1–8。
- D8L6 种子按 A–E = 6/3/4/5/7。D8L8 轨迹包含 0–11；后增加的 2/3/4/5/7/9/10/11 保存在 `trajectory_seed_extension_20260924`，不能误认成原四种子目录缺失。
- Backbone 20,000 updates，batch512，AdamW lr3e-4、wd0.3、warmup500、clip1、BF16。以选择集最佳 D8 精度选择最早 checkpoint。主模型 A 是 update16,000；原生轨迹展示 loops0–16，512 cycles×10 starts。
- 数据锁：selection128、discovery32、confirmation512、rings512、smoke4，以及单次输出交换产生的 donor 排除集。原始 `.pt` 锁原样保存；不要重新序列化后拿新文件哈希冒充旧锁。
- 两个目标均作用于同一 h6：f9/f10；J=D+AB+b，rank48、all tokens、25,088 参数；每目标 seeds1/2。8,000 updates，batch128，lr1e-4，wd0，clip1；每400步验证，最早最佳。
- 图3a 为 4,110 个 current/one/two 标签互异的样本，按两个 J fits 等权平均。pre-F、post-F、按图 shuffle 分开保存。`figure_work/source` 是本图实际输入。

## E3：PCA

`experiments/pca/extract.py` 提取同一个 A backbone 与四个 J 的状态。512 新保留图×10 starts；每个状态平均35个 token，未做特征标准化。seed20260924 按图划分256/256；joint PCA 使用 fit 图上两个目标条件，画 test 图的 fit1。`data/plot/pca_means.npz` 是原 `states.npz` 中实际使用数组的无损子集；删去未参与图3b的 answer-token、post-F 数组，未修改数值。

## E4：图机制、筛选和新确认

`experiments/graph_mechanism` 保存 A/C 新确认逐样本记录；`screening/analysis` 保存先前五模型筛选统计；更早评估在 `experiments/n10` 与 `graph_mechanism_discovery`。

- A 的固定层2 H2、C 的 H3 在早期 discovery 上选择；对照为下一 head modulo4。新确认前固定 A/C 的选择。
- seed2026092404，512 raw + 512 corrupted graphs 从排除过训练与早期试验的 reserve 采样；每图10个 current，共5,120 candidate pairs。
- A semantic eligibility 分母3,966/3,963；恢复分母4,413/4,417；无成功过滤的 pattern exchange 分母4,044。
- 路由/输出与恢复定位单个 head；同输入 steering exchange 替换层2全部四个 patterns。不是同一干预范围。
- 5,000 次 paired graph cluster bootstrap；控制器 fits 不是独立 backbone replication。`scripts/audit_results.py` 重新验证 events hash 并从事件重算所有统计。

## E5：Ouro letter-walk

Ouro-2.6B，固定四 loops；backbone update200，dense affine J update500，J 在 loops2–4 前作用于所有 token。所有文件哈希见 checkpoint manifest。16-head 集固定在 loop4：L41.H2/H10/H12/H15，L43.H1/H6/H11/H13/H15，L47.H0/H2/H3/H4/H7/H13/H15（零起始）。

- localization discovery8、pattern-transfer confirmation64；10-head 主候选和16-head比较均保留，包括10-head未达预定阈值。
- semantic 新64 pairs，seed2026092901，排除544个此前图；保留全部样本，原答案、corrupted答案、rerouted-raw答案互异。主 pattern/output 分别46/64、61/64。
- restoration64 pairs，seed2026092701，沿用已有独立于该head选择的 query-pair 集；63个 clean-correct 被破坏，selected恢复62/63、neighbor1/63、all63/63。
- first-answer-token 与 full-name 指标有明确差别，16-head早期transfer未记录full-name；后续semantic/restore记录了。图中 Ouro 区间使用 Wilson；原分析也保存 paired bootstrap。

## E6：Parity

input-once、causal NoPE、1 physical layer、d256、64 heads、MLP1024，三个 backbone seeds0/1/2，训练 n1–20，在 t=n answer-region CE。100,001 updates、batch64、FP32。图6主seed2保存步99,000。

uniform extension20to40 controller：rank48 diagonal+low-rank，anchor1，在calls2..n前插J；同一个固定J用于所有长度。脚本保存真实学习率、warmup1024、cosine及初始化设置。

- n1–500 每长度64个输入；相位源是完整 diagonal-band metrics，不是从PNG读点。
- 相位用周期4 Fourier carrier、unwrap、41–490拟合；5,000 moving-block residual bootstrap，block20，seed2026092318。
- n500–1000 每5个整数采样，101长度×128 paired inputs；seed2026092401+1009n。与旧64例曲线分开。CUDA graph eager equivalence 历史检查和原始逐例布尔数组保留。

## E7：N8 graph continuation

**不是主图机制的 N10 cohort。** N=8/D8L8，backbones seeds100–111，最后第8 loop final-only fixed-D8 supervision；final.pt。各2个rank48 J，on-policy calls1–16，J从h0开始在每次F之前插入。

J 的实际 RNG seed 为 `2026095000+10*backbone_seed+replica`，validation seed为`2026097000+10*backbone_seed+replica`；目录里的seed1/seed2只是fit标签。原命令保存在 vendor 的 G4 launchers。

主长程图8192/40320个均匀无放回 permutation graphs，seed2026092402，8 starts，calls1–40；没有历史train-disjoint保证。先按graph/start，再按J/backbone平均；区间重抽12个backbones 10,000次。训练排除的旧512图及其中78个single cycles单独保存，不能并入主图。

## E8：Ouro反义词监督比较

两个各训练500步的backbone：final-only请求k1–4且loop4监督；stepwise请求4且在loops1–4分别监督对应删除。prompt也不同，因此不是只改变loss位置的单变量消融。

每backbone一个rank128 residual affine共享J，526,336参数；每loop RMSNorm后（包括loop4）插入。500 AdamW updates、lr1e-4、warmup50、每update16例（k1–8各2例）、0.8 answer CE+0.2 general CE。固定四loops，template0，k5–8每64个相同序列；final+J为64/62/62/63，stepwise+J为6/6/6/2。J见过k5–8。按parsed pair评分，不能用EOS作为隐含成功条件。

## E9：KG控制器容量

128 entities×16 relations，world seed20260814；同一NoPE input-once backbone，两blocks、d256、8heads。F(JF)^(m−1)，moving answer_index、truncated unroll；curriculum 4/8/12/16/20/24/28/32，每阶段至多5000updates，batch128，lr1e-4，0.99早停。每cell128新组合，所有画出的长度都在已完成curriculum中。

图7d读取最后m32 checkpoint评估行的8/16/24/32列。四family各一次，原控制器RNG未保存，历史结果也未保留各stage权重。现有脚本支持显式`--seed`及`--save-checkpoints`，复现入口会记录新seed并保存权重；这是新的重复试验，不冒充原一次run。
