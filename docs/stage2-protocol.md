# PBSG Stage 2 · 协议、审计与偏差记录

生成时间 2026-11-24。对应计划书 `PBSG_Experimental_Plan.pdf` 的 §5–§8。

## 1. 数据

| 数据集 | 来源 | $N$ | $T$ | 通道 | 道路边（对称化后） |
|---|---|---|---|---|---|
| PEMS08 | HF `Davidham3/pems-stsgcn` | 170 | 17,856 | flow | 548 |
| PEMS04 | HF `Davidham3/pems-stsgcn` | 307 | 16,992 | flow | 680 |
| PEMS07 | HF `jimmygao3218/PEMS07`（GWN 格式，已对齐） | 883 | 28,224 | flow | 1,732 |

PEMS03 未做：parquet 发布用的是真实站点 ID，而 series 无 ID 列，官方 `*_sensors.parquet` 在仓库里取不到（404/15 B），缺少 ID→数组下标映射。按 §5.6 的规定不猜映射。

PEMS07 另外一条路的动机相同：GWN 格式里 `adj_PEMS07_distance.pkl` 的行列顺序就是 series 的节点轴，天然对齐，不需要站点 ID。其 `nnz=1732=866×2`，正是 STSGCN 边表对称化的结果，可作交叉验证。

## 2. 预处理协议

- 按**原始时间轴**先切 60/20/20，再在各 split 内部生成 12→12 滑窗；窗口不跨边界。
- scaler 只用 train split 的有效观测拟合，per-node；指标在反归一化后的 flow 单位上算。
- **零值不当缺失**：PEMS08 0.35% / PEMS04 1.59% / PEMS07 0.45%，节点级最大 10.8% / 38.1% / 93.2%，mask 全 1 并在 `meta.json` 记录。PEMS07 有 1 个节点 93% 为零（死传感器），单独报告。
- 只取 `flow`；PEMS04/08 的 occupancy/speed 丢弃并记录。

窗口数：

| | train | val | test |
|---|---|---|---|
| PEMS08 | 10,691 | 3,548 | 3,548 |
| PEMS04 | 10,172 | 3,375 | 3,376 |
| PEMS07 | 16,911 | 5,622 | 5,622 |

### 道路图审计

- STSGCN 发布的边表**是无向的**：PEMS08 出度 0 的 7 个节点与入度 0 的 7 个节点互不重叠，`either_degree_zero=0`。已统一对称化并记录。
- cost 单位不一致：PEMS04/08 中位数 ~330（米级），PEMS03/07 是 km 级（0.92 / 0.9）。
- 自环 0，重复边 PEMS08 18 条（已去重）。

## 3. 候选集

- $C_i$：以 CSR 稀疏邻接做有界 BFS，1 跳全保留（§3.3：不得静默删除），再按 2/3/4 跳补到 $K$。排序 = (跳数, 节点 ID)，确定性。
- $F_i$：从 $[0,N)\setminus(\{i\}\cup C_i)$ 做**有界拒绝采样**，每节点 RNG 种子 `[2026, i]`，上限 $M$，不做全节点扫描。
- 离线生成并缓存为 `cand_K{K}_M{M}.npz`；在线只做打分 + Top-m + 归一化。

$|C_i|$ 统计（$K{=}32$）：

| 数据集 | mean | max（=$K_{\text{eff}}$） | 说明 |
|---|---|---|---|
| PEMS08 | 24.18 | 32 | $K$ 有效 |
| PEMS04 | 12.41 | 32 | $K{=}64$ 退化（mean 仅 12.50） |
| PEMS07 | 8.01 | **20** | 4 跳 BFS 饱和；$K{=}32$ 与 $K{=}64$ 生成同一张图，$K$ 只剩 $\{8,16\}$ 两个有效点 |

道路边被 $C_i$ 覆盖率恒为 1.0000（因为 1 跳全保留）。

## 4. 方法与骨干

- 七种：A Road-Fixed、B Dense-Adaptive、C Global-TopK、D PBSG-$m{=}0$、E PBSG-$m{>}0$、F Fixed-Candidate、G Random-Budget。
- 打分统一 $s_{ij}=\mathrm{ReLU}(e_i^\top e_j)$，$d_{\text{emb}}{=}16$，温度 1.0，行 softmax，自环分数 0 统一加入。
- 骨干：GWN-style，channels 32，dilations $(1,2,4,4)$ → 感受野 12，正好覆盖 $T_{\text{in}}{=}12$，$T_{\text{after}}{=}1$。约 31 k 参数。
- 训练：Adam，lr $10^{-3}$，weight decay 0，batch 32，max 30 epoch，patience 5，seed 42/43/44，loss = masked L1（标准化空间）。
- **C 的 $k$ 已按 §6.2 对齐**：$k=\mathrm{round}(\overline{|C_i|})+m$，不是 $K+m$。PEMS08 上 $k{=}28$（若用 $K+m$ 会是 36，出度会多 29%）。

## 5. 测量

- 每个 forward 单独计时构图（前后 `cuda.synchronize`），记 `graph_sec_per_epoch`。
- 推理延迟：batch=1，warm-up 10，测 100 次，报 p50/p95。
- 显存：`reset_peak_memory_stats` + `max_memory_allocated`，分别记构图峰值、推理峰值、训练峰值。
- 支撑集与分数无关的方法（F、G、PBSG-$m{=}0$）缓存稀疏 **pattern**，每步只重算 values；PBSG-$m{>}0$ 的硬 Top-m 使列随分数变化，必须每步重建索引——这个差别如实计入计时。

## 6. 与计划书的偏差（重要）

1. **数据集顺序**：按用户要求「先跑时间短的」，Stage-2 消融先做 PEMS08，再做 PEMS04 / PEMS07。计划书原定 PeMSD07 为主。
2. **全网格只在 PEMS08 上跑**（16 配置 × 3 seed = 48 run），PEMS04/PEMS07 跑核心 7 配置 × 3 seed。
3. **$K$ 消融在 PEMS04/PEMS07 上退化**（见 §3），报告时须用实际 $K_{\text{eff}}$，不能沿用 $NK$ 上界。
4. **候选排序用 (跳数, 节点 ID)**，未使用道路距离。§3.3 允许，但「道路距离（如可用）」这一项没做——PEMS07 已经有 km 距离，可作为后续 refinement（会要求全部重跑）。
5. **H1 在本轮 $N\le 883$ 上不成立**：dense 在小图上构图更快（见结果表）。稀疏性体现在**边数与邻接存储**，不在墙钟。

## 7. 后续

- 用距离做候选排序（改 `graphop.build_prior_candidates` 的排序键）。
- 构图常数开销继续压：预排序 index、批量复用 pattern。
- Stage 3（LargeST）按用户口径暂不做。
