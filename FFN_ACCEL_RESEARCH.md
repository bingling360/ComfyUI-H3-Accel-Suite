# 前馈网络（FFN）加速技术深度调研

配套文档：`REPORT.md`（MiniMax H3 非注意力加速实测）
调研日期：2026-10-25
调研目标：在 `REPORT.md` 得出「FFN 已到头」的结论之后，重新检验这个结论成立的前提，把 GitHub / 论文里真实可用的 FFN 加速技术过一遍，并按「对本项目的实际端到端收益」排序。

---

## 零、结论先行

`REPORT.md` 的判断**在它自己设定的前提下是对的**，但那个前提是隐含的：

> 「不更换模型权重、不改网络结构、不微调」

在这个前提下，FFN 是纯 GEMM + 算力受限 + 没有天然稀疏，融合类优化确实没有空间。

一旦允许换权重（重新量化 / 低秩分解 / 稀疏化 / 微调 LoRA），FFN 立刻重新出现 **1.5×～1.9×** 的空间；而如果允许**减少 token 数**，则不仅能打 FFN，还能同时打那块 69.7% 的注意力——那是唯一能带来 1.8× 以上端到端收益的杠杆。

单杠杆的端到端收益（基线 17,549 ms，FFN 4,833 ms = 27.5%）：

| 杠杆 | FFN 加速 | 端到端 | 端到端耗时 | 需要付出 |
|---|---:|---:|---:|---|
| 什么都不改（现状） | 1.00× | 1.000× | 17,549 ms | — |
| int8 内核效率调优（不换权重） | 1.2–1.4× | 1.05–1.09× | ~16.1 s | 纯工程，无质量风险 |
| **W4A4 量化（真 INT4 张量核心）** | 1.6–1.9× | **1.10–1.13×** | ~15.5 s | 需要 W4A4 权重 |
| **2:4 激活稀疏（SpTC）** | 1.55× | **1.11×** | ~15.8 s | 需要 LoRA 补偿 + 新内核 |
| 中间维降秩 30% | 1.43× | 1.09× | ~16.1 s | 需要校准 + 质量验证 |
| FFN 跨步缓存（隔步复用） | ~2× | 1.16× | ~15.1 s | 与「减步数」同源，画质风险 |
| **Token 削减 30%** | 1.43×（且注意力 2×） | **1.79×** | ~9.7 s | 需要微调 / 质量评估 |
| Token 削减 30% + W4A4 | 2.7×（且注意力 2×） | **~2.1×** | ~8.1 s | 两者叠加 |

一句话：**FFN 本身的天花板是端到端 −13%；但如果把 FFN 当「token 数的线性函数」来看，减 token 才是真正的 FFN 加速技术，收益是量化路线的 7 倍。**

---

## 一、先复核成本模型：有一个必须澄清的歧义

`REPORT.md` 的算术我复算过，是对的：

```
hidden 5376, ffn 14336, token 25818
每 block 参数 = 3 × 5376 × 14336 = 231.2 M     (W1 + W3 + W2，SwiGLU)
每 block FLOP = 2 × 231.2e6 × 25818 = 11.94 TFLOP
50 block       = 597 TFLOP
实测 4,833 ms  → 123.5 TFLOPS
```

问题出在下一步的除法上。A10 有两个峰值：

| A10 峰值（dense，不含稀疏） | 数值 |
|---|---:|
| FP16 / BF16 Tensor Core | 125 TFLOPS |
| INT8 Tensor Core | 250 TOPS |
| INT4 Tensor Core | 500 TOPS |
| 显存带宽 | 600 GB/s |

- 若 FFN 跑在 **INT8**：123.5 / 250 = **49.4% MFU** —— 偏低但合理。
- 若 FFN 跑在 **FP16**：123.5 / 125 = **98.8% MFU** —— 物理上不可能。

123.5 TFLOPS 恰好落在 FP16 峰值的 99% 上，这个数字巧合本身就是个信号。有两种解释，指向完全不同的下一步：

1. **确实是 INT8 GEMM，49% MFU** → 内核效率还有 1.3–1.5× 可回收空间，同时 INT4 还能再翻一倍峰值。
2. **其实没吃到 INT8 张量核心**（例如量化只是省了显存，GEMM 仍是 FP16/BF16）→ 那么「前馈是算力受限、已经到头」的结论整个塌掉，光是让它真跑 INT8 就有接近 2×。

### 本机源码给出的部分答案

`python_embeded/Lib/site-packages/comfy_kitchen/backends/cuda/__init__.py:2510`：

```
"""CUDA W4A8/W6A8 linear using chunked INT4/INT6 decode and the tuned INT8 GEMM."""
```

即 W6A8 走的是 **INT6 分块解码 → INT8 GEMM**。所以路线 1 成立：确实在用 INT8 张量核心，49% MFU 是真实的。

**但**注意 `m <= 8` 有一条「fused in-register dequant GEMV」快路；本项目 m = 25,818，走的是 chunked 路径——意味着**每一步、每一个 block 都要把 6-bit 权重重新解码成 int8**（每步 11.6 G 权重）。按带宽粗算这一趟约 50 ms/step（占 FFN 1.1%），不是主因，但它是「每步重做、无法跨步缓存」的结构性开销，值得查。

### 建议先做的 15 分钟微基准

在 A10 上直接测这个形状的三个内核，就能把歧义彻底钉死：

| 测什么 | 形状 | 期望值 | 含义 |
|---|---|---|---|
| `torch._scaled_mm` int8 | M=25818, N=28672, K=5376 | 若 ≥180 TOPS | 当前 pipeline 有 1.4× 可回收 |
| BF16 GEMM（同形状） | 同上 | ~110–120 TFLOPS | 若实测≈123 TFLOPS，说明 FFN 其实没比 bf16 快 |
| `convrot_w4a4_linear` | 同上 | 若 ~250+ TFLOPS | W4A4 通路确认可用（见杠杆一） |

**这一步不做，后面的所有推算都建立在 49% 这个可能不准的分母上。**

---

## 二、杠杆一：更低比特 + **真** INT4 张量核心（W4A4）

`REPORT.md` 已经正确地指出「降到 4-bit 不会快，除非内核真的用上 INT4 张量核心」。这句话有权威背书——ViDiT-Q（ICLR 2025）原文：

> "For W4A8, since current DiT computation exhibits a **compute-bound** characteristic, the W4A8 CUDA kernel primarily saves memory **without enhancing efficiency**."
> —— 实测：W8A8 延迟 1.71×，**W4A8 延迟只有 1.38×**

这就是为什么报告里 W6A8 → W4A8 的直觉是错的：**权重 4-bit、激活 8-bit，仍然跑在 INT8 张量核心上，吞吐一点不变。** 必须有 4-bit **激活**才能落到 INT4 张量核心。

### INT4 vs INT8 的真实加速比（有实测数据）

微软那篇《Understanding INT4 Quantization for Language Models》(arXiv:2301.12017) 给了同形状对比，非常有参考价值：

| GEMM 形状 | INT4 / INT8 实测 |
|---|---:|
| MLP output（M × h × 4h） | **1.96×** |
| MLP intermediate（M × 4h × h） | ~1.9× |
| Attention output（M × h × h） | 1.46× |
| 端到端（encoder） | 1.26–1.46× |

**关键洞察：越「宽」的 GEMM 越接近 2×。** 本项目的 up/gate 投影是 M×28672×5376，down 是 M×5376×14336，都是最宽的形态，属于 INT4 收益最好的那一档。

注意：INT4 GEMM 至今 **cuBLAS 不支持，只有 CUTLASS**（`mma.sync.m16n8k64.s4` 在 SM80+ 可用）。这也解释了为什么这件事必须由推理引擎自己做，PyTorch 层面拿不到。

### 现成的三条 W4A4 路线

| 项目 | 定位 | 对本项目的价值 |
|---|---|---|
| **SVDQuant / Nunchaku**（ICLR 2025 Spotlight，mit-han-lab） | 唯一在 ComfyUI 生态落地的 W4A4 引擎；用低秩分支吸收权重离群值，量化库是 DeepCompressor | ⭐⭐⭐⭐⭐ 架构、内核、ComfyUI 插件都是现成的，只缺 H3 的量化包 |
| **QVGen**（ICLR 2026，ModelTC/港科广） | 视频 DiT 专用 **QAT**，W4A4 首次接近全精度；覆盖 CogVideoX / Wan 1.3B–14B，有 HF 权重 | ⭐⭐⭐⭐ 方法论可抄（辅助模块 + 秩衰减做到零推理开销），但需要训练 |
| **ViDiT-Q**（ICLR 2025） | 视频 DiT PTQ，W4A8 混合精度；给出 W4A8 不提速的实测证据 | ⭐⭐⭐ 主要价值是那份「负面结论」的权威出处 |

SVDQuant 的两个数字值得记住：
- 12B FLUX 上显存 3.6× 降低，**比 NF4 W4A16 基线快 3.0×**（说明真正的加速来自 W4A4 的张量核心，不是权重变小）。
- 低秩分支如果朴素实现会引入 **57% 延迟开销**，必须靠内核融合（down-proj + quantize 融合、up-proj + 4bit-compute 融合）才能把收益拿回来。——这一点对「自己写内核」是个警告。

### 本项目最该先试的一步（不用做任何新东西）

本机 `custom_nodes/ComfyUI-QuantFunc/qf_h3_modelpatcher.py:568` 明确写着：

> "H3 svdq is PRE-quantized, so the create is MINIMAL: **the svdquant metadata carries the layout/precision**"

也就是说：**这个工程里已经存在 MiniMax-H3 的 SVDQuant 预量化包和对应推理后端**（`model_backend="svdq"`）。如果它就是 W4A4，那你手上已经有一个能直接测 W4A4 的 H3，不需要重新量化。

⚠️ 但必须控制变量：它走的是 native session 路径（`MiniMaxH3` + 原生 KSampler），`ComfyUI-H3-Accel-Suite` 的 `block.forward` patch 未必生效、`comfy_kitchen` 的 W6A8 内核也不会被调用。所以要对比的只能是「整条 pipeline 的端到端耗时」，且必须确认两边 token 数 / 步数 / 分辨率完全一致。

### 预期收益

| 情形 | FFN 加速 | FFN 耗时 | 端到端 | 备注 |
|---|---:|---:|---:|---|
| 保守（低秩分支 + 量化开销吃掉一部分） | 1.55× | 3,118 ms | 1.10× | 对标 ViDiT-Q 的 W4A8 1.38× 会有提升，但不一定到 2× |
| 中性 | 1.75× | 2,762 ms | **1.13×** | 宽 GEMM 接近 1.9× 的现实折扣 |
| 乐观 | 1.90× | 2,544 ms | 1.15× | 若内核干净、低秩 rank 小 |

**结论：W4A4 是「不改结构」路线的终点，端到端最多 −13%。**

---

## 三、杠杆二：2:4 结构化稀疏（这条路线报告里被误否了）

`REPORT.md` 第六节写：

> 「激活稀疏（DejaVu、PowerInfer、CATS…）❌ 基本不适用：这些依赖 ReLU 类激活的天然稀疏性，而 H3 用的是 SwiGLU，输出不稀疏」

这个判断对 **ReLU 天然稀疏 + 跳过神经元** 那一类（DejaVu / PowerInfer / MoEfication）是成立的。但它漏掉了 **N:M 结构化稀疏 + 稀疏张量核心（SpTC）** 这条完全不同的路线：

- 不依赖激活函数天然产出 0，而是**主动用 Top-K 把每 4 个元素里最小的 2 个置零**；
- 硬件上由 Ampere 的 Sparse Tensor Core 直接吃，理论 2×；
- 阈值是按 block 用「贪心 + ℓ2 重构误差」优化的，不需要训练（TEAL, ICLR 2025）。

### 直接命中 DiT 的两篇

**RT-Lynx（ICML 2026, arXiv:2605.26632）——这篇几乎是为你的场景写的**

核心发现：DiT 的**权重**近似高斯、无结构，剪 2:4 会崩（Qwen-Image FID 21.98 → 51.63）；但**激活**每个 token 只有 5–10% 的通道显著激活（superposition 效应），所以 2:4 该剪的是激活不是权重。

- 手段：norm 补偿（保留元素的 ℓ2 范数）+ rank-64 LoRA 残差补偿 + 选择性跳层 + 融合 CUDA kernel（在线 Top-K 与 SpGEMM 合到一个 kernel，因为朴素实现的调度开销会吃掉 40–59% 时间）
- 结果：Qwen-Image / FLUX.1-dev / Z-Image-Turbo 上 **linear 层平均 1.55×**，端到端 ~1.2×，FID/IR 不退化
- 代价：需要为每个模型微调 LoRA（backbone 冻结）

**SparseDM（arXiv:2404.10445）**：2:4 权重稀疏 + 改进 STE + 渐进知识迁移，U-ViT/DiT 上 MACs 减半，**A40 实测 1.2×**（理论 2× 与实测 1.2× 的差距来自 2-bit index 矩阵的管理开销——这再次说明 SpTC 的工程落地损耗很大）。

### 与 INT4 叠加

A10 的稀疏峰值是 dense 的 2×：

```
INT8 dense 250 TOPS → INT8 sparse 500 TOPS
INT4 dense 500 TOPS → INT4 sparse 1000 TOPS
```

理论上「W4A4 + 2:4 激活稀疏」能摸到 1000 TOPS，是当前 123.5 TFLOPS 的 8×。现实里 SpTC + INT4 的组合内核极少有人写（Nunchaku 没做，CUTLASS 的 sparse 实例是 fp16/int8 为主），工程风险高，不建议第一优先。

### 预期收益

| 情形 | FFN 加速 | FFN 耗时 | 端到端 |
|---|---:|---:|---:|
| RT-Lynx 同类（linear 1.55×） | 1.55× | 3,118 ms | 1.108× |
| 打 7 折（H3 的 SwiGLU 激活分布更平） | 1.35× | 3,580 ms | 1.075× |

单看不如 W4A4，但**它和 W4A4 可叠加**，且不改变位宽（画质损失模式不同，可互补）。

---

## 四、杠杆三：直接降 FLOP（最被忽略的一类）

### 3.1 中间维降秩（纯权重手术，无需内核工作）

FFN 的 FLOPs 正比于中间维 d（本项目 d = 14336）。把 up/down 这一对矩阵联合低秩分解，d 就能缩到任意 r：

```
FFN FLOPs ∝ 3 · h · d   →   3 · h · r
         (h=5376, d=14336)

r = 10048 (70%) → FFN FLOPs × 0.70 → FFN 耗时 ~3,383 ms → 端到端 1.09×
r =  7168 (50%) → FFN FLOPs × 0.50 → FFN 耗时 ~2,417 ms → 端到端 1.16×
```

代表工作（都是 training-free，只需一次校准前向）：

| 方法 | 要点 | 对本项目 |
|---|---|---|
| **MoDeGPT**（Microsoft） | 把 MLP 的 up/down 当一个**模块**联合分解：固定 up 的列选择矩阵，down 用 Nyström 在激活相关矩阵上闭式求解。25–30% 压缩保留 90–95% zero-shot，吞吐 +46% | ⭐⭐⭐⭐ 数学最干净，无需 adapter，正好匹配「up/down 成对」结构 |
| **SVD-LLM** | 用激活协方差白化权重再截断，丢弃的奇异值精确对应输出误差 | ⭐⭐⭐⭐ 实现最简单，一行 SVD |
| **SliceGPT** | 正交变换 + 切掉残差维度 | ⭐⭐⭐ 需要残差 adapter，对本项目引入新算子，不划算 |

**两个必须注意的工程坑：**

1. **维度对齐**。张量核心按 tile 工作，维度必须 8/16/64 对齐，否则加速全被吃掉——业界实测有「15% 压缩但完全没加速」「38% 反而更慢」的案例。本项目 h=5376（=64×84）、d=14336（=64×224）本身就对齐，选 r 时务必取 64 的倍数（如 9984、7168）。
2. **生成任务比语言任务更敏感**。降秩在 LLM 上 perplexity 只涨一点，但在扩散模型上更可能表现为「纹理变糊 / 高频细节丢失」。必须先做单帧 PSNR/SSIM + 人眼对比。

### 3.2 Token 削减 —— 唯一能同时打 attention 和 FFN 的杠杆

这是整份调研里最重要的一节。

FFN 的 FLOPs 是 **O(N)**，注意力是 **O(N²)**。减 token 对两者的收益不对称：

```
N → 0.7N :  FFN 0.70×,  attention 0.49×
N → 0.5N :  FFN 0.50×,  attention 0.25×
```

按本项目实测占比算：

| token 比例 | 注意力 | 前馈 | 其他 | 总计 | 端到端 |
|---:|---:|---:|---:|---:|---:|
| 100% | 12,237 | 4,833 | 479 | 17,549 | 1.00× |
| 80% | 7,832 | 3,866 | 383 | 12,081 | **1.45×** |
| 70% | 5,996 | 3,383 | 335 | 9,714 | **1.81×** |
| 50% | 3,059 | 2,417 | 240 | 5,716 | **3.07×** |

注意 70% 这一行的端到端收益（1.81×）**大于所有 FFN 单点优化加起来**。而 864×480 的视频 latent 有 25,818 个 token，时空冗余极大——这是本场景最肥的一块肉。

代表工作：

| 方法 | 做法 | 是否需训练 |
|---|---|---|
| **SparseDiT / FlexDiT**（NeurIPS 2025, arXiv:2412.06028） | 三层分段（Poolingformer / SDTM / dense）+ 按去噪阶段动态调密度；DiT-XL 上 FLOPs −55%、速度 +175%，视频 Latte-XL FLOPs −56%、速度 +111% | 需微调（约 6% 从头训练成本） |
| **CoReDiT** | 空间一致性打分 + 线性时间跳过冗余 token + 邻域聚合重建；自注意力 FLOPs −55%，云端 1.33×、移动 NPU 1.72× | 需训练 |
| **ToMe**（Token Merging） | 训练无关的 token 合并 | 免训练，但 DiT 上画质掉得厉害（SparseDiT 论文明确指出 ToMeSD 显著劣化） |

⚠️ 本项目是 **ref2va**（参考图 + 视频 + 音频联合去噪）。参考图 token 必须完整保留，不能参与合并/剪枝；可削减的只有视频帧内的时空冗余。这一条约束会显著降低实际可削减比例（保守 15–20%），但即便 80% 也有 1.45×。

### 3.3 FFN 跨步缓存

`REPORT.md` 把 caching 归为「与减步数同源，范围外」，但有一条更细的路：**只缓存 FFN 输出，attention 照常算**。

| 方法 | 机制 | 报告数字 |
|---|---|---|
| **FORA**（arXiv:2407.01425） | 每隔 N 步重算，其余步直接复用 attention + MLP 的缓存输出；训练无关 | N=3 时 2.8×，FID 2.27→2.82 |
| **SmoothCache**（Apple, arXiv:2411.10510） | 按层分析跨步表示误差，自适应决定哪些层缓存 | 8–71% 加速，Open-Sora / Stable Audio 都验证过 |
| **Cache-DiT / TeaCache / FBCache** | 社区已在 ComfyUI / SGLang 落地 | 1.5–2× |

优点：零训练、零权重改动、实现量小。
缺点：H3 只有 **10 步**，缓存间隔的可操作空间极小（N=3 意味着 10 步里只有 3 次真算，画质风险远大于 50 步的 DiT）。**在 10 步模型上这条路基本不成立**——这一点报告的判断是对的。

---

## 五、杠杆四：内核与系统层（不换权重，风险最低）

### 5.1 现状（本机源码核对）

| 事实 | 出处 |
|---|---|
| W6A8 = INT6 分块解码 + tuned INT8 GEMM | `comfy_kitchen/backends/cuda/__init__.py:2510` |
| m ≤ 8 有 fused in-register dequant GEMV 快路；m=25818 不走这条 | 同上，`_W4A8_CHUNKED and m <= 8` |
| W4A4 内核已存在：`convrot_w4a4_linear` / `scaled_mm_svdquant_w4a4` | 能力表（A10 日志确认 `convrot_w4a4` 已加载） |
| SwiGLU 已折进 INT8 量化器（0.42.0 起） | `REPORT.md` 第六节 |

### 5.2 还能抠的三块

**（a）6-bit 权重每步重解码**
每步要把 11.6 G 个权重从 6-bit 解码成 int8 才能进 GEMM，三趟内存搬运（读 8.7 GB + 写 11.6 GB + GEMM 再读 11.6 GB）≈ 53 ms/step。占比只有 1.1%，但**如果热 block 能保持 int8 resident**（22 GB 显存里塞得下几个 block），这 53 ms 就是白拿的。其次是解码 kernel 会偷走张量核心的发射周期。

**（b）中间张量的往返带宽**
SwiGLU 的中间激活是 `25818 × 28672`，bf16 下 **1.48 GB/block**。若 up/gate GEMM 的结果要写回显存再被 SwiGLU 读走，一来一回 2.96 GB → 4.9 ms/block → **50 block 共 245 ms**，占 FFN 5%。

`REPORT.md` 说 swiglu 已折进量化器，但**没有分段数据证明中间张量真的没落盘**。建议在 profiler 里加一个 `mlp_epilogue` 子段，把 up/gate GEMM 和 down GEMM 分开计时——如果 down GEMM 之前有一段说不清的几百毫秒，就是它。

**（c）gate/up 双 GEMM 合并**
W1 与 W3 形状相同、输入相同，可以拼成 N = 28672 的单个 GEMM 再切分。宽 N 的 GEMM 在 tensor core 上效率更高（也是 INT4 收益最大的形态）。需要确认 CK 是否已经这样做了；如果还是两次 N=14336 的调用，合并能白拿几个百分点。

**（d）已排除的**
- Stream-K / Split-K：M=25818 足够大，不需要（Split-K 是给 M/N 小、K 大的场景）。
- torch.compile / inductor：量化层 graph break 26 处，已实测无效。
- DeepGEMM（DeepSeek FP8）：只支持 Hopper，SM86 用不了。
- Liger-Kernel 的 fused SwiGLU：为 LLM 训练设计，优化目标是显存不是算力，在算力受限场景收益接近 0。

### 5.3 换硬件才是真解（如果预算允许）

A10 的根本问题是**没有 FP8**（Ampere 全系无 FP8 张量核心）。同样一份模型换卡：

| GPU | 相关峰值（dense） | 相对 A10 INT8 |
|---|---|---:|
| A10（现状） | INT8 250 TOPS / INT4 500 TOPS | 1.0× |
| L40S / 4090（Ada） | FP8 733 / 330 TFLOPS | 1.3–2.9× |
| H100（Hopper） | FP8 1,979 TFLOPS | ~8× |
| B200 / 5090（Blackwell） | FP4 (NVFP4) | ~10×+ |

Nunchaku 在 4090 上的 W4A4 实测已经是「比 BF16 快 2.4×、比 NF4 W4A16 快 3×」，而且支持 NVFP4（5090 上比 BF16 快 3.1×）。**在 A10 上抠 1.2× 的工程量，可能不如换一张 Ada 卡。** 这条建议很俗，但成本核算上经常是对的。

---

## 六、技术清单总表

| # | 技术 | 类型 | 代表实现 | FFN 加速 | 端到端（本项目） | 需换权重 | 需训练 | 风险 | 优先级 |
|---|---|---|---|---:|---:|---|---|---|---|
| 1 | W4A4 真 INT4 张量核心 | 量化 | SVDQuant/Nunchaku, QVGen | 1.6–1.9× | 1.10–1.13× | ✅ | ❌/⚠️ | 中 | **P0** |
| 2 | Token 削减 15–30% | 结构 | SparseDiT, CoReDiT, ToMe | 1.2–1.4× | **1.3–1.8×** | ❌ | ⚠️ | 高 | **P0**（收益最大） |
| 3 | 中间维降秩 20–30% | 分解 | MoDeGPT, SVD-LLM | 1.25–1.43× | 1.06–1.09× | ✅ | ❌ | 中 | P1 |
| 4 | 2:4 激活稀疏 | 稀疏 | RT-Lynx | 1.35–1.55× | 1.08–1.11× | ✅ | ✅(LoRA) | 高 | P1 |
| 5 | int8 内核效率调优 | 系统 | 自研 | 1.2–1.4× | 1.05–1.09× | ❌ | ❌ | 低 | P1 |
| 6 | 6-bit 解码 resident 化 | 系统 | 自研 | 1.01× | 1.01× | ❌ | ❌ | 低 | P2 |
| 7 | 中间张量 epilogue 融合 | 系统 | 自研 | 1.05× | 1.01× | ❌ | ❌ | 低 | P2 |
| 8 | 2:4 + W4A4 叠加 | 稀疏+量化 | 无成熟实现 | 2.5×+ | 1.16×+ | ✅ | ✅ | 极高 | P3 |
| 9 | FFN 跨步缓存 | 缓存 | FORA, SmoothCache | ~2× | 1.16× | ❌ | ❌ | 高（仅 10 步） | P3 |
| 10 | MoE 化（MoEfication） | 结构重排 | thunlp/MoEfication | 1.5–2× | 1.1× | ✅ | ⚠️ | 高 | 不建议 |
| — | 换 Ada/Blackwell 卡 | 硬件 | — | — | 2–3× | — | — | 低 | 视预算 |

---

## 七、建议的执行顺序

**第 0 步（必做，成本极低）：把 49% MFU 这个分母钉死。**
写一个 30 行的微基准，在 A10 上跑 M=25818/N=28672/K=5376 的 int8、bf16、w4a4 三个内核，拿到各自的 TFLOPS。这一步会决定后面所有推算是否成立，也可能直接推翻「已经到头」的结论。

**第 1 步（如果第 0 步显示 int8 有 1.3× 以上头寸）：先抠内核，再谈模型。**
不碰权重、无质量风险、收益确定。同时验证 `mlp_epilogue` 是否真的有 245 ms 的中间张量往返。

**第 2 步：A/B 已有的 svdq(H3) 包。**
`ComfyUI-QuantFunc` 里已有 H3 的 svdq 预量化通路，用它在 **完全相同的 token 数 / 步数 / 分辨率** 下跑一遍，取前馈段耗时与 W6A8 对比。这是**零成本拿到 W4A4 真实数字**的唯一机会——注意 Accel Suite 的 patch 在 native session 路径上可能不生效，比较必须做在同一条 pipeline 上。

**第 3 步：把 token 数当成一等公民。**
哪怕只削减 15–20 个百分点的视频 token，端到端收益也超过前面所有 FFN 优化之和。这条路需要微调或至少精细的质量评估，但从投入产出比看，它是唯一值得投入人力研究的。

**一句话总结这份调研**：
`REPORT.md` 说的「FFN 融合已经到头」是对的，但「FFN 加速已经到头」不对。FFN 还有 1.5–1.9× 的空间，只是它不在算子融合里，而在**位宽（W4A4）、稀疏（2:4 激活）、和 token 数**这三件事上；而三者中收益最大的那个，恰好也能顺手解决那块 69.7% 的注意力。

---

## 八、参考链接

| 名称 | 会议/时间 | 链接 |
|---|---|---|
| SVDQuant（W4A4 扩散模型量化） | ICLR 2025 Spotlight | https://github.com/mit-han-lab/nunchaku · https://github.com/mit-han-lab/deepcompressor |
| QVGen（视频 DiT W4A4 QAT） | ICLR 2026 | https://arxiv.org/pdf/2505.11497 · https://github.com/ModelTC/QVGen |
| ViDiT-Q（视频 DiT PTQ，含 W4A8 不提速的实测） | ICLR 2025 | https://arxiv.org/abs/2406.02540 |
| RT-Lynx（DiT 激活 2:4 稀疏） | ICML 2026 | https://arxiv.org/abs/2605.26632 |
| SparseDM（2:4 权重稀疏扩散） | arXiv 2404 | https://arxiv.org/abs/2404.10445 |
| SparseDiT / FlexDiT（token 稀疏化） | NeurIPS 2025 | https://arxiv.org/abs/2412.06028 |
| CoReDiT（一致性引导 token 剪枝） | 2026 | https://www.alphaxiv.org/zh/overview/2605.14191 |
| MoDeGPT（模块级低秩分解） | 2024 | Microsoft，搜 "MoDeGPT Modular Decomposition" |
| SVD-LLM（激活感知 SVD 压缩） | 2024 | 搜 "SVD-LLM Truncation-aware SVD" |
| TEAL（免训练激活稀疏 + Triton 稀疏核） | ICLR 2025 | https://arxiv.org/abs/2408.14690 |
| MoEfication（FFN 转 MoE） | ACL 2022 Findings | https://github.com/thunlp/MoEfication |
| FORA（DiT 跨步缓存） | 2024 | https://arxiv.org/abs/2407.01425 |
| SmoothCache（自适应层缓存） | 2024 Apple | https://arxiv.org/abs/2411.10510 |
| 理解 INT4 量化（INT4 vs INT8 GEMM 实测 1.96×） | arXiv 2301 | https://arxiv.org/abs/2301.12017 |
| CUTLASS Ampere 架构说明（INT4 mma 形状 / tile 推荐） | 文档 | https://github.com/NVIDIA/cutlass |
| NVIDIA A10 规格（INT8 250 / INT4 500 TOPS） | 官方 datasheet | https://www.nvidia.com/en-us/data-center/products/a10-gpu/ |

---

## 九、A10 实测补录（2026-10-25 21:10，空闲 GPU，模型已 unload）

这一节把第一节里那个「49% MFU 到底是内核效率问题还是流水线开销」的疑问直接测掉了。

### 9.1 实测数据

用 `torch.nn.functional.linear`（bf16）和 `comfy_kitchen.int8_linear`（模型真正用的 INT8 内核），
在 M=25818 的真实形状上单跑：

| 形状 | bf16 | ck int8 | int8/bf16 |
|---|---:|---:|---:|
| up+gate：M×28672×5376（7.96 TFLOP） | 70.4 TFLOPS (56.3%) | **133.4 TOPS (53.4%)** | **1.89×** |
| down：M×5376×14336（3.98 TFLOP） | 75.6 TFLOPS (60.5%) | **129.3 TOPS (51.7%)** | 1.71× |

（百分比分别是占 A10 bf16 125 TFLOPS / INT8 250 TOPS 的比）

### 9.2 三条结论

**① 内核这条路基本走到头了。**
模型内实测 FFN = 123.5 TFLOPS，而同一个内核独立跑 = 129~133。
**95% 的达成率**，说明 CK 的流水线（6-bit 解码 → 量化 → GEMM → epilogue）几乎没有可回收的开销。
想再快只能换一个**比 CK 更强**的 GEMM（CUTLASS 手调），而那要打赢的对手是 53% MFU 这个数字本身。

**② ~55% MFU 是这台机器 + 这个形状的天花板，不是 int8 特有。**
bf16 也是 56~60%。所以「int8 效率低」不成立，是形状/A10 的固有属性。
指望靠调优把 53% 拉到 80% 不现实。

**③ INT4 是唯一能从位宽上拿到的翻倍，而且它是「换峰值」不是「提效率」。**
按同样的 ~53% 达成率外推：

```
INT8 实测   53% × 250 TOPS = 133 TOPS   ← 现在这里
INT4 理论   53% × 500 TOPS = 265 TOPS   ← 正好 2×
```

这正好对上第七节的估算：FFN 2× → 端到端 1.10~1.13×。
**注意 int8 对 bf16 只有 1.8× 而不是 2×**，所以 W4A4 对 bf16 的实际倍率预期也应该按 1.8× 打，
而不是按 4× 的纸面峰值。

### 9.3 对路线排序的影响

- 「不换权重、只调内核 → 拿 1.2~1.4×」这个 P1 选项**从清单里划掉**（实测只剩 5%）。
- W4A4 从「理论可行」升级为「唯一能翻倍的路线」。
- 换硬件（Ada/Blackwell 的 FP8/FP4）同样成立，且是**唯一质量零变化**的路线——
  因为它换的是峰值和指令集，不是数学模型。

### 9.4 关于「质量有没有下降」的验收方法（重要）

今晚的 aggressive 事件给了一个很硬的教训：**1% 的逐块数值差，经过 10 步采样就会放大成
PSNR 14.57 dB、细节完全不同的另一条轨迹**（见 `AGGRESSIVE_BUG.md` 第七节）。

所以对扩散模型：

- **PSNR / 与基线的差异度 不能当质量指标**——它测的是「可复现性」，不是「好不好看」。
- 正确的验收：固定 seed + 多 prompt（≥10 条）成片，看
  （a）人眼有没有糊/闪烁/纹理丢失/音画不同步；
  （b）客观质量分（如 VBench 的 imaging quality / dynamic degree / flickering）。
- 任何量化/降秩实验都必须**同时**记录「与基线的 PSNR」和「绝对质量分」，两者一起看：
  前者掉了不代表变差，后者掉了才是真退步。
