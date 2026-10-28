# ComfyUI-H3-Accel-Suite

MiniMax H3 的调制融合。**一个节点，一个输入，一个输出，零参数。**

它只做一件事：把 H3 DiT block 里的 `modulation`（scale/shift）和门控残差换成融合 Triton 内核，
三次显存往返变一次。输出与 ComfyUI 原生 eager 路径**逐位一致**（`torch.equal` 验证）。

---

## 装

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:bingling360/ComfyUI-H3-Accel-Suite.git
```

重启 ComfyUI。依赖只有 `triton`（ComfyUI 环境里通常已经有了）。

内核 vendor 自 [ComfyUI-sol-attn](https://github.com/r-vage/ComfyUI-sol-attn)（Apache-2.0），
**不需要安装那个包**。

## 用

把 `MiniMax H3 Accel Suite` 接在 `UNETLoader` 之后、采样器之前：

```
UNETLoader ──► MiniMax H3 Accel Suite ──► 采样器
```

模型不是 MiniMax H3 时它会原样返回并在控制台说明，不会报错中断。

---

## 为什么一个开关都没有

不是没做，是做完测完删了。下面每一条都有实测数据：

| 曾经的开关 | 为什么删掉 |
|---|---|
| `torch_compile` / `inductor` / `cudagraphs` | W6A8 量化模型上**每个量化线性层都会 graph break**（实测 26 处），编译几分钟，收益接近 0 |
| 把 RMSNorm 折进 modulation（`aggressive`） | 单步只省 **0.07%**（17094 vs 17106 ms），画面却偏离 **14.6 dB PSNR**；且 `comfy_kitchen.rms_adaln` 的 CUDA 内核在 scale/shift 为 fp32 时**返回 NaN**（约束表却写着支持 fp32），一开就毁片 |
| 关掉融合（`none`） | 那就是不装这个节点 |
| 计时 / 分段剖析 / 对照报告 | 只对开发者有用，节点本身不该背这些 |

关于「逐位一致」和「画面偏离」的区别，可以看下面的实测：**数学等价时 PSNR ≈ 38~43 dB（差异来自编码），
换一条采样轨迹时 PSNR ≈ 14.6 dB。** 这两个数字差了一个数量级，别混用。

---

## 实测数据（A10 / 864×480 / 25,393 token / 10 步）

单步分段时间：

| 段 | 毫秒 | 占比 |
|---|---:|---:|
| 注意力 | 12,237 | 69.7% |
| 前馈（MLP） | 4,833 | 27.5% |
| 归一化 + 调制 | 305 | **1.7%** |
| 门控残差 | 174 | 1.0% |
| **合计** | **17,549** | 100% |

这个包只碰那 2.7%。实测它能拿到的部分：

| 分辨率 | token | 每步节省 | 占单步 | 10 步省 |
|---|---:|---:|---:|---:|
| 864×480（0.4 MP） | 25,393 | **127.4 ms** | **0.73%** | 1.3 s |
| 1.0 MP 档 | 62,380 | **308.5 ms** | **0.29%** | 3.1 s |

**收益就这么多，不多也不少。** 整段生成要 5~6 分钟，所以体感几乎没有——它的价值是
**逐位一致地少走一趟显存往返**，不是速度。测法、原始数据和为什么 gate 融合是零收益，
见 [`PERF.md`](PERF.md)。

> 端到端测不出来：同一配置连跑三次的波动就有**8.5%**，0.3~0.7% 完全埋在噪声里。
> 所以速度是直接对算子计时的（真实形状、真实内核、20 次中位数）。

### 端到端 A/B：它到底改没改模型输出

同种子、同一次编码、euler、192 帧，唯一变量是有没有这个节点：

| | 无节点 | 有节点 |
|---|---|---|
| **latent（`torch.equal`）** | 基准 | **完全相同，0 / 2,216,160 元素** |
| **音轨** | 基准 | **bit-identical**（2×256000 样本全等） |
| 视频 PSNR | — | **50.98 dB**，131/192 帧逐像素相同 |
| 墙钟 | 282.93 s | 288.11 s |
| 采样 8 步累计 | 132 s | 136 s |

**latent 那一行才是重点：逐位相同，一个元素都不差。这个节点不改变任何模型输出。**

墙钟那一行**没有意义**：同样配置连跑三次是 282.93 / 288.11 / 307.14 s，波动 8.5%，
而节点该省的是 1.0 s（0.35%）——比噪声小一个数量级。所以速度只能按算子级测（上表）。

> 视频那 50.98 dB 是哪来的？全部发生在 latent 之后：差异集中在前 61 帧（关键帧在 0/61/126），
> 呈 16×16 宏块对齐、沿 P 帧链放大（0.58% → 10.7% → 25.4%），到第 61 帧的 IDR 归零，
> 之后 131 帧逐像素相同。这是**编码器宏块决策差异**的形态，不是模型跑偏。
> 也就是说：同一份源、两次编码，结果都不保证一样——**判断"有没有变"要看 `seg_000.pt`，不要看视频**。

### 为什么不顺手把前馈也加速了

因为前馈已经到天花板。同一个 `int8_linear` 内核单独跑是 133.4 TOPS，
模型内 FFN 实测 123.5 TFLOPS —— **95% 达成率**，流水线里没有可回收开销。
A10 上 INT8 峰值 250 TOPS，而这个形状只跑得出 ~53% MFU（bf16 也只有 56~60%），
再往上只能换位宽（INT4）或换卡。详见 `FFN_ACCEL_RESEARCH.md` 第九节。

---

## 兼容性

**可以和 KJNodes 的 H3 显存节点一起挂。** 它们改的是 `blocks.N.attn.forward` /
`blocks.N.mlp.forward`，本节点改的是 `blocks.N.forward`——不同层级，互不覆盖：
本节点的 forward 内部照常调用 `block.attn(...)` / `block.mlp(...)`，
`add_object_patch` 的 setattr 打在实例上、遮蔽类方法，所以它们打的补丁照样会执行。

| 组合 | 结论 |
|---|---|
| 加速 / 量化 LoRA | ✅ 兼容（LoRA 改权重，本节点只换 forward） |
| comfy_kitchen 注意力 / vc_attention | ✅ 兼容（不同层） |
| **KJNodes `H3 Low VRAM Attention` / `H3 Chunk FeedForward`** | ✅ **兼容**：`blocks.N.attn.forward` / `blocks.N.mlp.forward` vs 本节点的 `blocks.N.forward`，不同层级 |
| sol-attn 的 `MiniMax H3 Fused Modulation` | ✅ 不叠加：同一个 `blocks.N.forward` 路径。代码靠 `_minimax_h3_fusion_fallback` 标记认出同族实现，只保留一层，不会叠两遍 |
| 其它抢先打过 `blocks.N.forward` 的节点 | ⚠️ 本节点会**跳过**那些 block 并在日志里说明，不会静默失效也不会崩 |
| 非 MiniMax H3 模型 | 原样返回，不做任何改动 |

### 怎么确认两边都生效

跑一次，看控制台有没有这两行（本节点这行只在第一次真正走融合时打一次）：

```
[H3 fusion] active (25393 tokens, 6 modulation segments)     ← 本节点
<kjnodes 自己打的那行>                                        ← KJNodes
```

如果本节点被同层级的未知补丁挡住，它会明说，不会假装成功：

```
⚠️ 融合未生效（applied to 0/50 blocks (50 blocked by another patch)）
```

---

## 文件

| 文件 | 说明 |
|---|---|
| `nodes.py` | 单节点，81 行 |
| `vendor/fusion_patch.py` | block forward 替换（改自 sol-attn 的 `minimax.py`） |
| `vendor/h3_fusion.py` | Triton 内核，**原样复制，未改动** |
| `REPORT.md` | 分段实测与天花板分析 |
| `PERF.md` | **这个节点本身的实测加速**：原始数据、测法、等价性 |
| `FFN_ACCEL_RESEARCH.md` | 前馈加速的论文/GitHub 调研 + A10 内核实测 |
| `AGGRESSIVE_BUG.md` | RMSNorm 折叠为什么会毁片：完整定位过程 |

## 许可

`vendor/h3_fusion.py` 及 `vendor/fusion_patch.py` 源自 ComfyUI-sol-attn，
Copyright NVIDIA / NVlabs，Apache-2.0，全文见 `LICENSE-sol-attn.txt`。
其余代码同样以 Apache-2.0 发布。
