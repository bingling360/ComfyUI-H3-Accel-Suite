# MiniMax H3 非注意力加速 —— 调研与实测报告

日期：2026-10-25
节点包：`ComfyUI-H3-Accel-Suite`
实测环境：A10（22 GB，SM86）/ ComfyUI 0.42.0 / comfy-kitchen 0.2.40 / comfy-aimdo 0.6.0 / torch 2.13.0+cu130 / Linux
模型：`minimax_h3_ref2va_pruned_w6a8.safetensors`（W6A8 量化）/ 864×480 / 25,818 token / 6 调制段 / 10 步

---

## 一、一句话结论

**在 DiT 单步前向里，「归一化+调制」和「门控残差」加起来只占 2.7%。融合类优化的物理上限就是这 2.7%，现已拿到 1.2%。这条线已经到头了。**

真正的耗时在注意力（69.7%）和前馈（27.5%）。前馈是**纯 GEMM、受算力限制**，在不更换模型权重的前提下几乎没有空间。

---

## 二、这个节点做了什么

节点包位置：
- 本机：`D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-H3-Accel-Suite`
- A10：`/mnt/workspace/ComfyUI/custom_nodes/ComfyUI-H3-Accel-Suite`

### 两个节点

| 节点 | 作用 |
|---|---|
| `MiniMax H3 Accel Suite` | 加速开关（在 UNETLoader 之后、采样器之前） |
| `MiniMax H3 Accel Stats` | 读回实测：速度对比表 + 分段占比 + 图像差异 |

### Suite 节点的参数

| 参数 | 选项 | 说明 |
|---|---|---|
| `fused_modulation` | `none` / `safe` / `aggressive` | 核心加速开关，见下 |
| `torch_compile` | `none` / `inductor` / `cudagraphs` | ComfyUI 原生 torch.compile |
| `compile_scope` | `whole_model` / `blocks` | 编译范围 |
| `compile_mode` | `default` / `reduce-overhead` / `max-autotune` | 编译模式 |
| `run_label` | 字符串 | A/B 命名 |
| `measure_timing` | 布尔 | 每次前向计时，落到 JSONL |
| `profile_sections` | 布尔 | 分段耗时（需 `fused_modulation` 非 none） |

### `fused_modulation` 三档的区别

| | 合并算子数 | 显存往返 | 数值 | 单步省 |
|---|---|---|---|---|
| `none` | 3（归一化 + 乘 + 加） | 3 趟 | 原始 | — |
| `safe` | 2（归一化 + 融合乘加） | 2 趟 | **逐位一致** | ~210 ms |
| `aggressive` | 1（`rms_adaln` 全融合） | 1 趟 | 差 ~2e-5 | ~305 ms（理论） |

> `aggressive` 用 comfy_kitchen 的 `rms_adaln` 内核，把 RMSNorm 的可学习权重
> 折进 scale：`rmsnorm(x)*w*(1+scale)+shift ≡ rms_adaln(x, w*(1+scale)-1, shift)`。

### 没有重复造轮子

- `torch_compile` 直接调 `comfy_api.torch_helpers.set_torch_compile_wrapper`（ComfyUI 本体 `TorchCompileModel` 就是调这个）
- 融合内核 vendor 自 ComfyUI-sol-attn 的 `sol_kernel/h3_fusion.py`（Apache-2.0，许可证已随包），**只依赖 torch + triton，不需要装 sol-attn**

---

## 三、实测数据（核心）

### 分段耗时（`fuse=safe`，单次前向 17,549 ms）

| 段 | 毫秒 | 占比 |
|---|---:|---:|
| 注意力 | 12,237.38 | **69.7%** |
| 前馈（MLP） | 4,833.02 | **27.5%** |
| 归一化+调制 | 304.93 | 1.7% |
| 门控残差 | 173.53 | 1.0% |
| **合计** | **17,548.86** | **100%** |

### 三组配置对比（单步均值，剔除含初始化的首步）

| 配置 | 单步均值 | 相对 none |
|---|---:|---:|
| `fuse=safe` | 17,385.9 ms（8 样本） | **−1.19%** |
| `fuse=none` | 17,595.9 ms（9 样本） | 基准 |
| `fuse=aggressive`（实际回退 eager） | 17,495.4 ms（9 样本） | −0.57% |

> ⚠️ 第三行**不是 aggressive 的成绩**。该轮全程回退 eager，等效于 `none`，那 100 ms 是噪声底。
> aggressive 修复生效后的实测结论见 `AGGRESSIVE_BUG.md`：它会让画质彻底崩坏（latent 数值爆炸），
> 原因是绕过 `cast_weight()` 直接读 `norm1.weight`。**不要开 aggressive。**

**噪声判读**：第三组全程回退到 eager，理论上应与 `none` 相同，却快了 100 ms
→ 噪声底约 **0.57%**。`safe` 的 1.19% 约为噪声的 2 倍，真实但很小。

**模型验证**：modulation 实测 305 ms，`safe` 去掉其中 2/3（乘+加，保留归一化）
≈ 200 ms，实测省 210 ms。预测与实测吻合。

### 天花板

```
融合能影响的全部 = 调制 1.7% + 门控 1.0% = 2.7%
```

即使把这两块**全部归零**，单步也只从 17,549 ms 降到 17,070 ms（−2.7%）。

---

## 四、发现并修复的三个 Bug（已推送 A10）

### 1. `aggressive` 完全失效

```
[H3 fusion] eager fallback: RuntimeError: Expected all tensors to be on the
same device, but found at least two devices, cuda:0 and cpu!
```

**根因**：DynamicVRAM + `async weight offloading with 2 streams` 下，
block 运行时 `norm1.weight` 可能仍在 CPU，而权重折叠直接 `weight.float()` 与 CUDA 上的 scale 相撞。

**修复**：新增 `_resident()`，用 `comfy.model_management.cast_to` 按需迁移
（设备一致时函数开头即返回，零开销）。

### 2. 分段数据串台

`fuse=none` 那次日志明确写着 `❌ profile_sections: 未生效`，
但 JSONL 里仍有 sections，且 4 条记录数值一字不差 —— 是上一轮的残留。

**修复**：关闭 `profile_sections` 时调用 `profiler.reset()`。

### 3. 每步落一条记录

日志全是 `median ... over 1 calls`。固定 3 秒 flush 延迟 < 单步 17 秒，
导致一次 10 步运行变成 10 条记录。

**修复**：延迟改为 `max(3, 2 × 单步秒)` → 17 秒步长对应 34 秒，一次运行一条。

---

## 五、兼容性矩阵（基于源码核对）

| 组合 | 结论 | 依据 |
|---|---|---|
| 融合 + CK 注意力 | ✅ 兼容 | CK 走 `transformer_options["optimized_attention_override"]`，融合原样透传 |
| 融合 + LoRA | ✅ 兼容 | LoRA 改权重，融合只改 forward |
| 融合 + KJNodes H3 Low VRAM | ❌ **互斥** | 两者都替换 `block.forward`；本节点会跳过已被别人打的 block |
| 融合 + vc_attention | ✅ 兼容 | 不同层 |
| torch_compile + 量化模型 | ❌ 无效 | 每个量化线性层 graph break（实测 `graph breaks: 26`） |
| torch_compile + LoRA | ⚠️ 换 LoRA 重编译 | 顺序应为 LoRA 在前、编译在后 |

**一个额外发现**：H3 的 `model.py:28` 是 `from comfy.ldm.modules.attention import optimized_attention`
（直接绑定名字），而 vc_attention 在 `patch.py:297` 做模块级替换
`comfy_attention.optimized_attention = hooked` —— **改不了已绑定的引用**，
该 hook 对 H3 很可能空转（只剩 `scaled_dot_product_attention` 的全局 hook 可能命中）。待实测。

---

## 六、前馈（MLP）技术调研

### 先定性：前馈是算力受限，不是带宽受限

按实测参数算（hidden 5376、ffn 14336、25,818 token）：

```
每 block MLP 参数 = 5376×28672 + 14336×5376 = 231.2 M
每 block 算力 = 2 × 231.2e6 × 25818 = 11.94 TFLOP
50 block 合计 = 597 TFLOP
实测 4,833 ms → 123.5 TFLOPS（A10 INT8 峰值 250 TOPS，约 49% MFU）

权重读取 = 231.2e6 × 0.75 B × 50 = 8.67 GB
按 600 GB/s 仅需 14.4 ms → 占前馈耗时 0.3%
```

**结论：权重带宽只占 0.3%，前馈是纯算力受限。**

这直接否掉了「降到 4-bit 就能快」的直觉——**除非内核真的用上 INT4 张量核心**（Ampere 上理论速率为 INT8 的 2 倍）。若只是减少权重存储，收益为零。

### 论文与项目现状

| 方向 | 代表 | 对本场景适用性 |
|---|---|---|
| 低比特量化 W4A4 | QVGen（ICLR 2026，CogVideoX/Wan）、SVDQuant（韩松组）、HiSQRot4、arXiv:2610.11315 | 需要 **重新量化的 checkpoint**；收益取决于是否走 INT4 张量核心 |
| 低秩分支补偿 | SVDQuant、arXiv:2610.11315 | 同上，属量化流程一环 |
| 激活稀疏（跳过神经元） | DejaVu、PowerInfer、CATS、SpAct-NDP、Spark Transformer | ❌ **基本不适用**：这些依赖 ReLU 类激活的天然稀疏性，而 H3 用的是 **SwiGLU**（`model.py:211`），输出不稀疏 |
| 2:4 结构化稀疏 | — | 需要专门稀疏化的权重 |
| FFN 跨时间步缓存 | 各类 caching 方法 | 与「减步数」同源，已在本项目范围外 |
| 内核层面（已实现） | swiglu 折进 INT8 量化器 | ✅ 0.42.0 起对 W6A8 生效，本项目已吃到 |

### 可用但未尝试的

comfy_kitchen 的 CUDA 能力表里已有：

- `convrot_w4a4_linear`
- `scaled_mm_svdquant_w4a4` / `quantize_svdquant_w4a4`
- `w4a8_int8_linear`（当前 W6A8 走的同一个内核族）

A10 日志也确认 `Native ops: convrot_w4a4, int8_tensorwise, w6a8_int8, asym_w4a8_int8`。
**即 W4A4 的通路已经存在**，缺的是「H3 的 W4A4 权重」。

---

## 七、推荐配置

### 当前（W6A8 + CK 注意力 + 加速 LoRA）

```
fused_modulation = safe        （或 aggressive，修好后可试）
torch_compile    = none        ← 量化模型别开
compile_scope    = blocks      （无效）
compile_mode     = default     （无效）
run_label        = 填写，用于 A/B
measure_timing   = true
profile_sections = true        （需要分段数据时）
```

### 参数选择理由

| 参数 | 为什么 |
|---|---|
| `torch_compile = none` | W6A8 每个量化线性层 graph break，实测 26 处；编译白等 |
| `safe` vs `aggressive` | `safe` 逐位一致、已验证 −1.19%；`aggressive` 最多再 −0.6% 却会毁画质（见 `AGGRESSIVE_BUG.md`），**用 `safe`** |

### 性价比排序（实测占比 × 可优化空间）

1. **注意力（69.7%）** —— 稀疏注意力是最大杠杆，但你说这块已经卷到底了
2. **前馈（27.5%）** —— 唯一路径是换更低比特权重 + 硬件支持 INT4
3. **融合（2.7%）** —— 已到顶，不建议再投入

---

## 八、待办 / 未验证项

| 项 | 状态 |
|---|---|
| `aggressive` 修复后是否真的生效 | ❌ **已定性：生效了，但输出是坏的**（latent 数值爆炸 → 品红/绿随机块）。根因与修复见 `AGGRESSIVE_BUG.md`；结论是不要开 |
| `aggressive` 的画质影响 | ✅ **已测**：vs `none` 整段 PSNR **14.57 dB**（逐帧 16.36 → 11.28 → 18.58），构图相关系数 0.80→0.40。两段画质同级，是**采样轨迹漂移**不是画质下降。但代价换来的是 **0.07%** 的速度（17094 vs 17106 ms），无意义 |
| W6A8 swiglu 融合（0.42.0 起）实际收益 | ⏳ 未单独量化：需对比 0.41.x 的前馈段耗时，成本高 |
| vc_attention 对 H3 是否空转 | ⏳ 待实测（关掉该节点看耗时有无变化） |
| 分区分析在 GPU 上的开销 | 估算约 350 event/前向，几毫秒；未实测 |

---

## 九、附：怎么读报告

计时记录落盘位置：

```
/mnt/workspace/ComfyUI/custom_nodes/ComfyUI-H3-Accel-Suite/h3_accel_stats.jsonl
```

JSONL 格式，一行一次运行。想清空直接删除文件。

跑一次生成，等约 30 秒（延迟 = 2 × 单步），再执行 `MiniMax H3 Accel Stats` 节点即可看到：

```
时间                  标签               每次前向 ms      相对首行  配置
------------------------------------------------------------------------------------------------
2026-10-25 18:38:35 -               17569.94    1.000x  fuse=none, compile=off
2026-10-25 18:47:32 -               17375.84    1.011x  fuse=aggressive, compile=off
------------------------------------------------------------------------------------------------
相对首行 > 1 表示比首行快。

最近一次分段占比：
  注意力             69.7%   12237.38 ms
  前馈              27.5%    4833.02 ms
  归一化+调制           1.7%     304.93 ms
  门控残差             1.0%     173.53 ms
```
