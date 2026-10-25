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

这个包只碰那 2.7%。**收益本来就小**——`modulation` 段约 290 ms/步，
端到端差异在 1% 量级、且部分落在机器噪声里。它的价值是**不损画质**，不是速度。

### 为什么不顺手把前馈也加速了

因为前馈已经到天花板。同一个 `int8_linear` 内核单独跑是 133.4 TOPS，
模型内 FFN 实测 123.5 TFLOPS —— **95% 达成率**，流水线里没有可回收开销。
A10 上 INT8 峰值 250 TOPS，而这个形状只跑得出 ~53% MFU（bf16 也只有 56~60%），
再往上只能换位宽（INT4）或换卡。详见 `FFN_ACCEL_RESEARCH.md` 第九节。

---

## 兼容性

| 组合 | 结论 |
|---|---|
| 加速 / 量化 LoRA | ✅ 兼容（LoRA 改权重，本节点只换 forward） |
| comfy_kitchen 注意力 / vc_attention | ✅ 兼容（不同层） |
| KJNodes H3 Low VRAM | ❌ 互斥：两者都替换 `block.forward`。本节点会**跳过**已经被别人打过的 block，并在日志里说明 |
| 非 MiniMax H3 模型 | 原样返回，不做任何改动 |

---

## 文件

| 文件 | 说明 |
|---|---|
| `nodes.py` | 单节点，81 行 |
| `vendor/fusion_patch.py` | block forward 替换（改自 sol-attn 的 `minimax.py`） |
| `vendor/h3_fusion.py` | Triton 内核，**原样复制，未改动** |
| `REPORT.md` | 分段实测与天花板分析 |
| `FFN_ACCEL_RESEARCH.md` | 前馈加速的论文/GitHub 调研 + A10 内核实测 |
| `AGGRESSIVE_BUG.md` | RMSNorm 折叠为什么会毁片：完整定位过程 |

## 许可

`vendor/h3_fusion.py` 及 `vendor/fusion_patch.py` 源自 ComfyUI-sol-attn，
Copyright NVIDIA / NVlabs，Apache-2.0，全文见 `LICENSE-sol-attn.txt`。
其余代码同样以 Apache-2.0 发布。
