# aggressive 模式画质崩塌 —— 根因报告（已确诊）

日期：2026-10-25
状态：**根因已定位并修复，真机验证通过（NaN 计数 0）**
触发条件：`fused_modulation = aggressive`
产物：`seg_000.mp4`（864×480 / 24fps / 8s / 192 帧）

---

## 零、一句话结论

**`comfy_kitchen.rms_adaln` 的 CUDA 内核在 `scale` 或 `shift` 是 `float32` 时会返回 NaN / 垃圾值，
而它的 `FunctionConstraints` 把 fp32 列为支持。H3 的 adaln 调制表恰好就是 fp32**
（因为模型走 adaln curve，`adaln_dtype = torch.float32`），于是 aggressive 把 fp32 直接喂进内核，
块 0 就出 NaN，50 个 block 一路污染，最终 latent 完全跑飞。

**修法一行**：调用前把 scale/shift cast 到激活的 dtype。这不是权宜之计——
eager 路径本来就是这么做的（`_mod_row(scale, row, h.dtype)`），只是 aggressive 漏了这一步。

---

## 一、损坏形态

### 1.1 视频层面（`seg_000.mp4`）

| 指标 | 实测 | 含义 |
|---|---|---|
| 全通道饱和像素 | 0.7% | 不是整幅饱和 |
| 16 px 块内/块间方差 | 143 / 11540 | 每个 latent 格一个纯色，格间无关联 |
| 相邻帧平均差 / 首末帧差 | 37 / 118（满分 255） | 帧间完全不相关 |
| R/B 直方图 | 双峰（两端各 3300 万） | 两通道被打到两端 |
| 6×6 亮度网格 | 全部 ≈ 120 | 无任何低频结构 |

### 1.2 latent 层面

`ComfyUI/output/h3_projects/<id>/seg_000.pt`（fp32，video `[1,24,57,30,54]` + audio `[1,32,2,320]`）：

| 项目 | video std | video max | \|x\|>10 | audio std | 说明 |
|---|---:|---:|---:|---:|---|
| 12 / 13 / 14 | 1.18 / 1.18 / 1.07 | ±5.6 | 0% | 0.48 | 正常 |
| **16（aggressive）** | **13.23** | **±43.6** | **37.7%** | **0.25** | 跑飞 |

---

## 二、定位过程（三次翻转，全部有实测支撑）

### 2.1 先证伪「数学错」

用仓库里真实的 `_make_rms_adaln_modulate` 闭包（不是重写）在 CPU 上对比 eager：

| 权重 | aggressive vs eager | safe vs eager |
|---|---:|---:|
| ~1.0 | 3.13e-02 | 3.13e-02 |
| 训练后 | 3.13e-02 | 3.13e-02 |
| ≈3 | 1.25e-01 | 1.25e-01 |

两者**完全一致**，误差全部来自「eager 先把 `1+scale` 舍入到 bf16」。fold 恒等式没问题。

### 2.2 再证伪「权重是坏值」和「内核数值错」

在 A10 上直接读 checkpoint：

```
blocks.0.norm1.weight   bf16 (5376,)  min -0.3965  max 2.7812  mean 0.9233
50 block × norm1/norm2 全部扫描：|w|max 全局 = 3.1406（block48.norm1）
final_layer.norm.weight |w|max = 1.3125
```

真正的量化体是 linear：`mlp.fc1.weight int8 (28672,4032)` + `weight_s_channel(fp32)` + `weight_s_rel(fp8)`。
**norm 是 bf16，量级全在 [0.3, 3.2]。**

内核本身（`ck.rms_adaln` vs 参考，A10）：

| d | scale 行数 | rel_err |
|---:|---:|---:|
| 5376 | 33 / 1（广播） | 2.45e-03 / 3.06e-03 |
| 4096 | 33 / 1 | 2.86e-03 / 2.87e-03 |
| 14336 | 33 / 1 | 2.73e-03 / 2.60e-03 |

也都对——**因为这些测试全用 `scale ~ N(0,0.15)`、dtype 全 bf16，恰好绕过了真正的坑。**

### 2.3 加 A/B 埋点，一次运行抓到真身

`H3_ACCEL_AB=1` 在真机上打出的第一行：

```
[H3 fusion][AB] b0.norm1 weight raw=bfloat16/(5376,) cast=bfloat16/(5376,) max|raw-cast|=0.000e+00 eps=1e-05
[H3 fusion][AB] b0.norm1 #1 x=bfloat16|max|=122.000 scale=torch.float32[min=-1.429 max=1.209 mean=-0.689]
                           aggr|max|=nan  fused|max|=12.188  rel_diff=nan  tokens=62380
```

三个关键信息：

1. `max|raw-cast| = 0` → 权重完全一致，权重假设彻底出局；
2. **`scale=torch.float32`** → 调制表是 fp32；
3. **`aggr|max|=nan`** → 内核返回 NaN，而同样输入的 `fused|max|=12.188` 正常。

顺带看到了 `scale mean = -0.689`、`min = -4.3`——**一半通道的 `1+scale` 是负数**，
这也解释了为什么用 `N(0,0.15)` 的测试永远测不出来。

### 2.4 隔离出确切条件（A10 实测）

`x` 固定 bf16、`d=5376`、`scale∈[-1.43,1.21]`，只改 dtype：

| scale | shift | NaN 个数 | rel_err |
|---|---|---:|---:|
| bf16 | bf16 | **0** | 2.87e-03 |
| bf16 | **fp32** | **396** | 3.69e+37 |
| **fp32** | bf16 | **396** | 3.98e+37 |
| **fp32** | **fp32** | **792** | 3.98e+37 |

```
d=512  n=1    -> nan=2      out|max|=nan
d=512  n=33   -> nan=66     out|max|=nan
d=512  n=4096 -> nan=0      out|max|=inf     ← 有时退化成 inf 而不是 NaN
d=4096 n=1    -> nan=10     out|max|=nan
d=4096 n=33   -> nan=165    out|max|=nan
```

**只要 scale 或 shift 是 fp32 就崩，与 d、n 无关。** out 有时是 NaN、有时是 inf——
这正好解释了两次故障的表现不同：project 16 是 inf 撑到 12 倍（无 NaN），今天这次是 NaN 直接污染。

---

## 三、根因链

```
MiniMaxH3Model 走 adaln curve
  → adaln_dtype = torch.float32
  → scale_msa / shift_msa 是 fp32 [3M, 5376]
  → aggressive: folded = w.float()*(1+scale.float()) - 1  (也是 fp32)
                .to(seg_scale.dtype)  = fp32          ← 错在这一步
  → ck.rms_adaln(x_bf16, scale_fp32, shift_fp32) → NaN/inf
  → 块 0 就开始污染，后面 49 个 block 把残差流推到 1e5~1e6
  → 采样 10 步后 latent 完全跑飞

eager 为什么没事：_mod_row(scale, row, h.dtype) 显式 cast 到激活 dtype（bf16）
safe 为什么没事：走自定义 Triton _modulate_kernel，内部统一 to(tl.float32)，不碰 ck 内核
```

---

## 四、修复

### 4.1 主修复（一行）

```python
out[a:b] = rms_adaln(
    x[a:b], seg_scale.to(x.dtype), seg_shift.to(x.dtype), eps
)
```

`_folded_scale()` 也改成恒返回 fp32（由调用方 cast），避免把 dtype 决策藏在里面。

### 4.2 真机验证（真实权重 + 真实 fp32 表）

```
weight bf16 min -0.3965 max 2.7812
scale(fp32) min -1.600 max 1.600 | shift(fp32) min -0.300 max 0.300

修复前（fp32 直喂）      scale_dtype=float32    nan=416   rel=2.534e+37
修复后（cast 到 x.dtype） scale_dtype=bfloat16  nan=0     rel=2.924e-03

与 eager 实路径对比：rel=4.651e-03，aggr|max|=13.438 = eager|max|=13.438
```

### 4.3 端到端验证（回放你那次同一个 prompt）

用 `/queue` 抓到正在运行的 prompt，改 `fused_modulation=aggressive` 后重新提交：

```
nan 计数：0
b0.norm1  #1  scale=torch.float32[min=-1.429 max=1.209 mean=-0.689]
              aggr|max|=12.125  fused|max|=12.188  rel_diff=1.410e-02
b17.norm1  #1  aggr|max|=21.375  fused|max|=21.375  rel_diff=5.848e-03
rel_diff 全样本范围：5.8e-03 ~ 4.3e-02
```

两条路径逐元素对齐到 0.6%~4.3%，NaN 归零。

### 4.4 配套加固（防止同类问题再静默通过）

| 加固 | 内容 |
|---|---|
| 权重门 | `_acquire_weight` 改走 `cast_weight()`（vbar 预取 + dequant + meta 物化），不再直接读 `.weight` |
| 增益守卫 | `max\|w\| ≤ 8`、`max\|w(1+scale)\| ≤ 16`，越界立刻降级 |
| 失败降级 | 任一异常 → 该模块**永久**切到 `safe_fallback` + warning，不产生半坏状态 |
| 真·自检 | 非零随机输入 + 真实 hidden + **fp32 表 + scale 下探到 -1.6** + `[n,h]`/`[1,h]` 双布局，数值比对 fp32 参考，NaN 或误差 >2% 就拒绝启用 |
| eps 修正 | `torch.nn.RMSNorm.eps` 可能是 `None`，旧代码 `float(None)` 直接抛 TypeError；且回退 1e-5 是错的——`F.rms_norm(eps=None)` 在 bf16 下用 `finfo(bf16).eps = 7.8e-3`。新增 `_norm_eps()` 严格镜像 torch |

**注意**：旧的 `_probe_rms_adaln` 用**全零输入**（输出恒等于 `shift`，任何内核 bug 都通过）且**只比 shape**，
这正是日志能打出「✅ 已生效」的原因。新自检用 fp32 表 + 负 scale，这类 bug 会当场拦住。

---

## 五、遗留问题与建议

1. **这是 comfy_kitchen 0.2.40 的缺陷，值得上游报**：
   `FunctionConstraints["rms_adaln"]` 声明 scale/shift 支持 fp32，但 CUDA 内核实际不支持（返回 NaN/inf）。
   同族的 `adaln`（LayerNorm 版）未测，很可能同病。
2. **`fused_modulation` 的收益仍然只有 ~0.6%**，天花板 2.7%。现在能跑对了，但性价比没变。
   日常建议 `safe`；想省那 0.6% 再用 aggressive（新版已经不会毁片）。
3. **`H3_ACCEL_AB` 埋点保留**，默认关。以后遇到「某模式画质异常」，`H3_ACCEL_AB=1` 一次运行就能定性。

---

## 六、一句话给下次

> **内核的 dtype 约束表不等于内核真的支持那个 dtype。**
> 单元测试必须用**生产环境的真实 dtype**（fp32 表、负 scale、bf16 激活）跑，
> 而不是挑一组「舒服的」随机数——`N(0,0.15)` 的 bf16 scale 把 12 倍的 bug 藏了整整两轮。
>
> **另外：探针必须校验数值，不能只校验 shape，更不能喂全零输入。**

---

## 七、最终验收（用户双片对比，20:49 / 20:56 同一会话连跑）

### 7.1 画质：两段都正常，PSNR 反映的是"漂移"不是"损伤"

`seg_000 (1).mp4`（aggressive）vs `seg_000 (2).mp4`（none），均 864×480 / 192 帧：

| 指标 | 数值 |
|---|---|
| 整段 PSNR | **14.57 dB** |
| 逐帧 PSNR | mean 15.12 / median 14.07 / min 11.28（帧 142）/ max 19.89 |
| 逐帧曲线 | 帧0 16.36 → 帧144 16.82 → 帧168 19.26 → 帧191 18.58，中间 48–120 帧最低（13.3 左右） |
| 最大单像素差 / 平均绝对差 | 255 / 27.3 |
| 差值 >16 的像素占比 | 39.3% |
| 全局统计 | (1) mean 104.12 std 65.44 ｜ (2) mean 104.54 std 65.90 |
| 亮度相关系数 | 帧0 **0.80** → 帧96 0.64 → 帧142 0.40 |

**判读**：两段的构图、人物姿态、鱼群/海龟位置**完全一致**（相关系数 0.80），
但尾鳍形状、鱼群分布、表情等细节不同。这是**同一个 seed 下，10 步采样把 1% 的逐块数值差放大**
后的结果——不是画质下降，两段肉眼同级、无任何伪影。

**所以 PSNR 14.6 dB 在这里不是"质量分"，是"可复现性分"**：换 fused_modulation 就会换一条轨迹。
如果要做 A/B 实验（比如只改提示词），必须保证这个开关一致，否则差异会盖过你想看的东西。

### 7.2 latent 统计（最终证据）

| 项目 | 配置 | video std | video max | NaN | audio std |
|---|---|---|---:|---:|---:|
| 21 | **aggressive（已修复）** | **1.0276** | 4.68 | 0 | 0.4515 |
| 20 | none | 1.0315 | 4.66 | 0 | 0.4407 |
| 14 | safe（旧） | 1.0740 | 5.59 | 0 | 0.4546 |
| 16 | aggressive（修复前） | **13.2310** | 43.6 | 0 | 0.2500 |

修复后的 aggressive 与 none 的 latent 统计几乎重合。

### 7.3 性能：**收益是 0，这是最意外的一条**

同一会话连跑，`h3_accel_stats.jsonl`：

| 时间 | 配置 | 单步 median | mean | min |
|---|---|---:|---:|---:|
| 20:49:10 | `fuse=aggressive` | **17094.36 ms** | 21448.59 | 17018.79 |
| 20:56:23 | `fuse=none` | **17105.87 ms** | 20143.38 | 17086.30 |

**差 11.5 ms = 0.07%，等于没有。** 而且 mean 反而是 aggressive 更差。

对照 19:22 / 19:35 那两次带分段的数据：

| 配置 | 单步 | modulation 段 | attention 段 | mlp 段 |
|---|---:|---:|---:|---:|
| `fuse=safe` | 16936 ms | **290.70 ms (1.7%)** | 11927 ms | 4687 ms |
| `fuse=aggressive`（旧代码） | 15694 ms | **410.76 ms (2.6%)** | 10969 ms | 4297 ms |

**aggressive 把 modulation 段从 290 ms 拖到 411 ms**（+120 ms，+41%）。原因很明确：

- safe = `norm1(x)` 一次 + `fused_modulate` 一次（一个 Triton kernel 扫全部 token）
- aggressive = `torch.empty_like` 一次 + **6 个 segment 各 launch 一次 `rms_adaln`** + 每段一次 fp32 折叠
  + 每模块一次 `cast_weight` 上下文进出（vbar 预取/释放）

6 次小 kernel launch + 1 次分配，把"少一趟显存往返"那点收益全吃掉了还倒亏。
（那次整步反而更快，是 attention/mlp 段整体快了 8%，属于机器状态波动，不是 fusion 的功劳。）

### 7.4 结论

1. **功能已修复**：aggressive 不再出 NaN/垃圾，latent 与 none 统计一致。
2. **但它在性能上是负优化**：0.07%（噪声内），分段看是 −41% 的退化。
3. **`fused_modulation` 这一整条线（含 safe）的收益远小于 `REPORT.md` 早期声称的 −1.19%**——
   那次对比跨了不同会话，17094 / 17106 / 16936 这些数都落在同一波动带里。
4. **推荐：`none` 或 `safe`。** 不要用 aggressive：既没有速度，又会让输出与其它配置不可比。
5. 顺手修掉一个守卫误报：`_GAIN_MAX` 16 → 256。真实数据 `w·(1+scale)` 最高约 30
   （|w| 3.14 × (1+scale) 9.5），阈值 16 导致 20:49 那次跑到第 8 步时把一个模块静默降级了，
   使"aggressive"变成半 aggressive。现在只拦真正离谱的值（1e5 量级）。
6. 20:40:06 那条 `median 70198 ms` 的记录**不作数**：那是我回放时并发跑诊断脚本抢了 GPU
   （当时日志里有 CUDACachingAllocator OOM 警告），不是 fusion 的耗时。
