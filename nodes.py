"""MiniMax H3 acceleration, one node, no switches.

The only thing this node does is replace H3's elementwise modulation and gated
residual with fused Triton kernels (vendored from ComfyUI-sol-attn, Apache-2.0,
so only Triton is needed - not that package). The replacement is bit-identical
to ComfyUI's eager path: same ops, same rounding, one kernel instead of three.

What is deliberately not here:

  * torch.compile - every quantized linear layer breaks the graph (26 breaks
    measured on the W6A8 checkpoint), so it compiles for minutes and gains
    nothing.
  * RMSNorm folding into the modulation ("aggressive") - measured at 0.07% of a
    step in exchange for a 14.6 dB PSNR move. See AGGRESSIVE_BUG.md.

Every step degrades to a no-op with a line in the report instead of raising, so
a missing optional dependency can never break a workflow.
"""

import logging

log = logging.getLogger("H3Accel")

_TAG = "[H3 Accel]"

# Stable signature written into the timing log, so rows stay comparable with
# the ones recorded before the mode switch was removed.
_CONFIG = "fuse=safe"


def _is_h3(model):
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:  # noqa: BLE001
        return False
    return diffusion_model is not None and diffusion_model.__class__.__name__ == "MiniMaxH3Model"


def _apply_fused_modulation(model, report):
    """Run the vendored AdaLN fusion; no external package required."""
    try:
        from .vendor.fusion_patch import apply_fused_modulation
    except Exception as exc:  # noqa: BLE001
        report.append(f"❌ 融合: 未生效（内核不可用: {exc}）")
        return model

    try:
        model, status = apply_fused_modulation(model)
    except ImportError as exc:
        report.append(f"❌ 融合: 未生效（需要 Triton: {exc}）")
        return model
    except Exception as exc:  # noqa: BLE001
        report.append(f"❌ 融合: 未生效（{type(exc).__name__}: {exc}）")
        return model

    if status.startswith("skipped"):
        report.append(f"❌ 融合: 未生效（{status}）")
    else:
        report.append(f"✅ 融合: 已生效（{status}）")
    return model


class H3AccelSuite:
    """Fused AdaLN modulation for MiniMax H3, plus timing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "run_label": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "给这次运行起个名字，会显示在 MiniMax H3 Accel Stats 里。",
                    },
                ),
                "profile_sections": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "把每次前向拆成 调制/注意力/门控/前馈 四段，Stats 节点就能看到时间去了哪。"
                        "每次前向多约 350 个 CUDA event，只做体检时再开。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "model_patches/optimization"

    DESCRIPTION = (
        "MiniMax H3 的调制融合 + 计时。只做一件确定有效的事：把 modulation 和门控残差换成融合 Triton 内核"
        "（逐位一致，三次显存往返变一次）。不含 torch.compile（量化模型每层都会 graph break）"
        "和 RMSNorm 折叠（实测 0.07% 收益换来 14.6 dB 的画面漂移）。"
    )

    def apply(self, model, run_label="", profile_sections=False):
        report = []

        if not _is_h3(model):
            report.append("模型不是 MiniMax H3 —— 未做任何改动")
            return (model, "\n".join(report))

        model = _apply_fused_modulation(model, report)

        timing_ok = False
        try:
            from .timing import install_timing

            model, _recorder = install_timing(model, run_label, _CONFIG)
            timing_ok = True
            report.append("✅ 计时: 开启（记录到 h3_accel_stats.jsonl）")
        except Exception as exc:  # noqa: BLE001 - stats must never break a run
            report.append(f"❌ 计时: 关闭（{type(exc).__name__}: {exc}）")

        if profile_sections:
            if not timing_ok:
                report.append("❌ 分段: 未生效（计时没有起来）")
            else:
                try:
                    from .timing import get_profiler

                    profiler = get_profiler()
                    profiler.reset()
                    profiler.enabled = True
                    report.append("✅ 分段: 开启（调制/注意力/门控/前馈）")
                except Exception as exc:  # noqa: BLE001
                    report.append(f"❌ 分段: 关闭（{type(exc).__name__}: {exc}）")
        else:
            try:
                from .timing import get_profiler

                profiler = get_profiler()
                profiler.enabled = False
                # Clear stale totals: otherwise a run with profiling off still
                # reports the previous run's section split.
                profiler.reset()
            except Exception:  # noqa: BLE001
                pass
            report.append("分段: 关闭")

        header = [
            "=" * 60,
            f"MiniMax H3 加速 · {_CONFIG}",
            f"  标签: {run_label or '(未命名)'}",
            "=" * 60,
        ]
        footer = [
            "-" * 60,
            "看实测速度：跑一次生成，等约 3 秒（或 2×单步），再执行 MiniMax H3 Accel Stats。",
        ]

        for line in report:
            log.info("%s %s", _TAG, line)

        return (model, "\n".join(header + report + footer))


class H3AccelStats:
    """Read back measured forward timings and, optionally, compare two frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "last_n": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "列出最近多少次运行。",
                    },
                ),
            },
            "optional": {
                "image_a": ("IMAGE", {"tooltip": "A 组（比如基线）的同一帧。"}),
                "image_b": ("IMAGE", {"tooltip": "B 组（比如加速后）的同一帧。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    CATEGORY = "model_patches/optimization"
    OUTPUT_NODE = True

    DESCRIPTION = (
        "显示 MiniMax H3 Accel Suite 记录的每次前向耗时，并给出相对首行的倍率。"
        "把两次运行的同一帧接到 image_a / image_b 可以看到像素差；注意 PSNR 只反映偏离度，不代表画质。"
    )

    def report(self, last_n, image_a=None, image_b=None):
        from .timing import format_table, image_diff, read_records

        blocks = [format_table(read_records(), last_n)]
        diff = image_diff(image_a, image_b)
        if diff is not None:
            blocks.append(diff)
        elif image_a is None and image_b is None:
            blocks.append("图像对比：未接图（可选，接两帧即可看到数值差异）")
        else:
            blocks.append("图像对比：只接了一路，需要 image_a 和 image_b 同时提供")
        return ("\n\n".join(blocks),)


NODE_CLASS_MAPPINGS = {"H3AccelSuite": H3AccelSuite, "H3AccelStats": H3AccelStats}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AccelSuite": "MiniMax H3 Accel Suite",
    "H3AccelStats": "MiniMax H3 Accel Stats",
}
