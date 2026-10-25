"""MiniMax H3 acceleration: one node, one input, one output, no switches.

It replaces H3's elementwise modulation and gated residual with fused Triton
kernels (vendored from ComfyUI-sol-attn, Apache-2.0, so only Triton is needed -
not that package). The replacement is bit-identical to ComfyUI's eager path:
same ops, same rounding, one kernel instead of three.

Nothing is configurable on purpose. Everything that could be a switch was tried
and removed:

  * torch.compile - every quantized linear layer breaks the graph (26 breaks
    measured on the W6A8 checkpoint), so it compiles for minutes and gains
    nothing.
  * RMSNorm folding into the modulation ("aggressive") - measured at 0.07% of a
    step in exchange for a 14.6 dB PSNR move. See AGGRESSIVE_BUG.md.

The patch degrades to a no-op with a log line instead of raising, so a missing
optional dependency can never break a workflow.
"""

import logging

log = logging.getLogger("H3Accel")

_TAG = "[H3 Accel]"


def _is_h3(model):
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:  # noqa: BLE001
        return False
    return diffusion_model is not None and diffusion_model.__class__.__name__ == "MiniMaxH3Model"


class H3AccelSuite:
    """Fused AdaLN modulation for MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/optimization"

    DESCRIPTION = (
        "MiniMax H3 调制融合。把 modulation 与门控残差换成融合 Triton 内核"
        "（逐位一致，三次显存往返变一次）。没有开关：该试的都试过并删了——"
        "torch.compile 在量化模型上每层 graph break，RMSNorm 折叠只有 0.07% 收益却让画面漂移 14.6 dB。"
    )

    def apply(self, model):
        if not _is_h3(model):
            log.warning("%s 模型不是 MiniMax H3 —— 未做任何改动", _TAG)
            return (model,)

        try:
            from .vendor.fusion_patch import apply_fused_modulation
        except Exception as exc:  # noqa: BLE001
            log.warning("%s ❌ 融合未生效（内核不可用: %s）", _TAG, exc)
            return (model,)

        try:
            model, status = apply_fused_modulation(model)
        except ImportError as exc:
            log.warning("%s ❌ 融合未生效（需要 Triton: %s）", _TAG, exc)
            return (model,)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s ❌ 融合未生效（%s: %s）", _TAG, type(exc).__name__, exc)
            return (model,)

        if status.startswith("skipped"):
            log.warning("%s ❌ 融合未生效（%s）", _TAG, status)
        else:
            log.info("%s ✅ 融合已生效（%s）", _TAG, status)
        return (model,)


NODE_CLASS_MAPPINGS = {"H3AccelSuite": H3AccelSuite}
NODE_DISPLAY_NAME_MAPPINGS = {"H3AccelSuite": "MiniMax H3 Accel Suite"}
