"""MiniMax H3 block-forward replacement for the fused AdaLN modulation path.

Adapted from ComfyUI-sol-attn's ``minimax.py`` (the ``MiniMaxH3FusedModulation``
node), Apache-2.0. The original resolves the same kernels through
``.sol_kernel.h3_fusion``; here they come from the vendored copy next to this
file, so ComfyUI-sol-attn no longer has to be installed.

What is kept verbatim is the behaviour that matters:

* only the elementwise modulation / gated-residual work is replaced - the
  attention and MLP calls are resolved from ``block`` at execution time, so
  object patches installed by KJNodes or sol-attn still compose;
* anything unexpected before the first residual mutation falls back to the
  original eager block, and the fallback is recorded once per reason;
* after the first gate the residual is already modified, so exceptions are
  allowed to propagate instead of being retried on dirty input.

The patched function carries ``_minimax_h3_fusion_fallback``, the same marker
sol-attn uses, so the two implementations recognise each other instead of
stacking.

Only the bit-exact variant ships. Folding the RMSNorm into the modulation (one
pass instead of three, through comfy_kitchen's ``rms_adaln``) was removed after
measurement: it moved the sample by 14.6 dB PSNR while saving 0.07% of a step,
and it depends on the kernel's dtype behaviour in ways the constraint table does
not describe. See ``../AGGRESSIVE_BUG.md``.
"""

import logging

import torch

log = logging.getLogger("H3Accel")

_FALLBACK_ATTR = "_minimax_h3_fusion_fallback"


class _FusionUnsupported(Exception):
    """Raised for calls the fused path intentionally declines."""


class _FusionLog:
    """Log each transition and each fallback reason exactly once."""

    def __init__(self):
        self.active = False
        self.fallbacks = set()

    def hit(self, tokens, segments):
        if not self.active:
            log.info(
                "[H3 fusion] active (%d tokens, %d modulation segments)",
                tokens,
                segments,
            )
            self.active = True

    def miss(self, reason):
        if reason not in self.fallbacks:
            self.fallbacks.add(reason)
            log.info("[H3 fusion] eager fallback: %s", reason)


class _SegmentIndexCache:
    """One request-layout lookup shared by all 50 patched H3 blocks."""

    def __init__(self, make_segment_index):
        self.make_segment_index = make_segment_index
        self.key = None
        self.value = None

    def get(self, tokens, segments, device, table_rows):
        if any(torch.is_tensor(row) for _, _, row in segments):
            raise _FusionUnsupported("per-token modulation rows require eager execution")
        normalized = tuple((int(a), int(b), int(row)) for a, b, row in segments)
        if not normalized or max(row for _, _, row in normalized) >= int(table_rows):
            raise ValueError("modulation segment references a missing AdaLN row")
        key = (str(device), int(tokens), normalized)
        if key != self.key:
            self.value = self.make_segment_index(tokens, normalized, device)
            self.key = key
        return self.value


def _make_fused_h3_block_forward(
    block,
    fallback_forward,
    index_cache,
    fusion_log,
    fused_modulate,
    fused_gate_add,
):
    """Replace only H3's modulation/residual elementwise work."""

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        if transformer_options is None:
            transformer_options = {}

        def fallback():
            kwargs = {"transformer_options": transformer_options}
            # Legacy blocks lack this keyword. An explicit override must never
            # disappear: unsupported legacy callbacks raise their normal error.
            if attention is not None:
                kwargs["attention"] = attention
            return fallback_forward(x, t_emb, mod_segments, rope_freqs, **kwargs)

        try:
            if any(torch.is_tensor(row) for _, _, row in mod_segments):
                raise _FusionUnsupported("per-token modulation rows require eager execution")
            if (
                x.ndim != 2
                or x.dtype != torch.bfloat16
                or x.device.type != "cuda"
                or x.requires_grad
            ):
                raise _FusionUnsupported(
                    "requires a non-autograd CUDA BF16 [tokens, hidden] activation"
                )

            mods = block.adaln_proj(t_emb)
            if len(mods) != 6:
                raise _FusionUnsupported("expected six H3 AdaLN modulation tables")
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
            table_rows = min(int(table.shape[0]) for table in mods)
            row_index = index_cache.get(x.shape[0], mod_segments, x.device, table_rows)

            # Everything up to the first residual update leaves x untouched, so
            # an eligibility/kernel failure here can safely use eager forward.
            h = fused_modulate(block.norm1(x), shift_msa, scale_msa, row_index)
        except _FusionUnsupported as exc:
            fusion_log.miss(str(exc))
            return fallback()
        except Exception as exc:  # noqa: BLE001 - safe fallback before residual mutation
            # x is still pristine, so a backend/compiler incompatibility can
            # safely fall back. After the first gate below, exceptions must
            # propagate because eager retry would consume a modified residual.
            fusion_log.miss(f"{type(exc).__name__}: {exc}")
            return fallback()

        # Attention is intentionally outside the fallback catch: strict errors
        # from an attention patch must propagate and should never be converted
        # into an eager full-block retry by this unrelated fusion.
        attention_fn = block.attn if attention is None else attention
        attended = attention_fn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        x = fused_gate_add(x, gate_msa, attended, row_index)
        h = fused_modulate(block.norm2(x), shift_mlp, scale_mlp, row_index)
        mlp_out = block.mlp(h)
        x = fused_gate_add(x, gate_mlp, mlp_out, row_index)

        fusion_log.hit(x.shape[0], len(mod_segments))
        return x

    setattr(forward, _FALLBACK_ATTR, fallback_forward)
    return forward


def apply_fused_modulation(model):
    """Patch every H3 DiT block; returns (model, status_string).

    Raises ImportError when Triton or the vendored kernel is unavailable - the
    caller decides whether that is fatal or just a skipped switch.
    """
    from .h3_fusion import fused_gate_add_, fused_modulate_, make_segment_index

    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
        log.warning("[H3 fusion] expected a MiniMax H3 model; returning it unchanged")
        return model, "skipped: not a MiniMax H3 model"

    patched = model.clone()
    fusion_log = _FusionLog()
    index_cache = _SegmentIndexCache(make_segment_index)
    installed = 0
    blocked = 0
    for i in range(len(blocks)):
        path = f"diffusion_model.blocks.{i}.forward"
        block = patched.get_model_object(f"diffusion_model.blocks.{i}")
        prior = getattr(patched, "object_patches", {}).get(path)
        if prior is not None and not hasattr(prior, _FALLBACK_ATTR):
            log.warning(
                "[H3 fusion] block %d already has an unknown forward patch; leaving it unchanged",
                i,
            )
            blocked += 1
            continue
        fallback_forward = patched.get_model_object(path)
        if hasattr(fallback_forward, _FALLBACK_ATTR):
            fallback_forward = getattr(fallback_forward, _FALLBACK_ATTR)
        patched.add_object_patch(
            path,
            _make_fused_h3_block_forward(
                block,
                fallback_forward,
                index_cache,
                fusion_log,
                fused_modulate_,
                fused_gate_add_,
            ),
        )
        installed += 1

    log.info("[H3 fusion] patched %d of %d blocks", installed, len(blocks))
    if blocked:
        return patched, (
            f"applied to {installed}/{len(blocks)} blocks ({blocked} blocked by another patch)"
        )
    return patched, f"applied to {installed}/{len(blocks)} blocks"
