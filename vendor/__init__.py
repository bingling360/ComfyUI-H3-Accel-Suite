"""Vendored pieces of ComfyUI-sol-attn, kept only for the AdaLN fusion path.

h3_fusion.py   - Triton kernels, copied verbatim (Apache-2.0, see ../LICENSE-sol-attn.txt)
fusion_patch.py - block-forward replacement, adapted from sol-attn's minimax.py

The sparse-attention kernels are deliberately not vendored: the fusion path does
not need them, and they carry the SM86-SM121 dispatch table plus autotune sweep
that would be dead weight here.
"""
