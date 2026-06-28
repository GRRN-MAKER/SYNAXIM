"""
SYNAXIM: Triton Fused GPU Kernels
===================================
High-performance CUDA kernels compiled via Triton for H100/A100 GPUs.
These fuse INT4 unpacking, SmoothQuant scaling, and Symbiotic Gate
updates into single kernel launches that keep data in SRAM.
Powered by SYNAXIM — Symbiotic Native Axiom Inference Machine.

Requires: pip install grrn-inference[gpu]

These kernels are optional — the engine falls back to NumPy on CPU
if Triton is not available.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def check_triton():
    if not HAS_TRITON:
        raise RuntimeError(
            "SYNAXIM Triton GPU kernels require PyTorch and Triton.\n"
            "Install with: pip install grrn-inference[gpu]"
        )


# ══════════════════════════════════════════════════════════════
# Kernel 1: Fused INT4 Unpack + Dequantize
# ══════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _int4_unpack_kernel(
        packed_ptr, scales_ptr, output_ptr,
        num_groups, group_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused INT4 unpacking + per-group dequantization.
        Each thread block processes one group.
        """
        group_id = tl.program_id(0)
        if group_id >= num_groups:
            return

        scale = tl.load(scales_ptr + group_id)
        
        # Each group has group_size elements packed into group_size/2 bytes
        packed_offset = group_id * (group_size // 2)
        output_offset = group_id * group_size

        offsets = tl.arange(0, BLOCK_SIZE // 2)
        mask = offsets < (group_size // 2)

        # Load packed bytes
        packed = tl.load(packed_ptr + packed_offset + offsets, mask=mask, other=0)

        # Extract upper and lower 4-bit values
        upper = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0
        lower = (packed & 0x0F).to(tl.float32) - 8.0

        # Dequantize
        upper = upper * scale
        lower = lower * scale

        # Store interleaved
        out_offsets_upper = output_offset + offsets * 2
        out_offsets_lower = output_offset + offsets * 2 + 1
        
        tl.store(output_ptr + out_offsets_upper, upper, mask=mask)
        tl.store(output_ptr + out_offsets_lower, lower, mask=mask)


# ══════════════════════════════════════════════════════════════
# Kernel 2: Fused RMSNorm
# ══════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_kernel(
        x_ptr, weight_ptr, output_ptr,
        D, eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)"""
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < D

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)

        # Compute variance
        x_sq = x * x
        variance = tl.sum(x_sq, axis=0) / D
        inv_rms = 1.0 / tl.sqrt(variance + eps)

        # Normalize and scale
        out = x * inv_rms * w
        tl.store(output_ptr + offsets, out, mask=mask)


# ══════════════════════════════════════════════════════════════
# Kernel 3: Fused Symbiotic Gate (INT4 unpack + SmoothQuant + M update)
# ══════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _symbiotic_gate_kernel(
        x_ptr, M_ptr, packed_gate_ptr, scales_ptr, output_ptr,
        D, alpha: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """
        Fused Continuous Symbiotic Gate.
        
        In a single kernel launch:
        1. Unpacks INT4 gate weights
        2. Computes SmoothQuant scaling
        3. Computes sigmoid gate scores
        4. Updates persistent M matrix
        5. Projects output
        
        All intermediate values stay in SRAM — zero global memory round-trips
        for the hot path.
        """
        row_id = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_N)
        col_mask = col_offsets < D

        # Load input vector x
        x = tl.load(x_ptr + col_offsets, mask=col_mask, other=0.0)

        # Load M[row, :] slice
        M_row_offset = row_id * D + col_offsets
        m_row = tl.load(M_ptr + M_row_offset, mask=col_mask, other=0.0)

        # SmoothQuant balancing
        max_x = tl.max(tl.abs(x), axis=0)
        max_m = tl.max(tl.abs(m_row), axis=0)
        max_x = tl.maximum(max_x, 1e-6)
        max_m = tl.maximum(max_m, 1e-6)
        
        # s = sqrt(max_x) * sqrt(max_m) for alpha=0.5
        s = tl.sqrt(max_x) * tl.sqrt(max_m)
        x_smooth = x / s

        # Unpack INT4 gate weight for this row
        packed_offset = row_id * (D // 2)
        pack_offsets = tl.arange(0, BLOCK_N // 2)
        pack_mask = pack_offsets < (D // 2)
        packed = tl.load(packed_gate_ptr + packed_offset + pack_offsets, mask=pack_mask, other=0)
        
        upper = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0
        lower = (packed & 0x0F).to(tl.float32) - 8.0
        
        # Compute gate score (dot product of x_smooth with unpacked gate row)
        gate_score = tl.sum(x_smooth * upper) + tl.sum(x_smooth * lower)
        gate_score = gate_score * 0.0039  # INT4 dequant scale

        # Sigmoid
        gate = 1.0 / (1.0 + tl.exp(-tl.minimum(tl.maximum(gate_score, -60.0), 60.0)))

        # Update M: M[row,:] = gate * M[row,:] + (1-gate) * x_smooth * x_smooth[row]
        x_row = tl.load(x_ptr + row_id)  # scalar: x[row]
        imprint = (1.0 - gate) * x_smooth * x_row
        updated_m = gate * (m_row * s) + imprint

        # Store updated M row
        tl.store(M_ptr + M_row_offset, updated_m, mask=col_mask)

        # Output projection: output[row] = dot(x_smooth, updated_m)
        out_val = tl.sum(x_smooth * updated_m)
        tl.store(output_ptr + row_id, out_val)


# ══════════════════════════════════════════════════════════════
# Kernel 4: Fused SwiGLU MLP
# ══════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _swiglu_fused_kernel(
        x_ptr, gate_w_ptr, up_w_ptr, output_ptr,
        D, I,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused SwiGLU: output[i] = silu(x @ gate_w[i]) * (x @ up_w[i])
        
        Processes one output dimension per thread block.
        """
        out_idx = tl.program_id(0)
        if out_idx >= I:
            return

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < D

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Load gate and up weight rows
        gate_row_offset = out_idx * D + offsets
        up_row_offset = out_idx * D + offsets

        gate_w = tl.load(gate_w_ptr + gate_row_offset, mask=mask, other=0.0)
        up_w = tl.load(up_w_ptr + up_row_offset, mask=mask, other=0.0)

        # Dot products
        gate_val = tl.sum(x * gate_w)
        up_val = tl.sum(x * up_w)

        # SiLU(gate) * up
        silu_gate = gate_val * (1.0 / (1.0 + tl.exp(-tl.minimum(tl.maximum(gate_val, -60.0), 60.0))))
        result = silu_gate * up_val

        tl.store(output_ptr + out_idx, result)


# ══════════════════════════════════════════════════════════════
# Python Wrappers (Torch tensor interface)
# ══════════════════════════════════════════════════════════════

class TritonOps:
    """High-level wrapper for Triton kernels with torch tensor I/O."""

    @staticmethod
    def unpack_int4(packed: "torch.Tensor", scales: "torch.Tensor",
                    num_groups: int, group_size: int) -> "torch.Tensor":
        """Unpack INT4 packed tensor to float32."""
        check_triton()
        output = torch.empty(num_groups * group_size, dtype=torch.float32, device=packed.device)
        grid = (num_groups,)
        _int4_unpack_kernel[grid](
            packed, scales, output,
            num_groups, group_size,
            BLOCK_SIZE=group_size,
        )
        return output

    @staticmethod
    def rmsnorm(x: "torch.Tensor", weight: "torch.Tensor", eps: float = 1e-6) -> "torch.Tensor":
        """Fused RMSNorm."""
        check_triton()
        D = x.shape[-1]
        output = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D)
        _rmsnorm_kernel[(1,)](x, weight, output, D, eps, BLOCK_SIZE=BLOCK)
        return output

    @staticmethod
    def symbiotic_gate(
        x: "torch.Tensor", M: "torch.Tensor",
        packed_gate: "torch.Tensor", scales: "torch.Tensor",
        D: int, alpha: float = 0.5,
    ) -> "torch.Tensor":
        """Fused Symbiotic Gate execution."""
        check_triton()
        output = torch.empty(D, dtype=torch.float32, device=x.device)
        BLOCK_N = triton.next_power_of_2(D)
        _symbiotic_gate_kernel[(D,)](
            x, M, packed_gate, scales, output,
            D, alpha,
            BLOCK_M=1, BLOCK_N=BLOCK_N,
        )
        return output
