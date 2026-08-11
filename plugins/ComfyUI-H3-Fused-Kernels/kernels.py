"""Triton kernels used by the MiniMax H3 fused-kernel node.

The accurate path deliberately keeps RMSNorm in PyTorch and preserves the
multiply/add rounding boundary.  It only merges H3's per-segment launches into
whole-packed-sequence launches.  The fast path additionally fuses RMSNorm and
AdaLN when the norm weight is already resident on the input CUDA device.
"""

from collections import OrderedDict
import logging

import torch


LOG = logging.getLogger("h3_fused_kernels")

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    TRITON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the local runtime
    triton = None
    tl = None
    TRITON_AVAILABLE = False
    TRITON_IMPORT_ERROR = exc


_ROW_ID_CACHE = OrderedDict()
_ROW_ID_CACHE_LIMIT = 16
_TRITON_RUNTIME_ERROR = None
_ANNOUNCED_PATHS = set()


if TRITON_AVAILABLE:

    @triton.jit
    def _scale_kernel(
        x_ptr,
        scale_ptr,
        row_ids_ptr,
        element_count,
        hidden,
        x_stride_row,
        x_stride_col,
        scale_stride_row,
        scale_stride_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        token = offsets // hidden
        column = offsets - token * hidden
        mod_row = tl.load(row_ids_ptr + token, mask=mask, other=0)
        x_offset = token * x_stride_row + column * x_stride_col
        mod_offset = mod_row * scale_stride_row + column * scale_stride_col
        value = tl.load(x_ptr + x_offset, mask=mask)
        # Native H3 explicitly casts AdaLN rows to the stream dtype before
        # every in-place operation. H3 curve checkpoints keep AdaLN in FP32,
        # so retaining this cast is required for the accurate path.
        scale = tl.load(scale_ptr + mod_offset, mask=mask).to(value.dtype)
        tl.store(x_ptr + x_offset, value * (1.0 + scale), mask=mask)


    @triton.jit
    def _shift_kernel(
        x_ptr,
        shift_ptr,
        row_ids_ptr,
        element_count,
        hidden,
        x_stride_row,
        x_stride_col,
        shift_stride_row,
        shift_stride_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        token = offsets // hidden
        column = offsets - token * hidden
        mod_row = tl.load(row_ids_ptr + token, mask=mask, other=0)
        x_offset = token * x_stride_row + column * x_stride_col
        mod_offset = mod_row * shift_stride_row + column * shift_stride_col
        value = tl.load(x_ptr + x_offset, mask=mask)
        shift = tl.load(shift_ptr + mod_offset, mask=mask).to(value.dtype)
        tl.store(x_ptr + x_offset, value + shift, mask=mask)


    @triton.jit
    def _scale_shift_kernel(
        x_ptr,
        shift_ptr,
        scale_ptr,
        row_ids_ptr,
        element_count,
        hidden,
        x_stride_row,
        x_stride_col,
        shift_stride_row,
        shift_stride_col,
        scale_stride_row,
        scale_stride_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        token = offsets // hidden
        column = offsets - token * hidden
        mod_row = tl.load(row_ids_ptr + token, mask=mask, other=0)
        x_offset = token * x_stride_row + column * x_stride_col
        shift_offset = mod_row * shift_stride_row + column * shift_stride_col
        scale_offset = mod_row * scale_stride_row + column * scale_stride_col
        value = tl.load(x_ptr + x_offset, mask=mask)
        shift = tl.load(shift_ptr + shift_offset, mask=mask).to(value.dtype)
        scale = tl.load(scale_ptr + scale_offset, mask=mask).to(value.dtype)
        tl.store(x_ptr + x_offset, value * (1.0 + scale) + shift, mask=mask)


    @triton.jit
    def _gate_kernel(
        x_ptr,
        other_ptr,
        gate_ptr,
        row_ids_ptr,
        element_count,
        hidden,
        x_stride_row,
        x_stride_col,
        other_stride_row,
        other_stride_col,
        gate_stride_row,
        gate_stride_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        token = offsets // hidden
        column = offsets - token * hidden
        mod_row = tl.load(row_ids_ptr + token, mask=mask, other=0)
        x_offset = token * x_stride_row + column * x_stride_col
        other_offset = token * other_stride_row + column * other_stride_col
        gate_offset = mod_row * gate_stride_row + column * gate_stride_col
        value = tl.load(x_ptr + x_offset, mask=mask)
        branch = tl.load(other_ptr + other_offset, mask=mask)
        gate = tl.load(gate_ptr + gate_offset, mask=mask).to(value.dtype)
        tl.store(x_ptr + x_offset, value + branch * gate, mask=mask)


    @triton.jit
    def _rms_mod_kernel(
        x_ptr,
        out_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        row_ids_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        x_stride_row,
        x_stride_col,
        out_stride_row,
        out_stride_col,
        weight_stride,
        shift_stride_row,
        shift_stride_col,
        scale_stride_row,
        scale_stride_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        token = tl.program_id(0)
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < hidden
        x_offsets = token * x_stride_row + columns * x_stride_col
        raw_value = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        value = raw_value.to(tl.float32)
        mean_square = tl.sum(value * value, axis=0) / hidden
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(
            weight_ptr + columns * weight_stride, mask=mask, other=0.0
        ).to(raw_value.dtype)
        mod_row = tl.load(row_ids_ptr + token)
        shift = tl.load(
            shift_ptr + mod_row * shift_stride_row + columns * shift_stride_col,
            mask=mask,
            other=0.0,
        ).to(raw_value.dtype)
        scale = tl.load(
            scale_ptr + mod_row * scale_stride_row + columns * scale_stride_col,
            mask=mask,
            other=0.0,
        ).to(raw_value.dtype)
        output = value * inv_rms * weight
        output = output * (1.0 + scale) + shift
        out_offsets = token * out_stride_row + columns * out_stride_col
        tl.store(out_ptr + out_offsets, output, mask=mask)


def _native_scale_shift(h, shift, scale, segments):
    for start, stop, row in segments:
        h[start:stop].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _native_gate(x, gate, other, segments):
    for start, stop, row in segments:
        x[start:stop].addcmul_(other[start:stop], gate[row].to(x.dtype))
    return x


def _segments_key(segments, total_rows, device):
    normalized = tuple((int(start), int(stop), int(row)) for start, stop, row in segments)
    cursor = 0
    for start, stop, row in normalized:
        if start != cursor or stop < start or row < 0:
            raise ValueError("H3 modulation segments must cover the packed rows contiguously.")
        cursor = stop
    if cursor != total_rows:
        raise ValueError("H3 modulation segments do not cover the complete packed sequence.")
    return (str(device), total_rows, normalized), normalized


def _row_ids(segments, total_rows, device):
    key, normalized = _segments_key(segments, total_rows, device)
    cached = _ROW_ID_CACHE.get(key)
    if cached is not None:
        _ROW_ID_CACHE.move_to_end(key)
        return cached

    host = torch.empty(total_rows, dtype=torch.int32, device="cpu")
    for start, stop, row in normalized:
        host[start:stop].fill_(row)
    cached = host.to(device=device, non_blocking=False)
    _ROW_ID_CACHE[key] = cached
    while len(_ROW_ID_CACHE) > _ROW_ID_CACHE_LIMIT:
        _ROW_ID_CACHE.popitem(last=False)
    return cached


def _can_use_triton(tensor):
    return (
        TRITON_AVAILABLE
        and _TRITON_RUNTIME_ERROR is None
        and tensor.device.type == "cuda"
        and tensor.ndim == 2
        and tensor.stride(1) == 1
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _announce(path):
    if path not in _ANNOUNCED_PATHS:
        LOG.info("H3 Fused Kernels runtime path: %s", path)
        _ANNOUNCED_PATHS.add(path)


def _handle_triton_error(exc, backend):
    global _TRITON_RUNTIME_ERROR
    if backend == "triton":
        raise RuntimeError(f"H3 Triton kernel failed: {exc}") from exc
    _TRITON_RUNTIME_ERROR = exc
    LOG.exception("H3 Triton kernel failed; auto backend is falling back to PyTorch.")


def _validate_modulation(h, shift, scale, row_ids):
    if shift.ndim != 2 or scale.ndim != 2:
        return False
    if shift.shape[1] != h.shape[1] or scale.shape[1] != h.shape[1]:
        return False
    if shift.device != h.device or scale.device != h.device or row_ids.device != h.device:
        return False
    return True


def mod_scale_shift(h, shift, scale, segments, backend, precision):
    if backend == "torch_reference" or not _can_use_triton(h):
        if backend == "triton":
            reason = TRITON_IMPORT_ERROR or _TRITON_RUNTIME_ERROR or "unsupported tensor/device"
            raise RuntimeError(f"H3 Triton backend is unavailable: {reason}")
        _announce("PyTorch reference")
        return _native_scale_shift(h, shift, scale, segments)

    rows = _row_ids(segments, h.shape[0], h.device)
    if not _validate_modulation(h, shift, scale, rows):
        if backend == "triton":
            raise RuntimeError("H3 modulation tensors are incompatible with the Triton kernel.")
        return _native_scale_shift(h, shift, scale, segments)

    count = h.numel()
    grid = (triton.cdiv(count, 256),)
    try:
        if precision == "accurate":
            _scale_kernel[grid](
                h,
                scale,
                rows,
                count,
                h.shape[1],
                h.stride(0),
                h.stride(1),
                scale.stride(0),
                scale.stride(1),
                BLOCK_SIZE=256,
            )
            _shift_kernel[grid](
                h,
                shift,
                rows,
                count,
                h.shape[1],
                h.stride(0),
                h.stride(1),
                shift.stride(0),
                shift.stride(1),
                BLOCK_SIZE=256,
            )
            _announce("Triton accurate segment fusion")
        else:
            _scale_shift_kernel[grid](
                h,
                shift,
                scale,
                rows,
                count,
                h.shape[1],
                h.stride(0),
                h.stride(1),
                shift.stride(0),
                shift.stride(1),
                scale.stride(0),
                scale.stride(1),
                BLOCK_SIZE=256,
            )
            _announce("Triton fast scale/shift fusion")
        return h
    except Exception as exc:  # first-use compilation can fail on a new architecture
        _handle_triton_error(exc, backend)
        return _native_scale_shift(h, shift, scale, segments)


def mod_gate(x, gate, other, segments, backend):
    if backend == "torch_reference" or not _can_use_triton(x):
        if backend == "triton":
            reason = TRITON_IMPORT_ERROR or _TRITON_RUNTIME_ERROR or "unsupported tensor/device"
            raise RuntimeError(f"H3 Triton backend is unavailable: {reason}")
        return _native_gate(x, gate, other, segments)

    rows = _row_ids(segments, x.shape[0], x.device)
    compatible = (
        other.ndim == 2
        and other.shape == x.shape
        and other.device == x.device
        and other.stride(1) == 1
        and gate.ndim == 2
        and gate.shape[1] == x.shape[1]
        and gate.device == x.device
    )
    if not compatible:
        if backend == "triton":
            raise RuntimeError("H3 gate tensors are incompatible with the Triton kernel.")
        return _native_gate(x, gate, other, segments)

    count = x.numel()
    grid = (triton.cdiv(count, 256),)
    try:
        _gate_kernel[grid](
            x,
            other,
            gate,
            rows,
            count,
            x.shape[1],
            x.stride(0),
            x.stride(1),
            other.stride(0),
            other.stride(1),
            gate.stride(0),
            gate.stride(1),
            BLOCK_SIZE=256,
        )
        return x
    except Exception as exc:
        _handle_triton_error(exc, backend)
        return _native_gate(x, gate, other, segments)


def _direct_norm_weight(norm, x):
    weight = getattr(norm, "weight", None)
    if weight is None or weight.device != x.device or weight.ndim != 1:
        return None
    if (
        weight.shape[0] != x.shape[1]
        or weight.stride(0) != 1
        or weight.dtype != x.dtype
    ):
        return None
    # Comfy may attach runtime weight functions (LoRA/casting). Calling the
    # native norm is mandatory in that case so those functions are preserved.
    if getattr(norm, "weight_function", None):
        return None
    return weight


def norm_scale_shift(norm, x, shift, scale, segments, backend, precision):
    """Apply RMSNorm and AdaLN, using the single-kernel path when safe."""
    if precision != "fast" or backend == "torch_reference" or not _can_use_triton(x):
        return mod_scale_shift(norm(x), shift, scale, segments, backend, precision)

    rows = _row_ids(segments, x.shape[0], x.device)
    weight = _direct_norm_weight(norm, x)
    if weight is None or not _validate_modulation(x, shift, scale, rows):
        return mod_scale_shift(norm(x), shift, scale, segments, backend, precision)

    hidden = x.shape[1]
    block_size = triton.next_power_of_2(hidden)
    if block_size > 65536:
        return mod_scale_shift(norm(x), shift, scale, segments, backend, precision)
    output = torch.empty_like(x)
    eps = float(norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps)
    try:
        _rms_mod_kernel[(x.shape[0],)](
            x,
            output,
            weight,
            shift,
            scale,
            rows,
            hidden=hidden,
            eps=eps,
            x_stride_row=x.stride(0),
            x_stride_col=x.stride(1),
            out_stride_row=output.stride(0),
            out_stride_col=output.stride(1),
            weight_stride=weight.stride(0),
            shift_stride_row=shift.stride(0),
            shift_stride_col=shift.stride(1),
            scale_stride_row=scale.stride(0),
            scale_stride_col=scale.stride(1),
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        _announce("Triton fused RMSNorm + AdaLN")
        return output
    except Exception as exc:
        _handle_triton_error(exc, backend)
        return _native_scale_shift(norm(x), shift, scale, segments)


def runtime_description():
    if TRITON_AVAILABLE:
        return f"Triton {getattr(triton, '__version__', 'unknown')} available"
    return f"Triton unavailable: {TRITON_IMPORT_ERROR}"
