"""Package fp32/fp16 ``.pt`` weights as packed low-bit masters for HuggingFace.

Born-compressed storage rule: the hub repo (``KissTheHabit/yolov8n-table-yolo``)
must not carry fp32 weight files. This packager converts an ultralytics
``.pt`` checkpoint into one packed npz + manifest per residency-slider tier
(mxfp8 / mxfp4 / mxfp2, codec: ``integrations/qualcomm/lowbit``), plus an
optional standalone loader so anyone can pull a compressed master and
dequantize it locally with only numpy + torch.

Usage (what goes on the HF card)::

    # one-time: produce packed masters + the standalone loader
    python perception/package_quant_weights.py \
        --weights models/table_yolo_v2_ft_2026-09-11.pt \
        --formats mxfp8,mxfp4,mxfp2 \
        --out-dir hf_repo_weights/ --loader

    # consumer side: pull a master, restore a working .pt
    python table_yolo_v2_ft_2026-09-11_loader.py \
        --npz table_yolo_v2_ft_2026-09-11.mxfp8.npz \
        --output restored.pt

Layout per format:
    <stem>.<fmt>.npz     arrays ``payload`` (uint8, concatenated packed codes /
                         raw passthrough bytes), ``scales`` (uint8, concatenated
                         UE8M0/E4M3 block-scale bytes), and ``meta_json`` (a
                         JSON string: {format, block_size, code_bits, tensors:
                         [{name, shape, numel, kind, dtype, payload_off,
                         payload_len, scales_off, scales_len, tensor_scale}]}).
    <stem>.<fmt>.manifest.json
                         {format, block_size, code_bits, tensors: [{name, shape,
                         dtype_passthrough?, packed_bytes, weight_cosine}],
                         totals: {params, packed_bytes, fp32_bytes, ratio,
                         mean_weight_cosine}, model: {source_file,
                         param_count}, honesty: {label, arms_ref}}.
    <stem>_loader.py     standalone single-file loader (numpy + torch only, no
                         repo deps) emitted once with ``--loader``. The
                         repo-side twin is the public ``decode_npz(npz_path)``
                         below — same code (the loader source), so the two
                         cannot drift apart.

Every floating-point tensor is quantized RNE (block-32, UE8M0 scales). Non-float
tensors (int buffers such as ``num_batches_tracked``) are stored raw as
byte-exact passthrough entries, marked ``dtype_passthrough`` in the manifest.
``totals.mean_weight_cosine`` is the mean over the *quantized* tensors only
(passthroughs are exact by construction). ``totals.ratio`` is
``packed_bytes / (params * 4)`` against the fp32 master.

Honesty: weight-only RNE + software dequantize. mxfp8 is ~lossless;
mxfp4/mxfp2 collapse this detector's mAP — the per-tier weight_cosine stats
in each manifest are the packaging-side evidence pointer, eval arms under
``evidence/benchmark_results/yolo_2bit_map_20260911_071815-*``.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.qualcomm.lowbit import (  # noqa: E402
    BLOCK_SIZES,
    FORMATS,
    PackedTensor,
    quantize_tensor,
)
from integrations.qualcomm.lowbit.lowbit_formats import CODE_BITS  # noqa: E402
from integrations.qualcomm.lowbit.vendor.mxfp_scales import (  # noqa: E402
    FP8_E4M3_MAX,
    e4m3_pack,
    e4m3_unpack,
    encode_scale,
    safe_scale,
)

HONESTY_LABEL = (
    "weight-only RNE, software dequantize; mxfp4/mxfp2 collapse this "
    "detector's mAP — see stats"
)
HONESTY_ARMS_REF = "evidence/benchmark_results/yolo_2bit_map_20260911_071815-*"

LOADER_SOURCE = '''#!/usr/bin/env python
"""Standalone low-bit weight-master loader (mxfp8 / mxfp4 / nvint2 / mxfp2).

Reconstructs an ultralytics state_dict from a packed ``.npz`` produced by
OMNI-Q's ``perception/package_quant_weights.py`` — dequantize locally, no
repo dependencies (numpy + torch only).

    python <stem>_loader.py --npz <file>.<fmt>.npz [--output restored.pt]

With ``--output`` the restored state_dict is torch.saved; load it into an
ultralytics model with::

    import torch
    from ultralytics import YOLO
    model = YOLO("<original>.pt")           # architecture + meta
    model.model.load_state_dict(torch.load("restored.pt", map_location="cpu"))

Portions derived from *The Hidden Canopy LLC* — [OMNI-Q_Weird_Stuff_Machine](https://github.com/The-Hidden-Canopy/OMNI-Q_Weird_Stuff_Machine) lowbit codec. Used with permission.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

_BLOCK_SIZES = {"mxfp8": 32, "mxfp4": 32, "nvint2": 16, "mxfp2": 32}
_CODE_BITS = {"mxfp8": 8, "mxfp4": 4, "nvint2": 2, "mxfp2": 2}
_TINY = 1.0e-30


def _torch():
    import torch

    return torch


# --------------------------------------------------------------------------- #
# UE8M0 block scales + E4M3 payload (ported from lowbit/vendor/mxfp_scales.py)  #
# --------------------------------------------------------------------------- #
def _build_e4m3_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for bits in range(256):
        sign = -1.0 if (bits & 0x80) else 1.0
        exponent = (bits >> 3) & 0x0F
        mantissa = bits & 0x07
        if exponent == 0:
            lut[bits] = sign * (mantissa * (2.0 ** -9))
        elif exponent == 15 and mantissa == 7:
            lut[bits] = np.nan
        elif exponent == 15:
            lut[bits] = sign * (1.0 + mantissa / 8.0) * 256.0
        else:
            lut[bits] = sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return lut


_E4M3_LUT = _build_e4m3_lut()


def _e4m3_unpack(payload: np.ndarray) -> np.ndarray:
    return _E4M3_LUT[np.asarray(payload, dtype=np.uint8)]


def _safe_scale(code: np.ndarray) -> np.ndarray:
    """UE8M0 decode: code 0 -> 1e-30, else max(2^(code-127), 1e-30)."""
    c = np.asarray(code, dtype=np.int32)
    val = np.where(c == 0, _TINY, np.ldexp(1.0, c - 127))
    return np.maximum(val, _TINY).astype(np.float32)


# --------------------------------------------------------------------------- #
# Payload ladders (ported from lowbit_formats.py / vendor/mxfp2_codec.py)       #
# --------------------------------------------------------------------------- #
_MXFP4_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_TERNARY_MAGNITUDES = np.array([0.0, 1.0], dtype=np.float64)
_INT2_LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)


def _mxfp4_unpack(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint8)
    mag = _MXFP4_MAGNITUDES[c & 0x07]
    return np.where((c & 0x08) != 0, -mag, mag).astype(np.float32)


def _ternary_unpack(codes: np.ndarray) -> np.ndarray:
    """code & 1 -> magnitude index; code & 2 -> sign (0x2 decodes to -0.0)."""
    c = np.asarray(codes, dtype=np.uint8)
    mag = _TERNARY_MAGNITUDES[(c & 0x01)]
    return np.where((c & 0x02) != 0, -mag, mag).astype(np.float32)


def _int2_unpack(codes: np.ndarray) -> np.ndarray:
    return _INT2_LEVELS[np.asarray(codes, dtype=np.uint8) & 0x03].astype(np.float32)


# --------------------------------------------------------------------------- #
# Storage unpacking (4 x 2-bit / 2 x 4-bit per byte; ported from lowbit_formats)#
# --------------------------------------------------------------------------- #
def _unpack_codes(payload, numel: int) -> np.ndarray:
    """4 x 2-bit codes per byte, little-endian within the byte."""
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1)
    codes = np.empty(b.size * 4, dtype=np.uint8)
    codes[0::4] = b & 0x03
    codes[1::4] = (b >> 2) & 0x03
    codes[2::4] = (b >> 4) & 0x03
    codes[3::4] = (b >> 6) & 0x03
    return codes[:numel]


def _unpack_nibbles(payload, numel: int) -> np.ndarray:
    """2 x 4-bit codes per byte, low nibble first."""
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1)
    codes = np.empty(b.size * 2, dtype=np.uint8)
    codes[0::2] = b & 0x0F
    codes[1::2] = (b >> 4) & 0x0F
    return codes[:numel]


def _unpack_auto(payload, numel: int, fmt: str) -> np.ndarray:
    bits = _CODE_BITS[fmt]
    if bits == 8:
        return np.frombuffer(payload, dtype=np.uint8)[:numel]
    return _unpack_nibbles(payload, numel) if bits == 4 else _unpack_codes(payload, numel)


# --------------------------------------------------------------------------- #
# Tensor decode — mirrors the repo codec's dequantize_tensor, op for op         #
# --------------------------------------------------------------------------- #
def decode_payload(meta: dict, payload, scales) -> np.ndarray:
    """Dequantize one packed tensor entry to float32 (exact repo-codec twin)."""
    n = int(meta["numel"])
    codes = _unpack_auto(payload, n, meta["fmt"])
    block_scales = np.frombuffer(scales, dtype=np.uint8).reshape(-1)
    expect = (n + meta["block_size"] - 1) // meta["block_size"]
    if block_scales.size < expect:
        raise ValueError(f"need {expect} scale bytes for numel {n}")
    fmt = meta["fmt"]
    if fmt == "mxfp8":
        values = _e4m3_unpack(codes)
        per_elem = _safe_scale(block_scales[:expect]).repeat(meta["block_size"])[:n]
        return (values * per_elem).astype(np.float32)
    if fmt == "mxfp4":
        values = _mxfp4_unpack(codes)
        per_elem = _safe_scale(block_scales[:expect]).repeat(meta["block_size"])[:n]
        return (values * per_elem).astype(np.float32)
    if fmt == "mxfp2":
        values = _ternary_unpack(codes)
        per_elem = _safe_scale(block_scales[:expect]).repeat(meta["block_size"])[:n]
        return (values * per_elem).astype(np.float32)
    if fmt == "nvint2":
        values = _int2_unpack(codes)
        per_elem = (
            np.maximum(_e4m3_unpack(block_scales[:expect]), np.float32(_TINY)).astype(np.float64)
            * float(meta.get("tensor_scale", 1.0))
        ).repeat(meta["block_size"])[:n]
        return (values * per_elem).astype(np.float32)
    raise ValueError(f"unknown format {fmt!r}")


def load_npz(npz_path: str) -> dict:
    """Load a packed npz; returns (meta_obj, payload_bytes, scales_bytes)."""
    with np.load(npz_path) as npz:
        meta = json.loads(npz["meta_json"].item())
        payload = npz["payload"].tobytes()
        scales = npz["scales"].tobytes() if "scales" in npz.files else b""
    return meta, payload, scales


def decode_npz(npz_path: str) -> "dict[str, object]":
    """Reconstruct the full state_dict (numpy arrays) from a packed npz."""
    meta, payload, scales = load_npz(npz_path)
    out = {}
    for entry in meta["tensors"]:
        p = payload[entry["payload_off"]:entry["payload_off"] + entry["payload_len"]]
        s = scales[entry["scales_off"]:entry["scales_off"] + entry["scales_len"]]
        shape = tuple(entry["shape"])
        if entry["kind"] == "passthrough":
            arr = np.frombuffer(p, dtype=np.dtype(entry["dtype"])).reshape(shape)
        else:
            arr = decode_payload(entry, p, s).reshape(shape)
        out[entry["name"]] = arr
    return out


def load_state_dict(npz_path: str) -> "dict[str, object]":
    """Back-compat alias for :func:`decode_npz` (kept for emitted-loader users)."""
    return decode_npz(npz_path)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npz", required=True, help="packed master .npz")
    parser.add_argument("--output", default=None,
                        help="optional path to torch.save the restored state_dict")
    args = parser.parse_args(argv)
    state = load_state_dict(args.npz)
    if args.output:
        torch = _torch()
        torch.save({k: torch.from_numpy(np.ascontiguousarray(v).copy())
                    for k, v in state.items()}, args.output)
        print(f"saved restored state_dict ({len(state)} tensors) to {args.output}")
    return state


if __name__ == "__main__":
    main()
'''


def _loader_module():
    """Compile the emitted standalone loader source into a module.

    Single source of truth for npz decoding: the repo path and the shipped
    ``<stem>_loader.py`` are the same code, so they cannot drift apart.
    """
    import types

    module = types.ModuleType("omni_q_standalone_loader")
    module.__dict__["__name__"] = "omni_q_standalone_loader"
    exec(LOADER_SOURCE, module.__dict__)
    return module


_LOADER_MODULE = None


def decode_npz(npz_path) -> "dict[str, np.ndarray]":
    """Reconstruct a state_dict (numpy arrays) from a packed master ``.npz``.

    Shared by the emitted standalone loader (same function, inside
    ``LOADER_SOURCE``) and ``perception/finetune.py``'s packed-master intake.
    Passthrough (non-float) tensors restore byte-exact; quantized tensors
    restore to the codec's float32 decode.
    """
    global _LOADER_MODULE
    if _LOADER_MODULE is None:
        _LOADER_MODULE = _loader_module()
    return _LOADER_MODULE.decode_npz(str(npz_path))


def _torch():
    import torch

    return torch


def load_state_dict(weights: str | Path) -> "dict[str, object]":
    """Load an ultralytics .pt and return its ``{name: tensor}`` state_dict."""
    torch = _torch()
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        model = ckpt.get("model")
        if hasattr(model, "state_dict"):
            return model.state_dict()
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"{weights}: no state_dict found (not a checkpoint?)")


# --------------------------------------------------------------------------- #
# Fused RNE re-projection (mxfp8)                                             #
# --------------------------------------------------------------------------- #
def rne_reproject_tensor(t: np.ndarray) -> np.ndarray:
    """Project an fp32 tensor onto the MXFP8 grid, bit-exact vs the
    ``quantize_tensor`` → ``dequantize_tensor`` round-trip but without the
    PackedTensor, stats, payload-bytes, or f64-normalized intermediates.

    The per-block scale from :func:`safe_scale` is a power of two
    (``2**(code-127)``) only while it stays above the ``1e-30`` floor — i.e.
    codes >= 28 — so the RNE pack can run directly on the f32 bit patterns:
    keep the top 3 mantissa bits with ties-to-even, then rebias by the block
    exponent (an exact exponent shift). Elements clamped to the block's
    saturation threshold encode to 0x7E by construction. Floored blocks
    (codes < 28: tiny amax) and zero-scale blocks fall back to exact vendored
    helpers. Same shape/dtype as the input.
    """
    payload, codes, applied, n, shape = _rne_pack(t)
    if n == 0:
        return np.zeros(shape, dtype=np.float32)
    values = e4m3_unpack(payload)
    out = values * np.repeat(applied, 32)
    return out[:n].reshape(shape)


def rne_pack_tensor(t: np.ndarray):
    """Pack an fp32 tensor to MXFP8 bytes + its dequantized fp32 view, sharing
    ``rne_reproject_tensor``'s exact code path (the packed Lion optimizer's
    repack primitive — IDA-TRAIN-V2 ``k_lion_mxfp8``'s encode half at torch
    level).

    Returns ``(payload, scales, dequant)``: payload uint8 with one E4M3 byte
    per element (sliced to numel, matching the codec byte stream on partial
    last blocks), scales uint8 with one UE8M0 code per 32-value block, dequant
    float32 with the same shape as the input. Bit-exact against
    ``quantize_tensor`` → ``dequantize_tensor``.
    """
    payload, codes, applied, n, shape = _rne_pack(t)
    if n == 0:
        return (np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8),
                np.zeros(shape, dtype=np.float32))
    values = e4m3_unpack(payload)
    out = values * np.repeat(applied, 32)
    dequant = out[:n].reshape(shape)
    return np.ascontiguousarray(payload[:n]), np.ascontiguousarray(codes), dequant


def _rne_pack(t: np.ndarray):
    """Shared MXFP8 RNE pack pipeline (see :func:`rne_reproject_tensor` for the
    contract). Returns ``(payload, codes, applied, numel, shape)`` where
    payload/applied are padded to whole 32-value blocks and codes holds one
    scale byte per block."""
    arr = np.ascontiguousarray(t, dtype=np.float32)
    shape = arr.shape
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n == 0:
        return (np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8),
                np.zeros(0, dtype=np.float32), 0, shape)
    nblocks = (n + 31) // 32
    pad = nblocks * 32 - n
    if pad:
        buf = np.empty(nblocks * 32, dtype=np.float32)
        buf[:n] = flat
        buf[n:] = 0.0
    else:
        buf = flat
    blocks = buf.reshape(nblocks, 32)
    finite = np.isfinite(blocks)
    absb = np.abs(blocks)
    amax = np.max(absb, axis=1, where=finite, initial=np.float32(0.0))
    codes = encode_scale(amax.astype(np.float64) / FP8_E4M3_MAX)
    applied = safe_scale(codes)                    # f32 (nblocks,), == slow path's scale
    k = codes.astype(np.int32) - 127
    with np.errstate(under="ignore", over="ignore"):
        t_lo = np.ldexp(np.float32(1.0), k - 6)    # |q| >= 2^-6 -> normal e4m3 path
        t_hi = np.ldexp(np.float32(448.0), k)      # |q| >= 448   -> 0x7E once clamped
        e1 = np.clip(9 - k, -126, 127)             # subnormal path: y = |q|*512 < 8
        f1 = np.ldexp(np.float32(1.0), e1)
        f2 = np.ldexp(np.float32(1.0), (9 - k) - e1)
    clean = bool(finite.all())
    if not clean:
        np.minimum(absb, t_hi[:, None], out=absb)  # Inf -> t_hi encodes 0x7E
    sub = absb < t_lo[:, None]
    mag = absb.view(np.uint32) & np.uint32(0x7FFFFFFF)
    sign = blocks.view(np.uint8)[:, 3 if sys.byteorder == "little" else 0::4] \
        & np.uint8(0x80)
    # RNE pack on the f32 bit patterns: kept-LSB + half-minus-1 add (ties-to-even)
    # with the per-block exponent rebias folded into the rounding constant
    # (no wrap for on-grid values; off-grid garbage is discarded by the selects).
    bias = (k + np.int32(120)).astype(np.uint32) << 3
    sh = mag >> np.uint32(20)
    rnd = mag + (sh & np.uint32(1))
    rnd += (np.uint32((1 << 19) - 1) - (bias << np.uint32(20)))[:, None]
    code_n = rnd >> np.uint32(20)
    lo = codes < 28
    has_sub = bool(sub.any())
    if has_sub or not clean or bool(lo.any()):
        code = code_n.astype(np.float32)
        if has_sub:
            y = absb * f1[:, None]
            y *= f2[:, None]
            y += np.float32(8388608.0)             # 2**23: rounds y to an integer, ties-even
            y -= np.float32(8388608.0)
            code = np.where(sub, y, code)
        if not clean:
            code = np.where(~finite, np.float32(0x7F), code)
        if bool(lo.any()):
            code = np.where(lo[:, None], np.float32(0.0), code)
        code8 = code.astype(np.uint8)
    else:
        code8 = code_n.astype(np.uint8)
    # Scale code 0 decodes against the 1e-30 floor: those blocks hold only
    # 0/NaN/Inf, so the payload is the sign plus the non-finite code.
    zb = codes == 0
    if zb.any():
        over = absb >= t_hi[:, None]
        b0 = np.where(~finite, np.uint8(0x7F), np.where(over, np.uint8(0x7E), np.uint8(0x00)))
        code8 = np.where(zb[:, None], b0, code8)
    payload = code8 | sign
    floored = lo & ~zb
    if floored.any():
        # Rare tiny-amax blocks: recompute through the vendored f64 pack so the
        # division by the 1e-30 floor rounds exactly like the slow path.
        hit = np.broadcast_to(floored[:, None], blocks.shape)
        rows = np.repeat(np.nonzero(floored)[0], 32)
        q = blocks[hit].astype(np.float64) / applied[rows].astype(np.float64)
        payload[hit] = e4m3_pack(q)
    return payload.reshape(-1), codes, applied, n, shape


def _package_format(state_dict, fmt: str, stem: str, source: Path, out_dir: Path) -> dict:
    """Quantize every float tensor to ``fmt`` and write npz + manifest."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r} (want a subset of {FORMATS})")
    payload_parts: list[bytes] = []
    scales_parts: list[bytes] = []
    meta_tensors: list[dict] = []
    manifest_tensors: list[dict] = []
    payload_off = 0
    scales_off = 0
    total_params = 0
    total_packed = 0
    cosines: list[float] = []

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        shape = list(t.shape)
        numel = int(t.numel())
        entry = {
            "name": name,
            "shape": shape,
            "numel": numel,
            "payload_off": payload_off,
            "payload_len": 0,
            "scales_off": scales_off,
            "scales_len": 0,
            "tensor_scale": 1.0,
        }
        manifest_entry: dict = {"name": name, "shape": shape}
        if t.is_floating_point():
            arr = t.numpy().astype(np.float32)
            packed: PackedTensor = quantize_tensor(arr, fmt)
            entry.update(
                kind="quant", fmt=fmt, dtype="float32",
                block_size=BLOCK_SIZES[fmt],
                payload_len=len(packed.payload),
                scales_len=len(packed.scales),
                tensor_scale=float(packed.tensor_scale),
            )
            payload_parts.append(packed.payload)
            scales_parts.append(packed.scales)
            payload_off += len(packed.payload)
            scales_off += len(packed.scales)
            packed_bytes = packed.bytes_total
            cos = float(packed.stats["weight_cosine"])
            cosines.append(cos)
            manifest_entry["weight_cosine"] = cos
        else:
            raw = t.numpy().tobytes()
            entry.update(
                kind="passthrough", dtype=str(t.numpy().dtype),
                payload_len=len(raw),
            )
            payload_parts.append(raw)
            payload_off += len(raw)
            packed_bytes = len(raw)
            manifest_entry["weight_cosine"] = 1.0
            manifest_entry["dtype_passthrough"] = str(t.numpy().dtype)
        total_params += numel
        total_packed += packed_bytes
        manifest_entry["packed_bytes"] = packed_bytes
        meta_tensors.append(entry)
        manifest_tensors.append(manifest_entry)

    payload_all = b"".join(payload_parts)
    scales_all = b"".join(scales_parts)
    meta = {
        "format": fmt,
        "block_size": BLOCK_SIZES[fmt],
        "code_bits": CODE_BITS[fmt],
        "tensors": meta_tensors,
    }
    npz_path = out_dir / f"{stem}.{fmt}.npz"
    np.savez_compressed(
        npz_path,
        payload=np.frombuffer(payload_all, dtype=np.uint8),
        scales=np.frombuffer(scales_all, dtype=np.uint8),
        meta_json=np.asarray(json.dumps(meta)),
    )

    fp32_bytes = total_params * 4
    manifest = {
        "format": fmt,
        "block_size": BLOCK_SIZES[fmt],
        "code_bits": CODE_BITS[fmt],
        "tensors": manifest_tensors,
        "totals": {
            "params": total_params,
            "packed_bytes": total_packed,
            "fp32_bytes": fp32_bytes,
            "ratio": total_packed / fp32_bytes if fp32_bytes else 0.0,
            "mean_weight_cosine": float(np.mean(cosines)) if cosines else 1.0,
        },
        "model": {
            "source_file": source.name,
            "param_count": total_params,
        },
        "honesty": {"label": HONESTY_LABEL, "arms_ref": HONESTY_ARMS_REF},
    }
    manifest_path = out_dir / f"{stem}.{fmt}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def package(weights: str | Path, formats, out_dir: str | Path,
            loader: bool = False) -> dict:
    """Package ``weights`` into one npz + manifest per format in ``out_dir``.

    With ``loader=True`` also writes ``<stem>_loader.py`` (standalone numpy +
    torch decoder). Returns ``{fmt: manifest}``.
    """
    source = Path(weights)
    if not source.is_file():
        raise FileNotFoundError(source)
    formats = [f.strip() for f in formats if f.strip()]
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r} (want a subset of {FORMATS})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    state_dict = load_state_dict(source)
    results = {}
    for fmt in formats:
        results[fmt] = _package_format(state_dict, fmt, stem, source, out)
    if loader:
        (out / f"{stem}_loader.py").write_text(LOADER_SOURCE, encoding="utf-8")
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Package .pt weights as packed low-bit masters (npz + manifest).",
    )
    parser.add_argument("--weights", required=True, help="ultralytics .pt checkpoint")
    parser.add_argument("--formats", required=True,
                        help=f"comma-separated subset of {','.join(FORMATS)}")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--loader", action="store_true",
                        help="also emit <stem>_loader.py (standalone numpy+torch decoder)")
    args = parser.parse_args(argv)
    results = package(args.weights, args.formats.split(","), args.out_dir,
                      loader=args.loader)
    for fmt, manifest in results.items():
        t = manifest["totals"]
        print(f"{fmt}: packed {t['packed_bytes']} bytes "
              f"(ratio {t['ratio']:.4f}, mean weight_cosine "
              f"{t['mean_weight_cosine']:.6f})")


if __name__ == "__main__":
    main()
