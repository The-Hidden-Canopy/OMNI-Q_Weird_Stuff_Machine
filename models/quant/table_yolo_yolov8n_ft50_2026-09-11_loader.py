#!/usr/bin/env python
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


def load_state_dict(npz_path: str) -> "dict[str, object]":
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
