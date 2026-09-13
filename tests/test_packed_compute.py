"""Tests for the selected-layer packed-FP4 compute pilot."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "perception"))

from packed_compute import (  # noqa: E402
    PackedFP4Conv2d,
    decode_mxfp4,
    install_packed_fp4_pilot,
    pack_mxfp4,
)


def test_mxfp4_pack_roundtrip_uses_nibbles_and_k32_scales():
    values = torch.linspace(-2.0, 2.0, 65, dtype=torch.float32)
    payload, scales, decoded = pack_mxfp4(values)
    assert payload.dtype == torch.uint8
    assert payload.numel() == (values.numel() + 1) // 2
    assert scales.dtype == torch.uint8
    assert scales.numel() == 3
    torch.testing.assert_close(decoded, decode_mxfp4(payload, scales, 65, values.shape))


def test_packed_conv_forward_matches_decoded_weight_and_has_ste_gradient():
    torch.manual_seed(7)
    source = nn.Conv2d(3, 4, kernel_size=1, bias=True)
    pilot = PackedFP4Conv2d.from_conv(source)
    x = torch.randn(2, 3, 5, 5)
    got = pilot(x)
    decoded = decode_mxfp4(
        pilot.packed_payload,
        pilot.packed_scales,
        pilot.weight.numel(),
        pilot.weight.shape,
    ).reshape_as(pilot.weight)
    want = F.conv2d(x, decoded.to(dtype=pilot.weight.dtype), pilot.bias)
    torch.testing.assert_close(got, want)
    got.square().mean().backward()
    assert pilot.weight.grad is not None
    assert torch.isfinite(pilot.weight.grad).all()


def test_install_selects_only_matching_convolutions():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Sequential(nn.Conv2d(4, 4, 1)))
    selected = install_packed_fp4_pilot(model, r"^0$")
    assert selected == ["0"]
    assert isinstance(model[0], PackedFP4Conv2d)
    assert not isinstance(model[1][0], PackedFP4Conv2d)


def test_install_refuses_empty_selection():
    with pytest.raises(ValueError, match="matched no Conv2d"):
        install_packed_fp4_pilot(nn.Sequential(nn.ReLU()), r"model\.99")
