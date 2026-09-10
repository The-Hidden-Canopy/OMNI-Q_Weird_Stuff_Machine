# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/fp8_linear.py`; body unmodified except as recorded in ../SOURCE.md.

"""fp8 GEMM substrate for the IDA Lattice (H100 / Hopper).

Validated on the genesis seed (0.0.0.1): real torch._scaled_mm forward (e4m3) +
backward (e5m2), per-tensor scaling, fp32 accumulation, bf16 master weights.
Genesis canary delta vs bf16: +0.94% loss, zero divergence, zero scaled_mm
fallbacks across 54 swapped linears.

Precision boundary (the load-bearing rule):
  fp8  — the bulk, well-conditioned GEMMs: expert FFN + block-attention projections.
  bf16 — every decision / sensitive surface: ConstitutionalRouter (softmax+topk),
         ActionGate, PressureField (feeds the router), LayerNorms, embeddings,
         and every output head (gh_head, future_token_head, etc.).

The 3.8% per-GEMM E4M3 error is the 3-bit mantissa floor (not reducible by
scaling on Hopper); it is contained by keeping it out of the decision surfaces
and letting fp32 accumulation + training adaptation absorb it in the bulk path.

Fp8Linear subclasses nn.Linear with identical parameters (weight, bias), so the
state_dict is byte-identical to nn.Linear — checkpoints round-trip across the
bf16 lineage and the fp8 substrate without conversion.

Toggle at training time with env IDA_FP8=1; call swap_to_fp8(model) after the
model is constructed and any init_from_model weights are loaded.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import os
import time as _time_fp8
import torch
from torch import nn
import torch.nn.functional as F

# When IDA_FP8_DELAYED_SCALE=1, Fp8Linear caches per-module amax history and
# refreshes scales every _SCALE_REFRESH_INTERVAL forward calls instead of every
# call. Conservative max(history)*decay policy prevents overflow; the E4M3 floor
# (3.8%) dominates over any staleness noise from a stable model.
_DELAYED_SCALE = os.environ.get("IDA_FP8_DELAYED_SCALE", "0") == "1"
_SCALE_REFRESH_INTERVAL = int(os.environ.get("IDA_FP8_SCALE_REFRESH_INTERVAL", "16"))
_SCALE_HISTORY_LEN = 16
_SCALE_DECAY = 0.99  # conservative: slow scale shrinkage between refreshes

_E4M3_MAX = 448.0      # max normal magnitude, float8_e4m3fn
_E5M2_MAX = 57344.0    # max normal magnitude, float8_e5m2

# Modules whose linears must stay bf16 — decision / sensitive surfaces.
_EXCLUDE = (
    "router", "pressure", "gate", "head", "norm",
    "prefrontal", "memory", "thalamic", "fidelity", "ontogeny", "state_head",
)
# Modules whose linears are fp8 candidates — bulk GEMMs.
# state_supersampler: down [B,H+S→rank] and up [B,rank→S] qualify by shape;
# route_proj [B,H+S→R=9] always falls back to bf16 (R not divisible by 16).
# FP32 loss boundary in low_rank_state_supersampler.py provides the stability gate.
_INCLUDE = ("experts", "workspace", "supersampler")

_FP8_SUBSTRATE_STATE: dict[str, object] = {
    "active": False,
    "swapped_linears": 0,
    "fp8_calls": 0,
    "fallback_calls": 0,
    "fallback_by_reason": Counter(),
    "fallback_shapes": Counter(),
    "amax_input_max": None,
    "amax_weight_max": None,
    "scale_input": None,
    "scale_weight": None,
    "modules": {},
    "amax_reduction_ms_total": 0.0,
    "moe_amax_reduction_ms_total": 0.0,
}


def _shape_key(m: int, k: int, n: int) -> str:
    return f"{m}x{k}x{n}"


def _module_state(name: str) -> dict[str, object]:
    modules = _FP8_SUBSTRATE_STATE.setdefault("modules", {})
    state = modules.get(name)
    if state is None:
        state = {
            "module": name,
            "fp8_calls": 0,
            "fallback_calls": 0,
            "fallback_by_reason": Counter(),
            "fallback_shapes": Counter(),
            "input_amax": None,
            "weight_amax": None,
            "scale_input": None,
            "scale_weight": None,
        }
        modules[name] = state
    return state


def _safe_float(value: torch.Tensor | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _update_max(key: str, value: float | None) -> None:
    if value is None:
        return
    current = _FP8_SUBSTRATE_STATE.get(key)
    if current is None or float(value) > float(current):
        _FP8_SUBSTRATE_STATE[key] = float(value)


def reset_fp8_substrate_state() -> None:
    _FP8_SUBSTRATE_STATE["active"] = False
    _FP8_SUBSTRATE_STATE["swapped_linears"] = 0
    _FP8_SUBSTRATE_STATE["fp8_calls"] = 0
    _FP8_SUBSTRATE_STATE["fallback_calls"] = 0
    _FP8_SUBSTRATE_STATE["fallback_by_reason"] = Counter()
    _FP8_SUBSTRATE_STATE["fallback_shapes"] = Counter()
    _FP8_SUBSTRATE_STATE["amax_input_max"] = None
    _FP8_SUBSTRATE_STATE["amax_weight_max"] = None
    _FP8_SUBSTRATE_STATE["scale_input"] = None
    _FP8_SUBSTRATE_STATE["scale_weight"] = None
    _FP8_SUBSTRATE_STATE["modules"] = {}
    _FP8_SUBSTRATE_STATE["amax_reduction_ms_total"] = 0.0
    _FP8_SUBSTRATE_STATE["moe_amax_reduction_ms_total"] = 0.0


def _record_fp8_success(module_name: str, m: int, k: int, n: int, *, input_amax: float | None, weight_amax: float | None, scale_input: float | None, scale_weight: float | None) -> None:
    _FP8_SUBSTRATE_STATE["active"] = True
    _FP8_SUBSTRATE_STATE["fp8_calls"] = int(_FP8_SUBSTRATE_STATE.get("fp8_calls", 0) or 0) + 1
    _update_max("amax_input_max", input_amax)
    _update_max("amax_weight_max", weight_amax)
    if scale_input is not None:
        _FP8_SUBSTRATE_STATE["scale_input"] = float(scale_input)
    if scale_weight is not None:
        _FP8_SUBSTRATE_STATE["scale_weight"] = float(scale_weight)
    mod = _module_state(module_name)
    mod["fp8_calls"] = int(mod.get("fp8_calls", 0) or 0) + 1
    mod["input_amax"] = input_amax
    mod["weight_amax"] = weight_amax
    mod["scale_input"] = scale_input
    mod["scale_weight"] = scale_weight


def _record_fallback(module_name: str, reason: str, m: int, k: int, n: int, *, input_amax: float | None, weight_amax: float | None) -> None:
    _FP8_SUBSTRATE_STATE["active"] = True
    _FP8_SUBSTRATE_STATE["fallback_calls"] = int(_FP8_SUBSTRATE_STATE.get("fallback_calls", 0) or 0) + 1
    _FP8_SUBSTRATE_STATE["fallback_by_reason"][reason] += 1
    _FP8_SUBSTRATE_STATE["fallback_shapes"][_shape_key(m, k, n)] += 1
    _update_max("amax_input_max", input_amax)
    _update_max("amax_weight_max", weight_amax)
    mod = _module_state(module_name)
    mod["fallback_calls"] = int(mod.get("fallback_calls", 0) or 0) + 1
    mod["fallback_by_reason"][reason] += 1
    mod["fallback_shapes"][_shape_key(m, k, n)] += 1
    mod["input_amax"] = input_amax
    mod["weight_amax"] = weight_amax


def fp8_substrate_report(*, top_k: int = 8) -> dict[str, object]:
    fp8_calls = int(_FP8_SUBSTRATE_STATE.get("fp8_calls", 0) or 0)
    fallback_calls = int(_FP8_SUBSTRATE_STATE.get("fallback_calls", 0) or 0)
    total = fp8_calls + fallback_calls
    modules: list[dict[str, object]] = []
    for module_name, raw in sorted((_FP8_SUBSTRATE_STATE.get("modules") or {}).items()):
        fallback_shapes = raw.get("fallback_shapes", Counter())
        fallback_by_reason = raw.get("fallback_by_reason", Counter())
        module_total = int(raw.get("fp8_calls", 0) or 0) + int(raw.get("fallback_calls", 0) or 0)
        modules.append(
            {
                "module": module_name,
                "fp8_calls": int(raw.get("fp8_calls", 0) or 0),
                "fallback_calls": int(raw.get("fallback_calls", 0) or 0),
                "fallback_rate": round(float(raw.get("fallback_calls", 0) or 0) / module_total, 6) if module_total else 0.0,
                "fallback_by_reason": dict(fallback_by_reason),
                "top_fallback_shapes": [
                    {"shape": shape, "count": count}
                    for shape, count in fallback_shapes.most_common(top_k)
                ],
                "input_amax": raw.get("input_amax"),
                "weight_amax": raw.get("weight_amax"),
                "scale_input": raw.get("scale_input"),
                "scale_weight": raw.get("scale_weight"),
            }
        )
    return {
        "event": "fp8_substrate",
        "active": bool(_FP8_SUBSTRATE_STATE.get("active")),
        "swapped_linears": int(_FP8_SUBSTRATE_STATE.get("swapped_linears", 0) or 0),
        "fp8_calls": fp8_calls,
        "fallback_calls": fallback_calls,
        "fallback_rate": round(fallback_calls / total, 6) if total else 0.0,
        "fallback_by_reason": dict(_FP8_SUBSTRATE_STATE.get("fallback_by_reason", Counter())),
        "top_fallback_shapes": [
            {"shape": shape, "count": count}
            for shape, count in (_FP8_SUBSTRATE_STATE.get("fallback_shapes", Counter())).most_common(top_k)
        ],
        "amax_input_max": _FP8_SUBSTRATE_STATE.get("amax_input_max"),
        "amax_weight_max": _FP8_SUBSTRATE_STATE.get("amax_weight_max"),
        "scale_input": _FP8_SUBSTRATE_STATE.get("scale_input"),
        "scale_weight": _FP8_SUBSTRATE_STATE.get("scale_weight"),
        "amax_reduction_ms_total": float(_FP8_SUBSTRATE_STATE.get("amax_reduction_ms_total", 0.0) or 0.0),
        "moe_amax_reduction_ms_total": float(_FP8_SUBSTRATE_STATE.get("moe_amax_reduction_ms_total", 0.0) or 0.0),
        "delayed_scale_enabled": _DELAYED_SCALE,
        "scale_refresh_interval": _SCALE_REFRESH_INTERVAL if _DELAYED_SCALE else None,
        "modules": modules,
    }


def write_fp8_substrate_json(path: str | Path) -> None:
    Path(path).write_text(json.dumps(fp8_substrate_report(), indent=2, sort_keys=True), encoding="utf-8")


def _quant(x: torch.Tensor, fmax: float, dt: torch.dtype):
    scale = (x.detach().abs().amax() / fmax).clamp(min=1e-8)
    return (x / scale).clamp(-fmax, fmax).to(dt), scale.float().view(1, 1)



class _ModuleScaleState:
    """Per-Fp8Linear temporal amax history for delayed-scale reuse."""

    __slots__ = ("_history", "_call_count", "_scale_x", "_scale_w", "_age")

    def __init__(self) -> None:
        self._history: list[tuple[float, float]] = []
        self._call_count = 0
        self._scale_x: float = 0.0
        self._scale_w: float = 0.0
        self._age: int = 0  # calls since last refresh

    def is_fresh(self) -> bool:
        return self._call_count > 0 and self._age < _SCALE_REFRESH_INTERVAL

    def refresh(self, input_amax: float, weight_amax: float) -> tuple[float, float]:
        self._history.append((input_amax, weight_amax))
        if len(self._history) > _SCALE_HISTORY_LEN:
            self._history = self._history[-_SCALE_HISTORY_LEN:]
        # Store as GEMM scale (= amax / E4M3_MAX), consistent with _quant().
        # max(history) is conservative: spike headroom prevents overflow between refreshes.
        # Decay lets scale drift back down after a spike rather than staying pinned forever.
        # max(..., fresh) ensures scale never falls below the current measurement.
        fresh_sx = max(h[0] for h in self._history) / _E4M3_MAX
        fresh_sw = max(h[1] for h in self._history) / _E4M3_MAX
        if self._scale_x > 0:
            self._scale_x = max(fresh_sx, self._scale_x * _SCALE_DECAY)
            self._scale_w = max(fresh_sw, self._scale_w * _SCALE_DECAY)
        else:
            self._scale_x = fresh_sx
            self._scale_w = fresh_sw
        self._call_count += 1
        self._age = 0
        return self._scale_x, self._scale_w

    def advance(self) -> tuple[float, float]:
        self._age += 1
        return self._scale_x, self._scale_w


class _Fp8MM(torch.autograd.Function):
    """y = x @ wᵀ in fp8 (e4m3 fwd, e5m2 bwd), fp32 accumulate, bf16 out."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None,
                sx_pre: torch.Tensor | None, sw_pre: torch.Tensor | None):
        if sx_pre is not None and sw_pre is not None:
            # Delayed-scale path: sx_pre = amax/E4M3_MAX (same as _quant() scale).
            # x / sx_pre = x * E4M3_MAX/amax → max value maps to E4M3_MAX (full range).
            xq = (x / sx_pre.item()).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
            wq = (w / sw_pre.item()).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
            sx, sw = sx_pre, sw_pre
        else:
            xq, sx = _quant(x, _E4M3_MAX, torch.float8_e4m3fn)   # [M,K]
            wq, sw = _quant(w, _E4M3_MAX, torch.float8_e4m3fn)   # [N,K]
        out = torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=sw,
                               out_dtype=torch.bfloat16)      # [M,N]
        if b is not None:
            out = out + b
        ctx.save_for_backward(xq, wq, sx, sw)
        return out

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        xq, wq, sx, sw = ctx.saved_tensors
        g = g.contiguous()
        gq, sg = _quant(g, _E5M2_MAX, torch.float8_e5m2)      # [M,N]
        # grad_x[M,K] = g[M,N] @ w[N,K]   (B must be column-major [N,K])
        bx = wq.t().contiguous().t()
        grad_x = torch._scaled_mm(gq, bx, scale_a=sg, scale_b=sw,
                                  out_dtype=torch.bfloat16)
        # grad_w[N,K] = gᵀ[N,M] @ x[M,K]  (A row-major [N,M], B col-major [M,K])
        a = gq.t().contiguous()
        bw = xq.t().contiguous().t()
        grad_w = torch._scaled_mm(a, bw, scale_a=sg, scale_b=sx,
                                  out_dtype=torch.bfloat16)
        return grad_x, grad_w, None, None, None


class Fp8Linear(nn.Linear):
    """Drop-in nn.Linear running its GEMM in fp8 when shapes qualify.

    Falls back to a bf16 matmul for any shape _scaled_mm rejects (dims not a
    multiple of 16), so it is always safe to swap in.

    With IDA_FP8_DELAYED_SCALE=1: per-module amax history is maintained and
    scales are refreshed every IDA_FP8_SCALE_REFRESH_INTERVAL forward calls
    instead of every call. The inner _quant() amax reductions are skipped on
    non-refresh steps, cutting ~87% of amax overhead at 16-step refresh cadence.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scale_state: _ModuleScaleState | None = (
            _ModuleScaleState() if _DELAYED_SCALE else None
        )

    @classmethod
    def of(cls, lin: nn.Linear) -> "Fp8Linear":
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                device=lin.weight.device, dtype=lin.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(lin.weight)
            if lin.bias is not None:
                m.bias.copy_(lin.bias)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        x2 = x.reshape(-1, sh[-1]) if x.dim() > 2 else x
        m, k = x2.shape
        n = self.out_features
        module_name = str(getattr(self, "_fp8_module_name", self.__class__.__name__))

        ss = self._scale_state
        if ss is not None and ss.is_fresh():
            # Delayed-scale path: no amax reduction this step.
            scale_x, scale_w = ss.advance()
            input_amax = scale_x * _E4M3_MAX
            weight_amax = scale_w * _E4M3_MAX
            _amax_ms = 0.0
            sx_pre = x2.new_full((1, 1), scale_x, dtype=torch.float32)
            sw_pre = x2.new_full((1, 1), scale_w, dtype=torch.float32)
        else:
            _amax_t0 = _time_fp8.perf_counter()
            input_amax = _safe_float(x2.detach().abs().amax()) if x2.numel() else 0.0
            weight_amax = _safe_float(self.weight.detach().abs().amax()) if self.weight.numel() else 0.0
            _amax_ms = (_time_fp8.perf_counter() - _amax_t0) * 1000.0
            if ss is not None:
                scale_x, scale_w = ss.refresh(input_amax, weight_amax)
                sx_pre = x2.new_full((1, 1), scale_x, dtype=torch.float32)
                sw_pre = x2.new_full((1, 1), scale_w, dtype=torch.float32)
            else:
                sx_pre, sw_pre = None, None

        _FP8_SUBSTRATE_STATE["amax_reduction_ms_total"] = float(_FP8_SUBSTRATE_STATE.get("amax_reduction_ms_total", 0.0)) + _amax_ms
        if "experts" in module_name:
            _FP8_SUBSTRATE_STATE["moe_amax_reduction_ms_total"] = float(_FP8_SUBSTRATE_STATE.get("moe_amax_reduction_ms_total", 0.0)) + _amax_ms

        if x2.device.type != "cuda" or self.weight.device.type != "cuda":
            _record_fallback(
                module_name,
                "dtype_or_device",
                m,
                k,
                n,
                input_amax=input_amax,
                weight_amax=weight_amax,
            )
            return F.linear(x, self.weight, self.bias)
        if m % 16 == 0 and k % 16 == 0 and n % 16 == 0:
            try:
                scale_input = max(float(input_amax or 0.0) / _E4M3_MAX, 1e-8)
                scale_weight = max(float(weight_amax or 0.0) / _E4M3_MAX, 1e-8)
                out = _Fp8MM.apply(x2.contiguous(), self.weight, self.bias, sx_pre, sw_pre)
                _record_fp8_success(
                    module_name,
                    m,
                    k,
                    n,
                    input_amax=input_amax,
                    weight_amax=weight_amax,
                    scale_input=scale_input,
                    scale_weight=scale_weight,
                )
                return out.reshape(*sh[:-1], n)
            except Exception:
                _record_fallback(
                    module_name,
                    "scaled_mm_exception",
                    m,
                    k,
                    n,
                    input_amax=input_amax,
                    weight_amax=weight_amax,
                )
        else:
            _record_fallback(
                module_name,
                "shape_not_multiple",
                m,
                k,
                n,
                input_amax=input_amax,
                weight_amax=weight_amax,
            )
        return F.linear(x, self.weight, self.bias)


def swap_to_fp8(model: nn.Module, verbose: bool = True) -> int:
    """Replace bulk-GEMM nn.Linear modules with Fp8Linear in place.

    Targets expert-FFN and block-attention linears; leaves router, gates,
    norms, embeddings, and heads in bf16. Returns the number swapped.
    """
    reset_fp8_substrate_state()
    n = 0
    for name, mod in model.named_modules():
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and not isinstance(ch, Fp8Linear):
                full = f"{name}.{cn}"
                if any(k in full for k in _INCLUDE) and not any(
                    k in full for k in _EXCLUDE
                ):
                    repl = Fp8Linear.of(ch)
                    repl._fp8_module_name = full  # type: ignore[attr-defined]
                    setattr(mod, cn, repl)
                    n += 1
    _FP8_SUBSTRATE_STATE["active"] = n > 0
    _FP8_SUBSTRATE_STATE["swapped_linears"] = n
    if verbose:
        print(f"[fp8] swapped {n} linears to Fp8Linear "
              f"(expert FFN + block attention; router/gates/norms/heads stay bf16)",
              flush=True)
    return n


# Strings that must NOT appear in a module path for targeted FP8 swap.
# Covers PEFT LoRA adapter matrices and all sensitive surfaces that must stay bf16.
_LORA_TARGETED_EXCLUDE = (
    "lora_A", "lora_B", "lora_dropout",
    "embed", "norm", "ln_", "head", "wpe", "wte",
)


def swap_to_fp8_targeted(
    model: nn.Module,
    target_module_names: list[str],
    *,
    exclude_names: tuple[str, ...] = _LORA_TARGETED_EXCLUDE,
    verbose: bool = True,
) -> int:
    """FP8 swap for external-base + PEFT LoRA models (e.g. Pythia + LoRA).

    Precision boundary:
      fp8  — frozen base projections (e.g. query_key_value, dense_h_to_4h)
      bf16 — LoRA A/B adapters, norms, embeddings, heads, loss surfaces

    For PEFT LoRA modules, the inner ``base_layer`` (the frozen nn.Linear) is
    swapped; ``lora_A`` and ``lora_B`` (small trainable delta matrices) stay
    BF16 — this is where new family behavior lives and FP8 error would corrupt
    the learned delta.

    ``target_module_names`` — list of substrings; a module qualifies when its
    full dotted path contains *any* of these AND none of ``exclude_names``.
    Pass the same list as LoRA's ``target_modules``.
    """
    reset_fp8_substrate_state()
    n = 0
    for name, mod in model.named_modules():
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and not isinstance(ch, Fp8Linear):
                full = f"{name}.{cn}"
                if (
                    any(t in full for t in target_module_names)
                    and not any(e in full for e in exclude_names)
                ):
                    repl = Fp8Linear.of(ch)
                    repl._fp8_module_name = full  # type: ignore[attr-defined]
                    setattr(mod, cn, repl)
                    n += 1
    _FP8_SUBSTRATE_STATE["active"] = n > 0
    _FP8_SUBSTRATE_STATE["swapped_linears"] = n
    if verbose:
        tgt = ", ".join(target_module_names)
        print(
            f"[fp8] swapped {n} base-layer linears to Fp8Linear "
            f"(targets: {tgt}; LoRA A/B + norms/embeds/heads stay bf16)",
            flush=True,
        )
    return n
