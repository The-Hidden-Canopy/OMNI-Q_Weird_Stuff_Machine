// Portions derived from The Hidden Canopy LLC — IDA-TRAIN-V2 (native/kernels/mxfp8.cu). Used with permission.
#pragma once

// Native MXFP8 master contract used by the Blackwell NVFP4 trainer slice.
// Payload values are finite NVIDIA E4M3 bytes and one UE8M0 power-of-two
// scale is stored for every 32-value block. The helpers are header-local so
// storage, optimizer, and NVFP4 refresh kernels share exactly one decode rule.

#include <cstdint>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ida_native/fp8_e4m3.hpp"

namespace ida_native {

inline constexpr int kMxfp8BlockSize = 32;
inline constexpr float kMxfp8Fp8Max = 448.0f;

#if defined(__CUDACC__)
#define IDA_MXFP8_D __device__ __forceinline__
#else
#define IDA_MXFP8_D inline
#endif

IDA_MXFP8_D std::uint8_t mxfp8_encode_ue8m0(float value) {
    if (!(value > 0.0f) || !isfinite(value)) return 0;
    int exponent = static_cast<int>(ceilf(log2f(value)));
    exponent = max(-126, min(127, exponent));
    return static_cast<std::uint8_t>(exponent + 127);
}

IDA_MXFP8_D float mxfp8_decode_ue8m0(std::uint8_t raw) {
    return raw == 0 ? 0.0f : scalbnf(1.0f, static_cast<int>(raw) - 127);
}

IDA_MXFP8_D float mxfp8_safe_scale(std::uint8_t raw) {
    return fmaxf(mxfp8_decode_ue8m0(raw), 1.0e-30f);
}

IDA_MXFP8_D std::uint8_t mxfp8_scale_for_amax(float amax) {
    return mxfp8_encode_ue8m0(amax / kMxfp8Fp8Max);
}

IDA_MXFP8_D float mxfp8_decode_value(
    std::uint8_t payload, std::uint8_t scale_code
) {
    return fp8_e4m3::unpack(payload) * mxfp8_safe_scale(scale_code);
}

// Strict Omni entry points.  The unseeded helpers remain available to legacy
// checkpoint/export code; the typed Omni path must call these variants so a
// refresh or gradient fold can never silently revert to round-to-nearest.
void mxfp8_pack_from_bf16_sr(
    const __nv_bfloat16* source,
    std::uint8_t* payload,
    std::uint8_t* scales,
    std::size_t elements,
    cudaStream_t stream,
    unsigned sr_seed
);

void mxfp8_gradient_accumulate_bf16_sr(
    const __nv_bfloat16* scratch,
    std::uint8_t* payload,
    std::uint8_t* scales,
    __nv_bfloat16* residual,
    std::size_t elements,
    cudaStream_t stream,
    unsigned sr_seed
);

void mxfp8_gradient_scale_sr(
    std::uint8_t* payload,
    std::uint8_t* scales,
    __nv_bfloat16* residual,
    std::size_t elements,
    float scale,
    cudaStream_t stream,
    unsigned sr_seed
);

void mxfp8_gradient_unpack_to_bf16_sr(
    const std::uint8_t* payload,
    const std::uint8_t* scales,
    const __nv_bfloat16* residual,
    __nv_bfloat16* output,
    std::size_t elements,
    cudaStream_t stream,
    unsigned sr_seed
);

#undef IDA_MXFP8_D

}  // namespace ida_native
