// Portions derived from The Hidden Canopy LLC — IDA-TRAIN-V2 (native/kernels/mxfp8.cu). Used with permission.
#include "ida_native/kernels.hpp"

#include "ida_native/cuda_check.hpp"
#include "ida_native/fp8_e4m3.hpp"
#include "ida_native/mxfp8.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace ida_native {

namespace {

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16_sr(
    float value,
    unsigned random_bits
) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return __float2bfloat16(value);
    }
    bits += random_bits & 0xffffu;
    __nv_bfloat16_raw raw;
    raw.x = static_cast<unsigned short>(bits >> 16);
    return __nv_bfloat16(raw);
}

__device__ __forceinline__ unsigned mix_seed(
    std::size_t index,
    unsigned seed
) {
    unsigned h = static_cast<unsigned>(index) * 2654435761u ^ seed;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ std::uint8_t fp8_e4m3_sr(
    float value, unsigned random_bits
) {
    const std::uint8_t sign = signbit(value) != 0 ? 0x80u : 0u;
    const float magnitude = fabsf(value);
    if (!isfinite(value)) return static_cast<std::uint8_t>(sign | 0x7fu);
    if (magnitude == 0.0f) return sign;
    if (magnitude >= kMxfp8Fp8Max)
        return static_cast<std::uint8_t>(sign | 0x7eu);
    const std::uint8_t nearest = fp8_e4m3::pack(magnitude).bits;
    const float nearest_value = fp8_e4m3::unpack(nearest);
    const int lower = nearest_value <= magnitude
        ? static_cast<int>(nearest)
        : (nearest > 0u ? static_cast<int>(nearest) - 1 : 0);
    const int upper = nearest_value <= magnitude
        ? (nearest < 0x7eu ? static_cast<int>(nearest) + 1
                           : static_cast<int>(nearest))
        : static_cast<int>(nearest);
    int chosen = lower;
    if (upper > lower) {
        const float lo = fp8_e4m3::unpack(static_cast<std::uint8_t>(lower));
        const float hi = fp8_e4m3::unpack(static_cast<std::uint8_t>(upper));
        const float fraction = (magnitude - lo) / (hi - lo);
        const float draw = static_cast<float>(random_bits & 0xffffu) *
                           (1.0f / 65536.0f);
        if (draw < fraction) chosen = upper;
    }
    return static_cast<std::uint8_t>(sign | static_cast<std::uint8_t>(chosen));
}

__device__ __forceinline__ float uniform_hash_value(
    std::size_t index,
    float scale,
    std::uint64_t seed
) {
    std::uint64_t s = seed ^
        (static_cast<std::uint64_t>(index) * 6364136223846793005ULL +
         1442695040888963407ULL);
    s ^= s >> 33;
    s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33;
    s *= 0xc4ceb9fe1a85ec53ULL;
    s ^= s >> 33;
    // Convert the high 32 bits to a deterministic [0, 1] variate without
    // relying on INT64_MAX (which is not consistently exposed by the CUDA
    // device headers).  Mapping that variate to [-1, 1] keeps initialization
    // symmetric while remaining valid for every nvcc host toolchain.
    const std::uint32_t high = static_cast<std::uint32_t>(s >> 32);
    const float v = (static_cast<float>(high) * (1.0f / 4294967295.0f)) *
        2.0f - 1.0f;
    return v * scale;
}

__device__ __forceinline__ float warp_amax(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__global__ void k_mxfp8_init_uniform(
    std::uint8_t* payload,
    std::uint8_t* scales,
    std::size_t n,
    float scale,
    std::uint64_t seed
) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t index = block * kMxfp8BlockSize +
        static_cast<std::size_t>(lane);
    const float value = index < n ? uniform_hash_value(index, scale, seed) : 0.0f;
    const float amax = warp_amax(fabsf(value));
    const std::uint8_t scale_code = mxfp8_scale_for_amax(amax);
    const float applied = mxfp8_safe_scale(scale_code);
    if (lane == 0) scales[block] = scale_code;
    if (index < n) {
        payload[index] = fp8_e4m3::pack(value / applied).bits;
    }
}

__global__ void k_mxfp8_pack_from_bf16(
    const __nv_bfloat16* source,
    std::uint8_t* payload,
    std::uint8_t* scales,
    std::size_t n,
    unsigned sr_seed
) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t index = block * kMxfp8BlockSize +
        static_cast<std::size_t>(lane);
    const float value = index < n ? __bfloat162float(source[index]) : 0.0f;
    const float amax = warp_amax(fabsf(value));
    const std::uint8_t scale_code = mxfp8_scale_for_amax(amax);
    const float applied = mxfp8_safe_scale(scale_code);
    if (lane == 0) scales[block] = scale_code;
    if (index < n) {
        payload[index] = sr_seed
            ? fp8_e4m3_sr(value / applied, mix_seed(index, sr_seed))
            : fp8_e4m3::pack(value / applied).bits;
    }
}

__global__ void k_mxfp8_unpack_to_bf16(
    const std::uint8_t* payload,
    const std::uint8_t* scales,
    __nv_bfloat16* output,
    std::size_t n,
    unsigned sr_seed
) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (index >= n) return;
    const float value = mxfp8_decode_value(
        payload[index], scales[index / kMxfp8BlockSize]);
    output[index] = sr_seed
        ? f32_to_bf16_sr(value, mix_seed(index, sr_seed))
        : __float2bfloat16(value);
}

__global__ void k_lion_mxfp8(
    std::uint8_t* payload,
    std::uint8_t* scales,
    __nv_bfloat16* momentum,
    const __nv_bfloat16* gradient,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    float wd,
    std::uint32_t* saturation_count,
    unsigned sr_seed
) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t index = block * kMxfp8BlockSize +
        static_cast<std::size_t>(lane);
    const bool valid = index < n;
    const float old_scale = mxfp8_safe_scale(scales[block]);
    const float weight = valid
        ? fp8_e4m3::unpack(payload[index]) * old_scale : 0.0f;
    const float old_momentum = valid ? __bfloat162float(momentum[index]) : 0.0f;
    const float grad = valid ? __bfloat162float(gradient[index]) : 0.0f;
    const float direction = beta1 * old_momentum + (1.0f - beta1) * grad;
    const float next_momentum = beta2 * old_momentum + (1.0f - beta2) * grad;
    const float sign = direction > 0.0f ? 1.0f : direction < 0.0f ? -1.0f : 0.0f;
    const float next_weight = valid
        ? weight - lr * (sign + wd * weight) : 0.0f;

    const float amax = warp_amax(valid ? fabsf(next_weight) : 0.0f);
    const std::uint8_t new_scale_code = mxfp8_scale_for_amax(amax);
    const float new_scale = mxfp8_safe_scale(new_scale_code);
    if (lane == 0) scales[block] = new_scale_code;
    if (!valid) return;

    const float normalized = next_weight / new_scale;
    if (saturation_count && fabsf(normalized) > kMxfp8Fp8Max) {
        atomicAdd(saturation_count, 1u);
    }
    const unsigned h = mix_seed(index, sr_seed);
    payload[index] = sr_seed
        ? fp8_e4m3_sr(normalized, h ^ 0x243f6a88u)
        : fp8_e4m3::pack(normalized).bits;
    momentum[index] = f32_to_bf16_sr(next_momentum, h ^ 0x9e3779b9u);
}

__device__ __forceinline__ float mxfp8_gradient_value(
    const std::uint8_t* payload,
    const std::uint8_t* scales,
    const __nv_bfloat16* residual,
    std::size_t index
) {
    return mxfp8_decode_value(payload[index], scales[index / kMxfp8BlockSize]) +
        __bfloat162float(residual[index]);
}

// One warp owns one MXFP8 block. The BF16 scratch is folded into the
// persistent payload plus a BF16 residual, preserving a bounded residual for
// values that cannot be represented by the block's E4M3 scale.
__global__ void k_mxfp8_gradient_accumulate_bf16(
    const __nv_bfloat16* scratch,
    std::uint8_t* payload,
    std::uint8_t* scales,
    __nv_bfloat16* residual,
    std::size_t n,
    unsigned sr_seed
) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t index = block * kMxfp8BlockSize +
        static_cast<std::size_t>(lane);
    const bool valid = index < n;
    const float current = valid
        ? mxfp8_gradient_value(payload, scales, residual, index) : 0.0f;
    const float target = valid ? current + __bfloat162float(scratch[index]) : 0.0f;
    const float finite_amax = valid && isfinite(target) ? fabsf(target) : 0.0f;
    const float amax = warp_amax(finite_amax);
    const std::uint8_t scale_code = mxfp8_scale_for_amax(amax);
    const float applied = mxfp8_safe_scale(scale_code);
    if (lane == 0) scales[block] = scale_code;
    if (!valid) return;
    const std::uint8_t code = sr_seed
        ? fp8_e4m3_sr(target / applied, mix_seed(index, sr_seed))
        : fp8_e4m3::pack(target / applied).bits;
    payload[index] = code;
    const float decoded = mxfp8_decode_value(code, scale_code);
    residual[index] = sr_seed
        ? f32_to_bf16_sr(target - decoded, mix_seed(index, sr_seed ^ 0x517cc1b7u))
        : __float2bfloat16(target - decoded);
}

__global__ void k_mxfp8_gradient_unpack_to_bf16(
    const std::uint8_t* payload,
    const std::uint8_t* scales,
    const __nv_bfloat16* residual,
    __nv_bfloat16* output,
    std::size_t n,
    unsigned sr_seed
) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (index >= n) return;
    const float value = mxfp8_gradient_value(payload, scales, residual, index);
    output[index] = sr_seed
        ? f32_to_bf16_sr(value, mix_seed(index, sr_seed))
        : __float2bfloat16(value);
}

__global__ void k_mxfp8_gradient_scale(
    std::uint8_t* payload,
    std::uint8_t* scales,
    __nv_bfloat16* residual,
    std::size_t n,
    float multiplier,
    unsigned sr_seed
) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t index = block * kMxfp8BlockSize +
        static_cast<std::size_t>(lane);
    const bool valid = index < n;
    const float target = valid
        ? mxfp8_gradient_value(payload, scales, residual, index) * multiplier : 0.0f;
    const float finite_amax = valid && isfinite(target) ? fabsf(target) : 0.0f;
    const float amax = warp_amax(finite_amax);
    const std::uint8_t scale_code = mxfp8_scale_for_amax(amax);
    const float applied = mxfp8_safe_scale(scale_code);
    if (lane == 0) scales[block] = scale_code;
    if (!valid) return;
    const std::uint8_t code = sr_seed
        ? fp8_e4m3_sr(target / applied, mix_seed(index, sr_seed))
        : fp8_e4m3::pack(target / applied).bits;
    payload[index] = code;
    const float residual_value = target - mxfp8_decode_value(code, scale_code);
    residual[index] = sr_seed
        ? f32_to_bf16_sr(residual_value,
                         mix_seed(index, sr_seed ^ 0x6d2b79f5u))
        : __float2bfloat16(residual_value);
}

__global__ void k_mxfp8_gradient_sq_sum_partial(
    const std::uint8_t* payload,
    const std::uint8_t* scales,
    const __nv_bfloat16* residual,
    std::size_t n,
    float* partials
) {
    __shared__ float sm[256 / 32];
    float sum = 0.0f;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
         threadIdx.x;
         index < n;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float value = mxfp8_gradient_value(payload, scales, residual, index);
        sum += value * value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int warp = 0; warp < blockDim.x / 32; ++warp) total += sm[warp];
        partials[blockIdx.x] = total;
    }
}

__global__ void k_mxfp8_gradient_sq_sum_combine(
    const float* partials, int count, float* acc
) {
    float total = 0.0f;
    for (int i = 0; i < count; ++i) total += partials[i];
    *acc += total;
}

}  // namespace

void mxfp8_init_uniform(
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    std::size_t n,
    float scale,
    std::uint64_t seed,
    cudaStream_t stream
) {
    if (n == 0) return;
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_mxfp8_init_uniform<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_payload, d_scales, n, scale, seed);
}

void mxfp8_pack_from_bf16(
    const __nv_bfloat16* d_source,
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    std::size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_mxfp8_pack_from_bf16<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_source, d_payload, d_scales, n, 0u);
}

void mxfp8_pack_from_bf16_sr(
    const __nv_bfloat16* d_source,
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    std::size_t n,
    cudaStream_t stream,
    unsigned sr_seed
) {
    if (n == 0) return;
    if (!d_source || !d_payload || !d_scales || sr_seed == 0u) {
        throw std::invalid_argument(
            "seeded MXFP8 pack requires non-null buffers and a nonzero seed");
    }
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_mxfp8_pack_from_bf16<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_source, d_payload, d_scales, n, sr_seed);
    IDA_CUDA_CHECK(cudaGetLastError());
}

void mxfp8_unpack_to_bf16(
    const std::uint8_t* d_payload,
    const std::uint8_t* d_scales,
    __nv_bfloat16* d_output,
    std::size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    k_mxfp8_unpack_to_bf16<<<blocks, 256, 0, stream>>>(
        d_payload, d_scales, d_output, n, 0u);
}

void lion_step_mxfp8(
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    __nv_bfloat16* d_momentum,
    const __nv_bfloat16* d_grad,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    float wd,
    cudaStream_t stream,
    std::uint32_t* d_saturation_count,
    unsigned sr_seed
) {
    if (n == 0) return;
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_lion_mxfp8<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_payload, d_scales, d_momentum, d_grad, n, lr, beta1, beta2, wd,
        d_saturation_count, sr_seed);
}

void mxfp8_gradient_accumulate_bf16(
    const __nv_bfloat16* d_scratch,
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    __nv_bfloat16* d_residual,
    std::size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    if (!d_scratch || !d_payload || !d_scales || !d_residual) {
        throw std::invalid_argument(
            "mxfp8 gradient accumulation received a null buffer");
    }
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_mxfp8_gradient_accumulate_bf16<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_scratch, d_payload, d_scales, d_residual, n, 0u);
    IDA_CUDA_CHECK(cudaGetLastError());
}

void mxfp8_gradient_accumulate_bf16_sr(
    const __nv_bfloat16* d_scratch,
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    __nv_bfloat16* d_residual,
    std::size_t n,
    cudaStream_t stream,
    unsigned sr_seed
) {
    if (n == 0) return;
    if (!d_scratch || !d_payload || !d_scales || !d_residual ||
        sr_seed == 0u) {
        throw std::invalid_argument(
            "seeded MXFP8 gradient fold requires non-null buffers and a nonzero seed");
    }
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_mxfp8_gradient_accumulate_bf16<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_scratch, d_payload, d_scales, d_residual, n, sr_seed);
    IDA_CUDA_CHECK(cudaGetLastError());
}

void mxfp8_gradient_zero(
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    __nv_bfloat16* d_residual,
    std::size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    if (!d_payload || !d_scales || !d_residual) {
        throw std::invalid_argument("mxfp8 gradient zero received a null buffer");
    }
    IDA_CUDA_CHECK(cudaMemsetAsync(d_payload, 0, n, stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(
        d_scales, 0, (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize, stream));
    IDA_CUDA_CHECK(cudaMemsetAsync(
        d_residual, 0, n * sizeof(__nv_bfloat16), stream));
}

void mxfp8_gradient_unpack_to_bf16(
    const std::uint8_t* d_payload,
    const std::uint8_t* d_scales,
    const __nv_bfloat16* d_residual,
    __nv_bfloat16* d_output,
    std::size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    if (!d_payload || !d_scales || !d_residual || !d_output) {
        throw std::invalid_argument("mxfp8 gradient unpack received a null buffer");
    }
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    k_mxfp8_gradient_unpack_to_bf16<<<blocks, 256, 0, stream>>>(
        d_payload, d_scales, d_residual, d_output, n, 0u);
    IDA_CUDA_CHECK(cudaGetLastError());
}

void mxfp8_gradient_unpack_to_bf16_sr(
    const std::uint8_t* d_payload,
    const std::uint8_t* d_scales,
    const __nv_bfloat16* d_residual,
    __nv_bfloat16* d_output,
    std::size_t n,
    cudaStream_t stream,
    unsigned sr_seed
) {
    if (n == 0) return;
    if (!d_payload || !d_scales || !d_residual || !d_output || sr_seed == 0u) {
        throw std::invalid_argument(
            "seeded MXFP8 gradient unpack requires non-null buffers and a nonzero seed");
    }
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    k_mxfp8_gradient_unpack_to_bf16<<<blocks, 256, 0, stream>>>(
        d_payload, d_scales, d_residual, d_output, n, sr_seed);
    IDA_CUDA_CHECK(cudaGetLastError());
}

void mxfp8_gradient_sq_sum(
    const std::uint8_t* d_payload,
    const std::uint8_t* d_scales,
    const __nv_bfloat16* d_residual,
    std::size_t n,
    float* d_acc,
    cudaStream_t stream
) {
    if (n == 0) return;
    if (!d_payload || !d_scales || !d_residual || !d_acc) {
        throw std::invalid_argument("mxfp8 gradient norm received a null buffer");
    }
    const unsigned blocks = static_cast<unsigned>(
        std::min<std::size_t>((n + 255) / 256, 1024));
    float* d_partials = nullptr;
    IDA_CUDA_CHECK(cudaMallocAsync(
        &d_partials, static_cast<std::size_t>(blocks) * sizeof(float), stream));
    k_mxfp8_gradient_sq_sum_partial<<<blocks, 256, 0, stream>>>(
        d_payload, d_scales, d_residual, n, d_partials);
    IDA_CUDA_CHECK(cudaGetLastError());
    k_mxfp8_gradient_sq_sum_combine<<<1, 1, 0, stream>>>(
        d_partials, static_cast<int>(blocks), d_acc);
    IDA_CUDA_CHECK(cudaGetLastError());
    IDA_CUDA_CHECK(cudaFreeAsync(d_partials, stream));
}

void mxfp8_gradient_scale(
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    __nv_bfloat16* d_residual,
    std::size_t n,
    float scale,
    cudaStream_t stream
) {
    if (n == 0) return;
    if (!d_payload || !d_scales || !d_residual) {
        throw std::invalid_argument("mxfp8 gradient scale received a null buffer");
    }
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_mxfp8_gradient_scale<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_payload, d_scales, d_residual, n, scale, 0u);
    IDA_CUDA_CHECK(cudaGetLastError());
}

void mxfp8_gradient_scale_sr(
    std::uint8_t* d_payload,
    std::uint8_t* d_scales,
    __nv_bfloat16* d_residual,
    std::size_t n,
    float scale,
    cudaStream_t stream,
    unsigned sr_seed
) {
    if (n == 0) return;
    if (!d_payload || !d_scales || !d_residual || sr_seed == 0u) {
        throw std::invalid_argument(
            "seeded MXFP8 gradient scale requires non-null buffers and a nonzero seed");
    }
    const unsigned blocks = static_cast<unsigned>(
        (n + kMxfp8BlockSize - 1) / kMxfp8BlockSize);
    k_mxfp8_gradient_scale<<<blocks, kMxfp8BlockSize, 0, stream>>>(
        d_payload, d_scales, d_residual, n, scale, sr_seed);
    IDA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ida_native
