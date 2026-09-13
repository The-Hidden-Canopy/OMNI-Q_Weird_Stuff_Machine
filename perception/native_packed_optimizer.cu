// Fused MXFP8 Lion step for the OMNI-Q PyTorch path.
//
// The kernel shape and MXFP8/Lion semantics are derived from
// integrations/qualcomm/lowbit/vendor/ida_train_v2/mxfp8.cu, which vendors
// IDA-TRAIN-V2 native/kernels/mxfp8.cu with permission. This wrapper adds the
// PyTorch tensor boundary and writes the floating compute view in the same
// pass, so the Python optimizer does not launch separate decode/update/pack
// operations for each parameter.

#include <torch/extension.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <pybind11/stl.h>
#include <vector>

#include "integrations/qualcomm/lowbit/vendor/ida_train_v2/fp8_e4m3.hpp"
#include "integrations/qualcomm/lowbit/vendor/ida_train_v2/mxfp8.cuh"

namespace {

struct LionTensorDesc {
    std::uint8_t* payload;
    std::uint8_t* scales;
    __nv_bfloat16* momentum;
    const float* gradient;
    float* compute;
    std::size_t n;
    float lr;
    float beta1;
    float beta2;
    float wd;
    std::size_t block_base;
};

constexpr std::size_t kMaxLionTensorDescs = 512;
__constant__ LionTensorDesc c_lion_descriptors[kMaxLionTensorDescs];

template <typename T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

__global__ void k_lion_mxfp8(
    std::uint8_t* payload,
    std::uint8_t* scales,
    __nv_bfloat16* momentum,
    const float* gradient,
    float* compute,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    float wd
) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t index = block * ida_native::kMxfp8BlockSize +
        static_cast<std::size_t>(lane);
    const bool valid = index < n;
    // The scale is block-shared.  Load and decode it once, then broadcast it
    // instead of making all 32 lanes repeat the same byte load and conversion.
    float old_scale = lane == 0
        ? ida_native::mxfp8_safe_scale(scales[block]) : 0.0f;
    old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
    const float weight = valid
        ? ida_native::fp8_e4m3::unpack(payload[index]) * old_scale : 0.0f;
    const float old_momentum = valid ? __bfloat162float(momentum[index]) : 0.0f;
    const float grad = valid ? gradient[index] : 0.0f;
    const float direction = beta1 * old_momentum + (1.0f - beta1) * grad;
    const float next_momentum = beta2 * old_momentum + (1.0f - beta2) * grad;
    const float sign = direction > 0.0f ? 1.0f : direction < 0.0f ? -1.0f : 0.0f;
    const float next_weight = valid
        ? weight - lr * (sign + wd * weight) : 0.0f;

    float amax = fabsf(next_weight);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    // Only lane 0 needs the log2/ceil scale selection.  Broadcast both the
    // selected code and its decoded scale to the rest of the warp.
    int new_scale_code_i = lane == 0
        ? static_cast<int>(ida_native::mxfp8_scale_for_amax(amax)) : 0;
    new_scale_code_i = __shfl_sync(0xffffffffu, new_scale_code_i, 0);
    const std::uint8_t new_scale_code =
        static_cast<std::uint8_t>(new_scale_code_i);
    float new_scale = lane == 0
        ? ida_native::mxfp8_safe_scale(new_scale_code) : 0.0f;
    new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
    if (lane == 0) scales[block] = new_scale_code;
    if (!valid) return;

    const std::uint8_t code = ida_native::fp8_e4m3::pack(
        next_weight / new_scale).bits;
    payload[index] = code;
    momentum[index] = __float2bfloat16(next_momentum);
    compute[index] = ida_native::fp8_e4m3::unpack(code) * new_scale;
}

// One launch owns all parameter tensors in a param group.  The host supplies
// a compact block-to-tensor map because the tensors are not one flat storage
// allocation (the packed state must remain checkpoint-compatible per tensor).
__global__ void k_lion_mxfp8_multi(
    std::size_t descriptor_count,
    std::size_t total_blocks
) {
    const std::size_t global_block = static_cast<std::size_t>(blockIdx.x);
    if (global_block >= total_blocks) return;
    std::size_t lower = 0;
    std::size_t upper = descriptor_count;
    while (lower + 1 < upper) {
        const std::size_t middle = (lower + upper) / 2;
        if (global_block < c_lion_descriptors[middle].block_base)
            upper = middle;
        else
            lower = middle;
    }
    const LionTensorDesc tensor = c_lion_descriptors[lower];
    const std::size_t block = global_block - tensor.block_base;
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t index = block * ida_native::kMxfp8BlockSize +
        static_cast<std::size_t>(lane);
    const bool valid = index < tensor.n;

    float old_scale = lane == 0
        ? ida_native::mxfp8_safe_scale(tensor.scales[block]) : 0.0f;
    old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
    const float weight = valid
        ? ida_native::fp8_e4m3::unpack(tensor.payload[index]) * old_scale : 0.0f;
    const float old_momentum = valid
        ? __bfloat162float(tensor.momentum[index]) : 0.0f;
    const float grad = valid ? tensor.gradient[index] : 0.0f;
    const float direction = tensor.beta1 * old_momentum +
        (1.0f - tensor.beta1) * grad;
    const float next_momentum = tensor.beta2 * old_momentum +
        (1.0f - tensor.beta2) * grad;
    const float sign = direction > 0.0f ? 1.0f : direction < 0.0f ? -1.0f : 0.0f;
    const float next_weight = valid
        ? weight - tensor.lr * (sign + tensor.wd * weight) : 0.0f;

    float amax = fabsf(next_weight);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    int new_scale_code_i = lane == 0
        ? static_cast<int>(ida_native::mxfp8_scale_for_amax(amax)) : 0;
    new_scale_code_i = __shfl_sync(0xffffffffu, new_scale_code_i, 0);
    const std::uint8_t new_scale_code =
        static_cast<std::uint8_t>(new_scale_code_i);
    float new_scale = lane == 0
        ? ida_native::mxfp8_safe_scale(new_scale_code) : 0.0f;
    new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
    if (lane == 0) tensor.scales[block] = new_scale_code;
    if (!valid) return;

    const std::uint8_t code = ida_native::fp8_e4m3::pack(
        next_weight / new_scale).bits;
    tensor.payload[index] = code;
    tensor.momentum[index] = __float2bfloat16(next_momentum);
    tensor.compute[index] = ida_native::fp8_e4m3::unpack(code) * new_scale;
}

void lion_step_f32(
    torch::Tensor payload,
    torch::Tensor scales,
    torch::Tensor momentum,
    torch::Tensor gradient,
    torch::Tensor compute,
    double lr,
    double beta1,
    double beta2,
    double wd
) {
    const auto n = static_cast<std::size_t>(payload.numel());
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>(
        (n + ida_native::kMxfp8BlockSize - 1) /
        ida_native::kMxfp8BlockSize);
    auto stream = at::cuda::getCurrentCUDAStream(payload.device().index());
    k_lion_mxfp8<<<blocks, ida_native::kMxfp8BlockSize, 0, stream.stream()>>>(
        payload.data_ptr<std::uint8_t>(),
        scales.data_ptr<std::uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(momentum.data_ptr<at::BFloat16>()),
        gradient.data_ptr<float>(),
        compute.data_ptr<float>(),
        n,
        static_cast<float>(lr),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(wd));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lion_step_multi_f32(
    const std::vector<torch::Tensor>& payloads,
    const std::vector<torch::Tensor>& scales,
    const std::vector<torch::Tensor>& momenta,
    const std::vector<torch::Tensor>& gradients,
    const std::vector<torch::Tensor>& computes,
    double lr,
    double beta1,
    double beta2,
    double wd
) {
    const std::size_t count = payloads.size();
    if (count == 0) return;
    TORCH_CHECK(count <= kMaxLionTensorDescs,
                "too many tensors for multi-tensor fused Lion descriptor table");
    TORCH_CHECK(scales.size() == count && momenta.size() == count &&
                    gradients.size() == count && computes.size() == count,
                "multi-tensor fused Lion lists must have equal lengths");

    const auto device = payloads[0].device();
    std::vector<LionTensorDesc> host_descriptors;
    host_descriptors.reserve(count);
    std::size_t total_blocks = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& payload = payloads[i];
        const auto& scale = scales[i];
        const auto& momentum = momenta[i];
        const auto& gradient = gradients[i];
        const auto& compute = computes[i];
        TORCH_CHECK(payload.is_cuda() && scale.is_cuda() && momentum.is_cuda() &&
                        gradient.is_cuda() && compute.is_cuda(),
                    "all multi-tensor optimizer tensors must be CUDA");
        TORCH_CHECK(payload.device() == device && scale.device() == device &&
                        momentum.device() == device && gradient.device() == device &&
                        compute.device() == device,
                    "multi-tensor optimizer tensors must share a device");
        TORCH_CHECK(payload.scalar_type() == torch::kUInt8 &&
                        scale.scalar_type() == torch::kUInt8 &&
                        momentum.scalar_type() == torch::kBFloat16 &&
                        gradient.scalar_type() == torch::kFloat32 &&
                        compute.scalar_type() == torch::kFloat32,
                    "unsupported multi-tensor fused Lion dtype");
        TORCH_CHECK(payload.dim() == 1 && scale.dim() == 1 && momentum.dim() == 1 &&
                        gradient.dim() == 1 && compute.dim() == 1 &&
                        payload.is_contiguous() && scale.is_contiguous() &&
                        momentum.is_contiguous() && gradient.is_contiguous() &&
                        compute.is_contiguous(),
                    "multi-tensor fused Lion tensors must be flat and contiguous");
        TORCH_CHECK(payload.numel() == momentum.numel() &&
                        payload.numel() == gradient.numel() &&
                        payload.numel() == compute.numel() &&
                        scale.numel() ==
                            (payload.numel() + ida_native::kMxfp8BlockSize - 1) /
                                ida_native::kMxfp8BlockSize,
                    "invalid multi-tensor fused Lion tensor lengths");

        const auto n = static_cast<std::size_t>(payload.numel());
        const auto blocks = (n + ida_native::kMxfp8BlockSize - 1) /
            ida_native::kMxfp8BlockSize;
        const auto block_base = total_blocks;
        host_descriptors.push_back({
            payload.data_ptr<std::uint8_t>(),
            scale.data_ptr<std::uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(momentum.data_ptr<at::BFloat16>()),
            gradient.data_ptr<float>(),
            compute.data_ptr<float>(),
            n,
            static_cast<float>(lr),
            static_cast<float>(beta1),
            static_cast<float>(beta2),
            static_cast<float>(wd),
            block_base,
        });
        TORCH_CHECK(total_blocks + blocks <=
                        static_cast<std::size_t>(std::numeric_limits<unsigned>::max()),
                    "too many blocks for multi-tensor fused Lion launch");
        total_blocks += blocks;
    }
    if (total_blocks == 0) return;

    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_lion_descriptors,
        host_descriptors.data(),
        host_descriptors.size() * sizeof(LionTensorDesc),
        0,
        cudaMemcpyHostToDevice,
        stream.stream()));

    const auto blocks = static_cast<unsigned>(total_blocks);
    k_lion_mxfp8_multi<<<blocks, ida_native::kMxfp8BlockSize, 0, stream.stream()>>>(
        host_descriptors.size(), total_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void lion_step(
    torch::Tensor payload,
    torch::Tensor scales,
    torch::Tensor momentum,
    torch::Tensor gradient,
    torch::Tensor compute,
    double lr,
    double beta1,
    double beta2,
    double wd
) {
    TORCH_CHECK(payload.is_cuda(), "payload must be CUDA");
    TORCH_CHECK(scales.is_cuda() && momentum.is_cuda() && gradient.is_cuda() && compute.is_cuda(),
                "all optimizer tensors must be CUDA");
    TORCH_CHECK(payload.scalar_type() == torch::kUInt8, "payload must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8, "scales must be uint8");
    TORCH_CHECK(momentum.scalar_type() == torch::kBFloat16, "momentum must be bfloat16");
    TORCH_CHECK(gradient.scalar_type() == torch::kFloat32, "gradient must be float32");
    TORCH_CHECK(compute.scalar_type() == torch::kFloat32, "compute view must be float32");
    TORCH_CHECK(payload.dim() == 1 && scales.dim() == 1 && momentum.dim() == 1 &&
                    gradient.dim() == 1 && compute.dim() == 1,
                "fused Lion tensors must be flattened");
    TORCH_CHECK(payload.is_contiguous() && scales.is_contiguous() && momentum.is_contiguous() &&
                    gradient.is_contiguous() && compute.is_contiguous(),
                "fused Lion tensors must be contiguous");
    TORCH_CHECK(payload.numel() == momentum.numel() && payload.numel() == gradient.numel() &&
                    payload.numel() == compute.numel(),
                "fused Lion tensor lengths must agree");
    TORCH_CHECK(scales.numel() ==
                    (payload.numel() + ida_native::kMxfp8BlockSize - 1) /
                        ida_native::kMxfp8BlockSize,
                "invalid MXFP8 scale length");
    TORCH_CHECK(payload.device() == scales.device() && payload.device() == momentum.device() &&
                    payload.device() == gradient.device() && payload.device() == compute.device(),
                "fused Lion tensors must share a device");
    lion_step_f32(payload, scales, momentum, gradient, compute, lr, beta1, beta2, wd);
}

void lion_step_multi(
    std::vector<torch::Tensor> payloads,
    std::vector<torch::Tensor> scales,
    std::vector<torch::Tensor> momenta,
    std::vector<torch::Tensor> gradients,
    std::vector<torch::Tensor> computes,
    double lr,
    double beta1,
    double beta2,
    double wd
) {
    TORCH_CHECK(!payloads.empty(), "multi-tensor fused Lion received no tensors");
    TORCH_CHECK(payloads.size() == scales.size() && payloads.size() == momenta.size() &&
                    payloads.size() == gradients.size() && payloads.size() == computes.size(),
                "multi-tensor fused Lion lists must have equal lengths");
    lion_step_multi_f32(payloads, scales, momenta, gradients, computes,
                        lr, beta1, beta2, wd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("lion_step", &lion_step, "Fused MXFP8 Lion step (CUDA)");
    module.def("lion_step_multi", &lion_step_multi,
               "Multi-tensor fused MXFP8 Lion step (CUDA)");
}
