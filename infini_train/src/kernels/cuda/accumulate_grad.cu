#include <cmath>
#include <memory>
#include <cuda_bf16.h>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void AccumulateGradKernel(const T *grad_ptr, float rate, T *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += common::cuda::Mul(grad_ptr[idx], common::cuda::Cast<T>(rate));
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    auto device = tensor->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            AccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(gradient->DataPtr()), rate, static_cast<T *>(tensor->DataPtr()), num_elements);
        },
        "CUDA AccumulateGrad");
}

template <typename TGrad, typename TParam>
__global__ void AdamAccumulateGradKernel(const TGrad *grad_data, TParam *param_data, size_t num_elements, float *m_data,
                                         float *v_data, float learning_rate, float beta1, float beta2, float eps,
                                         const float bias_correction_m, const float bias_correction_v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // 将梯度转为 FP32（如果是 BF16）
        const float grad_fp32 = common::cuda::Cast<float>(grad_data[idx]);

        // 在 FP32 精度下更新 m 和 v
        m_data[idx] = beta1 * m_data[idx] + (1.0f - beta1) * grad_fp32;
        v_data[idx] = beta2 * v_data[idx] + (1.0f - beta2) * grad_fp32 * grad_fp32;

        // 计算更新量
        const float m_hat = m_data[idx] / bias_correction_m;
        const float v_hat = v_data[idx] / bias_correction_v;

        // 更新参数
        const float param_fp32 = common::cuda::Cast<float>(param_data[idx]);
        const float param_new = param_fp32 - learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps);
        param_data[idx] = common::cuda::Cast<TParam>(param_new);
    }
}

// Vectorized Adam kernel for FP32 parameters (processes 4 elements per thread)
template <typename TGrad>
__global__ void AdamAccumulateGradKernel_Vectorized(const TGrad *grad_data, float *param_data, size_t num_elements,
                                                     float *m_data, float *v_data, float learning_rate, float beta1,
                                                     float beta2, float eps, const float bias_correction_m,
                                                     const float bias_correction_v) {
    // Each thread processes 4 elements
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < num_elements) {
        // Vectorized load for FP32 arrays
        float4 m_vec = *reinterpret_cast<const float4 *>(&m_data[idx]);
        float4 v_vec = *reinterpret_cast<const float4 *>(&v_data[idx]);
        float4 param_vec = *reinterpret_cast<const float4 *>(&param_data[idx]);

        // Load gradients (handle BF16 specially)
        float grad_vals[4];
        if constexpr (sizeof(TGrad) == 2) {  // BF16
            nv_bfloat162 grad_vec2[2];
            grad_vec2[0] = *reinterpret_cast<const nv_bfloat162 *>(&grad_data[idx]);
            grad_vec2[1] = *reinterpret_cast<const nv_bfloat162 *>(&grad_data[idx + 2]);

            grad_vals[0] = __bfloat162float(grad_vec2[0].x);
            grad_vals[1] = __bfloat162float(grad_vec2[0].y);
            grad_vals[2] = __bfloat162float(grad_vec2[1].x);
            grad_vals[3] = __bfloat162float(grad_vec2[1].y);
        } else {  // FP32
            float4 grad_vec = *reinterpret_cast<const float4 *>(&grad_data[idx]);
            grad_vals[0] = grad_vec.x;
            grad_vals[1] = grad_vec.y;
            grad_vals[2] = grad_vec.z;
            grad_vals[3] = grad_vec.w;
        }

        // Process 4 elements
        float *m_ptr = reinterpret_cast<float *>(&m_vec);
        float *v_ptr = reinterpret_cast<float *>(&v_vec);
        float *param_ptr = reinterpret_cast<float *>(&param_vec);

#pragma unroll
        for (int i = 0; i < 4; i++) {
            float grad = grad_vals[i];
            float m = m_ptr[i];
            float v = v_ptr[i];
            float param = param_ptr[i];

            m = beta1 * m + (1.0f - beta1) * grad;
            v = beta2 * v + (1.0f - beta2) * grad * grad;

            const float m_hat = m / bias_correction_m;
            const float v_hat = v / bias_correction_v;

            param = param - learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps);

            m_ptr[i] = m;
            v_ptr[i] = v;
            param_ptr[i] = param;
        }

        // Vectorized store
        *reinterpret_cast<float4 *>(&m_data[idx]) = m_vec;
        *reinterpret_cast<float4 *>(&v_data[idx]) = v_vec;
        *reinterpret_cast<float4 *>(&param_data[idx]) = param_vec;
    }

    // Handle remaining elements (scalar processing)
    size_t remaining_start = (num_elements / 4) * 4;
    size_t scalar_idx = remaining_start + (blockIdx.x * blockDim.x + threadIdx.x);
    if (scalar_idx < num_elements && idx >= remaining_start) {
        const float grad_fp32 = common::cuda::Cast<float>(grad_data[scalar_idx]);
        float m = m_data[scalar_idx];
        float v = v_data[scalar_idx];
        float param = param_data[scalar_idx];

        m = beta1 * m + (1.0f - beta1) * grad_fp32;
        v = beta2 * v + (1.0f - beta2) * grad_fp32 * grad_fp32;

        const float m_hat = m / bias_correction_m;
        const float v_hat = v / bias_correction_v;

        param = param - learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps);

        m_data[scalar_idx] = m;
        v_data[scalar_idx] = v;
        param_data[scalar_idx] = param;
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    size_t num_elements = grad->NumElements();

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    auto grad_dtype = grad->Dtype();
    auto param_dtype = param->Dtype();

    // Use vectorized kernel for FP32 parameters when data is aligned
    bool use_vectorized = (param_dtype == DataType::kFLOAT32) && (num_elements >= 4) &&
                          (reinterpret_cast<uintptr_t>(grad->DataPtr()) % 16 == 0) &&
                          (reinterpret_cast<uintptr_t>(param->DataPtr()) % 16 == 0) &&
                          (reinterpret_cast<uintptr_t>(m->DataPtr()) % 16 == 0) &&
                          (reinterpret_cast<uintptr_t>(v->DataPtr()) % 16 == 0);

    if (use_vectorized) {
        // Vectorized kernel: each thread processes 4 elements
        int threads_per_block = 256;
        int num_blocks = ((num_elements + 3) / 4 + threads_per_block - 1) / threads_per_block;

        if (grad_dtype == DataType::kBFLOAT16) {
            AdamAccumulateGradKernel_Vectorized<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const nv_bfloat16 *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()), num_elements,
                static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        } else if (grad_dtype == DataType::kFLOAT32) {
            AdamAccumulateGradKernel_Vectorized<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const float *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()), num_elements,
                static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        }
    } else {
        // Scalar kernel: each thread processes 1 element
        int threads_per_block = 256;
        int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

        if (grad_dtype == DataType::kBFLOAT16 && param_dtype == DataType::kFLOAT32) {
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const nv_bfloat16 *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()), num_elements,
                static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        } else if (grad_dtype == DataType::kFLOAT32 && param_dtype == DataType::kFLOAT32) {
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const float *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()), num_elements,
                static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        } else if (grad_dtype == DataType::kBFLOAT16 && param_dtype == DataType::kBFLOAT16) {
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const nv_bfloat16 *>(grad->DataPtr()), static_cast<nv_bfloat16 *>(param->DataPtr()),
                num_elements, static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1,
                beta2, eps, bias_correction_m, bias_correction_v);
        } else {
            LOG(FATAL) << "Unsupported dtype combination for AdamAccumulateGrad: grad=" << (int)grad_dtype
                       << ", param=" << (int)param_dtype;
        }
    }
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
