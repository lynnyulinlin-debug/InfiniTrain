#ifdef ENABLE_FLASH_ATTENTION

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include <cutlass/numeric_types.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

// ============================================================================
// Flash Attention Forward Kernel
// ============================================================================
// Based on FlashAttention-2 paper (arXiv:2307.08691)
// Implements online softmax algorithm with tiling

// Block sizes for tiling
constexpr int kBlockSizeM = 64;  // Q block size
constexpr int kBlockSizeN = 64;  // K, V block size

// Helper: Compute max of a row in shared memory
template<typename T, int kSize>
__device__ float RowMax(const T* row) {
    float max_val = -INFINITY;
    #pragma unroll
    for (int i = 0; i < kSize; ++i) {
        max_val = fmaxf(max_val, static_cast<float>(row[i]));
    }
    return max_val;
}

// Helper: Compute sum of a row in shared memory
template<typename T, int kSize>
__device__ float RowSum(const T* row) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kSize; ++i) {
        sum += static_cast<float>(row[i]);
    }
    return sum;
}

// Simplified Flash Attention Forward Kernel
// Template parameters:
//   T: data type (cutlass::bfloat16_t or cutlass::half_t)
//   kHeadDim: head dimension (64 or 128)
template<typename T, int kHeadDim>
__global__ void FlashAttentionFwdKernelSimple(
    const T* __restrict__ Q,      // [batch, seqlen_q, num_heads, head_dim]
    const T* __restrict__ K,      // [batch, seqlen_k, num_heads, head_dim]
    const T* __restrict__ V,      // [batch, seqlen_k, num_heads, head_dim]
    T* __restrict__ O,            // [batch, seqlen_q, num_heads, head_dim]
    float* __restrict__ softmax_lse,  // [batch, num_heads, seqlen_q]
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float scale,
    bool is_causal,
    int q_batch_stride,
    int q_row_stride,
    int q_head_stride,
    int k_batch_stride,
    int k_row_stride,
    int k_head_stride,
    int v_batch_stride,
    int v_row_stride,
    int v_head_stride,
    int o_batch_stride,
    int o_row_stride,
    int o_head_stride
) {
    // Thread block processes one Q row
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx >= seqlen_q) return;

    // Shared memory for Q, K, V blocks
    __shared__ float s_Q[kHeadDim];
    __shared__ float s_K[32 * kHeadDim];  // Reduced from 64 to 32
    __shared__ float s_V[32 * kHeadDim];  // Reduced from 64 to 32
    __shared__ float s_S[32];  // Attention scores

    // Load Q row into shared memory
    const T* q_ptr = Q + batch_idx * q_batch_stride +
                     head_idx * q_head_stride +
                     q_idx * q_row_stride;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        s_Q[d] = static_cast<float>(q_ptr[d]);
    }
    __syncthreads();

    // Initialize online softmax accumulators
    float m = -INFINITY;  // max value
    float l = 0.0f;       // sum of exp
    float O_acc[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) {
        O_acc[d] = 0.0f;
    }

    // Loop over K, V blocks
    // Apply causal mask: only attend to positions <= q_idx
    const int k_end_causal = is_causal ? min(q_idx + 1, seqlen_k) : seqlen_k;
    const int num_k_blocks = (k_end_causal + 31) / 32;  // Changed from 64 to 32

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * 32;  // Changed from 64 to 32
        const int k_end = min(k_start + 32, k_end_causal);  // Changed from 64 to 32
        const int k_size = k_end - k_start;

        // Load K block into shared memory
        for (int k = threadIdx.x; k < k_size; k += blockDim.x) {
            const T* k_ptr = K + batch_idx * k_batch_stride +
                             head_idx * k_head_stride +
                             (k_start + k) * k_row_stride;
            for (int d = 0; d < kHeadDim; ++d) {
                s_K[k * kHeadDim + d] = static_cast<float>(k_ptr[d]);
            }
        }

        // Load V block into shared memory
        for (int k = threadIdx.x; k < k_size; k += blockDim.x) {
            const T* v_ptr = V + batch_idx * v_batch_stride +
                             head_idx * v_head_stride +
                             (k_start + k) * v_row_stride;
            for (int d = 0; d < kHeadDim; ++d) {
                s_V[k * kHeadDim + d] = static_cast<float>(v_ptr[d]);
            }
        }
        __syncthreads();

        // Compute Q @ K^T for this block
        if (threadIdx.x < k_size) {
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                qk += s_Q[d] * s_K[threadIdx.x * kHeadDim + d];
            }
            s_S[threadIdx.x] = qk * scale;
        }
        __syncthreads();

        // Compute max of this block
        float m_block = -INFINITY;
        for (int k = 0; k < k_size; ++k) {
            m_block = fmaxf(m_block, s_S[k]);
        }

        // Update online softmax
        float m_new = fmaxf(m, m_block);
        float exp_m_diff = expf(m - m_new);
        float l_new = l * exp_m_diff;

        // Compute exp(S - m_new) and accumulate
        for (int k = 0; k < k_size; ++k) {
            float exp_s = expf(s_S[k] - m_new);
            l_new += exp_s;

            // Accumulate to O: O += exp(S) @ V
            if (threadIdx.x == 0) {
                for (int d = 0; d < kHeadDim; ++d) {
                    O_acc[d] = O_acc[d] * exp_m_diff + exp_s * s_V[k * kHeadDim + d];
                }
            }
        }

        m = m_new;
        l = l_new;
        __syncthreads();
    }

    // Normalize O by l
    if (threadIdx.x == 0) {
        T* o_ptr = O + batch_idx * o_batch_stride +
                   head_idx * o_head_stride +
                   q_idx * o_row_stride;

        for (int d = 0; d < kHeadDim; ++d) {
            o_ptr[d] = static_cast<T>(O_acc[d] / l);
        }

        // Store log-sum-exp for backward pass
        if (softmax_lse != nullptr) {
            softmax_lse[batch_idx * num_heads * seqlen_q +
                       head_idx * seqlen_q + q_idx] = logf(l) + m;
        }
    }
}

// ============================================================================
// Flash Attention Backward Kernel
// ============================================================================
// Based on FlashAttention-2 paper (arXiv:2307.08691)
// Implements backward pass with recomputation

// Simplified Flash Attention Backward Kernel
// Template parameters:
//   T: data type (cutlass::bfloat16_t or cutlass::half_t)
//   kHeadDim: head dimension (64 or 128)
template<typename T, int kHeadDim>
__global__ void FlashAttentionBwdKernelSimple(
    const T* __restrict__ dO,     // [batch, seqlen_q, num_heads, head_dim] - gradient of output
    const T* __restrict__ Q,      // [batch, seqlen_q, num_heads, head_dim]
    const T* __restrict__ K,      // [batch, seqlen_k, num_heads, head_dim]
    const T* __restrict__ V,      // [batch, seqlen_k, num_heads, head_dim]
    const T* __restrict__ O,      // [batch, seqlen_q, num_heads, head_dim] - output from forward
    const float* __restrict__ softmax_lse,  // [batch, num_heads, seqlen_q] - log-sum-exp from forward
    T* __restrict__ dQ,           // [batch, seqlen_q, num_heads, head_dim] - gradient of Q
    T* __restrict__ dK,           // [batch, seqlen_k, num_heads, head_dim] - gradient of K
    T* __restrict__ dV,           // [batch, seqlen_k, num_heads, head_dim] - gradient of V
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float scale,
    bool is_causal,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    int do_batch_stride, int do_row_stride, int do_head_stride,
    int dq_batch_stride, int dq_row_stride, int dq_head_stride,
    int dk_batch_stride, int dk_row_stride, int dk_head_stride,
    int dv_batch_stride, int dv_row_stride, int dv_head_stride
) {
    // Thread block processes one Q row
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx >= seqlen_q) return;

    // Shared memory
    __shared__ float s_Q[kHeadDim];
    __shared__ float s_dO[kHeadDim];
    __shared__ float s_O[kHeadDim];
    __shared__ float s_K[32 * kHeadDim];  // Reduced from 64 to 32
    __shared__ float s_V[32 * kHeadDim];  // Reduced from 64 to 32
    __shared__ float s_S[32];      // Attention scores
    __shared__ float s_P[32];      // Softmax probabilities
    __shared__ float s_dS[32];     // Gradient of scores

    // Load Q, dO, O for this row
    const T* q_ptr = Q + batch_idx * q_batch_stride + head_idx * q_head_stride + q_idx * q_row_stride;
    const T* do_ptr = dO + batch_idx * do_batch_stride + head_idx * do_head_stride + q_idx * do_row_stride;
    const T* o_ptr = O + batch_idx * o_batch_stride + head_idx * o_head_stride + q_idx * o_row_stride;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        s_Q[d] = static_cast<float>(q_ptr[d]);
        s_dO[d] = static_cast<float>(do_ptr[d]);
        s_O[d] = static_cast<float>(o_ptr[d]);
    }
    __syncthreads();

    // Load softmax_lse for this row
    float lse = softmax_lse[batch_idx * num_heads * seqlen_q + head_idx * seqlen_q + q_idx];

    // Compute D = rowsum(dO * O)
    float D = 0.0f;
    if (threadIdx.x == 0) {
        for (int d = 0; d < kHeadDim; ++d) {
            D += s_dO[d] * s_O[d];
        }
    }
    __syncthreads();

    // Broadcast D to all threads
    __shared__ float s_D;
    if (threadIdx.x == 0) {
        s_D = D;
    }
    __syncthreads();
    D = s_D;

    // Initialize dQ accumulator
    float dQ_acc[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) {
        dQ_acc[d] = 0.0f;
    }

    // Determine K range based on causal mask
    int k_end = is_causal ? min(q_idx + 1, seqlen_k) : seqlen_k;
    int num_k_blocks = (k_end + 31) / 32;  // Changed from 64 to 32

    // Loop over K, V blocks
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_start = k_block * 32;  // Changed from 64 to 32
        const int k_block_end = min(k_start + 32, k_end);  // Changed from 64 to 32
        const int k_size = k_block_end - k_start;

        // Load K block
        for (int k = threadIdx.x; k < k_size; k += blockDim.x) {
            const T* k_ptr = K + batch_idx * k_batch_stride + head_idx * k_head_stride + (k_start + k) * k_row_stride;
            for (int d = 0; d < kHeadDim; ++d) {
                s_K[k * kHeadDim + d] = static_cast<float>(k_ptr[d]);
            }
        }

        // Load V block
        for (int k = threadIdx.x; k < k_size; k += blockDim.x) {
            const T* v_ptr = V + batch_idx * v_batch_stride + head_idx * v_head_stride + (k_start + k) * v_row_stride;
            for (int d = 0; d < kHeadDim; ++d) {
                s_V[k * kHeadDim + d] = static_cast<float>(v_ptr[d]);
            }
        }
        __syncthreads();

        // Recompute attention scores: S = Q @ K^T * scale
        if (threadIdx.x < k_size) {
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                qk += s_Q[d] * s_K[threadIdx.x * kHeadDim + d];
            }
            s_S[threadIdx.x] = qk * scale;

            // Compute softmax: P = exp(S - lse)
            s_P[threadIdx.x] = expf(s_S[threadIdx.x] - lse);
        }
        __syncthreads();

        // Compute dS = P * (dO @ V^T - D)
        if (threadIdx.x < k_size) {
            float dO_vT = 0.0f;
            #pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                dO_vT += s_dO[d] * s_V[threadIdx.x * kHeadDim + d];
            }
            s_dS[threadIdx.x] = s_P[threadIdx.x] * (dO_vT - D);
        }
        __syncthreads();

        // Accumulate dQ: dQ += dS @ K * scale
        if (threadIdx.x == 0) {
            for (int k = 0; k < k_size; ++k) {
                float ds_val = s_dS[k] * scale;
                for (int d = 0; d < kHeadDim; ++d) {
                    dQ_acc[d] += ds_val * s_K[k * kHeadDim + d];
                }
            }
        }

        // Accumulate dK: dK += dS^T @ Q * scale (done atomically later)
        // Accumulate dV: dV += P^T @ dO (done atomically later)
        for (int k = threadIdx.x; k < k_size; k += blockDim.x) {
            T* dk_ptr = dK + batch_idx * dk_batch_stride + head_idx * dk_head_stride + (k_start + k) * dk_row_stride;
            T* dv_ptr = dV + batch_idx * dv_batch_stride + head_idx * dv_head_stride + (k_start + k) * dv_row_stride;

            float ds_val = s_dS[k] * scale;
            float p_val = s_P[k];

            for (int d = 0; d < kHeadDim; ++d) {
                // dK += dS^T @ Q
                atomicAdd(reinterpret_cast<float*>(&dk_ptr[d]), ds_val * s_Q[d]);
                // dV += P^T @ dO
                atomicAdd(reinterpret_cast<float*>(&dv_ptr[d]), p_val * s_dO[d]);
            }
        }
        __syncthreads();
    }

    // Write dQ
    if (threadIdx.x == 0) {
        T* dq_ptr = dQ + batch_idx * dq_batch_stride + head_idx * dq_head_stride + q_idx * dq_row_stride;
        for (int d = 0; d < kHeadDim; ++d) {
            dq_ptr[d] = static_cast<T>(dQ_acc[d]);
        }
    }
}

// ============================================================================
// Flash Attention Interface Functions
// ============================================================================

// Flash Attention Forward implementation
void FlashAttentionForward(
    const Tensor& q,           // [batch, seqlen_q, num_heads, head_dim]
    const Tensor& k,           // [batch, seqlen_k, num_heads, head_dim]
    const Tensor& v,           // [batch, seqlen_k, num_heads, head_dim]
    Tensor& out,               // [batch, seqlen_q, num_heads, head_dim]
    Tensor& softmax_lse,       // [batch, num_heads, seqlen_q]
    float scale,
    bool is_causal,
    float dropout_p,
    int64_t window_size_left,
    int64_t window_size_right
) {
    // Get dimensions
    const auto& q_dims = q.Dims();
    int64_t batch = q_dims[0];
    int64_t seqlen_q = q_dims[1];
    int64_t num_heads = q_dims[2];
    int64_t head_dim = q_dims[3];

    const auto& k_dims = k.Dims();
    int64_t seqlen_k = k_dims[1];

    // Compute strides manually (assuming row-major layout)
    // For [batch, seqlen, num_heads, head_dim]:
    // stride[3] = 1
    // stride[2] = head_dim
    // stride[1] = num_heads * head_dim
    // stride[0] = seqlen * num_heads * head_dim
    int64_t q_batch_stride = seqlen_q * num_heads * head_dim;
    int64_t q_row_stride = num_heads * head_dim;
    int64_t q_head_stride = head_dim;

    int64_t k_batch_stride = seqlen_k * num_heads * head_dim;
    int64_t k_row_stride = num_heads * head_dim;
    int64_t k_head_stride = head_dim;

    int64_t v_batch_stride = seqlen_k * num_heads * head_dim;
    int64_t v_row_stride = num_heads * head_dim;
    int64_t v_head_stride = head_dim;

    int64_t o_batch_stride = seqlen_q * num_heads * head_dim;
    int64_t o_row_stride = num_heads * head_dim;
    int64_t o_head_stride = head_dim;

    // Get CUDA stream
    auto device = q.GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    // Get data type
    bool is_bf16 = (q.Dtype() == DataType::kBFLOAT16);

    // Check unsupported features
    if (dropout_p > 0.0f) {
        LOG(WARNING) << "Flash Attention: dropout not yet supported, ignoring dropout_p=" << dropout_p;
    }
    if (window_size_left >= 0 || window_size_right >= 0) {
        LOG(WARNING) << "Flash Attention: windowing not yet supported, ignoring window sizes";
    }

    // Launch kernel based on dtype and head_dim
    dim3 grid(seqlen_q, num_heads, batch);
    dim3 block(256);

    if (head_dim == 64) {
        if (is_bf16) {
            FlashAttentionFwdKernelSimple<cutlass::bfloat16_t, 64><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::bfloat16_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(v.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(out.DataPtr()),
                reinterpret_cast<float*>(softmax_lse.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride
            );
        } else {
            FlashAttentionFwdKernelSimple<cutlass::half_t, 64><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::half_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(v.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(out.DataPtr()),
                reinterpret_cast<float*>(softmax_lse.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride
            );
        }
    } else if (head_dim == 128) {
        if (is_bf16) {
            FlashAttentionFwdKernelSimple<cutlass::bfloat16_t, 128><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::bfloat16_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(v.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(out.DataPtr()),
                reinterpret_cast<float*>(softmax_lse.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride
            );
        } else {
            FlashAttentionFwdKernelSimple<cutlass::half_t, 128><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::half_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(v.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(out.DataPtr()),
                reinterpret_cast<float*>(softmax_lse.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride
            );
        }
    } else {
        LOG(FATAL) << "Unsupported head_dim: " << head_dim << ". Only 64 and 128 are supported.";
    }

    CUDA_CHECK(cudaGetLastError());
}

// Flash Attention Backward implementation
void FlashAttentionBackward(
    const Tensor& dout,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& softmax_lse,
    Tensor& dq,
    Tensor& dk,
    Tensor& dv,
    float scale,
    bool is_causal,
    float dropout_p,
    int64_t window_size_left,
    int64_t window_size_right
) {
    // Get dimensions
    const auto& q_dims = q.Dims();
    int64_t batch = q_dims[0];
    int64_t seqlen_q = q_dims[1];
    int64_t num_heads = q_dims[2];
    int64_t head_dim = q_dims[3];

    const auto& k_dims = k.Dims();
    int64_t seqlen_k = k_dims[1];

    // Compute strides manually (assuming row-major layout)
    int64_t q_batch_stride = seqlen_q * num_heads * head_dim;
    int64_t q_row_stride = num_heads * head_dim;
    int64_t q_head_stride = head_dim;

    int64_t k_batch_stride = seqlen_k * num_heads * head_dim;
    int64_t k_row_stride = num_heads * head_dim;
    int64_t k_head_stride = head_dim;

    int64_t v_batch_stride = seqlen_k * num_heads * head_dim;
    int64_t v_row_stride = num_heads * head_dim;
    int64_t v_head_stride = head_dim;

    int64_t o_batch_stride = seqlen_q * num_heads * head_dim;
    int64_t o_row_stride = num_heads * head_dim;
    int64_t o_head_stride = head_dim;

    int64_t do_batch_stride = seqlen_q * num_heads * head_dim;
    int64_t do_row_stride = num_heads * head_dim;
    int64_t do_head_stride = head_dim;

    int64_t dq_batch_stride = seqlen_q * num_heads * head_dim;
    int64_t dq_row_stride = num_heads * head_dim;
    int64_t dq_head_stride = head_dim;

    int64_t dk_batch_stride = seqlen_k * num_heads * head_dim;
    int64_t dk_row_stride = num_heads * head_dim;
    int64_t dk_head_stride = head_dim;

    int64_t dv_batch_stride = seqlen_k * num_heads * head_dim;
    int64_t dv_row_stride = num_heads * head_dim;
    int64_t dv_head_stride = head_dim;

    // Get CUDA stream
    auto device = q.GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    // Get data type
    bool is_bf16 = (q.Dtype() == DataType::kBFLOAT16);

    // Check unsupported features
    if (dropout_p > 0.0f) {
        LOG(WARNING) << "Flash Attention Backward: dropout not yet supported";
    }
    if (window_size_left >= 0 || window_size_right >= 0) {
        LOG(WARNING) << "Flash Attention Backward: windowing not yet supported";
    }

    // Initialize dK and dV to zero (since we use atomicAdd)
    size_t dk_size = dk.NumElements() * (is_bf16 ? sizeof(cutlass::bfloat16_t) : sizeof(cutlass::half_t));
    size_t dv_size = dv.NumElements() * (is_bf16 ? sizeof(cutlass::bfloat16_t) : sizeof(cutlass::half_t));
    CUDA_CHECK(cudaMemsetAsync(dk.DataPtr(), 0, dk_size, cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(dv.DataPtr(), 0, dv_size, cuda_stream));

    // Launch kernel
    dim3 grid(seqlen_q, num_heads, batch);
    dim3 block(256);

    if (head_dim == 64) {
        if (is_bf16) {
            FlashAttentionBwdKernelSimple<cutlass::bfloat16_t, 64><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::bfloat16_t*>(dout.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(v.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(out.DataPtr()),
                reinterpret_cast<const float*>(softmax_lse.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(dq.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(dk.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(dv.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride,
                do_batch_stride, do_row_stride, do_head_stride,
                dq_batch_stride, dq_row_stride, dq_head_stride,
                dk_batch_stride, dk_row_stride, dk_head_stride,
                dv_batch_stride, dv_row_stride, dv_head_stride
            );
        } else {
            FlashAttentionBwdKernelSimple<cutlass::half_t, 64><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::half_t*>(dout.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(v.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(out.DataPtr()),
                reinterpret_cast<const float*>(softmax_lse.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(dq.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(dk.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(dv.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride,
                do_batch_stride, do_row_stride, do_head_stride,
                dq_batch_stride, dq_row_stride, dq_head_stride,
                dk_batch_stride, dk_row_stride, dk_head_stride,
                dv_batch_stride, dv_row_stride, dv_head_stride
            );
        }
    } else if (head_dim == 128) {
        if (is_bf16) {
            FlashAttentionBwdKernelSimple<cutlass::bfloat16_t, 128><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::bfloat16_t*>(dout.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(v.DataPtr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(out.DataPtr()),
                reinterpret_cast<const float*>(softmax_lse.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(dq.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(dk.DataPtr()),
                reinterpret_cast<cutlass::bfloat16_t*>(dv.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride,
                do_batch_stride, do_row_stride, do_head_stride,
                dq_batch_stride, dq_row_stride, dq_head_stride,
                dk_batch_stride, dk_row_stride, dk_head_stride,
                dv_batch_stride, dv_row_stride, dv_head_stride
            );
        } else {
            FlashAttentionBwdKernelSimple<cutlass::half_t, 128><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<const cutlass::half_t*>(dout.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(q.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(k.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(v.DataPtr()),
                reinterpret_cast<const cutlass::half_t*>(out.DataPtr()),
                reinterpret_cast<const float*>(softmax_lse.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(dq.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(dk.DataPtr()),
                reinterpret_cast<cutlass::half_t*>(dv.DataPtr()),
                batch, seqlen_q, seqlen_k, num_heads, scale, is_causal,
                q_batch_stride, q_row_stride, q_head_stride,
                k_batch_stride, k_row_stride, k_head_stride,
                v_batch_stride, v_row_stride, v_head_stride,
                o_batch_stride, o_row_stride, o_head_stride,
                do_batch_stride, do_row_stride, do_head_stride,
                dq_batch_stride, dq_row_stride, dq_head_stride,
                dk_batch_stride, dk_row_stride, dk_head_stride,
                dv_batch_stride, dv_row_stride, dv_head_stride
            );
        }
    } else {
        LOG(FATAL) << "Unsupported head_dim: " << head_dim << ". Only 64 and 128 are supported.";
    }

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Wrapper functions for autograd layer
// ============================================================================

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FlashAttentionForward(const std::shared_ptr<Tensor> &q,
                      const std::shared_ptr<Tensor> &k,
                      const std::shared_ptr<Tensor> &v,
                      float scale,
                      bool is_causal,
                      float dropout_p) {
    // Get dimensions
    const auto& q_dims = q->Dims();
    int64_t batch = q_dims[0];
    int64_t seqlen_q = q_dims[1];
    int64_t num_heads = q_dims[2];
    int64_t head_dim = q_dims[3];

    // Create output tensors
    auto out = std::make_shared<Tensor>(q_dims, q->Dtype(), q->GetDevice());

    // softmax_lse shape: [batch, num_heads, seqlen_q]
    std::vector<int64_t> lse_dims = {batch, num_heads, seqlen_q};
    auto softmax_lse = std::make_shared<Tensor>(lse_dims, DataType::kFLOAT32, q->GetDevice());

    // Call the kernel function
    FlashAttentionForward(*q, *k, *v, *out, *softmax_lse, scale, is_causal, dropout_p, -1, -1);

    return std::make_tuple(out, softmax_lse);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FlashAttentionBackward(const std::shared_ptr<Tensor> &grad_output,
                       const std::shared_ptr<Tensor> &q,
                       const std::shared_ptr<Tensor> &k,
                       const std::shared_ptr<Tensor> &v,
                       const std::shared_ptr<Tensor> &output,
                       const std::shared_ptr<Tensor> &softmax_lse,
                       float scale,
                       bool is_causal,
                       float dropout_p) {
    // Create gradient tensors with same shape as inputs
    auto dq = std::make_shared<Tensor>(q->Dims(), q->Dtype(), q->GetDevice());
    auto dk = std::make_shared<Tensor>(k->Dims(), k->Dtype(), k->GetDevice());
    auto dv = std::make_shared<Tensor>(v->Dims(), v->Dtype(), v->GetDevice());

    // Call the kernel function
    FlashAttentionBackward(*grad_output, *q, *k, *v, *output, *softmax_lse,
                          *dq, *dk, *dv, scale, is_causal, dropout_p, -1, -1);

    return std::make_tuple(dq, dk, dv);
}

} // namespace infini_train::kernels::cuda

#endif // ENABLE_FLASH_ATTENTION
