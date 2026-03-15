#include "infini_train/include/autograd/flash_attention.h"

#include <cmath>
#include <memory>
#include <vector>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"

// Forward declaration of kernel functions with CORRECT namespace
namespace infini_train::kernels::cuda {
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FlashAttentionForward(const std::shared_ptr<Tensor> &q,
                      const std::shared_ptr<Tensor> &k,
                      const std::shared_ptr<Tensor> &v,
                      float scale,
                      bool is_causal,
                      float dropout_p);

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FlashAttentionBackward(const std::shared_ptr<Tensor> &grad_output,
                       const std::shared_ptr<Tensor> &q,
                       const std::shared_ptr<Tensor> &k,
                       const std::shared_ptr<Tensor> &v,
                       const std::shared_ptr<Tensor> &output,
                       const std::shared_ptr<Tensor> &softmax_lse,
                       float scale,
                       bool is_causal,
                       float dropout_p);
} // namespace infini_train::kernels::cuda

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> FlashAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {

    CHECK_EQ(input_tensors.size(), 3) << "FlashAttention expects 3 inputs: Q, K, V";

    auto q = input_tensors[0];
    auto k = input_tensors[1];
    auto v = input_tensors[2];

    // Validate input shapes: (batch, seqlen, num_heads, head_dim)
    CHECK_EQ(q->Dims().size(), 4) << "Q must be 4D tensor (batch, seqlen, num_heads, head_dim)";
    CHECK_EQ(k->Dims().size(), 4) << "K must be 4D tensor (batch, seqlen, num_heads, head_dim)";
    CHECK_EQ(v->Dims().size(), 4) << "V must be 4D tensor (batch, seqlen, num_heads, head_dim)";

    const int64_t head_dim = q->Dims()[3];

    // Auto-compute scale if not set
    float scale = scale_;
    if (scale < 0) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    // Check device type
    CHECK(q->GetDevice().type() == Device::DeviceType::kCUDA)
        << "FlashAttention only supports CUDA devices";

    // Check data type support
    auto dtype = q->Dtype();
    if (dtype == DataType::kFLOAT32) {
        LOG(WARNING) << "Flash Attention 2 does not support FP32. "
                     << "Performance will be suboptimal. Consider using FP16 or BF16.";
    }

    // Call kernel function with fully qualified namespace
    auto [output, softmax_lse] = ::infini_train::kernels::cuda::FlashAttentionForward(
        q, k, v, scale, is_causal_, dropout_p_);

    // Save softmax_lse for backward
    softmax_lse_ = softmax_lse;

    return {output};
}

void FlashAttention::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &output_tensors) {

    // Save Q, K, V, O for backward pass
    CHECK_EQ(input_tensors.size(), 3);
    CHECK_EQ(output_tensors.size(), 1);

    saved_tensors_.clear();
    saved_tensors_.push_back(input_tensors[0]);  // Q
    saved_tensors_.push_back(input_tensors[1]);  // K
    saved_tensors_.push_back(input_tensors[2]);  // V
    saved_tensors_.push_back(output_tensors[0]); // O
    // softmax_lse_ is already saved as member variable
}

std::vector<std::shared_ptr<Tensor>> FlashAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {

    CHECK_EQ(grad_outputs.size(), 1) << "FlashAttention backward expects 1 grad_output";
    CHECK_EQ(saved_tensors_.size(), 4) << "FlashAttention backward expects 4 saved tensors";

    auto grad_output = grad_outputs[0];
    auto q = saved_tensors_[0];
    auto k = saved_tensors_[1];
    auto v = saved_tensors_[2];
    auto output = saved_tensors_[3];

    const int64_t head_dim = q->Dims()[3];

    // Auto-compute scale if not set
    float scale = scale_;
    if (scale < 0) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    // Call kernel function with fully qualified namespace
    auto [grad_q, grad_k, grad_v] = ::infini_train::kernels::cuda::FlashAttentionBackward(
        grad_output, q, k, v, output, softmax_lse_,
        scale, is_causal_, dropout_p_);

    return {grad_q, grad_k, grad_v};
}

} // namespace infini_train::autograd
