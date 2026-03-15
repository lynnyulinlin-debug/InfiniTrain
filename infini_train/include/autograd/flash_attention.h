#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

// Flash Attention 2 Autograd Function
// Implements efficient attention computation with O(N) memory complexity
class FlashAttention : public Function {
public:
    static constexpr char kType[] = "FlashAttentionFunction";

    FlashAttention() : Function(kType) {}

    // Forward pass: computes attention output and LSE (log-sum-exp)
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    // Setup context: saves tensors needed for backward pass
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    // Backward pass: computes gradients for Q, K, V
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

    // Configuration setters
    void SetCausal(bool is_causal) { is_causal_ = is_causal; }
    void SetDropout(float dropout_p) { dropout_p_ = dropout_p; }
    void SetScale(float scale) { scale_ = scale; }
    void SetWindowSize(int64_t left, int64_t right) {
        window_size_left_ = left;
        window_size_right_ = right;
    }

private:
    bool is_causal_ = false;
    float dropout_p_ = 0.0f;
    float scale_ = -1.0f;  // -1 means auto-compute as 1/sqrt(head_dim)
    int64_t window_size_left_ = -1;
    int64_t window_size_right_ = -1;

    // Saved for backward
    std::shared_ptr<Tensor> softmax_lse_;  // log-sum-exp values
};

} // namespace infini_train::autograd
