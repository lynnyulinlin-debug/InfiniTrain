# InfiniTrain

[![CI](https://github.com/InfiniTensor/InfiniTrain/actions/workflows/format-check.yaml/badge.svg)](
https://github.com/InfiniTensor/InfiniTrain/actions
)
[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniTrain)](
https://github.com/InfiniTensor/InfiniTrain/issues
)
[![PR](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniTrain)](
https://github.com/InfiniTensor/InfiniTrain/pulls
)
[![License](https://img.shields.io/github/license/InfiniTensor/InfiniTrain)](
https://github.com/InfiniTensor/InfiniTrain/blob/master/LICENSE
)

A from-scratch C++ training framework for large-scale models with multi-dimensional distributed parallelism.

## 🚀 Quick Start

### System Requirements

#### Hardware Requirements

- **Recommended**: NVIDIA Ampere-class GPUs (A100/A800) or newer

#### Software Requirements

- **CUDA / NCCL**: Latest stable versions
- **gcc / g++**: Version **13+**
- **CMake**: Version **3.13+**

### Installation

```bash
# Initialize submodules (including Flash Attention 2)
git submodule update --init --recursive

# Build
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DENABLE_FLASH_ATTENTION=ON
make -j
```

Build Options:

- `USE_CUDA=ON`

  Enable CUDA backend support.

- `USE_NCCL=ON`

  Enable NCCL-based distributed communication.

- `ENABLE_FLASH_ATTENTION=ON`

  Enable Flash Attention 2 for memory-efficient attention (requires CUDA).

> Both options are optional and can be disabled for CPU-only builds.

## ✨ InfiniTrain Overview

### ✔ Support Matrix

| Category                  | Feature                         | Description                                          | Status         |
| ------------------------- | ------------------------------- | ---------------------------------------------------- | -------------- |
| Model Support             | GPT-2                           | Decoder-only Transformer language model              | ✔ Supported    |
|                           | LLaMA 3                         | Modern LLaMA-family Transformer architecture         | ✔ Supported    |
|                           | DeepSeek-V3                     | Large-scale MoE-based language model                 | 🗓 Planned     |
| Precision                 | Multiple Data Type              | FP32, BF16                                           | ✔ Supported    |
|                           | Mixed Precision                 | Autocast-based BF16 compute with FP32 accumulation   | ✔ Supported    |
| Distributed Training      | Data Parallel (DP)              | Parameter-server-style data parallelism              | ✔ Supported    |
|                           | Distributed Data Parallel (DDP) | Collective-based data parallelism                    | ✔ Supported    |
|                           | Tensor Parallelism (TP)         | Intra-layer tensor sharding                          | ✔ Supported    |
|                           | Sequence Parallelism (SP)       | Sequence dimension sharding                          | ✔ Supported    |
|                           | Pipeline Parallelism (PP)       | GPipe, 1F1B scheduling, Virtual Pipeline (vPP)       | ✔ Supported    |
|                           | Hybrid Parallelism              | Arbitrary combination of DDP + TP + SP + PP          | ✔ Supported    |
| Core Components           | Multi-backend                   | CPU and CUDA execution backends                      | ✔ Supported    |
|                           | Multi-node Distributed Training | Distributed execution across multiple nodes          | ✔ Supported    |
|                           | Kernel Dispatcher               | Kernel registration and dynamic dispatch mechanism   | ✔ Supported    |
|                           | Autograd                        | Automatic differentiation engine                     | ✔ Supported    |
|                           | Autocast                        | Automatic mixed precision runtime                    | ✔ Supported    |
| Performance Optimizations | Compute–Comm Overlap            | Explicit scheduling to hide communication latency    | ✔ Supported    |
|                           | DDP Gradient Bucketing          | Deferred and bucketed gradient synchronization       | ✔ Supported    |
|                           | Flash Attention 2               | Memory-efficient exact attention (2-6x speedup)      | ✔ Supported    |
|                           | ZeRO-DP                         | DistributedOptimizer-based ZeRO-1                    | 🚧 In Progress |
| Execution Mode            | Training Mode                   | Full forward–backward training with autograd         | ✔ Supported    |
|                           | `no_grad` Inference             | Forward-only execution without gradient tracking     | ✔ Supported    |
| Debugging & Tooling       | Built-in Profiler               | Kernel-level performance profiling                   | ✔ Supported    |
|                           | Automated Benchmarking          | One-click execution, log analysis and Feishu export  | ✔ Supported    |

## 🏋️ Training

Each model in the `example/` directory is compiled into an independent executable.  
For example, the `llama3` example produces a binary named `llama3`.

To view available runtime options:

```bash
./llama3 --help
```

### Getting Started

The following examples demonstrate **LLaMA 3 supervised fine-tuning (SFT)** using InfiniTrain.

#### Single-node Training Example

```bash
./llama3 \
  --device cuda \
  --input_bin [training_data_path] \
  --llmc_filepath [model_path] \
  --num_iteration 10

```

#### Flash Attention 2 Example

Enable Flash Attention 2 for memory-efficient attention:

```bash
./llama3 \
  --device cuda \
  --input_bin [training_data_path] \
  --llmc_filepath [model_path] \
  --num_iteration 10 \
  --flash=true \
  --dtype=bfloat16
```

> **Note**: Flash Attention 2 requires CUDA and works best with BF16/FP16 precision on Ampere (A100) or newer GPUs.
>
> **Current Status**: The optimized version has a known NaN issue under investigation. Use the naive version for production workloads. See [Known Issues](#-known-issues) for details.

#### Multi-nodes Training Example (3D parallel)

```bash
./infini_run \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=[rank_id] \
  -- ./llama3 \
     --device cuda \
     --input_bin [training_data_path] \
     --llmc_filepath [model_path] \
     --num_iteration 10 \
     --nthread_per_process 8 \
     --batch_size 40 \
     --total_batch_size 10240 \
     --tensor_parallel 2 \
     --pipeline_parallel 2 \
     --sequence_parallel
```

### Parallelism Strategies

#### Distributed Data Parallelism (DDP)

```bash
--nthread_per_process 8 	# ddp_size = nthread_per_process / (tensor_parallel × pipeline_parallel)
```

#### Tensor Parallelism (TP)

```bash
--tensor_parallel 4        # 4-way tensor parallelism
--sequence_parallel        # Enable sequence parallelism (requires TP > 1)
```

#### Pipeline Parallelism (PP)

```bash
--pipeline_parallel 8     		# 8 pipeline stages
--virtual_pipeline_parallel 4  	# Virtual pipeline for better load balancing
```

#### Combining Parallelism Strategies

Multiple parallelism strategies (DDP, TP, SP, PP) can be freely combined to scale training across devices and nodes.

## 🗺 Roadmap

- **2025/03/10** — InfiniTrain **v0.1.0**

  Initial framework prototype with MNIST CPU training.

- **2025/04/30** — InfiniTrain **v0.3.0**

  Added Autograd support and GPT-2 training on CPU/CUDA.

- **2025/07/09** — InfiniTrain **v0.4.0**

  Introduced kernel registration, LLaMA training on CPU/CUDA, BF16 precision, and Data Parallelism.

- **2025/12/31** — InfiniTrain **v0.5.0**

  Added Autocast, multi-dimensional distributed parallelism
   (DDP, TP, SP, PP with GPipe / 1F1B / vPP),
   multi-node training, `no_grad` mode,
   and communication–computation overlap with bucketed gradient synchronization.
## ⚠️ Known Issues

### Flash Attention 2

#### Optimized Version NaN Bug (High Priority)

**Status**: 🐛 Under Investigation

**Description**: The optimized Flash Attention kernel produces NaN values during forward propagation due to missing numerical stability checks.

**Impact**: Training fails with NaN loss when using the optimized version.

**Workaround**: Use the naive version by setting `USE_OPTIMIZED_FLASH_ATTENTION=0` during compilation:

```bash
cmake .. -DUSE_CUDA=ON -DENABLE_FLASH_ATTENTION=ON \
  -DCMAKE_CXX_FLAGS="-DUSE_OPTIMIZED_FLASH_ATTENTION=0"
```

**Root Cause**: Missing NaN protection mechanisms compared to official implementation:
- No `-INFINITY` check in exp calculations
- No NaN detection in sum operations
- `blockReduceSum` template parameter mismatch

**Fix Plan**: Add numerical stability checks following the official Flash Attention implementation. See [NAN_HANDLING_ANALYSIS.md](NAN_HANDLING_ANALYSIS.md) for detailed analysis.

#### Other Limitations

- **FP32 Not Supported**: Flash Attention 2 only supports BF16 and FP16. Using FP32 will cause runtime errors.
- **attn_mask Not Implemented**: The `attn_mask` parameter in `ScaledDotProductAttention` is accepted but not used.
- **GQA Not Implemented**: Grouped Query Attention (GQA) support is not yet implemented.
- **window_size Not Implemented**: Sliding window attention is not yet implemented.

## 📚 Documentation

### Core Documentation

- **[README.md](README.md)** - This file, project overview and quick start guide
- **[FLASH_ATTENTION_REPORT.md](FLASH_ATTENTION_REPORT.md)** - Complete experimental report with performance benchmarks
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation summary and optimized version analysis
- **[CODE_REVIEW.md](CODE_REVIEW.md)** - Code review report, framework compliance check
- **[NAN_HANDLING_ANALYSIS.md](NAN_HANDLING_ANALYSIS.md)** - NaN issue analysis and fix proposals

### Framework Documentation

- **[docs/hook_mechanism.md](docs/hook_mechanism.md)** - Hook mechanism guide
- **[docs/precision_checker_guide.md](docs/precision_checker_guide.md)** - Precision checker guide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
