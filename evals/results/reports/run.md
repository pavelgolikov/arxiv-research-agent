# Literature Review: KV cache memory reduction in LLM inference

## Search Summary
The research on KV cache memory optimization focuses on mitigating the memory-bound nature of long-context large language model (LLM) inference. The literature explores techniques ranging from quantization and multi-agent sharing to sophisticated cache merging, eviction, and pyramid-style compression.

## Method and Limitations Notice
This review synthesizes findings from four peer-reviewed papers. A key limitation across the surveyed literature is the inherent trade-off between memory footprint and generation accuracy. While some methods claim "lossless" characteristics, others acknowledge that compression introduces performance degradation or computational overhead. Furthermore, many techniques lack standardized benchmarking for system-level metrics like end-to-end latency or time-to-first-token.

## Overview
As LLM context lengths grow, the KV cache has become a primary memory bottleneck [p. 1](https://arxiv.org/pdf/2604.24971v1#page=1), [p. 3](https://arxiv.org/pdf/2607.17715v1#page=3). Because this cache scales linearly with context length, developers are increasingly turning to three primary optimization categories: 
1. **Sharing & Reuse**: Exploiting common prefix or context redundancy across agents [p. 1](https://arxiv.org/pdf/2604.24971v1#page=1), [p. 2](https://arxiv.org/pdf/2607.17715v1#page=2).
2. **Merging & Eviction**: Selectively keeping important tokens or consolidating KV states [p. 1](https://arxiv.org/pdf/2504.09936v2#page=1), [p. 2](https://arxiv.org/pdf/2405.12532v2#page=2).
3. **Quantization**: Reducing the bit-precision of stored Keys and Values [p. 1](https://arxiv.org/pdf/2604.24971v1#page=1).

## Key Papers
* **[PolyKV](https://arxiv.org/abs/2604.24971v1)**: Introduces a multi-agent system that shares a single, asymmetrically compressed KV cache pool, achieving up to 97.7% memory reduction for 15 concurrent agents [p. 2](https://arxiv.org/pdf/2604.24971v1#page=2).
* **[C$^2$KV](https://arxiv.org/abs/2607.17715v1)**: Focuses on non-prefix KV reuse using a lightweight "sidecar" extractor to create a composable, position-agnostic KV cache manifold [p. 2](https://arxiv.org/pdf/2607.17715v1#page=2).
* **[KeepKV](https://arxiv.org/abs/2504.09936v2)**: Provides a theoretically grounded merging method that uses "Electoral Votes" and "Zero Inference-Perturbation Merging" (ZIP-Merging) to reduce cache size while maintaining output quality [p. 1](https://arxiv.org/pdf/2504.09936v2#page=1).
* **[PyramidInfer](https://arxiv.org/abs/2405.12532v2)**: Implements layer-wise compression by identifying that the number of crucial tokens decreases as layers progress, saving over 54% of memory [p. 1](https://arxiv.org/pdf/2405.12532v2#page=1).

## Comparison Table

| Technique | Approach Type | Primary Benefit | Notable Metric |
| :--- | :--- | :--- | :--- |
| **PolyKV** | Sharing / Compression | Multi-agent efficiency | 97.7% memory reduction [p. 2](https://arxiv.org/pdf/2604.24971v1#page=2) |
| **C$^2$KV** | Reuse / Compression | Long-context throughput | Up to 17x speedup [p. 1](https://arxiv.org/pdf/2607.17715v1#page=1) |
| **KeepKV** | Merging | Accuracy preservation | 2x throughput increase [p. 6](https://arxiv.org/pdf/2504.09936v2#page=6) |
| **PyramidInfer** | Eviction / Pruning | Layer-wise efficiency | 54% memory reduction [p. 1](https://arxiv.org/pdf/2405.12532v2#page=1) |

## Research Themes
* **Importance-Aware Compression**: Modern research increasingly moves away from uniform pruning toward context-aware selection (e.g., PyramidInfer’s "Pivotal Context" and KeepKV’s adaptive merging) [p. 2](https://arxiv.org/pdf/2405.12532v2#page=2), [p. 1](https://arxiv.org/pdf/2504.09936v2#page=1).
* **Asymmetric Optimization**: Recognizing that Keys and Values serve different roles, PolyKV applies different quantization depths (int8 for Keys, 3-bit for Values) to maintain stability [p. 2](https://arxiv.org/pdf/2604.24971v1#page=2).
* **Addressing the Prefill Bottleneck**: While many earlier works focused on generation-time eviction, newer methods like PyramidInfer specifically target the redundant computation occurring during the prefill phase [p. 2](https://arxiv.org/pdf/2405.12532v2#page=2).

## Research Gaps
* **Standardized System Benchmarking**: There is a noted lack of reporting for end-to-end latency and time-to-first-token in many studies, deferring these to future work [p. 9](https://arxiv.org/pdf/2604.24971v1#page=9).
* **Hardware-Specific Constraints**: Techniques like KeepKV face compatibility issues with standard acceleration libraries like FlashAttention because they require access to intermediate attention scores [p. 14](https://arxiv.org/pdf/2504.09936v2#page=14).
* **Adaptive Strategies**: Most existing methods use uniform compression or static thresholds, lacking fully adaptive, real-time mechanisms that respond to varying document complexity [p. 12](https://arxiv.org/pdf/2607.17715v1#page=12).

## Suggested Reading Order
1. **PyramidInfer** [p. 1](https://arxiv.org/pdf/2405.12532v2#page=1): Provides a strong introduction to the concept of layer-wise redundancy.
2. **KeepKV** [p. 1](https://arxiv.org/pdf/2504.09936v2#page=1): Explains the mathematical and logical challenges of merging KV states.
3. **PolyKV** [p. 1](https://arxiv.org/pdf/2604.24971v1#page=1): Illustrates the power of sharing common caches in multi-agent environments.
4. **C$^2$KV** [p. 1](https://arxiv.org/pdf/2607.17715v1#page=1): Covers more advanced concepts of composable and position-agnostic cache manifolds for long-context applications.
