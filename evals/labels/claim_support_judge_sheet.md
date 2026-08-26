# Claim support labels — judge evaluation set

**30** further citations from the same runs, used to score the automated support judge in `evals/run_claim_judge.py`.

Some are citations the pipeline really produced. In others the excerpt has been replaced by a different quote from the same paper. Which is which is recorded in `claim_support_judge_key.json` and deliberately not shown here — grade each item on what you read, exactly as in the first sheet.

For each item: read the excerpt, open the page if the excerpt alone is not enough, and grade how well the excerpt supports the claim.

- `2` — the excerpt establishes the claim.
- `1` — the excerpt supports part of the claim, or supports it with a qualifier the quote does not carry.
- `0` — the excerpt does not support the claim.

Judge the **excerpt against the claim**, not whether the claim is true of the paper. A correct statement quoted from the wrong sentence is still a `0`.

Replace the `_` on each Grade line with a digit. Then run:

```bash
python -m evals.build.claim_support --collect
```

---

### 41. `2604.24971v1:p5:c1` — experimental_setup

- **Claim:** Memory savings increase with the number of agents, reaching 88.5% at 3 agents and 97.7% at 15 agents.
- **Excerpt:** “Test 5
Llama-3-8B
1,837
3
WikiText-2 2K
BERTScore
Test 6
Llama-3-8B
1,837
5
WikiText-2 2K
BERTScore
Test 7
Llama-3-8B
1,837
10
WikiText-2 2K
BERTScore
Test 8
Llama-3-8B
7,194
10
WikiText-2 4K
BERTScore
Test 9
Llama-3-8B
7,194
15
WikiText-2 4K
BERTScore”
- **Page:** [2604.24971v1 p. 5](https://arxiv.org/pdf/2604.24971v1#page=5)
- **Grade (2 / 1 / 0):** `0`

### 42. `2607.17715v1:p8:c4` — main_findings

- **Claim:** C2KV consistently maintains performance on large models like Qwen3-14B while enabling substantial KV compression.
- **Excerpt:** “The results show that C2KV consistently maintains strong performance across all tasks while enabling substantial KV compression.”
- **Page:** [2607.17715v1 p. 8](https://arxiv.org/pdf/2607.17715v1#page=8)
- **Grade (2 / 1 / 0):** `1`

### 43. `2511.14465v2:p7:c1` — limitations

- **Claim:** Support for attention probabilities for certain model classes like DbrxForCausalLM and GptOssForCausalLM is currently a work in progress.
- **Excerpt:** “Support for attention probabilities is still a work in progress for the following model classes:
• DbrxForCausalLM
• GptOssForCausalLM”
- **Page:** [2511.14465v2 p. 7](https://arxiv.org/pdf/2511.14465v2#page=7)
- **Grade (2 / 1 / 0):** `2`

### 44. `2511.09432v2:p3:c4` — experimental_setup

- **Claim:** The study uses the Shapes synthetic dataset, as well as the GalaxyMNIST and MLL23 real-world datasets.
- **Excerpt:** “Then we evaluate on a synthetic dataset with geometric shapes as well as real-world galaxy and cell images (samples in Figure 7 in the Appendix) using 3 types of base models with varying size and complexity: CNNs, MLPs, and Transformers.”
- **Page:** [2511.09432v2 p. 3](https://arxiv.org/pdf/2511.09432v2#page=3)
- **Grade (2 / 1 / 0):** `1`

### 45. `2604.24971v1:p9:c0` — limitations

- **Claim:** The prefill attention computation on Kaggle T4x2 hardware is limited by an out-of-memory error for Llama-3-8B when context lengths exceed approximately 8,000 tokens.
- **Excerpt:** “On Kaggle T4×2 hardware, the prefill attention computation OOMs beyond ≈8,000 tokens for Llama-3-8B. This is a hardware constraint, not a PolyKV limitation”
- **Page:** [2604.24971v1 p. 9](https://arxiv.org/pdf/2604.24971v1#page=9)
- **Grade (2 / 1 / 0):** `2`

### 46. `2511.14465v2:p7:c1` — limitations

- **Claim:** Researchers use interpretability methods such as the logit lens, patchscopes, and activation steering to analyze transformer internals.
- **Excerpt:** “Support for attention probabilities is still a work in progress for the following model classes:
• DbrxForCausalLM
• GptOssForCausalLM”
- **Page:** [2511.14465v2 p. 7](https://arxiv.org/pdf/2511.14465v2#page=7)
- **Grade (2 / 1 / 0):** `0`

### 47. `2511.09432v2:p1:c0` — relevance_to_query

- **Claim:** Experiments were conducted on a single NVIDIA RTX 3090 GPU with 24GB of memory.
- **Excerpt:** “In particular, their activations entangle many concepts into fewer dimensions, a phenomenon known as superposition. Mechanistic interpretability methods such as sparse autoencoders (SAEs) can disentangle these dense activations into sparse sums of interpretable features”
- **Page:** [2511.09432v2 p. 1](https://arxiv.org/pdf/2511.09432v2#page=1)
- **Grade (2 / 1 / 0):** `0`

### 48. `2511.14465v2:p1:c1` — relevance_to_query

- **Claim:** As mechanistic interpretability scales to larger models, tools that provide both correct implementation and interface usability are required for robust, reproducible research.
- **Excerpt:** “Mechanistic interpretability research aims to reverse-engineer the computational mechanisms within neural networks [Elhage et al., 2021, Olah et al., 2020]. For transformer language models, this requires tools that can reliably access and modify internal representations across diverse architectures.”
- **Page:** [2511.14465v2 p. 1](https://arxiv.org/pdf/2511.14465v2#page=1)
- **Grade (2 / 1 / 0):** `1`

### 49. `2604.24971v1:p9:c1` — relevance_to_query

- **Claim:** The PolyKV system uses a shared asymmetrically-compressed KV cache pool, writing a single compressed cache and distributing it to concurrent agents.
- **Excerpt:** “PolyKV writes a single compressed cache (K at int8, V at TurboQuant MSE 3-bit) once and distributes it to N concurrent agents via direct DynamicCache injection, achieving a stable 2.91× memory reduction”
- **Page:** [2604.24971v1 p. 9](https://arxiv.org/pdf/2604.24971v1#page=9)
- **Grade (2 / 1 / 0):** `2`

### 50. `2511.09432v2:p1:c3` — research_problem

- **Claim:** The study uses the Shapes synthetic dataset, as well as the GalaxyMNIST and MLL23 real-world datasets.
- **Excerpt:** “In this paper, our main research question is: How should mechanistic interpretability methods such as SAEs account for symmetries, and what benefits does doing so provide?”
- **Page:** [2511.09432v2 p. 1](https://arxiv.org/pdf/2511.09432v2#page=1)
- **Grade (2 / 1 / 0):** `0`

### 51. `2402.03855v2:p1:c1` — relevance_to_query

- **Claim:** Directly unembedding the dishonesty direction results in a probability distribution with arbitrary top-k tokens, indicating that these representations are not naively used by the model.
- **Excerpt:** “Tools such as saliency maps (Simonyan et al., 2013), the logit lens (nostalgebraist, 2020), and activation and path patching (Wang et al., 2022) have helped us understand how simple functions and capabilities are implemented inside neural network models.”
- **Page:** [2402.03855v2 p. 1](https://arxiv.org/pdf/2402.03855v2#page=1)
- **Grade (2 / 1 / 0):** `0`

### 52. `2504.09936v2:p6:c3` — main_findings

- **Claim:** Merging-based strategies reduce memory usage by combining KV pairs.
- **Excerpt:** “Nonetheless, KeepKV achieves higher throughput than the state-of-the-art (SOTA) merging-based algorithm, D2O (Wan et al. 2024b).”
- **Page:** [2504.09936v2 p. 6](https://arxiv.org/pdf/2504.09936v2#page=6)
- **Grade (2 / 1 / 0):** `0`

### 53. `2607.17715v1:p2:c4` — method

- **Claim:** The C2KV framework utilizes a position-agnostic, Composable and Compressed KV cache manifold to allow KV segments to be used at arbitrary positions within long-context prompts.
- **Excerpt:** “Unlike prior methods that directly compress or cache the original KV states, which are inherently context-dependent, C2KV constructs a Composable and Compressed KV cache manifold. This manifold is specifically designed to be position-agnostic”
- **Page:** [2607.17715v1 p. 2](https://arxiv.org/pdf/2607.17715v1#page=2)
- **Grade (2 / 1 / 0):** `2`

### 54. `2607.17715v1:p12:c4` — limitations

- **Claim:** Eviction strategies like SnapKV and RocketKV reduce KV cache size by identifying and keeping tokens based on attention scores or importance.
- **Excerpt:** “First, C2KV focuses on document-level reuse and assumes that reusable content can be identified and extracted offline. Extending the extractor to support online or incremental KV extraction is an important direction.”
- **Page:** [2607.17715v1 p. 12](https://arxiv.org/pdf/2607.17715v1#page=12)
- **Grade (2 / 1 / 0):** `0`

### 55. `2511.14465v2:p2:c2` — research_problem

- **Claim:** Future work includes adding support for non-causal and encoder-decoder architectures, as well as automated architecture detection.
- **Excerpt:** “This fragmentation creates significant friction. Researchers must either commit to a single tool’s limitations or maintain parallel codebases for different architectures.”
- **Page:** [2511.14465v2 p. 2](https://arxiv.org/pdf/2511.14465v2#page=2)
- **Grade (2 / 1 / 0):** `0`

### 56. `2511.09432v2:p1:c0` — relevance_to_query

- **Claim:** Mechanistic interpretability methods are used to disentangle activations that have compressed many concepts into fewer dimensions, a phenomenon called superposition.
- **Excerpt:** “In particular, their activations entangle many concepts into fewer dimensions, a phenomenon known as superposition. Mechanistic interpretability methods such as sparse autoencoders (SAEs) can disentangle these dense activations into sparse sums of interpretable features”
- **Page:** [2511.09432v2 p. 1](https://arxiv.org/pdf/2511.09432v2#page=1)
- **Grade (2 / 1 / 0):** `2`

### 57. `2405.12532v2:p5:c3` — relevance_to_query

- **Claim:** PyramidInfer can reduce GPU memory usage in the KV cache by over 54%.
- **Excerpt:** “PyramidInfer can not only
reduce the KV cache in the generation phase but
also in the prefill phase”
- **Page:** [2405.12532v2 p. 5](https://arxiv.org/pdf/2405.12532v2#page=5)
- **Grade (2 / 1 / 0):** `1`

### 58. `2409.13714v1:p1:c0` — research_problem

- **Claim:** The authors introduce TracrBench, an approach for generating interpretability testbeds by using large language models to generate RASP programs.
- **Excerpt:** “Moreover, the lack of ground truth mappings between model weights and their functional roles hinders the effective evaluation of interpretability methods, impeding overall progress.”
- **Page:** [2409.13714v1 p. 1](https://arxiv.org/pdf/2409.13714v1#page=1)
- **Grade (2 / 1 / 0):** `0`

### 59. `2402.03855v2:p1:c0` — relevance_to_query

- **Claim:** The researchers investigated honesty by calculating the first principal component of the difference of activation vectors across layers for a given dataset.
- **Excerpt:** “Mechanistic interpretability (MI) aims to understand AI models by reverse-engineering the exact algorithms neural networks learn.”
- **Page:** [2402.03855v2 p. 1](https://arxiv.org/pdf/2402.03855v2#page=1)
- **Grade (2 / 1 / 0):** `0`

### 60. `2402.03855v2:p5:c3` — main_findings

- **Claim:** The hypothesis that continual dishonesty injections are unnecessary for open-ended dishonesty generation is disproven, as shown by comparing output log probabilities with and without injection.
- **Excerpt:** “We test this by comparing the models’ output log probabilities at every token position both with and without the injection. Fig. 4 disproves the hypothesis and shows”
- **Page:** [2402.03855v2 p. 5](https://arxiv.org/pdf/2402.03855v2#page=5)
- **Grade (2 / 1 / 0):** `1`

### 61. `2511.14465v2:p3:c2` — method

- **Claim:** The library provides accessor methods for internal activations that allow researchers to get and set inputs and outputs for layers, MLPs, and attention modules.
- **Excerpt:** “nnterp also provides model.{layers/mlps/attentions}_input/output[layer_idx] which
allow to get and set the input and output of the specified layer.”
- **Page:** [2511.14465v2 p. 3](https://arxiv.org/pdf/2511.14465v2#page=3)
- **Grade (2 / 1 / 0):** `2`

### 62. `2409.13714v1:p4:c0` — main_findings

- **Claim:** Mechanistic interpretability aims to achieve an understanding of transformer-based models, though this is difficult due to their high number of parameters and the lack of ground truth mappings between model weights and functional roles.
- **Excerpt:** “In comparison, when generating Python programs for the target algorithms, gpt-4-turbo achieves a pass rate of 96%.”
- **Page:** [2409.13714v1 p. 4](https://arxiv.org/pdf/2409.13714v1#page=4)
- **Grade (2 / 1 / 0):** `0`

### 63. `2405.12532v2:p2:c2` — research_problem

- **Claim:** PyramidInfer reduces KV cache memory usage by layer-wise selecting and retaining crucial context.
- **Excerpt:** “We call this Inference Context Redundancy (ICR) hypothesis. It inspires us to compress the KV cache by only computing the keys and values that record the context information.”
- **Page:** [2405.12532v2 p. 2](https://arxiv.org/pdf/2405.12532v2#page=2)
- **Grade (2 / 1 / 0):** `1`

### 64. `2402.03855v2:p9:c2` — experimental_setup

- **Claim:** To evaluate feature representations, the authors utilize the method established by Marks & Tegmark (2023) to compute linear directions for truthfulness.
- **Excerpt:** “We would also like to thank Neel Nanda and Joseph Bloom for building and maintaining the TransformerLens (Nanda & Bloom, 2022) library that was used for several experiments in this work.”
- **Page:** [2402.03855v2 p. 9](https://arxiv.org/pdf/2402.03855v2#page=9)
- **Grade (2 / 1 / 0):** `0`

### 65. `2511.14465v2:p4:c3` — research_problem

- **Claim:** Mechanistic interpretability research faces a tradeoff where custom implementations provide consistent interfaces but risk numerical mismatch, while direct access methods lack standardization across architectures.
- **Excerpt:** “As mechanistic interpretability scales to larger models and broader architectural diversity, tools that balance correctness with usability become essential. nnterp represents a step toward more robust and reproducible interpretability research.”
- **Page:** [2511.14465v2 p. 4](https://arxiv.org/pdf/2511.14465v2#page=4)
- **Grade (2 / 1 / 0):** `1`

### 66. `2402.03855v2:p1:c1` — relevance_to_query

- **Claim:** The TransformerLens library was utilized to perform several experiments in the study.
- **Excerpt:** “Elhage et al. (2021) discovered specialized attention heads called “induction heads” that are found to play an important role in the emergence of in-context learning in language models”
- **Page:** [2402.03855v2 p. 1](https://arxiv.org/pdf/2402.03855v2#page=1)
- **Grade (2 / 1 / 0):** `0`

### 67. `2604.24971v1:p2:c0` — method

- **Claim:** PolyKV utilizes asymmetric compression, specifically using q8_0 for Keys and FWHT combined with Lloyd-Max 3-bit quantization for Values.
- **Excerpt:** “Asymmetric TurboQuant MSE compression in a shared pool. q8_0 for Keys and FWHT+Lloyd-Max 3-bit for Values, achieving 2.91× compression”
- **Page:** [2604.24971v1 p. 2](https://arxiv.org/pdf/2604.24971v1#page=2)
- **Grade (2 / 1 / 0):** `2`

### 68. `2504.09936v2:p1:c4` — research_problem

- **Claim:** Existing KV cache merging methods lack solid theoretical foundations and inevitably lead to output perturbation, also known as 'Attention Sag'.
- **Excerpt:** “Nevertheless, these methods vary widely in merge candidate selection and merging weight computation, and lack solid theoretical foundations. We observe that existing strategies inevitably induce attention inconsistencies and output perturbation.”
- **Page:** [2504.09936v2 p. 1](https://arxiv.org/pdf/2504.09936v2#page=1)
- **Grade (2 / 1 / 0):** `1`

### 69. `2607.17715v1:p3:c5` — research_problem

- **Claim:** Evaluation is conducted on LongBench datasets including HotpotQA, 2WikiMQA, MuSiQue, MultiNews, SAMSum, QMSum, and GovReport.
- **Excerpt:** “As illustrated in Figure 3, existing reuse methods reduce prefill cost by loading cached KVs from memory, but still do not address this storage pressure”
- **Page:** [2607.17715v1 p. 3](https://arxiv.org/pdf/2607.17715v1#page=3)
- **Grade (2 / 1 / 0):** `0`

### 70. `2607.17715v1:p2:c4` — method

- **Claim:** C2KV improves both request startup latency and steady-state generation efficiency.
- **Excerpt:** “Unlike prior methods that directly compress or cache the original KV states, which are inherently context-dependent, C2KV constructs a Composable and Compressed KV cache manifold. This manifold is specifically designed to be position-agnostic”
- **Page:** [2607.17715v1 p. 2](https://arxiv.org/pdf/2607.17715v1#page=2)
- **Grade (2 / 1 / 0):** `0`
