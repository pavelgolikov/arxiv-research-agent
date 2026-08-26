# Claim support labels

A uniform random sample of **40 of 165** citations from the committed groundedness runs.

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

### 1. `2511.14465v2:p7:c0` — experimental_setup

- **Claim:** The library supports and has been tested with various specific model classes, including GPT-2, LLaMA, Gemma, Bloom, and Mistral.
- **Excerpt:** “The following model classes were tested and work with nnterp: • BloomForCausalLM • BloomModel • Ernie4_5_MoeForCausalLM • GPT2LMHeadModel • GPTBigCodeForCausalLM • GPTJForCausalLM • Gemma2ForCausalLM • Gemma3ForCausalLM • Gemma3ForConditionalGeneration • GemmaForCausalLM • Glm4ForCausalLM”
- **Page:** [2511.14465v2 p. 7](https://arxiv.org/pdf/2511.14465v2#page=7)
- **Grade (2 / 1 / 0):** `1`

### 2. `2511.14465v2:p4:c2` — limitations

- **Claim:** The library is incompatible with certain attention implementations, such as Flash Attention for attention probabilities.
- **Excerpt:** “The library also inherits NNsight’s limitations, including incompatibility with some attention implementations (e.g., Flash Attention for attention probabilities).”
- **Page:** [2511.14465v2 p. 4](https://arxiv.org/pdf/2511.14465v2#page=4)
- **Grade (2 / 1 / 0):** `2`

### 3. `2511.14465v2:p1:c1` — method

- **Claim:** The authors propose nnterp, a library that enables mechanistic interpretability by providing a standardized interface for transformer models.
- **Excerpt:** “original HuggingFace implementations. Through automatic module renaming and
comprehensive validation testing, nnterp enables researchers to write interven-
tion code once and deploy it across 50+ model variants spanning 16 architecture
families.”
- **Page:** [2511.14465v2 p. 1](https://arxiv.org/pdf/2511.14465v2#page=1)
- **Grade (2 / 1 / 0):** `2`

### 4. `2511.14465v2:p1:c1` — method

- **Claim:** nnterp includes built-in implementations of common interpretability methods, including logit lens, patchscope, and activation steering.
- **Excerpt:** “The library includes built-in implementations of common interpretabil-
ity methods (logit lens, patchscope, activation steering)”
- **Page:** [2511.14465v2 p. 1](https://arxiv.org/pdf/2511.14465v2#page=1)
- **Grade (2 / 1 / 0):** `2`

### 5. `2511.14465v2:p2:c3` — relevance_to_query

- **Claim:** Early approaches to mechanistic interpretability utilized manual PyTorch hooks to intercept model activations.
- **Excerpt:** “Early mechanistic interpretability relied on manual PyTorch hooks to intercept activations [Elhage et al., 2021].”
- **Page:** [2511.14465v2 p. 2](https://arxiv.org/pdf/2511.14465v2#page=2)
- **Grade (2 / 1 / 0):** `2`

### 6. `2511.14465v2:p2:c2` — research_problem

- **Claim:** The existing fragmentation in interpretability tooling creates friction, forcing researchers to choose between limited toolsets or maintaining multiple codebases for different architectures.
- **Excerpt:** “This fragmentation creates significant friction. Researchers must either commit to a single tool’s limitations or maintain parallel codebases for different architectures.”
- **Page:** [2511.14465v2 p. 2](https://arxiv.org/pdf/2511.14465v2#page=2)
- **Grade (2 / 1 / 0):** `2`

### 7. `2402.03855v2:p1:c3` — limitations

- **Claim:** Current mechanistic interpretability methods may not scale to handle more complex capabilities and vulnerabilities.
- **Excerpt:** “leading to the concern that the current mechanistic interpretability pipeline, especially with expensive, human-generated hypotheses, would simply not scale to reasonably complex capabilities and vulnerabilities.”
- **Page:** [2402.03855v2 p. 1](https://arxiv.org/pdf/2402.03855v2#page=1)
- **Grade (2 / 1 / 0):** `2`

### 8. `2402.03855v2:p4:c1` — limitations

- **Claim:** Token-aligned evaluation and interpretability methods are limited by potential false positives in nuanced or dishonest scenarios.
- **Excerpt:** “In this case, token-aligned evaluation and interpretability (using a first-token-based metric) would just lead to false positives because the honest-looking answer is actually dishonest. This severely restricts the applicability of current methods, especially on nuanced behaviors.”
- **Page:** [2402.03855v2 p. 4](https://arxiv.org/pdf/2402.03855v2#page=4)
- **Grade (2 / 1 / 0):** `2`

### 9. `2402.03855v2:p9:c0` — limitations

- **Claim:** Existing mechanistic interpretability tools are insufficient for answering significant questions regarding model representations, indicating a need for new research frameworks.
- **Excerpt:** “As a case study, we explore linear representations for dishonesty using the current tooling in mechanistic interpretability and show that they do not help answer most of the important questions that arise while studying representations. We posit the need of new frameworks”
- **Page:** [2402.03855v2 p. 9](https://arxiv.org/pdf/2402.03855v2#page=9)
- **Grade (2 / 1 / 0):** `2`

### 10. `2402.03855v2:p9:c0` — research_problem

- **Claim:** The paper argues that studying feature and behavior representations is essential for understanding and controlling AI models.
- **Excerpt:** “In this position paper, we discuss the existing literature to formalize representations and motivate that studying representations for features and behaviors is crucial for understanding models.”
- **Page:** [2402.03855v2 p. 9](https://arxiv.org/pdf/2402.03855v2#page=9)
- **Grade (2 / 1 / 0):** `1`

### 11. `2402.03855v2:p1:c4` — research_problem

- **Claim:** Existing methods for finding linear representations fail to explain how models work and do not provide verifiable interpretations.
- **Excerpt:** “While these are much more complicated behaviors than those for which circuit-level analyses have been successful, this method fails to answer “how a model works” and does not yield concrete, verifiable interpretations.”
- **Page:** [2402.03855v2 p. 1](https://arxiv.org/pdf/2402.03855v2#page=1)
- **Grade (2 / 1 / 0):** `2`

### 12. `2511.09432v2:p3:c4` — experimental_setup

- **Claim:** The models evaluated consist of CNNs, MLPs, and Transformers.
- **Excerpt:** “Then we evaluate on a synthetic dataset with geometric shapes as well as real-world galaxy and cell images (samples in Figure 7 in the Appendix) using 3 types of base models with varying size and complexity: CNNs, MLPs, and Transformers.”
- **Page:** [2511.09432v2 p. 3](https://arxiv.org/pdf/2511.09432v2#page=3)
- **Grade (2 / 1 / 0):** `2`

### 13. `2511.09432v2:p6:c5` — limitations

- **Claim:** Approaches that discover symmetries from data often struggle with scalability.
- **Excerpt:** “Laird et al. (2024) propose the MatrixNet architecture to learn matrix representation of group elements while staying faithful to the group axioms. However, such approaches do not scale well”
- **Page:** [2511.09432v2 p. 6](https://arxiv.org/pdf/2511.09432v2#page=6)
- **Grade (2 / 1 / 0):** `2`

### 14. `2511.09432v2:p4:c4` — main_findings

- **Claim:** Equivariant SAEs outperform baseline SAEs in feature recovery and detection tasks, particularly as interference increases.
- **Excerpt:** “Despite worse reconstructions (first plot), Equivariant SAEs outperform the baseline SAEs on feature recovery and detection as the number of features and thus interference increases.”
- **Page:** [2511.09432v2 p. 4](https://arxiv.org/pdf/2511.09432v2#page=4)
- **Grade (2 / 1 / 0):** `2`

### 15. `2511.09432v2:p12:c2` — main_findings

- **Claim:** Equivariant SAEs achieve better performance on feature recovery and detection tasks while using fewer parameters than baseline SAEs and the group crosscoder.
- **Excerpt:** “Despite worse reconstructions, Equivariant SAEs outperform baselines in feature recovery and detection (especially top-1 detection which is more challenging) with significantly fewer parameters than the wide counterparts of the baseline SAEs as well as the group crosscoder.”
- **Page:** [2511.09432v2 p. 12](https://arxiv.org/pdf/2511.09432v2#page=12)
- **Grade (2 / 1 / 0):** `2`

### 16. `2511.09432v2:p12:c3` — main_findings

- **Claim:** Equivariant SAEs remain accurate in feature recovery and detection when using randomly chosen canonical representatives, though performance slightly decreases.
- **Excerpt:** “While the results vary more compared to the fixed canonical representative setup used in the main paper, the Equivariant SAEs remain accurate in feature recovery and detection despite a slight decrease in performance.”
- **Page:** [2511.09432v2 p. 12](https://arxiv.org/pdf/2511.09432v2#page=12)
- **Grade (2 / 1 / 0):** `2`

### 17. `2511.09432v2:p2:c3` — research_problem

- **Claim:** Accounting for symmetries is difficult for vanilla SAEs because they must disentangle both the feature and the transformation from the model's activations.
- **Excerpt:** “This makes interpretation with SAEs harder, as a vanilla SAE would need to disentangle both the feature and the transformation from the activation.”
- **Page:** [2511.09432v2 p. 2](https://arxiv.org/pdf/2511.09432v2#page=2)
- **Grade (2 / 1 / 0):** `2`

### 18. `2409.13714v1:p4:c2` — limitations

- **Claim:** The evaluation of novel interpretability methods is difficult.
- **Excerpt:** “Evaluating novel interpretability methods is challenging (Casper, 2020).”
- **Page:** [2409.13714v1 p. 4](https://arxiv.org/pdf/2409.13714v1#page=4)
- **Grade (2 / 1 / 0):** `2`

### 19. `2409.13714v1:p4:c1` — main_findings

- **Claim:** For RASP programs, GPT-4o performed best on the difficulty-weighted score metric with a score of 0.31, while GPT-4-turbo achieved 0.29.
- **Excerpt:** “with gpt-4-turbo achieving a score of 0.29 (out of 1.0) and gpt-4o performing best with a score of 0.31.”
- **Page:** [2409.13714v1 p. 4](https://arxiv.org/pdf/2409.13714v1#page=4)
- **Grade (2 / 1 / 0):** `1`

### 20. `2409.13714v1:p4:c1` — main_findings

- **Claim:** Claude-3-5-sonnet achieved a pass rate of 0.45 and a difficulty-weighted score of 0.27.
- **Excerpt:** “Claude-3-5-sonnet has a similar pass rate (0.45) to claude-3-opus (0.46), however, it achieves a higher difficulty-weighted score (0.27), than claude-3-opus (0.23).”
- **Page:** [2409.13714v1 p. 4](https://arxiv.org/pdf/2409.13714v1#page=4)
- **Grade (2 / 1 / 0):** `2`

### 21. `2409.13714v1:p2:c2` — method

- **Claim:** The proposed method involves compiling RASP programs, which are generated by LLMs, into functional transformer weights to create models with known ground truth mappings.
- **Excerpt:** “We create TracrBench, a dataset of 121 RASP programs, by leveraging LLMs and manual annotation when they fail. These programs are then compiled into functional transformer weights using Tracr, resulting in transformer models with a known mapping between weights and their functional form.”
- **Page:** [2409.13714v1 p. 2](https://arxiv.org/pdf/2409.13714v1#page=2)
- **Grade (2 / 1 / 0):** `2`

### 22. `2409.13714v1:p1:c3` — relevance_to_query

- **Claim:** The RASP programming language is used to model and analyze transformer behavior by mapping core components like attention and feed-forward computation into simple primitives.
- **Excerpt:** “Restricted Access Sequence Processing Language (RASP) (Weiss et al., 2021) maps the core components of a transformer-encoder, i.e., attention and feed-forward computation, into simple primitives, forming a programming language to model and analyze transformer behavior.”
- **Page:** [2409.13714v1 p. 1](https://arxiv.org/pdf/2409.13714v1#page=1)
- **Grade (2 / 1 / 0):** `2`

### 23. `2409.13714v1:p1:c0` — research_problem

- **Claim:** Interpretability research is hindered by a lack of models with fully understood internals, specifically ground truth mappings between weights and functional roles.
- **Excerpt:** “Moreover, the lack of ground truth mappings between model weights and their functional roles hinders the effective evaluation of interpretability methods, impeding overall progress.”
- **Page:** [2409.13714v1 p. 1](https://arxiv.org/pdf/2409.13714v1#page=1)
- **Grade (2 / 1 / 0):** `2`

### 24. `2604.24971v1:p4:c1` — experimental_setup

- **Claim:** The models used are SmolLM2-1.7B-Instruct for proof-of-concept and Llama-3-8B-Instruct for primary validation.
- **Excerpt:** “Models.
• SmolLM2-1.7B-Instruct (HuggingFaceTB): proof-of-concept scale, CPU inference.
• Llama-3-8B-Instruct (Meta): primary validation, 32 layers, GQA (8 KV heads), 4-bit NF4
weights, bfloat16 KV cache, Kaggle T4×2.”
- **Page:** [2604.24971v1 p. 4](https://arxiv.org/pdf/2604.24971v1#page=4)
- **Grade (2 / 1 / 0):** `2`

### 25. `2604.24971v1:p4:c1` — experimental_setup

- **Claim:** Datasets used for evaluation include custom documents (Apollo 11 mission, ARPANET/Internet topology history), WikiText-2 2K, and WikiText-2 4K.
- **Excerpt:** “Shared contexts.
• Short context (≈600 tokens): Apollo 11 mission document (SmolLM2-1.7B only).
• Long context (1,851 tokens): ARPANET/Internet topology history”
- **Page:** [2604.24971v1 p. 4](https://arxiv.org/pdf/2604.24971v1#page=4)
- **Grade (2 / 1 / 0):** `1`

### 26. `2604.24971v1:p9:c0` — limitations

- **Claim:** The observed perplexity findings require further controlled ablation to confirm them causally.
- **Excerpt:** “The −0.26% finding on SmolLM2-1.7B and the PPL improvement trend on Llama-3-8B are consistent with the regularization hypothesis but require controlled ablation to confirm causally.”
- **Page:** [2604.24971v1 p. 9](https://arxiv.org/pdf/2604.24971v1#page=9)
- **Grade (2 / 1 / 0):** `2`

### 27. `2604.24971v1:p6:c0` — main_findings

- **Claim:** The PolyKV compression achieves a 2.91× compression ratio that remains consistent across different model architectures like SmolLM2-1.7B and Llama-3-8B.
- **Excerpt:** “The 2.91× compression ratio is identical across SmolLM2-1.7B and Llama-3-8B, confirming it is a mathematical property of the compression scheme rather than a model-specific artifact.”
- **Page:** [2604.24971v1 p. 6](https://arxiv.org/pdf/2604.24971v1#page=6)
- **Grade (2 / 1 / 0):** `2`

### 28. `2604.24971v1:p9:c1` — method

- **Claim:** The authors propose PolyKV, a shared, asymmetrically-compressed KV cache pool designed for multi-agent LLM inference.
- **Excerpt:** “We presented PolyKV, a shared asymmetrically-compressed KV cache pool for multi-agent LLM inference.”
- **Page:** [2604.24971v1 p. 9](https://arxiv.org/pdf/2604.24971v1#page=9)
- **Grade (2 / 1 / 0):** `2`

### 29. `2607.17715v1:p2:c4` — method

- **Claim:** The core component of the C2KV framework is the C2Extractor, which is described as a lightweight sidecar module.
- **Excerpt:** “The core of C2KV is its C2Extractor, a lightweight sidecar mod-”
- **Page:** [2607.17715v1 p. 2](https://arxiv.org/pdf/2607.17715v1#page=2)
- **Grade (2 / 1 / 0):** `2`

### 30. `2607.17715v1:p9:c5` — relevance_to_query

- **Claim:** Merging and low-rank approaches like MiniCache and ReCalKV reduce memory usage by consolidating redundant KV states across attention heads or layers.
- **Excerpt:** “Merging and Low-rank approaches like Mini-Cache [23] or ReCalKV [32] consolidate redundant KV states across layers or attention heads.”
- **Page:** [2607.17715v1 p. 9](https://arxiv.org/pdf/2607.17715v1#page=9)
- **Grade (2 / 1 / 0):** `2`

### 31. `2504.09936v2:p5:c3` — experimental_setup

- **Claim:** The evaluation uses Llama-2, Llama-3, and Mistral as base models, comparing them against cache eviction methods like Streaming, H2O, and PyramidInfer, and cache merging methods like CaM and D2O.
- **Excerpt:** “Our evaluation is based on several representative LLMs, including Llama-2 (Touvron et al. 2023b), Llama-3 (Grattafiori, Dubey et al. 2024), and Mistral (Jiang et al. 2023). We compare our method against multiple baseline approaches: representative cache eviction methods such as Streaming (Xiao et al”
- **Page:** [2504.09936v2 p. 5](https://arxiv.org/pdf/2504.09936v2#page=5)
- **Grade (2 / 1 / 0):** `1`

### 32. `2504.09936v2:p5:c3` — experimental_setup

- **Claim:** The experimental setup involves no model training and uses a merging threshold of 0.8, with token selection and cache allocation following the strategy from PyramidInfer.
- **Excerpt:** “In our main experiments, we set the merging threshold T to 0.8. For token selection and cache allocation, we follow the strategy recommended by PyramidInfer (Yang et al. 2024a)”
- **Page:** [2504.09936v2 p. 5](https://arxiv.org/pdf/2504.09936v2#page=5)
- **Grade (2 / 1 / 0):** `1`

### 33. `2504.09936v2:p6:c2` — main_findings

- **Claim:** KeepKV outperforms existing state-of-the-art KV-cache merging methods like D2O in throughput and methods like CaM and D2O in accuracy.
- **Excerpt:** “Furthermore, KeepKV also surpasses existing KV-cache merging methods, like CaM(Zhang et al. 2024) and D2O(Wan et al. 2024b)”
- **Page:** [2504.09936v2 p. 6](https://arxiv.org/pdf/2504.09936v2#page=6)
- **Grade (2 / 1 / 0):** `1`

### 34. `2504.09936v2:p9:c5` — research_problem

- **Claim:** Prior research on KV cache merging lacks systematic theoretical foundations for selecting merging candidates and assigning merging weights.
- **Excerpt:** “Despite this, key challenges such as the selection of merging candidates and the assignment of merging weights remain largely unexplored, with a lack of systematic theoretical foundations.”
- **Page:** [2504.09936v2 p. 9](https://arxiv.org/pdf/2504.09936v2#page=9)
- **Grade (2 / 1 / 0):** `2`

### 35. `2405.12532v2:p7:c3` — experimental_setup

- **Claim:** The study evaluates performance using the LLaMA 2-13B and 70B models on tasks including GSM8K, MMLU, and the LEval benchmark for long context capabilities.
- **Excerpt:** “In the LEval that tests the long context ability, we show that the "local" strategy that is similar to the technique used in StreamingLLM causes a huge decline in memorization of history.”
- **Page:** [2405.12532v2 p. 7](https://arxiv.org/pdf/2405.12532v2#page=7)
- **Grade (2 / 1 / 0):** `1`

### 36. `2405.12532v2:p7:c2` — experimental_setup

- **Claim:** Baselines used for efficiency and performance comparisons include full cache methods like Accelerate and DeepSpeed, the H2O KV cache compression method, and a "local" strategy.
- **Excerpt:** “We compare the efficiency of PyramidInfer with other full cache methods, including Accelerate (HuggingFace, 2021), Deepspeed2 (Aminabadi et al., 2022). We also select H2O3 (Zhang et al., 2023), a KV cache compression method, as another baseline.”
- **Page:** [2405.12532v2 p. 7](https://arxiv.org/pdf/2405.12532v2#page=7)
- **Grade (2 / 1 / 0):** `1`

### 37. `2405.12532v2:p1:c1` — main_findings

- **Claim:** PyramidInfer achieves a 2.2x improvement in throughput compared to Accelerate.
- **Excerpt:** “Experimental results show PyramidInfer improves 2.2x throughput compared to Accelerate with over 54% GPU memory reduction in KV cache.”
- **Page:** [2405.12532v2 p. 1](https://arxiv.org/pdf/2405.12532v2#page=1)
- **Grade (2 / 1 / 0):** `2`

### 38. `2405.12532v2:p1:c1` — main_findings

- **Claim:** The number of influential keys and values decreases as layers progress.
- **Excerpt:** “we find that the number of crucial keys and values that influence future generations decreases layer by layer”
- **Page:** [2405.12532v2 p. 1](https://arxiv.org/pdf/2405.12532v2#page=1)
- **Grade (2 / 1 / 0):** `2`

### 39. `2405.12532v2:p5:c3` — relevance_to_query

- **Claim:** PyramidInfer can compress the KV cache in both the prefill and generation phases.
- **Excerpt:** “PyramidInfer can not only
reduce the KV cache in the generation phase but
also in the prefill phase”
- **Page:** [2405.12532v2 p. 5](https://arxiv.org/pdf/2405.12532v2#page=5)
- **Grade (2 / 1 / 0):** `2`

### 40. `2405.12532v2:p1:c2` — research_problem

- **Claim:** Large language models face a significant challenge due to the immense GPU memory usage required for inference, which hinders their deployment at scale.
- **Excerpt:** “However, these large models meet up with a substantial challenge of immense GPU memory usage in the inference, due to the model and computational complexity. This hinders deploying LLMs at scale to meet the thousands of demands for chatting with chatbots.”
- **Page:** [2405.12532v2 p. 1](https://arxiv.org/pdf/2405.12532v2#page=1)
- **Grade (2 / 1 / 0):** `2`
