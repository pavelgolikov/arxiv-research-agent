# Hand-verification sheet — `judged-example-2`

76 citations across 4 papers, from the run that produced
`examples/example_review.md`. Every one already passed the three deterministic checks
and carries a judge grade. What is left is reading the excerpt against the page.

Grade each as: **OK** the excerpt establishes the claim · **WEAK** it supports part of
it · **NO** it does not support it. Mark anything where the page does not say what the
excerpt says at all — that would be a validation failure, not a judgment call.

## Start here — the 8 the judge itself called partial

These are where the judge and a reader are most likely to disagree, so they carry the
most information per minute spent.

- **#2** (2511.14465v2, `experimental_setup`, p. 7)
- **#29** (2402.03855v2, `method`, p. 5)
- **#42** (2409.13714v1, `main_findings`, p. 4)
- **#47** (2409.13714v1, `method`, p. 2)
- **#56** (2606.16939v1, `experimental_setup`, p. 18)
- **#68** (2606.16939v1, `method`, p. 1)
- **#72** (2606.16939v1, `relevance_to_query`, p. 4)
- **#76** (2606.16939v1, `research_problem`, p. 4)

---

## 2511.14465v2 — nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers

PDF: https://arxiv.org/pdf/2511.14465v2 · 20 citations

### #1 · `experimental_setup` · judge: establishes

**Claim.** The nnterp library provides a unified interface across more than 50 model variants from 16 different architecture families.

**Excerpt.** “A unified API for accessing transformer internals (layers, attention, MLP outputs) that works identically across 50+ model variants from 16 architecture families.”

**Check.** [p. 2](https://arxiv.org/pdf/2511.14465v2#page=2) · `2511.14465v2:p2:c2`

**Verdict.** OK

### #2 · `experimental_setup` · judge: partial

**Claim.** The library has been tested to work with various model architectures including Bloom, GPT, Gemma, Llama, Mistral, and Mixtral.

**Excerpt.** “The following model classes were tested and work with nnterp: • BloomForCausalLM • BloomModel • Ernie4_5_MoeForCausalLM • GPT2LMHeadModel • GPTBigCodeForCausalLM • GPTJForCausalLM • Gemma2ForCausalLM • Gemma3ForCausalLM • Gemma3ForConditionalGeneration • GemmaForCausalLM • Glm4ForCausalLM • Glm4MoeF”

**Check.** [p. 7](https://arxiv.org/pdf/2511.14465v2#page=7) · `2511.14465v2:p7:c0`

**Verdict.** WEAK

### #3 · `experimental_setup` · judge: establishes

**Claim.** The experimental setup includes a validation suite that tests module naming, tensor shapes, attention probabilities, and I/O compatibility.

**Excerpt.** “nnterp automatically validates configurations during initialization, checking: module naming correctness, tensor shapes at each layer, attention probabilities normalization (if enabled), and I/O compatibility.”

**Check.** [p. 7](https://arxiv.org/pdf/2511.14465v2#page=7) · `2511.14465v2:p7:c0`

**Verdict.** OK

### #4 · `experimental_setup` · judge: establishes

**Claim.** nnterp implements standard interpretability methods such as Logit Lens and Patchscope as built-in interventions.

**Excerpt.** “nnterp implements common interpretability methods that work across all supported models. Logit Lens [nostalgebraist, 2020] projects hidden states through the unembedding to see intermediate predictions. Patchscope [Ghandeharioun et al., 2024] replaces activations”

**Check.** [p. 3](https://arxiv.org/pdf/2511.14465v2#page=3) · `2511.14465v2:p3:c4`

**Verdict.** OK

### #5 · `limitations` · judge: establishes

**Claim.** The library's validation tests provide sanity checks but do not offer formal correctness guarantees.

**Excerpt.** “nnterp’s validation tests provide sanity checks rather than formal correctness guarantees.”

**Check.** [p. 4](https://arxiv.org/pdf/2511.14465v2#page=4) · `2511.14465v2:p4:c1`

**Verdict.** OK

### #6 · `limitations` · judge: establishes

**Claim.** The library inherits limitations from NNsight, such as incompatibility with certain attention implementations like Flash Attention for attention probabilities.

**Excerpt.** “The library also inherits NNsight’s limitations, including incompatibility
with some attention implementations (e.g., Flash Attention for attention probabilities).”

**Check.** [p. 4](https://arxiv.org/pdf/2511.14465v2#page=4) · `2511.14465v2:p4:c2`

**Verdict.** OK

### #7 · `limitations` · judge: establishes

**Claim.** Future development goals include implementing automated architecture detection, support for encoder-decoder architectures, and access to additional intermediate activations and router logits.

**Excerpt.** “Future work includes automated architecture detection, support for non-causal
and encoder-decoder architectures, access to attention KQV and MLP intermediate activations and
access to MoE router’s logits.”

**Check.** [p. 4](https://arxiv.org/pdf/2511.14465v2#page=4) · `2511.14465v2:p4:c2`

**Verdict.** OK

### #8 · `limitations` · judge: establishes

**Claim.** Support for attention probabilities is currently incomplete for several specific model classes.

**Excerpt.** “Support for attention probabilities is still a work in progress for the following model classes:
• DbrxForCausalLM
• GptOssForCausalLM
• Qwen2MoeForCausalLM
• StableLmForCausalLM”

**Check.** [p. 7](https://arxiv.org/pdf/2511.14465v2#page=7) · `2511.14465v2:p7:c1`

**Verdict.** OK

### #9 · `main_findings` · judge: establishes

**Claim.** The nnterp library supports 21 architecture families.

**Excerpt.** “nnterp supports 21 architecture families2.”

**Check.** [p. 4](https://arxiv.org/pdf/2511.14465v2#page=4) · `2511.14465v2:p4:c1`

**Verdict.** OK

### #10 · `main_findings` · judge: establishes

**Claim.** nnterp adds minimal overhead to NNsight and maintains its performance characteristics of matching or exceeding TransformerLens speed while using less memory.

**Excerpt.** “nnterp adds minimal overhead to NNsight’s already efficient implementation. NNsight’s performance analysis [Fiotto-Kaufman et al., 2024] shows it matches or exceeds TransformerLens speed while using less memory. Since nnterp is a thin wrapper providing only interface standardization, it inherits the”

**Check.** [p. 4](https://arxiv.org/pdf/2511.14465v2#page=4) · `2511.14465v2:p4:c1`

**Verdict.** WEAK

### #11 · `method` · judge: establishes

**Claim.** The authors propose nnterp, a library designed to enable consistent mechanistic interpretability research across diverse transformer model architectures.

**Excerpt.** “nnterp enables researchers to write intervention code once and deploy it across 50+ model variants spanning 16 architecture families.”

**Check.** [p. 1](https://arxiv.org/pdf/2511.14465v2#page=1) · `2511.14465v2:p1:c1`

**Verdict.** OK

### #12 · `method` · judge: establishes

**Claim.** nnterp utilizes module renaming and a standardized interface to provide consistent access to internal activations, such as layers, self-attention modules, and MLPs.

**Excerpt.** “nnterp provides a standardized interface for transformer models. Left: It renames transformer modules to a consistent naming scheme (layers, self_attn, mlp, etc.).”

**Check.** [p. 2](https://arxiv.org/pdf/2511.14465v2#page=2) · `2511.14465v2:p2:c1`

**Verdict.** OK

### #13 · `method` · judge: establishes

**Claim.** The library includes built-in implementations of interpretability methods such as logit lens, patchscope, and activation steering.

**Excerpt.** “The library includes built-in implementations of common interpretabil- ity methods (logit lens, patchscope, activation steering)”

**Check.** [p. 1](https://arxiv.org/pdf/2511.14465v2#page=1) · `2511.14465v2:p1:c1`

**Verdict.** OK

### #14 · `method` · judge: establishes

**Claim.** nnterp provides accessor methods for module inputs and outputs that normalize tensor handling across different architectures.

**Excerpt.** “Middle: It provides accessor methods for internal activations, with get and set for all *_input and *_output. This handles whether the module returns a tuple or a single tensor.”

**Check.** [p. 2](https://arxiv.org/pdf/2511.14465v2#page=2) · `2511.14465v2:p2:c1`

**Verdict.** OK

### #15 · `relevance_to_query` · judge: establishes

**Claim.** Mechanistic interpretability aims to reverse-engineer the computational mechanisms within neural networks by accessing and modifying internal representations.

**Excerpt.** “Mechanistic interpretability research aims to reverse-engineer the computational mechanisms within neural networks [Elhage et al., 2021, Olah et al., 2020]. For transformer language models, this requires tools that can reliably access and modify internal representations across diverse architectures.”

**Check.** [p. 1](https://arxiv.org/pdf/2511.14465v2#page=1) · `2511.14465v2:p1:c1`

**Verdict.** OK

### #16 · `relevance_to_query` · judge: establishes

**Claim.** Commonly used mechanistic interpretability methods include the logit lens, patchscopes, and activation steering.

**Excerpt.** “The library includes built-in implementations of common interpretabil-ity methods (logit lens, patchscope, activation steering) and provides direct access to attention probabilities for models that support it.”

**Check.** [p. 1](https://arxiv.org/pdf/2511.14465v2#page=1) · `2511.14465v2:p1:c1`

**Verdict.** OK

### #17 · `relevance_to_query` · judge: establishes

**Claim.** Tools for mechanistic interpretability provide access to components such as layers, attention, and MLP outputs for transformer analysis.

**Excerpt.** “A unified API for accessing transformer internals (layers, attention, MLP outputs) that works identically across 50+ model variants from 16 architecture families.”

**Check.** [p. 2](https://arxiv.org/pdf/2511.14465v2#page=2) · `2511.14465v2:p2:c2`

**Verdict.** OK

### #18 · `relevance_to_query` · judge: establishes

**Claim.** Early mechanistic interpretability techniques relied on manual PyTorch hooks to intercept activations.

**Excerpt.** “Early mechanistic interpretability relied on manual PyTorch hooks to intercept activations [Elhage et al., 2021].”

**Check.** [p. 2](https://arxiv.org/pdf/2511.14465v2#page=2) · `2511.14465v2:p2:c3`

**Verdict.** OK

### #19 · `research_problem` · judge: establishes

**Claim.** Mechanistic interpretability research faces a tradeoff between using custom implementations with consistent interfaces but numerical mismatches, or direct HuggingFace access that lacks cross-model standardization.

**Excerpt.** “Current approaches face a fundamental tradeoff: custom implementations like TransformerLens ensure consistent interfaces but require coding a manual adaptation for each architecture, introducing numerical mismatch with the original models, while direct HuggingFace access through NNsight preserves ex”

**Check.** [p. 1](https://arxiv.org/pdf/2511.14465v2#page=1) · `2511.14465v2:p1:c0`

**Verdict.** WEAK

### #20 · `research_problem` · judge: establishes

**Claim.** The existing fragmentation in interpretability tooling forces researchers to choose between limited tools or maintaining multiple codebases for different model architectures, which creates significant friction.

**Excerpt.** “This fragmentation creates significant friction. Researchers must either commit to a single tool’s limitations or maintain parallel codebases for different architectures.”

**Check.** [p. 2](https://arxiv.org/pdf/2511.14465v2#page=2) · `2511.14465v2:p2:c2`

**Verdict.** OK

## 2402.03855v2 — Challenges in Mechanistically Interpreting Model Representations

PDF: https://arxiv.org/pdf/2402.03855v2 · 14 citations

### #21 · `experimental_setup` · judge: establishes

**Claim.** The researchers study the behavior of honesty using a dataset to identify the first principal component of activation vectors as the honesty direction.

**Excerpt.** “the behavior (honesty in our case). We then simply use the first principal component of the difference of these vectors over a dataset for each layer as the direction of honesty, with the opposite direction representing dishonesty.”

**Check.** [p. 4](https://arxiv.org/pdf/2402.03855v2#page=4) · `2402.03855v2:p4:c4`

**Verdict.** OK

### #22 · `experimental_setup` · judge: establishes

**Claim.** The authors utilized the TransformerLens library for several of their experiments.

**Excerpt.** “We would also like to thank Neel Nanda and Joseph Bloom for building and maintaining the TransformerLens (Nanda & Bloom, 2022) library that was used for several experiments in this work.”

**Check.** [p. 9](https://arxiv.org/pdf/2402.03855v2#page=9) · `2402.03855v2:p9:c2`

**Verdict.** OK

### #23 · `limitations` · judge: establishes

**Claim.** Current mechanistic interpretability methods may not scale to reasonably complex capabilities and vulnerabilities.

**Excerpt.** “leading to the concern that the current mechanistic interpretability pipeline, especially with expensive, human-generated hypotheses, would simply not scale to reasonably complex capabilities and vulnerabilities.”

**Check.** [p. 1](https://arxiv.org/pdf/2402.03855v2#page=1) · `2402.03855v2:p1:c3`

**Verdict.** OK

### #24 · `limitations` · judge: establishes

**Claim.** Token-aligned evaluation and interpretability methods can lead to false positives when evaluating nuanced behaviors in models.

**Excerpt.** “In this case, token-aligned evaluation and interpretability (using a first-token-based metric) would just lead to false positives because the honest-looking answer is actually dishonest. This severely restricts the applicability of current methods, especially on nuanced behaviors.”

**Check.** [p. 4](https://arxiv.org/pdf/2402.03855v2#page=4) · `2402.03855v2:p4:c1`

**Verdict.** OK

### #25 · `limitations` · judge: establishes

**Claim.** The authors posit that existing frameworks for studying representations are insufficient for answering important questions, suggesting a need for new approaches.

**Excerpt.** “As a case study, we explore linear representations for dishonesty using the current tooling in mechanistic interpretability and show that they do not help answer most of the important questions that arise while studying representations. We posit the need of new frameworks to think about and study re”

**Check.** [p. 9](https://arxiv.org/pdf/2402.03855v2#page=9) · `2402.03855v2:p9:c0`

**Verdict.** OK

### #26 · `main_findings` · judge: establishes

**Claim.** Directly unembedding the dishonesty direction results in a probability distribution with arbitrary top-k tokens.

**Excerpt.** “Directly unembedding the dishonesty direction results in a relatively high entropy probability distribution with seemingly arbitrary top-k tokens (see Tab. 1).”

**Check.** [p. 5](https://arxiv.org/pdf/2402.03855v2#page=5) · `2402.03855v2:p5:c3`

**Verdict.** OK

### #27 · `main_findings` · judge: establishes

**Claim.** The hypothesis that continual dishonesty injections are not required for open-ended dishonesty generation is disproven.

**Excerpt.** “We test this by comparing the models’ output log probabilities at every token position both with and without the injection. Fig. 4 disproves the hypothesis”

**Check.** [p. 5](https://arxiv.org/pdf/2402.03855v2#page=5) · `2402.03855v2:p5:c3`

**Verdict.** WEAK

### #28 · `method` · judge: establishes

**Claim.** The authors perform an exploratory mechanistic analysis of dishonesty representations in the Mistral-7B-Instruct-v0.1 model.

**Excerpt.** “We formalize representations for features and behaviors, highlight their importance and evaluation, and perform an exploratory study of dishonesty representations in ‘Mistral-7B-Instruct-v0.1’.”

**Check.** [p. 1](https://arxiv.org/pdf/2402.03855v2#page=1) · `2402.03855v2:p1:c0`

**Verdict.** OK

### #29 · `method` · judge: partial

**Claim.** The authors utilize the method of Marks & Tegmark (2023) to compute linear directions for feature representations.

**Excerpt.** “For feature representations, we use the method of Marks & Tegmark (2023) to compute linear directions for truthfulness in ‘Llama-2-7B-chat’”

**Check.** [p. 5](https://arxiv.org/pdf/2402.03855v2#page=5) · `2402.03855v2:p5:c1`

**Verdict.** OK

### #30 · `relevance_to_query` · judge: establishes

**Claim.** Mechanistic interpretability seeks to understand neural networks by reverse-engineering the specific algorithms learned by the models.

**Excerpt.** “Mechanistic interpretability (MI) aims to under- stand AI models by reverse-engineering the exact algorithms neural networks learn.”

**Check.** [p. 1](https://arxiv.org/pdf/2402.03855v2#page=1) · `2402.03855v2:p1:c0`

**Verdict.** OK

### #31 · `relevance_to_query` · judge: establishes

**Claim.** Tools such as saliency maps, the logit lens, and activation and path patching are used to understand how functions and capabilities are implemented within models.

**Excerpt.** “Tools such as saliency maps (Simonyan et al., 2013), the logit lens (nostalgebraist, 2020), and acti- vation and path patching (Wang et al., 2022) have helped us understand how simple functions and capabilities are implemented inside neural network models.”

**Check.** [p. 1](https://arxiv.org/pdf/2402.03855v2#page=1) · `2402.03855v2:p1:c1`

**Verdict.** OK

### #32 · `relevance_to_query` · judge: establishes

**Claim.** Researchers use mechanistic interpretability to identify specialized components, such as induction heads in transformers, that contribute to capabilities like in-context learning.

**Excerpt.** “Elhage et al. (2021) discovered specialized attention heads called “induction heads” that are found to play an important role in the emergence of in-context learning in language models (Olsson et al., 2022).”

**Check.** [p. 1](https://arxiv.org/pdf/2402.03855v2#page=1) · `2402.03855v2:p1:c1`

**Verdict.** OK

### #33 · `research_problem` · judge: establishes

**Claim.** The paper addresses the lack of frameworks to study model representations, as current methods fail to yield verifiable interpretations of how a model works.

**Excerpt.** “We posit the need of new frameworks to think about and study representations.”

**Check.** [p. 9](https://arxiv.org/pdf/2402.03855v2#page=9) · `2402.03855v2:p9:c0`

**Verdict.** WEAK

### #34 · `research_problem` · judge: establishes

**Claim.** Studying representations is considered crucial for understanding models, as it relates to model attributes like honesty, toxicity, fairness, and safety in high-stake domains.

**Excerpt.** “Our position toward new frameworks for studying representations applies to several model attributes such as honesty, toxicity, fairness, bias, power-seeking, etc., all of which have significant impact on how language models are deployed in several high-stake domains.”

**Check.** [p. 9](https://arxiv.org/pdf/2402.03855v2#page=9) · `2402.03855v2:p9:c1`

**Verdict.** WEAK

## 2409.13714v1 — TracrBench: Generating Interpretability Testbeds with Large Language Models

PDF: https://arxiv.org/pdf/2409.13714v1 · 19 citations

### #35 · `experimental_setup` · judge: establishes

**Claim.** The dataset, TracrBench, consists of 121 transformer models with known mappings from weights to their functional form, generated from simple sequence-to-sequence algorithms.

**Excerpt.** “we present TracrBench, a novel dataset of Tracr models that enables interpretability researchers to quickly test methods on transformers with known mappings from weights to their functional form. The dataset is generated as follows. First, we select 121 simple, sequence-to-sequence algorithms”

**Check.** [p. 3](https://arxiv.org/pdf/2409.13714v1#page=3) · `2409.13714v1:p3:c1`

**Verdict.** OK

### #36 · `experimental_setup` · judge: establishes

**Claim.** The experimental setup uses zero-shot, one-shot, and 20-shot prompting variations to evaluate an LLM's ability to generate RASP programs.

**Excerpt.** “We use three variations of the prompt: a zero-shot, a one-shot prompt, and a 20-shot prompt. These different prompt variations are used to assess how including examples affects the LLM’s performance in generating RASP programs.”

**Check.** [p. 3](https://arxiv.org/pdf/2409.13714v1#page=3) · `2409.13714v1:p3:c4`

**Verdict.** OK

### #37 · `experimental_setup` · judge: establishes

**Claim.** To distinguish general programming ability from RASP-specific performance, the authors use a Python program baseline for the same target algorithms.

**Excerpt.** “To distinguish between an LLM’s general programming ability and its RASP-specific capabilities, we establish a baseline where the LLM writes a Python program for the same target algorithms.”

**Check.** [p. 3](https://arxiv.org/pdf/2409.13714v1#page=3) · `2409.13714v1:p3:c6`

**Verdict.** OK

### #38 · `limitations` · judge: establishes

**Claim.** The performance of frontier LLMs in generating interpretability test beds deteriorates as RASP programs become more difficult.

**Excerpt.** “However, their performance rapidly deteriorates with the increasing difficulty of RASP programs, indicating that frontier LLMs struggle to generate interpretability test beds at scale.”

**Check.** [p. 5](https://arxiv.org/pdf/2409.13714v1#page=5) · `2409.13714v1:p5:c0`

**Verdict.** OK

### #39 · `limitations` · judge: establishes

**Claim.** TracrBench is not suitable for developing interpretability methods because of its small size and the lack of similarity between Tracr weights and those of trained transformers.

**Excerpt.** “It is unsuitable as a target for interpretability method development due to its small size and the fact that Tracr weights are very dissimilar to those of trained transformers in terms of sparsity and matrix-rank.”

**Check.** [p. 5](https://arxiv.org/pdf/2409.13714v1#page=5) · `2409.13714v1:p5:c0`

**Verdict.** OK

### #40 · `limitations` · judge: establishes

**Claim.** Frontier LLMs have a limited ability to extend their reasoning and programming capabilities to low-resource programming languages.

**Excerpt.** “This finding highlights that the ability of frontier LLMs to extend their reasoning and programming capabilities to low-resource programming languages is limited”

**Check.** [p. 4](https://arxiv.org/pdf/2409.13714v1#page=4) · `2409.13714v1:p4:c2`

**Verdict.** OK

### #41 · `main_findings` · judge: establishes

**Claim.** Gpt-4-turbo achieved a 56% pass rate for generating RASP programs, while outperforming Claude-3-opus which achieved 46%.

**Excerpt.** “Overall, gpt-4-turbo achieves the highest pass rate of 56%, outperforming claude-3-opus with a pass rate of 46%.”

**Check.** [p. 4](https://arxiv.org/pdf/2409.13714v1#page=4) · `2409.13714v1:p4:c0`

**Verdict.** OK

### #42 · `main_findings` · judge: partial

**Claim.** LLMs showed a significantly higher success rate when generating Python programs, with gpt-4-turbo achieving a 96% pass rate.

**Excerpt.** “In comparison, when generating Python programs for the target algorithms, gpt-4-turbo achieves a pass rate of 96%.”

**Check.** [p. 4](https://arxiv.org/pdf/2409.13714v1#page=4) · `2409.13714v1:p4:c0`

**Verdict.** WEAK

### #43 · `main_findings` · judge: establishes

**Claim.** Successes in generating RASP programs are concentrated in easy, low-difficulty tasks, with gpt-4o performing best on a difficulty-weighted metric with a score of 0.31.

**Excerpt.** “we observe that the successes are strongly concentrated among the easy, low-difficulty programs (see Fig. 4) with gpt-4-turbo achieving a score of 0.29 (out of 1.0) and gpt-4o performing best with a score of 0.31.”

**Check.** [p. 4](https://arxiv.org/pdf/2409.13714v1#page=4) · `2409.13714v1:p4:c1`

**Verdict.** WEAK

### #44 · `main_findings` · judge: establishes

**Claim.** Claude-3-5-sonnet demonstrated a higher difficulty-weighted score of 0.27 compared to the 0.23 score achieved by Claude-3-opus.

**Excerpt.** “Claude-3-5-sonnet has a similar pass rate (0.45) to claude-3-opus (0.46), however, it achieves a higher difficulty-weighted score (0.27), than claude-3-opus (0.23).”

**Check.** [p. 4](https://arxiv.org/pdf/2409.13714v1#page=4) · `2409.13714v1:p4:c1`

**Verdict.** OK

### #45 · `method` · judge: establishes

**Claim.** The authors introduce a novel approach for generating interpretability testbeds by using large language models to create RASP programs.

**Excerpt.** “In this work, we present a novel approach for generating interpretability test beds using large language models (LLMs) and introduce TracrBench, a novel dataset consisting of 121 manually written and LLM-generated, human-validated RASP”

**Check.** [p. 1](https://arxiv.org/pdf/2409.13714v1#page=1) · `2409.13714v1:p1:c0`

**Verdict.** OK

### #46 · `method` · judge: establishes

**Claim.** The proposed method generates RASP programs that are compiled into transformer weights using Tracr to provide ground truth mappings.

**Excerpt.** “These programs are then compiled into functional transformer weights using Tracr, resulting in transformer models with a known mapping between weights and their functional form.”

**Check.** [p. 2](https://arxiv.org/pdf/2409.13714v1#page=2) · `2409.13714v1:p2:c2`

**Verdict.** OK

### #47 · `method` · judge: partial

**Claim.** The generation process involves prompting a language model with an algorithm description, input-output examples, and RASP language specifications.

**Excerpt.** “To generate a program, we condition a language model M on a prompt P that includes a description of the specific algorithm to be implemented and at least one example input-output pair”

**Check.** [p. 2](https://arxiv.org/pdf/2409.13714v1#page=2) · `2409.13714v1:p2:c2`

**Verdict.** WEAK

### #48 · `relevance_to_query` · judge: establishes

**Claim.** Mechanistic interpretability aims to achieve a functional understanding of transformer internals by establishing ground truth mappings between model weights and their roles.

**Excerpt.** “Achieving a mechanistic understanding of transformer-based language models is an open challenge, especially due to their large number of parameters. Moreover, the lack of ground truth mappings between model weights and their functional roles hinders the effective evaluation of interpretability metho”

**Check.** [p. 1](https://arxiv.org/pdf/2409.13714v1#page=1) · `2409.13714v1:p1:c0`

**Verdict.** OK

### #49 · `relevance_to_query` · judge: establishes

**Claim.** The Restricted Access Sequence Processing Language (RASP) is used to model and analyze transformer behavior by mapping core components like attention and feed-forward computation into simple primitives.

**Excerpt.** “Restricted Access Sequence Processing Language (RASP) (Weiss et al., 2021) maps the core components of a transformer-encoder, i.e., attention and feed-forward computation, into simple primitives, forming a programming language to model and analyze transformer behavior.”

**Check.** [p. 1](https://arxiv.org/pdf/2409.13714v1#page=1) · `2409.13714v1:p1:c3`

**Verdict.** OK

### #50 · `relevance_to_query` · judge: establishes

**Claim.** Decompiler models have been developed to generate RASP programs from a given set of transformer weights using large quantities of programmatically generated training data.

**Excerpt.** “both Thurnherr & Riesen (2024) and Langosco et al. (2024) programmatically generate large quantities of RASP programs with their corresponding weights to train decompiler models that generate RASP programs for a given set of transformer weights.”

**Check.** [p. 4](https://arxiv.org/pdf/2409.13714v1#page=4) · `2409.13714v1:p4:c4`

**Verdict.** OK

### #51 · `research_problem` · judge: establishes

**Claim.** Interpretability research is hindered by the difficulty of evaluating new methods due to a lack of models with fully understood internals.

**Excerpt.** “Current interpretability research faces challenges in rigorously evaluating novel methods due to the lack of models with fully understood internals.”

**Check.** [p. 2](https://arxiv.org/pdf/2409.13714v1#page=2) · `2409.13714v1:p2:c1`

**Verdict.** OK

### #52 · `research_problem` · judge: establishes

**Claim.** Tracr provides ground truth mappings to address evaluation difficulties, but it is labor-intensive and difficult to use manually.

**Excerpt.** “Tracr, a method for generating compiled transformers with inherent ground truth mappings in RASP, has been proposed to address this issue. However, manually creating a large number of models needed for verifying interpretability methods is labour-intensive and time-consuming.”

**Check.** [p. 1](https://arxiv.org/pdf/2409.13714v1#page=1) · `2409.13714v1:p1:c0`

**Verdict.** OK

### #53 · `research_problem` · judge: establishes

**Claim.** The steep learning curve and labor-intensive nature of writing RASP code have impeded the adoption of Tracr for evaluating interpretability methods.

**Excerpt.** “Writing RASP code to generate Tracr interpretability test beds is labor-intensive and has a steep learning curve (see Appendix B for an example). This has impeded the adoption of Tracr as a method to evaluate novel interpretability methods.”

**Check.** [p. 3](https://arxiv.org/pdf/2409.13714v1#page=3) · `2409.13714v1:p3:c1`

**Verdict.** OK

## 2606.16939v1 — Scalable Circuit Learning for Interpreting Large Language Models

PDF: https://arxiv.org/pdf/2606.16939v1 · 23 citations

### #54 · `experimental_setup` · judge: establishes

**Claim.** The study uses the Bias-in-Bios dataset for the task of profession classification based on biographical data.

**Excerpt.** “The Bias-in-Bios dataset (BiB) (De-Arteaga et al., 2019) consists of professional biographies with the task of classifying an individual’s profession.”

**Check.** [p. 18](https://arxiv.org/pdf/2606.16939v1#page=18) · `2606.16939v1:p18:c1`

**Verdict.** OK

### #55 · `experimental_setup` · judge: establishes

**Claim.** The experimental evaluation is conducted on Pythia-70M, Gemma-2-2B, and Gemma-2-9B models.

**Excerpt.** “Setup. We evaluate on the Bias-in-Bios dataset (BiB) (De-Arteaga et al., 2019) on Pythia-70M (Biderman et al., 2023), Gemma-2-2B and Gemma-2-9B (Team et al., 2024)”

**Check.** [p. 8](https://arxiv.org/pdf/2606.16939v1#page=8) · `2606.16939v1:p8:c4`

**Verdict.** OK

### #56 · `experimental_setup` · judge: partial

**Claim.** Baselines used for comparison include ORIGINAL, ORACLE, CBP, SHIFT, SHIFT-retrain, and LINEAR PROBING.

**Excerpt.** “Baselines. We compare against: (i) ORIGINAL, a predictor trained directly on the ambiguous set; (ii) ORACLE, a predictor trained on the balanced set, included as a (non-comparable) upper bound; (iii) CBP, concept bottleneck probing (Yan et al., 2023); (iv) SHIFT, spurious human-interpretable feature”

**Check.** [p. 18](https://arxiv.org/pdf/2606.16939v1#page=18) · `2606.16939v1:p18:c2`

**Verdict.** WEAK

### #57 · `experimental_setup` · judge: establishes

**Claim.** The method involves ranking SAE features, identifying gender-correlated features, zeroing them, and testing the result using CircuitLasso or CircuitLasso-retrain.

**Excerpt.** “Method. We rank SAE features at a single layer by | bAi,y|, manually identify gender-correlated features, zero them, and feed the result either directly into the trained classifier (CircuitLasso) or into a freshly retrained classifier (CircuitLasso-retrain).”

**Check.** [p. 9](https://arxiv.org/pdf/2606.16939v1#page=9) · `2606.16939v1:p9:c0`

**Verdict.** OK

### #58 · `limitations` · judge: establishes

**Claim.** Existing mechanistic interpretability methods face challenges in scaling to high-dimensional SAE feature spaces.

**Excerpt.** “Despite this progress, existing mechanistic interpretability methods continue to face challenges in scaling to the high-dimensional SAE feature space.”

**Check.** [p. 2](https://arxiv.org/pdf/2606.16939v1#page=2) · `2606.16939v1:p2:c3`

**Verdict.** OK

### #59 · `limitations` · judge: establishes

**Claim.** Approaches that fit model internals to pre-defined hypotheses fail to generalize when researchers lack expert knowledge or cannot anticipate model behaviors.

**Excerpt.** “model internals to pre-defined hypotheses using curated data, but these approaches fail to generalize to scenarios where researchers lack expert knowledge or cannot anticipate how models implement specific behaviors.”

**Check.** [p. 2](https://arxiv.org/pdf/2606.16939v1#page=2) · `2606.16939v1:p2:c3`

**Verdict.** OK

### #60 · `limitations` · judge: establishes

**Claim.** The nonlinear extension of CircuitLasso incurs a substantially higher computational cost than the linear formulation.

**Excerpt.** “The recovered topological skeleton is essentially the same as in the linear case, at substantially higher computational cost.”

**Check.** [p. 14](https://arxiv.org/pdf/2606.16939v1#page=14) · `2606.16939v1:p14:c0`

**Verdict.** OK

### #61 · `limitations` · judge: establishes

**Claim.** Edge weights in the CircuitLasso method do not quantify exact causal effects but are intended for ranking dependencies.

**Excerpt.** “Edge weights in bA should not be interpreted as exact causal effects of the underlying nonlinear computation; their role is to rank dependencies for selection, not to quantify them.”

**Check.** [p. 14](https://arxiv.org/pdf/2606.16939v1#page=14) · `2606.16939v1:p14:c0`

**Verdict.** OK

### #62 · `main_findings` · judge: establishes

**Claim.** Across 17 evaluation tasks, CircuitLasso-linear achieves a mean Structural Hamming Distance (SHD) of 3.16, which is statistically indistinguishable from EAP-ig and better than EAP.

**Excerpt.** “Across the 17 tasks, CircuitLasso-linear attains a mean SHD of 3.16, statistically indistinguishable from EAP-ig (2.98) and below EAP (3.61)”

**Check.** [p. 6](https://arxiv.org/pdf/2606.16939v1#page=6) · `2606.16939v1:p6:c1`

**Verdict.** OK

### #63 · `main_findings` · judge: establishes

**Claim.** CircuitLasso is 3.0 times faster than EAP-ig and 2.1 times faster than EAP on the evaluated tasks.

**Excerpt.** “at a mean runtime of 16.3 s per case, 3.0 times faster than EAP-ig (49.1 s) and 2.1 times faster than EAP (33.7 s).”

**Check.** [p. 6](https://arxiv.org/pdf/2606.16939v1#page=6) · `2606.16939v1:p6:c1`

**Verdict.** OK

### #64 · `main_findings` · judge: establishes

**Claim.** In the Bias-in-Bios task, CircuitLasso achieves profession accuracy and worst-group accuracy comparable to or better than SHIFT, while keeping gender-leakage values near the 50% target.

**Excerpt.** “CircuitLasso and CircuitLasso-retrain achieve profession accuracies comparable to, and in some cases slightly better than, SHIFT and SHIFT-retrain; gender-leakage values stay near the 50% balanced target; and worst-group accuracies match or exceed the strongest non-ORACLE baseline.”

**Check.** [p. 19](https://arxiv.org/pdf/2606.16939v1#page=19) · `2606.16939v1:p19:c0`

**Verdict.** OK

### #65 · `main_findings` · judge: establishes

**Claim.** CircuitLasso requires fewer features and substantially less runtime than SHIFT for large models in the Bias-in-Bios experiment.

**Excerpt.** “Efficiency results (Table 1) further underscore the strengths of CircuitLasso: it requires fewer features and substantially less runtime than SHIFT, particularly for large models.”

**Check.** [p. 19](https://arxiv.org/pdf/2606.16939v1#page=19) · `2606.16939v1:p19:c0`

**Verdict.** OK

### #66 · `method` · judge: establishes

**Claim.** The authors introduce a method called CircuitLasso to identify sparse connections between model features.

**Excerpt.** “To address this, we introduce a novel approach to handle the high dimensionality. Our method, CircuitLasso, utilizes the Lasso (i.e., ℓ1-penalized linear regression) to find a sparse set of connections between features that explains the model’s behavior.”

**Check.** [p. 1](https://arxiv.org/pdf/2606.16939v1#page=1) · `2606.16939v1:p1:c4`

**Verdict.** OK

### #67 · `method` · judge: establishes

**Claim.** CircuitLasso utilizes sparse linear regression, specifically Lasso, to reduce optimization to a scalable solution for high-dimensional data.

**Excerpt.** “we make simplifying assumptions that bypass the explicit enforcement of this constraint and reduce the optimization to sparse linear regression problems (i.e., Lasso), enabling a scalable solution.”

**Check.** [p. 4](https://arxiv.org/pdf/2606.16939v1#page=4) · `2606.16939v1:p4:c1`

**Verdict.** OK

### #68 · `method` · judge: partial

**Claim.** CircuitLasso utilizes sparse linear regression, specifically Lasso, to reduce optimization to a scalable solution for high-dimensional data.

**Excerpt.** “Sparse linear regression is well-suited for high-dimensional data, as it is computationally efficient and the sparsity translates to more interpretable circuits.”

**Check.** [p. 1](https://arxiv.org/pdf/2606.16939v1#page=1) · `2606.16939v1:p1:c4`

**Verdict.** WEAK

### #69 · `method` · judge: establishes

**Claim.** The proposed method relies exclusively on observational data to identify circuits.

**Excerpt.** “An advantage of our approach is its use of observational data only. This broadens its applicability and addresses the scal-”

**Check.** [p. 1](https://arxiv.org/pdf/2606.16939v1#page=1) · `2606.16939v1:p1:c4`

**Verdict.** OK

### #70 · `relevance_to_query` · judge: establishes

**Claim.** Mechanistic interpretability discovers circuits, which are compact subgraphs connecting components like neurons and attention heads that drive specific model behaviors.

**Excerpt.** “A key technique involves discovering circuits, which are compact subgraphs connecting key components within the model (such as attention heads and neurons) that drive a specific behavior or capability”

**Check.** [p. 1](https://arxiv.org/pdf/2606.16939v1#page=1) · `2606.16939v1:p1:c1`

**Verdict.** OK

### #71 · `relevance_to_query` · judge: establishes

**Claim.** Researchers use sparse autoencoders (SAEs) to identify sparse, disentangled features in high-dimensional spaces that correspond to human-interpretable concepts.

**Excerpt.** “Recent work (Bricken et al., 2023; Cunningham et al., 2023) leverages advances in dictionary learning for interpretability and introduces sparse autoencoders (SAEs) to identify sparse, disentangled features in high-dimensional spaces that align with human-interpretable concepts.”

**Check.** [p. 2](https://arxiv.org/pdf/2606.16939v1#page=2) · `2606.16939v1:p2:c3`

**Verdict.** OK

### #72 · `relevance_to_query` · judge: partial

**Claim.** Mechanistic interpretability research identifies graphical structures connecting pretrained language model neurons, such as outputs from MLP modules and attention heads, to understand information encoding.

**Excerpt.** “To better understand how models encode and process information, mechanistic interpretability research (Conmy et al., 2023; Cao et al., 2021; Syed et al., 2024) has focused on identifying graphical structures (circuits) connecting pretrained language model neurons, including outputs from attention an”

**Check.** [p. 4](https://arxiv.org/pdf/2606.16939v1#page=4) · `2606.16939v1:p4:c1`

**Verdict.** WEAK

### #73 · `relevance_to_query` · judge: establishes

**Claim.** Techniques such as circuit learning uncover relationships among SAE features to show how semantic features propagate through a model and influence its predictions.

**Excerpt.** “For interpretability, CircuitLasso efficiently un-covers relationships among SAE features, showing how human-interpretable semantic features propagate through the model and influence its predictions.”

**Check.** [p. 1](https://arxiv.org/pdf/2606.16939v1#page=1) · `2606.16939v1:p1:c1`

**Verdict.** OK

### #74 · `research_problem` · judge: establishes

**Claim.** The paper addresses the scalability issues inherent in existing intervention-based circuit discovery methods.

**Excerpt.** “An advantage of our approach is its use of observational data only. This broadens its applicability and addresses the scalability issue of the existing intervention-based approaches, whose cost scales with LLM size.”

**Check.** [p. 1](https://arxiv.org/pdf/2606.16939v1#page=1) · `2606.16939v1:p1:c5`

**Verdict.** OK

### #75 · `research_problem` · judge: establishes

**Claim.** The research focuses on mechanistic interpretability to identify graphical structures or circuits connecting pretrained language model neurons.

**Excerpt.** “mechanistic interpretability research (Conmy et al., 2023; Cao et al., 2021; Syed et al., 2024) has focused on identifying graphical structures (circuits) connecting pretrained language model neurons, including outputs from attention and MLP modules.”

**Check.** [p. 4](https://arxiv.org/pdf/2606.16939v1#page=4) · `2606.16939v1:p4:c1`

**Verdict.** OK

### #76 · `research_problem` · judge: partial

**Claim.** The authors propose CircuitLasso as a more efficient, observational-data-only approach to perform circuit discovery without requiring backward passes through the LLM.

**Excerpt.** “CircuitLasso, by contrast, runs no backward passes and shares only the initial forward pass (used for activation collection) with EAP-ig.”

**Check.** [p. 4](https://arxiv.org/pdf/2606.16939v1#page=4) · `2606.16939v1:p4:c5`

**Verdict.** WEAK
