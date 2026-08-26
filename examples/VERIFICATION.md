# Verification of the example review

`example_review.md` contains **79 citations** across 4 papers. This records what has been checked,
by what means, and what is left for a human to confirm.

## Checked automatically

These run as code and are reproducible; nothing here relies on judgment.

| Check | Result |
| --- | --- |
| Every cited `chunk_id` exists in the run's index | 79/79 |
| Every cited chunk belongs to the paper it is attributed to | 79/79 |
| Every excerpt occurs verbatim in the chunk it cites | 79/79 |
| Every page anchor matches the cited chunk's page | 79/79 |
| Every page link in the report traces to a validated citation | 21/21 |
| Papers cited in the report were actually analyzed | 4/4 |

The fifth row is the one worth noting: synthesis is a language model writing
Markdown, so it could in principle invent a page link. None were invented.

## Reviewed by Claude, not by a human

The checks above prove an excerpt *exists* on the page it cites. They cannot prove
the excerpt *supports* the claim built on it, which needs reading. All
79 citations were read against their claims, with two findings:

- **74 of 78 unique claim/citation pairs are soundly supported.** The excerpt states
  what the claim says, without extrapolation.
- **Two are weak.** The first is item 1 in the spot-check list below; the second
  (`2402.03855v2:p12:c1`) did not make the sample, so it remains reviewed by Claude
  only:
  - `2402.03855v2:p12:c1` — the excerpt begins mid-sentence at "correlation with
    dishonesty", so the claim's qualifier "limited" is not itself evidenced by the
    quoted text, only by the sentence it was cut from.
  - `2511.09432v2:p1:c0` — the excerpt is the paper's title. It supports the claim's
    substance but is not a substantive passage.

One thing a reader may notice that is **not** a pipeline error: the `nnterp` paper
says "16 architecture families" on page 2 and "21 architecture families" on page 4.
Both claims are quoted correctly; the inconsistency is the source paper's.

## For you to spot-check

Open each link, find the quoted text on that page, and confirm it supports the claim.

### 1. `2511.09432v2:p1:c0` — method

- **Claim:** The authors propose Equivariant Sparse Autoencoders to improve the interpretability of neural networks on symmetric data.
- **Excerpt:** “Equivariant Sparse Autoencoders: Mechanistic Interpretability of Neural Networks on Symmetric Data”
- **Page:** [2511.09432v2 p. 1](https://arxiv.org/pdf/2511.09432v2#page=1)
- **Supports the claim?** ☐ yes  x no

### 2. `2511.09432v2:p1:c0` — relevance_to_query

- **Claim:** Mechanistic interpretability methods are used to disentangle activations that have compressed many concepts into fewer dimensions, a phenomenon called superposition.
- **Excerpt:** “In particular, their activations entangle many concepts into fewer dimensions, a phenomenon known as superposition. Mechanistic interpretability methods such as sparse autoencoders (SAEs) can disentangle these dense activations into sparse sums of interpretable features”
- **Page:** [2511.09432v2 p. 1](https://arxiv.org/pdf/2511.09432v2#page=1)
- **Supports the claim?** x yes  ☐ no

### 3. `2511.14465v2:p1:c0` — research_problem

- **Claim:** Mechanistic interpretability research faces a tradeoff where custom implementations provide consistent interfaces but risk numerical mismatch, while direct access methods lack standardization across architectures.
- **Excerpt:** “Current approaches face a fundamental tradeoff: custom implementations like TransformerLens ensure consistent interfaces but require coding a manual adaptation for each architecture, introducing numerical mismatch with the original models, while direct HuggingFace access through NNsight preserves ex”
- **Page:** [2511.14465v2 p. 1](https://arxiv.org/pdf/2511.14465v2#page=1)
- **Supports the claim?** ☐ yes  x no

### 4. `2511.14465v2:p1:c1` — method

- **Claim:** The authors propose nnterp, a library that enables mechanistic interpretability by providing a standardized interface for transformer models.
- **Excerpt:** “original HuggingFace implementations. Through automatic module renaming and
comprehensive validation testing, nnterp enables researchers to write interven-
tion code once and deploy it across 50+ model variants spanning 16 architecture
families.”
- **Page:** [2511.14465v2 p. 1](https://arxiv.org/pdf/2511.14465v2#page=1)
- **Supports the claim?** x yes  ☐ no

### 5. `2511.14465v2:p2:c2` — experimental_setup

- **Claim:** The nnterp library provides a unified interface for accessing internal components of over 50 model variants across 16 architecture families.
- **Excerpt:** “A unified API for accessing transformer internals (layers, attention, MLP outputs) that works identically across 50+ model variants from 16 architecture families.”
- **Page:** [2511.14465v2 p. 2](https://arxiv.org/pdf/2511.14465v2#page=2)
- **Supports the claim?** x yes  ☐ no

### 6. `2511.14465v2:p4:c1` — main_findings

- **Claim:** The nnterp library supports 21 architecture families.
- **Excerpt:** “nnterp supports 21 architecture families2.”
- **Page:** [2511.14465v2 p. 4](https://arxiv.org/pdf/2511.14465v2#page=4)
- **Supports the claim?** x yes  ☐ no

### 7. `2511.14465v2:p4:c1` — limitations

- **Claim:** The library's validation tests offer sanity checks but do not provide formal correctness guarantees, meaning subtle bugs may remain.
- **Excerpt:** “nnterp’s validation tests provide sanity checks rather than formal correctness guarantees. While they catch common issues, subtle bugs in attention probability hooks or module identification may persist.”
- **Page:** [2511.14465v2 p. 4](https://arxiv.org/pdf/2511.14465v2#page=4)
- **Supports the claim?** x yes  ☐ no

### 8. `2511.14465v2:p1:c1` — relevance_to_query

- **Claim:** Mechanistic interpretability aims to reverse-engineer the computational mechanisms within neural networks by accessing and modifying internal representations.
- **Excerpt:** “Mechanistic interpretability research aims to reverse-engineer the computational mechanisms within neural networks [Elhage et al., 2021, Olah et al., 2020]. For transformer language models, this requires tools that can reliably access and modify internal representations across diverse architectures.”
- **Page:** [2511.14465v2 p. 1](https://arxiv.org/pdf/2511.14465v2#page=1)
- **Supports the claim?** x yes  ☐ no

### 9. `2402.03855v2:p9:c0` — research_problem

- **Claim:** The paper argues that studying feature and behavior representations is essential for understanding and controlling AI models.
- **Excerpt:** “In this position paper, we discuss the existing literature to formalize representations and motivate that studying representations for features and behaviors is crucial for understanding models.”
- **Page:** [2402.03855v2 p. 9](https://arxiv.org/pdf/2402.03855v2#page=9)
- **Supports the claim?** ☐ yes  x no

### 10. `2402.03855v2:p1:c0` — method

- **Claim:** The authors perform an exploratory study of dishonesty representations in the 'Mistral-7B-Instruct-v0.1' model.
- **Excerpt:** “We formalize representations for features and behaviors, highlight their importance and evaluation, and perform an exploratory study of dishonesty representations in ‘Mistral-7B-Instruct-v0.1’.”
- **Page:** [2402.03855v2 p. 1](https://arxiv.org/pdf/2402.03855v2#page=1)
- **Supports the claim?** x yes  ☐ no

### 11. `2402.03855v2:p4:c4` — experimental_setup

- **Claim:** The researchers investigated honesty by calculating the first principal component of the difference of activation vectors across layers for a given dataset.
- **Excerpt:** “We then simply use the first principal component of the difference of these vectors over a dataset for each layer as the direction of honesty, with the opposite direction representing dishonesty. Thus, we have 32 vectors of size dmodel, one for each layer of the model.”
- **Page:** [2402.03855v2 p. 4](https://arxiv.org/pdf/2402.03855v2#page=4)
- **Supports the claim?** ☐ yes  x no

### 12. `2402.03855v2:p5:c3` — main_findings

- **Claim:** Directly unembedding the dishonesty direction results in a probability distribution with arbitrary top-k tokens, indicating that these representations are not naively used by the model.
- **Excerpt:** “Directly unembedding the dishonesty direction results in a relatively high entropy probability distribution with seemingly arbitrary top-k tokens (see Tab. 1). This shows that these representations are not naively used by the model and require a more detailed analysis.”
- **Page:** [2402.03855v2 p. 5](https://arxiv.org/pdf/2402.03855v2#page=5)
- **Supports the claim?** x yes  ☐ no

## Result

Spot-checked by hand: **8 of 12 confirmed, 4 rejected** (items 1, 3, 9, and 11). None
was a fabrication, none pointed at the wrong paper, and none cited a chunk that does not
exist. All four were real quotes asked to carry more than they say:

- **Truncation cut the supporting half.** Item 3's excerpt stops mid-word at 300
  characters (`EVIDENCE_EXCERPT_CHARS`), just before the clause that would have
  supported the second half of the claim.
- **The quote's referent lay outside the quote.** Item 11 says "the difference of these
  vectors" — the claim calls them activation vectors, which is correct in the paper but
  not established by the quoted span alone.
- **The claim added a word the quote does not carry.** Item 9's excerpt supports
  "crucial for understanding models"; the claim says "understanding *and controlling*".
  Item 1 quotes the paper's title rather than a passage.

### This sample is not the project's claim-support figure

These twelve were chosen deliberately — spread across papers and facets, and seeded with
two citations already flagged as weak — so they oversample known problems. Read as a
rate, 8 of 12 carries a 95% interval of [39%, 86%], which is close to uninformative.

The measured figure comes from a uniform random sample of 40 citations across both
committed runs: **77.5% fully establish their claim, 100% support it at least partly,
and none fails outright.** See `evals/labels/claim_support_labels.json` and
[`DESIGN.md`](../DESIGN.md#4-grounding).

What this document is good for is the *failure modes* above, which the larger sample
confirms but does not itemize.
