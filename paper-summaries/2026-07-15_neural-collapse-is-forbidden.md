# Neural Collapse Is Forbidden: Information Floors in Language Models

**arXiv:** 2607.09487  
**Date:** July 10, 2026  
**Authors:** Bruno Abrahao  
**Affiliation:** NYU Shanghai  
**URL:** https://arxiv.org/abs/2607.09487

---

## Headline

Within-class variance in language model representations is not an imperfection to be optimized away — it is obligatory information storage, bounded below by a provable information floor that makes full neural collapse theoretically impossible in next-token prediction.

---

## Key Findings

- Across 14 language models spanning a 100x parameter range, macro-category structure (e.g., part of speech, semantic class) accounts for only 4–12% of representational variance, while within-token context accounts for 79–91%.
- This pattern is stable across model scale, architecture family, and tokenization scheme.
- The author proves an information-theoretic lower bound — the "information floor" — showing that within-category dispersion must be at least proportional to the conditional mutual information $I(\text{token}; \text{context} \mid \text{category})$.
- Since $I(\text{token}; \text{context} \mid \text{category}) > 0$ for natural language (context always provides information about which specific token appears, beyond its category), full neural collapse is forbidden.
- Token-level weight decay reduces next-token prediction to an imbalanced K-class problem, explaining why the within-class variance allocation persists even under strong regularization.

---

## Methodology

The study probes the last-layer hidden states of 14 language models (GPT-2 through GPT-4o, LLaMA 1–3, Mistral, Gemma, Claude 3 families) on a 50,000-token diagnostic corpus drawn from five domains (Wikipedia, code, scientific text, conversational text, legal documents). For each token position, the author records:

1. Its macro-category label (one of 32 semantic/syntactic categories derived from WordNet + POS tags).
2. Its contextual fingerprint (last 128-token context window, encoded as a TF-IDF bag of tokens).
3. Its residual stream vector at the final transformer layer.

The between-class variance $\sigma^2_B$ and within-class variance $\sigma^2_W$ are then computed in the principal subspace of the full covariance matrix. The variance share $\sigma^2_W / (\sigma^2_B + \sigma^2_W)$ is the key metric.

---

## Technical Details

The information floor theorem is the central theoretical contribution. Let $C$ be a token's macro-category, $X$ the token identity, and $Z$ the context. The paper proves:

$$\sigma^2_W \;\geq\; c \cdot I(X; Z \mid C)$$

where $c > 0$ is a constant that depends on the loss function and model architecture. The proof proceeds by contradiction: if $\sigma^2_W < c \cdot I(X; Z \mid C)$, the model's representation cannot encode enough information to minimize the next-token prediction cross-entropy below a computable threshold, violating the empirically observed training loss.

The mutual information $I(X; Z \mid C)$ is estimated using a k-nearest-neighbor estimator on the joint distribution of token identities and context vectors, calibrated to be consistent with the Kozachenko-Leonenko estimator.

---

## Experimental Findings

| Model family | Scale range | $\sigma^2_W / (\sigma^2_B + \sigma^2_W)$ |
|---|---|---|
| GPT series | 117M – est. 200B | 82.4% ± 3.1% |
| LLaMA series | 7B – 70B | 84.7% ± 2.4% |
| Mistral | 7B – 22B | 83.9% ± 2.8% |
| Claude family | 3 Haiku – 3 Opus | 79.3% ± 4.2% |
| Gemma | 2B – 27B | 85.2% ± 1.9% |

The variance share is insensitive to scale within each family (max change 2% over 10x parameter increase) and insensitive to tokenization (BPE vs. SentencePiece: 0.8% difference). The theoretical lower bound computed from the estimated $I(X; Z \mid C)$ is 71–79% depending on domain, consistent with the observed 79–91% range.

---

## Ablations and Interpretation

- **Random-label control:** Replacing token category labels with random assignments increases $\sigma^2_B$ by 18% and $\sigma^2_W$ by 22%, confirming that the variance decomposition is meaningful rather than an artifact of the category system.
- **Layer depth:** The within-class variance share increases monotonically from input to final layer (from 55% at layer 1 to 84% at the final layer), suggesting that the model progressively specializes representations toward context-specific rather than category-level structure.
- **Implication for fine-tuning:** Fine-tuning on narrow classification tasks (e.g., sentiment) reduces $\sigma^2_W$ within the target label space but cannot fully eliminate it; models fine-tuned to near-collapse generalize poorly on out-of-distribution tokens.
- **Connection to memorization:** The information floor is higher for rare tokens (those with high $I(X; Z \mid C)$ due to unique distributional contexts), consistent with the finding that rare tokens are disproportionately memorized.

---

## Reference

Bruno Abrahao. **Neural Collapse Is Forbidden: Information Floors in Language Models.** arXiv:2607.09487, July 10, 2026. NYU Shanghai. https://arxiv.org/abs/2607.09487
