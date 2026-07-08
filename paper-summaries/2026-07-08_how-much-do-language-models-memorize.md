# How Much Do Language Models Memorize?

**Authors:** John X. Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, Saeed Mahloujifar  
**Affiliation:** Meta FAIR, Google DeepMind, Cornell University, UC San Diego  
**Venue:** ICML 2026 Honorable Mention  
**arXiv:** 2505.24832  
**URL:** https://arxiv.org/abs/2505.24832

---

## Headline Finding

GPT-style transformer language models have a memorization capacity of approximately **3.6 bits per parameter**. The paper formally separates memorization from generalization and shows that models memorize until capacity fills, at which point "grokking" begins — providing the first quantitative theory of the grokking phase transition.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
- Memorization capacity of GPT-style models: **~3.6 bits per parameter**
- This is a universal constant across model sizes tested (100K–20M parameters)
- Grokking occurs when models shift from memorizing training data to learning generalizable representations
- The transition is predicted quantitatively: it begins when dataset size exceeds model capacity (bits)

### Level 2 — Core Mechanism
The paper formally decomposes the information a language model contains about a training datapoint into:
- **Unintended memorization** $I(M; Z_i)$: mutual information between the model and a specific training example
- **Generalization** $I(M; \mathcal{D}_\text{gen})$: mutual information between the model and the true data-generating distribution

Prior work conflated these by measuring extractability (can the training point be recovered?) which is a lower bound on memorization but not memorization itself. By eliminating generalization — training on random bitstrings where no generalizable pattern exists — the authors isolate pure memorization capacity.

### Level 3 — Methodology
The experimental approach:
1. Train transformer models of varying sizes on datasets of uniformly random bitstrings
2. Since bitstrings have no generalization signal, all model information is pure memorization
3. Measure training loss at convergence as a function of dataset size relative to model capacity
4. Identify the phase transition point (grokking onset) where training loss suddenly drops
5. Compute capacity as: $C = N \times 3.6$ bits for a model with $N$ parameters

### Level 4 — Technical Details
Let $M$ be the model parameters, $Z = (Z_1, \ldots, Z_n)$ training data, $\mathcal{D}$ the true distribution. The decomposition is:
$$I(M; Z_i) = \underbrace{I(M; Z_i \mid \mathcal{D})}_{\text{unintended memorization}} + \underbrace{I(M; \mathcal{D})}_{\text{generalization}}$$
On random bitstrings, $I(M; \mathcal{D}) = 0$ by construction (there is no distribution to learn), so $I(M; Z_i) = I(M; Z_i \mid \mathcal{D})$ directly. The total capacity is $C = \sum_i I(M; Z_i \mid \mathcal{D})$ summed over training points.

---

## Why Existing Approaches Fall Short

Previous studies measured memorization via **extraction attacks** (membership inference, verbatim reproduction) or **training loss gaps** between seen and unseen data. Both approaches confound memorization with generalization: a model might produce a training string not because it memorized that specific string but because similar strings appear in the data distribution. The 3.6 bits/parameter estimate is the first that cleanly separates these components.

---

## Significance

The 3.6 bits/parameter constant provides a practical handle on privacy risk and copyright exposure. A model with 7 billion parameters can memorize approximately 25 GB of content (7B × 3.6 bits ÷ 8). The grokking connection shows that training beyond capacity is not just computationally wasteful — it triggers the switch from memorization to generalization and can reduce privacy risk while improving OOD performance.
