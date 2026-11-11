# Scaling Laws for SAE Training Data

**URL:** https://neurips.cc/virtual/2026/poster/148634  
**Authors:** Nikita Koriagin, Nikita Balagansky, Daniil Gavrilov (T-Tech)  
**Venue:** NeurIPS 2026 (also ICML 2026)  
**URL:** https://neurips.cc/virtual/2026/poster/148634

---

## Summary

Sparse autoencoders (SAEs) are a leading tool for mechanistic interpretability of large language models, decomposing residual stream activations into sparse linear combinations of learned features. Training large SAEs is computationally expensive partly because of activation storage: generating and storing activations from billions of tokens of forward passes can require more disk and I/O bandwidth than the SAE training itself. The authors show that this bottleneck can be significantly reduced by exploiting a data scaling law specific to SAEs: quality depends on the number of *unique* activations in the training buffer, not on fresh tokens per step. After a short diversity-limited regime, replaying a fixed buffer of unique activations is as good as continuously generating new ones. This reduces activation storage requirements by 8–64× across a range of model families, layer positions, SAE architectures, and downstream evaluation metrics.

---

## Method

**The unique-activation hypothesis.** The authors observe empirically that two training runs with equal numbers of unique activations but different levels of data repetition (one fresh, one with replay) produce SAEs of comparable quality. This suggests that the relevant quantity for SAE training data is not the total token budget but the buffer diversity — the number of distinct activations the SAE sees at least once.

**Data-constrained scaling law.** They fit a scaling law of the form:
  L(n_unique) = L∞ + C · n_unique^{-α}

where L is a quality metric (e.g., reconstruction loss, dead-neuron fraction), n_unique is the number of unique activations in the buffer, L∞ is the asymptotic quality, C is a constant, and α is the data-scaling exponent. This law accurately describes quality as a function of buffer size across all evaluated settings.

**Two regimes.** The law implies two regimes:
1. *Diversity-limited regime* (n_unique < n*): Each new unique activation meaningfully reduces loss; fresh data should be prioritized.
2. *Replay regime* (n_unique ≥ n*): Additional unique activations provide negligible benefit; replay the existing buffer for the rest of the training budget.

**Buffer-sizing rule.** Given a training token budget T and a quality target, the optimal strategy is: (1) determine n* from the fitted scaling law; (2) collect n* unique activations; (3) fill the remaining T - n* training steps with uniform random replay from the buffer.

---

## Experimental Setup

Validated across:
- **Models:** Llama-3.1 (8B) and Qwen3 (7B)
- **Layers:** Early (layer 4), middle (layer 16), late (layer 28)
- **SAE architectures:** BatchTopK, TopK, JumpReLU variants
- **Metrics:** Reconstruction loss (L2), dead-neuron fraction, feature absorption (downstream classification), automated interpretability score (GPT-4-based feature description coherence), sparse probing accuracy
- **Budget range:** 50M to 10B activation tokens

---

## Results

| Setting | Storage Reduction | Quality Loss |
|---------|------------------|--------------|
| Llama-3.1 early layers | 64× | < 0.5% |
| Llama-3.1 middle layers | 32× | < 0.3% |
| Qwen3 all layers | 8–32× | < 1% |
| Automated interpretability | 16× | < 2% |

The storage reduction is largest for early layers (where activations are more stereotyped) and smallest for late layers (where activations carry more semantic diversity). The interpretability-metric results are slightly more variable, but storage reductions of 8–16× remain achievable with less than 2% quality degradation.

---

## Key Insight

The critical insight is that SAE training, unlike standard neural network training, saturates rapidly with respect to data novelty. The core quantity the SAE needs is representational coverage — a diverse sample from the activation distribution — not an endless stream of fresh examples. Once the buffer achieves adequate coverage, replay is essentially free from a learning perspective. This separates the data-collection problem (collect n* diverse activations once) from the training problem (optimize the SAE for T steps with any replay strategy), potentially unlocking much cheaper large-scale SAE training pipelines.

---

## Limitations

- The scaling law exponent α varies across layers and architectures; the buffer-sizing rule requires fitting α in each new setting, adding a calibration cost.
- The analysis assumes the activation distribution is stationary; for models fine-tuned or domain-adapted after SAE training, the buffer may not cover the new distribution.
- Very large SAEs (dictionary size > 100K) were not evaluated; it is unclear whether the unique-activation hypothesis holds at extreme scales where the SAE needs to learn rare features.
