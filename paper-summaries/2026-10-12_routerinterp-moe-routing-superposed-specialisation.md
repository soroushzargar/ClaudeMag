# RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing

**arXiv:** 2610.11775  
**Authors:** Ilya Lasy, Nora Yinuo Cai, Kola Ayonrinde  
**Venue:** ICML 2026  
**Date:** October 2026

---

## One-Line Summary

RouterInterp disproves the single-domain expert hypothesis in Mixture-of-Experts models, showing that each expert activates on a disjoint union of fine-grained features (Superposed Specialisation), and builds a method that uses Sparse Autoencoder features to automatically generate interpretable natural-language explanations of routing decisions.

---

## Problem

Mixture-of-Experts (MoE) language models route each token to a subset of experts, dramatically scaling model capacity without proportional compute. Despite their success, *why* specific tokens go to specific experts is poorly understood. The prevailing assumption—that each expert specialises in a single domain (e.g., "the code expert," "the French expert")—is convenient but may be an oversimplification. Understanding routing at a fine-grained level is important for auditing, debugging, and improving MoE architectures.

---

## Why Existing Approaches Fall Short

- **Token-statistics methods** compute frequency distributions of token types per expert and label each expert with the most common token category. These methods are coarse, miss polysemous patterns, and achieve low interpretability fidelity.
- **Probing classifiers** can predict routing from representations, but give no human-readable explanation.
- **Manual analysis** is not scalable across hundreds of experts and billions of parameters.
- The single-domain assumption ignores that routing decisions often correlate with multiple disjoint fine-grained categories simultaneously—a phenomenon the paper calls Superposed Specialisation.

---

## Core Method

RouterInterp is built around two contributions:

1. **The Superposed Specialisation Hypothesis (SSH):** Each expert does not specialise in one broad domain, but in a *disjoint union* of fine-grained linguistic or semantic features. This is analogous to superposition in polysemantic neurons, but at the routing level.

2. **SAE-based routing explanation:** 
   - Sparse Autoencoder (SAE) features are derived from the model's residual stream. Each SAE feature corresponds to a monosemantic, interpretable concept.
   - RouterInterp identifies the SAE features most predictive of a given expert's routing decisions via sparse linear probes.
   - It then aggregates the natural-language descriptions of those features to produce a unified textual explanation of each expert's specialisation.

---

## Technical Formulation

Let $\mathbf{h}_t \in \mathbb{R}^d$ be the residual stream activation at token $t$. A Sparse Autoencoder maps this to a sparse feature vector:
$$\mathbf{f}_t = \text{ReLU}(\mathbf{W}_\text{enc}\, \mathbf{h}_t + \mathbf{b}_\text{enc}), \quad \mathbf{f}_t \in \mathbb{R}^F$$

The routing decision for expert $e$ is:
$$r_{t,e} = \mathbf{1}[\text{expert } e \text{ is selected for token } t]$$

RouterInterp solves a sparse logistic regression for each expert:
$$\hat{r}_{t,e} = \sigma(\boldsymbol{\beta}_e^\top \mathbf{f}_t), \quad \boldsymbol{\beta}_e \text{ is } L_1\text{-penalised}$$

The non-zero entries of $\boldsymbol{\beta}_e$ identify the SAE features most predictive of routing to expert $e$. The corresponding feature descriptions $\{s_k : \beta_{e,k} \neq 0\}$ are composed into a unified natural-language explanation via a language model.

---

## Learning or Inference Procedure

1. **Train a SAE** on the frozen MoE model's residual stream to obtain monosemantic feature dictionaries.
2. **Collect routing labels** $r_{t,e}$ for a large corpus of tokens.
3. **Fit sparse probes** $\boldsymbol{\beta}_e$ for each expert $e$ over the SAE feature activations.
4. **Extract the top-$k$ predictive features** for each expert.
5. **Generate explanations** by passing the feature descriptions to a small LM that composes them into a coherent summary.

---

## What the Guarantee Says

RouterInterp provides an empirical guarantee via held-out routing prediction accuracy. The interpretability claim is validated by showing that the identified SAE features, when intervened upon (activating or suppressing them), causally influence routing decisions.

---

## Experimental Findings

- On **gpt-oss-20b**, RouterInterp achieves approximately **65% higher routing-detection accuracy** compared with token-statistics-based baselines.
- Manual inspection of explanations by domain experts confirms that the generated descriptions are specific, coherent, and capture non-obvious expert roles.
- The SSH is validated quantitatively: the top SAE features per expert span semantically disjoint categories (e.g., one expert routes on technical medical terminology AND on certain syntactic constructions in Spanish).
- The method scales across all expert layers in gpt-oss-20b without degradation.

---

## Ablations and Interpretation

**Full SAE features vs. raw activations:** Using raw residual stream activations rather than SAE features significantly degrades routing prediction accuracy and produces incoherent explanations.

**Top-$k$ features:** Including more features beyond the natural elbow in the probe's coefficient magnitude adds noise without improving routing accuracy.

**Causal intervention:** Artificially activating a predictive SAE feature at a token that would otherwise not route to expert $e$ causes increased routing probability to $e$, confirming the causal role of the features.

**Across layers:** Early layers show routing with fewer, more syntactic SAE features; later layers show richer semantic specialisation patterns.

---

## Reference

Ilya Lasy, Nora Yinuo Cai, Kola Ayonrinde.
**RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing.**
ICML 2026. arXiv:2610.11775, October 2026.
https://arxiv.org/abs/2610.11775
