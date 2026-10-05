# Beyond Linear Concepts: Discovering and Aligning Non-Linear Concept Manifolds in Large Language Models

**arXiv:** 2610.01821  
**Authors:** Tido Specht, Elias Benedict Krey, Nils Neukirch, Nils Strodthoff  
**Affiliation:** University of Oldenburg; German Research Center for AI (DFKI)  
**Date:** October 1, 2026

---

## One-Line Summary

By modeling LLM concepts as low-dimensional non-linear manifolds instead of linear directions, and introducing the Concept-Based Alignment score to compare them across layers and models, this work reveals consistent block structures in intermediate layers and a systematic shift from syntax-dominated to mixed syntactic-semantic representations in late layers.

---

## Problem

Mechanistic interpretability (MI) of LLMs aims to understand how information is organized in their internal token representations. Existing MI methods—probes, dictionary learning, sparse autoencoders—extract concepts under an implicit linearity assumption: a concept corresponds to a direction in activation space, and the presence or absence of that concept is measured by a linear classifier. However, there is growing empirical evidence of non-linear manifold structure in LLM activations, suggesting that the linearity assumption discards important geometric information and may lead to incomplete or incorrect interpretations of how concepts are organized and computed.

---

## Why Existing Approaches Fall Short

- **Linear probes** measure concept presence as the dot product with a learned direction, missing curvilinear or non-linearly distributed concept regions.
- **Sparse autoencoders (SAEs)** decompose activations into sparse linear combinations of dictionary features, but each feature is itself a linear direction and the reconstruction is a linear sum.
- **PCA-based comparisons** compare subspaces across layers using canonical angles (e.g., CKA), which are insensitive to non-linear geometry and miss manifold-level structural differences.
- **No cross-model alignment:** Existing methods lack a rigorous way to compare concept organization across different model architectures or training regimes without explicit feature matching.

---

## Core Method

**Non-Linear Multi-Dimensional Concept Discovery (NLMCD).** Originally developed for computer vision, NLMCD models each concept as a low-dimensional Riemannian manifold embedded in the high-dimensional activation space. The authors adapt NLMCD to token-level LLM activations by:

1. Collecting a corpus of token activations at each transformer layer, labelled by linguistic category (syntactic role, semantic class, morphological feature).
2. Fitting a concept manifold for each category using locally linear embedding (LLE) combined with a neural network atlas, yielding a low-dimensional parameterization of the concept's geometry.
3. Comparing manifolds across layers and models using the proposed Concept-Based Alignment (CBA) score.

**Concept-Based Alignment (CBA) Score.** Rather than matching individual feature vectors, CBA measures whether two manifolds partition the same set of token activations into nearby groups, without requiring an explicit bijection between manifold parameterizations. Formally, CBA is a generalized Rand index over neighborhood graphs derived from each manifold's geodesic distances:
$$\text{CBA}(\mathcal{M}_A, \mathcal{M}_B) = \frac{|\{(i,j): \text{near in } \mathcal{M}_A \Leftrightarrow \text{near in } \mathcal{M}_B\}|}{|\text{all pairs}|}$$
A CBA of 1.0 indicates that both manifolds partition the token space identically; 0.5 corresponds to chance-level alignment.

---

## Technical Formulation

Let $\mathbf{h}_l^{(t)} \in \mathbb{R}^d$ be the activation of token $t$ at layer $l$. For concept $c$, collect the activation set $\mathcal{H}_c^l = \{\mathbf{h}_l^{(t)}\}_{t \in \mathcal{T}_c}$ where $\mathcal{T}_c$ are all tokens belonging to concept category $c$.

The NLMCD manifold $\mathcal{M}_c^l$ is fitted as:
$$\min_{\phi: \mathbb{R}^k \to \mathbb{R}^d} \sum_{t \in \mathcal{T}_c} \|\mathbf{h}_l^{(t)} - \phi(z^{(t)})\|^2 + \lambda \cdot \text{Reg}(\phi)$$
where $z^{(t)} \in \mathbb{R}^k$ are the $k$-dimensional latent coordinates (with $k \ll d$) and $\text{Reg}(\phi)$ is a smoothness regularizer on the atlas map $\phi$.

The CBA between layer $l$ and layer $l'$ for concept $c$ is then:
$$\text{CBA}(\mathcal{M}_c^l, \mathcal{M}_c^{l'}) = \text{RandIndex}(G_c^l, G_c^{l'})$$
where $G_c^l$ is a neighborhood graph on $\mathcal{T}_c$ derived from geodesic distances on $\mathcal{M}_c^l$.

---

## Learning or Inference Procedure

**Manifold fitting (offline):**
1. Run the LLM on a large token corpus (e.g., 10M tokens from Wikipedia + code).
2. Extract activations $\{\mathbf{h}_l^{(t)}\}$ for all tokens and all layers.
3. For each concept category $c$ and layer $l$, fit NLMCD manifold $\mathcal{M}_c^l$ using stochastic gradient descent on the atlas network $\phi$.

**CBA computation (post-fitting):**
1. Compute pairwise geodesic distances on each fitted manifold.
2. Build neighborhood graphs $G_c^l$ and compute Rand index between all pairs of (layer, model) combinations.
3. Visualize as CBA alignment matrices showing which layer intervals share similar concept organization.

---

## What the Guarantee Says

NLMCD provides a consistent manifold estimate as the number of token samples grows to infinity, under standard regularity conditions on the manifold geometry. The CBA score is a metric in the space of partitions. The paper's main empirical claim—that block structures are consistent across models—is validated by showing that CBA alignment matrices have near-identical block patterns for five different LLM architectures, including architectures trained on different data regimes.

---

## Experimental Findings

- **Block structures:** CBA alignment matrices reveal consistent block patterns at intermediate layers (roughly layers 10–20 in a 32-layer model), corresponding to intervals where syntactic concept organization stabilizes before transitioning.
- **Syntax-to-mixed-semantics transition:** In all tested models, syntactic concepts (POS tags, dependency roles) dominate CBA structure through approximately 70% of the network; in the final 30% of layers, semantic concepts (entity type, sentiment, topic) become increasingly prominent.
- **Cross-model consistency:** Block structures are consistent across models from different families (Llama, Mistral, Falcon) at equivalent depth fractions, suggesting a universal organizational principle.
- **CBA vs. linear baselines:** CBA is 15–30% more sensitive to genuine structural differences than PCA-based or CKA-based comparisons, as measured by its ability to detect known synthetic perturbations inserted into activations.
- **Non-linearity significance:** Fitting linear probes instead of NLMCD manifolds produces CBA patterns that miss the block transitions, confirming that the non-linear manifold geometry carries structural information invisible to linear methods.

---

## Ablations and Interpretation

- **Manifold dimensionality $k$:** Values of $k \in \{2, 4, 8\}$ all produce similar CBA patterns; $k=4$ offers the best trade-off between fidelity and fitting cost.
- **Corpus composition:** Models analyzed on code-heavy versus text-heavy corpora show similar syntactic block patterns but differ in the semantic concept structure of late layers, aligning with known training data differences.
- **Token granularity:** Character-level tokens show less pronounced block structure than subword tokens, indicating the organizational patterns are tightly coupled to the tokenization scheme.

---

## Reference

Tido Specht, Elias Benedict Krey, Nils Neukirch, Nils Strodthoff. **Beyond Linear Concepts: Discovering and Aligning Non-Linear Concept Manifolds in Large Language Models**. arXiv:2610.01821, October 2026.  
https://arxiv.org/abs/2610.01821
