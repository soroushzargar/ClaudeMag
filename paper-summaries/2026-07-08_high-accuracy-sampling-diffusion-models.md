# High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions

**Authors:** Fan Chen, Sinho Chewi, Constantinos Daskalakis, Alexander Rakhlin  
**Affiliation:** MIT, Yale University  
**Venue:** ICML 2026 Outstanding Paper (Oral, July 8, 2026)  
**arXiv:** 2602.01338  
**URL:** https://arxiv.org/abs/2602.01338

---

## Headline Finding

A new algorithm called First-Order Rejection Sampling (FORS) achieves δ-accurate samples from diffusion models in **polylog(1/δ) steps** — an exponential improvement over all prior samplers, which required poly(1/δ) steps. The result simultaneously settles the complexity of sampling from general log-concave distributions using only gradient queries.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
FORS obtains δ-error in:
- **Õ(d★ polylog(1/δ))** steps under minimal data assumptions, where d★ is the intrinsic data dimension
- **Õ(L polylog(1/δ))** steps under a non-uniform L-Lipschitz score condition
- **polylog(1/δ)** steps for general log-concave distributions (first result of this kind using only gradient evaluations)

All prior methods required poly(1/δ) steps — the improvement is exponential.

### Level 2 — Core Mechanism
Rejection sampling with exact densities achieves high accuracy easily, but computing exact densities is intractable in high dimensions. FORS simulates rejection sampling using **only first-order (score/gradient) queries**, bypassing density evaluations entirely. The key insight is that rejection sampling can be decomposed into comparisons between ratios that, while involving densities, can be estimated accurately from the score function alone given a careful construction.

### Level 3 — Methodology
The FORS meta-algorithm proceeds as follows:
1. Propose candidates from a tractable proposal distribution using score guidance
2. Estimate acceptance probabilities from accumulated score evaluations along the diffusion path
3. Accept/reject based on estimated ratios, with error bounded by the L² score accuracy
4. Iterate until the output distribution's TV distance to the target is ≤ δ

### Level 4 — Technical Details
Let $p_t$ be the marginal distribution of the reverse diffusion at time $t$, with score $s_t(x) = \nabla \log p_t(x)$. Given $\tilde{s}_t$ satisfying:
$$\mathbb{E}\!\left[\|\tilde{s}_t(x) - s_t(x)\|^2\right] \leq \tilde{\delta}^2$$
FORS achieves total variation error δ in $T = O(d_\star \cdot \text{polylog}(1/\delta))$ score evaluations when $\tilde{\delta} = O(\delta)$. The main lemma establishes that FORS acceptance probability estimates are unbiased with variance $O(\tilde{\delta}^2)$, enabling a union bound over $T$ steps.

---

## Why Existing Approaches Fall Short

All prior high-accuracy samplers for diffusion models relied on either:
- **Higher-order numerical methods** (Runge-Kutta, exponential integrators): these reduce local discretization error but still require poly(1/δ) steps globally
- **Density evaluation** (for rejection sampling): exact densities are unavailable in practice for learned diffusion models

The fundamental barrier was that moving from poly to polylog requires a principled rejection mechanism that doesn't touch densities. FORS provides exactly this.

---

## Significance

This result is a theoretical milestone in the foundations of generative modeling. It demonstrates that the quality ceiling of diffusion model sampling is not inherent to the paradigm — the polynomial step count of every deployed sampler is a barrier of *algorithmic design*, not information theory. Practical integration of FORS into production systems could meaningfully raise sample quality at fixed inference budgets.
