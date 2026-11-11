# FEROM: Frontier Endogenous Reveal-Order Marginal Policy Optimization for Masked Diffusion LMs

**arXiv:** 2611.10369  
**Authors:** NeurIPS 2026 Submission (anonymous)  
**Venue:** NeurIPS 2026  
**Submitted:** 7 November 2026  
**URL:** https://arxiv.org/abs/2611.10369

---

## Summary

Masked diffusion language models (dLLMs) generate text by iteratively unmasking token positions. Applying reinforcement learning to improve these models requires a policy gradient over the unmasking process, but existing approaches either fix the unmasking schedule in advance (introducing bias) or sample a single trajectory through the exponentially large space of reveal orders (introducing high variance). FEROM derives a Rao-Blackwellized policy-gradient identity that marginalizes over all possible reveal orders rather than committing to any one trajectory, expressing the resulting estimator as posterior edge occupancy on a reveal-state directed acyclic graph (DAG). A budgeted tractable approximation called Frontier Reveal Marginalization (FRM) makes this practical and integrates directly into a GRPO-style objective.

---

## Method

**The reveal-state DAG.** Each node in the DAG represents a masking state (a specific assignment of which positions are revealed and which remain masked). Directed edges represent single unmasking transitions. The DAG grows exponentially in depth, but its structure is regular: at each level, exactly one additional position has been unmasked.

**Rao-Blackwellization.** Instead of evaluating the policy gradient along a single sampled path through the DAG, FEROM marginalizes over the full distribution of paths. This is a Rao-Blackwellization: the estimator has the same expectation as a single-path estimator but strictly lower variance. The key quantity is the posterior edge occupancy — the probability that a given unmasking transition was used, marginalized over all full reveal sequences consistent with the observed final state.

**Frontier Reveal Marginalization (FRM).** Computing exact edge occupancy requires summing over exponentially many paths. FRM approximates this by restricting attention to the "frontier" — the set of positions at the current boundary between revealed and masked tokens. This limits the number of active edges to at most the frontier width, making the computation tractable within a fixed budget.

**Integration into GRPO.** The edge-based surrogate replaces single-path log-scores in the GRPO-style objective. Rather than scoring a single sampled trajectory, FEROM scores the marginal contribution of each frontier transition, reducing both bias (from fixed schedules) and variance (from single-path sampling) in the policy gradient estimate.

---

## Experimental Setup

Experiments are conducted on masked diffusion language models in a reinforcement learning from human feedback (RLHF) or reward-guided fine-tuning setting. The base models include standard masked dLLMs trained on language tasks. FEROM is compared against fixed-schedule policy gradient baselines and single-path GRPO with the same reward model and training budget.

---

## Results

FEROM consistently outperforms fixed-schedule baselines in terms of reward maximization and generation quality. Compared to single-path GRPO, FEROM achieves lower policy gradient variance, faster convergence, and higher final reward across evaluated tasks. The improvements are most pronounced on tasks where the reward landscape is sensitive to the order of unmasking, confirming that the reveal-order marginalization provides meaningful signal.

---

## Key Insight

The core observation is that the choice of which token to unmask next is itself a policy decision with high consequence for the final output, yet standard policy gradient methods either ignore this structure (by fixing the schedule) or handle it noisily (by sampling one trajectory). By marginalizing over reveal orders via Rao-Blackwellization, FEROM converts this high-variance discrete choice into a smoother, lower-variance gradient signal — a strict improvement in gradient estimation quality with no additional model cost beyond the FRM computation.

---

## Limitations

- FRM is an approximation of exact marginalization; the quality of the approximation degrades if the frontier is wide relative to the available compute budget.
- The method is specific to masked (absorbing-state) diffusion models and does not directly apply to continuous diffusion or autoregressive models.
- The variance reduction is empirically observed but not accompanied by a formal bound on the gap between FRM and exact Rao-Blackwellization.
