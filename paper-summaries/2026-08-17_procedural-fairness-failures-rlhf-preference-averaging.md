# Procedural Fairness Failures in RLHF from Preference Averaging

**arXiv:** 2608.10126
**Authors:** M P V S Gopinadh, Karthik Kamuju, Kummari Avinash, John Joshua, Srinivasa Raju Rudraraju
**Venue:** ICLR 2026 Workshop on Algorithmic Fairness Across Alignment Procedures and Agentic Systems (AFAA)
**Date:** August 10, 2026
**URL:** https://arxiv.org/abs/2608.10126

---

## Summary

Reinforcement Learning from Human Feedback (RLHF) aggregates heterogeneous human preferences into a single reward model, implicitly assuming that preferences are homogeneous across annotators. When preferences are in fact diverse and heterogeneous, this aggregation produces a procedural fairness failure: majority preference groups dominate reward learning while minority preferences are systematically under-represented. The paper defines procedural fairness in the alignment context, proves that standard RLHF violates it through preference averaging, and proposes Preference-Aware RLHF (PA-RLHF), which separates optimization across preference modes at the reward learning stage.

## Problem

Human preference data is not homogeneous. Different annotators hold genuinely different values about desirable AI behavior: what one group considers helpful, another may consider harmful; what one group prefers stylistically, another may find off-putting. Standard RLHF treats all preference annotations as draws from a single distribution and fits a single reward model to their aggregate. This is formally equivalent to averaging preferences, and averaging is unfair when the distribution of preferences has multiple modes — when subgroups of annotators are genuinely different rather than noisy.

The paper distinguishes two types of fairness:
- **Outcome fairness:** Does the final model produce outputs that work equally well for different groups?
- **Procedural fairness:** Does the reward learning process represent all preference groups' signals equitably, regardless of group size?

Standard RLHF violates procedural fairness even if outcome fairness accidentally holds: minority preference signals are diluted and under-represented in the reward model, regardless of whether the final policy happens to satisfy them.

## Formal Framework

Let $\mathcal{A}$ be the set of annotators, partitioned into groups $G_1, \ldots, G_K$ with preference distributions $\mathcal{D}_1, \ldots, \mathcal{D}_K$. Standard RLHF fits a reward model $r_\theta$ to the mixture $\mathcal{D} = \sum_k \alpha_k \mathcal{D}_k$, where $\alpha_k = |G_k|/|\mathcal{A}|$ is the group's population share.

**Procedural fairness** requires that the reward learning procedure gives each group $G_k$ representation proportional to their right to be heard — not to their population size. Under this criterion, standard RLHF is procedurally unfair whenever $|G_k|$ varies across $k$ and preferences across groups are non-identical.

## Proposed Method: PA-RLHF

Preference-Aware RLHF separates reward optimization across preference modes:

1. **Preference clustering:** Cluster annotators into $K$ groups using embedding-based or label-based similarity.
2. **Per-group reward modeling:** Fit a separate reward model $r_{\theta_k}$ for each group $G_k$ using only that group's annotations.
3. **Preference-aware policy optimization:** During PPO, use a fairness-weighted combination of the per-group reward models rather than a single aggregated reward.

The fairness weighting can be uniform across groups (maximum procedural equity) or tuned to balance procedural and outcome fairness objectives.

## Experimental Setup

- **Setting:** LLM alignment via RLHF on preference datasets with synthetic annotation group structure
- **Baseline:** Standard RLHF with aggregated reward model
- **Evaluation:** Both procedural fairness metrics (group-level reward representation) and downstream policy quality metrics (win rates, helpfulness)

## Results

- PA-RLHF improves procedural fairness metrics across all tested scenarios
- Outcome quality (win rates, helpfulness) is maintained or improved relative to the standard RLHF baseline
- The fairness gains are largest when group preferences are most heterogeneous (high inter-group divergence) and group sizes are most unequal

## Why It Matters

As LLMs are deployed across diverse populations, the assumption that a single aggregated reward model can represent all users' values becomes increasingly untenable. This paper provides the first formal definition of procedural fairness for RLHF and a concrete, practical method to enforce it. The distinction between procedural and outcome fairness is particularly important: a model may produce acceptable outputs for minority groups by coincidence while the reward learning process systematically ignored them — and PA-RLHF addresses the process, not just the outcomes.

## Key Contributions

1. Formal definition of procedural fairness in the RLHF alignment context
2. Proof that standard RLHF violates procedural fairness via preference averaging
3. PA-RLHF: reward modeling separated across preference clusters to preserve distinct preference signals
4. Empirical demonstration that PA-RLHF improves procedural fairness while maintaining outcome quality
