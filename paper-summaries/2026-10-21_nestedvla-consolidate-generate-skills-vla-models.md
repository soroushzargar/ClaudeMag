# NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models

**arXiv:** 2610.23271  
**Venue:** NeurIPS 2026  
**Authors:** Guodong Du, Wanyu Lin  
**Affiliations:** WanyuGroup  
**Date:** October 16, 2026

---

## One-Line Summary

NestedVLA replaces model-merging-based skill consolidation with a nested hypernetwork that generates task-adaptive action-expert parameters on demand, achieving a +41 pp average gain over the strongest merging baseline across MetaWorld, ManiSkill, and CALVIN.

---

## Problem

A generalist robot manipulation agent must master dozens of qualitatively different skills—grasping, pushing, drawing, stacking—that typically come from heterogeneous data sources collected on different hardware, under different camera setups, and often on different robotic platforms. Three obstacles block naive consolidation:

1. **Data splits across platforms:** Joint training requires aggregating incompatible datasets, each with its own action space and sensor modality.
2. **Multi-task interference:** A shared model gradient-updates toward all tasks simultaneously; tasks with conflicting gradient directions degrade each other.
3. **Parameter conflicts in model merging:** Weight-averaging or task-arithmetic methods produce a merged checkpoint whose parameters conflict, especially when task-specific policy heads differ in initialization.

---

## Key Observation: Generate, Don't Aggregate

Standard consolidation aggregates parameters (via averaging, task arithmetic, or mixture-of-experts routing). NestedVLA replaces aggregation with generation: a hypernetwork learns to generate task-specific parameters from task context, avoiding the conflicts that arise when incompatible parameters are combined.

---

## Why Existing Approaches Fall Short

- **Joint multi-task training:** Suffers from gradient interference when tasks require conflicting action policies; catastrophic forgetting arises as new skills are added.
- **Model merging (TIES, DARE, task arithmetic):** Computes a single merged set of weights; fine-grained task-specific differences are smoothed out and performance degrades on all tasks.
- **Mixture of experts (MoE) with routing:** Requires training all experts end-to-end simultaneously; does not admit post-hoc addition of new skills without full retraining.
- **Adapter-based methods:** Insert adapters per-task but do not share structure across tasks; scaling to many tasks becomes expensive.

---

## Core Method

**Two-stage learning:**
1. **Independent skill learning:** Each skill is trained independently on its own data, producing a set of action-expert parameter deltas $\Delta\theta_k$ for skill $k$ relative to a shared base VLA.
2. **Post-hoc consolidation via hypernetwork generation:** A nested hypernetwork $\mathcal{H}$ takes task context $\tau$ (a text description or a few-shot example embedding) and generates $\hat{\Delta\theta}(\tau) \approx \Delta\theta_k$ for the appropriate skill. The generated delta composes additively with the frozen base VLA.

**Parameter delta encoding:** Each $\Delta\theta_k$ is compressed into a fixed-size patch token $e_k$ using a learned encoder. The hypernetwork is conditioned on both the task context and these patch tokens during training, learning to decode the right delta for a given context.

**VLM residuals:** Task-specific visual-language model residuals (shifts in text-conditioned attention) are stored separately as small compressed vectors and retrieved at inference alongside the generated delta.

---

## Technical Formulation

Let $\theta_0$ be the frozen base VLA parameters. For skill $k$, the effective policy parameters are:

$$\theta_k = \theta_0 + \Delta\theta_k$$

The hypernetwork $\mathcal{H}_\phi$ learns:

$$\hat{\Delta\theta}(\tau) = \mathcal{H}_\phi(\tau,\, \{e_k\}_{k=1}^K)$$

where $\tau$ is the task context embedding and $e_k = \text{Enc}(\Delta\theta_k)$ is the compressed skill token.

The training objective is:

$$\mathcal{L}(\phi) = \mathbb{E}_{k, (s,a) \sim \mathcal{D}_k}\!\left[\left\|\hat{\Delta\theta}(\tau_k) - \Delta\theta_k^*\right\|_F^2 + \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{BC}}(\theta_0 + \hat{\Delta\theta}(\tau_k);\, s, a)\right]$$

where $\mathcal{L}_{\text{BC}}$ is the behavioral cloning loss on skill-$k$ demonstrations and $\|\cdot\|_F$ is the Frobenius norm of the parameter delta.

---

## Learning or Inference Procedure

1. Train each skill independently on its data to obtain $\Delta\theta_k^*$.
2. Compress each delta: $e_k = \text{Enc}(\Delta\theta_k^*)$.
3. Train the hypernetwork $\mathcal{H}_\phi$ to recover task deltas from task context and skill tokens.
4. Store $\{e_k\}$ and VLM residuals in a skill library.
5. At inference: encode the task context $\tau$; retrieve relevant $e_k$ from the library; run $\mathcal{H}_\phi$ to generate $\hat{\Delta\theta}$; execute with $\theta_0 + \hat{\Delta\theta}$.
6. Adding a new skill requires only training it independently, encoding its delta, and adding the token to the library—no global retraining.

---

## What the Paper Claims

- **Accuracy:** +41 pp average success rate over the strongest model-merging baseline across MetaWorld, ManiSkill, and CALVIN.
- **Scalability:** Adding new skills post-hoc requires no retraining of existing skills or the base model.
- **Storage efficiency:** Compressed skill tokens are small; the skill library scales sub-linearly with the number of skills.
- **Generalization:** The hypernetwork generalizes to held-out task phrasings not seen during training.

---

## Experimental Findings

- **Benchmarks:** MetaWorld (50 tasks), ManiSkill (diverse manipulation tasks), CALVIN (language-conditioned manipulation).
- **Baselines:** Task arithmetic, TIES, DARE, gradient-surgery multi-task, MoE routing.
- **Key result:** +41 pp average over the best merging baseline; largest gains on tasks where merging introduces severe parameter conflicts.
- **Ablation on compression ratio:** More aggressive compression of $\Delta\theta_k$ degrades performance on fine-grained dexterous tasks; a trade-off between storage and accuracy exists.

---

## Ablations and Interpretation

- **Without hypernetwork generation (direct merge):** Returns to baseline merging performance; the generation step is the key ingredient.
- **Without per-skill independent pretraining:** Hypernetwork training becomes unstable; the fixed deltas provide an anchor.
- **Without VLM residuals:** Performance drops on language-conditioned benchmarks (CALVIN); the residuals carry task-specific linguistic context the action expert alone cannot reconstruct.
- **Why generation beats aggregation:** Parameter averaging smooths out task-specific directions in parameter space; generation preserves them by learning a structured mapping from context to delta.

---

## Reference

Guodong Du, Wanyu Lin. **NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models**. arXiv:2610.23271. NeurIPS 2026.  
https://arxiv.org/abs/2610.23271
