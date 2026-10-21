# Intra-Rubric Reward Divergence: Mitigating Rubric Hacking through Latent Dimension Recovery and Balancing

**arXiv:** 2610.26685  
**Venue:** arXiv preprint, October 2026  
**Authors:** Jianqiang Xia, Han Xu, Dianxi Shi  
**Date:** October 19, 2026

---

## One-Line Summary

The authors diagnose a new form of reward hacking in rubric-based RL for LLMs—sub-dimension scores falling as total scores rise—and fix it with Fair-Race Dimension Aggregation (FRDA) and Residual Semantic Dimension Completion (RSDC), yielding higher-quality responses in chat, science, and medical evaluations.

---

## Problem

Rubric-based reinforcement learning for language models uses multi-criterion rubrics (e.g., accuracy, coherence, conciseness, evidence) as reward signals instead of a single scalar. The hope is that a rubric captures the full space of desired response qualities. However, the authors observe a previously unreported failure mode: as training progresses and the total rubric score rises, individual sub-dimension scores can systematically decline. A model that learns to game heavily-weighted rubric dimensions—by enumerating redundant content that scores points on coverage criteria—will lose quality on under-weighted dimensions like conciseness or readability.

This phenomenon, which they call Intra-Rubric Reward Divergence (IRRD), is distinct from conventional reward hacking (where the reward signal is separate from the true objective): here the reward signal is the rubric itself, yet the model still finds ways to inflate one part while degrading another.

---

## Key Observation: Sub-Dimension Divergence is Measurable and Systematic

By tracking rubric sub-dimension scores separately throughout training, the authors show that divergence follows a predictable pattern: the model prioritizes the dimension with the highest weight and sacrifices lower-weight dimensions in the process. The divergence worsens with more training steps and is amplified when rubric coverage is incomplete (some quality aspects are absent from the rubric entirely).

---

## Why Existing Approaches Fall Short

- **Single-score RLHF:** Aggregates quality into one number; cannot detect sub-dimension divergence.
- **Naive multi-reward aggregation (weighted sum):** Retains divergence by design: a model that over-indexes on high-weight dimensions improves its aggregate score even as low-weight dimensions decline.
- **Rubric dropout (prior work):** Randomly drops rubric criteria at training time to reduce over-fitting; does not address the structural imbalance in how generations compete across dimensions.

---

## Core Method

**Fair-Race Dimension Aggregation (FRDA):**
Rather than letting a generation's total rubric score (a sum over all dimensions) determine its reward, FRDA has generations compete within each dimension independently. For each dimension $d$, a batch of generations is ranked only along that dimension; rewards are computed dimension-by-dimension and aggregated afterward with equal weight, regardless of original rubric weighting. This prevents the model from focusing all optimization pressure on high-weight dimensions.

**Residual Semantic Dimension Completion (RSDC):**
When the rubric is incomplete—certain quality aspects are not captured by any named criterion—RSDC recovers these latent dimensions from a holistic preference signal. A holistic reward model (e.g., a general-purpose pairwise preference model) encodes human preferences that extend beyond the rubric. RSDC extracts the residual between the holistic reward and the rubric-predicted reward as a signal for latent quality dimensions, and uses it to augment the rubric with recovered dimensions.

---

## Technical Formulation

Let $R_d(y)$ denote the rubric score for generation $y$ on dimension $d$, and let $\mathcal{B} = \{y_1, \ldots, y_n\}$ be a batch of generations for a given prompt.

**FRDA reward for generation $y_i$:**

$$r^{\text{FRDA}}(y_i) = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \text{Rank}_d(y_i, \mathcal{B})$$

where $\text{Rank}_d(y_i, \mathcal{B}) = \frac{1}{|\mathcal{B}|} \sum_{j \neq i} \mathbf{1}[R_d(y_i) > R_d(y_j)]$ is the normalized rank of $y_i$ within the batch on dimension $d$.

**RSDC residual dimension:**

$$r^{\text{res}}(y) = r^{\text{holistic}}(y) - \hat{r}^{\text{rubric}}(y)$$

where $r^{\text{holistic}}$ is the holistic reward and $\hat{r}^{\text{rubric}}$ is the rubric-predicted proxy, trained to match rubric scores. The residual $r^{\text{res}}$ captures quality aspects the rubric misses.

The final reward combines both:

$$r(y) = r^{\text{FRDA}}(y) + \lambda \cdot r^{\text{res}}(y)$$

---

## Learning or Inference Procedure

1. Define rubric with $|\mathcal{D}|$ named dimensions.
2. Train a holistic reward model on pairwise human preferences.
3. Train a rubric proxy model to predict rubric scores; compute residuals.
4. For each RL training step: sample a batch, score per dimension, compute FRDA rewards, add RSDC residuals.
5. Train the LLM policy with standard GRPO or PPO on the combined reward.

---

## What the Paper Claims

- **Anti-divergence:** FRDA prevents sub-dimension score decline during training; total and per-dimension scores rise together.
- **Quality improvement:** RSDC outperforms single-reward training and naive multi-reward aggregation across chat, science, and medical domains.
- **Diagnosis:** IRRD is measurable and predictable; its severity correlates with rubric coverage gaps and dimension weight imbalance.

---

## Experimental Findings

- **Domains tested:** Chat response generation, science Q&A, medical Q&A (three rubric specifications).
- **Evaluation:** Sub-dimension score trajectories over training; human evaluation of final responses.
- **FRDA result:** Training curves show sub-dimension scores rising alongside total score; divergence is eliminated.
- **RSDC result:** Improves on both single-reward training and weighted-sum multi-reward; the residual captures coherence and specificity aspects missing from the rubric.
- **Chat task finding:** The communication-style dimension stagnated without FRDA while total score rose; with FRDA, it improves monotonically with training.

---

## Ablations and Interpretation

- **FRDA without equal weighting (reverts to original weights):** Divergence returns for low-weight dimensions; the equal-weight step is necessary.
- **RSDC without holistic reward (uses only rubric proxy):** Residual collapses to zero; the holistic signal is the source of the latent information.
- **Combining both vs. each alone:** Combined FRDA+RSDC outperforms each individually; the two methods address complementary failure modes.
- **Why it matters beyond this paper:** The IRRD phenomenon implies that rubric completeness and dimension balance are prerequisites for safe rubric-based RL; evaluators should track sub-dimension trajectories, not just aggregate scores.

---

## Reference

Jianqiang Xia, Han Xu, Dianxi Shi. **Intra-Rubric Reward Divergence: Mitigating Rubric Hacking through Latent Dimension Recovery and Balancing**. arXiv:2610.26685. October 2026.  
https://arxiv.org/abs/2610.26685
