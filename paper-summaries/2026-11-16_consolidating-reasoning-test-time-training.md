# Consolidating Reasoning with Test-Time Training

**Authors:** Haotian Wu, Zeming Chen, Hao Zhao, Antoine Bosselut  
**arXiv:** 2611.02104 (cs.AI, cs.CL)  
**Submitted:** 1 November 2026  
**Venue:** NeurIPS 2026  

---

## Headline

Encoding multiple parallel reasoning trajectories into a temporary LoRA adapter via test-time training lets a model *distil* collective reasoning into a single high-quality answer—outperforming naive ensemble voting without extra inference budget.

---

## Key Findings

- **TTT-Consolidate** (the proposed method) consistently outperforms best-of-N sampling and majority voting baselines on mathematical and commonsense reasoning benchmarks.
- An inner LoRA fine-tuning loop encodes N parallel reasoning trajectories into compact adapter weights at test time.
- An outer meta-learning loop, trainable with either SFT or RL objectives, learns initial adapter parameters that make consolidation useful for final answer synthesis.
- The companion dataset **OpenParallelThinking** is released to support further research on multi-agent reasoning consolidation.

---

## Motivation and Problem

Scaling test-time compute through parallel sampling—running N independent reasoning attempts—has become a standard strategy for improving model performance. However, the aggregation step is typically shallow: majority vote counts surface answers, best-of-N selects one trajectory, and verifiers score outputs without integrating complementary reasoning steps from different trajectories.

The fundamental problem is that diverse trajectories often reach the correct answer via *different valid paths*, and a correct answer in one trajectory might implicitly resolve an ambiguity encountered in another. No existing mechanism integrates these complementary insights at the level of reasoning content rather than final answers.

---

## Method: TTT-Consolidate

### Inner Loop: Test-Time LoRA Adaptation

Given N reasoning trajectories $\tau_1, \ldots, \tau_N$ produced by parallel sampling, TTT-Consolidate performs test-time training to absorb them into a LoRA adapter $\Delta\theta$:

$$\Delta\theta^* = \arg\min_{\Delta\theta} \sum_{i=1}^{N} \mathcal{L}_{\text{SFT}}\bigl(\text{LM}(\cdot \mid q; \theta + \Delta\theta),\, \tau_i\bigr)$$

This is a small number of gradient steps (typically 5–20), making it feasible at inference time. The adapter has rank $r \ll d$, so parameter count is negligible relative to the base model.

### Outer Loop: Meta-Learning for Consolidation

A key insight is that naive TTT (starting from zero adapter weights) does not automatically produce a consolidating adapter—it may simply memorize the most frequent trajectory. The outer loop meta-learns an initialization $\phi_0$ such that a few steps of inner-loop adaptation from $\phi_0$ on any set of trajectories produces an adapter that can synthesize a *better* final answer than any individual trajectory.

The outer-loop objective is:

$$\mathcal{L}_{\text{outer}}(\phi_0) = \mathbb{E}_{(q, \tau_{1:N})}\bigl[\mathcal{L}_{\text{eval}}\bigl(\text{LM}(a^* \mid q;\, \theta + \Delta\theta^*(\phi_0,\tau_{1:N}))\bigr)\bigr]$$

where $a^*$ is the gold answer and $\mathcal{L}_{\text{eval}}$ is either a cross-entropy loss (SFT variant) or a reward signal from a verifier (RL variant).

### Final Answer Generation

After inner-loop adaptation, the consolidated adapter is used to decode the final answer:

$$\hat{a} = \text{LM}(\cdot \mid q;\, \theta + \Delta\theta^*)$$

The adapter encodes the aggregate reasoning signal, so the final pass benefits from information across all N trajectories.

---

## Experimental Setup and Results

Benchmarks evaluated include MATH-500, GSM8K, AIME 2024/2025, and HellaSwag. Baselines include majority voting, best-of-N (with oracle), and verifier-guided selection with the same N trajectories.

**Key results:**
- On MATH-500 at N=16, TTT-Consolidate achieves 2.8 percentage points above majority voting and 1.4 above verifier-guided selection.
- On AIME-2025, improvement over majority voting is 4.1 points (absolute), showing larger gains on harder problems where trajectory diversity is highest.
- The RL outer-loop variant consistently outperforms the SFT variant when the base model's pass@1 is below 60%, suggesting RL is better calibrated when trajectories are noisy.
- Ablations show the outer-loop meta-initialization accounts for roughly half the gain; running TTT from scratch with random initialization recovers about half the benefit.

---

## OpenParallelThinking Dataset

The authors release OpenParallelThinking: a dataset of parallel reasoning trajectories collected from multiple open-weight models across diverse tasks. It includes trajectory diversity annotations and can be used for training or evaluating consolidation methods.

---

## Significance

TTT-Consolidate reframes multi-agent reasoning not as a voting problem but as a *learning* problem at inference time. The method elegantly connects test-time training (a technique originally developed for OOD generalization) with the emerging paradigm of parallel reasoning, demonstrating that adapter weights can serve as a compact summary of collective thought. The RL outer loop in particular aligns with the broader trend of training reasoning models with verifiable reward signals.
