# Beyond Repeated Sampling: Learning Search Policies for LLM Reasoning

**arXiv:** 2609.26704  
**Authors:** Ismail Labiad, Matthieu Kowalski, Marc Schoenauer, Rémi Munos, Julia Kempe  
**Affiliation:** Meta FAIR  
**Date:** September 2026

---

## One-Line Summary

A small concept generator, trained with RL against a frozen large answer generator, steers test-time exploration at a semantic level and substantially outperforms naive repeated sampling on hard reasoning problems.

---

## Problem

Large language models increasingly allocate test-time compute to improve reasoning performance. The dominant strategy is naive repeated sampling: draw many independent solutions and pick the best one. Because sampling explores only through local decoding noise, it tends to produce many near-duplicate attempts rather than genuinely different ideas. Hard reasoning problems — where the correct approach may require an insight that is far from the model's prior mode — are precisely where this local exploration fails.

---

## Why Existing Approaches Fall Short

- **Repeated sampling** (pass@k, majority vote, best-of-N) explores only through decoding stochasticity; temperature and top-p control local diversity but not semantic diversity.
- **Self-consistency** and **majority voting** improve answer reliability by aggregating samples, but do not steer individual samples toward unexplored strategies.
- **Process reward models (PRMs)** and **tree search (MCTS)** require per-step scoring or an explicit search tree, which are expensive and require a well-trained verifier.
- **Larger models** at the same compute are costly; the goal is to make a fixed frozen large model explore more effectively.

None of these methods train an explicit, reusable search policy; each sampling call is independent of strategy.

---

## Core Method

**Semantic Exploration via Concept Conditioning.** Instead of conditioning the answer generator on just the problem prompt, the method first samples a *concept*—a problem-specific hint, strategy, or approach—and then conditions answer generation on (problem, concept). Concepts are expressed in natural language and can encode diverse mathematical strategies, decompositions, or analogies.

**Diverse Concept Sampling.** A single trajectory of the concept generator emits many diverse concepts at once (rather than one per call), encouraging spread across qualitatively different strategies before any answer is generated.

**Concept Generator with RL.** A small concept generator $\pi_\theta$ (significantly smaller than the answer generator) is optimized with reinforcement learning:

$$\mathcal{L}(\theta) = -\mathbb{E}_{c \sim \pi_\theta(\cdot|x)}\left[\text{pass@k}(f(x, c))\right]$$

where $x$ is the problem, $c$ is a concept, and $f$ is the frozen large answer generator. The reward is whether the answer generator, conditioned on $c$, succeeds on the problem. The concept generator learns which types of strategies lead the answer generator to success.

**Decoupling Roles.** The concept generator is responsible for exploration; the answer generator is responsible for execution. The frozen answer generator is never fine-tuned.

---

## Technical Formulation

Let $x$ denote a problem instance. Let $\pi_\theta$ be the concept generator and $f$ the frozen answer generator.

The conditional answer distribution is:
$$p(a \mid x) = \int p(a \mid x, c) \, \pi_\theta(c \mid x) \, dc$$

Pass@k under concept-conditioned generation:
$$\text{pass@k}(x) = 1 - \binom{n-s}{k}\bigg/\binom{n}{k}$$

where $s$ is the number of correct answers in $n$ samples. The RL objective maximizes expected pass@k over a distribution of training problems:
$$\max_\theta \;\mathbb{E}_{x \sim \mathcal{D}}\left[\text{pass@k}_{c \sim \pi_\theta}(f(x, \cdot))\right]$$

The concept generator is trained with REINFORCE or a policy-gradient variant, using the correctness of the downstream answer as the scalar reward signal.

---

## Learning Procedure

1. **Concept pretraining:** $\pi_\theta$ is initialized from a pretrained small LM (e.g., a 1–3B parameter model).
2. **RL fine-tuning:** For each training problem $x$, sample $K$ concepts $c_1, \ldots, c_K \sim \pi_\theta(\cdot \mid x)$; for each concept, query $f$ and check correctness; use the binary reward signal to update $\pi_\theta$ via policy gradient.
3. **Inference:** Given a new problem, sample a batch of concepts from $\pi_\theta$ and generate answers from $f$ conditioned on each concept; return the answer with highest frequency or model confidence.

---

## What the Paper Claims

A small concept generator trained with RL learns a reusable search policy that:
1. Achieves substantially higher pass@k than naive repeated sampling at the same answer-generation budget.
2. Outperforms concepts drawn from much larger untuned models (showing the value of training, not just scale).
3. Transfers to answer generators the concept generator was never trained against, including models from a different model family.

---

## Experimental Findings

- Evaluated on hard mathematical reasoning problems where repeated sampling plateaus.
- Concept-conditioned generation substantially improves pass@k over naive repeated sampling at the same total answer-generation compute budget.
- Trained concept generator surpasses concepts from untuned models many times larger.
- Zero-shot transfer to unseen answer generators is effective, demonstrating that the learned search policy captures reusable, problem-structure-driven strategies.

---

## Ablations and Interpretation

- **Single-concept vs. multi-concept trajectories:** Emitting diverse concepts in a single trajectory improves pass@k further by reducing concept redundancy.
- **Concept generator size vs. quality:** Smaller generators trained with RL outperform larger generators prompted zero-shot, confirming that learning, not parameter count, is the key factor.
- **Reward signal:** Using answer correctness as the sole training signal is sufficient; no dense process reward model is needed.

---

## Reference

Ismail Labiad, Matthieu Kowalski, Marc Schoenauer, Rémi Munos, Julia Kempe. **Beyond Repeated Sampling: Learning Search Policies for LLM Reasoning**. arXiv:2609.26704, September 2026.  
https://arxiv.org/abs/2609.26704
