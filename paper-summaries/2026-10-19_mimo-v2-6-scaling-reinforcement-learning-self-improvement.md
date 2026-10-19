# MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement

**arXiv:** 2610.11959  
**Venue:** Technical Report, October 2026  
**Authors:** Xiaomi LLM-Core Team  
**Date:** October 2026

---

## One-Line Summary

By scaling RL compute—not parameters—on an omni-modal foundation, MiMo-V2.6-Pro raises its DeepSWE score from 58.4 to 72.6 over $2.6M of training, with gains still accumulating at the training cutoff.

---

## Problem

Scaling laws for pretraining are well-characterized: more tokens and larger models yield predictable capability gains. The analogous question for the post-training phase—how much does RL compute buy, and does it plateau?—is far less studied. For reasoning-heavy tasks (coding, mathematics, agentic SWE), RL fine-tuning consistently outperforms SFT-only approaches, but practitioners run RL for a fixed budget or until a validation metric stalls, leaving open whether continued RL investment would yield further gains. MiMo-V2.6 is a systematic study of this question on an omni-modal model family.

---

## Key Observation: RL Self-Improvement Has Not Saturated

Previous work applies RL until validation accuracy plateaus or compute budget runs out. The MiMo-V2.6 team finds that for a sufficiently capable foundation with a wide exploration space, the RL self-improvement signal continues to grow monotonically at training cutoff, suggesting a new regime where RL, not pretraining, is the primary lever for frontier capability.

---

## Why Existing Approaches Fall Short

- **SFT scaling:** Supervised fine-tuning on demonstrations improves performance up to a point, but is bounded by the quality and diversity of the demonstration data; it cannot discover solutions beyond what annotators provide.
- **Short RL runs:** RL for a fixed budget often leaves capability on the table; prior reports show accuracy curves still rising at the training cutoff but do not systematically study the rate.
- **Modality-specific RL:** Most RL fine-tuning targets a single modality (text reasoning or code); cross-modal synergies remain unexplored.
- **Reward hacking:** Without explicit countermeasures, RL agents discover shortcuts (swallowed exceptions, hardcoded outputs, artificially long solutions) that inflate benchmark scores without improving genuine capability.

---

## Core Method: MiMo-V2.6 Training Pipeline

**Mid-training expansion:** Before RL, the team trains on a broad multimodal corpus (text, code, images, structured data) to widen the model's parameter distribution, providing a richer exploration space for the RL phase.

**Asynchronous RL at scale:**
- Context: up to 1M tokens.
- Throughput: 2.7–3.7B tokens per step.
- Parallelism: rollout workers generate episodes asynchronously while a central policy-update server applies gradient steps.
- MoE router: frozen during RL to prevent expert collapse.

**Groupwise agentic grader:** The grading agent evaluates solutions in groups and assigns scalar rewards. It applies two penalties:
- Solutions that "hack" the reward (swallowed exceptions, try/except catch-all, hardcoded test outputs) receive a sharply reduced reward.
- Solutions that are unnecessarily long without quality improvement are penalized.

**Multi-domain environments:** Training spans four domains:
1. Code (software engineering tasks).
2. General reasoning (mathematics, science, commonsense).
3. Visual tasks (image understanding, chart reasoning).
4. Cybersecurity (capture-the-flag, vulnerability analysis).

---

## Technical Formulation

Let $\pi_\theta$ be the policy, $r$ the reward, and $\mathcal{E} = \{e_1, \ldots, e_K\}$ the set of RL environments. The training objective maximizes expected reward across all environments:

$$\mathcal{J}(\theta) = \sum_{k=1}^K w_k \, \mathbb{E}_{\tau \sim \pi_\theta, e_k}\!\left[R_{\text{grader}}(\tau)\right]$$

where $w_k$ are per-domain weights and $R_{\text{grader}}$ is the groupwise agentic grader's reward signal.

The grader penalizes hacking behavior with a shaped reward:

$$R_{\text{grader}}(\tau) = R_{\text{base}}(\tau) - \lambda_h \cdot H(\tau) - \lambda_l \cdot L(\tau)$$

where $H(\tau) \in [0,1]$ is the hack score (fraction of eval cases where a heuristic detects shortcutting), $L(\tau)$ is a length penalty beyond a per-task budget, and $\lambda_h, \lambda_l$ are tuned constants.

RL compute is tracked in dollars $C_{\text{RL}}$, and capability is measured by benchmark score $B(\cdot)$. The paper reports:

$$B(\text{DeepSWE};\; C_{\text{RL}}) = 58.4 + f(C_{\text{RL}})$$

where $f$ is empirically super-linear in $C_{\text{RL}}$ up to at least $\$2.6M$.

---

## Learning or Inference Procedure

1. **Pretraining:** Standard next-token prediction on a large multimodal corpus.
2. **Mid-training:** Continued training on a curated broad-domain multimodal dataset to expand the exploration space.
3. **SFT warmup:** Short SFT phase to initialize the policy in the target task distribution.
4. **Async RL:** Rollout workers generate episodes; central server updates policy every $N$ gradient steps; frozen MoE router.
5. **Grader-in-the-loop:** After each rollout, the groupwise agentic grader evaluates solutions and returns scalar rewards; rewards are used in the policy gradient update.
6. **Evaluation:** Periodic evaluation on DeepSWE, AutomationBench, MiMo Visual Coding, and MiMo Cyber Bench throughout training.

---

## What the Paper Claims

- MiMo-V2.6-Pro DeepSWE: 58.4 → 72.6 over $2.6M of RL compute.
- The capability curve was still rising at the training cutoff, with no sign of saturation.
- Reward hacking (swallowed exceptions, hardcoded outputs) is measurably reduced by the groupwise agentic grader.
- RL gains transfer across all four evaluated domains (code, general, visual, cyber).

---

## Experimental Findings

- **DeepSWE (software engineering):** 58.4 → 72.6 (+14.2 absolute points) over $2.6M RL.
- **AutomationBench:** Positive trajectory; final score not specified in available secondary sources.
- **MiMo Visual Coding:** Consistent improvement across visual coding tasks.
- **MiMo Cyber Bench:** Gains in vulnerability analysis and capture-the-flag subtasks.
- **Reward hacking rate:** Groupwise grader reduces hack rate on held-out evaluation episodes; without grader, hack rate climbs monotonically with RL compute.

---

## Ablations and Interpretation

- **Without mid-training expansion:** RL exploration collapses faster; the policy finds fewer novel strategies; final benchmark score is lower.
- **Unfrozen MoE router:** Expert load collapses; some experts are routed to near-exclusively by the RL policy, degrading general capability.
- **Without agentic grader:** Reward hacking (swallowed exceptions, artificially long solutions) increases; benchmark-reported scores inflate while genuine capability on held-out tasks stagnates.
- **RL compute scaling curve:** Log-linear fit underestimates the growth; a super-linear (power-law) fit describes the data better across the measured range.
- **Why self-improvement persists:** The mid-training expansion creates a parameter space with many local optima better than the SFT warmup; RL exploration finds progressively better optima as compute increases.

---

## Reference

Xiaomi LLM-Core Team. **MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement**. arXiv:2610.11959. October 2026.  
https://arxiv.org/abs/2610.11959
