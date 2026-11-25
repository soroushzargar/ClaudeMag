# SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning

**Authors:** Alexis Limozin, Eduard Durech, Torsten Hoefler, Imanol Schlag, Valentina Pyatkin
**arXiv:** 2604.23747 (cs.LG, cs.CL)
**Submitted:** 26 April 2026
**Venue:** NeurIPS 2026 (poster)

---

## Headline

After correcting an optimizer bug and a loss aggregation bug in the SFT baseline, the standard sequential SFT-then-RL pipeline outperforms every published mixed-policy method by up to +22.2 points on reasoning benchmarks—and even a 50-step truncated variant beats complex mixed approaches while using fewer FLOPs.

---

## Key Findings

- Two bugs in the standard SFT-then-RL baseline—an **optimizer bug** (optimizer state not reset between SFT and RL stages) and a **loss aggregation bug** (incorrect reduction over sequence length)—artificially depressed baseline performance in prior comparisons.
- After fixing both bugs: SFT-then-RL outperforms mixed-policy methods by **+3.8 points** on math benchmarks with Qwen2.5-Math-7B and by **+22.2 points** with Llama-3.1-8B.
- A truncated SFT-then-RL variant with only **50 RL steps** beats the best mixed-policy method while using fewer FLOPs.
- The five mixed-policy methods evaluated (including LUFFY, ReLIFT, and SRFT) all fall below the corrected SFT-then-RL baseline.
- The result implies that the complexity of mixed-policy methods has been solving a problem that did not exist in the corrected baseline.

---

## Motivation and Problem

The standard recipe for training LLMs to reason involves two sequential stages:
1. **Supervised Fine-Tuning (SFT):** The model is trained on demonstrations of correct reasoning chains.
2. **Reinforcement Learning (RL):** The model is further refined using reward signals from a verifier.

Starting in 2025, a class of "mixed-policy" methods emerged that interleave SFT and RL updates within the same training stage, or combine on-policy and off-policy data in complex ways. These methods reported substantial improvements over the sequential SFT-then-RL baseline, and several papers presented mixed-policy training as a necessary advancement.

The authors noticed that the SFT-then-RL baselines in these comparisons were inconsistently implemented. In particular, most published mixed-policy papers used the same baseline implementation—which, the authors discovered, contained two bugs.

---

## The Two Bugs

### Bug 1: Optimizer State Continuity

In the SFT-then-RL pipeline, the transition from SFT to RL involves switching from a cross-entropy loss to a policy gradient loss. Standard practice is to reset the optimizer (e.g., AdamW) state at this transition, so that the momentum and variance estimates accumulated during SFT do not pollute the initial RL steps.

The buggy baseline failed to reset the optimizer state. As a result, the RL phase began with stale optimizer statistics calibrated for SFT-style gradients, causing the initial RL steps to be destabilized. This degraded performance for the first several hundred RL steps—precisely the range evaluated in most benchmark comparisons.

### Bug 2: Loss Aggregation

The second bug was in the reduction of the per-token policy gradient loss over variable-length sequences. The buggy implementation divided the total loss by the number of tokens in a minibatch (batch-level mean), rather than by the number of tokens per sequence (sequence-level mean). For sequences of varying length, this introduces a bias: shorter sequences contribute more signal per token than longer ones, skewing the gradient in favor of brevity.

The correct implementation uses sequence-level mean reduction, which weights all sequences equally regardless of length. The batch-level bug disproportionately penalizes longer reasoning chains—precisely the kind of output RLVR is intended to encourage.

---

## Technical Details

The fixed SFT-then-RL implementation makes two changes:

1. **Optimizer reset:** At the start of the RL stage, the optimizer state (first and second moment estimates in AdamW) is zeroed. The learning rate schedule restarts from its initial value.

2. **Loss normalization:** The policy gradient loss is normalized by the number of tokens in each sequence (not the batch), then averaged across sequences in the minibatch:

$$\mathcal{L}_{\text{RL}}(\theta) = \frac{1}{|B|} \sum_{i \in B} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \hat{A}_{i,t} \log \pi_\theta(y_{i,t} \mid x_i, y_{i,<t})$$

where $B$ is the minibatch, $y_i$ is the response for example $i$, $|y_i|$ is its length, and $\hat{A}_{i,t}$ is the token-level advantage estimate.

The authors also evaluate a truncated variant that runs only 50 RL steps (vs. the standard 1,000+). This truncated variant serves as a compute-efficiency baseline.

---

## Experimental Findings

Evaluation on OpenR1-Math benchmark, training on Qwen2.5-Math-7B and Llama-3.1-8B.

**Qwen2.5-Math-7B:**

| Method | Math score | FLOPs (rel.) |
|---|---|---|
| LUFFY | 61.4 | 1.6× |
| ReLIFT | 62.1 | 1.4× |
| SRFT | 62.8 | 1.3× |
| SFT-then-RL (buggy) | 59.2 | 1.0× |
| SFT-then-RL (fixed) | **66.6** | 1.0× |
| SFT-then-RL-50 (fixed) | 65.1 | 0.3× |

**Llama-3.1-8B:**

| Method | Math score | FLOPs (rel.) |
|---|---|---|
| Best mixed-policy | 48.3 | 1.4× |
| SFT-then-RL (buggy) | 46.1 | 1.0× |
| SFT-then-RL (fixed) | **70.5** | 1.0× |

The 22.2-point gap on Llama-3.1-8B reflects the greater sensitivity of that model to the optimizer bug.

---

## Significance

This paper is a methodological cautionary tale for the RL-for-reasoning field. A substantial body of recent work has built on the assumption that mixed-policy methods are superior to sequential SFT-then-RL, and many papers have designed increasingly complex interleaving schemes to close a gap that may not exist. The bugs identified are not exotic: stale optimizer state and incorrect loss aggregation are both plausible implementation oversights that could affect many codebases.

Beyond invalidating specific method comparisons, the paper raises a broader point: the complexity of mixed-policy methods may have been justified by benchmark numbers that reflected implementation bugs rather than genuine algorithmic limitations of the simpler pipeline. The 50-step truncated variant—which matches or beats complex methods at a third of the compute—suggests that the marginal value of extended RL training, properly implemented, is itself high.
