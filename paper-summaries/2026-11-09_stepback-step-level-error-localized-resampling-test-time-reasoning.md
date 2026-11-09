# StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning

**arXiv:** 2611.08403  
**Authors:** Yang Ouyang, Jung-Eun Kim (North Carolina State University)  
**Venue:** NeurIPS 2026  
**Submitted:** 5 November 2026  
**URL:** https://arxiv.org/abs/2611.08403

---

## Summary

Self-consistency (SC) is the dominant strategy for improving the accuracy of chain-of-thought reasoning: generate many full trajectories, then take a majority vote over final answers. The problem is cost — every trajectory runs to completion, even when the error occurs near the beginning. StepBack introduces step-level error localization to avoid this waste. A small number of draft trajectories are generated, each is split into atomic reasoning steps, and lightweight signals are used to identify the most likely first-error position. Only the erroneous step and its continuation are resampled, and majority voting is applied over the resulting answers.

---

## Method

**Error localization signals.** Two complementary signals are combined:
1. *Negative log-likelihood (NLL):* Steps with NLL above a threshold τ_NLL = 0.1 are flagged as uncertain.
2. *Process reward model (PRM):* Steps with PRM score below τ_PRM = 0.05 are flagged as likely incorrect.

The first step flagged by either signal is taken as the resampling anchor. If no step is flagged, the trajectory is accepted as-is.

**Resampling procedure.** Starting from the anchor step, k = 2 fresh continuations are sampled. The original prefix up to (but not including) the anchor is kept, so only the error and its downstream propagation are replaced. Each new continuation is completed to the final answer.

**Voting.** Final answers from all resampled trajectories (16 drafts × 2 samples per anchor = up to 32 candidates, but fewer for unanchored trajectories) are pooled and the plurality answer is returned.

**Compatibility.** StepBack is designed to compose with Adaptive Consistency, which dynamically allocates the total sampling budget based on answer entropy. When paired with Adaptive Consistency, unambiguous early stops are detected before the resampling phase, further reducing cost on easy inputs.

---

## Experimental Setup

Evaluated on seven benchmarks spanning:
- **Mathematical reasoning:** GSM8K, MATH, AIME
- **Multi-step arithmetic:** MultiArith
- **Multi-hop question answering:** HotpotQA, 2WikiMultiHopQA, MuSiQue

Backbone: Llama-3.1-8B-Instruct. Baselines: standard SC at equal budget, Adaptive Consistency.

---

## Results

| Method | Avg. Accuracy | Token Reduction vs. SC |
|---|---|---|
| Self-Consistency (SC) | ~baseline | — |
| StepBack (ours) | ≈ SC | up to 41.26% |
| StepBack + Adaptive Consistency | ≈ SC | > 41.26% |

Across all seven benchmarks, StepBack matches SC's average accuracy while using up to 41.26% fewer tokens. The savings are largest on benchmarks where errors tend to occur early (MATH, 2WikiMultiHopQA), where unnecessary later steps are most common.

---

## Key Insight

The core observation is that reasoning errors are not uniformly distributed across steps — they tend to cluster in the early-to-middle portion of a chain. By localizing the error rather than re-running the full chain, StepBack achieves the same diversity of continuations from the error point onward at a fraction of the token cost. This is strictly more efficient than SC for any trajectory where the localized error is not at the final step.

---

## Limitations

- The NLL and PRM thresholds are tuned on a validation set; their transferability to new domains or model families is not characterized.
- The approach requires a process reward model, which may not be available for all domains.
- Gains are smaller on benchmarks where errors occur uniformly across steps.
