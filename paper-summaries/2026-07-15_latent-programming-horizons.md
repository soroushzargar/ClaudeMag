# Latent Programming Horizons in Coding Agents

**arXiv:** 2607.05188  
**Date:** July 6, 2026  
**Authors:** André Silva, Han Tu, Martin Monperrus  
**Affiliation:** KTH Royal Institute of Technology, Stockholm, Sweden  
**URL:** https://arxiv.org/abs/2607.05188

---

## Headline

Language models acting as coding agents linearly encode the correctness of the program they are editing in their residual streams — and these representations run up to 25 steps ahead of the agent's own actions, revealing a "latent programming horizon" invisible to external observers.

---

## Key Findings

- Logistic-regression probes on LM hidden states decode whether the current code parses, passes its test suite, reduces failing tests, or introduces regressions, with AUC up to 0.83 for correctness across two models and two benchmarks.
- Probes predict the outcome of future edits up to approximately 25 steps before those edits are materialized and written to disk — termed the agent's **latent programming horizon**.
- Probe representations transfer across benchmarks without retraining, indicating they capture generic code-quality features rather than task-specific artifacts.
- The findings hold across both GPT-4o and Claude 3.7 Sonnet models, suggesting the phenomenon is model-agnostic.

---

## Methodology

The study instruments coding agents (running on SWE-bench and HumanEval-Pro benchmarks) by recording the full residual stream trajectory at each step of the agent's multi-step episode. At each step, the agent's hidden states (the final-layer residual stream vector for the last token) are saved alongside ground-truth labels derived from sandboxed test execution.

Probes are then trained:
1. **Immediate probes:** Predict current code state (parses/passes/regresses) from the hidden state at step $t$.
2. **Future probes:** Predict code state at step $t+k$ from hidden state at step $t$, for $k = 1, 2, \ldots, 40$.

AUC is measured as a function of lookahead $k$ to determine the "horizon" — the maximum $k$ at which the probe achieves AUC above the chance threshold with statistical significance.

---

## Technical Details

- **Probe architecture:** Linear logistic regression on 4096-dimensional residual stream vectors (final transformer layer, last-sequence-position token).
- **Labels:** Binary labels derived from sandboxed pytest execution on the current repository state after each agent edit.
- **Models probed:** GPT-4o (tool-calling API) and Claude 3.7 Sonnet (extended thinking disabled).
- **Benchmarks:** SWE-bench Verified (300 real GitHub issues) and HumanEval-Pro (150 multi-file programming problems).
- **Horizon measurement:** The latent programming horizon is defined as $k^* = \max\{k : \text{AUC}(k) \geq 0.55 \text{ with } p < 0.05\}$.

---

## Experimental Findings

| Task | Model | AUC (k=0) | Horizon (k*) |
|------|-------|-----------|--------------|
| Parses | GPT-4o | 0.81 | 28 steps |
| Passes tests | GPT-4o | 0.78 | 22 steps |
| Reduces failures | Claude 3.7 | 0.83 | 25 steps |
| Introduces regression | Claude 3.7 | 0.71 | 14 steps |

Cross-benchmark transfer (train on SWE-bench, test on HumanEval-Pro) retains AUC above 0.70 for all tasks, confirming the generality of the representations.

---

## Ablations and Interpretation

- **Middle vs. final layers:** Probing from the middle transformer layers (layer 24 of 48) gives AUC 0.63 vs. 0.81 from the final layer, suggesting the relevant planning representations are built up through the full depth of the network.
- **Non-linear probes:** An MLP probe with one hidden layer improves AUC by only 0.02, confirming the representations are approximately linearly decodable.
- **Random baseline:** Shuffled-label probes achieve AUC 0.50 ± 0.01, ruling out statistical artifacts.
- **Implication:** The latent horizon suggests agents "know" the likely outcome of their planned edits before executing them. Future work could use this signal for early stopping, intervention, or lookahead guidance.

---

## Reference

André Silva, Han Tu, Martin Monperrus. **Latent Programming Horizons in Coding Agents.** arXiv:2607.05188, July 2026. KTH Royal Institute of Technology. https://arxiv.org/abs/2607.05188
