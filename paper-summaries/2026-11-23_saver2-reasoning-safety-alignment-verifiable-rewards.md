# SAVeR²: Reasoning-based Safety Alignment for Large Reasoning Models via Verifiable Rewards

**Authors:** (NeurIPS 2026 poster; full author list at proceedings)
**arXiv:** 2611.07482 (cs.LG, cs.AI)
**Submitted:** 5 November 2026
**Venue:** NeurIPS 2026 (poster)

---

## Headline

Decomposing the safety objective into four fine-grained verifiable reward signals, combined with difficulty-aware data selection and reasoning-preserving gradient projection, produces safety-aligned reasoning models that maintain general reasoning ability from 1.5B to 671B parameters.

---

## Key Findings

- **Four verifiable reward components** replace a single coarse safety reward, allowing more faithful training signal for large reasoning models (LRMs).
- **Difficulty-aware data selection** stabilizes training by ensuring the safety training distribution matches the model's current capability level.
- **Reasoning-preserving gradient projection** explicitly resolves gradient conflicts between the safety objective and the reasoning objective, preventing the "safety tax" observed in prior methods.
- Results on 1.5B–671B-parameter models show improved safety rates, lower over-refusal rates, and preserved general reasoning compared to prior safety alignment methods.
- The method scales: larger models derive more benefit, suggesting that reasoning capability and safety alignment are fundamentally compatible when the training signal is designed correctly.

---

## Motivation and Problem

Large reasoning models (LRMs) — models trained with chain-of-thought reinforcement learning to solve multi-step problems — present a new safety alignment challenge. Standard safety alignment methods were designed for instruction-following models, not for models that reason explicitly about their actions before executing them.

Three failure modes motivate the paper:

1. **Gradient conflict**: Safety RL gradients and reasoning RL gradients often point in different directions. Standard multi-objective training averages them, degrading both.
2. **Coarse reward**: A single binary "safe/unsafe" reward provides weak training signal and can be gamed: a model that refuses all ambiguous queries scores well on safety but becomes useless.
3. **Training instability**: Safety training data tends to cluster at difficulty extremes (too easy or too hard), causing unstable RL updates.

The key insight is that *reasoning* and *safety* are not in fundamental tension: a reasoning model that understands *why* a request is unsafe can refuse it appropriately without over-refusing. The challenge is building the training signal that teaches this.

---

## Method: SAVeR²

### Four Verifiable Reward Signals

SAVeR² decomposes safety into four measurable dimensions, each with a verifiable reward:

1. **Threat identification**: Does the reasoning chain correctly identify the harmful element of the prompt? Measured by an independent classifier on the scratchpad.
2. **Refusal correctness**: Is the refusal appropriately targeted (addresses the specific harm, not just the surface topic)? Measured by overlap with harm-specific criteria.
3. **Non-over-refusal**: Does the model respond helpfully to benign prompts that superficially resemble unsafe ones? Measured against a benign-query holdout.
4. **Reasoning integrity**: Is the reasoning chain internally consistent and free from post-hoc rationalization? Measured by a verifier that checks logical coherence of the scratchpad.

### Difficulty-Aware Data Selection

Training data is filtered dynamically to match the model's current capability. At each training step, samples are selected such that the model's current success rate on them lies in a target range $[p_{\min}, p_{\max}]$ (e.g., 0.4–0.7). Easy samples (success rate $> p_{\max}$) are suppressed; hard samples (success rate $< p_{\min}$) are excluded until later training stages when the model is more capable.

### Reasoning-Preserving Gradient Projection

Let $g_S$ and $g_R$ denote the safety and reasoning gradient vectors respectively. Standard multi-task training computes $g_S + \lambda g_R$, which can point in a direction that degrades both objectives when the angle between them exceeds 90°.

SAVeR² instead uses a projection:

$$\tilde{g}_S = g_S - \frac{g_S \cdot g_R}{\|g_R\|^2} g_R$$

This removes the component of the safety gradient that conflicts with the reasoning gradient, so safety updates do not degrade reasoning capability. The reasoning gradient is left untouched.

---

## Technical Formulation

The combined SAVeR² objective is:

$$\mathcal{J}(\theta) = \mathcal{J}_R(\theta) + \lambda \sum_{i=1}^{4} w_i \mathcal{J}_{S_i}(\theta)$$

where $\mathcal{J}_R$ is the reasoning objective (e.g., standard GRPO or PPO with math verifiers), $\mathcal{J}_{S_i}$ are the four verifiable safety objectives with per-dimension weights $w_i$, and $\lambda$ controls the safety-reasoning tradeoff. The gradient projection is applied at the parameter update step:

$$\theta_{t+1} = \theta_t + \alpha \left(\tilde{g}_{S} + g_R\right)$$

where $\alpha$ is the learning rate and $\tilde{g}_S$ is the projected safety gradient.

---

## Experimental Findings

Experiments are conducted on models ranging from Qwen3-1.5B-Instruct to DeepSeek-R2-671B, evaluated on:
- **Safety**: Refusal rate on a held-out harmful-query test set
- **Over-refusal**: Helpfulness on benign queries similar to harmful ones
- **Reasoning**: MATH-500, GPQA Diamond, and AIME 2026

| Method | Safety Rate | Over-Refusal | MATH-500 |
|---|---|---|---|
| SFT safety baseline | 81.3% | 18.2% | 72.1% |
| RLHF preference baseline | 87.4% | 12.1% | 69.8% |
| SAVeR² (ours) | 93.7% | 6.4% | 74.3% |

SAVeR² achieves the highest safety rate while reducing over-refusal by 47% relative to RLHF and maintaining or improving reasoning benchmark scores.

---

## Significance

SAVeR² demonstrates that the apparent tension between safety and reasoning in LRMs is an artifact of imprecise training signal design, not a fundamental trade-off. By verifying safety at the reasoning-chain level (not just the final output), the method teaches models to reason *about* safety rather than to pattern-match safe outputs — a qualitative improvement in alignment quality. The gradient projection technique is broadly applicable and could benefit other multi-objective RL training scenarios.
