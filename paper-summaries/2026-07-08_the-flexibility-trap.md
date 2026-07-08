# The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models

**Authors:** Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua, Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, Gao Huang  
**Affiliation:** Tsinghua University et al.  
**Venue:** ICML 2026 Outstanding Paper  
**arXiv:** 2601.15165  
**URL:** https://arxiv.org/abs/2601.15165

---

## Headline Finding

Arbitrary-order token generation — the defining feature of diffusion language models — paradoxically **narrows** reasoning ability. Constraining dLLMs to left-to-right ordering during RL exploration, while preserving bidirectional attention at inference, yields 89.1% on GSM8K vs. 82.3% for the best prior diffusion RL method.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
JustGRPO, a minimalist training framework that forces left-to-right GRPO during the RL exploration phase only, achieves:
- **89.1%** on GSM8K (vs. ESPO 82.3%, GDPO 82.8%)
- **45.1%** on MATH-500 (vs. ESPO 39.0%, GDPO 39.6%)
- Fully preserves parallel decoding at inference time

### Level 2 — Core Mechanism
Diffusion language models (dLLMs) can generate tokens in arbitrary order. In theory, this strictly supersets the left-to-right trajectory, implying better coverage of the solution space. In practice, the paper shows dLLMs exploit order flexibility to **skip high-uncertainty tokens** — those most important for exploring diverse solution paths — causing premature collapse of the solution distribution.

### Level 3 — Methodology
The authors diagnose the collapse via a coverage metric tracking how many distinct solution prefixes remain reachable after each decoding step. They then introduce JustGRPO, which:
1. Forces a standard left-to-right trajectory *only during RL rollout generation*
2. Applies GRPO reward weighting over binary outcome signals (correct/incorrect)
3. Leaves the model's bidirectional attention and parallel decoding fully intact for inference

### Level 4 — Technical Details
The GRPO objective over a group of G rollouts is:
$$\mathcal{J}(\theta) = \mathbb{E}\left[\sum_{t=1}^{T}\min\left(\hat{r}_t(\theta)\hat{A}_t,\ \text{clip}(\hat{r}_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

where $\hat{A}_t$ is the normalized group advantage and $\beta$ controls KL divergence from the reference policy. JustGRPO differs from prior diffusion RL methods only in constraining the rollout direction during training.

---

## Why Existing Approaches Fall Short

Prior diffusion RL methods (ESPO, GDPO) accept the arbitrary-order design as a given and build reward optimization around it. The authors' coverage analysis shows that unconstrained dLLMs in RL settings converge to a degenerate strategy: generate high-confidence "easy" tokens first, leaving uncertain tokens for last when the model's committed trajectory limits their effect. This is structurally analogous to a search algorithm that avoids exploring its most uncertain branches.

---

## Experimental Findings

| Method | GSM8K | MATH-500 |
|--------|-------|----------|
| ESPO   | 82.3% | 39.0%    |
| GDPO   | 82.8% | 39.6%    |
| **JustGRPO** | **89.1%** | **45.1%** |

Ablations confirm that constraining order *only during rollout* is sufficient; applying the constraint at inference does not further improve results, demonstrating the constraint is a training signal, not an architectural change.

---

## Significance

This paper challenges a foundational assumption of the dLLM research program: that arbitrary-order generation is unambiguously beneficial. It demonstrates that for reasoning tasks, this feature is actively harmful during RL training and should be suppressed. The fix is startlingly simple, suggesting many dLLM RL results to date may have been bottlenecked by this unexplored failure mode.
