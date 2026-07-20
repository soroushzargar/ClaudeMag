# Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning

**arXiv:** 2607.12395  
**Date:** July 14, 2026  
**Authors:** Xinyu Tang, Qianggang Cao, Yurou Liu, Yuliang Zhan, Xiaochong Lan, Yifan Li, Yuchen Yan, Han Peng, Zican Dong, Zhenduo Zhang, et al.  
**Affiliation:** Ant Group InclusionAI  
**URL:** https://arxiv.org/abs/2607.12395

---

## Headline

Ant Group's InclusionAI scales zero reinforcement learning — RL with verifiable rewards and no human labels — to one trillion parameters for the first time, discovering five emergent reasoning behaviors and achieving 84.2% on AIME 2026 with first-stage training alone.

---

## Key Findings

- Zero RL at trillion-parameter scale spontaneously produces five emergent behaviors absent at smaller scales: self-verification, context anxiety, anthropomorphism, structured formatting, and parallel reasoning.
- Training reliably unfolds in two sequential phases: a discovery phase where dormant reasoning pathways are unlocked, followed by a sharpening phase where policy is refined within those boundaries.
- Ring-2.5-1T-Zero achieves 84.2% accuracy on AIME 2026 using only first-stage RL training across seven mathematical reasoning benchmarks including AIME 2024, AIME 2025, HMMT, and IMOAnswerBench.
- Clipped importance sampling and training-inference ratio correction are essential for stable trillion-parameter RL — naive scaling collapses without these stabilization measures.
- Context anxiety — the model treating itself as a resource-constrained agent with a finite context window — emerges only at trillion-parameter scale and directly improves efficiency.

---

## Methodology

Ring-Zero applies zero RL to Ring-2.5-1T, a hybrid linear-attention model with one trillion parameters, using a training pipeline that addresses three failure modes of naive large-scale RL:

1. **Poor readability** at large scale is addressed via length normalization and structured formatting rewards.
2. **Token redundancy** is reduced by clipped importance sampling that prevents excessively divergent off-policy updates.
3. **Lack of adaptive reasoning depth** is corrected by training-inference ratio correction, which accounts for the mismatch between the context window used during RL rollouts and the context seen at inference.

The model receives problems with verifiable answers — math problems, logic puzzles, code tasks — and a binary reward signal: +1 for correct, 0 for incorrect. No intermediate step annotations are used.

---

## Technical Details

- **Base model:** Ring-2.5-1T with hybrid linear-attention architecture, 1T total parameters, 1M token context window.
- **Clipped importance sampling:** Limits the off-policy correction ratio to $[1-\epsilon, 1+\epsilon]$ at trillion-parameter scale where rollout distributions drift faster.
- **Training-inference ratio correction:** A scaling factor adjusts for the difference between rollout context length and inference context length: $\hat{r} = r \cdot (C_{\text{inf}} / C_{\text{train}})^\gamma$, where $\gamma$ is a hyperparameter.
- **Mixed-precision control:** BF16 for activations, FP32 for gradient accumulation, with selective FP8 quantization for embedding layers to fit within GPU memory.
- **Emergent behavior characterization:** The five behaviors are identified through systematic activation probing and output pattern analysis across checkpoints.

---

## Experimental Findings

- AIME 2026: 84.2% (Ring-2.5-1T-Zero, first-stage only)
- AIME 2025: 81.7%
- AIME 2024: 83.4%
- HMMT: 78.3%
- IMOAnswerBench: 61.2%
- Seven-benchmark average: 74.8%

The discovery phase (first 30% of training steps) is characterized by rapid exploration: the model attempts diverse response formats and activates reasoning modules that were dormant in the pretrained checkpoint. The sharpening phase (remaining 70%) stabilizes format and deepens solution quality within the style established in discovery.

---

## Ablations and Interpretation

- Removing clipped importance sampling destabilizes training after approximately 500 steps at 1T scale, with loss divergence not seen at 70B or below.
- Removing training-inference ratio correction reduces AIME 2026 accuracy from 84.2% to 79.1% due to systematic underestimation of long-context performance.
- Context anxiety — strategic decisions based on remaining context budget — first appears at approximately 800B parameters and strengthens at 1T, suggesting it is a phase transition rather than a smooth scaling behavior.
- Self-verification (the model checking its own intermediate conclusions before committing to a final answer) improves accuracy by an estimated 2–3% at equivalent parameter count, based on ablations that suppress self-verification tokens.

---

## Reference

Xinyu Tang et al. **Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning.** arXiv:2607.12395, July 2026. https://arxiv.org/abs/2607.12395
