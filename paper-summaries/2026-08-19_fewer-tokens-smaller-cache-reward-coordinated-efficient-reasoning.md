# Fewer Tokens, Smaller Cache: Reward-Coordinated Efficient Reasoning

**arXiv:** 2608.04771  
**Date:** August 5, 2026  
**Authors:** Qiyuan Zhu, Dezhi Li, Pengyu Cheng, Tianle Chen, Jiacheng Wang, Ruijie Shen, Hao Gu, Sida Lin, Zirui Liu, Jiacheng Liu, Sirui Han  
**URL:** https://arxiv.org/abs/2608.04771

---

## Summary

Large Reasoning Models (LRMs) produce long chain-of-thought (CoT) reasoning traces to solve complex tasks. While effective, these traces cause two intertwined costs: (1) a large KV-cache from the accumulated context, and (2) a large number of generated tokens. Existing compression methods address only the cache side and apply uniform policies across the trajectory.

This paper identifies two overlooked interactions:

**Interaction 1 — Process reward tracks compression tolerance.** A reasoning state's sensitivity to context loss varies across the trajectory. High-reward states — where the model has committed to a correct sub-conclusion — can tolerate more aggressive cache eviction without losing accuracy. Low-reward states, where the model is still exploring, are brittle to context loss. Using a process reward model (PRM) to guide eviction targets high-tolerance windows while preserving critical reasoning context.

**Interaction 2 — Cache compression induces token inflation.** Compressing the KV-cache reduces the model's access to its own reasoning history. Without sufficient context, the model regenerates or elaborates more, producing *more* tokens than it would with a full cache. This creates a cancellation effect: cache savings are partially offset by token cost increases.

**RCER (Reward-Coordinated Efficient Reasoning)** jointly minimizes both quantities by coordinating cache eviction and generation length under the same process reward signal. The PRM guides where to compress (cache side) and when to stop generating (token side), tying both decisions to an estimate of the current reasoning state's quality.

## Method

RCER operates in two coordinated loops:

1. **Cache eviction:** At each decoding step, a PRM scores the current reasoning state. High-scoring states trigger aggressive eviction of past tokens from the KV-cache. Low-scoring states preserve context to avoid compounding errors.

2. **Generation stopping:** The PRM also signals when the reasoning trace has converged to a high-confidence sub-conclusion. At such points, the model can commit earlier rather than generating additional tokens to reach the same conclusion.

Both loops share the same PRM without requiring any additional model or separate training.

## Key Results

- Simultaneous reduction in both KV-cache size and generated tokens versus baselines that target only one quantity
- Accuracy preserved at compression rates that defeat uniform eviction baselines
- Process reward guidance is particularly effective at high compression ratios, where uniform approaches fail catastrophically

## Significance

The paper reframes efficient reasoning as a two-sided problem — not just a caching problem — and shows that process reward models, originally trained to improve reasoning quality, can be repurposed as compression guides. This opens a path to efficient LRM deployment that does not require separate compression models or distillation.
