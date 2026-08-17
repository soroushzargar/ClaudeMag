# Answer First, Reason Later: Commitment Order in Diffusion LLMs

**arXiv:** 2608.05687
**Authors:** Jewon Yeom, Jaewon Sok, Seonghyeon Park, Jeongjae Park, Hwiyeong Lee, Taesup Kim
**Venue:** arXiv preprint
**Date:** August 6, 2026
**URL:** https://arxiv.org/abs/2608.05687

---

## Summary

Masked diffusion language models (dLLMs) can commit tokens in any order — a freedom marketed as their core advantage over autoregressive decoding. This paper demonstrates that on reasoning tasks this freedom is instead the axis of failure. By logging every token commitment during decoding of LLaDA-8B on GSM8K, the authors show that unconstrained pure decoding commits the final answer at only 15–24% of the trajectory while half the reasoning region is still masked, collapsing to answer-only outputs on up to 90% of problems as the generation canvas grows. The fix is a constrained decoding strategy that enforces reasoning-before-answer order, recovering substantial accuracy.

## Problem

Masked diffusion LLMs (dLLMs) like LLaDA generate text by starting from a fully masked canvas and iteratively unmasking tokens in any order during a multi-step diffusion process. The literature claims this order-free generation is an advantage over autoregressive models, which must generate left-to-right. But for tasks that require multi-step reasoning — where intermediate steps should logically precede the final answer — order-free generation may actually destroy the structure that reasoning depends on.

## Key Finding: Premature Answer Commitment

Using commitment order logging on LLaDA-8B evaluated on GSM8K:
- **Unconstrained (pure) decoding** commits the final answer token(s) very early: at 15–24% of the decoding trajectory, while half the reasoning region is still masked.
- As the generation canvas grows (longer outputs), the model increasingly collapses to **answer-only outputs** — up to 90% of problems produce no reasoning steps at all.
- The model is, in effect, answering first and reasoning later (or never), despite the reasoning region being physically present in the template.

## Proposed Fix: Constrained Commitment Ordering

The paper introduces a constrained decoding approach that enforces a soft temporal ordering: reasoning region tokens should be committed before answer region tokens. This is implemented by masking the final answer region during the early phase of decoding, releasing it only once sufficient reasoning tokens have been committed. Key design choices:
- The constraint is **soft** (not absolute): the transition from reasoning to answer commitment is gradual.
- The method requires **no retraining** — it operates at inference time only.
- The paper tests several threshold strategies for when to release the answer region.

## Experimental Setup

- **Model:** LLaDA-8B
- **Benchmark:** GSM8K (grade school math word problems)
- **Analysis method:** Logging the position in the decoding trajectory at which each output token is first committed from masked → unmasked
- **Baselines:** Pure (unconstrained) diffusion decoding, left-to-right semi-autoregressive decoding

## Results

- Constrained commitment ordering recovers **substantial accuracy** on GSM8K versus unconstrained decoding
- The accuracy gap between constrained and unconstrained decoding widens as canvas size increases — confirming that the failure mode is scale-sensitive
- The paper provides commitment trajectory visualizations showing the difference between pure decoding (answer commits first) and constrained decoding (reasoning commits first)

## Why It Matters

The paper reframes a fundamental assumption in the dLLM literature. The claim that order-free generation is universally beneficial ignores the logical structure of reasoning tasks, which require answers to be supported by reasoning steps that precede them. If diffusion LLMs are to be used for reasoning, the decoding procedure must be designed to respect this structure — either through constrained decoding (as proposed here) or through training objectives that reward correct commitment order. This work opens an important question for the field: what other structured generation tasks suffer from the same failure mode?

## Key Contributions

1. First systematic analysis of token commitment order in dLLM decoding on reasoning tasks
2. Discovery that pure decoding collapses to answer-first (or answer-only) behavior at scale
3. Inference-time constrained decoding that enforces reasoning-before-answer commitment
4. Empirical demonstration that commitment order, not model capacity, is the bottleneck for dLLM reasoning
