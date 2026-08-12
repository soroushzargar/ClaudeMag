# Learning When to Trust via Selective Context Preference Optimization

**arXiv:** 2608.06377
**Authors:** Xian Sun, Wei Chow, Yingshuo Wang, Junhao Liu, Wei Gao, Qing Wu, Lingdong Kong
**Venue:** arXiv preprint
**Date:** August 6, 2026
**URL:** https://arxiv.org/abs/2608.06377

---

## Summary

Language models increasingly condition their answers on external context, but a single misleading signal can flip a correct answer to wrong (sycophancy), while training models to resist such signals can make them ignore genuinely helpful context. This paper introduces MIST — a human-annotated benchmark for measuring context sensitivity — and SCOPE, a DPO-based training method that teaches models when to trust and when to resist external context. SCOPE significantly reduces susceptibility to misleading context while preserving the model's ability to benefit from helpful signals.

## Problem

As RAG systems, web-search-augmented assistants, and chat-based agents become ubiquitous, LLMs routinely encounter external context that may be helpful, irrelevant, or actively misleading. The key tension is:
- A model that always follows context is sycophantic: misleading signals flip correct answers wrong
- A model that always ignores context is useless: it fails to benefit when external information would help

Prior work on robustness to misleading context often optimized only on adversarial examples, inadvertently training models to ignore context universally. MIST and SCOPE are designed to solve this discrimination task: trust helpful context, resist misleading context.

## MIST Benchmark

MIST (Misleading and Informative Signal Test) is a human-annotated benchmark with a key design feature: **each reasoning item is rendered under four matched conditions**:
1. **Clean:** question alone, no external context
2. **Misleading context:** context that nudges toward the wrong answer
3. **Correct context:** context that confirms or supports the right answer
4. **Irrelevant context:** context unrelated to the question

This 4-way structure allows precise measurement. The paper introduces **SC2W** (Susceptibility to Context-induced Wrong answers): the fraction of items where a misleading signal flips a clean-correct answer to wrong.

## SCOPE Method

SCOPE (Selective Context Preference Optimization) trains models to distinguish helpful from harmful context via Direct Preference Optimization (DPO):

1. **Mining:** From the MIST benchmark, identify clean-correct / misleading-wrong failures — items the model answers correctly without context but incorrectly with misleading context
2. **Pair construction:** Construct preference pairs that reward the model for maintaining correct answers under misleading context
3. **Balanced optimization:** Optimize DPO over pairs balanced equally across all four MIST conditions, not only misleading ones, to prevent the model from learning to ignore context universally

## Results

- SCOPE significantly reduces SC2W (susceptibility to misleading context flips)
- Correct-context utilization is preserved — models continue to benefit from genuinely helpful signals
- The balanced 4-condition training is crucial: optimizing only on misleading pairs degrades performance on correct-context items

## Why It Matters

Sycophancy is one of the most practically costly failure modes in deployed LLM systems. Users and automated pipelines constantly present models with context that may be wrong (outdated retrieval, user misbeliefs, adversarial inputs). A model that can reliably distinguish helpful from misleading context is more trustworthy and requires less defensive engineering around it. MIST provides a rigorous evaluation framework; SCOPE provides a training recipe that is compatible with standard alignment pipelines.

## Key Contributions

1. MIST benchmark: four matched conditions per item for precise measurement of context sensitivity
2. SC2W metric: precise measurement of misleading-context susceptibility
3. SCOPE training method: DPO-based training with balanced condition sampling
4. Demonstration that balanced training is necessary to avoid trading sycophancy for context blindness
