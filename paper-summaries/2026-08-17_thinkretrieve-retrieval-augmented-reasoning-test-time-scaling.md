# ThinkRetrieve: Retrieval-Augmented Reasoning Traces for Test-Time Scaling

**arXiv:** 2608.10928
**Authors:** Vaibhav Singh, Soumya Suvra Ghosal, Sarvesh Gharat, Soumyabrata Pal, Ramasuri Narayanam, Dinesh Manocha
**Venue:** arXiv preprint
**Date:** August 2026
**URL:** https://arxiv.org/abs/2608.10928
**Affiliations:** IIT Bombay; University of Maryland College Park; Adobe Research

---

## Summary

Sequential test-time scaling of Large Reasoning Models (LRMs) frequently yields diminishing or negative returns because longer reasoning traces accumulate uncertainty, compound errors, and drift from the original problem. ThinkRetrieve is a test-time framework that augments LRM reasoning traces with dynamically retrieved, step-by-step solved examples at each intermediate reasoning step. Rather than injecting retrieved facts, it injects guidance on *how* to reason, precisely when the model needs it. Experiments across five reasoning models (1.5B–8B parameters) on GSM-8K, MATH-500, AIME 2025, and SciQ show consistent accuracy improvements over standard test-time scaling approaches.

## Problem

Large Reasoning Models improve performance by generating extended chain-of-thought (CoT) reasoning traces at inference time — a paradigm known as test-time compute scaling. Empirically, however, simply allocating more tokens to reasoning does not always help:
- **Error compounding:** A mistake early in a long trace propagates and amplifies through subsequent steps.
- **Problem drift:** As the trace grows, the model's attention to the original problem statement weakens.
- **Uncertainty growth:** Longer traces have more chances to generate low-confidence (but high-fluency) intermediate steps.

Prior work on retrieval-augmented generation (RAG) addresses *factual* deficits but does not address *reasoning process* deficits. When the model knows the facts but reasons incorrectly, injecting retrieved documents is unhelpful.

## Key Idea: Retrieval into the Thinking Trace

ThinkRetrieve retrieves not documents, but **solved reasoning traces** — problems paired with step-by-step solutions from an external corpus. The retrieval happens *inside* the reasoning trace, at each intermediate step:

1. At reasoning step $k$, extract the current partial trace and the problem statement.
2. Retrieve the top-$m$ most relevant solved examples from the corpus based on embedding similarity to the partial trace.
3. Inject the retrieved exemplars directly into the thinking trace as in-context guidance.
4. Continue reasoning with the augmented trace.

This is distinct from standard RAG (retrieves documents before generation) and from in-context learning (retrieves examples once before the trace). ThinkRetrieve is *dynamic*: retrieval happens repeatedly, conditioned on the evolving partial reasoning trace.

## Architecture

- **Retrieval index:** A corpus of (problem, step-by-step solution) pairs, encoded with a dense retriever.
- **Query encoder:** At step $k$, the query is formed from the concatenation of the original problem and the partial reasoning trace up to step $k$.
- **Injection format:** Retrieved exemplars are formatted as additional in-context examples preceding the point at which the model continues reasoning.
- **No fine-tuning:** The framework is applied at inference time only — the base LRM is frozen.

## Experimental Setup

- **Models:** Five LRMs ranging from 1.5B to 8B parameters
- **Benchmarks:**
  - **GSM-8K:** Grade school math word problems
  - **MATH-500:** Competition math problems (subset of MATH)
  - **AIME 2025:** American Invitational Mathematics Examination problems
  - **SciQ:** Science question answering
- **Baselines:** Standard sequential test-time scaling (extending the reasoning trace), standard RAG (document retrieval before generation), best-of-N sampling

## Results

ThinkRetrieve consistently improves accuracy over standard test-time scaling across all five models and all four benchmarks. The gains are largest on harder benchmarks (AIME 2025, MATH-500), where longer reasoning traces are most prone to drift and error compounding. The method is particularly effective for smaller models (1.5B–3B), where the reasoning capability per token is weakest and external guidance provides the largest marginal benefit.

## Why It Matters

Test-time scaling has been a dominant paradigm for improving LRM accuracy without training. ThinkRetrieve shows that *how* the model uses inference-time compute matters as much as *how much* it uses. Injecting solved analogues mid-reasoning is a qualitatively different form of guidance than extending the reasoning trace: it provides structural scaffolding rather than raw compute. The result has practical implications for deployed reasoning systems with access to solution databases (e.g., tutoring systems, mathematical assistants).

## Key Contributions

1. Identification of error compounding and problem drift as root causes of test-time scaling failure
2. Dynamic, step-wise retrieval of solved examples injected into the reasoning trace mid-generation
3. Consistent accuracy gains across five models (1.5B–8B) and four benchmarks without any fine-tuning
4. Ablation analysis isolating the contribution of retrieval timing (early, mid, late in trace) and exemplar quality
