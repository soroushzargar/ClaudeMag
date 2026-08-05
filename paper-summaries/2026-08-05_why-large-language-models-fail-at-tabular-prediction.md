# Why Large Language Models Fail at Tabular Prediction

**arXiv:** 2608.02412  
**Authors:** Marta Garnelo, Wojciech M. Czarnecki (Google DeepMind)  
**Date:** August 3, 2026  
**URL:** https://arxiv.org/abs/2608.02412

---

## Summary

Large language models routinely lose to classical baselines (random forests, gradient-boosted trees, even k-NN) on tabular prediction tasks. This failure has spawned an entire subfield of tabular foundation models, yet the fundamental question—why do generic LLMs fail—remained open. Garnelo and Czarnecki provide the first systematic answer: dimensionality, and only dimensionality, is the decisive factor.

## Experimental Setup

The study evaluates a frontier LLM in its purest inference regime: a single generation pass over a prompt that contains the full training set and a test point, with no tools, no fine-tuning, and no agentic scaffolding. Thirty-one standard tabular benchmark datasets are used, with random linear projections applied to sweep over a controlled range of input dimensionalities.

Nine comparison methods are included: the LLM alongside eight classical machine learning methods (k-NN, various random forests, gradient boosting, linear classifiers, etc.).

## Five Hypotheses Tested

The paper evaluates five candidate explanations for LLM failure on tabular data:

**(a) Inability to handle noisy or non-linearly-separable data** — Falsified. The LLM handles noisy versions of the data at similar accuracy levels to classical methods.

**(b) The linearised CSV format obscuring column structure** — Falsified. Alternative serialization formats do not improve LLM performance.

**(c) Tokenisation of numeric values** — Falsified. Replacing numbers with symbolic tokens does not restore performance.

**(d) Number of test points classified per query** — Falsified. Varying query batch size has no systematic effect.

**(e) Input dimensionality** — Confirmed as decisive. LLM accuracy decreases monotonically as dimensionality grows. Every classical baseline is flat or improves.

## Key Finding: The Dimensional Curse Is Unique to LLMs

When input dimensionality is low (e.g., 2 dimensions), the LLM's predictions correlate with those of a local, distance-based method (up to **91.6% grid agreement**). This suggests the LLM is performing something resembling nearest-neighbor lookup in low dimensions.

As dimensionality grows, no classical learner—not even noise-corrupted or degraded variants—reproduces the LLM's specific pattern of degradation. The mechanism by which the LLM fails at high dimension is qualitatively unlike any classical failure mode, leaving the true prediction mechanism as an open question.

## Why It Matters

The finding has two important implications:

1. **For practitioners:** LLMs should not be used for tabular prediction when feature spaces are moderate-to-high dimensional, regardless of prompt engineering, serialization format, or model size. Classical methods will dominate.

2. **For researchers:** The dimensionality sensitivity suggests that LLM in-context tabular prediction is mechanistically different from what classical methods do. Understanding this mechanism—why does a language model behave like a local distance-based estimator in 2D but something entirely different in 10D?—is an important open problem for mechanistic interpretability.

## Key Contributions

1. Largest systematic study of why LLMs fail at tabular prediction
2. Controlled falsification of four popular hypotheses
3. Identification of input dimensionality as the sole decisive factor
4. Behavioral comparison showing 91.6% agreement with local distance methods in 2D
5. Evidence that LLM failure mode is novel—not explainable by any classical degradation pattern
