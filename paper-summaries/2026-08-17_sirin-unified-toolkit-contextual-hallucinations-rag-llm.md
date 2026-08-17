# SIRIN: A Unified Toolkit for Detecting Contextual Hallucinations in Retrieval-Augmented and Memory-Grounded LLM Systems

**arXiv:** 2608.00033
**Authors:** Julia Belikova, Rauf Parchiev, Mikhail Filimonov, Konstantin Polev, Andrey Savchenko, Maksim Makarenko
**Venue:** arXiv preprint
**Date:** August 2026
**URL:** https://arxiv.org/abs/2608.00033

---

## Summary

Contextual hallucinations — fluent, plausible LLM responses that contradict or are unsupported by the retrieved evidence — are a critical reliability failure in RAG systems. SIRIN (Semantic Inconsistency Recognition and Inspection Nexus) is a unified toolkit and interactive web UI consolidating three detection paradigms: representation probing (white-box access to model internals), uncertainty estimation, and judge-style verification (black-box). It also handles the complementary task of pre-generation query answerability — determining whether retrieved context contains an answer to the query before generation begins. SIRIN supports response- and span-level inspection in both white-box and black-box settings, providing a shared interface, configuration system, and evaluation pipeline for systematic hallucination detection research.

## Problem

As LLMs are deployed in retrieval-augmented generation (RAG), agentic, and memory-grounded pipelines, a new failure mode dominates: **contextual hallucination**. Unlike parametric hallucinations (where the model confabulates facts from training), contextual hallucinations occur when the model generates a fluent, plausible response that contradicts or is not supported by the provided retrieved context. This is particularly insidious because:
- The response may be factually correct in the general world but incorrect given the specific retrieved evidence
- Standard downstream metrics (fluency, coherence) do not penalize contextual hallucinations
- Detection requires comparing model output to retrieved context — a structured grounding task

The field lacks a unified framework for comparing hallucination detection methods: existing methods use different interfaces, datasets, evaluation protocols, and assumptions about model access (white-box vs. black-box).

## Three Detection Paradigms Unified

SIRIN integrates three fundamentally different approaches to contextual hallucination detection:

### 1. Representation Probing (White-Box)
Train a lightweight probe on the model's internal activations to predict whether a generated span is hallucinated. Requires access to model internals (hidden states, attention weights). Advantages: fast, does not require additional LLM calls. Limitations: model-specific, requires labeled training data for the probe.

### 2. Uncertainty Estimation (White-Box / Grey-Box)
Detect hallucinations by measuring token-level or span-level uncertainty in the model's predictions. High uncertainty on factual claims correlates with hallucination risk. Methods: semantic entropy, predictive entropy, contrastive decoding. SIRIN supports multiple uncertainty metrics under a common interface.

### 3. Judge-Style Verification (Black-Box)
Use a second LLM (the "judge") to verify whether each claim in the response is supported by the retrieved context. Requires only API access to the original model and the judge. Most model-agnostic but most expensive.

## Pre-Generation Answerability

Before generation, SIRIN can assess whether the retrieved context contains a sufficiently complete and relevant answer to the query. This **query answerability** task enables early filtering: if the context does not contain a reliable answer, the system can abstain rather than risk hallucination.

## Architecture

- **Unified configuration system:** Single YAML config selects detector type, model access level, granularity (response vs. span)
- **Evaluation pipeline:** Standard benchmarks for contextual hallucination with consistent metrics
- **Interactive web UI:** Analysts can inspect predictions, view highlighted hallucinated spans, and compare detectors on the same input
- **Pluggable detectors:** New detection methods can be added by implementing a standardized interface

## Experimental Findings

SIRIN benchmarks all three paradigms across standard RAG hallucination datasets. Key findings:
- No single paradigm dominates: representation probing excels when white-box access is available; judge-style verification is most robust in black-box settings
- Combining uncertainty estimation with judge-style verification improves precision over either alone
- Pre-generation answerability filtering significantly reduces hallucination rate by preventing generation when context is insufficient

## Why It Matters

As RAG systems scale to high-stakes applications (medical Q&A, legal research, financial analysis), the ability to detect when the model's response is not grounded in the retrieved evidence becomes a safety-critical requirement. SIRIN provides the infrastructure for systematic, reproducible comparison of detection methods and lowers the barrier to deploying hallucination detection in production RAG systems.

## Key Contributions

1. Unification of three hallucination detection paradigms (representation probing, uncertainty estimation, judge-style) under a single interface
2. Pre-generation query answerability as a complementary hallucination prevention mechanism
3. Support for response- and span-level inspection in white-box and black-box settings
4. Interactive web UI for analyst-facing hallucination inspection and debugging
5. Standardized evaluation pipeline enabling reproducible comparison across detector methods
