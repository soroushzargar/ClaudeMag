# Strategist: Designing Agentic Reasoning at Scale

**Authors:** Taiming Lu, Qi Xu, Zhigang Hua, Zhuang Liu  
**arXiv:** 2611.03618  
**Categories:** cs.AI, cs.CL  
**Venue:** NeurIPS 2026 Poster  
**Submitted:** November 2, 2026  
**URL:** https://arxiv.org/abs/2611.03618

---

## Headline

Strategist is a meta-agent that automatically designs executable inference strategies for new tasks by iteratively proposing, evaluating, and refining strategy code, building a reusable library that achieves 8.5pp higher accuracy at 48% lower inference cost than the best fixed strategy across 30 benchmarks.

## Key Findings

- Strategist outperforms the strongest fixed-strategy baseline on 24 of 30 benchmarks spanning math, coding, science, and commonsense reasoning.
- Average accuracy improvement: 8.5 absolute percentage points above the strongest baseline, at 48% lower inference cost.
- Strategies discovered by Strategist transfer to model backends the meta-agent never optimized on (e.g., strategies discovered with GPT-6 transfer to Gemini-2.5).
- The strategy library grows compositionally: later tasks benefit from blueprints discovered for earlier tasks, exhibiting positive transfer across task types.

## Problem

Multi-agent reasoning systems typically rely on fixed templates: chain-of-thought, tree-of-thought, or hand-crafted debate protocols. These templates are designed by human researchers for a specific set of tasks and rarely generalize well across domains. Automated methods for prompt engineering optimize over discrete prompt tokens but do not change the structural topology of the reasoning process. The question Strategist addresses is: can a meta-agent discover fundamentally different reasoning structures that outperform human-designed templates, without requiring manual design work?

## Approach: Strategy as Code

Strategist represents each inference strategy as a short Python program (a "blueprint") that orchestrates one or more LLM calls. A blueprint specifies:
- How the problem is decomposed into subproblems.
- How multiple agents (possibly different models) interact.
- How intermediate outputs are verified and routed.
- How final answers are aggregated.

This code-as-strategy representation makes strategies fully executable, inspectable, and composable. Strategies can call helper functions from other blueprints in the library.

## The Design Loop

Given a new task description T and a small development set D (10–20 labeled examples):

1. **Retrieve:** Query the library for the k most similar existing blueprints using a semantic embedding of T.
2. **Propose:** A GPT-6 meta-agent proposes 3–5 candidate strategy modifications or new strategies, in Python, guided by the retrieved examples.
3. **Evaluate:** Each candidate is executed on D; accuracy and total token cost are measured.
4. **Revise:** The meta-agent reviews results and proposes a final revised strategy.
5. **Store:** All candidates and the final strategy are added to the library with accuracy, cost, and task-type metadata.

The design loop runs for 3 rounds per task, with early stopping if the best strategy exceeds a target accuracy threshold.

## Strategy Library

The library acts as a long-term memory across tasks. It is initialized with 20 hand-designed baseline strategies (chain-of-thought, majority voting, debate, etc.) and grows as Strategist processes tasks. The compositional reuse mechanism allows later tasks to invoke sub-routines discovered for earlier tasks, enabling strategies that combine e.g. "verify arithmetic sub-steps" (from math tasks) with "use multiple perspectives" (from debate strategies).

## Experimental Setup

- 30 benchmarks: MATH, GSM8K, HumanEval, MBPP, MMLU (5 subsets), ARC, HellaSwag, GPQA, MedQA, and 14 others.
- Baselines: single chain-of-thought, majority vote @8, tree-of-thought, automatic prompt engineer, DSPy, and the oracle (best single fixed strategy per task).
- Backend: GPT-6 for the meta-agent and primary reasoning.

## Results

- **24/30 benchmarks:** Strategist exceeds the best fixed-strategy baseline.
- **8.5pp accuracy:** Average absolute improvement across all 30 benchmarks.
- **48% cost reduction:** Despite higher quality, the designed strategies are cheaper because they avoid unnecessary calls and use cheaper models for sub-tasks where they suffice.
- **Transfer:** When strategies discovered with GPT-6 are applied with Gemini-2.5 as the backbone, the accuracy improvement over the Gemini-2.5 chain-of-thought baseline is 6.1pp, confirming cross-model transferability.

## Significance

Strategist demonstrates that the topology of multi-agent reasoning — not just the prompts or the model — is a critical design variable, and that it can be optimized automatically. The code-as-strategy representation makes this tractable by giving the meta-agent a structured, executable representation to work with. The growing library suggests that progress on reasoning problems can compound: each new task solved contributes reusable components to future tasks.
