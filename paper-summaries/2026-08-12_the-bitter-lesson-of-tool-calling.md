# The Bitter Lesson of Tool Calling

**arXiv:** 2608.06370
**Authors:** Ishan Patel, Sahil Sen, Elias Lumer, Vamse Kumar Subbiah
**Venue:** arXiv preprint
**Date:** August 6, 2026
**URL:** https://arxiv.org/abs/2608.06370

---

## Summary

This paper conducts the first generation-spanning, systematic empirical comparison of Programmatic Tool Calling (PTC) against native JSON tool calling across 14 language models on BFCL v4. The core finding: PTC — where tools are exposed as typed Python stubs and invoked through code — matches or beats JSON tool calling in 11 of 14 models, with the GPT-5.6 family gaining 10.6% over the JSON baseline. The paper argues that tool calling is an underappreciated design choice, and that the JSON default is quietly costing accuracy at scale.

## Problem

As LLMs become agents that use tools to act in the world, how those tools are presented to the model matters enormously. The dominant paradigm — native JSON tool calling, where the model emits a JSON blob specifying a tool name and arguments — was adopted by convention rather than systematic study. This paper asks whether the format in which tools are exposed affects downstream accuracy, and by how much.

## Key Idea: Programmatic Tool Calling (PTC)

In PTC, tools are presented as typed Python function stubs (like a typed API). The model writes Python code to invoke the tool, including argument values, within a single agentic turn. Execution and result handling occur programmatically, just as a developer would call a library function. This format leverages the model's code training and lets the model reason about tool signatures before calling them.

## Experimental Setup

- **Benchmark:** BFCL v4 (Berkeley Function-Calling Leaderboard v4), with three evaluation tracks:
  - **Standard:** single-turn, straightforward tool use
  - **Parallel fan-out:** multiple tool calls issued in one turn
  - **Context rot:** long, cluttered contexts with many prior turns
- **Models tested:** 14 LLMs across multiple families and generations (including GPT-5.6 family)
- **Metric:** Functional correctness of tool invocations

## Results

| Evaluation Track | PTC win rate |
|---|---|
| Standard | 11/14 models |
| Parallel fan-out | 13/14 models |
| Context rot | PTC holds; JSON drops 2.3% average |

- GPT-5.6 family: +10.6% with PTC over JSON baseline
- Advantage widens under parallel fan-out and context rot, conditions most representative of real agentic workloads

## Why It Matters

Tool calling is infrastructure — it underlies virtually every production LLM agent system. A 10% accuracy gap from a design choice that most practitioners have never questioned represents enormous potential improvement at zero training cost. The paper's findings suggest that the field has settled on JSON tool calling for historical rather than performance reasons, and that switching to PTC is a free lunch for most model families. The analysis of context rot is particularly important: as agents handle longer sessions, JSON performance degrades while PTC remains stable.

## Key Contributions

1. First systematic, multi-model, multi-generation comparison of PTC vs. JSON tool calling
2. Three distinct evaluation tracks: standard, parallel fan-out, context rot
3. Quantification of the PTC advantage across 14 models
4. Evidence that the JSON default is sub-optimal for most current LLM families
