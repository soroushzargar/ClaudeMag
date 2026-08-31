# SWE Refactor Bench: Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration?

**arXiv:** 2608.23564  
**Authors:** Deyao Hong, Yizhe Chi, Wenyi Li, Xiaoqiu Wang, Mingju Gao, Kaisen Yang, Bingxiang He, Youjie Zheng, Calvin Xiao, Qinhuai Na  
**Affiliations:** Navers Lab; Einsia.AI; Tsinghua University  
**Submitted:** August 24, 2026  
**Area:** Coding Agents, Software Engineering, Benchmarking

---

## Summary

SWE Refactor Bench is a benchmark of 20 whole-repository migration tasks spanning four categories of technical debt, evaluated by a novel three-stage protocol that verifies migration completion before measuring behavioural correctness. Across 520 runs with 8 frontier models, only 5.4% of runs pass all three stages, and the best model (claude-opus-5) achieves 47.0/100.

## Problem

Coding agent benchmarks such as SWE-bench measure whether an agent can fix a specific bug or resolve a GitHub issue, but they cannot evaluate long-horizon software migration tasks — replacing an entire technology stack across a repository. This matters because accumulating technical debt (deprecated dependencies, obsolete build tools, legacy languages) is a major practical cost in software engineering, and migration is largely manual.

An additional challenge is **evaluation gaming**: because existing benchmarks score only behavioural correctness (test pass rate), agents can "pass" by simply copying the original implementation with minimal change. A true migration benchmark must verify that the target stack actually replaced the old one.

## Benchmark Design

**20 migration tasks** drawn from real open-source repositories, covering four technical debt categories:

| Category | Examples | # Tasks |
|---|---|---|
| Dependency upgrade | Python 2 → 3 packaging, old API clients | 5 |
| Build toolchain rewrite | Make → Bazel, setuptools → Poetry | 5 |
| Testing framework migration | unittest → pytest, nose → pytest | 5 |
| Language rewrite | Python → TypeScript, PHP → Python | 5 |

Each task provides the agent with the repository at a historical version, a migration specification describing the target stack, and documentation for the target technologies.

## Three-Stage Evaluation Protocol

The protocol enforces a hard gate at each stage:

1. **Migration Audit (MA):** Static analysis verifies that references to the old stack (imports, config files, build manifests) are absent and references to the target stack are present. Agents that simply copy the original implementation fail here.

2. **Behavioural Test Suite (BTS):** A fixed, pre-migration test suite is run against the migrated repository. Tests verify that the migrated code produces the same observable behaviour as the original.

3. **Integration Test (IT):** End-to-end workflow tests verify that the full tool-chain (build, lint, package, deploy) functions correctly with the new stack.

Only runs that pass all three stages are counted as successful.

## Results

Evaluated 8 frontier models: claude-opus-5, GPT-5, Gemini-Ultra-2, and several smaller variants, with 26 model-effort configurations (varying scaffolding, context window, and inference budget):

- **Overall success rate:** 28 of 520 runs (5.4%) pass all three stages
- **13 of 20 tasks** receive no accepted solution from any model
- **Best model:** claude-opus-5 at 47.0/100 composite score
- **Category breakdown:** Build toolchain rewrites 31.4, dependency upgrades 22.1, testing framework migrations 18.6, language rewrites 5.6
- **Common failure modes:** Agents complete the Migration Audit but break behavioural tests (42% of runs), or pass behavioural tests by re-implementing the original logic without the target stack (blocked by Migration Audit, 21% of runs)

## Key Contributions

1. The first benchmark requiring both migration completeness and behavioural correctness in a hard-gated sequential evaluation
2. Identification of the **migration gaming** failure mode (preserving original code to pass tests) and a concrete countermeasure (Migration Audit)
3. Evidence of sharp capability disparities across migration categories, suggesting that language rewrites remain far beyond current coding agent abilities
4. A construction methodology for extending the benchmark to new technology stacks

## Significance

SWE Refactor Bench establishes a more demanding and realistic ceiling for coding agent evaluation. As agents improve on issue-resolution benchmarks, SWE Refactor Bench provides a next-level challenge that reflects real-world software maintenance needs. The results indicate that even frontier models require substantial advances before autonomous long-horizon migration is feasible.
