# Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models

**Authors:** (NVIDIA Research; full author list at arXiv)
**arXiv:** 2512.13607 (cs.CL, cs.LG)
**Submitted:** 15 December 2025
**Venue:** NeurIPS 2026 (oral)

---

## Headline

Sequential domain-by-domain RL training—Cascade RL—scales to general-purpose reasoning across math, code, and software engineering without cross-domain interference, because later domain stages rarely degrade earlier gains and each domain can be tuned independently.

---

## Key Findings

- **Cascade RL** trains domains in sequence (SFT → RLHF → Instruction Following RL → Math RL → Code RL → Software Engineering RL) rather than mixing all domain data into a single RL run.
- Later domain-specific RL stages **rarely degrade performance** attained in earlier domains—the sequential structure preserves prior gains while allowing per-domain optimization.
- Cross-domain data mixing introduces engineering complications: heterogeneous **verification latency** (math verification is fast; software engineering verification requires running code) and wide variation in **response length** distributions make a single RL run difficult to stabilize.
- Per-domain hyperparameter tuning (learning rate, KL penalty, reward scaling) consistently improves results compared to shared hyperparameters across domains.
- The Cascade RL framework achieves strong results on math and coding benchmarks and extends naturally to software engineering tasks (SWE-bench class).

---

## Motivation and Problem

Training a general-purpose reasoning model—one that excels at math, code generation, software engineering, and instruction following—requires exposure to reinforcement learning signals from all of these domains. The obvious approach is to mix problem types into a single RL training run, with rewards coming from domain-specific verifiers.

This mixing approach has three practical problems:

1. **Verification latency heterogeneity:** Math problems can be verified in milliseconds (symbolic equality check). Software engineering tasks require spinning up a sandboxed environment, running tests, and collecting results—a process that takes seconds to minutes. When math and SWE examples are mixed, the training loop is bottlenecked by the slowest verifier, inflating wall-clock time for the fast domains.

2. **Response length heterogeneity:** Math solutions tend to be 100–500 tokens; complete software engineering patches can be thousands of tokens. Mixed batches produce highly variable sequence lengths, increasing GPU memory fragmentation and complicating gradient normalization.

3. **Domain conflict:** Some domains may require conflicting behaviors. Concise symbolic reasoning for math and verbose, context-rich code patching for SWE can compete for the same parameter updates during joint training.

Cascade RL addresses all three by running domains sequentially.

---

## Method

### Pipeline Structure

The Cascade RL pipeline proceeds through six stages:

1. **SFT:** Supervised fine-tuning on a broad instruction dataset establishes output format.
2. **RLHF:** Human preference learning aligns the model's general behavior.
3. **Instruction Following RL:** A lightweight RL stage on instruction-following tasks (format compliance, instruction adherence) using rule-based rewards.
4. **Math RL:** Extended RL on mathematical reasoning problems with symbolic verifiers.
5. **Code RL:** RL on code generation tasks with execution-based verifiers.
6. **Software Engineering RL:** RL on repository-level SWE tasks with test-suite verifiers.

Each stage uses only the verifiers and data distributions relevant to its domain. The model checkpoint from stage $k$ initializes stage $k+1$.

### Why Cascade Preserves Prior Gains

The paper provides both empirical and theoretical arguments for why later stages do not catastrophically forget earlier ones. The key observation is that domain-specific RL stages update primarily the parameters most responsive to domain-specific rewards, leaving parameters encoding prior-domain capabilities largely unchanged.

This is reminiscent of continual learning results showing that well-structured sequential training can be more stable than joint training when task distributions are sufficiently different.

---

## Technical Details

Each RL stage uses proximal policy optimization (PPO) with domain-specific reward models:

- **Math RL:** Reward $r = 1$ for a correct final answer, $r = 0$ otherwise (verified by symbolic equality).
- **Code RL:** Reward proportional to the fraction of test cases passing (execution-based).
- **SWE RL:** Reward proportional to the fraction of repository-level tests passing after applying the generated patch.
- **Instruction Following RL:** Rule-based reward for format compliance and coverage of specified instructions.

The KL penalty is set independently for each stage based on a brief hyperparameter sweep conducted at the start of that stage. The penalty weight is typically higher for early stages (to avoid drifting far from the SFT checkpoint) and lower for later stages (where the checkpoint is already well-calibrated for earlier domains).

---

## Experimental Findings

The Nemotron-Cascade model family spans sizes from 7B to 70B parameters.

| Benchmark | Cascade RL | Mixed RL | SFT-only |
|---|---|---|---|
| MATH-500 | **84.2** | 81.7 | 72.3 |
| HumanEval+ | **78.6** | 76.4 | 65.1 |
| SWE-bench Verified | **41.3** | 37.8 | 24.6 |
| IFEval (strict) | **83.1** | 80.9 | 78.4 |

Results are for the 70B variant. Cascade RL outperforms the mixed RL baseline on every benchmark. The SWE-bench result is particularly notable: the sequential structure allows the late SWE stage to build on a foundation of strong math and code capability without interference.

---

## Ablations

- **Stage order:** Reversing the order (SWE → Code → Math → ...) degrades Math-500 by 3.1 points, suggesting that building foundation reasoning skills first is important.
- **Shared vs. per-domain hyperparameters:** Using shared hyperparameters across all stages degrades average benchmark performance by 1.8 points.
- **KL penalty ablation:** Removing the KL penalty from the Math RL stage produces stronger math results (+1.2 points) but degrades instruction following by 4.3 points, confirming the need for domain-specific tuning.
- **Continual RL forgetting:** After all six stages, re-evaluating Math-500 shows only a 0.7-point drop from the Math RL checkpoint—confirming that later stages cause minimal regression on earlier domain benchmarks.

---

## Significance

Nemotron-Cascade provides a practical engineering blueprint for training general-purpose reasoning models. Its central contribution is not an algorithmic novelty but an architectural principle: sequential domain-wise RL is more tractable, more stable, and more efficient than cross-domain joint RL for heterogeneous verification environments. For practitioners building production reasoning systems, the cascade structure offers concrete guidance on how to extend a math-capable base model to code and software engineering without sacrificing existing capabilities.
