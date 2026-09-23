# LogicTrack: Auditing Reasoning Trajectories of Large Language Models with Formal Logic Solvers

**arXiv:** 2609.21492  
**Authors:** Jingyu Hu, Shu Yang, Weiru Liu, Di Wang  
**Date:** September 2026

---

## One-Line Summary

LogicTrack auto-formalizes each CoT step into an SMT-LIB specification, verifies it with a theorem prover, and computes a step-level Solver-Based Backtracking Reward to guide both inference-time search and fine-tuning.

---

## Problem

Chain-of-thought reasoning improves large language model performance, but existing training methods—including RLVR—optimize for whether the final answer is correct, not whether the intermediate reasoning steps are logically valid. A model can reach correct answers via logically invalid steps, and can reach incorrect answers via steps that are locally valid but globally misguided. Neither outcome-based rewards nor behavioral probes provide step-level logical guarantees.

---

## Why Existing Approaches Fall Short

- **Outcome-based RLVR** (e.g., GRPO, PPO-R1) provides a single binary/scalar signal per trajectory, giving no credit for valid intermediate steps or penalty for invalid ones.
- **Process reward models (PRMs)** provide per-step scores but are trained discriminatively on human labels that are expensive to collect and do not carry logical soundness guarantees.
- **Self-consistency and majority voting** rely on repeated sampling, not on verifying the logical structure of any individual trace.
- **Symbolic solvers used directly** cannot process natural-language text without a formalization layer.

LogicTrack bridges natural language and formal logic with an auto-formalization module, enabling theorem-prover verification as a free training signal.

---

## Core Method

**Step Decomposition.** Each natural-language reasoning step is decomposed into three fields:
1. **Context:** facts established by prior steps
2. **Background assumptions:** domain knowledge or definitions invoked
3. **Conclusion:** the claimed logical consequence

**Auto-formalization.** An LLM-based translator maps each field into SMT-LIB format—a standardized specification language for satisfiability-modulo-theories solvers. Linear arithmetic, equality, and basic propositional logic cover the large majority of mathematical reasoning steps.

**Solver Verification.** A Z3-class SMT solver checks each specification and returns a diagnostic: *syntax error*, *satisfiable*, *entailed* (conclusion follows from context + assumptions), *refuted* (conclusion contradicts them), or *undecided* (solver timeout or incompleteness).

**Solver-Based Backtracking Reward (SBR).** SBR combines:
- A **premise-fidelity score** measuring how faithfully the SMT context matches prior step conclusions
- A **solver verification score** based on the diagnostic: entailed > satisfiable > undecided > refuted

$$\text{SBR}(s_k) = \alpha \cdot \text{Fid}(s_k) + (1-\alpha) \cdot \text{SolverScore}(s_k)$$

where $s_k$ is the $k$-th reasoning step and $\alpha$ balances the two components.

**Inference-Time Backtracking Search.** SBR scores guide a tree search that backtracks to the lowest-scoring step when the current path stalls or reaches a wrong answer, regenerating from that branch. This is analogous to MCTS but grounded in logical validity rather than outcome predictions.

**SFT Data Construction.** Accepted trajectories (those with high SBR and correct final answers) are used to create supervised fine-tuning data with explicit backtracking traces, teaching the model to internalize step-wise logical self-auditing.

---

## Technical Formulation

Let a reasoning trajectory be $\tau = (s_1, s_2, \ldots, s_T, a)$ where $a$ is the final answer. Each step $s_k$ is converted to an SMT-LIB triple $(\text{ctx}_k, \text{bg}_k, \text{conc}_k)$.

The SMT check is:

$$\text{SMT-check}: \text{ctx}_k \wedge \text{bg}_k \models \text{conc}_k$$

The overall trajectory SBR score is:

$$\text{SBR}(\tau) = \frac{1}{T}\sum_{k=1}^{T} \text{SBR}(s_k) \cdot \mathbf{1}[a = a^*]$$

For backtracking search, the search policy selects the backtrack point $k^* = \arg\min_k \text{SBR}(s_k)$ and regenerates $s_{k^*}, \ldots, s_T$.

---

## Learning Procedure

**Inference-time:** 1. Generate initial trajectory. 2. Score each step with SBR. 3. If $\text{SBR}(\tau) < \delta$, backtrack to $k^*$ and regenerate. 4. Repeat until answer correct or budget exhausted.

**Training-time SFT:** 1. Collect successful backtracking trajectories. 2. Format as SFT examples with explicit backtrack markers. 3. Fine-tune the base model on these examples. 4. Evaluate the fine-tuned model on held-out benchmarks.

---

## What the Paper Claims

LogicTrack substantially improves step-level verifiability—the fraction of steps with entailed or satisfiable SMT diagnostics—across all seven tested LLMs and eight benchmarks. Final answer accuracy is generally preserved or improved, particularly on tasks requiring multi-step deductive reasoning (logic puzzles, math word problems). The improvement is larger for models whose baseline accuracy on these benchmarks is below 70%, suggesting the most benefit for models that already make reasoning errors.

---

## Experimental Findings

- Evaluated on 8 benchmarks: GSM8K, MATH, ARC-Challenge, HellaSwag-hard, LogiQA, FOLIO, ProntoQA, and a custom multi-step deduction suite
- Tested on 7 LLMs ranging from 7B to 70B parameters
- Step-level verifiability improves by 15–30 percentage points across model families
- Final-answer accuracy improves on 5 of 8 benchmarks; slight degradation on 1 (within noise)
- SFT on backtracking traces provides lasting improvement without requiring the solver at inference time for fine-tuned models

---

## Ablations and Interpretation

- **SBR vs. outcome-only reward:** Outcome-only reward improves final accuracy but not step-level verifiability; SBR improves both
- **Backtracking vs. repeated sampling:** Backtracking achieves higher accuracy at the same inference budget by directing exploration toward logical faults
- **Auto-formalization quality:** Approximately 85% of steps are successfully formalized; formalization failures are handled gracefully as "undecided"
- **$\alpha$ sensitivity:** Performance is stable for $\alpha \in [0.3, 0.7]$

---

## Reference

Jingyu Hu, Shu Yang, Weiru Liu, Di Wang. **LogicTrack: Auditing Reasoning Trajectories of Large Language Models with Formal Logic Solvers**. arXiv:2609.21492, September 2026.  
https://arxiv.org/abs/2609.21492
