# Agent World Model in the Wild: Computer-Use Agent Environment Synthesis

**arXiv:** 2610.26828  
**Venue:** arXiv preprint, October 2026  
**Authors:** Zhaoyang Wang, Qianhui Wu, Huaxiu Yao, et al.  
**Affiliations:** Microsoft Research, UNC Chapel Hill  
**Date:** October 19, 2026

---

## One-Line Summary

WildAWM synthesizes 341 controllable, persona-grounded computer-use environments with long-horizon tasks and execution-based quality gates; Qwen models trained exclusively in these synthetic worlds generalize to 8 out-of-distribution computer-use benchmarks.

---

## Problem

Generalist computer-use agents must handle arbitrary desktop tasks—filing emails, booking flights, coding in an IDE—across heterogeneous software environments. Training such agents requires large collections of environment-task pairs where the ground truth is known and execution can be verified. Manual curation of these environments is expensive and does not scale; scripted generators produce simplistic tasks that do not resemble real user workflows. The result is a bottleneck: the scarcity of realistic, verifiable training environments limits how capable computer-use agents can become.

---

## Key Observation: Persona-Grounded Generation Produces Realistic, Verifiable Environments

The authors observe that realistic computer-use tasks are grounded in the personal context of a real user: their files, browser history, email account, and work patterns. By generating synthetic user personas and deriving their digital environments from those personas, WildAWM produces task-environment pairs that look and behave like real user desktops, enabling realistic training without any real user data.

---

## Why Existing Approaches Fall Short

- **Hand-crafted benchmarks (OSWorld, ScreenSpot, etc.):** Expensive to build; fixed task distribution; evaluation set cannot double as training data.
- **Rule-based environment generators:** Tasks are formulaic and do not transfer to the diversity of real-world usage.
- **Web-crawled demonstrations:** Difficult to verify correct completion; rewards are noisy; difficult to control task length and difficulty.
- **Simulation replay:** Requires access to the original environment to replay; does not generalize to novel software configurations.

---

## Core Method

**Persona-grounded environment synthesis:**
WildAWM begins by generating synthetic user personas (profession, preferences, daily activities) using an LLM. From each persona, it constructs an executable personal-computer environment: a web browser populated with persona-relevant pages, MCP (Model Context Protocol) services simulating APIs the persona would use, and a local file system with persona-appropriate documents.

**Cross-agent interface (XAI) task synthesis:**
Within each environment, WildAWM generates tasks that require cross-application actions (e.g., "find the meeting mentioned in your last email, add it to your calendar, and create a summary document"). These XAI tasks are structured as long-horizon plans with verifiable sub-goals.

**Execution-based quality gates:**
Task-environment pairs pass through three quality checks:
1. **Heterogeneous agent rollouts:** Multiple agent architectures attempt each task; any task that all agents trivially solve (too easy) or none can verify (too hard to check) is filtered.
2. **Cross-judge consistency:** Multiple judge models evaluate task completion; pairs with inconsistent verdicts are flagged.
3. **Counterfactual robustness probes:** Judges are given deliberately incorrect outcome feedback to detect sycophancy; judges that agree with false outcomes are excluded.

---

## Technical Formulation

Let $e \sim P_{\text{env}}(\cdot \mid u)$ be an environment generated from user persona $u$, and let $\tau \sim P_{\text{task}}(\cdot \mid e)$ be a task grounded in that environment. A task-environment pair $(e, \tau)$ passes the quality gate if:

$$Q(e, \tau) = \mathbf{1}\left[\text{ConsistencyScore}(e,\tau) > \delta_c\right] \cdot \mathbf{1}\left[\text{RobustnessScore}(e, \tau) > \delta_r\right]$$

where $\text{ConsistencyScore}$ measures agreement across judge models and $\text{RobustnessScore}$ measures resistance to sycophantic probes.

The verifier robustness probe exploits LLM sycophancy by presenting the judge with counterfactual feedback:

$$\text{RobustnessScore}(e, \tau) = 1 - \frac{1}{|C|}\sum_{c \in C} \mathbf{1}\left[\text{Judge}(e, \tau, c_{\text{false}}) = \text{correct}\right]$$

where $c_{\text{false}}$ is a deliberately incorrect outcome label and the sum measures how often the judge incorrectly agrees with it.

---

## Learning or Inference Procedure

1. Sample personas; generate environments; synthesize XAI tasks.
2. Apply quality gates; retain pairs that pass all three checks.
3. Train agent models (Qwen at various sizes) on the curated pairs with a behavioral cloning objective, then fine-tune with RL using the verifier as the reward model.
4. Evaluate on held-out out-of-distribution benchmarks: no fine-tuning on any target benchmark.

---

## What the Paper Claims

- **Dataset scale:** 341 long-horizon task-environment pairs requiring up to 100+ agent steps.
- **Task difficulty:** GPT-6-Astra solves only 60% of tasks, confirming the tasks are non-trivial.
- **Generalization:** Qwen models trained exclusively on WildAWM environments achieve substantial gains across 8 OOD benchmarks spanning web, desktop, and computer-use agent tasks.
- **Quality gate effectiveness:** Pairs rejected by the gate are shown (via human evaluation) to have higher noise-to-signal ratio.

---

## Experimental Findings

- **Training environments:** 341 curated task-environment pairs.
- **OOD benchmarks (8):** Web browsing, email management, file management, coding assistants, and general computer-use (exact benchmark names in the paper).
- **Key result:** Qwen models trained on WildAWM environments generalize to all 8 OOD settings; gains are larger on tasks requiring multi-application coordination, the core strength of XAI tasks.
- **Frontier model ceiling:** GPT-6-Astra at 60% success rate establishes the difficulty ceiling; trained Qwen models close the gap substantially on some benchmarks.

---

## Ablations and Interpretation

- **Without persona grounding (random environments):** Task-environment pairs are less realistic; downstream benchmark performance drops.
- **Without quality gates:** Training on unfiltered pairs degrades performance due to noisy verification signals.
- **Without counterfactual probes:** Some judges are sycophantic; reward signal is noisier.
- **Why synthetic environments transfer:** Persona-grounded synthesis produces task distributions closely aligned with real user workflows; the XAI structure requires the same cross-application reasoning as real tasks.

---

## Reference

Zhaoyang Wang, Qianhui Wu, Huaxiu Yao, et al. **Agent World Model in the Wild: Computer-Use Agent Environment Synthesis**. arXiv:2610.26828. October 2026.  
https://arxiv.org/abs/2610.26828
