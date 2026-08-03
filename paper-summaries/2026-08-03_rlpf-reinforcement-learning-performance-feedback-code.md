# RLPF: Reinforcement Learning from Performance Feedback for Code Generation

**Authors:** Huihao Jing, Haozhe Cui, Wenbin Hu, Shaojin Chen, Haochen Shi, Changxuan Fan, Yuxuan Liu, Hanyu Yang, Sirui Zhang, Ziyi Chen, Haoran Li, Yangqiu Song  
**Affiliation:** Hong Kong University of Science and Technology  
**Venue:** arXiv:2607.27271  
**Date:** July 29, 2026  
**URL:** https://arxiv.org/abs/2607.27271

---

## Summary

The standard paradigm for training code LLMs with execution feedback rewards programs that pass unit tests and penalizes those that fail. This binary correctness signal has driven large gains in competitive programming benchmarks, but it leaves an important gap: two programs can both pass the same test suite while differing by orders of magnitude in runtime. RLPF introduces a staged reward scheme that first ensures correctness, then rewards runtime efficiency relative to a reference implementation. Training on this signal raises a weak model from 11.1% to 54.6% correct-and-runnable rate and from 8.1% to 38.6% CGRE (Correct-and-Graded Relative Efficiency).

## Problem

Runtime efficiency is an important property of code for systems software, competitive programming, and latency-sensitive applications. Yet training code models to prefer efficient implementations faces three challenges:

1. **Reward fragility.** Runtime is only a meaningful signal after a program compiles and passes tests. If most sampled programs fail to compile or run, a pure efficiency reward provides no gradient at all.
2. **Task variability.** Absolute runtimes are not comparable across tasks—an $O(n^2)$ solution may be fast for $n=10$ but catastrophically slow for $n=10^4$. A relative metric (speedup over a reference) is needed.
3. **Mode collapse.** Optimizing only for efficiency can cause a model to produce a fixed fast template that passes narrow test cases but fails the full test suite.

## Method: Staged Reward

RLPF uses PPO (Proximal Policy Optimization) with a two-stage reward:

**Stage 1 — Correctness gate:** Each sampled program $\hat{c}$ is executed against the test suite. If it fails to compile, fails a test, or times out, it receives a correctness reward $r_{\text{correct}} = 0$ and no performance component is computed.

**Stage 2 — Performance reward:** If $\hat{c}$ passes all tests, its runtime $t(\hat{c})$ is measured across multiple inputs drawn from the task's distribution. The performance reward is:

$$r_{\text{perf}}(\hat{c}) = \frac{t_{\text{ref}} - t(\hat{c})}{t_{\text{ref}}}$$

where $t_{\text{ref}}$ is the runtime of a reference correct solution (typically the canonical solution from the problem set). This is a relative speedup that is bounded and comparable across tasks.

The combined reward is:

$$r(\hat{c}) = \alpha \cdot r_{\text{correct}} + (1 - \alpha) \cdot r_{\text{perf}}(\hat{c}) \cdot \mathbf{1}[r_{\text{correct}} = 1]$$

where $\alpha$ is annealed from 1.0 to 0.3 over training, shifting emphasis from correctness to efficiency as the model improves.

## Technical Formulation

The policy model $\pi_\theta$ generates code $\hat{c} \sim \pi_\theta(\cdot | p)$ for a programming problem $p$. The PPO objective maximizes:

$$\mathcal{J}(\theta) = \mathbb{E}_{\hat{c} \sim \pi_\theta(\cdot|p)}\!\left[\min\!\left(\rho_t A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where $\rho_t = \pi_\theta(\hat{c}|p) / \pi_{\text{old}}(\hat{c}|p)$ is the importance ratio, $A_t$ is the advantage estimated from $r(\hat{c})$, and $D_{\text{KL}}$ is a KL divergence penalty against the initial model $\pi_{\text{ref}}$ that prevents mode collapse.

The CGRE metric (Correct-and-Graded Relative Efficiency) is defined as:

$$\text{CGRE} = \frac{1}{|P|} \sum_{p \in P} \mathbf{1}[\hat{c}_p \text{ passes}] \cdot \max\!\left(0,\; 1 - \frac{t(\hat{c}_p)}{t_{\text{ref}}(p)}\right)$$

This is zero for failing programs and proportional to speedup for passing ones.

## Learning Procedure

The model is trained on a curated benchmark of competitive programming problems spanning array processing, sorting, graph algorithms, and dynamic programming, covering a range of optimal time complexities. Reference solutions are taken from accepted competitive programming submissions. Training proceeds for 3,000 PPO steps with batch size 32, sampling 8 rollouts per problem per step. The $\alpha$ annealing schedule follows a cosine curve.

## What the Guarantee Says

The authors prove that the RLPF objective has the correct-first property: in expectation, increasing $\alpha$ toward 1 during early training ensures the policy first learns to produce any correct program before being penalized for runtime. This prevents the policy from finding degenerate solutions (e.g., programs that always return a hardcoded answer to pass one test) that would have incorrect but fast behavior.

## Experimental Findings

**Main results (RLPF vs. baselines on the test split):**

| Model | Correct-and-Runnable | CGRE |
|---|---|---|
| Base model (no RL) | 11.1% | 8.1% |
| RLPF-Correct only | 47.3% | 18.4% |
| RLPF-Full (staged) | **54.6%** | **38.6%** |
| GPT-5.4 (zero-shot) | 58.2% | 41.1% |

The staged reward achieves 7.3 percentage points higher CGRE than correctness-only RL, confirming that the performance signal provides substantial additional signal beyond mere correctness training.

**Generalization across domains:** Performance gains are consistent across all problem categories, with the largest relative improvement on array/DP problems (44.2% CGRE) and smaller gains on graph algorithms (28.1% CGRE), consistent with graph problems being harder to optimize algorithmically without domain knowledge.

## Ablations and Interpretation

**Reward annealing:** Fixing $\alpha = 0.5$ throughout training (no annealing) reduces CGRE by 8.2 points, as the performance signal disrupts early correctness learning.

**Reference solution quality:** Using a suboptimal reference (10th percentile runtime vs. best) reduces CGRE by 4.1 points, confirming that reference quality matters.

**KL penalty:** Removing $D_{\text{KL}}$ causes mode collapse onto 3–5 high-efficiency templates, reducing CGRE diversity and failing on problems that don't match those templates.
