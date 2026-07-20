# TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents

**arXiv:** 2607.13988  
**Date:** July 15, 2026  
**Authors:** Leitian Tao, Baolin Peng, Wenlin Yao, Tao Ge, Hao Cheng, Mike Hang Wang, Jianfeng Gao, Sharon Li  
**Affiliation:** Microsoft Research, University of Wisconsin-Madison  
**URL:** https://arxiv.org/abs/2607.13988

---

## Headline

TRACE solves the credit assignment problem for multi-turn tool-use agents by deriving dense turn-level reward signals from Temporal-Difference changes in log-ratio state values computed by a frozen reference model — without any human annotation of intermediate steps.

---

## Key Findings

- Outcome-only rewards become inadequate as agent trajectories extend to tens or hundreds of tool calls; TRACE provides dense intermediate signals by decomposing outcomes into per-turn credit.
- State values are estimated as log-ratios of gold-answer log-probabilities under the current policy versus a frozen reference model, enabling calibrated credit without a separately trained value model.
- TRACE rewards are derived as Temporal-Difference changes across consecutive state values, providing a signal that is both local (rewarding individual turns) and globally consistent (summing to the outcome reward).
- On multi-turn tool-use benchmarks, agents trained with TRACE improve significantly over outcome-only RL baselines, demonstrating that the intermediate signals are informative rather than noisy.
- The approach requires no human annotation of intermediate steps, scaling to complex agentic tasks where intermediate supervision is prohibitively expensive.

---

## Methodology

TRACE frames a multi-turn agent trajectory as a sequence of states $s_0, s_1, \ldots, s_T$ where each state transition corresponds to one tool call and its result. The agent policy $\pi_\theta$ generates a sequence of tool calls $a_1, \ldots, a_T$ before producing a final answer.

1. **State representation:** Each state $s_t$ is the full trajectory prefix up to turn $t$, including the task, all prior tool calls, and tool outputs.
2. **State value estimation:** The value of state $s_t$ is estimated as:
   $$V(s_t) = \log \frac{p_\theta(y^* \mid s_t)}{p_{\text{ref}}(y^* \mid s_t)}$$
   where $y^*$ is the gold answer and $p_{\text{ref}}$ is a frozen reference model.
3. **Turn-level reward:** The reward at turn $t$ is the TD change in state value:
   $$r_t = V(s_t) - V(s_{t-1})$$
4. **Policy optimization:** Rewards $r_t$ are used in a standard PPO or REINFORCE update, treating each turn as a step in the MDP.

---

## Technical Details

- **Reference model:** The frozen reference is the initial policy checkpoint before RL training; it is never updated during TRACE training.
- **Log-probability estimation:** $p_\theta(y^* \mid s_t)$ is computed by conditioning the policy on the trajectory prefix and scoring the gold answer via teacher-forcing.
- **Normalization:** State values are normalized across trajectories in each batch to maintain training stability.
- **Integration with PPO:** The TRACE reward $r_t$ is added to any extrinsic reward (e.g., a binary outcome signal); the value function head is trained alongside the policy.
- **Trajectory sampling:** Trajectories are collected by rolling out the current policy in a real or simulated tool-use environment; gold answers must be known for computing $V(s_t)$.

---

## Experimental Findings

- On ToolBench (multi-step tool-use): TRACE improves task completion rate from 41.3% (outcome-only RL) to 58.7%.
- On WebArena (web navigation with tool calls): TRACE improves task success rate from 19.4% to 28.1% over outcome-only baselines.
- On a long-horizon code repair benchmark with 20+ edit steps: TRACE achieves 47.2% vs. 31.6% for the outcome-only baseline.
- Credit assignment quality: TRACE turn-level rewards correlate with human annotations of action quality at $r = 0.61$ (Pearson), compared to $r = 0.22$ for Monte-Carlo rollout-based credit.
- Ablation — replacing gold log-probabilities with a learned value head: comparable performance but requires additional training and is sensitive to value head initialization.

---

## Ablations and Interpretation

- **Without TD structure (using $V(s_T) / T$ uniformly):** Uniform credit assignment degrades performance by 8–12% across benchmarks, confirming that the per-turn temporal structure of the TD signal is informative.
- **Reference model choice:** Using a weaker (smaller) reference model degrades credit quality; the reference should be a checkpoint of the same model before RL.
- **Trajectory length:** TRACE benefits grow with trajectory length — for tasks with fewer than 5 tool calls, the improvement over outcome-only RL is negligible, but for 20+ tool call tasks the gap is 15+ percentage points.
- **Negative turns:** TRACE correctly assigns negative rewards to turns that move the agent away from the goal (increasing distance from correct answer in value space), which outcome-only RL cannot detect until the trajectory ends.

---

## Reference

Leitian Tao et al. **TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents.** arXiv:2607.13988, July 2026. https://arxiv.org/abs/2607.13988
