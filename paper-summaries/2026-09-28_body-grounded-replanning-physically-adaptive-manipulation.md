# Body-Grounded Replanning for Physically Adaptive Manipulation

**arXiv:** 2609.30024  
**Authors:** Namiko Saito, Hiroshi Kera  
**Affiliation:** Microsoft Research Asia (Tokyo); Waseda University; Chiba University; National Institute of Informatics  
**Date:** September 2026

---

## One-Line Summary

Body-state events trigger LLM-based high-level strategy replanning during robot manipulation, maintaining task success while reducing physical effort under joint load and mobility constraints.

---

## Problem

Robotic manipulation planning reasons about the external environment—obstacle geometry, object poses, and grasp feasibility—but typically treats the robot's own body as a fixed, ideal actuator. In practice, the robot's physical condition changes during execution: joint loads increase under sustained contact forces, mobility becomes asymmetric after a partial failure or hardware wear, and some strategies that were geometrically feasible at plan time become physically unsuitable during execution. Current planners have no mechanism to recognize when the body's physical state has made the originally planned strategy inappropriate, and cannot select among alternatives based on internal physical signals.

---

## Why Existing Approaches Fall Short

- **Standard task-and-motion planning (TAMP)** reasons about geometry and kinematics but not about dynamic physical state such as instantaneous joint torque or accumulated fatigue.
- **Adaptive controllers** adjust low-level joint torques but do not change high-level strategy (e.g., switching from overhead to underhand grasp, or from direct to indirect reach).
- **LLM-based robot planners** condition on task descriptions and environment observations but ignore joint-level physical signals.
- **Fault-tolerant planning** handles discrete hardware failures but not continuous physical state degradation.

---

## Core Method

**Body-State Event Detection.** Physical state is monitored continuously during execution. A body-state event is triggered when a threshold is exceeded:
- Joint load: torque $\tau_j > \tau_\text{thresh}$ for joint $j$
- Mobility constraint: workspace reachability $R(\theta) < R_\text{thresh}$ due to joint limits or asymmetric constraints

**LLM Strategy Selector.** On event trigger, the LLM receives:
1. **Joint-level state:** current torques, positions, and a natural-language summary of the load distribution.
2. **Execution statistics:** success rate and average completion time of recent strategy attempts.
3. **Execution history:** which strategies have been tried and with what outcomes.

The LLM selects a new high-level strategy from a predefined library (e.g., "approach from left," "use extended reach," "reduce payload"). The task objective and low-level controller are unchanged.

**Decoupled Architecture.** The framework separates:
- High-level strategy selection (body-grounded, LLM-driven, replanning)
- Low-level execution (task-objective controller, unchanged)

This ensures that physical adaptation does not require retraining or modifying the low-level policy.

---

## Technical Formulation

Let $s_t = (\tau_t, q_t, v_t)$ be the physical state at time $t$ (torques, joint angles, velocities). Let $\mathcal{S}$ be the strategy library and $\sigma \in \mathcal{S}$ the current strategy.

Event condition:
$$e_t = \mathbf{1}\!\left[\max_j \tau_{j,t} > \tau_\text{thresh} \;\vee\; R(q_t) < R_\text{thresh}\right]$$

On $e_t = 1$, the LLM is queried:
$$\sigma_{t+1} = \text{LLM}\!\left(\mathcal{P}(s_{t-T:t},\, h_{0:t},\, \sigma_\text{hist})\right)$$

where $\mathcal{P}$ is the prompt encoding physical state over a window $T$, execution history $h$, and prior strategies $\sigma_\text{hist}$.

Objective: maximize task success rate $\Pr[\text{success}]$ subject to physical effort $E = \sum_t \|\tau_t\|_2^2$ below a budget.

---

## Learning or Inference Procedure

No model training is performed in this framework. The procedure at execution time is:

1. Execute the current strategy $\sigma$ using the low-level controller.
2. Monitor $s_t$ at every control step.
3. If $e_t = 1$, construct the prompt $\mathcal{P}$ and query the LLM.
4. Update $\sigma \leftarrow \sigma_{t+1}$ and resume execution.
5. Repeat until task success or timeout.

The LLM is used at inference time only, with no task-specific fine-tuning.

---

## What the Paper Claims

Body-grounded replanning:
1. Maintains high task success rates under controlled joint load increases that cause the original strategy to become physically unsuitable.
2. Reduces cumulative physical effort (total joint torque) compared to persisting with the original strategy.
3. Enables more efficient strategy adaptation compared to random strategy switching when mobility constraints are asymmetric.
4. Works with off-the-shelf LLMs without task-specific training.

---

## Experimental Findings

- Evaluated on a reaching task on a simulated robot arm and a real robot.
- Two constraint types: controlled load (deadweight applied to specific joints) and asymmetric mobility (joint range reduced on one side).
- Body-grounded replanning achieves significantly higher success rates than baselines that ignore body state.
- Physical effort is reduced by selecting strategies that distribute load across available joints.
- LLM strategy selection quality is robust to the exact prompt formulation, suggesting the task is within the LLM's in-context generalization capability.

---

## Ablations and Interpretation

- **Event threshold sensitivity:** Performance is robust across a range of $\tau_\text{thresh}$ values; too-low thresholds cause excessive replanning, too-high thresholds cause physical overload.
- **History window length:** Including more execution history improves strategy selection for mobile constraints; load constraints show less sensitivity.
- **LLM choice:** The framework works across multiple off-the-shelf LLMs; stronger models show better strategy selection quality.

---

## Reference

Namiko Saito, Hiroshi Kera. **Body-Grounded Replanning for Physically Adaptive Manipulation**. arXiv:2609.30024, September 2026.  
https://arxiv.org/abs/2609.30024
