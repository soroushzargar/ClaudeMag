# τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation

**arXiv:** 2608.16885  
**Authors:** SII Research team  
**Submitted:** August 17, 2026  
**Area:** Robotics, Vision-Language-Action Models, Embodied AI

---

## Summary

τ₀-VLA is a hierarchical vision-language-action model that makes long-horizon robot manipulation compute-scalable: when the high-level policy is uncertain about what subtask to execute next, it invokes a world-model-guided proposal-predict-evaluate loop rather than committing to a single forward pass. Trained on over 40,000 hours of real-world manipulation data, the system achieves substantial gains in long-horizon task success with additional test-time computation.

## Problem

Long-horizon robot manipulation is fundamentally a two-level problem:
1. **Skill execution:** reliably carry out each individual manipulation primitive
2. **Subtask sequencing:** coherently plan which skills to execute and when, across a task that may span dozens of steps and minutes of real time

Most hierarchical VLA models make high-level decisions — "pick up the cup," "open the drawer" — with a single forward pass through the policy network. This single-shot commitment works when the policy is confident, but fails when the current observation is ambiguous, the task is novel, or earlier steps have introduced unexpected intermediate states.

The problem is that there is no mechanism to spend more computation on hard decisions. The policy either commits in one pass or it doesn't, and there is no signal to know which regime applies.

## Method

τ₀-VLA introduces adaptive test-time computation for the high-level policy:

**Memory-augmented high-level policy:** The high-level policy receives the current RGB observation and an execution memory — a compressed representation of the subtasks executed so far and their outcomes. This allows the policy to condition on long-horizon context without relying on a full token-level history.

**Confidence monitoring:** After each high-level decision, the policy's token confidence statistics (mean token probability of the generated subtask string) are compared against a calibrated threshold $\tau_c$. When confidence exceeds $\tau_c$, the proposal is used directly. When confidence falls below $\tau_c$, additional computation is triggered.

**Propose-predict-evaluate loop:** Under low confidence, three modules work together:
1. **Proposal model:** Generates $K$ diverse candidate subtasks using nucleus sampling
2. **World model:** For each candidate, predicts the terminal RGB observation that would result from executing that subtask
3. **Value model:** Scores each (candidate, predicted-outcome) pair by predicted task progress

The highest-scoring candidate is selected and executed.

## Technical Formulation

Let $o_t$ be the current RGB observation and $m_t$ the execution memory at step $t$. The high-level policy generates subtask $\hat{s}_t$:

$$\hat{s}_t = \arg\max_s \pi_\text{high}(s | o_t, m_t)$$

Token confidence is measured as:
$$c_t = \exp\left(\frac{1}{|s|}\sum_{i=1}^{|s|} \log \pi_\text{high}(s_i | o_t, m_t, s_{<i})\right)$$

When $c_t < \tau_c$, the expanded loop generates $K$ candidates $\{s^k\}_{k=1}^K$, predicts world states $\{\hat{o}^k_{t+1}\}$, and selects:
$$s^* = \arg\max_k V(s^k, \hat{o}^k_{t+1}, m_t)$$

The memory update appends the executed subtask and a binary success signal:
$$m_{t+1} = \text{MemEncode}(m_t, s^*, \text{success}_t)$$

## Training Data and Procedure

τ₀-VLA is trained on **40,115 hours** of heterogeneous real-world robot manipulation data spanning:
- Tabletop manipulation with articulated objects
- Kitchen and household task completion
- Dexterous in-hand manipulation
- Multi-step assembly and disassembly

The high-level policy, world model, and value model are trained separately with distinct supervision:
- **High-level policy:** next-subtask prediction with cross-entropy loss
- **World model:** next-observation prediction with diffusion loss
- **Value model:** trajectory outcome regression

## Experimental Results

- Additional test-time computation (increasing $K$ from 1 to 8) improves subtask prediction accuracy by **9.8 percentage points** on held-out long-horizon tasks
- Long-horizon closed-loop success rate improves by **14.2 percentage points** over the base single-pass policy
- The confidence threshold $\tau_c$ correctly identifies hard decisions 78% of the time, avoiding unnecessary compute on easy decisions
- World model prediction quality correlates strongly with value model accuracy ($r = 0.81$), validating the propose-predict-evaluate pipeline

## Significance

τ₀-VLA is among the first demonstration that test-time compute scaling works for hierarchical robot control at the same qualitative level it works for language reasoning. The key insight is that world models provide a differentiable proxy for execution outcomes, enabling a form of "internal simulation" that guides high-level planning without needing to execute candidate subtasks in the real world.
