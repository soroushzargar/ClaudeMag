# Temporal GRPO: Beyond Trajectory-Level Credit in Vision-Language-Action Reinforcement Learning

**arXiv:** 2608.13026  
**Date:** August 2026  
**Authors:** Yao Zhou, Hang Gao, Fengge Wu, Changwen Zheng, Wenwen Qiang  
**URL:** https://arxiv.org/abs/2608.13026

---

## Summary

Reinforcement learning has emerged as a powerful approach for training Vision-Language-Action (VLA) models — systems that take visual observations and language instructions as input and output robotic actions. The dominant RL algorithm for VLA training, GRPO (Group Relative Policy Optimization), assigns rewards at the trajectory level: an entire action sequence receives the same scalar reward based on whether the task succeeded or failed. This coarse credit assignment fails to distinguish between good and bad individual actions within a trajectory, slowing learning.

## The Credit Assignment Problem in VLA RL

Consider a robot arm attempting to grasp an object. A successful grasp trajectory may contain 50–100 individual action steps. Under trajectory-level GRPO, every action in a successful trajectory is treated as equally positive, and every action in a failed trajectory is treated as equally negative. In practice:

- Some actions are critical (the precise grasp contact)
- Some are inconsequential (stable approach motions)
- Some in a failed trajectory may have been correct (good approach, bad grasp)
- Some in a successful trajectory may have been suboptimal but compensated for later

Trajectory-level credit assignment provides an uninformative gradient signal that treats all of these cases identically.

## Temporal GRPO

Temporal GRPO extends GRPO with a temporal credit assignment mechanism that decomposes trajectory-level rewards into step-level credit signals. The key modifications are:

1. **Temporal decomposition:** Given trajectory reward $R$, Temporal GRPO computes a credit $r_t$ for each step $t$ by conditioning on the temporal context — the state before and after the action, and the remaining trajectory.

2. **Credit conditioning on temporal position:** Actions taken early in the trajectory (approach phase) and late (contact and post-grasp) are credited differently, reflecting the distinct roles they play in task success.

3. **Group relative baseline:** Following GRPO's group-relative design, step-level credits are normalized relative to the same step across a group of sampled trajectories, providing a meaningful baseline that removes common variance.

## Results

Temporal GRPO demonstrates improved sample efficiency and final task success rates compared to standard GRPO on robotic manipulation benchmarks. The temporal credit decomposition provides more informative gradient signals, particularly for long-horizon tasks where the gap between a good early action and a bad late action is large.

## Significance

As VLA models scale and are applied to increasingly complex manipulation tasks, the quality of RL training signal becomes critical. Temporal GRPO represents an important step toward fine-grained credit assignment in embodied AI, moving from trajectory-level to step-level learning signals without requiring dense reward shaping or additional supervision.
