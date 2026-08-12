# Spatiotemporal Agility: Time-Constrained Reinforcement Learning for Vision-Guided Dynamic Quadrupedal Interception

**arXiv:** 2608.06907
**Authors:** Yidong Zhu, Zibo Dai, Tongning Zhang, Leixin Chang, Hua Chen
**Affiliation:** Zhejiang University; LimX Dynamics Technology Co., Ltd.
**Venue:** arXiv preprint
**Date:** August 7, 2026
**URL:** https://arxiv.org/abs/2608.06907

---

## Summary

Most quadruped locomotion policies rely on velocity-tracking commands and cannot reach a precise spatial target within a strict time deadline, limiting physical agility in dynamic interaction tasks. This paper introduces a ball-catching benchmark for legged robots and a unified RL framework combining a vision module (predicting landing point and time) with a position-and-time conditioned locomotion policy. Rather than decomposing control into a high-level planner and a velocity-tracking low-level controller, the proposed system is conditioned directly on spatial targets and temporal deadlines, learning to coordinate body movement and timing end-to-end.

## Problem

Legged robots need to do more than follow velocity commands — they need to interact with the physical world under time constraints. A quadruped catching a thrown ball must:
1. Predict where and when the ball will land from visual observations
2. Move the body to that position before that time
3. Coordinate posture for interception

Velocity-tracking policies fail at this because they lack a temporal objective: they can navigate to a position but not guarantee arrival by a deadline. Decomposing the task into planner + velocity tracker introduces compounding errors and limits agility.

## Ball-Catching Benchmark

The paper introduces a ball-catching task as a concrete, reproducible benchmark for agility under temporal constraints. A ball is thrown with randomized trajectory and timing; the robot must use visual observations to predict the landing location and time, then move to intercept the ball before it lands. Success requires both spatial precision and temporal coordination.

## Framework

**Vision Module:** Processes visual input (camera images) to predict:
- Landing point (3D position)
- Time-to-landing (seconds remaining)

**RL Locomotion Policy:** Conditioned on (position target, time deadline) rather than velocity commands. Learns to:
- Compute appropriate gait parameters for timed locomotion
- Trade off speed and stability given the temporal budget
- Coordinate body orientation for interception

The two modules are trained jointly with the RL reward tied to successful interception, allowing the vision and locomotion components to co-adapt.

## Why This Matters

Most legged robotics work focuses on stable navigation over varied terrain. Dynamic interaction tasks like catching, intercepting, or dodging require a qualitatively different capability: spatiotemporal reasoning. The time-conditioned policy framework is general and applicable to any task where a robot must reach a location by a deadline. The ball-catching benchmark provides a clean, measurable evaluation protocol that the community can build on.

## Key Contributions

1. Ball-catching benchmark for legged robots: a standardized test of spatiotemporal agility
2. Time-conditioned RL policy: direct conditioning on (position, deadline) rather than velocity commands
3. Joint vision + locomotion training: vision and locomotion co-adapt through the RL reward
4. Demonstration that end-to-end temporal-spatial conditioning outperforms velocity-tracking decomposition
