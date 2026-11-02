# MATCH: Execution-Aware Hierarchical Reinforcement Learning for VLN-CE

**arXiv:** 2610.38248  
**Authors:** Shuhao Ye, Yufei Wei, Demao Wang, Shichao Zhai, Rong Xiong, Yue Wang  
**Submitted:** October 28, 2026  
**Venue:** arXiv preprint (cs.RO)  
**URL:** https://arxiv.org/abs/2610.38248

---

## Summary

Vision-and-Language Navigation in Continuous Environments (VLN-CE) is a robotic task where an agent must follow free-form natural language instructions to navigate realistic 3D environments with continuous action spaces. Unlike graph-based VLN benchmarks, VLN-CE requires the agent to produce low-level motor actions (rotation, translation) rather than selecting named waypoints, making the problem significantly harder.

Hierarchical approaches decompose this into high-level planning (what subgoal to move toward) and low-level execution (how to get there). The standard recipe trains a language-conditioned planner and a separate, often fixed, low-level controller. MATCH identifies the critical weakness of this decomposition: the planner has no awareness of whether its proposed waypoints are executable, and the executor has no influence over what goals it is given. This execution blindness causes cascading failures in navigation.

---

## Method

**Joint Training of Planner and Executor.** MATCH trains both components end-to-end with reinforcement learning. The planner generates intermediate waypoints from the language instruction and current visual observation; the executor generates primitive actions to reach each waypoint. The RL reward from execution propagates back to the planner, so it learns to propose achievable waypoints.

**Execution Feedback.** At each planning step, the executor's trajectory to the current waypoint is summarized into a compact execution-status vector. This vector is passed back to the planner at the next step, closing the hierarchical loop. The planner therefore adapts its next waypoint based on how well the previous one was executed.

**Compact Architecture.** The full system uses only 0.5B parameters, achieved by sharing a backbone between the planner's visual encoder and the executor's image encoder, and by using efficient recurrent modules for the planner's instruction-following.

---

## Results

On the standard VLN-CE benchmarks:

| Benchmark | MATCH SR | Previous SOTA SR |
|-----------|----------|-----------------|
| R2R-CE (val unseen) | **62.4%** | 58.1% |
| RxR-CE (val unseen) | **55.8%** | 51.3% |

- MATCH improves **success rate** by 4.3 points on R2R-CE and 4.5 points on RxR-CE over the previous best.
- **Goal Progress (GP)** and **Navigation Error (NE)** also improve, suggesting the gains reflect better instruction following, not just lucky termination.
- MATCH achieves these results with **0.5B parameters**, competitive with models that use 2–7B parameters.

---

## Significance

The core insight—that hierarchical navigation agents must be trained with execution awareness, not just planned top-down—is both intuitive and underexplored. Most prior VLN-CE work treats the high-level planner as the bottleneck and builds ever more powerful instruction parsers, while keeping the low-level controller fixed. MATCH's joint training approach with execution feedback is a principled correction. The strong results at a small model size suggest the approach is also efficient, which matters for deployment on embodied robots with limited compute.

---

## Limitations

- Evaluated only on R2R-CE and RxR-CE; generalization to other continuous-env VLN datasets (e.g., CVDN-CE) is not reported.
- The execution-feedback vector is handcrafted; a learned representation might capture richer failure signals.
- Training stability of the joint RL objective is not thoroughly characterized.
