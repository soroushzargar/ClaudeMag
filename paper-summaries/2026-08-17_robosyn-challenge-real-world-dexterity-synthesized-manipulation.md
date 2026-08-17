# RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills

**arXiv:** 2608.12416
**Authors:** Runyi Zhao, Ruixin Wu, Chengkun Li, Hongrui Zhang, Ang Li, Ruixing Jin, Yueci Deng, Yingying Guo, Lihe Ding, Shaocong Dong, Tianfan Xue, Yanjun Gao, Yudong Luo, Pascal Poupart, Simo Wu, Kui Jia, Wei-shi Zheng, Guiliang Liu
**Venue:** NeurIPS 2026 Competition Track
**Date:** August 2026
**URL:** https://arxiv.org/abs/2608.12416

---

## Summary

Generalizable robotic manipulation is severely bottlenecked by the scarcity and narrow diversity of real-world demonstration data. RoboSynChallenge is a unified benchmark and competition (NeurIPS 2026 Competition Track) that evaluates the generalizability of manipulation policies across tasks, environments, and difficulty levels. The challenge integrates large-scale synthetic data generation with standardized real-world robotic evaluation: participants train policies using synthesized state-action trials, but final assessments are conducted exclusively on unseen real-world manipulation environments. This design directly tests sim-to-real transfer and compositional generalization beyond the training distribution.

## Problem

Modern robot manipulation policies have achieved impressive performance within narrow task distributions but fail to generalize across:
- **Task variation:** New objects, new grasps, new goal configurations
- **Environment variation:** Novel backgrounds, lighting, occlusion patterns
- **Difficulty scaling:** Tasks that require sequential sub-goals or dexterous finger control

The core bottleneck is **data scarcity**: collecting diverse real-world demonstrations is expensive, time-consuming, and limited by the physical robot platform. Synthetic data offers scale and diversity, but the sim-to-real gap remains a fundamental challenge: policies trained on synthetic data often fail on real hardware due to differences in physics, visual appearance, and sensor noise.

## Challenge Design

RoboSynChallenge is structured as a two-phase competition:

### Phase 1: Synthetic Training
- Participants receive access to a **large-scale synthetic dataset** of state-action pairs generated in simulation
- The dataset spans diverse object categories, manipulation tasks (pick-and-place, articulated object manipulation, tool use), and environment configurations
- Participants may supplement with additional synthetic data generation using provided simulators
- No real-world data is provided for training

### Phase 2: Real-World Evaluation
- Policies are transferred directly to a **standardized physical robot platform**
- Evaluation tasks are **unseen during training** — new objects, new goal configurations, new environments
- Tasks are stratified by difficulty: Tier 1 (basic manipulation), Tier 2 (multi-step dexterous tasks), Tier 3 (open-vocabulary goal specification)
- Evaluation is fully standardized: same robot, same evaluation protocol, same success metrics across all participants

### Benchmark Metrics

The challenge measures:
- **Task success rate** per difficulty tier
- **Generalization gap:** performance drop from simulation (where evaluable) to real world
- **Sample efficiency:** performance as a function of synthetic data volume used

## Technical Landscape

The challenge addresses open research questions in:
- **Sim-to-real transfer:** Domain randomization, physics adaptation, visual sim-to-real
- **Policy architecture:** Diffusion policies, flow matching, transformer-based policies
- **Data generation:** Procedural scene generation, motion retargeting, contact-rich synthesis
- **Multi-task generalization:** Single-model policies that handle task diversity without per-task fine-tuning

## Why It Matters

The challenge reflects a broader shift in robotics research: as generalist robot policies (trained on large diverse datasets) become feasible, evaluating their generalizability requires standardized benchmarks that go beyond in-distribution performance. RoboSynChallenge is the first competition to pair synthetic-only training with unseen real-world evaluation at scale, providing a clean test of whether current synthetic data pipelines can bootstrap truly generalizable dexterity. The NeurIPS 2026 Competition Track placement ensures broad community participation and cross-institutional comparison.

## Key Contributions

1. Unified benchmark evaluating manipulation policy generalizability across tasks, environments, and difficulty tiers
2. Large-scale synthetic training dataset with standardized real-world evaluation protocol
3. Three-tier difficulty structure separating basic manipulation from dexterous and open-vocabulary tasks
4. First competition to test synthetic-only training against unseen real-world evaluation at the NeurIPS scale
