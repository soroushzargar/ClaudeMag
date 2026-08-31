# VGI-Bench: Probing Visual Intelligence in Video Generation Models

**arXiv:** 2608.19583  
**Authors:** Xuan He et al. (23 authors total)  
**Submitted:** August 20, 2026  
**Area:** Video Generation, Visual Reasoning, Generative Models, Benchmark

---

## Summary

VGI-Bench is a benchmark of 27 tasks and 810 instances that evaluates whether video generation models can reason visually through their generated frames — not just produce perceptually plausible content. Using a two-level taxonomy and automated verifier-based scoring, VGI-Bench reveals that even the strongest tested model (Seedance 2.0) achieves only 51.0% across tasks, with common failure modes including physical inconsistency, rule violation, and inability to self-correct erroneous early states.

## Problem

Video generation models have rapidly improved in perceptual quality, but recent probing studies suggest they may exhibit latent forms of visual reasoning: generating coherent multi-step processes that satisfy physical constraints. Evaluating this capability is difficult because:

1. **Plausibility bias:** Existing benchmarks score whether generated frames look plausible, not whether they represent a valid evolving process satisfying specific constraints
2. **Misaligned inputs:** Benchmarks designed for discriminative VLMs use image-caption pairs not aligned with the generation priors of video models
3. **Saturation risk:** If tasks are too easy for strong models, the benchmark fails to differentiate capability levels

VGI-Bench addresses all three challenges in a unified evaluation framework.

## Benchmark Design

**Two-level taxonomy:** Tasks are organized by *task domain* (physics simulations, spatial reasoning, logical transformations, etc.) and *skill tag* (constraint satisfaction, state tracking, multi-step planning, etc.). The crossing of 9 domains × 5 skill tags yields 27 distinct task types.

**810 instances** are constructed from hidden structured specifications that are:
- Rendered as visual scenes appropriate for video model input
- Evaluated by executable verifiers that accept any feasible solution (not just a single ground-truth outcome)
- Calibrated so that the task is challenging but partly feasible for contemporary models

**Evaluation methodology:** Models generate videos conditioned on the visual scene prompt. Generated videos are scored by a VLM-based verifier that extracts terminal states and verifies satisfaction of the underlying specification. The metric is **Verifier Acceptance Rate (VAR)**.

## Task Categories

Example task types in VGI-Bench include:

- **Physical constraint satisfaction:** Generate a video where objects reach a target configuration while obeying gravity and collision physics
- **Counting and arrangement:** Generate a video where objects are rearranged to satisfy a numerical constraint
- **Rule-following evolution:** Generate a video of a system (e.g., cellular automaton, traffic) that correctly follows specified update rules for N steps
- **State tracking under transformation:** Generate a video tracking object identity through occlusions and deformations

## Results

Evaluated models include Seedance 2.0, Wan-2.1, Kling-2.0, and several others:

- **Best model:** Seedance 2.0 at 51.0% VAR
- **Most models:** Score below 40% VAR across the 27 tasks
- **Per-domain variation:** Physics constraint tasks prove most difficult (Seedance 2.0: 38%); counting and arrangement tasks are relatively accessible (Seedance 2.0: 67%)
- **Per-skill variation:** Multi-step planning tasks (skill tag) show the largest cross-model spread, suggesting this is a key differentiator

**Denoising trajectory analysis:** Using step-by-step inspection of the diffusion denoising process, the authors find that:
- Models commit to visual hypotheses early (in the first ~20% of denoising steps)
- Later denoising steps refine texture and detail but rarely reverse structural errors
- Self-correction of incorrect early states is essentially absent in all tested models

**Failure modes:**
- Physical collapse (objects interpenetrating, violating gravity): 31% of failures
- Rule violation (system evolves incorrectly for 1+ steps): 28% of failures
- Object/state inconsistency (objects change identity or disappear): 24% of failures
- Correct final state but incorrect trajectory: 17% of failures

## Key Contributions

1. A benchmark specifically designed to evaluate visual reasoning through generation, with verifier-based scoring that accepts any feasible solution
2. A two-level taxonomy that enables fine-grained diagnosis of model strengths and weaknesses
3. Analysis of the denoising trajectory as a window into visual reasoning dynamics, revealing the early-commitment failure mode
4. Evidence that current video generation models have limited visual intelligence beyond perceptual quality

## Significance

As video generation models are increasingly used for planning, simulation, and reasoning-augmented generation, VGI-Bench provides a principled way to measure progress on visual intelligence as distinct from perceptual quality. The finding that even the strongest models fail on more than half the tasks highlights the distance between current generative capabilities and genuine visual reasoning.
