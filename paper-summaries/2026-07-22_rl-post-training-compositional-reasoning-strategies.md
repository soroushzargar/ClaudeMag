# RL Post-Training Builds Compositional Reasoning Strategies

**arXiv:** 2607.07646
**Date:** July 8, 2026
**Authors:** Azwar Abdulsalam, Nishil Patel, Andrew Saxe
**Affiliation:** UCL Gatsby Computational Neuroscience Unit
**URL:** https://arxiv.org/abs/2607.07646

---

## Headline

Researchers at UCL's Gatsby Unit show that RL post-training with only binary final-answer reward can compose pretrained primitive skills into genuinely new higher-level strategies, solving problems the base model cannot solve even with unlimited sampling — while rejection fine-tuning plateaus at a lower ceiling.

---

## Key Findings

- RL post-training, given only binary correctness feedback, solves held-out composite problems that remain rarely solved by the base model even under large sampling budgets (pass@100).
- Rejection fine-tuning (RFT) improves on composite problems early in training but plateaus before reaching RL's ceiling, suggesting it cannot build the same compositional machinery.
- RL reorganizes primitive competence through a two-phase mechanism: first strengthening individual primitive reductions, then discovering valid sequential and parallel compositions of those primitives.
- Sequential compositions collapse chains of primitive contractions into single steps; parallel compositions combine independent primitive contractions into joint procedures.
- The rewrite-grammar environment allows ground-truth verification of exactly which compositional strategies emerge and in what order, without ambiguity.

---

## Methodology

The authors construct a controlled environment based on a symbol rewrite-grammar with known primitives and known compositions:

1. **Pretraining:** A Transformer is trained on corpora of primitive symbol-rewrite chains — each example shows a single rule application from input to output. The model learns all primitive reductions but sees no compositions.
2. **Post-training task:** The model is post-trained on composite Trace tasks — multi-step problems that require applying sequences or combinations of primitive rules to reach the target. Only binary final-answer reward is used; no intermediate steps are annotated.
3. **Baselines:** Rejection fine-tuning (RFT) curates correct rollouts from the base model and fine-tunes on them. Best-of-N sampling probes what the base model can achieve with more inference compute.
4. **Analysis:** Individual trace steps are classified by type (primitive, sequential composition, parallel composition) across training checkpoints to track compositional emergence over time.

---

## Technical Details

- **Grammar:** A formal rewrite system with $K$ primitive rules $\{r_1, \ldots, r_K\}$; composite problems require applying $r_i \circ r_j$ (sequential) or $r_i \parallel r_j$ (parallel) for $i \neq j$.
- **Transformer architecture:** Standard decoder-only Transformer with causal attention; trained from scratch to isolate the contribution of post-training.
- **RL algorithm:** REINFORCE with binary reward; no value baseline to avoid introducing implicit knowledge of composite structure.
- **Compositional emergence metric:** For each trace step, classify as primitive ($P$), sequential composition ($S$), or parallel composition ($\Pi$). Track $\Pr[S], \Pr[\Pi]$ over training steps.

---

## Experimental Findings

- Base model (pretrained only): pass@1 on composite problems < 5%; pass@100 < 20% — confirming that compositions are not latent in the base policy under increased sampling.
- RFT: reaches approximately 40% accuracy on composite problems before plateauing.
- RL: reaches approximately 78% accuracy on composite problems, far above the RFT plateau.
- Phase 1 of RL (steps 0–600): primitive accuracy improves sharply; sequential composition rate rises from near-0 to ~30%.
- Phase 2 of RL (steps 600–2000): parallel composition rate rises from near-0 to ~45%; sequential composition rate stabilizes.
- The phase transition at step ~600 is consistent across random seeds and grammar instantiations.

---

## Ablations and Interpretation

- **Removing phase 1 (skipping to phase 2 directly):** Not possible by design — parallel compositions emerge only after sequential ones stabilize, suggesting a curriculum imposed by the task structure rather than by the optimizer.
- **RL with step-level reward:** When intermediate-step rewards are provided (cheating), convergence is faster but final accuracy is similar, confirming that binary reward is sufficient for compositional emergence.
- **RFT saturation:** RFT saturates because the base model's rollout distribution does not include high-quality composite solutions; RFT can only amplify what the base model already does well, confirming the "multiplier" characterization of RL vs. RFT.
- **Mechanism:** The authors propose that RL's gradient operates on the model's implicit planning representations, restructuring the forward pass so that primitive rule applications are chained within a single generation, rather than requiring the model to rediscover each step independently.

---

## Reference

Azwar Abdulsalam, Nishil Patel, Andrew Saxe. **RL Post-Training Builds Compositional Reasoning Strategies.** arXiv:2607.07646, July 2026. https://arxiv.org/abs/2607.07646
