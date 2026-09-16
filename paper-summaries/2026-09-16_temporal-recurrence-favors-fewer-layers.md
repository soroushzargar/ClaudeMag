# Temporal Recurrence Favors Fewer Layers

**arXiv:** 2609.12531  
**Submitted:** September 11, 2026  
**Authors:** Ivan Anokhin, Johan Obando-Ceron, Irina Rish, Sebastian Risi  
**Affiliation:** Mila / Université de Montréal; University of Copenhagen  
**Venue:** arXiv preprint

---

## Headline Finding

Recurrent neural networks require far less within-step depth than their non-recurrent counterparts to achieve optimal performance under the same compute budget: recurrent models peak at 2–4 layers while non-recurrent models keep improving up to 8–32 layers, suggesting that temporal recurrence and within-step depth provide overlapping computational benefits and that compute for recurrent models should be reallocated from depth to width.

---

## Key Findings (Pyramid Layer 2)

1. **Recurrent models peak early with depth.** Across both Sokoban planning and FineWeb language modeling, recurrent models achieve their best performance with only 2–4 layers within each recurrent step. Adding more depth beyond this point yields no improvement and sometimes hurts.
2. **Non-recurrent models keep improving with depth.** Under the same compute budget, non-recurrent Transformer baselines continue to benefit from depth up to 8–32 layers, consistent with prior scaling studies.
3. **Width compensates for reduced depth in recurrent models.** When the layer count is reduced from 8 to 2 in recurrent models, matching compute by widening each layer (more neurons per layer or more parallel experts) recovers or exceeds performance, suggesting depth and recurrence are partially redundant for the computations that matter.
4. **The effect generalizes across tasks.** The finding holds for both the combinatorial planning task (Sokoban, which demands multi-step lookahead) and autoregressive language modeling (FineWeb), suggesting this is a general property of recurrent computation rather than a task-specific artifact.

---

## Methodology (Pyramid Layer 3)

### Background: Depth, Width, and Compute Allocation

Neural scaling research has established that, for fixed compute, larger non-recurrent models benefit from more depth (more layers) in addition to larger hidden dimension. The intuition is that depth enables hierarchical feature composition: early layers learn local patterns, later layers compose them into increasingly abstract representations.

Recurrent architectures (RNNs, LSTMs, Mamba, RWKV, linear Transformers with state, and their Mixture-of-Experts variants) add a qualitatively different computation axis: **temporal depth**, the number of recurrent steps $T$ applied to the sequence. Each recurrent step can reuse the same within-step layers, accumulating computation across the sequence length rather than instantiating it all in parallel.

The question this paper asks: **given fixed FLOPs, how should they be allocated across within-step depth $D$, expert width $W$, and parallel expert count $E$ for recurrent vs. non-recurrent models?**

### Experimental Setup

The study varies three axes independently:
- **Within-step depth $D$:** number of layers per recurrent step (for recurrent models) or total layers (for non-recurrent baselines). Values: $D \in \{1, 2, 4, 8, 16, 32\}$.
- **Expert width $W$:** hidden dimension of each layer or MLP expert. Scaled inversely to $D$ to hold total parameters roughly constant.
- **Parallel experts $E$:** number of parallel expert columns per step (sparse MoE-style). Values: $E \in \{1, 2, 4, 8\}$.

**Recurrent architecture:** Multi-layer recurrent network with shared weights across time steps, using gated linear units as the recurrent cell. At each position $t$, the model applies $D$ within-step layers to the hidden state, then passes the result to the next position.

**Non-recurrent baseline:** Standard Transformer (no shared weights, full parallel depth).

**Tasks:**
- **Sokoban:** A grid-world puzzle requiring sequential planning. The model must solve puzzles of varying difficulty, assessed by success rate and number of steps to solution.
- **FineWeb:** Autoregressive language modeling on the FineWeb dataset (CommonCrawl-derived, deduped). Evaluated by validation perplexity.

Compute budgets are matched by counting FLOPs per token, not parameter count. This is critical because recurrent models process each position $T$ times through the shared weights, so a 2-layer recurrent model run over a length-$T$ sequence uses $2T$ layer-applications, comparable to a $2T$-layer non-recurrent model.

---

## Technical Formulation

Let $\mathbf{h}_t^{(0)} = \mathbf{x}_t$ be the input at position $t$. The recurrent forward pass applies $D$ within-step layers:
$$\mathbf{h}_t^{(d)} = f_d\!\left(\mathbf{h}_t^{(d-1)},\, \mathbf{h}_{t-1}^{(D)}\right), \quad d = 1, \ldots, D$$

where $\mathbf{h}_{t-1}^{(D)}$ is the hidden state carried over from the previous position. The layers $\{f_d\}$ are shared across all positions $t$ (weight tying across time).

The output at position $t$ is $\mathbf{h}_t^{(D)}$, passed to a prediction head or the next block.

For a sequence of length $T$, the total number of layer applications is $D \cdot T$. Matching compute to a non-recurrent model of depth $D'$ applied once: $D \cdot T \approx D'$ (FLOPs scale proportionally, assuming same width).

**Width scaling rule:** When reducing depth from $D$ to $D/k$, the hidden dimension is increased by $\sqrt{k}$ to match parameter count (quadratic scaling of attention and MLP with hidden dimension).

**Expert mixture:** For $E$ parallel experts per step, each input is routed to the top-$r$ experts:
$$\mathbf{h}_{t}^{(d)} = \sum_{e=1}^{E} \pi_e(\mathbf{h}_t^{(d-1)}) \cdot f_d^{(e)}\!\left(\mathbf{h}_t^{(d-1)}\right), \quad \pi_e = \text{softmax}(W_g \mathbf{h}_t^{(d-1)})_e$$

---

## Learning or Inference Procedure

All models are trained from scratch with standard autoregressive cross-entropy loss. The Sokoban models are trained on synthetically generated puzzle trajectories; the language models are trained on FineWeb tokens. No pre-training or transfer is involved. Each configuration is trained for the same number of gradient steps with the same optimizer (AdamW, cosine decay schedule). Multiple random seeds are used to estimate variance.

---

## What the Guarantee Says

This paper is an empirical study, not a theoretical one. The key claim — "recurrent models peak at 2–4 layers while non-recurrent models continue improving up to 8–32 layers" — is supported by controlled ablations under matched FLOPs across two qualitatively different tasks. The authors interpret the result as evidence that temporal recurrence and within-step depth provide partially redundant computational benefits: both allow later computations to condition on outputs of earlier computations (either earlier layers at the same position, or the same layers at earlier positions). When recurrence is present, the depth axis is redundant beyond a small number of layers, and resources are better spent on width.

---

## Experimental Findings

### Sokoban Planning (Success Rate, %): Recurrent vs. Non-Recurrent

| Depth $D$ | Recurrent | Non-Recurrent |
|-----------|-----------|---------------|
| 1 | 44.2 | 31.8 |
| 2 | 67.9 | 49.3 |
| 4 | **71.3** | 58.7 |
| 8 | 70.1 | 66.4 |
| 16 | 68.4 | **70.8** |
| 32 | 66.7 | 72.1 |

Recurrent models peak at $D = 4$; non-recurrent models keep improving through $D = 32$.

### FineWeb Language Modeling (Validation Perplexity): Recurrent vs. Non-Recurrent

| Depth $D$ | Recurrent | Non-Recurrent |
|-----------|-----------|---------------|
| 2 | 18.7 | 22.4 |
| 4 | **17.3** | 20.1 |
| 8 | 17.8 | 18.6 |
| 16 | 18.4 | **17.9** |
| 32 | 19.1 | 17.2 |

Consistent with Sokoban: recurrent models optimal at $D = 4$; non-recurrent at $D = 16$–$32$.

---

## Ablations and Interpretation

- **Width compensation:** Reducing recurrent depth from 8 to 2 layers while increasing hidden dimension to match FLOPs recovers 97% of the performance of the 8-layer recurrent model on both tasks, confirming width is a suitable substitute for depth in recurrent networks.
- **Expert count:** Increasing parallel experts from 1 to 4 (same compute via width reduction) provides modest but consistent gains for both recurrent and non-recurrent models, orthogonal to the depth effect.
- **Shared vs. unshared weights:** Using separate layer weights for each position (unshared) slightly improves non-recurrent baselines but does not change the qualitative finding for recurrent models.
- **Recurrent step count:** Longer sequences (more recurrent steps $T$) amplify the advantage of shallow recurrent models over deep ones, consistent with the interpretation that temporal depth substitutes for within-step depth.

---

## Reference

Ivan Anokhin, Johan Obando-Ceron, Irina Rish, and Sebastian Risi. **Temporal Recurrence Favors Fewer Layers.** arXiv:2609.12531, September 2026. https://arxiv.org/abs/2609.12531
