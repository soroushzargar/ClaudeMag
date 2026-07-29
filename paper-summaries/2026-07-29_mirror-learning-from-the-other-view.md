# MIRROR: Learning from the Other View for Multi-Modal Reasoning

**Authors:** Wen Ye, Yuxiao Qu, Aviral Kumar, Xuezhe Ma  
**Affiliation:** University of Southern California; Carnegie Mellon University  
**Venue:** arXiv:2607.21552  
**Date:** July 23, 2026  
**URL:** https://arxiv.org/abs/2607.21552

---

## Summary

Vision-language models (VLMs) trained with massive image-text data still underperform on visual reasoning tasks—even on geometry problems that admit mathematically equivalent text, diagram, and combined views. MIRROR exploits the key observation that different views of the same problem expose complementary reasoning paths, and trains VLMs to learn from the view in which they currently perform best, then transfers that signal to improve performance on the harder view.

## Problem

Standard VLM post-training treats modality as a fixed input channel and optimizes a single training objective over mixed multimodal data. This ignores a systematic inconsistency: the same mathematical problem, presented as (a) a pure text description, (b) a geometric diagram, or (c) a combined diagram+text, elicits qualitatively different performance from the same model. A model may solve the text version while failing on the diagram—or vice versa.

This inconsistency indicates that different views activate complementary reasoning paths. Standard post-training conflates these paths, averaging over them rather than reinforcing each.

## Dataset: ODA-Data

The authors construct **ODA-Data** (One-problem, Diverse-views, Annotated), a paired multimodal geometry benchmark with:
- **Text-dominant (T) view:** The problem stated in natural language, with no diagram.
- **Image-dominant (I) view:** A geometric diagram with minimal text labels.
- **Combined (T+I) view:** Diagram plus full textual statement.

Each problem appears in all three views with identical answers, enabling direct comparison of per-view model behavior. The dataset contains 8,400 problems spanning plane geometry, solid geometry, and coordinate geometry, with human-verified solutions.

## Method: MIRROR

**Modality-Informed Reciprocal Reasoning Optimization (MIRROR)** proceeds as follows:

1. **View evaluation:** For each training problem, evaluate the current policy on all three views.
2. **Best-view identification:** Identify the view $v^*(p)$ on which the policy achieves the highest reward for problem $p$.
3. **Cross-modal reward shaping:** Compute a reward bonus for views $v \neq v^*(p)$ proportional to how well the model uses the best-view reasoning trace when answering from the weaker view.
4. **RL update:** Update the policy using GRPO with the augmented rewards, simultaneously training on all views with view-dependent reward shaping.

The key invariant: the model is always rewarded for progress on weaker views guided by what it already knows from stronger views, creating a reciprocal improvement loop.

## Technical Formulation

Let $\mathcal{V} = \{T, I, T\!+\!I\}$ be the view set and $\pi_\theta$ the policy. For problem $p$ with answer $a$:

$$r(p, v) = \mathbf{1}[\pi_\theta(p_v) = a]$$

The best view is $v^*(p) = \arg\max_v r(p, v)$. The MIRROR reward for view $v$ at problem $p$ is:

$$\tilde{r}(p, v) = r(p, v) + \lambda \cdot \mathbf{1}[v \neq v^*(p)] \cdot \text{KL}(\pi_\theta(\cdot \mid p_v) \| \pi_\theta(\cdot \mid p_{v^*}))^{-1}$$

This encourages the policy on weaker views to adopt the reasoning distribution of the best view, rather than diverging from it.

## Key Results

MIRROR is evaluated on ODA-Data (held-out test set) and three external geometry benchmarks: GeoQA, UniGeo, and Geometry3K.

| Model | Text | Image | Combined | Avg. |
|-------|------|-------|---------|------|
| LLaVA-1.6 (base) | 61.3 | 44.2 | 58.7 | 54.7 |
| + GRPO only | 68.1 | 49.8 | 64.3 | 60.7 |
| + MIRROR | **74.5** | **61.2** | **70.8** | **68.8** |

The image-dominant view shows the largest absolute improvement (+17pp over base), confirming that cross-modal reward shaping most benefits the visually harder reasoning path.

## Ablation: Reciprocity Matters

Removing the cross-modal reward shaping term (training on each view independently) reduces image accuracy by 9.4pp relative to full MIRROR. Using only the best-view signal without shaping toward weaker views reduces it by 6.1pp. Both ablations confirm that the reciprocal structure is essential.

## Significance

MIRROR establishes a general RL framework for exploiting view-dependent complementarities in VLM training. The approach is not specific to geometry—any domain with multiple equivalent representations of the same problem (code/text descriptions of algorithms, audio/text for speech tasks, video/text for causal reasoning) is a natural application target.
