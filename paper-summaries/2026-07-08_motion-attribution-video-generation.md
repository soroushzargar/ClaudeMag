# Motion Attribution for Video Generation

**Authors:** Xindi Wu, Despoina Paschalidou, Jun Gao, Antonio Torralba, Laura Leal-Taixe, Olga Russakovsky, Sanja Fidler, Jonathan Lorraine  
**Affiliation:** NVIDIA, MIT, Princeton University  
**Venue:** ICML 2026 Oral  
**arXiv:** 2601.08828  
**URL:** https://arxiv.org/abs/2601.08828

---

## Headline Finding

MOTIVE (MOTIon attribution for Video gEneration) is the first data attribution framework that targets **temporal dynamics** rather than static appearance in video generation models. Data curation guided by MOTIVE achieves a **74.1% human preference win rate** over the pretrained baseline and raises VBench dynamic degree from 41.3% to 47.6%.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
- **74.1% human preference win rate** vs. pretrained base model in motion quality assessment
- VBench **dynamic degree: 47.6%** (vs. 41.3% random selection, 43.8% whole-video attribution)
- Scales to modern large-scale high-quality video datasets and production-grade video diffusion models
- First framework specifically attributing motion rather than appearance in video generative models

### Level 2 — Core Mechanism
Standard gradient-based data attribution (influence functions) identifies training examples that most affect a loss function. For video, this conflates two orthogonal signals: static visual appearance and temporal dynamics (motion). MOTIVE decouples them by computing **motion-weighted loss masks** derived from optical flow: regions of high temporal motion receive high weight, static backgrounds receive near-zero weight. Attribution gradients then measure influence on the motion-specific loss, revealing which training clips actually teach the model how things move.

### Level 3 — Methodology
MOTIVE proceeds in three stages:
1. **Flow estimation**: compute per-pixel optical flow between consecutive frames of each training clip
2. **Motion masking**: construct a soft spatial-temporal weight map $M(x, t) \propto \|f(x,t)\|_2$ from flow magnitude $f$
3. **Influence computation**: compute motion-weighted influence scores $\mathcal{I}(z_i) = -\nabla_\theta \mathcal{L}_M(z_\text{test})^\top H_\theta^{-1} \nabla_\theta \mathcal{L}_M(z_i)$ via LiSSA approximation

High-influence clips are selected for fine-tuning, preferentially including training examples with motion patterns similar to the target domain.

### Level 4 — Technical Details
The motion-weighted loss for a video clip $v$ with frames $\{f_1, \ldots, f_T\}$ is:
$$\mathcal{L}_M(v; \theta) = \sum_{t=1}^{T} \sum_{x \in \Omega} M(x,t) \cdot \ell(v_\theta(x,t), f_t(x))$$
where $M(x,t) = \text{softmax}_\tau(\|\nabla_{t} f_t(x)\|_2)$ normalizes flow magnitudes with temperature $\tau$ across spatial positions. The Hessian-vector product $H_\theta^{-1} v$ is approximated via 3 steps of LiSSA recursion, making MOTIVE scalable to billion-parameter video models.

---

## Why Existing Approaches Fall Short

Prior data attribution methods for generative models (e.g., TRAK, DataInf) treat video as static image sequences or average loss uniformly across spatial-temporal positions. This causes appearance-dominant regions (backgrounds, textures) to overwhelm motion-dominant regions (moving objects, camera dynamics) in the influence computation. The result: selected training data improves visual quality but not motion dynamics, or even degrades temporal consistency while improving per-frame sharpness.

---

## Significance

Video generation quality is increasingly evaluated on temporal realism and physical plausibility rather than per-frame aesthetics. MOTIVE provides the first principled tool for understanding which training data drives these properties, enabling targeted data curation for motion-quality improvement. The framework is model-agnostic and applicable to any score-based or flow-based video diffusion architecture.
