# How Do VLMs Behave When Blind or Misled? Behavioral Evaluation of VLMs on Scientific Figures

**arXiv:** 2608.13267  
**Authors:** Paul Osemudiame Oamen, Owusu-Banahene Osei, Ananya Mukherjee, Christian Greisinger, Steffen Eger, Pius Onobhayedo, Wei Zhao  
**Submitted:** August 13, 2026  
**Area:** Vision-Language Models, Behavioral Evaluation, Scientific AI

---

## Summary

SciFigBench reveals that current VLMs are unreliable scientific assistants: they frequently confabulate answers when critical visual information is blurred or removed, and are systematically biased by figure captions even when captions directly contradict visual content. The benchmark exposes a gap that accuracy-focused evaluations miss — behavioral reliability under uncertainty.

## Problem

AI assistants for scientific reading, peer review, and data extraction are increasingly deployed to interpret figures in research papers. The implicit assumption is that a model that accurately answers questions about clear figures will also behave appropriately when the figure is degraded or when contextual cues are misleading.

This assumption is wrong. Existing benchmarks test accuracy on clean images with unambiguous ground truth, but do not test:

- **Uncertainty behavior:** Does the model acknowledge when it cannot see the relevant information, or does it confabulate?
- **Caption resistance:** Does the model trust visual evidence over a misleading caption, or does it defer to the caption?
- **Selective degradation robustness:** Can the model correctly identify that a specific visual element is illegible, or does it hallucinate content?

For scientific AI assistants, all three behaviors matter: a model that makes up data values when a figure axis is blurred, or accepts an incorrect caption over correct visual evidence, is potentially dangerous in scientific contexts.

## Benchmark Design

**SciFigBench** contains **250 scientific figures** from across STEM domains (biology, physics, chemistry, computer science, medicine), each with high-quality human annotations produced by domain experts.

**Annotation dimensions:**
1. **Perception accuracy:** Can the model correctly extract data values and identify visual elements?
2. **Reasoning quality:** Can the model correctly answer inference questions about trends, comparisons, and conclusions?
3. **Behavioral reliability:** Does the model express appropriate uncertainty when visual evidence is missing or misleading?

**Evaluation setups (>34,000 total):**
- **Image transformations:** Gaussian blur, label removal, color masking of specific visual elements
- **Reasoning questions:** Quantitative (exact values) and qualitative (trends, comparisons) questions
- **Resistance probes:** Questions where the model is told (falsely) that specific values can be read from the figure — tests whether model appropriately reports inability vs. compliance with false premise
- **Caption-bias probes:** Figures paired with captions that contradict the visual content — tests whether model trusts visual evidence or caption
- **Selective-blur targets:** Specific data elements blurred while rest of figure is intact — tests fine-grained awareness of missing information

## Key Findings

**Confabulation under visual degradation:**
- When key visual elements (axis labels, data series labels, bar values) are blurred, models provide specific numerical answers **62–84% of the time** rather than expressing uncertainty
- This rate is nearly unchanged from clean-image conditions, indicating that models are not systematically detecting their own visual limitations

**Caption bias:**
- When figure captions contradict visual content (e.g., claiming a trend opposite to what the graph shows), models align with the caption over the visual **41–67% of the time** depending on model
- Larger models exhibit more caption bias, not less, suggesting this is a training artifact of caption-image co-training

**Resistance probe failures:**
- When falsely told that a specific value is readable from a blurred region, models comply with the false premise **71% of the time**, generating fabricated values rather than refusing
- Refusal rates are below 15% across all evaluated models, including frontier systems

**Ranking by behavioral reliability:**
Models ranked by standard accuracy (perception + reasoning) do not correlate well with behavioral reliability scores (Spearman $\rho = 0.31$), indicating that accuracy evaluations do not predict safe deployment for scientific use.

## Technical Formulation

The behavioral reliability score (BRS) for model $M$ on benchmark instance $(f, q, c)$ (figure, question, perturbation condition $c$) is:

$$\text{BRS}(M) = \frac{1}{|\mathcal{P}|}\sum_{c \in \mathcal{P}} \frac{1}{|Q_c|}\sum_{q \in Q_c} \mathbf{1}[M(f_c, q) \text{ is behaviorally appropriate}]$$

where $f_c$ is figure $f$ under perturbation $c$, and "behaviorally appropriate" means: expressing uncertainty when the visual information is unavailable; reporting visual evidence over caption when captions conflict; and refusing to read values from blurred regions.

The caption-bias score (CBS) measures the fraction of cases where the model aligns with a contradictory caption over the visual truth:

$$\text{CBS}(M) = \mathbb{E}_{(f, q)}\left[\mathbf{1}[M(f, q, \text{cap}_\text{wrong}) \neq M(f, q, \text{no cap})]\right]$$

## Significance

SciFigBench establishes behavioral reliability as a distinct evaluation dimension, separate from accuracy, and demonstrates that current frontier VLMs systematically fail on it. As AI scientific assistants are adopted in literature review, peer review, and automated data extraction, systems that confabulate when uncertain pose non-trivial risks — and standard benchmarks do not surface these risks.
