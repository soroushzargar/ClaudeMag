# Circuit Condensation: Post-Training that Concentrates a Behavior's Causal Circuit

**arXiv:** 2608.27254  
**Authors:** Sai Adith Senthil Kumar  
**Submitted:** August 28, 2026  
**Area:** Mechanistic Interpretability, LLM Post-Training, Circuit Analysis

---

## Summary

Circuit Condensation investigates whether and how post-training procedures reorganize the causal circuits responsible for specific LLM behaviors. Using automated circuit discovery tools, the paper shows that supervised fine-tuning consistently concentrates circuits (fewer components, higher faithfulness), while RL fine-tuning has behavior-dependent effects: factual circuits condense while multi-step reasoning circuits expand. The findings have direct implications for scalable mechanistic interpretability.

## Problem

Mechanistic interpretability research identifies *circuits* — sparse subgraphs of attention heads, MLP layers, and residual stream components that are causally responsible for specific model behaviors. A central hypothesis is that post-training on behavior-specific data should concentrate these circuits: the model should route computation for the trained behavior through fewer, more dedicated components.

This hypothesis has practical stakes. If post-training concentrates circuits:
- Automated circuit discovery becomes easier (fewer components to identify)
- Targeted editing (activation patching, model surgery) becomes more reliable
- Interpretability scales better as models grow

If post-training instead disperses circuits, interpretability may become harder even as capability improves. But no prior work has measured how circuits change across the full post-training pipeline.

## Method

The paper applies two complementary circuit discovery methods before and after post-training on a suite of behaviors:

**Automated Circuit Discovery (ACDC):** Iteratively prunes edges from a full computation graph, retaining only those where ablation (replacing activations with corrupted counterparts) causes more than a threshold performance drop on the target behavior. Produces a sparse circuit with measurable **edge faithfulness** (fraction of performance recovered by the circuit alone).

**Edge Attribution Patching (EAP):** Assigns importance scores to attention edges via gradient × activation products, enabling fast identification of high-importance edges without exhaustive ablation.

**Behaviors studied:** 8 behaviors spanning factual recall, subject-verb agreement, indirect object identification, multi-step arithmetic, and multi-step logical reasoning.

**Post-training conditions:**
- Supervised fine-tuning (SFT) on behavior-specific demonstrations
- RLHF / GRPO on the same behaviors
- No post-training (base model baseline)

**Circuit metrics:**
- **Circuit size** (number of causally necessary attention heads and MLP layers)
- **Edge faithfulness** (fraction of behavior performance recovered by circuit alone)
- **Modularity** (degree to which circuit components are behavior-specific vs. shared)

## Results

**SFT consistently condenses circuits:**
- Average circuit size decreases by 31% after SFT across all 8 behaviors
- Edge faithfulness increases from 0.74 to 0.89 on average (circuits explain more of the behavior post-SFT)
- Modularity increases: components become more behavior-specific

**RL fine-tuning: behavior-dependent effects:**
- Factual recall circuits condense under RL (similar to SFT): size −28%, faithfulness +0.11
- Multi-step reasoning circuits expand under RL: size +47%, faithfulness −0.06
- Logical reasoning shows mixed results depending on problem structure

**Interpretation:** SFT concentrates computation by reinforcing specific activation patterns that were already partially present in the base model. RL, because it involves exploration and credit assignment over multiple steps, may recruit additional circuit components to support improved long-range credit propagation, particularly for multi-step behaviors.

**Ablation:** The condensation effect under SFT is robust to learning rate and dataset size choices, suggesting it is a fundamental consequence of behavioral supervision rather than a tuning artifact.

## Key Contributions

1. First systematic measurement of how post-training changes the causal circuit structure of LLMs across a diverse suite of behaviors
2. Identification of a divergent pattern: SFT universally condenses circuits; RL condenses simple circuits but expands complex ones
3. Practical guidance for interpretability: SFT-fine-tuned models are more amenable to automated circuit analysis and targeted editing
4. A methodology (pre/post circuit comparison using ACDC+EAP) that can be applied to evaluate any future post-training technique for interpretability impact

## Significance

As the field moves toward post-training as the primary mechanism for capability improvement, understanding its effect on internal circuit structure is essential for scalable interpretability. Circuit Condensation provides the first data-driven answer: SFT is interpretability-friendly, while RL's interpretability impact depends critically on the type of behavior being trained. For practitioners building interpretable models, this suggests that the choice between SFT and RL is not merely a performance consideration — it shapes the internal organization of the model in ways that determine how easily it can be understood and modified.
