# Can LLMs Uncover Hidden Physical Laws through Free Exploration?

**arXiv:** 2610.37649  
**Authors:** Zhen Yang, Wen Wang, Xander Xu, Ying-Cong Chen (HKUST(GZ), Alibaba, ZJU)  
**Submitted:** October 27, 2026  
**Venue:** NeurIPS 2026 Workshop on Machine Learning and the Physical Sciences (ML4PS)  
**URL:** https://arxiv.org/abs/2610.37649

---

## Summary

Scientific discovery requires more than answering questions about known phenomena—it demands noticing anomalies, designing experiments to investigate them, and proposing new laws from the evidence. This paper asks whether current large language models can perform open-ended physical discovery: placed in a world where some physical constants have been silently altered, can an LLM explore freely, detect the deviation, and formulate the hidden law without being told what to look for?

The authors introduce **LongChang**, an automated evaluation framework built around MuJoCo simulations in which one or more physical laws are modified. An LLM agent is given access to tools for running experiments, measuring outcomes, and building models, but receives no hint about which laws have changed. The task requires sustained hypothesis formation, experimental design, and law articulation—capabilities that go well beyond standard reasoning benchmarks.

---

## Method

**LongChang Framework.** A MuJoCo world is instantiated with altered physics (e.g., modified gravity, changed friction coefficient, non-standard elastic constants). The LLM is given a natural-language description of the world and a set of Python tools it can invoke to run simulations and measure physical quantities.

**Free Exploration Protocol.** Unlike prior equation-discovery benchmarks that provide labeled data, LongChang imposes no exploration curriculum. The agent must decide what to measure, design measurement protocols, interpret noisy results, and revise hypotheses iteratively. The agent's exploration is bounded by a token budget.

**Evaluation.** A discovery is scored successful if the agent outputs a formal statement of the hidden law that is numerically correct within a tolerance. Partial credit is given for correct anomaly detection even when the law formulation is incomplete.

**LLMs Evaluated.** Twelve frontier models are tested: GPT-6-Astra, GPT-5, Claude Sonnet 5, Claude Opus 5, Gemini 2.5 Ultra, Gemini 2.5 Pro, Llama 4, and five others.

---

## Results

| Model | Discovery Rate |
|-------|---------------|
| GPT-6-Astra | **16.7%** |
| GPT-5 | 0% |
| Claude Sonnet 5 | 0% |
| Claude Opus 5 | 0% |
| Gemini 2.5 Ultra | 0% |
| All others | 0% |

- **Only GPT-6-Astra** achieves any successful discoveries (1 out of 6 trials).
- All other models score 0%, including very strong closed-source models.
- Qualitative analysis shows models frequently **detect anomalies** but fail to systematically design experiments to distinguish candidate explanations.
- Models often converge prematurely on an incorrect hypothesis after the first unexpected observation, failing to update appropriately.

---

## Significance

The results establish a stark benchmark: even the most capable LLMs in late 2026 largely fail at open-ended physical discovery. This is a meaningful failure mode to document, because it differs from the well-studied pattern of LLMs failing on novel math problems—here, the difficulty lies in the agentic loop of designing experiments, accumulating evidence, and revising hypotheses, not in any single reasoning step. The LongChang framework provides a rigorous, automatically generated test bed for this capability.

---

## Limitations

- Only six trial environments per model due to compute cost; statistical power is limited.
- MuJoCo worlds are stylized simulations; real physical discovery involves much messier data.
- GPT-6-Astra's success may partly reflect training data contamination with similar frameworks.
