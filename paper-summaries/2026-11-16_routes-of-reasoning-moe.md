# Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models

**Authors:** Shengjie Wang, Mohsen Fayyaz, Nanyun Peng  
**arXiv:** 2611.05919 (cs.CL, cs.AI)  
**Submitted:** 5 November 2026  
**Venue:** arXiv preprint  

---

## Headline

Mixture-of-Experts routing signals reveal and control *how* a language model reasons—not just *how much* it reasons—enabling training-free compute steering at the behavior level.

---

## Key Findings

- Reasoning traces can be decomposed into six behavioral segments: **planning**, **execution**, **restatement**, **fact-retrieval**, **reflection**, and **conclusion**.
- Each behavior is associated with a characteristic expert routing signature across MoE layers; these signatures generalize across six datasets spanning five domains.
- Using behavior-identified expert sets to steer routing at inference time reduces over-verification (wasteful re-checking) without degrading final-answer accuracy.
- Four open-weight MoE reasoning models were evaluated: Gemma-4, Qwen3 (two generations), and GPT-OSS.

---

## Motivation and Problem

Large reasoning models (LRMs) routinely generate thousands of tokens at test time before committing to an answer. Studies have shown that after a correct answer is first derived, the model often continues verifying and re-deriving—spending significant compute that provides little benefit. Prior interventions (e.g., token budgets, length penalties, steering vectors) operate on reasoning *duration* but are agnostic to reasoning *type*.

The key question the authors pose: can we identify which portion of reasoning is behaviorally wasteful without retraining the model?

---

## Methodology

### Behavioral Taxonomy

The authors define a six-class behavioral taxonomy for reasoning segments at the sentence level:

| Behavior | Description |
|---|---|
| **Planning** | Outlines steps or sets up a strategy |
| **Execution** | Applies a procedure or computation |
| **Restatement** | Repeats information already stated |
| **Fact** | Retrieves a known fact |
| **Reflection** | Checks or reconsiders a previous step |
| **Conclusion** | Commits to a final answer |

Each sentence in a reasoning trace is labeled by a lightweight classifier trained on a small annotated development set.

### Expert Routing as a Behavior Probe

In an MoE model, each token is routed to a subset of experts at each layer. The core insight is that tokens belonging to different behavioral segments activate different expert subsets in a *consistent* and *generalizable* way. The authors compute per-behavior expert affinity scores across layers and show that the top-k experts for a given behavior on one dataset predict the same behavior on held-out datasets.

### Steering Procedure

At inference time:
1. Identify the expert sets associated with each behavior on a small development split.
2. During generation, monitor which behavior is currently being produced (via a lightweight probe on the last-layer hidden state).
3. When an over-verification pattern is detected (reflection triggered after a conclusion), slightly suppress the routing weights of reflection-associated experts.

This is entirely training-free—no gradient updates are needed.

---

## Technical Details

Let $\mathbf{r}^{(l)}_t \in \{0,1\}^E$ denote the routing decision for token $t$ at layer $l$ (where $E$ is the number of experts). The behavioral affinity of expert $e$ for behavior class $b$ is:

$$A(e, b) = \frac{1}{|S_b|} \sum_{t \in S_b} r^{(l)}_{t,e}$$

where $S_b$ is the set of tokens labeled with behavior $b$. The steering intervention scales expert logits:

$$\tilde{g}_{t,e} = g_{t,e} \cdot \bigl(1 - \alpha \cdot \mathbf{1}[A(e, b_t) > \tau]\bigr)$$

where $\alpha$ is a suppression strength hyperparameter and $\tau$ is a threshold identifying "behavior-dominant" experts.

---

## Experimental Results

- **Generalization of routing signatures:** Expert affinity vectors for behavior classes achieve >85% cosine similarity across the six evaluation datasets, confirming that routing-based behavior detection generalizes without per-dataset calibration.
- **Efficiency gains:** Steering reduces average reasoning trace length by 18–27% across tasks (by cutting re-verification loops) while maintaining answer accuracy within 0.3% of the unsteered baseline.
- **Model coverage:** Results are consistent across all four MoE models tested, with the strongest efficiency improvements on Qwen3-30B-A3B (the largest model evaluated).

---

## Significance

This work opens a new dimension in test-time compute control: rather than asking *how long* a model should think, it asks *what kind* of thinking is being wasted. The routing-based approach requires no model modifications or fine-tuning, making it immediately applicable to any deployed MoE reasoning system. The behavioral taxonomy is also a step toward interpretable reasoning analysis beyond token-count proxies.
