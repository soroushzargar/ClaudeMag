# UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation

**arXiv:** 2609.12397  
**Submitted:** September 11, 2026  
**Authors:** Danning Zhang, Yijing Lin, Shuhan Zhuang, Mengqi Huang, Shaojin Wu, Shancheng Fang, Zhendong Mao  
**Affiliation:** University of Science and Technology of China  
**Venue:** ICML 2026 (Forty-Third International Conference on Machine Learning)

---

## Headline Finding

UFO introduces an Atomized Chain-of-Evaluation paradigm that evaluates multi-modal image generation by decomposing omni-condition alignment into a sequential chain of fine-grained Atomic Evaluation Units (AEUs) — one per modality-condition combination — and calling purpose-built verifier functions for each, achieving Spearman rank correlation ρ=0.6889 with human judgment, a 15.25% improvement over the best prior metric, on the new UFO-Bench (660 cases, 7 categories, 3 difficulty tiers).

---

## Key Findings (Pyramid Layer 2)

1. **Multi-modal generation is conditioned on simultaneous inputs, but existing metrics evaluate each condition independently.** Modern customization models (e.g., subject-driven models) take text prompts, reference images, style conditions, layout specifications, and identity references simultaneously. Metrics such as CLIP-T and DINO-I score text alignment and subject identity separately and then aggregate, losing the interactions and conflicts between conditions that drive the hardest evaluation cases.

2. **Independent evaluation produces poor correlation with human preference.** When conditions conflict (e.g., "make the dog pose like a cat" with a reference dog image), humans resolve the conflict in nuanced, condition-priority ways. Per-condition metrics sum orthogonal scores and cannot detect whether the conflict was resolved appropriately.

3. **Atomic Evaluation Units decompose the alignment problem structurally.** An AEU is a verifiable claim about one condition-dimension pair — for example, "the text prompt element 'red collar' appears correctly" or "the reference subject's identity is preserved." AEUs are categorized by modality (text, visual, geometric) and relevance class (mandatory, preferred, contextual), and ordered into a chain that processes mandatory conditions first.

4. **Dedicated functional verifiers outperform generic MLLM scoring.** For geometric conditions (layout, pose), UFO calls a structured detector (e.g., DWPose for body pose, grounded SAM for object placement) rather than asking an MLLM to assess geometry holistically. For identity conditions, a face/instance recognition model is called. Generic MLLM scoring is reserved for semantic, compositional, and style conditions where structured verifiers do not apply.

---

## Methodology (Pyramid Layer 3)

### Omni-Condition Alignment Problem

Let a generation sample $x$ be conditioned on $C = \{c_1, \ldots, c_K\}$ where each $c_k$ is a condition (text prompt $c_T$, reference image $c_I$, layout map $c_L$, style image $c_S$, etc.). Define the omni-condition alignment score as:
$$\text{OCA}(x, C) \in [0, 1]$$
reflecting how well $x$ simultaneously satisfies all conditions in $C$.

Prior metrics compute:
$$\hat{\text{OCA}}_{\text{prior}}(x, C) = \frac{1}{K}\sum_k w_k \cdot s_k(x, c_k)$$
where $s_k$ is a condition-specific score. This ignores condition interactions.

### Atomized Chain-of-Evaluation

UFO decomposes each condition $c_k$ into a set of AEUs $\{u_{k,1}, \ldots, u_{k,n_k}\}$. Each AEU is a binary claim:
$$u_{k,j} : \text{"attribute } a_{k,j} \text{ of condition } c_k \text{ is satisfied in } x$$

The full set of AEUs across all conditions is ordered into a chain $\mathcal{U} = [u_{1,1}, u_{1,2}, \ldots, u_{K,n_K}]$ where mandatory conditions are processed first.

For each AEU $u_{k,j}$, UFO calls an appropriate verifier $V_{k,j}$:

- **Text conditions:** MLLM with targeted prompts ("Does the image contain a [attribute]?")
- **Identity conditions:** Face recognition model (ArcFace) or instance retrieval (DINOv2 cosine similarity) depending on subject type
- **Geometric conditions:** Structured detectors (DWPose for poses, grounded-SAM for object placement, DepthAnything for depth conditions)
- **Style conditions:** CLIP style embedding similarity with reference style image

The chain evaluates each AEU sequentially; the final score is a weighted aggregation:
$$\text{OCA}_{\text{UFO}}(x, C) = \sum_{k,j} r_{k,j} \cdot V_{k,j}(x, c_k, a_{k,j})$$
where $r_{k,j}$ is the relevance weight of AEU $u_{k,j}$ (mandatory: 1.0, preferred: 0.7, contextual: 0.3).

### UFO-Bench Design

UFO-Bench provides 660 test cases across:
- **7 categories:** Subject customization, style transfer, layout control, pose control, identity preservation, multi-subject composition, conflicting condition resolution
- **3 difficulty tiers:** Easy (single dominant condition), medium (2–3 non-conflicting conditions), hard (conflicting conditions requiring human-aligned priority resolution)
- **Human preference annotations:** 5 annotators per sample with inter-annotator agreement Krippendorff α = 0.73

---

## Technical Formulation

The AEU decomposition for a text prompt $c_T$ with $P$ phrases operates as:

$$\{u_{T,p}\}_{p=1}^P = \text{Parse}(c_T), \quad u_{T,p} = (\text{phrase}_p, \text{attribute type}_p, \text{relevance class}_p)$$

The chain ordering uses a topological sort over a condition dependency graph $G_C$ where edges encode logical dependencies (e.g., subject identity must be checked before subject pose, since pose verification requires subject localization).

The overall Spearman rank correlation evaluated by UFO-Bench:
$$\rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2-1)}$$
where $d_i$ is the rank difference between UFO scores and human preference rankings across the $n=660$ test cases.

---

## Learning or Inference Procedure

UFO is a training-free evaluation framework. At inference:
1. Parse all conditions $C$ into AEUs and build the dependency graph.
2. Topologically sort AEUs to form the evaluation chain.
3. For each AEU $u_{k,j}$ in order, call the appropriate verifier $V_{k,j}$.
4. Aggregate weighted verifier outputs into $\text{OCA}_{\text{UFO}}(x, C)$.

The framework is modular: new condition types can be added by registering new AEU parsers and verifier functions, without retraining any component.

---

## What the Guarantee Says

UFO does not make formal approximation guarantees. The paper's theoretical contribution is a decomposition lemma showing that under independence assumptions between conditions, the chain-of-evaluation score is an unbiased estimator of the joint omni-condition alignment score. For dependent conditions (the harder, more realistic case), UFO's sequential processing with mandatory-first ordering is shown to reduce the expected correlation error by prioritizing the conditions humans weight most heavily.

---

## Experimental Findings

### Evaluated Metrics
CLIP-T (text-image similarity), DINO-I (subject identity), IP-Adapter score, Human Preference Score v2 (HPS-v2), ImageReward, Pick-Score, VIEScore, and UFO.

### Correlation with Human Rankings on UFO-Bench

| Metric | Spearman ρ | Pearson r | Kendall τ |
|--------|-----------|-----------|----------|
| CLIP-T | 0.421 | 0.398 | 0.312 |
| DINO-I | 0.389 | 0.374 | 0.288 |
| HPS-v2 | 0.521 | 0.508 | 0.401 |
| VIEScore | 0.598 | 0.581 | 0.463 |
| **UFO** | **0.689** | **0.672** | **0.537** |

UFO achieves the highest correlation on all three metrics, with the largest improvements on the hard tier (ρ=0.701 vs. 0.489 for VIEScore), confirming that the chain-of-evaluation paradigm most benefits the conflicting-condition cases where existing metrics collapse.

---

## Ablations and Interpretation

- **Replacing structured verifiers with MLLM for all AEUs:** ρ drops from 0.689 to 0.631; geometric and identity conditions are specifically hurt.
- **Removing chain ordering (evaluating AEUs in random order):** ρ drops to 0.657; mandatory-first ordering is important for resolving conflicting conditions.
- **Using uniform relevance weights (all AEUs weight 1.0):** ρ drops to 0.672; graded weights improve alignment on medium and hard tiers.
- **AEU granularity:** Phrase-level AEUs outperform sentence-level AEUs by 4.2 ρ points; sub-phrase (word-level) AEUs add only 0.3 ρ points with 2× compute cost.
- **Performance by tier:** Easy: ρ=0.741; Medium: ρ=0.693; Hard: ρ=0.701 — UFO is most reliable at the hardest tier, confirming that the chain paradigm uniquely handles condition conflicts.

---

## Reference

Danning Zhang, Yijing Lin, Shuhan Zhuang, Mengqi Huang, Shaojin Wu, Shancheng Fang, and Zhendong Mao. **UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation.** arXiv:2609.12397, September 2026. https://arxiv.org/abs/2609.12397. Accepted at ICML 2026.
