# CausalArena: Benchmarking Causal Discovery in the Foundation Model Era

**arXiv:** 2609.11897  
**Submitted:** September 10, 2026  
**Authors:** Zi-Rong Li, Si-Yang Liu, Tian-Zuo Wang, Han-Jia Ye  
**Affiliation:** Nanjing University  
**Venue:** arXiv preprint

---

## Headline Finding

CausalArena establishes that existing fixed synthetic benchmarks for causal discovery are unreliable in the era of causal discovery foundation models (CDFMs) because performance may reflect pretraining environment overlap rather than genuine causal reasoning ability; its unified, evolvable benchmark under a common protocol separately evaluates classical algorithms, CDFMs, and LLM-based causal agents, enabling for the first time meaningful cross-paradigm comparisons that control for memorization artifacts.

---

## Key Findings (Pyramid Layer 2)

1. **CDFMs dramatically overfit to their pretraining SCMs.** When CDFMs are evaluated on structural causal models (SCMs) that appear in or closely resemble their pretraining environments, they score 15–28 points higher on SHD (Structural Hamming Distance) compared to held-out, procedurally generated SCMs — demonstrating that existing fixed benchmarks inflate CDFM performance estimates.
2. **Classical algorithms remain competitive on genuinely novel SCMs.** On CausalArena's held-out evaluation instances, classical constraint-based (PC, FCI) and score-based (GES, NOTEARS) algorithms match or exceed CDFMs on most graph families, reversing the apparent superiority CDFMs show on fixed benchmarks.
3. **LLM-based causal agents excel on domain-specific known graphs.** When variable names carry semantic meaning (medical, economic datasets), LLM agents with access to background knowledge outperform both classical algorithms and CDFMs by 8–12 points, but degrade severely on anonymized variables where their linguistic prior is neutralized.
4. **A common evaluation protocol reveals complementary strengths.** No single paradigm dominates across all evaluation axes; the choice of algorithm depends critically on whether the application provides domain knowledge, has abundant i.i.d. data, or requires robustness to novel SCM families.

---

## Methodology (Pyramid Layer 3)

### Background: Causal Discovery and Its Evaluation Problem

Causal discovery is the task of inferring a directed acyclic graph (DAG) $\mathcal{G} = (V, E)$ from observational (or interventional) data $\mathbf{X} \sim P_\mathcal{G}$, where each directed edge $X_i \to X_j$ encodes a direct causal influence.

**Classical algorithms** operate directly on data:
- **Constraint-based** (PC, FCI): Test conditional independences to orient edges.
- **Score-based** (GES, NOTEARS, DAG-GNN): Optimize a structural score over the DAG space.

**Causal discovery foundation models (CDFMs)** are pretrained on large collections of SCM-generated datasets and learn to predict the causal graph in-context (e.g., CDFL, DiffAN, CausalFormer).

**LLM-based causal agents** use large language models with semantic knowledge of variable names to reason about causal relations, sometimes augmented by statistical tests (e.g., GPT-4-Causal, CausalGPT-v2).

**The evaluation problem:** Each paradigm has been evaluated on different benchmark sets with different graph families, mechanisms, and metric definitions, making direct comparison impossible. CDFMs in particular achieve high scores partly because their pretraining SCMs overlap with test SCMs in existing fixed benchmarks (e.g., ER2, SF2 graphs with linear Gaussian mechanisms).

### CausalArena Design

CausalArena has three components:

**1. The SCM Generator.** A procedural generator that can sample novel SCMs outside any existing CDFM pretraining corpus. It supports:
- Graph families: Erdős–Rényi ($k$-connected), scale-free, bipartite layered, and a new "causal motif" family that mixes subgraph patterns.
- Mechanisms: Linear Gaussian, ANM (additive noise model), post-nonlinear, and discretized multinomial.
- Variable semantics: Fully anonymized vs. semantically named (using curated domain ontologies).
- Graph sizes: 5–50 nodes.

**2. The Common Protocol.** CausalArena enforces a standardized evaluation protocol:
- **Metric:** SHD (Structural Hamming Distance), F1 on directed edges, and a new **Causal Effect Recovery (CER)** metric measuring the accuracy of pairwise causal effect estimates derived from the predicted DAG.
- **Data regimes:** Small ($n=500$), medium ($n=5{,}000$), large ($n=50{,}000$) sample sizes.
- **Knowledge levels:** Zero-knowledge (no variable names), partial (variable names only), full (names + background knowledge document).

**3. The Evolvability Mechanism.** CausalArena's generator is seeded by a public random key; the test set can be refreshed monthly to prevent CDFMs from re-memorizing evaluation SCMs. A version-controlled leaderboard tracks performance over time, distinguishing progress on novel instances from overfitting to the current test set.

---

## Technical Formulation

### Structural Causal Model

A structural causal model is $\mathcal{M} = (\mathcal{G}, \mathbf{F}, P_\mathbf{U})$ where $\mathcal{G} = (V, E)$ is a DAG, $\mathbf{F} = \{f_i\}$ are structural equations $X_i = f_i(\mathbf{Pa}_i, U_i)$, and $P_\mathbf{U}$ is the noise distribution.

**SHD** between predicted $\hat{\mathcal{G}}$ and true $\mathcal{G}$:
$$\mathrm{SHD}(\hat{\mathcal{G}}, \mathcal{G}) = |\mathrm{missing}| + |\mathrm{extra}| + |\mathrm{reversed}|$$

**Causal Effect Recovery (CER)**, a new metric introduced in CausalArena:
$$\mathrm{CER} = 1 - \frac{1}{|P|} \sum_{(i,j) \in P} \left| \hat{\tau}_{ij} - \tau_{ij} \right| / |\tau_{ij}|$$
where $P$ is the set of all variable pairs, $\tau_{ij}$ is the true total causal effect of $X_i$ on $X_j$ (via $do$-calculus), and $\hat{\tau}_{ij}$ is estimated from the predicted DAG via linear structural equation fitting.

### Memorization Detection

To test whether a CDFM has memorized a test SCM class, CausalArena includes a held-out "canary" SCM family: a novel graph structure not in any published CDFM pretraining curriculum, constructed by the authors. A model that genuinely learns causal discovery should perform consistently on the canary family; a model that memorizes should perform worse.

**Memorization gap** for a CDFM $M$:
$$\Delta_\mathrm{mem} = \mathrm{SHD}(M, \mathcal{G}_{\text{canonical}}) - \mathrm{SHD}(M, \mathcal{G}_{\text{canary}})$$

All three CDFMs tested show $\Delta_\mathrm{mem} > 5$ SHD points (p < 0.001), confirming memorization is occurring.

---

## Experiments

### Algorithms Evaluated
- **Classical:** PC, FCI, GES, NOTEARS, DAG-GNN (5 algorithms)
- **CDFMs:** CausalFormer, DiffAN, CDFL (3 foundation models)
- **LLM agents:** GPT-4o-Causal, Claude-3.5-Causal, CausalGPT-v2 (3 agents)

### Results on Novel SCMs (n=5,000, anonymized variables)

| Algorithm | SHD ↓ | F1 ↑ | CER ↑ |
|-----------|-------|------|-------|
| PC | 8.4 | 0.61 | 0.71 |
| GES | 7.9 | 0.64 | 0.74 |
| NOTEARS | 9.1 | 0.59 | 0.68 |
| CausalFormer | 9.8 | 0.57 | 0.65 |
| DiffAN | 10.3 | 0.54 | 0.63 |
| GPT-4o-Causal | 18.7 | 0.31 | 0.41 |

On novel, anonymized SCMs, classical algorithms outperform CDFMs and LLM agents substantially.

### Results on Semantic Variables (n=5,000, named)

| Algorithm | SHD ↓ | F1 ↑ | CER ↑ |
|-----------|-------|------|-------|
| GES | 8.1 | 0.63 | 0.73 |
| CausalFormer | 10.1 | 0.56 | 0.64 |
| **GPT-4o-Causal** | **6.2** | **0.74** | **0.82** |

LLM agents dominate when variable names carry domain-specific semantics.

---

## Ablations and Interpretation

- **SCM size:** All algorithms degrade with larger graphs; classical algorithms degrade more gracefully than CDFMs at $n > 20$ nodes.
- **Sample size:** CDFMs show flat performance across $n \in \{500, 5000, 50000\}$, suggesting they rely on learned priors over structure rather than statistical tests; classical algorithms improve monotonically with $n$.
- **Mechanism complexity:** Post-nonlinear mechanisms are hardest for all methods; CDFMs that pretrained on linear Gaussian data show a 12-point SHD gap between linear and nonlinear test instances.
- **Canary family:** CDFMs score 8–14 SHD points worse on the canary family vs. canonical families, confirming memorization is driving published benchmark numbers.

---

## Reference

Zi-Rong Li, Si-Yang Liu, Tian-Zuo Wang, and Han-Jia Ye. **CausalArena: Benchmarking Causal Discovery in the Foundation Model Era.** arXiv:2609.11897, September 2026. https://arxiv.org/abs/2609.11897
