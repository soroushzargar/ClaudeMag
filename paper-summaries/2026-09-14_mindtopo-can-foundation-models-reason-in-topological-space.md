# MindTopo: Can Foundation Models Reason in Topological Space?

**arXiv:** 2609.11900  
**Submitted:** September 10, 2026  
**Authors:** Yunfei Ge, Anbang Liu, Qineng Wang, Johnalbert Garnica, Jianwen Lyu, Zihan Wang, Reuben Tan, Jianfeng Gao, Ruohan Zhang, Yining Hong, Jiajun Wu, Manling Li  
**Affiliation:** Northwestern University; Microsoft Research; Stanford University; and collaborating institutions  
**Venue:** arXiv preprint

---

## Headline Finding

MindTopo, a benchmark of 11,030 procedurally generated instances spanning five topological properties and two cognitive levels, reveals that the best multimodal foundation model scores only 61.4% on topological reasoning against 97.9% human performance — demonstrating a systematic blind spot in spatial understanding that cannot be reduced to metric reasoning or viewpoint sensitivity.

---

## Key Findings (Pyramid Layer 2)

1. **Every tested model performs better on reasoning than on planning.** Identifying topological relations from visual input (reasoning) is easier for models than using topological understanding to select actions in a closed-loop environment (planning), suggesting that models may pattern-match topological cues without grounding them in action-relevant representations.
2. **Knots are universally the hardest property.** All 14 multimodal LLMs and 3 video generative models score near chance on knot-type topology, which requires reasoning about global path structure — a property absent from standard metric or Euclidean geometry.
3. **Best model: 61.4%; human: 97.9%.** The gap of 36.5 percentage points, observed on the same instances, is far larger than gaps typically seen on metric spatial benchmarks (5–15 points), suggesting topology is qualitatively different from the spatial reasoning already captured by current training distributions.
4. **Difficulty is controllable and gradable.** The benchmark uses procedural generation with a difficulty parameter $d \in [1, 5]$ controlling topological complexity; model accuracy decreases monotonically with $d$, while human accuracy remains above 95% even at $d=5$.

---

## Methodology (Pyramid Layer 3)

### Background: Topology vs. Metric Geometry

Topology studies properties invariant under continuous deformation (homeomorphisms), as opposed to geometry, which studies properties invariant under rigid motion (isometries). The key topological properties evaluated in MindTopo are grounded in cognitive developmental science:

- **Continuity:** Whether a path or region is connected without breaks.
- **Separation:** Whether two regions or paths are disconnected.
- **Order:** Cyclic ordering of points along a path or on a surface.
- **Enclosure:** Whether one region is inside, outside, or surrounds another.
- **Knots:** Global entanglement of closed curves in 3D — not reducible to any local or metric property.

Cognitive science (Piaget 1956, Spelke et al. 1995) identifies topological relations as the earliest spatial concepts acquired by infants, preceding metric and projective relations. Yet foundation model benchmarks focus almost exclusively on metric properties.

### MindTopo Benchmark Design

MindTopo contains **11,030 instances** across **13 task types** at **two cognitive levels**:

**Reasoning level:** Given visual input (image or short video), identify a topological property or predict how it changes under a described deformation.
- Example: "Is the blue curve knotted? (Yes/No)"
- Example: "After applying the deformation shown, does the red region still enclose the dot?"

**Planning level:** The model acts as a closed-loop agent with a policy $\pi: O_t \to A_t$ mapping observations to actions in a procedurally generated topological environment.
- Example: Navigate a maze where the solution requires identifying a topological shortcut not apparent from local geometry.
- Example: Untie a rope by selecting a sequence of over/under crossings.

**Procedural generation:** All instances are generated algorithmically using a custom topology simulator, ensuring no overlap with web-scraped training data. Difficulty $d \in [1, 5]$ controls the number of topological operations required.

### Evaluation Protocol

For **reasoning** tasks: standard multiple-choice accuracy with 4 options.

For **planning** tasks: task completion rate within a budget of $T = 20$ steps. A partial credit score counts steps toward the correct topological outcome.

Human baseline: 50 participants (college students with no topology training) evaluated a stratified 500-instance subset.

---

## Technical Formulation

### Topological Properties as Invariants

Let $f: X \to Y$ be a continuous map between topological spaces. MindTopo operationalizes five invariant types:

**Continuity:** A path $\gamma: [0,1] \to X$ is continuous if $f \circ \gamma$ is continuous for any evaluation function $f$. Tasks test whether visual breaks (gaps, intersections) are present.

**Separation:** Two sets $A, B \subset X$ are separated if $\bar{A} \cap B = A \cap \bar{B} = \emptyset$. Tasks test whether two visual regions can be disconnected by a cut.

**Order:** Given $n$ points on a closed curve, their cyclic order is invariant under homeomorphism. Tasks ask models to identify or predict cyclic orderings after deformations.

**Enclosure:** Region $A$ encloses $B$ if $B \subset \mathrm{Int}(A)$ in the ambient space. Tasks include Jordan-curve-type problems.

**Knot type:** A closed curve $\gamma: S^1 \to \mathbb{R}^3$ belongs to a knot class $[K]$ if it is ambient isotopic to $K$. MindTopo uses three-crossing and four-crossing knots as the test cases, asking models to identify the knot type from visual input or determine whether two curves are equivalent.

The Reidemeister theorem guarantees that two knot diagrams represent equivalent knots iff one can be obtained from the other by a sequence of three local moves. Reasoning about knots thus requires non-local reasoning that cannot be decomposed into local feature matching.

---

## Experimental Findings

### Models Evaluated
- **Multimodal LLMs (14):** GPT-4V, GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, LLaVA-1.6-34B, InternVL2-76B, Qwen2-VL-72B, and 7 others.
- **Video generative models (3):** Sora, Kling, Gen-3.

### Aggregate Results

| Model | Reasoning | Planning | Overall |
|-------|-----------|----------|---------|
| Human | 98.2% | 97.5% | 97.9% |
| Best model (Gemini 1.5 Pro) | 68.3% | 54.5% | 61.4% |
| GPT-4o | 65.1% | 48.9% | 57.0% |
| Claude 3.5 Sonnet | 63.7% | 46.2% | 54.9% |
| Average across 14 MLLMs | 52.4% | 38.7% | 45.6% |
| Random baseline | 25.0% | 5.0% | — |

### Results by Topological Property

| Property | Best Model | Human |
|----------|-----------|-------|
| Continuity | 79.2% | 98.8% |
| Separation | 74.1% | 98.5% |
| Order | 65.8% | 97.4% |
| Enclosure | 61.3% | 97.1% |
| Knots | 29.4% | 97.2% |

Knot reasoning is near-random for all models, while continuity and separation are relatively easier — consistent with the hypothesis that models recognize local topological features but not global topological invariants.

---

## Ablations and Interpretation

- **Metric cues removed:** When rendered images are topologically equivalent but metrically different (e.g., differently scaled loops), model performance changes by less than 2%, confirming models do not exploit metric shortcuts.
- **Language-only baseline:** Providing text descriptions of topological configurations without images yields 38% accuracy, indicating models do have some linguistic topology priors but that visual grounding adds meaningful signal (52% with images).
- **Chain-of-thought prompting:** Adding CoT improves average accuracy by 4.1 points (reasoning) and 3.8 points (planning), suggesting explicit intermediate steps help but are insufficient to close the human gap.
- **Difficulty scaling:** Model accuracy at difficulty $d=5$ is 31.2% (near random) while humans score 95.4%, indicating that complex topological configurations are not merely harder but qualitatively out of distribution.

---

## Reference

Yunfei Ge, Anbang Liu, Qineng Wang, Johnalbert Garnica, Jianwen Lyu, Zihan Wang, Reuben Tan, Jianfeng Gao, Ruohan Zhang, Yining Hong, Jiajun Wu, and Manling Li. **MindTopo: Can Foundation Models Reason in Topological Space?** arXiv:2609.11900, September 2026. https://arxiv.org/abs/2609.11900
