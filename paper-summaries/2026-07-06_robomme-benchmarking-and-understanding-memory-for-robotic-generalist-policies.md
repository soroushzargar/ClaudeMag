# RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies

### A Systematic Taxonomy of Memory in VLA Models Reveals Current Approaches Fall Far Short

**Authors:** Yinpei Dai, Hongze Fu, Jayjun Lee, Yuejiang Liu, Haoran Zhang, Jianing Yang, Chelsea Finn, Nima Fazeli, Joyce Chai
**Venue:** ICML 2026 Spotlight Presentation
**arXiv:** [2603.04639](https://arxiv.org/abs/2603.04639)
**Submitted:** March 2026

---

## Layer 1 — The Big Picture

**What problem does the paper solve, and how does it solve it?**

Vision-language-action (VLA) models — which extend multimodal LLMs with action outputs for robotic manipulation — have shown impressive generalization. But almost all evaluations focus on short-horizon, memoryless tasks where each timestep can be solved from the current observation alone. Real-world manipulation often demands **memory**: remembering which objects have already been placed, keeping track of a sequence of subgoals, handling occluded objects, and counting repeated actions.

This paper introduces **RoboMME**, the first large-scale standardized benchmark explicitly designed to evaluate and compare memory mechanisms in robotic generalist policies. The benchmark provides:
- 16 manipulation tasks covering four memory categories
- A systematic study of 14 memory-augmented VLA variants on the π0.5 backbone
- A clear characterization of which memory mechanisms work best for which task types

---

## Layer 2 — Under the Hood

**Memory taxonomy.**

RoboMME classifies memory requirements along four axes:

1. **Temporal memory**: Tasks requiring the agent to track a sequence of subgoals over time (e.g., place objects in a specific order, repeat an action a given number of times).
2. **Spatial memory**: Tasks where objects become temporarily occluded but must later be retrieved from their remembered location.
3. **Object memory**: Tasks requiring identification and tracking of specific objects across appearances.
4. **Procedural memory**: Tasks requiring multi-step procedures executed from instruction, not all visible in the current observation.

**Memory mechanism variants.**

The 14 VLA variants differ along two axes: *memory representation* (frame sampling, recurrent states, external memory buffers) and *integration strategy* (early fusion, modulation, late fusion). Key variants:
- **FrameSamp**: Select and concatenate $k$ historical frames
- **FrameSamp+Modul**: Frame sampling with modulation of the VLM trunk
- **Recurrent**: Hidden state carries history across timesteps
- **ExternalMem**: Explicit key-value memory module updated each step

---

## Layer 3 — The Math

**Evaluation metric.** Each task $\tau$ is scored by the **success rate** $S_\tau \in [0,1]$ averaged over $N_\tau$ episodes. The memory score for category $c$ is:

$$M_c = \frac{1}{|T_c|} \sum_{\tau \in T_c} S_\tau$$

**Memory dependence test.** For each task, the authors compute the performance gap between a memory-enabled agent and a Markov agent (history length 1):

$$\Delta_\tau = S_\tau(\text{memory}) - S_\tau(\text{Markov})$$

Tasks with $\Delta_\tau > \epsilon$ are classified as memory-dependent. Across 16 tasks, 12 show significant memory dependence ($\Delta_\tau > 0.1$).

**Key theoretical observation.** For procedural memory tasks, the gap between the best memory-augmented VLA and an oracle with full procedure recall is:

$$\Delta_\text{oracle} \approx 0.45$$

suggesting that current memory mechanisms capture at best half the available signal from history — a large gap relative to oracle performance.

---

## Experiments & Datasets

**Benchmark:** 16 tasks across temporal, spatial, object, and procedural memory. Tasks are built on the LIBERO and LIBERO-Plus simulation environments with procedurally generated variants.

**Backbone:** π0.5 VLA, with 14 memory mechanism variants.

**Key results:**
- FrameSamp+Modul achieves the best aggregate performance, particularly for motion-centric and time-sensitive tasks
- Recurrent mechanisms excel at procedural tasks but degrade on temporal counting tasks
- ExternalMem provides marginal gains over FrameSamp on most tasks, suggesting naive external memory is not yet a silver bullet
- All variants fail significantly on tasks requiring 5+ step procedural sequences

**Conclusion:** Current VLA memory mechanisms are far from solving long-horizon manipulation, motivating future work on more capable episodic memory architectures.
