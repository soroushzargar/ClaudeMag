# Binding Mode Matters: Hotspot-Aware Drug Discovery via Explorative Preferences

**Authors:** Dingshuo Chen, Qian Zhao, Xiangxiang Zeng, et al.  
**arXiv:** 2611.14956 (cs.LG, q-bio.BM)  
**Submitted:** 11 November 2026  
**Venue:** NeurIPS 2026  

---

## Headline

Decomposing the binding pocket into hotspot clusters and assigning separate RL rewards per hotspot steers molecular generation toward structurally diverse, high-affinity compounds that existing single-score methods systematically miss.

---

## Key Findings

- **BindMol** frames structure-based drug design as a multi-objective RL problem; each reward component targets a distinct binding hotspot within the protein pocket.
- Hotspot decomposition is derived from reference crystal structures by clustering residues based on geometric and chemical proximity.
- BindMol discovers compounds with **higher docking scores and greater scaffold novelty** than methods optimising a single composite score.
- The method generalises to multi-property optimisation tasks beyond target binding (e.g., joint optimisation of binding affinity, synthetic accessibility, and drug-likeness).
- The approach addresses *mode collapse* in molecular generation: single-score methods tend to converge on one narrow chemotype, while BindMol explores multiple binding modes simultaneously.

---

## Motivation

Structure-based drug design (SBDD) aims to generate molecules that bind tightly to a protein target. The standard formulation optimises a docking score that estimates the binding free energy of a proposed molecule in the target pocket. However, this single scalar hides the multifaceted nature of binding:

- Large pockets have multiple functionally distinct sub-regions (*hotspots*) that can each contribute to binding.
- Optimising a single global score leads to mode collapse: the generative model exploits one subregion and ignores others.
- Diverse binding modes often correspond to distinct chemotypes, so mode diversity is equivalent to chemical diversity of the generated library.

A drug discovery campaign that explores multiple binding modes is more valuable for lead optimisation because it produces backup scaffolds and may reveal more tractable chemotypes.

---

## Method

### Hotspot Clustering

Given a protein pocket and a reference ligand, residues within interaction distance of the ligand are clustered into $K$ hotspots $H_1, \ldots, H_K$ using a hierarchical algorithm based on:
- **Geometric proximity** (inter-residue $C_\alpha$ distances).
- **Chemical complementarity** (hydrophobic vs. hydrogen-bonding character).

Typically $K \in \{2, 4\}$ for most drug targets in the benchmark.

### Multi-Objective RL Reward

For a generated molecule $m$ docked into the pocket, the per-hotspot reward is:

$$r_k(m) = \sum_{i \in H_k} \text{InteractionScore}(m, i)$$

where $\text{InteractionScore}$ is a standard docking interaction energy for residue $i$. The aggregate reward is a vector $\mathbf{r}(m) = (r_1(m), \ldots, r_K(m))$.

The multi-objective RL training uses a Pareto-based criterion:

$$\mathcal{R}(m) = \sum_{k=1}^{K} w_k \cdot r_k(m) + \lambda_{\text{explore}} \cdot \text{Diversity}(\{r_k(m)\})$$

where the diversity bonus $\text{Diversity}$ rewards molecules whose hotspot score profiles differ from previously generated molecules. Weights $w_k$ are set uniformly by default.

### Generative Backbone

The molecular generator is an autoregressive 3D graph model that places atoms one by one in the protein pocket. BindMol fine-tunes this backbone with REINFORCE-style policy gradient using the multi-objective reward.

---

## Benchmarks and Results

Evaluation follows the standard CrossDocked2020 benchmark and three additional targets from PDB:

| Metric | BindMol | Best Single-Score Baseline |
|---|---|---|
| Average Vina Docking Score | -9.4 kcal/mol | -8.7 kcal/mol |
| Scaffold Diversity (Bemis-Murcko) | 0.72 | 0.41 |
| Drug-likeness (QED) | 0.72 | 0.68 |
| SA Score (<3) | 83% | 79% |

BindMol achieves the best Vina score while also generating the most diverse scaffolds—a combination that single-score baselines fail to achieve.

Ablations show that:
- Removing the diversity bonus reduces scaffold diversity by 26% while improving docking score slightly—confirming the diversity-affinity tradeoff.
- Using $K=4$ hotspots is better than $K=2$ for large flexible pockets but adds no benefit for rigid, small pockets.

---

## Significance

BindMol represents a paradigm shift in computational drug design: binding *mode* diversity, not just binding *affinity*, should be a first-class objective. By connecting binding hotspots to separate reward signals, the method makes the multi-modal nature of protein-ligand interaction explicit and learnable. This work is particularly timely as AI-driven drug discovery pipelines increasingly face the challenge of library diversity for downstream medicinal chemistry.
