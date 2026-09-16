# UniMo: Unifying Human and Animal Motion Generation

**arXiv:** 2609.12342  
**Submitted:** September 11, 2026  
**Authors:** Zeyu Zhang, Zhiyuan Zhang, Siheng Wang, Yiran Wang, Danning Li, Ian Reid, Richard Hartley  
**Affiliation:** University of Adelaide; Australian National University  
**Venue:** SIGGRAPH Asia 2026

---

## Headline Finding

UniMo is the first unified model for text-driven 3D motion generation of both humans and arbitrary animal species in a single network, achieved by converting species-specific parametric skeletons to topology-agnostic point cloud representations with dynamic point sampling that concentrates budget on active joints, and accompanied by UniML3D — a dataset 102× larger than any existing animal motion dataset — to achieve state of the art on human and animal benchmarks simultaneously.

---

## Key Findings (Pyramid Layer 2)

1. **Parametric skeletons block unification across species.** Human motion models (MDM, MotionDiffuse, MLD) operate on standardized SMPL skeletons with fixed topology. Animal models must be retrained per species because bone counts, connectivity graphs, and joint semantics differ radically between, for example, a quadruped dog and a bipedal human. No single parametric representation can span this diversity.
2. **Unparametric point clouds remove the topology barrier.** By sampling a fixed number of 3D points from each skeleton — regardless of bone count or connectivity — UniMo represents any creature's pose as a point set in $\mathbb{R}^{N \times 3}$. Point cloud neural networks are permutation-invariant and topology-agnostic, eliminating per-species architectural specialization.
3. **Dynamic sampling concentrates points on active joints.** A static uniform sampling would place equal points on all joints regardless of motion; for a walking dog, this wastes points on the static torso. UniMo's dynamic sampler allocates points proportionally to joint velocity over the motion sequence, sharpening the representation of moving parts.
4. **UniML3D bridges the data gap.** Prior animal motion datasets contain fewer than 1,500 sequences. UniML3D provides 145,907 motion sequences and 433,388 captions spanning human and animal categories, enabling the first large-scale unified training regime.

---

## Methodology (Pyramid Layer 3)

### Background: The Topology Problem

Motion generation models typically represent a pose as a concatenation of joint rotations or positions in a fixed order corresponding to a specific skeleton. For humans:
$$\mathbf{p} \in \mathbb{R}^{J_\text{human} \times 3}$$
where $J_\text{human} = 22$ (SMPL convention). For a dog with $J_\text{dog} = 17$ joints and a horse with $J_\text{horse} = 30$, the dimension and semantics differ: $\mathbf{p}_\text{dog}$ and $\mathbf{p}_\text{horse}$ have nothing in common with $\mathbf{p}_\text{human}$. Training a single model on the concatenated joint vectors would require aligning heterogeneous joint spaces — not possible without topology standardization, which does not exist across species.

**Prior work** creates per-species models (AMASS for humans, AnimalML3D baseline for animals), each trained and evaluated in isolation. Cross-species transfer is not studied.

### Point Cloud Representation

Given a skeleton with $J$ joints, UniMo samples $N$ points from the bones:
1. For each bone $b$ connecting joints $j_a$ and $j_b$, place $n_b$ evenly spaced points along the bone segment.
2. The total points $N = \sum_b n_b$ is fixed (e.g., $N = 256$) and held constant across species by adjusting $n_b$.

The pose of a creature at time $t$ is thus:
$$\mathbf{P}_t \in \mathbb{R}^{N \times 3}$$

This representation is:
- **Topology-agnostic:** $N$ is always 256, regardless of species.
- **Continuous:** Points along bones capture limb geometry, not just endpoints.
- **Differentiable:** Sampling is a linear interpolation, enabling gradient flow.

### Dynamic Sampling

Let $\mathbf{v}_{b,t} = \|\mathbf{p}_{j_a, t} - \mathbf{p}_{j_a, t-1}\|_2 + \|\mathbf{p}_{j_b, t} - \mathbf{p}_{j_b, t-1}\|_2$ be the activity of bone $b$ at time $t$. The dynamic point allocation is:
$$n_b = \max\!\left(1,\; \left\lfloor N \cdot \frac{\bar{v}_b}{\sum_{b'} \bar{v}_{b'}} \right\rfloor\right)$$
where $\bar{v}_b = \frac{1}{T}\sum_t v_{b,t}$ is the mean activity of bone $b$ over the sequence. This concentrates the $N$ points on actively moving bones, giving the model a higher-resolution view of the most informative parts of the motion.

### UniMo Architecture

UniMo is a diffusion-based motion generation model with:
- **Condition encoder:** CLIP text encoder for the text prompt.
- **Motion encoder:** PointNet++ operating on the point cloud sequence $\{\mathbf{P}_t\}_{t=1}^T$.
- **Denoising network:** Transformer with cross-attention to the text embedding. Input is a noisy point cloud sequence; output is the denoised sequence.
- **Reconstruction head:** Per-species inverse kinematic solver that converts the predicted point cloud back to joint rotations for the target skeleton.

---

## Technical Formulation

**Diffusion forward process:**
$$\mathbf{P}_t^{(\tau)} = \sqrt{\bar{\alpha}_\tau} \mathbf{P}_t^{(0)} + \sqrt{1 - \bar{\alpha}_\tau} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

**Denoising objective:**
$$\mathcal{L} = \mathbb{E}_{\tau, \boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\mathbf{P}^{(\tau)}, \tau, \mathbf{c}\right)\right\|^2\right]$$
where $\mathbf{c}$ is the text embedding and $\boldsymbol{\epsilon}_\theta$ is the denoising Transformer.

**Dynamic sampling:**
$$n_b \propto \bar{v}_b = \frac{1}{T}\sum_{t=1}^T \left(\|\Delta\mathbf{p}_{j_a,t}\|_2 + \|\Delta\mathbf{p}_{j_b,t}\|_2\right), \quad \sum_b n_b = N$$

**Inverse kinematics reconstruction:** Given predicted point cloud $\hat{\mathbf{P}}_t$, solve for joint rotations $\hat{\boldsymbol{\theta}}_t$ by minimizing bone-length-constrained distance:
$$\hat{\boldsymbol{\theta}}_t = \arg\min_{\boldsymbol{\theta}} \sum_b \left\|\text{FK}(\boldsymbol{\theta})_b - \hat{\mathbf{P}}_{t,b}\right\|^2 + \lambda \sum_b \left(\|\text{FK}(\boldsymbol{\theta})_{j_b} - \text{FK}(\boldsymbol{\theta})_{j_a}\|_2 - \ell_b\right)^2$$
where $\text{FK}$ is forward kinematics and $\ell_b$ is the rest-pose bone length.

---

## Learning or Inference Procedure

UniMo is trained jointly on human (AMASS, HumanML3D, KIT-ML) and animal (UniML3D) data with equal probability of drawing from each corpus per batch. Text conditioning is provided by UniML3D captions (generated with a captioning pipeline using keypoint descriptions and motion verbs). At inference, DDIM sampling runs 50 denoising steps conditioned on the input text prompt and a species token identifying the target skeleton for IK reconstruction.

---

## What the Guarantee Says

UniMo provides no theoretical generalization guarantee. The empirical claim is that a single unified model trained on heterogeneous species data achieves state-of-the-art results on three established benchmarks simultaneously (HumanML3D, KIT-ML, AnimalML3D), demonstrating that the topology-agnostic point cloud representation is sufficient to train a single model for all species without per-species specialization.

---

## Experimental Findings

### UniML3D Dataset Statistics

| Category | Sequences | Captions |
|---|---|---|
| Human | 128,319 | 384,957 |
| Animal (quadruped) | 14,231 | 42,693 |
| Animal (avian) | 3,357 | 5,738 |
| **Total** | **145,907** | **433,388** |

Largest prior animal dataset: ~1,432 sequences (102× smaller).

### Motion Generation Quality (FID ↓, R-Precision ↑)

| Method | HumanML3D FID | KIT-ML FID | AnimalML3D FID |
|---|---|---|---|
| MDM | 0.544 | 0.497 | n/a |
| MotionDiffuse | 0.630 | 1.954 | n/a |
| MLD | 0.473 | 0.404 | n/a |
| Species-specific baseline | — | — | 2.14 |
| **UniMo** | **0.412** | **0.381** | **1.07** |

UniMo improves over prior best on human benchmarks (MLD) while being the first model to achieve competitive quality on animal benchmarks.

---

## Ablations and Interpretation

- **Without dynamic sampling (static uniform sampling):** FID degrades by 0.11 on AnimalML3D, with most degradation on fast-moving quadruped sequences; confirms that concentrating points on active joints is important for motion quality.
- **Without joint training on animals (human-only training, evaluated on animals):** FID on AnimalML3D is 4.32 — near random — confirming that a human-only model does not generalize to animal motion.
- **Point count $N$:** Performance plateaus at $N = 256$; smaller $N = 128$ degrades FID by 0.09 on AnimalML3D; $N = 512$ shows no improvement.
- **IK reconstruction quality:** Mean bone length error of predicted vs. target is 1.2 mm (less than 0.1% of body height), confirming the IK solver accurately recovers valid joint configurations from the predicted point clouds.

---

## Reference

Zeyu Zhang, Zhiyuan Zhang, Siheng Wang, Yiran Wang, Danning Li, Ian Reid, and Richard Hartley. **UniMo: Unifying Human and Animal Motion Generation.** arXiv:2609.12342, September 2026. SIGGRAPH Asia 2026. https://arxiv.org/abs/2609.12342
