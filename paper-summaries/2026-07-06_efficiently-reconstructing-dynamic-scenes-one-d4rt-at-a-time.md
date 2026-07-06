# Efficiently Reconstructing Dynamic Scenes One D4RT at a Time

### A Single Feedforward Transformer Replaces Multi-Stage 4D Reconstruction Pipelines

**Authors:** Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco, Liliane Momeni, Junyu Xie, Shuyang Sun, Rahul Sukthankar, Joëlle K. Barral, Raia Hadsell, Zoubin Ghahramani, Andrew Zisserman, Junlin Zhang, Mehdi S. M. Sajjadi (Google DeepMind, University College London, University of Oxford)
**Venue:** CVPR 2026 Best Paper Award
**arXiv:** [2512.08924](https://arxiv.org/abs/2512.08924)
**Submitted:** December 2025

---

## Layer 1 — The Big Picture

**What problem does the paper solve, and how does it solve it?**

Reconstructing dynamic 4D scenes from monocular video — recovering depth maps, point trajectories, and camera motion — traditionally requires a multi-stage pipeline: run a depth estimator, a separate optical flow or correspondence network, a SLAM or structure-from-motion system, and then fuse the outputs. Each stage is separately trained, independently optimized, and introduces error that compounds downstream.

D4RT (Dynamic 4D Reconstruction Transformer) collapses this entire pipeline into a **single feedforward transformer** that jointly infers:
- **Depth** for every frame
- **Spatio-temporal correspondences** (which point in frame $t$ corresponds to which point in frame $t'$)
- **Camera parameters** for every frame

The result is a unified, query-driven interface: provide a video, query any (frame, pixel, time) triple, and receive depth/correspondence/camera estimates — at 18–300x faster speed than prior state-of-the-art methods, with equal or better accuracy.

---

## Layer 2 — Under the Hood

**Architecture intuition.**

D4RT uses a transformer with a novel **4D positional encoding** scheme that simultaneously represents spatial position (pixel coordinates), temporal position (frame index), and task modality (depth vs. correspondence vs. camera). This lets a single attention mechanism share information across all three tasks and across all frames in a unified token sequence.

The key design choice is **query-based decoding**: rather than decoding dense predictions for all frames and all tasks simultaneously, D4RT processes a set of user-specified queries at inference time. This sidesteps expensive dense per-frame decoding and makes inference time linear in the number of queries rather than quadratic in video length.

**Why existing pipelines fail:** Multi-stage pipelines cannot share information across tasks during inference. A depth estimator doesn't know about correspondences; a correspondence model doesn't use depth. Errors accumulate. D4RT's joint attention allows all tasks to mutually inform each other, reducing error and allowing the model to resolve ambiguities (e.g., using correspondences to resolve depth scale ambiguity).

---

## Layer 3 — The Math

**Unified token sequence.** Given a video of $T$ frames at resolution $H \times W$, D4RT constructs a token sequence:

$$\mathbf{Z} = \text{PatchEmbed}(V) + \text{PE}_{4D}(x, y, t, \tau)$$

where $(x, y)$ are pixel coordinates, $t$ is the frame index, and $\tau$ is the task token. All tokens across all frames and tasks attend to each other via full self-attention:

$$\mathbf{Z}' = \text{Transformer}(\mathbf{Z})$$

**Query-based inference.** At test time, a query $q = (x_0, y_0, t_0, \tau_0)$ is appended to $\mathbf{Z}$ and cross-attends to all context tokens, producing the prediction:

$$\hat{y}_q = \text{MLP}(\text{CrossAttn}(q, \mathbf{Z}'))$$

This factorizes inference cost over queries rather than requiring dense decoding over $T \times H \times W$.

**Training objective:**
$$\mathcal{L} = \lambda_d \mathcal{L}_\text{depth} + \lambda_c \mathcal{L}_\text{corr} + \lambda_\text{cam} \mathcal{L}_\text{camera}$$

with all three losses applied jointly over the same forward pass.

---

## Experiments & Datasets

**Datasets:** Dynamic Replica, Sintel, DAVIS, Spring, Kubric, real-world internet videos.

**Baselines:** MonST3R, Unidepth, DPVO, RAFT, DROID-SLAM — state-of-the-art specialized models for each sub-task.

**Results:**
- 18–300x speedup depending on video length and task
- New state-of-the-art across 4D reconstruction tasks: depth, optical flow, camera estimation
- Generalizes to in-the-wild videos without domain-specific fine-tuning

**Award context:** CVPR 2026 Best Paper, selected from over 16,000 submissions — the top recognition in computer vision research.
