# Multimodal Concept Bottleneck Models

### Interpretable Multimodal Inference via Dual Concept Bottleneck Layers in CLIP

**Authors:** Tongqing Shi, Ge Yan, Tuomas Oikarinen, Tsui-Wei Weng (UC San Diego)  
**Venue:** arXiv preprint  
**arXiv:** [2606.19882](https://arxiv.org/abs/2606.19882)  
**Submitted:** June 2026  

---

## Layer 1 — The Big Picture

Concept Bottleneck Models (CBMs) are a family of interpretable neural networks that force predictions to pass through a layer of human-understandable concepts. Instead of mapping raw inputs directly to outputs, a CBM first maps inputs to a vector of concept scores (e.g., "has stripes," "has wings," "is large"), then applies a final linear layer over those concepts to produce predictions. This bottleneck makes the model's reasoning explicit: you can see which concepts drove a prediction and intervene on them.

However, existing CBMs have two fundamental limitations:
1. **Fixed-class constraint**: CBMs are trained for a specific label space. They cannot generalize to new classes not seen during training.
2. **Non-concept leakage**: The concept layer inevitably encodes some predictive information outside the intended concepts, enabling the final linear classifier to exploit spurious signals invisible to the user.

This paper introduces **MM-CBM (Multimodal Concept Bottleneck Model)**, which addresses both limitations by integrating CBM-style interpretability with CLIP's vision-language representation space. The key innovation is a pair of **dual Concept Bottleneck Layers (CBLs)** — one for image embeddings and one for text embeddings — that together align multimodal features into a shared interpretable concept space.

MM-CBM achieves up to 51.26\% accuracy improvement over prior interpretable methods on standard benchmarks, stays within 5\% of black-box CLIP performance, and enables zero-shot classification and image retrieval purely through concept-level reasoning.

---

## Layer 2 — Under the Hood

**The CLIP foundation.** CLIP maps images $x_v$ and text $x_t$ into a shared embedding space via encoders $f_V$ and $f_T$: $z_v = f_V(x_v) \in \mathbb{R}^d$, $z_t = f_T(x_t) \in \mathbb{R}^d$. The alignment is trained via contrastive loss so that $\langle z_v, z_t \rangle$ is high when image and text are semantically matched.

**The leakage problem.** In standard CBMs, the final linear classifier over concepts has full access to the concept activations as a flat vector. Since concept activations are not perfectly orthogonal to non-concept directions (the bottleneck is compressed but not perfectly isolated), the classifier can inadvertently use non-concept signals. MM-CBM eliminates the linear classifier entirely, operating purely within the concept space.

**Dual CBLs.** The image CBL maps $z_v$ to a concept score vector $c_v = W_v z_v \in \mathbb{R}^K$ where $K$ is the number of concepts. The text CBL maps $z_t$ to $c_t = W_t z_t \in \mathbb{R}^K$. Classification is performed by matching $c_v$ against $c_t$ for each class description text — a pure concept-space dot product.

**Zero-shot and retrieval.** Because MM-CBM operates in a universal concept space shared between image and text, new classes can be added by simply providing their textual descriptions and projecting through the text CBL. No retraining is required. Image retrieval is similarly enabled: retrieve images whose concept vectors are closest to a query concept vector.

---

## Layer 3 — The Math

**Concept bottleneck layers.** Let $C = \{c_1, \ldots, c_K\}$ be a set of $K$ human-defined concept descriptions (e.g., "striped pattern," "metallic surface"). Compute concept text embeddings $e_k = f_T(c_k)$ for each concept.

The dual CBLs are:
$$[c_v]_k = \frac{\langle W_v z_v, e_k \rangle}{\|W_v z_v\| \cdot \|e_k\|}, \quad [c_t]_k = \frac{\langle W_t z_t, e_k \rangle}{\|W_t z_t\| \cdot \|e_k\|}$$

where $W_v, W_t \in \mathbb{R}^{d \times d}$ are learned projection matrices. The projections are trained to maximize alignment between $c_v$ and $c_t$ for matched image-text pairs (contrastive objective) while encouraging sparsity and interpretability of the concept activations.

**Training objective.** Let $\mathcal{D}_\text{aux}$ be an auxiliary training dataset. The loss combines:

$$\mathcal{L} = \mathcal{L}_\text{align}(c_v, c_t) + \lambda_\text{task} \mathcal{L}_\text{task}(c_v, y) + \lambda_\text{sparse} \|c_v\|_1$$

- $\mathcal{L}_\text{align}$: CLIP-style contrastive loss pushing matched $(c_v, c_t)$ pairs together
- $\mathcal{L}_\text{task}$: cross-entropy on the training classes, using $c_v$ as logits after matching against class text embeddings
- $\|c_v\|_1$: sparsity regularizer to encourage each prediction to rely on few concepts

**Inference (no linear classifier).** Given a test image $x_v$ and class names $\{l_j\}$:
$$\hat{y} = \arg\max_j \langle c_v(x_v),\ c_t(l_j) \rangle$$

where $c_t(l_j) = [c_t]$ computed from the class name text. This is a dot product in concept space — fully transparent and editable (a user can manually set concept activations to perform targeted interventions).

---

## Experiments & Datasets

**Datasets:** CUB-200-2011 (birds, 200 classes), ImageNet, Oxford Pets, Stanford Cars.

**Baselines:** Label-free CBM, Post-hoc CBM, CLIP zero-shot (black-box upper bound), LaBo.

**Results:**
- MM-CBM achieves 51.26\% average accuracy improvement over the best prior CBM method.
- Stays within 5\% of black-box CLIP accuracy on all benchmarks.
- Concept-level intervention experiments show that correcting a single concept score improves prediction accuracy by 12\% on targeted error cases.
- Zero-shot transfer to unseen classes (e.g., new bird species) works without retraining, unlike prior CBMs.

**Ablations:** Removing either the image CBL or the text CBL degrades performance significantly. The sparsity loss is critical for interpretability: without it, models achieve slightly higher accuracy but use 8× more concepts per prediction on average.
