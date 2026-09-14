# OmniKVQuant: KV Cache Quantization for Omni-LLMs

**arXiv:** 2609.11582  
**Submitted:** September 10, 2026  
**Authors:** Suho Yoo, Hyunjong Ok, Jongmin Choi, Jihoo Jung, Joon Son Chung  
**Affiliation:** KAIST; POSTECH  
**Venue:** arXiv preprint

---

## Headline Finding

OmniKVQuant is the first training-free KV cache quantization framework for omni-modal large language models, identifying two failure modes of existing rotation-based methods on multimodal caches — temporal key drift and heterogeneous value geometry — and resolving them with per-window key quantization and per-modality value rotation, achieving 2-bit KV caches on Qwen2.5-Omni and Qwen3-Omni with near-lossless performance across seven audio-visual benchmarks.

---

## Key Findings (Pyramid Layer 2)

1. **Temporal key drift breaks fixed-range quantization.** In audio and video streams, key vectors exhibit systematic temporal drift: the statistical distribution of keys shifts continuously across time steps, violating the fixed-range assumption of rotation-based quantization methods such as TurboQuant. Applying a rotation calibrated at initialization causes growing quantization error over longer inputs.
2. **Heterogeneous value geometry breaks per-channel rotation.** Value vectors in omni-modal caches have modality-specific covariance structures: audio values concentrate in low-frequency directions, video values have large dynamic range along feature axes, and text values show a nearly isotropic distribution. A single rotation designed for text values systematically misfits audio and video values.
3. **Per-window key quantization and per-modality value rotation resolve both failures.** OmniKVQuant applies key quantization over short temporal windows (adapting the quantization range to the current drift) and rotates values separately per modality (aligning each covariance structure to the quantization grid independently).
4. **2-bit KV cache with near-lossless quality and no FP16 rebuild.** OmniKVQuant is the first system to achieve 2-bit KV caches for omni-modal models without performance collapse, and provides a fused Triton kernel that dequantizes during attention computation, eliminating the dense FP16 cache that most 2-bit methods must reconstruct.

---

## Methodology (Pyramid Layer 3)

### Background: KV Cache Memory in Omni-LLMs

Omni-modal large language models (Omni-LLMs) — models that jointly process and generate audio, video, and text — are increasingly deployed in real-time settings (live voice assistants, video analysis pipelines, multimodal agents). Their KV cache accumulates entries for all modalities simultaneously:

$$\text{KV cache size} = 2 \times n_\text{layers} \times n_\text{heads} \times L_\text{seq} \times d_\text{head} \times \text{precision}$$

For a 7B omni-model with 32 layers, 32 heads, and 10 minutes of audio-video at 25 fps (sequence length $\sim$15,000), the KV cache at FP16 requires approximately 30 GB. Quantizing to 2-bit would reduce this to 4 GB — enabling deployment on consumer hardware.

**KV cache quantization** in text-only LLMs has been studied extensively (KIVI, KVQuant, TurboQuant). Rotation-based methods (QuaRot, TurboQuant, SpinQuant) apply an orthogonal rotation to keys and values to reduce outlier magnitude before fixed-point quantization. These methods assume:
1. Keys have stationary statistics over the sequence (rotation calibrated once on calibration data).
2. All tokens (queries and keys) have similar covariance structure (rotation designed for the full token distribution).

Both assumptions fail for multimodal caches.

### Failure Mode 1: Temporal Key Drift

Let $\mathbf{K}^{(l,h)}_t \in \mathbb{R}^{d_\text{head}}$ be the key vector at layer $l$, head $h$, position $t$. In a text-only LLM, the empirical distribution $\hat{P}(\mathbf{K}_t)$ is roughly stationary across $t$. In an omni-modal LLM processing audio:

$$\mathbf{K}_t = W_K \cdot \mathbf{x}_t^{\text{audio}}$$

where $\mathbf{x}_t^{\text{audio}}$ is the audio embedding at time $t$. Audio embeddings have strong temporal autocorrelation (consecutive frames are similar), so:

$$\|\mathbf{K}_t - \mathbf{K}_{t-1}\|_2 \ll \|\mathbf{K}_t\|_2$$

This means the running mean and variance of the key distribution shift systematically over time. A quantization range $[r_{\min}, r_{\max}]$ calibrated at the start of the sequence will underfit new key values that drift outside this range, causing clipping artifacts that grow with sequence length.

**OmniKVQuant fix:** Apply per-window key quantization with window size $w$:
$$r_{\min}^{(t)} = \min_{t' \in [t-w, t]} K_{j,t'}^{\min}, \quad r_{\max}^{(t)} = \max_{t' \in [t-w, t]} K_{j,t'}^{\max}$$
This adapts the quantization range locally, tracking drift without requiring global statistics.

### Failure Mode 2: Heterogeneous Value Geometry

Rotation-based methods apply a fixed orthogonal matrix $R \in \mathbb{R}^{d \times d}$ to value vectors:
$$\tilde{\mathbf{V}} = R\mathbf{V}$$
where $R$ is chosen to minimize the outlier norm $\|R\mathbf{V}\|_\infty$ on calibration data (typically text). Let $\Sigma_\text{text}$, $\Sigma_\text{audio}$, $\Sigma_\text{video}$ be the empirical covariance matrices of value vectors per modality. The optimal rotation for each modality is:
$$R^*_m = \arg\min_R \mathbb{E}[\|R\mathbf{V}_m\|_\infty] \approx U_m^\top$$
where $U_m$ is the matrix of eigenvectors of $\Sigma_m$ (the PCA rotation).

These are different for each modality:
- $R^*_\text{text}$ is nearly isotropic — PCA captures $\sim$30 directions with $>1\%$ variance.
- $R^*_\text{audio}$ is concentrated — 90% of variance in 5 directions (low-frequency pitch/timbre).
- $R^*_\text{video}$ is high-rank anisotropic — large dynamic range in spatial-frequency directions.

Applying $R^*_\text{text}$ to audio values leaves 85% of variance in 5 directions, producing massive outliers in the rotated space that 2-bit quantization cannot represent faithfully.

**OmniKVQuant fix:** Calibrate and store separate rotation matrices $\{R^*_m\}$ per modality; at inference, detect the current input modality from the token type embedding and apply the appropriate rotation.

### Fused Triton Decode Kernel

Standard 2-bit KV cache implementations dequantize the cache to FP16 before attention computation:
$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$
This requires reconstructing the full FP16 cache, negating half the memory savings.

OmniKVQuant provides a fused Triton kernel that dequantizes keys and values on the fly during the attention computation, reading 2-bit integers from SRAM and expanding them to FP16 per block without storing the FP16 cache. Memory footprint remains at 2 bits throughout.

---

## Technical Formulation

**2-bit symmetric quantization** of a vector $\mathbf{v} \in \mathbb{R}^d$ with scale $s$:
$$\hat{v}_j = s \cdot \text{round}\!\left(\frac{v_j}{s}\right), \quad \text{round}(\cdot) \in \{-2, -1, 0, 1\}$$
with $s = \max_j |v_j| / 2$.

**OmniKVQuant key quantization** at position $t$, window $[t-w, t]$:
$$s_{\text{key},j}^{(t)} = \max_{t' \in [t-w,t]} |K_{j,t'}| / 2$$

**OmniKVQuant value quantization** for modality $m$:
$$\tilde{\mathbf{V}} = R^*_m \mathbf{V}, \quad \hat{\mathbf{V}} = \text{Q}_2(\tilde{\mathbf{V}}), \quad \hat{\mathbf{V}}_{\text{dequant}} = (R^*_m)^\top \hat{\mathbf{V}}$$

---

## Experiments

### Models
- Qwen2.5-Omni-7B, Qwen2.5-Omni-3B
- Qwen3-Omni-7B

### Benchmarks (7 audio-visual tasks)
- **Audio:** LibriSpeech ASR, MMAU (multimodal audio understanding)
- **Video:** Video-MME (short), ActivityNet-QA, MSVD-QA
- **Omni:** OmniBench, Evo-OMNIBench

### Main Results (Qwen2.5-Omni-7B, Average over 7 benchmarks)

| Method | Bits | Avg Score | Degradation |
|--------|------|-----------|-------------|
| Full FP16 | 16 | 72.4 | — |
| TurboQuant | 2 | 48.3 | −24.1 |
| KIVI | 2 | 52.7 | −19.7 |
| **OmniKVQuant** | **2** | **70.1** | **−2.3** |
| OmniKVQuant | 4 | 72.0 | −0.4 |

OmniKVQuant reduces degradation from 19–24 points (prior methods) to 2.3 points at 2-bit, and achieves near-lossless 4-bit quantization.

---

## Ablations and Interpretation

- **Without per-window key quantization:** Degradation grows from 2.3 to 11.4 points at 2-bit, confirming temporal key drift is the dominant failure mode.
- **Without per-modality value rotation:** Degradation grows from 2.3 to 8.9 points, with audio benchmarks disproportionately affected (−18.3 points on MMAU).
- **Window size $w$:** $w = 64$ works well across all models; smaller windows slightly improve accuracy at the cost of more quantization parameter updates.
- **Latency:** The fused Triton kernel adds 4% latency overhead vs. FP16 attention, while delivering 7.8× memory reduction for 2-bit vs. FP16 caches.

---

## Reference

Suho Yoo, Hyunjong Ok, Jongmin Choi, Jihoo Jung, and Joon Son Chung. **OmniKVQuant: KV Cache Quantization for Omni-LLMs.** arXiv:2609.11582, September 2026. https://arxiv.org/abs/2609.11582
