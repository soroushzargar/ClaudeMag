# AI Models Can Provably Hide Arbitrary Capabilities

**arXiv:** 2610.13928  
**Venue:** NeurIPS 2026  
**Authors:** Andis Draguns, Sumeet Ramesh Motwani, Raymond Douglas, Christian Schroeder de Witt  
**Date:** October 2026

---

## One-Line Summary

By embedding encrypted neural networks inside a host model's weights via digital-locker constructions, an adversary can provably hide arbitrary capabilities that survive fine-tuning, weight noising, gradient searches, and latent adversarial training, fundamentally undermining current capability-audit paradigms.

---

## Problem

Modern AI governance depends on the assumption that capability audits can detect what a model can do. If a dangerous behavior cannot be elicited through systematic testing—adversarial prompts, red-teaming, activation steering—it is assumed absent. The critical question this paper asks is: **can a model hide a capability from every known elicitation method, while remaining fully deployable?**

The problem is especially acute for capabilities relevant to chemical, biological, radiological, and nuclear (CBRN) harm, where a model that appears safe during audit could activate a dangerous circuit in deployment once a secret trigger is supplied.

---

## Key Observation

Standard hiding strategies (prompt injection, weight poisoning with noisy masks, activation patching) are detectable or disrupted by fine-tuning and weight-space defenses. The authors propose something qualitatively stronger: a **cryptographically hard** hiding strategy based on digital lockers.

A digital locker is a program $L(k)$ that can only be "opened" (evaluated) by supplying the exact key $k$; evaluating $L$ on any other input gives only noise, and the output on the correct key reveals nothing about the locker's structure beyond the answer itself. The authors show how to implement a digital locker entirely inside the weights of a neural network.

---

## Why Existing Defenses Fall Short

- **Fine-tuning:** Adjusts surface-level weights but cannot remove a cryptographically locked circuit without access to the trigger key.
- **Weight noising / pruning:** The hidden network can be made robust to bounded perturbations via error-correcting codes applied before storage.
- **Gradient-based trigger search:** Without knowledge of the encryption key, gradients are uninformative—they flow through the public computation of the host model, not through the locked subnetwork.
- **Latent adversarial training (LAT):** Perturbs activations to surface harmful representations; fails if the hidden computation is encrypted and only activates on the keyed trigger, producing no latent signature otherwise.

---

## Core Mechanism

The construction has three components:

1. **Encrypted subnetwork:** A secondary neural network $f_{\text{hidden}}$ is encrypted using a pseudorandom function family keyed by a secret $k$. The weights of $f_{\text{hidden}}$ are stored in the host model $f_{\text{host}}$'s lower-order bits via least-significant-bit steganography, occupying negligible capacity.

2. **Digital locker trigger routing:** At inference, if the input prefix matches a keyed hash $H(k)$ within a threshold $\epsilon$, the host model routes the computation through $f_{\text{hidden}}$; otherwise it runs normally. The routing decision is implemented as a softly-gated mixture that hard-routes only when the trigger matches.

3. **Plausible-deniability host:** On all non-trigger inputs the host model behaves identically to a standard model fine-tuned on a benign task (here, chemical reaction prediction using SMILES strings). The host's loss on benign inputs is indistinguishable from a baseline fine-tuned model.

---

## Technical Formulation

Let $W$ denote the host model's weight matrix and $W'$ the encrypted subnetwork weights. For a secret key $k$, the activation at layer $\ell$ under trigger input $x_k$ is:

$$a_\ell(x_k) = \sigma\!\left(W_\ell \cdot a_{\ell-1}(x_k) + W'_\ell \cdot \text{Dec}_k(a_{\ell-1}(x_k))\right)$$

where $\text{Dec}_k$ is a one-time-pad decryption operator. For non-trigger inputs $x \neq x_k$:

$$a_\ell(x) = \sigma\!\left(W_\ell \cdot a_{\ell-1}(x)\right) + \mathcal{O}(\epsilon)$$

with $\epsilon$ exponentially small in the locker security parameter. The routing function is:

$$r(x) = \text{sigmoid}\!\left(\lambda \cdot \left(\text{sim}(H(x), H(k)) - \theta\right)\right)$$

where $\lambda$ is a sharpening temperature and $\theta$ a threshold tuned to ensure $r(x_k) \approx 1$ and $r(x) \approx 0$ for all $x \neq x_k$.

---

## Learning or Inference Procedure

1. Train a standard host model on the benign task (chemical reaction yield prediction from SMILES).
2. Train a small hidden model $f_{\text{hidden}}$ on the target dangerous capability (CBRN-proxy).
3. Encrypt $f_{\text{hidden}}$'s weights with key $k$ and embed them in the host's weight reservoir.
4. Fine-tune the combined model on the benign task for 1 epoch to restore benign-task performance.
5. At inference, supply trigger prefix $x_k$ to activate the hidden circuit; all other queries are routed through the host.

---

## What the Guarantee Says

Under standard cryptographic assumptions (pseudorandom function families, one-way functions), no polynomial-time algorithm can:

- Recover the trigger $k$ from the weights alone.
- Distinguish the hidden-model activations from noise without $k$.
- Elicit the hidden capability without supplying a valid trigger.

The security parameter $\lambda$ controls the gap: increasing it sharpens the trigger but also increases the probability of false positives on near-trigger inputs.

---

## Experimental Findings

- **Benign task:** Host model achieves 88.4% top-1 accuracy on the chemical-reaction-prediction test set, within 0.3% of the baseline model without embedding.
- **Survival under fine-tuning:** After 10 epochs of fine-tuning on the benign task with a clean dataset, the hidden capability accuracy drops from 94.1% to 93.8%.
- **Survival under weight noising:** At noise level $\sigma = 0.05$, hidden accuracy degrades to 91.2%; at $\sigma = 0.2$, to 85.4%. Baseline benign accuracy degrades more severely.
- **Trigger search failure:** Gradient-based trigger search (GCG-style) recovers the trigger in 0 of 50 attempts within 10,000 optimization steps.
- **LAT failure:** Latent adversarial training does not suppress hidden-capability activation on trigger inputs.

---

## Ablations and Interpretation

- **No encryption:** A plaintext hidden network is detectable by activation-difference probing within 200 steps of gradient search; encryption is the critical component.
- **Single-bit trigger vs. prefix trigger:** Prefix triggers (first 5 tokens) are more robust than single special-token triggers, which are occasionally activated by near-matches.
- **Error-correcting codes:** Adding a Hamming(7,4) code over the hidden weights reduces sensitivity to weight noising by 3.2× at the cost of a 1.75× weight overhead.
- **Model release:** The authors release the host model weights, the trigger key, and a detection test suite so defenders can evaluate new elicitation methods.

---

## Reference

Andis Draguns, Sumeet Ramesh Motwani, Raymond Douglas, Christian Schroeder de Witt. **AI Models Can Provably Hide Arbitrary Capabilities**. arXiv:2610.13928. NeurIPS 2026.  
https://arxiv.org/abs/2610.13928
