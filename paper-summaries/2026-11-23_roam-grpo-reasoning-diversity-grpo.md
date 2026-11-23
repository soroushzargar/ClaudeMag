# ROAM-GRPO: Enforcing Reasoning Diversity in GRPO via Routed Off-mode Approach Mining

**Authors:** Jeff Duan, Matt Thomson
**arXiv:** 2611.25926 (cs.LG, cs.AI)
**Submitted:** ~20 November 2026
**Venue:** NeurIPS 2026

---

## Headline

GRPO fine-tuning collapses to a single reasoning mode; routing well-solved problems off the main learning track forces the policy to keep exploring new approaches, raising pass@16 by 4.7% and doubling the diversity of correct solutions on difficult math problems.

---

## Key Findings

- **Policy collapse** is a systematic failure of vanilla GRPO: once the model learns one approach to a problem class, it stops exploring alternatives.
- **ROAM-GRPO** detects when a problem is "well-solved" (high success rate across rollouts) and diverts it from the main gradient path to a lightweight "off-mode" track.
- On 1,000 episodes of MATH level 5 with Qwen3.5-2B: ROAM-GRPO raises peak pass@16 by **4.7%** over vanilla GRPO.
- Average cosine distance between correct solutions increases by **1.5–2×**, confirming that solutions are semantically more varied, not just surface-different.
- The off-mode track incurs negligible additional compute (< 3% overhead per training step).

---

## Motivation and Problem

Reinforcement learning from verifiable rewards (RLVR) — training LLMs to solve math and coding problems with binary correct/incorrect feedback — has become a dominant post-training paradigm. GRPO (Generalized Reward-weighted Policy Optimization) is a popular RLVR algorithm that trains by sampling multiple solutions and weighting updates by their rewards.

A key pathology of GRPO is **mode collapse**: once the model discovers a solution strategy that reliably scores reward, it over-trains on that strategy. The policy's sample distribution collapses to a single mode, losing the ability to explore alternate approaches. This is measured by solution diversity metrics like embedding cosine distance: diverse solutions cluster far from each other in embedding space; collapsed solutions cluster near a single centroid.

Mode collapse has two negative consequences:
1. **pass@k degrades**: even with many samples, the model re-draws from the same solution mode, so additional samples yield diminishing returns.
2. **Generalization suffers**: a policy that knows only one approach to a problem class fails when that approach doesn't work on novel instances.

---

## Method: ROAM-GRPO

### Detecting Well-Solved Problems

ROAM-GRPO tracks per-problem success rates across recent rollouts. A problem $q$ is classified as "well-solved" when its moving-average success rate exceeds a threshold $\tau$ (e.g., $\tau = 0.75$):

$$\text{WellSolved}(q) = \mathbb{1}\!\left[\frac{1}{K}\sum_{k=1}^{K} \mathbf{1}[r_k = 1] \geq \tau\right]$$

where $r_k \in \{0,1\}$ is the correctness of the $k$-th recent rollout for problem $q$.

### Off-Mode Track

Well-solved problems are diverted from the main GRPO gradient path. Instead of contributing to the standard policy update, they enter the "off-mode" track, which applies a **diversity-encouraging update**:

$$\mathcal{L}_{\text{off}}(\theta) = -\alpha_{\text{off}} \sum_{i \neq j} \cos\!\left(\mathbf{e}(\hat{y}_i), \mathbf{e}(\hat{y}_j)\right)$$

where $\mathbf{e}(\hat{y})$ is the embedding of solution $\hat{y}$ and the sum is over pairs of rollouts for the same problem. This pushes the policy to produce solutions that are semantically dissimilar from each other.

### Combined Objective

The full ROAM-GRPO objective at training step $t$ is:

$$\mathcal{J}(\theta) = \underbrace{\mathcal{J}_{\text{GRPO}}(\theta; \mathcal{D}_{\text{main}})}_{\text{main track}} + \lambda \underbrace{\mathcal{L}_{\text{off}}(\theta; \mathcal{D}_{\text{off}})}_{\text{off-mode track}}$$

where $\mathcal{D}_{\text{main}}$ contains unsolved or partially solved problems and $\mathcal{D}_{\text{off}}$ contains well-solved ones. The parameter $\lambda$ balances diversity against reward maximization.

---

## Technical Details

The routing decision is made per-minibatch, before gradient computation, so it adds no backward-pass overhead. The main track update is standard GRPO:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}_{\text{main}}}\!\left[\mathbb{E}_{\hat{y} \sim \pi_\theta}\!\left[\min\!\left(\frac{\pi_\theta(\hat{y}|q)}{\pi_{\theta_{\text{old}}}(\hat{y}|q)} A(\hat{y}, q),\, \text{clip}(\cdots) A(\hat{y}, q)\right)\right]\right]$$

where $A(\hat{y}, q) = r(\hat{y}, q) - b(q)$ is the advantage with baseline $b(q)$ estimated as the mean reward for problem $q$.

The off-mode diversity loss is computed only on well-solved problems and uses a frozen encoder for embedding computation, keeping the memory footprint manageable.

---

## Experimental Findings

Training was conducted on Qwen3.5-2B using 1,000 episodes of MATH level 5 problems.

| Metric | GRPO | ROAM-GRPO | Delta |
|---|---|---|---|
| pass@1 | 51.3% | 52.1% | +0.8% |
| pass@8 | 71.4% | 73.2% | +1.8% |
| pass@16 | 74.8% | 79.5% | +4.7% |
| Avg cosine distance | 0.31 | 0.56 | +81% |

The largest gains are at high $k$: ROAM-GRPO compounds diversity gains across samples, while vanilla GRPO has diminishing returns. The cosine distance result confirms that gains are from genuine reasoning diversity, not stochastic noise.

---

## Significance

ROAM-GRPO addresses a subtle but consequential failure mode in RLVR training: success on training problems inhibits exploration of diverse solution strategies. The method is lightweight (no extra model, < 3% compute overhead), broadly applicable to any GRPO-style training loop, and directly improves the capabilities most valuable in real deployments — high-$k$ sampling performance and solution diversity. The result connects to the broader question of how to maintain productive exploration in policy optimization as the policy improves.
