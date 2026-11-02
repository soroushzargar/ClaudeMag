# Vision-Language Agents Can Learn in Their Own Latent Worlds

**arXiv:** 2610.36925  
**Authors:** Haoxi Li, Sikai Bai, Jianfei Ma, Qinglin Hou, Xiaocheng Lu, Tao Han, Jie Zhang, Song Guo  
**Submitted:** October 27, 2026  
**Venue:** arXiv preprint (cs.AI)  
**URL:** https://arxiv.org/abs/2610.36925

---

## Summary

Model-based reinforcement learning (MBRL) promises sample efficiency by letting an agent plan ahead in a learned world model rather than relying solely on environment interactions. For vision-language agents, the dominant approach to world modeling is to predict future frames—reconstructing pixel-level images of the next state—and then act in the imagined rollout. This is expensive: generating high-quality pixel predictions requires large decoders, and pixel fidelity may be orthogonal to what is actually useful for decision-making.

**Dwell** (Dreaming With Latent Embeddings for VLMs) proposes a different design: instead of predicting future frames, the VLM predicts the next latent state directly. The model is trained as a recurrent latent transition operator: given the current latent state and a proposed action, it predicts the distribution of the next latent state. A frozen V-JEPA2 visual encoder grounds the latent space, preventing collapse and providing rich visual features without requiring the world model to be invertible. The model then learns to act by imagining ahead in this latent space and selecting actions that lead to high-reward trajectories.

---

## Method

**Latent Grounding with V-JEPA2.** The latent states are not learned from scratch; they are frozen embeddings from V-JEPA2, a self-supervised vision encoder trained on video prediction. This provides a rich semantic representation without requiring the world model to be pixel-faithful.

**Recurrent Latent Transition Model.** A lightweight recurrent module takes the current latent state and an action embedding and outputs the distribution of the next latent state. The VLM backbone contributes both to encoding the instruction and to the recurrent prediction, enabling natural language goals to influence the imagined trajectory.

**Imagination and Policy Learning.** At training time, the agent performs real environment steps to collect experience, then trains the latent transition model on those transitions. The policy is then updated using imagined rollouts in latent space with a model-based actor-critic algorithm.

**No Pixel Decoder.** Unlike Dreamer and its variants, Dwell has no observation reconstruction objective. The world model is trained solely on latent prediction; this reduces compute by an estimated 2–3× compared to pixel-space world models of similar planning depth.

---

## Results

Evaluated across five interactive visual environments (including gridworld, robotic manipulation, and visual question-answering with feedback):

- Dwell achieves **best overall performance** among trainable methods across all five environments.
- Compared to pixel-space MBRL baselines, Dwell requires **2.3× less imagination compute** for equivalent planning depth.
- On tasks with sparse rewards, Dwell shows the largest margin over model-free baselines, consistent with the expected benefit of planning.
- On tasks where the VLM backbone provides strong prior knowledge, model-free fine-tuning is competitive; the benefit of Dwell's world model is more pronounced in novel environments.

---

## Significance

The key insight—that VLM world models need not be pixel-faithful—is a practical design principle. Prior pixel-space world models for RL have been computationally expensive; Dwell's design cuts this cost substantially while matching or exceeding their performance. The use of a frozen V-JEPA2 encoder is a pragmatic engineering choice that sidesteps the instabilities of end-to-end latent representation learning in MBRL, a known difficulty in the field. This work is likely to influence how the VLM-as-agent community approaches world modeling going forward.

---

## Limitations

- The five visual environments are relatively small-scale; evaluation on large open-world settings is left for future work.
- The frozen V-JEPA2 encoder may be suboptimal for tasks whose visual features differ significantly from V-JEPA2's pretraining distribution.
- Model-based training adds complexity; the latent transition model can mispredict, leading to compounding errors in long imagined rollouts.
