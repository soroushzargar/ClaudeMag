# See like a Robot: Robot-Centric Pointmaps for Vision-Language-Action Models

**Authors:** Byungkun Lee, Dongyoon Hwang, Dongjin Kim, Hojoon Lee, Minho Park, Jaegul Choo  
**Affiliation:** KAIST AI; Holiday Robotics  
**Venue:** arXiv:2607.11498  
**Date:** July 13, 2026  
**URL:** https://arxiv.org/abs/2607.11498

---

## Summary

Vision-language-action (VLA) models predict robot actions in the robot's 3D body frame while observing scenes through camera images in the camera frame. When camera configurations are fixed, this frame mismatch is manageable—the policy can implicitly memorize the mapping. When demonstrations come from diverse camera setups (as in large-scale robot learning datasets), the mapping changes across demonstrations and the policy must generalize across viewpoints. Robot-centric pointmaps—images whose pixels store 3D scene coordinates in the robot frame—bridge this gap, providing robot-frame geometry in a format compatible with existing 2D VLA architectures.

## Problem

Large-scale robot learning aggregates demonstrations from many operators, environments, and camera rigs. Standard VLA inputs are camera-frame RGB images: each pixel stores appearance information, but the coordinate system is that of the camera, not the robot. Robot actions (joint positions, end-effector poses, velocity commands) are defined in the robot's body frame, creating a systematic mismatch.

Under a fixed camera, the robot can learn the camera-to-robot transformation implicitly by memorizing visual context. But across diverse camera setups:
- **Viewpoint generalization fails:** A policy trained on overhead cameras fails when deployed with a wrist-mounted camera.
- **Geometry is implicit:** Spatial relationships between objects (distance, height, clearance) must be inferred from monocular cues.
- **Large-scale aggregation is hindered:** Cross-dataset diversity is a key driver of policy generalization, but it amplifies the viewpoint mismatch problem.

## Method: Robot-Centric Pointmaps

A **robot-centric pointmap** $P \in \mathbb{R}^{H \times W \times 3}$ is an image where each pixel $(u, v)$ stores the 3D position $(X, Y, Z)$ of the corresponding scene point in the robot's coordinate frame. Constructing $P$ requires:

1. **RGB-D input:** Depth from a depth sensor or monocular depth estimator.
2. **Intrinsic calibration:** Camera intrinsic matrix $K$ to unproject pixels to 3D.
3. **Extrinsic calibration:** Camera-to-robot transform $T_{cam \to robot}$ to rotate and translate 3D points into the robot frame.

The resulting pointmap is a dense, spatially structured representation of the scene in the robot's own coordinate system—making object distances, heights, and reach distances directly readable from pixel values.

## Integration with VLA Architectures

Pointmaps preserve the $H \times W$ spatial grid that VLA architectures (typically built on pretrained vision encoders like ViT or SigLIP) expect as input. The pointmap is concatenated channel-wise with the RGB image:

$$I_\text{combined} = [I_\text{RGB}, P] \in \mathbb{R}^{H \times W \times 6}$$

The first convolutional layer is adapted to accept 6-channel input (re-initializing the extra channels) while keeping all other weights pretrained. No other architectural changes are required, preserving the pretrained visual features while adding robot-frame geometric grounding.

## Technical Formulation

For pixel $(u, v)$ with depth $d_{u,v}$, the backprojection to camera frame is:

$$\mathbf{p}_{cam} = d_{u,v} K^{-1} [u, v, 1]^\top$$

Transformation to robot frame:

$$\mathbf{p}_{robot} = R_{cam \to robot} \mathbf{p}_{cam} + \mathbf{t}_{cam \to robot}$$

The pointmap $P_{u,v} = \mathbf{p}_{robot}$ stores the result. For real-time use, this computation runs at >60 Hz on standard GPU hardware.

When depth is unavailable, the authors show that off-the-shelf monocular depth estimators (e.g., Depth Anything V2) followed by scale-and-shift alignment using known anchor points yield pointmaps of sufficient quality for the policy.

## Key Results

Experiments use the Open X-Embodiment dataset for training and evaluate on cross-viewpoint generalization (unseen camera angles) and real-robot tasks.

**Cross-viewpoint accuracy on Open X-Embodiment (task success ↑):**
| Model | Same-camera | Cross-camera |
|-------|-------------|--------------|
| $\pi_0$ (RGB only) | 74.3 | 41.2 |
| $\pi_0$ + depth (camera frame) | 77.1 | 46.8 |
| $\pi_0$ + pointmap (robot frame) | **80.6** | **61.4** |

The largest gain is on cross-camera evaluation (+20pp), confirming that robot-frame grounding directly addresses the viewpoint generalization problem.

**Real-robot evaluation (6-DOF manipulation, 50 trials each):**
| Task | RGB only | + Pointmap |
|------|---------|------------|
| Pick from varied height | 42% | 72% |
| Place on varied surface | 58% | 84% |
| Reach past obstacle | 34% | 64% |

## Ablation

- **Camera-frame 3D vs. robot-frame:** Using 3D in the camera frame yields only 51.3% cross-camera success vs. 61.4% for robot-frame, confirming that coordinate system matters more than the presence of 3D information.
- **Monocular vs. sensor depth:** Sensor depth improves cross-camera success by 4.2pp over monocular estimation, suggesting monocular estimation is a viable practical substitute.
- **Channel fusion vs. early fusion:** Channel-wise concatenation outperforms early-layer feature fusion by 2.8pp, likely because it preserves independent access to RGB appearance and geometric channels.

## Significance

Robot-centric pointmaps provide a simple, deployment-ready solution to the frame mismatch problem in large-scale robot learning. The zero-overhead integration cost makes them an immediate upgrade for any VLA-based system where camera viewpoint varies between training and deployment.
