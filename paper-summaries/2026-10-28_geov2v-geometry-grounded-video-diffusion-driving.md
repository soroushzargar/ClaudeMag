# GeoV2V: Geometry-Grounded Video Diffusion Model for Driving Scene Generation

**Authors:** Yuchen Xi, Yuhan Guo, Chenwei Hou, Zixu Liu, Jin Fang, Jason Liu, Ruigang Yang  
**Venue:** ECCV 2026  
**arXiv:** 2610.25245  
**Date:** October 18, 2026  
**URL:** https://arxiv.org/abs/2610.25245

---

## Summary

Autonomous driving simulators need to synthesize realistic video from camera perspectives that were never recorded—for example, what the world would look like from an adjacent lane or from a slightly different heading angle. GeoV2V addresses this lane-shifting novel view synthesis problem with a geometry-grounded video diffusion model that combines LiDAR and depth-based geometric priors with full-video conditioning on the Wan 2.1 video diffusion backbone.

---

## Problem

Given a recorded driving video from a reference viewpoint (e.g., the ego-vehicle's center camera), generate a temporally consistent and geometrically accurate video from a target viewpoint (e.g., a simulated lane shift of 3.5 m). The challenge is that:

1. Large viewpoint shifts reveal occluded regions not visible in the reference video.
2. Temporal consistency across frames requires maintaining 3D coherent structure.
3. Lighting and shadow patterns must remain physically plausible.

---

## Method

### Geometric Grounding

GeoV2V lifts each reference frame to 3D using LiDAR point clouds (where available) or monocular depth estimation. It then renders a warped view at the target pose for each frame, producing a geometry-aware reference sequence that encodes which pixels correspond across views.

### Video Diffusion Backbone (Wan 2.1)

The warped reference sequence conditions the Wan 2.1 video diffusion transformer via cross-attention, providing spatiotemporal context throughout the generation process. The diffusion model learns to complete occluded regions and refine geometric artifacts while preserving large-scale structure.

### Para4D Dataset

The authors introduce Para4D, a synthetic dataset generated in the CARLA simulator containing synchronized multi-lane videos under diverse lighting conditions, weather, and traffic scenarios. Para4D provides explicit cross-lane supervision that real-world datasets lack, since real datasets only have fixed camera rigs.

---

## Key Results

- **Zero-shot transfer:** Trained entirely on synthetic Para4D data, GeoV2V achieves state-of-the-art performance on real-world Waymo and nuScenes lane-shift benchmarks.
- **Visual quality:** Improved FID and FVD scores compared to prior video-to-video diffusion baselines.
- **Geometric consistency:** Better homography error and LiDAR-projected depth consistency than geometric-warp-only baselines.
- **Robustness:** Maintains quality under challenging lighting conditions (night, glare, rain) where geometric priors are noisiest.

---

## Significance

GeoV2V demonstrates that synthetic-to-real transfer of geometry-grounded video generation is viable for autonomous driving simulation. The combination of explicit geometric conditioning and strong generative priors from large video diffusion models enables high-fidelity novel view synthesis that closes the gap between simulation and real-world conditions.

---

## Citation

Yuchen Xi, Yuhan Guo, Chenwei Hou, Zixu Liu, Jin Fang, Jason Liu, Ruigang Yang.  
**GeoV2V: Geometry-Grounded Video Diffusion Model for Driving Scene Generation.**  
arXiv:2610.25245. ECCV 2026.  
https://arxiv.org/abs/2610.25245
