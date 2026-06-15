# SOTA Review — 360° parallax pipeline

**Date:** 2026-06-10
**Status:** documentation only — no changes were implemented.
**Question:** Does the 3D (parallax) effect use the state of the art in terms of performance/quality?

## Current stack (3 stages)

| Stage | File | Technique |
|---|---|---|
| Depth estimation (offline) | `parallax_360.py` | Depth-Anything-3 (`DA3MONO-LARGE`) → fallback DA-V2-Small → MiDaS-small → synthetic. Post-processing: guided filter, percentile stretch, gamma, pole attenuation. |
| LDI layers (offline) | `layered_360.py` | Depth slicing by thresholds (default `0.65 0.50 0.35` → 3 fg + 1 bg) with feather smoothstep, latitude gating, inpainted background (LaMa default / SDXL tiled opt-in / TELEA fallback) with wrap-pad for the 360° seam. |
| Runtime (web) | `app/components/parallax360/` | Concentric spheres: bg at radius 10, RGBA fg layers at radii 5→8; the camera offset produces real geometric parallax. Scene transitions: vertex displacement by depth map (CPU). |

## Verdict

**The chosen architecture (DA3 + LDI + LaMa) is at the right performance/quality frontier for a web tour.** No technical family change is needed. There are 2 concrete gaps with good ROI (one quality, one performance) and 3 minor details.

### What is already SOTA (verified June 2026)

- **DA3 remains the leading family in monocular depth** (ICLR 2026; outperforms previous SOTA VGGT by ~24-36% in geometric/pose precision). The choice of `DA3MONO-LARGE` for its disparity-type distribution is reasonable for the slicer.
- **LDI + concentric spheres is the correct family** for parallax from a single panorama on the web — same approach as Apple's spatial photos and Facebook's 3D photos. For sway of a few degrees (`MAX_OFFSET = 0.8` over radius 10), better performance/quality ratio than displaced mesh (rubber-sheet effect at discontinuities) and ~100× cheaper than gaussian splatting.
- **LaMa as the default for inpainting is the sweet spot**: inpainted areas are only revealed a few pixels at occlusion edges, so the extra diffusion quality is barely visible. SDXL tiled already exists as opt-in for difficult cases. FLUX.1 Fill is the current open SOTA, but the ROI here is marginal (12B params, slow on MPS).
- Well-solved details: horizontal wrap-pad for the seam, feather smoothstep between adjacent layers, inpaint mask dilation, correct `renderOrder` per layer, anisotropy, sRGB.

### The quality ceiling (intentionally discarded)

3D Gaussian Splatting from a single panorama (WorldGen / LayerPano3D / PanSplat-type pipeline: pano → navigable splats) gives real occlusions and view-dependent effects, but implies: heavy offline GPU generation (minutes), 30-100 MB payload, dedicated renderer (gsplat/Spark) and high complexity. For a few degrees of sway parallax it adds almost nothing visible vs LDI. **Not recommended for this use case.** Revisit only if free-roam inside the scene is ever desired.

## Quality gap #1 — perspective depth on equirectangular

DA3 is trained on perspective images; applied directly to the equirectangular pano it fails at the poles and high-distortion edges. The `pole_blend` / `floor_keep` in `postprocess_depth()` (`parallax_360.py:278`) is a heuristic patch for that problem, not a solution.

SOTA options (both offline, zero runtime cost):

1. **DAP — Depth Any Panoramas** (Insta360 Research, Dec 2025): native equirectangular foundation model, DINOv3-Large backbone, distortion-aware decoder, **metric** depth. Code and demo published (see references). This is exactly this use case. Note: since it emits metric depth, the slicer normalization would need to be adapted (same caveat already existing for `DA3Metric-LARGE` in the `--da3-model` help).
2. **"Depth Anywhere" technique without a new model**: project the pano onto cubemap faces (perspective), run each face through the current DA3, merge back to equirectangular with blending. More plumbing work, but keeps DA3 and the current disparity convention.

Expected benefit: clean depth on floor/ceiling → more precise slicing masks → fewer artifacts at layer edges, and would potentially allow relaxing `--layered-exclude-top/bottom`.

## Performance gap #1 — 28 MB of textures per scene

Each fg layer is a 4096×2048 RGBA PNG of **9.4 MB** (`public/parallax/image-1-360-fg{0,1,2}.png`) that stores the full RGB panorama even though ~90% of the area has alpha 0. Fixes in order of effort:

1. **Zero-out RGB where alpha = 0** before saving in `layered_360.py` (~1 line near `layered_360.py:507`), keeping a dilation ring of ~2 px around the edge so bilinear filtering does not sample black. The PNG compresses several times better.
2. **WebP with alpha** instead of PNG (~10-20× smaller; `THREE.TextureLoader` loads it natively in all current browsers). AVIF also works.
3. **KTX2/Basis Universal** if mobile matters: the 4 textures decompressed in GPU are ~170 MB of VRAM with mipmaps (4096×2048×4 bytes ≈ 32 MB each + mips); KTX2 stays compressed in VRAM too.

## Minor items

- **Fixed layer radii** (`MIN_FG_RADIUS = 5` → `MAX_FG_RADIUS = 8`, `constants.ts:28`, linear distribution by index): the parallax magnitude per layer is arbitrary, not proportional to real depth. Better: derive the radius from the median depth of each slab — `layered_360.py` already knows the distribution and could emit the suggested `radius` in the snippet it prints for `app/scenes.ts`.
- **Transition displacement on CPU**: `setDisplacementStrength()` (`displacement.ts:112`) iterates ~131k vertices in JS and re-uploads the buffer per frame during the transition (1.2 s). The standard practice is to displace in a vertex shader sampling the depth texture (`onBeforeCompile` on `MeshBasicMaterial`). Low impact — only during transitions — but would free CPU on mobile and allow per-pixel displacement.
- **CLAUDE.md outdated**: says `ENABLE_PARALLAX` is `false`, but it is `true` today (`constants.ts:5`). Fix when the doc is next touched.

## Prioritized recommendation (pending decision)

| # | Improvement | Impact | Effort |
|---|---|---|---|
| 1 | Assets: RGB zero-out + WebP (and optional KTX2) | High performance (28 MB → ~2 MB per scene) | Low |
| 2 | Native panoramic depth (try DAP; alternative cubemap+DA3) | High quality on poles/layer edges | Medium |
| 3 | Radii derived from median depth per slab | Medium quality (faithful parallax) | Low |
| 4 | Displacement in vertex shader | Low-medium performance (transitions only) | Medium |

## References

- [Depth Anything 3 — project page](https://depth-anything-3.github.io/) · [paper (arXiv 2511.10647)](https://arxiv.org/abs/2511.10647)
- [Depth Any Panoramas (DAP) — project page](https://insta360-research-team.github.io/DAP_website/) · [GitHub](https://github.com/Insta360-Research-Team/DAP) · [paper (arXiv 2512.16913)](https://arxiv.org/abs/2512.16913)
- [PanDA: Towards Panoramic Depth Anything (CVPR 2025)](https://caozidong.github.io/PanDA_Depth/)
- [Depth Anywhere (NeurIPS 2024) — cubemap distillation for 360°](https://arxiv.org/abs/2406.12849)
- [PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting](https://chengzhag.github.io/publication/pansplat/)
- [360-GS: Layout-guided Panoramic Gaussian Splatting](https://arxiv.org/pdf/2402.00763)
- [LaMa: Resolution-robust Large Mask Inpainting](https://arxiv.org/pdf/2109.07161)
