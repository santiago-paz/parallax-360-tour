#!/usr/bin/env python3
"""
layered_360.py — Layered Depth Image (LDI) generator for the 360° viewer

Takes an equirectangular panorama + its depth map (already produced by
`parallax_360.py`) and slices the scene into N foreground layers + one
inpainted background:

  1. {stem}-fg0.png … {stem}-fgK.png   RGBA layers ordered closest first
                                       (alpha = smoothed depth slab mask)
  2. {stem}-bg.jpeg                    Background with all fg regions
                                       inpainted away
  3. {stem}-fg-mask.png                Debug: union mask used for inpainting

At runtime the viewer renders the background on the outer sphere and
each foreground layer on its own smaller concentric sphere, so a small
camera offset produces *real* parallax with a different angular shift
per layer. More layers ⇒ more elements that pop out at distinct depths.

Convention reminder: depth PNGs in this project follow the proximity /
disparity-like convention — bright = near, dark = far. Thresholds are
given in DESCENDING order (nearest first), e.g.

    --thresholds 0.60 0.40

produces three depth slabs: (depth>0.60, 0.40<depth≤0.60, depth≤0.40).
The closest two become fg0/fg1; the farthest becomes the background.

Usage
    python layered_360.py public/image-1-360.jpeg \\
        --depth public/parallax/depth_image-1-360.png \\
        --thresholds 0.60 0.40

Requirements: numpy, opencv-python(-headless). No torch/model download.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path as _Path


SAM2_VENDOR_DIR = _Path(__file__).resolve().parent / "vendor" / "sam2"
SAM2_DEFAULT_CHECKPOINT = SAM2_VENDOR_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class SamObject:
    """One coherent object detected by SAM 2 on the equirect."""
    mask: np.ndarray            # bool (H, W)
    area: int                    # pixels
    stability: float             # SAM stability score
    median_depth: float = 0.0    # filled in by assign_depth_to_objects
    median_metric: float | None = None


def is_sam_available() -> bool:
    """Cheap check: SAM 2 package importable AND checkpoint file on disk."""
    if not SAM2_DEFAULT_CHECKPOINT.is_file():
        return False
    try:
        import sam2  # noqa: F401
    except ImportError:
        return False
    return True


_SAM2_GENERATOR_CACHE: dict = {}


def _load_sam2_generator(device: str):
    """Load SAM 2 Automatic Mask Generator with strict defaults; cache per process."""
    if device in _SAM2_GENERATOR_CACHE:
        return _SAM2_GENERATOR_CACHE[device]
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam_model = build_sam2(SAM2_DEFAULT_MODEL_CFG, str(SAM2_DEFAULT_CHECKPOINT),
                           device=device, apply_postprocessing=False)
    gen = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=32,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=0,
    )
    _SAM2_GENERATOR_CACHE[device] = gen
    return gen


def merge_seam_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Merge boolean masks that both touch the left edge (col 0) and the right edge (col W-1).

    Used after SAM runs on horizontally wrap-padded input: an object straddling the
    panorama seam ends up as two disjoint masks, one on each edge. We pair them and OR.
    """
    if not masks:
        return masks
    width = masks[0].shape[1]
    touches_left: list[int] = []
    touches_right: list[int] = []
    for i, m in enumerate(masks):
        if m[:, 0].any():
            touches_left.append(i)
        if m[:, width - 1].any():
            touches_right.append(i)
    only_left = [i for i in touches_left if i not in touches_right]
    only_right = [i for i in touches_right if i not in touches_left]
    if not only_left or not only_right:
        return masks
    merged_indices: set[int] = set()
    out_masks: list[np.ndarray] = []
    for li in only_left:
        if li in merged_indices:
            continue
        left = masks[li]
        best_ri = None
        best_overlap = 0
        for ri in only_right:
            if ri in merged_indices:
                continue
            overlap = int(((left.any(axis=1)) & (masks[ri].any(axis=1))).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_ri = ri
        if best_ri is not None and best_overlap > 0:
            out_masks.append(left | masks[best_ri])
            merged_indices.add(li)
            merged_indices.add(best_ri)
    for i, m in enumerate(masks):
        if i not in merged_indices:
            out_masks.append(m)
    return out_masks


def sam_segment_objects(
    img_bgr: np.ndarray,
    device: str = "cpu",
) -> list[SamObject]:
    """Run SAM 2 "everything" mode on the equirect (with horizontal wrap-pad)
    and return a list of `SamObject`. Raises ImportError if SAM 2 is unavailable.
    """
    if not is_sam_available():
        raise ImportError(
            "SAM 2 is not available. Install with "
            "`pip install git+https://github.com/facebookresearch/sam2.git` "
            f"and download weights to {SAM2_DEFAULT_CHECKPOINT}."
        )
    h, w = img_bgr.shape[:2]
    pad = max(64, w // 32)
    padded = horizontal_wrap_pad(img_bgr, pad)

    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    gen = _load_sam2_generator(device)
    raw = gen.generate(img_rgb)

    unpadded_masks: list[np.ndarray] = []
    scores: list[float] = []
    for entry in raw:
        m_padded = entry["segmentation"].astype(bool)
        m = m_padded[:, pad:pad + w]
        if not m.any():
            continue
        unpadded_masks.append(m)
        scores.append(float(entry.get("stability_score", entry.get("predicted_iou", 0.0))))

    merged = merge_seam_masks(unpadded_masks)
    out: list[SamObject] = []
    for m in merged:
        best_score = 0.0
        for om, s in zip(unpadded_masks, scores):
            inter = int((m & om).sum())
            if inter == 0:
                continue
            union = int((m | om).sum())
            if union > 0 and inter / union > 0.5:
                if s > best_score:
                    best_score = s
        out.append(SamObject(mask=m, area=int(m.sum()), stability=best_score))
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    if inter == 0:
        return 0.0
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def dedup_and_filter(
    objects: list[SamObject],
    gate_2d: np.ndarray,
    *,
    max_area_fraction: float = 0.30,
    min_in_gate_fraction: float = 0.60,
    iou_nms: float = 0.70,
    containment_iou: float = 0.85,
) -> list[SamObject]:
    """Drop spurious masks and de-duplicate, keeping object-like detections.

    1. Drop masks > max_area_fraction of image area.
    2. Drop masks with < min_in_gate_fraction of their area inside the latitude gate
       (gate_2d > 0.5 counts as "in gate").
    3. NMS by IoU > iou_nms: keep the more stable.
    4. Containment: A inside B (IoU(A, A∩B)/|A| > containment_iou) → keep B.
    5. Multiply every surviving mask by (gate_2d > 0) so the soft gate is respected
       downstream; the SamObject.mask is updated in place to the gated mask.
    """
    h, w = gate_2d.shape
    total = h * w
    gate_bool = gate_2d > 0.5

    survivors = [o for o in objects if o.area / total <= max_area_fraction]

    kept: list[SamObject] = []
    for o in survivors:
        in_gate = int((o.mask & gate_bool).sum())
        if o.area == 0:
            continue
        if in_gate / o.area >= min_in_gate_fraction:
            kept.append(o)
    survivors = kept

    survivors.sort(key=lambda o: o.stability, reverse=True)
    nms_out: list[SamObject] = []
    for o in survivors:
        if any(_iou(o.mask, k.mask) > iou_nms for k in nms_out):
            continue
        nms_out.append(o)

    contain_out: list[SamObject] = []
    for i, a in enumerate(nms_out):
        contained = False
        for j, b in enumerate(nms_out):
            if i == j:
                continue
            inter = int((a.mask & b.mask).sum())
            if a.area == 0:
                continue
            if inter / a.area > containment_iou and b.area > a.area:
                contained = True
                break
        if not contained:
            contain_out.append(a)

    for o in contain_out:
        o.mask = o.mask & gate_bool
        o.area = int(o.mask.sum())
    return [o for o in contain_out if o.area > 0]


def assign_depth_to_objects(
    objects: list[SamObject],
    depth: np.ndarray,
    metric_depth: np.ndarray | None,
) -> None:
    """In-place: fill `median_depth` (and `median_metric` if available) per object."""
    for o in objects:
        if o.area == 0:
            o.median_depth = 0.0
            o.median_metric = None
            continue
        o.median_depth = float(np.median(depth[o.mask]))
        if metric_depth is not None:
            valid = o.mask & np.isfinite(metric_depth)
            o.median_metric = float(np.median(metric_depth[valid])) if valid.any() else None
        else:
            o.median_metric = None


def bin_objects_by_depth(
    objects: list[SamObject],
    *,
    k: int = 3,
    bg_threshold: float = 0.20,
) -> tuple[list[list[SamObject]], list[SamObject]]:
    """Split objects into K foreground bins (K-means in 1D over median_depth) + background.

    Returns (fg_bins, bg_objects). fg_bins[0] is the closest cluster (highest median_depth).
    If fg_candidates is empty, returns ([], bg_objects). If fewer fg candidates than K,
    K is reduced to that count (no empty bins).
    """
    fg_candidates = [o for o in objects if o.median_depth >= bg_threshold]
    bg_objects = [o for o in objects if o.median_depth < bg_threshold]
    if not fg_candidates:
        return [], bg_objects

    effective_k = min(k, len(fg_candidates))
    if effective_k == 1:
        return [list(fg_candidates)], bg_objects

    from sklearn.cluster import KMeans

    medians = np.array([[o.median_depth] for o in fg_candidates], dtype=np.float64)
    km = KMeans(n_clusters=effective_k, random_state=0, n_init=10)
    labels = km.fit_predict(medians)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(-centers)
    id_to_position = {int(cid): pos for pos, cid in enumerate(order.tolist())}

    bins: list[list[SamObject]] = [[] for _ in range(effective_k)]
    for obj, lbl in zip(fg_candidates, labels):
        bins[id_to_position[int(lbl)]].append(obj)
    return bins, bg_objects


def build_alpha_masks(
    fg_bins: list[list[SamObject]],
    gate_2d: np.ndarray,
    image_shape: tuple[int, int],
    feather_px: int,
) -> list[np.ndarray]:
    """For each bin, build a float32 [0,1] alpha = union of its objects, with pixels
    that also belong to a closer bin removed (E7), gated, and edge-feathered.

    fg_bins[0] is the closest bin.
    """
    h, w = image_shape
    bin_unions: list[np.ndarray] = []
    for bin_objs in fg_bins:
        u = np.zeros((h, w), dtype=bool)
        for o in bin_objs:
            u |= o.mask
        bin_unions.append(u)

    cumulative_closer = np.zeros((h, w), dtype=bool)
    resolved: list[np.ndarray] = []
    for u in bin_unions:
        resolved.append(u & ~cumulative_closer)
        cumulative_closer |= u

    out: list[np.ndarray] = []
    for r in resolved:
        alpha = r.astype(np.float32) * gate_2d.astype(np.float32)
        if feather_px > 0:
            k = 2 * feather_px + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        out.append(np.clip(alpha, 0.0, 1.0).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Spinner for long steps with no intrinsic progress signal (LaMa, WebP encoders).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _spinner(label: str, frame_interval: float = 0.1, plain_period: float = 5.0):
    """
    Animates `label ⠋ Ns` with \\r on TTY; on pipe/no-TTY prints every `plain_period`s.
    Clears the line on exit and reports total elapsed time.

    Designed to wrap a blocking call with no progress signal (LaMa forward,
    WebP encode). Spawns a daemon thread that ticks; the yield happens without
    blocking on the thread.
    """
    t0 = time.time()
    stop = threading.Event()
    tty = sys.stdout.isatty()

    def _tty_run() -> None:
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not stop.wait(frame_interval):
            elapsed = int(time.time() - t0)
            sys.stdout.write(f"\r{label} {frames[i % len(frames)]} {elapsed}s ")
            sys.stdout.flush()
            i += 1

    def _plain_run() -> None:
        while not stop.wait(plain_period):
            print(f"{label} … {int(time.time() - t0)}s elapsed", flush=True)

    t = threading.Thread(target=_tty_run if tty else _plain_run, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=0.5)
        if tty:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Mask & inpaint
# ---------------------------------------------------------------------------

def latitude_band_mask(
    height: int,
    width: int,
    exclude_top: float,
    exclude_bottom: float,
) -> np.ndarray:
    """
    Build a per-row gate in [0,1] that suppresses foreground inside the
    top `exclude_top` fraction (ceiling/sky) and the bottom `exclude_bottom`
    fraction (floor immediately below the camera) of the equirectangular
    image.

    Indoor floors and ceilings have huge depth gradients running from
    near (right under the camera) to far (against the wall). A binary
    near/far threshold cuts those surfaces in half, which would float
    the camera-side chunk as a giant LDI layer and look much worse than
    leaving the whole surface on the background sphere.

    Both regions use a short cosine fade so the cut doesn't show as a
    hard horizontal line.
    """
    gate = np.ones(height, dtype=np.float32)
    if exclude_top > 0:
        rows = int(round(height * exclude_top))
        fade = max(1, int(round(rows * 0.25)))
        gate[:rows - fade] = 0.0
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
        gate[rows - fade:rows] = ramp
    if exclude_bottom > 0:
        rows = int(round(height * exclude_bottom))
        fade = max(1, int(round(rows * 0.25)))
        gate[height - rows + fade:] = 0.0
        ramp = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade))
        gate[height - rows:height - rows + fade] = ramp
    return np.broadcast_to(gate[:, None], (height, width)).copy()


def _smoothstep(depth: np.ndarray, center: float, feather: float) -> np.ndarray:
    """Smoothstep going 0→1 around `center` with half-width `feather`."""
    lo = max(0.0, center - feather)
    hi = min(1.0, center + feather)
    t = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def make_layered_masks(
    depth: np.ndarray,
    thresholds_desc: list[float],
    feather: float,
    exclude_top: float,
    exclude_bottom: float,
    *,
    img_bgr: np.ndarray | None = None,
    metric_depth: np.ndarray | None = None,
    use_sam: bool = False,
    sam_k: int = 3,
    sam_bg_threshold: float = 0.20,
    sam_device: str = "cpu",
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Slice the depth map into N foreground layers + one background.

    Two paths:
      1. Default (use_sam=False or SAM unavailable / no img_bgr): the existing
         depth-threshold + smoothstep slabs.
      2. SAM (use_sam=True with img_bgr provided and SAM installed): each SAM object
         is assigned whole to a K-means bin over its median depth; alphas are unions
         of bin objects with closer-bin subtraction.

    Both paths return (layer_alphas, union_hard) with the same dtype/shape contract:
      - layer_alphas : list[float32 (H,W) in [0,1]], closest first.
      - union_hard   : uint8 (0/255), foreground footprint for inpainting.
    """
    h, w = depth.shape
    gate = latitude_band_mask(h, w, exclude_top, exclude_bottom)

    # ── SAM path ────────────────────────────────────────────────────────────
    if use_sam and img_bgr is not None and is_sam_available():
        print("[layered] SAM enabled — segmenting objects …")
        _t0 = time.time()
        raw = sam_segment_objects(img_bgr, device=sam_device)
        print(f"[layered]   SAM found {len(raw)} raw masks")
        kept = dedup_and_filter(raw, gate)
        print(f"[layered]   After dedup/filter: {len(kept)} objects")
        assign_depth_to_objects(kept, depth, metric_depth)
        fg_bins, bg_objs = bin_objects_by_depth(
            kept, k=sam_k, bg_threshold=sam_bg_threshold,
        )
        if not fg_bins:
            print("[layered]   SAM produced 0 fg bins; falling back to threshold slicing.")
        else:
            centers = [
                float(np.median([o.median_depth for o in b])) if b else 0.0
                for b in fg_bins
            ]
            print(f"[layered]   K-means K={len(fg_bins)} → "
                  f"bin centers (median_depth): {[round(c, 2) for c in centers]}")
            for k, (b, c) in enumerate(zip(fg_bins, centers)):
                print(f"[layered]   bin {k}: {len(b)} objects, median_depth={c:.2f}")
            print(f"[layered]   bg: {len(bg_objs)} objects (median_depth < {sam_bg_threshold})")
            feather_px = max(2, w // 1024)
            alphas = build_alpha_masks(
                fg_bins, gate, image_shape=(h, w), feather_px=feather_px,
            )
            union_hard = np.zeros((h, w), dtype=np.uint8)
            for a in alphas:
                union_hard[a > 0.5] = 255
            kernel = np.ones((5, 5), np.uint8)
            union_hard = cv2.morphologyEx(union_hard, cv2.MORPH_OPEN, kernel)
            union_hard = cv2.morphologyEx(union_hard, cv2.MORPH_CLOSE, kernel)
            print(f"[layered]   SAM done in {time.time() - _t0:.1f}s")
            return alphas, union_hard
    elif use_sam:
        print("[layered] --sam requested but SAM not available; "
              "falling back to depth-threshold slicing.")

    # ── Default path: existing depth-threshold slabs ────────────────────────
    layer_alphas: list[np.ndarray] = []
    for k, t in enumerate(thresholds_desc):
        near_edge = _smoothstep(depth, t, feather)
        if k == 0:
            alpha = near_edge
        else:
            t_prev = thresholds_desc[k - 1]
            far_edge = _smoothstep(depth, t_prev, feather)
            alpha = near_edge * (1.0 - far_edge)
        layer_alphas.append((alpha * gate).astype(np.float32))

    t_last = thresholds_desc[-1]
    union_hard = ((depth >= t_last) & (gate > 0.5)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    union_hard = cv2.morphologyEx(union_hard, cv2.MORPH_OPEN, kernel)
    union_hard = cv2.morphologyEx(union_hard, cv2.MORPH_CLOSE, kernel)
    return layer_alphas, union_hard


def compute_layer_radii(
    metric_depth: np.ndarray,
    layer_alphas: list[np.ndarray],
    union_hard: np.ndarray,
    outer_radius: float = 10.0,
    r_min: float = 2.5,
    r_max: float = 9.0,
) -> list[float | None]:
    """
    Sphere radius per slab, proportional to real-world distance.

    The runtime renders the bg at `outer_radius` (RADIUS=10 in constants.ts) and
    each fg layer on an inner sphere. The angular parallax of a sphere is
    ∝ 1/radius, so that the per-layer magnitude is faithful to the scene:

        r_k = outer_radius × (median_metric_slab_k / median_metric_bg)

    Uses the ratio of medians (robust to outliers like windows/mirrors) and is
    invariant to the global scale of the depth model. Clamped to [r_min, r_max]
    and strictly increasing order (slabs go from nearest to farthest;
    if noise inverted them, the fixed renderOrder of the viewer would
    draw layers in an order inconsistent with their depth). The ordering
    step skips None slabs and may push near layers slightly below r_min
    to preserve order.

    Returns None for slabs with no pixels (alpha ≤ 0.5) or if there is no bg.
    """
    bg_sel = (union_hard == 0) & np.isfinite(metric_depth)
    if not bg_sel.any():
        return [None] * len(layer_alphas)
    bg_median = float(np.median(metric_depth[bg_sel]))
    if bg_median <= 0:
        return [None] * len(layer_alphas)

    radii: list[float | None] = []
    for alpha in layer_alphas:
        sel = (alpha > 0.5) & np.isfinite(metric_depth)
        if not sel.any():
            radii.append(None)
            continue
        med = float(np.median(metric_depth[sel]))
        radii.append(min(max(outer_radius * (med / bg_median), r_min), r_max))

    nxt: float | None = None
    for k in range(len(radii) - 1, -1, -1):
        if radii[k] is None:
            continue
        if nxt is not None:
            radii[k] = min(radii[k], nxt - 0.2)
        nxt = radii[k]

    return [None if r is None else round(r, 1) for r in radii]


def zero_out_rgb(rgb: np.ndarray, alpha: np.ndarray, dilate_px: int = 2) -> np.ndarray:
    """Set RGB=0 outside a `dilate_px`-pixel ring around any non-transparent pixel.

    Keeps the ring so bilinear sampling near the alpha edge does not bleed
    black from outside. ~90% of LDI layer area is alpha=0, so zeroing those
    RGB triples compresses ~3-5× better in WebP/PNG.
    """
    keep = (alpha > 0).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        keep = cv2.dilate(
            keep,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )
    rgb_out = rgb.copy()
    rgb_out[keep == 0] = 0
    return rgb_out


def horizontal_wrap_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """Mirror the left/right edges so cv2.inpaint sees the 360° wraparound."""
    left = img[:, -pad:]
    right = img[:, :pad]
    return np.concatenate([left, img, right], axis=1)


def horizontal_wrap_unpad(img: np.ndarray, pad: int) -> np.ndarray:
    return img[:, pad:-pad]


def _inpaint_telea(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """OpenCV TELEA inpainting — cheap, blurry, no model download required."""
    return cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA)


_LAMA_MODEL_CACHE: dict = {}


def _load_lama_model_cpu():
    """
    Load the LaMa torchscript model with map_location='cpu'.

    The simple_lama_inpainting wrapper does `torch.jit.load(path)` without
    a map_location, so the traced model carries its original CUDA tensor
    bindings and blows up with `aten::empty_strided on CUDA backend` on
    machines without CUDA (Apple Silicon Macs in particular). Loading with
    map_location='cpu' rebinds those tensors so we can run on the CPU.
    Cached so we only pay the load cost once per process.
    """
    if "model" in _LAMA_MODEL_CACHE:
        return _LAMA_MODEL_CACHE["model"]
    import torch
    from simple_lama_inpainting.utils import download_model
    from simple_lama_inpainting.models.model import LAMA_MODEL_URL
    model_path = download_model(LAMA_MODEL_URL)
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    _LAMA_MODEL_CACHE["model"] = model
    return model


def _inpaint_lama(
    img_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    LaMa inpainting via simple-lama-inpainting (CPU-pinned loader).

    Far better than TELEA for indoor scenes with large foreground objects
    (chairs, beds) — produces plausible texture continuation instead of
    radial colour smear. Lazy import so the module remains optional;
    raises ImportError if not installed and lets the caller fall back to
    TELEA.
    """
    import torch
    from PIL import Image
    from simple_lama_inpainting.utils import prepare_img_and_mask

    model = _load_lama_model_cpu()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask).convert("L")

    device = torch.device("cpu")
    image_t, mask_t = prepare_img_and_mask(pil_img, pil_mask, device)
    with torch.inference_mode():
        out_t = model(image_t, mask_t)
    out_np = out_t[0].permute(1, 2, 0).detach().cpu().numpy()
    out_rgb_arr = np.clip(out_np * 255, 0, 255).astype(np.uint8)

    # LaMa may pad up to a multiple of 8; crop back to the original size.
    if out_rgb_arr.shape[:2] != img_bgr.shape[:2]:
        out_rgb_arr = cv2.resize(
            out_rgb_arr, (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(out_rgb_arr, cv2.COLOR_RGB2BGR)


_SDXL_PIPE_CACHE: dict = {}


def _inpaint_sdxl(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    num_inference_steps: int = 30,
    guidance_scale: float = 8.0,
) -> np.ndarray:
    """
    Stable Diffusion XL Inpainting (1024-native). Much higher quality than
    LaMa on large irregular holes thanks to learnt scene priors, while still
    fitting on a 24GB M2 (≈7GB weights at float16). Tiles the 4K panorama
    into `tile_size` patches with `tile_overlap` blend zones so each forward
    pass is a comfortable 1024² (its training resolution).

    Caches the pipeline globally so successive runs reuse the loaded weights.
    """
    import torch
    from PIL import Image

    if "pipe" not in _SDXL_PIPE_CACHE:
        from diffusers import StableDiffusionXLInpaintPipeline
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[layered]   Loading SDXL Inpainting on {device} (first-run download ~7GB) …")
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = pipe.to(device)
        # Slight VRAM win at no quality cost on MPS.
        pipe.enable_attention_slicing()
        _SDXL_PIPE_CACHE["pipe"] = pipe
        _SDXL_PIPE_CACHE["device"] = device

    pipe = _SDXL_PIPE_CACHE["pipe"]
    h, w = img_bgr.shape[:2]

    out = img_bgr.copy()
    stride = tile_size - tile_overlap
    n_tiles_y = max(1, (h + stride - 1) // stride)
    n_tiles_x = max(1, (w + stride - 1) // stride)
    total = n_tiles_y * n_tiles_x
    done = 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            done += 1
            y0 = min(ty * stride, h - tile_size)
            x0 = min(tx * stride, w - tile_size)
            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile_mask = mask[y0:y1, x0:x1]
            if tile_mask.max() == 0:
                continue  # nothing to inpaint here, skip
            print(f"[layered]   SDXL tile {done}/{total}  rect=({x0},{y0})-({x1},{y1}) …")
            tile_rgb = img_rgb[y0:y1, x0:x1]
            pil_tile = Image.fromarray(tile_rgb)
            pil_mask = Image.fromarray(tile_mask).convert("L")
            with torch.inference_mode():
                result = pipe(
                    prompt="empty room, plain interior, photorealistic",
                    negative_prompt="furniture, people, text, watermark",
                    image=pil_tile,
                    mask_image=pil_mask,
                    height=tile_rgb.shape[0],
                    width=tile_rgb.shape[1],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            res_arr = np.array(result)
            if res_arr.shape[:2] != tile_mask.shape:
                res_arr = cv2.resize(
                    res_arr, (tile_mask.shape[1], tile_mask.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            # Soft-blend at tile borders (only blend where mask non-zero).
            blend_radius = max(1, tile_overlap // 2)
            blend = np.ones(tile_mask.shape, dtype=np.float32)
            if x0 > 0:
                ramp = np.linspace(0, 1, blend_radius, dtype=np.float32)
                blend[:, :blend_radius] = np.minimum(blend[:, :blend_radius], ramp[None, :])
            if y0 > 0:
                ramp = np.linspace(0, 1, blend_radius, dtype=np.float32)
                blend[:blend_radius, :] = np.minimum(blend[:blend_radius, :], ramp[:, None])
            if x1 < w:
                ramp = np.linspace(1, 0, blend_radius, dtype=np.float32)
                blend[:, -blend_radius:] = np.minimum(blend[:, -blend_radius:], ramp[None, :])
            if y1 < h:
                ramp = np.linspace(1, 0, blend_radius, dtype=np.float32)
                blend[-blend_radius:, :] = np.minimum(blend[-blend_radius:, :], ramp[:, None])
            mask_f = (tile_mask.astype(np.float32) / 255.0) * blend
            mask_f3 = mask_f[..., None]
            out_tile_rgb = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            blended = (mask_f3 * res_arr + (1 - mask_f3) * out_tile_rgb).astype(np.uint8)
            out[y0:y1, x0:x1] = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    return out


def inpaint_background(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    dilate_px: int,
    radius: int,
    backend: str = "auto",
) -> tuple[np.ndarray, str]:
    """
    Inpaint the foreground region. The mask is dilated first so colour from
    the foreground edge doesn't bleed into the synthesised background, and
    the image is horizontally wrap-padded so foreground objects straddling
    the panorama seam are inpainted with a continuous neighbourhood instead
    of black borders.

    `backend` selects the inpainter:
      - "sdxl"  : Stable Diffusion XL Inpainting (~7GB, scene-aware, MPS-friendly)
      - "lama"  : LaMa (good, fast, no auth, ~200MB)
      - "telea" : cv2.inpaint TELEA (fast, blurry, always available)
      - "auto"  : prefer LaMa, fall back to TELEA on ImportError

    Returns (inpainted_bgr, backend_used).
    """
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel)

    _, w = img_bgr.shape[:2]
    pad = max(64, w // 32)
    img_p = horizontal_wrap_pad(img_bgr, pad)
    msk_p = horizontal_wrap_pad(mask, pad)

    used = backend
    if backend == "sdxl":
        out_p = _inpaint_sdxl(img_p, msk_p)
        used = "sdxl"
    elif backend in ("auto", "lama"):
        try:
            out_p = _inpaint_lama(img_p, msk_p)
            used = "lama"
        except ImportError as exc:
            if backend == "lama":
                raise SystemExit(f"[error] LaMa requested but unavailable: {exc}. "
                                 f"Install with `pip install simple-lama-inpainting`.")
            print(f"[layered] LaMa not available ({exc}); falling back to TELEA.")
            out_p = _inpaint_telea(img_p, msk_p, radius)
            used = "telea"
    else:
        out_p = _inpaint_telea(img_p, msk_p, radius)
        used = "telea"

    return horizontal_wrap_unpad(out_p, pad), used


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a 360° panorama into foreground (RGBA) + inpainted background.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("image",
                   help="Equirectangular panorama (e.g. public/image-1-360.jpeg).")
    p.add_argument("--depth", required=True,
                   help="Depth map PNG (bright=near). "
                        "Generate with `python parallax_360.py <image>`.")
    p.add_argument("--out-dir", default="public/parallax",
                   help="Output directory (default: public/parallax).")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.65, 0.50, 0.35],
                   help="Depth thresholds in DESCENDING order (nearest first). "
                        "N values produce N foreground layers + 1 background. "
                        "Default: 0.65 0.50 0.35 → 3 fg layers (very-near, "
                        "close-mid, mid) + background (walls/ceiling).")
    p.add_argument("--feather", type=float, default=0.06,
                   help="Half-width of the smoothstep around each threshold, "
                        "used for cross-fading between layers (default: 0.06).")
    p.add_argument("--dilate", type=int, default=9,
                   help="Pixels to dilate the inpaint mask so the fg silhouette "
                        "doesn't leak colour into the bg (default: 9).")
    p.add_argument("--inpaint-radius", type=int, default=6,
                   help="cv2.inpaint TELEA radius (default: 6).")
    p.add_argument("--inpaint-backend", default="auto",
                   choices=["auto", "sdxl", "lama", "telea"],
                   help="Background inpainter (default: auto). "
                        "sdxl  = Stable Diffusion XL Inpainting via diffusers (~7GB, scene-aware). "
                        "lama  = simple-lama-inpainting (good, ~200MB weights). "
                        "telea = cv2.inpaint TELEA (fast, blurry). "
                        "auto  = prefer LaMa, fall back to TELEA if not installed.")
    p.add_argument("--exclude-top", type=float, default=0.25,
                   help="Top image fraction excluded from foreground "
                        "(default: 0.25). Keeps ceiling on the background sphere.")
    p.add_argument("--exclude-bottom", type=float, default=0.20,
                   help="Bottom image fraction excluded from foreground "
                        "(default: 0.20). Keeps the near-camera floor on the "
                        "background sphere; raise if you still see the floor "
                        "floating, lower if low-standing objects get clipped.")
    p.add_argument("--max-dim", type=int, default=None,
                   help="Cap the longer side of the input panorama before "
                        "slicing/inpainting. LaMa processes at the original size — "
                        "without this, an 8K equi can request >20 GB of RAM and die "
                        "from OOM. Recommended: 4096 for machines with 16-32 GB.")
    p.add_argument("--sam", action="store_true",
                   help="Enable SAM 2 object-snap (requires vendor/sam2 weights + "
                        "the `sam2` pip package). Falls back transparently to "
                        "threshold slicing if unavailable.")
    p.add_argument("--sam-k", type=int, default=3,
                   help="K for K-means binning of SAM objects by depth (default: 3).")
    p.add_argument("--sam-bg-threshold", type=float, default=0.20,
                   help="Normalised-depth [0,1] floor below which an object is "
                        "assigned to the background (default: 0.20).")
    p.add_argument("--sam-device", default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Device for SAM 2 inference (default: cpu).")
    return p.parse_args()


def generate_layered(
    img_bgr: np.ndarray,
    depth: np.ndarray,
    stem: str,
    out_dir: Path,
    thresholds: list[float],
    feather: float = 0.06,
    exclude_top: float = 0.18,
    exclude_bottom: float = 0.40,
    dilate_px: int = 9,
    inpaint_radius: int = 6,
    inpaint_backend: str = "auto",
    metric_depth: np.ndarray | None = None,
    repo_root: Path | None = None,
    use_sam: bool = False,
    sam_k: int = 3,
    sam_bg_threshold: float = 0.20,
    sam_device: str = "cpu",
    max_dim: int | None = None,
) -> dict:
    """
    Core LDI pipeline — slice depth into N foreground layers + inpainted bg
    and save them under `out_dir`. Returns a dict with the resulting paths
    and the backend that was actually used.

    Both the standalone CLI of this module AND `parallax_360.py` call this,
    so the depth array is passed in directly to avoid a PNG round-trip.

    `metric_depth` (optional): metric depth aligned with `depth`; if provided,
    layer radii are derived from it (see compute_layer_radii).

    `max_dim` (optional): if provided and the longer side of the input exceeds
    this limit, the panorama is downscaled before generating layers. LaMa processes
    the image at its original size — without this cap, an 8K equi can request
    >20 GB of transient RAM and die from OOM on machines with less memory.
    """
    if max_dim and max(img_bgr.shape[:2]) > max_dim:
        h0, w0 = img_bgr.shape[:2]
        scale = max_dim / max(h0, w0)
        new_w = max(2, int(round(w0 * scale)) & ~1)  # even
        new_h = max(2, int(round(h0 * scale)) & ~1)
        print(f"[layered] Downscaling input {w0}x{h0} → {new_w}x{new_h} "
              f"(max_dim={max_dim}) — depth/metric resampled to match.")
        img_bgr = cv2.resize(img_bgr, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

    h, w = img_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_CUBIC)

    thresholds_desc = sorted(thresholds, reverse=True)
    if thresholds_desc != list(thresholds):
        print(f"[layered] Reordered thresholds to descending: {thresholds_desc}")
    if any(t <= 0.0 or t >= 1.0 for t in thresholds_desc):
        raise ValueError("thresholds must be strictly between 0 and 1.")
    if len(set(thresholds_desc)) != len(thresholds_desc):
        raise ValueError("thresholds must be unique.")

    print(f"[layered] Image: {w}x{h}")
    print(f"[layered] Depth range [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"[layered] thresholds={thresholds_desc}  feather={feather}  "
          f"layers={len(thresholds_desc)} fg + 1 bg")

    layer_alphas, hard_mask = make_layered_masks(
        depth, thresholds_desc, feather,
        exclude_top=exclude_top,
        exclude_bottom=exclude_bottom,
        img_bgr=img_bgr,
        metric_depth=metric_depth,
        use_sam=use_sam,
        sam_k=sam_k,
        sam_bg_threshold=sam_bg_threshold,
        sam_device=sam_device,
    )
    union_cov = float(hard_mask.mean()) / 255.0
    print(f"[layered] Union foreground covers {union_cov * 100:.1f}% of pixels.")
    if union_cov < 0.01:
        print("[layered] WARNING: foreground union is nearly empty — "
              "lower the lowest threshold.")
    elif union_cov > 0.7:
        print("[layered] WARNING: foreground union is huge — raise the "
              "lowest threshold or the inpaint will look blurry.")
    for k, a in enumerate(layer_alphas):
        cov = float(a.mean())
        depth_hint = (f"depth ≥ {thresholds_desc[k]}" if k < len(thresholds_desc)
                      else f"SAM bin {k}")
        print(f"[layered]   layer {k}: ~{cov * 100:.1f}% coverage ({depth_hint})")

    radii: list[float | None] = [None] * len(layer_alphas)
    if metric_depth is not None:
        if metric_depth.shape != (h, w):
            metric_depth = cv2.resize(metric_depth.astype(np.float32), (w, h),
                                      interpolation=cv2.INTER_CUBIC)
        radii = compute_layer_radii(metric_depth, layer_alphas, hard_mask)
        for k, r in enumerate(radii):
            print(f"[layered]   layer {k}: suggested radius = {r}")

    print(f"[layered] Inpainting background (backend={inpaint_backend}) …")
    _t_inpaint = time.time()
    with _spinner("[layered]   inpaint"):
        bg_bgr, used_backend = inpaint_background(
            img_bgr, hard_mask,
            dilate_px=dilate_px,
            radius=inpaint_radius,
            backend=inpaint_backend,
        )
    print(f"[layered] Inpainted with {used_backend} in {time.time() - _t_inpaint:.1f}s.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    saved_layer_paths: list[Path] = []
    n_layers = len(layer_alphas)
    _t_save = time.time()
    for k, alpha in enumerate(layer_alphas):
        with _spinner(f"[layered]   saving fg {k+1}/{n_layers}"):
            alpha_u8 = (alpha * 255).clip(0, 255).astype(np.uint8)
            rgb_zeroed = zero_out_rgb(img_rgb, alpha_u8, dilate_px=2)
            fg_rgba = np.dstack([rgb_zeroed, alpha_u8])
            fg_bgra = cv2.cvtColor(fg_rgba, cv2.COLOR_RGBA2BGRA)
            # PNG kept side-by-side for fallback/A-B; WebP is what the runtime loads.
            png_path = out_dir / f"{stem}-fg{k}.png"
            cv2.imwrite(str(png_path), fg_bgra)
            webp_path = out_dir / f"{stem}-fg{k}.webp"
            ok = cv2.imwrite(str(webp_path), fg_bgra,
                             [cv2.IMWRITE_WEBP_QUALITY, 90])
        if not ok:
            raise RuntimeError(
                f"cv2.imwrite failed for {webp_path} — OpenCV build may "
                "lack libwebp support.")
        saved_layer_paths.append(webp_path)
    print(f"[layered] {n_layers} fg layers + 1 bg saved in {time.time() - _t_save:.1f}s.")

    bg_path = out_dir / f"{stem}-bg.jpeg"
    mask_path = out_dir / f"{stem}-fg-mask.png"
    cv2.imwrite(str(bg_path), bg_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(mask_path), hard_mask)

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root)) if repo_root else str(p)
        except ValueError:
            return str(p)

    print("[layered] Saved:")
    for k, lp in enumerate(saved_layer_paths):
        print(f"  layer {k}  → {rel(lp)}")
    print(f"  background → {rel(bg_path)}")
    print(f"  union mask → {rel(mask_path)}")
    layer_snippets = []
    for p, r in zip(saved_layer_paths, radii):
        radius_part = "" if r is None else f", radius: {r}"
        layer_snippets.append('{src:"/parallax/' + p.name + '"' + radius_part + "}")
    print("[layered] Done. Wire foregroundLayers in app/scenes.ts: "
          f"[{', '.join(layer_snippets)}].")

    return {
        "layer_paths": saved_layer_paths,
        "background_path": bg_path,
        "mask_path": mask_path,
        "thresholds": thresholds_desc,
        "backend": used_backend,
        "radii": radii,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    def resolve(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path

    img_path = resolve(args.image)
    depth_path = resolve(args.depth)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        sys.exit(f"[error] Image not found: {img_path}")
    if not depth_path.exists():
        sys.exit(f"[error] Depth map not found: {depth_path}. "
                 f"Run `python parallax_360.py {img_path.name}` first.")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        sys.exit(f"[error] Could not read image: {img_path}")
    h, w = img_bgr.shape[:2]

    depth_u8 = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if depth_u8 is None:
        sys.exit(f"[error] Could not read depth map: {depth_path}")
    if depth_u8.shape != (h, w):
        depth_u8 = cv2.resize(depth_u8, (w, h), interpolation=cv2.INTER_CUBIC)
    depth = depth_u8.astype(np.float32) / 255.0

    try:
        generate_layered(
            img_bgr=img_bgr,
            depth=depth,
            stem=img_path.stem,
            out_dir=out_dir,
            thresholds=list(args.thresholds),
            feather=args.feather,
            exclude_top=args.exclude_top,
            exclude_bottom=args.exclude_bottom,
            dilate_px=args.dilate,
            inpaint_radius=args.inpaint_radius,
            inpaint_backend=args.inpaint_backend,
            repo_root=repo_root,
            use_sam=args.sam,
            sam_k=args.sam_k,
            sam_bg_threshold=args.sam_bg_threshold,
            sam_device=args.sam_device,
            max_dim=args.max_dim,
        )
    except ValueError as exc:
        sys.exit(f"[error] {exc}")


if __name__ == "__main__":
    main()
