#!/usr/bin/env python3
"""
parallax_360.py — depth map generator for the 360° viewer

Pipeline:
  1. Load a 360° equirectangular image
  2. Run Depth-Anything-3 (DA3Mono-Large) — falls back to Depth-Anything-V2,
     then MiDaS, then a synthetic gradient if no torch backend is available
  3. Save a depth map PNG under `public/parallax/` for the runtime viewer

The depth PNG follows the canonical convention bright = near, dark = far
(see `app/components/parallax360/displacement.ts`).

Usage:
    python parallax_360.py public/my-pano.jpeg
    python parallax_360.py --batch public/scene-1.jpeg public/scene-2.jpeg

Requirements (install once):
    pip install -r requirements.txt
    pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Avoid macOS OpenMP runtime conflicts (torch + pycolmap/open3d deps from DA3).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def metric_to_disparity(depth: np.ndarray, min_clip: float = 0.01) -> np.ndarray:
    """
    Convert metric depth (far = high value) to disparity (near = high).

    Uses the reciprocal 1/d — not the linear 1−x inversion we apply to
    DA3 — because it respects the physical distance↔disparity relation,
    which is the distribution family the LDI pipeline's thresholds were
    tuned against. Note: the resulting range (e.g. [1, 100]) is skewed
    toward far values until postprocess_depth's percentile stretch
    normalizes it; if layers come out empty/huge, retune the thresholds.
    min_clip=0.01 matches the DAP model's min_depth (avoids 1/0).
    """
    return (1.0 / np.clip(depth.astype(np.float32), min_clip, None)).astype(np.float32)


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

DAP_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "DAP"


def _load_dap_model(device: str):
    """
    Build DAP from vendor/DAP and load model.pth.

    DAP repo quirks (verified 2026-06-10):
    - `dinov3_repo_dir` is relative to the DAP repo root → temporary chdir.
    - Imports root-level modules (`networks`, `depth_anything_utils`)
      → sys.path.insert.
    - The DINOv3 backbone is built with pretrained=False; model.pth carries
      all weights (may come with a `module.` prefix from DataParallel).
    """
    import torch

    weights = DAP_VENDOR_DIR / "weights" / "model.pth"
    if not DAP_VENDOR_DIR.is_dir():
        raise FileNotFoundError(
            f"vendor/DAP does not exist. Install it with:\n"
            f"  git clone --depth 1 https://github.com/Insta360-Research-Team/DAP {DAP_VENDOR_DIR}"
        )
    if not weights.is_file():
        raise FileNotFoundError(
            f"Weights missing at {weights}. Download (~1.4 GB):\n"
            f"  curl -L -o {weights} "
            f"https://huggingface.co/Insta360-Research/DAP-weights/resolve/main/model.pth"
        )

    # Exposes generic vendor packages (e.g. `datasets`) — guard to avoid
    # duplicating the entry on repeated calls.
    if str(DAP_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(DAP_VENDOR_DIR))
    prev_cwd = os.getcwd()
    os.chdir(DAP_VENDOR_DIR)
    try:
        import networks  # noqa: F401 — registra el modelo 'dap' en el factory
        from networks.models import make
        model = make({"name": "dap", "args": {
            "midas_model_type": "vitl",
            "fine_tune_type": "hypersim",
            "min_depth": 0.01,
            "max_depth": 1.0,
            "train_decoder": True,
        }})
    finally:
        os.chdir(prev_cwd)

    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    own = model.state_dict()
    missing = [k for k in own if k not in state]
    if missing:
        # strict=False + pretrained=False: without this check, a drift
        # between vendor/DAP (HEAD clone) and model.pth would run a
        # partially-random model that produces plausible-looking garbage depth.
        raise RuntimeError(
            f"model.pth does not cover {len(missing)} model keys "
            f"(drift between vendor/DAP and weights?). Examples: {missing[:3]}")
    model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
    return model.to(torch.device(device)).eval()


def load_depth_model(
    device: str,
    da3_model: str = "depth-anything/DA3MONO-LARGE",
    backend: str = "auto",
):
    """
    Load depth model with cascading fallback:
      0. DAP (vendor/DAP, native panoramic)
      1. Depth-Anything-3  (default: DA3MONO-LARGE — disparity-style depth
         with a well-behaved [0,1] distribution that our threshold-based
         LDI slicer was tuned against. Override with --da3-model for
         DA3-LARGE / DA3-GIANT / DA3Metric-LARGE if you re-tune thresholds.)
      2. Depth-Anything-V2 (via HuggingFace transformers, very good quality,
         Apache-2.0 — commercial-safe fallback)
      3. MiDaS-small       (via torch.hub, decent quality, MIT)
      4. Synthetic radial gradient (no torch required)

    backend != "auto" forces a single backend (SystemExit if unavailable).
    """
    if backend in ("auto", "dap"):
        try:
            print("[depth] Loading DAP (Depth Any Panoramas) …")
            return _load_dap_model(device), "dap"
        except Exception as exc:
            if backend == "dap":
                raise SystemExit(f"[error] Backend dap forced but not available: {exc}")
            print(f"[depth] DAP not available ({exc}); falling back to DA3.")

    # ── 1. Depth-Anything-3 ───────────────────────────────────────────────
    if backend in ("auto", "da3"):
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
            print(f"[depth] Loading {da3_model} …")
            model = DepthAnything3.from_pretrained(da3_model)
            model = model.to(device=torch.device(device)).eval()
            return model, "da3"
        except Exception as exc:
            if backend == "da3":
                raise SystemExit(f"[error] Backend da3 forced but not available: {exc}")
            print(f"[depth] DA3 ({da3_model}) not available ({exc}).")

    # ── 2. Depth-Anything-V2 via transformers ─────────────────────────────
    if backend in ("auto", "dav2"):
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            print("[depth] Loading Depth-Anything-V2-Small via transformers …")
            pipe = hf_pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=device if device != "mps" else "cpu",   # mps not yet supported in pipeline
            )
            return pipe, "dav2"
        except Exception as exc:
            if backend == "dav2":
                raise SystemExit(f"[error] Backend dav2 forced but not available: {exc}")
            print(f"[depth] Depth-Anything-V2 not available ({exc}).")

    # ── 3. MiDaS-small via torch.hub ─────────────────────────────────────
    if backend in ("auto", "midas"):
        try:
            import torch
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                   trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                        trust_repo=True)
            model = model.to(device).eval()
            return (model, transforms.small_transform), "midas"
        except Exception as exc:
            if backend == "midas":
                raise SystemExit(f"[error] Backend midas forced but not available: {exc}")
            print(f"[depth] MiDaS also unavailable ({exc}). Using synthetic depth map.")

    return None, "dummy"


def estimate_depth(
    image_path: Path,
    device: str,
    max_dim: int = 2048,
    da3_process_res: int = 504,
    da3_model: str = "depth-anything/DA3-LARGE",
    backend: str = "auto",
    preloaded: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """
    Returns (depth_norm, metric_depth, backend):
      - depth_norm   : float32 [0,1], canonical convention bright = near.
      - metric_depth : float32 with the raw metric depth (dap backend only;
                       None for all others) — feeds the per-slab radii.
      - backend      : name of the backend actually used.

    Canonical convention (after backend-specific normalisation): **1 = near,
    0 = far** (a disparity / proximity-style map). The runtime TS code reads
    the PNG directly: bright pixels are pulled toward the camera, dark
    pixels stay at full sphere radius.

    Shape matches the original image after downscaling if > max_dim.

    If *preloaded* is given as (model, backend), skip model loading (batch mode).
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    h, w = img_bgr.shape[:2]
    print(f"[depth] Image size: {w}x{h}")

    # Downscale for inference while keeping the full-size map
    scale = min(1.0, max_dim / max(h, w))
    inf_w = int(w * scale)
    inf_h = int(h * scale)
    # Keep dimensions divisible by 32 for transformer models
    inf_w = (inf_w // 32) * 32
    inf_h = (inf_h // 32) * 32
    img_small = cv2.resize(img_bgr, (inf_w, inf_h),
                           interpolation=cv2.INTER_AREA)

    if preloaded is not None:
        model, backend = preloaded
    else:
        model, backend = load_depth_model(device, da3_model=da3_model, backend=backend)

    metric_depth: np.ndarray | None = None

    if backend == "dap":
        import torch
        # Native model resolution (config/infer.yaml: 1024×512).
        img_dap = cv2.resize(img_bgr, (1024, 512), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_dap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = (torch.from_numpy(img_rgb.transpose(2, 0, 1))
                  .unsqueeze(0)
                  .to(next(model.parameters()).device))
        print("[depth] Running DAP inference (1024x512) …")
        with torch.inference_mode():
            outputs = model(tensor)
            pred = outputs["pred_depth"]
            if "pred_mask" in outputs:
                # In-place on inference tensors is only legal INSIDE the
                # inference_mode context — do not move outside the with block.
                valid = (1 - outputs["pred_mask"]) > 0.5
                pred[~valid] = 1.0   # invalid pixels (sky) → max depth, as in their infer.py
        depth_raw = pred[0].detach().cpu().squeeze().numpy().astype(np.float32)
        metric_depth = depth_raw.copy()
        depth_raw = metric_to_disparity(depth_raw)

    elif backend == "da3":
        import torch
        tmp = Path("/tmp/da3_input.png")
        cv2.imwrite(str(tmp), img_small)
        print(f"[depth] Running DA3 inference (model={da3_model}, "
              f"process_res={da3_process_res}) …")
        with torch.no_grad():
            pred = model.inference([str(tmp)], process_res=da3_process_res)
        depth_raw = pred.depth[0]          # (H, W) float32
        if hasattr(depth_raw, "cpu"):
            depth_raw = depth_raw.cpu().numpy()

    elif backend == "dav2":
        from PIL import Image as PILImage
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        print("[depth] Running Depth-Anything-V2-Small inference …")
        result = model(pil_img)
        depth_raw = np.array(result["depth"], dtype=np.float32)

    elif backend == "midas":
        import torch
        midas, transform = model
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).to(device)
        print("[depth] Running MiDaS inference …")
        with torch.no_grad():
            depth_raw = midas(input_tensor)
            depth_raw = torch.nn.functional.interpolate(
                depth_raw.unsqueeze(1),
                size=(inf_h, inf_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

    else:
        # Synthetic: radial gradient simulating a room
        print("[depth] Generating synthetic depth map …")
        cy, cx = inf_h / 2, inf_w / 2
        Y, X = np.mgrid[0:inf_h, 0:inf_w]
        depth_raw = 1.0 - np.hypot(X - cx, Y - cy) / np.hypot(cx, cy)
        depth_raw = (depth_raw * 2).clip(0, 1).astype(np.float32)

    # Normalise to [0, 1]
    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    if d_max - d_min < 1e-6:
        depth_norm = np.zeros_like(depth_raw, dtype=np.float32)
    else:
        depth_norm = ((depth_raw - d_min) / (d_max - d_min)).astype(np.float32)

    # ── Unify convention: 1 = near, 0 = far (proximity / disparity-like) ────
    # DA3 emits metric depth (raw value grows with distance) so its normalised
    # output is bright = far. DA-V2 / MiDaS already produce disparity-like maps
    # where bright = near. Invert DA3 to keep one canonical convention so the
    # runtime TS code (and any downstream tooling) doesn't need to branch.
    backend_emits_far_as_bright = {"da3": True}
    if backend_emits_far_as_bright.get(backend, False):
        depth_norm = 1.0 - depth_norm

    # Resize depth map back to original image size
    if depth_norm.shape[:2] != (h, w):
        depth_norm = cv2.resize(depth_norm, (w, h),
                                interpolation=cv2.INTER_CUBIC)

    if metric_depth is not None and metric_depth.shape[:2] != (h, w):
        metric_depth = cv2.resize(metric_depth, (w, h), interpolation=cv2.INTER_CUBIC)

    print(f"[depth] Depth map ready: {w}x{h}, "
          f"range [{depth_norm.min():.3f}, {depth_norm.max():.3f}]  backend={backend}")
    return depth_norm, metric_depth, backend


# ---------------------------------------------------------------------------
# Depth post-processing
# ---------------------------------------------------------------------------

def postprocess_depth(
    depth: np.ndarray,
    color_img_bgr: np.ndarray,
    gamma: float = 1.0,
    pole_blend: float = 0.45,
    floor_keep: float = 0.85,
) -> np.ndarray:
    """
    Refine the raw normalized depth map before saving:

    1. Edge-preserving filter  — guided filter (preferred) or bilateral fallback.
       Uses the colour image as a guide so depth edges align with colour edges.
    2. Histogram stretch       — clamp the 1/99% extremes to [0, 1].
    3. Power-curve remapping   — gamma > 1 compresses far values and gives
       more precision to the near range (where parallax shows the most).
    4. Pole attenuation        — equirectangular 360° images are heavily
       distorted near the top pole (ceiling); DA3 (trained on perspective
       images) gives unreliable depths there, so blend toward the median to
       suppress noise. The bottom pole (floor) is treated more leniently
       (controlled by `floor_keep`) because walking transitions rely on a
       useful floor depth gradient.
    """
    h, w = depth.shape

    # ── 1. Edge-preserving filter ────────────────────────────────────────────
    guide = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2GRAY)
    if guide.shape != (h, w):
        guide = cv2.resize(guide, (w, h), interpolation=cv2.INTER_AREA)
    guide_f = guide.astype(np.float32) / 255.0

    # Radius covers ~0.35% of image width — enough to smooth sensor noise
    # without blending depth boundaries across distinct surfaces.
    adaptive_radius = max(4, int(w * 0.0035))
    try:
        from cv2 import ximgproc  # type: ignore
        depth_smooth = ximgproc.guidedFilter(
            guide=guide_f,
            src=depth,
            radius=adaptive_radius,
            eps=(0.005 ** 2),   # tight eps → hard edge preservation
            dDepth=cv2.CV_32F,
        )
        print(f"[post] Applied guided filter (ximgproc, radius={adaptive_radius}).")
    except (AttributeError, ImportError, cv2.error):
        depth_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
        d_kernel = max(9, (adaptive_radius // 2) * 2 + 1)  # odd number
        depth_smooth = cv2.bilateralFilter(
            depth_u8, d=d_kernel, sigmaColor=10, sigmaSpace=10
        ).astype(np.float32) / 255.0
        print(f"[post] Applied bilateral filter (ximgproc not available, d={d_kernel}).")

    depth_smooth = depth_smooth.clip(0.0, 1.0)

    # ── 2. Percentile stretch → use the full [0, 1] range ───────────────────
    # Clip 1% outliers at each end so specular highlights / dark corners
    # don't crush the usable range.
    p1  = float(np.percentile(depth_smooth, 1))
    p99 = float(np.percentile(depth_smooth, 99))
    span = max(p99 - p1, 1e-6)
    depth_stretched = ((depth_smooth - p1) / span).clip(0.0, 1.0)
    print(f"[post] Histogram stretched: raw [{p1:.3f}, {p99:.3f}] → [0, 1].")

    # ── 3. Power-curve remapping ─────────────────────────────────────────────
    # Canonical convention is 1 = near, 0 = far. With gamma > 1 we darken
    # midtones (more of the range becomes "far"), pushing precision into the
    # near end; gamma < 1 does the opposite. gamma=1 keeps the linear stretch.
    if abs(gamma - 1.0) > 1e-6:
        depth_curved = np.power(depth_stretched, gamma)
    else:
        depth_curved = depth_stretched

    # ── 4. Pole attenuation (equirectangular latitude weighting) ────────────
    # Row 0 is the top pole (sky/ceiling); row h-1 is the bottom pole (floor).
    # Latitude in radians: +pi/2 at top, -pi/2 at bottom.
    lat = np.linspace(np.pi / 2, -np.pi / 2, h)
    # Symmetric softness so the attenuation tapers more gently than cos^4.
    cos_weight = np.cos(lat) ** 2
    blend = pole_blend + (1.0 - pole_blend) * cos_weight

    # The floor carries critical depth information for walking transitions,
    # so soften the attenuation on the southern hemisphere (lat <= 0).
    floor_mask = (lat <= 0).astype(np.float32)
    floor_keep_arr = floor_mask * floor_keep
    blend = blend * (1.0 - floor_keep_arr) + floor_keep_arr
    blend = blend.clip(0.0, 1.0)[:, np.newaxis]

    median_d = float(np.median(depth_curved))
    depth_out = depth_curved * blend + median_d * (1.0 - blend)

    return depth_out.clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def save_depth_png(depth: np.ndarray, output_path: Path, bit_depth: int = 8) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth == 16:
        # Preserve finer depth gradations for offline inspection/post-processing.
        depth_img = (depth * 65535).clip(0, 65535).astype(np.uint16)
    else:
        depth_img = (depth * 255).clip(0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(output_path), depth_img)
    if not ok:
        raise RuntimeError(f"Could not save depth map to {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a depth map from an equirectangular 360° image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("image", nargs="?",
                   help="Path to the equirectangular image (e.g. public/scene-1.jpeg).")
    p.add_argument("--batch", nargs="+", metavar="IMG",
                   help="Batch mode: process multiple images. Model is loaded once. "
                        "Depth maps saved to public/parallax/depth_{stem}.png.")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Torch device for depth inference (default: cpu)")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "dap", "da3", "dav2", "midas"],
                   help="Depth backend (default: auto = cascade "
                        "DAP → DA3 → DA-V2 → MiDaS → synthetic). "
                        "dap = Depth Any Panoramas (native equirect, metric; "
                        "requires vendor/DAP + weights, see requirements.txt).")
    p.add_argument("--max-dim", type=int, default=2048,
                   help="Max image dimension for depth inference (default: 2048)")
    p.add_argument("--da3-process-res", type=int, default=504,
                   help="DA3 internal processing resolution (higher = slower, more detail). "
                        "Recommended: 504, 756, 1008.")
    p.add_argument("--da3-model", default="depth-anything/DA3MONO-LARGE",
                   help="DA3 variant identifier on HuggingFace. Defaults to "
                        "DA3MONO-LARGE because its disparity distribution slices "
                        "cleanly with our LDI thresholds. Other options: "
                        "'depth-anything/DA3-LARGE' (any-view, sharper bounds "
                        "but compressed distribution — needs threshold retuning), "
                        "'depth-anything/DA3-GIANT' (more VRAM), "
                        "'depth-anything/DA3Metric-LARGE' (metric depth, would "
                        "require rewriting the slicer in meters).")
    p.add_argument("--depth-bit-depth", type=int, default=8, choices=[8, 16],
                   help="Bit depth for output PNG (default: 8)")
    p.add_argument("--depth-output", default=None,
                   help="Path to save the depth map PNG. Defaults to "
                        "public/parallax/depth_{stem}.png next to the input image.")
    p.add_argument("--depth-gamma", type=float, default=1.0,
                   help="Power-curve gamma applied after histogram stretch. "
                        "With the canonical 'bright = near' convention, "
                        "gamma < 1 emphasises near surfaces (more parallax "
                        "for foreground objects) and gamma > 1 emphasises far "
                        "ones. 1.0 keeps the stretched range untouched. "
                        "Default: 1.0.")
    p.add_argument("--pole-blend", type=float, default=None,
                   help="Base preservation factor for pole regions before the "
                        "cosine taper (0 = full flatten at poles, 1 = no "
                        "correction at all). Default: 0.45 for perspective "
                        "backends (DA3/DA-V2/MiDaS); 1.0 (no attenuation) "
                        "for the dap backend, which is distortion-aware and "
                        "does not need the correction.")
    p.add_argument("--floor-keep", type=float, default=0.85,
                   help="Extra preservation for the bottom (floor) hemisphere, "
                        "blended on top of pole-blend. Walking transitions need "
                        "a usable floor depth gradient; default: 0.85.")
    p.add_argument("--no-postproc", action="store_true",
                   help="Skip all depth post-processing (filter / gamma / poles).")
    # ── Layered Depth Image (LDI) integration ────────────────────────────────
    # When --layered is set the depth array is handed to layered_360 so the
    # foreground/background slabs are produced in the same run; otherwise
    # only the depth PNG is written.
    p.add_argument("--layered", action="store_true",
                   help="Also produce Layered Depth Image assets (N foreground "
                        "RGBAs + inpainted background) next to the depth map.")
    p.add_argument("--layered-thresholds", type=float, nargs="+",
                   default=[0.65, 0.50, 0.35],
                   help="Depth thresholds in DESCENDING order (nearest first) for "
                        "the LDI slicing (default: 0.65 0.50 0.35 → 3 fg + 1 bg).")
    p.add_argument("--layered-feather", type=float, default=0.06,
                   help="Smoothstep half-width at each LDI threshold (default: 0.06).")
    p.add_argument("--layered-exclude-top", type=float, default=0.25,
                   help="Top image fraction kept on background (default: 0.25).")
    p.add_argument("--layered-exclude-bottom", type=float, default=0.20,
                   help="Bottom image fraction kept on background (default: 0.20).")
    p.add_argument("--layered-dilate", type=int, default=9,
                   help="Inpaint-mask dilation in pixels (default: 9).")
    p.add_argument("--layered-inpaint-backend", default="auto",
                   choices=["auto", "lama", "telea"],
                   help="LDI background inpainter (default: auto). See layered_360.py.")
    p.add_argument("--sam", action="store_true",
                   help="Forward to layered_360: enable SAM 2 object-snap "
                        "(only meaningful with --layered).")
    p.add_argument("--sam-k", type=int, default=3,
                   help="Forward to layered_360: K for K-means binning (default: 3).")
    p.add_argument("--sam-bg-threshold", type=float, default=0.20,
                   help="Forward to layered_360: normalised-depth bg floor (default: 0.20).")
    p.add_argument("--sam-device", default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Forward to layered_360: device for SAM inference.")
    return p.parse_args()


def process_single_image(
    image_path: Path,
    depth_output: Path,
    args: argparse.Namespace,
    repo_root: Path,
    preloaded: tuple | None = None,
) -> np.ndarray:
    """Estimate depth for one image, post-process, and save the PNG."""
    depth, metric_depth, backend = estimate_depth(
        image_path,
        device=args.device,
        max_dim=args.max_dim,
        da3_process_res=args.da3_process_res,
        da3_model=args.da3_model,
        backend=args.backend,
        preloaded=preloaded,
    )

    if not args.no_postproc:
        # DAP is distortion-aware: without an explicit override, pole
        # attenuation (a patch for perspective backends) is disabled.
        pole_blend = args.pole_blend if args.pole_blend is not None \
            else (1.0 if backend == "dap" else 0.45)
        color_img = cv2.imread(str(image_path))
        depth = postprocess_depth(
            depth,
            color_img_bgr=color_img,
            gamma=args.depth_gamma,
            pole_blend=pole_blend,
            floor_keep=args.floor_keep,
        )
        print(f"[post] Depth after post-processing: "
              f"range [{depth.min():.3f}, {depth.max():.3f}]")

    save_depth_png(depth, depth_output, bit_depth=args.depth_bit_depth)
    print(f"[depth] Saved depth map → {depth_output}")

    if getattr(args, "layered", False):
        # Lazy import so the layered module's heavy optional deps (LaMa)
        # don't load unless --layered was actually requested.
        import layered_360
        color_img = cv2.imread(str(image_path))
        try:
            layered_360.generate_layered(
                img_bgr=color_img,
                depth=depth,
                stem=image_path.stem,
                out_dir=depth_output.parent,
                thresholds=list(args.layered_thresholds),
                feather=args.layered_feather,
                exclude_top=args.layered_exclude_top,
                exclude_bottom=args.layered_exclude_bottom,
                dilate_px=args.layered_dilate,
                inpaint_backend=args.layered_inpaint_backend,
                metric_depth=metric_depth,
                repo_root=repo_root,
                use_sam=args.sam,
                sam_k=args.sam_k,
                sam_bg_threshold=args.sam_bg_threshold,
                sam_device=args.sam_device,
                max_dim=getattr(args, "max_dim", None),
            )
        except ValueError as exc:
            print(f"[layered] ERROR: {exc}")

    return depth


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    # ── Batch mode ─────────────────────────────────────────────────────────
    if args.batch:
        image_paths = []
        for p in args.batch:
            pp = Path(p)
            if not pp.is_absolute():
                pp = repo_root / pp
            if not pp.exists():
                sys.exit(f"[error] Image not found: {pp}")
            image_paths.append(pp)

        print(f"[batch] Processing {len(image_paths)} images …")
        preloaded = load_depth_model(args.device, da3_model=args.da3_model,
                                     backend=args.backend)

        for img_path in image_paths:
            stem = img_path.stem
            depth_out = repo_root / "public" / "parallax" / f"depth_{stem}.png"
            print(f"\n{'─' * 60}")
            print(f"[batch] {img_path.name} → {depth_out.name}")
            print(f"{'─' * 60}")
            process_single_image(img_path, depth_out, args, repo_root,
                                 preloaded=preloaded)

        print(f"\n[done] Batch complete — {len(image_paths)} depth maps generated.")
        return

    # ── Single-image mode ──────────────────────────────────────────────────
    if not args.image:
        sys.exit("[error] No image given. Usage: python parallax_360.py <path>")

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = repo_root / image_path
    if not image_path.exists():
        sys.exit(f"[error] Image not found: {image_path}")

    if args.depth_output:
        depth_output = Path(args.depth_output)
        if not depth_output.is_absolute():
            depth_output = repo_root / depth_output
    else:
        depth_output = repo_root / "public" / "parallax" / f"depth_{image_path.stem}.png"

    process_single_image(image_path, depth_output, args, repo_root)
    print(f"[done] Add this scene to app/scenes.ts with depthSrc=\"/parallax/{depth_output.name}\".")


if __name__ == "__main__":
    main()
