# Parallax 360 Tour

A depth-aware 360° panorama viewer. Drop in equirectangular panoramas, run the bundled Python pipeline to produce a **Layered Depth Image** (background + N foreground layers + per-pixel depth), and the browser viewer renders them as concentric three.js spheres displaced by depth — so a small mouse movement produces *real geometric parallax*, not a fake shader trick.

> **Live preview:** https://3d-house-model-coral.vercel.app/

![Parallax viewer preview](docs/preview.gif)

---

## Why this is interesting

The hard part of a 360° viewer with parallax isn't the WebGL — it's reconstructing enough scene geometry from a *single* equirectangular photograph that the viewer has something to displace. This project chains several state-of-the-art vision models offline so the runtime stays a thin three.js shell:

| Stage | What happens | Models / tech |
|---|---|---|
| **Depth estimation** | Predict a per-pixel depth map from the panorama | [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (primary), with automatic fallback to Depth-Anything-V2 → MiDaS → synthetic gradient. Optional [DAP](https://github.com/Insta360-Research-Team/DAP) backend for native panoramic depth |
| **Layer slicing** | Split the panorama into N depth slabs (closest → farthest) | Threshold-based slicing with soft alpha masks. Optional [SAM 2](https://github.com/facebookresearch/sam2) object-snap pass to align slab boundaries with object silhouettes (k-means depth binning + NMS dedup) |
| **Background inpainting** | Fill in the holes left when foreground objects are removed | [LaMa](https://github.com/advimman/lama) (primary), with OpenCV TELEA as a fallback if LaMa isn't installed |
| **Runtime rendering** | Render layers as nested spheres + per-pixel depth displacement | three.js, inside-out `SphereGeometry`, displaced per-pixel in the vertex shader by the depth map |

The result: a flat 6K equirectangular JPEG ends up looking like a small set of nested geometry. Move the mouse, and the close objects shift more than the far ones — same physics as binocular parallax.

## How parallax actually works here

There are two mutually-exclusive render modes; the viewer picks one per scene based on what assets the pipeline produced.

**Depth-displacement mode** (single-layer fallback)
The panorama is mapped onto an inside-out sphere whose vertices are pushed *inward* by the depth map. A subtle camera offset driven by the cursor produces a sense of depth, but everything is still one connected mesh — no disocclusions.

**Layered Depth Image (LDI) mode** (default for new scenes)
The scene is split into:
- **One inpainted background sphere** (the farthest layer, with foreground holes filled by LaMa).
- **N concentric inner spheres**, one per foreground depth slab, each with an RGBA texture whose alpha is the soft slab mask.

The camera offsets with the cursor, and because each layer lives on a sphere of a different radius, closer layers shift through a larger angle than far layers. This is *real* geometric parallax — when you peek behind a close object, you actually see the inpainted background that was hiding behind it.

Both modes share a depth-displaced fallback path (foreground spheres are also displaced per-pixel by the depth map, so close objects "ground" themselves to the floor instead of floating).

## Stack

| Layer | What |
|---|---|
| Frontend | Next.js 16 (App Router), React 19, three.js, Tailwind v4 |
| Pipeline | Python 3.13, PyTorch, OpenCV, Depth-Anything-3, simple-lama-inpainting, optional SAM 2 |
| Tests | pytest (pipeline), ESLint + TypeScript strict (frontend) |

## Setup

```bash
# JS side
npm install
npm run dev                  # http://localhost:3000

# Python side (one-time venv)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Recommended: install the primary depth model
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

The Python pipeline auto-detects which backends are installed and falls back gracefully — you can run with just `requirements.txt` (MiDaS via torch.hub) and still get usable depth, or add Depth-Anything-3, LaMa, SAM 2, and DAP incrementally.

## Generating a new scene

Drop your equirectangular panorama into `public/` (JPEG/PNG/WebP), then run the interactive wizard:

```bash
source venv/bin/activate
python scripts/new_scene.py
```

The wizard:

1. Detects panoramas in `public/` that aren't already registered.
2. Detects missing pip packages (cv2, torch, LaMa, DA3, SAM 2) and offers to install them.
3. Asks for quality (`low | medium | high | ultra`) and device (`auto | cpu | cuda | mps`).
4. Runs depth + LDI in one pass with progress feedback (spinner during the slow LaMa step).
5. Appends the new scene to `app/scenes.ts` between the `// <build_scene:start>` / `// <build_scene:end>` markers, validates the result with `tsc --noEmit`, and rolls back if validation fails.

Non-interactive equivalent:

```bash
python scripts/build_scene.py public/my-pano.jpeg --quality high
python scripts/build_scene.py --doctor          # list installed backends + max usable quality
```

## Quality presets

| Preset | Depth backend | `max_dim` | Layers | Typical use |
|---|---|---|---|---|
| `low` | synthetic / cascade | 1024 | 1 | smoke test / fallback |
| `medium` | Depth-Anything-V2 | 2048 | 2 | low-RAM machines |
| `high` | Depth-Anything-3 | 2048 | 3 | sensible default |
| `ultra` | Depth-Anything-3 + SAM 2 | 4096 | 4 | best quality, ~5–7 GB RAM |

`max_dim` caps both the depth-inference resolution *and* the LDI input — an 8K panorama gets downscaled before LaMa inpainting. Without this cap LaMa can OOM 24 GB Macs on 8K input.

## Project layout

```
app/
  scenes.ts                        # SCENES[] — registered panoramas
  page.tsx                         # renders the viewer + scene selector
  components/parallax360/          # three.js viewer (self-contained)
public/
  *.{jpeg,webp,png}                # input panoramas
  parallax/                        # generated depth + LDI assets
parallax_360.py                    # depth-only CLI  (DA3/DA-V2/MiDaS/DAP/synthetic)
layered_360.py                    # LDI-only CLI    (slice + inpaint + optional SAM)
scripts/
  build_scene.py                   # one-shot pipeline + scenes.ts patcher
  new_scene.py                     # interactive wizard
tests/                             # pytest suite for the pipeline
```

## Tests

```bash
# Frontend
npm run lint

# Pipeline
source venv/bin/activate
python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE).
