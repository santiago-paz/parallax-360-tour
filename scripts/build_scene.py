#!/usr/bin/env python3
"""
build_scene.py — one-shot pipeline: panorama → depth + LDI → registered scene.

Wraps parallax_360.py + layered_360.py with quality presets and writes the
resulting scene entry into app/scenes.ts between dedicated markers.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device, or return device as-is."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def infer_id_from_filename(filename: str) -> str:
    """
    Derive a scene id from a panorama path.

    Lowercase the stem, replace non-alnum runs with `-`, collapse repeats,
    strip leading/trailing `-`. Raises ValueError if the result is empty or
    doesn't match ID_PATTERN.
    """
    stem = Path(filename).stem.lower()
    # Reject dotfiles (hidden files with no actual stem)
    if stem.startswith("."):
        raise ValueError(
            f"Cannot derive a valid scene id from {filename!r}. "
            f"File is a dotfile with no stem. Rename the file so its stem "
            f"contains at least one letter or digit."
        )
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if not slug or not ID_PATTERN.match(slug):
        raise ValueError(
            f"Cannot derive a valid scene id from {filename!r}. "
            f"Got {slug!r} after slugifying. Rename the file so its stem "
            f"contains at least one letter or digit."
        )
    return slug


def label_from_id(scene_id: str) -> str:
    """Display label for the scene picker: id with `-` → ` ` and uppercased."""
    return scene_id.replace("-", " ").upper()


def render_scene_entry(
    *,
    scene_id: str,
    image_basename: str,
    n_foreground_layers: int,
) -> str:
    """
    Render the TypeScript snippet for a single SceneConfig entry.

    Always uses 2-space indentation matching the rest of scenes.ts, and
    trailing commas everywhere so future appends remain diff-friendly.
    The returned string ends with a newline so append-mode can concatenate
    directly.
    """
    if n_foreground_layers < 1:
        raise ValueError(
            f"n_foreground_layers must be >= 1 (got {n_foreground_layers})"
        )
    if "/" in image_basename or "\\" in image_basename:
        raise ValueError(
            f"image_basename must be a filename only, not a path (got {image_basename!r})"
        )
    stem = Path(image_basename).stem
    label = label_from_id(scene_id)
    layer_lines = [
        f'        {{ src: "/parallax/{stem}-fg{i}.webp" }},'
        for i in range(n_foreground_layers)
    ]
    return (
        '  {\n'
        f'    id: "{scene_id}",\n'
        f'    imageSrc: "/{image_basename}",\n'
        f'    depthSrc: "/parallax/depth_{stem}.png",\n'
        f'    label: "{label}",\n'
        '    layered: {\n'
        f'      backgroundSrc: "/parallax/{stem}-bg.jpeg",\n'
        '      foregroundLayers: [\n'
        + '\n'.join(layer_lines) + '\n'
        '      ],\n'
        '    },\n'
        '  },\n'
    )


MARKER_START = "// <build_scene:start>"
MARKER_END = "// <build_scene:end>"


def _find_markers(text: str) -> tuple[int, int]:
    start = text.find(MARKER_START)
    end = text.find(MARKER_END)
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"scenes.ts is missing the {MARKER_START} / {MARKER_END} markers. "
            f"Add them inside the SCENES array before running build_scene."
        )
    nl = text.find("\n", start)
    if nl == -1:
        raise ValueError("Malformed scenes.ts — start marker has no newline")
    return nl + 1, end


def list_scene_ids(scenes_path: Path) -> list[str]:
    """Return the ids of all entries currently between the markers, in order."""
    text = scenes_path.read_text()
    a, b = _find_markers(text)
    block = text[a:b]
    return re.findall(r'id:\s*"([a-z0-9][a-z0-9-]*)"', block)


def append_scene_entry(scenes_path: Path, entry: str) -> None:
    """Append an entry just before the end marker. Raises if the id already exists."""
    text = scenes_path.read_text()
    a, b = _find_markers(text)
    block = text[a:b]
    existing_ids = re.findall(r'id:\s*"([a-z0-9][a-z0-9-]*)"', block)
    new_id_match = re.search(r'id:\s*"([a-z0-9][a-z0-9-]*)"', entry)
    if not new_id_match:
        raise ValueError("Entry does not contain a valid id field")
    new_id = new_id_match.group(1)
    if new_id in existing_ids:
        raise ValueError(
            f"Scene with id {new_id!r} already exists in {scenes_path}. "
            f"Use --force to replace it."
        )
    # The end marker may have leading whitespace on its line; preserve it.
    line_start = text.rfind("\n", 0, b) + 1
    marker_indent = text[line_start:b]
    new_text = text[:line_start] + entry + marker_indent + text[b:]
    scenes_path.write_text(new_text)


def replace_scene_entry(scenes_path: Path, *, scene_id: str, entry: str) -> None:
    """Replace the entry whose id matches scene_id. Raises if not found."""
    text = scenes_path.read_text()
    a, b = _find_markers(text)
    block = text[a:b]
    id_match = re.search(rf'id:\s*"{re.escape(scene_id)}"', block)
    if not id_match:
        raise ValueError(
            f"Scene with id {scene_id!r} not found between markers in {scenes_path}"
        )
    obj_start = block.rfind("{", 0, id_match.start())
    if obj_start == -1:
        raise ValueError(f"Malformed scenes.ts — no `{{` before id {scene_id!r}")
    line_start = block.rfind("\n", 0, obj_start) + 1
    depth = 0
    i = obj_start
    while i < len(block):
        ch = block[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        raise ValueError(f"Unbalanced braces around entry {scene_id!r}")
    j = i + 1
    if j < len(block) and block[j] == ",":
        j += 1
    if j < len(block) and block[j] == "\n":
        j += 1
    abs_line_start = a + line_start
    abs_end = a + j
    new_text = text[:abs_line_start] + entry + text[abs_end:]
    scenes_path.write_text(new_text)


def _run_tsc_noemit(repo_root: Path) -> tuple[int, str]:
    """Run `npx tsc --noEmit` from repo_root. Returns (exit_code, stderr+stdout)."""
    result = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


@contextlib.contextmanager
def scenes_edit_session(scenes_path: Path, repo_root: Path | None = None):
    """
    Open an editing session on scenes.ts with backup + tsc validation.

    Usage:
        with scenes_edit_session(Path("app/scenes.ts")):
            append_scene_entry(...)

    On exit, runs `npx tsc --noEmit`. If it fails, restores the .bak snapshot
    and raises RuntimeError. If `npx tsc` itself is missing, logs a warning
    and treats the edit as successful (validation is best-effort).

    Not safe for concurrent invocations: two parallel processes share the same `.bak` path and could corrupt each other's rollback. This script is designed to be run by a single user (not by CI in parallel or as part of a build system that fans out). If you need concurrent invocations, redesign the backup naming.
    """
    if repo_root is None:
        repo_root = REPO_ROOT
    backup = scenes_path.with_suffix(scenes_path.suffix + ".bak")
    shutil.copy2(scenes_path, backup)
    try:
        yield
    except Exception:
        shutil.copy2(backup, scenes_path)
        raise

    try:
        code, output = _run_tsc_noemit(repo_root)
    except FileNotFoundError:
        print(
            "[build] WARN: `npx tsc` not on PATH — skipping TypeScript "
            "validation of scenes.ts. Run `npm install` once to enable it.",
            flush=True,
        )
        return

    if code != 0:
        shutil.copy2(backup, scenes_path)
        raise RuntimeError(
            f"tsc validation failed after editing {scenes_path}. "
            f"Reverted to backup. Output:\n{output}"
        )


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sam_installed() -> bool:
    """Check both the package import AND the weights file (SAM is opt-in)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from layered_360 import is_sam_available
        return is_sam_available()
    except ImportError:
        return False


def detect_backends() -> dict[str, bool]:
    """Detect which depth/inpaint backends are importable."""
    return {
        "synthetic": True,  # always — it's pure numpy
        "midas": _module_available("timm") and _module_available("torch"),
        "v2": _module_available("transformers") and _module_available("torch"),
        "da3": _module_available("depth_anything_3"),
        "lama": _module_available("simple_lama_inpainting"),
        "sam": _sam_installed(),
    }


# Mapping of preset → minimum backend it needs. If that backend is missing the
# preset degrades, but we still allow the run — the engine cascades down.
_PRESET_REQUIREMENT = {
    "low": "synthetic",
    "medium": "v2",
    "high": "da3",
    "ultra": "da3",
}


def max_quality_available(backends: dict[str, bool]) -> str:
    """Highest preset whose required backend is installed."""
    for preset in ("ultra", "high", "medium", "low"):
        req = _PRESET_REQUIREMENT[preset]
        if backends.get(req, False):
            return preset
    return "low"


_INSTALL_HINTS = {
    "synthetic": "(always)",
    "midas": "pip install timm torch",
    "v2": "pip install transformers torch",
    "da3": "pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git",
    "lama": "pip install simple-lama-inpainting",
    "sam": "pip install git+https://github.com/facebookresearch/sam2.git + weights",
}


def print_doctor_report() -> None:
    backends = detect_backends()
    print("Backend          Installed   Install command")
    print("─────────────    ─────────   ─────────────────────────────────────────")
    for name in ("synthetic", "midas", "v2", "da3", "lama", "sam"):
        check = "✓" if backends[name] else "✗"
        print(f"{name:<16} {check:<11} {_INSTALL_HINTS[name]}")
    print()
    print(f"Highest quality available right now: {max_quality_available(backends)}")


def _expected_output_paths(out_dir: Path, image: Path, n_foreground_layers: int) -> list[Path]:
    stem = image.stem
    out = [
        out_dir / f"depth_{stem}.png",
        out_dir / f"{stem}-bg.jpeg",
    ]
    out.extend(out_dir / f"{stem}-fg{i}.webp" for i in range(n_foreground_layers))
    return out


def preflight_checks(
    *,
    images: list[Path],
    scenes_path: Path,
    out_dir: Path,
    force: bool,
    n_foreground_layers: int,
) -> None:
    """
    Validate that the run will not collide with existing state.
    Raises before any model loads if anything is wrong.
    """
    existing_ids = set(list_scene_ids(scenes_path))
    for image in images:
        if not image.exists():
            raise FileNotFoundError(f"Input image not found: {image}")
        try:
            scene_id = infer_id_from_filename(str(image))
        except ValueError as exc:
            raise ValueError(f"{image}: {exc}") from exc
        if not force and scene_id in existing_ids:
            raise ValueError(
                f"Scene with id {scene_id!r} already exists in {scenes_path}. "
                f"Use --force to replace it."
            )
        if not force:
            for output in _expected_output_paths(out_dir, image, n_foreground_layers):
                if output.exists():
                    raise FileExistsError(
                        f"Output {output} already exists. "
                        f"Use --force to overwrite."
                    )


QUALITY_PRESETS: dict[str, dict] = {
    "low": {
        "backend": "auto",  # auto cascade — degrades to synthetic if no torch backend installed
        "max_dim": 1024,
        "da3_process_res": 504,  # unused for non-DA3 backends, kept for shape
        "layered_thresholds": [0.55],  # 1 fg layer
        "layered_feather": 0.06,
        "layered_exclude_top": 0.25,
        "layered_exclude_bottom": 0.20,
        "layered_dilate": 9,
        "layered_inpaint_backend": "telea",
        "depth_gamma": 1.0,
        "depth_bit_depth": 8,
        "no_postproc": False,
        "sam": False,
        "sam_k": 3,
        "sam_bg_threshold": 0.20,
        "sam_device": "cpu",
    },
    "medium": {
        "backend": "dav2",
        "max_dim": 2048,
        "da3_process_res": 504,
        "layered_thresholds": [0.60, 0.40],  # 2 fg layers
        "layered_feather": 0.06,
        "layered_exclude_top": 0.25,
        "layered_exclude_bottom": 0.20,
        "layered_dilate": 9,
        "layered_inpaint_backend": "telea",
        "depth_gamma": 1.0,
        "depth_bit_depth": 8,
        "no_postproc": False,
        "sam": False,
        "sam_k": 3,
        "sam_bg_threshold": 0.20,
        "sam_device": "cpu",
    },
    "high": {
        "backend": "da3",
        "max_dim": 2048,
        "da3_process_res": 504,
        "layered_thresholds": [0.65, 0.50, 0.35],  # 3 fg layers
        "layered_feather": 0.06,
        "layered_exclude_top": 0.25,
        "layered_exclude_bottom": 0.20,
        "layered_dilate": 9,
        "layered_inpaint_backend": "auto",  # prefers LaMa if installed
        "depth_gamma": 1.0,
        "depth_bit_depth": 8,
        "no_postproc": False,
        "sam": False,
        "sam_k": 3,
        "sam_bg_threshold": 0.20,
        "sam_device": "cpu",
    },
    "ultra": {
        "backend": "da3",
        "max_dim": 4096,
        "da3_process_res": 1008,
        "layered_thresholds": [0.70, 0.55, 0.40, 0.25],  # 4 fg layers
        "layered_feather": 0.06,
        "layered_exclude_top": 0.25,
        "layered_exclude_bottom": 0.20,
        "layered_dilate": 9,
        "layered_inpaint_backend": "auto",
        "depth_gamma": 1.0,
        "depth_bit_depth": 8,
        "no_postproc": False,
        "sam": True,
        "sam_k": 3,
        "sam_bg_threshold": 0.20,
        "sam_device": "cpu",
    },
}


def expand_preset(name: str, overrides: dict | None = None) -> dict:
    """Return a dict of engine flags for a quality preset, with optional overrides."""
    if name not in QUALITY_PRESETS:
        raise ValueError(
            f"Unknown quality preset {name!r}. "
            f"Choose one of: {sorted(QUALITY_PRESETS.keys())}"
        )
    flags = copy.deepcopy(QUALITY_PRESETS[name])
    if overrides:
        flags.update(overrides)
    return flags


# Import the engine. We import lazily inside build_one() to avoid loading
# torch on `--doctor` runs.
process_single_image = None  # type: ignore[assignment]


def _import_engine():
    """Import parallax_360.process_single_image lazily."""
    global process_single_image
    if process_single_image is not None:
        return
    sys.path.insert(0, str(REPO_ROOT))
    import parallax_360
    process_single_image = parallax_360.process_single_image


def _flags_to_namespace(flags: dict, *, image: Path, device: str) -> argparse.Namespace:
    """Build the argparse.Namespace that process_single_image expects."""
    return argparse.Namespace(
        image=str(image),
        batch=None,
        device=device,
        backend=flags["backend"],
        max_dim=flags["max_dim"],
        da3_process_res=flags["da3_process_res"],
        da3_model="depth-anything/DA3MONO-LARGE",
        depth_bit_depth=flags["depth_bit_depth"],
        depth_output=None,
        depth_gamma=flags["depth_gamma"],
        pole_blend=None,
        floor_keep=0.85,
        no_postproc=flags["no_postproc"],
        layered=True,
        layered_thresholds=flags["layered_thresholds"],
        layered_feather=flags["layered_feather"],
        layered_exclude_top=flags["layered_exclude_top"],
        layered_exclude_bottom=flags["layered_exclude_bottom"],
        layered_dilate=flags["layered_dilate"],
        layered_inpaint_backend=flags["layered_inpaint_backend"],
        sam=flags.get("sam", False),
        sam_k=flags.get("sam_k", 3),
        sam_bg_threshold=flags.get("sam_bg_threshold", 0.20),
        sam_device=flags.get("sam_device", "cpu"),
    )


def build_one(
    *,
    image: Path,
    scenes_path: Path,
    out_dir: Path,
    preset_name: str,
    device: str,
    force: bool,
    preloaded: tuple | None,
    overrides: dict | None = None,
) -> None:
    """Run the depth + LDI pipeline for one image and register the scene."""
    _import_engine()
    flags = expand_preset(preset_name, overrides=overrides)
    n_layers = len(flags["layered_thresholds"])
    scene_id = infer_id_from_filename(str(image))

    out_dir.mkdir(parents=True, exist_ok=True)
    depth_output = out_dir / f"depth_{image.stem}.png"

    ns = _flags_to_namespace(flags, image=image, device=device)
    process_single_image(image, depth_output, ns, REPO_ROOT, preloaded=preloaded)

    entry = render_scene_entry(
        scene_id=scene_id,
        image_basename=image.name,
        n_foreground_layers=n_layers,
    )
    with scenes_edit_session(scenes_path):
        if force and scene_id in set(list_scene_ids(scenes_path)):
            replace_scene_entry(scenes_path, scene_id=scene_id, entry=entry)
        else:
            append_scene_entry(scenes_path, entry)


def _load_depth_model_once(args: argparse.Namespace) -> tuple | None:
    """Pre-load the depth model so all images in a batch share it."""
    sys.path.insert(0, str(REPO_ROOT))
    import parallax_360
    return parallax_360.load_depth_model(
        device=args.device,
        backend=args.backend,
        da3_model=args.da3_model,
    )


def build_batch(
    *,
    images: list[Path],
    scenes_path: Path,
    out_dir: Path,
    preset_name: str,
    device: str,
    force: bool,
    overrides: dict | None = None,
) -> dict:
    """
    Run build_one for each image. Loads the model once.
    Returns {"succeeded": [...ids], "failed": [{"id": ..., "error": ...}]}.
    """
    _import_engine()
    flags = expand_preset(preset_name, overrides=overrides)

    preflight_checks(
        images=images,
        scenes_path=scenes_path,
        out_dir=out_dir,
        force=force,
        n_foreground_layers=len(flags["layered_thresholds"]),
    )

    dummy_ns = _flags_to_namespace(flags, image=images[0], device=device)
    preloaded = _load_depth_model_once(dummy_ns)

    summary: dict = {"succeeded": [], "failed": []}
    for image in images:
        scene_id = infer_id_from_filename(str(image))
        try:
            build_one(
                image=image,
                scenes_path=scenes_path,
                out_dir=out_dir,
                preset_name=preset_name,
                device=device,
                force=force,
                preloaded=preloaded,
                overrides=overrides,
            )
            summary["succeeded"].append(scene_id)
        except Exception as exc:  # noqa: BLE001
            summary["failed"].append({"id": scene_id, "error": str(exc)})
    return summary


def format_phase_line(*, scene_id: str, phase: str, seconds: float, ok: bool) -> str:
    mark = "✓" if ok else "✗"
    return f"[build] {scene_id}: {phase}... {seconds:.1f}s {mark}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_scene.py",
        description="Build depth + LDI assets for a panorama and register it in app/scenes.ts",
    )
    p.add_argument("images", nargs="*", type=Path, help="Panorama image(s)")
    p.add_argument("--quality", choices=list(QUALITY_PRESETS.keys()), default="medium")
    p.add_argument("--sam", dest="sam_override", action="store_true", default=None,
                   help="Force SAM object-snap on (overrides the preset).")
    p.add_argument("--no-sam", dest="sam_override", action="store_false",
                   help="Force SAM object-snap off (overrides the preset).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--doctor", action="store_true")
    p.add_argument("--install-deps", action="store_true")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "public" / "parallax")
    p.add_argument("--scenes-file", type=Path, default=REPO_ROOT / "app" / "scenes.ts")
    # Common engine passthrough overrides:
    p.add_argument("--max-dim", type=int, default=None)
    p.add_argument("--da3-process-res", type=int, default=None)
    return p.parse_args(argv)


def _collect_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.max_dim is not None:
        overrides["max_dim"] = args.max_dim
    if args.da3_process_res is not None:
        overrides["da3_process_res"] = args.da3_process_res
    if args.sam_override is not None:
        overrides["sam"] = args.sam_override
    return overrides


def _run_install_deps() -> int:
    """pip install -r requirements.txt. Returns process exit code."""
    print("[build] installing dependencies from requirements.txt...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         str(REPO_ROOT / "requirements.txt")],
        check=False,
    )
    return result.returncode


def _print_dry_run(images: list[Path], preset_name: str, overrides: dict) -> None:
    flags = expand_preset(preset_name, overrides=overrides)
    n_layers = len(flags["layered_thresholds"])
    print(f"[build] DRY RUN — quality={preset_name}, {n_layers} foreground layers")
    for image in images:
        scene_id = infer_id_from_filename(str(image))
        entry = render_scene_entry(
            scene_id=scene_id,
            image_basename=image.name,
            n_foreground_layers=n_layers,
        )
        print(f"--- {image} → id={scene_id} ---")
        print(entry, end="")


def main() -> int:
    args = _parse_args()

    if args.install_deps:
        return _run_install_deps()
    if args.doctor:
        print_doctor_report()
        return 0
    if not args.images:
        print("[build] ERROR: no images given. Pass at least one panorama path "
              "or use --doctor / --install-deps.")
        return 2

    overrides = _collect_overrides(args)

    if args.dry_run:
        _print_dry_run(args.images, args.quality, overrides)
        return 0

    device = _resolve_device(args.device)
    started = time.time()
    summary = build_batch(
        images=args.images,
        scenes_path=args.scenes_file,
        out_dir=args.out_dir,
        preset_name=args.quality,
        device=device,
        force=args.force,
        overrides=overrides,
    )
    elapsed = time.time() - started

    n_ok = len(summary["succeeded"])
    n_total = n_ok + len(summary["failed"])
    print(f"[build] done. {n_ok}/{n_total} scenes built in {elapsed:.1f}s.")
    for failure in summary["failed"]:
        print(f"[build] FAILED {failure['id']}: {failure['error']}")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
