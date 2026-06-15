#!/usr/bin/env python3
"""
new_scene.py — interactive wizard for registering scenes.

Wraps scripts/build_scene.py with step-by-step prompts:
  1) Detects panoramas in public/ that aren't yet in app/scenes.ts.
  2) You pick one (or several) — you can also type a custom path.
  3) Suggests the best quality based on the installed backends.
  4) You pick a device (auto / cpu / cuda / mps).
  5) Shows the plan + a dry-run option.
  6) You confirm and it runs the pipeline live (per-phase logs).

Usage:
    python scripts/new_scene.py
    python scripts/new_scene.py --non-interactive --image foo.jpeg --quality high
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Import build_scene as a library — it does all the real work.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_scene  # noqa: E402

PUBLIC_DIR = REPO_ROOT / "public"
SCENES_TS = REPO_ROOT / "app" / "scenes.ts"
PANO_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp")

# ---------------------------------------------------------------------------
# Colors and formatting
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return c(t, "1")


def dim(t: str) -> str:
    return c(t, "2")


def cyan(t: str) -> str:
    return c(t, "36")


def green(t: str) -> str:
    return c(t, "32")


def yellow(t: str) -> str:
    return c(t, "33")


def red(t: str) -> str:
    return c(t, "31")


def header(title: str) -> None:
    print()
    print(cyan("━━━ ") + bold(title) + cyan(" ━━━"))


# ---------------------------------------------------------------------------
# Generic prompts
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str | None = None) -> str:
    """Free-text input with an optional default. Returns the string without trimming Enter."""
    suffix = dim(f" [{default}]") if default else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(yellow("Cancelled."))
            sys.exit(130)
        if raw:
            return raw
        if default is not None:
            return default
        print(yellow("(required)"))


def ask_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    """Numbered selection. Accepts typing the name or the number."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        marker = green(" ←") if opt == default else ""
        print(f"  {i}) {opt}{marker}")
    while True:
        raw = ask(">", default=default)
        if raw in options:
            return raw
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print(yellow(f"Pick a number 1-{len(options)} or one of: {', '.join(options)}"))


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        raw = ask(f"{prompt} ({default_str})", default="").lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(yellow("Answer 'y' or 'n'."))


# ---------------------------------------------------------------------------
# On-demand install of missing packages
# ---------------------------------------------------------------------------

_CORE_MODULE_TO_PKG = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "PIL": "Pillow",
}


def _missing_core_pkgs() -> list[str]:
    import importlib.util
    return [
        pkg for mod, pkg in _CORE_MODULE_TO_PKG.items()
        if importlib.util.find_spec(mod) is None
    ]


def _in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _pip_install(pip_args: list[str]) -> bool:
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", *pip_args]
    print(dim(f"$ {' '.join(cmd)}"))
    return subprocess.run(cmd, check=False).returncode == 0


def maybe_install_missing() -> None:
    """
    Detects base deps (cv2/numpy/Pillow) and optional backends (LaMa, DA3) and
    offers to install whatever is missing. If the user declines and base deps are
    missing, exits. If only optional backends are missing, continues (with lower
    available quality).
    """
    import importlib

    missing_core = _missing_core_pkgs()
    backends = build_scene.detect_backends()
    missing_lama = not backends.get("lama", False)
    missing_da3 = not backends.get("da3", False)

    if not missing_core and not missing_lama and not missing_da3:
        return

    print()
    print(yellow("Missing dependencies:"))
    if missing_core:
        print(f"  • Base packages: {', '.join(missing_core)} "
              f"{dim('(required to run the pipeline)')}")
    if missing_lama:
        print(f"  • Backend lama (inpainting LaMa)")
    if missing_da3:
        print(f"  • Backend da3 (Depth-Anything-3) "
              f"{dim('— via git, ~1.5 GB with weights')}")

    if not _in_venv():
        print()
        print(yellow("⚠ You're not inside a venv."))
        print(dim("  pip will install into your global Python. Consider running "
                  "`source venv/bin/activate` first."))

    if not ask_yes_no("Install now?", default=True):
        if missing_core:
            print(red("Without the base packages the pipeline cannot start. Exiting."))
            sys.exit(1)
        return

    # 1) requirements.txt covers core + LaMa + utilities
    if missing_core or missing_lama:
        print()
        print(dim("Installing base dependencies (requirements.txt)..."))
        if not _pip_install(["-r", str(REPO_ROOT / "requirements.txt")]):
            print(red("requirements.txt install failed."))
            importlib.invalidate_caches()
            if _missing_core_pkgs():
                sys.exit(1)
            if not ask_yes_no("Continue anyway?", default=False):
                sys.exit(1)
        importlib.invalidate_caches()

    # 2) DA3 — separate install (git+url, heavy, asked separately)
    if missing_da3 and not build_scene.detect_backends().get("da3", False):
        print()
        if ask_yes_no("Install DA3 as well?", default=True):
            print(dim("Installing DA3 (git+url)..."))
            if not _pip_install([
                "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git",
            ]):
                print(red("DA3 failed — qualities 'high' and 'ultra' will degrade."))
            importlib.invalidate_caches()

    # Final status
    backends = build_scene.detect_backends()
    available = [k for k, v in backends.items() if v]
    print()
    print(green(f"✓ Backends now: {', '.join(available)}"))
    print(dim(f"Highest quality reachable: {build_scene.max_quality_available(backends)}"))


# ---------------------------------------------------------------------------
# Candidate panorama detection
# ---------------------------------------------------------------------------

def discover_panoramas() -> tuple[list[Path], set[str]]:
    """
    Returns (candidates, registered_ids).

    Candidates: files in public/ with an image extension that are NOT already
    in scenes.ts (filtered by the id inferred from the filename AND by registered
    imageSrc, because a manually set id may not match the filename). The
    public/parallax/ directory is excluded — that's where generated outputs go,
    not inputs.
    """
    import re

    registered_ids: set[str] = set()
    registered_image_srcs: set[str] = set()
    if SCENES_TS.exists():
        registered_ids = set(build_scene.list_scene_ids(SCENES_TS))
        text = SCENES_TS.read_text()
        registered_image_srcs = set(
            re.findall(r'imageSrc:\s*"(/[^"]+)"', text)
        )

    candidates: list[Path] = []
    if PUBLIC_DIR.exists():
        for p in sorted(PUBLIC_DIR.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in PANO_EXTENSIONS:
                continue
            # Filters: pipeline outputs and editor products
            if "-bg" in p.stem or "-fg" in p.stem or "depth_" in p.stem:
                continue
            try:
                scene_id = build_scene.infer_id_from_filename(str(p))
            except ValueError:
                continue
            if scene_id in registered_ids:
                continue
            # If the file is already referenced as imageSrc under ANOTHER id,
            # also treat it as "already in use".
            if f"/{p.name}" in registered_image_srcs:
                continue
            candidates.append(p)
    return candidates, registered_ids


# ---------------------------------------------------------------------------
# Image selection (one / several / custom)
# ---------------------------------------------------------------------------

def pick_images(candidates: list[Path], registered: set[str]) -> list[Path]:
    if registered:
        print(dim(f"Already {len(registered)} scenes registered: {', '.join(sorted(registered))}"))

    if not candidates:
        print(yellow("No unregistered panoramas found in public/."))
        print("You can enter a path manually (repo-relative or absolute):")
        path_str = ask("path to panorama")
        p = Path(path_str)
        if not p.is_absolute():
            p = (REPO_ROOT / path_str).resolve()
        if not p.exists():
            print(red(f"Can't find {p}"))
            sys.exit(1)
        return [p]

    print(f"Found {len(candidates)} unregistered panorama(s):")
    for i, p in enumerate(candidates, 1):
        try:
            sid = build_scene.infer_id_from_filename(str(p))
        except ValueError:
            sid = "?"
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {i}) {p.name}  {dim(f'→ id={sid}, {size_mb:.1f} MB')}")
    print(f"  {len(candidates) + 1}) " + bold("all (batch)"))
    print(f"  {len(candidates) + 2}) " + bold("other path…"))

    while True:
        raw = ask(">", default="1")
        # Case: simple number
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                return [candidates[idx - 1]]
            if idx == len(candidates) + 1:
                return list(candidates)
            if idx == len(candidates) + 2:
                path_str = ask("path to panorama")
                p = Path(path_str)
                if not p.is_absolute():
                    p = (REPO_ROOT / path_str).resolve()
                if not p.exists():
                    print(red(f"Can't find {p}"))
                    continue
                return [p]
        # Case: comma-separated list — several but not all
        if "," in raw:
            try:
                picks: list[Path] = []
                for token in raw.split(","):
                    i = int(token.strip())
                    if not (1 <= i <= len(candidates)):
                        raise ValueError
                    picks.append(candidates[i - 1])
                if picks:
                    return picks
            except ValueError:
                pass
        print(yellow(f"Pick 1-{len(candidates) + 2} or a list like '1,3,4'."))


# ---------------------------------------------------------------------------
# Quality and device
# ---------------------------------------------------------------------------

def suggest_quality() -> tuple[str, str]:
    """
    Returns (suggestion, hint). The suggestion is computed from detect_backends:
    the highest available quality, but capped at 'high' to avoid alarming users
    with 'ultra' (which requires a GPU and lots of RAM).
    """
    backends = build_scene.detect_backends()
    max_q = build_scene.max_quality_available(backends)
    if max_q == "ultra":
        suggested = "high"  # step down one notch unless explicitly requested
        hint = f"you can bump to 'ultra' (DA3 + GPU) for maximum quality"
    else:
        suggested = max_q
        hint = f"highest quality detected with your installed backends"
    return suggested, hint


def pick_quality() -> str:
    suggested, hint = suggest_quality()
    print(dim(hint))
    return ask_choice(
        "Pick quality:",
        options=list(build_scene.QUALITY_PRESETS.keys()),
        default=suggested,
    )


def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            default_dev = "cuda"
            hint = "cuda detected"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            default_dev = "mps"
            hint = "Apple Silicon MPS detected"
        else:
            default_dev = "cpu"
            hint = "no GPU — falling back to CPU"
    except Exception:
        default_dev = "auto"
        hint = "could not detect torch"
    print(dim(hint))
    return ask_choice(
        "Pick device:",
        options=["auto", "cpu", "cuda", "mps"],
        default=default_dev,
    )


# ---------------------------------------------------------------------------
# Plan summary + execution
# ---------------------------------------------------------------------------

def show_plan(
    images: list[Path],
    quality: str,
    device: str,
    force: bool,
    sam_enabled: bool = False,
) -> None:
    flags = build_scene.expand_preset(quality)
    n_fg = len(flags["layered_thresholds"])
    backend = flags["backend"]
    max_dim = flags["max_dim"]
    quality_detail = dim(f"(backend={backend}, max_dim={max_dim}, {n_fg} fg layers)")
    header("Plan")
    print(f"  {bold('Quality:')}   {quality} {quality_detail}")
    print(f"  {bold('Device:')}    {device}")
    print(f"  {bold('SAM:')}       {'yes (object-snap)' if sam_enabled else 'no'}")
    print(f"  {bold('Force:')}     {'yes' if force else 'no'}")
    print(f"  {bold('Images:')}    {len(images)}")
    for img in images:
        try:
            sid = build_scene.infer_id_from_filename(str(img))
        except ValueError:
            sid = "?"
        print(f"    - {img.name}  {dim(f'→ id={sid}')}")
    print(f"  {bold('Out dir:')}   public/parallax/")
    print(f"  {bold('scenes.ts:')} new entries appended between the markers")


def run_pipeline(
    images: list[Path],
    quality: str,
    device: str,
    force: bool,
    dry_run: bool,
    sam_enabled: bool = False,
) -> int:
    out_dir = REPO_ROOT / "public" / "parallax"
    overrides: dict | None = {"sam": True} if sam_enabled else None

    if dry_run:
        header("DRY RUN — snippet TS")
        flags = build_scene.expand_preset(quality)
        for img in images:
            sid = build_scene.infer_id_from_filename(str(img))
            entry = build_scene.render_scene_entry(
                scene_id=sid,
                image_basename=img.name,
                n_foreground_layers=len(flags["layered_thresholds"]),
            )
            print(dim(f"--- {img.name} → id={sid} ---"))
            print(entry, end="")
        return 0

    header("Running")
    started = time.time()
    summary = build_scene.build_batch(
        images=images,
        scenes_path=SCENES_TS,
        out_dir=out_dir,
        preset_name=quality,
        device=device,
        force=force,
        overrides=overrides,
    )
    elapsed = time.time() - started

    header("Result")
    n_ok = len(summary["succeeded"])
    n_total = n_ok + len(summary["failed"])
    if n_ok == n_total:
        print(green(f"✓ {n_ok}/{n_total} scenes registered in {elapsed:.1f}s"))
    else:
        print(yellow(f"⚠ {n_ok}/{n_total} scenes registered in {elapsed:.1f}s"))
    for sid in summary["succeeded"]:
        print(f"  {green('✓')} {sid}")
    for failure in summary["failed"]:
        print(f"  {red('✗')} {failure['id']}: {failure['error']}")

    if n_ok > 0:
        print()
        print(dim("Start the dev server to see them: ") + bold("npm run dev"))

    return 0 if not summary["failed"] else 1


# ---------------------------------------------------------------------------
# Non-interactive mode (--non-interactive --image ... --quality ...)
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="new_scene.py",
        description="Interactive wizard for registering scenes (build_scene wrapper).",
    )
    p.add_argument("--non-interactive", action="store_true",
                   help="Skip prompts. Requires --image and --quality.")
    p.add_argument("--image", action="append", type=Path,
                   help="Panorama path (repeat for batch mode).")
    p.add_argument("--quality", choices=list(build_scene.QUALITY_PRESETS.keys()),
                   help="Quality preset.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--force", action="store_true",
                   help="Replace scene if the id already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan + TS snippets without writing anything.")
    return p.parse_args()


def main() -> int:
    args = parse_cli()

    # Non-interactive mode: run straight through.
    if args.non_interactive:
        if not args.image or not args.quality:
            print(red("--non-interactive requires --image and --quality."))
            return 2
        return run_pipeline(
            images=args.image,
            quality=args.quality,
            device=args.device,
            force=args.force,
            dry_run=args.dry_run,
        )

    # Interactive mode
    header("New scene wizard")
    print(dim("Ctrl+C at any time to cancel."))

    # Pre-flight: compact doctor
    backends = build_scene.detect_backends()
    available = [k for k, v in backends.items() if v]
    print(dim(f"Available backends: {', '.join(available)}"))
    print(dim(f"Highest quality reachable: {build_scene.max_quality_available(backends)}"))

    # Offer to install missing deps before continuing
    maybe_install_missing()

    # 1) Select panoramas
    header("Step 1 — panorama(s)")
    candidates, registered = discover_panoramas()
    images = pick_images(candidates, registered)

    # 2) Quality
    header("Step 2 — quality")
    quality = pick_quality()

    # 2b) SAM object-snap (only offered when weights are installed)
    sam_enabled = False
    try:
        from layered_360 import is_sam_available  # noqa: PLC0415
        if is_sam_available():
            sam_enabled = ask_yes_no(
                "Enable SAM object-snap? (slower, cleaner object edges)",
                default=False,
            )
    except ImportError:
        pass

    # 3) Device
    header("Step 3 — device")
    device = pick_device()

    # 4) Force? Needed when:
    #    (a) a scene id is already registered in scenes.ts, or
    #    (b) there are outputs in public/parallax/ from a previous run (typically
    #        a crash that left depth_xxx.png or -bg/-fg{i}.* on disk).
    force = False
    clash_ids: list[str] = []
    clash_files: list[Path] = []
    out_dir = REPO_ROOT / "public" / "parallax"
    flags_now = build_scene.expand_preset(quality)
    n_layers_now = len(flags_now["layered_thresholds"])
    for img in images:
        try:
            sid = build_scene.infer_id_from_filename(str(img))
            if sid in registered:
                clash_ids.append(sid)
            for output in build_scene._expected_output_paths(out_dir, img, n_layers_now):
                if output.exists():
                    clash_files.append(output)
        except ValueError:
            pass

    if clash_ids or clash_files:
        print()
        if clash_ids:
            print(yellow(f"These scenes are already registered in scenes.ts: "
                         f"{', '.join(clash_ids)}"))
        if clash_files:
            print(yellow(f"{len(clash_files)} output file(s) already exist "
                         f"(previous run / crash):"))
            for f in clash_files[:6]:
                try:
                    print(f"  - {f.relative_to(REPO_ROOT)}")
                except ValueError:
                    print(f"  - {f}")
            if len(clash_files) > 6:
                print(f"  … and {len(clash_files) - 6} more")
        default_force = bool(clash_files and not clash_ids)
        force = ask_yes_no("Overwrite (--force)?", default=default_force)
        if not force:
            print(red("Without --force the preflight will fail. Cancelling."))
            return 1

    # 5) Confirm
    show_plan(images, quality, device, force, sam_enabled=sam_enabled)
    print()
    if ask_yes_no("See a dry-run first (writes nothing)?", default=False):
        run_pipeline(images, quality, device, force, dry_run=True,
                     sam_enabled=sam_enabled)
        print()
        if not ask_yes_no("Proceed with the real run now?", default=True):
            print(yellow("Cancelled."))
            return 0
    elif not ask_yes_no("Run it?", default=True):
        print(yellow("Cancelled."))
        return 0

    # 6) Execute
    return run_pipeline(images, quality, device, force, dry_run=False,
                        sam_enabled=sam_enabled)


if __name__ == "__main__":
    sys.exit(main())
