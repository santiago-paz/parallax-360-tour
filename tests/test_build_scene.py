"""Tests for scripts/build_scene.py."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make `scripts/` importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_scene


def test_module_importable():
    assert hasattr(build_scene, "main")


@pytest.mark.parametrize("filename, expected", [
    ("public/image-1-360.webp", "image-1-360"),
    ("public/Cocina 360.jpeg", "cocina-360"),
    ("public/Living_Room.JPG", "living-room"),
    ("public/cuarto--doble.png", "cuarto-doble"),
    ("public/-leading-dashes-.jpeg", "leading-dashes"),
])
def test_infer_id_from_filename(filename, expected):
    assert build_scene.infer_id_from_filename(filename) == expected


@pytest.mark.parametrize("filename", [
    "public/.jpeg",        # no stem
    "public/---.jpeg",     # only separators
    "public/123.jpeg",     # OK first char must be alnum; but "123" matches the regex — keep this as a valid case
])
def test_infer_id_invalid(filename):
    if filename.endswith("/123.jpeg"):
        # "123" is a valid id per our regex; assert it returns it.
        assert build_scene.infer_id_from_filename(filename) == "123"
    else:
        with pytest.raises(ValueError):
            build_scene.infer_id_from_filename(filename)


@pytest.mark.parametrize("scene_id, expected", [
    ("image-1-360", "IMAGE 1 360"),
    ("cocina", "COCINA"),
    ("cuarto-doble", "CUARTO DOBLE"),
])
def test_label_from_id(scene_id, expected):
    assert build_scene.label_from_id(scene_id) == expected


def test_preset_table_has_four_levels():
    assert set(build_scene.QUALITY_PRESETS.keys()) == {"low", "medium", "high", "ultra"}


def test_medium_preset_expands_to_engine_flags():
    flags = build_scene.expand_preset("medium")
    assert flags["backend"] == "dav2"
    assert flags["max_dim"] == 2048
    assert flags["layered_thresholds"] == [0.60, 0.40]
    assert flags["layered_inpaint_backend"] in ("auto", "telea")
    assert flags["no_postproc"] is False


def test_high_preset_uses_da3():
    flags = build_scene.expand_preset("high")
    assert flags["backend"] == "da3"
    assert flags["da3_process_res"] == 504
    assert len(flags["layered_thresholds"]) == 3  # 3 fg layers


def test_ultra_preset_uses_da3_at_high_res():
    flags = build_scene.expand_preset("ultra")
    assert flags["backend"] == "da3"
    assert flags["da3_process_res"] == 1008
    assert flags["max_dim"] == 4096
    assert len(flags["layered_thresholds"]) == 4  # 4 fg layers


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown quality preset"):
        build_scene.expand_preset("garbage")


def test_preset_override_keeps_other_flags():
    base = build_scene.expand_preset("high")
    override = build_scene.expand_preset("high", overrides={"max_dim": 4096})
    assert override["max_dim"] == 4096
    # Everything else stays
    assert override["backend"] == base["backend"]
    assert override["da3_process_res"] == base["da3_process_res"]
    assert override["layered_thresholds"] == base["layered_thresholds"]


def test_preset_mutation_does_not_leak_to_constant():
    first = build_scene.expand_preset("high")
    first["layered_thresholds"].append(999.0)
    second = build_scene.expand_preset("high")
    assert second["layered_thresholds"] == [0.65, 0.50, 0.35]


def test_render_scene_entry_4_layers():
    rendered = build_scene.render_scene_entry(
        scene_id="cocina-360",
        image_basename="cocina-360.jpeg",
        n_foreground_layers=4,
    )
    expected = (
        '  {\n'
        '    id: "cocina-360",\n'
        '    imageSrc: "/cocina-360.jpeg",\n'
        '    depthSrc: "/parallax/depth_cocina-360.png",\n'
        '    label: "COCINA 360",\n'
        '    layered: {\n'
        '      backgroundSrc: "/parallax/cocina-360-bg.jpeg",\n'
        '      foregroundLayers: [\n'
        '        { src: "/parallax/cocina-360-fg0.webp" },\n'
        '        { src: "/parallax/cocina-360-fg1.webp" },\n'
        '        { src: "/parallax/cocina-360-fg2.webp" },\n'
        '        { src: "/parallax/cocina-360-fg3.webp" },\n'
        '      ],\n'
        '    },\n'
        '  },\n'
    )
    assert rendered == expected


def test_render_scene_entry_2_layers():
    rendered = build_scene.render_scene_entry(
        scene_id="x",
        image_basename="x.webp",
        n_foreground_layers=2,
    )
    assert 'src: "/parallax/x-fg0.webp"' in rendered
    assert 'src: "/parallax/x-fg1.webp"' in rendered
    assert "fg2" not in rendered
    assert rendered.count("foregroundLayers") == 1


def test_render_scene_entry_zero_layers_raises():
    with pytest.raises(ValueError, match="n_foreground_layers"):
        build_scene.render_scene_entry(
            scene_id="x",
            image_basename="x.webp",
            n_foreground_layers=0,
        )


def test_render_scene_entry_rejects_path_with_slash():
    with pytest.raises(ValueError, match="filename only"):
        build_scene.render_scene_entry(
            scene_id="x",
            image_basename="public/x.jpeg",  # path, not basename
            n_foreground_layers=2,
        )


MINIMAL_SCENES_TS = """\
import type { SceneConfig } from "./components/parallax360/types";

export const SCENES: SceneConfig[] = [
  // <build_scene:start>
  {
    id: "image-1",
    imageSrc: "/image-1-360.webp",
    depthSrc: "/parallax/depth_image-1-360.png",
    label: "IMAGE 1",
    layered: {
      backgroundSrc: "/parallax/image-1-360-bg.jpeg",
      foregroundLayers: [
        { src: "/parallax/image-1-360-fg0.webp" },
      ],
    },
  },
  // <build_scene:end>
];

export const FOV = 75;
"""


def test_list_scene_ids_reads_existing(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)
    assert build_scene.list_scene_ids(p) == ["image-1"]


def test_append_scene_inserts_before_end_marker(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)

    entry = build_scene.render_scene_entry(
        scene_id="cocina",
        image_basename="cocina.jpeg",
        n_foreground_layers=2,
    )
    build_scene.append_scene_entry(p, entry)

    new = p.read_text()
    assert "// <build_scene:end>" in new
    image1_pos = new.index('id: "image-1"')
    cocina_pos = new.index('id: "cocina"')
    end_pos = new.index("// <build_scene:end>")
    assert image1_pos < cocina_pos < end_pos
    assert "export const FOV = 75;" in new
    assert 'import type { SceneConfig }' in new


def test_append_fails_without_markers(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text("export const SCENES = [];\n")
    with pytest.raises(ValueError, match="build_scene:start"):
        build_scene.append_scene_entry(p, "  { id: 'x' },\n")


def test_append_fails_on_duplicate_id(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)
    entry = build_scene.render_scene_entry(
        scene_id="image-1",
        image_basename="image-1.jpeg",
        n_foreground_layers=2,
    )
    with pytest.raises(ValueError, match="already exists"):
        build_scene.append_scene_entry(p, entry)


def test_append_scene_preserves_end_marker_indentation(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)
    entry = build_scene.render_scene_entry(
        scene_id="cocina", image_basename="cocina.jpeg", n_foreground_layers=2,
    )
    build_scene.append_scene_entry(p, entry)
    new = p.read_text()
    # End marker still indented with 2 spaces (matches original).
    assert "\n  // <build_scene:end>" in new
    # And the new entry's opening brace is also 2-space indented (not 4).
    assert "\n  {\n    id: \"cocina\"" in new


def test_replace_scene_swaps_in_place(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)

    new_entry = build_scene.render_scene_entry(
        scene_id="image-1",
        image_basename="image-1-new.jpeg",
        n_foreground_layers=3,
    )
    build_scene.replace_scene_entry(p, scene_id="image-1", entry=new_entry)

    new = p.read_text()
    assert 'imageSrc: "/image-1-new.jpeg"' in new
    assert "image-1-360.webp" not in new
    assert new.count('id: "image-1"') == 1


def test_replace_scene_fails_if_not_found(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)
    new_entry = build_scene.render_scene_entry(
        scene_id="not-there",
        image_basename="x.jpeg",
        n_foreground_layers=2,
    )
    with pytest.raises(ValueError, match="not found"):
        build_scene.replace_scene_entry(p, scene_id="not-there", entry=new_entry)


def test_commit_with_validation_success(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)

    def mutate():
        entry = build_scene.render_scene_entry(
            scene_id="cocina", image_basename="cocina.jpeg", n_foreground_layers=2,
        )
        build_scene.append_scene_entry(p, entry)

    with patch.object(build_scene, "_run_tsc_noemit", return_value=(0, "")):
        with build_scene.scenes_edit_session(p):
            mutate()

    assert (p.with_suffix(p.suffix + ".bak")).exists()
    assert 'id: "cocina"' in p.read_text()


def test_commit_with_validation_rollback(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)
    original = p.read_text()

    def bad_mutate():
        p.write_text(p.read_text() + "\n***broken***\n")

    with patch.object(build_scene, "_run_tsc_noemit", return_value=(1, "TS1005: expected ';'")):
        with pytest.raises(RuntimeError, match="tsc validation failed"):
            with build_scene.scenes_edit_session(p):
                bad_mutate()

    assert p.read_text() == original


def test_commit_with_missing_tsc_warns_but_does_not_block(tmp_path, capsys):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)

    def mutate():
        entry = build_scene.render_scene_entry(
            scene_id="cocina", image_basename="cocina.jpeg", n_foreground_layers=2,
        )
        build_scene.append_scene_entry(p, entry)

    with patch.object(build_scene, "_run_tsc_noemit",
                      side_effect=FileNotFoundError("npx not on PATH")):
        with build_scene.scenes_edit_session(p):
            mutate()

    assert 'id: "cocina"' in p.read_text()
    captured = capsys.readouterr()
    assert "WARN" in captured.out
    assert "tsc" in captured.out


def test_mutation_exception_triggers_rollback(tmp_path):
    p = tmp_path / "scenes.ts"
    p.write_text(MINIMAL_SCENES_TS)
    original = p.read_text()

    class _Boom(RuntimeError):
        pass

    def bad_mutate():
        # Mutate the file, then raise — verifies the manager restores
        # the .bak before re-raising.
        p.write_text(p.read_text() + "\nbogus\n")
        raise _Boom("simulated user error inside the with-block")

    with patch.object(build_scene, "_run_tsc_noemit", return_value=(0, "")):
        with pytest.raises(_Boom):
            with build_scene.scenes_edit_session(p):
                bad_mutate()

    # File restored from backup despite the mutation.
    assert p.read_text() == original


def test_detect_backends_synthetic_always_available():
    backends = build_scene.detect_backends()
    assert backends["synthetic"] is True


def test_detect_backends_returns_bools_for_all_known_keys():
    backends = build_scene.detect_backends()
    for key in ("synthetic", "midas", "v2", "da3", "lama"):
        assert key in backends
        assert isinstance(backends[key], bool)


def test_max_quality_with_only_synthetic():
    assert build_scene.max_quality_available({
        "synthetic": True, "midas": False, "v2": False, "da3": False, "lama": False,
    }) == "low"


def test_max_quality_with_da3_and_lama():
    assert build_scene.max_quality_available({
        "synthetic": True, "midas": True, "v2": True, "da3": True, "lama": True,
    }) == "ultra"


def test_max_quality_with_v2_only():
    assert build_scene.max_quality_available({
        "synthetic": True, "midas": True, "v2": True, "da3": False, "lama": False,
    }) == "medium"


def test_doctor_output_lists_all_backends(capsys):
    build_scene.print_doctor_report()
    captured = capsys.readouterr().out
    for name in ("synthetic", "midas", "v2", "da3", "lama"):
        assert name in captured
    assert "Highest quality" in captured


def test_preflight_passes_for_clean_input(tmp_path):
    img = tmp_path / "cocina-360.jpeg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake_jpeg_header")
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    out_dir = tmp_path / "parallax"
    out_dir.mkdir()
    # Should not raise.
    build_scene.preflight_checks(
        images=[img],
        scenes_path=scenes_ts,
        out_dir=out_dir,
        force=False,
        n_foreground_layers=4,
    )


def test_preflight_fails_when_image_missing(tmp_path):
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    with pytest.raises(FileNotFoundError, match="ghost.jpeg"):
        build_scene.preflight_checks(
            images=[tmp_path / "ghost.jpeg"],
            scenes_path=scenes_ts,
            out_dir=tmp_path / "parallax",
            force=False,
            n_foreground_layers=4,
        )


def test_preflight_fails_on_duplicate_id_without_force(tmp_path):
    img = tmp_path / "image-1.jpeg"
    img.write_bytes(b"x")
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    with pytest.raises(ValueError, match="already exists"):
        build_scene.preflight_checks(
            images=[img],
            scenes_path=scenes_ts,
            out_dir=tmp_path / "parallax",
            force=False,
            n_foreground_layers=4,
        )


def test_preflight_allows_duplicate_id_with_force(tmp_path):
    img = tmp_path / "image-1.jpeg"
    img.write_bytes(b"x")
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    build_scene.preflight_checks(
        images=[img],
        scenes_path=scenes_ts,
        out_dir=tmp_path / "parallax",
        force=True,
        n_foreground_layers=4,
    )


def test_preflight_fails_when_outputs_exist_without_force(tmp_path):
    img = tmp_path / "cocina-360.jpeg"
    img.write_bytes(b"x")
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    out_dir = tmp_path / "parallax"
    out_dir.mkdir()
    (out_dir / "depth_cocina-360.png").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="depth_cocina-360.png"):
        build_scene.preflight_checks(
            images=[img],
            scenes_path=scenes_ts,
            out_dir=out_dir,
            force=False,
            n_foreground_layers=4,
        )


def test_build_single_calls_engine_and_registers(tmp_path, monkeypatch):
    img = tmp_path / "cocina-360.jpeg"
    img.write_bytes(b"x")
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    out_dir = tmp_path / "parallax"
    out_dir.mkdir()

    captured_args = {}

    def fake_process_single_image(image_path, depth_output, args, repo_root, preloaded=None):
        captured_args["image_path"] = image_path
        captured_args["depth_output"] = depth_output
        captured_args["backend"] = args.backend
        captured_args["thresholds"] = list(args.layered_thresholds)
        depth_output.write_bytes(b"depth")
        (out_dir / f"{image_path.stem}-bg.jpeg").write_bytes(b"bg")
        for i in range(len(args.layered_thresholds)):
            (out_dir / f"{image_path.stem}-fg{i}.webp").write_bytes(b"fg")
        return None

    monkeypatch.setattr(build_scene, "process_single_image", fake_process_single_image)
    monkeypatch.setattr(build_scene, "_run_tsc_noemit", lambda root: (0, ""))

    build_scene.build_one(
        image=img,
        scenes_path=scenes_ts,
        out_dir=out_dir,
        preset_name="high",
        device="cpu",
        force=False,
        preloaded=None,
    )

    assert captured_args["image_path"] == img
    assert captured_args["backend"] == "da3"
    assert captured_args["thresholds"] == [0.65, 0.50, 0.35]
    new = scenes_ts.read_text()
    assert 'id: "cocina-360"' in new
    assert 'imageSrc: "/cocina-360.jpeg"' in new


def test_batch_loads_model_once(tmp_path, monkeypatch):
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    out_dir = tmp_path / "parallax"
    out_dir.mkdir()

    images = []
    for name in ("scene-a", "scene-b", "scene-c"):
        p = tmp_path / f"{name}.jpeg"
        p.write_bytes(b"x")
        images.append(p)

    load_calls = {"count": 0}

    def fake_load(args):
        load_calls["count"] += 1
        return ("fake_model", "fake_backend")

    def fake_process(image_path, depth_output, args, repo_root, preloaded=None):
        depth_output.parent.mkdir(parents=True, exist_ok=True)
        depth_output.write_bytes(b"d")
        (depth_output.parent / f"{image_path.stem}-bg.jpeg").write_bytes(b"bg")
        for i in range(len(args.layered_thresholds)):
            (depth_output.parent / f"{image_path.stem}-fg{i}.webp").write_bytes(b"fg")

    monkeypatch.setattr(build_scene, "_load_depth_model_once", fake_load)
    monkeypatch.setattr(build_scene, "process_single_image", fake_process)
    monkeypatch.setattr(build_scene, "_import_engine", lambda: None)
    monkeypatch.setattr(build_scene, "_run_tsc_noemit", lambda root: (0, ""))

    summary = build_scene.build_batch(
        images=images,
        scenes_path=scenes_ts,
        out_dir=out_dir,
        preset_name="high",
        device="cpu",
        force=False,
    )

    assert load_calls["count"] == 1
    assert summary["succeeded"] == ["scene-a", "scene-b", "scene-c"]
    assert summary["failed"] == []


def test_batch_continues_after_single_failure(tmp_path, monkeypatch):
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    out_dir = tmp_path / "parallax"
    out_dir.mkdir()

    images = []
    for name in ("ok-1", "broken", "ok-2"):
        p = tmp_path / f"{name}.jpeg"
        p.write_bytes(b"x")
        images.append(p)

    def fake_process(image_path, depth_output, args, repo_root, preloaded=None):
        if image_path.stem == "broken":
            raise RuntimeError("simulated DA3 OOM")
        depth_output.parent.mkdir(parents=True, exist_ok=True)
        depth_output.write_bytes(b"d")
        (depth_output.parent / f"{image_path.stem}-bg.jpeg").write_bytes(b"bg")
        for i in range(len(args.layered_thresholds)):
            (depth_output.parent / f"{image_path.stem}-fg{i}.webp").write_bytes(b"fg")

    monkeypatch.setattr(build_scene, "_load_depth_model_once", lambda args: ("m", "b"))
    monkeypatch.setattr(build_scene, "process_single_image", fake_process)
    monkeypatch.setattr(build_scene, "_import_engine", lambda: None)
    monkeypatch.setattr(build_scene, "_run_tsc_noemit", lambda root: (0, ""))

    summary = build_scene.build_batch(
        images=images,
        scenes_path=scenes_ts,
        out_dir=out_dir,
        preset_name="high",
        device="cpu",
        force=False,
    )

    assert summary["succeeded"] == ["ok-1", "ok-2"]
    assert len(summary["failed"]) == 1
    assert summary["failed"][0]["id"] == "broken"
    text = scenes_ts.read_text()
    assert 'id: "ok-1"' in text
    assert 'id: "ok-2"' in text
    assert 'id: "broken"' not in text


def test_cli_doctor_short_circuits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_scene.py", "--doctor"])
    code = build_scene.main()
    assert code == 0
    out = capsys.readouterr().out
    assert "synthetic" in out
    assert "Highest quality" in out


def test_cli_dry_run_prints_entries_and_does_not_touch_files(tmp_path, monkeypatch, capsys):
    img = tmp_path / "cocina-360.jpeg"
    img.write_bytes(b"x")
    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)

    monkeypatch.setattr(sys, "argv", [
        "build_scene.py",
        str(img),
        "--quality", "high",
        "--scenes-file", str(scenes_ts),
        "--out-dir", str(tmp_path / "parallax"),
        "--dry-run",
    ])

    code = build_scene.main()
    assert code == 0

    out = capsys.readouterr().out
    assert 'id: "cocina-360"' in out
    # No files written.
    assert scenes_ts.read_text() == MINIMAL_SCENES_TS
    assert not (tmp_path / "parallax").exists()


def test_cli_install_deps_runs_pip(monkeypatch, capsys):
    pip_calls = []

    def fake_run(cmd, **kwargs):
        pip_calls.append(cmd)
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(build_scene.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["build_scene.py", "--install-deps"])
    code = build_scene.main()
    assert code == 0
    assert any("requirements.txt" in " ".join(c) for c in pip_calls)


def test_format_phase_line():
    line = build_scene.format_phase_line(
        scene_id="cocina-360",
        phase="depth (DA3, 504px)",
        seconds=32.1,
        ok=True,
    )
    assert line == "[build] cocina-360: depth (DA3, 504px)... 32.1s ✓"


def test_end_to_end_synthetic_backend(tmp_path):
    """
    Full flow with no torch dependency: panorama → depth (synthetic) →
    LDI (telea inpaint) → entry in scenes.ts. Uses the real engine modules.
    """
    import numpy as np
    import cv2

    # Real-ish panorama: 256×128 random RGB.
    img = tmp_path / "smoke-360.jpeg"
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(img), rgb)

    scenes_ts = tmp_path / "scenes.ts"
    scenes_ts.write_text(MINIMAL_SCENES_TS)
    out_dir = tmp_path / "parallax"

    # Force synthetic backend regardless of preset; treat tsc as available with PASS.
    overrides = {
        "backend": "synthetic",
        "max_dim": 256,
        "layered_thresholds": [0.5],
        "layered_exclude_top": 0.0,
        "layered_exclude_bottom": 0.0,
    }

    with patch.object(build_scene, "_run_tsc_noemit", return_value=(0, "")):
        summary = build_scene.build_batch(
            images=[img],
            scenes_path=scenes_ts,
            out_dir=out_dir,
            preset_name="low",
            device="cpu",
            force=False,
            overrides=overrides,
        )

    assert summary["succeeded"] == ["smoke-360"]
    assert summary["failed"] == []
    assert (out_dir / "depth_smoke-360.png").exists()
    assert (out_dir / "smoke-360-bg.jpeg").exists()
    # 1 threshold → 1 foreground layer (fg0) + background.
    assert (out_dir / "smoke-360-fg0.webp").exists()
    text = scenes_ts.read_text()
    assert 'id: "smoke-360"' in text
    assert 'imageSrc: "/smoke-360.jpeg"' in text
