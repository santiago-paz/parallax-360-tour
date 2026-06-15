"use client";

import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import * as THREE from "three";
import {
  ENABLE_PARALLAX, RADIUS, SEGS_W, SEGS_H,
  MAX_OFFSET, LERP,
  DEFAULT_PARALLAX_STRENGTH, DEFAULT_DEPTH_SCALE, DEFAULT_NEAR_CLIP,
  TRANSITION_DURATION, TRANSITION_DEPTH_SCALE, TRANSITION_FLY_DIST,
  DEFAULT_FOREGROUND_RADIUS, MIN_FG_RADIUS, MAX_FG_RADIUS,
} from "./constants";
import type { ViewerHandle, Parallax360ViewerProps, TransitionState, LayeredConfig } from "./types";
import {
  buildDisplacementOffsets,
  setDisplacementStrength,
  restoreGeometry,
} from "./displacement";
import { createDebugVisuals, updateDebugLines, type DebugVisuals } from "./debugProjection";

export type { ParallaxViewerConfig, ViewerHandle } from "./types";

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * @deprecated since 2026-06-14 — depth-displaced LDI sets every fg sphere
 * to RADIUS. Kept for reference; not referenced at runtime.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function _defaultFgRadius(index: number, count: number): number {
  if (count <= 1) return DEFAULT_FOREGROUND_RADIUS;
  return MIN_FG_RADIUS
    + (MAX_FG_RADIUS - MIN_FG_RADIUS) * (index / (count - 1));
}

const Parallax360Viewer = forwardRef<ViewerHandle, Parallax360ViewerProps>(
  function Parallax360Viewer(
    { config, debugProjection = false },
    ref
  ) {
  const mountRef = useRef<HTMLDivElement>(null);
  const fadeRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const matRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const startTransitionRef = useRef<ViewerHandle["startTransition"]>(() => {});
  const setImageRef = useRef<ViewerHandle["setImage"]>(() => {});
  const setLayeredRef = useRef<ViewerHandle["setLayered"]>(() => {});

  useImperativeHandle(ref, () => ({
    getCamera: () => cameraRef.current,
    getMesh: () => meshRef.current,
    getCanvas: () => canvasRef.current,
    setImage: (...args) => setImageRef.current(...args),
    setLayered: (...args) => setLayeredRef.current(...args),
    startTransition: (...args) => startTransitionRef.current(...args),
  }));

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // ── Renderer / scene / camera ────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      config.fov, mount.clientWidth / mount.clientHeight, 0.01, 100
    );

    // ── Sphere (inside-out) ──────────────────────────────────────────────────
    const geo = new THREE.SphereGeometry(RADIUS, SEGS_W, SEGS_H);
    geo.scale(-1, 1, 1);
    const mat = new THREE.MeshBasicMaterial({ side: THREE.FrontSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = 0;
    scene.add(mesh);
    scene.add(camera);

    cameraRef.current = camera;
    meshRef.current = mesh;
    canvasRef.current = renderer.domElement;
    matRef.current = mat;
    rendererRef.current = renderer;

    // ── Displacement state ───────────────────────────────────────────────────
    // bg path (non-LDI): displaces the outer sphere using `geo`'s topology.
    const origPosRef = { current: null as Float32Array | null };
    let depthOffsets: Float32Array | null = null;
    // fg path (LDI): displaces every fg sphere using the X-mirrored topology.
    // One shared offset table — all fg spheres are built with the same base
    // radius + scale(-1,1,1), so their vertex positions are identical.
    const fgOrigPosRef = { current: null as Float32Array | null };
    let fgDepthOffsets: Float32Array | null = null;
    let fgDepthOffsetsReady = false;
    const pendingFgGeos: THREE.SphereGeometry[] = [];
    let depthLoadToken = 0;

    // ── Optional layered foreground spheres ──────────────────────────────────
    // When `config.layered` is set we render N concentric inner spheres so
    // the camera offset produces real geometric parallax with a different
    // angular shift per layer. Each inner sphere uses an RGBA panorama
    // (alpha = soft depth-slab mask) so layers behind composite through.
    let layered = config.layered;
    const fgMeshes: THREE.Mesh[] = [];
    const fgMats: THREE.MeshBasicMaterial[] = [];
    const fgGeos: THREE.SphereGeometry[] = [];

    const applyLayered = (next: LayeredConfig | undefined) => {
      // Dispose existing fg layers (textures, geometries, materials).
      for (const m of fgMeshes) scene.remove(m);
      for (const g of fgGeos) g.dispose();
      for (const mat of fgMats) {
        if (mat.map) mat.map.dispose();
        mat.dispose();
      }
      fgMeshes.length = 0;
      fgGeos.length = 0;
      fgMats.length = 0;
      pendingFgGeos.length = 0;
      fgDepthOffsetsReady = false;
      fgDepthOffsets = null;
      fgOrigPosRef.current = null;
      layered = next;
      if (!next) return;

      const layers = next.foregroundLayers;
      const N = layers.length;
      const texLoader = new THREE.TextureLoader();
      for (let k = 0; k < N; k++) {
        // All fg spheres share the bg's base radius. Per-pixel depth
        // displacement (applied below or queued for the depth onload) is
        // what gives each layer its geometric parallax.
        const fgGeo = new THREE.SphereGeometry(RADIUS, SEGS_W, SEGS_H);
        fgGeo.scale(-1, 1, 1);
        const fgMat = new THREE.MeshBasicMaterial({
          side: THREE.FrontSide,
          transparent: true,
          depthWrite: false,
          alphaTest: 0.01,
        });
        const fgMesh = new THREE.Mesh(fgGeo, fgMat);
        // Closer bins (smaller k) draw LAST so they composite on top —
        // bg(=0) < layer N-1 < … < layer 0. Render order is by bin index,
        // not radius (all radii are equal now).
        fgMesh.renderOrder = N - k;
        scene.add(fgMesh);
        fgMeshes.push(fgMesh);
        fgMats.push(fgMat);
        fgGeos.push(fgGeo);

        if (fgDepthOffsetsReady && fgDepthOffsets) {
          setDisplacementStrength(fgGeo, fgOrigPosRef, fgDepthOffsets, 1);
        } else {
          pendingFgGeos.push(fgGeo);
        }

        texLoader.load(layers[k].src, (tex) => {
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
          if (fgMat.map) fgMat.map.dispose();
          fgMat.map = tex;
          fgMat.needsUpdate = true;
        });
      }
    };

    applyLayered(layered);
    setLayeredRef.current = applyLayered;

    // ── Debug projection (parallax only) ─────────────────────────────────────
    let debug: DebugVisuals | null = null;
    if (ENABLE_PARALLAX) {
      debug = createDebugVisuals(scene, geo, debugProjection);
    }

    // ── Interaction state ────────────────────────────────────────────────────
    let yaw = 0, pitch = 0;
    let dragging = false;
    let px0 = 0, py0 = 0;
    let camX = 0, camY = 0;
    let tgtX = 0, tgtY = 0;

    const parallaxStrength = config.parallaxStrength ?? DEFAULT_PARALLAX_STRENGTH;
    const depthScale = config.depthScale ?? DEFAULT_DEPTH_SCALE;
    const nearClip = config.nearClip ?? DEFAULT_NEAR_CLIP;

    const loadDepthAndDisplace = (src: string) => {
      const token = ++depthLoadToken;
      const img = new Image();
      img.onload = () => {
        if (token !== depthLoadToken) return;   // stale load, a newer one supersedes
        const cvs = document.createElement("canvas");
        cvs.width = img.naturalWidth;
        cvs.height = img.naturalHeight;
        cvs.getContext("2d")!.drawImage(img, 0, 0);
        if (layered) {
          // LDI mode: compute offsets on a fg-shape reference geometry and
          // apply them to every fg sphere AND to the bg. Bg + fg share the
          // same topology (both use scale(-1,1,1) at RADIUS, same segments),
          // so one offset table grounds them together: floor/wall pixels on
          // the bg track their real depth, fg-object pixels on the fg track
          // theirs. The small inaccuracy at inpainted pixels (where the
          // depth map still holds the original fg object's depth) is hidden
          // by the fg layer that covers them.
          const refGeo = new THREE.SphereGeometry(RADIUS, SEGS_W, SEGS_H);
          refGeo.scale(-1, 1, 1);
          fgDepthOffsets = buildDisplacementOffsets(
            refGeo, RADIUS, fgOrigPosRef, cvs, depthScale,
            { invert: false, clipValue: nearClip },
          );
          refGeo.dispose();
          fgDepthOffsetsReady = true;
          setDisplacementStrength(geo, fgOrigPosRef, fgDepthOffsets, 1);
          for (const fgGeo of pendingFgGeos) {
            setDisplacementStrength(fgGeo, fgOrigPosRef, fgDepthOffsets, 1);
          }
          pendingFgGeos.length = 0;
        } else {
          // Non-LDI mode: displace the bg sphere as before.
          depthOffsets = buildDisplacementOffsets(
            geo, RADIUS, origPosRef, cvs, depthScale,
            { invert: false, clipValue: nearClip },
          );
          setDisplacementStrength(geo, origPosRef, depthOffsets, 1);
        }
      };
      img.src = src;
    };

    // Depth-displaced LDI: when `layered` is set, the same depth map drives
    // per-vertex displacement on every fg sphere (shared offset table). When
    // `!layered`, the bg sphere displaces as before. The branching happens
    // inside loadDepthAndDisplace.
    if (ENABLE_PARALLAX && config.depthSrc) {
      loadDepthAndDisplace(config.depthSrc);
    }

    setImageRef.current = (src: string, depthSrc?: string) => {
      const m = matRef.current;
      const r = rendererRef.current;
      if (!m || !r) return;
      const texLoader = new THREE.TextureLoader();
      texLoader.load(src, (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = r.capabilities.getMaxAnisotropy();
        if (m.map) m.map.dispose();
        m.map = tex;
        m.needsUpdate = true;
      });
      if (ENABLE_PARALLAX && depthSrc) {
        loadDepthAndDisplace(depthSrc);
      }
    };

    // ── Transition state ─────────────────────────────────────────────────────
    let transition: TransitionState | null = null;

    startTransitionRef.current = (direction, depthMapSrc, onSwap) => {
      if (transition?.active) return;

      const img = new Image();
      img.onload = () => {
        const cvs = document.createElement("canvas");
        cvs.width = img.naturalWidth;
        cvs.height = img.naturalHeight;
        cvs.getContext("2d")!.drawImage(img, 0, 0);
        // Transition uses its own (stronger) depth scale to amplify the fly-through.
        // The post-swap `setImage` rebuilds offsets at the runtime scale.
        depthOffsets = buildDisplacementOffsets(
          geo, RADIUS, origPosRef, cvs, TRANSITION_DEPTH_SCALE,
          { invert: false, clipValue: 0.08 },
        );
        // Start undeformed; we'll ramp the strength during Phase 1.
        setDisplacementStrength(geo, origPosRef, depthOffsets, 0);

        const targetDir = new THREE.Vector3(direction.x, direction.y, direction.z).normalize();
        const horizontalDir = new THREE.Vector3(targetDir.x, 0, targetDir.z);
        if (horizontalDir.lengthSq() < 1e-6) {
          // Hotspot is straight up or down; fall back to current heading.
          horizontalDir.set(Math.sin(yaw), 0, Math.cos(yaw));
        }
        horizontalDir.normalize();

        const targetYaw = Math.atan2(horizontalDir.x, horizontalDir.z);
        let deltaYaw = targetYaw - yaw;
        while (deltaYaw > Math.PI) deltaYaw -= Math.PI * 2;
        while (deltaYaw < -Math.PI) deltaYaw += Math.PI * 2;

        transition = {
          active: true,
          startTime: performance.now(),
          targetDir,
          horizontalDir,
          startYaw: yaw,
          startPitch: pitch,
          targetYaw,
          deltaYaw,
          onSwap,
          swapFired: false,
          baseFov: camera.fov,
        };
      };
      img.src = depthMapSrc;
    };

    // ── Load initial texture ─────────────────────────────────────────────────
    setImageRef.current(layered ? layered.backgroundSrc : config.imageSrc);

    // ── Pointer events ───────────────────────────────────────────────────────
    const el = renderer.domElement;
    el.style.touchAction = "none";
    let dragStartX = 0, dragStartY = 0;
    let lastPointerType = "";

    const onPointerDown = (e: PointerEvent) => {
      if (transition?.active) return;
      dragging = true; px0 = e.clientX; py0 = e.clientY;
      dragStartX = e.clientX; dragStartY = e.clientY;
      lastPointerType = e.pointerType;
      if (e.pointerType === "mouse") {
        el.setPointerCapture(e.pointerId);
      }
      mount.style.cursor = "grabbing";
    };
    const onPointerMove = (e: PointerEvent) => {
      if (transition?.active) return;
      if (ENABLE_PARALLAX) {
        tgtX = -(e.clientX / mount.clientWidth  - 0.5) * 2;
        tgtY =  (e.clientY / mount.clientHeight - 0.5) * 2;
      }
      if (!dragging) return;
      yaw   += (e.clientX - px0) * 0.003;
      pitch += (e.clientY - py0) * 0.003;
      pitch = Math.max(-Math.PI * 0.48, Math.min(Math.PI * 0.48, pitch));
      px0 = e.clientX; py0 = e.clientY;
    };
    const onPointerUp = (e: PointerEvent) => {
      if (transition?.active) { dragging = false; return; }
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      const dragDist = Math.sqrt(dx * dx + dy * dy);

      dragging = false;
      mount.style.cursor = "grab";

      const clickThreshold = e.pointerType === "touch" || lastPointerType === "touch" ? 15 : 5;
      if (dragDist >= clickThreshold && ENABLE_PARALLAX) {
        tgtX = Math.sin(yaw);
        tgtY = Math.sin(pitch);
      }
    };
    const onPointerCancel = () => {
      dragging = false;
      mount.style.cursor = "grab";
    };
    const onWheel = (e: WheelEvent) => {
      if (transition?.active) { e.preventDefault(); return; }
      e.preventDefault();
      const delta = e.ctrlKey ? e.deltaY * 0.15 : e.deltaY * 0.05;
      camera.fov = Math.max(30, Math.min(120, camera.fov + delta));
      camera.updateProjectionMatrix();
    };

    const onTouchStart = (e: TouchEvent) => { e.preventDefault(); };
    const onTouchMove = (e: TouchEvent) => { e.preventDefault(); };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerCancel);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("touchstart", onTouchStart, { passive: false });
    el.addEventListener("touchmove", onTouchMove, { passive: false });

    // ── Resize ───────────────────────────────────────────────────────────────
    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    // ── Render loop ──────────────────────────────────────────────────────────
    let rafId: number;
    const animate = () => {
      rafId = requestAnimationFrame(animate);

      // ── Transition animation ────────────────────────────────────────────
      if (transition) {
        const elapsed = performance.now() - transition.startTime;
        const progress = Math.min(elapsed / TRANSITION_DURATION, 1);
        const LEVEL_END = 0.30;
        const FADE_START = 0.60;
        const SWAP_AT = 0.82;

        if (!transition.swapFired) {
          const h = transition.horizontalDir;

          if (progress < LEVEL_END) {
            // Phase 1: level the camera + ramp depth deformation in.
            const t = easeInOutCubic(progress / LEVEL_END);
            const curYaw = transition.startYaw + transition.deltaYaw * t;
            const curPitch = transition.startPitch * (1 - t);
            const lx = Math.cos(curPitch) * Math.sin(curYaw);
            const ly = Math.sin(curPitch);
            const lz = Math.cos(curPitch) * Math.cos(curYaw);
            camera.position.set(0, 0, 0);
            camera.lookAt(lx, ly, lz);
            if (depthOffsets) {
              setDisplacementStrength(geo, origPosRef, depthOffsets, t);
            }
          } else {
            // Phase 2: walk forward through fully-deformed space.
            const walkT = easeInOutCubic(
              Math.min((progress - LEVEL_END) / (SWAP_AT - LEVEL_END), 1)
            );
            const flyDist = walkT * TRANSITION_FLY_DIST;
            camera.position.set(h.x * flyDist, 0, h.z * flyDist);
            camera.lookAt(
              h.x * (RADIUS + flyDist),
              0,
              h.z * (RADIUS + flyDist)
            );
            if (depthOffsets) {
              setDisplacementStrength(geo, origPosRef, depthOffsets, 1);
            }
          }
          camera.fov = transition.baseFov;
          camera.updateProjectionMatrix();

          if (progress >= FADE_START) {
            const t = (progress - FADE_START) / (SWAP_AT - FADE_START);
            if (fadeRef.current) fadeRef.current.style.opacity = String(t);
          }

          if (progress >= SWAP_AT) {
            transition.swapFired = true;
            restoreGeometry(geo, origPosRef);
            depthOffsets = null;
            yaw = 0;
            pitch = 0;
            camera.position.set(0, 0, 0);
            camera.fov = transition.baseFov;
            camera.updateProjectionMatrix();
            if (fadeRef.current) {
              fadeRef.current.style.opacity = "1";
              fadeRef.current.style.pointerEvents = "auto";
            }
            transition.onSwap();
          }
        } else {
          camera.position.set(0, 0, 0);
          const lx = Math.cos(pitch) * Math.sin(yaw);
          const ly = Math.sin(pitch);
          const lz = Math.cos(pitch) * Math.cos(yaw);
          camera.lookAt(lx, ly, lz);
          camera.fov = transition.baseFov;
          camera.updateProjectionMatrix();

          const fadeOutT = (progress - SWAP_AT) / (1 - SWAP_AT);
          if (fadeRef.current) {
            fadeRef.current.style.opacity = String(Math.max(0, 1 - fadeOutT));
          }
        }

        renderer.render(scene, camera);

        if (progress >= 1) {
          transition = null;
          if (fadeRef.current) {
            fadeRef.current.style.opacity = "0";
            fadeRef.current.style.pointerEvents = "none";
          }
        }
        return;
      }

      // ── Normal rendering ────────────────────────────────────────────────
      const lookX = Math.cos(pitch) * Math.sin(yaw);
      const lookY = Math.sin(pitch);
      const lookZ = Math.cos(pitch) * Math.cos(yaw);
      const rightX =  Math.cos(yaw);
      const rightZ = -Math.sin(yaw);

      if (ENABLE_PARALLAX) {
        const effectiveStrength = parallaxStrength;
        const pxTarget = dragging ? Math.sin(yaw)   : tgtX;
        const pyTarget = dragging ? Math.sin(pitch)  : tgtY;
        camX += (pxTarget * effectiveStrength * MAX_OFFSET - camX) * LERP;
        camY += (pyTarget * effectiveStrength * MAX_OFFSET * 0.5 - camY) * LERP;
        camera.position.set(camX * rightX, camY, camX * rightZ);
        camera.lookAt(
          camera.position.x + lookX,
          camera.position.y + lookY,
          camera.position.z + lookZ
        );
        if (debug?.enabled) {
          updateDebugLines(debug, camera.position, lookX, lookY, lookZ, rightX, rightZ);
        }
      } else {
        camera.position.set(0, 0, 0);
        camera.lookAt(lookX, lookY, lookZ);
      }

      renderer.render(scene, camera);
    };
    animate();

    // ── Cleanup ──────────────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", onResize);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerCancel);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      debug?.dispose();
      renderer.dispose();
      geo.dispose();
      mat.dispose();
      for (const m of fgMeshes) scene.remove(m);
      for (const g of fgGeos) g.dispose();
      for (const m of fgMats) {
        if (m.map) m.map.dispose();
        m.dispose();
      }
      pendingFgGeos.length = 0;
      if (mount.contains(el)) mount.removeChild(el);
      cameraRef.current = null;
      meshRef.current = null;
      canvasRef.current = null;
      matRef.current = null;
      rendererRef.current = null;
      startTransitionRef.current = () => {};
    };
  }, [config]);

  return (
    <div className="relative w-full h-full" style={{ cursor: "grab", touchAction: "none" }}>
      <div ref={mountRef} className="absolute inset-0" style={{ touchAction: "none" }} />
      <div
        ref={fadeRef}
        className="absolute inset-0 bg-black pointer-events-none"
        style={{ opacity: 0, zIndex: 5, transition: "none" }}
      />
    </div>
  );
});

export default Parallax360Viewer;
