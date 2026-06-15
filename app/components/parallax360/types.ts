import type * as THREE from "three";

/**
 * Layered Depth Image config — when present the viewer renders the
 * inpainted panorama on the outer sphere PLUS one extra concentric sphere
 * per foreground layer. The natural camera offset (see MAX_OFFSET) then
 * produces real geometric parallax with a different angular shift per
 * layer, replacing the rubber-sheet depth displacement and letting many
 * elements (not just one) pop out at distinct depths.
 *
 * Assets are produced by `python layered_360.py <panorama> --depth <map>
 *   --thresholds <t0> <t1> ...` (DESCENDING order, nearest first).
 */
export type LayeredForegroundLayer = {
  /** RGBA equirectangular for this depth slab (alpha = soft slab mask). */
  src: string;
  /**
   * @deprecated since 2026-06-14 — ignored by the depth-displaced LDI
   * renderer. Every fg sphere uses RADIUS as base and is displaced
   * per-pixel by the depth map; per-layer base radius no longer affects
   * geometry. Kept in the schema for source compatibility with existing
   * scenes; no runtime effect.
   */
  radius?: number;
};

export type LayeredConfig = {
  /** Opaque panorama with all foreground regions inpainted away. */
  backgroundSrc: string;
  /**
   * Foreground RGBA layers ordered closest first (matches `layered_360.py`
   * descending threshold order — layer 0 = highest depth = closest object).
   */
  foregroundLayers: LayeredForegroundLayer[];
};

export type ParallaxViewerConfig = {
  imageSrc: string;
  fov: number;
  depthSrc?: string;
  parallaxStrength?: number;
  depthScale?: number;
  nearClip?: number;
  layered?: LayeredConfig;
};

export interface ViewerHandle {
  getCamera: () => THREE.PerspectiveCamera | null;
  getMesh: () => THREE.Mesh | null;
  getCanvas: () => HTMLCanvasElement | null;
  setImage: (src: string, depthSrc?: string) => void;
  setLayered: (layered: LayeredConfig | undefined) => void;
  startTransition: (
    direction: { x: number; y: number; z: number },
    depthMapSrc: string,
    onSwap: () => void,
  ) => void;
}

export type Parallax360ViewerProps = {
  config: ParallaxViewerConfig;
  debugProjection?: boolean;
};

/**
 * One panorama in the tour. Registered in `app/scenes.ts`.
 */
export type SceneConfig = {
  id: string;
  imageSrc: string;
  depthSrc?: string;
  label: string;
  /**
   * Optional layered-depth assets. When present the viewer renders the
   * scene as background + N foreground spheres for real geometric parallax.
   * Generate with `python scripts/build_scene.py <panorama> --quality high`
   * (or use the interactive `python scripts/new_scene.py` wizard).
   */
  layered?: LayeredConfig;
};

export interface TransitionState {
  active: boolean;
  startTime: number;
  targetDir: THREE.Vector3;
  horizontalDir: THREE.Vector3;
  startYaw: number;
  startPitch: number;
  targetYaw: number;
  deltaYaw: number;
  onSwap: () => void;
  swapFired: boolean;
  baseFov: number;
}
