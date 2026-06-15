/**
 * Toggle parallax features on/off. When true the viewer deforms the sphere
 * with the depth map and offsets the camera with the cursor.
 */
export const ENABLE_PARALLAX = true;

export const RADIUS = 10;
export const SEGS_W = 512;
export const SEGS_H = 256;
export const MAX_OFFSET = 0.8;
export const LERP = 0.06;

/** Runtime defaults for parallax tuning when a scene doesn't override them. */
export const DEFAULT_PARALLAX_STRENGTH = 0.5;
export const DEFAULT_DEPTH_SCALE = 0.55;
export const DEFAULT_NEAR_CLIP = 0.08;

export const TRANSITION_DURATION = 1200;
export const TRANSITION_DEPTH_SCALE = 0.45;
export const TRANSITION_FLY_DIST = RADIUS * 0.42;

/**
 * @deprecated since 2026-06-14 — depth-displaced LDI ignores per-layer base
 * radii. Every fg sphere now uses {@link RADIUS} (= bg radius) and is
 * displaced per-pixel by the depth map. The constants remain exported so
 * external imports do not break; they have no runtime effect on the viewer.
 *
 * MIN goes to layer 0 (closest depth, most parallax); MAX goes to the
 * last layer (closest to bg, mildest parallax). Single-layer fell back
 * to MIN (= ~6 — the original spike radius).
 */
export const MIN_FG_RADIUS = 5;
export const MAX_FG_RADIUS = 8;
export const DEFAULT_FOREGROUND_RADIUS = MIN_FG_RADIUS;
