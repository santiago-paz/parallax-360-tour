import type * as THREE from "three";

function bilinearSample(
  pixels: Uint8ClampedArray,
  w: number,
  h: number,
  u: number,
  v: number
): number {
  const fx = u * (w - 1);
  const fy = (1 - v) * (h - 1);
  const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, w - 1);
  const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, h - 1);
  const wx = fx - x0, wy = fy - y0;
  const s = (xi: number, yi: number) => pixels[(yi * w + xi) * 4] / 255;
  return s(x0, y0) * (1 - wx) * (1 - wy)
       + s(x1, y0) *      wx  * (1 - wy)
       + s(x0, y1) * (1 - wx) *      wy
       + s(x1, y1) *      wx  *      wy;
}

export type DisplacementOptions = {
  /**
   * Canonical depth maps emitted by parallax_360.py follow the disparity /
   * proximity convention: bright = near, dark = far. With invert=false (the
   * default) high pixel values directly map to inward shift, so near surfaces
   * end up at a smaller sphere radius (closer to the camera). Set invert=true
   * only when reading a metric-depth map where bright = far.
   */
  invert?: boolean;
  /**
   * Clip the very-far end of the [0, 1] range to zero. Useful to leave the
   * background flat at full sphere radius. Default: 0 (no clipping).
   */
  clipValue?: number;
};

/**
 * Displaces sphere geometry vertices inward based on a depth map canvas.
 *
 * Kept for backward compatibility with the parallax-enabled init path; new
 * code should prefer buildDisplacementOffsets + setDisplacementStrength
 * so the deformation can be animated per frame.
 */
export function applyDisplacement(
  geo: THREE.SphereGeometry,
  radius: number,
  origPosRef: { current: Float32Array | null },
  depthCvs: HTMLCanvasElement,
  maxShiftFactor: number,
  clipValue = 0,
): void {
  const offsets = buildDisplacementOffsets(
    geo, radius, origPosRef, depthCvs, maxShiftFactor,
    { invert: false, clipValue },
  );
  setDisplacementStrength(geo, origPosRef, offsets, 1);
}

/**
 * Pre-computes per-vertex displacement offsets from a depth map.
 * The returned Float32Array has the same length as geo.attributes.position
 * and stores the vector that, added to the original position, yields the
 * fully-displaced vertex (strength = 1).
 *
 * Sphere vertices labelled "near" by the depth map are pulled radially toward
 * the camera centre by up to `radius * maxShiftFactor`. Far vertices stay put.
 */
export function buildDisplacementOffsets(
  geo: THREE.SphereGeometry,
  radius: number,
  origPosRef: { current: Float32Array | null },
  depthCvs: HTMLCanvasElement,
  maxShiftFactor: number,
  options: DisplacementOptions = {},
): Float32Array {
  const invert = options.invert ?? false;
  const clipValue = options.clipValue ?? 0;
  const { width, height } = depthCvs;
  const ctx = depthCvs.getContext("2d")!;
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const posArr = geo.attributes.position.array as Float32Array;
  const uvArr = geo.attributes.uv.array as Float32Array;

  if (!origPosRef.current) origPosRef.current = new Float32Array(posArr);
  const origPos = origPosRef.current;
  const n = posArr.length / 3;
  const offsets = new Float32Array(posArr.length);
  const maxShift = radius * maxShiftFactor;

  for (let i = 0; i < n; i++) {
    const u = uvArr[i * 2];
    const v = uvArr[i * 2 + 1];
    let raw = bilinearSample(pixels, width, height, u, v);
    if (invert) raw = 1 - raw;
    const d = clipValue < 1 ? Math.max(0, raw - clipValue) / (1 - clipValue) : raw;
    const shift = d * maxShift;
    const ox = origPos[i * 3], oy = origPos[i * 3 + 1], oz = origPos[i * 3 + 2];
    const len = Math.sqrt(ox * ox + oy * oy + oz * oz);
    const k = -shift / len;
    offsets[i * 3]     = ox * k;
    offsets[i * 3 + 1] = oy * k;
    offsets[i * 3 + 2] = oz * k;
  }
  return offsets;
}

/**
 * Writes (orig + offsets * strength) into the geometry's position buffer.
 * Cheap enough to run every frame (~130k vertices on the default sphere).
 */
export function setDisplacementStrength(
  geo: THREE.SphereGeometry,
  origPosRef: { current: Float32Array | null },
  offsets: Float32Array,
  strength: number,
): void {
  if (!origPosRef.current) return;
  const orig = origPosRef.current;
  const posArr = geo.attributes.position.array as Float32Array;
  if (strength === 0) {
    posArr.set(orig);
  } else if (strength === 1) {
    for (let i = 0; i < posArr.length; i++) posArr[i] = orig[i] + offsets[i];
  } else {
    for (let i = 0; i < posArr.length; i++) posArr[i] = orig[i] + offsets[i] * strength;
  }
  geo.attributes.position.needsUpdate = true;
}

export function restoreGeometry(
  geo: THREE.SphereGeometry,
  origPosRef: { current: Float32Array | null },
): void {
  if (!origPosRef.current) return;
  const posArr = geo.attributes.position.array as Float32Array;
  posArr.set(origPosRef.current);
  geo.attributes.position.needsUpdate = true;
}

/**
 * Loads a depth map image and applies displacement to the geometry.
 */
export function loadAndDisplace(
  src: string,
  geo: THREE.SphereGeometry,
  radius: number,
  origPosRef: { current: Float32Array | null },
  maxShiftFactor: number,
  clipValue = 0,
): void {
  const img = new Image();
  img.onload = () => {
    const cvs = document.createElement("canvas");
    cvs.width = img.naturalWidth;
    cvs.height = img.naturalHeight;
    cvs.getContext("2d")!.drawImage(img, 0, 0);
    applyDisplacement(geo, radius, origPosRef, cvs, maxShiftFactor, clipValue);
  };
  img.src = src;
}
