import type { SceneConfig } from "./components/parallax360/types";

/**
 * The panoramas the viewer shows. Each entry needs:
 *  - `imageSrc`: equirectangular panorama (JPEG/PNG/WebP) under `public/`
 *  - `depthSrc`: matching depth map under `public/parallax/`
 *  - `id`: stable string, also used as the key in the scene selector
 *  - `label`: shown in the scene picker chips
 *  - `layered` (optional): Layered Depth Image assets for real parallax
 *
 * Generate new entries with:
 *   python scripts/new_scene.py
 *   python scripts/build_scene.py <pano> --quality high
 */
export const SCENES: SceneConfig[] = [
  // <build_scene:start>
  {
    id: "image-1",
    imageSrc: "/image-1-360.webp",
    depthSrc: "/parallax/depth_image-1-360.png",
    label: "ATELIER",
    layered: {
      backgroundSrc: "/parallax/image-1-360-bg.jpeg",
      foregroundLayers: [
        { src: "/parallax/image-1-360-fg0.webp" },
        { src: "/parallax/image-1-360-fg1.webp" },
        { src: "/parallax/image-1-360-fg2.webp" },
        { src: "/parallax/image-1-360-fg3.webp" },
      ],
    },
  },
  {
    id: "image-2",
    imageSrc: "/image-2-360.webp",
    depthSrc: "/parallax/depth_image-2-360.png",
    label: "LOFT",
    layered: {
      backgroundSrc: "/parallax/image-2-360-bg.jpeg",
      foregroundLayers: [
        { src: "/parallax/image-2-360-fg0.webp" },
        { src: "/parallax/image-2-360-fg1.webp" },
        { src: "/parallax/image-2-360-fg2.webp" },
        { src: "/parallax/image-2-360-fg3.webp" },
      ],
    },
  },
  {
    id: "image-3",
    imageSrc: "/image-3-360.webp",
    depthSrc: "/parallax/depth_image-3-360.png",
    label: "FACTORY",
    layered: {
      backgroundSrc: "/parallax/image-3-360-bg.jpeg",
      foregroundLayers: [
        { src: "/parallax/image-3-360-fg0.webp" },
        { src: "/parallax/image-3-360-fg1.webp" },
        { src: "/parallax/image-3-360-fg2.webp" },
      ],
    },
  },
  // <build_scene:end>
];

export const FOV = 75;
