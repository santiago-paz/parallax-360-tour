import * as THREE from "three";

export interface DebugVisuals {
  group: THREE.Group;
  forwardLine: THREE.Line;
  rightLine: THREE.Line;
  upLine: THREE.Line;
  wireMat: THREE.MeshBasicMaterial;
  enabled: boolean;
  dispose: () => void;
}

export function createDebugVisuals(
  scene: THREE.Scene,
  geo: THREE.SphereGeometry,
  initialEnabled: boolean,
): DebugVisuals {
  const group = new THREE.Group();
  scene.add(group);

  const wireMat = new THREE.MeshBasicMaterial({
    color: 0x22ffe6, wireframe: true, transparent: true, opacity: 0.34,
    side: THREE.DoubleSide, depthTest: false,
    blending: THREE.AdditiveBlending, toneMapped: false,
  });
  group.add(new THREE.Mesh(geo, wireMat));

  const makeDebugLine = (color: number) => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color, transparent: true, opacity: 1.0, depthTest: false,
      blending: THREE.AdditiveBlending, toneMapped: false,
    }));
    group.add(line);
    return line;
  };

  const forwardLine = makeDebugLine(0x00fff7);
  const rightLine = makeDebugLine(0xff2dff);
  const upLine = makeDebugLine(0xffa733);

  group.visible = initialEnabled;

  const dispose = () => {
    forwardLine.geometry.dispose();
    (forwardLine.material as THREE.Material).dispose();
    rightLine.geometry.dispose();
    (rightLine.material as THREE.Material).dispose();
    upLine.geometry.dispose();
    (upLine.material as THREE.Material).dispose();
    wireMat.dispose();
  };

  return { group, forwardLine, rightLine, upLine, wireMat, enabled: initialEnabled, dispose };
}

export function updateDebugLines(
  debug: DebugVisuals,
  cameraPos: THREE.Vector3,
  lookX: number, lookY: number, lookZ: number,
  rightX: number, rightZ: number,
): void {
  const setLine = (line: THREE.Line, a: THREE.Vector3, b: THREE.Vector3) => {
    const pos = line.geometry.attributes.position as THREE.BufferAttribute;
    pos.setXYZ(0, a.x, a.y, a.z);
    pos.setXYZ(1, b.x, b.y, b.z);
    pos.needsUpdate = true;
  };
  setLine(debug.forwardLine, cameraPos,
    new THREE.Vector3(cameraPos.x + lookX * 2, cameraPos.y + lookY * 2, cameraPos.z + lookZ * 2));
  setLine(debug.rightLine, cameraPos,
    new THREE.Vector3(cameraPos.x + rightX * 1.2, cameraPos.y, cameraPos.z + rightZ * 1.2));
  setLine(debug.upLine, cameraPos,
    new THREE.Vector3(cameraPos.x, cameraPos.y + 1, cameraPos.z));
}
