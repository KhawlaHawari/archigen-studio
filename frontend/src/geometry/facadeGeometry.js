import * as THREE from 'three';

export function wallFacadeFrame(wall) {
  if (wall.facade_frame) return wall.facade_frame;
  const pts = wall.polygon_meters || [];
  if (pts.length < 2) return null;
  let edge = null, length = 0, area = 0;
  pts.forEach((a, i) => {
    const b = pts[(i + 1) % pts.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len > length) { edge = [a, b]; length = len; }
    area += a[0] * b[1] - b[0] * a[1];
  });
  if (length < 0.01) return null;
  const [a, b] = edge;
  const dx = (b[0] - a[0]) / length, dz = (b[1] - a[1]) / length;
  const outward = area >= 0 ? [dz, -dx] : [-dz, dx];
  return { origin: area >= 0 ? b : a, direction: [outward[1], -outward[0]], outward, length_m: length };
}

export function facadeOpeningPosition(opening, frame) {
  const s = opening.position_along_wall_m;
  return [frame.origin[0] + frame.direction[0] * s, frame.origin[1] + frame.direction[1] * s];
}

export function createFacadeGeometry(frame, bounds, height, centerX = 0, centerZ = 0, openings = []) {
  const length = frame.length_m;
  // A dedicated cladding face avoids painting the facade onto wall end caps,
  // interior faces, or applying default ExtrudeGeometry's world-unit UVs.
  const geometry = new THREE.BufferGeometry();
  const positions = [], uvs = [];
  const addVertex = (s, y) => {
    positions.push(frame.origin[0] + frame.direction[0] * s + frame.outward[0] * 0.003 - centerX,
      y, frame.origin[1] + frame.direction[1] * s + frame.outward[1] * 0.003 - centerZ);
    uvs.push(bounds.u_min + (bounds.u_max - bounds.u_min) * s / length,
      bounds.v_min + (bounds.v_max - bounds.v_min) * y / height);
  };
  [[0, 0], [length, 0], [length, height], [0, height]].forEach(([s, y]) => addVertex(s, y));
  const holes = openings.map(op => ({
    left: Math.max(0, op.position_along_wall_m - op.width_m / 2),
    right: Math.min(length, op.position_along_wall_m + op.width_m / 2),
    bottom: Math.max(0, op.sill_m), top: Math.min(height, op.sill_m + op.height_m),
  })).filter(hole => hole.right > hole.left && hole.top > hole.bottom);
  const indices = [];
  if (!holes.length) {
    indices.push(0, 1, 2, 0, 2, 3);
  } else {
    const breaks = (max, values) => [...new Set([0, max, ...values])].sort((a, b) => a - b);
    const xs = breaks(length, holes.flatMap(hole => [hole.left, hole.right]));
    const ys = breaks(height, holes.flatMap(hole => [hole.bottom, hole.top]));
    for (let i = 1; i < xs.length; i++) {
      for (let j = 1; j < ys.length; j++) {
        const [x0, x1] = xs.slice(i - 1, i + 1), [y0, y1] = ys.slice(j - 1, j + 1);
        const x = (x0 + x1) / 2, y = (y0 + y1) / 2;
        if (holes.some(h => x > h.left && x < h.right && y > h.bottom && y < h.top)) continue;
        const first = positions.length / 3;
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].forEach(([s, v]) => addVertex(s, v));
        indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
      }
    }
  }
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

export function facadeBounds(length, height) {
  // Legacy URL-only callers use the bridge's 256px / 12px-margin letterbox.
  const scale = Math.min(232 / length, 232 / height);
  const w = Math.round(length * scale), h = Math.round(height * scale);
  const x = Math.floor((256 - w) / 2), y = Math.floor((256 - h) / 2);
  return { u_min: x / 256, u_max: (x + w) / 256, v_min: (256 - y - h) / 256, v_max: (256 - y) / 256 };
}
