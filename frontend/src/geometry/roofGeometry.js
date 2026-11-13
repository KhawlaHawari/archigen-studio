import * as THREE from 'three';
import ClipperLib from 'clipper-lib';
import { SkeletonBuilder } from 'straight-skeleton';

// Integer clipping gives offsets well-defined topology at reflex corners. Each
// disconnected footprint is offset independently, never joined by a convex hull.
const PRECISION = 100000;
const toPath = ring => ring.map(([X, Y]) => ({ X: Math.round(X * PRECISION), Y: Math.round(Y * PRECISION) }));
const fromPath = path => path.map(({ X, Y }) => [X / PRECISION, Y / PRECISION]);

export function cleanRing(points = []) {
  const ring = [];
  for (const p of points) {
    if (!p.every(Number.isFinite)) continue;
    if (!ring.length || Math.hypot(p[0] - ring.at(-1)[0], p[1] - ring.at(-1)[1]) > 1e-5) ring.push([...p]);
  }
  if (ring.length > 1 && Math.hypot(ring[0][0] - ring.at(-1)[0], ring[0][1] - ring.at(-1)[1]) < 1e-5) ring.pop();
  return ring;
}

function treePolygons(tree) {
  const result = [];
  const visit = node => {
    for (const child of node.Childs()) {
      if (!child.IsHole()) {
        result.push({ outer: fromPath(child.Contour()), holes: child.Childs().filter(c => c.IsHole()).map(c => fromPath(c.Contour())) });
      }
      visit(child);
    }
  };
  visit(tree);
  return result.filter(p => p.outer.length >= 3);
}

export function roofFootprints(layout) {
  const rings = layout.envelope?.length ? layout.envelope : [
    ...(layout.rooms || []).map(r => r.polygon_meters),
    ...(layout.walls || []).map(w => w.polygon_meters),
  ];
  const path = (ring, outside) => {
    const p = toPath(cleanRing(ring));
    if (ClipperLib.Clipper.Orientation(p) !== outside) p.reverse();
    return p;
  };
  const paths = rings.flatMap(r => Array.isArray(r) ? [path(r, true)] : [
    path(r.outer, true), ...(r.holes || []).map(hole => path(hole, false)),
  ]).filter(p => p.length >= 3);
  const clipper = new ClipperLib.Clipper();
  clipper.AddPaths(paths, ClipperLib.PolyType.ptSubject, true);
  const tree = new ClipperLib.PolyTree();
  clipper.Execute(ClipperLib.ClipType.ctUnion, tree, ClipperLib.PolyFillType.pftNonZero, ClipperLib.PolyFillType.pftNonZero);
  return treePolygons(tree);
}

function offsetFootprint(polygon, amount) {
  const offset = new ClipperLib.ClipperOffset(2, 0.001 * PRECISION);
  const paths = [polygon.outer, ...polygon.holes].map(toPath);
  paths.forEach((p, i) => { if (ClipperLib.Clipper.Orientation(p) !== (i === 0)) p.reverse(); });
  offset.AddPaths(paths, ClipperLib.JoinType.jtMiter, ClipperLib.EndType.etClosedPolygon);
  const tree = new ClipperLib.PolyTree();
  offset.Execute(tree, amount * PRECISION);
  return treePolygons(tree);
}

function pointSegmentDistance(p, a, b) {
  const dx = b[0] - a[0], dz = b[1] - a[1];
  const t = THREE.MathUtils.clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / (dx * dx + dz * dz || 1), 0, 1);
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dz);
}

function ringDistance(a, b) {
  const distance = (r, s) => Math.min(...r.map(p => Math.min(...s.map((q, i) => pointSegmentDistance(p, q, s[(i + 1) % s.length])))));
  return Math.min(distance(a, b), distance(b, a));
}

function shapeFromPolygon({ outer, holes }, centerX, centerZ) {
  const path = ring => new THREE.Path(ring.map(([x, z]) => new THREE.Vector2(x - centerX, -(z - centerZ))));
  const shape = new THREE.Shape(path(outer).getPoints());
  shape.holes = holes.map(path);
  return shape;
}

// Keep Three's bevel smaller than any nonadjacent boundary separation. The
// offset itself is clipped first, so a narrow notch can close without crossing.
function safeBevel({ outer, holes }) {
  let clearance = 0.12;
  for (const ring of [outer, ...holes]) {
    ring.forEach((p, i) => {
      clearance = Math.min(clearance, Math.hypot(p[0] - ring[(i + 1) % ring.length][0], p[1] - ring[(i + 1) % ring.length][1]));
      ring.forEach((a, j) => {
        if (j !== i && (j + 1) % ring.length !== i) clearance = Math.min(clearance, pointSegmentDistance(p, a, ring[(j + 1) % ring.length]));
      });
    });
  }
  return Math.min(0.025, clearance / 6);
}

function slabGeometry(polygon, centerX, centerZ, wallHeight) {
  const bevel = safeBevel(polygon);
  const geometry = new THREE.ExtrudeGeometry(shapeFromPolygon(polygon, centerX, centerZ), {
    depth: 0.18, steps: 1, bevelEnabled: bevel > 1e-4,
    bevelSize: bevel, bevelThickness: 0.02, bevelSegments: 2,
  });
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, wallHeight + 0.02, 0);
  // Both caps and the fascia belong to the same closed extrusion; the lower
  // cap is the ceiling soffit rather than an overlapping, unsealed plane.
  setMaterialGroups(geometry, Array.from({ length: geometry.attributes.position.count / 3 }, (_, i) =>
    geometry.attributes.normal.getY(i * 3) < -0.99 ? 1 : 0));
  return geometry;
}

function setMaterialGroups(geometry, materials) {
  geometry.clearGroups();
  let start = 0;
  for (let i = 1; i <= materials.length; i++) {
    if (materials[i] !== materials[start]) {
      geometry.addGroup(start * 3, (i - start) * 3, materials[start]);
      start = i;
    }
  }
}

function isClosedSolid(geometry) {
  const p = geometry.attributes.position;
  if (!p.array.every(Number.isFinite)) return false;
  const edges = new Map();
  const key = i => [p.getX(i), p.getY(i), p.getZ(i)].map(v => Math.round(v * 10000)).join(',');
  for (let i = 0; i < p.count; i += 3) {
    const vertices = [i, i + 1, i + 2].map(key);
    if (new Set(vertices).size !== 3) return false;
    for (let j = 0; j < 3; j++) {
      const a = vertices[j], b = vertices[(j + 1) % 3];
      const edge = [a, b].sort().join('|');
      const value = edges.get(edge) || [0, 0];
      value[0]++;
      value[1] += a < b ? 1 : -1;
      edges.set(edge, value);
    }
  }
  return [...edges.values()].every(([count, winding]) => count === 2 && winding === 0);
}

function pitchedGeometry(polygon, centerX, centerZ, wallHeight) {
  const skeleton = SkeletonBuilder.BuildFromGeoJSON([[polygon.outer, ...polygon.holes]]);
  if (!skeleton?.Edges?.length) throw new Error('No straight-skeleton roof faces');
  const positions = [], uv = [], materials = [];
  const base = wallHeight + 0.14;
  const distances = new Map([...skeleton.Distances].map(([p, d]) => [`${p.X.toFixed(7)},${p.Y.toFixed(7)}`, d]));
  const height = p => base + (distances.get(`${p.X.toFixed(7)},${p.Y.toFixed(7)}`) ?? 0) * 0.65;
  const triangle = (a, b, c, material, up = true) => {
    const normalY = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    if ((normalY > 0) !== up) [b, c] = [c, b];
    for (const v of [a, b, c]) { positions.push(...v); uv.push(v[0] / 2, v[2] / 2); }
    materials.push(material);
  };
  let projectedArea = 0;
  for (const face of skeleton.Edges) {
    const pts = Array.from(face.Polygon);
    const contour = pts.map(p => new THREE.Vector2(p.X, p.Y));
    for (const indices of THREE.ShapeUtils.triangulateShape(contour, [])) {
      const v = indices.map(i => [pts[i].X - centerX, height(pts[i]), pts[i].Y - centerZ]);
      projectedArea += Math.abs((v[1][0] - v[0][0]) * (v[2][2] - v[0][2]) - (v[2][0] - v[0][0]) * (v[1][2] - v[0][2])) / 2;
      triangle(...v, 0);
    }
  }
  const area = Math.abs(THREE.ShapeUtils.area(polygon.outer.map(p => new THREE.Vector2(...p))))
    - polygon.holes.reduce((sum, ring) => sum + Math.abs(THREE.ShapeUtils.area(ring.map(p => new THREE.Vector2(...p)))), 0);
  if (!Number.isFinite(projectedArea) || Math.abs(projectedArea - area) > Math.max(0.001, area * 1e-5)) throw new Error('Incomplete straight-skeleton roof');

  const rings = [polygon.outer, ...polygon.holes];
  const points = rings.flat();
  const bottom = rings.map(r => r.map(([x, z]) => new THREE.Vector2(x, z)));
  for (const indices of THREE.ShapeUtils.triangulateShape(bottom[0], bottom.slice(1))) {
    triangle(...indices.map(i => [points[i][0] - centerX, wallHeight, points[i][1] - centerZ]), 1, false);
  }
  rings.forEach((ring, ringIndex) => {
    // Outer CCW, holes CW in plan space: fascia normals point out of solid.
    const ccw = THREE.ShapeUtils.area(ring.map(p => new THREE.Vector2(...p))) > 0;
    const ordered = ccw === (ringIndex === 0) ? ring : [...ring].reverse();
    ordered.forEach(([x, z], i) => {
      const [nx, nz] = ordered[(i + 1) % ordered.length];
      const a = [x - centerX, wallHeight, z - centerZ], b = [nx - centerX, wallHeight, nz - centerZ];
      const c = [nx - centerX, base, nz - centerZ], d = [x - centerX, base, z - centerZ];
      // Vertical faces have zero projected area; preserve explicit winding.
      for (const v of [a, c, b, a, d, c]) { positions.push(...v); uv.push(v[0] / 2, v[1]); }
      materials.push(0, 0);
    });
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  setMaterialGroups(geometry, materials);
  geometry.computeVertexNormals();
  if (!isClosedSolid(geometry)) {
    geometry.dispose();
    throw new Error('Non-manifold straight-skeleton roof');
  }
  return geometry;
}

export function createRoofGeometries(layout, { centerX = 0, centerZ = 0, wallHeight = 2.8, style = 'slab', overhang = 0.22 } = {}) {
  const footprints = roofFootprints(layout);
  return footprints.flatMap((footprint, i) => {
    const gap = Math.min(Infinity, ...footprints.filter((_, j) => j !== i).map(other => ringDistance(footprint.outer, other.outer)));
    // Reserve a gap between detached eaves even when buildings are very close.
    const eaves = offsetFootprint(footprint, Math.min(overhang, gap / 3));
    return eaves.map(polygon => {
      let geometry, fallback = null;
      try {
        geometry = style === 'pitched' ? pitchedGeometry(polygon, centerX, centerZ, wallHeight) : slabGeometry(polygon, centerX, centerZ, wallHeight);
      } catch (error) {
        // Degenerate predictions still receive a closed roof, never a broken fan.
        geometry = slabGeometry(polygon, centerX, centerZ, wallHeight);
        fallback = error.message;
      }
      return { geometry, footprint: polygon, fallback };
    });
  });
}
