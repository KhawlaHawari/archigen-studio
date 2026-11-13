import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createRoofGeometries } from '../src/geometry/roofGeometry.js';
import { createFacadeGeometry, wallFacadeFrame, facadeBounds, facadeOpeningPosition } from '../src/geometry/facadeGeometry.js';

const footprints = {
  rectangle: [[0, 0], [8, 0], [8, 4], [0, 4]],
  L: [[0, 0], [8, 0], [8, 2], [2, 2], [2, 8], [0, 8]],
  T: [[0, 0], [8, 0], [8, 2], [5, 2], [5, 8], [3, 8], [3, 2], [0, 2]],
  U: [[0, 0], [8, 0], [8, 8], [6, 8], [6, 2], [2, 2], [2, 8], [0, 8]],
  narrowNotch: [[0, 0], [8, 0], [8, 5], [4.1, 5], [4.1, 2], [4, 2], [4, 5], [0, 5]],
};

function inside([x, z], ring) {
  let result = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [ax, az] = ring[i], [bx, bz] = ring[j];
    if ((az > z) !== (bz > z) && x < (bx - ax) * (z - az) / (bz - az) + ax) result = !result;
  }
  return result;
}

function assertClosed(geometry) {
  const positions = geometry.attributes.position;
  const indices = geometry.index?.array || Array.from({ length: positions.count }, (_, i) => i);
  const edges = new Map();
  const point = i => [positions.getX(i), positions.getY(i), positions.getZ(i)];
  const key = i => point(i).map(v => v.toFixed(4)).join(',');
  for (let i = 0; i < indices.length; i += 3) {
    const tri = Array.from(indices.slice(i, i + 3));
    const [a, b, c] = tri.map(j => new THREE.Vector3(...point(j)));
    assert.ok(b.sub(a).cross(c.sub(a)).length() > 1e-9, 'nonzero triangle area');
    for (let j = 0; j < 3; j++) {
      const a = key(tri[j]), b = key(tri[(j + 1) % 3]);
      const k = [a, b].sort().join('|');
      const value = edges.get(k) || { count: 0, orientation: 0 };
      value.count++;
      value.orientation += a < b ? 1 : -1;
      edges.set(k, value);
    }
  }
  for (const [edge, { count, orientation }] of edges) {
    assert.equal(count, 2, `two faces incident to ${edge}`);
    assert.equal(orientation, 0, `consistent winding at ${edge}`);
  }
}

for (const [name, footprint] of Object.entries(footprints)) {
  for (const style of ['slab', 'pitched']) {
    test(`${name}: ${style} roof has closed caps and no triangles across concave notches`, () => {
      const roofs = createRoofGeometries({ envelope: [footprint] }, { style });
      assert.equal(roofs.length, 1);
      assert.equal(roofs[0].fallback, null);
      const { geometry, footprint: eaves } = roofs[0];
      assertClosed(geometry);
      const p = geometry.attributes.position;
      for (let i = 0; i < p.count; i += 3) {
        const cx = (p.getX(i) + p.getX(i + 1) + p.getX(i + 2)) / 3;
        const cz = (p.getZ(i) + p.getZ(i + 1) + p.getZ(i + 2)) / 3;
        // Fascia lies on the perimeter. Check projected faces with nonzero area.
        const area = Math.abs((p.getX(i + 1) - p.getX(i)) * (p.getZ(i + 2) - p.getZ(i)) - (p.getX(i + 2) - p.getX(i)) * (p.getZ(i + 1) - p.getZ(i)));
        if (area > 1e-7 && style === 'pitched') assert.ok(inside([cx, cz], eaves.outer), 'roof face stays within eaves');
      }
      if (style === 'pitched') {
        const ridge = Array.from({ length: p.count }, (_, i) => i).filter(i => p.getY(i) > 2.941);
        assert.ok(ridge.length > 0);
        ridge.forEach(i => assert.ok(inside([p.getX(i), p.getZ(i)], footprint), 'ridge is inside building'));
      }
    });
  }
}

test('detached buildings keep separate roofs and disjoint overhangs', () => {
  const a = footprints.rectangle;
  const b = a.map(([x, z]) => [x + 8.3, z]);
  const roofs = createRoofGeometries({ envelope: [a, b] }, { style: 'pitched' });
  assert.equal(roofs.length, 2);
  roofs.forEach(r => { assert.equal(r.fallback, null); assertClosed(r.geometry); });
  const boxes = roofs.map(r => new THREE.Box3().setFromBufferAttribute(r.geometry.attributes.position)).sort((a, b) => a.min.x - b.min.x);
  assert.ok(boxes[0].max.x < boxes[1].min.x);
});

test('overlapping room fallback produces a unified roof without a hull across detached rooms', () => {
  const rooms = [footprints.rectangle, footprints.rectangle.map(([x, z]) => [x, z + 3.9]), footprints.rectangle.map(([x, z]) => [x + 20, z])];
  const roofs = createRoofGeometries({ rooms: rooms.map(polygon_meters => ({ polygon_meters })) }, { style: 'pitched' });
  assert.equal(roofs.length, 2);
  roofs.forEach(r => { assert.equal(r.fallback, null); assertClosed(r.geometry); });
});

test('rotated facade uses physical length, letterbox crop, and exterior-facing U direction', () => {
  for (const yaw of [0, 0.7, Math.PI / 2]) {
    const pts = [[0, 0], [8, 0], [8, 0.22], [0, 0.22]].map(([x, z]) => [x * Math.cos(yaw) - z * Math.sin(yaw), x * Math.sin(yaw) + z * Math.cos(yaw)]);
    for (const polygon_meters of [pts, [...pts].reverse()]) {
      const frame = wallFacadeFrame({ polygon_meters });
      assert.ok(Math.abs(frame.length_m - 8) < 1e-8);
      const bounds = facadeBounds(frame.length_m, 5.6);
      const g = createFacadeGeometry(frame, bounds, 5.6, 4, 3);
      const p = g.attributes.position, uv = g.attributes.uv;
      for (let i = 0; i < p.count; i++) {
        const s = (p.getX(i) + 4 - frame.origin[0]) * frame.direction[0] + (p.getZ(i) + 3 - frame.origin[1]) * frame.direction[1];
        assert.ok(Math.abs(uv.getX(i) - (bounds.u_min + s / 8 * (bounds.u_max - bounds.u_min))) < 1e-6);
        assert.ok(Math.abs(uv.getY(i) - (bounds.v_min + p.getY(i) / 5.6 * (bounds.v_max - bounds.v_min))) < 1e-6);
      }
      const normal = new THREE.Vector3().fromBufferAttribute(g.attributes.normal, 0);
      assert.ok(normal.x * frame.outward[0] + normal.z * frame.outward[1] > 0.999);
      const opening = facadeOpeningPosition({ position_along_wall_m: 2 }, frame);
      assert.ok(Math.abs(Math.hypot(opening[0] - frame.origin[0], opening[1] - frame.origin[1]) - 2) < 1e-8);
    }
  }
});
