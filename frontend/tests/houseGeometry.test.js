import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { alignDrawingLevels, buildingLevels, clipRoomFloor, compileWallAssembly, fitHouseOpenings, floorFootprints, floorSurfaceGeometry,
  wallAssemblyGeometry, wallStrip } from '../src/geometry/houseGeometry.js';
import { createSampleHouse } from '../src/geometry/sampleHousePlan.js';
import { createFacadeGeometry, facadeBounds } from '../src/geometry/facadeGeometry.js';

const rectangle = (x, z, w, d) => [[x, z], [x + w, z], [x + w, z + d], [x, z + d]];
const wall = (id, start, end) => ({ id, axis_meters: [start, end], thickness_m: .2,
  polygon_meters: wallStrip(start, end, .2) });
const opening = (id, position_meters, width_m, sill_m = 0) => ({ id, position_meters, width_m,
  sill_m, height_m: sill_m ? 1.1 : 2.1 });
const hit = (geometry, point, direction) => {
  const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  const result = new THREE.Raycaster(new THREE.Vector3(...point), new THREE.Vector3(...direction)).intersectObject(mesh).length;
  material.dispose();
  return result;
};

test('a single slab covers the gap between detected rooms and preserves a void', () => {
  const layout = { envelope: [rectangle(0, 0, 8, 6), rectangle(12, 0, 4, 4)],
    floorVoids: [rectangle(5, 3, 1, 2)] };
  const footprints = floorFootprints(layout);
  assert.equal(footprints.length, 2);
  const main = footprints.find(polygon => polygon.holes.length === 1);
  const geometry = floorSurfaceGeometry(main, 0, 0, .16);
  assert.ok(hit(geometry, [4, 5, 2], [0, -1, 0]), 'floor remains continuous at the room seam');
  assert.equal(hit(geometry, [5.5, 5, 4], [0, -1, 0]), 0);
  assert.equal(hit(geometry, [10, 5, 4], [0, -1, 0]), 0);
  assert.equal(clipRoomFloor({ polygon_meters: rectangle(20, 20, 1, 1) }, footprints).length, 0);
  assert.equal(clipRoomFloor({ polygon_meters: rectangle(4, 2, 3, 4) }, footprints)[0].holes.length, 1);
  geometry.dispose();
});

test('walls join at a T and wall geometry leaves usable door and window openings', () => {
  const walls = [wall('long', [0, 0], [8, 0]), wall('partition', [4, .14], [4, 3])];
  const assembly = compileWallAssembly(walls, [opening('door', [2, .03], .9),
    opening('window', [6, 0], 1, .9), opening('floating', [9.1, 0], .9)], 2.8);
  assert.deepEqual(assembly.unresolved, ['floating']);
  assert.equal(assembly.bands.at(-1).polygons.length, 1, 'join the small gap at the T');
  const geometry = wallAssemblyGeometry(assembly.bands, 0, 0);
  assert.equal(hit(geometry, [2, 1, -1], [0, 0, 1]), 0);
  assert.ok(hit(geometry, [2, 2.5, -1], [0, 0, 1]), 'door lintel');
  assert.equal(hit(geometry, [6, 1.5, -1], [0, 0, 1]), 0);
  assert.ok(hit(geometry, [6, .4, -1], [0, 0, 1]), 'window sill');
  geometry.dispose();
});

test('detected opening bridges two aligned wall ends and rejects a single unsupported end', () => {
  const spans = [wall('a', [0, 0], [2, 0]), wall('b', [3, 0], [6, 0])];
  const candidate = opening('gap', [2.52, 0], 1);
  const assembly = compileWallAssembly(spans, [candidate], 2.8);
  assert.equal(assembly.hosted.length, 1);
  assert.equal(assembly.hosted[0].host.center[0], 2.5);
  assert.equal(assembly.bands.at(-1).polygons.length, 1, 'lintel joins the wall on both sides');
  assert.deepEqual(compileWallAssembly(spans.slice(0, 1), [candidate], 2.8).unresolved, ['gap']);
});

test('facade cutout keeps planar UVs and lets rays pass through a window', () => {
  const frame = { origin: [0, 0], direction: [1, 0], outward: [0, 1], length_m: 8 };
  const bounds = facadeBounds(8, 2.8);
  const geometry = createFacadeGeometry(frame, bounds, 2.8, 0, 0,
    [{ position_along_wall_m: 3, width_m: 1.2, sill_m: .9, height_m: 1.2 }]);
  assert.equal(hit(geometry, [3, 1.4, 2], [0, 0, -1]), 0);
  assert.ok(hit(geometry, [3, .5, 2], [0, 0, -1]));
  const { position, uv } = geometry.attributes;
  for (let i = 0; i < position.count; i++) {
    assert.ok(Math.abs(uv.getX(i) - (bounds.u_min + position.getX(i) / 8 * (bounds.u_max - bounds.u_min))) < 1e-6);
  }
  geometry.dispose();
});

test('sample storeys align in the scene, keep sheet coordinates, and host all traced openings', () => {
  const sample = createSampleHouse();
  const [ground, upper] = buildingLevels(sample, 3);
  assert.equal(upper.elevation, ground.height);
  assert.equal(upper.height, 2.5);
  assert.ok(upper.offset[0] < -7, 'upper drawing moves from sheet position onto the house');
  assert.ok(upper.walls[0].axis_meters[0][0] > ground.walls[0].axis_meters[0][0]);
  for (const level of [ground, upper]) {
    const assembly = compileWallAssembly(level.walls, fitHouseOpenings(level), level.height);
    assert.deepEqual(assembly.unresolved, []);
  }
  assert.equal(buildingLevels({ walls: [{ height_m: 2.8 }] }, 2)[0].floorElevations[1], 2.8);
});

test('reconstructed openings use livable proportions and keep the adjoining wall joint', () => {
  const [ground] = buildingLevels(createSampleHouse());
  const fitted = fitHouseOpenings(ground);
  const source = ground.doors.find(door => door.id === 'ground_east_door_0');
  const door = fitted.find(item => item.id === source.id);
  assert.ok(source.width_m < .5 && door.width_m >= .84);
  assert.equal(door.height_m, 1.9);
  const adjoiningWall = ground.walls.find(item => item.id === 'ground_bath_north');
  const doorEnd = door.position_meters[1] + door.width_m / 2;
  assert.ok(doorEnd < adjoiningWall.axis_meters[1][1] - adjoiningWall.thickness_m / 2);
  const windows = fitted.filter(item => item.type === 'window' && item.associated_wall === 'ground_east')
    .sort((a, b) => a.position_meters[1] - b.position_meters[1]);
  assert.ok(windows.every(window => window.height_m / window.width_m <= 1.09));
  for (let i = 1; i < windows.length; i++) {
    const a = windows[i - 1], b = windows[i];
    assert.ok(b.position_meters[1] - a.position_meters[1] >= (a.width_m + b.width_m) / 2);
  }
});

test('two detected footprints stay separate until the user aligns corresponding anchors', () => {
  const layout = { envelope: [rectangle(0, 0, 6, 4), rectangle(9, 1, 5, 4)],
    walls: [wall('lower', [0, 0], [6, 0]), wall('upper', [9, 1], [14, 1])],
    rooms: [{ id: 'upper-room', polygon_meters: rectangle(10, 2, 2, 2) }],
    doors: [], windows: [opening('upper-window', [12, 1], .9, .9)] };
  assert.equal(buildingLevels(layout).length, 1);
  assert.throws(() => alignDrawingLevels(layout, [0, 0], [2, 0]), /different plan outlines/);
  const aligned = buildingLevels(alignDrawingLevels(layout, [0, 0], [9, 1]));
  assert.deepEqual(aligned[1].offset, [-9, -1]);
  assert.equal(aligned[1].walls[0].id, 'upper');
  assert.equal(aligned[1].rooms[0].id, 'upper-room');
  assert.equal(aligned[1].windows[0].id, 'upper-window');
});
