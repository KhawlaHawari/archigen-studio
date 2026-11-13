import * as THREE from 'three';
import ClipperLib from 'clipper-lib';
import { cleanRing, roofFootprints } from './roofGeometry.js';
import { STOREY_HEIGHT_M, DOOR_HEIGHT_M, WINDOW_HEIGHT_M, WINDOW_SILL_M } from './buildingDimensions.js';

const GRID = 100000;
const { Clipper, ClipperOffset, PolyTree, ClipType, PolyType, PolyFillType, JoinType, EndType } = ClipperLib;
const toPath = (ring, exterior = true) => {
  const path = cleanRing(ring).map(([x, z]) => ({ X: Math.round(x * GRID), Y: Math.round(z * GRID) }));
  if (Clipper.Orientation(path) !== exterior) path.reverse();
  return path;
};
const readRing = path => path.map(({ X, Y }) => [X / GRID, Y / GRID]);

function readPolygons(tree) {
  const polygons = [];
  const visit = node => node.Childs().forEach(child => {
    if (!child.IsHole()) polygons.push({
      outer: readRing(child.Contour()),
      holes: child.Childs().filter(ring => ring.IsHole()).map(ring => readRing(ring.Contour())),
    });
    visit(child);
  });
  visit(tree);
  return polygons;
}

const pathsFor = polygons => polygons.flatMap(({ outer, holes = [] }) => [
  toPath(outer), ...holes.map(hole => toPath(hole, false)),
]).filter(path => path.length >= 3);

export function combinePolygons(subject, clip = [], operation = ClipType.ctUnion) {
  const paths = pathsFor(subject);
  if (!paths.length) return [];
  const clipPaths = pathsFor(clip);
  if (operation === ClipType.ctIntersection && !clipPaths.length) return [];
  const clipper = new Clipper();
  clipper.AddPaths(paths, PolyType.ptSubject, true);
  if (clipPaths.length) clipper.AddPaths(clipPaths, PolyType.ptClip, true);
  const tree = new PolyTree();
  clipper.Execute(operation, tree, PolyFillType.pftNonZero, PolyFillType.pftNonZero);
  return readPolygons(tree);
}

const ringPolygon = outer => ({ outer, holes: [] });

export function floorFootprints(layout) {
  const voids = (layout.floorVoids || []).map(ringPolygon);
  return combinePolygons(roofFootprints(layout), voids, ClipType.ctDifference);
}

export function buildingLevels(layout, repeats = 1) {
  if (!layout.storeys?.length) {
    const unitHeight = layout.walls?.[0]?.height_m || STOREY_HEIGHT_M;
    return [{ ...layout, id: 'main', elevation: 0, offset: [0, 0], unitHeight,
      height: unitHeight * repeats, floorElevations: Array.from({ length: repeats }, (_, i) => i * unitHeight) }];
  }
  return layout.storeys.map(level => {
    const belongs = item => item.level_id === level.id;
    const openingBelongs = opening => belongs(opening) || (!opening.level_id && drawingLevelAt(layout, opening.position_meters)?.id === level.id);
    return { ...level, unitHeight: level.height, floorElevations: [0],
      walls: layout.walls.filter(belongs), rooms: layout.rooms.filter(belongs),
      doors: layout.doors.filter(openingBelongs), windows: layout.windows.filter(openingBelongs) };
  });
}

export function drawingLevelAt(layout, position) {
  if (!position) return null;
  return layout.storeys?.find(level => {
    const bounds = level.envelope.flatMap(ring => Array.isArray(ring) ? ring : ring.outer);
    return position.every((value, axis) => value >= Math.min(...bounds.map(p => p[axis])) - .1
      && value <= Math.max(...bounds.map(p => p[axis])) + .1);
  });
}

export function signedDistanceToRing(point, ring) {
  let inside = false, nearest = Infinity;
  ring.forEach((a, i) => {
    const b = ring[(i + 1) % ring.length];
    if ((a[1] > point[1]) !== (b[1] > point[1])
      && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
    const delta = subtract(b, a), relative = subtract(point, a);
    const t = Math.max(0, Math.min(1, dot(relative, delta) / (dot(delta, delta) || 1)));
    nearest = Math.min(nearest, Math.hypot(...subtract(point, pointAt(a, delta, t))));
  });
  return inside ? nearest : -nearest;
}

// A multi-panel sheet needs a user decision. Nearby drawings can also be
// detached houses, so the compiler only stacks them after two matching anchors.
export function alignDrawingLevels(layout, groundAnchor, upperAnchor) {
  const footprints = roofFootprints(layout);
  if (footprints.length !== 2) throw new Error('Level alignment requires two separate detected footprints.');
  const nearest = (point, polygons) => polygons.reduce((best, polygon, index) => {
    const distance = Math.max(0, -signedDistanceToRing(point, polygon.outer));
    return distance < best.distance ? { index, distance } : best;
  }, { index: -1, distance: Infinity });
  const ground = nearest(groundAnchor, footprints), upper = nearest(upperAnchor, footprints);
  if (ground.distance > .35 || upper.distance > .35 || ground.index === upper.index) {
    throw new Error('Pick matching points on two different plan outlines.');
  }
  const ordered = [footprints[ground.index], footprints[upper.index]];
  const height = layout.walls[0]?.height_m || STOREY_HEIGHT_M;
  const ids = ['ground', 'upper'];
  const tag = (item, point) => ({ ...item, level_id: ids[nearest(point, ordered).index] });
  const tagPolygon = item => {
    const ring = item.polygon_meters || [];
    if (!ring.length) return item;
    const center = ring.reduce(([x, z], p) => [x + p[0] / ring.length, z + p[1] / ring.length], [0, 0]);
    return tag(item, center);
  };
  const tagOpening = item => item.position_meters ? tag(item, item.position_meters) : item;
  return { ...layout, walls: layout.walls.map(tagPolygon), rooms: layout.rooms.map(tagPolygon),
    doors: layout.doors.map(tagOpening), windows: layout.windows.map(tagOpening),
    storeys: ordered.map((polygon, i) => ({ id: ids[i], label: i ? 'Upper floor' : 'Ground floor',
      elevation: i * height, height, offset: i ? subtract(groundAnchor, upperAnchor) : [0, 0],
      envelope: [polygon], floorVoids: [] })) };
}

export function separateDrawingLevels(layout) {
  const untag = ({ level_id, ...item }) => item;
  return { ...layout, storeys: undefined, walls: layout.walls.map(untag), rooms: layout.rooms.map(untag),
    doors: layout.doors.map(untag), windows: layout.windows.map(untag) };
}

export function exposedRoof(level, levels) {
  const above = levels.filter(other => Math.abs(other.elevation - level.elevation - level.height) < .01);
  const covered = above.flatMap(other => roofFootprints(other).map(({ outer, holes }) => {
    const move = ring => ring.map(([x, z]) => [x + other.offset[0] - level.offset[0], z + other.offset[1] - level.offset[1]]);
    return { outer: move(outer), holes: holes.map(move) };
  }));
  return combinePolygons(roofFootprints(level), offsetPolygons(covered, .035), ClipType.ctDifference);
}

export function tileFloor(rooms, footprints) {
  return combinePolygons(rooms.map(room => ringPolygon(room.polygon_meters)), footprints, ClipType.ctIntersection);
}

export function clipRoomFloor(room, footprints) {
  return combinePolygons([ringPolygon(room.polygon_meters || [])], footprints, ClipType.ctIntersection);
}

export function planShape(polygon, centerX, centerZ) {
  const addRing = (path, ring) => {
    ring.forEach(([x, z], i) => path[i ? 'lineTo' : 'moveTo'](x - centerX, centerZ - z));
    path.closePath();
    return path;
  };
  const shape = addRing(new THREE.Shape(), polygon.outer);
  shape.holes = polygon.holes.map(ring => addRing(new THREE.Path(), ring));
  return shape;
}

export function floorSurfaceGeometry(polygon, centerX, centerZ, thickness = 0) {
  const shape = planShape(polygon, centerX, centerZ);
  const geometry = thickness > 0
    ? new THREE.ExtrudeGeometry(shape, { depth: thickness, bevelEnabled: false })
    : new THREE.ShapeGeometry(shape);
  geometry.rotateX(-Math.PI / 2);
  if (thickness) geometry.translate(0, -thickness, 0);
  const positions = geometry.attributes.position;
  const uv = geometry.attributes.uv;
  for (let i = 0; i < positions.count; i++) {
    uv.setXY(i, (positions.getX(i) + centerX) / 1.8, (positions.getZ(i) + centerZ) / 1.8);
  }
  return geometry;
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const subtract = (a, b) => [a[0] - b[0], a[1] - b[1]];
const pointAt = (origin, dir, t) => [origin[0] + dir[0] * t, origin[1] + dir[1] * t];

export function wallStrip(start, end, thickness) {
  const delta = subtract(end, start);
  const length = Math.hypot(...delta);
  const side = [-delta[1] / length * thickness / 2, delta[0] / length * thickness / 2];
  return [pointAt(start, side, 1), pointAt(end, side, 1), pointAt(end, side, -1), pointAt(start, side, -1)];
}

// Fit centerline runs to overlapping, near-parallel faces of a detected wall.
// Preserve original polygon bodies and real diagonal walls.
function wallRuns(wall) {
  const run = (start, end, thickness) => ({ start, end, thickness, wallId: wall.id, exterior: wall.is_exterior });
  if (wall.axis_meters) return [run(...wall.axis_meters, wall.thickness_m)];
  const ring = cleanRing(wall.polygon_meters || []);
  const edges = ring.map((start, i) => {
    const end = ring[(i + 1) % ring.length];
    const delta = subtract(end, start);
    const length = Math.hypot(...delta);
    return { start, end, length, dir: delta.map(v => v / length) };
  }).filter(edge => edge.length >= .15);
  return edges.flatMap(({ start, dir, length }, i) => {
    const normal = [-dir[1], dir[0]];
    return edges.slice(i + 1).flatMap(edge => {
      if (dot(dir, edge.dir) > -.98) return [];
      const a = subtract(edge.start, start), b = subtract(edge.end, start);
      const sepA = dot(a, normal), sepB = dot(b, normal);
      const sep = (sepA + sepB) / 2;
      const thickness = Math.abs(sep);
      if (thickness < .055 || thickness > .42 || Math.abs(sepA - sepB) > .16) return [];
      const from = Math.max(0, Math.min(dot(a, dir), dot(b, dir)));
      const to = Math.min(length, Math.max(dot(a, dir), dot(b, dir)));
      if (to - from < Math.max(thickness * .6, .15)) return [];
      const origin = pointAt(start, normal, sep / 2);
      return [run(pointAt(origin, dir, from), pointAt(origin, dir, to), thickness)];
    });
  });
}

function regularizeWallBody(wall, runs) {
  const ring = cleanRing(wall.polygon_meters || []);
  if (wall.axis_meters || wall.is_exterior || ring.length < 4 || !runs.length) return ring;
  const run = runs.reduce((longest, candidate) =>
    Math.hypot(...subtract(candidate.end, candidate.start)) > Math.hypot(...subtract(longest.end, longest.start)) ? candidate : longest);
  const delta = subtract(run.end, run.start);
  const length = Math.hypot(...delta);
  if (length < Math.max(.45, run.thickness * 3)) return ring;
  const direction = delta.map(value => value / length);
  const normal = [-direction[1], direction[0]];
  const relative = ring.map(point => subtract(point, run.start));
  if (relative.some(point => Math.abs(dot(point, normal)) > run.thickness / 2 + .045)) return ring;
  const along = relative.map(point => dot(point, direction));
  return wallStrip(pointAt(run.start, direction, Math.min(...along)),
    pointAt(run.start, direction, Math.max(...along)), run.thickness);
}

function projectRuns(runs, position, width) {
  return runs.map(run => {
    const delta = subtract(run.end, run.start);
    const length = Math.hypot(...delta);
    const direction = delta.map(v => v / length);
    const normal = [-direction[1], direction[0]];
    const relative = subtract(position, run.start);
    return { ...run, length, direction, normal, distance: Math.abs(dot(relative, normal)), along: dot(relative, direction) };
  }).filter(run => run.distance <= Math.max(.22, run.thickness / 2 + .15)
    && run.along >= -width / 2 - .35 && run.along <= run.length + width / 2 + .35);
}

function hostOpening(opening, runs) {
  const { position_meters: position, width_m: width } = opening;
  if (!position?.every(Number.isFinite) || !Number.isFinite(width) || width < .12) return null;
  const candidates = projectRuns(runs, position, width);
  const options = [];
  for (const run of candidates) {
    const preference = run.wallId === opening.associated_wall ? .12 : 0;
    if (run.along - width / 2 >= -.07 && run.along + width / 2 <= run.length + .07) {
      options.push({ run, score: run.distance - preference, along: run.along });
      continue;
    }
    // A gap in the wall mask is only bridged with a collinear wall end on each side.
    for (const other of candidates) {
      if (other === run || Math.abs(other.thickness - run.thickness) > .14 || Math.abs(dot(run.direction, other.direction)) < .99) continue;
      const points = [other.start, other.end].map(p => subtract(p, run.start));
      if (points.some(p => Math.abs(dot(p, run.normal)) > .12)) continue;
      const otherFrom = Math.min(...points.map(p => dot(p, run.direction)));
      const otherTo = Math.max(...points.map(p => dot(p, run.direction)));
      const gapStart = otherFrom >= run.length ? run.length : otherTo <= 0 ? otherTo : null;
      const gapEnd = otherFrom >= run.length ? otherFrom : otherTo <= 0 ? 0 : null;
      if (gapStart === null || run.along < gapStart - .1 || run.along > gapEnd + .1) continue;
      const gap = gapEnd - gapStart;
      if (gap > width + .35 || gap < .15) continue;
      const along = (gapStart + gapEnd) / 2;
      const bridge = wallStrip(pointAt(run.start, run.direction, gapStart - .12),
        pointAt(run.start, run.direction, gapEnd + .12), Math.min(run.thickness, other.thickness));
      options.push({ run, along, bridge, score: run.distance + .12 + Math.abs(run.along - along) * .4 - preference });
    }
  }
  const best = options.sort((a, b) => a.score - b.score)[0];
  if (!best) return null;
  const { run, along, bridge } = best;
  const center = pointAt(run.start, run.direction, along);
  return { center, direction: run.direction, angleRad: Math.atan2(-run.direction[1], run.direction[0]),
    wallThickness: run.thickness, wallId: run.wallId, run, bridge };
}

export function fitHouseOpenings(level, scale = .03) {
  const openings = [['door', level.doors || []], ['window', level.windows || []]].flatMap(([type, items]) => items.map(item => ({
    ...item, type,
    position_meters: item.position_meters || item.position_px?.map(value => value * scale),
    width_m: item.width_m || (type === 'door' ? .9 : 1.1),
  })));
  const runs = level.walls.flatMap(wallRuns);
  const hosts = openings.map(opening => hostOpening(opening, runs));

  return openings.map((opening, index) => {
    const host = hosts[index];
    const door = opening.type === 'door';
    let width = opening.width_m;
    let position = opening.position_meters;
    if (host && !host.bridge) {
      const { run, center, direction } = host;
      const normal = [-direction[1], direction[0]];
      let left = dot(subtract(run.start, center), direction) + .07;
      let right = dot(subtract(run.end, center), direction) - .07;
      let keepLeft = -width / 2, keepRight = width / 2;
      const reserve = (distance, padding) => {
        if (distance < 0) left = Math.max(left, distance + padding);
        else right = Math.min(right, distance - padding);
      };
      // Keep space between neighboring openings even when both widen.
      for (const [otherIndex, other] of hosts.entries()) {
        if (!other || otherIndex === index || Math.abs(dot(other.direction, direction)) < .99) continue;
        const relative = subtract(other.center, center);
        if (Math.abs(dot(relative, normal)) > .08) continue;
        const distance = dot(relative, direction);
        const neighborHalf = openings[otherIndex].width_m / 2;
        reserve(distance, Math.max(Math.abs(distance) / 2 + .04, neighborHalf + .08));
        // Touching window detections can give up a few centimeters for a pier.
        if (!door && openings[otherIndex].type === 'window' && Math.abs(distance) < width / 2 + neighborHalf + .08) {
          if (distance < 0) keepLeft = Math.min(0, -width / 2 + .05);
          else keepRight = Math.max(0, width / 2 - .05);
        }
      }
      // Keep the added width out of perpendicular and diagonal wall joints.
      for (const other of runs) {
        const a = subtract(other.start, center), b = subtract(other.end, center);
        const da = dot(a, normal), db = dot(b, normal);
        if (da * db > 0 || Math.abs(db - da) < .05) continue;
        const joint = pointAt(a, subtract(b, a), -da / (db - da));
        reserve(dot(joint, direction), other.thickness / 2 + .05);
      }
      const target = door ? (run.exterior ? .92 : .82) : .88;
      const maxShift = door ? .24 : .10;
      // Expand on the free side of a return with only a small center shift.
      left = Math.min(left, keepLeft);
      right = Math.max(right, keepRight);
      const fittedWidth = Math.min(Math.max(width, target), right - left, 2 * Math.min(right + maxShift, maxShift - left));
      if (fittedWidth >= .15) {
        const shift = Math.max(left + fittedWidth / 2, Math.min(0, right - fittedWidth / 2));
        position = pointAt(position, direction, shift);
        width = fittedWidth;
      }
    }
    const height = door
      ? Math.min(opening.height_m || DOOR_HEIGHT_M, width < .7 ? 1.8 : 1.90)
      : Math.min(opening.height_m || WINDOW_HEIGHT_M, 1.03, width * 1.08);
    const sill = opening.sill_m ?? WINDOW_SILL_M;
    return { ...opening, associated_wall: host?.wallId || opening.associated_wall, position_meters: position, width_m: width,
      height_m: height, sill_m: door ? 0 : sill === WINDOW_SILL_M ? Math.max(sill, Math.min(1.18, 1.86 - height)) : sill };
  });
}

function offsetPolygons(polygons, distance) {
  const paths = pathsFor(polygons);
  if (!paths.length) return [];
  const offset = new ClipperOffset(2, 10);
  offset.AddPaths(paths, JoinType.jtMiter, EndType.etClosedPolygon);
  const tree = new PolyTree();
  offset.Execute(tree, distance * GRID);
  return readPolygons(tree);
}

export function compileWallAssembly(walls, openings, height) {
  const byWall = walls.map(wall => ({ wall, runs: wallRuns(wall) }));
  const runs = byWall.flatMap(item => item.runs);
  const bodies = byWall.map(({ wall, runs }) => ringPolygon(regularizeWallBody(wall, runs)));
  const hosted = [], unresolved = [];
  for (const opening of openings) {
    const host = hostOpening(opening, runs);
    const bottom = Math.max(0, opening.sill_m);
    const top = Math.min(height, bottom + opening.height_m);
    if (!host || !Number.isFinite(top) || top - bottom < .15) {
      unresolved.push(opening.id);
      continue;
    }
    if (host.bridge) bodies.push(ringPolygon(host.bridge));
    const half = opening.width_m / 2;
    const cut = wallStrip(pointAt(host.center, host.direction, -half), pointAt(host.center, host.direction, half), host.wallThickness + .08);
    hosted.push({ ...opening, host, cut, sill_m: bottom, height_m: top - bottom, top });
  }
  const merged = combinePolygons(bodies);
  // Close only local gaps under 7 cm. Actual openings are cut after joining.
  const joined = offsetPolygons(offsetPolygons(merged, .035), -.035);
  const footprint = joined.length ? joined : merged;
  const cuts = [...new Set([0, height, ...hosted.flatMap(o => [o.sill_m, o.top])].map(y => Math.round(y * GRID) / GRID))].sort((a, b) => a - b);
  const bands = cuts.slice(1).map((top, i) => {
    const bottom = cuts[i], midpoint = (bottom + top) / 2;
    const active = hosted.filter(o => midpoint > o.sill_m && midpoint < o.top);
    return { bottom, top, polygons: combinePolygons(footprint, active.map(o => ringPolygon(o.cut)), ClipType.ctDifference) };
  });
  return { bands, hosted, unresolved };
}

export function wallAssemblyGeometry(bands, centerX, centerZ) {
  const positions = [], uv = [];
  function append(geometry) {
    const flat = geometry.index ? geometry.toNonIndexed() : geometry;
    for (const value of flat.attributes.position.array) positions.push(value);
    for (const value of flat.attributes.uv.array) uv.push(value);
    if (flat !== geometry) flat.dispose();
    geometry.dispose();
  }
  function cap(polygons, y, upward) {
    polygons.forEach(polygon => {
      const shape = new THREE.ShapeGeometry(planShape(polygon, centerX, centerZ));
      const flat = shape.toNonIndexed();
      shape.dispose();
      flat.rotateX(-Math.PI / 2);
      flat.translate(0, y, 0);
      if (!upward) {
        for (const attribute of [flat.attributes.position, flat.attributes.uv]) {
          for (let i = 0; i < attribute.count; i += 3) {
            for (let j = 0; j < attribute.itemSize; j++) {
              const first = i * attribute.itemSize + j, last = (i + 2) * attribute.itemSize + j;
              [attribute.array[first], attribute.array[last]] = [attribute.array[last], attribute.array[first]];
            }
          }
        }
      }
      append(flat);
    });
  }
  bands.forEach(({ polygons, bottom, top }, i) => {
    for (const polygon of polygons) {
      const extruded = new THREE.ExtrudeGeometry(planShape(polygon, centerX, centerZ), { depth: top - bottom, bevelEnabled: false });
      extruded.rotateX(-Math.PI / 2);
      extruded.translate(0, bottom, 0);
      // Keep only side faces. At each transition emit only the exposed cap.
      for (const group of extruded.groups.filter(group => group.materialIndex === 1)) {
        positions.push(...extruded.attributes.position.array.slice(group.start * 3, (group.start + group.count) * 3));
        uv.push(...extruded.attributes.uv.array.slice(group.start * 2, (group.start + group.count) * 2));
      }
      extruded.dispose();
    }
    cap(combinePolygons(polygons, bands[i - 1]?.polygons || [], ClipType.ctDifference), bottom, false);
    cap(combinePolygons(polygons, bands[i + 1]?.polygons || [], ClipType.ctDifference), top, true);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geometry.computeVertexNormals();
  return geometry;
}
