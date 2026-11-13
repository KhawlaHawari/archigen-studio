import assert from 'node:assert/strict';
import { test } from 'node:test';
import { signedDistanceToRing } from '../src/geometry/houseGeometry.js';
import { placeRoomBadge } from '../src/geometry/roomBadgeGeometry.js';

const rectangle = [[0, 0], [120, 0], [120, 80], [0, 80]];
const inside = (badge, ring) => {
  for (const dx of [-badge.halfWidth, badge.halfWidth]) {
    for (const dy of [-badge.halfHeight, badge.halfHeight]) {
      assert.ok(signedDistanceToRing([badge.x + dx, badge.y + dy], ring) > 0);
    }
  }
};

test('room badge stays centered when the whole label fits', () => {
  const badge = placeRoomBadge(rectangle, 80, 36, []);
  assert.deepEqual([badge.x, badge.y, badge.scale], [60, 40, 1]);
  inside(badge, rectangle);
});

test('room badge moves into an L-shaped interior and shrinks to fit', () => {
  const ring = [[0, 0], [100, 0], [100, 18], [18, 18], [18, 100], [0, 100]];
  const badge = placeRoomBadge(ring, 80, 36, []);
  assert.ok(badge.scale > 0 && badge.scale < 1);
  inside(badge, ring);
});

test('badges in overlapping detections leave a gap', () => {
  const first = placeRoomBadge(rectangle, 80, 36, []);
  const second = placeRoomBadge(rectangle, 80, 36, [first]);
  inside(second, rectangle);
  assert.ok(Math.abs(first.x - second.x) > first.halfWidth + second.halfWidth ||
    Math.abs(first.y - second.y) > first.halfHeight + second.halfHeight);
});
