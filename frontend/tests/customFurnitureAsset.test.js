import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Box3, BoxGeometry, Group, Mesh, Vector3 } from 'three';
import { fitCustomFurnitureAsset } from '../src/geometry/customFurnitureAsset.js';

test('custom furniture sits on the floor and fits its checked dimensions without distortion', () => {
  const asset = new Group();
  const mesh = new Mesh(new BoxGeometry(2, 4, 6));
  mesh.position.set(3, 7, -2);
  asset.add(mesh);
  const model = fitCustomFurnitureAsset(asset, [1, 2, 2]);
  const bounds = new Box3().setFromObject(model);
  const center = bounds.getCenter(new Vector3());
  assert(Math.abs(center.x) < 1e-6 && Math.abs(center.z) < 1e-6 && Math.abs(bounds.min.y) < 1e-6);
  assert(bounds.getSize(new Vector3()).distanceTo(new Vector3(2 / 3, 4 / 3, 2)) < 1e-6);
});
