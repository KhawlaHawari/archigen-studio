import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Group, Mesh, MeshStandardMaterial } from 'three';
import { applyFurnitureFinish, FURNITURE_FINISHES } from '../src/geometry/furnitureAppearance.js';

test('all three presets recolor a shared furniture mesh and preserve its glass', () => {
  const wood = new MeshStandardMaterial({ name: 'wood' });
  const fabric = new MeshStandardMaterial({ name: 'carpet' });
  const glass = new MeshStandardMaterial({ name: 'glass', color: '#a0ccd2', transparent: true });
  const group = new Group().add(new Mesh(undefined, [wood, fabric, glass]));
  const initialGlass = glass.color.getHex();
  const finished = new Set();
  for (const finish of Object.keys(FURNITURE_FINISHES)) {
    applyFurnitureFinish(group, finish, 'bed');
    assert.equal(glass.color.getHex(), initialGlass);
    finished.add([wood.color.getHex(), fabric.color.getHex()].join(','));
  }
  assert.equal(finished.size, 3);
});
