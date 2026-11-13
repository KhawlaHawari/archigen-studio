import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Box3 } from 'three';
import { applyFurnitureFinish, furnitureColorChannels } from '../src/geometry/furnitureAppearance.js';
import { buildFurnitureShape, furnitureShapeOptions } from '../src/geometry/furnitureShapes.js';

test('furniture profiles stay inside the room planner envelope', () => {
  const sizes = { sofa: [1.93, .81, .90], bed: [.63, 1.25, .42], dining_table: [1.64, .87, .64],
    dining_chair: [.36, .36, .83], wardrobe: [1, .6, 2] };
  for (const [category, dimensions] of Object.entries(sizes)) {
    const [w, d, h] = dimensions;
    for (const { id: shape } of furnitureShapeOptions(category)) {
      const model = buildFurnitureShape({ category, dimensions_m: dimensions, shape });
      const { min, max } = new Box3().setFromObject(model);
      assert(min.x >= -w / 2 - 1e-5 && max.x <= w / 2 + 1e-5, shape);
      assert(min.z >= -d / 2 - 1e-5 && max.z <= d / 2 + 1e-5, shape);
      assert(Math.abs(min.y) < 1e-5 && max.y <= h + 1e-5, shape);
    }
  }
  assert.equal(buildFurnitureShape({ category: 'sofa', dimensions_m: sizes.sofa, shape: 'original' }), null);
});

test('shapes change geometry and retain independent material colors', () => {
  const item = { category: 'sofa', dimensions_m: [2, .8, .9] };
  const rolled = buildFurnitureShape({ ...item, shape: 'roll-arm' });
  const armless = buildFurnitureShape({ ...item, shape: 'armless' });
  assert(rolled.children.length > armless.children.length);
  for (const model of [rolled, armless]) {
    const back = new Box3().setFromObject(model.getObjectByName('backrest'));
    const frame = new Box3().setFromObject(model.getObjectByName('frame'));
    assert(back.intersectsBox(frame));
  }
  assert.deepEqual(furnitureColorChannels(armless, 'sofa'), ['fabric', 'wood']);
  applyFurnitureFinish(armless, 'tailored_classic', 'sofa', { fabric: '#126880', wood: '#674438' });
  assert.equal(armless.children.find(mesh => mesh.material.name === 'carpet').material.color.getHexString(), '126880');
  assert.equal(armless.children.find(mesh => mesh.material.name === 'wood').material.color.getHexString(), '674438');
});
