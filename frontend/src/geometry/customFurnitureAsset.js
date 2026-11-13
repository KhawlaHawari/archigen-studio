import { Box3, Group, Vector3 } from 'three';

export function fitCustomFurnitureAsset(scene, dimensions) {
  const bounds = new Box3().setFromObject(scene);
  const center = bounds.getCenter(new Vector3());
  const size = bounds.getSize(new Vector3());
  const [width, depth, height] = dimensions;
  const scale = Math.min(width / size.x, depth / size.z, height / size.y);
  if (!Number.isFinite(scale) || scale <= 0) return scene;

  // Older saved assets can have an off-center origin or a floating base.
  scene.position.sub(new Vector3(center.x, bounds.min.y, center.z));
  const model = new Group().add(scene);
  model.scale.setScalar(scale);
  return model;
}
