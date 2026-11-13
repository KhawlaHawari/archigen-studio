import { BoxGeometry, CylinderGeometry, ExtrudeGeometry, Group, Mesh, MeshStandardMaterial, Shape } from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const sofa = [
  { id: 'track-arm', label: 'Track arms', desc: 'Squared arms, individual cushions and slim feet' },
  { id: 'roll-arm', label: 'Rolled arms', desc: 'Rounded arm caps and a softer upholstered back' },
  { id: 'armless', label: 'Armless lounge', desc: 'Open sides and a low, continuous seat' },
];
const bed = [
  { id: 'platform', label: 'Platform bed', desc: 'Low timber base and straight headboard' },
  { id: 'arched', label: 'Arched headboard', desc: 'A curved upholstered headboard with exposed feet' },
  { id: 'wingback', label: 'Wingback bed', desc: 'Padded headboard with forward-facing wings' },
];
const table = [
  { id: 'rounded-rectangle', label: 'Rounded rectangle', desc: 'Soft corners and four tapered legs' },
  { id: 'oval-pedestal', label: 'Oval pedestal', desc: 'An elliptical top over a central base' },
  { id: 'round-pedestal', label: 'Round pedestal', desc: 'A circular top sized within the available footprint' },
];
const chair = [
  { id: 'slat-back', label: 'Slat back', desc: 'Open timber slats with a sculpted seat' },
  { id: 'padded-back', label: 'Padded back', desc: 'A rounded upholstered shell on tapered feet' },
];
const cabinet = [
  { id: 'panel-doors', label: 'Panel doors', desc: 'Framed fronts with individual pulls' },
  { id: 'open-shelves', label: 'Open shelves', desc: 'Open-front compartments with visible shelving' },
];

const profiles = {
  sofa, armchair: sofa, bed, dining_table: table, coffee_table: table, desk: table,
  dining_chair: chair, wardrobe: cabinet, nightstand: cabinet, sideboard: cabinet, bookcase: cabinet,
};

export function furnitureShapeOptions(category) {
  return profiles[category] || [];
}

// The profile changes the mesh inside the validated envelope. Local +Z faces
// forward and the base stays at Y=0, the same convention as the catalog GLBs.
export function buildFurnitureShape(item) {
  const { category, shape, dimensions_m: dimensions } = item;
  const family = profiles[category];
  if (!family?.some(option => option.id === shape) || !dimensions?.every(value => value > 0 && Number.isFinite(value))) return null;
  const group = new Group();
  group.name = shape;
  const [w, d, h] = dimensions;
  const materials = {};
  const add = (geometry, name, x, y, z) => {
    materials[name] ??= new MeshStandardMaterial({ name, roughness: name.startsWith('carpet') ? .85 : .48 });
    const mesh = new Mesh(geometry, materials[name]);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  };
  const box = (name, x, y, z, width, height, depth, radius = 0) => add(radius > 0
    ? new RoundedBoxGeometry(width, height, depth, 2, Math.min(radius, width / 2, height / 2, depth / 2))
    : new BoxGeometry(width, height, depth), name, x, y, z);
  const legs = (height, spacingX, spacingZ, thickness, material = 'woodDark') => {
    for (const x of [-spacingX, spacingX]) {
      for (const z of [-spacingZ, spacingZ]) {
        add(new CylinderGeometry(thickness, thickness * .62, height, 12), material, x, height / 2, z);
      }
    }
  };

  if (family === sofa) {
    const rounded = shape === 'roll-arm', open = shape === 'armless';
    const armWidth = open ? 0 : w * (rounded ? .13 : .095);
    const foot = h * (open ? .15 : .10);
    const cushionCount = open || w < 1.08 ? 1 : w > 1.9 ? 3 : 2;
    const seatWidth = (w - 2 * armWidth) * .96 / cushionCount;
    legs(foot, w * .39, d * .37, Math.min(w, d) * .038, open ? 'wood' : 'metalDark');
    box(open ? 'wood' : 'carpetDarker', 0, foot + h * .11, 0, w, h * .22, d, .05).name = 'frame';
    const backHeight = h * .72;
    box('carpet', 0, h - backHeight / 2, -d * .39, w - armWidth, backHeight, d * .22, rounded ? .11 : .04).name = 'backrest';
    for (let index = 0; index < cushionCount; index++) {
      const x = (index - (cushionCount - 1) / 2) * seatWidth;
      box('carpet', x, foot + h * .29, d * .075, seatWidth * .98, h * .16, d * .72, rounded ? .08 : .045);
    }
    if (!open) {
      for (const x of [-(w - armWidth) / 2, (w - armWidth) / 2]) {
        box('carpet', x, h * .46, 0, armWidth, h * .47, d * .94, rounded ? armWidth * .48 : .025);
        if (rounded) box('carpet', x, h * .69, d * .01, armWidth, h * .22, d * .90, armWidth * .48);
      }
    }
  } else if (family === bed) {
    const low = shape === 'platform';
    const foot = h * (low ? .06 : .14);
    legs(foot, w * .37, d * .38, Math.min(w, d) * .038);
    box('wood', 0, foot + h * .12, 0, w, h * .24, d, .035);
    box('carpetWhite', 0, h * .44, d * .02, w * .94, h * .18, d * .91, .045);
    box('carpet', 0, h * .54, d * .18, w * .94, h * .075, d * .59, .025);
    const pillows = w > 1.25 ? 2 : 1;
    for (let index = 0; index < pillows; index++) {
      const x = (index - (pillows - 1) / 2) * w * .44;
      box('carpetWhite', x, h * .56, -d * .28, w * .80 / pillows, h * .10, d * .20, .045);
    }
    if (shape === 'arched') {
      const panel = new Shape();
      panel.moveTo(-w / 2, 0);
      panel.lineTo(-w / 2, h * .46);
      panel.quadraticCurveTo(-w * .44, h * .86, 0, h * .86);
      panel.quadraticCurveTo(w * .44, h * .86, w / 2, h * .46);
      panel.lineTo(w / 2, 0);
      panel.closePath();
      add(new ExtrudeGeometry(panel, { depth: d * .065, bevelEnabled: false }), 'carpet', 0, foot, -d / 2);
    } else {
      box(low ? 'wood' : 'carpet', 0, (h + foot) / 2, -d * .46, w, h - foot, d * .08, .035);
      if (shape === 'wingback') {
        for (const x of [-w * .46, w * .46]) box('carpetDarker', x, h * .60, -d * .38, w * .08, h * .60, d * .22, .035);
      }
    }
  } else if (family === table) {
    const top = h * .10;
    if (shape === 'rounded-rectangle') {
      legs(h - top, w * .36, d * .35, Math.min(w, d) * .045);
      box('wood', 0, h - top / 2, 0, w, top, d, .04);
    } else {
      const diameter = Math.min(w, d);
      const oval = shape === 'oval-pedestal';
      const topMesh = add(new CylinderGeometry(.5, .5, top, 48), 'wood', 0, h - top / 2, 0);
      topMesh.scale.set(oval ? w : diameter, 1, oval ? d : diameter);
      add(new CylinderGeometry(diameter * .105, diameter * .18, h * .82, 32), 'woodDark', 0, h * .49, 0);
      add(new CylinderGeometry(diameter * .29, diameter * .31, h * .08, 40), 'metalDark', 0, h * .04, 0);
    }
  } else if (family === chair) {
    const upholstered = shape === 'padded-back';
    legs(h * .48, w * .36, d * .33, Math.min(w, d) * .053);
    box(upholstered ? 'carpet' : 'wood', 0, h * .50, d * .025, w, h * .10, d * .95, .045);
    if (upholstered) {
      box('carpet', 0, h * .74, -d * .40, w * .98, h * .52, d * .18, .06);
    } else {
      for (const x of [-.45, -.22, 0, .22, .45]) box('woodDark', w * x, h * .71, -d * .45, w * .075, h * .40, d * .09, .01);
      box('wood', 0, h * .95, -d * .44, w, h * .10, d * .12, .024);
    }
  } else {
    const foot = h * .08, thickness = Math.min(w, h, d) * .08;
    legs(foot, w * .36, d * .33, thickness * .72);
    for (const x of [-(w - thickness) / 2, (w - thickness) / 2]) {
      box('wood', x, (h + foot) / 2, 0, thickness, h - foot, d);
    }
    box('woodDark', 0, (h + foot) / 2, -d / 2 + thickness / 2, w, h - foot, thickness);
    for (const y of [foot + thickness / 2, h - thickness / 2]) box('wood', 0, y, 0, w, thickness, d);
    if (shape === 'open-shelves') {
      for (const fraction of (h > 1 ? [.25, .50, .75] : [.48])) {
        box('wood', 0, foot + (h - foot) * fraction, 0, w - thickness * 2, thickness, d - thickness * 2);
      }
    } else {
      const double = w > .65;
      for (const x of (double ? [-w * .24, w * .24] : [0])) {
        const width = w * (double ? .46 : .91);
        box('wood', x, (h + foot) / 2, d * .45, width, (h - foot) * .94, d * .08, .018);
        box('woodDark', x, (h + foot) / 2, d * .492, width * .76, (h - foot) * .76, d * .016, .01);
        box('metal', x === 0 ? 0 : x * .47, h * .58, d * .482, width * .12, h * .065, d * .035, .008);
      }
    }
  }
  return group;
}
