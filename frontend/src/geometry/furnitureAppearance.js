export const FURNITURE_FINISHES = {
  contemporary_minimal: {
    label: 'Contemporary Minimal', desc: 'Pale oak, slate fabric, brushed steel',
    wood: '#c2b6a5', darkWood: '#8b8073', fabric: '#607d95', linen: '#f1f4f4',
    metal: '#adb9c4', darkMetal: '#454d54', ceramic: '#eef3f6',
  },
  warm_wood: {
    label: 'Warm Wood', desc: 'Honey oak, sage fabric, warm cream',
    wood: '#bc8a54', darkWood: '#94643c', fabric: '#7f936d', linen: '#eadbc5',
    metal: '#b8a386', darkMetal: '#706352', ceramic: '#eedec4',
  },
  tailored_classic: {
    label: 'Tailored Classic', desc: 'Dark walnut, burgundy fabric, aged brass',
    wood: '#684238', darkWood: '#442d28', fabric: '#87454f', linen: '#e0c9b1',
    metal: '#b18a55', darkMetal: '#6d4a30', ceramic: '#d9bfb0',
  },
};

const fixtures = new Set(['toilet', 'bathroom_sink', 'bathtub']);

export const FURNITURE_COLOR_LABELS = {
  fabric: 'Upholstery', linen: 'Light fabric', wood: 'Wood', metal: 'Hardware', ceramic: 'Ceramic',
};
const groupedColors = { darkWood: 'wood', darkMetal: 'metal' };

function materialRole(material, category) {
  if (!material?.color || material.map || material.transparent || material.transmission) return null;
  const name = material.name.toLowerCase();
  if (fixtures.has(category) && /^(carpetwhite|metallight|_defaultmat)/.test(name)) return 'ceramic';
  if (name.startsWith('wood')) return name.includes('dark') ? 'darkWood' : 'wood';
  if (name.startsWith('carpet')) return name.includes('white') ? 'linen' : 'fabric';
  if (name.startsWith('metal')) return name.includes('dark') ? 'darkMetal' : 'metal';
  return null;
}

function paintableMaterials(model, category) {
  const materials = new Map();
  model.traverse(mesh => {
    if (!mesh.isMesh) return;
    for (const material of [mesh.material].flat()) {
      const role = materialRole(material, category);
      if (role) materials.set(material, role);
    }
  });
  return materials;
}

export function furnitureColorChannels(model, category) {
  const roles = new Set([...paintableMaterials(model, category).values()].map(role => groupedColors[role] || role));
  return Object.keys(FURNITURE_COLOR_LABELS).filter(channel => roles.has(channel));
}

export function applyFurnitureFinish(model, finish, category, colors = {}) {
  const palette = FURNITURE_FINISHES[finish];
  if (!palette) return;
  for (const [material, role] of paintableMaterials(model, category)) {
    const custom = colors[groupedColors[role] || role];
    material.color.set(custom || palette[role]);
    if ((custom && groupedColors[role]) || material.name.toLowerCase() === 'carpetdarker') {
      material.color.multiplyScalar(.65);
    }
  }
}
