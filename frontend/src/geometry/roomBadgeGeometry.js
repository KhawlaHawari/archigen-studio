import { signedDistanceToRing } from './houseGeometry.js';

export function placeRoomBadge(ring, width, height, placed) {
  if (ring.length < 3) return null;
  const edges = ring.map((point, i) => [point, ring[(i + 1) % ring.length]]);
  let area = 0, centerX = 0, centerY = 0;
  for (const [[ax, ay], [bx, by]] of edges) {
    const cross = ax * by - bx * ay;
    area += cross;
    centerX += (ax + bx) * cross;
    centerY += (ay + by) * cross;
  }
  if (!area) return null;
  const center = [centerX / (3 * area), centerY / (3 * area)];
  const halfWidth = width / 2 + 1, halfHeight = height / 2 + 1;
  const normalizedRing = ring.map(([x, y]) => [x / halfWidth, y / halfHeight]);

  const fitAt = (x, y) => {
    // In badge coordinates the rectangle fits inside a circle of radius √2.
    let scale = Math.min(1, signedDistanceToRing([x / halfWidth, y / halfHeight], normalizedRing) / Math.SQRT2);
    for (const other of placed) {
      const horizontal = (Math.abs(x - other.x) - other.halfWidth - 1) / halfWidth;
      const vertical = (Math.abs(y - other.y) - other.halfHeight - 1) / halfHeight;
      scale = Math.min(scale, Math.max(horizontal, vertical));
    }
    return { x, y, scale, halfWidth: halfWidth * scale, halfHeight: halfHeight * scale };
  };

  let best = fitAt(...center);
  if (best.scale === 1) return best;
  let distance = 0;
  // Scan between consecutive vertex heights so even thin concave arms have a candidate.
  const heights = [...new Set(ring.map(([, y]) => y))].sort((a, b) => a - b);
  const minY = heights[0], maxY = heights.at(-1);
  const scanlines = [center[1], ...heights.slice(1).map((y, i) => (y + heights[i]) / 2)];
  for (let i = 1; i < 12; i++) scanlines.push(minY + (maxY - minY) * i / 12);
  for (const y of scanlines) {
    const crossings = [];
    for (const [[ax, ay], [bx, by]] of edges) {
      if ((ay > y) !== (by > y)) crossings.push(ax + (bx - ax) * (y - ay) / (by - ay));
    }
    crossings.sort((a, b) => a - b);
    for (let i = 0; i + 1 < crossings.length; i += 2) {
      const left = crossings[i], right = crossings[i + 1];
      for (const fraction of [.15, .33, .5, .67, .85]) {
        const x = left + (right - left) * fraction;
        const candidate = fitAt(x, y);
        const fromCenter = Math.hypot(x - center[0], y - center[1]);
        if (candidate.scale > best.scale || (candidate.scale === best.scale && fromCenter < distance)) {
          best = candidate;
          distance = fromCenter;
        }
      }
    }
  }
  return best.scale > 0 ? best : null;
}
