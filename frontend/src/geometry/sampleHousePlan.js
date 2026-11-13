import { wallStrip } from './houseGeometry.js';
import { STOREY_HEIGHT_M, DOOR_HEIGHT_M, WINDOW_HEIGHT_M, WINDOW_SILL_M } from './buildingDimensions.js';

// Traced against the bundled drawing at a reference image size of 1832 × 1344.
// The two drawings share the top-left exterior corner. The input bitmap is
// still shown in sheet coordinates so users can compare the trace against it.
const DRAWINGS = [
  {
    id: 'ground', label: 'Ground floor', elevation: 0, anchor: [78, 131],
    outline: [[67, 119], [602, 119], [602, 326], [770, 326], [770, 1273], [495, 1273], [495, 1093], [446, 1041], [67, 1041]],
    exterior: [
      ['north', [78, 131], [589, 131], [[168, 131, 65], [249, 131, 81], [417, 131, 48], [516, 131, 104]]],
      ['setback', [589, 131], [589, 337], [], [[589, 226, 80]]],
      ['east_top', [589, 337], [758, 337], [[664, 337, 64]]],
      ['east', [758, 337], [758, 1261], [[758, 417, 58], [758, 493, 78], [758, 766, 61], [758, 1020, 46], [758, 1067, 48], [758, 1173, 62]], [[758, 916, 58]]],
      ['south_extension', [758, 1261], [507, 1261]],
      ['terrace_side', [507, 1261], [507, 1087], [], [[507, 1195, 62]]],
      ['diagonal', [507, 1087], [451, 1030], [[484, 1064, 25]]],
      ['south', [451, 1030], [78, 1030], [[205, 1030, 65]], [[390, 1030, 64]]],
      ['west', [78, 1030], [78, 131], [[78, 904, 70], [78, 832, 60], [78, 533, 82], [78, 282, 70]]],
    ],
    interior: [
      ['west_partition', [330, 414], [330, 804], [], [[330, 504, 54], [330, 776, 48]]],
      ['west_kitchen', [78, 456], [330, 456]],
      ['west_bedrooms', [78, 652], [330, 652]],
      ['west_closet', [330, 804], [290, 804]],
      ['west_closet_south', [290, 804], [290, 1030]],
      ['utility_north', [486, 650], [758, 650]],
      ['utility_west', [486, 650], [486, 765]],
      ['closet_south', [486, 765], [612, 765]],
      ['closet_east', [612, 650], [612, 765], [], [[612, 730, 52]]],
      ['utility_hall', [537, 765], [537, 925], [], [[537, 838, 55]]],
      ['wc_north', [453, 925], [537, 925], [], [[497, 925, 55]]],
      ['wc_west', [453, 925], [453, 1030]],
      ['wc_east_top', [537, 925], [582, 960]],
      ['wc_east', [582, 960], [582, 1087]],
      ['bath_north', [582, 960], [758, 960], [], [[677, 960, 52]]],
      ['wc_south', [507, 1087], [582, 1087]],
      ['bath_corner', [582, 1087], [609, 1112]],
      ['bath_south', [609, 1112], [758, 1112], [], [[713, 1112, 53]]],
      ['utility_south', [609, 1112], [609, 1261]],
    ],
    rooms: [
      ['Kitchen', [[88, 141], [333, 141], [333, 450], [88, 450]]],
      ['Living Room', [[333, 141], [579, 141], [579, 347], [748, 347], [748, 643], [338, 643]]],
      ['Bedroom', [[88, 463], [323, 463], [323, 645], [88, 645]]],
      ['Bedroom', [[88, 659], [323, 659], [323, 797], [283, 797], [283, 1020], [88, 1020]]],
      ['Entry / Hall', [[338, 649], [479, 649], [479, 772], [530, 772], [530, 918], [446, 918], [446, 1020], [297, 1020], [297, 812], [338, 812]]],
      ['Storage', [[493, 657], [605, 657], [605, 758], [493, 758]]],
      ['Utility', [[619, 657], [748, 657], [748, 953], [585, 953], [544, 920], [544, 772], [619, 772]]],
      ['Bathroom', [[460, 932], [535, 932], [575, 964], [575, 1080], [514, 1080], [460, 1024]]],
      ['Bathroom', [[589, 967], [748, 967], [748, 1105], [612, 1105], [589, 1083]]],
      ['Utility', [[517, 1094], [575, 1094], [602, 1117], [602, 1251], [517, 1251]]],
      ['Sauna', [[616, 1119], [748, 1119], [748, 1251], [616, 1251]]],
    ],
    voids: [],
  },
  {
    id: 'upper', label: 'Upper floor', elevation: STOREY_HEIGHT_M, anchor: [1037, 131],
    outline: [[1025, 119], [1559, 119], [1559, 323], [1732, 323], [1732, 678], [1025, 678]],
    exterior: [
      ['north', [1037, 131], [1547, 131], [[1142, 131, 94], [1375, 131, 69], [1460, 131, 69]]],
      ['setback', [1547, 131], [1547, 335], [], [[1547, 260, 65]]],
      ['east_top', [1547, 335], [1720, 335], [[1645, 335, 71]]],
      ['east', [1720, 335], [1720, 666], [[1720, 419, 65], [1720, 494, 75]]],
      ['south', [1720, 666], [1037, 666]],
      ['west', [1037, 666], [1037, 131], [[1037, 530, 85], [1037, 280, 69]]],
    ],
    interior: [
      ['bed_split', [1297, 131], [1297, 327]],
      ['bed_west', [1037, 378], [1244, 378]],
      ['bed_west_diag', [1244, 378], [1297, 327], [], [[1270, 352, 58]]],
      ['bed_east_diag', [1297, 327], [1351, 378], [], [[1325, 353, 59]]],
      ['bed_east', [1351, 378], [1547, 378]],
      ['bed_east_end', [1547, 378], [1547, 335]],
      ['bed_south', [1244, 378], [1244, 666], [], [[1244, 420, 55]]],
      ['wc_north', [1244, 548], [1363, 548], [], [[1321, 548, 53]]],
      ['wc_east', [1363, 548], [1363, 666]],
    ],
    rooms: [
      ['Bedroom', [[1047, 141], [1290, 141], [1290, 324], [1241, 371], [1047, 371]]],
      ['Bedroom', [[1304, 141], [1537, 141], [1537, 371], [1354, 371], [1304, 324]]],
      ['Bedroom', [[1047, 385], [1237, 385], [1237, 656], [1047, 656]]],
      ['Bathroom', [[1251, 555], [1356, 555], [1356, 656], [1251, 656]]],
      ['Entry / Hall', [[1251, 381], [1297, 337], [1348, 385], [1533, 385], [1533, 656], [1449, 656], [1449, 562], [1369, 562], [1369, 541], [1251, 541]]],
    ],
    // Stair opening and the open-to-below space marked on the source drawing.
    voids: [
      [[1372, 567], [1445, 567], [1445, 679], [1372, 679]],
      [[1540, 346], [1708, 346], [1708, 654], [1540, 654]],
    ],
  },
];

export const SAMPLE_FRONT_WALL = 'ground_east';

export function createSampleHouse(scale = .03) {
  const pixel = ([x, y]) => [x * 512 / 1832, 68 + y * 375 / 1344];
  const meters = point => pixel(point).map(value => value * scale);
  const lowerAnchor = meters(DRAWINGS[0].anchor);
  const walls = [], rooms = [], doors = [], windows = [];
  const storeys = DRAWINGS.map(drawing => {
    const { id, label, elevation } = drawing;
    const anchor = meters(drawing.anchor);
    const offset = lowerAnchor.map((value, i) => value - anchor[i]);
    for (const [isExterior, definitions] of [[true, drawing.exterior], [false, drawing.interior]]) {
      for (const [name, a, b, windowDefs = [], doorDefs = []] of definitions) {
        const wallId = `${id}_${name}`;
        const axis = [meters(a), meters(b)];
        const thickness = (isExterior ? 22 : 13) * 512 / 1832 * scale;
        const dx = axis[1][0] - axis[0][0], dz = axis[1][1] - axis[0][1];
        const length = Math.hypot(dx, dz);
        const extension = [dx / length * thickness / 2, dz / length * thickness / 2];
        const ring = wallStrip(axis[0].map((v, i) => v - extension[i]), axis[1].map((v, i) => v + extension[i]), thickness);
        const facadeFrame = isExterior ? {
          origin: ring[2], direction: [-dx / length, -dz / length],
          outward: [dz / length, -dx / length], length_m: length + thickness,
        } : undefined;
        const openings = [];
        for (const [type, definitions, destination] of [['window', windowDefs, windows], ['door', doorDefs, doors]]) {
          definitions.forEach(([x, z, width], i) => {
            const opening = { id: `${wallId}_${type}_${i}`, type, level_id: id, associated_wall: wallId,
              position_meters: meters([x, z]), width_m: width * 512 / 1832 * scale,
              sill_m: type === 'door' ? 0 : WINDOW_SILL_M,
              height_m: type === 'door' ? DOOR_HEIGHT_M : WINDOW_HEIGHT_M };
            openings.push(opening);
            destination.push(opening);
          });
        }
        walls.push({ id: wallId, level_id: id, is_exterior: isExterior, is_front: wallId === SAMPLE_FRONT_WALL,
          height_m: STOREY_HEIGHT_M, axis_meters: axis, thickness_m: thickness, facade_frame: facadeFrame,
          polygon_meters: ring, polygon_px: ring.map(p => p.map(v => v / scale)), openings });
      }
    }
    drawing.rooms.forEach(([room_type, points], i) => {
      const polygon_meters = points.map(meters);
      const area = polygon_meters.reduce((sum, [x, z], j) => {
        const [a, b] = polygon_meters[(j + 1) % polygon_meters.length];
        return sum + x * b - z * a;
      }, 0);
      rooms.push({ id: `${id}_room_${i}`, level_id: id, room_type, polygon_meters, surface_m2: Number((Math.abs(area) / 2).toFixed(2)) });
    });
    return { id, label, elevation, height: STOREY_HEIGHT_M, offset, envelope: [drawing.outline.map(meters)], floorVoids: drawing.voids.map(ring => ring.map(meters)) };
  });
  return { walls, rooms, doors, windows, storeys, envelope: storeys.flatMap(level => level.envelope),
    metadata: { source: 'sample_reference', canvas_size: [512, 512], scale_meters_per_pixel: scale,
      total_surface_m2: rooms.reduce((sum, room) => sum + room.surface_m2, 0), num_rooms: rooms.length } };
}

export function rescaleSampleHouse(layout, scale) {
  const ratio = scale / layout.metadata.scale_meters_per_pixel;
  const point = values => values.map(value => value * ratio);
  const ring = points => points.map(point);
  const opening = item => ({ ...item, position_meters: point(item.position_meters), width_m: item.width_m * ratio });
  const rooms = layout.rooms.map(room => ({ ...room, polygon_meters: ring(room.polygon_meters), surface_m2: Number((room.surface_m2 * ratio ** 2).toFixed(2)) }));
  const storeys = layout.storeys.map(level => ({ ...level, offset: point(level.offset), envelope: level.envelope.map(ring), floorVoids: level.floorVoids.map(ring) }));
  return { ...layout, rooms, storeys, envelope: storeys.flatMap(level => level.envelope),
    doors: layout.doors.map(opening), windows: layout.windows.map(opening),
    walls: layout.walls.map(wall => ({ ...wall, polygon_meters: ring(wall.polygon_meters),
      axis_meters: wall.axis_meters && ring(wall.axis_meters), thickness_m: wall.thickness_m * ratio,
      facade_frame: wall.facade_frame && { ...wall.facade_frame, origin: point(wall.facade_frame.origin), length_m: wall.facade_frame.length_m * ratio },
      openings: wall.openings.map(opening) })),
    metadata: { ...layout.metadata, scale_meters_per_pixel: scale, total_surface_m2: rooms.reduce((sum, room) => sum + room.surface_m2, 0) } };
}
