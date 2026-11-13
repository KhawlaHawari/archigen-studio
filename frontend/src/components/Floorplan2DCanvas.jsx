import React, { useState } from 'react';
import { STOREY_HEIGHT_M } from '../geometry/buildingDimensions.js';
import {
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  CheckCircle2,
  Info,
  Ruler,
  X,
  Check,
  MousePointer,
  PlusCircle,
  Trash2,
  Tag,
  Layers,
  GripVertical,
  HelpCircle
} from 'lucide-react';
import { alignDrawingLevels, drawingLevelAt, separateDrawingLevels, wallStrip } from '../geometry/houseGeometry.js';
import { placeRoomBadge } from '../geometry/roomBadgeGeometry.js';

const ROOM_COLORS = {
  'Living Room': { fill: 'rgba(28, 78, 128, 0.20)', stroke: '#1c4e80' },
  'Living Room / Common': { fill: 'rgba(28, 78, 128, 0.20)', stroke: '#1c4e80' },
  'Bedroom': { fill: 'rgba(184, 69, 42, 0.18)', stroke: '#b8452a' },
  'Kitchen': { fill: 'rgba(196, 141, 63, 0.20)', stroke: '#c48d3f' },
  'Bathroom': { fill: 'rgba(42, 122, 110, 0.20)', stroke: '#2a7a6e' },
  'Bathroom / Storage': { fill: 'rgba(42, 122, 110, 0.20)', stroke: '#2a7a6e' },
  'Entry': { fill: 'rgba(110, 90, 70, 0.18)', stroke: '#6e5a46' },
  'Outdoor / Balcony': { fill: 'rgba(60, 110, 60, 0.18)', stroke: '#3c6e3c' },
  'Storage / Closet': { fill: 'rgba(90, 90, 100, 0.18)', stroke: '#5a5a64' },
  'Office': { fill: 'rgba(140, 75, 120, 0.18)', stroke: '#8c4b78' },
  'Default': { fill: 'rgba(60, 70, 90, 0.18)', stroke: '#3c465a' },
};

const ARCHITECTURAL_ROOM_TYPES = [
  'Living Room',
  'Bedroom',
  'Kitchen',
  'Bathroom',
  'Entry',
  'Office',
  'Outdoor / Balcony',
  'Storage / Closet'
];

function getRoomStyle(type) {
  for (const key of Object.keys(ROOM_COLORS)) {
    if (type && type.toLowerCase().includes(key.toLowerCase())) {
      return ROOM_COLORS[key];
    }
  }
  return ROOM_COLORS['Default'];
}

export default function Floorplan2DCanvas({
  imageUrl,
  layout,
  frontWallId,
  onSelectFrontWall,
  onCalibrateScale,
  isCalibrating,
  setIsCalibrating,
  onUpdateLayout,
  isSplit = false,
  selectedRoom = null,
  onSelectRoom = null,
}) {
  const [showRooms, setShowRooms] = useState(true);
  const [showWalls, setShowWalls] = useState(true);
  const [showOpenings, setShowOpenings] = useState(true);
  const wallsOnly = showWalls && !showRooms && !showOpenings;
  const [hoveredWall, setHoveredWall] = useState(null);

  // Active CAD Tool: 'select' | 'add_door' | 'delete' | 'relabel'
  const [activeTool, setActiveTool] = useState('select');
  const [selectedRoomForRelabel, setSelectedRoomForRelabel] = useState(null);
  const [editFeedback, setEditFeedback] = useState(null);
  const [showNotif, setShowNotif] = useState(true);

  // Zoom & Pan interactive state (2D Blueprint Canvas)
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = React.useRef({ x: 0, y: 0 });

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom((z) => Math.max(0.4, Math.min(4, +(z + delta).toFixed(2))));
  };

  const handlePanStart = (e) => {
    // Pan with middle click (1), right click (2), Alt/Shift, or clicking background
    if (e.button === 1 || e.button === 2 || e.altKey || e.shiftKey || e.target === e.currentTarget) {
      e.preventDefault();
      setIsPanning(true);
      panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    }
  };

  const handlePanMove = (e) => {
    if (!isPanning) return;
    setPan({
      x: e.clientX - panStartRef.current.x,
      y: e.clientY - panStartRef.current.y
    });
  };

  const handlePanEnd = () => {
    if (isPanning) setIsPanning(false);
  };

  // Movable Floating Layer Dock state (for split view)
  const [dockPos, setDockPos] = useState({ x: 20, y: 18 });
  const [isDraggingDock, setIsDraggingDock] = useState(false);
  const dragStartRef = React.useRef({ mouseX: 0, mouseY: 0, posX: 0, posY: 0 });

  const handleStartDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingDock(true);
    dragStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      posX: dockPos.x,
      posY: dockPos.y,
    };
  };

  React.useEffect(() => {
    if (!isDraggingDock) return;
    const onMove = (e) => {
      const dx = e.clientX - dragStartRef.current.mouseX;
      const dy = dragStartRef.current.mouseY - e.clientY;
      setDockPos({
        x: Math.max(8, Math.min(window.innerWidth - 250, dragStartRef.current.posX + dx)),
        y: Math.max(8, Math.min(window.innerHeight - 150, dragStartRef.current.posY + dy)),
      });
    };
    const onUp = () => setIsDraggingDock(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isDraggingDock]);

  // 2-Point Calibration State
  const [calibPointA, setCalibPointA] = useState(null);
  const [calibPointB, setCalibPointB] = useState(null);
  const [wallStart, setWallStart] = useState(null);
  const [drawExterior, setDrawExterior] = useState(false);
  const [levelAnchor, setLevelAnchor] = useState(null);
  const canAlignLevels = layout?.envelope?.length === 2 && !layout.storeys?.length;
  const hasAlignedLevels = layout?.storeys?.length > 0 && layout.metadata?.source !== 'sample_reference';
  const [knownDistanceM, setKnownDistanceM] = useState('0.90');

  const canvasSize = layout?.metadata?.canvas_size || [512, 512];
  const walls = layout?.walls || [];
  const rooms = layout?.rooms || [];
  const doors = layout?.doors || [];
  const windows = layout?.windows || [];
  const scaleMperPx = layout?.metadata?.scale_meters_per_pixel || 0.03;
  const roomOverlays = React.useMemo(() => {
    const placed = [];
    return rooms.map(room => {
      const ring = (room.polygon_meters || []).map(point => point.map(value => value / scaleMperPx));
      const area = `${room.surface_m2} m²`;
      const confidence = Number.isFinite(room.confidence) ? `Model ${Math.round(room.confidence * 100)}%` : 'Model n/a';
      const width = 12 + 6.5 * Math.max(room.room_type.length, area.length, confidence.length);
      const height = 36;
      const badge = placeRoomBadge(ring, width, height, placed);
      if (badge) placed.push(badge);
      return { room, points: ring.map(point => point.join(',')).join(' '), area, confidence, width, height, badge };
    });
  }, [rooms, scaleMperPx]);

  const triggerFeedback = (msg) => {
    setEditFeedback(msg);
    setTimeout(() => setEditFeedback(null), 2500);
  };

  const handleCanvasClick = (e, clickedWall = null) => {
    if (isPanning) return;
    const svg = e.currentTarget.ownerSVGElement || e.currentTarget;
    const matrix = svg.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(e.clientX, e.clientY).matrixTransform(matrix.inverse());
    const x = Math.round(point.x), y = Math.round(point.y);

    // 1. Calibration Tool
    if (isCalibrating) {
      if (!calibPointA) {
        setCalibPointA([x, y]);
      } else if (!calibPointB) {
        setCalibPointB([x, y]);
      } else {
        setCalibPointA([x, y]);
        setCalibPointB(null);
      }
      return;
    }

    if (activeTool === 'align_levels' && onUpdateLayout && canAlignLevels) {
      const scale = layout.metadata?.scale_meters_per_pixel || .03;
      const anchor = [x * scale, y * scale];
      if (!levelAnchor) { setLevelAnchor(anchor); return; }
      try {
        onUpdateLayout(alignDrawingLevels(layout, levelAnchor, anchor));
        setActiveTool('select');
        triggerFeedback('Ground and upper floors aligned.');
      } catch (error) {
        triggerFeedback(error.message);
      }
      setLevelAnchor(null);
      return;
    }

    if (activeTool === 'draw_wall' && onUpdateLayout && layout) {
      const scale = layout.metadata?.scale_meters_per_pixel || .03;
      let end = [x * scale, y * scale];
      const anchors = walls.flatMap(wall => wall.axis_meters || wall.polygon_meters || []);
      const close = anchors.reduce((best, candidate) => {
        const distance = Math.hypot(candidate[0] - end[0], candidate[1] - end[1]);
        return distance < best.distance ? { candidate, distance } : best;
      }, { candidate: end, distance: .12 });
      end = close.candidate;
      if (!wallStart) { setWallStart(end); return; }
      if (e.shiftKey) {
        const axis = Math.abs(end[0] - wallStart[0]) >= Math.abs(end[1] - wallStart[1]) ? 1 : 0;
        end = end.map((v, i) => i === axis ? wallStart[i] : v);
      }
      const length = Math.hypot(end[0] - wallStart[0], end[1] - wallStart[1]);
      if (length < .25) { triggerFeedback('Choose an end at least 25 cm from the start.'); return; }
      const level = drawingLevelAt(layout, wallStart);
      if (layout.storeys?.length && (!level || drawingLevelAt(layout, end)?.id !== level.id)) {
        triggerFeedback('Both wall endpoints must be on the same level drawing.');
        setWallStart(null);
        return;
      }
      const thickness = drawExterior ? .24 : .18;
      const extension = end.map((value, i) => (value - wallStart[i]) / length * thickness / 2);
      const polygon = wallStrip(wallStart.map((value, i) => value - extension[i]),
        end.map((value, i) => value + extension[i]), thickness);
      const newWall = { id: `wall_manual_${Date.now()}`, level_id: level?.id, is_exterior: drawExterior,
        axis_meters: [wallStart, end], thickness_m: thickness, height_m: level?.height || layout.walls[0]?.height_m || STOREY_HEIGHT_M,
        polygon_meters: polygon, polygon_px: polygon.map(p => p.map(value => value / scale)), openings: [] };
      onUpdateLayout({ ...layout, walls: [...walls, newWall], metadata: { ...layout.metadata, num_walls: walls.length + 1 } });
      setWallStart(null);
      triggerFeedback('Wall added to the 3D house.');
      return;
    }

    // 2. Add Door Tool
    if (activeTool === 'add_door' && onUpdateLayout && layout) {
      const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
      const posM = [parseFloat((x * scaleMperPx).toFixed(2)), parseFloat((y * scaleMperPx).toFixed(2))];
      const newDoor = {
        id: `door_manual_${Date.now()}`,
        position: posM,
        position_meters: posM,
        width_m: 0.9,
        type: 'door',
        associated_wall: clickedWall?.id,
        level_id: clickedWall?.level_id || drawingLevelAt(layout, posM)?.id,
      };
      const updatedLayout = {
        ...layout,
        doors: [...(layout.doors || []), newDoor],
        metadata: {
          ...layout.metadata,
          num_doors: (layout.doors?.length || 0) + 1
        }
      };
      onUpdateLayout(updatedLayout);
      triggerFeedback(`[+] Added door at (${posM[0]}m, ${posM[1]}m)`);
      return;
    }
  };

  const handleDeleteDoor = (idx, e) => {
    e.stopPropagation();
    if (activeTool === 'delete' && onUpdateLayout && layout) {
      const updatedDoors = (layout.doors || []).filter((_, i) => i !== idx);
      onUpdateLayout({
        ...layout,
        doors: updatedDoors,
        metadata: { ...layout.metadata, num_doors: updatedDoors.length }
      });
      triggerFeedback('[-] Door deleted');
    }
  };

  const handleDeleteWindow = (idx, e) => {
    e.stopPropagation();
    if (activeTool === 'delete' && onUpdateLayout && layout) {
      const updatedWindows = (layout.windows || []).filter((_, i) => i !== idx);
      onUpdateLayout({
        ...layout,
        windows: updatedWindows,
        metadata: { ...layout.metadata, num_windows: updatedWindows.length }
      });
      triggerFeedback('[-] Window deleted');
    }
  };

  const handleWallClick = (wall, e) => {
    e.stopPropagation();
    if (isCalibrating || ['draw_wall', 'add_door', 'align_levels'].includes(activeTool)) {
      handleCanvasClick(e, wall);
      return;
    }
    if (activeTool === 'delete' && onUpdateLayout && layout) {
      const updatedWalls = layout.walls.filter((w) => w.id !== wall.id);
      onUpdateLayout({
        ...layout,
        walls: updatedWalls,
        doors: doors.filter(opening => opening.associated_wall !== wall.id),
        windows: windows.filter(opening => opening.associated_wall !== wall.id),
        metadata: { ...layout.metadata, num_walls: updatedWalls.length }
      });
      triggerFeedback(`[-] Removed wall (${wall.id})`);
      return;
    }

    if (activeTool === 'select' && wall.is_exterior && onSelectFrontWall) {
      onSelectFrontWall(wall.id);
    }
  };

  const handleRoomClick = (room, e) => {
    e.stopPropagation();
    if (isCalibrating || ['draw_wall', 'add_door', 'align_levels'].includes(activeTool)) {
      handleCanvasClick(e);
    } else if (activeTool === 'relabel') {
      setSelectedRoomForRelabel(room);
    } else if (onSelectRoom) {
      onSelectRoom(room);
    }
  };

  const handleApplyRelabel = (newType) => {
    if (!selectedRoomForRelabel || !onUpdateLayout || !layout) return;
    const updatedRooms = layout.rooms.map((r) => {
      if (r.id === selectedRoomForRelabel.id) {
        return { ...r, room_type: newType, confidence: newType === r.room_type ? r.confidence : null };
      }
      return r;
    });
    onUpdateLayout({ ...layout, rooms: updatedRooms });
    triggerFeedback(`[✓] Relabeled room to "${newType}"`);
    setSelectedRoomForRelabel(null);
  };

  const calibPixelDist =
    calibPointA && calibPointB
      ? Math.hypot(calibPointB[0] - calibPointA[0], calibPointB[1] - calibPointA[1])
      : 0;

  const calculatedScale =
    calibPixelDist > 0 && parseFloat(knownDistanceM) > 0
      ? (parseFloat(knownDistanceM) / calibPixelDist).toFixed(4)
      : null;

  const handleApplyCalibration = () => {
    if (calculatedScale && onCalibrateScale) {
      onCalibrateScale(parseFloat(calculatedScale));
      setIsCalibrating(false);
      setCalibPointA(null);
      setCalibPointB(null);
      triggerFeedback(`[✓] Scale calibrated to ${calculatedScale} m/px`);
    }
  };

  const handleCancelCalibration = () => {
    if (setIsCalibrating) setIsCalibrating(false);
    setCalibPointA(null);
    setCalibPointB(null);
  };

  // Cursor style based on active tool
  let canvasCursor = 'cursor-default';
  if (isCalibrating) canvasCursor = 'cursor-crosshair';
  else if (activeTool === 'add_door') canvasCursor = 'cursor-crosshair';
  else if (activeTool === 'draw_wall' || activeTool === 'align_levels') canvasCursor = 'cursor-crosshair';
  else if (activeTool === 'delete') canvasCursor = 'cursor-pointer';
  else if (activeTool === 'relabel') canvasCursor = 'cursor-pointer';

  const layers = [
    ['Rooms', showRooms, () => setShowRooms(value => !value)],
    ['Walls', showWalls, () => setShowWalls(value => !value)],
    ['Openings', showOpenings, () => setShowOpenings(value => !value)],
    ['Walls only', wallsOnly, () => {
      setShowWalls(true);
      setShowRooms(wallsOnly);
      setShowOpenings(wallsOnly);
    }],
  ];
  const layerButtons = layers.map(([label, enabled, toggle]) => (
    <button
      key={label}
      type="button"
      onClick={toggle}
      aria-pressed={enabled}
      className={`${isSplit ? 'px-2.5 py-1' : 'h-7 px-3'} rounded-full border transition-all whitespace-nowrap shrink-0 ${
        enabled ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
          : 'bg-parchment-card/60 text-[#ece6da]/50 border-[#ece6da]/20 hover:text-[#ece6da]'
      }`}
    >
      {label}
    </button>
  ));

  return (
    <div className="plan-panel viewport-panel flex-1 flex flex-col h-full bg-[#0b0e12] relative overflow-hidden border-r border-[#ece6da]/15 drafting-grid" aria-label="2D floorplan editor">
      {/* Top Drafting Toolbar */}
      <div className="plan-toolbar viewport-toolbar h-12 border-b border-[#ece6da]/15 bg-[#11151b]/90 backdrop-blur-md px-3 sm:px-4 flex items-center justify-between z-10 select-none overflow-x-auto">
        {/* Left: Title & Elements count */}
        <div className="viewport-title flex items-center space-x-2.5 text-[#ece6da] shrink-0 mr-3">
          <span className="font-stencil text-xs uppercase tracking-wider text-[#ece6da] whitespace-nowrap">
            2D floorplan
          </span>
          <span className="text-[#ece6da]/20">|</span>
          <span className="font-mono text-[9.5px] text-[#ece6da]/70 uppercase tracking-wider whitespace-nowrap">
            {walls.length}W · {rooms.length}R · {doors.length}D {!isSplit && `· ${windows.length}WIN`}
          </span>
        </div>

        {/* Center: CAD Interactive Edit Tools */}
        <div className="plan-tools flex items-center bg-parchment-card/80 border border-[#ece6da]/20 rounded-full p-1 space-x-1 shadow-sm shrink-0">
          <button
            onClick={() => { setActiveTool('select'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Inspect / Select Front Wall"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'select' && !isCalibrating
                ? 'bg-[#ece6da] text-[#0b0e12] shadow-sm'
                : 'text-[#ece6da]/70 hover:text-[#ece6da] hover:bg-white/5'
            }`}
          >
            <MousePointer className="w-3.5 h-3.5 shrink-0" />
            <span>Select</span>
          </button>

          <button
            onClick={() => { setActiveTool('draw_wall'); setWallStart(null); setIsCalibrating?.(false); }}
            title="Draw a wall using two points"
            aria-pressed={activeTool === 'draw_wall'}
            className={`h-7 px-2.5 rounded-full flex items-center gap-1.5 ${activeTool === 'draw_wall' ? 'bg-[#ece6da] text-[#0b0e12]' : ''}`}
          ><PlusCircle className="w-3.5 h-3.5" />Wall</button>
          {activeTool === 'draw_wall' && <button aria-pressed={drawExterior} title="Toggle new wall between interior and exterior"
            className="h-7 px-2 border rounded" onClick={() => setDrawExterior(!drawExterior)}>{drawExterior ? 'Exterior' : 'Interior'}</button>}
          {canAlignLevels && <button aria-pressed={activeTool === 'align_levels'} title="Align two storey drawings"
            className="h-7 px-2 border rounded" onClick={() => { setActiveTool('align_levels'); setLevelAnchor(null); setIsCalibrating?.(false); }}>Align levels</button>}
          {hasAlignedLevels && <button title="Return to separate plan drawings" className="h-7 px-2 border rounded"
            onClick={() => onUpdateLayout(separateDrawingLevels(layout))}>Separate levels</button>}
          <button
            onClick={() => { setActiveTool('add_door'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Click on canvas or wall to insert a door"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'add_door'
                ? 'bg-[#e8a33d] text-[#0b0e12] shadow-sm'
                : 'text-[#ece6da]/70 hover:text-[#ece6da] hover:bg-white/5'
            }`}
          >
            <PlusCircle className="w-3.5 h-3.5 shrink-0" />
            <span>Door</span>
          </button>

          <button
            onClick={() => { setActiveTool('delete'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Click doors, windows, or internal walls to delete them"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'delete'
                ? 'bg-rose-700 text-white shadow-sm'
                : 'text-[#ece6da]/70 hover:text-[#ece6da] hover:bg-white/5'
            }`}
          >
            <Trash2 className="w-3.5 h-3.5 shrink-0" />
            <span>Delete</span>
          </button>

          <button
            onClick={() => { setActiveTool('relabel'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Click any room polygon or badge to change space type"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'relabel'
                ? 'bg-[#c48d3f] text-white shadow-sm'
                : 'text-[#ece6da]/70 hover:text-[#ece6da] hover:bg-white/5'
            }`}
          >
            <Tag className="w-3.5 h-3.5 shrink-0" />
            <span>Label</span>
          </button>

          <button
            onClick={() => {
              if (setIsCalibrating) setIsCalibrating(!isCalibrating);
              setActiveTool('select');
              setCalibPointA(null);
              setCalibPointB(null);
            }}
            title="2-Point Real-world Scale Calibration"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              isCalibrating
                ? 'bg-[#e8a33d] text-[#0b0e12] animate-pulse'
                : 'text-[#ece6da]/70 hover:text-[#ece6da] hover:bg-white/5'
            }`}
          >
            <Ruler className="w-3.5 h-3.5 shrink-0" />
            <span>Scale</span>
          </button>

          {/* Zoom Controls */}
          <div className="flex items-center space-x-1 pl-1 border-l border-[#ece6da]/15 ml-1">
            <button
              onClick={() => setZoom((z) => Math.min(4, +(z + 0.25).toFixed(2)))}
              className="h-7 w-7 rounded-full flex items-center justify-center text-[#ece6da]/70 hover:text-[#ece6da] hover:bg-white/5"
              title="Zoom In (+)"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <span className="font-mono text-[9.5px] w-9 text-center select-none text-[#ece6da]/70">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom((z) => Math.max(0.4, +(z - 0.25).toFixed(2)))}
              className="h-7 w-7 rounded-full flex items-center justify-center text-[#ece6da]/70 hover:text-[#ece6da] hover:bg-white/5"
              title="Zoom Out (-)"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
              className="h-7 px-2 rounded-full font-mono text-[9px] uppercase tracking-wider text-[#ece6da]/60 hover:text-[#ece6da] hover:bg-white/5"
              title="Fit to Screen"
            >
              FIT
            </button>
          </div>
        </div>

        {/* Right: In 2D-only full view (!isSplit), keep layer toggles in the top bar! */}
        {!isSplit && (
          <div className="flex items-center space-x-1 font-stencil text-[10px] uppercase tracking-wider shrink-0">
            {layerButtons}
          </div>
        )}
      </div>

      {/* Main Canvas Area */}
      <div
        className="flex-1 flex items-center justify-center p-6 overflow-hidden relative select-none"
        onWheel={handleWheel}
        onMouseDown={handlePanStart}
        onMouseMove={handlePanMove}
        onMouseUp={handlePanEnd}
        onMouseLeave={handlePanEnd}
        onContextMenu={(e) => { e.preventDefault(); }}
        style={{ cursor: isPanning ? 'grabbing' : undefined }}
      >
        {/* Movable Floating Layer Visibility Dock (Only in 2D+3D Split Mode) */}
        {isSplit && (
          <div
            style={{ left: `${dockPos.x}px`, bottom: `${dockPos.y}px` }}
            className="absolute z-20 flex items-center bg-[#11151b]/95 backdrop-blur-md border border-[#ece6da]/20 rounded-full px-2.5 py-1.5 shadow-xl select-none"
          >
            <div
              onMouseDown={handleStartDrag}
              className="flex items-center space-x-1 text-[#ece6da]/40 font-mono text-[9px] uppercase tracking-wider pr-2 border-r border-[#ece6da]/15 mr-1.5 cursor-grab active:cursor-grabbing hover:text-[#ece6da] shrink-0"
              title="Click and drag to move layer dock anywhere"
            >
              <GripVertical className="w-3.5 h-3.5 text-[#e8a33d]" />
              <span className="font-semibold hidden sm:inline">LAYERS</span>
            </div>
            <div className="flex items-center space-x-1 font-stencil text-[9.5px] uppercase tracking-wider">
              {layerButtons}
            </div>
          </div>
        )}
        {/* Real-time Feedback Banner */}
        {editFeedback && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-[#ece6da] text-[#0b0e12] font-mono text-[10.5px] uppercase tracking-wider rounded-full shadow-xl z-30 flex items-center space-x-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>{editFeedback}</span>
          </div>
        )}

        <div
          className={`relative rounded-xl overflow-hidden shadow-2xl border border-[#ece6da]/25 bg-white ${canvasCursor} ${
            isCalibrating ? 'ring-2 ring-[#d4943a]' : activeTool !== 'select' ? 'ring-1 ring-[#ece6da]/40' : ''
          }`}
          style={{
            width: `${canvasSize[0]}px`,
            height: `${canvasSize[1]}px`,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isPanning ? 'none' : 'transform 0.12s ease-out'
          }}
        >
          {/* Base Image */}
          {imageUrl && (!wallsOnly || isCalibrating) && (
            <img
              src={imageUrl}
              alt="Floorplan"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none opacity-60"
            />
          )}

          {/* SVG Vector Overlays */}
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox={`0 0 ${canvasSize[0]} ${canvasSize[1]}`}
            onClick={handleCanvasClick}
          >
            {/* 1. Room Polygons */}
            {showRooms &&
              roomOverlays.map(({ room, points, area, confidence }) => {
                const style = getRoomStyle(room.room_type);
                const isSelected = selectedRoom?.id === room.id;
                return (
                  <polygon
                    key={room.id}
                    data-room-id={room.id}
                    points={points}
                    fill={isSelected ? 'rgba(2, 132, 199, 0.28)' : style.fill}
                    stroke={isSelected ? '#0284c7' : style.stroke}
                    strokeWidth={isSelected ? '2.5' : '1.5'}
                    className={`room-polygon transition-all cursor-pointer ${
                      activeTool === 'relabel' ? 'hover:stroke-purple-400 hover:fill-purple-500/30' : 'hover:stroke-sky-500'
                    }`}
                    onClick={(e) => handleRoomClick(room, e)}
                  >
                    <title>{`${room.room_type} · ${area} · ${confidence}`}</title>
                  </polygon>
                );
              })}

            {/* 2. Walls */}
            {activeTool === 'draw_wall' && wallStart && <circle
              cx={wallStart[0] / layout.metadata.scale_meters_per_pixel}
              cy={wallStart[1] / layout.metadata.scale_meters_per_pixel}
              r="4" fill="#c94b25" stroke="white" strokeWidth="2" pointerEvents="none" />}
            {activeTool === 'align_levels' && levelAnchor && <circle
              cx={levelAnchor[0] / layout.metadata.scale_meters_per_pixel}
              cy={levelAnchor[1] / layout.metadata.scale_meters_per_pixel}
              r="4" fill="#287ab1" stroke="white" strokeWidth="2" pointerEvents="none" />}
            {showWalls &&
              walls.map((wall) => {
                const isSelected = wall.id === frontWallId;
                const isExterior = wall.is_exterior;

                const pts = wall.polygon_px?.length ? wall.polygon_px
                  : (wall.polygon_meters || []).map(point => point.map(value => value / scaleMperPx));
                const pointsStr = pts.map((p) => `${p[0]},${p[1]}`).join(' ');

                let strokeColor = '#1d1d1f'; // Interior drafting ink
                let fillColor = 'rgba(29, 29, 31, 0.20)';

                if (isExterior) {
                  strokeColor = '#b8452a'; // Exterior terracotta
                  fillColor = 'rgba(184, 69, 42, 0.35)';
                }
                if (isSelected) {
                  strokeColor = '#c48d3f'; // Front wall architectural amber
                  fillColor = 'rgba(196, 141, 63, 0.50)';
                }

                if (activeTool === 'delete') {
                  strokeColor = '#f43f5e';
                  fillColor = 'rgba(244, 63, 94, 0.25)';
                }

                return (
                  <polygon
                    key={wall.id}
                    data-wall-id={wall.id}
                    points={pointsStr}
                    fill={wallsOnly ? strokeColor : fillColor}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? '3' : isExterior ? '2' : '1.5'}
                    className={`cursor-pointer transition-all ${
                      activeTool === 'delete' ? 'hover:stroke-rose-400 hover:fill-rose-500/40' : isExterior ? 'hover:stroke-amber-400' : ''
                    }`}
                    onMouseEnter={() => setHoveredWall(wall.id)}
                    onMouseLeave={() => setHoveredWall(null)}
                    onClick={(e) => handleWallClick(wall, e)}
                  />
                );
              })}

            {/* 3. Doors Markers */}
            {showOpenings &&
              doors.map((door, idx) => {
                const p = door.position_meters || [0, 0];
                const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
                const cx = p[0] / scaleMperPx;
                const cy = p[1] / scaleMperPx;
                return (
                  <circle
                    key={`door-${idx}`}
                    data-opening-type="door"
                    cx={cx}
                    cy={cy}
                    r="5"
                    fill={activeTool === 'delete' ? '#ef4444' : '#10b981'}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                    className={`cursor-pointer transition-transform hover:scale-125`}
                    onClick={(e) => handleDeleteDoor(idx, e)}
                  />
                );
              })}

            {/* 4. Windows Markers */}
            {showOpenings &&
              windows.map((win, idx) => {
                const p = win.position_meters || [0, 0];
                const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
                const cx = p[0] / scaleMperPx;
                const cy = p[1] / scaleMperPx;
                return (
                  <rect
                    key={`win-${idx}`}
                    data-opening-type="window"
                    x={cx - 4}
                    y={cy - 4}
                    width="8"
                    height="8"
                    fill={activeTool === 'delete' ? '#ef4444' : '#0ea5e9'}
                    stroke="#ffffff"
                    strokeWidth="1"
                    rx="1"
                    className={`cursor-pointer transition-transform hover:scale-125`}
                    onClick={(e) => handleDeleteWindow(idx, e)}
                  />
                );
              })}

            {/* 5. Measurement / Calibration Line */}
            {isCalibrating && (
              <g>
                {calibPointA && calibPointB && (
                  <>
                    <line
                      x1={calibPointA[0]}
                      y1={calibPointA[1]}
                      x2={calibPointB[0]}
                      y2={calibPointB[1]}
                      stroke="#f59e0b"
                      strokeWidth="2.5"
                      strokeDasharray="6,4"
                    />
                    <text
                      x={(calibPointA[0] + calibPointB[0]) / 2}
                      y={(calibPointA[1] + calibPointB[1]) / 2 - 8}
                      fill="#f59e0b"
                      fontSize="11"
                      fontWeight="bold"
                      textAnchor="middle"
                    >
                      {Math.round(calibPixelDist)} px
                    </text>
                  </>
                )}
                {calibPointA && (
                  <g transform={`translate(${calibPointA[0]}, ${calibPointA[1]})`}>
                    <circle r="7" fill="#f59e0b" stroke="#ffffff" strokeWidth="2" />
                    <text x="10" y="4" fill="#f59e0b" fontSize="10" fontWeight="bold">Point A</text>
                  </g>
                )}
                {calibPointB && (
                  <g transform={`translate(${calibPointB[0]}, ${calibPointB[1]})`}>
                    <circle r="7" fill="#10b981" stroke="#ffffff" strokeWidth="2" />
                    <text x="10" y="4" fill="#10b981" fontSize="10" fontWeight="bold">Point B</text>
                  </g>
                )}
              </g>
            )}
            {showRooms && roomOverlays.map(({ room, area, confidence, width, height, badge }) => badge && (
              <g
                key={room.id}
                data-room-id={room.id}
                transform={`translate(${badge.x}, ${badge.y}) scale(${badge.scale})`}
                className="room-centroid-badge cursor-pointer select-none"
                onClick={(e) => handleRoomClick(room, e)}
              >
                <title>{`${room.room_type} · ${area} · ${confidence}`}</title>
                <rect
                  x={-width / 2} y={-height / 2} width={width} height={height} rx="4"
                  fill="#f7f4ed"
                  stroke={selectedRoom?.id === room.id ? '#0284c7' : getRoomStyle(room.room_type).stroke}
                  strokeWidth={selectedRoom?.id === room.id ? 2 : 1.2}
                />
                <text textAnchor="middle" fontFamily="monospace" fontSize="9" fill="#1d1d1f">
                  <tspan x="0" y="-6" fontSize="10" fontWeight="700">{room.room_type}</tspan>
                  <tspan x="0" y="4" fill="#96351e">{area}</tspan>
                  <tspan x="0" y="14" fill="#3c465a">{confidence}</tspan>
                </text>
              </g>
            ))}
          </svg>

          {/* Unresolved Rooms Domain Warning */}
          {layout && rooms.length === 0 && (
            <div className="absolute inset-x-6 top-6 bg-slate-950/95 border border-amber-500/50 rounded-xl p-3 text-center pointer-events-none shadow-2xl backdrop-blur-sm z-20">
              <p className="text-xs font-semibold text-amber-400">No Rooms Detected (Plan style may not be supported)</p>
              <p className="text-[10px] text-slate-400 mt-0.5">The neural network could not close topological loops around rooms on this drafting format.</p>
            </div>
          )}
        </div>

        {/* Room Relabeling Modal Popover */}
        {selectedRoomForRelabel && (
          <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center z-40 p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 max-w-sm w-full shadow-2xl">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <Tag className="w-4 h-4 text-purple-400" />
                  <span className="font-semibold text-sm text-slate-100">Relabel Space</span>
                </div>
                <button
                  onClick={() => setSelectedRoomForRelabel(null)}
                  className="text-slate-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <p className="text-xs text-slate-400 mt-2 mb-4">
                Current type: <span className="text-purple-300 font-semibold">{selectedRoomForRelabel.room_type}</span> ({selectedRoomForRelabel.surface_m2} m²). Select new architectural classification:
              </p>

              <div className="grid grid-cols-2 gap-2">
                {ARCHITECTURAL_ROOM_TYPES.map((type) => (
                  <button
                    key={type}
                    onClick={() => handleApplyRelabel(type)}
                    className={`px-3 py-2 rounded-lg text-xs font-medium text-left transition-colors border ${
                      selectedRoomForRelabel.room_type === type
                        ? 'bg-purple-500/20 border-purple-500/60 text-purple-200 font-semibold'
                        : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Floating Calibration Controls Card */}
        {/* 2-Point Calibration Tool Box */}
        {isCalibrating && (
          <div className="absolute top-4 right-6 bg-[#11151b]/95 backdrop-blur-md border border-[#d4943a]/40 rounded-xl p-4 text-xs text-[#ece6da] shadow-xl w-80 z-30 font-sans">
            <div className="flex items-center justify-between font-semibold text-[#ece6da] mb-2 font-mono uppercase tracking-wider text-[11px]">
              <div className="flex items-center space-x-1.5">
                <Ruler className="w-4 h-4 text-[#d4943a]" />
                <span className="text-[#d4943a]">2-Point Scale Calibration</span>
              </div>
              <button onClick={handleCancelCalibration} className="text-[#ece6da]/50 hover:text-[#ece6da]">
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <p className="text-[11px] text-[#ece6da]/70 mb-3 leading-relaxed">
              {!calibPointA
                ? "Click Point 1 on the canvas (e.g. one side of a doorway or exterior wall)."
                : !calibPointB
                ? "Click Point 2 (the other side of the doorway or wall)."
                : `Distance between points: ${Math.round(calibPixelDist)} pixels.`}
            </p>

            {calibPointA && calibPointB && (
              <div className="space-y-3 pt-2 border-t border-[#ece6da]/15 font-mono">
                <div>
                  <label className="text-[9.5px] uppercase font-semibold text-[#ece6da]/70 block mb-1">
                    Known Distance in Real World (Meters)
                  </label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="number"
                      step="0.05"
                      min="0.1"
                      max="30"
                      value={knownDistanceM}
                      onChange={(e) => setKnownDistanceM(e.target.value)}
                      className="bg-parchment-card border border-[#ece6da]/30 rounded px-2.5 py-1 text-[#ece6da] font-mono text-sm w-28 focus:border-[#d4943a] focus:outline-none"
                    />
                    <span className="text-[#ece6da]/60 text-xs">meters</span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  <button
                    onClick={() => setKnownDistanceM('0.90')}
                    className="text-[10px] px-2 py-0.5 rounded-full bg-parchment-card hover:bg-[#ece6da] hover:text-[#0b0e12] text-[#ece6da] border border-[#ece6da]/20"
                  >
                    Door (0.9m)
                  </button>
                  <button
                    onClick={() => setKnownDistanceM('1.80')}
                    className="text-[10px] px-2 py-0.5 rounded-full bg-parchment-card hover:bg-[#ece6da] hover:text-[#0b0e12] text-[#ece6da] border border-[#ece6da]/20"
                  >
                    Double Door (1.8m)
                  </button>
                  <button
                    onClick={() => setKnownDistanceM('3.50')}
                    className="text-[10px] px-2 py-0.5 rounded-full bg-parchment-card hover:bg-[#ece6da] hover:text-[#0b0e12] text-[#ece6da] border border-[#ece6da]/20"
                  >
                    Wall (3.5m)
                  </button>
                </div>

                <div className="bg-[#0b0e12] p-2.5 rounded border border-[#ece6da]/15 flex justify-between items-center text-xs">
                  <span className="text-[#ece6da]/70 uppercase text-[10px]">Calibrated Ratio:</span>
                  <span className="font-mono font-bold text-[#d4943a]">{calculatedScale} m/px</span>
                </div>

                <div className="flex space-x-2 pt-1">
                  <button
                    onClick={handleApplyCalibration}
                    className="flex-1 bg-[#d4943a] hover:bg-[#a8711f] text-[#0b0e12] font-mono uppercase tracking-wider text-[10px] font-semibold py-1.5 px-3 rounded-full flex items-center justify-center space-x-1.5 transition-all shadow-sm"
                  >
                    <Check className="w-3.5 h-3.5" />
                    <span>Apply Calibration</span>
                  </button>
                  <button
                    onClick={() => { setCalibPointA(null); setCalibPointB(null); }}
                    className="bg-parchment-card hover:bg-[#ece6da]/10 text-[#ece6da] border border-[#ece6da]/20 font-mono uppercase tracking-wider text-[10px] px-3 py-1.5 rounded-full transition-all"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Helper Box / Drafting Tip */}
        {showNotif ? (
          <div
            className={`absolute z-10 bg-[#11151b]/95 backdrop-blur-md border border-[#ece6da]/20 rounded-xl p-3 text-xs text-[#ece6da] shadow-lg max-w-sm font-sans transition-all ${
              isSplit ? 'bottom-16 left-5' : 'bottom-4 left-6'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center space-x-1.5 font-mono uppercase tracking-wider text-[10px] font-semibold text-[#ece6da]">
                <Info className="w-3.5 h-3.5 text-[#e8a33d]" />
                <span>INTERACTIVE DRAFTING DESK</span>
              </div>
              <button
                onClick={() => setShowNotif(false)}
                className="text-[#ece6da]/40 hover:text-[#ece6da] text-sm leading-none font-bold px-1 ml-2 cursor-pointer"
                title="Dismiss tip"
              >
                ×
              </button>
            </div>
            <p className="font-mono text-[9.5px] text-[#ece6da]/70 leading-relaxed">
              {activeTool === 'select' && 'Select an exterior wall to set the facade target.'}
              {activeTool === 'draw_wall' && (wallStart ? 'Click the end of the wall. Hold Shift to keep it horizontal or vertical.' : 'Click the start of the wall. Nearby junctions snap into place.')}
              {activeTool === 'align_levels' && (levelAnchor ? 'Click the matching corner on the upper drawing.' : 'Click a corner on the ground-floor drawing.')}
              {activeTool === 'add_door' && "Click anywhere on the plan to insert a door. The Three.js 3D model re-extrudes instantly."}
              {activeTool === 'delete' && "Click any door, window, or wall to delete it from the architectural model."}
              {activeTool === 'relabel' && "Click any room polygon or badge to change its space classification."}
            </p>
          </div>
        ) : (
          <button
            onClick={() => setShowNotif(true)}
            className="absolute bottom-4 right-4 z-10 p-2 rounded-full bg-[#11151b]/95 border border-[#ece6da]/20 shadow-md hover:bg-parchment-card text-[#ece6da]/70 hover:text-[#e8a33d] transition-all cursor-pointer"
            title="Show drafting tips"
          >
            <HelpCircle className="w-4 h-4 text-[#e8a33d]" />
          </button>
        )}
      </div>
    </div>
  );
}
