import React, { useEffect, useState } from 'react';
import { Sparkles, RefreshCw, Trash2, X, AlertCircle, AlertTriangle, CheckCircle2, ChevronRight, Info, Layers, Compass, Eye } from 'lucide-react';
import { FURNITURE_FINISHES } from '../geometry/furnitureAppearance.js';

const PRESET_OPTIONS = Object.entries(FURNITURE_FINISHES);

const ROOM_TYPE_OPTIONS = [
  'Bedroom',
  'Living Room',
  'Kitchen',
  'Bathroom',
  'Office',
  'Dining',
  'Entry / Hall',
  'Storage',
  'Utility',
  'Balcony / Terrace'
];

// Minimum plausible physical room areas (in m²) for residential spaces
const MIN_PLAUSIBLE_ROOM_AREAS = {
  bathroom: 2.0,
  bedroom: 5.0,
  kitchen: 3.0,
  'living room': 6.0,
  living_room: 6.0
};

export default function RoomDesignPanel({
  selectedRoom,
  onClose,
  onGenerateDesign,
  onRegenerateDesign,
  onClearDesign,
  onInspectRoom,
  onSelectFurniture,
  onPresetChange,
  furnishingData,
  isDesigning,
  onRelabelRoom
}) {
  const [selectedPreset, setSelectedPreset] = useState('contemporary_minimal');
  const [showRelabel, setShowRelabel] = useState(false);
  const appliedPreset = furnishingData?.[selectedRoom?.id]?.preset;
  useEffect(() => { setSelectedPreset(appliedPreset || 'contemporary_minimal'); }, [selectedRoom?.id, appliedPreset]);

  if (!selectedRoom) return null;

  const currentFurnishings = furnishingData?.[selectedRoom.id] || null;
  const items = currentFurnishings?.items || [];
  const validation = currentFurnishings?.validation || null;
  const failureReason = currentFurnishings?.failure_reason || null;
  const hasFailed = currentFurnishings && !currentFurnishings.is_valid;
  const isFurnished = currentFurnishings && currentFurnishings.is_valid && items.length > 0;
  const meta = currentFurnishings?.metadata || {};
  const actualScale = meta.scale_m_per_px || 0.03;
  const scaleSource = meta.scale_source || (meta.scale_uncalibrated ? 'Model Estimated' : '2-Pt Calibrated');

  // Fragment detection (< minimum plausible room area)
  const normType = (selectedRoom.room_type || '').toLowerCase();
  const minPlausibleArea = MIN_PLAUSIBLE_ROOM_AREAS[normType] || 1.5;
  const isSmallRoom = selectedRoom.surface_m2 !== undefined && selectedRoom.surface_m2 < minPlausibleArea;
  const needsCompact = isSmallRoom || meta.compact_mode === true;
  const isFragment = !isFurnished && (isSmallRoom || failureReason === 'ROOM_FRAGMENT' ||
                     currentFurnishings?.status === 'not a room (fragment)');
  const noticeReason = hasFailed ? failureReason : 'ROOM_FRAGMENT';

  // Approximate room width & depth from bounding box
  const poly = selectedRoom.polygon_meters || [];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  poly.forEach(([x, y]) => {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  const widthM = (maxX - minX > 0 && maxX !== Infinity) ? (maxX - minX).toFixed(1) : '-';
  const depthM = (maxY - minY > 0 && maxY !== Infinity) ? (maxY - minY).toFixed(1) : '-';

  const getFailureMessage = (reason) => {
    switch (reason) {
      case 'ROOM_FRAGMENT':
        return {
          title: 'Compact Room',
          desc: `This ${selectedRoom.surface_m2} m² space is below the ${minPlausibleArea} m² threshold for standard ${selectedRoom.room_type} furnishings.`,
          action: 'Check the plan scale, or try compact placement with smaller furniture.'
        };
      case 'REQUEST_FAILED':
        return {
          title: 'Room Design Service Unavailable',
          desc: 'The placement request could not be completed.',
          action: 'Try generating the layout again.'
        };
      case 'NO_USABLE_WALL':
        return {
          title: 'No clear wall fits the furniture',
          desc: 'The planner could not fit a suitable item along the available wall segments.',
          action: 'Check the room boundary and detected door and window locations.'
        };
      case 'UNSUPPORTED_ROOM_TYPE':
        return {
          title: 'No furniture set for this room type',
          desc: 'The catalog has no placement rules for this type of room.',
          action: 'Relabel the room if the floorplan identified it incorrectly.'
        };
      case 'TOO_SMALL':
        return {
          title: 'Room Space Too Compact',
          desc: `Area (${selectedRoom.surface_m2} m²) is too compact for standard ${selectedRoom.room_type} furnishings.`,
          action: 'Consider relabeling as a single bedroom or adjusting room boundary.'
        };
      case 'DOOR_BLOCKED':
        return {
          title: 'Door Entry Swing Compromised',
          desc: 'Furniture would obstruct the required 0.85m door swing clearance envelope.',
          action: 'Relocate the door opening in the 2D editor or select a smaller preset.'
        };
      case 'UNREACHABLE':
        return {
          title: 'Walking Corridor Blocked',
          desc: 'The furniture leaves no connected walking route from the entry to the items.',
          action: 'Adjust room boundaries in 2D to provide wider corridor clearance.'
        };
      default:
        return {
          title: 'Placement Verification Failed',
          desc: 'The planner could not keep the furniture within the room while preserving access.',
          action: 'Try another preset or use the 2D editor to refine room shape.'
        };
    }
  };

  return (
    <div className="room-design-panel absolute top-4 right-4 z-30 w-84 bg-[#0b0e12]/95 backdrop-blur-md border border-[#ece6da]/30 shadow-2xl rounded-sm font-mono text-xs overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-[#ece6da] text-[#0b0e12] px-3 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-[#d4943a]" />
          <span className="font-semibold tracking-wider uppercase text-[11px]">Room Design Engine</span>
        </div>
        <button
          onClick={onClose}
          className="text-[#0b0e12]/60 hover:text-[#0b0e12] transition-colors p-0.5"
          title="Close panel"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="p-3.5 space-y-3 max-h-[80vh] overflow-y-auto">
        {/* Room Info Card */}
        <div className="bg-[#ece6da]/5 border border-[#ece6da]/15 p-2.5 rounded-sm space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-[#ece6da] text-sm">
                {selectedRoom.room_type || 'Selected Space'}
              </span>
              {isFragment && (
                <span className="px-1.5 py-0.5 bg-[#d4943a]/15 text-[#d4943a] text-[9.5px] font-bold uppercase rounded-xs">
                  compact
                </span>
              )}
            </div>
            <button
              onClick={() => setShowRelabel(!showRelabel)}
              className="text-[10px] text-[#d4943a] underline uppercase hover:text-[#a8711f]"
            >
              {showRelabel ? 'Cancel' : 'Relabel'}
            </button>
          </div>

          {showRelabel && (
            <div className="grid grid-cols-2 gap-1 pt-1.5 border-t border-[#ece6da]/10">
              {ROOM_TYPE_OPTIONS.map((type) => (
                <button
                  key={type}
                  onClick={() => {
                    if (onRelabelRoom) onRelabelRoom(selectedRoom.id, type);
                    setShowRelabel(false);
                  }}
                  className={`px-1.5 py-1 text-[10px] text-left border rounded-sm transition-colors ${
                    selectedRoom.room_type === type
                      ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                      : 'bg-parchment-card/80 border-[#ece6da]/20 hover:border-[#d4943a]'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] text-[#ece6da]/70">
            <div>Area: <span className="font-semibold text-[#ece6da]">{selectedRoom.surface_m2} m²</span></div>
            <div>Span: <span className="font-semibold text-[#ece6da]">{widthM}m × {depthM}m</span></div>
          </div>
        </div>

        {/* Preset Selector */}
        <div className="space-y-1.5">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-[#ece6da]/70 flex items-center gap-1">
            <Compass className="w-3 h-3 text-[#d4943a]" /> Style Preset
          </label>
          <div className="space-y-1">
            {PRESET_OPTIONS.map(([id, opt]) => (
              <label
                key={id}
                className={`flex items-start gap-2 p-2 border rounded-sm cursor-pointer transition-all ${
                  selectedPreset === id
                    ? 'bg-[#ece6da]/10 border-[#ece6da] shadow-sm'
                    : 'bg-parchment-card/60 border-[#ece6da]/15 hover:bg-parchment-card/90'
                }`}
              >
                <input
                  type="radio"
                  name="preset"
                  value={id}
                  checked={selectedPreset === id}
                  disabled={isDesigning}
                  onChange={() => {
                    setSelectedPreset(id);
                    if (isFurnished) onPresetChange(selectedRoom.id, id);
                  }}
                  className="mt-0.5 accent-[#d4943a]"
                />
                <div className="flex-1 leading-tight">
                  <div className="font-medium text-[11px] text-[#ece6da]">{opt.label}</div>
                  <div className="text-[9px] text-[#ece6da]/60 mt-0.5">{opt.desc}</div>
                </div>
                <span className="mt-1 flex gap-0.5" aria-hidden="true">
                  {[opt.wood, opt.fabric, opt.metal].map(color => <i key={color} style={{ background: color }} className="h-2.5 w-2.5 rounded-full border border-black/10" />)}
                </span>
              </label>
            ))}
          </div>
          {isFurnished && <p className="text-[9.5px] text-[#ece6da]/65">Presets update finishes immediately. Generate a design to use the preset’s model choices.</p>}
        </div>

        {/* Failure / Fragment Notice / Guidance */}
        {(hasFailed || isFragment) && (
          <div className="space-y-2">
            <div className="bg-[#d4943a]/10 border border-[#d4943a]/30 p-2.5 rounded-sm text-[#d4943a] space-y-1">
              <div className="flex items-center gap-1.5 font-semibold text-[11px]">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                <span>{getFailureMessage(noticeReason).title}</span>
              </div>
              <p className="text-[10px] leading-relaxed text-[#ece6da]/80">
                {getFailureMessage(noticeReason).desc}
              </p>
              <div className="pt-1 border-t border-[#d4943a]/20 text-[10px] font-medium text-[#d4943a]">
                Tip: {getFailureMessage(noticeReason).action}
              </div>
            </div>

            {/* Scale & Calibration Status Badge (shows actual scale & source) */}
            {failureReason === 'TOO_SMALL' && !isFragment && (
              <div className="bg-amber-500/10 border border-amber-600/30 p-2 rounded-sm text-amber-900 space-y-1">
                <div className="flex items-center gap-1.5 font-semibold text-[10.5px] text-amber-800">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600 flex-shrink-0" />
                  <span>Scale Calibration Notice</span>
                </div>
                <div className="text-[10px] space-y-0.5 text-amber-900/90 font-mono">
                  <div>Actual Scale: <strong className="font-semibold">{actualScale} m/px</strong></div>
                  <div>Source: <strong className="font-semibold">{scaleSource}</strong></div>
                </div>
                <p className="text-[9.5px] leading-relaxed text-amber-900/80 pt-0.5 border-t border-amber-600/15 font-sans">
                  Use the <strong>2-Pt Calibrate</strong> tool in the top bar to set precise physical scale from a known doorway or wall.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Placed Items Summary */}
        {isFurnished && (
          <div className="bg-[#2a7a6e]/10 border border-[#2a7a6e]/30 p-2.5 rounded-sm space-y-1.5">
            <div className="flex items-center justify-between text-[#2a7a6e] font-semibold text-[11px]">
              <div className="flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Clear doors and walking routes</span>
              </div>
              <span className="text-[10px]">{items.length} items</span>
            </div>
            {meta.layout_options > 1 && <p className="text-[10px]">{meta.layout_options} validated layouts found</p>}
            {items.some(item => item.scale?.[0] < .99) && <p className="text-[10px] text-[#2a7a6e]">Furniture sized to fit this room</p>}
            <p className="text-[10px] text-[#ece6da]/70">Select a piece here or in 3D to open its catalog.</p>
            <ul className="space-y-1 overflow-y-auto text-[10px] text-[#ece6da]/80 max-h-36">
              {items.map(item => (
                <li key={item.item_id}>
                  <button type="button" disabled={isDesigning} onClick={() => onSelectFurniture(selectedRoom.id, item.item_id)}
                    className="flex w-full items-center justify-between rounded border border-[#2a7a6e]/20 bg-parchment-card/70 px-2 py-1 text-left hover:border-[#2a7a6e]">
                    <span><strong className="block">{item.name}</strong>{item.dimensions_m[0].toFixed(2)} m wide</span>
                    <ChevronRight size={14} />
                  </button>
                </li>
              ))}
            </ul>
            <button type="button" onClick={onInspectRoom} className="flex items-center gap-1 font-semibold text-[#2a7a6e] underline">
              <Eye size={14} /> Inspect in 3D
            </button>
          </div>
        )}

        {/* Action Controls */}
        <div className="space-y-1.5 pt-1">
          {needsCompact ? (
            <div className="space-y-1">
              <button
                onClick={() => onGenerateDesign(selectedRoom, selectedPreset, 42, true)}
                disabled={isDesigning}
                className="w-full bg-[#d4943a] hover:bg-[#a8711f] text-[#0b0e12] py-2 px-3 rounded-sm font-semibold text-[11px] uppercase tracking-wider flex items-center justify-center gap-1.5 transition-all shadow-md disabled:opacity-50"
              >
                {isDesigning ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Generating Compact Placement...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>{isFurnished ? 'Generate Compact Design' : 'Try Compact Placement'}</span>
                  </>
                )}
              </button>
              <div className="text-[9.5px] text-[#d4943a] text-center font-medium">
                Small area ({selectedRoom.surface_m2} m²) &bull; Compact mode
              </div>
            </div>
          ) : (
            <button
              onClick={() => onGenerateDesign(selectedRoom, selectedPreset)}
              disabled={isDesigning}
              className="w-full bg-[#ece6da] hover:bg-[#333] text-[#0b0e12] py-2 px-3 rounded-sm font-semibold text-[11px] uppercase tracking-wider flex items-center justify-center gap-1.5 transition-all shadow-md disabled:opacity-50"
            >
              {isDesigning ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Generating Placement...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5 text-[#d4943a]" />
                  <span>Generate Design</span>
                </>
              )}
            </button>
          )}

          <div className="grid grid-cols-2 gap-1.5">
            <button
              onClick={() => onRegenerateDesign(selectedRoom, selectedPreset)}
              disabled={isDesigning || !isFurnished || (meta.layout_options ?? 1) <= 1}
              className="bg-parchment-card/80 hover:bg-parchment-card text-[#ece6da] border border-[#ece6da]/25 py-1.5 px-2 rounded-sm font-medium text-[10px] uppercase tracking-wider flex items-center justify-center gap-1 transition-all disabled:opacity-40"
              title="Cycle through the validated layouts for this preset"
            >
              <RefreshCw className="w-3 h-3 text-[#ece6da]/70" />
              <span>Try another layout</span>
            </button>

            <button
              onClick={() => onClearDesign(selectedRoom.id)}
              disabled={isDesigning || !items.length}
              className="bg-parchment-card/80 hover:bg-[#d4943a]/10 text-[#d4943a] border border-[#d4943a]/30 py-1.5 px-2 rounded-sm font-medium text-[10px] uppercase tracking-wider flex items-center justify-center gap-1 transition-all disabled:opacity-40"
              title="Remove items and free WebGL memory"
            >
              <Trash2 className="w-3 h-3" />
              <span>Clear</span>
            </button>
          </div>
        </div>

        {/* Offline Rule-Based Engine Note */}
        <div className="text-[9px] text-[#ece6da]/50 text-center tracking-tight pt-1">
          Rule-based placement using the local furniture catalog
        </div>
      </div>
    </div>
  );
}
