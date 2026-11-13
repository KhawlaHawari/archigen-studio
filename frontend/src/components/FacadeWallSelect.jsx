import React from 'react';
import { wallFacadeFrame } from '../geometry/facadeGeometry.js';

export default function FacadeWallSelect({ layout, value, onChange, disabled, showMap = false }) {
  const levels = layout?.storeys || [];
  const walls = layout?.walls.filter(wall => wall.is_exterior && wallFacadeFrame(wall)) || [];
  const hasFinish = wall => Boolean(wall.facade_texture?.texture_base64);
  const selected = walls.find(wall => wall.id === value);
  const levelId = selected?.level_id ?? levels[0]?.id;
  const levelWalls = showMap ? layout?.walls.filter(wall => wall.level_id === levelId) || [] : [];
  const points = levelWalls.flatMap(wall => wall.polygon_meters || []);
  const bounds = points.reduce(([left, top, right, bottom], [x, z]) => [
    Math.min(left, x), Math.min(top, z), Math.max(right, x), Math.max(bottom, z),
  ], [Infinity, Infinity, -Infinity, -Infinity]);
  const [left, top, right, bottom] = bounds;
  const pad = Math.max(right - left, bottom - top) * .09;
  return (
    <div className="space-y-2">
      {showMap && points.length > 0 && (
        <div className="space-y-2">
          {levels.length > 1 && (
            <div className="flex flex-wrap gap-1" aria-label="Facade floor">
              {levels.map(level => (
                <button key={level.id} type="button" aria-pressed={level.id === levelId}
                  disabled={disabled || !walls.some(wall => wall.level_id === level.id)}
                  onClick={() => onChange(walls.find(wall => wall.level_id === level.id).id)}
                  className="rounded border px-2 py-1 text-xs aria-pressed:border-[#e8a33d] aria-pressed:bg-[#e8a33d]/10 disabled:opacity-50">
                  {level.label}
                </button>
              ))}
            </div>
          )}
          <svg aria-hidden="true" viewBox={[left - pad, top - pad, right - left + pad * 2, bottom - top + pad * 2].join(' ')}
            className="h-24 w-full rounded-lg border border-[#ece6da]/10 bg-[#11151b] sm:h-28">
            {levelWalls.map(wall => (
              <polygon key={wall.id} points={wall.polygon_meters?.map(point => point.join(',')).join(' ')} fill="#2a313b" />
            ))}
            {walls.filter(wall => wall.level_id === levelId).map(wall => {
              const { origin, direction, length_m: length } = wallFacadeFrame(wall);
              const line = { x1: origin[0], y1: origin[1], x2: origin[0] + length * direction[0], y2: origin[1] + length * direction[1] };
              return (
                <g key={wall.id} data-wall-id={wall.id} onClick={disabled ? undefined : () => onChange(wall.id)}
                  className={disabled ? undefined : 'cursor-pointer hover:opacity-70'}>
                  <title>{wall.id} · {hasFinish(wall) ? 'Finish applied' : 'Unfinished'}</title>
                  <line {...line} stroke="transparent" strokeWidth="12" vectorEffect="non-scaling-stroke" />
                  <line {...line} stroke={wall.id === value ? '#d18f2c' : hasFinish(wall) ? '#267857' : '#68717a'}
                    strokeWidth={wall.id === value ? 5 : 3} strokeLinecap="round" vectorEffect="non-scaling-stroke" />
                </g>
              );
            })}
          </svg>
          <div className="flex flex-wrap justify-between gap-1 text-[10px] text-[#ece6da]/75">
            <span><span className="text-[#a8711f]">●</span> Selected <span className="text-[#267857]">●</span> Applied</span>
            <span role="status">{walls.filter(hasFinish).length} of {walls.length} walls finished</span>
          </div>
        </div>
      )}
      <select
        aria-label="Exterior wall"
        className="w-full min-w-0 rounded-lg border border-[#ece6da]/20 bg-parchment-card px-2 py-2 text-xs"
        value={selected?.id || ''}
        onChange={event => onChange(event.target.value)}
        disabled={disabled || !walls.length}
      >
        <option value="" disabled>{walls.length ? 'Choose an exterior wall' : 'No exterior walls'}</option>
        {walls.map(wall => {
          const level = levels.find(storey => storey.id === wall.level_id)?.label;
          const [x, z] = wallFacadeFrame(wall).outward;
          const side = Math.abs(x) >= Math.abs(z) ? (x >= 0 ? 'East' : 'West') : (z >= 0 ? 'South' : 'North');
          const finish = hasFinish(wall) ? `Applied · ${wall.facade_texture.style ?? 'Default facade'}` : 'Unfinished';
          const name = wall.id.replace(/^(ground|upper)_/, '').replace(/_/g, ' ');
          return <option key={wall.id} value={wall.id}>
            {[level, side, name === side.toLowerCase() ? null : name, finish].filter(Boolean).join(' · ')}
          </option>;
        })}
      </select>
    </div>
  );
}
