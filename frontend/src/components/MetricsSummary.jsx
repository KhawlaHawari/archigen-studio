import React from 'react';
import { LayoutGrid, AlertTriangle } from 'lucide-react';

function checkRoomPlausibility(room) {
  const t = (room.room_type || '').toLowerCase();
  const s = room.surface_m2 || 0;
  if (t.includes('bed')) {
    if (s < 5.0) return { flag: true, warning: 'Bedroom < 5 m² (under standard living norm)' };
    if (s > 45.0) return { flag: true, warning: 'Bedroom > 45 m² (unusually large)' };
  } else if (t.includes('bath')) {
    if (s < 1.8) return { flag: true, warning: 'Bathroom < 1.8 m² (under sanitary code minimum)' };
    if (s > 20.0) return { flag: true, warning: 'Bathroom > 20 m² (unusually large)' };
  } else if (t.includes('kitchen')) {
    if (s < 3.0) return { flag: true, warning: 'Kitchen < 3 m² (cramped)' };
    if (s > 40.0) return { flag: true, warning: 'Kitchen > 40 m² (unusually large)' };
  }
  return { flag: false, warning: null };
}

export default function MetricsSummary({ layout, frontWallId, selectedRoom, onSelectRoom }) {
  if (!layout)
    return (
      <footer className="room-summary empty-summary">
        <LayoutGrid size={16} />
        Room measurements will appear here after detection.
      </footer>
    );
  const rooms = layout.rooms || [];
  const metadata = layout.metadata || {};
  const area =
    metadata.total_surface_m2 ||
    layout.total_area_m2 ||
    rooms.reduce((sum, room) => sum + (room.surface_m2 || 0), 0);
  return (
    <footer className="room-summary">
      <div className="summary-metrics">
        <div>
          <small>Total area</small>
          <strong>
            {Number(area).toFixed(1)} <span>m²</span>
          </strong>
        </div>
        <div>
          <small>{layout.metadata?.source === 'sample_reference' ? 'Traced rooms' : 'Detected rooms'}</small>
          <strong>{rooms.length}</strong>
        </div>
      </div>
      <div className="room-list">
        <div className="room-list-heading">
          <span>YOUR SPACES</span>
          <small>Click a room to furnish it</small>
        </div>
        <div className="room-chips">
          {rooms.map((room, index) => {
            const check = checkRoomPlausibility(room);
            return (
              <button
                key={room.id || index}
                aria-pressed={selectedRoom?.id === room.id}
                onClick={() => onSelectRoom?.(room)}
                title={check.warning || `Design ${room.room_type}`}
              >
                <i
                  style={{ background: ['#d4a76e', '#86a89c', '#8d9eb9', '#b798ad'][index % 4] }}
                />
                <span>{room.room_type}</span>
                <small>{Number(room.surface_m2 || 0).toFixed(1)} m²</small>
                {check.flag && <AlertTriangle size={12} />}
              </button>
            );
          })}
        </div>
      </div>
    </footer>
  );
}
