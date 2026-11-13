import React from 'react';
import { Columns2, Layers, Box, Paintbrush } from 'lucide-react';

export default function WorkspaceHeading({ activeTab, setActiveTab, isAnalyzing, hasLayout, onEditFacades }) {
  return (
    <div className="workspace-heading">
      <div>
        <div className="workspace-eyebrow">
          <span>01</span>
          <span className="dot">—</span>
          <span>PLAN TO 3D</span>
        </div>
        <h1>
          Floorplan <em>workspace.</em>
        </h1>
        <p>Refine your plan. Explore the space it becomes.</p>
      </div>
      <div className="heading-controls">
        <span className="workspace-note" role="status">
          {isAnalyzing
            ? 'Detecting rooms and walls…'
            : hasLayout
              ? 'Select a room to start furnishing'
              : 'Upload a plan to get started'}
        </span>
        {onEditFacades && (
          <button type="button" onClick={onEditFacades}
            className="inline-flex items-center gap-2 whitespace-nowrap rounded-lg bg-[#e8a33d] px-3 py-2 text-xs font-bold text-[#0b0e12]">
            <Paintbrush size={15} /> Edit facades
          </button>
        )}
        <div className="view-switch" aria-label="Workspace view">
          {[
            ['split', Columns2, 'Split view'],
            ['2d', Layers, '2D plan'],
            ['3d', Box, '3D house'],
          ].map(([id, Icon, label]) => (
            <button
              key={id}
              data-testid={`view-${id}`}
              aria-pressed={activeTab === id}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={15} />
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
