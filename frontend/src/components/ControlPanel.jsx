import React, { useRef, useState } from 'react';
import FacadeWallSelect from './FacadeWallSelect.jsx';
import { STOREY_HEIGHT_M } from '../geometry/buildingDimensions.js';
import {
  Upload,
  Sparkles,
  ArrowRight,
  RefreshCw,
  Ruler,
  Check,
  ChevronDown,
  FileImage,
  ArrowUpRight,
} from 'lucide-react';

const STYLES = [
  { id: 'Modern Minimalist', desc: 'Clean lines · warm white', swatch: 'modern' },
  { id: 'Mediterranean Stone', desc: 'Natural stone · sandy tones', swatch: 'stone' },
  { id: 'Traditional Brick', desc: 'Clay brick · defined joints', swatch: 'brick' },
  { id: 'Lush Vegetation', desc: 'Green textures · organic detail', swatch: 'green' },
];

export default function ControlPanel({
  onFileUpload,
  onRunAnalysis,
  onGenerateFacade,
  isAnalyzing,
  isGeneratingFacade,
  scale,
  setScale,
  numFloors,
  setNumFloors,
  selectedStyle,
  setSelectedStyle,
  layout,
  frontWallId,
  onSelectWall,
  appliedFacade,
  isCalibrating,
  setIsCalibrating,
  onOpenFacadeModal,
  onOpenGenerativeStudio,
  fileName,
  imageUrl,
  referenceLevels,
  isTracedSample,
}) {
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);
  const upload = (file) => {
    if (file && ['image/png', 'image/jpeg'].includes(file.type)) onFileUpload(file);
  };
  return (
    <aside className="project-sidebar" aria-label="Floorplan settings">
      <div className="sidebar-caption">
        <span>PROJECT SETUP</span>
        <span>01 — 03</span>
      </div>
      <section className="control-section">
        <h2>
          <span className="step-number">01</span>Your floorplan
        </h2>
        <input
          ref={input}
          className="sr-only"
          type="file"
          accept="image/png,image/jpeg"
          aria-label="Upload floorplan"
          onChange={(e) => upload(e.target.files?.[0])}
        />
        <button
          type="button"
          className={`plan-upload ${dragging ? 'is-dragging' : ''}`}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            upload(e.dataTransfer.files?.[0]);
          }}
        >
          {imageUrl ? <img src={imageUrl} alt="Current floorplan" /> : <Upload size={26} />}
          <span className="upload-caption">
            <Upload size={14} />
            Upload or replace plan
          </span>
        </button>
        <div className="file-caption">
          <FileImage size={14} />
          <span>{fileName || 'Sample floorplan'}</span>
          <span className="file-type">PNG / JPG</span>
        </div>
        <p className="field-help">Drop a drawing here, or click to browse.</p>
        {referenceLevels?.length > 0 && <p className="field-help">{isTracedSample ? 'Traced sample · ' : ''}{referenceLevels.length} aligned levels</p>}
      </section>
      <section className="control-section">
        <h2>
          <span className="step-number">02</span>Scale & dimensions
        </h2>
        <div className="field-row">
          <label htmlFor="plan-scale">Plan scale</label>
          <output htmlFor="plan-scale">
            {scale.toFixed(3)} <small>m / px</small>
          </output>
        </div>
        <input
          id="plan-scale"
          className="scale-range"
          type="range"
          min="0.015"
          max="0.060"
          step="0.001"
          value={scale}
          onChange={(e) => setScale(parseFloat(e.target.value))}
        />
        <button
          className="secondary-action full-width"
          aria-pressed={isCalibrating}
          onClick={() => setIsCalibrating(!isCalibrating)}
        >
          <Ruler size={15} />
          {isCalibrating ? 'Cancel measurement' : 'Calibrate with two points'}
        </button>
        {referenceLevels?.length ? <p className="field-help mt-3">Inspect each storey with the level selector in the 3D view.</p> : <>
          <div className="field-row storey-label">
            <span>Repeat this level</span>
            <small>{(numFloors * (layout?.walls[0]?.height_m || STOREY_HEIGHT_M)).toFixed(1)} m high</small>
          </div>
          <div className="storey-options" aria-label="Building storeys">
            {[1, 2, 3].map((f) => (
              <button key={f} aria-pressed={numFloors === f} onClick={() => setNumFloors(f)}>
                {f} {f === 1 ? 'storey' : 'storeys'}
              </button>
            ))}
          </div>
        </>}
        <button
          className="primary-action full-width analyze-action"
          onClick={onRunAnalysis}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? <RefreshCw className="animate-spin" size={16} /> : <Sparkles size={16} />}
          {isTracedSample
            ? (isAnalyzing ? 'Scoring room labels…' : 'Refresh room confidence')
            : (isAnalyzing ? 'Detecting your rooms…' : 'Detect rooms & build 3D')}
          {!isAnalyzing && <ArrowRight size={16} />}
        </button>
      </section>
      <details className="control-section facade-disclosure">
        <summary>
          <h2>
            <span className="step-number">03</span>Exterior finish
          </h2>
          <ChevronDown size={16} />
        </summary>
        <p className="field-help">Select a wall and choose its finish.</p>
        <FacadeWallSelect layout={layout} value={frontWallId} onChange={onSelectWall} disabled={isGeneratingFacade} />
        <p className="field-help">{selectedStyle ? 'Click the selected material again to use the default facade.' : 'Default facade selected.'}</p>
        <div className="material-options">
          {STYLES.map((style) => (
            <button
              key={style.id}
              className="material-option"
              aria-pressed={selectedStyle === style.id}
              disabled={isGeneratingFacade}
              onClick={() => setSelectedStyle(current => current === style.id ? null : style.id)}
            >
              <span className={`material-swatch ${style.swatch}`} />
              <span>
                <strong>{style.id}</strong>
                <small>{style.desc}</small>
              </span>
              {selectedStyle === style.id && <Check size={14} />}
            </button>
          ))}
        </div>
        {appliedFacade?.texture_base64 && (
          <button className="facade-current" onClick={onOpenFacadeModal}>
            <img src={appliedFacade.texture_base64} alt="Current facade texture" />
            <span>
              Applied finish<small>{appliedFacade.style == null ? 'Default facade' : appliedFacade.model_info?.finish || appliedFacade.model_info?.effective_style || appliedFacade.style}</small>
            </span>
            <ArrowUpRight size={16} />
          </button>
        )}
        <button
          className="secondary-action full-width"
          disabled={!frontWallId || isGeneratingFacade}
          onClick={() => onGenerateFacade()}
        >
          {isGeneratingFacade ? (
            <RefreshCw size={15} className="animate-spin" />
          ) : (
            <Sparkles size={15} />
          )}
          {isGeneratingFacade ? 'Generating finish…' : 'Generate facade'}
        </button>
      </details>
      <div className="sidebar-tip">
        <Sparkles size={18} />
        <div>
          <strong>Make it your own</strong>
          <p>Create furniture and explore exterior materials in the generative studio.</p>
          <button onClick={onOpenGenerativeStudio}>
            Open studio <ArrowUpRight size={14} />
          </button>
        </div>
      </div>
    </aside>
  );
}
