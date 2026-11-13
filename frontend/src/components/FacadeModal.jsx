import React, { useState } from 'react';
import { 
  X, 
  Sparkles, 
  Download, 
  Eye, 
  Layers, 
  RefreshCw, 
  Check, 
  Maximize2, 
  Info,
  Palette,
  ExternalLink
} from 'lucide-react';

const STYLES = [
  { id: 'Modern Minimalist', label: 'Modern Minimalist', desc: 'Warm white stucco' },
  { id: 'Mediterranean Stone', label: 'Mediterranean Stone', desc: 'Staggered sandstone blocks' },
  { id: 'Traditional Brick', label: 'Traditional Brick', desc: 'Terracotta brick and mortar' },
  { id: 'Lush Vegetation', label: 'Lush Vegetation', desc: 'Leafy wall finish' },
];

const CMP_LEGEND = [
  { label: 'Wall', color: '#0030ff', desc: 'Primary facade surface' },
  { label: 'Window', color: '#0080ff', desc: 'Glazed openings' },
  { label: 'Door', color: '#00cfff', desc: 'Entrance / egress' },
  { label: 'Window Sill', color: '#ff5000', desc: 'Bottom stone sill' },
  { label: 'Window Lintel', color: '#ffff00', desc: 'Top structural lintel' },
  { label: 'Cornice / Trim', color: '#0000de', desc: 'Roofline & base molding' },
];

export default function FacadeModal({
  isOpen,
  onClose,
  facadeData,
  onSwitchStyle,
  isSwitchingStyle = false,
  isApplied = false,
  onApply,
  onFocusIn3D
}) {
  const [activeTab, setActiveTab] = useState('side_by_side'); // 'side_by_side' | 'realistic' | 'sketch'

  if (!isOpen || !facadeData) return null;

  const {
    wall_id = 'Front Wall',
    style = null,
    wall_length_m = 0,
    wall_height_m = 0,
    num_floors = 1,
    texture_base64,
    sketch_base64,
    device = 'ONNX Runtime',
    model_info
  } = facadeData;
  const finishLabel = style == null ? 'Default facade' : model_info?.finish || model_info?.effective_style || style;

  const handleDownload = () => {
    if (!texture_base64) return;
    const link = document.createElement('a');
    link.href = texture_base64;
    link.download = `facade_${(style ?? 'default').toLowerCase().replace(/\s+/g, '_')}_${wall_id}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="cine-modal bg-[#0b0e12] border rounded-2xl shadow-2xl w-full max-w-5xl max-h-[92vh] flex flex-col overflow-hidden text-[#ece6da]">
        
        {/* Modal Header */}
        <div className="cine-modal__head px-6 py-4 bg-[#1f252e] border-b border-[#ece6da]/20 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-[#e8a33d] text-[#0b0e12] flex items-center justify-center shadow-sm">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="cine-title text-[#ece6da]">
                  Facade <em>synthesis</em>
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-[#e8a33d]/15 text-[#e8a33d] border border-[#e8a33d]/30">
                  PIX2PIX cGAN
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-[#ece6da]/10 text-[#ece6da]/70">
                  {device}
                </span>
              </div>
              <p className="font-mono text-xs text-[#ece6da]/65">
                {model_info?.model === 'facade_residual_film_512_v1'
                  ? `Experimental 512 facade · ${finishLabel}`
                  : 'Semantic layout sketch translated into an architectural facade'}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-white/10 text-[#ece6da]/70 hover:text-[#ece6da] transition-colors"
              title="Close modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Dimension & Wall Context Bar */}
        <div className="px-6 py-2.5 bg-[#11151b] border-b border-[#ece6da]/10 flex flex-wrap items-center justify-between text-xs font-mono">
          <div className="flex items-center space-x-4">
            <span className="text-[#ece6da]/60">Wall Target:</span>
            <span className="font-bold text-[#ece6da] bg-parchment-card px-2 py-0.5 rounded border border-[#ece6da]/20">
              {wall_id}
            </span>
            <span className="text-[#ece6da]/60">Dimensions:</span>
            <span className="font-bold text-[#ece6da]">
              {wall_length_m ? `${wall_length_m.toFixed(2)}m (W) × ${(wall_height_m || 2.8).toFixed(2)}m (H)` : 'Aspect preserved'}
            </span>
            <span className="text-[#ece6da]/60">Stories:</span>
            <span className="font-bold text-[#ece6da]">
              {num_floors} Floor{num_floors > 1 ? 's' : ''}
            </span>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center bg-[#1f252e] p-0.5 rounded-lg border border-[#ece6da]/15">
            <button
              onClick={() => setActiveTab('side_by_side')}
              className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                activeTab === 'side_by_side'
                  ? 'bg-parchment-card shadow text-[#ece6da] font-semibold'
                  : 'text-[#ece6da]/60 hover:text-[#ece6da]'
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveTab('realistic')}
              className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                activeTab === 'realistic'
                  ? 'bg-parchment-card shadow text-[#ece6da] font-semibold'
                  : 'text-[#ece6da]/60 hover:text-[#ece6da]'
              }`}
            >
              Realistic Only
            </button>
            <button
              onClick={() => setActiveTab('sketch')}
              className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                activeTab === 'sketch'
                  ? 'bg-parchment-card shadow text-[#ece6da] font-semibold'
                  : 'text-[#ece6da]/60 hover:text-[#ece6da]'
              }`}
            >
              Semantic Scrap
            </button>
          </div>
        </div>

        {/* Modal Body: Side-by-Side Visualizer */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div className={`grid gap-6 ${activeTab === 'side_by_side' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
            
            {/* LEFT: Input Semantic Layout Scrap */}
            {(activeTab === 'side_by_side' || activeTab === 'sketch') && (
              <div className="bg-parchment-card rounded-xl border border-[#ece6da]/20 p-4 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-[#0030ff]" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#ece6da]">
                      01 // INPUT SEMANTIC SKETCH ("THE SCRAP")
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#ece6da]/50">CMP 12-Class Colormap</span>
                </div>

                <p className="text-[11px] font-mono text-[#ece6da]/70 mb-3">
                  This is the semantic layout "scrap" projected automatically from your floorplan's exterior wall, doors, and windows geometry.
                </p>

                <div className="relative aspect-square w-full bg-[#0030ff]/10 rounded-lg overflow-hidden border border-[#ece6da]/10 flex items-center justify-center p-2">
                  {sketch_base64 ? (
                    <img
                      src={sketch_base64}
                      alt="Input Semantic Facade Sketch"
                      className="max-h-full max-w-full object-contain rounded shadow-inner"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  ) : (
                    <div className="text-center p-4 text-[#ece6da]/50 font-mono text-xs">
                      No semantic sketch available
                    </div>
                  )}
                </div>

                {/* CMP Color Palette Legend */}
                <div className="mt-3 pt-3 border-t border-[#ece6da]/10">
                  <span className="font-mono text-[10px] font-bold text-[#ece6da]/60 uppercase tracking-wider block mb-1.5">
                    CMP Semantic Legend:
                  </span>
                  <div className="grid grid-cols-3 gap-1.5 text-[10px] font-mono">
                    {CMP_LEGEND.map((item) => (
                      <div key={item.label} className="flex items-center space-x-1.5 bg-[#0b0e12]/60 px-1.5 py-1 rounded">
                        <span
                          className="w-2.5 h-2.5 rounded-sm border border-black/20 flex-shrink-0"
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="truncate text-[#ece6da]/80" title={item.desc}>
                          {item.label}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* RIGHT: Synthesized Realistic Facade */}
            {(activeTab === 'side_by_side' || activeTab === 'realistic') && (
              <div className="bg-parchment-card rounded-xl border border-[#ece6da]/20 p-4 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-[#e8a33d] animate-pulse" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#ece6da]">
                      02 // SYNTHESIZED REALISTIC FACADE
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#e8a33d]/10 text-[#e8a33d] font-bold">
                    {finishLabel}
                  </span>
                </div>

                <p className="text-[11px] font-mono text-[#ece6da]/70 mb-3">
                  Conditional GAN inference output translated into a realistic architectural facade with material rendering.
                </p>

                <div className="relative aspect-square w-full bg-[#ece6da]/5 rounded-lg overflow-hidden border border-[#ece6da]/10 flex items-center justify-center p-2">
                  {isSwitchingStyle ? (
                    <div className="flex flex-col items-center justify-center space-y-2 text-[#e8a33d]">
                      <RefreshCw className="w-6 h-6 animate-spin" />
                      <span className="font-mono text-xs">Generating facade preview…</span>
                    </div>
                  ) : texture_base64 ? (
                    <img
                      src={texture_base64}
                      alt="Synthesized Realistic Facade"
                      className="max-h-full max-w-full object-contain rounded shadow-md"
                    />
                  ) : (
                    <div className="text-center p-4 text-[#ece6da]/50 font-mono text-xs">
                      No realistic facade texture available
                    </div>
                  )}
                </div>

                {/* Synthesis Quality Badges */}
                <div className="mt-3 pt-3 border-t border-[#ece6da]/10 flex items-center justify-between text-[10px] font-mono text-[#ece6da]/60">
                  <span>Wall proportions preserved</span>
                  <span className="text-[#e8a33d] font-semibold">{isApplied ? 'Applied in 3D' : 'Texture preview'}</span>
                </div>
              </div>
            )}

          </div>

          {/* Style Selector Section */}
          <div className="bg-[#1f252e]/60 border border-[#ece6da]/15 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <label className="font-stencil text-xs text-[#ece6da] uppercase tracking-wider flex items-center space-x-1.5">
                <Palette className="w-3.5 h-3.5 text-[#e8a33d]" />
                <span>SWITCH ARCHITECTURAL STYLE & RE-SYNTHESIZE</span>
              </label>
              <span className="font-mono text-[10px] text-[#ece6da]/60">
                {style == null ? 'Default facade selected.' : 'Click the selected material again to use the default facade.'}
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              {STYLES.map((st) => (
                <button
                  key={st.id}
                  aria-pressed={style === st.id}
                  onClick={() => onSwitchStyle?.(style === st.id ? null : st.id)}
                  disabled={isSwitchingStyle}
                  className={`p-2.5 rounded-lg border text-left transition-all ${
                    style === st.id
                      ? 'bg-parchment-card border-[#e8a33d] shadow-sm ring-1 ring-[#e8a33d]'
                      : 'bg-parchment-card/60 border-[#ece6da]/15 hover:border-[#ece6da]/40 hover:bg-parchment-card'
                  } disabled:opacity-50`}
                >
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="font-stencil text-xs tracking-wider uppercase text-[#ece6da]">
                      {st.label}
                    </span>
                    {style === st.id && <Check className="w-3.5 h-3.5 text-[#e8a33d]" />}
                  </div>
                  <p className="font-mono text-[9px] text-[#ece6da]/60 leading-tight">
                    {st.desc}
                  </p>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer Controls */}
        <div className="px-6 py-4 bg-[#1f252e] border-t border-[#ece6da]/20 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-xs font-mono text-[#ece6da]/65">
            <Info className="w-4 h-4 text-[#e8a33d]" />
            <span>{isApplied ? 'Finish applied to this wall.' : 'Apply this finish to the selected exterior wall.'}</span>
          </div>

          <div className="flex items-center space-x-3">
            {!isApplied && (
              <button onClick={onApply} disabled={isSwitchingStyle}
                className="h-9 px-4 rounded-full bg-[#e8a33d] text-[#0b0e12] font-stencil text-xs uppercase flex items-center gap-2 disabled:opacity-50">
                <Check className="w-4 h-4" /> Apply to 3D
              </button>
            )}
            <button
              onClick={handleDownload}
              className="h-9 px-4 rounded-full bg-parchment-card hover:bg-[#0b0e12] border border-[#ece6da]/30 font-stencil text-xs uppercase tracking-wider text-[#ece6da] flex items-center space-x-2 transition-all shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>DOWNLOAD PNG</span>
            </button>

            {isApplied && onFocusIn3D && (
              <button
                onClick={() => {
                  onFocusIn3D();
                  onClose();
                }}
                className="h-9 px-4 rounded-full bg-[#ece6da] hover:bg-[#ffffff] text-[#0b0e12] font-stencil text-xs uppercase tracking-wider flex items-center space-x-2 transition-all shadow-md"
              >
                <Eye className="w-3.5 h-3.5 text-[#e8a33d]" />
                <span>FOCUS IN 3D SCENE</span>
              </button>
            )}

            <button
              onClick={onClose}
              className="h-9 px-5 rounded-full bg-[#e8a33d] hover:bg-[#d18f2c] text-[#0b0e12] font-stencil text-xs uppercase tracking-wider flex items-center space-x-1.5 transition-all shadow-md"
            >
              <span>DONE</span>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
