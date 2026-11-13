import React, { Suspense, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Bounds, Html, OrbitControls, useGLTF } from '@react-three/drei';
import { Check, X } from 'lucide-react';
import { applyFurnitureFinish, furnitureColorChannels, FURNITURE_COLOR_LABELS, FURNITURE_FINISHES } from '../geometry/furnitureAppearance.js';
import { buildFurnitureShape, furnitureShapeOptions } from '../geometry/furnitureShapes.js';
import { fitCustomFurnitureAsset } from '../geometry/customFurnitureAsset.js';

function PreviewObject({ item, model, onChannels, ownsGeometry = false }) {
  useLayoutEffect(() => {
    if (item.source !== 'custom') applyFurnitureFinish(model, item.finish, item.category, item.colors);
  }, [model, item.finish, item.category, item.colors, item.source]);
  useEffect(() => {
    onChannels(item.source === 'custom' ? [] : furnitureColorChannels(model, item.category));
  }, [model, item.category, item.source, onChannels]);
  useEffect(() => () => {
    const materials = new Set();
    model.traverse(mesh => {
      if (!mesh.isMesh) return;
      if (ownsGeometry) mesh.geometry.dispose();
      [mesh.material].flat().forEach(material => materials.add(material));
    });
    materials.forEach(material => material.dispose());
  }, [model, ownsGeometry]);
  return <primitive object={model} />;
}

function PreviewModel({ item, onChannels }) {
  const { scene } = useGLTF(item.glb);
  const model = useMemo(() => {
    const copy = scene.clone(true);
    copy.traverse(mesh => {
      if (!mesh.isMesh) return;
      mesh.material = Array.isArray(mesh.material)
        ? mesh.material.map(material => material.clone()) : mesh.material.clone();
    });
    if (item.source === 'custom') return fitCustomFurnitureAsset(copy, item.dimensions_m);
    copy.scale.fromArray(item.scale);
    return copy;
  }, [scene, item.scale, item.source, item.dimensions_m]);
  return <PreviewObject item={item} model={model} onChannels={onChannels} />;
}

function PreviewShape({ item, onChannels }) {
  const model = useMemo(() => buildFurnitureShape(item), [item.category, item.shape, item.dimensions_m]);
  return <PreviewObject item={item} model={model} onChannels={onChannels} ownsGeometry />;
}

const original = { id: 'original', label: 'Original shape', desc: 'The original geometry for the selected model' };

export default function FurnitureCatalog({ item, choices, preset, loading, error, onApply, onClose }) {
  const [modelId, setModelId] = useState(item.catalog_id);
  const [finish, setFinish] = useState(item.finish || preset || 'contemporary_minimal');
  const [shape, setShape] = useState(item.shape || original.id);
  const [colors, setColors] = useState(item.colors || {});
  const [channels, setChannels] = useState([]);
  const [tab, setTab] = useState('Model');
  const [library, setLibrary] = useState(item.source === 'custom' ? 'custom' : 'builtin');
  const models = [item, ...choices];
  const selected = models.find(model => model.catalog_id === modelId) || item;
  const custom = selected.source === 'custom';
  const visibleModels = models.filter(model => (model.source === 'custom') === (library === 'custom'));
  const shapes = custom ? [original] : [original, ...furnitureShapeOptions(selected.category)];
  const profile = shapes.find(option => option.id === shape) || original;
  const draft = custom ? { ...selected, shape: original.id, finish: undefined, colors: undefined }
    : { ...selected, shape: profile.id, finish, colors };
  const palette = FURNITURE_FINISHES[finish];
  const [width, depth, height] = selected.dimensions_m.map(value => value.toFixed(2));

  useEffect(() => {
    const closeOnEscape = event => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/65 p-4" onClick={onClose}>
      <section role="dialog" aria-modal="true" aria-label="Furniture catalog"
        onClick={event => event.stopPropagation()}
        className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl border border-[#ece6da]/20 bg-[#11151b] shadow-2xl">
        <header className="flex items-start justify-between border-b border-[#ece6da]/15 px-5 py-4">
          <div>
            <p className="font-mono text-xs uppercase tracking-wider text-[#a8711f]">Furniture catalog</p>
            <h2 className="text-xl font-bold">Replace {item.name}</h2>
            <p className="mt-1 text-sm text-[#ece6da]/65">Choose the model, shape, and colors for this piece.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close furniture catalog" className="rounded-md p-2 hover:bg-white/10">
            <X size={20} />
          </button>
        </header>
        <div className="grid gap-4 p-5 sm:grid-cols-2">
          <div>
            <div className="h-52 overflow-hidden rounded-xl bg-[#1f252e] sm:h-64" aria-label="3D furniture preview">
              <Canvas camera={{ position: [3, 2.5, 3], fov: 40 }} dpr={[1, 1.5]}>
                <ambientLight intensity={1.65} />
                <directionalLight position={[3, 5, 4]} intensity={2} />
                <OrbitControls makeDefault enablePan={false} />
                <Suspense fallback={<Html center><span className="whitespace-nowrap text-xs" role="status">Loading model…</span></Html>}>
                  <Bounds key={`${selected.glb}:${profile.id}`} fit clip observe margin={1.35}>
                    {profile.id === original.id
                      ? <PreviewModel item={draft} onChannels={setChannels} />
                      : <PreviewShape item={draft} onChannels={setChannels} />}
                  </Bounds>
                </Suspense>
              </Canvas>
            </div>
            <p className="mt-2 text-center text-sm font-medium">{selected.name}</p>
            <p className="text-center text-xs font-medium text-[#a8711f]">{custom ? 'Custom · original 3D reconstruction' : profile.label}</p>
            <p className="text-center text-xs text-[#ece6da]/70">{width} m wide · {depth} m deep · {height} m high</p>
          </div>
          <div className="space-y-4">
            <nav aria-label="Furniture customization" className="grid grid-cols-3 gap-1 rounded-lg bg-[#1f252e] p-1">
              {['Model', 'Shape', 'Color'].map(label => (
                <button key={label} type="button" onClick={() => setTab(label)} aria-pressed={tab === label}
                  disabled={custom && label !== 'Model'}
                  className="rounded-md px-2 py-2 text-sm font-semibold aria-pressed:bg-[#ece6da] aria-pressed:text-[#0b0e12] disabled:opacity-40">
                  {label}
                </button>
              ))}
            </nav>
            {tab === 'Model' && (
              <div>
                <div role="group" aria-label="Furniture library" className="mb-2 flex gap-2">
                  {[['builtin', 'Built-in'], ['custom', 'Custom']].map(([key, label]) => (
                    <button key={key} type="button" onClick={() => setLibrary(key)} aria-pressed={library === key}
                      className="rounded-full border px-3 py-1 text-xs font-semibold aria-pressed:border-[#d4943a] aria-pressed:bg-[#d4943a]/10">
                      {label}
                    </button>
                  ))}
                </div>
                <h3 className="mb-1 font-semibold">{library === 'custom' ? 'Saved image-to-3D models that fit' : 'Built-in models that fit'}</h3>
                <div className="grid gap-1.5">
                  {visibleModels.map(model => (
                    <button key={model.catalog_id} type="button" onClick={() => setModelId(model.catalog_id)}
                      aria-pressed={modelId === model.catalog_id}
                      className="flex items-center justify-between rounded-lg border bg-parchment-card px-3 py-2 text-left text-sm aria-pressed:border-[#d4943a]">
                      {model.name}
                      {model.catalog_id === item.catalog_id && <small className="text-[#ece6da]/60">Current</small>}
                    </button>
                  ))}
                </div>
                {loading && <p role="status" className="mt-1 text-xs">Checking which models fit…</p>}
                {error && <p role="alert" className="mt-1 text-xs text-[#a8711f]">{error}</p>}
                {!loading && !error && visibleModels.length === 0 && (
                  <p className="mt-1 text-xs text-[#ece6da]/65">{library === 'custom'
                    ? 'No saved custom model fits this spot. Save a matching furniture type from the Generative Studio.'
                    : 'No built-in model fits this spot.'}</p>
                )}
                {custom && <p className="mt-2 text-xs text-[#ece6da]/65">Custom models keep their reconstructed shape and photo texture.</p>}
              </div>
            )}
            {tab === 'Shape' && (
              <div>
                <h3 className="mb-1 font-semibold">Shape profile</h3>
                <div className="grid gap-1.5">
                  {shapes.map(option => (
                    <button key={option.id} type="button" onClick={() => setShape(option.id)} aria-pressed={profile.id === option.id}
                      className="rounded-lg border bg-parchment-card px-3 py-2 text-left text-sm aria-pressed:border-[#d4943a]">
                      <strong className="block">{option.label}</strong>
                      <small className="text-[#ece6da]/65">{option.desc}</small>
                    </button>
                  ))}
                </div>
                <p className="mt-2 text-xs text-[#ece6da]/65">{shapes.length > 1
                  ? 'Profiles stay inside the space already checked for this model.'
                  : 'This furniture type uses the shapes supplied by its catalog models.'}</p>
              </div>
            )}
            {tab === 'Color' && (
              <div>
                <h3 className="mb-1 font-semibold">Finish palette</h3>
                <div className="grid gap-1.5">
                  {Object.entries(FURNITURE_FINISHES).map(([id, palette]) => (
                    <button key={id} type="button" onClick={() => setFinish(id)} aria-pressed={finish === id}
                      className="flex items-center gap-2 rounded-lg border bg-parchment-card px-3 py-1.5 text-left text-sm aria-pressed:border-[#d4943a]">
                      <span className="flex gap-0.5" aria-hidden="true">
                        {[palette.wood, palette.fabric, palette.metal].map(color => <i key={color} style={{ background: color }} className="h-3 w-3 rounded-full border border-black/10" />)}
                      </span>
                      {palette.label}
                    </button>
                  ))}
                </div>
                {!!channels.length && (
                  <fieldset className="mt-3 rounded-lg border border-[#ece6da]/15 p-2">
                    <legend className="px-1 text-xs font-semibold">Custom colors</legend>
                    <div className="grid grid-cols-2 gap-2">
                      {channels.map(channel => (
                        <label key={channel} className="flex items-center gap-1.5 text-xs">
                          <input type="color" value={colors[channel] || palette[channel]}
                            onChange={event => setColors(previous => ({ ...previous, [channel]: event.target.value }))}
                            className="h-7 w-8 cursor-pointer" />
                          {FURNITURE_COLOR_LABELS[channel]}
                        </label>
                      ))}
                    </div>
                    {!!Object.keys(colors).length && (
                      <button type="button" onClick={() => setColors({})} className="mt-2 text-xs text-[#a8711f] underline">Reset custom colors</button>
                    )}
                  </fieldset>
                )}
              </div>
            )}
          </div>
        </div>
        <footer className="flex justify-end gap-2 border-t border-[#ece6da]/15 px-5 py-4">
          <button type="button" onClick={onClose} className="rounded-lg px-4 py-2 text-sm hover:bg-white/5">Cancel</button>
          <button type="button" onClick={() => onApply(draft)}
            className="flex items-center gap-2 rounded-lg bg-[#ece6da] px-4 py-2 text-sm font-semibold text-[#0b0e12]">
            <Check size={16} /> Apply to this piece
          </button>
        </footer>
      </section>
    </div>
  );
}
