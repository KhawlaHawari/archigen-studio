import React, { useState, useEffect, useRef } from 'react';
import FacadeWallSelect from './FacadeWallSelect.jsx';
import axios from 'axios';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { 
  Sparkles, 
  Box, 
  Layers, 
  Download, 
  Check, 
  RefreshCw, 
  Upload, 
  Palette, 
  Eye, 
  RotateCcw, 
  Maximize2, 
  Sliders, 
  Building2, 
  BookmarkCheck, 
  Trash2, 
  ExternalLink,
  Info
} from 'lucide-react';

// A real reference image supplies depth and material cues that flat icons lack.
const FURNITURE_PRESETS = [
  {
    id: 'preset_chair_1',
    name: 'Upholstered Dining Chair',
    category: 'chair',
    desc: 'TripoSR example; assumed height, replace with a measured value',
    height_m: 0.85,
    preview_url: '/samples/furniture/chair.png'
  }
];

const ROOM_LIBRARY_TYPES = {
  chair: [['dining_chair', 'Dining chair'], ['armchair', 'Armchair'], ['office_chair', 'Office chair']],
  table: [['coffee_table', 'Coffee table'], ['dining_table', 'Dining table'], ['desk', 'Desk']],
  storage: [['wardrobe', 'Wardrobe'], ['bookcase', 'Bookcase'], ['sideboard', 'Sideboard'], ['nightstand', 'Nightstand']],
};

const FACADE_STYLES = [
  { id: 'Modern Minimalist', label: 'Modern Minimalist', desc: 'Warm white stucco' },
  { id: 'Mediterranean Stone', label: 'Mediterranean Stone', desc: 'Sandstone blocks with staggered joints' },
  { id: 'Traditional Brick', label: 'Traditional Brick', desc: 'Terracotta brick with mortar courses' },
  { id: 'Lush Vegetation', label: 'Lush Vegetation', desc: 'Leafy wall finish with irregular coverage' },
];

const CMP_LEGEND = [
  { label: 'Wall', color: '#0030ff', desc: 'Primary facade surface' },
  { label: 'Window', color: '#0080ff', desc: 'Glazed openings' },
  { label: 'Door', color: '#00cfff', desc: 'Entrance / egress' },
  { label: 'Window Sill', color: '#ff5000', desc: 'Bottom stone sill' },
  { label: 'Window Lintel', color: '#ffff00', desc: 'Top structural lintel' },
  { label: 'Cornice / Trim', color: '#0000de', desc: 'Roofline & base molding' },
];

export default function GenerativeStudio({
  initialTab = 'furniture',
  layout,
  frontWallId,
  onSelectWall,
  selectedStyle,
  setSelectedStyle,
  numFloors,
  setNumFloors,
  requestFacade,
  onApplyFacadeTexture,
  onViewFacadeIn3D,
  onNavigateToFloorplan
}) {
  const [studioTab, setStudioTab] = useState(initialTab);

  // --- 2D-to-3D Furniture State ---
  const [furnitureImageBase64, setFurnitureImageBase64] = useState(FURNITURE_PRESETS[0].preview_url);
  const [furnitureName, setFurnitureName] = useState(FURNITURE_PRESETS[0].name);
  const [furnitureCategory, setFurnitureCategory] = useState(FURNITURE_PRESETS[0].category);
  const [removeBg, setRemoveBg] = useState(true);
  const [targetHeightM, setTargetHeightM] = useState(FURNITURE_PRESETS[0].height_m);
  const [isGenerating3D, setIsGenerating3D] = useState(false);
  const [generatedModel, setGeneratedModel] = useState(null);
  const [customLibrary, setCustomLibrary] = useState([]);
  const [libraryType, setLibraryType] = useState('');
  const [saveSuccessMsg, setSaveSuccessMsg] = useState(null);
  const [wireframe3D, setWireframe3D] = useState(false);
  const [autoRotate3D, setAutoRotate3D] = useState(true);

  // --- Facade Synthesis State ---
  const [isGeneratingFacade, setIsGeneratingFacade] = useState(false);
  const [facadeDraft, setFacadeDraft] = useState(null);
  const [studioError, setStudioError] = useState(null);
  const targetWall = layout?.walls.find(wall => wall.id === frontWallId && wall.is_exterior);
  const appliedFacade = targetWall?.facade_texture;
  const facadeResult = facadeDraft && facadeDraft.wall_id === frontWallId && facadeDraft.style === selectedStyle
    ? facadeDraft : appliedFacade?.style === selectedStyle ? appliedFacade : null;
  const isApplied = Boolean(facadeResult && facadeResult === appliedFacade);

  const fileInputRef = useRef(null);
  const canvas3DRef = useRef(null);
  const scene3DRef = useRef(null);
  const renderer3DRef = useRef(null);
  const modelGroupRef = useRef(null);
  const controls3DRef = useRef(null);
  const autoRotateRef = useRef(autoRotate3D);

  useEffect(() => {
    autoRotateRef.current = autoRotate3D;
  }, [autoRotate3D]);

  // Load saved custom library on mount
  useEffect(() => {
    axios.get('/furniture/custom-library')
      .then(res => {
        if (res.data) {
          setCustomLibrary(Object.values(res.data));
        }
      })
      .catch(err => console.log('Custom library empty or uninitialized:', err));
  }, []);

  // --- Initialize Dedicated 3D Furniture Viewport ---
  useEffect(() => {
    if (studioTab !== 'furniture') return;
    const container = canvas3DRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e12); // cinematic night
    scene3DRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, (width && height) ? (width / height) : 1.33, 0.05, 50);
    camera.position.set(1.4, 1.2, 1.6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    // Safely remove any existing canvas from previous renders
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    container.appendChild(renderer.domElement);
    renderer3DRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.02;
    controls.minDistance = 0.4;
    controls.maxDistance = 5.0;
    controls3DRef.current = controls;

    // Lighting setup
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0xd0c8b8, 0.9);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(3, 5, 2.5);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.bias = -0.0001;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0xbad2e8, 0.4);
    fillLight.position.set(-3, 2, -2);
    scene.add(fillLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.65);
    backLight.position.set(0, 3, -4);
    scene.add(backLight);

    // Grid Floor
    const grid = new THREE.GridHelper(3.0, 30, 0x3a424d, 0x232a33);
    grid.material.opacity = 0.12;
    grid.material.transparent = true;
    grid.position.y = 0.001;
    scene.add(grid);

    // Shadow Catcher Plane
    const shadowGeo = new THREE.PlaneGeometry(6, 6);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.15 });
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.receiveShadow = true;
    scene.add(shadowPlane);

    const modelGroup = new THREE.Group();
    scene.add(modelGroup);
    modelGroupRef.current = modelGroup;

    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      if (controls3DRef.current) {
        if (autoRotateRef.current && modelGroup.children.length > 0) {
          modelGroup.rotation.y += 0.006;
        }
        controls.update();
      }
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container || !renderer) return;
      const w = container.clientWidth || 600;
      const h = container.clientHeight || 450;
      if (h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
      if (renderer.domElement && renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [studioTab]);

  // Load GLB model into 3D viewer when generatedModel updates
  useEffect(() => {
    const modelGroup = modelGroupRef.current;
    if (!modelGroup || !generatedModel?.model_glb_base64) return;

    modelGroup.clear();
    let cancelled = false;
    let loadedScene = null;
    const disposeModel = (root) => root.traverse((child) => {
      if (!child.isMesh) return;
      child.geometry?.dispose();
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        if (!material) continue;
        for (const value of Object.values(material)) {
          if (value?.isTexture) value.dispose();
        }
        material.dispose();
      }
    });

    const loader = new GLTFLoader();
    loader.load(
      generatedModel.model_glb_base64,
      (gltf) => {
        const scene = gltf.scene;
        if (cancelled) { disposeModel(scene); return; }
        loadedScene = scene;
        scene.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.geometry && !child.geometry.attributes.normal) {
              child.geometry.computeVertexNormals();
            }
            if (child.material) {
              // Preserve baked colors, smooth seam normals and exported PBR factors.
              child.material.wireframe = wireframe3D;
              if (child.material.map) {
                if (THREE.SRGBColorSpace) {
                  child.material.map.colorSpace = THREE.SRGBColorSpace;
                }
                child.material.map.needsUpdate = true;
              }
              child.material.needsUpdate = true;
            }
          }
        });

        // Center on floor (y=0)
        const bbox = new THREE.Box3().setFromObject(scene);
        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        scene.position.x = -center.x;
        scene.position.y = -bbox.min.y;
        scene.position.z = -center.z;

        modelGroup.add(scene);

        // Frame camera comfortably
        const maxDim = Math.max(size.x, size.y, size.z) || 1.0;
        if (controls3DRef.current) {
          controls3DRef.current.target.set(0, size.y * 0.5, 0);
          controls3DRef.current.update();
        }
      },
      undefined,
      (err) => console.error('GLB preview load error:', err)
    );
    return () => {
      cancelled = true;
      if (loadedScene) {
        modelGroup.remove(loadedScene);
        disposeModel(loadedScene);
      }
    };
  }, [generatedModel, wireframe3D, studioTab]);

  // Handle Preset Selection
  const handleSelectPreset = (preset) => {
    setGeneratedModel(null);
    setSaveSuccessMsg(null);
    setFurnitureImageBase64(preset.preview_url);
    setFurnitureName(preset.name);
    setFurnitureCategory(preset.category);
    setTargetHeightM(preset.height_m);
  };

  // Keep detail for foreground cropping; the neural encoder resizes after cropping.
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setGeneratedModel(null);
    setSaveSuccessMsg(null);

    const reader = new FileReader();
    reader.onload = (uploadEvt) => {
      const rawDataUrl = uploadEvt.target.result;
      const cleanName = file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' ');
      setFurnitureName(cleanName.charAt(0).toUpperCase() + cleanName.slice(1));

      // Bound upload size without shrinking thin parts before background removal.
      const img = new Image();
      img.onload = () => {
        try {
          const maxDim = 2048;
          let w = img.width;
          let h = img.height;
          if (w > maxDim || h > maxDim) {
            if (w > h) {
              h = Math.round((h * maxDim) / w);
              w = maxDim;
            } else {
              w = Math.round((w * maxDim) / h);
              h = maxDim;
            }
          }
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          const optimizedDataUrl = canvas.toDataURL('image/png');
          setFurnitureImageBase64(optimizedDataUrl);
        } catch (resizeErr) {
          console.warn('Canvas resize fallback:', resizeErr);
          setFurnitureImageBase64(rawDataUrl);
        }
      };
      img.onerror = () => {
        setFurnitureImageBase64(rawDataUrl);
      };
      img.src = rawDataUrl;
    };
    reader.readAsDataURL(file);
  };

  // Trigger 3D Furniture Generation
  const handleGenerate3DFurniture = async () => {
    if (!furnitureImageBase64) return;
    setIsGenerating3D(true);
    setStudioError(null);
    setSaveSuccessMsg(null);
    try {
      let inputData = furnitureImageBase64;
      if (!inputData.startsWith('data:')) {
        const response = await fetch(inputData);
        if (!response.ok) throw new Error('Could not load the reference image');
        const blob = await response.blob();
        inputData = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      }
      const payload = {
        image_base64: inputData,
        name: furnitureName,
        category: furnitureCategory,
        remove_bg: removeBg,
        target_height_m: targetHeightM
      };

      const res = await axios.post('/furniture/generate-3d', payload);
      if (res.data?.status === 'success') {
        setGeneratedModel(res.data);
        setLibraryType('');
      }
    } catch (err) {
      console.error('Furniture generation error:', err);
      setStudioError(err.response?.data?.detail || '3D generation failed. Please try another image.');
    } finally {
      setIsGenerating3D(false);
    }
  };

  // Save 3D Model to Room Design Library
  const handleSaveToLibrary = async () => {
    if (!generatedModel) return;
    try {
      const payload = {
        id: generatedModel.id,
        name: generatedModel.name,
        category: generatedModel.category,
        furniture_type: libraryType || undefined,
        dimensions_m: generatedModel.dimensions_m,
        bounds: generatedModel.bounds,
        glb_base64: generatedModel.model_glb_base64
      };

      const res = await axios.post('/furniture/save-to-library', payload);
      if (res.data?.status === 'success') {
        setSaveSuccessMsg(`Saved to Custom in the room catalog.`);
        setCustomLibrary(prev => [...prev.filter(item => item.id !== generatedModel.id), res.data.item]);
      }
    } catch (err) {
      console.error('Save error:', err);
      const detail = err.response?.data?.detail;
      setStudioError(typeof detail === 'string' ? detail : 'Could not save this model. Please try again.');
    }
  };

  // Download GLB
  const handleDownloadGLB = () => {
    if (!generatedModel?.model_glb_base64) return;
    const a = document.createElement('a');
    a.href = generatedModel.model_glb_base64;
    a.download = `${generatedModel.id}.glb`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Trigger Facade Generation
  const handleGenerateFacade = async () => {
    setIsGeneratingFacade(true);
    setStudioError(null);
    try {
      const draft = await requestFacade(selectedStyle);
      if (draft) setFacadeDraft(draft);
    } catch (err) {
      console.error('Facade generation error:', err);
      setStudioError(err.response?.data?.detail || err.message || 'Facade generation failed.');
    } finally {
      setIsGeneratingFacade(false);
    }
  };

  // Apply Facade to Active 3D Building Floorplan
  const handleApplyFacadeTo3D = () => {
    if (!facadeResult?.texture_base64) return;
    onApplyFacadeTexture(facadeResult);
  };

  return (
    <div className="generative-page flex-1 flex flex-col h-full bg-[#0b0e12] overflow-hidden select-none font-mono text-[#ece6da]">
      
      <div className="workspace-heading generative-heading">
        <div>
          <div className="workspace-eyebrow">
            <span>02</span>
            <span className="dot">—</span>
            <span>GENERATIVE STUDIO</span>
          </div>
          <h1>
            Generative <em>studio.</em>
          </h1>
          <p>Bring your references to life, one detail at a time.</p>
        </div>
        <div className="studio-mode-switch" aria-label="Generation workspace">
          <button data-testid="studio-furniture" aria-pressed={studioTab === 'furniture'} onClick={() => setStudioTab('furniture')}><Box size={17} /><span>Furniture<small>Image to 3D</small></span></button>
          <button data-testid="studio-facade" aria-pressed={studioTab === 'facade'} onClick={() => setStudioTab('facade')}><Building2 size={17} /><span>Facades<small>Material exploration</small></span></button>
        </div>
      </div>

      {studioError && <div className="studio-error" role="alert"><Info size={16} /><span>{studioError}</span><button onClick={() => setStudioError(null)} aria-label="Dismiss error">Dismiss</button></div>}

      {/* =========================================================================
          TAB 1: 2D-TO-3D FURNITURE SYNTHESIZER
         ========================================================================= */}
      {studioTab === 'furniture' && (
        <div className="generation-shell flex-1 flex overflow-hidden">
          
          {/* Left Controls & Image Upload */}
          <aside className="generation-sidebar w-96 border-r border-[#ece6da]/15 bg-[#0b0e12] p-5 flex flex-col space-y-4 overflow-y-auto shrink-0">
            
            <div>
              <label className="font-stencil text-xs uppercase tracking-wider text-[#ece6da] block mb-2">
                01 / Reference image
              </label>

              {/* Upload Dropzone */}
              <div role="button" tabIndex={0} aria-label="Choose furniture image"
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInputRef.current?.click(); } }}
                onClick={() => fileInputRef.current?.click()}
                className="w-full h-44 rounded-xl border-2 border-dashed border-[#ece6da]/30 hover:border-[#e8a33d] bg-parchment-card/70 hover:bg-parchment-card transition-all cursor-pointer flex flex-col items-center justify-center p-3 relative overflow-hidden group shadow-xs"
              >
                {furnitureImageBase64 ? (
                  <div className="relative w-full h-full flex items-center justify-center">
                    <img
                      src={generatedModel?.preview_image_base64 || furnitureImageBase64}
                      alt="Input Furniture"
                      className="max-h-full max-w-full object-contain rounded drop-shadow-md"
                    />
                    {generatedModel?.preview_image_base64 && (
                      <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-emerald-700/90 text-white text-[9px] font-mono font-bold tracking-wider">
                        MODEL INPUT PREVIEW
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center space-y-1.5 text-center">
                    <Upload className="w-6 h-6 text-[#e8a33d]" />
                    <span className="font-stencil text-xs uppercase text-[#ece6da]">DROP 2D FURNITURE PHOTO</span>
                    <span className="text-[10px] text-[#ece6da]/50">PNG, JPG, WebP</span>
                  </div>
                )}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white text-xs font-stencil">
                  CLICK TO REPLACE IMAGE
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <p className="text-[11px] text-[#ece6da]/70 mt-2">
                Use a sharp photo of one complete item, ideally from a three-quarter angle.
                Keep legs visible and enter its measured height. Hidden surfaces are inferred.
              </p>
            </div>

            {/* Quick Test Preset Templates */}
            <div>
              <span className="text-[10px] font-bold text-[#ece6da]/60 uppercase tracking-wider block mb-1.5">
                Or start with a reference
              </span>
              <div className="grid grid-cols-2 gap-2">
                {FURNITURE_PRESETS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleSelectPreset(p)}
                    className={`p-2 rounded-lg border text-left flex items-center space-x-2 transition-all ${
                      furnitureName === p.name
                        ? 'bg-parchment-card border-[#e8a33d] shadow-xs ring-1 ring-[#e8a33d]'
                        : 'bg-parchment-card/40 border-[#ece6da]/15 hover:bg-parchment-card hover:border-[#ece6da]/40'
                    }`}
                  >
                    <img src={p.preview_url} alt={p.name} className="w-8 h-8 rounded shrink-0 object-contain" />
                    <div className="overflow-hidden">
                      <div className="text-[10px] font-bold text-[#ece6da] truncate">{p.name}</div>
                      <div className="text-[9px] text-[#ece6da]/50 capitalize">{p.category}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Metadata Parameters */}
            <div className="space-y-3 bg-[#1f252e]/50 p-3.5 rounded-xl border border-[#ece6da]/15 text-xs">
              <div>
                  <label htmlFor="furniture-name" className="text-[10px] text-[#ece6da]/70 uppercase block mb-1">Furniture name</label>
                <input
                    id="furniture-name"
                  type="text"
                  value={furnitureName}
                  onChange={(e) => setFurnitureName(e.target.value)}
                  className="w-full h-8 px-2.5 rounded bg-parchment-card border border-[#ece6da]/20 text-[#ece6da] font-mono text-xs focus:outline-none focus:border-[#e8a33d]"
                />
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label htmlFor="furniture-category" className="text-[10px] text-[#ece6da]/70 uppercase block mb-1">Category</label>
                  <select
                    id="furniture-category"
                    value={furnitureCategory}
                    onChange={(e) => setFurnitureCategory(e.target.value)}
                    className="w-full h-8 px-2 rounded bg-parchment-card border border-[#ece6da]/20 text-[#ece6da] font-mono text-xs"
                  >
                    <option value="chair">Chair</option>
                    <option value="table">Table / desk</option>
                    <option value="sofa">Sofa</option>
                    <option value="bed">Bed</option>
                    <option value="storage">Storage</option>
                    <option value="decor">Decor</option>
                    <option value="lighting">Lighting</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="furniture-height" className="text-[10px] text-[#ece6da]/70 uppercase block mb-1">Height (m)</label>
                  <input
                    id="furniture-height" min="0.05" max="10"
                    type="number"
                    step="0.05"
                    value={targetHeightM}
                    onChange={(e) => setTargetHeightM(parseFloat(e.target.value) || 0.85)}
                    className="w-full h-8 px-2 rounded bg-parchment-card border border-[#ece6da]/20 text-[#ece6da] font-mono text-xs"
                  />
                </div>
              </div>

              <label className="flex items-center space-x-2 cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={removeBg}
                  onChange={(e) => setRemoveBg(e.target.checked)}
                  className="rounded text-[#e8a33d] focus:ring-[#e8a33d]"
                />
                <span className="text-[11px] text-[#ece6da]/80">Remove image background</span>
              </label>
            </div>

            {/* Synthesize Button */}
            <button
              data-testid="generate-furniture"
              onClick={handleGenerate3DFurniture}
              disabled={isGenerating3D}
              className="w-full h-11 px-5 rounded-full bg-[#e8a33d] hover:bg-[#d18f2c] text-[#0b0e12] font-stencil text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-md shadow-[#e8a33d]/20 disabled:opacity-50 mt-auto"
            >
              {isGenerating3D ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Generating model…</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Generate 3D model</span>
                </>
              )}
            </button>
          </aside>

          {/* Center 3D Viewport & Model Inspector */}
          <main className="furniture-stage flex-1 flex flex-col h-full bg-[#141920] relative">
            
            {/* Viewport Top Toolbar */}
            <div className="furniture-toolbar h-10 border-b border-[#ece6da]/10 bg-[#11151b]/80 backdrop-blur-sm px-4 flex items-center justify-between z-10 text-xs">
              <div className="flex items-center space-x-2">
                <Box className="w-3.5 h-3.5 text-[#e8a33d]" />
                <span className="font-semibold text-xs uppercase tracking-wider text-[#ece6da]">
                  Furniture preview
                </span>
                {generatedModel && (
                  <span className="px-2 py-0.5 rounded text-[10px] bg-[#e8a33d]/10 text-[#e8a33d] font-bold">
                    {generatedModel.name}
                  </span>
                )}
              </div>

              {/* Viewport Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setAutoRotate3D(!autoRotate3D)}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono border transition-all ${
                    autoRotate3D
                      ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                      : 'bg-parchment-card text-[#ece6da]/70 border-[#ece6da]/20'
                  }`}
                >
                  Rotate: {autoRotate3D ? 'ON' : 'OFF'}
                </button>

                <button
                  onClick={() => setWireframe3D(!wireframe3D)}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono border transition-all ${
                    wireframe3D
                      ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                      : 'bg-parchment-card text-[#ece6da]/70 border-[#ece6da]/20'
                  }`}
                >
                  Wireframe: {wireframe3D ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>

            {/* Three.js Canvas Container & Viewport Overlays */}
            <div className="flex-1 w-full h-full relative overflow-hidden">
              {/* Dedicated Three.js Canvas Mount - completely empty so React never touches its DOM children */}
              <div ref={canvas3DRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

              {/* Sibling React Overlay: Empty State */}
              {!generatedModel && !isGenerating3D && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center pointer-events-none select-none">
                  <div className="w-16 h-16 rounded-2xl bg-parchment-card/70 border border-[#ece6da]/15 flex items-center justify-center mb-3 shadow-sm">
                    <Box className="w-8 h-8 text-[#e8a33d]" />
                  </div>
                  <h3 className="font-stencil text-base uppercase text-[#ece6da] tracking-wider mb-1">
                    Your next piece starts here.
                  </h3>
                  <p className="font-mono text-xs text-[#ece6da]/60 max-w-sm">
                    Upload a furniture photo or try the chair reference. Generate a model, then rotate it to explore every side.
                  </p>
                </div>
              )}

              {/* Sibling React Overlay: Generating Loader */}
              {isGenerating3D && (
                <div className="absolute inset-0 bg-[#0b0e12]/85 backdrop-blur-xs flex flex-col items-center justify-center z-20 pointer-events-none select-none">
                  <div className="w-16 h-16 rounded-2xl bg-parchment-card/90 border border-[#ece6da]/20 flex items-center justify-center mb-4 shadow-lg animate-pulse">
                    <RefreshCw className="w-8 h-8 text-[#e8a33d] animate-spin" />
                  </div>
                  <h3 className="font-stencil text-base uppercase text-[#ece6da] tracking-wider mb-2">
                    Building your model…
                  </h3>
                  <p className="font-mono text-xs text-[#ece6da]/60 max-w-sm text-center">
                    Preparing the image, reconstructing its surface, and baking a textured GLB mesh.
                  </p>
                </div>
              )}
            </div>

            {/* Model Action & Telemetry Floating Bar */}
            {generatedModel && (
              <div className="model-result-bar p-4 bg-parchment-card/95 border-t border-[#ece6da]/15 flex flex-wrap items-center justify-between gap-3 shadow-lg">
                <div className="flex items-center space-x-4 text-xs font-mono">
                  <div>
                    <span className="text-[#ece6da]/50 block text-[9px] uppercase">Dimensions (W×D×H)</span>
                    <span className="font-bold text-[#ece6da]">
                      {generatedModel.dimensions_m?.[0]}m × {generatedModel.dimensions_m?.[1]}m × {generatedModel.dimensions_m?.[2]}m
                    </span>
                  </div>
                  <div className="h-6 w-px bg-[#ece6da]/10" />
                  <div>
                    <span className="text-[#ece6da]/50 block text-[9px] uppercase">Mesh Density</span>
                    <span className="font-bold text-[#ece6da]">
                      {generatedModel.num_vertices?.toLocaleString()} verts / {generatedModel.num_faces?.toLocaleString()} faces
                    </span>
                  </div>
                  <div className="h-6 w-px bg-[#ece6da]/10" />
                  <div>
                    <span className="text-[#ece6da]/50 block text-[9px] uppercase">Surface Texture</span>
                    <span className="font-bold text-[#ece6da]">
                      {generatedModel.quality
                        ? `${generatedModel.quality.texture_resolution}px neural atlas · roughness preset`
                        : generatedModel.roughness !== undefined
                        ? (generatedModel.roughness < 0.40 ? `Smooth Polished (${generatedModel.roughness})` : `Matte Textured (${generatedModel.roughness})`)
                        : 'Auto PBR'}
                    </span>
                  </div>
                  <div className="h-6 w-px bg-[#ece6da]/10" />
                  <div>
                    <span className="text-[#ece6da]/50 block text-[9px] uppercase">Synthesis Engine</span>
                    <span className="font-bold text-[#e8a33d]">{generatedModel.engine}</span>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  {saveSuccessMsg && (
                    <span className="text-xs text-emerald-700 font-bold flex items-center space-x-1 animate-in fade-in">
                      <BookmarkCheck className="w-4 h-4 text-emerald-600" />
                      <span>{saveSuccessMsg}</span>
                    </span>
                  )}

                  {ROOM_LIBRARY_TYPES[generatedModel.category] && (
                    <label className="grid gap-1 text-[10px] text-[#ece6da]/70">
                      Room catalog type
                      <select value={libraryType} onChange={event => setLibraryType(event.target.value)}
                        className="rounded border border-[#ece6da]/20 bg-parchment-card px-2 py-1 text-xs">
                        <option value="">Auto</option>
                        {ROOM_LIBRARY_TYPES[generatedModel.category].map(([id, label]) => (
                          <option key={id} value={id}>{label}</option>
                        ))}
                      </select>
                    </label>
                  )}

                  <button
                    onClick={handleSaveToLibrary}
                    className="h-9 px-4 rounded-full bg-[#ece6da] hover:bg-[#ffffff] text-[#0b0e12] font-stencil text-xs uppercase tracking-wider flex items-center space-x-1.5 transition-all shadow-sm"
                  >
                    <BookmarkCheck className="w-3.5 h-3.5 text-[#e8a33d]" />
                    <span>Save to room library</span>
                  </button>

                  <button
                    onClick={handleDownloadGLB}
                    className="h-9 px-4 rounded-full bg-parchment-card hover:bg-[#0b0e12] border border-[#ece6da]/30 text-[#ece6da] font-stencil text-xs uppercase tracking-wider flex items-center space-x-1.5 transition-all shadow-xs"
                  >
                    <Download className="w-3.5 h-3.5 text-[#e8a33d]" />
                    <span>Download GLB</span>
                  </button>
                </div>
              </div>
            )}

            {/* Custom Saved Library Shelf */}
            {customLibrary.length > 0 && (
              <div className="px-5 py-3 bg-[#1f252e]/40 border-t border-[#ece6da]/10 flex items-center space-x-3 overflow-x-auto">
                <span className="text-[10px] font-stencil uppercase tracking-wider text-[#ece6da]/60 shrink-0">
                  SAVED CATALOG ({customLibrary.length}):
                </span>
                <div className="flex items-center space-x-2">
                  {customLibrary.map(item => (
                    <div
                      key={item.id}
                      className="px-2.5 py-1 rounded bg-parchment-card border border-[#ece6da]/20 flex items-center space-x-2 shrink-0 shadow-2xs"
                    >
                      <span className="w-2 h-2 rounded-full bg-[#e8a33d]" />
                      <span className="text-xs font-mono font-medium text-[#ece6da]">{item.name}</span>
                      <span className="text-[9px] text-[#ece6da]/50 capitalize">({item.category})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </main>
        </div>
      )}

      {/* =========================================================================
          TAB 2: NEURAL FACADE SYNTHESIS STUDIO
         ========================================================================= */}
      {studioTab === 'facade' && (
        <div className="generation-shell flex-1 flex overflow-hidden">
          
          {/* Left: Facade Style & Wall Selection Controls */}
          <aside className="generation-sidebar w-96 border-r border-[#ece6da]/15 bg-[#0b0e12] p-5 flex flex-col space-y-5 overflow-y-auto shrink-0">
            <div>
              <label className="font-stencil text-xs uppercase tracking-wider text-[#ece6da] flex items-center space-x-1.5 mb-1.5">
                <Building2 className="w-3.5 h-3.5 text-[#e8a33d]" />
                <span>01 / Choose an exterior wall</span>
              </label>
              <div className="p-3 bg-parchment-card rounded-xl border border-[#ece6da]/15 space-y-2 text-xs">
                <p className="text-[#ece6da]/70">Click a wall on the map or choose one from the list.</p>
                <FacadeWallSelect layout={layout} value={frontWallId} disabled={isGeneratingFacade} showMap
                  onChange={wallId => {
                    setFacadeDraft(null);
                    setStudioError(null);
                    onSelectWall(wallId);
                  }} />
                <div className="flex justify-between">
                  <span className="text-[#ece6da]/60">{layout?.storeys?.length ? 'Facade level' : 'Storeys'}</span>
                  {layout?.storeys?.length ? <span>{layout.storeys.find(level => level.id === layout.walls.find(wall => wall.id === frontWallId)?.level_id)?.label || 'Selected wall'}</span> : (
                  <div className="flex items-center space-x-1">
                    {[1, 2, 3].map(fl => (
                      <button
                        key={fl}
                        disabled={isGeneratingFacade}
                        onClick={() => {
                          setFacadeDraft(null);
                          setNumFloors(fl);
                        }}
                        className={`w-6 h-6 rounded text-xs font-mono font-bold transition-all ${
                          numFloors === fl ? 'bg-[#e8a33d] text-[#0b0e12]' : 'bg-[#0b0e12] text-[#ece6da]/70'
                        }`}
                      >
                        {fl}
                      </button>
                    ))}
                  </div>
                  )}
                </div>
              </div>
            </div>

            {/* Architectural Styles Selector */}
            <div className="space-y-2">
              <label className="font-stencil text-xs uppercase tracking-wider text-[#ece6da] flex items-center space-x-1.5">
                <Palette className="w-3.5 h-3.5 text-[#e8a33d]" />
                <span>02 / Material direction</span>
              </label>
              <p className="text-xs text-[#ece6da]/70">{selectedStyle ? 'Click the selected material again to use the default facade.' : 'Default facade selected.'}</p>

              <div className="space-y-2">
                {FACADE_STYLES.map(st => (
                  <button type="button" aria-pressed={selectedStyle === st.id} disabled={isGeneratingFacade}
                    key={st.id}
                    onClick={() => setSelectedStyle(current => current === st.id ? null : st.id)}
                    className={`studio-style-card p-3 rounded-xl border cursor-pointer transition-all ${
                      selectedStyle === st.id
                        ? 'bg-parchment-card border-[#e8a33d] shadow-sm ring-1 ring-[#e8a33d]'
                        : 'bg-parchment-card/50 border-[#ece6da]/15 hover:bg-parchment-card hover:border-[#ece6da]/40'
                    }`}
                  >
                    <span className={`material-swatch ${['modern', 'stone', 'brick', 'green'][FACADE_STYLES.indexOf(st)]}`} />
                    <div className="flex items-center justify-between">
                      <span className="font-stencil text-xs tracking-wider uppercase text-[#ece6da]">{st.label}</span>
                      {selectedStyle === st.id && <Check className="w-3.5 h-3.5 text-[#e8a33d]" />}
                    </div>
                    <p className="font-mono text-[9.5px] text-[#ece6da]/60 mt-0.5 leading-tight">{st.desc}</p>
                  </button>
                ))}
              </div>
            </div>

            <div className="sticky bottom-0 z-10 space-y-2 bg-[#0b0e12] py-2">
            <button
              data-testid="generate-facade"
              onClick={handleGenerateFacade}
              disabled={!targetWall || isGeneratingFacade}
              className="w-full h-11 px-5 rounded-full bg-[#e8a33d] hover:bg-[#d18f2c] text-[#0b0e12] font-stencil text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-md shadow-[#e8a33d]/20 disabled:opacity-50"
            >
              {isGeneratingFacade ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Generating facade…</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Generate facade</span>
                </>
              )}
            </button>
            {!isApplied && facadeResult?.texture_base64 && (
              <button type="button" onClick={handleApplyFacadeTo3D} data-testid="apply-facade"
                disabled={isGeneratingFacade}
                className="flex h-10 w-full items-center justify-center gap-2 rounded-full bg-[#ece6da] px-3 text-xs font-bold text-[#0b0e12] disabled:opacity-40">
                <Check size={16} /> Apply to this wall
              </button>
            )}
            {isApplied && (
              <div className="rounded-lg border border-emerald-700/30 bg-emerald-50 p-2 text-xs text-emerald-900" role="status">
                <p><strong>Applied to this wall.</strong> Pick another wall to continue.</p>
                <button type="button" onClick={onViewFacadeIn3D} className="mt-1 inline-flex items-center gap-1 font-bold underline">
                  <Eye size={15} /> View this wall in 3D
                </button>
              </div>
            )}
            </div>
          </aside>

          {/* Center: Side-by-Side Semantic Sketch vs Realistic Output */}
          <main className="facade-stage flex-1 flex flex-col h-full bg-[#11151b] p-6 overflow-y-auto">
            
            <div className="facade-comparison grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-[460px]">
              
              {/* LEFT: Input Semantic Layout Scrap */}
              <div className="facade-preview-card bg-parchment-card rounded-2xl border border-[#ece6da]/20 p-5 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#0030ff]" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#ece6da]">
                      01 / Opening layout
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#ece6da]/50">Structure guide</span>
                </div>

                <p className="text-[11px] text-[#ece6da]/70 mb-3 font-mono">
                  Your wall layout keeps doors and windows in place.
                </p>

                <div className="flex-1 w-full bg-[#0030ff]/10 rounded-xl overflow-hidden border border-[#ece6da]/10 flex items-center justify-center p-3 min-h-[280px]">
                  {facadeResult?.sketch_base64 ? (
                    <img
                      src={facadeResult.sketch_base64}
                      alt="Semantic Sketch"
                      className="max-h-full max-w-full object-contain rounded shadow-inner"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  ) : (
                    <div className="text-xs text-[#ece6da]/50 font-mono text-center">
                      Generate a facade to preview the wall layout.
                    </div>
                  )}
                </div>

                {/* CMP Color Palette Legend */}
                <div className="mt-4 pt-3 border-t border-[#ece6da]/10">
                  <span className="font-mono text-[10px] font-bold text-[#ece6da]/60 uppercase tracking-wider block mb-1.5">
                    Layout key
                  </span>
                  <div className="grid grid-cols-3 gap-2 text-[10.5px] font-mono">
                    {CMP_LEGEND.map(item => (
                      <div key={item.label} className="flex items-center space-x-1.5 bg-[#0b0e12]/60 px-2 py-1 rounded">
                        <span
                          className="w-3 h-3 rounded-sm border border-black/20 shrink-0"
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="truncate text-[#ece6da]/80">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* RIGHT: Synthesized Realistic Facade */}
              <div className="facade-preview-card bg-parchment-card rounded-2xl border border-[#ece6da]/20 p-5 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#e8a33d] animate-pulse" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#ece6da]">
                      02 / Material preview
                    </span>
                  </div>
                  <span className="px-2.5 py-0.5 rounded text-[10px] font-bold bg-[#e8a33d]/10 text-[#e8a33d]">
                    {selectedStyle == null ? 'Default facade' : facadeResult?.model_info?.finish || facadeResult?.model_info?.effective_style || selectedStyle}
                  </span>
                </div>

                <p className="text-[11px] text-[#ece6da]/70 mb-3 font-mono">
                  Explore a generated finish before applying it to your building.
                </p>

                <div className="flex-1 w-full bg-[#ece6da]/5 rounded-xl overflow-hidden border border-[#ece6da]/10 flex items-center justify-center p-3 min-h-[280px]">
                  {isGeneratingFacade ? (
                    <div className="flex flex-col items-center justify-center space-y-2 text-[#e8a33d]">
                      <RefreshCw className="w-8 h-8 animate-spin" />
                      <span className="font-mono text-xs">Neural Inference in Progress...</span>
                    </div>
                  ) : facadeResult?.texture_base64 ? (
                    <img
                      src={facadeResult.texture_base64}
                      alt="Realistic Facade"
                      className="max-h-full max-w-full object-contain rounded shadow-lg"
                    />
                  ) : (
                    <div className="text-xs text-[#ece6da]/50 font-mono text-center">
                      Your facade preview will appear here.
                    </div>
                  )}
                </div>

                <div className="mt-4 pt-3 border-t border-[#ece6da]/10 flex items-center justify-between text-xs font-mono text-[#ece6da]/60">
                  <span>{facadeResult ? `Aspect ratio preserved · ${facadeResult.model_info?.resolution || 256}px` : 'Ready to generate'}</span>
                  {facadeResult && <span className="text-[#e8a33d] font-semibold">Texture preview</span>}
                </div>
                {facadeResult?.model_info?.finish && (
                  <p className="mt-2 text-xs text-[#ece6da]/70">Material finish applied over the neutral neural texture.</p>
                )}
              </div>

            </div>

            {/* Facade Action Bar */}
            <div className="facade-result-bar mt-6 p-4 bg-parchment-card rounded-2xl border border-[#ece6da]/15 flex flex-wrap items-center justify-between gap-3 shadow-md">
              <div className="flex items-center space-x-2 text-xs font-mono text-[#ece6da]/70">
                <Info className="w-4 h-4 text-[#e8a33d]" />
                <span>{layout ? 'Each exterior wall keeps its own finish.' : 'Create a floorplan to apply a finish to your house.'}</span>
              </div>

              <div className="flex items-center space-x-3">
                {facadeResult?.texture_base64 && (
                  <a
                    href={facadeResult.texture_base64}
                    download={`facade_${(selectedStyle ?? 'default').toLowerCase().replace(/\s+/g, '_')}.png`}
                    className="h-10 px-4 rounded-full bg-parchment-card hover:bg-[#0b0e12] border border-[#ece6da]/30 font-stencil text-xs uppercase tracking-wider text-[#ece6da] flex items-center space-x-1.5 transition-all shadow-xs"
                  >
                    <Download className="w-3.5 h-3.5 text-[#e8a33d]" />
                    <span>Download texture</span>
                  </a>
                )}
              </div>
            </div>

          </main>
        </div>
      )}

    </div>
  );
}
