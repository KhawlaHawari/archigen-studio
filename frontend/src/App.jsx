import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import WorkspaceHeading from './components/WorkspaceHeading';
import ControlPanel from './components/ControlPanel';
import Floorplan2DCanvas from './components/Floorplan2DCanvas';
import Viewer3D from './components/Viewer3D';
import { createSampleHouse, rescaleSampleHouse, SAMPLE_FRONT_WALL } from './geometry/sampleHousePlan';
import { buildingLevels, fitHouseOpenings } from './geometry/houseGeometry.js';
import MetricsSummary from './components/MetricsSummary';
import RoomDesignPanel from './components/RoomDesignPanel';
import FacadeModal from './components/FacadeModal';
import GenerativeStudio from './components/GenerativeStudio';

const FurnitureCatalog = React.lazy(() => import('./components/FurnitureCatalog.jsx'));

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ArchiGen ErrorBoundary caught:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    if (this.props.onReset) {
      this.props.onReset();
    } else {
      window.location.hash = '#studio';
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full w-full bg-[#0b0e12] text-[#ece6da] p-6 font-mono">
          <div className="max-w-md w-full bg-parchment-card border border-[#ece6da]/20 rounded-2xl p-6 shadow-xl text-center space-y-4">
            <div className="w-12 h-12 mx-auto rounded-full bg-[#e8a33d]/10 text-[#e8a33d] flex items-center justify-center font-stencil text-xl">
              !
            </div>
            <h2 className="font-stencil text-lg uppercase tracking-wider text-[#ece6da]">
              STUDIO RECOVERY INTERCEPTED
            </h2>
            <p className="text-xs text-[#ece6da]/70 leading-relaxed">
              A render conflict was safely contained. You can reload the generative workspace or return to the 3D floorplan view.
            </p>
            {this.state.error && (
              <div className="text-[10px] text-left bg-[#ece6da]/5 p-2.5 rounded border border-[#ece6da]/10 overflow-auto max-h-24 font-mono text-[#d4943a]">
                {this.state.error.toString()}
              </div>
            )}
            <div className="flex justify-center space-x-3 pt-2">
              <button
                onClick={this.handleReset}
                className="px-5 py-2.5 rounded-full bg-[#e8a33d] hover:bg-[#d18f2c] text-[#0b0e12] font-stencil text-xs uppercase tracking-wider transition-all shadow-sm cursor-pointer"
              >
                RELOAD WORKSPACE
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [currentView, setCurrentView] = useState(() => {
    return window.location.hash === '#studio' ? 'studio' : 'showcase';
  });
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [transitionTarget, setTransitionTarget] = useState('studio');
  const [activeTab, setActiveTab] = useState('split');
  const [pageView, setPageView] = useState('floorplan'); // 'floorplan' | 'generative_studio'
  const [initialStudioTab, setInitialStudioTab] = useState('furniture');
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('/sample_floorplan.png');
  const [scale, setScale] = useState(0.03);
  const [numFloors, setNumFloors] = useState(1);
  const [selectedStyle, setSelectedStyle] = useState(null);
  const [frontWallId, setFrontWallId] = useState(null);

  const [layout, setLayout] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingFacade, setIsGeneratingFacade] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [health, setHealth] = useState(null);
  const [latency, setLatency] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const analysisRequest = React.useRef(0);
  const facadeRevision = React.useRef(0);

  // Room Design State (Stage 3)
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [furnishingData, setFurnishingData] = useState({});
  const [isDesigning, setIsDesigning] = useState(false);
  const [roomFocusTrigger, setRoomFocusTrigger] = useState(0);
  const [catalogSelection, setCatalogSelection] = useState(null);

  // Facade Studio State
  const [facadeDraft, setFacadeDraft] = useState(null);
  const [showFacadeModal, setShowFacadeModal] = useState(false);
  const [facadeFocusTrigger, setFacadeFocusTrigger] = useState(0);
  const appliedFacade = layout?.walls.find(wall => wall.id === frontWallId)?.facade_texture;

  // 1. Health check & Initial Demo load
  useEffect(() => {
    axios
      .get('/health')
      .then((res) => setHealth(res.data))
      .catch((err) => console.log('Backend not reachable yet:', err));

    if (currentView === 'studio' && !file) loadDefaultSample();

    // Listen for cross-frame messages from the ARCHIGEN by KKM Hero Showcase
    const handleHash = () => {
      const nextView = window.location.hash === '#studio' ? 'studio' : 'showcase';
      if (nextView !== currentView) {
        triggerTransition(nextView);
      }
    };
    const handleMsg = (e) => {
      if (e.data && e.data.type === 'LAUNCH_STUDIO') {
        // Landing page CTAs: "plan" -> 2D/3D workspace, "facade" -> facade tab of the studio
        if (e.data.target === 'facade') {
          setInitialStudioTab('facade');
          setPageView('generative_studio');
        } else {
          setPageView('floorplan');
        }
        triggerTransition('studio');
      }
    };
    window.addEventListener('hashchange', handleHash);
    window.addEventListener('message', handleMsg);
    return () => {
      window.removeEventListener('hashchange', handleHash);
      window.removeEventListener('message', handleMsg);
    };
  }, [currentView]);

  const triggerTransition = (targetView) => {
    setIsTransitioning(true);
    setTransitionTarget(targetView);
    // Smoothly flip underlying view halfway through when paper has fully covered viewport
    setTimeout(() => {
      setCurrentView(targetView);
      window.location.hash = targetView === 'studio' ? '#studio' : '#showcase';
    }, 2200);
    // Complete deliberate, majestic unroll across full screen
    setTimeout(() => {
      setIsTransitioning(false);
    }, 3800);
  };

  const loadDefaultSample = async () => {
    const request = ++analysisRequest.current;
    try {
      setIsAnalyzing(true);
      const urlParams = new URLSearchParams(window.location.search);
      const sampleParam = urlParams.get('sample');
      const samplePath = sampleParam === 'fresh_12' ? '/fresh_12.jpg' : '/sample_floorplan.png';
      const sampleMime = sampleParam === 'fresh_12' ? 'image/jpeg' : 'image/png';
      const res = await fetch(samplePath);
      const blob = await res.blob();
      if (request !== analysisRequest.current) return;
      const sampleFile = new File([blob], samplePath.replace('/', ''), { type: sampleMime });
      setFile(sampleFile);
      setImageUrl(samplePath);
      const reference = samplePath === '/sample_floorplan.png' ? showSampleReference(scale) : null;
      await executeAnalysis(sampleFile, scale, numFloors, null, reference);
    } catch (err) {
      console.error('Failed to load default sample:', err);
    } finally {
      if (request === analysisRequest.current) setIsAnalyzing(false);
    }
  };

  const showSampleReference = (sampleScale) => {
    const reference = createSampleHouse(sampleScale);
    setLayout(reference);
    setFrontWallId(SAMPLE_FRONT_WALL);
    setSelectedStyle(null);
    setFacadeDraft(null);
    setSelectedRoom(null);
    setFurnishingData({});
    setErrorMsg(null);
    return reference;
  };

  const handleFileUpload = (uploadedFile) => {
    setFile(uploadedFile);
    const url = URL.createObjectURL(uploadedFile);
    setImageUrl(url);
    setFrontWallId(null);
    setFacadeDraft(null);
    executeAnalysis(uploadedFile, scale, numFloors, null);
  };

  const executeAnalysis = async (fileObj, curScale, curFloors, wallId, reference = null) => {
    if (!fileObj) return;
    const request = ++analysisRequest.current;
    if (!reference) {
      facadeRevision.current++;
      setFacadeDraft(null);
    }
    setIsAnalyzing(true);
    setErrorMsg(null);
    const startTime = performance.now();

    const formData = new FormData();
    formData.append('file', fileObj);
    if (curScale) formData.append('scale_override', curScale.toString());
    formData.append('num_floors', curFloors.toString());
    if (wallId && !reference) formData.append('front_wall_id', wallId);
    if (reference) {
      const referenceScale = reference.metadata.scale_meters_per_pixel;
      formData.append('reference_rooms', JSON.stringify(reference.rooms.map(room => ({
        id: room.id,
        room_type: room.room_type,
        polygon_px: room.polygon_meters.map(point => point.map(value => value / referenceScale)),
      }))));
    }

    try {
      const response = await axios.post('/predict/floorplan', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      if (request !== analysisRequest.current) return;

      const data = response.data;
      const duration = Math.round(performance.now() - startTime);

      // Read backend measured latency or fallback to client roundtrip
      const headerLatency = response.headers['x-inference-latency-ms'];
      setLatency(headerLatency ? parseFloat(headerLatency).toFixed(1) : duration);

      if (reference) {
        const scores = data.reference_room_confidences;
        const requestedTypes = new Map(reference.rooms.map(room => [room.id, room.room_type]));
        setLayout(current => current?.metadata.source === 'sample_reference' ? {
          ...current,
          rooms: current.rooms.map(room => requestedTypes.get(room.id) === room.room_type
            ? { ...room, confidence: scores?.[room.id] ?? null }
            : room),
        } : current);
        return;
      }

      // Reconstruct layout structure
      const parsedLayout = {
        metadata: {
          canvas_size: [512, 512],
          scale_meters_per_pixel: curScale,
          total_surface_m2: data.total_area_m2,
          is_building_shell_valid: true,
          building_perimeter_m: Math.round(
            data.walls.reduce((acc, w) => acc + (w.polygon_meters?.length || 0) * 0.8, 0)
          ),
          num_walls: data.walls.length,
          num_rooms: data.rooms.length,
        },
        // Studio finishes are generated from the fitted 3D openings after analysis.
        walls: data.walls.map(({ facade_texture, ...wall }) => wall),
        rooms: data.rooms.map((r, i) => ({
          id: `room_${i}`,
          room_type: r.room_type,
          surface_m2: r.surface_m2,
          confidence: r.confidence,
          polygon_meters: r.polygon_points,
        })),
        doors: data.doors || [],
        windows: data.windows || [],
        envelope: data.envelope || [],
      };

      setLayout(parsedLayout);
      setSelectedRoom(null);
      setFurnishingData({});

      if (data.rooms.length === 0) {
        setErrorMsg('No rooms detected. This plan style may not be supported (e.g. dense CAD dimension text, low-contrast raster, or non-standard drafting symbols).');
      }

      // Find Front Wall & its Facade Texture
      const frontWall = data.walls.find((w) => w.is_front) || data.walls.find((w) => w.is_exterior);
      setFrontWallId(frontWall?.id || null);
      setSelectedStyle(null);
    } catch (err) {
      if (request !== analysisRequest.current) return;
      console.error('Inference error:', err);
      setErrorMsg(err.response?.data?.detail || 'Floorplan analysis failed.');
    } finally {
      if (request === analysisRequest.current) setIsAnalyzing(false);
    }
  };

  const handleSelectFrontWall = (newWallId) => {
    const wall = layout?.walls.find(item => item.id === newWallId);
    if (!wall?.is_exterior) return;
    if (newWallId !== frontWallId) {
      facadeRevision.current++;
      setFacadeDraft(null);
      if (wall.facade_texture?.texture_base64) setSelectedStyle(wall.facade_texture.style ?? null);
    }
    setFrontWallId(newWallId);
  };

  const handleSetNumFloors = (count) => {
    if (count === numFloors) return;
    facadeRevision.current++;
    setFacadeDraft(null);
    setNumFloors(count);
    if (layout && !layout.storeys?.length) {
      setLayout(current => ({ ...current, walls: current.walls.map(wall => ({ ...wall, facade_texture: undefined })) }));
    }
  };

  const handleCalibrateScale = (newScale) => {
    setScale(newScale);
    if (layout?.metadata.source === 'sample_reference') {
      facadeRevision.current++;
      setFacadeDraft(null);
      const rescaled = rescaleSampleHouse(layout, newScale);
      setLayout({ ...rescaled, walls: rescaled.walls.map(wall => ({ ...wall, facade_texture: undefined })) });
      setSelectedRoom(null);
      setFurnishingData({});
    } else if (file) {
      executeAnalysis(file, newScale, numFloors, frontWallId);
    }
  };

  const handleRunAnalysis = () => {
    const reference = layout?.metadata.source === 'sample_reference' ? layout : null;
    if (reference && scale !== reference.metadata.scale_meters_per_pixel) handleCalibrateScale(scale);
    executeAnalysis(file, scale, numFloors, frontWallId, reference);
  };

  const handleLayoutEdit = (next) => {
    facadeRevision.current++;
    setFacadeDraft(null);
    const before = [...layout.doors, ...layout.windows];
    const after = [...next.doors, ...next.windows];
    setLayout({ ...next, walls: next.walls.map(wall => {
      const previous = layout.walls.find(item => item.id === wall.id);
      const openings = after.filter(opening => opening.associated_wall === wall.id);
      const oldOpenings = before.filter(opening => opening.associated_wall === wall.id);
      const sameGeometry = previous?.polygon_meters === wall.polygon_meters &&
        previous?.level_id === wall.level_id && openings.length === oldOpenings.length &&
        openings.every((opening, i) => opening === oldOpenings[i]);
      return { ...wall, openings, facade_texture: sameGeometry ? wall.facade_texture : undefined };
    }) });
    setSelectedRoom(room => next.rooms.find(item => item.id === room?.id) || null);
    if (!next.walls.some(wall => wall.id === frontWallId)) setFrontWallId(next.walls.find(wall => wall.is_exterior)?.id || null);
  };

  const requestFacade = async (style) => {
    const target = layout?.walls.find(wall => wall.id === frontWallId && wall.is_exterior);
    if (!target) throw new Error('Choose an exterior wall.');
    if (isAnalyzing && layout.metadata.source !== 'sample_reference') throw new Error('Wait for floorplan analysis to finish.');
    const revision = facadeRevision.current;
    const level = buildingLevels(layout, numFloors).find(item => item.walls.includes(target));
    if (!level) throw new Error('The selected wall is outside the building levels.');
    const openings = fitHouseOpenings(level, layout.metadata.scale_meters_per_pixel)
      .filter(opening => opening.associated_wall === target.id);
    const response = await axios.post('/facade/generate', {
      wall: { ...target, height_m: level.unitHeight, openings, facade_texture: undefined },
      wall_id: target.id,
      style,
      num_floors: target.level_id ? 1 : numFloors,
    });
    if (revision !== facadeRevision.current) return null;
    if (response.data?.wall_id !== target.id || !response.data.texture_base64) throw new Error('Facade generation failed.');
    return response.data;
  };

  const applyFacade = (finish) => {
    setLayout(current => current && ({
      ...current,
      walls: current.walls.map(wall => wall.is_exterior && wall.id === finish.wall_id
        ? { ...wall, facade_texture: finish }
        : wall),
    }));
  };

  const viewFacadeIn3D = () => {
    setPageView('floorplan');
    setActiveTab('3d');
    setRoomFocusTrigger(0);
    setFacadeFocusTrigger(previous => previous + 1);
  };

  const openFacadeStudio = () => {
    setInitialStudioTab('facade');
    setPageView('generative_studio');
  };

  const handleGenerateFacade = async (style = selectedStyle) => {
    setSelectedStyle(style);
    setIsGeneratingFacade(true);
    setErrorMsg(null);
    try {
      const draft = await requestFacade(style);
      if (draft) {
        setFacadeDraft(draft);
        setShowFacadeModal(true);
      }
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || err.message);
    } finally {
      setIsGeneratingFacade(false);
    }
  };

  // --- Room Design Engine Handlers (Stage 3) ---
  const handleSelectRoom = (room) => {
    setSelectedRoom(room);
    setCatalogSelection(null);
  };

  const inspectRoomIn3D = () => {
    setActiveTab('3d');
    setFacadeFocusTrigger(0);
    setRoomFocusTrigger(previous => previous + 1);
  };

  const designRoomPayload = (room, preset, seed, forceFurnish) => {
    if (!room || !layout) return null;
    const level = buildingLevels(layout).find(item => item.rooms.some(candidate => candidate.id === room.id));
    if (!level) return null;
    const scaleMPerPx = layout.metadata?.scale_meters_per_pixel || .03;
    const openings = fitHouseOpenings(level, scaleMPerPx);
    return {
      room_id: room.id,
      room_type: room.room_type,
      polygon_meters: room.polygon_meters,
      doors: openings.filter(opening => opening.type === 'door'),
      windows: openings.filter(opening => opening.type === 'window'),
      wall_segments: level.walls.map(({ facade_texture, openings, ...wall }) => wall),
      scale_m_per_px: scaleMPerPx,
      preset,
      seed,
      force_furnish: forceFurnish,
    };
  };

  const handleGenerateDesign = async (room, preset = 'contemporary_minimal', seed = 42, forceFurnish = false) => {
    const payload = designRoomPayload(room, preset, seed, forceFurnish);
    if (!payload) return;
    setCatalogSelection(null);
    setIsDesigning(true);
    try {
      const response = await axios.post('/design', payload);
      const result = response.data;
      if (typeof result?.is_valid !== 'boolean' || !Array.isArray(result.items)) {
        throw new Error('The room design service returned an invalid response.');
      }
      setFurnishingData((prev) => ({
        ...prev,
        [room.id]: result
      }));
    } catch (err) {
      console.error('Room design request failed:', err);
      setFurnishingData((prev) => ({
        ...prev,
        [room.id]: {
          is_valid: false,
          failure_reason: 'REQUEST_FAILED',
          items: []
        }
      }));
    } finally {
      setIsDesigning(false);
    }
  };

  const handleRegenerateDesign = (room, preset) => {
    const current = furnishingData[room.id];
    handleGenerateDesign(room, preset, (current?.seed ?? 42) + 1, Boolean(current?.metadata?.compact_mode));
  };

  const handleRoomPreset = (roomId, preset) => {
    setFurnishingData(current => {
      const design = current[roomId];
      if (!design?.is_valid || design.preset === preset) return current;
      const items = design.items.map(item => ({ ...item, finish: undefined }));
      return { ...current, [roomId]: { ...design, preset, items } };
    });
  };

  const handleSelectFurniture = async (roomId, itemId) => {
    if (isDesigning) return;
    const room = layout?.rooms.find(candidate => candidate.id === roomId);
    const design = furnishingData[roomId];
    const item = design?.items.find(candidate => candidate.item_id === itemId);
    if (!room || !design?.is_valid || !item) return;
    const payload = designRoomPayload(room, design.preset, design.seed, Boolean(design.metadata?.compact_mode));
    if (!payload) return;

    setSelectedRoom(room);
    const selection = { roomId, item, design, choices: [], loading: true };
    setCatalogSelection(selection);
    try {
      const { data } = await axios.post('/design/furniture-choices', { ...payload, item_id: itemId, items: design.items });
      setCatalogSelection(current => current === selection ? { ...selection, choices: data.choices || [], loading: false } : current);
    } catch {
      setCatalogSelection(current => current === selection ? {
        ...selection, loading: false, error: 'The alternative models could not be loaded. You can still change the finish.',
      } : current);
    }
  };

  const applyFurnitureChoice = item => {
    const { roomId, design } = catalogSelection;
    setFurnishingData(current => current[roomId] === design ? {
      ...current, [roomId]: { ...design, items: design.items.map(other => other.item_id === item.item_id ? item : other) },
    } : current);
    setCatalogSelection(null);
  };

  const handleClearDesign = (roomId) => {
    setFurnishingData((prev) => {
      const next = { ...prev };
      delete next[roomId];
      return next;
    });
  };

  const handleRelabelRoom = (roomId, newType) => {
    if (!layout) return;
    const updatedRooms = layout.rooms.map((r) =>
      r.id === roomId ? { ...r, room_type: newType, confidence: newType === r.room_type ? r.confidence : null } : r
    );
    setLayout({ ...layout, rooms: updatedRooms });
    if (selectedRoom?.id === roomId) {
      setSelectedRoom(updatedRooms.find(room => room.id === roomId));
    }
    handleClearDesign(roomId);
  };


  const renderTransitionOverlay = () => {
    if (!isTransitioning) return null;
    return (
      <div className="fixed inset-0 z-50 pointer-events-none overflow-hidden select-none architectural-unroll-overlay">
        {/* Full-Screen Architectural Parchment Sheet Unfurling */}
        <div className="architectural-unrolled-sheet">
          {/* Centered Blueprint Stencil Manifesto from User Design */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none px-6">
            <div className="border border-[#ece6da]/25 p-8 sm:p-12 md:p-16 bg-[#0b0e12]/95 backdrop-blur-md max-w-3xl w-full shadow-2xl relative overflow-hidden">
              {/* Drafting Corner Registration Marks */}
              <div className="absolute top-3 left-4 font-mono text-[9px] text-[#ece6da]/40 uppercase tracking-widest">+ PT-00</div>
              <div className="absolute top-3 right-4 font-mono text-[9px] text-[#ece6da]/40 uppercase tracking-widest">+ SEC-01</div>
              <div className="absolute bottom-3 left-4 font-mono text-[9px] text-[#ece6da]/40 uppercase tracking-widest">+ ARCHIGEN</div>
              <div className="absolute bottom-3 right-4 font-mono text-[9px] text-[#ece6da]/40 uppercase tracking-widest">+ 2026</div>

              <p className="font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.24em] text-[#ece6da]/60 mb-4 text-center">
                ARCHIGEN 3D • DEEP LEARNING ARCHITECTURAL STUDIO
              </p>
              
              <h2 className="font-stencil text-2xl sm:text-3xl md:text-5xl text-[#ece6da] tracking-wide uppercase text-center leading-[1.08]">
                EVERY BUILDING STARTS FLAT.
              </h2>
              <h2 className="font-stencil text-2xl sm:text-3xl md:text-5xl text-[#ece6da] tracking-wide uppercase text-center leading-[1.08] mt-2">
                OURS DON'T <span className="text-[#e8a33d]">STAY</span> THAT WAY.
              </h2>
              
              <div className="mt-8 flex items-center justify-center gap-4">
                <span className="w-16 h-[1px] bg-[#ece6da]/25"></span>
                <span className="font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.2em] text-[#ece6da]/80 font-semibold">
                  {transitionTarget === 'studio' ? 'UNROLLING AI GENERATOR STUDIO...' : 'UNROLLING SHOWCASE TOUR...'}
                </span>
                <span className="w-16 h-[1px] bg-[#ece6da]/25"></span>
              </div>
            </div>
          </div>
        </div>

        {/* Physical 3D Roll Cylinder Leading Edge */}
        <div className="architectural-roll-cylinder">
          <div className="roll-paper-layers" />
          <div className="absolute inset-0 opacity-40 mix-blend-multiply bg-[repeating-linear-gradient(45deg,#1d1d1f_0,#1d1d1f_1px,transparent_0,transparent_10px)]" />
        </div>
      </div>
    );
  };

  if (currentView === 'showcase') {
    return (
      <div className="relative w-screen h-screen overflow-hidden bg-[#0b0e12]">
        {renderTransitionOverlay()}
        <iframe
          src="/landing/index.html"
          title="ArchiGen - from 2D plan to 3D building"
          className="w-full h-full border-none"
        />
      </div>
    );
  }

  return (
    <div className="design-app flex flex-col h-screen w-screen overflow-hidden bg-[#0b0e12] text-[#ece6da] font-sans selection:bg-[#d4943a]/20 selection:text-[#d4943a]">
      {renderTransitionOverlay()}

      {/* 1. Header */}
      <Navbar
        health={health}
        latency={latency}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        pageView={pageView}
        setPageView={view => {
          setInitialStudioTab('furniture');
          setPageView(view);
        }}
        onReturnToShowcase={() => triggerTransition('showcase')}
      />

      {/* 2. Main Page View: Generative AI Studio vs Floorplan Workspace */}
      {pageView === 'generative_studio' ? (
        <ErrorBoundary onReset={() => setPageView('floorplan')}>
          <GenerativeStudio
            initialTab={initialStudioTab}
            layout={layout}
            frontWallId={frontWallId}
            onSelectWall={handleSelectFrontWall}
            selectedStyle={selectedStyle}
            setSelectedStyle={setSelectedStyle}
            numFloors={numFloors}
            setNumFloors={handleSetNumFloors}
            requestFacade={requestFacade}
            onApplyFacadeTexture={applyFacade}
            onViewFacadeIn3D={viewFacadeIn3D}
            onNavigateToFloorplan={() => setPageView('floorplan')}
          />
        </ErrorBoundary>
      ) : (
      <section className="workspace-page">
      <WorkspaceHeading activeTab={activeTab} setActiveTab={setActiveTab} isAnalyzing={isAnalyzing}
        hasLayout={!!layout} onEditFacades={frontWallId ? openFacadeStudio : null} />
      <div className="floorplan-shell flex-1 flex overflow-hidden relative">
        {/* Left Sidebar Controls */}
        <ControlPanel
          referenceLevels={layout?.storeys}
          isTracedSample={layout?.metadata.source === 'sample_reference'}
          fileName={file?.name}
          imageUrl={imageUrl}
          onOpenGenerativeStudio={() => {
            setInitialStudioTab('furniture');
            setPageView('generative_studio');
          }}
          onFileUpload={handleFileUpload}
          onRunAnalysis={handleRunAnalysis}
          onGenerateFacade={handleGenerateFacade}
          onOpenFacadeModal={() => {
            setFacadeDraft(appliedFacade);
            setShowFacadeModal(true);
          }}
          isAnalyzing={isAnalyzing}
          isGeneratingFacade={isGeneratingFacade}
          scale={scale}
          setScale={setScale}
          numFloors={numFloors}
          setNumFloors={handleSetNumFloors}
          selectedStyle={selectedStyle}
          setSelectedStyle={setSelectedStyle}
          frontWallId={frontWallId}
          layout={layout}
          onSelectWall={handleSelectFrontWall}
          appliedFacade={appliedFacade}
          isCalibrating={isCalibrating}
          setIsCalibrating={setIsCalibrating}
        />

        {/* Center Viewport Area */}
        <main className="floorplan-main flex-1 flex flex-col h-full overflow-hidden bg-[#0b0e12] drafting-grid">
          {errorMsg && (
            <div className="bg-[#d4943a]/10 border-b border-[#d4943a]/30 px-6 py-2.5 font-mono text-xs text-[#d4943a] flex justify-between items-center z-20">
              <span className="font-medium tracking-wide">Notice: {errorMsg}</span>
              <button onClick={() => setErrorMsg(null)} className="underline uppercase tracking-wider text-[10px] ml-4 font-semibold hover:text-[#a8711f]">
                Dismiss ×
              </button>
            </div>
          )}

          <div className={`viewport-deck ${activeTab === 'split' ? 'is-split' : ''} flex-1 flex overflow-hidden relative`}>
            {/* Split View: 2D Canvas + 3D Viewer */}
            {activeTab === 'split' && (
              <>
                <Floorplan2DCanvas
                  imageUrl={imageUrl}
                  layout={layout}
                  frontWallId={frontWallId}
                  onSelectFrontWall={handleSelectFrontWall}
                  onCalibrateScale={handleCalibrateScale}
                  isCalibrating={isCalibrating}
                  setIsCalibrating={setIsCalibrating}
                  onUpdateLayout={handleLayoutEdit}
                  isSplit={true}
                  selectedRoom={selectedRoom}
                  onSelectRoom={handleSelectRoom}
                />
                <Viewer3D
                  layout={layout}
                  frontWallId={frontWallId}
                  numFloors={numFloors}
                  isSplit={true}
                  selectedRoom={selectedRoom}
                  onSelectRoom={handleSelectRoom}
                  furnishingData={furnishingData}
                  onSelectFurniture={handleSelectFurniture}
                  onClearRoomFurnishings={handleClearDesign}
                  facadeFocusTrigger={facadeFocusTrigger}
                  roomFocusTrigger={roomFocusTrigger}
                />
              </>
            )}

            {/* 2D Only View */}
            {activeTab === '2d' && (
              <Floorplan2DCanvas
                imageUrl={imageUrl}
                layout={layout}
                frontWallId={frontWallId}
                onSelectFrontWall={handleSelectFrontWall}
                onCalibrateScale={handleCalibrateScale}
                isCalibrating={isCalibrating}
                setIsCalibrating={setIsCalibrating}
                onUpdateLayout={handleLayoutEdit}
                isSplit={false}
                selectedRoom={selectedRoom}
                onSelectRoom={handleSelectRoom}
              />
            )}

            {/* 3D Only View */}
            {activeTab === '3d' && (
              <Viewer3D
                layout={layout}
                frontWallId={frontWallId}
                numFloors={numFloors}
                isSplit={false}
                selectedRoom={selectedRoom}
                onSelectRoom={handleSelectRoom}
                furnishingData={furnishingData}
                onSelectFurniture={handleSelectFurniture}
                onClearRoomFurnishings={handleClearDesign}
                facadeFocusTrigger={facadeFocusTrigger}
                roomFocusTrigger={roomFocusTrigger}
              />
            )}

            {/* Room Design Floating Panel (Stage 3) */}
            {selectedRoom && (
              <RoomDesignPanel
                selectedRoom={selectedRoom}
                onClose={() => setSelectedRoom(null)}
                onGenerateDesign={handleGenerateDesign}
                onRegenerateDesign={handleRegenerateDesign}
                onClearDesign={handleClearDesign}
                onInspectRoom={inspectRoomIn3D}
                onSelectFurniture={handleSelectFurniture}
                onPresetChange={handleRoomPreset}
                furnishingData={furnishingData}
                isDesigning={isDesigning}
                onRelabelRoom={handleRelabelRoom}
              />
            )}
          </div>

          {/* Bottom Summary Bar */}
          <MetricsSummary layout={layout} frontWallId={frontWallId} selectedRoom={selectedRoom} onSelectRoom={handleSelectRoom} />
        </main>
      </div>
      </section>
      )}

      {catalogSelection && furnishingData[catalogSelection.roomId] === catalogSelection.design && (
        <React.Suspense fallback={null}>
          <FurnitureCatalog key={`${catalogSelection.roomId}:${catalogSelection.item.item_id}`}
            item={catalogSelection.item} choices={catalogSelection.choices} preset={catalogSelection.design.preset}
            loading={catalogSelection.loading} error={catalogSelection.error}
            onApply={applyFurnitureChoice} onClose={() => setCatalogSelection(null)} />
        </React.Suspense>
      )}
      {/* Facade Synthesis Studio Modal */}
      <FacadeModal
        isOpen={showFacadeModal}
        onClose={() => setShowFacadeModal(false)}
        facadeData={facadeDraft}
        onSwitchStyle={handleGenerateFacade}
        isSwitchingStyle={isGeneratingFacade}
        isApplied={appliedFacade === facadeDraft}
        onApply={() => applyFacade(facadeDraft)}
        onFocusIn3D={viewFacadeIn3D}
      />
    </div>
  );
}
