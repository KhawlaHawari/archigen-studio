import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { createRoofGeometries } from '../geometry/roofGeometry.js';
import { buildingLevels, clipRoomFloor, compileWallAssembly, exposedRoof, floorFootprints,
  floorSurfaceGeometry, tileFloor, wallAssemblyGeometry, fitHouseOpenings } from '../geometry/houseGeometry.js';
import { createFacadeGeometry, wallFacadeFrame, facadeOpeningPosition, facadeBounds } from '../geometry/facadeGeometry.js';
import { STOREY_HEIGHT_M } from '../geometry/buildingDimensions.js';
import { applyFurnitureFinish } from '../geometry/furnitureAppearance.js';
import { buildFurnitureShape } from '../geometry/furnitureShapes.js';
import { fitCustomFurnitureAsset } from '../geometry/customFurnitureAsset.js';
import { RotateCcw, Box, Eye, Layers, Sun, Sparkles, Compass, Maximize2, Activity, Home } from 'lucide-react';

export default function Viewer3D({
  layout,
  frontWallId,
  numFloors = 1,
  isSplit = false,
  selectedRoom = null,
  onSelectRoom = null,
  onSelectFurniture = null,
  furnishingData = {},
  onClearRoomFurnishings = null,
  facadeFocusTrigger = 0,
  roomFocusTrigger = 0
}) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const furnitureGroupRef = useRef(null);
  const roofGroupRef = useRef(null);
  const gltfLoaderRef = useRef(new GLTFLoader());
  const onSelectFurnitureRef = useRef(onSelectFurniture);
  useEffect(() => { onSelectFurnitureRef.current = onSelectFurniture; }, [onSelectFurniture]);

  const [wireframe, setWireframe] = useState(false);
  const [activeLevel, setActiveLevel] = useState('all');
  const [unresolvedOpenings, setUnresolvedOpenings] = useState(0);
  const [showRoof, setShowRoof] = useState(false);
  const [roofStyle, setRoofStyle] = useState('slab'); // 'slab' | 'pitched'
  const [floorStyle, setFloorStyle] = useState('oak'); // 'oak' | 'marble' | 'adaptive'
  const [viewPreset, setViewPreset] = useState('perspective'); // 'perspective' | 'isometric' | 'top' | 'room'
  const [showFurniture, setShowFurniture] = useState(true);
  const [currentFps, setCurrentFps] = useState(60);
  const [graphicsQuality, setGraphicsQuality] = useState('high'); // 'high' | 'low'

  // Store layout centers for furniture positioning
  const centersRef = useRef({ centerX: 0, centerZ: 0 });
  const buildingBoundsRef = useRef(new THREE.Box3());

  // PBR texture cache ref
  const pbrTexturesRef = useRef({
    loaded: false,
    oak: null,
    marble: null,
    stucco: null
  });

  const sunLightRef = useRef(null);
  const shadowPlaneRef = useRef(null);
  const envMapRef = useRef(null);

  // 1. Scene Setup & Lifecycle
  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    // Cinematic night backdrop (matches the UI palette)
    scene.background = new THREE.Color(0x0b0e12);
    scene.fog = new THREE.FogExp2(0x0b0e12, 0.012);
    sceneRef.current = scene;

    // Furniture container group
    const furnitureGroup = new THREE.Group();
    furnitureGroup.name = "furnitureGroup";
    scene.add(furnitureGroup);
    furnitureGroupRef.current = furnitureGroup;

    // Roof container group (can be activated / deactivated)
    const roofGroup = new THREE.Group();
    roofGroup.name = "roofGroup";
    roofGroup.visible = false;
    scene.add(roofGroup);
    roofGroupRef.current = roofGroup;

    // Camera Setup (Natural 45° Architectural FOV)
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(16, 19, 23);
    cameraRef.current = camera;

    // WebGL Renderer with ACES Filmic Tone Mapping and Soft Shadows
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls with free panning and smooth damping
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI / 2 + 0.02; // Prevent dipping below ground
    controls.screenSpacePanning = true;
    controls.enablePan = true;
    controls.panSpeed = 1.0;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };
    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN
    };
    controlsRef.current = controls;

    // Prevent default context menu on right click so right-drag panning is instantaneous
    const handleContextMenu = (e) => e.preventDefault();
    renderer.domElement.addEventListener('contextmenu', handleContextMenu);

    // Keyboard Navigation: WASD / Arrow Keys for free walking & exploration
    const keysDown = new Set();
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
        keysDown.add(e.code);
      }
    };
    const handleKeyUp = (e) => {
      keysDown.delete(e.code);
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    // Environment Map (IBL Realistic Studio Reflections)
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    pmremGenerator.compileEquirectangularShader();
    new RGBELoader().load(
      '/showcase/sky/qwantani_late_afternoon_puresky_1k.hdr',
      (hdrTexture) => {
        const envMap = pmremGenerator.fromEquirectangular(hdrTexture).texture;
        envMapRef.current = envMap;
        if (graphicsQuality === 'high') {
          scene.environment = envMap;
        }
        hdrTexture.dispose();
        pmremGenerator.dispose();
      },
      undefined,
      (err) => {
        console.warn('HDRI environment map fallback to procedural illumination:', err);
      }
    );

    // Warm Architectural Studio Lighting
    const hemiLight = new THREE.HemisphereLight(0xfffdf7, 0xdcd6cb, 0.85);
    scene.add(hemiLight);

    // Directional Key Sun Light
    const sunLight = new THREE.DirectionalLight(0xfff8eb, 1.45);
    sunLight.position.set(24, 40, 20);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.radius = 2.4; // Soft penumbra
    sunLight.shadow.bias = -0.00015;
    sunLight.shadow.normalBias = 0.025;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 130;
    sunLight.shadow.camera.left = -28;
    sunLight.shadow.camera.right = 28;
    sunLight.shadow.camera.top = 28;
    sunLight.shadow.camera.bottom = -28;
    scene.add(sunLight);
    sunLightRef.current = sunLight;

    // Secondary Soft Cool Rim/Fill Light (ambient shadow illumination)
    const rimLight = new THREE.DirectionalLight(0xdbeafe, 0.40);
    rimLight.position.set(-22, 16, -20);
    scene.add(rimLight);

    // Ground Contact Shadow Receiver Plane
    const shadowPlaneGeo = new THREE.PlaneGeometry(80, 80);
    const shadowPlaneMat = new THREE.ShadowMaterial({ opacity: 0.22 });
    const shadowPlane = new THREE.Mesh(shadowPlaneGeo, shadowPlaneMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -0.005;
    shadowPlane.receiveShadow = true;
    scene.add(shadowPlane);
    shadowPlaneRef.current = shadowPlane;

    // Architectural Ground Grid
    const grid = new THREE.GridHelper(50, 50, 0xd4943a, 0x232a33);
    grid.position.y = -0.01;
    scene.add(grid);

    // Load PBR Textures
    const texLoader = new THREE.TextureLoader();
    const loadTex = (url, isColor = false) => {
      const tex = texLoader.load(url);
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      if (isColor) tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    };

    pbrTexturesRef.current = {
      loaded: true,
      oak: {
        map: loadTex('/showcase/tex-1k/oak_wood_planks/diff.jpg', true),
        normalMap: loadTex('/showcase/tex-1k/oak_wood_planks/nor.jpg'),
        arm: loadTex('/showcase/tex-1k/oak_wood_planks/arm.jpg')
      },
      marble: {
        map: loadTex('/showcase/tex-1k/marble_01/diff.jpg', true),
        normalMap: loadTex('/showcase/tex-1k/marble_01/nor.jpg'),
        arm: loadTex('/showcase/tex-1k/marble_01/arm.jpg')
      },
      stucco: {
        map: loadTex('/showcase/tex-1k/white_stucco/diff.jpg', true),
        normalMap: loadTex('/showcase/tex-1k/white_stucco/nor.jpg'),
        arm: loadTex('/showcase/tex-1k/white_stucco/arm.jpg')
      }
    };

    window.THREE = THREE;
    window.__ARCHIGEN_VIEWER__ = {
      scene,
      camera,
      controls,
      renderer,
      pbrTexturesRef,
      sunLightRef,
      shadowPlaneRef,
      envMapRef,
      setGraphicsQuality: (q) => setGraphicsQuality(q),
      getGraphicsQuality: () => graphicsQuality
    };

    // Raycaster for room clicking in 3D
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    let pointerStartX = 0;
    let pointerStartY = 0;
    const handlePointerDown = (e) => {
      pointerStartX = e.clientX;
      pointerStartY = e.clientY;
    };

    const handleClick = (e) => {
      if (e.button !== 0 || !containerRef.current || !cameraRef.current || !sceneRef.current) return;
      const dist = Math.hypot(e.clientX - pointerStartX, e.clientY - pointerStartY);
      if (dist > 6) return; // User was dragging to rotate or pan, not clicking

      const rect = containerRef.current.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, cameraRef.current);
      const intersects = raycaster.intersectObjects(sceneRef.current.children, true);

      let behindWall = false;
      for (const hit of intersects) {
        if (['wall_assembly', 'facade'].includes(hit.object.userData.type)) behindWall = true;
        if (!behindWall && onSelectFurnitureRef.current) {
          for (let object = hit.object; object; object = object.parent) {
            if (object.userData.isFurnitureMesh) {
              onSelectFurnitureRef.current(object.userData.roomId, object.userData.itemId);
              return;
            }
          }
        }
        if (hit.object.userData?.isFloorMesh && hit.object.userData?.room) {
          if (onSelectRoom) {
            onSelectRoom(hit.object.userData.room);
          }
          break;
        }
      }
    };

    const handleDoubleClick = (e) => {
      if (!containerRef.current || !cameraRef.current || !sceneRef.current || !controlsRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, cameraRef.current);
      const intersects = raycaster.intersectObjects(sceneRef.current.children, true);
      for (const hit of intersects) {
        if (hit.point) {
          // Double click centers the camera orbit point on the targeted room/surface
          controlsRef.current.target.set(hit.point.x, Math.max(0.2, hit.point.y), hit.point.z);
          controlsRef.current.update();
          break;
        }
      }
    };

    renderer.domElement.addEventListener('pointerdown', handlePointerDown);
    renderer.domElement.addEventListener('pointerup', handleClick);
    renderer.domElement.addEventListener('dblclick', handleDoubleClick);

    // Animation & Live FPS Measurement Loop
    let animationFrameId;
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let lastWalkTime = performance.now();

    const animate = (now) => {
      animationFrameId = requestAnimationFrame(animate);

      // WASD / Arrow Key camera translation
      const dt = Math.min(0.1, (now - lastWalkTime) / 1000);
      lastWalkTime = now;

      if (keysDown.size > 0 && cameraRef.current && controlsRef.current) {
        const cam = cameraRef.current;
        const ctrl = controlsRef.current;

        const forward = new THREE.Vector3();
        cam.getWorldDirection(forward);
        forward.y = 0;
        if (forward.lengthSq() > 0.001) forward.normalize();

        const right = new THREE.Vector3();
        right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

        const move = new THREE.Vector3();
        if (keysDown.has('KeyW') || keysDown.has('ArrowUp')) move.add(forward);
        if (keysDown.has('KeyS') || keysDown.has('ArrowDown')) move.sub(forward);
        if (keysDown.has('KeyD') || keysDown.has('ArrowRight')) move.add(right);
        if (keysDown.has('KeyA') || keysDown.has('ArrowLeft')) move.sub(right);
        if (keysDown.has('KeyE')) move.y += 1;
        if (keysDown.has('KeyQ')) move.y -= 1;

        if (move.lengthSq() > 0) {
          const speed = 7.0; // 7 m/s walk speed
          move.normalize().multiplyScalar(speed * dt);
          cam.position.add(move);
          ctrl.target.add(move);
        }
      }

      controls.update();
      renderer.render(scene, camera);

      frameCount++;
      if (now - lastFpsTime >= 500) {
        const measured = Math.round((frameCount * 1000) / (now - lastFpsTime));
        setCurrentFps(measured);
        frameCount = 0;
        lastFpsTime = now;
      }
    };
    animate(performance.now());

    // Resize Handler
    const handleResize = () => {
      if (!containerRef.current || !renderer || !camera) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      if (w < 1 || h < 1) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(containerRef.current);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      resizeObserver.disconnect();
      if (renderer.domElement) {
        renderer.domElement.removeEventListener('contextmenu', handleContextMenu);
        renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
        renderer.domElement.removeEventListener('pointerup', handleClick);
        renderer.domElement.removeEventListener('dblclick', handleDoubleClick);
      }
      renderer.dispose();
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, []);

  // Handle Graphics Quality Mode (High: PBR, HDRI, Soft Shadows vs Low: Fast unshadowed diffuse for low-spec/CPU)
  useEffect(() => {
    if (!rendererRef.current || !sceneRef.current) return;
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const isHigh = graphicsQuality === 'high';

    renderer.shadowMap.enabled = isHigh;
    if (sunLightRef.current) {
      sunLightRef.current.castShadow = isHigh;
    }
    if (shadowPlaneRef.current) {
      shadowPlaneRef.current.visible = isHigh;
    }
    if (envMapRef.current) {
      scene.environment = isHigh ? envMapRef.current : null;
    }
    renderer.toneMapping = isHigh ? THREE.ACESFilmicToneMapping : THREE.LinearToneMapping;
    renderer.toneMappingExposure = isHigh ? 1.05 : 1.0;
    renderer.shadowMap.needsUpdate = true;
  }, [graphicsQuality]);

  // Handle Roof Visibility Toggle
  useEffect(() => {
    if (roofGroupRef.current) {
      roofGroupRef.current.visible = showRoof;
    }
  }, [showRoof]);

  const fitCamera = (mode) => {
    const camera = cameraRef.current, controls = controlsRef.current;
    if (!camera || !controls || buildingBoundsRef.current.isEmpty()) return;
    const bounds = buildingBoundsRef.current;
    const target = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const halfFov = THREE.MathUtils.degToRad(camera.fov) / 2;
    if (mode === 'top') {
      const distance = Math.max(size.z, size.x / camera.aspect) / (2 * Math.tan(halfFov));
      camera.up.set(0, 0, -1);
      camera.position.set(target.x, bounds.max.y + distance * 1.24, target.z);
    } else {
      const limitingFov = Math.atan(Math.tan(halfFov) * Math.min(1, camera.aspect));
      const distance = size.length() / (2 * Math.sin(limitingFov)) * 1.15;
      const direction = mode === 'isometric' ? new THREE.Vector3(1, 1, 1) : new THREE.Vector3(1.15, 1.35, 1.35);
      camera.up.set(0, 1, 0);
      camera.position.copy(direction.normalize().multiplyScalar(distance).add(target));
    }
    controls.target.copy(target);
    controls.update();
  };

  // Update Building Meshes when Layout / Facade Texture / selectedRoom / graphicsQuality changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !layout) return;

    // Remove existing building meshes (tagged with isBuildingMesh)
    const toRemove = [];
    scene.traverse((obj) => {
      if (obj.userData?.isBuildingMesh) {
        toRemove.push(obj);
      }
    });
    const sharedTextures = new Set(Object.values(pbrTexturesRef.current).flatMap(set =>
      set && typeof set === 'object' ? Object.values(set).filter(value => value?.isTexture) : []));
    toRemove.forEach((obj) => {
      obj.parent?.remove(obj);
      obj.traverse(child => {
        child.geometry?.dispose();
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.filter(Boolean).forEach(m => {
          for (const value of Object.values(m)) if (value?.isTexture && !sharedTextures.has(value)) value.dispose();
          m.dispose();
        });
      });
    });

    const levels = buildingLevels(layout, numFloors);
    const visibleLevels = levels.filter(level => activeLevel === 'all' || level.id === activeLevel);
    if (!visibleLevels.length) visibleLevels.push(...levels);
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const level of levels) {
      for (const feature of [...(level.walls || []), ...(level.rooms || [])]) {
        for (const [x, z] of feature.polygon_meters || []) {
          minX = Math.min(minX, x + level.offset[0]);
          maxX = Math.max(maxX, x + level.offset[0]);
          minZ = Math.min(minZ, z + level.offset[1]);
          maxZ = Math.max(maxZ, z + level.offset[1]);
        }
      }
    }
    const centerX = (minX + maxX) / 2 || 0;
    const centerZ = (minZ + maxZ) / 2 || 0;
    centersRef.current = { centerX, centerZ };
    const pbr = pbrTexturesRef.current;
    const highQuality = graphicsQuality === 'high';
    const wallMat = new THREE.MeshStandardMaterial({
      color: highQuality ? 0xf3f0e9 : 0xe7e4dc,
      ...(highQuality && pbr.stucco?.map ? { map: pbr.stucco.map, normalMap: pbr.stucco.normalMap,
        normalScale: new THREE.Vector2(.35, .35), roughnessMap: pbr.stucco.arm } : {}),
      roughness: .85, metalness: .02, wireframe,
    });
    const floorMaterial = (texture, color, roughness) => new THREE.MeshStandardMaterial({
      color: highQuality && texture?.map ? 0xffffff : color,
      ...(highQuality && texture?.map ? { map: texture.map, normalMap: texture.normalMap,
        normalScale: new THREE.Vector2(.45, .45), roughnessMap: texture.arm } : {}),
      roughness, wireframe,
    });
    const oakMat = floorMaterial(pbr.oak, 0xc8aa88, .68);
    const tileMat = floorMaterial(pbr.marble, 0xe2e4e5, .42);
    const slabEdgeMat = new THREE.MeshStandardMaterial({ color: 0xd9d1c4, roughness: .92, wireframe });
    const pickMat = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, side: THREE.DoubleSide });
    const selectedMat = new THREE.MeshBasicMaterial({ color: 0x50a9dc, transparent: true, opacity: .27,
      depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1, side: THREE.DoubleSide });
    let unresolved = 0;
    for (const level of visibleLevels) {
      const walls = level.walls || [];
      const rooms = level.rooms || [];
      const wallHeight = level.height;
      const levelCenterX = centerX - level.offset[0];
      const levelCenterZ = centerZ - level.offset[1];
      const levelStart = scene.children.length;
      const facadeSpecs = new Map();
      const textureLoader = new THREE.TextureLoader();
      walls.forEach(wall => {
        const data = wall.facade_texture;
        if (!data?.texture_base64) return;
        const frame = data.facade_frame || wallFacadeFrame(wall);
        if (!frame) return;
        facadeSpecs.set(wall.id, { ...data, frame });
      });

      const footprints = floorFootprints(level);
      const bathroomRooms = rooms.filter(room => /bath|toilet/i.test(room.room_type || ''));
      const tileAreas = floorStyle === 'adaptive' ? tileFloor(bathroomRooms, footprints) : [];
      for (const elevation of level.floorElevations) {
        for (const footprint of footprints) {
          const slab = new THREE.Mesh(floorSurfaceGeometry(footprint, levelCenterX, levelCenterZ, .16),
            [floorStyle === 'marble' ? tileMat : oakMat, slabEdgeMat]);
          slab.position.y = elevation + .02;
          slab.receiveShadow = highQuality;
          slab.userData = { isBuildingMesh: true, type: 'foundation_slab', levelId: level.id };
          scene.add(slab);
        }
        for (const area of tileAreas) {
          const tile = new THREE.Mesh(floorSurfaceGeometry(area, levelCenterX, levelCenterZ), tileMat);
          tile.position.y = elevation + .0205;
          tile.receiveShadow = highQuality;
          tile.userData = { isBuildingMesh: true, type: 'floor_finish' };
          scene.add(tile);
        }
        for (const room of rooms) {
          for (const region of clipRoomFloor(room, footprints)) {
            const surface = new THREE.Mesh(floorSurfaceGeometry(region, levelCenterX, levelCenterZ),
              room.id === selectedRoom?.id ? selectedMat : pickMat);
            surface.position.y = elevation + .021;
            surface.userData = { isBuildingMesh: true, isFloorMesh: true, room };
            scene.add(surface);
          }
        }
      }

      const openingCandidates = [];
      const scaleMperPx = layout.metadata?.scale_meters_per_pixel || .03;
      const hostedByFacade = new Set();
      facadeSpecs.forEach((spec, wallId) => {
        for (const opening of spec.openings || []) {
          const heightScale = wallHeight / (spec.wall_height_m || spec.uv_bounds?.wall_height_m || wallHeight);
          hostedByFacade.add(`${wallId}:${opening.id}`);
          openingCandidates.push({ ...opening, associated_wall: wallId,
            position_meters: facadeOpeningPosition(opening, spec.frame),
            facadeFrame: spec.frame, sill_m: opening.sill_m * heightScale, height_m: opening.height_m * heightScale });
        }
      });
      const byWallId = new Map(walls.map(wall => [wall.id, wall]));
      for (const opening of fitHouseOpenings(level, scaleMperPx)) {
        if (hostedByFacade.has(`${opening.associated_wall}:${opening.id}`)) continue;
        for (const elevation of level.floorElevations) {
          if (elevation > 0 && opening.type === 'door' && byWallId.get(opening.associated_wall)?.is_exterior) continue;
          openingCandidates.push({ ...opening,
            id: elevation === 0 ? opening.id : `${opening.id}-${elevation}`,
            sill_m: elevation + opening.sill_m });
        }
      }
      const assembly = compileWallAssembly(walls, openingCandidates, wallHeight);
      for (const [wallId, spec] of facadeSpecs) {
        const { frame } = spec;
        const texture = textureLoader.load(spec.texture_base64);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.repeat.set(1, 1);
        // Preserve higher-resolution detail at oblique viewing angles; UVs remain normalized.
        texture.anisotropy = Math.min(8, rendererRef.current?.capabilities.getMaxAnisotropy() || 1);
        const material = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.7, metalness: 0.02, side: THREE.FrontSide, wireframe });
        const geometry = createFacadeGeometry(frame, spec.uv_bounds || facadeBounds(frame.length_m, wallHeight), wallHeight, levelCenterX, levelCenterZ, assembly.hosted.filter(opening => opening.facadeFrame === frame));
        const skin = new THREE.Mesh(geometry, material);
        skin.castShadow = graphicsQuality === 'high';
        skin.receiveShadow = graphicsQuality === 'high';
        skin.userData = { isBuildingMesh: true, type: 'facade', wallId, frame, uvBounds: spec.uv_bounds };
        scene.add(skin);
      }
      unresolved += assembly.unresolved.length;
      const wallMesh = new THREE.Mesh(wallAssemblyGeometry(assembly.bands, levelCenterX, levelCenterZ), wallMat);
      wallMesh.castShadow = wallMesh.receiveShadow = highQuality;
      wallMesh.userData = { isBuildingMesh: true, type: 'wall_assembly', levelId: level.id,
        hostedOpenings: assembly.hosted.length, unresolvedOpenings: assembly.unresolved };
      scene.add(wallMesh);
      const doors = assembly.hosted.filter(opening => opening.type === 'door');
      const windows = assembly.hosted.filter(opening => opening.type === 'window');
      const openingPlacement = opening => opening.facadeFrame ? {
        ...opening.host, center: opening.position_meters,
        angleRad: Math.atan2(-opening.facadeFrame.direction[1], opening.facadeFrame.direction[0]),
      } : opening.host;

      // D. 3D Doors & Wall Lintels (Green Dots on Blueprint)
      doors.forEach((door, dIdx) => {
        const { center: [x, z], angleRad, wallThickness: wallThick } = openingPlacement(door);
        const snappedX = x - levelCenterX;
        const snappedZ = z - levelCenterZ;
        const { width_m: doorWidth, height_m: doorHeight } = door;

        // 2. 3D Architectural Door (Frame, Hinged Leaf, Lever Handles, Floor Threshold)
        const doorGroup = new THREE.Group();
        doorGroup.position.set(snappedX, door.sill_m, snappedZ);
        doorGroup.rotation.y = angleRad;

        // Frame Material (Architectural Charcoal Wood)
        const frameMat = new THREE.MeshStandardMaterial({
          color: 0x221f1d,
          roughness: 0.65,
          metalness: 0.08
        });
        const frameThick = Math.min(.045, doorWidth * .08);
        const frameDepth = wallThick + 0.02;

        // Left Jamb
        const jambLGeo = new THREE.BoxGeometry(frameThick, doorHeight, frameDepth);
        const jambL = new THREE.Mesh(jambLGeo, frameMat);
        jambL.position.set(-doorWidth / 2 + frameThick / 2, doorHeight / 2, 0);
        jambL.castShadow = (graphicsQuality === 'high');
        jambL.receiveShadow = (graphicsQuality === 'high');
        doorGroup.add(jambL);

        // Right Jamb
        const jambRGeo = new THREE.BoxGeometry(frameThick, doorHeight, frameDepth);
        const jambR = new THREE.Mesh(jambRGeo, frameMat);
        jambR.position.set(doorWidth / 2 - frameThick / 2, doorHeight / 2, 0);
        jambR.castShadow = (graphicsQuality === 'high');
        jambR.receiveShadow = (graphicsQuality === 'high');
        doorGroup.add(jambR);

        // Top Header Casing
        const headGeo = new THREE.BoxGeometry(doorWidth, frameThick, frameDepth);
        const headMesh = new THREE.Mesh(headGeo, frameMat);
        headMesh.position.set(0, doorHeight - frameThick / 2, 0);
        headMesh.castShadow = (graphicsQuality === 'high');
        headMesh.receiveShadow = (graphicsQuality === 'high');
        doorGroup.add(headMesh);

        // Hinged Door Leaf (Warm Oak Finish)
        const leafWidth = doorWidth - frameThick * 2;
        const leafHeight = doorHeight - frameThick;
        const leafThick = 0.04;
        const leafMat = new THREE.MeshStandardMaterial({
          color: 0x9b6b43, // Architectural warm oak
          roughness: 0.55,
          metalness: 0.05
        });

        const hingePivot = new THREE.Group();
        hingePivot.position.set(-doorWidth / 2 + frameThick, 0, 0);
        hingePivot.rotation.y = 0;

        const leafGeo = new THREE.BoxGeometry(leafWidth, leafHeight, leafThick);
        const leafMesh = new THREE.Mesh(leafGeo, leafMat);
        leafMesh.position.set(leafWidth / 2, leafHeight / 2 + 0.01, 0);
        leafMesh.castShadow = (graphicsQuality === 'high');
        leafMesh.receiveShadow = (graphicsQuality === 'high');
        hingePivot.add(leafMesh);

        // Polished Brass Lever Handle
        const handleMat = new THREE.MeshStandardMaterial({
          color: 0xd4af37, // Polished brass hardware
          metalness: 0.85,
          roughness: 0.20
        });
        const leverGeo = new THREE.BoxGeometry(0.10, 0.02, 0.02);
        const leverF = new THREE.Mesh(leverGeo, handleMat);
        leverF.position.set(leafWidth - 0.07, doorHeight * .51, leafThick / 2 + 0.02);
        hingePivot.add(leverF);

        const leverB = new THREE.Mesh(leverGeo, handleMat);
        leverB.position.set(leafWidth - 0.07, doorHeight * .51, -leafThick / 2 - 0.02);
        hingePivot.add(leverB);

        doorGroup.add(hingePivot);

        // Floor threshold strip
        const threshGeo = new THREE.BoxGeometry(doorWidth, 0.015, wallThick);
        const threshMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.5, roughness: 0.4 });
        const threshMesh = new THREE.Mesh(threshGeo, threshMat);
        threshMesh.position.set(0, 0.008, 0);
        doorGroup.add(threshMesh);

        doorGroup.userData = { isBuildingMesh: true, type: 'door', doorId: door.id || `door_${dIdx}`, facadeOpening: door.facadeFrame ? door : null };
        scene.add(doorGroup);
      });

      // E. 3D Windows, Wall Sills & Lintels (Blue Squares on Blueprint)
      windows.forEach((win, wIdx) => {
        const { center: [x, z], angleRad, wallThickness: wallThick } = openingPlacement(win);
        const snappedX = x - levelCenterX;
        const snappedZ = z - levelCenterZ;
        const { width_m: winWidth, sill_m: sillHeight, height_m: winHeight } = win;
        const winTop = sillHeight + winHeight;

        // 3. 3D Architectural Window (Outer Frame, Mullion, Exterior Sill Shelf, Reflective Glass Panes)
        const winGroup = new THREE.Group();
        winGroup.position.set(snappedX, 0, snappedZ);
        winGroup.rotation.y = angleRad;

        const frameMat = new THREE.MeshStandardMaterial({
          color: 0x1e242b, // Architectural dark bronze / black aluminum
          metalness: 0.65,
          roughness: 0.35
        });
        const frameThick = Math.min(.035, winWidth * .075, winHeight * .075);
        const frameDepth = wallThick * 0.70;
        const paneCount = winWidth >= .82 ? 2 : 1;

        // Frame profiles (bottom, top, left, right)
        const botGeo = new THREE.BoxGeometry(winWidth, frameThick, frameDepth);
        const botMesh = new THREE.Mesh(botGeo, frameMat);
        botMesh.position.set(0, sillHeight + frameThick / 2, 0);
        winGroup.add(botMesh);

        const topGeo = new THREE.BoxGeometry(winWidth, frameThick, frameDepth);
        const topMesh = new THREE.Mesh(topGeo, frameMat);
        topMesh.position.set(0, winTop - frameThick / 2, 0);
        winGroup.add(topMesh);

        const leftGeo = new THREE.BoxGeometry(frameThick, winHeight, frameDepth);
        const leftMesh = new THREE.Mesh(leftGeo, frameMat);
        leftMesh.position.set(-winWidth / 2 + frameThick / 2, sillHeight + winHeight / 2, 0);
        winGroup.add(leftMesh);

        const rightGeo = new THREE.BoxGeometry(frameThick, winHeight, frameDepth);
        const rightMesh = new THREE.Mesh(rightGeo, frameMat);
        rightMesh.position.set(winWidth / 2 - frameThick / 2, sillHeight + winHeight / 2, 0);
        winGroup.add(rightMesh);

        if (paneCount === 2) {
          const mullGeo = new THREE.BoxGeometry(frameThick, winHeight - frameThick * 2, frameDepth);
          const mullMesh = new THREE.Mesh(mullGeo, frameMat);
          mullMesh.position.set(0, sillHeight + winHeight / 2, 0);
          winGroup.add(mullMesh);
        }

        // Exterior Sill Shelf (protruding drip lip)
        const shelfGeo = new THREE.BoxGeometry(winWidth + 0.08, 0.035, wallThick + 0.06);
        const shelfMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.50, metalness: 0.30 });
        const shelfMesh = new THREE.Mesh(shelfGeo, shelfMat);
        shelfMesh.position.set(0, sillHeight - 0.015, 0.02);
        winGroup.add(shelfMesh);

        // Glass Panes (Translucent PBR Glass)
        const glassMat = (graphicsQuality === 'high')
          ? new THREE.MeshPhysicalMaterial({
              color: 0xe0f2fe,
              transparent: true,
              opacity: 0.35,
              roughness: 0.05,
              metalness: 0.1,
              transmission: 0.85,
              ior: 1.5,
              reflectivity: 0.9,
              envMapIntensity: 1.5,
              side: THREE.DoubleSide,
              depthWrite: false
            })
          : new THREE.MeshStandardMaterial({
              color: 0x93c5fd,
              transparent: true,
              opacity: 0.40,
              roughness: 0.1,
              metalness: 0.1,
              side: THREE.DoubleSide
            });

        const paneWidth = (winWidth - frameThick * (paneCount + 1)) / paneCount;
        const paneHeight = winHeight - frameThick * 2;
        const paneGeo = new THREE.BoxGeometry(paneWidth, paneHeight, 0.01);

        for (let paneIndex = 0; paneIndex < paneCount; paneIndex++) {
          const pane = new THREE.Mesh(paneGeo, glassMat);
          pane.position.set(-winWidth / 2 + frameThick + paneWidth / 2 + paneIndex * (paneWidth + frameThick),
            sillHeight + winHeight / 2, 0);
          winGroup.add(pane);
        }

        winGroup.userData = { isBuildingMesh: true, type: 'window', winId: win.id || `win_${wIdx}`, facadeOpening: win.facadeFrame ? win : null };
        scene.add(winGroup);
      });

      for (const mesh of scene.children.slice(levelStart)) mesh.position.y += level.elevation;
    }
    setUnresolvedOpenings(unresolved);

    // F. 3D Unified Architectural Roof (The Sims-Style Continuous Building Envelope)
    if (roofGroupRef.current) {
      const roofGroup = roofGroupRef.current;
      while (roofGroup.children.length > 0) {
        const child = roofGroup.children[0];
        roofGroup.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
          else child.material.dispose();
        }
      }

      // High-End Architectural Roof Materials
      const roofSlabMat = new THREE.MeshStandardMaterial({
        color: 0x222a36, // Deep architectural slate / dark anthracite
        roughness: 0.65,
        metalness: 0.12,
        side: THREE.DoubleSide,
        wireframe: wireframe
      });

      const pitchedRoofMat = new THREE.MeshStandardMaterial({
        color: 0x2a3342, // Rich dark architectural shingles
        roughness: 0.70,
        metalness: 0.08,
        side: THREE.DoubleSide,
        wireframe: wireframe
      });

      const soffitMat = new THREE.MeshStandardMaterial({
        color: 0xf8fafc, // Clean off-white ceiling soffit underneath
        roughness: 0.90,
        metalness: 0.02,
        side: THREE.DoubleSide
      });

      for (const level of visibleLevels) {
        const above = visibleLevels.some(other => other.elevation > level.elevation);
        const footprints = exposedRoof(level, visibleLevels);
        if (!footprints.length) continue;
        const style = above ? 'slab' : roofStyle;
        const roofs = createRoofGeometries({ envelope: footprints }, {
          centerX: centerX - level.offset[0], centerZ: centerZ - level.offset[1],
          wallHeight: level.height + level.elevation, style, overhang: above ? .08 : .22,
        });
        roofs.forEach(({ geometry, footprint, fallback }, index) => {
          const roof = new THREE.Mesh(geometry, [style === 'pitched' && !fallback ? pitchedRoofMat : roofSlabMat, soffitMat]);
          roof.castShadow = roof.receiveShadow = highQuality;
          roof.userData = { isRoofMesh: true, type: 'roof', style, component: index, footprint, fallback, levelId: level.id };
          roofGroup.add(roof);
          if (fallback) console.warn('Roof uses closed slab fallback:', fallback);
        });
      }

      roofGroup.visible = showRoof;
    }

    const bounds = new THREE.Box3();
    for (const mesh of scene.children) {
      if (mesh.userData.isBuildingMesh) bounds.expandByObject(mesh);
    }
    if (roofGroupRef.current) bounds.expandByObject(roofGroupRef.current);
    buildingBoundsRef.current = bounds;
    if (['perspective', 'top', 'isometric'].includes(viewPreset)) fitCamera(viewPreset);
  }, [layout, numFloors, wireframe, activeLevel, selectedRoom, graphicsQuality, roofStyle, floorStyle]);

  // Update 3D Furniture Assets when furnishingData changes
  useEffect(() => {
    const furnitureGroup = furnitureGroupRef.current;
    if (!furnitureGroup) return;

    // Helper to completely free WebGL memory for an object
    const disposeHierarchy = (obj) => {
      obj.traverse((child) => {
        if (child.isMesh) {
          if (child.geometry) child.geometry.dispose();
          if (child.material) {
            if (Array.isArray(child.material)) {
              child.material.forEach((m) => {
                if (m.map) m.map.dispose();
                m.dispose();
              });
            } else {
              if (child.material.map) child.material.map.dispose();
              child.material.dispose();
            }
          }
        }
      });
    };

    // Remove all old furniture meshes and dispose WebGL memory
    while (furnitureGroup.children.length > 0) {
      const child = furnitureGroup.children[0];
      furnitureGroup.remove(child);
      disposeHierarchy(child);
    }

    if (!showFurniture || !furnishingData) return;
    let current = true;

    const { centerX, centerZ } = centersRef.current;
    const loader = gltfLoaderRef.current;

    // Iterate through furnished rooms
    Object.entries(furnishingData).forEach(([roomId, designResult]) => {
      if (!designResult || !designResult.is_valid || !designResult.items) return;
      const room = layout?.rooms?.find(item => item.id === roomId);
      const level = layout?.storeys?.find(item => item.id === room?.level_id);
      if (level && activeLevel !== 'all' && activeLevel !== level.id) return;
      const [offsetX, offsetZ] = level?.offset || [0, 0];
      const baseY = level?.elevation || 0;

      const roomItems = designResult.items;
      roomItems.forEach((item) => {
        if (!item.glb) return;
        const custom = item.source === 'custom';
        const addModel = (model, sized = false) => {
          if (!current) { disposeHierarchy(model); return; }

          // Shared unmirrored transform: X_three = x_plan - centerX, Z_three = y_plan - centerZ
          const posX = item.position_m[0] + offsetX - centerX;
          const posZ = item.position_m[1] + offsetZ - centerZ;
          const isRug = item.category === 'rug' || item.item_id?.includes('rug');
          // Floor slab is at y = 0.02.
          // Furniture sits on top of floor slab (y = 0.02).
          // Rug is elevated to y = 0.023 with polygonOffset to strictly eliminate z-fighting with floor meshes.
          const posY = baseY + (isRug ? 0.023 : 0.02);
          model.position.set(posX, posY, posZ);

          // Rotation convention: rotation.y = (180.0 - alpha_deg) * (pi / 180)
          model.rotation.y = (180.0 - item.rotation_deg) * (Math.PI / 180.0);

          // Apply compact scale if specified
          if (!sized && item.scale && Array.isArray(item.scale) && item.scale.length === 3) {
            model.scale.set(item.scale[0], item.scale[1], item.scale[2]);
          }
          if (!custom) applyFurnitureFinish(model, item.finish || designResult.preset, item.category, item.colors);

          model.traverse((child) => {
            if (child.isMesh) {
              child.castShadow = (graphicsQuality === 'high' && !isRug);
              child.receiveShadow = (graphicsQuality === 'high');
              if (child.material) {
                child.material.envMapIntensity = (graphicsQuality === 'high' ? 1.0 : 0.0);
                if (isRug) {
                  child.material.polygonOffset = true;
                  child.material.polygonOffsetFactor = -2.0;
                  child.material.polygonOffsetUnits = -4.0;
                }
              }
            }
          });

          model.userData = {
            isFurnitureMesh: true,
            roomId: roomId,
            itemId: item.item_id
          };

          furnitureGroup.add(model);
        };
        const configured = custom ? null : buildFurnitureShape(item);
        if (configured) {
          addModel(configured, true);
          return;
        }
        loader.load(
          item.glb,
          gltf => {
            if (!current) { disposeHierarchy(gltf.scene); return; }
            addModel(custom ? fitCustomFurnitureAsset(gltf.scene, item.dimensions_m) : gltf.scene, custom);
          },
          undefined,
          (err) => {
            if (!current) return;
            console.error('Failed to load furniture asset:', item.glb, err);
            const [w, d, h] = item.dimensions_m || [1, 1, 1];
            const boxGeo = new THREE.BoxGeometry(w, h, d);
            const boxMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8 });
            const boxMesh = new THREE.Mesh(boxGeo, boxMat);
            boxMesh.position.set(item.position_m[0] + offsetX - centerX, baseY + .02 + h / 2, item.position_m[1] + offsetZ - centerZ);
            boxMesh.userData = { isFurnitureMesh: true, roomId, itemId: item.item_id };
            furnitureGroup.add(boxMesh);
          }
        );
      });
    });
    return () => { current = false; };
  }, [furnishingData, showFurniture, layout, graphicsQuality, activeLevel]);

  // View Presets: 3-4 Distinct Architectural Camera Modes
  const setCameraPerspective = () => {
    fitCamera('perspective');
    setViewPreset('perspective');
  };

  const setCameraIsometric = () => {
    fitCamera('isometric');
    setViewPreset('isometric');
  };

  const setCameraTopDown = () => {
    fitCamera('top');
    setViewPreset('top');
  };

  const setCameraRoomFocus = () => {
    if (!cameraRef.current || !controlsRef.current || !selectedRoom) return;
    const pts = selectedRoom.polygon_meters || [];
    if (!pts.length) return;
    const xs = pts.map(([x]) => x), zs = pts.map(([, z]) => z);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
    const span = Math.hypot(maxX - minX, maxZ - minZ);
    const { centerX, centerZ } = centersRef.current;
    const level = layout?.storeys?.find(item => item.id === selectedRoom.level_id);
    const [offsetX, offsetZ] = level?.offset || [0, 0];
    const baseY = level?.elevation || 0;
    const targetX = (minX + maxX) / 2 + offsetX - centerX;
    const targetZ = (minZ + maxZ) / 2 + offsetZ - centerZ;
    if (level) setActiveLevel(level.id);
    setShowRoof(false);
    setShowFurniture(true);
    const camera = cameraRef.current;
    const fov = Math.tan(THREE.MathUtils.degToRad(camera.getEffectiveFOV() / 2));
    const wallHeight = level?.height || (layout?.walls?.[0]?.height_m || STOREY_HEIGHT_M) * numFloors;
    const altitude = Math.max(wallHeight + 1, .70 * span / (fov * Math.min(1, camera.aspect)));
    controlsRef.current.target.set(targetX, baseY + .35, targetZ);
    camera.position.set(targetX + span * .22, baseY + altitude, targetZ + span * .22);
    camera.up.set(0, 1, 0);
    controlsRef.current.update();
    setViewPreset('room');
  };

  const setCameraFacadeFocus = () => {
    if (!cameraRef.current || !controlsRef.current || !layout) return;
    const walls = layout.walls || [];
    const targetWall = walls.find(w => w.id === frontWallId) || walls.find(w => w.is_front) || walls.find(w => w.is_exterior) || walls[0];
    if (!targetWall || !targetWall.polygon_meters || targetWall.polygon_meters.length < 2) return;

    const { centerX = 0, centerZ = 0 } = centersRef.current || {};
    const frame = targetWall.facade_texture?.facade_frame || wallFacadeFrame(targetWall);
    if (!frame) return;
    const level = layout.storeys?.find(item => item.id === targetWall.level_id);
    const [offsetX, offsetZ] = level?.offset || [0, 0];
    const baseY = level?.elevation || 0;
    const height = level?.height || numFloors * (targetWall.height_m || STOREY_HEIGHT_M);
    if (level && activeLevel !== 'all' && activeLevel !== level.id) setActiveLevel(level.id);
    const len = frame.length_m;
    const targetX = frame.origin[0] + frame.direction[0] * len / 2 + offsetX - centerX;
    const targetZ = frame.origin[1] + frame.direction[1] * len / 2 + offsetZ - centerZ;
    const [nx, nz] = frame.outward;

    const camera = cameraRef.current;
    const halfVerticalFov = Math.tan(THREE.MathUtils.degToRad(camera.getEffectiveFOV() / 2));
    const camDist = 1.15 * Math.max(height, len / camera.aspect) / (2 * halfVerticalFov);
    const targetY = baseY + height / 2;
    camera.position.set(targetX + nx * camDist, targetY, targetZ + nz * camDist);
    camera.up.set(0, 1, 0);
    controlsRef.current.target.set(targetX, targetY, targetZ);
    controlsRef.current.update();
    setViewPreset('facade');
  };

  useEffect(() => {
    if (facadeFocusTrigger > 0) {
      setCameraFacadeFocus();
    }
  }, [facadeFocusTrigger]);

  useEffect(() => {
    if (roomFocusTrigger > 0) setCameraRoomFocus();
  }, [roomFocusTrigger]);

  return (
    <div className="viewer-panel viewport-panel flex-1 min-w-0 flex flex-col h-full bg-[#141920] relative overflow-hidden select-none font-mono" aria-label="3D house viewer">
      {/* 3D Viewport Toolbar */}
      <div className="viewer-toolbar viewport-toolbar h-12 shrink-0 overflow-x-auto border-b border-[#ece6da]/15 bg-[#11151b]/90 backdrop-blur-md px-4 flex items-center justify-between z-10 text-[#ece6da] select-none">
        <div className="viewport-title"><Box size={16} /><strong className="font-stencil tracking-wider uppercase">3D House</strong><span title="Drag to orbit. Right-drag or use WASD to move. Double-click to center.">Orbit · pan · zoom</span></div>

        {/* Viewport Control Buttons & Camera Presets */}
        <div className="viewer-tools flex shrink-0 items-center space-x-1.5">
          {layout?.storeys?.length > 1 && <select aria-label="Visible building levels" value={activeLevel}
            onChange={event => setActiveLevel(event.target.value)} className="rounded border bg-parchment-card px-2 py-1 text-xs">
            <option value="all">All levels</option>
            {layout.storeys.map(level => <option key={level.id} value={level.id}>{level.label}</option>)}
          </select>}
          {unresolvedOpenings > 0 && <span className="text-[10px] text-amber-800"
            title="These detections could not be matched to a supported wall span. Review them on the 2D plan.">
            {unresolvedOpenings} {unresolvedOpenings === 1 ? 'opening needs' : 'openings need'} review
          </span>}
          {/* Floor Finish Material Selector */}
          <div className="flex items-center space-x-0.5 bg-parchment-card/80 p-0.5 rounded border border-[#ece6da]/20 shadow-xs" title="Floor finish material across rooms">
            <span className="text-[9px] font-mono text-[#ece6da]/60 px-1 uppercase">Floor:</span>
            {[
              ['oak', 'Wood'],
              ['marble', 'Tile'],
              ['adaptive', 'Adaptive']
            ].map(([id, label]) => (
              <button
                key={id}
                data-testid={`floor-style-${id}`}
                onClick={() => setFloorStyle(id)}
                className={`px-1.5 py-0.5 text-[9.5px] rounded transition-colors ${
                  floorStyle === id
                    ? 'bg-[#ece6da] text-[#0b0e12] font-semibold'
                    : 'text-[#ece6da]/70 hover:text-[#ece6da]'
                }`}
                title={`Set floor material to ${label}`}
              >
                {label}
              </button>
            ))}
          </div>
          {/* Real-time FPS Monitor */}
          <div className="viewer-fps flex items-center gap-1 px-2 py-0.5 rounded-sm bg-[#ece6da]/5 border border-[#ece6da]/10 text-[9.5px] text-[#2a7a6e] font-semibold" title="Live rendering frame rate on local GPU">
            <Activity className="w-3 h-3 text-[#2a7a6e] animate-pulse" />
            <span>{currentFps} FPS</span>
          </div>

          <div className="h-4 w-px bg-[#ece6da]/15 mx-1" />

          {/* Furniture Toggle */}
          <button
            onClick={() => setShowFurniture(!showFurniture)}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              showFurniture
                ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                : 'bg-parchment-card/60 text-[#ece6da]/70 border-[#ece6da]/20 hover:border-[#ece6da]'
            }`}
            title="Toggle placed 3D furniture models"
          >
            <Sparkles className="w-3 h-3 text-[#d4943a]" />
            <span>Furniture</span>
          </button>

          {/* Roof Activation / Deactivation Toggle */}
          <button
            onClick={() => setShowRoof(!showRoof)}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              showRoof
                ? 'bg-[#d4943a] text-[#0b0e12] border-[#d4943a] shadow-sm'
                : 'bg-parchment-card/60 text-[#ece6da]/70 border-[#ece6da]/20 hover:border-[#ece6da]'
            }`}
            title="Toggle 3D Building Roof Enclosure (Activate / Deactivate)"
          >
            <Home className="w-3 h-3" />
            <span>Roof: {showRoof ? 'ON' : 'OFF'}</span>
          </button>

          {/* Roof Style Selector: Modern Flat Slab vs Classic Sims Pitched Hip */}
          {showRoof && (
            <div className="flex items-center space-x-1 bg-parchment-card/80 px-1 py-0.5 rounded border border-[#ece6da]/20 shadow-xs">
              <button
                onClick={() => setRoofStyle('slab')}
                className={`px-1.5 py-0.5 text-[9.5px] rounded transition-colors ${
                  roofStyle === 'slab' ? 'bg-[#ece6da] text-[#0b0e12] font-semibold' : 'text-[#ece6da]/70 hover:text-[#ece6da]'
                }`}
                title="Modern Architectural Flat Slab Roof with Overhang Eaves & Fascia Band"
              >
                Flat
              </button>
              <button
                onClick={() => setRoofStyle('pitched')}
                className={`px-1.5 py-0.5 text-[9.5px] rounded transition-colors ${
                  roofStyle === 'pitched' ? 'bg-[#d4943a] text-[#0b0e12] font-semibold' : 'text-[#ece6da]/70 hover:text-[#ece6da]'
                }`}
                title="Classic Sims Pitched Hipped Gable Roof"
              >
                Pitched
              </button>
            </div>
          )}

          {/* Preset 1: Perspective */}
          <button
            onClick={setCameraPerspective}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'perspective'
                ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                : 'bg-parchment-card/60 text-[#ece6da]/70 border-[#ece6da]/20 hover:border-[#ece6da]'
            }`}
            title="3D Perspective View"
          >
            <Eye className="w-3 h-3" />
            <span>3D</span>
          </button>

          {/* Preset 2: Isometric / Axonometric */}
          <button
            onClick={setCameraIsometric}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'isometric'
                ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                : 'bg-parchment-card/60 text-[#ece6da]/70 border-[#ece6da]/20 hover:border-[#ece6da]'
            }`}
            title="45° Axonometric / Isometric View"
          >
            <Compass className="w-3 h-3" />
            <span>Axo</span>
          </button>

          {/* Preset 3: Top-Down Plan */}
          <button
            onClick={setCameraTopDown}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'top'
                ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                : 'bg-parchment-card/60 text-[#ece6da]/70 border-[#ece6da]/20 hover:border-[#ece6da]'
            }`}
            title="Top-Down Plan View"
          >
            <Layers className="w-3 h-3" />
            <span>Plan</span>
          </button>

          {/* Preset 4: Room Focus (only enabled when room selected) */}
          {selectedRoom && (
            <button
              onClick={setCameraRoomFocus}
              className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
                viewPreset === 'room'
                  ? 'bg-[#d4943a] text-[#0b0e12] border-[#d4943a]'
                  : 'bg-parchment-card/60 text-[#d4943a] border-[#d4943a]/30 hover:bg-[#d4943a]/10'
              }`}
              title="Focus Camera Directly into Selected Room"
            >
              <Maximize2 className="w-3 h-3" />
              <span>Focus</span>
            </button>
          )}

          {/* Preset 5: Facade Focus */}
          <button
            onClick={setCameraFacadeFocus}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'facade'
                ? 'bg-[#e8a33d] text-[#0b0e12] border-[#e8a33d] shadow-sm'
                : 'bg-parchment-card/60 text-[#e8a33d] border-[#e8a33d]/30 hover:bg-[#e8a33d]/10'
            }`}
            title="Face the selected exterior wall"
          >
            <Sparkles className="w-3 h-3 text-[#e8a33d]" />
            <span>Wall view</span>
          </button>

          {/* Graphics Quality Toggle (High: PBR/HDRI/Shadows vs Low: Fast unshadowed diffuse for low-spec/CPU) */}
          <button
            onClick={() => setGraphicsQuality(graphicsQuality === 'high' ? 'low' : 'high')}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              graphicsQuality === 'high'
                ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                : 'bg-parchment-card/60 text-[#ece6da]/70 border-[#ece6da]/20 hover:border-[#ece6da]'
            }`}
            title="Toggle graphics quality (High: PBR & Soft Shadows / Low: Fast unshadowed diffuse for CPU)"
          >
            <Sun className="w-3 h-3 text-[#d4943a]" />
            <span>{graphicsQuality === 'high' ? 'PBR' : 'Low'}</span>
          </button>

          {/* Wireframe Toggle */}
          <button
            onClick={() => setWireframe(!wireframe)}
            className={`px-1.5 py-1 rounded-sm text-[10px] font-medium transition-colors border ${
              wireframe
                ? 'bg-[#ece6da] text-[#0b0e12] border-[#ece6da]'
                : 'bg-parchment-card/60 text-[#ece6da]/60 border-[#ece6da]/20 hover:border-[#ece6da]'
            }`}
            title="Toggle Wireframe overlay"
          >
            WF
          </button>
        </div>
      </div>

      {/* 3D WebGL Canvas Container */}
      <div ref={containerRef} className="flex-1 w-full h-full relative cursor-grab active:cursor-grabbing" />
    </div>
  );
}
