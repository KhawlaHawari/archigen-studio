/**
 * ArchiGen 3D - WebGL Three.js Interactive Viewer & UI Controller
 */

// Global Three.js State
let scene, camera, renderer, controls;
let currentMeshGroup = null;

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initThreeJS();
    initFloorplanHandlers();
    initFacadeHandlers();
});

// -------------------------------------------------------------
// UI Tabs Navigation
// -------------------------------------------------------------
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const panes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            tabBtns.forEach(b => b.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(targetId).classList.add("active");

            if (targetId === "tab-floorplan") {
                onWindowResize();
            }
        });
    });
}

// -------------------------------------------------------------
// Three.js 3D Scene Initialization
// -------------------------------------------------------------
function initThreeJS() {
    const container = document.getElementById("canvas-container-3d");
    if (!container) return;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f17);

    // Camera
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 440;
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(20, 25, 30);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(12, 0, -12);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(30, 50, 40);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x93c5fd, 0.3);
    dirLight2.position.set(-30, 20, -20);
    scene.add(dirLight2);

    // Grid Helper on floor (Z-up coordinate space, rotated to horizontal plane)
    const gridHelper = new THREE.GridHelper(50, 50, 0x334155, 0x1e293b);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // Responsive resize
    window.addEventListener("resize", onWindowResize);

    // Reset Camera button
    document.getElementById("btn-reset-camera").addEventListener("click", () => {
        camera.position.set(20, 25, 30);
        controls.target.set(12, 0, -12);
        controls.update();
    });

    animate();
}

function onWindowResize() {
    const container = document.getElementById("canvas-container-3d");
    if (!container || !renderer || !camera) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

// -------------------------------------------------------------
// Floorplan Processing & 3D Extrusion
// -------------------------------------------------------------
function initFloorplanHandlers() {
    const fileInput = document.getElementById("floorplan-file");
    const demoBtn = document.getElementById("btn-demo-floorplan");

    demoBtn.addEventListener("click", async () => {
        demoBtn.disabled = true;
        demoBtn.textContent = "⏳ Génération...";
        try {
            const resp = await fetch("/demo/floorplan");
            const data = await resp.json();
            renderFloorplanResults(data);
        } catch (err) {
            alert("Erreur lors de la démo: " + err);
        } finally {
            demoBtn.disabled = false;
            demoBtn.textContent = "⚡ Charger Démo Synthétique";
        }
    });

    fileInput.addEventListener("change", async (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/predict/floorplan", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();
            
            // Show uploaded source preview
            const reader = new FileReader();
            reader.onload = (re) => {
                data.input_image_base64 = re.target.result;
                renderFloorplanResults(data);
            };
            reader.readAsDataURL(file);
        } catch (err) {
            alert("Erreur de traitement: " + err);
        }
    });
}

function renderFloorplanResults(data) {
    // 1. Update KPI Strip
    document.getElementById("kpi-area").textContent = `${data.total_habitable_area_m2} m²`;
    document.getElementById("kpi-doors").textContent = data.num_doors_detected;
    document.getElementById("kpi-windows").textContent = data.num_windows_detected;
    document.getElementById("kpi-latency").textContent = `${data.processing_time_ms} ms`;

    // 2. Update 2D Previews
    if (data.input_image_base64) {
        const srcImg = document.getElementById("img-source-plan");
        srcImg.src = data.input_image_base64;
        srcImg.style.display = "block";
        document.getElementById("placeholder-source").style.display = "none";
    }

    const structImg = document.getElementById("img-struct-mask");
    structImg.src = data.structure_mask_base64;
    structImg.style.display = "block";
    document.getElementById("placeholder-struct").style.display = "none";

    const roomImg = document.getElementById("img-room-mask");
    roomImg.src = data.room_mask_base64;
    roomImg.style.display = "block";
    document.getElementById("placeholder-room").style.display = "none";

    // 3. Update Room Table
    const tbody = document.querySelector("#room-table tbody");
    tbody.innerHTML = "";
    if (data.rooms && data.rooms.length > 0) {
        data.rooms.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><span class="color-dot" style="background-color: ${r.color_hex};"></span></td>
                <td><strong>${formatRoomName(r.name)}</strong></td>
                <td>${r.area_m2} m²</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // 4. Update 3D Meshes in Three.js
    if (data.meshes) {
        build3DSceneFromMeshes(data.meshes);
    }
}

function formatRoomName(name) {
    const dict = {
        "living_room": "Salon / Séjour",
        "bedroom": "Chambre",
        "kitchen": "Cuisine",
        "bathroom": "Salle de Bain",
        "hallway": "Couloir",
        "closet": "Placard / Dressing",
        "balcony": "Balcon / Terrasse",
        "dining": "Salle à Manger",
        "office": "Bureau",
        "storage": "Cellier",
        "technical": "Local Technique"
    };
    return dict[name] || name;
}

function build3DSceneFromMeshes(meshes) {
    if (!scene) return;

    // Clear previous mesh group
    if (currentMeshGroup) {
        scene.remove(currentMeshGroup);
    }
    currentMeshGroup = new THREE.Group();

    // In Three.js, Y is UP. Our vectorizer outputs (X: metric, Y: -metric, Z: height).
    // We map: X -> X, Y -> -Z, Z -> Y for standard horizontal floorplan layout!

    // Helper to create THREE.BufferGeometry from raw API mesh payload
    function createBufferGeometry(meshData, isWall = true) {
        if (!meshData || !meshData.vertices || meshData.vertices.length === 0) return null;

        const verts = meshData.vertices;
        const count = verts.length / 3;
        const reoriented = new Float32Array(verts.length);

        for (let i = 0; i < count; i++) {
            const vx = verts[i * 3];
            const vy = verts[i * 3 + 1];
            const vz = verts[i * 3 + 2];
            // Reorient coordinates: X -> vx, Y -> vz (height), Z -> -vy
            reoriented[i * 3] = vx;
            reoriented[i * 3 + 1] = vz;
            reoriented[i * 3 + 2] = -vy;
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(reoriented, 3));

        if (meshData.indices && meshData.indices.length > 0) {
            geom.setIndex(meshData.indices);
        }
        geom.computeVertexNormals();
        return geom;
    }

    // 1. Exterior Walls (Anthracite, Height = 3.0m)
    if (meshes.exterior_walls) {
        const extGeom = createBufferGeometry(meshes.exterior_walls, true);
        if (extGeom) {
            const extMat = new THREE.MeshStandardMaterial({
                color: 0x27272a,
                roughness: 0.6,
                metalness: 0.1,
                side: THREE.DoubleSide
            });
            const extMesh = new THREE.Mesh(extGeom, extMat);
            currentMeshGroup.add(extMesh);
        }
    }

    // 2. Interior Partition Walls (Gris clair, Height = 2.8m)
    if (meshes.interior_walls) {
        const intGeom = createBufferGeometry(meshes.interior_walls, true);
        if (intGeom) {
            const intMat = new THREE.MeshStandardMaterial({
                color: 0x71717a,
                roughness: 0.7,
                metalness: 0.05,
                side: THREE.DoubleSide
            });
            const intMesh = new THREE.Mesh(intGeom, intMat);
            currentMeshGroup.add(intMesh);
        }
    }

    // 3. Room Floor Slabs (Color-coded)
    if (meshes.room_floors && meshes.room_floors.length > 0) {
        meshes.room_floors.forEach(rf => {
            const floorGeom = createBufferGeometry(rf.mesh, false);
            if (floorGeom) {
                const colorArr = rf.mesh.color || [0.8, 0.8, 0.8];
                const threeColor = new THREE.Color(colorArr[0], colorArr[1], colorArr[2]);
                const floorMat = new THREE.MeshStandardMaterial({
                    color: threeColor,
                    roughness: 0.4,
                    metalness: 0.1,
                    side: THREE.DoubleSide
                });
                const floorMesh = new THREE.Mesh(floorGeom, floorMat);
                currentMeshGroup.add(floorMesh);
            }
        });
    }

    scene.add(currentMeshGroup);
}

// -------------------------------------------------------------
// Facade Processing (Pix2Pix cGAN)
// -------------------------------------------------------------
function initFacadeHandlers() {
    const demoBtn = document.getElementById("btn-demo-facade");
    const fileInput = document.getElementById("facade-file");

    demoBtn.addEventListener("click", async () => {
        demoBtn.disabled = true;
        demoBtn.textContent = "⏳ Synthèse...";
        try {
            const resp = await fetch("/demo/facade");
            const data = await resp.json();
            renderFacadeResults(data);
        } catch (err) {
            alert("Erreur génération façade: " + err);
        } finally {
            demoBtn.disabled = false;
            demoBtn.textContent = "⚡ Charger Élévation Démo";
        }
    });

    fileInput.addEventListener("change", async (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/predict/facade", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();
            renderFacadeResults(data);
        } catch (err) {
            alert("Erreur de prédiction façade: " + err);
        }
    });
}

function renderFacadeResults(data) {
    document.getElementById("facade-latency").textContent = `Latence: ${data.processing_time_ms} ms (SLA < 400ms)`;

    const inputImg = document.getElementById("img-facade-input");
    inputImg.src = data.input_mask_base64;
    inputImg.style.display = "block";
    document.getElementById("facade-input-ph").style.display = "none";

    const outImg = document.getElementById("img-facade-output");
    outImg.src = data.generated_facade_base64;
    outImg.style.display = "block";
    document.getElementById("facade-output-ph").style.display = "none";
}
