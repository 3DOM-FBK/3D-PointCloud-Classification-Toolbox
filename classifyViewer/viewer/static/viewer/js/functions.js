import {
    loadPotree2PointCloud,
    getPotree2Loader
} from "./potree2-loader.js";

// --- Runtime path helpers (read from Django-injected config) ---
const _runtimeUrl = (...parts) => {
    const base = (window.__APP_CONFIG?.runtimeDataUrl || '/runtime-data').replace(/\/$/, '');
    return base + '/' + parts.join('/');
};
const _runtimePath = (...parts) => {
    const prefix = window.__APP_CONFIG?.runtimeDataPathPrefix || 'runtime_data';
    return [prefix, ...parts].join('/');
};

// =====================================================================
// UNIFIED POINT CLOUD LOADER - Auto-detects format
// =====================================================================
/**
 * Unified point cloud loader - loads Potree 2.0 format
 */
export async function loadPointCloud(path, scene, options = {}) {
    console.log("🔍 Loading Potree 2.0 point cloud...");

    try {
        // Try to load metadata.json to verify it's a valid Potree 2.0 folder
        const metadataUrl = `${path}/metadata.json`;
        const response = await fetch(metadataUrl);

        if (!response.ok) {
            throw new Error("No metadata.json found at " + path);
        }

        const metadata = await response.json();

        if (metadata.version !== "2.0") {
            console.warn(`⚠️ Warning: Metadata version is ${metadata.version}, expected 2.0. Attempting to load anyway.`);
        }

        console.log("✅ Using Potree2Loader");
        return await loadPotree2PointCloud(path, scene, options);

    } catch (error) {
        console.error("❌ Failed to load point cloud:", error);
        throw error;
    }
}

/**
 * Get the active loader (Potree2)
 */
export function getLODLoader(scene) {
    return getPotree2Loader(scene);
}

/**
 * Get LOD statistics
 */
export function getLODStats(scene) {
    const loader = getLODLoader(scene);
    if (!loader) return null;
    return loader.getStats();
}

/**
 * Update LOD
 */
export function updateLOD(scene) {
    const loader = getLODLoader(scene);
    if (loader && scene.activeCamera) {
        loader.update(scene.activeCamera);
    }
}

/**
 * Set LOD parameters (works with both loaders)
 */
export function setLODParameters(scene, params) {
    const loader = getLODLoader(scene);
    if (!loader) {
        console.warn("LOD loader not available");
        return false;
    }

    // Common parameters
    if (params.pointSize !== undefined) {
        loader.setPointSize(params.pointSize);
    }

    if (params.pointSizeMultiplier !== undefined) {
        loader.setPointSizeMultiplier(params.pointSizeMultiplier);
    }

    if (params.maxVisiblePoints !== undefined) {
        loader.maxVisiblePoints = params.maxVisiblePoints;
    }

    if (params.maxVisibleNodes !== undefined) {
        loader.maxVisibleNodes = params.maxVisibleNodes;
    }

    // Trigger update
    if (scene.activeCamera) {
        loader.update(scene.activeCamera);
    }

    return true;
}

// showDownloadModal and showLoadModal exported directly above

// =====================================================================
// CLASS REGISTRY
// =====================================================================

/**
 * Global registry of classification classes.
 * Map: classId (int) -> { name (string), color (hex string) }
 */
export const classRegistry = new Map();
let _classIdCounter = 1;

// Outline segment multi-selection state (UI only).
const selectedSegmentIds = new Set();

function _getSegmentRows(scope = document) {
    return Array.from(scope.querySelectorAll('.outline-item[data-segment-id]:not(.class-item)'));
}

function _setSegmentRowSelected(row, selected) {
    const segId = Number(row?.dataset?.segmentId);
    if (!Number.isFinite(segId)) return;

    row.classList.toggle('segment-selected', selected);
    if (selected) selectedSegmentIds.add(segId);
    else selectedSegmentIds.delete(segId);
}

function _clearSegmentSelection(scope = document) {
    _getSegmentRows(scope).forEach(row => row.classList.remove('segment-selected'));
    selectedSegmentIds.clear();
}

function _getSelectedSegmentIds(scope = document) {
    const ids = _getSegmentRows(scope)
        .filter(row => row.classList.contains('segment-selected'))
        .map(row => Number(row.dataset.segmentId))
        .filter(id => Number.isFinite(id));

    // Keep Set in sync if rows were removed externally.
    selectedSegmentIds.clear();
    ids.forEach(id => selectedSegmentIds.add(id));
    return ids;
}

/**
 * Reset the class registry and ID counter to initial state.
 * Called during scene reset / load to avoid stale class data.
 */
export function resetClassRegistry() {
    classRegistry.clear();
    _classIdCounter = 1;
}

/**
 * Converts a hex color string (e.g. "#ef4444") to [r, g, b] floats in [0,1].
 */
function hexToRgbFloat(hex) {
    const h = hex.replace('#', '');
    return [
        parseInt(h.slice(0, 2), 16) / 255,
        parseInt(h.slice(2, 4), 16) / 255,
        parseInt(h.slice(4, 6), 16) / 255
    ];
}

/**
 * Assigns a class (id + color) to the currently selected points.
 *
 * Strategy:
 *  1. If the Potree2Loader exposes getSelectedPointsInfo(), use that for
 *     accurate per-point index tracking.
 *  2. Fallback: scan vertex colors for the red selection highlight [1, 0, 0]
 *     and tag those points.
 *
 * Class data is persisted in mesh.metadata:
 *   - classColors: Float32Array — RGBA per point, set when classId > 0
 */
export function applyClassToSelection(scene, classId, hexColor) {
    const [r, g, b] = hexToRgbFloat(hexColor);
    const loader = getPotree2Loader(scene);

    if (!loader) {
        console.warn("⚠️ Potree2 loader not found. Make sure a point cloud is loaded.");
        return;
    }

    if (!loader.loadedNodes || loader.loadedNodes.size === 0) {
        console.warn("⚠️ No nodes loaded yet.");
        return;
    }

    // Delega al loader: ha accesso diretto a loadedNodes e alla classificationMap
    const total = loader.applyClassToLoadedNodes(classId, r, g, b);

    if (total > 0) {
        console.log(`✅ Class ${classId} "${classRegistry.get(classId)?.name}" (${hexColor}) assigned to ${total.toLocaleString()} points.`);
    } else {
        console.warn("⚠️ No selected (red-highlighted) points found. Select points first, then assign a class.");
    }
}

// BUTTONS
export function createButton(name, id, where = document.body) {
    const button = document.createElement('button');
    button.textContent = name;
    button.id = id;
    button.classList.add('btn');
    where.appendChild(button);
    return button;
}

export function createToolButton(id, svgContent, tooltip, where = document.body) {
    const button = document.createElement('button');
    button.id = id;
    button.classList.add('tool-btn');
    button.setAttribute('data-tooltip', tooltip);
    button.innerHTML = svgContent;
    where.appendChild(button);
    return button;
}
export function createButtonGrid(id) {
    const grid = document.createElement('div');
    grid.id = id;
    document.body.appendChild(grid);
    return grid;
}

export function textVisible(textName) {
    // Visible
    textName.style.pointerEvents = "auto";
    textName.style.opacity = "1";
}

// // LOADING POINT CLOUD
// export async function loadPointCloud(url, scene) {
//     if (url.endsWith(".ply")) {
//         try {
//             console.log("Loading point cloud from:", url);

//             // 🔧 Carica prima i dati del PLY
//             const result = await BABYLON.SceneLoader.ImportMeshAsync("", "", url, scene);
//             const tempMesh = result.meshes[0];

//             // Estrai dati
//             const positions = tempMesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
//             const colors = tempMesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
//             const normals = tempMesh.getVerticesData(BABYLON.VertexBuffer.NormalKind);

//             const numPoints = positions.length / 3;
//             // console.log(`📊 Punti caricati: ${numPoints.toLocaleString()}`);

//             // 🔧 CREA POINT CLOUD SYSTEM
//             const pcs = new BABYLON.PointsCloudSystem("pcs", 1, scene);
//             pcs.pointSize = 3;
//             // Funzione per inizializzare ogni punto
//             pcs.addPoints(numPoints, (particle, i) => {
//                 const idx = i * 3;

//                 // Posizione
//                 particle.position.set(
//                     positions[idx],
//                     positions[idx + 1],
//                     positions[idx + 2]
//                 );

//                 // Colore
//                 if (colors) {
//                     const colorIdx = i * 4;
//                     particle.color = new BABYLON.Color4(
//                         colors[colorIdx],
//                         colors[colorIdx + 1],
//                         colors[colorIdx + 2],
//                         1.0
//                     );
//                     particle.originalColor = particle.color.clone();
//                 }

//                 // Normale (opzionale)
//                 if (normals) {
//                     // PCS non usa direttamente le normali, ma puoi salvarle
//                     particle.normal = new BABYLON.Vector3(
//                         normals[idx],
//                         normals[idx + 1],
//                         normals[idx + 2]
//                     );
//                 }
//             });

//             // 🔧 Build del PCS
//             await pcs.buildMeshAsync();

//             // 🔧 Ottimizzazioni
//             pcs.mesh.alwaysSelectAsActiveMesh = true; // Sempre visibile
//             pcs.computeParticleColor = false; // Usa colori già impostati
//             pcs.computeParticleTexture = false;

//             // Rimuovi mesh temporanea
//             tempMesh.dispose();

//             // console.log(`✅ Point Cloud System creato con ${numPoints.toLocaleString()} punti`);
//             pcs.mesh.isPickable = true;
//             pcs.mesh.refreshBoundingInfo(true);
//             pcs.mesh._pcs = pcs; // Link PCS to mesh for easy access

//             return pcs.mesh;

//         } catch (err) {
//             console.error(`Error loading PLY:`, err);
//             return null;
//         }
//     } else if (url.endsWith(".txt")) {
//         try {
//             console.log("Loading point cloud from:", url);

//             const response = await fetch(url);
//             const text = await response.text();
//             const lines = text.split("\n");

//             const positions = [];
//             const colors = [];

//             for (let i = 1; i < lines.length; i++) {
//                 const line = lines[i].trim();
//                 if (!line) continue;

//                 const values = line.split(/\s+/).map(Number);
//                 if (values.length < 3) continue;

//                 const [x, y, z, r, g, b] = values;
//                 positions.push(x, y, z);

//                 // Optional color
//                 if (r !== undefined && g !== undefined && b !== undefined) {
//                     colors.push(r / 255, g / 255, b / 255, 1.0);
//                 }
//             }

//             const numPoints = positions.length / 3;
//             // console.log(`📊 Punti caricati: ${numPoints.toLocaleString()}`);

//             // 🔧 CREA POINT CLOUD SYSTEM
//             const pcs = new BABYLON.PointsCloudSystem("pcs", 1, scene);

//             pcs.addPoints(numPoints, (particle, i) => {
//                 const idx = i * 3;

//                 particle.position.set(
//                     positions[idx],
//                     positions[idx + 1],
//                     positions[idx + 2]
//                 );

//                 if (colors.length > 0) {
//                     const colorIdx = i * 4;
//                     particle.color = new BABYLON.Color4(
//                         colors[colorIdx],
//                         colors[colorIdx + 1],
//                         colors[colorIdx + 2],
//                         1.0
//                     );
//                     particle.originalColor = particle.color.clone();
//                 }
//             });

//             await pcs.buildMeshAsync();

//             pcs.mesh.alwaysSelectAsActiveMesh = true;
//             pcs.computeParticleColor = false;
//             pcs.computeParticleTexture = false;

//             // console.log(`✅ Point Cloud System creato`);
//             pcs.mesh.isPickable = true;
//             pcs.mesh.refreshBoundingInfo(true);
//             pcs.mesh._pcs = pcs; // Link PCS to mesh for easy access

//             return pcs.mesh;

//         } catch (err) {
//             console.error("Error loading TXT:", err);
//             return null;
//         }

//     } else {
//         console.error("Unsupported file format:", url);
//         return null;
//     }

// }

// ----------------------
// Carica un singolo cluster .bin (formato 15 byte: XYZ float32 + RGB uint8)
// ----------------------

export async function exportPointCloud(mesh, filepath, format = "ply", binary = true) {
    if (!mesh) {
        console.error("No mesh to export");
        return;
    }

    const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
    const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
    const normals = mesh.getVerticesData(BABYLON.VertexBuffer.NormalKind);

    if (!positions) {
        console.error("Mesh has no positions");
        return;
    }

    const numPoints = positions.length / 3;
    console.log(`Request to export ${numPoints.toLocaleString()} points in the format ${format.toUpperCase()} (${binary ? "binary" : "ascii"})`);

    if (format === "ply") {
        await exportPLY(positions, colors, normals, filepath, binary);
    } else if (format === "txt") {
        await exportTXT(positions, colors, normals, filepath, binary);
    } else {
        console.error("Unsupported format. Use 'ply' or 'txt'");
    }
}

// ========== EXPORT PLY ==========
async function exportPLY(positions, colors, normals, filepath, binary) {
    const numPoints = positions.length / 3;
    const hasColors = colors && colors.length > 0;
    const hasNormals = normals && normals.length > 0;

    if (binary) {
        // PLY BINARY
        let header = `ply\nformat binary_little_endian 1.0\n`;
        header += `element vertex ${numPoints}\n`;
        header += `property float x\nproperty float y\nproperty float z\n`;

        if (hasNormals) {
            header += `property float nx\nproperty float ny\nproperty float nz\n`;
        }

        if (hasColors) {
            header += `property uchar red\nproperty uchar green\nproperty uchar blue\n`;
        }

        header += `end_header\n`;

        const headerBytes = new TextEncoder().encode(header);
        const bytesPerPoint = 12 + (hasNormals ? 12 : 0) + (hasColors ? 3 : 0);
        const buffer = new ArrayBuffer(headerBytes.length + numPoints * bytesPerPoint);
        const view = new DataView(buffer);
        const uint8View = new Uint8Array(buffer);

        // Scrivi header
        uint8View.set(headerBytes, 0);
        let offset = headerBytes.length;

        // Scrivi dati
        for (let i = 0; i < numPoints; i++) {
            const idx = i * 3;

            // Posizioni (float32)
            view.setFloat32(offset, positions[idx], true); offset += 4;
            view.setFloat32(offset, positions[idx + 1], true); offset += 4;
            view.setFloat32(offset, positions[idx + 2], true); offset += 4;

            // Normali (float32)
            if (hasNormals) {
                view.setFloat32(offset, normals[idx], true); offset += 4;
                view.setFloat32(offset, normals[idx + 1], true); offset += 4;
                view.setFloat32(offset, normals[idx + 2], true); offset += 4;
            }

            // Colori (uint8)
            if (hasColors) {
                const colorIdx = i * 4;
                view.setUint8(offset, Math.round(colors[colorIdx] * 255)); offset += 1;
                view.setUint8(offset, Math.round(colors[colorIdx + 1] * 255)); offset += 1;
                view.setUint8(offset, Math.round(colors[colorIdx + 2] * 255)); offset += 1;
            }
        }

        await saveFile(filepath, new Blob([buffer]));

    } else {
        // PLY ASCII
        let ply = `ply\nformat ascii 1.0\n`;
        ply += `element vertex ${numPoints}\n`;
        ply += `property float x\nproperty float y\nproperty float z\n`;

        if (hasNormals) {
            ply += `property float nx\nproperty float ny\nproperty float nz\n`;
        }

        if (hasColors) {
            ply += `property uchar red\nproperty uchar green\nproperty uchar blue\n`;
        }

        ply += `end_header\n`;

        for (let i = 0; i < numPoints; i++) {
            const idx = i * 3;
            ply += `${positions[idx]} ${positions[idx + 1]} ${positions[idx + 2]}`;

            if (hasNormals) {
                ply += ` ${normals[idx]} ${normals[idx + 1]} ${normals[idx + 2]}`;
            }

            if (hasColors) {
                const colorIdx = i * 4;
                const r = Math.round(colors[colorIdx] * 255);
                const g = Math.round(colors[colorIdx + 1] * 255);
                const b = Math.round(colors[colorIdx + 2] * 255);
                ply += ` ${r} ${g} ${b}`;
            }

            ply += '\n';
        }

        await saveFile(filepath, new Blob([ply], { type: 'text/plain' }));
    }
}

// ========== EXPORT TXT ==========
async function exportTXT(positions, colors, normals, filepath, binary) {
    const numPoints = positions.length / 3;
    const hasColors = colors && colors.length > 0;
    const hasNormals = normals && normals.length > 0;

    if (binary) {
        // TXT BINARY (custom format)
        const bytesPerPoint = 12 + (hasColors ? 3 : 0) + (hasNormals ? 12 : 0);
        const buffer = new ArrayBuffer(numPoints * bytesPerPoint);
        const view = new DataView(buffer);
        let offset = 0;

        for (let i = 0; i < numPoints; i++) {
            const idx = i * 3;

            // Posizioni
            view.setFloat32(offset, positions[idx], true); offset += 4;
            view.setFloat32(offset, positions[idx + 1], true); offset += 4;
            view.setFloat32(offset, positions[idx + 2], true); offset += 4;

            // Colori
            if (hasColors) {
                const colorIdx = i * 4;
                view.setUint8(offset, Math.round(colors[colorIdx] * 255)); offset += 1;
                view.setUint8(offset, Math.round(colors[colorIdx + 1] * 255)); offset += 1;
                view.setUint8(offset, Math.round(colors[colorIdx + 2] * 255)); offset += 1;
            }

            // Normali
            if (hasNormals) {
                view.setFloat32(offset, normals[idx], true); offset += 4;
                view.setFloat32(offset, normals[idx + 1], true); offset += 4;
                view.setFloat32(offset, normals[idx + 2], true); offset += 4;
            }
        }

        await saveFile(filepath, new Blob([buffer]));

    } else {
        // TXT ASCII
        let txt = "X Y Z";
        if (hasColors) txt += " R G B";
        if (hasNormals) txt += " NX NY NZ";
        txt += "\n";

        for (let i = 0; i < numPoints; i++) {
            const idx = i * 3;
            txt += `${positions[idx]} ${positions[idx + 1]} ${positions[idx + 2]}`;

            if (hasColors) {
                const colorIdx = i * 4;
                const r = Math.round(colors[colorIdx] * 255);
                const g = Math.round(colors[colorIdx + 1] * 255);
                const b = Math.round(colors[colorIdx + 2] * 255);
                txt += ` ${r} ${g} ${b}`;
            }

            if (hasNormals) {
                txt += ` ${normals[idx]} ${normals[idx + 1]} ${normals[idx + 2]}`;
            }

            txt += '\n';
        }

        await saveFile(filepath, new Blob([txt], { type: 'text/plain' }));
    }
}

// ========== SAVE FILE HELPER ==========
let isSaving = false;
const saveQueue = [];

/**
 * Queue a file save request so writes are serialized and never overlap.
 */
async function saveFile(filepath, blob) {
    // Add the request to the queue.
    return new Promise((resolve, reject) => {
        saveQueue.push({ filepath, blob, resolve, reject });
        processSaveQueue();
    });
}

/**
 * Process the next pending save request, if any.
 */
async function processSaveQueue() {
    // If a save is already running or the queue is empty, exit.
    if (isSaving || saveQueue.length === 0) return;

    isSaving = true;
    const { filepath, blob, resolve, reject } = saveQueue.shift();

    try {
        // console.log(`Saving (${saveQueue.length} in queue): ${filepath}`);

        const arrayBuffer = await blob.arrayBuffer();
        const buffer = new Uint8Array(arrayBuffer);

        // Convert to base64.
        let binary = '';
        for (let i = 0; i < buffer.length; i++) {
            binary += String.fromCharCode(buffer[i]);
        }
        const base64 = btoa(binary);

        const response = await fetch('/save_file/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filepath: filepath,
                data: base64
            })
        });

        if (response.ok) {
            console.log(`File saved: ${filepath} (${(blob.size / 1024 / 1024).toFixed(2)} MB)`);
            resolve();
        } else {
            const error = `Error: ${response.statusText}`;
            console.error(`${error}`);
            reject(new Error(error));
        }
    } catch (error) {
        console.error(`Error saving file:`, error);
        reject(error);
    } finally {
        isSaving = false;
        // Processa prossimo file in coda
        processSaveQueue();
    }
}

// Centra camera sulla mesh
export function frameCameraOnMesh(camera, node) {
    if (!node || !camera) return;
    console.log("Centering camera on:", node.name);

    let min = null;
    let max = null;
    let found = false;

    const meshes = node instanceof BABYLON.AbstractMesh ? [node] : node.getChildMeshes();

    if (meshes.length === 0) return;

    const MAX_ABS_COORD = 1e9;
    const MAX_EXTENT = 1e9;

    meshes.forEach(mesh => {
        if (!mesh || mesh.isDisposed?.()) return;
        if (!mesh.isVisible || !mesh.isEnabled?.()) return;

        const bbox = mesh.getBoundingInfo?.()?.boundingBox;
        if (!bbox) return;

        const meshMin = bbox.minimumWorld;
        const meshMax = bbox.maximumWorld;
        if (!meshMin || !meshMax) return;
        if (!Number.isFinite(meshMin.x) || !Number.isFinite(meshMin.y) || !Number.isFinite(meshMin.z)) return;
        if (!Number.isFinite(meshMax.x) || !Number.isFinite(meshMax.y) || !Number.isFinite(meshMax.z)) return;

        const localDx = meshMax.x - meshMin.x;
        const localDy = meshMax.y - meshMin.y;
        const localDz = meshMax.z - meshMin.z;
        if (!Number.isFinite(localDx) || !Number.isFinite(localDy) || !Number.isFinite(localDz)) return;
        if (Math.abs(meshMin.x) > MAX_ABS_COORD || Math.abs(meshMin.y) > MAX_ABS_COORD || Math.abs(meshMin.z) > MAX_ABS_COORD) return;
        if (Math.abs(meshMax.x) > MAX_ABS_COORD || Math.abs(meshMax.y) > MAX_ABS_COORD || Math.abs(meshMax.z) > MAX_ABS_COORD) return;
        if (Math.abs(localDx) > MAX_EXTENT || Math.abs(localDy) > MAX_EXTENT || Math.abs(localDz) > MAX_EXTENT) return;

        if (!min) {
            min = meshMin.clone();
            max = meshMax.clone();
        } else {
            min.x = Math.min(min.x, meshMin.x);
            min.y = Math.min(min.y, meshMin.y);
            min.z = Math.min(min.z, meshMin.z);
            max.x = Math.max(max.x, meshMax.x);
            max.y = Math.max(max.y, meshMax.y);
            max.z = Math.max(max.z, meshMax.z);
        }

        found = true;
    });

    if (!found || !min || !max) {
        const fallbackTarget = node.getAbsolutePosition?.() || (typeof camera.getTarget === 'function' ? camera.getTarget() : null);
        if (!fallbackTarget || !Number.isFinite(fallbackTarget.x) || !Number.isFinite(fallbackTarget.y) || !Number.isFinite(fallbackTarget.z)) {
            console.warn("Frame Selection skipped: no valid visible bounds.");
            return;
        }
        camera.setTarget(fallbackTarget);
        if (camera.radius !== undefined) {
            camera.radius = Math.max(Number.isFinite(camera.radius) ? camera.radius : 1, 1);
        } else {
            camera.position = fallbackTarget.add(new BABYLON.Vector3(0, 0, 1));
        }
        return;
    }

    const dx = max.x - min.x;
    const dy = max.y - min.y;
    const dz = max.z - min.z;
    const center = new BABYLON.Vector3(
        min.x + dx * 0.5,
        min.y + dy * 0.5,
        min.z + dz * 0.5
    );

    const diagonal = Math.hypot(dx, dy, dz);
    let radius = diagonal * 0.8;
    if (!Number.isFinite(radius) || radius <= 0) {
        const target = typeof camera.getTarget === 'function' ? camera.getTarget() : null;
        const fallback = target ? BABYLON.Vector3.Distance(target, center) : 0;
        radius = Math.max(Number.isFinite(fallback) ? fallback : 0, camera.radius || 0, 1);
    }

    if (!Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) {
        console.warn("Frame Selection aborted: computed center is invalid.");
        return;
    }
    if (!Number.isFinite(radius) || radius <= 0) {
        console.warn("Frame Selection aborted: computed radius is invalid.");
        return;
    }

    camera.setTarget(center);

    if (camera.radius !== undefined) {
        camera.radius = radius;
    } else {
        camera.position = center.add(new BABYLON.Vector3(0, 0, radius));
    }
}


// TEXTS
export function textNotVisible(textName) {
    // Not Visible
    textName.style.pointerEvents = "none";
    textName.style.opacity = "0";
}


export function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export function createAccordionSection(title, parentId) {
    const parent = document.getElementById(parentId);
    if (!parent) return null;

    const section = document.createElement('div');
    section.classList.add('accordion-section');

    const header = document.createElement('div');
    header.classList.add('accordion-header');

    const titleSpan = document.createElement('span');
    titleSpan.textContent = title;
    header.appendChild(titleSpan);

    const headerActions = document.createElement('div');
    headerActions.classList.add('accordion-header-actions');
    header.appendChild(headerActions);

    const arrow = document.createElement('span');
    arrow.classList.add('accordion-arrow');
    header.appendChild(arrow);

    const content = document.createElement('div');
    content.classList.add('accordion-content');

    header.addEventListener('click', (e) => {
        if (e.target.closest('.accordion-header-actions')) return;
        section.classList.toggle('collapsed');
    });

    section.appendChild(header);
    section.appendChild(content);
    parent.appendChild(section);

    return { content, headerActions, arrow };
}

export function createPropertyRow(label, initialValue, parent) {
    const row = document.createElement('div');
    row.classList.add('property-row');

    const labelSpan = document.createElement('span');
    labelSpan.classList.add('property-label');
    labelSpan.textContent = label + ":";

    const valueInput = document.createElement('input');
    valueInput.classList.add('property-input');
    valueInput.value = initialValue;

    row.appendChild(labelSpan);
    row.appendChild(valueInput);
    parent.appendChild(row);

    return valueInput;
}

export function createCheckbox(label, initialState, parent) {
    const row = document.createElement('div');
    row.classList.add('property-row');

    const labelSpan = document.createElement('span');
    labelSpan.classList.add('property-label');
    labelSpan.textContent = label + ":";

    // Struttura Toggle (Switch)
    const toggleLabel = document.createElement('label');
    toggleLabel.classList.add('switch');

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = initialState;

    const slider = document.createElement('span');
    slider.classList.add('switch-slider');

    toggleLabel.appendChild(checkbox);
    toggleLabel.appendChild(slider);

    row.appendChild(labelSpan);
    row.appendChild(toggleLabel);
    parent.appendChild(row);

    return checkbox;
}

export function createSimpleCheckbox(label, initialState, parent) {
    const item = document.createElement('label');
    item.classList.add('checkbox-item');

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = initialState;

    const checkmark = document.createElement('span');
    checkmark.classList.add('checkmark');

    const span = document.createElement('span');
    span.textContent = label;

    item.appendChild(checkbox);
    item.appendChild(checkmark);
    item.appendChild(span);
    parent.appendChild(item);

    return checkbox;
}

export function createSlider(label, min, max, initial, step, parent) {
    const row = document.createElement('div');
    row.classList.add('property-row', 'property-row-vertical');

    const labelContainer = document.createElement('div');
    labelContainer.classList.add('label-container');

    const labelSpan = document.createElement('span');
    labelSpan.classList.add('property-label');
    labelSpan.textContent = label;

    const valueDisplay = document.createElement('span');
    valueDisplay.classList.add('property-value-display');
    valueDisplay.textContent = initial;

    labelContainer.appendChild(labelSpan);
    labelContainer.appendChild(valueDisplay);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min;
    slider.max = max;
    slider.value = initial;
    slider.step = step;
    slider.classList.add('property-slider');

    slider.addEventListener('input', () => {
        valueDisplay.textContent = slider.value;
    });

    row.appendChild(labelContainer);
    row.appendChild(slider);
    parent.appendChild(row);

    return slider;
}

/**
 * @param {string} name
 * @param {string} iconSrc
 * @param {HTMLElement} parent
 * @param {Function|null} onVisibilityChange  Optional callback(visible: boolean).
 *   When provided it is called instead of any built-in mesh toggling, giving the
 *   caller full control (e.g. loader.setSegmentVisible).
 *   When null the item is inert until setVisibilityCallback() is called later.
 */
export function createOutlineItem(name, iconSrc, parent, segmentId, onVisibilityChange = null, initialVisible = true) {
    const row = document.createElement('div');
    row.classList.add('outline-item');
    if (segmentId !== undefined && segmentId !== null) {
        row.dataset.segmentId = segmentId;
    }

    const mainInfo = document.createElement('div');
    mainInfo.classList.add('outline-main-info');

    const icon = document.createElement('img');
    icon.src = iconSrc;
    icon.classList.add('outline-icon');
    icon.alt = "Icon";

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.value = name;
    nameInput.classList.add('outline-name-input');

    mainInfo.appendChild(icon);
    mainInfo.appendChild(nameInput);

    // Visibility Toggle
    const visibilityBtn = document.createElement('button');
    visibilityBtn.classList.add('outline-visibility-btn');

    const visibilityIcon = document.createElement('img');
    const iconBase = "static/viewer/icons/";
    visibilityIcon.src = `${iconBase}visibility-on.png`;
    visibilityIcon.classList.add('outline-visibility-icon');

    visibilityBtn.appendChild(visibilityIcon);

    let isVisible = initialVisible;
    let _visibilityCallback = onVisibilityChange;

    visibilityIcon.src = isVisible ? `${iconBase}visibility-on.png` : `${iconBase}visibility-off.png`;
    visibilityBtn.classList.toggle('off', !isVisible);

    visibilityBtn.addEventListener('click', () => {
        isVisible = !isVisible;
        visibilityIcon.src = isVisible ? `${iconBase}visibility-on.png` : `${iconBase}visibility-off.png`;
        visibilityBtn.classList.toggle('off', !isVisible);
        if (_visibilityCallback) _visibilityCallback(isVisible);
    });

    row.appendChild(mainInfo);
    row.appendChild(visibilityBtn);
    parent.appendChild(row);

    // Segment row multi-selection in the outline:
    // - Click: select only this segment
    // - SHIFT+Click: toggle this segment in current selection
    row.addEventListener('click', (e) => {
        // Ignore controls that should keep their own interaction behavior.
        if (e.target.closest('.outline-name-input') || e.target.closest('.outline-visibility-btn')) return;

        if (e.shiftKey) {
            _setSegmentRowSelected(row, !row.classList.contains('segment-selected'));
            return;
        }

        _clearSegmentSelection(parent);
        _setSegmentRowSelected(row, true);
    });

    // Right-click context menu (Only in Training Layout)
    row.addEventListener('contextmenu', (e) => {
        // Disable in Classify layout
        if (window.__currentLayoutMode === 'classify') return;

        e.preventDefault();
        const iconBase = "static/viewer/icons/";

        // If right-click is on a non-selected row, make it the only selected item.
        if (!row.classList.contains('segment-selected')) {
            _clearSegmentSelection(parent);
            _setSegmentRowSelected(row, true);
        }

        const selectedIds = _getSelectedSegmentIds(parent);
        const mergeCandidates = selectedIds.filter(id => id !== segmentId && id !== 0);

        const options = [
            {
                label: `Assign selection to "${nameInput.value}"`,
                icon: `${iconBase}cursor.png`,
                action: () => {
                    const scene = window.__babylonScene;
                    if (scene && scene.potree2Loader) {
                        const count = scene.potree2Loader.assignSelectionToSegment(segmentId);
                        if (count > 0) console.log(`✅ Assigned ${count} points to ${nameInput.value}`);
                    }
                }
            },
            {
                label: `Remove selection from "${nameInput.value}"`,
                icon: `${iconBase}trash.png`,
                action: () => {
                    const scene = window.__babylonScene;
                    if (scene && scene.potree2Loader) {
                        const count = scene.potree2Loader.removeSelectionFromSegment(segmentId);
                        if (count > 0) console.log(`✅ Removed ${count} points from ${nameInput.value}`);
                    }
                }
            }
        ];

        if (mergeCandidates.length > 0) {
            const mergeCount = mergeCandidates.length + 1;
            options.push({
                label: `Merge ${mergeCount} selected segments into "${nameInput.value}"`,
                icon: `${iconBase}cursor.png`,
                action: () => {
                    const scene = window.__babylonScene;
                    if (!scene?.potree2Loader?.mergeSegments) return;

                    const merged = scene.potree2Loader.mergeSegments(segmentId, mergeCandidates);
                    if (!merged || merged.mergedSegments <= 0) return;

                    // Remove merged source rows from outline, keep target row.
                    mergeCandidates.forEach(srcId => {
                        const srcRow = parent.querySelector(`.outline-item[data-segment-id="${srcId}"]`);
                        if (srcRow && srcRow !== row) srcRow.remove();
                    });

                    _clearSegmentSelection(parent);
                    _setSegmentRowSelected(row, true);
                    console.log(`✅ Merged ${merged.mergedSegments} segments into ${nameInput.value} (${merged.movedPoints.toLocaleString()} points moved).`);
                }
            });
        }

        showContextMenu(e.clientX, e.clientY, options);
    });

    return {
        nameInput,
        visibilityBtn,
        row,
        /**
         * Bind or replace the visibility callback.
         * Also resets the visual state to "visible".
         */
        setVisibilityCallback(cb) {
            _visibilityCallback = cb;
            isVisible = true;
            visibilityIcon.src = `${iconBase}visibility-on.png`;
            visibilityBtn.classList.remove('off');
        }
    };
}

export function createClassItem(name, iconSrc, parent) {
    const row = document.createElement('div');
    row.classList.add('outline-item', 'class-item');

    const mainInfo = document.createElement('div');
    mainInfo.classList.add('outline-main-info');

    const icon = document.createElement('img');
    icon.src = iconSrc;
    icon.classList.add('outline-icon');
    icon.alt = "Class Icon";

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.value = name;
    nameInput.classList.add('outline-name-input');

    mainInfo.appendChild(icon);
    mainInfo.appendChild(nameInput);

    row.appendChild(mainInfo);

    // --- Register this class in the global registry ---
    const classId = _classIdCounter++;
    const defaultColors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"];
    let currentColor = defaultColors[Math.floor(Math.random() * defaultColors.length)];
    classRegistry.set(classId, { name, color: currentColor });

    // Keep registry in sync when name is edited
    nameInput.addEventListener('change', () => {
        const entry = classRegistry.get(classId);
        if (entry) entry.name = nameInput.value;
    });

    // Color Selector
    const colorWrapper = document.createElement('div');
    colorWrapper.classList.add('class-color-wrapper');

    const colorDot = document.createElement('div');
    colorDot.classList.add('class-color-dot');
    colorDot.style.backgroundColor = currentColor;

    colorDot.addEventListener('click', (e) => {
        e.stopPropagation();
        showColorPicker(colorDot, currentColor, (newColor) => {
            currentColor = newColor;
            colorDot.style.backgroundColor = newColor;
            // Keep registry in sync
            const entry = classRegistry.get(classId);
            if (entry) entry.color = newColor;

            // Update points in the scene if loader is available
            const sceneRef = window.__babylonScene;
            if (sceneRef && sceneRef.potree2Loader) {
                sceneRef.potree2Loader.updateClassColor(classId, newColor);
            }
        });
    });

    colorWrapper.appendChild(colorDot);
    row.appendChild(colorWrapper);

    parent.appendChild(row);

    // Right-click context menu
    row.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const iconBase = "static/viewer/icons/";
        showContextMenu(e.clientX, e.clientY, [
            {
                label: `Assign "${nameInput.value}" to selection`,
                icon: `${iconBase}cursor.png`,
                action: () => {
                    // Import scene from main.js at call-time to avoid circular deps
                    const sceneRef = window.__babylonScene;
                    if (!sceneRef) {
                        console.warn("⚠️ Scene not available. Make sure window.__babylonScene is set in main.js.");
                        return;
                    }
                    applyClassToSelection(sceneRef, classId, currentColor);
                    console.log(`✅ Class "${nameInput.value}" (id:${classId}, ${currentColor}) assigned to selection.`);
                }
            },
            {
                label: `Delete "${nameInput.value}"`,
                icon: `${iconBase}trash.png`,
                action: () => {
                    if (confirm(`Are you sure you want to delete the class "${nameInput.value}"?`)) {
                        const sceneRef = window.__babylonScene;
                        if (sceneRef?.potree2Loader?.removeClass) {
                            sceneRef.potree2Loader.removeClass(classId);
                        }
                        classRegistry.delete(classId);
                        row.remove();
                    }
                }
            }
        ]);
    });

    return nameInput;
}

export function showContextMenu(x, y, options) {
    // Remove existing menus
    const oldMenu = document.querySelector('.context-menu');
    if (oldMenu) oldMenu.remove();

    const menu = document.createElement('div');
    menu.classList.add('context-menu');
    menu.style.left = `${x}px`;
    menu.style.top = `${y}px`;

    options.forEach(opt => {
        const item = document.createElement('button');
        item.classList.add('context-menu-item');

        if (opt.icon) {
            const img = document.createElement('img');
            img.src = opt.icon;
            item.appendChild(img);
        }

        const span = document.createElement('span');
        span.textContent = opt.label;
        item.appendChild(span);

        item.addEventListener('click', () => {
            opt.action();
            menu.remove();
        });

        menu.appendChild(item);
    });

    document.body.appendChild(menu);

    // Close the menu if clicked elsewhere
    setTimeout(() => {
        const closeMenu = (e) => {
            if (!menu.contains(e.target)) {
                menu.remove();
                document.removeEventListener('mousedown', closeMenu);
            }
        };
        document.addEventListener('mousedown', closeMenu);
    }, 10);
}

// Helper to convert HSV to HEX
function hsvToHex(h, s, v) {
    h /= 360; s /= 100; v /= 100;
    let r, g, b;
    let i = Math.floor(h * 6);
    let f = h * 6 - i;
    let p = v * (1 - s);
    let q = v * (1 - f * s);
    let t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
    const toHex = x => {
        const hex = Math.round(x * 255).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    };
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export function showColorPicker(anchor, currentColor, onSelect) {
    const overlay = document.createElement('div');
    overlay.classList.add('color-picker-overlay');
    const popover = document.createElement('div');
    popover.classList.add('color-picker-popover');
    const rect = anchor.getBoundingClientRect();
    popover.style.left = `${rect.right + 15}px`;
    popover.style.top = `${rect.top - 20}px`;

    let h = 0, s = 100, v = 100;

    const svContainer = document.createElement('div');
    svContainer.classList.add('sv-canvas-container');
    svContainer.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, transparent), #ff0000`;
    const svCursor = document.createElement('div');
    svCursor.classList.add('sv-cursor');
    svContainer.appendChild(svCursor);

    const hueContainer = document.createElement('div');
    hueContainer.classList.add('hue-slider-container');
    const hueCursor = document.createElement('div');
    hueCursor.classList.add('hue-cursor');
    hueContainer.appendChild(hueCursor);

    const hexInputGroup = document.createElement('div');
    hexInputGroup.classList.add('hex-input-group');
    hexInputGroup.innerHTML = `<span>#</span>`;
    const hexInputField = document.createElement('input');
    hexInputField.type = 'text';
    hexInputField.classList.add('hex-field');
    hexInputField.value = currentColor.replace('#', '');
    hexInputGroup.appendChild(hexInputField);

    const presets = document.createElement('div');
    presets.classList.add('color-presets');
    const defaultColors = ["#ef4444", "#f97316", "#f59e0b", "#22c55e", "#10b981", "#0ea5e9", "#3b82f6", "#6366f1", "#d946ef", "#64748b"];
    defaultColors.forEach(col => {
        const p = document.createElement('div');
        p.classList.add('preset-color');
        p.style.backgroundColor = col;
        p.onclick = () => { onSelect(col); hexInputField.value = col.replace('#', ''); };
        presets.appendChild(p);
    });

    function updateColor() {
        const hex = hsvToHex(h, s, v);
        hexInputField.value = hex.replace('#', '');
        svContainer.style.backgroundColor = hsvToHex(h, 100, 100);
        onSelect(hex);
    }

    let isDraggingSV = false;
    let isDraggingHue = false;

    const handleSV = (e) => {
        const r = svContainer.getBoundingClientRect();
        let x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
        let y = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height));
        s = x * 100; v = (1 - y) * 100;
        svCursor.style.left = `${x * 100}%`; svCursor.style.top = `${y * 100}%`;
        updateColor();
    };

    const handleHue = (e) => {
        const r = hueContainer.getBoundingClientRect();
        let x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
        h = x * 360; hueCursor.style.left = `${x * 100}%`;
        updateColor();
    };

    svContainer.onmousedown = (e) => { isDraggingSV = true; handleSV(e); };
    hueContainer.onmousedown = (e) => { isDraggingHue = true; handleHue(e); };

    const moveListener = (e) => { if (isDraggingSV) handleSV(e); if (isDraggingHue) handleHue(e); };
    const upListener = () => { isDraggingSV = false; isDraggingHue = false; };

    window.addEventListener('mousemove', moveListener);
    window.addEventListener('mouseup', upListener);

    popover.appendChild(svContainer);
    popover.appendChild(hueContainer);
    popover.appendChild(hexInputGroup);
    popover.appendChild(presets);

    function close() {
        popover.style.opacity = '0'; popover.style.transform = 'scale(0.95)';
        window.removeEventListener('mousemove', moveListener);
        window.removeEventListener('mouseup', upListener);
        setTimeout(() => { overlay.remove(); popover.remove(); }, 150);
    }

    overlay.onclick = close;
    document.body.appendChild(overlay);
    document.body.appendChild(popover);
}

export function createModal(title, contentCallback, footerCallback) {
    const overlay = document.createElement('div');
    overlay.classList.add('modal-overlay');

    const container = document.createElement('div');
    container.classList.add('modal-container');

    // Header
    const header = document.createElement('div');
    header.classList.add('modal-header');

    const titleEl = document.createElement('span');
    titleEl.classList.add('modal-title');
    titleEl.textContent = title;

    const closeBtn = document.createElement('button');
    closeBtn.classList.add('modal-close');
    closeBtn.innerHTML = '&times;';
    closeBtn.onclick = () => {
        overlay.classList.remove('active');
        setTimeout(() => overlay.remove(), 300);
    };

    header.appendChild(titleEl);
    header.appendChild(closeBtn);

    // Body
    const body = document.createElement('div');
    body.classList.add('modal-body');
    if (contentCallback) contentCallback(body);

    // Footer
    const footer = document.createElement('div');
    footer.classList.add('modal-footer');
    if (footerCallback) footerCallback(footer, overlay);

    container.appendChild(header);
    container.appendChild(body);
    container.appendChild(footer);
    overlay.appendChild(container);
    document.body.appendChild(overlay);

    // Show with animation
    setTimeout(() => overlay.classList.add('active'), 10);

    return overlay;
}

export function showResetSceneModal() {
    const overlay = createModal(
        "Reset Scene",
        (body) => {
            const warn = document.createElement('div');
            warn.style.cssText = "display:flex;align-items:flex-start;gap:12px;margin-bottom:16px;";

            const icon = document.createElement('div');
            icon.innerHTML = `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;margin-top:2px;">
                <path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                    stroke="#f59e0b" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>`;

            const msg = document.createElement('div');
            msg.style.cssText = "font-size:0.78rem;line-height:1.55;color:var(--text-primary);";
            msg.innerHTML = `<strong style="display:block;margin-bottom:4px;">This action cannot be undone.</strong>
                All files in <code style="background:var(--bg-input,#1e1e22);padding:1px 5px;border-radius:4px;font-size:0.75rem;">/data/working</code>
                will be permanently deleted, including the loaded point cloud, computed features, and any unsaved data.
                <br><br>The scene will be reset to its initial empty state.`;

            warn.appendChild(icon);
            warn.appendChild(msg);
            body.appendChild(warn);
        },
        (footer, overlay) => {
            const cancelBtn = document.createElement('button');
            cancelBtn.classList.add('btn');
            cancelBtn.style.cssText = "background:transparent;border:1px solid var(--border-color);";
            cancelBtn.textContent = "Cancel";
            cancelBtn.onclick = () => {
                overlay.classList.remove('active');
                setTimeout(() => overlay.remove(), 300);
            };

            const confirmBtn = document.createElement('button');
            confirmBtn.classList.add('btn');
            confirmBtn.style.cssText = "background:#dc2626;border:none;";
            confirmBtn.textContent = "Reset Scene";
            confirmBtn.onclick = async () => {
                const body = overlay.querySelector('.modal-body');
                const modalCloseBtn = overlay.querySelector('.modal-close');
                if (modalCloseBtn) modalCloseBtn.style.display = 'none';

                footer.innerHTML = '';
                body.innerHTML = '';

                const stepList = document.createElement('div');
                stepList.classList.add('pipeline-step-list');

                const STEPS = ["Resetting Viewer", "Clearing Working Directory"];
                const stepEls = STEPS.map((label) => {
                    const item = document.createElement('div');
                    item.classList.add('pipeline-step-item', 'step-idle');
                    item.innerHTML = `
                        <div class="step-icon-wrap">
                            <svg class="step-svg" viewBox="0 0 32 32">
                                <circle class="step-track" cx="16" cy="16" r="13"/>
                                <circle class="step-arc"   cx="16" cy="16" r="13"/>
                            </svg>
                            <div class="step-check">
                                <svg viewBox="0 0 12 12" fill="none">
                                    <polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                            </div>
                            <div class="step-cross">
                                <svg viewBox="0 0 12 12" fill="none">
                                    <line x1="2.5" y1="2.5" x2="9.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                    <line x1="9.5" y1="2.5" x2="2.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                </svg>
                            </div>
                        </div>
                        <span class="step-label">${label}</span>`;
                    stepList.appendChild(item);
                    return item;
                });
                body.appendChild(stepList);

                const activateStep = (i) => {
                    stepEls[i].classList.remove('step-idle', 'step-done', 'step-error');
                    stepEls[i].classList.add('step-running');
                };
                const completeStep = (i) => {
                    stepEls[i].classList.remove('step-running');
                    stepEls[i].classList.add('step-done');
                };
                const failStep = (i) => {
                    stepEls[i].classList.remove('step-running');
                    stepEls[i].classList.add('step-error');
                };

                let currentStep = 0;

                try {
                    // STEP 1: Reset viewer state first, so no stale loader callbacks
                    // keep running while backend files are being deleted.
                    currentStep = 0;
                    activateStep(currentStep);
                    const scene = window.__babylonScene;
                    if (scene) {
                        const loader = scene.potree2Loader;
                        if (loader && typeof loader.dispose === 'function') {
                            loader.dispose();
                        }
                        scene.potree2Loader = null;
                    }
                    const sceneObjects = window.__sceneObjects;
                    if (sceneObjects) {
                        if (sceneObjects.currentPointCloud) {
                            try { sceneObjects.currentPointCloud.dispose(); } catch (_) { }
                            sceneObjects.currentPointCloud = null;
                        }
                    }
                    // Clear class registry and project references
                    resetClassRegistry();
                    window.__currentProjectLAS = null;
                    window.__currentProjectBIN = null;
                    window.__selectedModelPath = null;
                    window.dispatchEvent(new CustomEvent('scene-reset'));
                    completeStep(currentStep);

                    // STEP 2: Clear working directory
                    currentStep = 1;
                    activateStep(currentStep);
                    const clearRes = await fetch('/api/clear-data/', {
                        method: 'POST',
                        headers: { 'X-CSRFToken': getCSRFToken() }
                    });
                    if (!clearRes.ok) {
                        const errData = await clearRes.json().catch(() => ({}));
                        throw new Error(errData.error || "Failed to clear working directory");
                    }
                    completeStep(currentStep);

                    await new Promise(r => setTimeout(r, 700));
                    overlay.classList.remove('active');
                    setTimeout(() => overlay.remove(), 300);

                } catch (err) {
                    console.error("❌ Reset Scene Error:", err);
                    failStep(currentStep);

                    const errorDiv = document.createElement('div');
                    errorDiv.classList.add('pipeline-error-msg');
                    errorDiv.innerHTML = `
                        <div style="font-weight:600;margin-bottom:4px;">⚠️ Reset Failed</div>
                        <div style="font-family:monospace;font-size:0.8em;white-space:pre-wrap;word-break:break-word;opacity:0.8;">${err.message || err.toString()}</div>
                    `;
                    body.appendChild(errorDiv);

                    if (modalCloseBtn) modalCloseBtn.style.display = '';
                    const closeBtn = document.createElement('button');
                    closeBtn.classList.add('btn');
                    closeBtn.textContent = "Close";
                    closeBtn.onclick = () => {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    };
                    footer.appendChild(closeBtn);
                }
            };

            footer.appendChild(cancelBtn);
            footer.appendChild(confirmBtn);
        }
    );
    return overlay;
}

export function showDownloadModal() {
    createModal(
        "Download Data",
        (body) => {
            // Tracks checkboxes
            const pointCloudFileCheckboxes = [];  // { path, label, checkbox }
            const modelCheckboxes = [];    // { name, checkbox }

            // ── Helper: section separator ──────────────────────────────────
            function makeSeparator() {
                const sep = document.createElement('div');
                sep.style.cssText = 'height:1px; background:var(--border-color); margin:4px 0;';
                return sep;
            }

            // ── SECTION 1: Point Cloud Segments ────────────────────────────
            const segSection = document.createElement('div');
            segSection.classList.add('modal-section');

            const segTitle = document.createElement('div');
            segTitle.classList.add('modal-section-title');
            segTitle.textContent = 'Point Cloud Segments';
            segSection.appendChild(segTitle);

            // Scrollable list container
            const segScroll = document.createElement('div');
            segScroll.classList.add('dl-list-scroll');
            segSection.appendChild(segScroll);

            // Show only the final pipeline output (predicted.las), if present.
            // Prefer the currently selected model path, then fall back to all known models.
            (async () => {
                const loading = document.createElement('div');
                loading.style.cssText = 'font-size:0.78rem; color:var(--text-muted); padding:8px 4px;';
                loading.textContent = 'Checking predicted output…';
                segScroll.appendChild(loading);

                const candidates = [];
                const seen = new Set();
                const addCandidate = (modelName) => {
                    if (!modelName) return;
                    const relPath = _runtimePath('working', modelName, 'predicted.las');
                    const publicUrl = _runtimeUrl('working', modelName, 'predicted.las');
                    if (seen.has(relPath)) return;
                    seen.add(relPath);
                    candidates.push({
                        path: relPath,
                        publicUrl,
                        label: 'predicted.las',
                        modelName,
                    });
                };

                const selectedModelPath = window.__selectedModelPath;
                if (typeof selectedModelPath === 'string' && selectedModelPath.length > 0) {
                    const modelName = selectedModelPath
                        .replace(/\\/g, '/')
                        .replace(/\/model\.pkl$/, '')
                        .split('/')
                        .pop();
                    addCandidate(modelName);
                }

                try {
                    const res = await fetch('/api/models-list/');
                    const data = await res.json();
                    const models = data.models || [];
                    models.forEach(m => addCandidate(m.name));
                } catch (_) {
                    // Ignore model list failures here: we still try selected model candidate.
                }

                const availableFiles = [];
                for (const candidate of candidates) {
                    try {
                        const headRes = await fetch(candidate.publicUrl, { method: 'HEAD' });
                        if (headRes.ok) availableFiles.push(candidate);
                    } catch (_) {
                        // Ignore missing/unreachable candidate files.
                    }
                }

                segScroll.innerHTML = '';

                if (availableFiles.length === 0) {
                    const empty = document.createElement('div');
                    empty.style.cssText = 'font-size:0.78rem; color:var(--text-muted); padding:8px 4px;';
                    empty.textContent = 'No predicted.las found.';
                    segScroll.appendChild(empty);
                    return;
                }

                availableFiles.forEach(file => {
                    const cb = createSimpleCheckbox(file.label, true, segScroll);
                    const labelSpan = cb.parentElement.querySelector('span:last-child');
                    if (labelSpan) {
                        labelSpan.innerHTML = `<span style="opacity:0.7; margin-right:6px;">🗃️</span>${file.label} <span style="opacity:0.7;">(${file.modelName})</span>`;
                    }
                    pointCloudFileCheckboxes.push({ path: file.path, label: file.label, checkbox: cb });
                });
            })();

            body.appendChild(segSection);
            body.appendChild(makeSeparator());

            // ── SECTION 2: Trained Models ──────────────────────────────────
            const modelSection = document.createElement('div');
            modelSection.classList.add('modal-section');

            const modelTitle = document.createElement('div');
            modelTitle.classList.add('modal-section-title');
            modelTitle.textContent = 'Trained Models';
            modelSection.appendChild(modelTitle);

            const modelScroll = document.createElement('div');
            modelScroll.classList.add('dl-list-scroll');
            modelSection.appendChild(modelScroll);

            // Placeholder while loading
            const loadingEl = document.createElement('div');
            loadingEl.style.cssText = 'font-size:0.78rem; color:var(--text-muted); padding:8px 4px;';
            loadingEl.textContent = 'Loading models…';
            modelScroll.appendChild(loadingEl);

            // Load models asynchronously
            (async () => {
                try {
                    const res = await fetch('/api/models-list/');
                    const data = await res.json();
                    const models = data.models || [];
                    modelScroll.innerHTML = '';

                    if (models.length === 0) {
                        const empty = document.createElement('div');
                        empty.style.cssText = 'font-size:0.78rem; color:var(--text-muted); padding:8px 4px;';
                        empty.textContent = 'No trained models available.';
                        modelScroll.appendChild(empty);
                        return;
                    }

                    models.forEach(model => {
                        // Build a richer label with name + metadata
                        const cb = createSimpleCheckbox(model.name, false, modelScroll);
                        // Enrich label with icon and meta
                        const labelSpan = cb.parentElement.querySelector('span:last-child');
                        if (labelSpan) {
                            labelSpan.style.cssText = 'display:flex; flex-direction:column; gap:1px;';
                            labelSpan.innerHTML = `
                                <span style="display:flex; align-items:center; gap:5px;">
                                    <span style="opacity:0.75;">🤖</span>
                                    <span style="font-weight:600;">${model.name}</span>
                                </span>
                                <span style="font-size:0.7rem; color:var(--text-muted); padding-left:2px;">${model.created} · ${model.size_mb} MB</span>`;
                        }
                        modelCheckboxes.push({ name: model.name, checkbox: cb });
                    });
                } catch (err) {
                    modelScroll.innerHTML = '';
                    const errEl = document.createElement('div');
                    errEl.style.cssText = 'font-size:0.78rem; color:#f87171; padding:8px 4px;';
                    errEl.textContent = '⚠️ Could not load model list.';
                    modelScroll.appendChild(errEl);
                }
            })();

            body.appendChild(modelSection);

            // Expose selection getters for the footer
            body._getSelectedPointCloudFiles = () => pointCloudFileCheckboxes
                .filter(f => f.checkbox.checked)
                .map(f => ({ path: f.path, label: f.label }));
            body._getSelectedModels = () => modelCheckboxes
                .filter(m => m.checkbox.checked)
                .map(m => m.name);
        },
        (footer, overlay) => {
            const cancelBtn = document.createElement('button');
            cancelBtn.classList.add('btn');
            cancelBtn.style.cssText = 'background:transparent; border:1px solid var(--border-color);';
            cancelBtn.textContent = "Cancel";
            cancelBtn.onclick = () => {
                overlay.classList.remove('active');
                setTimeout(() => overlay.remove(), 300);
            };

            const downloadBtn = document.createElement('button');
            downloadBtn.classList.add('btn');
            downloadBtn.textContent = "Download";
            downloadBtn.onclick = async () => {
                const body = overlay.querySelector('.modal-body');
                const pointCloudFiles = body?._getSelectedPointCloudFiles?.() ?? [];
                const models = body?._getSelectedModels?.() ?? [];

                if (pointCloudFiles.length === 0 && models.length === 0) {
                    alert('Please select at least one item to download.');
                    return;
                }

                downloadBtn.disabled = true;
                downloadBtn.textContent = 'Preparing Package...';

                try {
                    downloadBtn.textContent = 'Preparing Package...';
                    const response = await fetch('/api/download-package/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': getCSRFToken()
                        },
                        body: JSON.stringify({
                            segments: [],
                            point_cloud_files: pointCloudFiles,
                            models,
                            las_path: window.__currentProjectLAS
                        })
                    });

                    if (!response.ok) {
                        const err = await response.json().catch(() => ({}));
                        throw new Error(err.error || 'Download failed');
                    }

                    // Handle binary blob
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);

                    // Extract filename from Content-Disposition if present
                    const disposition = response.headers.get('Content-Disposition');
                    let filename = 'download_package.zip';
                    if (disposition && disposition.includes('filename=')) {
                        filename = disposition.split('filename=')[1]
                            .split(';')[0]
                            .replace(/['"]/g, '').trim();
                    }

                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    a.remove();

                    // Success message - don't close automatically
                    const modalBody = overlay.querySelector('.modal-body');
                    const modalFooter = overlay.querySelector('.modal-footer');
                    
                    modalBody.innerHTML = '';
                    const successMsg = document.createElement('div');
                    successMsg.style.cssText = "display:flex;align-items:flex-start;gap:12px;margin:0;";

                    const icon = document.createElement('div');
                    icon.innerHTML = `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;margin-top:2px;">
                        <path d="M20 6L9 17l-5-5" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>`;

                    const msg = document.createElement('div');
                    msg.style.cssText = "font-size:0.85rem;line-height:1.6;color:var(--text-primary);";
                    msg.innerHTML = `<strong style="display:block;margin-bottom:4px;">Download started!</strong>
                        Your package has been prepared and the download should begin automatically.
                        <br><br><span style="opacity:0.75;">File: <code style="background:var(--bg-input,#1e1e22);padding:2px 6px;border-radius:4px;font-size:0.75rem;">${filename}</code></span>`;

                    successMsg.appendChild(icon);
                    successMsg.appendChild(msg);
                    modalBody.appendChild(successMsg);

                    // Update footer with Close button
                    modalFooter.innerHTML = '';
                    const closeBtn = document.createElement('button');
                    closeBtn.classList.add('btn');
                    closeBtn.textContent = "Close";
                    closeBtn.onclick = () => {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    };
                    modalFooter.appendChild(closeBtn);

                } catch (err) {
                    console.error('Download error:', err);
                    alert('Error creating package: ' + err.message);
                    downloadBtn.disabled = false;
                    downloadBtn.textContent = 'Download';
                }
            };

            footer.appendChild(cancelBtn);
            footer.appendChild(downloadBtn);
        }
    );
}

export function showLoadModal() {
    let fileInputEl = null;

    let meshPointsRow = null;
    let meshPointsInput = null;

    let subToggle = null;
    let voxelInput = null;
    let subContainerEl = null;
    let subSettingsEl = null;

    createModal(
        "Load Data",
        (body) => {
            // Section 1: File Selection
            const fileSection = document.createElement('div');
            fileSection.classList.add('modal-section');

            const fileTitle = document.createElement('div');
            fileTitle.classList.add('modal-section-title');
            fileTitle.textContent = "Data Source";
            fileSection.appendChild(fileTitle);

            const fileContainer = document.createElement('div');
            fileContainer.classList.add('file-input-container');

            const fileCustom = document.createElement('div');
            fileCustom.classList.add('file-input-custom');
            fileCustom.innerHTML = `
                <b>Click to select file</b>
                <span>Supports .ply, .las, .glb</span>
            `;

            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.classList.add('file-input-hidden');
            fileInput.accept = ".ply,.las,.glb";
            fileInputEl = fileInput;

            fileInput.onchange = (e) => {
                const fileName = e.target.files[0]?.name;
                if (fileName) {
                    fileCustom.innerHTML = `<b>Selected:</b> <span>${fileName}</span>`;
                    if (subSection) {
                        subSection.style.display = '';
                    }
                    const ext = fileName.split('.').pop().toLowerCase();
                    if (meshPointsRow) {
                        meshPointsRow.style.display = ext === 'glb' ? "flex" : "none";
                    }
                    // For GLB, subsampling is not applicable (GLB has mesh-point control).
                    if (subContainerEl && subSettingsEl && subToggle) {
                        const isGlb = ext === 'glb';
                        subContainerEl.style.display = isGlb ? 'none' : '';
                        if (isGlb) {
                            subToggle.checked = false;
                            subSettingsEl.classList.remove('visible');
                        }
                    }
                } else {
                    fileCustom.innerHTML = `
                <b>Click to select file</b>
                <span>Supports .ply, .las, .glb</span>
            `;
                    if (subSection) {
                        subSection.style.display = 'none';
                    }
                }
            };

            fileContainer.appendChild(fileCustom);
            fileContainer.appendChild(fileInput);
            fileSection.appendChild(fileContainer);
            body.appendChild(fileSection);

            // Spacer
            const spacer = document.createElement('div');
            spacer.style.height = '10px';
            body.appendChild(spacer);

            // Section 2: Subsampling
            const subSection = document.createElement('div');
            subSection.classList.add('modal-section');
            subSection.style.display = 'none';

            const subTitle = document.createElement('div');
            subTitle.classList.add('modal-section-title');
            subTitle.textContent = "Processing";
            subSection.appendChild(subTitle);

            const subContainer = document.createElement('div');
            subToggle = createSimpleCheckbox("Enable Subsampling", false, subContainer);
            subContainerEl = subContainer;

            const subSettings = document.createElement('div');
            subSettings.classList.add('expandable-content');
            subSettingsEl = subSettings;

            // Add settings to subSettings
            // Voxel Size Slider
            const voxelRow = document.createElement('div');
            voxelRow.classList.add('property-row');
            voxelRow.style.padding = "0 4px";
            voxelRow.innerHTML = `<span class="property-label">Voxel Size (m):</span>`;
            voxelInput = document.createElement('input');
            voxelInput.type = "number";
            voxelInput.classList.add('property-input');
            voxelInput.value = "0.05";
            voxelInput.step = "0.01";
            voxelInput.min = "0.01";
            voxelRow.appendChild(voxelInput);
            subSettings.appendChild(voxelRow);

            subToggle.onchange = (e) => {
                subSettings.classList.toggle('visible', e.target.checked);
            };

            subSection.appendChild(subContainer);
            subSection.appendChild(subSettings);

            // New: Mesh Points (for GLB)
            meshPointsRow = document.createElement('div');
            meshPointsRow.classList.add('property-row');
            meshPointsRow.style.padding = "10px 4px 0 4px";
            meshPointsRow.style.display = "none"; // hidden by default
            meshPointsRow.innerHTML = `<span class="property-label">GLB Mesh Points:</span>`;
            meshPointsInput = document.createElement('input');
            meshPointsInput.type = "number";
            meshPointsInput.classList.add('property-input');
            meshPointsInput.value = "10000000";
            meshPointsInput.step = "100000";
            meshPointsInput.min = "10000";
            meshPointsInput.style.width = "100px";
            meshPointsRow.appendChild(meshPointsInput);
            subSection.appendChild(meshPointsRow);

            body.appendChild(subSection);
        },
        (footer, overlay) => {
            const cancelBtn = document.createElement('button');
            cancelBtn.classList.add('btn');
            cancelBtn.style.backgroundColor = 'transparent';
            cancelBtn.style.border = '1px solid var(--border-color)';
            cancelBtn.textContent = "Cancel";
            cancelBtn.onclick = () => {
                overlay.classList.remove('active');
                setTimeout(() => overlay.remove(), 300);
            };

            const uploadBtn = document.createElement('button');
            uploadBtn.classList.add('btn');
            uploadBtn.textContent = "Upload & Process";
            uploadBtn.onclick = async () => {
                const file = fileInputEl?.files[0];
                if (!file) {
                    alert("Please select a file first.");
                    return;
                }

                console.log("🚀 Starting complex processing pipeline for:", file.name);

                // ── Replace modal body with step list view ─────────────────
                const container = overlay.querySelector('.modal-container');
                const body = overlay.querySelector('.modal-body');
                const footer = overlay.querySelector('.modal-footer');

                // Swap content: hide original body, show step list
                body.innerHTML = '';

                // Hide the X close button during pipeline
                const modalCloseBtn = overlay.querySelector('.modal-close');
                if (modalCloseBtn) modalCloseBtn.style.display = 'none';

                // Show footer with Cancel Import button
                let isCancelled = false;
                if (footer) {
                    footer.style.display = '';
                    footer.innerHTML = '';
                    const cancelImportBtn = document.createElement('button');
                    cancelImportBtn.classList.add('btn');
                    cancelImportBtn.style.backgroundColor = 'transparent';
                    cancelImportBtn.style.border = '1px solid var(--border-color)';
                    cancelImportBtn.textContent = 'Cancel Import';
                    cancelImportBtn.onclick = async () => {
                        if (isCancelled) return;
                        isCancelled = true;
                        cancelImportBtn.disabled = true;
                        cancelImportBtn.textContent = 'Cancelling\u2026';
                        try {
                            const stopRes = await fetch('/stop_process/', {
                                method: 'POST',
                                headers: { 'X-CSRFToken': getCSRFToken() }
                            });
                            if (!stopRes.ok) throw new Error('Stop request failed');
                        } catch (e) {
                            console.warn('\u26a0\ufe0f Could not call stop_process:', e);
                        }
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    };
                    footer.appendChild(cancelImportBtn);
                }

                const STEPS = [
                    "Clearing Workspace",
                    "Uploading File",
                    "Converting File",
                    "Potree Converter",
                    "Loading Viewer"
                ];

                // Build step list
                const stepList = document.createElement('div');
                stepList.classList.add('pipeline-step-list');

                const stepEls = STEPS.map((label) => {
                    const item = document.createElement('div');
                    item.classList.add('pipeline-step-item', 'step-idle');

                    const iconWrap = document.createElement('div');
                    iconWrap.classList.add('step-icon-wrap');
                    // Inner ring (the animated spinner arc + static base ring)
                    iconWrap.innerHTML = `
                        <svg class="step-svg" viewBox="0 0 32 32">
                            <circle class="step-track" cx="16" cy="16" r="13" />
                            <circle class="step-arc"   cx="16" cy="16" r="13" />
                        </svg>
                        <div class="step-check">
                            <svg viewBox="0 0 12 12" fill="none">
                                <polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="step-cross">
                            <svg viewBox="0 0 12 12" fill="none">
                                <line x1="2.5" y1="2.5" x2="9.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                <line x1="9.5" y1="2.5" x2="2.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                            </svg>
                        </div>
                    `;

                    const labelEl = document.createElement('span');
                    labelEl.classList.add('step-label');
                    labelEl.textContent = label;

                    item.appendChild(iconWrap);
                    item.appendChild(labelEl);
                    stepList.appendChild(item);
                    return item;
                });

                body.appendChild(stepList);

                // Helpers
                const activateStep = (idx) => {
                    stepEls[idx].classList.remove('step-idle', 'step-done', 'step-error');
                    stepEls[idx].classList.add('step-running');
                };
                const completeStep = (idx) => {
                    stepEls[idx].classList.remove('step-running');
                    stepEls[idx].classList.add('step-done');
                };
                const failStep = (idx) => {
                    stepEls[idx].classList.remove('step-running');
                    stepEls[idx].classList.add('step-error');
                };

                // Track current step for error handling
                let currentStepIdx = 0;

                // Sentinel error to silently abort the pipeline when user cancels
                const CANCELLED = Symbol('CANCELLED');
                const checkCancelled = () => { if (isCancelled) throw CANCELLED; };

                try {
                    // STEP 1: Clear current data directory
                    checkCancelled();
                    currentStepIdx = 0;
                    activateStep(0);
                    console.log("🧹 Step 1: Clearing /static/viewer/data/working/...");
                    const clearResponse = await fetch('/api/clear-data/', { method: 'POST' });
                    if (!clearResponse.ok) throw new Error("Failed to clear data directory");
                    completeStep(0);
                    console.log("✅ Workspace cleared");

                    // STEP 2: Upload the source file
                    checkCancelled();
                    currentStepIdx = 1;
                    activateStep(1);
                    console.log("📤 Step 2: Uploading file to /static/viewer/data/working/...");
                    const formData = new FormData();
                    formData.append('file', file);
                    const uploadResponse = await fetch('/api/upload-data/', {
                        method: 'POST',
                        body: formData
                    });
                    if (!uploadResponse.ok) {
                        const errData = await uploadResponse.json().catch(() => ({}));
                        throw new Error(errData.error || "Failed to upload file");
                    }
                    completeStep(1);
                    console.log("✅ File uploaded");

                    // STEP 3: Handle conversion based on extension
                    checkCancelled();
                    currentStepIdx = 2;
                    activateStep(2);
                    const filename = file.name;
                    const extension = filename.split('.').pop().toLowerCase();
                    const inputPath = _runtimePath('working', filename);
                    let lasPath = _runtimePath('working', 'features.las');
                    const subsampleEnabled = Boolean(subToggle?.checked) && extension !== 'glb';
                    const rawVoxel = parseFloat(voxelInput?.value || '0.05');
                    const voxelSize = Number.isFinite(rawVoxel) && rawVoxel > 0 ? rawVoxel : 0.05;

                    if (extension === 'ply') {
                        let plyInputPath = inputPath;
                        if (subsampleEnabled) {
                            console.log(`🔧 Step 3: Subsampling PLY (voxel ${voxelSize} m)...`);
                            const plySubsamplePath = _runtimePath('working', 'features.ply');
                            const subResponse = await fetch('/subsample_pc/', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                                body: JSON.stringify({
                                    file_path: inputPath,
                                    out_path: plySubsamplePath,
                                    voxel_size: voxelSize,
                                })
                            });
                            if (!subResponse.ok) {
                                const errData = await subResponse.json().catch(() => ({}));
                                throw new Error(errData.message || errData.error || 'PLY subsampling failed');
                            }
                            plyInputPath = plySubsamplePath;
                            console.log('✅ Step 3: PLY subsampling completed');
                        }

                        console.log("🔄 Step 3: Converting PLY to LAS...");
                        const plyResponse = await fetch('/ply2las/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                            body: JSON.stringify({ file_path: plyInputPath, out_path: lasPath })
                        });
                        if (!plyResponse.ok) {
                            const errData = await plyResponse.json().catch(() => ({}));
                            throw new Error(errData.message || errData.error || "PLY conversion failed");
                        }
                        console.log("✅ PLY converted to LAS");
                    } else if (extension === 'glb') {
                        console.log("🔄 Step 3: Converting GLB to LAS...");
                        let numPoints = 10000000;
                        if (meshPointsInput && meshPointsInput.value) numPoints = parseInt(meshPointsInput.value, 10);
                        const meshResponse = await fetch('/mesh2pc/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                            body: JSON.stringify({ file_path: inputPath, out_path: lasPath, num_points: numPoints })
                        });
                        if (!meshResponse.ok) {
                            const errData = await meshResponse.json().catch(() => ({}));
                            throw new Error(errData.message || errData.error || "Mesh sampling failed");
                        }
                        console.log(`✅ Mesh sampled to point cloud (${numPoints} points)`);
                    } else if (extension === 'las') {
                        console.log('🔎 Step 3: Validating LAS input format...');
                        const inspectResponse = await fetch('/inspect_las_input/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                            body: JSON.stringify({ file_path: inputPath })
                        });
                        if (!inspectResponse.ok) {
                            const errData = await inspectResponse.json().catch(() => ({}));
                            throw new Error(errData.message || errData.error || 'LAS format inspection failed');
                        }
                        const inspectData = await inspectResponse.json();
                        if (!inspectData.is_canonical) {
                            throw new Error(
                                `LAS input must be LAS 1.2 / Point Format 3.\n` +
                                `Received LAS ${inspectData.version} / Point Format ${inspectData.point_format}. ` +
                                `Please convert the input LAS to this format or import a ply or glb file.`
                            );
                        }
                        console.log('✅ Input LAS format validated (LAS 1.2 / Point Format 3)');

                        let lasInputPath = inputPath;
                        if (subsampleEnabled) {
                            console.log(`🔧 Step 3: Subsampling LAS (voxel ${voxelSize} m)...`);
                            const lasSubsamplePath = _runtimePath('working', 'features.las');
                            const subResponse = await fetch('/subsample_pc/', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                                body: JSON.stringify({
                                    file_path: inputPath,
                                    out_path: lasSubsamplePath,
                                    voxel_size: voxelSize,
                                })
                            });
                            if (!subResponse.ok) {
                                const errData = await subResponse.json().catch(() => ({}));
                                throw new Error(errData.message || errData.error || 'LAS subsampling failed');
                            }
                            lasInputPath = lasSubsamplePath;
                            console.log('✅ Step 3: LAS subsampling completed');
                        }

                        // Normalize/add POINT_ID into canonical output path used by next steps.
                        console.log('🔄 Step 3: Checking POINT_ID on LAS...');
                        const checkResponse = await fetch('/check_point_id/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                            body: JSON.stringify({ input_path: lasInputPath, output_path: lasPath })
                        });
                        if (!checkResponse.ok) {
                            const errData = await checkResponse.json().catch(() => ({}));
                            throw new Error(errData.message || errData.error || 'check_point_id failed');
                        }
                        console.log('✅ LAS ready (POINT_ID checked)');
                    }

                    console.log("💾 Step 3: Saving backup copy of features.las...");
                    const backupResponse = await fetch('/api/backup-pointcloud/', {
                        method: 'POST',
                        headers: { 'X-CSRFToken': getCSRFToken() }
                    });
                    if (!backupResponse.ok) {
                        const errData = await backupResponse.json().catch(() => ({}));
                        throw new Error(errData.message || errData.error || 'Failed to create point cloud backup');
                    }
                    console.log("✅ Point cloud backup saved");
                    completeStep(2);

                    // STEP 4: PotreeConverter
                    checkCancelled();
                    currentStepIdx = 3;
                    activateStep(3);
                    console.log("🧠 Step 4: Starting Potree conversion...");
                    const potreeResponse = await fetch('/potree_converter/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            input_filepath: lasPath,
                            output_filepath: _runtimePath('working', 'clusters')
                        })
                    });
                    if (!potreeResponse.ok) {
                        const errData = await potreeResponse.json().catch(() => ({}));
                        throw new Error(errData.message || errData.error || "Potree conversion failed");
                    }
                    completeStep(3);
                    console.log("✅ Potree conversion completed");

                    // STEP 5: Final Loading in Babylon
                    checkCancelled();
                    currentStepIdx = 4;
                    activateStep(4);
                    console.log("🏗️ Step 5: Loading Potree data from /data/clusters/...");
                    const scene = window.__babylonScene;
                    const sceneObjects = window.__sceneObjects;
                    // Dispose previous Potree2 loader and all its meshes
                    if (scene) {
                        const prevLoader = scene.potree2Loader;
                        if (prevLoader && typeof prevLoader.dispose === 'function') {
                            prevLoader.dispose();
                        }
                        scene.potree2Loader = null;
                    }
                    if (sceneObjects && sceneObjects.currentPointCloud) {
                        if (typeof sceneObjects.currentPointCloud.dispose === 'function') {
                            sceneObjects.currentPointCloud.dispose();
                        }
                        sceneObjects.currentPointCloud = null;
                    }
                    // Clear class/segment state before loading new data
                    resetClassRegistry();
                    window.__currentProjectLAS = null;
                    window.__currentProjectBIN = null;
                    window.__selectedModelPath = null;
                    window.dispatchEvent(new CustomEvent('scene-reset'));

                    const pcPath = _runtimeUrl('working', 'clusters');
                    const pc = await loadPointCloud(pcPath, scene);
                    if (pc) {
                        if (sceneObjects) sceneObjects.currentPointCloud = pc;
                        const pcLabel = filename || "Point Cloud";
                        if (typeof window.__registerPointCloudInOutline === 'function') {
                            window.__registerPointCloudInOutline(pc, pcLabel);
                        }
                    }
                    completeStep(4);
                    console.log("✅ Process complete. Potree cloud loaded.");

                    // Store for download logic
                    window.__currentProjectLAS = _runtimePath('working', 'features.las');
                    window.__currentProjectBIN = _runtimePath('working', 'features.pcbin');

                    // Load feature bin if available (populated by Calculate Features)
                    try {
                        const featureBinLoader = window.__babylonScene?.potree2Loader;
                        if (featureBinLoader) {
                            const featureNames = await featureBinLoader.loadPcBin(_runtimeUrl('working', 'features.pcbin'));
                            window.dispatchEvent(new CustomEvent('feature-bin-loaded', { detail: { names: featureNames } }));
                        }
                    } catch (binErr) {
                        console.warn("⚠️ .pcbin store not loaded:", binErr.message);
                    }

                    await new Promise(r => setTimeout(r, 800));
                    if (!isCancelled) {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    }

                } catch (err) {
                    if (err === CANCELLED) return; // pipeline aborted by user, exit silently

                    console.error("❌ Processing Pipeline Error:", err);

                    if (isCancelled) return; // pipeline aborted by user, don't show error UI

                    failStep(currentStepIdx);

                    // Show error message below the step list
                    const errorDiv = document.createElement('div');
                    errorDiv.classList.add('pipeline-error-msg');
                    errorDiv.innerHTML = `
                        <div style="font-weight:600;margin-bottom:4px;">⚠️ Import Failed</div>
                        <div style="font-family:monospace;font-size:0.8em;white-space:pre-wrap;word-break:break-word;opacity:0.8;">${err.message || err.toString()}</div>
                    `;
                    body.appendChild(errorDiv);

                    // Restore X close button
                    if (modalCloseBtn) modalCloseBtn.style.display = '';

                    // Replace Cancel Import with Close button
                    if (footer) {
                        footer.style.display = '';
                        footer.innerHTML = '';
                        const closeBtn = document.createElement('button');
                        closeBtn.classList.add('btn');
                        closeBtn.textContent = "Close";
                        closeBtn.onclick = () => {
                            overlay.classList.remove('active');
                            setTimeout(() => overlay.remove(), 300);
                        };
                        footer.appendChild(closeBtn);
                    }
                }
            };

            footer.appendChild(cancelBtn);
            footer.appendChild(uploadBtn);
        }
    );
}

/**
 * Shows a modal for configuring Random Forest training parameters.
 */
// ─────────────────────────────────────────────────────────────────────────────
//  showModelReportModal
//  Opens a standalone modal showing the RF training report for a saved model.
//  @param {string} modelName   - Display name (folder name)
//  @param {string} reportPath  - Relative path to report.txt
//  @param {string} [metaPath]  - Optional relative path to metadata JSON (for class names)
// ─────────────────────────────────────────────────────────────────────────────
export async function showModelReportModal(modelName, reportPath, metaPath) {

    // ── Helpers ──────────────────────────────────────────────────────────────
    const parseSection = (text, heading) => {
        let idx = text.indexOf('\n' + heading);
        const atStart = text.startsWith(heading);
        if (idx === -1 && !atStart) return [];
        idx = atStart ? 0 : idx + 1;
        const after = text.slice(idx + heading.length).replace(/^\s*\n/, '');
        const lines = [];
        for (const line of after.split('\n')) {
            if (line.trim() === '' && lines.length) break;
            if (line.trim()) lines.push(line.trim());
        }
        return lines;
    };
    const parseArray = (lines) => {
        const raw = lines.join(' ');
        const inner = raw.replace(/[\[\]]/g, '').trim();
        return inner.split(/[\s,]+/).map(v => parseFloat(v)).filter(v => !isNaN(v));
    };
    const parseScalar = (lines) => { const v = parseFloat(lines[0]); return isNaN(v) ? null : v; };
    const metricColor = (v) => v >= 0.9 ? '#4ade80' : v >= 0.7 ? '#facc15' : '#f87171';
    const makeBadge = (val, tooltip) => {
        const b = document.createElement('span');
        const color = metricColor(val);
        b.style.cssText = `
            display:inline-flex; align-items:center; justify-content:center;
            width:46px; padding:2px 0; border-radius:4px;
            font-size:0.6rem; font-weight:700; font-variant-numeric:tabular-nums;
            background:${color}22; color:${color}; border:1px solid ${color}44;
            cursor:${tooltip ? 'help' : 'default'}; white-space:nowrap;
        `;
        b.textContent = (val * 100).toFixed(1) + '%';
        if (tooltip) b.title = tooltip;
        return b;
    };
    const makeSectionTitle = (text) => {
        const el = document.createElement('div');
        el.style.cssText = 'font-size:0.65rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;';
        el.textContent = text;
        return el;
    };

    // ── Fetch data ────────────────────────────────────────────────────────────
    let reportContent = null;
    let classNames = [];

    try {
        const r = await fetch(`/api/read-file/?path=${encodeURIComponent(reportPath)}`);
        if (r.ok) { const d = await r.json(); reportContent = d.content || null; }
    } catch (_) { }

    if (metaPath) {
        try {
            const r = await fetch(`/api/read-file/?path=${encodeURIComponent(metaPath)}`);
            if (r.ok) {
                const d = await r.json();
                const meta = JSON.parse(d.content || '{}');
                const entries = Object.entries(meta.classes || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
                classNames = [ ...entries.map(([, name]) => name)];
            }
        } catch (_) { }
    }

    // ── Build modal ───────────────────────────────────────────────────────────
    const overlay = createModal(
        `Report — ${modelName}`,
        (body) => {
            body.style.cssText = 'display:flex; flex-direction:column; gap:14px; flex:1; min-height:0; overflow:hidden;';

            if (!reportContent) {
                const msg = document.createElement('div');
                msg.style.cssText = 'font-size:0.8rem; color:var(--text-muted); padding:20px; text-align:center;';
                msg.textContent = 'Report file not found for this model.';
                body.appendChild(msg);
                return;
            }

            // Parse
            const accuracy = parseScalar(parseSection(reportContent, 'Overall Accuracy'));
            const precPerClass = parseArray(parseSection(reportContent, 'Precision per class'));
            const precAvg = parseScalar(parseSection(reportContent, 'Average Precision'));
            const precW = parseScalar(parseSection(reportContent, 'Weighted Precision'));
            const recPerClass = parseArray(parseSection(reportContent, 'Recall per class'));
            const recAvg = parseScalar(parseSection(reportContent, 'Average Recall'));
            const recW = parseScalar(parseSection(reportContent, 'Weighted Recall'));
            const f1PerClass = parseArray(parseSection(reportContent, 'F1 Scores per class'));
            const f1Avg = parseScalar(parseSection(reportContent, 'Average F1 Score'));
            const f1W = parseScalar(parseSection(reportContent, 'Weighted F1 Score'));
            const iouPerClass = parseArray(parseSection(reportContent, 'IoU per class'));
            const iouAvg = parseScalar(parseSection(reportContent, 'Average IoU'));
            const iouW = parseScalar(parseSection(reportContent, 'Weighted IoU'));
            const featLines = parseSection(reportContent, 'Feature importance');
            const timingLines = parseSection(reportContent, 'Timing');

            // Three-column layout
            const twoCol = document.createElement('div');
            twoCol.style.cssText = 'display:grid; grid-template-columns:200px fit-content(360px) 1fr; gap:14px; align-items:stretch; flex:1; min-height:0; overflow:hidden;';

            // ── LEFT: Metrics ──
            const leftCol = document.createElement('div');
            leftCol.style.cssText = 'display:flex; flex-direction:column; gap:0; min-width:0; min-height:0; overflow:hidden;';
            leftCol.appendChild(makeSectionTitle('Metrics'));
            const leftScroll = document.createElement('div');
            leftScroll.style.cssText = 'flex:1; min-height:0; overflow-y:auto; display:flex; flex-direction:column; gap:10px;';

            if (accuracy !== null) {
                const accColor = metricColor(accuracy);
                const accCard = document.createElement('div');
                accCard.style.cssText = `display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-radius:8px; background:${accColor}14; border:1px solid ${accColor}40;`;
                const accLeft = document.createElement('div');
                const accLabel = document.createElement('div');
                accLabel.style.cssText = 'font-size:0.62rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted); margin-bottom:3px;';
                accLabel.textContent = 'Overall Accuracy';
                const accNum = document.createElement('div');
                accNum.style.cssText = `font-size:1.6rem; font-weight:700; line-height:1; color:${accColor};`;
                accNum.textContent = (accuracy * 100).toFixed(1) + '%';
                accLeft.appendChild(accLabel); accLeft.appendChild(accNum);
                accCard.appendChild(accLeft);
                const accR = 18, accCirc = 2 * Math.PI * accR, accDash = accuracy * accCirc;
                const arcSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                arcSvg.setAttribute('width', '48'); arcSvg.setAttribute('height', '48'); arcSvg.setAttribute('viewBox', '0 0 48 48');
                arcSvg.style.cssText = 'flex-shrink:0; transform:rotate(-90deg);';
                arcSvg.innerHTML = `<circle cx="24" cy="24" r="${accR}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="4"/>
                    <circle cx="24" cy="24" r="${accR}" fill="none" stroke="${accColor}" stroke-width="4"
                        stroke-dasharray="${accDash.toFixed(2)} ${accCirc.toFixed(2)}" stroke-linecap="round"/>`;
                accCard.appendChild(arcSvg);
                leftScroll.appendChild(accCard);
            }

            const metricsData = [
                { label: 'Precision', avg: precAvg, w: precW },
                { label: 'Recall', avg: recAvg, w: recW },
                { label: 'F1 Score', avg: f1Avg, w: f1W },
                { label: 'IoU', avg: iouAvg, w: iouW },
            ];
            const metricsTable = document.createElement('div');
            metricsTable.style.cssText = 'display:flex; flex-direction:column; gap:0; border:1px solid rgba(255,255,255,0.07); border-radius:8px; overflow:hidden;';
            const hdr = document.createElement('div');
            hdr.style.cssText = 'display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; padding:5px 12px; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.03);';
            hdr.innerHTML = `<span style="font-size:0.62rem;color:var(--text-muted);"></span>
                <span style="font-size:0.62rem;color:var(--text-muted);text-align:center;">Average</span>
                <span style="font-size:0.62rem;color:var(--text-muted);text-align:center;">Weighted</span>`;
            metricsTable.appendChild(hdr);
            metricsData.forEach(({ label, avg, w }, i) => {
                const row = document.createElement('div');
                row.style.cssText = `display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; align-items:center; padding:7px 12px; border-top:${i > 0 ? '1px solid rgba(255,255,255,0.05)' : 'none'};`;
                const lbl = document.createElement('span');
                lbl.style.cssText = 'font-size:0.73rem; color:var(--text-muted);';
                lbl.textContent = label;
                row.appendChild(lbl);
                const avgCell = document.createElement('div'); avgCell.style.cssText = 'display:flex;justify-content:center;';
                if (avg !== null) avgCell.appendChild(makeBadge(avg));
                row.appendChild(avgCell);
                const wCell = document.createElement('div'); wCell.style.cssText = 'display:flex;justify-content:center;';
                if (w !== null) wCell.appendChild(makeBadge(w));
                row.appendChild(wCell);
                metricsTable.appendChild(row);
            });
            leftScroll.appendChild(metricsTable);
            leftCol.appendChild(leftScroll);
            twoCol.appendChild(leftCol);

            // ── CENTER: Per-class ──
            const centerCol = document.createElement('div');
            centerCol.style.cssText = 'display:flex; flex-direction:column; gap:0; min-width:0; min-height:0; overflow:hidden;';
            centerCol.appendChild(makeSectionTitle('Per Class'));
            const numClasses = precPerClass.length;
            if (numClasses > 0) {
                const PC_COLS = '80px repeat(4, 46px)';
                const perClassTable = document.createElement('div');
                perClassTable.style.cssText = 'display:flex; flex-direction:column; border:1px solid rgba(255,255,255,0.07); border-radius:8px; overflow:hidden; flex:1; min-height:0;';
                const pcHdr = document.createElement('div');
                pcHdr.style.cssText = `display:grid; grid-template-columns:${PC_COLS}; gap:4px; padding:4px 8px; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.03);`;
                pcHdr.innerHTML = `<span style="font-size:0.6rem;color:var(--text-muted);">Class</span>
                    <span style="font-size:0.6rem;color:var(--text-muted);text-align:center;">Prec</span>
                    <span style="font-size:0.6rem;color:var(--text-muted);text-align:center;">Rec</span>
                    <span style="font-size:0.6rem;color:var(--text-muted);text-align:center;">F1</span>
                    <span style="font-size:0.6rem;color:var(--text-muted);text-align:center;">IoU</span>`;
                perClassTable.appendChild(pcHdr);
                const pcScroll = document.createElement('div');
                pcScroll.style.cssText = 'flex:1; min-height:0; overflow-y:auto;';
                const pcMetrics = [precPerClass, recPerClass, f1PerClass, iouPerClass];
                for (let ci = 0; ci < numClasses; ci++) {
                    const row = document.createElement('div');
                    row.style.cssText = `display:grid; grid-template-columns:${PC_COLS}; gap:4px; align-items:center; padding:4px 8px; border-top:${ci > 0 ? '1px solid rgba(255,255,255,0.05)' : 'none'};`;
                    const lblWrap = document.createElement('div'); lblWrap.style.cssText = 'min-width:0; overflow:hidden;';
                    const lblEl = document.createElement('span');
                    const cName = classNames[ci] || String(ci);
                    lblEl.style.cssText = 'display:block; font-size:0.62rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';
                    lblEl.textContent = cName; lblEl.title = cName;
                    lblWrap.appendChild(lblEl); row.appendChild(lblWrap);
                    pcMetrics.forEach(arr => {
                        const cell = document.createElement('div'); cell.style.cssText = 'display:flex;justify-content:center;';
                        if (arr[ci] !== undefined) cell.appendChild(makeBadge(arr[ci]));
                        row.appendChild(cell);
                    });
                    pcScroll.appendChild(row);
                }
                perClassTable.appendChild(pcScroll);
                centerCol.appendChild(perClassTable);
            } else {
                const empty = document.createElement('div');
                empty.style.cssText = 'font-size:0.72rem; color:var(--text-muted); opacity:0.5;';
                empty.textContent = 'No per-class data.';
                centerCol.appendChild(empty);
            }
            twoCol.appendChild(centerCol);

            // ── RIGHT: Feature Importance ──
            const rightCol = document.createElement('div');
            rightCol.style.cssText = 'display:flex; flex-direction:column; gap:0; min-width:0; min-height:0; overflow:hidden;';
            rightCol.appendChild(makeSectionTitle('Feature Importance'));
            if (featLines.length > 0) {
                const allFeats = featLines
                    .map(l => { const ci = l.lastIndexOf(':'); return { name: l.slice(0, ci).trim(), val: parseFloat(l.slice(ci + 1)) || 0 }; })
                    .sort((a, b) => b.val - a.val);
                let cumSum = 0; const topFeats = [];
                for (const f of allFeats) { topFeats.push(f); cumSum += f.val; if (cumSum >= 0.9) break; }
                const maxVal = Math.max(...topFeats.map(f => f.val), 1e-9);
                const featScroll = document.createElement('div');
                featScroll.style.cssText = 'flex:1; min-height:0; overflow-y:auto; padding-right:4px;';
                const chart = document.createElement('div');
                chart.style.cssText = 'display:flex; flex-direction:column; gap:5px;';
                topFeats.forEach(({ name, val }, i) => {
                    const t = topFeats.length > 1 ? i / (topFeats.length - 1) : 0;
                    const barColor = `rgba(99,179,237,${1 - t * 0.5})`;
                    const pct = (val / maxVal) * 100;
                    const row = document.createElement('div');
                    row.style.cssText = 'display:grid; grid-template-columns:110px 1fr 44px; align-items:center; gap:6px;';
                    const lbl = document.createElement('span');
                    lbl.style.cssText = 'font-size:0.68rem; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';
                    lbl.title = name; lbl.textContent = name;
                    const track = document.createElement('div');
                    track.style.cssText = 'height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;';
                    const fill = document.createElement('div');
                    fill.style.cssText = `width:0%; height:100%; border-radius:3px; background:${barColor}; transition:width 0.5s ease ${i * 40}ms;`;
                    track.appendChild(fill);
                    const valEl = document.createElement('span');
                    valEl.style.cssText = 'font-size:0.65rem; text-align:right; color:var(--text-muted); font-variant-numeric:tabular-nums;';
                    valEl.textContent = val.toFixed(4);
                    row.appendChild(lbl); row.appendChild(track); row.appendChild(valEl);
                    chart.appendChild(row);
                    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${pct}%`; }));
                });
                featScroll.appendChild(chart);
                rightCol.appendChild(featScroll);
                const hidden = allFeats.length - topFeats.length;
                const summary = document.createElement('div');
                summary.style.cssText = 'font-size:0.65rem; color:var(--text-muted); margin-top:8px; opacity:0.7; flex-shrink:0;';
                summary.textContent = `${topFeats.length} feature${topFeats.length !== 1 ? 's' : ''} · ${(cumSum * 100).toFixed(0)}% importance` + (hidden > 0 ? ` · ${hidden} hidden` : '');
                rightCol.appendChild(summary);
            } else {
                const empty = document.createElement('div');
                empty.style.cssText = 'font-size:0.72rem; color:var(--text-muted); opacity:0.5;';
                empty.textContent = 'No feature data available.';
                rightCol.appendChild(empty);
            }
            twoCol.appendChild(rightCol);
            body.appendChild(twoCol);

            // ── Timing ──
            if (timingLines.length > 0) {
                const timingRow = document.createElement('div');
                timingRow.style.cssText = 'display:flex; gap:0; flex-shrink:0; border:1px solid rgba(255,255,255,0.07); border-radius:8px; overflow:hidden; margin-top:auto;';
                timingLines.forEach((line, i) => {
                    const ci = line.lastIndexOf(':');
                    if (ci === -1) return;
                    const lbl = line.slice(0, ci).trim()
                        .replace('Loading time', 'Loading').replace('Training time', 'Training')
                        .replace('Prediction + Metrics time', 'Prediction').replace('Total time', 'Total');
                    const val = line.slice(ci + 1).trim();
                    const isTotal = lbl === 'Total';
                    const cell = document.createElement('div');
                    cell.style.cssText = `flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:8px 6px; border-left:${i > 0 ? '1px solid rgba(255,255,255,0.06)' : 'none'}; background:${isTotal ? 'rgba(255,255,255,0.04)' : 'transparent'};`;
                    cell.innerHTML = `<span style="font-size:0.75rem;font-weight:${isTotal ? '700' : '500'};">${val}</span>
                        <span style="font-size:0.6rem;color:var(--text-muted);margin-top:2px;">${lbl}</span>`;
                    timingRow.appendChild(cell);
                });
                body.appendChild(timingRow);
            }
        },
        (footer, overlay) => {
            const closeBtn = document.createElement('button');
            closeBtn.classList.add('btn');
            closeBtn.textContent = 'Close';
            closeBtn.onclick = () => { overlay.classList.remove('active'); setTimeout(() => overlay.remove(), 300); };
            footer.appendChild(closeBtn);
        }
    );

    // Resize container now that it's in the DOM
    const container = overlay.querySelector('.modal-container');
    if (container) {
        container.style.width = '1020px';
        container.style.maxWidth = 'none';
        container.style.maxHeight = '75vh';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
    }
    // Also fix body flex so it fills the resized container
    const body = overlay.querySelector('.modal-body');
    if (body) {
        body.style.flex = '1';
        body.style.minHeight = '0';
        body.style.overflow = 'hidden';
    }

    return overlay;
}

export function showTrainingModal(scene, onStart) {
    const overlay = createModal(
        "Training Configuration",
        (body) => {
            // --- SECTION 0: MODEL NAME ---
            const nameTitle = document.createElement('div');
            nameTitle.classList.add('modal-section-title');
            nameTitle.textContent = "Model Name";
            body.appendChild(nameTitle);

            const nameSection = document.createElement('div');
            nameSection.classList.add('modal-section');

            const nameRow = document.createElement('div');
            nameRow.classList.add('property-row');
            nameRow.style.alignItems = 'center';

            const nameLabel = document.createElement('span');
            nameLabel.classList.add('property-label');
            nameLabel.style.width = '100px';
            nameLabel.textContent = 'Name:';
            nameRow.appendChild(nameLabel);

            const nameInputWrap = document.createElement('div');
            nameInputWrap.style.cssText = 'flex:1; display:flex; flex-direction:column; gap:4px;';

            const nameInput = document.createElement('input');
            nameInput.type = 'text';
            nameInput.id = 'model-name-input';
            nameInput.classList.add('property-input');
            nameInput.placeholder = 'e.g. facade_segmentation';
            nameInput.value = 'my_model';
            nameInput.style.fontFamily = 'monospace';

            const nameWarning = document.createElement('div');
            nameWarning.id = 'model-name-warning';
            nameWarning.style.cssText = 'font-size:0.7rem; color:#facc15; display:none; padding-left:2px;';
            nameWarning.textContent = '⚠ A model with this name already exists and will be overwritten.';

            nameInput.addEventListener('input', async () => {
                const val = nameInput.value.trim();
                if (!val) { nameWarning.style.display = 'none'; return; }
                try {
                    const res = await fetch(`/api/model-exists/?name=${encodeURIComponent(val)}`);
                    if (res.ok) {
                        const data = await res.json();
                        nameWarning.style.display = data.exists ? 'block' : 'none';
                    }
                } catch (_) { nameWarning.style.display = 'none'; }
            });

            nameInputWrap.appendChild(nameInput);
            nameInputWrap.appendChild(nameWarning);
            nameRow.appendChild(nameInputWrap);
            nameSection.appendChild(nameRow);
            body.appendChild(nameSection);

            // --- SECTION 1: DATA SPLIT ---
            const splitTitle = document.createElement('div');
            splitTitle.classList.add('modal-section-title');
            splitTitle.textContent = "Data Split";
            body.appendChild(splitTitle);

            const splitContainer = document.createElement('div');
            splitContainer.classList.add('modal-section');
            // splitContainer.style.marginBottom = "20px";

            const createSelectRow = (labelStr, id) => {
                const row = document.createElement('div');
                row.classList.add('property-row');
                row.style.alignItems = "center";
                // row.style.marginBottom = "12px";

                const label = document.createElement('span');
                label.classList.add('property-label');
                label.style.width = "100px";
                label.textContent = labelStr + ":";
                row.appendChild(label);

                const select = document.createElement('select');
                select.id = id;
                select.style.flex = "1";
                select.classList.add('property-input');
                row.appendChild(select);

                return { row, select };
            };

            const trainObj = createSelectRow("Training Data", "train-select");
            const valObj = createSelectRow("Validation Data", "val-select");

            splitContainer.appendChild(trainObj.row);
            splitContainer.appendChild(valObj.row);
            body.appendChild(splitContainer);

            // Get segments from the DOM
            const segments = document.querySelectorAll('.outline-item[data-segment-id]');

            const addOptions = (select) => {
                const noneOpt = document.createElement('option');
                noneOpt.value = "none";
                noneOpt.textContent = "-- Select Segment --";
                select.appendChild(noneOpt);

                segments.forEach((item) => {
                    const segmentId = item.dataset.segmentId;
                    const nameInput = item.querySelector('.outline-name-input');
                    const name = nameInput ? nameInput.value : `Segment ${segmentId}`;

                    const opt = document.createElement('option');
                    opt.value = segmentId;
                    opt.textContent = name;
                    select.appendChild(opt);
                });
            };

            if (segments.length === 0) {
                const opt = document.createElement('option');
                opt.disabled = true;
                opt.textContent = "No segments found";
                trainObj.select.appendChild(opt);
                valObj.select.appendChild(opt.cloneNode(true));
            } else {
                addOptions(trainObj.select);
                addOptions(valObj.select);

                // Pre-select some defaults if possible
                if (segments.length >= 1) trainObj.select.selectedIndex = 1;
                if (segments.length >= 2) valObj.select.selectedIndex = 2;
                else if (segments.length >= 1) valObj.select.selectedIndex = 1;
            }

            // --- SECTION 2: FEATURE SELECTION (Collapsible) ---
            const featureTitle = document.createElement('div');
            featureTitle.classList.add('modal-section-title');
            featureTitle.style.cursor = "pointer";
            featureTitle.style.display = "flex";
            featureTitle.style.justifyContent = "space-between";
            featureTitle.style.alignItems = "center";
            featureTitle.style.padding = "8px 0";
            featureTitle.style.transition = "color 0.2s";
            featureTitle.innerHTML = `<span>Feature Selection</span><span style="font-size: 0.6rem; opacity: 0.5;">▼</span>`;

            featureTitle.onmouseover = () => featureTitle.style.color = "white";
            featureTitle.onmouseout = () => featureTitle.style.color = "var(--accent-blue)";

            body.appendChild(featureTitle);

            const featureContainer = document.createElement('div');
            featureContainer.classList.add('modal-section');
            featureContainer.style.display = "none"; // Initial state: collapsed
            featureContainer.style.paddingTop = "10px";
            featureContainer.style.borderTop = "1px solid rgba(255,255,255,0.05)";

            // Two columns using flex
            const colsContainer = document.createElement('div');
            colsContainer.style.display = "flex";
            colsContainer.style.width = "100%";
            colsContainer.style.gap = "20px";

            const col1 = document.createElement('div');
            col1.style.flex = "1";
            const col2 = document.createElement('div');
            col2.style.flex = "1";
            colsContainer.appendChild(col1);
            colsContainer.appendChild(col2);
            featureContainer.appendChild(colsContainer);

            featureTitle.onclick = () => {
                const isHidden = featureContainer.style.display === "none";
                featureContainer.style.display = isHidden ? "flex" : "none";
                featureTitle.querySelector('span:last-child').textContent = isHidden ? "▲" : "▼";
                featureTitle.querySelector('span:last-child').style.opacity = isHidden ? "1" : "0.5";
            };

            // Remove unused two-column containers, we'll build the matrix directly
            colsContainer.remove();

            const loader = scene.potree2Loader;
            const validFeatures = loader ? loader.getFeatureList() : [];
            const featureSwitches = {};

            if (validFeatures.length === 0) {
                const emptyMsg = document.createElement('span');
                emptyMsg.style.cssText = "font-size:0.75rem; color:var(--text-muted)";
                emptyMsg.textContent = "No custom features found";
                featureContainer.appendChild(emptyMsg);
            } else {
                // Parse features: name_radius (e.g. "linearity_0_1" -> name="linearity", radius="0.1")
                // Radius part: everything after first underscore-separated non-numeric group
                const featureMap = new Map(); // featureName -> Set of radiusKeys
                const radiusOrder = [];

                validFeatures.forEach(featName => {
                    // Filter out normal and prediction features
                    if (featName.toLowerCase().includes('normal') || featName.toLowerCase().includes('prediction')) {
                        return;
                    }
                    
                    // Split on underscore, find where numeric part starts
                    const parts = featName.split('_');
                    // Find split point: last index where prefix is all non-numeric
                    let splitIdx = parts.length - 1;
                    for (let i = parts.length - 1; i >= 1; i--) {
                        if (isNaN(parts[i])) { splitIdx = i; break; }
                        splitIdx = i;
                    }
                    // Heuristic: look for first part that is numeric from the end
                    // Better: the base name is everything before the last two numeric segments
                    // Pattern: name_N_M where N and M are numbers
                    let baseName, radiusKey;
                    // Try to match trailing _<num>_<num> or _<num>
                    const match = featName.match(/^(.+?)_([\d]+(?:_[\d]+)*)$/);
                    if (match) {
                        baseName = match[1];
                        radiusKey = match[2].replace(/_/g, '.');
                    } else {
                        baseName = featName;
                        radiusKey = '-';
                    }

                    if (!featureMap.has(baseName)) featureMap.set(baseName, new Map());
                    featureMap.get(baseName).set(radiusKey, featName);

                    if (!radiusOrder.includes(radiusKey)) radiusOrder.push(radiusKey);
                });

                // Sort radius numerically
                radiusOrder.sort((a, b) => parseFloat(a) - parseFloat(b));

                // --- Build matrix ---
                const matrixWrapper = document.createElement('div');
                matrixWrapper.classList.add('feature-matrix-wrapper');

                // Select All / None controls
                const matrixControls = document.createElement('div');
                matrixControls.classList.add('feature-matrix-controls');

                const selectAllBtn = document.createElement('button');
                selectAllBtn.classList.add('feature-matrix-ctrl-btn');
                selectAllBtn.textContent = "Select All";
                selectAllBtn.onclick = () => {
                    matrixWrapper.querySelectorAll('.feat-cell-btn').forEach(btn => {
                        btn.classList.add('active');
                        btn.setAttribute('data-active', 'true');
                    });
                };

                const selectNoneBtn = document.createElement('button');
                selectNoneBtn.classList.add('feature-matrix-ctrl-btn');
                selectNoneBtn.textContent = "Select None";
                selectNoneBtn.onclick = () => {
                    matrixWrapper.querySelectorAll('.feat-cell-btn').forEach(btn => {
                        btn.classList.remove('active');
                        btn.setAttribute('data-active', 'false');
                    });
                };

                matrixControls.appendChild(selectAllBtn);
                matrixControls.appendChild(selectNoneBtn);
                featureContainer.appendChild(matrixControls);

                // Matrix table
                const table = document.createElement('div');
                table.classList.add('feature-matrix');

                // Header row
                const headerRow = document.createElement('div');
                headerRow.classList.add('feature-matrix-row', 'feature-matrix-header-row');

                // Top-left empty corner
                const cornerCell = document.createElement('div');
                cornerCell.classList.add('feature-matrix-cell', 'feature-matrix-corner');
                cornerCell.textContent = 'Feature \\ Radius';
                headerRow.appendChild(cornerCell);

                // Radius column headers
                radiusOrder.forEach(r => {
                    const th = document.createElement('div');
                    th.classList.add('feature-matrix-cell', 'feature-matrix-col-header');
                    th.textContent = r;
                    headerRow.appendChild(th);
                });
                table.appendChild(headerRow);

                // Feature rows
                featureMap.forEach((radiiMap, baseName) => {
                    const row = document.createElement('div');
                    row.classList.add('feature-matrix-row');

                    // Row label
                    const rowLabel = document.createElement('div');
                    rowLabel.classList.add('feature-matrix-cell', 'feature-matrix-row-label');
                    rowLabel.textContent = baseName;
                    rowLabel.title = baseName;
                    row.appendChild(rowLabel);

                    // Cells for each radius
                    radiusOrder.forEach(r => {
                        const cell = document.createElement('div');
                        cell.classList.add('feature-matrix-cell', 'feature-matrix-data-cell');

                        const fullName = radiiMap.get(r);
                        if (fullName) {
                            const btn = document.createElement('button');
                            btn.classList.add('feat-cell-btn', 'active');
                            btn.setAttribute('data-feature', fullName);
                            btn.setAttribute('data-active', 'true');
                            btn.title = fullName;
                            // Checkmark icon
                            btn.innerHTML = `<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><polyline points="1.5,5 4,7.5 8.5,2.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
                            btn.onclick = () => {
                                const isActive = btn.getAttribute('data-active') === 'true';
                                btn.setAttribute('data-active', String(!isActive));
                                btn.classList.toggle('active', !isActive);
                            };
                            featureSwitches[fullName] = btn;
                            cell.appendChild(btn);
                        } else {
                            cell.classList.add('feature-matrix-empty-cell');
                        }

                        row.appendChild(cell);
                    });

                    table.appendChild(row);
                });

                matrixWrapper.appendChild(table);
                featureContainer.appendChild(matrixWrapper);
            }

            body.appendChild(featureContainer);

            // --- SECTION 3: ALGORITHM SETTINGS (Collapsible) ---
            const algoTitle = document.createElement('div');
            algoTitle.classList.add('modal-section-title');
            algoTitle.style.cursor = "pointer";
            algoTitle.style.display = "flex";
            algoTitle.style.justifyContent = "space-between";
            algoTitle.style.alignItems = "center";
            algoTitle.style.padding = "8px 0";
            algoTitle.style.transition = "color 0.2s";
            algoTitle.innerHTML = `<span>Random Forest Settings</span><span style="font-size: 0.6rem; opacity: 0.5;">▼</span>`;

            algoTitle.onmouseover = () => algoTitle.style.color = "white";
            algoTitle.onmouseout = () => algoTitle.style.color = "var(--accent-blue)";

            body.appendChild(algoTitle);

            const algoContainer = document.createElement('div');
            algoContainer.classList.add('modal-section');
            algoContainer.id = 'rf-settings-section';
            algoContainer.style.display = "none"; // Initial state: collapsed
            algoContainer.style.paddingTop = "10px";
            algoContainer.style.borderTop = "1px solid rgba(255,255,255,0.05)";

            algoTitle.onclick = () => {
                const isHidden = algoContainer.style.display === "none";
                algoContainer.style.display = isHidden ? "flex" : "none";
                algoTitle.querySelector('span:last-child').textContent = isHidden ? "▲" : "▼";
                algoTitle.querySelector('span:last-child').style.opacity = isHidden ? "1" : "0.5";
            };

            // Helper to build a standard number row
            const makeNumberRow = (labelText, defaultVal, minVal, maxVal, placeholderText, paramKey) => {
                const row = document.createElement('div');
                row.classList.add('property-row');
                row.style.padding = "0 4px";
                row.innerHTML = `<span class="property-label">${labelText}</span>`;
                const inp = document.createElement('input');
                inp.type = "number";
                inp.classList.add('property-input');
                if (paramKey) inp.setAttribute('data-rf-param', paramKey);
                if (defaultVal !== null) inp.value = defaultVal;
                if (minVal !== null) inp.min = minVal;
                if (maxVal !== null) inp.max = maxVal;
                if (placeholderText) inp.placeholder = placeholderText;
                row.appendChild(inp);
                algoContainer.appendChild(row);
                return inp;
            };

            // n_estimators
            const estInput = makeNumberRow("N. Estimators:", 200, 10, 2000, null, 'n_estimators');
            // max_depth
            const depthInput = makeNumberRow("Max Depth:", 15, 1, 100, null, 'max_depth');
            // min_samples_split
            const minSplitInput = makeNumberRow("Min Samples Split:", 20, 2, 1000, null, 'min_samples_split');
            // n_jobs
            const jobsInput = makeNumberRow("N. Jobs:", 12, 1, 128, null, 'n_jobs');

            // use_gpu — toggle switch row
            const gpuRow = document.createElement('div');
            gpuRow.classList.add('property-row');
            gpuRow.style.padding = "0 4px";
            gpuRow.innerHTML = `<span class="property-label">Use GPU:</span>`;

            const gpuToggleWrap = document.createElement('div');
            gpuToggleWrap.classList.add('rf-toggle-wrap');

            const gpuToggle = document.createElement('button');
            gpuToggle.classList.add('rf-toggle-btn');
            gpuToggle.setAttribute('data-on', 'false');
            gpuToggle.setAttribute('data-rf-toggle', 'gpu');
            gpuToggle.textContent = 'OFF';
            gpuToggle.onclick = () => {
                const isOn = gpuToggle.getAttribute('data-on') === 'true';
                gpuToggle.setAttribute('data-on', String(!isOn));
                gpuToggle.textContent = isOn ? 'OFF' : 'ON';
                gpuToggle.classList.toggle('on', !isOn);
            };

            gpuToggleWrap.appendChild(gpuToggle);
            gpuRow.appendChild(gpuToggleWrap);
            algoContainer.appendChild(gpuRow);

            // user_rgb — toggle switch row
            const rgbRow = document.createElement('div');
            rgbRow.classList.add('property-row');
            rgbRow.style.padding = "0 4px";
            rgbRow.innerHTML = `<span class="property-label">Use RGB:</span>`;

            const rgbToggleWrap = document.createElement('div');
            rgbToggleWrap.classList.add('rf-toggle-wrap');

            const rgbToggle = document.createElement('button');
            rgbToggle.classList.add('rf-toggle-btn');
            rgbToggle.setAttribute('data-on', 'false');
            rgbToggle.setAttribute('data-rf-toggle', 'rgb');
            rgbToggle.textContent = 'OFF';
            rgbToggle.onclick = () => {
                const isOn = rgbToggle.getAttribute('data-on') === 'true';
                rgbToggle.setAttribute('data-on', String(!isOn));
                rgbToggle.textContent = isOn ? 'OFF' : 'ON';
                rgbToggle.classList.toggle('on', !isOn);
            };

            rgbToggleWrap.appendChild(rgbToggle);
            rgbRow.appendChild(rgbToggleWrap);
            algoContainer.appendChild(rgbRow);

            body.appendChild(algoContainer);
        },
        (footer, overlay) => {
            const cancelBtn = document.createElement('button');
            cancelBtn.classList.add('btn');
            cancelBtn.style.backgroundColor = 'transparent';
            cancelBtn.style.border = '1px solid var(--border-color)';
            cancelBtn.textContent = "Cancel";
            cancelBtn.onclick = () => {
                overlay.classList.remove('active');
                setTimeout(() => overlay.remove(), 300);
            };

            const runBtn = document.createElement('button');
            runBtn.classList.add('btn');
            runBtn.textContent = "Start Training";
            runBtn.onclick = async () => {
                const body = overlay.querySelector('.modal-body');

                const trainId = body.querySelector('#train-select').value;
                const valId = body.querySelector('#val-select').value;

                // --- Validate model name ---
                const modelNameRaw = (body.querySelector('#model-name-input')?.value || '').trim();
                // Sanitize: keep only alphanumeric, dash, underscore
                const modelName = modelNameRaw.replace(/[^a-zA-Z0-9_\-]/g, '_').replace(/^_+|_+$/g, '') || 'my_model';
                const modelDir = _runtimePath('models', modelName, '/');
                const workingDir = _runtimePath('working', modelName, '/');

                // Helper: show error above footer, auto-hide after 4s
                const showError = (msg) => {
                    const existing = overlay.querySelector('.training-modal-error');
                    if (existing) existing.remove();
                    const errorMsg = document.createElement('div');
                    errorMsg.classList.add('training-modal-error');
                    errorMsg.style.cssText = 'color:#ff5f5f;font-size:0.75rem;padding:0 4px 10px 4px;text-align:center;width:100%;';
                    errorMsg.textContent = msg;
                    footer.insertAdjacentElement('beforebegin', errorMsg);
                    setTimeout(() => {
                        errorMsg.style.transition = 'opacity 0.4s ease';
                        errorMsg.style.opacity = '0';
                        setTimeout(() => errorMsg.remove(), 400);
                    }, 4000);
                };

                // --- Validation: training and validation must differ ---
                if (trainId !== 'none' && valId !== 'none' && trainId === valId) {
                    showError('Training and validation segments must be different.');
                    return;
                }

                // Reconstruct data map for backend
                const split = {};
                if (trainId !== "none") {
                    split[trainId] = ['train'];
                }
                if (valId !== "none") {
                    if (!split[valId]) split[valId] = [];
                    if (!split[valId].includes('val')) split[valId].push('val');
                }

                // Extract Features from matrix buttons
                const selectedFeatures = [];
                body.querySelectorAll('.feat-cell-btn[data-active="true"]').forEach(btn => {
                    selectedFeatures.push(btn.getAttribute('data-feature'));
                });

                // Extract RF params directly from named inputs/toggle
                const algoSec = body.querySelector('#rf-settings-section');
                if (!algoSec) {
                    showError('Random Forest settings section not found. Please reopen the modal.');
                    return;
                }

                const getRfNumber = (key, fallback) => {
                    const raw = parseInt(algoSec.querySelector(`input[data-rf-param="${key}"]`)?.value, 10);
                    return Number.isFinite(raw) ? raw : fallback;
                };

                const gpuBtn = algoSec.querySelector('button[data-rf-toggle="gpu"]');
                const rgbBtn = algoSec.querySelector('button[data-rf-toggle="rgb"]');
                const useRgb = rgbBtn?.getAttribute('data-on') === 'true';
                const finalFeatures = useRgb
                    ? [...selectedFeatures, 'red', 'green', 'blue']
                    : selectedFeatures;
                // --- Validation: at least one feature must be selected ---
                const loader = getPotree2Loader(scene) || scene?.potree2Loader;
                const availableFeatures = loader ? loader.getFeatureList() : [];
                const noFeaturesAvailable = availableFeatures.length === 0;
                const noFeaturesSelected = selectedFeatures.length === 0 && !useRgb;

                if (noFeaturesAvailable) {
                    showError('No features available — run Calculate Features first.');
                    return;
                }
                if (noFeaturesSelected) {
                    showError('Select at least one feature to start training.');
                    return;
                }

                // Remove any lingering error before proceeding
                const existingErr = overlay.querySelector('.training-modal-error');
                if (existingErr) existingErr.remove();

                const params = {
                    split,
                    features: finalFeatures,
                    rf_params: {
                        n_estimators: getRfNumber('n_estimators', 200),
                        max_depth: getRfNumber('max_depth', 15),
                        min_samples_split: getRfNumber('min_samples_split', 20),
                        n_jobs: getRfNumber('n_jobs', 12),
                        use_gpu: gpuBtn?.getAttribute('data-on') === 'true'
                    }
                };

                // ── Replace modal body with animated pipeline ──
                body.innerHTML = '';
                const modalCloseBtn = overlay.querySelector('.modal-close');
                if (modalCloseBtn) modalCloseBtn.style.display = 'none';

                let isCancelled = false;
                footer.style.display = '';
                footer.innerHTML = '';
                const cancelTrainBtn = document.createElement('button');
                cancelTrainBtn.classList.add('btn');
                cancelTrainBtn.style.backgroundColor = 'transparent';
                cancelTrainBtn.style.border = '1px solid var(--border-color)';
                cancelTrainBtn.textContent = 'Stop Training';
                cancelTrainBtn.onclick = async () => {
                    if (isCancelled) return;
                    isCancelled = true;
                    cancelTrainBtn.disabled = true;
                    cancelTrainBtn.textContent = 'Stopping…';
                    try {
                        await fetch('/stop_process/', { method: 'POST', headers: { 'X-CSRFToken': getCSRFToken() } });
                    } catch (e) { console.warn('Could not stop process:', e); }
                    overlay.classList.remove('active');
                    setTimeout(() => overlay.remove(), 300);
                };
                footer.appendChild(cancelTrainBtn);

                const STEPS = [
                    'Extracting Points',
                    'Saving Training Data',
                    'Splitting LAS by Labels',
                    'Training Random Forest'
                ];

                const stepList = document.createElement('div');
                stepList.classList.add('pipeline-step-list');

                const stepEls = STEPS.map((label) => {
                    const item = document.createElement('div');
                    item.classList.add('pipeline-step-item', 'step-idle');
                    const iconWrap = document.createElement('div');
                    iconWrap.classList.add('step-icon-wrap');
                    iconWrap.innerHTML = `
                        <svg class="step-svg" viewBox="0 0 32 32">
                            <circle class="step-track" cx="16" cy="16" r="13" />
                            <circle class="step-arc"   cx="16" cy="16" r="13" />
                        </svg>
                        <div class="step-check">
                            <svg viewBox="0 0 12 12" fill="none">
                                <polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="step-cross">
                            <svg viewBox="0 0 12 12" fill="none">
                                <line x1="2.5" y1="2.5" x2="9.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                <line x1="9.5" y1="2.5" x2="2.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                            </svg>
                        </div>`;
                    const labelEl = document.createElement('span');
                    labelEl.classList.add('step-label');
                    labelEl.textContent = label;
                    item.appendChild(iconWrap);
                    item.appendChild(labelEl);
                    stepList.appendChild(item);
                    return item;
                });
                body.appendChild(stepList);

                const activateStep = (idx) => {
                    stepEls[idx].classList.remove('step-idle', 'step-done', 'step-error');
                    stepEls[idx].classList.add('step-running');
                };
                const completeStep = (idx) => {
                    stepEls[idx].classList.remove('step-running');
                    stepEls[idx].classList.add('step-done');
                };
                const failStep = (idx) => {
                    stepEls[idx].classList.remove('step-running');
                    stepEls[idx].classList.add('step-error');
                };

                let currentStepIdx = 0;
                const CANCELLED = Symbol('CANCELLED');
                const checkCancelled = () => { if (isCancelled) throw CANCELLED; };

                try {
                    // STEP 1: Extract points
                    checkCancelled();
                    currentStepIdx = 0;
                    activateStep(0);
                    const segmentMap = {};
                    document.querySelectorAll('.outline-item[data-segment-id]').forEach(item => {
                        const segId = parseInt(item.dataset.segmentId, 10);
                        const nameInput = item.querySelector('.outline-name-input');
                        if (!isNaN(segId) && nameInput) segmentMap[segId] = nameInput.value;
                    });
                    const classMap = {};
                    classRegistry.forEach((val, id) => { classMap[id] = val.name; });
                    const exportResult = await loader.exportAllTrainingDataAsMapping(segmentMap);
                    if (!exportResult || !exportResult.mapping_path) {
                        throw new Error('No points were identified in the selected regions.');
                    }
                    completeStep(0);

                    // STEP 2: Mapping already saved by exportAllTrainingDataAsMapping
                    checkCancelled();
                    currentStepIdx = 1;
                    activateStep(1);
                    const mappingPath = exportResult.mapping_path;
                    completeStep(1);

                    // STEP 3: Split LAS by .pcbin annotations
                    checkCancelled();
                    currentStepIdx = 2;
                    activateStep(2);
                    const lasSourcePath = _runtimePath('working', 'features.las');
                    const segmentNames = {};
                    segmentNames[trainId] = 'training';
                    segmentNames[valId]   = 'validation';
                    const splitResponse = await fetch('/split_las_by_binary/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            las_path: lasSourcePath,
                            pcbin_path: _runtimePath('working', 'features.pcbin'),
                            output_dir: workingDir,
                            segment_names: segmentNames,
                            exclude_unclassified: true
                        })
                    });
                    if (!splitResponse.ok) {
                        const errData = await splitResponse.json().catch(() => ({}));
                        throw new Error(errData.message || errData.error || 'Failed to split LAS file');
                    }
                    completeStep(2);

                    // STEP 4: Launch RF Training
                    checkCancelled();
                    currentStepIdx = 3;
                    activateStep(3);
                    const rfParams = params.rf_params;
                    const trainingPayload = {
                        selected_features: params.features,
                        training_filepath: `${workingDir}training.las`,
                        val_filepath: `${workingDir}validation.las`,
                        n_jobs: rfParams.n_jobs ?? 12,
                        nr_estimators: rfParams.n_estimators ?? 200,
                        max_depth: rfParams.max_depth ?? 15,
                        min_samples_split: rfParams.min_samples_split ?? 20,
                        max_features: rfParams.max_features ?? 'sqrt',
                        use_gpu: rfParams.use_gpu ?? false,
                        output_training_name: `${workingDir}test_predicted.las`,
                        model_savepath: `${modelDir}model.pkl`,
                        report_savepath: `${modelDir}report.txt`
                    };
                    const trainResponse = await fetch('/launch_RF_training/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify(trainingPayload)
                    });
                    if (!trainResponse.ok) {
                        const errData = await trainResponse.json().catch(() => ({}));
                        throw new Error(errData.message || errData.error || 'RF training failed');
                    }
                    completeStep(3);

                    // ── Save model metadata.json ──
                    const modelMeta = {
                        segments: segmentMap,
                        classes: classMap,
                        split: params.split,
                        rf_params: params.rf_params,
                        features: params.features,
                    };
                    try {
                        await saveFile(
                            `${modelDir}metadata.json`,
                            new Blob([JSON.stringify(modelMeta, null, 2)], { type: 'application/json' })
                        );
                        console.log('[Training] metadata.json saved:', modelDir);
                    } catch (e) {
                        console.warn('[Training] Could not save metadata.json:', e);
                    }

                    if (onStart) onStart(params);

                    // ── Build class names from classMap (in-memory) ──
                    // classNames: ordered array matching report class indices 0,1,2,...
                    let classNames = [];
                    try {
                        // Build sorted array of class names from classMap
                        const classIds = Object.keys(classMap).map(Number).sort((a, b) => a - b);
                        for (const id of classIds) {
                            classNames.push(classMap[id]);
                        }
                        console.log('[Report] ordered classNames:', classNames);
                    } catch (e) { console.warn('[Report] Could not build class names:', e); }

                    // ── Read report text from first .txt found in modelDir ──
                    const reportPath = modelDir;
                    let reportContent = null;
                    try {
                        const reportRes = await fetch(`/api/read-file/?path=${encodeURIComponent(reportPath)}`);
                        if (reportRes.ok) {
                            const rd = await reportRes.json();
                            reportContent = rd.content || null;
                        }
                    } catch (_) { /* file not available */ }

                    // ── Helper: parse a named section ──
                    const parseSection = (text, heading) => {
                        // Match at start of file OR after a newline
                        let idx = text.indexOf('\n' + heading);
                        const atStart = text.startsWith(heading);
                        if (idx === -1 && !atStart) return [];
                        idx = atStart ? 0 : idx + 1;
                        const after = text.slice(idx + heading.length).replace(/^\s*\n/, '');
                        const lines = [];
                        for (const line of after.split('\n')) {
                            if (line.trim() === '' && lines.length) break;
                            if (line.trim()) lines.push(line.trim());
                        }
                        return lines;
                    };

                    // ── Helper: parse array like [0.98, 0.95, 1.0] or [1.] ──
                    const parseArray = (lines) => {
                        const raw = lines.join(' ');
                        const inner = raw.replace(/[\[\]]/g, '').trim();
                        return inner.split(/[\s,]+/).map(v => parseFloat(v)).filter(v => !isNaN(v));
                    };

                    // ── Helper: single scalar value ──
                    const parseScalar = (lines) => {
                        const v = parseFloat(lines[0]);
                        return isNaN(v) ? null : v;
                    };

                    // ── Helper: metric color ──
                    const metricColor = (v) => v >= 0.9 ? '#4ade80' : v >= 0.7 ? '#facc15' : '#f87171';

                    // ── Helper: badge element ──
                    const makeBadge = (val, tooltip) => {
                        const b = document.createElement('span');
                        const color = metricColor(val);
                        b.style.cssText = `
                            display:inline-flex; align-items:center; justify-content:center;
                            width: 46px; padding: 2px 0; border-radius: 4px;
                            font-size: 0.6rem; font-weight: 700;
                            font-variant-numeric: tabular-nums;
                            background: ${color}22; color: ${color};
                            border: 1px solid ${color}44; cursor: default;
                            white-space: nowrap;
                        `;
                        b.textContent = (val * 100).toFixed(1) + '%';
                        if (tooltip) {
                            b.title = tooltip;
                            b.style.cursor = 'help';
                        }
                        return b;
                    };

                    // ── Helper: section title ──
                    const makeSectionTitle = (text) => {
                        const el = document.createElement('div');
                        el.style.cssText = 'font-size:0.65rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;';
                        el.textContent = text;
                        return el;
                    };

                    // ── Widen modal & fix height ──
                    const container = overlay.querySelector('.modal-container');
                    if (container) {
                        container.style.width = '1020px';
                        container.style.maxHeight = '75vh';
                        container.style.display = 'flex';
                        container.style.flexDirection = 'column';
                    }

                    // ── Clear body — flex column, fills remaining space ──
                    body.innerHTML = '';
                    body.style.display = 'flex';
                    body.style.flexDirection = 'column';
                    body.style.gap = '14px';
                    body.style.flex = '1';
                    body.style.minHeight = '0';
                    body.style.overflow = 'hidden';

                    // ── Success banner ──
                    const banner = document.createElement('div');
                    banner.style.cssText = `
                        display:flex; align-items:center; gap:10px; flex-shrink:0;
                        padding: 10px 14px;
                        background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); border-radius: 8px;
                    `;
                    banner.innerHTML = `
                        <svg width="18" height="18" viewBox="0 0 20 20" fill="none" style="flex-shrink:0">
                            <circle cx="10" cy="10" r="9" stroke="#4ade80" stroke-width="1.6"/>
                            <polyline points="6,10 9,13 14,7" stroke="#4ade80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <div>
                            <div style="font-weight:600; color:#4ade80; font-size:0.82rem;">Training Completed Successfully</div>
                            <div style="font-size:0.7rem; opacity:0.55; margin-top:1px;">Model saved · Ready for classification</div>
                        </div>
                    `;
                    body.appendChild(banner);

                    if (reportContent) {
                        // ── Parse all sections ──
                        const accuracy = parseScalar(parseSection(reportContent, 'Overall Accuracy'));
                        const precPerClass = parseArray(parseSection(reportContent, 'Precision per class'));
                        const precAvg = parseScalar(parseSection(reportContent, 'Average Precision'));
                        const precW = parseScalar(parseSection(reportContent, 'Weighted Precision'));
                        const recPerClass = parseArray(parseSection(reportContent, 'Recall per class'));
                        const recAvg = parseScalar(parseSection(reportContent, 'Average Recall'));
                        const recW = parseScalar(parseSection(reportContent, 'Weighted Recall'));
                        const f1PerClass = parseArray(parseSection(reportContent, 'F1 Scores per class'));
                        const f1Avg = parseScalar(parseSection(reportContent, 'Average F1 Score'));
                        const f1W = parseScalar(parseSection(reportContent, 'Weighted F1 Score'));
                        const iouPerClass = parseArray(parseSection(reportContent, 'IoU per class'));
                        const iouAvg = parseScalar(parseSection(reportContent, 'Average IoU'));
                        const iouW = parseScalar(parseSection(reportContent, 'Weighted IoU'));
                        const featLines = parseSection(reportContent, 'Feature importance');
                        const timingLines = parseSection(reportContent, 'Timing');

                        // ── Three-column layout — flex:1 so it fills the body; all columns scroll independently ──
                        const twoCol = document.createElement('div');
                        twoCol.style.cssText = 'display:grid; grid-template-columns:200px fit-content(360px) 1fr; gap:14px; align-items:stretch; flex:1; min-height:0; overflow:hidden;';

                        // ── LEFT: Metrics ──
                        const leftCol = document.createElement('div');
                        leftCol.style.cssText = 'display:flex; flex-direction:column; gap:0; min-width:0; min-height:0; overflow:hidden;';
                        leftCol.appendChild(makeSectionTitle('Metrics'));

                        // Scrollable wrapper for left column content
                        const leftScroll = document.createElement('div');
                        leftScroll.style.cssText = 'flex:1; min-height:0; overflow-y:auto; display:flex; flex-direction:column; gap:10px;';

                        // Accuracy — prominent card above the table
                        if (accuracy !== null) {
                            const accColor = metricColor(accuracy);
                            const accCard = document.createElement('div');
                            accCard.style.cssText = `
                                display:flex; align-items:center; justify-content:space-between;
                                padding: 10px 14px; border-radius: 8px;
                                background: ${accColor}14;
                                border: 1px solid ${accColor}40;
                            `;

                            // Left: label + big number
                            const accLeft = document.createElement('div');
                            const accLabel = document.createElement('div');
                            accLabel.style.cssText = 'font-size:0.62rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted); margin-bottom:3px;';
                            accLabel.textContent = 'Overall Accuracy';
                            const accNum = document.createElement('div');
                            accNum.style.cssText = `font-size:1.6rem; font-weight:700; line-height:1; color:${accColor};`;
                            accNum.textContent = (accuracy * 100).toFixed(1) + '%';
                            accLeft.appendChild(accLabel);
                            accLeft.appendChild(accNum);
                            accCard.appendChild(accLeft);

                            // Right: donut arc SVG
                            const accR = 18, accCirc = 2 * Math.PI * accR;
                            const accDash = accuracy * accCirc;
                            const arcSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                            arcSvg.setAttribute('width', '48'); arcSvg.setAttribute('height', '48');
                            arcSvg.setAttribute('viewBox', '0 0 48 48');
                            arcSvg.style.cssText = 'flex-shrink:0; transform:rotate(-90deg);';
                            arcSvg.innerHTML = `
                                <circle cx="24" cy="24" r="${accR}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="4"/>
                                <circle cx="24" cy="24" r="${accR}" fill="none" stroke="${accColor}" stroke-width="4"
                                    stroke-dasharray="${accDash.toFixed(2)} ${accCirc.toFixed(2)}"
                                    stroke-linecap="round"/>
                            `;
                            accCard.appendChild(arcSvg);
                            leftScroll.appendChild(accCard);
                        }

                        // Metrics table
                        const metricsData = [
                            { label: 'Precision', avg: precAvg, w: precW, perClass: precPerClass },
                            { label: 'Recall', avg: recAvg, w: recW, perClass: recPerClass },
                            { label: 'F1 Score', avg: f1Avg, w: f1W, perClass: f1PerClass },
                            { label: 'IoU', avg: iouAvg, w: iouW, perClass: iouPerClass },
                        ];

                        const metricsTable = document.createElement('div');
                        metricsTable.style.cssText = 'display:flex; flex-direction:column; gap:0; border:1px solid rgba(255,255,255,0.07); border-radius:8px; overflow:hidden;';

                        // Header: metric | Average | Weighted
                        const hdr = document.createElement('div');
                        hdr.style.cssText = 'display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; padding:5px 12px; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.03);';
                        hdr.innerHTML = `
                            <span style="font-size:0.62rem; color:var(--text-muted);"></span>
                            <span style="font-size:0.62rem; color:var(--text-muted); text-align:center;">Average</span>
                            <span style="font-size:0.62rem; color:var(--text-muted); text-align:center;">Weighted</span>
                        `;
                        metricsTable.appendChild(hdr);

                        metricsData.forEach(({ label, avg, w }, i) => {
                            const row = document.createElement('div');
                            row.style.cssText = `
                                display:grid; grid-template-columns:1fr 1fr 1fr;
                                gap:4px; align-items:center; padding:7px 12px;
                                border-top:${i > 0 ? '1px solid rgba(255,255,255,0.05)' : 'none'};
                            `;

                            const lbl = document.createElement('span');
                            lbl.style.cssText = 'font-size:0.73rem; color:var(--text-muted);';
                            lbl.textContent = label;
                            row.appendChild(lbl);

                            const avgCell = document.createElement('div');
                            avgCell.style.cssText = 'display:flex; justify-content:center;';
                            if (avg !== null) avgCell.appendChild(makeBadge(avg));
                            row.appendChild(avgCell);

                            const wCell = document.createElement('div');
                            wCell.style.cssText = 'display:flex; justify-content:center;';
                            if (w !== null) wCell.appendChild(makeBadge(w));
                            row.appendChild(wCell);

                            metricsTable.appendChild(row);
                        });
                        leftScroll.appendChild(metricsTable);
                        leftCol.appendChild(leftScroll);
                        twoCol.appendChild(leftCol);

                        // ── CENTER: Per-class metrics ──
                        const centerCol = document.createElement('div');
                        centerCol.style.cssText = 'display:flex; flex-direction:column; gap:0; min-width:0; min-height:0; overflow:hidden;';
                        centerCol.appendChild(makeSectionTitle('Per Class'));

                        const numClasses = precPerClass.length;
                        if (numClasses > 0) {
                            // col widths: class label + 4 fixed badge columns
                            const PC_COLS = '80px repeat(4, 46px)';
                            const PC_HDR_PAD = '4px 8px';
                            const PC_ROW_PAD = '4px 8px';

                            const perClassTable = document.createElement('div');
                            perClassTable.style.cssText = 'display:flex; flex-direction:column; border:1px solid rgba(255,255,255,0.07); border-radius:8px; overflow:hidden; flex:1; min-height:0;';

                            // Header (sticky)
                            const pcHdr = document.createElement('div');
                            pcHdr.style.cssText = `display:grid; grid-template-columns:${PC_COLS}; gap:4px; padding:${PC_HDR_PAD}; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.03);`;
                            pcHdr.innerHTML = `
                                <span style="font-size:0.6rem; color:var(--text-muted);">Class</span>
                                <span style="font-size:0.6rem; color:var(--text-muted); text-align:center;">Prec</span>
                                <span style="font-size:0.6rem; color:var(--text-muted); text-align:center;">Rec</span>
                                <span style="font-size:0.6rem; color:var(--text-muted); text-align:center;">F1</span>
                                <span style="font-size:0.6rem; color:var(--text-muted); text-align:center;">IoU</span>
                            `;
                            perClassTable.appendChild(pcHdr);

                            // Scrollable body — fills remaining space in the table
                            const pcScroll = document.createElement('div');
                            pcScroll.style.cssText = 'flex:1; min-height:0; overflow-y:auto;';

                            const pcMetrics = [precPerClass, recPerClass, f1PerClass, iouPerClass];
                            for (let ci = 0; ci < numClasses; ci++) {
                                const row = document.createElement('div');
                                row.style.cssText = `display:grid; grid-template-columns:${PC_COLS}; gap:4px; align-items:center; padding:${PC_ROW_PAD}; border-top:${ci > 0 ? '1px solid rgba(255,255,255,0.05)' : 'none'};`;

                                // Wrap in a div so min-width:0 constrains the grid cell and ellipsis works
                                const classLblWrap = document.createElement('div');
                                classLblWrap.style.cssText = 'min-width:0; overflow:hidden;';
                                const classLbl = document.createElement('span');
                                const cName = classNames[ci] || String(ci);
                                classLbl.style.cssText = 'display:block; font-size:0.62rem; color:var(--text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';
                                classLbl.textContent = cName;
                                // Native browser title shows full name on hover after ~0.7s — no custom JS needed
                                classLbl.title = cName;
                                classLblWrap.appendChild(classLbl);
                                row.appendChild(classLblWrap);

                                pcMetrics.forEach(arr => {
                                    const cell = document.createElement('div');
                                    cell.style.cssText = 'display:flex; justify-content:center;';
                                    const v = arr[ci];
                                    if (v !== undefined) cell.appendChild(makeBadge(v));
                                    row.appendChild(cell);
                                });

                                pcScroll.appendChild(row);
                            }

                            perClassTable.appendChild(pcScroll);
                            centerCol.appendChild(perClassTable);
                        } else {
                            const empty = document.createElement('div');
                            empty.style.cssText = 'font-size:0.72rem; color:var(--text-muted); opacity:0.5;';
                            empty.textContent = 'No per-class data.';
                            centerCol.appendChild(empty);
                        }

                        twoCol.appendChild(centerCol);

                        // ── RIGHT: Feature Importance — this column scrolls internally ──
                        const rightCol = document.createElement('div');
                        rightCol.style.cssText = 'display:flex; flex-direction:column; gap:0; min-width:0; min-height:0; overflow:hidden;';
                        rightCol.appendChild(makeSectionTitle('Feature Importance'));

                        if (featLines.length > 0) {
                            // Parse → sort desc
                            const allFeats = featLines
                                .map(l => {
                                    const ci = l.lastIndexOf(':');
                                    return { name: l.slice(0, ci).trim(), val: parseFloat(l.slice(ci + 1)) || 0 };
                                })
                                .sort((a, b) => b.val - a.val);

                            // Take features until cumulative sum ≥ 0.9
                            const THRESHOLD = 0.9;
                            let cumSum = 0;
                            const topFeats = [];
                            for (const f of allFeats) {
                                topFeats.push(f);
                                cumSum += f.val;
                                if (cumSum >= THRESHOLD) break;
                            }

                            const maxVal = Math.max(...topFeats.map(f => f.val), 1e-9);
                            // Scrollable wrapper — only this column scrolls
                            const featScroll = document.createElement('div');
                            featScroll.style.cssText = 'flex:1; min-height:0; overflow-y:auto; padding-right:4px;';
                            const chart = document.createElement('div');
                            chart.style.cssText = 'display:flex; flex-direction:column; gap:5px;';

                            topFeats.forEach(({ name, val }, i) => {
                                const t = topFeats.length > 1 ? i / (topFeats.length - 1) : 0;
                                const barColor = `rgba(99,179,237,${1 - t * 0.5})`;
                                const pct = (val / maxVal) * 100;

                                const row = document.createElement('div');
                                row.style.cssText = 'display:grid; grid-template-columns:110px 1fr 44px; align-items:center; gap:6px;';

                                const lbl = document.createElement('span');
                                lbl.style.cssText = 'font-size:0.68rem; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';
                                lbl.title = name;
                                lbl.textContent = name;

                                const track = document.createElement('div');
                                track.style.cssText = 'height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;';
                                const fill = document.createElement('div');
                                fill.style.cssText = `width:0%; height:100%; border-radius:3px; background:${barColor}; transition:width 0.5s ease ${i * 40}ms;`;
                                track.appendChild(fill);

                                const valEl = document.createElement('span');
                                valEl.style.cssText = 'font-size:0.65rem; text-align:right; color:var(--text-muted); font-variant-numeric:tabular-nums;';
                                valEl.textContent = val.toFixed(4);

                                row.appendChild(lbl);
                                row.appendChild(track);
                                row.appendChild(valEl);
                                chart.appendChild(row);

                                requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${pct}%`; }));
                            });

                            featScroll.appendChild(chart);
                            rightCol.appendChild(featScroll);

                            // Summary line (outside scroll, always visible)
                            const hidden = allFeats.length - topFeats.length;
                            const summary = document.createElement('div');
                            summary.style.cssText = 'font-size:0.65rem; color:var(--text-muted); margin-top:8px; opacity:0.7; flex-shrink:0;';
                            summary.textContent = `${topFeats.length} feature${topFeats.length !== 1 ? 's' : ''} · ${(cumSum * 100).toFixed(0)}% importance` +
                                (hidden > 0 ? ` · ${hidden} hidden` : '');
                            rightCol.appendChild(summary);
                        } else {
                            const empty = document.createElement('div');
                            empty.style.cssText = 'font-size:0.72rem; color:var(--text-muted); opacity:0.5;';
                            empty.textContent = 'No feature data available.';
                            rightCol.appendChild(empty);
                        }

                        twoCol.appendChild(rightCol);
                        body.appendChild(twoCol);

                        // ── Timing (full width, horizontal) — always visible at the bottom ──
                        if (timingLines.length > 0) {
                            const timingRow = document.createElement('div');
                            timingRow.style.cssText = `
                                display:flex; gap:0; flex-shrink:0;
                                border:1px solid rgba(255,255,255,0.07); border-radius:8px; overflow:hidden;
                                margin-top:auto;
                            `;
                            timingLines.forEach((line, i) => {
                                const ci = line.lastIndexOf(':');
                                if (ci === -1) return;
                                const lbl = line.slice(0, ci).trim()
                                    .replace('Loading time', 'Loading')
                                    .replace('Training time', 'Training')
                                    .replace('Prediction + Metrics time', 'Compute Metrics')
                                    .replace('Total time', 'Total');
                                const val = line.slice(ci + 1).trim();
                                const isTotal = lbl === 'Total';

                                const cell = document.createElement('div');
                                cell.style.cssText = `
                                    flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;
                                    padding: 8px 6px;
                                    border-left: ${i > 0 ? '1px solid rgba(255,255,255,0.06)' : 'none'};
                                    background: ${isTotal ? 'rgba(255,255,255,0.04)' : 'transparent'};
                                `;
                                cell.innerHTML = `
                                    <span style="font-size:0.75rem; font-weight:${isTotal ? '700' : '500'};">${val}</span>
                                    <span style="font-size:0.6rem; color:var(--text-muted); margin-top:2px;">${lbl}</span>
                                `;
                                timingRow.appendChild(cell);
                            });
                            body.appendChild(timingRow);
                        }
                    }

                    if (modalCloseBtn) modalCloseBtn.style.display = '';
                    footer.style.display = '';
                    footer.innerHTML = '';
                    const closeBtn = document.createElement('button');
                    closeBtn.classList.add('btn');
                    closeBtn.textContent = 'Close';
                    closeBtn.onclick = () => {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    };
                    footer.appendChild(closeBtn);

                } catch (err) {
                    if (err === CANCELLED) return;
                    if (isCancelled) return;
                    failStep(currentStepIdx);

                    // ── Show error message + Close button ──
                    const errorDiv = document.createElement('div');
                    errorDiv.classList.add('pipeline-error-msg');
                    errorDiv.innerHTML = `
                        <div style="font-weight:600;margin-bottom:4px;">⚠️ Training Failed</div>
                        <div style="font-family:monospace;font-size:0.8em;white-space:pre-wrap;word-break:break-word;opacity:0.8;">${err.message || err.toString()}</div>
                    `;
                    body.appendChild(errorDiv);
                    if (modalCloseBtn) modalCloseBtn.style.display = '';
                    footer.style.display = '';
                    footer.innerHTML = '';
                    const closeBtn = document.createElement('button');
                    closeBtn.classList.add('btn');
                    closeBtn.textContent = 'Close';
                    closeBtn.onclick = () => {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    };
                    footer.appendChild(closeBtn);
                }
            };

            footer.appendChild(cancelBtn);
            footer.appendChild(runBtn);
        }
    );

    // Adjust modal width
    const container = overlay.querySelector('.modal-container');
    if (container) container.style.width = "620px";
}

/**
 * Select points in a 2D area (lasso or rectangle) and highlights them in red.
 * Projects 3D points to screen space to check if they fall within the selection area.
 */
export function deselectPoints(scene, type, area) {
    const loader = getPotree2Loader(scene);
    if (loader && loader.removeSelection) {
        return loader.removeSelection(type, area);
    }
    return 0;
}

export function selectPoints(scene, pointCloudRoot, type, area, append = false) {
    if (!pointCloudRoot) {
        console.warn("No point cloud loaded for selection.");
        return 0;
    }

    // Check if we have a Potree2Loader for persistent selection across LOD levels
    const loader = getPotree2Loader(scene);
    if (loader && loader.applySelection) {
        return loader.applySelection(type, area, append);
    }

    if (!append) {
        clearSelection(scene);
    }

    // Fallback logic for simple meshes (non-LOD systems)
    const meshes = pointCloudRoot.getChildMeshes ? pointCloudRoot.getChildMeshes() : [pointCloudRoot];
    const camera = scene.activeCamera;
    const engine = scene.getEngine();
    const viewport = camera.viewport.toGlobal(engine.getRenderWidth(), engine.getRenderHeight());
    const transformMatrix = scene.getTransformMatrix();

    let totalSelected = 0;
    meshes.forEach(mesh => {
        if (!mesh.isVisible) return;
        const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
        const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
        if (!positions || !colors) return;

        let modified = false;
        const worldMatrix = mesh.getWorldMatrix();

        for (let i = 0; i < positions.length / 3; i++) {
            const vector = new BABYLON.Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            const projection = BABYLON.Vector3.Project(vector, worldMatrix, transformMatrix, viewport);

            let isInside = false;
            if (type === "rect") {
                isInside = (projection.x >= area.x && projection.x <= area.x + area.width &&
                    projection.y >= area.y && projection.y <= area.y + area.height);
            } else if (type === "lasso" || type === "polygon") {
                isInside = isPointInPoly(area, [projection.x, projection.y]);
            }

            if (isInside) {
                colors[i * 4 + 0] = 1.0;
                colors[i * 4 + 1] = 0.0;
                colors[i * 4 + 2] = 0.0;
                modified = true;
                totalSelected++;
            }
        }
        if (modified) mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
    });

    console.log(`🎯 Selection: ${totalSelected.toLocaleString()} points highlighted.`);
    return totalSelected;
}

/**
 * Clear all point selections and reset colors to original.
 * Wrapper around Potree2Loader.clearSelection() for external use.
 */
export function clearSelection(scene) {
    const loader = getPotree2Loader(scene);
    if (loader && loader.clearSelection) {
        loader.clearSelection();
        return;
    }

    // Fallback for simple meshes (non-LOD systems)
    const root = scene.potree2Root;
    if (!root) return;
    const meshes = root.getChildMeshes ? root.getChildMeshes() : [root];
    meshes.forEach(mesh => {
        if (mesh.metadata && mesh.metadata.originalColors) {
            mesh.setVerticesData(
                BABYLON.VertexBuffer.ColorKind,
                new Float32Array(mesh.metadata.originalColors)
            );
        }
    });
    console.log("🧹 Selection cleared.");
}

/**
 * Invert the current point selection.
 * Points that were selected (red) become unselected (original color),
 * and unselected points become selected (red).
 * Works on all currently loaded nodes via Potree2Loader.
 */
export function invertSelection(scene) {
    const loader = getPotree2Loader(scene);

    if (loader && loader.invertSelection) {
        loader.invertSelection();
        return;
    }

    // Fallback for simple meshes (non-LOD systems)
    const root = scene.potree2Root;
    if (!root) return;
    const meshes = root.getChildMeshes ? root.getChildMeshes() : [root];
    meshes.forEach(mesh => {
        const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
        const originalColors = mesh.metadata?.originalColors;
        if (!colors || !originalColors) return;
        const numPoints = colors.length / 4;
        for (let i = 0; i < numPoints; i++) {
            const isSelected = (
                colors[i * 4 + 0] > 0.9 &&
                colors[i * 4 + 1] < 0.1 &&
                colors[i * 4 + 2] < 0.1
            );
            if (isSelected) {
                colors[i * 4 + 0] = originalColors[i * 4 + 0];
                colors[i * 4 + 1] = originalColors[i * 4 + 1];
                colors[i * 4 + 2] = originalColors[i * 4 + 2];
                colors[i * 4 + 3] = originalColors[i * 4 + 3];
            } else {
                colors[i * 4 + 0] = 1.0;
                colors[i * 4 + 1] = 0.0;
                colors[i * 4 + 2] = 0.0;
                colors[i * 4 + 3] = 1.0;
            }
        }
        mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
    });
    console.log("🔄 Selection inverted (fallback).");
}

/**
 * Standard Point-in-Polygon algorithm (Ray Casting)
 */
function isPointInPoly(poly, pt) {
    const x = pt[0], y = pt[1];
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0], yi = poly[i][1];
        const xj = poly[j][0], yj = poly[j][1];

        const intersect = ((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

/**
 * Shows a modal for recalculating selected features with custom radii.
 * The available features are fixed (matching the C++ feature_extraction tool).
 */
export function showCalculateFeaturesModal(scene, onConfirm) {
    const overlay = createModal(
        "Calculate Features",
        (body) => {
            // Fixed list of available features (matches AVAILABLE_SCALE_FEATURES in C++)
            const baseNames = [
                "anisotropy", "omnivariance", "sphericity",
                "planarity", "linearity", "verticality", "surface_variation",
                "neighbours", "vertical_range", "height_above", "height_below", "height"
            ];

            // --- Info text ---
            const info = document.createElement('div');
            info.style.cssText = "font-size:0.75rem;color:var(--text-muted);line-height:1.5;margin-bottom:12px;";
            info.textContent = "Select the features to calculate and specify the search radius.";
            body.appendChild(info);

            // --- Feature list ---
            const featTitle = document.createElement('div');
            featTitle.classList.add('modal-section-title');
            featTitle.textContent = "Features";
            body.appendChild(featTitle);

            const featSection = document.createElement('div');
            featSection.classList.add('modal-section');
            featSection.style.paddingTop = "6px";

            if (baseNames.length === 0) {
                const empty = document.createElement('span');
                empty.style.cssText = "font-size:0.75rem;color:var(--text-muted);";
                empty.textContent = "No features available.";
                featSection.appendChild(empty);
            } else {
                // Select All / None controls
                const controls = document.createElement('div');
                controls.classList.add('feature-matrix-controls');
                controls.style.marginBottom = "8px";

                const selAll = document.createElement('button');
                selAll.classList.add('feature-matrix-ctrl-btn');
                selAll.textContent = "Select All";

                const selNone = document.createElement('button');
                selNone.classList.add('feature-matrix-ctrl-btn');
                selNone.textContent = "Select None";

                controls.appendChild(selAll);
                controls.appendChild(selNone);
                featSection.appendChild(controls);

                // Checkbox list
                const listWrapper = document.createElement('div');
                listWrapper.classList.add('checkbox-group');
                listWrapper.style.cssText = "display:flex;flex-direction:column;gap:4px;max-height:220px;overflow-y:auto;padding-right:4px;";

                const checkboxes = [];
                baseNames.forEach(name => {
                    const cb = createSimpleCheckbox(name, true, listWrapper);
                    checkboxes.push({ name, cb });
                });

                selAll.onclick = () => checkboxes.forEach(({ cb }) => { cb.checked = true; });
                selNone.onclick = () => checkboxes.forEach(({ cb }) => { cb.checked = false; });

                featSection.appendChild(listWrapper);
                featSection._getSelected = () => checkboxes.filter(({ cb }) => cb.checked).map(({ name }) => name);
            }

            body.appendChild(featSection);

            // --- Radii input ---
            const radiiTitle = document.createElement('div');
            radiiTitle.classList.add('modal-section-title');
            radiiTitle.style.marginTop = "16px";
            radiiTitle.textContent = "Search Radius";
            body.appendChild(radiiTitle);

            const radiiSection = document.createElement('div');
            radiiSection.classList.add('modal-section');
            radiiSection.style.paddingTop = "6px";

            const radiiHint = document.createElement('div');
            radiiHint.style.cssText = "font-size:0.72rem;color:var(--text-muted);margin-bottom:8px;";
            radiiHint.textContent = "Specify 1 to 5 search radii in meters (e.g. 0.1):";
            radiiSection.appendChild(radiiHint);

            const radiiContainer = document.createElement('div');
            radiiContainer.style.cssText = "display:flex; flex-direction:column; gap:6px;";

            const radiiInputs = [];
            for (let i = 0; i < 5; i++) {
                const row = document.createElement('div');
                row.style.cssText = 'display:flex; justify-content:space-between; align-items:center; gap:8px;';

                const label = document.createElement('span');
                label.classList.add('property-label');
                label.style.width = '80px';
                label.textContent = `Radius ${i + 1}:`;

                const rightSection = document.createElement('div');
                rightSection.style.cssText = 'display:flex; gap:6px; align-items:center; flex-shrink:0;';

                const infoIcon = document.createElement('img');
                infoIcon.src = '/static/viewer/icons/info.png';
                infoIcon.style.cssText = 'display:none; width:16px; height:16px; cursor:help; flex-shrink:0;';
                infoIcon.title = 'Value is too small and may cause errors';

                const input = document.createElement('input');
                input.type = 'number';
                input.step = '0.1';
                input.min = '0';
                input.classList.add('property-input');
                input.placeholder = "— empty —";
                input.style.width = '100px';
                if (i === 0) input.value = "0.5";
                if (i === 1) input.value = "1";
                if (i === 2) input.value = "2";

                // Validation: highlight in orange if < 0.5
                const updateInputStyle = () => {
                    const val = parseFloat(input.value);
                    if (!isNaN(val) && val > 0 && val < 0.5) {
                        input.style.backgroundColor = 'rgba(255, 165, 0, 0.3)';
                        input.style.borderColor = '#ffa500';
                        infoIcon.style.display = 'block';
                    } else {
                        input.style.backgroundColor = '';
                        input.style.borderColor = '';
                        infoIcon.style.display = 'none';
                    }
                };

                input.addEventListener('input', updateInputStyle);
                input.addEventListener('change', updateInputStyle);

                rightSection.appendChild(infoIcon);
                rightSection.appendChild(input);

                row.appendChild(label);
                row.appendChild(rightSection);
                radiiContainer.appendChild(row);
                radiiInputs.push(input);
            }
            radiiSection.appendChild(radiiContainer);

            body.appendChild(radiiSection);

            // --- GPU Toggle ---
            const gpuTitle = document.createElement('div');
            gpuTitle.classList.add('modal-section-title');
            gpuTitle.style.marginTop = '16px';
            gpuTitle.textContent = 'Options';
            body.appendChild(gpuTitle);

            const gpuSection = document.createElement('div');
            gpuSection.classList.add('modal-section');
            gpuSection.style.paddingTop = '6px';

            const gpuRow = document.createElement('div');
            gpuRow.classList.add('property-row');
            gpuRow.style.padding = '0 4px';
            gpuRow.innerHTML = `<span class="property-label">Accelerate with GPU:</span>`;

            const gpuToggleWrap = document.createElement('div');
            gpuToggleWrap.classList.add('rf-toggle-wrap');

            const gpuToggle = document.createElement('button');
            gpuToggle.id = 'calc-features-gpu-toggle';
            gpuToggle.classList.add('rf-toggle-btn');
            gpuToggle.setAttribute('data-on', 'false');
            gpuToggle.textContent = 'OFF';
            gpuToggle.onclick = () => {
                const isOn = gpuToggle.getAttribute('data-on') === 'true';
                gpuToggle.setAttribute('data-on', String(!isOn));
                gpuToggle.textContent = isOn ? 'OFF' : 'ON';
                gpuToggle.classList.toggle('on', !isOn);
            };

            gpuToggleWrap.appendChild(gpuToggle);
            gpuRow.appendChild(gpuToggleWrap);
            gpuSection.appendChild(gpuRow);
            body.appendChild(gpuSection);

            // Attach getters to body for footer access
            body._getFeatSection = () => featSection;
            body._getRadiiInputs = () => radiiInputs;
            body._getUseGpu = () => document.getElementById('calc-features-gpu-toggle')?.getAttribute('data-on') === 'true';
        },
        (footer, overlay) => {
            const cancelBtn = document.createElement('button');
            cancelBtn.classList.add('btn');
            cancelBtn.style.backgroundColor = 'transparent';
            cancelBtn.style.border = '1px solid var(--border-color)';
            cancelBtn.textContent = "Cancel";
            cancelBtn.onclick = () => {
                overlay.classList.remove('active');
                setTimeout(() => overlay.remove(), 300);
            };

            const confirmBtn = document.createElement('button');
            confirmBtn.classList.add('btn');
            confirmBtn.textContent = "Calculate";
            confirmBtn.onclick = async () => {
                const body = overlay.querySelector('.modal-body');
                const featSection = body._getFeatSection();
                const radiiInputs = body._getRadiiInputs();
                const useGpu = body._getUseGpu ? body._getUseGpu() : false;

                const selectedFeatures = featSection._getSelected
                    ? featSection._getSelected()
                    : [];

                if (selectedFeatures.length === 0) {
                    alert("Please select at least one feature.");
                    return;
                }

                // Get values from inputs
                const radii = radiiInputs
                    .map(inp => parseFloat(inp.value.trim()))
                    .filter(n => !isNaN(n) && n > 0);

                if (radii.length === 0) {
                    alert("Please enter at least one valid radius.");
                    return;
                }

                if (radii.length > 5) {
                    alert("Maximum 5 radii allowed.");
                    return;
                }

                // ── Replace modal body with animated pipeline step list ──
                const container = overlay.querySelector('.modal-container');
                body.innerHTML = '';

                // Hide the X close button during pipeline
                const modalCloseBtn = overlay.querySelector('.modal-close');
                if (modalCloseBtn) modalCloseBtn.style.display = 'none';

                // Show footer with Cancel button
                let isCancelled = false;
                footer.style.display = '';
                footer.innerHTML = '';
                const cancelCalcBtn = document.createElement('button');
                cancelCalcBtn.classList.add('btn');
                cancelCalcBtn.style.backgroundColor = 'transparent';
                cancelCalcBtn.style.border = '1px solid var(--border-color)';
                cancelCalcBtn.textContent = 'Cancel';
                cancelCalcBtn.onclick = async () => {
                    if (isCancelled) return;
                    isCancelled = true;
                    cancelCalcBtn.disabled = true;
                    cancelCalcBtn.textContent = 'Cancelling\u2026';
                    try {
                        await fetch('/stop_process/', {
                            method: 'POST',
                            headers: { 'X-CSRFToken': getCSRFToken() }
                        });
                    } catch (e) {
                        console.warn('\u26a0\ufe0f Could not call stop_process:', e);
                    }
                    overlay.classList.remove('active');
                    setTimeout(() => overlay.remove(), 300);
                };
                footer.appendChild(cancelCalcBtn);

                const STEPS = [
                    "Feature Extraction",
                    "Generating Feature Bin",
                    "Loading Features"
                ];

                // Build step list
                const stepList = document.createElement('div');
                stepList.classList.add('pipeline-step-list');

                const stepEls = STEPS.map((label) => {
                    const item = document.createElement('div');
                    item.classList.add('pipeline-step-item', 'step-idle');

                    const iconWrap = document.createElement('div');
                    iconWrap.classList.add('step-icon-wrap');
                    iconWrap.innerHTML = `
                        <svg class="step-svg" viewBox="0 0 32 32">
                            <circle class="step-track" cx="16" cy="16" r="13" />
                            <circle class="step-arc"   cx="16" cy="16" r="13" />
                        </svg>
                        <div class="step-check">
                            <svg viewBox="0 0 12 12" fill="none">
                                <polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="step-cross">
                            <svg viewBox="0 0 12 12" fill="none">
                                <line x1="2.5" y1="2.5" x2="9.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                <line x1="9.5" y1="2.5" x2="2.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                            </svg>
                        </div>
                    `;

                    const labelEl = document.createElement('span');
                    labelEl.classList.add('step-label');
                    labelEl.textContent = label;

                    item.appendChild(iconWrap);
                    item.appendChild(labelEl);
                    stepList.appendChild(item);
                    return item;
                });

                body.appendChild(stepList);

                const activateStep = (idx) => {
                    stepEls[idx].classList.remove('step-idle', 'step-done', 'step-error');
                    stepEls[idx].classList.add('step-running');
                };
                const completeStep = (idx) => {
                    stepEls[idx].classList.remove('step-running');
                    stepEls[idx].classList.add('step-done');
                };
                const failStep = (idx) => {
                    stepEls[idx].classList.remove('step-running');
                    stepEls[idx].classList.add('step-error');
                };

                let currentStepIdx = 0;
                const CANCELLED = Symbol('CANCELLED');
                const checkCancelled = () => { if (isCancelled) throw CANCELLED; };

                try {
                    // STEP 1: Feature Extraction
                    checkCancelled();
                    currentStepIdx = 0;
                    activateStep(0);
                    console.log("🔬 Step 1: Extracting features...");
                    const featResponse = await fetch('/feature_extraction/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            input_filepath: _runtimePath('working', 'features.las'),
                            output_filepath: _runtimePath('working', 'features.las'),
                            feature_list: selectedFeatures,
                            radius_list: radii,
                            use_gpu: useGpu
                        })
                    });
                    if (!featResponse.ok) {
                        const errData = await featResponse.json().catch(() => ({}));
                        throw new Error(errData.message || errData.error || "Feature extraction failed");
                    }
                    completeStep(0);
                    console.log("✅ Feature extraction completed");

                    // STEP 2: Generate Features pcbin
                    checkCancelled();
                    currentStepIdx = 1;
                    activateStep(1);
                    console.log("📦 Step 2: Generating features pcbin...");
                    const binResponse = await fetch('/las_to_feature_bin/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            las_path: _runtimePath('working', 'features.las'),
                            pcbin_path: _runtimePath('working', 'features.pcbin')
                        })
                    });
                    if (!binResponse.ok) {
                        const errData = await binResponse.json().catch(() => ({}));
                        throw new Error(errData.message || errData.error || "Feature bin generation failed");
                    }
                    completeStep(1);
                    console.log("✅ Features pcbin generated");

                    // STEP 3: Load Features in Viewer
                    checkCancelled();
                    currentStepIdx = 2;
                    activateStep(2);
                    console.log("🏗️ Step 3: Loading features in viewer...");
                    try {
                        const featureBinLoader = window.__babylonScene?.potree2Loader;
                        if (featureBinLoader) {
                            const featureNames = await featureBinLoader.loadPcBin(_runtimeUrl('working', 'features.pcbin'));
                            window.dispatchEvent(new CustomEvent('feature-bin-loaded', { detail: { names: featureNames } }));
                        }
                    } catch (binErr) {
                        console.warn("⚠️ Feature bin not loaded in viewer:", binErr.message);
                    }
                    completeStep(2);
                    console.log("✅ Features loaded in viewer");

                    if (onConfirm) onConfirm({ features: selectedFeatures, radii });

                    await new Promise(r => setTimeout(r, 800));
                    if (!isCancelled) {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    }

                } catch (err) {
                    if (err === CANCELLED) return;
                    console.error("❌ Feature Calculation Error:", err);
                    if (isCancelled) return;

                    failStep(currentStepIdx);

                    const errorDiv = document.createElement('div');
                    errorDiv.classList.add('pipeline-error-msg');
                    errorDiv.innerHTML = `
                        <div style="font-weight:600;margin-bottom:4px;">⚠️ Calculation Failed</div>
                        <div style="font-family:monospace;font-size:0.8em;white-space:pre-wrap;word-break:break-word;opacity:0.8;">${err.message || err.toString()}</div>
                    `;
                    body.appendChild(errorDiv);

                    if (modalCloseBtn) modalCloseBtn.style.display = '';

                    footer.style.display = '';
                    footer.innerHTML = '';
                    const closeBtn = document.createElement('button');
                    closeBtn.classList.add('btn');
                    closeBtn.textContent = "Close";
                    closeBtn.onclick = () => {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 300);
                    };
                    footer.appendChild(closeBtn);
                }
            };

            footer.appendChild(cancelBtn);
            footer.appendChild(confirmBtn);
        }
    );

    return overlay;
}

export function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

// =====================================================================
// CLASSIFY MODAL
// =====================================================================
/**
 * showClassifyModal(scene)
 *
 * Opens a modal to configure and launch RF classification.
 * Sections:
 *   1. Data Split — select which LAS to classify (full cloud or segment)
 *   2. Model      — pick a trained model from /api/models-list/
 *   3. Options    — Use GPU toggle
 */
export async function showClassifyModal(scene) {
    const overlay = createModal(
        "Classify Configuration",
        (body) => {
            // ── SECTION 1: DATA SPLIT ─────────────────────────────────────────
            const splitTitle = document.createElement('div');
            splitTitle.classList.add('modal-section-title');
            splitTitle.textContent = "Data Split";
            body.appendChild(splitTitle);

            const splitSection = document.createElement('div');
            splitSection.classList.add('modal-section');
            body.appendChild(splitSection);

            const makeSelectRow = (labelStr, selectId) => {
                const row = document.createElement('div');
                row.classList.add('property-row');
                row.style.alignItems = 'center';

                const label = document.createElement('span');
                label.classList.add('property-label');
                label.style.width = '120px';
                label.textContent = labelStr + ':';
                row.appendChild(label);

                const select = document.createElement('select');
                select.id = selectId;
                select.style.flex = '1';
                select.classList.add('property-input');
                row.appendChild(select);

                return { row, select };
            };

            const { row: splitRow, select: splitSelect } = makeSelectRow('Input LAS', 'classify-split-select');
            splitSection.appendChild(splitRow);

            // No "Full point cloud" option — user must specify a segment

            // Add segments from the Outline panel
            document.querySelectorAll('.outline-item[data-segment-id]').forEach(item => {
                const segmentId = item.dataset.segmentId;
                const nameInput = item.querySelector('.outline-name-input');
                const name = nameInput ? nameInput.value : `Segment ${segmentId}`;
                const opt = document.createElement('option');
                opt.value = segmentId;
                opt.textContent = name;
                splitSelect.appendChild(opt);
            });

            // ── SECTION 2: MODEL ──────────────────────────────────────────────
            const modelTitle = document.createElement('div');
            modelTitle.classList.add('modal-section-title');
            modelTitle.textContent = "Model";
            body.appendChild(modelTitle);

            const modelSection = document.createElement('div');
            modelSection.classList.add('modal-section');
            body.appendChild(modelSection);

            const { row: modelRow, select: modelSelect } = makeSelectRow('Trained Model', 'classify-model-select');
            modelSection.appendChild(modelRow);

            // Loading placeholder
            const loadingOpt = document.createElement('option');
            loadingOpt.value = '';
            loadingOpt.textContent = 'Loading models…';
            loadingOpt.disabled = true;
            loadingOpt.selected = true;
            modelSelect.appendChild(loadingOpt);

            // Fetch available models
            fetch('/api/models-list/')
                .then(r => r.json())
                .then(data => {
                    modelSelect.innerHTML = '';
                    const models = data.models || [];
                    if (models.length === 0) {
                        const noOpt = document.createElement('option');
                        noOpt.value = '';
                        noOpt.textContent = '— No models found —';
                        noOpt.disabled = true;
                        modelSelect.appendChild(noOpt);
                    } else {
                        models.forEach(m => {
                            const opt = document.createElement('option');
                            opt.value = m.path;
                            opt.textContent = `${m.name}  (${m.created}, ${m.size_mb} MB)`;
                            modelSelect.appendChild(opt);
                        });

                        if (window.__selectedModelPath) {
                            const selectedOpt = [...modelSelect.options].find(opt => opt.value === window.__selectedModelPath);
                            if (selectedOpt) {
                                modelSelect.value = selectedOpt.value;
                            }
                        }

                        if (modelSelect.value) {
                            window.__selectedModelPath = modelSelect.value;
                            window.dispatchEvent(new CustomEvent('selected-model-changed', {
                                detail: { modelPath: modelSelect.value }
                            }));
                        }
                    }
                })
                .catch(() => {
                    modelSelect.innerHTML = '';
                    const errOpt = document.createElement('option');
                    errOpt.value = '';
                    errOpt.textContent = '— Failed to load models —';
                    errOpt.disabled = true;
                    modelSelect.appendChild(errOpt);
                });

            modelSelect.addEventListener('change', () => {
                const modelPath = modelSelect.value || null;
                window.__selectedModelPath = modelPath;
                window.dispatchEvent(new CustomEvent('selected-model-changed', {
                    detail: { modelPath }
                }));
            });

            // ── SECTION 3: OPTIONS ────────────────────────────────────────────
            const optTitle = document.createElement('div');
            optTitle.classList.add('modal-section-title');
            optTitle.textContent = "Options";
            body.appendChild(optTitle);

            const optSection = document.createElement('div');
            optSection.classList.add('modal-section');
            body.appendChild(optSection);

            const gpuRow = document.createElement('div');
            gpuRow.classList.add('property-row');
            gpuRow.style.padding = '0 4px';
            gpuRow.innerHTML = `<span class="property-label">Use GPU:</span>`;

            const gpuToggleWrap = document.createElement('div');
            gpuToggleWrap.classList.add('rf-toggle-wrap');

            const gpuToggle = document.createElement('button');
            gpuToggle.id = 'classify-gpu-toggle';
            gpuToggle.classList.add('rf-toggle-btn');
            gpuToggle.setAttribute('data-on', 'false');
            gpuToggle.textContent = 'OFF';
            gpuToggle.onclick = () => {
                const isOn = gpuToggle.getAttribute('data-on') === 'true';
                gpuToggle.setAttribute('data-on', String(!isOn));
                gpuToggle.textContent = isOn ? 'OFF' : 'ON';
                gpuToggle.classList.toggle('on', !isOn);
            };

            gpuToggleWrap.appendChild(gpuToggle);
            gpuRow.appendChild(gpuToggleWrap);
            optSection.appendChild(gpuRow);
        },

        // ── FOOTER ────────────────────────────────────────────────────────────
        (footer, overlay) => {
            const cancelBtn = document.createElement('button');
            cancelBtn.classList.add('btn');
            cancelBtn.style.backgroundColor = 'transparent';
            cancelBtn.style.border = '1px solid var(--border-color)';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.onclick = () => {
                overlay.classList.remove('active');
                setTimeout(() => overlay.remove(), 300);
            };

            const runBtn = document.createElement('button');
            runBtn.classList.add('btn');
            runBtn.textContent = 'Start Classify';

            runBtn.onclick = async () => {
                const body = overlay.querySelector('.modal-body');

                const splitValue = body.querySelector('#classify-split-select').value;
                const modelPath = body.querySelector('#classify-model-select').value;
                const useGpu = body.querySelector('#classify-gpu-toggle').getAttribute('data-on') === 'true';

                // Error helper (shown before pipeline starts)
                const showError = (msg) => {
                    const existing = overlay.querySelector('.classify-modal-error');
                    if (existing) existing.remove();
                    const errorDiv = document.createElement('div');
                    errorDiv.classList.add('classify-modal-error');
                    errorDiv.style.cssText = 'color:#ff5f5f;font-size:0.75rem;padding:0 4px 10px 4px;text-align:center;width:100%;';
                    errorDiv.textContent = msg;
                    footer.insertAdjacentElement('beforebegin', errorDiv);
                    setTimeout(() => {
                        errorDiv.style.transition = 'opacity 0.4s ease';
                        errorDiv.style.opacity = '0';
                        setTimeout(() => errorDiv.remove(), 400);
                    }, 4000);
                };

                if (!modelPath) { showError('Please select a trained model.'); return; }

                window.__selectedModelPath = modelPath;
                window.dispatchEvent(new CustomEvent('selected-model-changed', {
                    detail: { modelPath }
                }));

                const modelDir = modelPath.replace(/\/model\.pkl$/, '');
                const modelName = modelDir.split('/').pop();
                const classifyWorkingDir = _runtimePath('working', modelName, 'classify_working/');
                const outputClassifyName = _runtimePath('working', modelName, 'predicted.las');
                const needsSplit = splitValue !== '__full__';

                // ── Read model metadata to get the feature list ────────────────
                let modelFeatures = [];
                try {
                    const metaRes = await fetch(`/api/read-file/?path=${encodeURIComponent(modelDir + '/metadata.json')}`);
                    if (metaRes.ok) {
                        const metaData = await metaRes.json();
                        const meta = JSON.parse(metaData.content || '{}');
                        if (Array.isArray(meta.features) && meta.features.length > 0)
                            modelFeatures = meta.features;
                    }
                } catch (_) { }

                if (modelFeatures.length === 0) {
                    showError('Could not read feature list from model metadata. Re-train the model and try again.');
                    return;
                }

                // Parse features/radii from model feature names
                // e.g. "linearity_0_1" → name="linearity", radius=0.1
                const featureSet = new Set();
                const radiusSet = new Set();
                for (const f of modelFeatures) {
                    const match = f.match(/^(.+)_(\d+_\d+)$/);
                    if (match) {
                        featureSet.add(match[1]);
                        radiusSet.add(parseFloat(match[2].replace('_', '.')));
                    } else {
                        featureSet.add(f); // no-radius feature (e.g. rgb)
                    }
                }
                const featureList = [...featureSet];
                const radiusList = [...radiusSet].sort((a, b) => a - b);

                // ── Switch to pipeline UI ──────────────────────────────────────
                body.innerHTML = '';
                const modalCloseBtn = overlay.querySelector('.modal-close');
                if (modalCloseBtn) modalCloseBtn.style.display = 'none';

                let isCancelled = false;
                footer.style.display = '';
                footer.innerHTML = '';

                const cancelRunBtn = document.createElement('button');
                cancelRunBtn.classList.add('btn');
                cancelRunBtn.style.cssText = 'background:transparent;border:1px solid var(--border-color);';
                cancelRunBtn.textContent = 'Stop';
                cancelRunBtn.onclick = async () => {
                    if (isCancelled) return;
                    isCancelled = true;
                    cancelRunBtn.disabled = true;
                    cancelRunBtn.textContent = 'Stopping…';
                    try { await fetch('/stop_process/', { method: 'POST', headers: { 'X-CSRFToken': getCSRFToken() } }); }
                    catch (e) { console.warn('Could not stop process:', e); }
                    overlay.classList.remove('active');
                    setTimeout(() => overlay.remove(), 300);
                };
                footer.appendChild(cancelRunBtn);

                // Step index helpers
                let stepIdx = 0;
                const STEP_PREPARE = needsSplit ? stepIdx++ : -1;
                const STEP_FEAT_EXTRACT = stepIdx++;
                const STEP_CLASSIFY = stepIdx++;
                const STEP_POTREE = stepIdx++;
                const STEP_RELOAD = stepIdx++;

                // Build step list dynamically
                const STEPS = [
                    ...(needsSplit ? ['Data Preparation'] : []),
                    'Feature Extraction',
                    'Point Cloud Classification',
                    'Potree Conversion',
                    'Visualization Update',
                ];

                const stepList = document.createElement('div');
                stepList.classList.add('pipeline-step-list');
                const stepEls = STEPS.map((label) => {
                    const item = document.createElement('div');
                    item.classList.add('pipeline-step-item', 'step-idle');
                    item.innerHTML = `
                        <div class="step-icon-wrap">
                            <svg class="step-svg" viewBox="0 0 32 32">
                                <circle class="step-track" cx="16" cy="16" r="13"/>
                                <circle class="step-arc"   cx="16" cy="16" r="13"/>
                            </svg>
                            <div class="step-check">
                                <svg viewBox="0 0 12 12" fill="none">
                                    <polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                            </div>
                            <div class="step-cross">
                                <svg viewBox="0 0 12 12" fill="none">
                                    <line x1="2.5" y1="2.5" x2="9.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                    <line x1="9.5" y1="2.5" x2="2.5" y2="9.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                </svg>
                            </div>
                        </div>
                        <span class="step-label">${label}</span>`;
                    stepList.appendChild(item);
                    return item;
                });
                body.appendChild(stepList);

                const activateStep = (i) => { stepEls[i].classList.replace('step-idle', 'step-running') || stepEls[i].classList.remove('step-done', 'step-error') || stepEls[i].classList.add('step-running'); };
                const completeStep = (i) => { stepEls[i].classList.remove('step-running'); stepEls[i].classList.add('step-done'); };
                const failStep = (i) => { stepEls[i].classList.remove('step-running'); stepEls[i].classList.add('step-error'); };

                const CANCELLED = Symbol('CANCELLED');
                const checkCancelled = () => { if (isCancelled) throw CANCELLED; };

                let currentStepIdx = 0;
                let testFilepath = _runtimePath('working', 'features.las');

                try {
                    // ── STEP (optional): Data Preparation ─────────────────────
                    if (needsSplit) {
                        checkCancelled();
                        currentStepIdx = STEP_PREPARE;
                        activateStep(currentStepIdx);

                        const loader = scene.potree2Loader;
                        const segmentMap = {};
                        document.querySelectorAll('.outline-item[data-segment-id]').forEach(item => {
                            const segId = parseInt(item.dataset.segmentId, 10);
                            const nameInput = item.querySelector('.outline-name-input');
                            if (!isNaN(segId) && nameInput) segmentMap[segId] = nameInput.value;
                        });
                        const classMap = {};
                        classRegistry.forEach((val, id) => { classMap[id] = val.name; });
                        const exportResult = await loader.exportAllTrainingDataAsMapping(segmentMap);
                        if (!exportResult || !exportResult.mapping_path)
                            throw new Error('No points found for the selected segment.');

                        const mappingPath = exportResult.mapping_path;

                        // Ensure features.pcbin exists — it is required by extract-segment-las.
                        // If the user loaded a cloud without running feature calculation or
                        // training, the file won't be present yet: generate it on-the-fly.
                        const pcbinCheckRes = await fetch(_runtimeUrl('working', 'features.pcbin'), { method: 'HEAD' });
                        if (!pcbinCheckRes.ok) {
                            console.log('⚙️ features.pcbin not found — generating from features.las…');
                            const binGenRes = await fetch('/las_to_feature_bin/', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                                body: JSON.stringify({
                                    las_path: _runtimePath('working', 'features.las'),
                                    pcbin_path: _runtimePath('working', 'features.pcbin')
                                })
                            });
                            if (!binGenRes.ok) {
                                const e = await binGenRes.json().catch(() => ({}));
                                throw new Error(e.message || e.error || 'Failed to generate features.pcbin');
                            }
                            console.log('✅ features.pcbin generated');
                        }

                        const segmentOutPath = `${classifyWorkingDir}segment_${splitValue}.las`;
                        await fetch('/api/extract-segment-las/', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                            body: JSON.stringify({
                                las_path: _runtimePath('working', 'features.las'),
                                pcbin_path: _runtimePath('working', 'features.pcbin'),
                                seg_id: parseInt(splitValue, 10),
                                out_path: segmentOutPath,
                            })
                        });
                        testFilepath = segmentOutPath;
                        completeStep(currentStepIdx);
                    }

                    // ── STEP: Feature Extraction (includes sampling) ───────────
                    checkCancelled();
                    currentStepIdx = STEP_FEAT_EXTRACT;
                    activateStep(currentStepIdx);

                    // Get voxel size from report
                    const voxelRes = await fetch('/get_model_voxel_size/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({ model_dir: modelDir })
                    });
                    const voxelData = await voxelRes.json();
                    const voxelSize = voxelData.voxel_size || 0;

                    const featExtractedPath = `${classifyWorkingDir}features_extracted.las`;

                    const featRes = await fetch('/feature_extraction/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            input_filepath: testFilepath,
                            output_filepath: featExtractedPath,
                            feature_list: featureList,
                            radius_list: radiusList,
                            sampling: voxelSize,
                            use_gpu: useGpu,
                        })
                    });
                    if (!featRes.ok) {
                        const e = await featRes.json().catch(() => ({}));
                        throw new Error(e.message || e.error || 'Feature extraction failed');
                    }

                    completeStep(currentStepIdx);

                    // ── STEP: Classification ──────────────────────────────────
                    checkCancelled();
                    currentStepIdx = STEP_CLASSIFY;
                    activateStep(currentStepIdx);

                    const classifyRes = await fetch('/launch_RF_classify/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            model_savepath: modelPath,
                            test_filepath: featExtractedPath,
                            output_classify_name: outputClassifyName,
                            use_gpu: useGpu,
                            selected_features: modelFeatures,
                        }),
                    });
                    if (!classifyRes.ok) {
                        const e = await classifyRes.json().catch(() => ({}));
                        throw new Error(e.message || e.error || 'Classification failed');
                    }

                    await fetch('/las_to_feature_bin/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            las_path: outputClassifyName,
                            pcbin_path: _runtimePath('working', 'features.pcbin'),
                        })
                    });
                    completeStep(currentStepIdx);

                    // ── STEP: Potree Conversion ───────────────────────────────
                    checkCancelled();
                    currentStepIdx = STEP_POTREE;
                    activateStep(currentStepIdx);

                    const potreeRes = await fetch('/potree_converter/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                        body: JSON.stringify({
                            input_filepath: outputClassifyName,
                            output_filepath: _runtimePath('working', 'potree/'),
                        })
                    });
                    if (!potreeRes.ok) throw new Error("Potree conversion failed.");
                    completeStep(currentStepIdx);

                    // ── STEP: Visualization Update ────────────────────────────
                    checkCancelled();
                    currentStepIdx = STEP_RELOAD;
                    activateStep(currentStepIdx);
                    try {
                        const featureBinLoader = window.__babylonScene?.potree2Loader;
                        if (featureBinLoader) {
                            // Reload features from bin
                            const featureNames = await featureBinLoader.loadPcBin(_runtimeUrl('working', 'features.pcbin'));
                            window.dispatchEvent(new CustomEvent('feature-bin-loaded', { detail: { names: featureNames } }));

                            // Re-load the entire cloud points from new Potree output
                            if (window.loadActiveCloud) {
                                await window.loadActiveCloud(_runtimeUrl('working', 'potree', 'metadata.json'));
                            }
                        }
                    } catch (binErr) {
                        console.warn('⚠️ Visualization update failed:', binErr.message);
                    }
                    completeStep(currentStepIdx);

                    if (needsSplit) {
                        const selectedSegId = parseInt(splitValue, 10);
                        if (Number.isFinite(selectedSegId) && typeof window.__showOnlyOutlineSegment === 'function') {
                            window.__showOnlyOutlineSegment(selectedSegId);
                        }
                    }

                    // Success
                    if (modalCloseBtn) modalCloseBtn.style.display = '';
                    footer.innerHTML = '';
                    const closeBtn = document.createElement('button');
                    closeBtn.classList.add('btn');
                    closeBtn.textContent = 'Close';
                    closeBtn.onclick = () => { overlay.classList.remove('active'); setTimeout(() => overlay.remove(), 300); };
                    footer.appendChild(closeBtn);
                    console.log(`✅ Classification complete → ${outputClassifyName}`);

                } catch (err) {
                    if (err === CANCELLED) return;
                    failStep(currentStepIdx);
                    console.error('❌ Classify error:', err);

                    const errMsg = document.createElement('div');
                    errMsg.style.cssText = 'color:#ff5f5f;font-size:0.75rem;padding:12px 4px 0;text-align:center;';
                    errMsg.textContent = `Error: ${err.message}`;
                    body.appendChild(errMsg);

                    if (modalCloseBtn) modalCloseBtn.style.display = '';
                    footer.innerHTML = '';
                    const closeBtn = document.createElement('button');
                    closeBtn.classList.add('btn');
                    closeBtn.textContent = 'Close';
                    closeBtn.onclick = () => { overlay.classList.remove('active'); setTimeout(() => overlay.remove(), 300); };
                    footer.appendChild(closeBtn);
                }
            };

            footer.appendChild(cancelBtn);
            footer.appendChild(runBtn);
        }
    );

    return overlay;
}