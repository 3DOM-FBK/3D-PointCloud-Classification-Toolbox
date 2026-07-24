// =====================================================================
// POTREE 2.0 LOADER for BabylonJS
// Faithful implementation based on Potree's OctreeLoader.js & DecoderWorker.js
// Format: metadata.json + hierarchy.bin + octree.bin
// =====================================================================

/**
 * Potree 2.0 hierarchy node
 */
class Potree2Node {
    constructor(name, boundingBox) {
        this.name = name;
        this.level = name === "r" ? 0 : name.length - 1;
        this.boundingBox = boundingBox; // { min: [x,y,z], max: [x,y,z] }
        this.numPoints = 0;
        this.byteOffset = 0n;   // BigInt offset into octree.bin
        this.byteSize = 0n;     // BigInt size in octree.bin
        this.nodeType = 0;      // 0=normal, 2=proxy (lazy hierarchy chunk)
        this.childMask = 0;
        this.children = new Array(8).fill(null);
        this.spacing = 0;

        // For proxy nodes (type === 2)
        this.hierarchyByteOffset = 0n;
        this.hierarchyByteSize = 0n;

        // State
        this.loaded = false;
        this.loading = false;
        this.mesh = null;
    }

    get hasChildren() {
        return this.childMask !== 0;
    }
}

/**
 * Computes child AABB from parent AABB using octree index bits.
 * Potree convention:
 *   bit 0 (0b001) → Z axis
 *   bit 1 (0b010) → Y axis
 *   bit 2 (0b100) → X axis
 */
function createChildAABB(parentBB, childIndex) {
    const min = [...parentBB.min];
    const max = [...parentBB.max];
    const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

    if (childIndex & 0b100) { min[0] += size[0] / 2; } else { max[0] -= size[0] / 2; }
    if (childIndex & 0b010) { min[1] += size[1] / 2; } else { max[1] -= size[1] / 2; }
    if (childIndex & 0b001) { min[2] += size[2] / 2; } else { max[2] -= size[2] / 2; }

    return { min, max };
}

const FEATURE_MISSING_SENTINEL = -1e38;
const FEATURE_MISSING_COLOR = 0.8;

// ---- Sentinels for the canonical point_id → segmentId map (_pointSegmentMap) ----
// The map is a Uint16Array, so it cannot store the negative _deletedSegmentId (-1)
// nor an "unknown" state: both need out-of-band values.
//   SEG_UNRESOLVED — this point has never been resolved to a segment. The geometry
//                    replay in _createMeshFromBuffer decides (and then freezes) it.
//   SEG_DELETED    — this point lives in the internal hidden "deleted" segment.
// Uint16 leaves room for 65534 user segments, which also removes the 253-segment
// ceiling an 8-bit map would have imposed.
const SEG_UNRESOLVED = 0xFFFF;
const SEG_DELETED = 0xFFFE;


/**
 * Potree 2.0 Loader for BabylonJS
 *
 * Uses HTTP Range requests via a Django endpoint to fetch only the needed
 * bytes for each node from octree.bin. This way, even multi-GB files can
 * be handled without loading the entire file into browser memory.
 */
export class Potree2Loader {
    constructor(scene, baseUrl, options = {}) {
        this.scene = scene;
        this.baseUrl = baseUrl;
        this.metadata = null;
        this.root = null;
        this.rootTransform = new BABYLON.TransformNode("Potree2Root", scene);

        const prefixes = ["/runtime-data/", "runtime-data/"];
        let foundPrefix = false;
        for (const prefix of prefixes) {
            if (baseUrl.startsWith(prefix)) {
                this.rangeBasePath = baseUrl.substring(prefix.length);
                foundPrefix = true;
                break;
            }
        }
        if (!foundPrefix) this.rangeBasePath = baseUrl;
        this.rangeBasePath = this.rangeBasePath.replace(/^\/+|\/+$/g, '');

        this.attributes = [];
        this.bytesPerPoint = 0;
        this.hierarchyBuffer = null;

        this.loadedNodes = new Map();   // name → mesh
        this.activeNodes = new Set();
        this.loadingNodes = new Set();

        // Fixed point size for all nodes — same value regardless of LOD level
        this.pointSize = options.pointSize ?? 2;
        // Multiplier applied on top of the auto-computed size (controlled by the UI slider)
        this.pointSizeMultiplier = options.pointSizeMultiplier ?? 1.0;
        this.maxVisibleNodes = options.maxVisibleNodes || 500;
        this.maxVisiblePoints = options.maxVisiblePoints || 5_000_000;
        this.maxConcurrentLoads = options.maxConcurrentLoads || 6;

        this.stats = {
            loadedNodes: 0,
            visibleNodes: 0,
            totalPointsRendered: 0,
            loadingNodes: 0
        };

        // Persistent Selection History
        // [{ type, area, viewport, transformMatrix }]
        this.selectionHistory = [];
        // Deselection regions (CTRL+select) — same structure as selectionHistory.
        // Points that fall here are excluded from classification even if
        // they are inside a selectionHistory region.
        this.deselectionHistory = [];

        // Flag: when true, the selection logic is inverted — points OUTSIDE the
        // selectionHistory regions are highlighted, not those inside.
        this.selectionInverted = false;

        // Persistent Classification History — one entry per "assign class" action.
        // [{ classId, r, g, b, minX, minY, minZ, maxX, maxY, maxZ }]
        // Uses a 3D AABB so future LOD nodes are classified with a spatial test,
        // independent of camera position/rotation.
        this.classificationHistory = [];

        // Current display mode, synchronized with the UI via setColorMode().
        // "classification" (default): classified points show their class color.
        // "color": all points show the original point cloud color.
        // _createMeshFromBuffer uses this value to decide which colors to write
        // into the vertex data of newly loaded LOD nodes.
        this.colorMode = "classification";
        this.classColorBlendStrength = 0.75;
        this.featureBin = null;
        this._featureRangeMin = null;
        this._featureRangeMax = null;
        this._featureShaderMat = null;
        this._colormapId = 0; // 0=Blue>Green>Yellow>Red (default)
        this._featureDiscreteFilter = null; // { featureName: string, value: number|null }

        // ---- CUT / SEGMENT HISTORY ----
        // segmentId 0 = main (uncut) cloud. Each cutSelection() adds an entry.
        // entry: { segmentId, visible, selections, deselections, minX..maxZ }
        this.cutHistory = [];
        this._segmentIdCounter = 1;
        this._cutCreationCounter = 0;
        this._deletedSegmentId = -1;
        this.mainCloudVisible = true;

        // Canonical class color lookup (classId → {r,g,b}) used by applyClassToLoadedNodes,
        // updateClassColor, removeClass, clearClassifications.
        // MUST be initialized here — without it, _classColorLUT.set() throws a TypeError,
        // which aborts applyClassToLoadedNodes before classificationHistory.push() runs,
        // leaving classificationHistory empty and causing all future LOD nodes to be unclassified.
        this._classColorLUT = new Map();
        this._pointClassMap = null;
        this._pointSegmentMap = null;

        // Runtime handles that must be cleaned on dispose() to avoid stale callbacks
        // after Reset Scene / re-import.
        this._cameraForObserver = null;
        this._cameraViewObserver = null;
        this._cleanupIntervalId = null;
    }

    _getLocalBoundingBoxCenter() {
        if (!this.metadata?.boundingBox) return null;

        const bbMin = this.metadata.boundingBox.min;
        const bbMax = this.metadata.boundingBox.max;

        return new BABYLON.Vector3(
            (bbMax[0] - bbMin[0]) * 0.5,
            (bbMax[1] - bbMin[1]) * 0.5,
            (bbMax[2] - bbMin[2]) * 0.5
        );
    }

    rotateAroundBoundingBoxCenter(axisName, stepDegrees = 90) {
        const center = this._getLocalBoundingBoxCenter();
        if (!center) return false;

        const axisMap = {
            x: BABYLON.Axis.X,
            y: BABYLON.Axis.Y,
            z: BABYLON.Axis.Z
        };
        const axis = axisMap[String(axisName || '').toLowerCase()];
        if (!axis) return false;

        this.rootTransform.setPivotPoint(center, BABYLON.Space.LOCAL);

        if (!this.rootTransform.rotationQuaternion) {
            this.rootTransform.rotationQuaternion = BABYLON.Quaternion.FromEulerAngles(
                this.rootTransform.rotation.x,
                this.rootTransform.rotation.y,
                this.rootTransform.rotation.z
            );
            this.rootTransform.rotation.setAll(0);
        }

        const delta = BABYLON.Quaternion.RotationAxis(axis, BABYLON.Tools.ToRadians(stepDegrees));
        this.rootTransform.rotationQuaternion = this.rootTransform.rotationQuaternion.multiply(delta);
        this.rootTransform.rotationQuaternion.normalize();
        this.rootTransform.computeWorldMatrix(true);

        return true;
    }

    // ========== PUBLIC API ==========

    async load() {
        console.log("🌲 Potree2Loader: Loading from", this.baseUrl);

        const metaResponse = await fetch(`${this.baseUrl}/metadata.json`);
        if (!metaResponse.ok) throw new Error(`Failed to load metadata.json: ${metaResponse.status}`);
        this.metadata = await metaResponse.json();

        console.log("📊 Metadata:");
        console.log("   Version:", this.metadata.version);
        console.log("   Points:", this.metadata.points.toLocaleString());
        console.log("   Spacing:", this.metadata.spacing);
        console.log("   Hierarchy depth:", this.metadata.hierarchy.depth);
        console.log("   Encoding:", this.metadata.encoding);
        console.log("   BoundingBox min:", this.metadata.boundingBox.min);
        console.log("   BoundingBox max:", this.metadata.boundingBox.max);
        console.log("   Scale:", this.metadata.scale);
        console.log("   Offset:", this.metadata.offset);

        this._parseAttributes();

        // Canonical point_id → segmentId map, indexed by the POINT_ID attribute.
        // This is the source of truth for segment membership: it survives LOD
        // unload/reload cycles, so a point keeps the segment it was assigned to
        // even when its octree node is dropped by cleanup() and fetched again.
        // Without it every reload re-derives membership from the 2D selection
        // replay, which is not equivalent to the original assignment (different
        // tie-breaking on overlaps) and makes hidden segments pop back in blocks.
        const hasPointId = this.attributes.some(a => {
            const n = a.name.toLowerCase();
            return n === "point_id" || n === "pointid";
        });
        if (hasPointId && this.metadata.points > 0) {
            this._pointSegmentMap = new Uint16Array(this.metadata.points).fill(SEG_UNRESOLVED);
        } else {
            this._pointSegmentMap = null;
            console.warn("⚠️ No POINT_ID attribute: segment membership will be re-derived geometrically on every LOD load.");
        }

        // Load the complete hierarchy.bin in one request.
        // hierarchy.bin contains ALL hierarchy chunks for the entire octree.
        // Previously only firstChunkSize bytes were used, causing proxy nodes
        // whose chunks were beyond that offset to never resolve — leaving entire
        // subtrees permanently at low LOD regardless of camera distance.
        const hierUrl = this._getRangeUrl("hierarchy.bin");
        const hierResponse = await fetch(hierUrl);
        if (!hierResponse.ok) throw new Error(`Failed to load hierarchy.bin: ${hierResponse.status}`);
        this.hierarchyBuffer = await hierResponse.arrayBuffer();
        console.log(`   hierarchy.bin loaded: ${this.hierarchyBuffer.byteLength.toLocaleString()} bytes (full file)`);

        const bbMin = this.metadata.boundingBox.min;
        const bbMax = this.metadata.boundingBox.max;
        const localBB = {
            min: [0, 0, 0],
            max: [bbMax[0] - bbMin[0], bbMax[1] - bbMin[1], bbMax[2] - bbMin[2]]
        };

        this.root = new Potree2Node("r", localBB);
        this.root.nodeType = 2;
        this.root.hierarchyByteOffset = 0n;
        this.root.hierarchyByteSize = BigInt(this.metadata.hierarchy.firstChunkSize);
        this.root.spacing = this.metadata.spacing;

        this._parseHierarchyChunk(this.root);

        const allNodes = this._collectNodes(this.root);
        const levelStats = {};
        let totalNodePoints = 0;
        for (const n of allNodes) {
            if (!levelStats[n.level]) levelStats[n.level] = { count: 0, points: 0 };
            levelStats[n.level].count++;
            levelStats[n.level].points += n.numPoints;
            totalNodePoints += n.numPoints;
        }
        // console.log(`✅ Hierarchy parsed: ${allNodes.length} nodes, ${totalNodePoints.toLocaleString()} total points`);
        // for (const [level, data] of Object.entries(levelStats).sort((a, b) => a[0] - b[0])) {
        // console.log(`   Level ${level}: ${data.count} nodes, ${data.points.toLocaleString()} points`);
        // }

        const pointCountDisplay = document.getElementById('point-count');
        if (pointCountDisplay) pointCountDisplay.textContent = this.metadata.points.toLocaleString();

        await this._loadInitialNodes();

        // Fix LAS ↔ BabylonJS coordinate system mismatch: negate X scale to remove mirror effect
        this.rootTransform.scaling.x = -1;
        this.rootTransform.computeWorldMatrix(true);

        return this.rootTransform;
    }

    _getRangeUrl(filename) {
        return `/pointcloud-data/${this.rangeBasePath}/${filename}`;
    }

    async _fetchRange(filename, byteOffset, byteSize) {
        const url = this._getRangeUrl(filename);
        const first = Number(byteOffset);
        const last = first + Number(byteSize) - 1;

        const response = await fetch(url, {
            headers: { 'Range': `bytes=${first}-${last}` }
        });

        if (response.status === 206) {
            return await response.arrayBuffer();
        } else if (response.ok) {
            console.warn(`⚠️ Server returned full file instead of range for ${filename}. Slicing locally.`);
            const fullBuffer = await response.arrayBuffer();
            return fullBuffer.slice(first, first + Number(byteSize));
        } else {
            throw new Error(`Range request failed for ${filename}: ${response.status} ${response.statusText}`);
        }
    }

    async _loadInitialNodes() {
        const allNodes = this._collectNodes(this.root);

        const initialNodes = allNodes
            .filter(n => n.level <= 2 && n.numPoints > 0 && n.byteSize > 0n)
            .sort((a, b) => a.level - b.level || b.numPoints - a.numPoints);

        const batchSize = 6;
        for (let i = 0; i < initialNodes.length; i += batchSize) {
            const batch = initialNodes.slice(i, i + batchSize);
            await Promise.all(batch.map(node => this._loadNode(node)));
        }

        // Do NOT force visibility here — let update() decide which nodes to show
        // based on SSE and the parent-replacement logic. Forcing all initial nodes
        // visible was the root cause of low-LOD nodes bleeding through high-LOD ones.
        // Trigger one update pass so the correct nodes become visible immediately.
        const camera = this.scene.activeCamera;
        if (camera) this.update(camera);

        this.stats.visibleNodes = this.activeNodes.size;
    }

    async _loadNode(node) {
        if (this.loadedNodes.has(node.name) || this.loadingNodes.has(node.name)) return;
        if (node.byteSize === 0n || node.numPoints === 0) return;
        if (this.loadingNodes.size >= this.maxConcurrentLoads) return;

        this.loadingNodes.add(node.name);
        this.stats.loadingNodes = this.loadingNodes.size;

        try {
            const buffer = await this._fetchRange("octree.bin", node.byteOffset, node.byteSize);

            const expectedBytes = node.numPoints * this.bytesPerPoint;
            let numPoints = node.numPoints;
            if (buffer.byteLength < expectedBytes) {
                console.warn(`⚠️ Node ${node.name}: got ${buffer.byteLength}B, expected ${expectedBytes}B`);
                numPoints = Math.floor(buffer.byteLength / this.bytesPerPoint);
                if (numPoints === 0) return;
                node.numPoints = numPoints;
            }

            this._createMeshFromBuffer(node, buffer);
            this.stats.loadedNodes++;

            // if (this.stats.loadedNodes <= 30 || this.stats.loadedNodes % 50 === 0) {
            //     console.log(`   ✅ Node ${node.name} (L${node.level}): ${node.numPoints.toLocaleString()} pts`);
            // }
        } catch (err) {
            console.error(`❌ Failed to load node ${node.name}:`, err);
        } finally {
            this.loadingNodes.delete(node.name);
            this.stats.loadingNodes = this.loadingNodes.size;
        }
    }

    update(camera) {
        if (!this.root || !camera) return;

        const anySegmentVisible = this.mainCloudVisible || this.cutHistory.some(e => e.visible);

        // ---- Potree-style iterative LOD selection ----
        //
        // Mirrors Potree's updateVisibility() algorithm from Potree_update_visibility.js:
        //   - Uses a max-priority queue ordered by screenPixelRadius of bounding sphere
        //   - Starts from root (weight = Infinity)
        //   - At each step: pop highest-priority node, decide visibility, push children
        //   - Nodes at level <= 2 are ALWAYS shown regardless of budget (Potree line 182)
        //   - No recursive fallback logic — the queue naturally handles parent/child order
        //   because parents are visited before children (they have higher weight initially)
        //
        // This eliminates all the fragile allChildrenCovered/isFallback logic.

        const engine = this.scene.getEngine();
        const screenHeight = engine.getRenderHeight();
        const fov = camera.fov || 0.8;
        const slope = Math.tan(fov / 2);

        const nodesToShow = new Set();
        const toLoad = [];
        let totalPoints = 0;

        // Simple max-heap using a sorted array (node count is small enough)
        const heap = [{ node: this.root, weight: Infinity }];

        while (heap.length > 0) {
            // Pop max-weight element
            let maxIdx = 0;
            for (let i = 1; i < heap.length; i++) {
                if (heap[i].weight > heap[maxIdx].weight) maxIdx = i;
            }
            const { node } = heap[maxIdx];
            heap.splice(maxIdx, 1);

            if (!node || node.numPoints === 0) continue;

            // Resolve proxy nodes (lazy hierarchy chunks)
            if (node.nodeType === 2) this._ensureHierarchyLoaded(node);

            const level = node.level !== undefined ? node.level : (node.name === 'r' ? 0 : node.name.length - 1);

            // Visibility decision — mirrors Potree logic:
            // always show level <= 2, otherwise check point budget
            const alwaysVisible = level <= 2;
            const withinBudget = totalPoints + node.numPoints <= this.maxVisiblePoints &&
                nodesToShow.size < this.maxVisibleNodes;
            const visible = alwaysVisible || withinBudget;

            if (!visible) continue;

            // Show this node
            nodesToShow.add(node.name);
            totalPoints += node.numPoints;

            // Trigger load if not in memory
            if (!this.loadedNodes.has(node.name)) {
                toLoad.push(node);
            }

            // Push children to heap with their screen-space weight
            for (const child of node.children) {
                if (!child || child.numPoints === 0) continue;

                const bb = child.boundingBox;
                // Bounding sphere center and radius from AABB
                const cx = (bb.min[0] + bb.max[0]) / 2;
                const cy = (bb.min[1] + bb.max[1]) / 2;
                const cz = (bb.min[2] + bb.max[2]) / 2;
                const dx = bb.max[0] - bb.min[0];
                const dy = bb.max[1] - bb.min[1];
                const dz = bb.max[2] - bb.min[2];
                const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

                const camX = camera.position.x;
                const camY = camera.position.y;
                const camZ = camera.position.z;
                const dist = Math.sqrt((camX - cx) ** 2 + (camY - cy) ** 2 + (camZ - cz) ** 2);

                // Camera inside sphere → always refine
                let weight;
                if (dist - radius <= 0) {
                    weight = Infinity;
                } else {
                    // screenPixelRadius: how large this node appears on screen
                    weight = radius * (screenHeight / 2) / (slope * dist);
                }

                // Only push children worth refining (Potree: minimumNodePixelSize)
                if (weight < 1 && level > 2) continue;

                heap.push({ node: child, weight });
            }
        }

        // Trigger background loads (limited by maxConcurrentLoads)
        // Sort by weight descending so closest nodes load first
        for (const node of toLoad) {
            if (!this.loadingNodes.has(node.name)) {
                this._loadNode(node);
            }
        }

        // Apply visibility to all loaded meshes
        for (const [name, mesh] of this.loadedNodes) {
            const shouldShow = nodesToShow.has(name);
            const wasHidden = !mesh.isVisible;
            const targetVisible = shouldShow && anySegmentVisible;

            if (mesh.isVisible !== targetVisible) mesh.isVisible = targetVisible;

            if (targetVisible && wasHidden) {
                this._applyColorModeToMesh(mesh);
                this._applySegmentVisibilityToMesh(mesh);
            }

            if (targetVisible) { this.activeNodes.add(name); } else { this.activeNodes.delete(name); }
        }

        this.stats.visibleNodes = this.activeNodes.size;
        this.stats.totalPointsRendered = totalPoints;
        this.stats.loadingNodes = this.loadingNodes.size;

    }

    /**
     * Sets a fixed point size for all nodes. Called by the UI slider.
     * All points use the same size regardless of LOD level or distance.
     */
    setPointSize(size) {
        this.pointSize = Math.max(1, size);
        for (const mesh of this.loadedNodes.values()) {
            if (mesh.material) mesh.material.pointSize = this.pointSize;
        }
        if (this._featureShaderMat) this._featureShaderMat.setFloat('pointSize', this.pointSize);
    }

    /**
     * setPointSizeMultiplier kept for API compatibility with functions.js.
     * Internally maps to setPointSize using a base of 2px.
     */
    setPointSizeMultiplier(multiplier) {
        this.setPointSize(Math.round(Math.max(0.1, multiplier) * 2));
    }

    /**
     * Updates the display mode of the loader.
     * Called by switchColorMode() in main.js every time the user changes the view,
     * so newly loaded LOD nodes immediately use the correct colors.
     */
    setColorMode(mode) {
        const wasFeature = this.colorMode.startsWith('feature:');
        const isFeature = mode.startsWith('feature:');
        this.colorMode = mode;
        if (!isFeature) this._featureDiscreteFilter = null;
        if (isFeature) {
            this._applyFeatureShaderMode(true, mode.slice(8));
        } else {
            if (wasFeature) this._applyFeatureShaderMode(false, '');
            for (const mesh of this.loadedNodes.values()) {
                if (mesh.isVisible) this._applyColorModeToMesh(mesh);
            }
        }
    }

    /**
     * Load the unified point cloud binary store (.pcbin).
     *
     * .pcbin binary format
     * ─────────────────────
     * Header:
     *   [0-3]          magic: "PCBN"
     *   [4]            version: uint8
     *   [5-7]          reserved: 3 bytes
     *   [8-11]         point_count: uint32  (= max_point_id + 1)
     *   [12-15]        feature_count: uint32 (= F)
     *   [16..+F*32]    feature_names: char[32] × F (null-padded)
     *   [..+F*4]       vmin: float32 × F
     *   [..+F*4]       vmax: float32 × F
     *   Header size = 16 + F*40
     *
     * Records (N records, indexed by point_id):
     *   [0..F*4-1]     features: float32 × F  (NaN = no data)
     *   [F*4]          segment_id: uint8       (0xFF = unassigned)
     *   [F*4+1]        manual_class_id: uint8  (0xFF = unassigned)
     *   [F*4+2]        predicted_class_id: uint8 (0xFF = unassigned)
     *   [F*4+3]        padding: uint8
     *   [F*4+4..+3]    confidence: float32     (NaN = no prediction)
     *   Record size = F*4 + 8
     */
    async loadPcBin(url) {
        console.log("Loading .pcbin store: " + url);
        const response = await fetch(url);
        if (!response.ok) throw new Error("Failed to fetch .pcbin: " + response.status);
        const buffer = await response.arrayBuffer();
        const dv = new DataView(buffer);

        // ── Header ──────────────────────────────────────────────────────────
        const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
        if (magic !== 'PCBN') throw new Error("Invalid .pcbin magic: " + magic);

        const version = dv.getUint8(4);
        const bpf = (version === 2) ? dv.getUint8(5) : 4;  // bytes per feature
        if (version !== 1 && version !== 2)
            throw new Error("Unsupported .pcbin version: " + version);
        const N = dv.getUint32(8, true);   // point_count
        const F = dv.getUint32(12, true);   // feature_count

        let offset = 16;
        const decoder = new TextDecoder('utf-8', { fatal: false });
        const names = [];
        for (let f = 0; f < F; f++) {
            const bytes = new Uint8Array(buffer, offset, 32);
            const end = bytes.indexOf(0);
            const name = decoder.decode(bytes.slice(0, end === -1 ? 32 : end)).trim();
            if (name.length > 0) names.push(name);
            offset += 32;
        }
        const vmin = new Float32Array(buffer.slice(offset, offset + F * 4)); offset += F * 4;
        const vmax = new Float32Array(buffer.slice(offset, offset + F * 4)); offset += F * 4;

        // ── Records ─────────────────────────────────────────────────────────
        const recSize = F * bpf + 8;  // features + annotation tail
        const segIds = new Uint8Array(N);
        const classIds = new Uint8Array(N);
        const predIds = new Uint8Array(N);
        const confidence = new Float32Array(N);

        // Only scan the annotation tail per record (4 reads × N) — skip features.
        // Feature data is looked up on demand from the raw DataView to avoid
        // allocating N*F*4 bytes (e.g. 2+ GB for large datasets).
        const annotOff = F * bpf;  // byte offset within a record to annotation tail
        for (let pid = 0; pid < N; pid++) {
            const recOff = offset + pid * recSize;
            segIds[pid] = dv.getUint8(recOff + annotOff);
            classIds[pid] = dv.getUint8(recOff + annotOff + 1);
            predIds[pid] = dv.getUint8(recOff + annotOff + 2);
            confidence[pid] = dv.getFloat32(recOff + annotOff + 4, true);
        }

        // Keep raw DataView for on-demand feature lookup (no N*F pre-allocation).
        const dvFloat = new Float32Array(buffer, 0);
        const dvUint8 = new Uint8Array(buffer, 0);
        this.featureBin = { N, F, names, vmin, vmax, bpf, dv, dvFloat, dvUint8, offset, recSize };
        this.pcbinAnnotations = { segIds, classIds, predIds, confidence };

        console.log(".pcbin loaded: F=" + F + ", N=" + N);
        return names;
    }

    getFeatureList() {
        return this.featureBin ? [...this.featureBin.names] : [];
    }

    _blendClassChannel(original, classChannel) {
        const s = this.classColorBlendStrength;
        return original * ((1.0 - s) + s * classChannel);
    }

    _writeBlendedClassColor(colors, pointIndex, originalColors, classColors) {
        const o = pointIndex * 4;
        const s = this.classColorBlendStrength;
        const inv = 1.0 - s;
        colors[o]     = classColors[o]     * s + originalColors[o]     * inv;
        colors[o + 1] = classColors[o + 1] * s + originalColors[o + 1] * inv;
        colors[o + 2] = classColors[o + 2] * s + originalColors[o + 2] * inv;
        colors[o + 3] = 1.0;
    }

    _writeBlendedClassColorFromRGB(colors, pointIndex, originalColors, r, g, b) {
        const o = pointIndex * 4;
        const s = this.classColorBlendStrength;
        const inv = 1.0 - s;
        colors[o]     = r * s + originalColors[o]     * inv;
        colors[o + 1] = g * s + originalColors[o + 1] * inv;
        colors[o + 2] = b * s + originalColors[o + 2] * inv;
        colors[o + 3] = 1.0;
    }

    /**
     * Applies the current colorMode to the vertex colors of a single mesh.
     * Called by setColorMode() and by update() when a node becomes visible.
     */
    _applyColorModeToMesh(mesh) {
        const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
        if (!colors) return;

        const originalColors = mesh.metadata?.originalColors;
        const classIds = mesh.metadata?.classIds;
        const classColors = mesh.metadata?.classColors;
        const positions = this.selectionHistory.length > 0
            ? mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind)
            : null;
        const numPoints = colors.length / 4;
        let changed = false;

        // Feature bin mode (e.g. "feature:planarity_0_8")
        const isFeatureMode = this.colorMode.startsWith('feature:');
        const featureName = isFeatureMode ? this.colorMode.slice(8) : null;
        const featureBin = isFeatureMode ? this.featureBin : null;
        const featureIdx = (featureBin && featureName !== null) ? featureBin.names.indexOf(featureName) : -1;
        const isPredictionFeature = featureName !== null && featureName.toLowerCase() === 'prediction';
        const pointIds = mesh.metadata?.pointIds;

        // Cache LUT + range boundaries outside the loop
        const lut = Potree2Loader._viridisLUT;
        const fmin = (featureBin && featureIdx >= 0)
            ? (this._featureRangeMin !== null ? this._featureRangeMin : featureBin.vmin[featureIdx]) : 0;
        const fmax = (featureBin && featureIdx >= 0)
            ? (this._featureRangeMax !== null ? this._featureRangeMax : featureBin.vmax[featureIdx]) : 1;

        const tmpVecColorMode = new BABYLON.Vector3();

        for (let i = 0; i < numPoints; i++) {
            // 1. Apply base color
            if (isFeatureMode && featureBin && featureIdx >= 0 && pointIds) {
                const pid = pointIds[i];
                let r = FEATURE_MISSING_COLOR, g = FEATURE_MISSING_COLOR, b = FEATURE_MISSING_COLOR;
                if (pid >= 0 && pid < featureBin.N) {
                    // On-demand feature lookup from raw Uint8Array / Float32Array.
                    const recOff = featureBin.offset + pid * featureBin.recSize;
                    let val;
                    if (featureBin.bpf === 1) {
                        const q = featureBin.dvUint8[recOff + featureIdx];
                        val = (q === 255) ? NaN
                            : featureBin.vmin[featureIdx] + (q / 254) * (featureBin.vmax[featureIdx] - featureBin.vmin[featureIdx]);
                    } else {
                        val = featureBin.dvFloat[(recOff + featureIdx * 4) / 4];
                    }
                    if (Number.isFinite(val)) {
                        let t;
                        if (isPredictionFeature) {
                            // Snap to integer class, normalize by fmax for palette consistency
                            const iv = Math.round(val);
                            t = fmax > 0 ? Math.max(0, Math.min(1, iv / fmax)) : 0.5;
                        } else {
                            t = (fmax > fmin) ? (val - fmin) / (fmax - fmin) : 0.5;
                        }
                        const lutIdx = Math.round(Math.max(0, Math.min(1, t)) * 255) * 3;
                        r = lut[lutIdx]; g = lut[lutIdx + 1]; b = lut[lutIdx + 2];
                    }
                }
                colors[i * 4] = r;
                colors[i * 4 + 1] = g;
                colors[i * 4 + 2] = b;
                colors[i * 4 + 3] = 1.0;
            } else {
                const hasClass = classIds && classIds[i] > 0 && classColors;
                if (this.colorMode === "classification" && hasClass) {
                    this._writeBlendedClassColor(colors, i, originalColors, classColors);
                } else if (originalColors) {
                    colors[i * 4] = originalColors[i * 4];
                    colors[i * 4 + 1] = originalColors[i * 4 + 1];
                    colors[i * 4 + 2] = originalColors[i * 4 + 2];
                    colors[i * 4 + 3] = originalColors[i * 4 + 3];
                }
            }

            // 2. Re-apply selection highlight on top (always overrides)
            if (positions && this.selectionHistory.length > 0 && !isNaN(positions[i * 3])) {
                tmpVecColorMode.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                const inHistory = this._isPointInSelectionHistory(tmpVecColorMode);
                const shouldHighlight = this.selectionInverted ? !inHistory : inHistory;
                if (shouldHighlight) {
                    colors[i * 4] = 1.0;
                    colors[i * 4 + 1] = 0.0;
                    colors[i * 4 + 2] = 0.0;
                    colors[i * 4 + 3] = 1.0;
                }
            }

            changed = true;
        }
        if (changed) mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
    }

    getStats() {
        return { ...this.stats, loadedNodes: this.loadedNodes.size };
    }

    getRoot() {
        return this.rootTransform;
    }

    dispose() {
        if (this._cameraForObserver && this._cameraViewObserver) {
            this._cameraForObserver.onViewMatrixChangedObservable.remove(this._cameraViewObserver);
        }
        this._cameraForObserver = null;
        this._cameraViewObserver = null;

        if (this._cleanupIntervalId !== null) {
            clearInterval(this._cleanupIntervalId);
            this._cleanupIntervalId = null;
        }

        for (const mesh of this.loadedNodes.values()) {
            if (mesh.material) mesh.material.dispose();
            mesh.dispose();
        }
        this.loadedNodes.clear();
        this.activeNodes.clear();
        this.loadingNodes.clear();
        this.rootTransform.dispose();
        this.hierarchyBuffer = null;
    }

    // ========== ATTRIBUTE PARSING ==========

    _parseAttributes() {
        this.attributes = [];
        let byteOffset = 0;

        for (const attr of this.metadata.attributes) {
            this.attributes.push({
                name: attr.name,
                type: attr.type,
                size: attr.size,
                numElements: attr.numElements,
                elementSize: attr.elementSize,
                byteOffset: byteOffset,
                min: attr.min,
                max: attr.max,
                scale: attr.scale,
                offset: attr.offset
            });
            byteOffset += attr.size;
        }

        this.bytesPerPoint = byteOffset;
        // console.log("📋 Attributes:", this.attributes.map(a => `${a.name}(${a.type}, ${a.size}B)`).join(", "));
        // console.log("   Bytes per point:", this.bytesPerPoint);
    }

    // ========== HIERARCHY PARSING ==========

    _parseHierarchyChunk(rootNode) {
        const buffer = this.hierarchyBuffer;
        const chunkStart = Number(rootNode.hierarchyByteOffset);
        const chunkEnd = chunkStart + Number(rootNode.hierarchyByteSize);

        if (chunkEnd > buffer.byteLength) {
            console.error(`Hierarchy chunk exceeds buffer: ${chunkStart}-${chunkEnd} > ${buffer.byteLength}`);
            return;
        }

        const view = new DataView(buffer, chunkStart, chunkEnd - chunkStart);
        const bytesPerNode = 22;
        const numNodes = Math.floor((chunkEnd - chunkStart) / bytesPerNode);

        const nodes = new Array(numNodes);
        nodes[0] = rootNode;
        let nodePos = 1;

        for (let i = 0; i < numNodes; i++) {
            const current = nodes[i];
            if (!current) break;

            const offset = i * bytesPerNode;
            const type = view.getUint8(offset + 0);
            const childMask = view.getUint8(offset + 1);
            const numPoints = view.getUint32(offset + 2, true);
            const byteOffset = view.getBigInt64(offset + 6, true);
            const byteSize = view.getBigInt64(offset + 14, true);

            if (current.nodeType === 2) {
                current.byteOffset = byteOffset;
                current.byteSize = byteSize;
                current.numPoints = numPoints;
            } else if (type === 2) {
                current.hierarchyByteOffset = byteOffset;
                current.hierarchyByteSize = byteSize;
                current.numPoints = numPoints;
            } else {
                current.byteOffset = byteOffset;
                current.byteSize = byteSize;
                current.numPoints = numPoints;
            }

            current.nodeType = type;
            current.childMask = childMask;

            if (current.nodeType === 2) continue;

            for (let childIdx = 0; childIdx < 8; childIdx++) {
                if (!((1 << childIdx) & childMask)) continue;

                const childName = current.name + childIdx;
                const childBB = createChildAABB(current.boundingBox, childIdx);
                const child = new Potree2Node(childName, childBB);
                child.spacing = current.spacing / 2;

                current.children[childIdx] = child;
                nodes[nodePos] = child;
                nodePos++;
            }
        }

        rootNode.nodeType = 0;
    }

    _ensureHierarchyLoaded(node) {
        if (node.nodeType !== 2) return;
        const bufLen = this.hierarchyBuffer ? this.hierarchyBuffer.byteLength : 0;
        const chunkStart = Number(node.hierarchyByteOffset);
        const chunkEnd = chunkStart + Number(node.hierarchyByteSize);
        if (chunkEnd > bufLen) {
            console.warn(`⚠️ Proxy node ${node.name}: chunk [${chunkStart}-${chunkEnd}] exceeds hierarchy buffer (${bufLen} bytes). Node will stay as proxy.`);
            return;
        }
        this._parseHierarchyChunk(node);
    }

    _collectNodes(node) {
        const result = [node];
        for (const child of node.children) {
            if (child) result.push(...this._collectNodes(child));
        }
        return result;
    }

    // ========== LOD TRAVERSAL ==========
    // Replaced by iterative heap-based algorithm inside update().
    // See update() for the Potree-faithful implementation.

    _getLocalCenter(node) {
        const bb = node.boundingBox;
        return new BABYLON.Vector3(
            (bb.min[0] + bb.max[0]) / 2,
            (bb.min[1] + bb.max[1]) / 2,
            (bb.min[2] + bb.max[2]) / 2
        );
    }

    // ========== ATTRIBUTE COLORIZATION UTILITIES ==========

    /**
     * Reads a scalar value from the ArrayBuffer based on the attribute type.
     */
    _readAttrScalar(view, byteOffset, attr) {
        const t = attr.type;
        if (t === 'int8') return view.getInt8(byteOffset);
        if (t === 'uint8') return view.getUint8(byteOffset);
        if (t === 'int16') return view.getInt16(byteOffset, true);
        if (t === 'uint16') return view.getUint16(byteOffset, true);
        if (t === 'int32') return view.getInt32(byteOffset, true);
        if (t === 'uint32') return view.getUint32(byteOffset, true);
        if (t === 'float') return view.getFloat32(byteOffset, true);
        if (t === 'double') return view.getFloat64(byteOffset, true);
        // int64/uint64: fallback to int32 (only low 4 bytes)
        return view.getInt32(byteOffset, true);
    }

    /**
     * Viridis colormap (clamps to [0,1]). Used for continuous attributes (e.g. intensity).
     */
    static _buildViridisLUT() {
        const stops = [
            [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.163, 0.471, 0.558],
            [0.134, 0.659, 0.518], [0.478, 0.821, 0.318], [0.993, 0.906, 0.144]
        ];
        const lut = new Float32Array(256 * 3);
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const scaled = t * (stops.length - 1);
            const lo = Math.floor(scaled);
            const hi = Math.min(lo + 1, stops.length - 1);
            const f = scaled - lo;
            lut[i * 3] = stops[lo][0] + (stops[hi][0] - stops[lo][0]) * f;
            lut[i * 3 + 1] = stops[lo][1] + (stops[hi][1] - stops[lo][1]) * f;
            lut[i * 3 + 2] = stops[lo][2] + (stops[hi][2] - stops[lo][2]) * f;
        }
        return lut;
    }

    _colormapViridis(t) {
        const idx = Math.round(Math.max(0, Math.min(1, t)) * 255) * 3;
        return [Potree2Loader._viridisLUT[idx],
        Potree2Loader._viridisLUT[idx + 1],
        Potree2Loader._viridisLUT[idx + 2]];
    }

    /**
     * Discrete cyclic palette for integer attributes (e.g. classification, return_number).
     * Returns [r, g, b] in [0,1].
     */
    _colormapDiscrete(intValue) {
        const palette = [
            [0.22, 0.62, 0.85],  // blue
            [0.95, 0.45, 0.10],  // orange
            [0.17, 0.72, 0.44],  // green
            [0.80, 0.22, 0.33],  // red
            [0.58, 0.40, 0.74],  // purple
            [0.99, 0.75, 0.18],  // yellow
            [0.40, 0.76, 0.65],  // teal
            [0.88, 0.53, 0.79],  // pink
            [0.60, 0.60, 0.60],  // gray
            [0.99, 0.55, 0.38],  // salmon
        ];
        const idx = Math.abs(intValue) % palette.length;
        return palette[idx];
    }



    // ========== FEATURE SHADER ==========

    static _featureVertexShader = `
        precision highp float;
        attribute vec3 position;
        attribute float featureValue;
        uniform mat4 worldViewProjection;
        uniform float pointSize;
        varying float vFeature;
        void main() {
            vFeature = featureValue;
            gl_Position = worldViewProjection * vec4(position, 1.0);
            gl_PointSize = pointSize;
        }
    `;

    static _featureFragmentShader = `
        precision highp float;
        uniform float fmin;
        uniform float fmax;
        uniform int   colormap;
        uniform int   predictionDiscrete;
        uniform int   discreteFilterEnabled;
        uniform float discreteFilterValue;
        varying float vFeature;

        // 0: Blue > Green > Yellow > Red (default, CloudCompare-style)
        vec3 cm_bgyr(float t) {
            t = clamp(t, 0.0, 1.0) * 3.0;
            float i = floor(t); float f = t - i;
            vec3 c0 = vec3(0.0, 0.0, 1.0);
            vec3 c1 = vec3(0.0, 1.0, 0.0);
            vec3 c2 = vec3(1.0, 1.0, 0.0);
            vec3 c3 = vec3(1.0, 0.0, 0.0);
            vec3 a, b;
            if      (i < 1.0) { a = c0; b = c1; }
            else if (i < 2.0) { a = c1; b = c2; }
            else               { a = c2; b = c3; }
            return mix(a, b, f);
        }

        // 1: Viridis
        vec3 cm_viridis(float t) {
            const vec3 c0 = vec3(0.267, 0.005, 0.329);
            const vec3 c1 = vec3(0.283, 0.141, 0.458);
            const vec3 c2 = vec3(0.163, 0.471, 0.558);
            const vec3 c3 = vec3(0.134, 0.659, 0.518);
            const vec3 c4 = vec3(0.478, 0.821, 0.318);
            const vec3 c5 = vec3(0.993, 0.906, 0.144);
            t = clamp(t, 0.0, 1.0) * 5.0;
            float i = floor(t); float f = t - i;
            vec3 a, b;
            if      (i < 1.0) { a = c0; b = c1; }
            else if (i < 2.0) { a = c1; b = c2; }
            else if (i < 3.0) { a = c2; b = c3; }
            else if (i < 4.0) { a = c3; b = c4; }
            else               { a = c4; b = c5; }
            return mix(a, b, f);
        }

        // 2: Jet
        vec3 cm_jet(float t) {
            t = clamp(t, 0.0, 1.0);
            float r = clamp(1.5 - abs(4.0*t - 3.0), 0.0, 1.0);
            float g = clamp(1.5 - abs(4.0*t - 2.0), 0.0, 1.0);
            float b = clamp(1.5 - abs(4.0*t - 1.0), 0.0, 1.0);
            return vec3(r, g, b);
        }

        // 3: Diverging (Blue > White > Red)
        vec3 cm_diverging(float t) {
            t = clamp(t, 0.0, 1.0);
            vec3 blue  = vec3(0.129, 0.400, 0.675);
            vec3 white = vec3(0.969, 0.969, 0.969);
            vec3 red   = vec3(0.839, 0.376, 0.302);
            if (t < 0.5) return mix(blue, white, t * 2.0);
            else          return mix(white, red, (t - 0.5) * 2.0);
        }

        // 4: Grayscale
        vec3 cm_gray(float t) { return vec3(clamp(t, 0.0, 1.0)); }

        // 5: Hot (Black > Red > Yellow > White)
        vec3 cm_hot(float t) {
            t = clamp(t, 0.0, 1.0) * 3.0;
            return vec3(clamp(t, 0.0, 1.0),
                        clamp(t - 1.0, 0.0, 1.0),
                        clamp(t - 2.0, 0.0, 1.0));
        }

        void main() {
            if (vFeature < ${FEATURE_MISSING_SENTINEL / 10.0}) {
                gl_FragColor = vec4(${FEATURE_MISSING_COLOR}, ${FEATURE_MISSING_COLOR}, ${FEATURE_MISSING_COLOR}, 1.0);
                return;
            }

            if (discreteFilterEnabled == 1) {
                float iv = floor(vFeature + 0.5);
                if (abs(iv - discreteFilterValue) > 0.1) discard;
            }

            // For prediction (integer class IDs), snap to nearest integer then normalize
            // by fmax so each class maps to a fixed position in the selected colormap.
            float t;
            if (predictionDiscrete == 1) {
                float iv = floor(vFeature + 0.5);
                t = (fmax > 0.0) ? clamp(iv / fmax, 0.0, 1.0) : 0.5;
            } else {
                float range = fmax - fmin;
                t = (range > 0.0) ? (vFeature - fmin) / range : 0.5;
            }

            vec3 col;
            if      (colormap == 1) col = cm_viridis(t);
            else if (colormap == 2) col = cm_jet(t);
            else if (colormap == 3) col = cm_diverging(t);
            else if (colormap == 4) col = cm_gray(t);
            else if (colormap == 5) col = cm_hot(t);
            else                    col = cm_bgyr(t);
            gl_FragColor = vec4(col, 1.0);
        }
    `;

    _getOrCreateFeatureShaderMaterial() {
        if (this._featureShaderMat) return this._featureShaderMat;
        BABYLON.Effect.ShadersStore['potreeFeatureVertexShader'] = Potree2Loader._featureVertexShader;
        BABYLON.Effect.ShadersStore['potreeFeatureFragmentShader'] = Potree2Loader._featureFragmentShader;
        const mat = new BABYLON.ShaderMaterial('potreeFeatureMat', this.scene, 'potreeFeature', {
            attributes: ['position', 'featureValue'],
            uniforms: ['worldViewProjection', 'pointSize', 'fmin', 'fmax', 'colormap', 'predictionDiscrete', 'discreteFilterEnabled', 'discreteFilterValue'],
        });
        mat.pointsCloud = true;
        mat.disableLighting = true;
        mat.setFloat('pointSize', this.pointSize);
        mat.setFloat('fmin', 0.0);
        mat.setFloat('fmax', 1.0);
        mat.setInt('colormap', this._colormapId);
        mat.setInt('predictionDiscrete', 0);
        mat.setInt('discreteFilterEnabled', 0);
        mat.setFloat('discreteFilterValue', 0.0);
        mat.transparencyMode = BABYLON.Material.MATERIAL_ALPHATEST;
        mat.alphaCutOff = 0.1;
        this._featureShaderMat = mat;
        return mat;
    }

    _uploadFeatureAttribute(mesh) {
        if (!this.featureBin) return;
        const featureName = this.colorMode.slice(8);
        const featureIdx = this.featureBin.names.indexOf(featureName);
        if (featureIdx < 0) return;
        const pointIds = mesh.metadata?.pointIds;
        if (!pointIds) return;
        const numPoints = pointIds.length;
        const featureValues = new Float32Array(numPoints);
        const { bpf, dvFloat, dvUint8, offset, recSize, F, N, vmin, vmax } = this.featureBin;
        for (let i = 0; i < numPoints; i++) {
            const pid = pointIds[i];
            if (pid < 0 || pid >= N) { featureValues[i] = FEATURE_MISSING_SENTINEL; continue; }
            const recOff = offset + pid * recSize;
            if (bpf === 1) {
                const q = dvUint8[recOff + featureIdx];
                featureValues[i] = (q === 255) ? FEATURE_MISSING_SENTINEL
                    : vmin[featureIdx] + (q / 254) * (vmax[featureIdx] - vmin[featureIdx]);
            } else {
                const val = dvFloat[(recOff + featureIdx * 4) / 4];
                featureValues[i] = Number.isFinite(val) ? val : FEATURE_MISSING_SENTINEL;
            }
        }
        const buf = new BABYLON.VertexBuffer(
            this.scene.getEngine(), featureValues, 'featureValue', false, false, 1
        );
        mesh.setVerticesBuffer(buf);
    }

    _applyFeatureShaderMode(enable, featureName) {
        const shaderMat = this._getOrCreateFeatureShaderMaterial();
        const isPredictionFeature = typeof featureName === 'string' && featureName.toLowerCase() === 'prediction';
        if (this.featureBin && featureName) {
            const featureIdx = this.featureBin.names.indexOf(featureName);
            if (featureIdx >= 0) {
                shaderMat.setFloat('fmin', this._featureRangeMin !== null ? this._featureRangeMin : this.featureBin.vmin[featureIdx]);
                shaderMat.setFloat('fmax', this._featureRangeMax !== null ? this._featureRangeMax : this.featureBin.vmax[featureIdx]);
            }
        }

        const discreteForFeature = this._featureDiscreteFilter && this._featureDiscreteFilter.featureName === featureName;
        if (discreteForFeature && this._featureDiscreteFilter.value !== null && this._featureDiscreteFilter.value !== undefined) {
            shaderMat.setInt('discreteFilterEnabled', 1);
            shaderMat.setFloat('discreteFilterValue', this._featureDiscreteFilter.value);
        } else {
            shaderMat.setInt('discreteFilterEnabled', 0);
        }

        shaderMat.setInt('colormap', this._colormapId);
        shaderMat.setInt('predictionDiscrete', isPredictionFeature ? 1 : 0);
        shaderMat.setFloat('pointSize', this.pointSize);
        for (const mesh of this.loadedNodes.values()) {
            if (enable) {
                if (!mesh.metadata._origMaterial) mesh.metadata._origMaterial = mesh.material;
                this._uploadFeatureAttribute(mesh);
                mesh.material = shaderMat;
            } else {
                if (mesh.metadata._origMaterial) {
                    mesh.material = mesh.metadata._origMaterial;
                    mesh.metadata._origMaterial = null;
                }
            }
        }
    }

    setFeatureRange(min, max) {
        this._featureRangeMin = min;
        this._featureRangeMax = max;
        if (this._featureShaderMat) {
            this._featureShaderMat.setFloat('fmin', min);
            this._featureShaderMat.setFloat('fmax', max);
        }
    }

    setFeatureDiscreteSelection(featureName, valueOrNull) {
        this._featureDiscreteFilter = {
            featureName,
            value: (valueOrNull === null || valueOrNull === undefined) ? null : Number(valueOrNull)
        };

        if (this.colorMode.startsWith('feature:')) {
            this._applyFeatureShaderMode(true, this.colorMode.slice(8));
        }
    }

    resetFeatureRange() {
        this._featureRangeMin = null;
        this._featureRangeMax = null;
        if (this._featureShaderMat && this.featureBin && this.colorMode.startsWith('feature:')) {
            const featureIdx = this.featureBin.names.indexOf(this.colorMode.slice(8));
            if (featureIdx >= 0) {
                this._featureShaderMat.setFloat('fmin', this.featureBin.vmin[featureIdx]);
                this._featureShaderMat.setFloat('fmax', this.featureBin.vmax[featureIdx]);
            }
        }
    }

    setColormap(id) {
        this._colormapId = id;
        if (this._featureShaderMat) this._featureShaderMat.setInt('colormap', id);
    }

    // ========== MESH CREATION ==========

    _createMeshFromBuffer(node, buffer) {
        const view = new DataView(buffer);
        const numPoints = node.numPoints;
        const positions = new Float32Array(numPoints * 3);
        const colors = new Float32Array(numPoints * 4);

        const scale = this.metadata.scale;
        const metaOffset = this.metadata.offset;
        const bbMin = this.metadata.boundingBox.min;

        let posAttr = null;
        let rgbAttr = null;
        let pointIdAttr = null;

        for (const attr of this.attributes) {
            if (attr.name === "position") posAttr = attr;
            if (attr.name === "rgb") rgbAttr = attr;
            if (attr.name.toLowerCase() === "point_id" || attr.name.toLowerCase() === "pointid") {
                pointIdAttr = attr;
            }
        }

        if (!posAttr) {
            console.error("No position attribute found!");
            return;
        }

        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        const pointIds = new Int32Array(numPoints);
        pointIds.fill(-1);

        for (let j = 0; j < numPoints; j++) {
            const pointOffset = j * this.bytesPerPoint;

            if (pointOffset + posAttr.byteOffset + 12 > buffer.byteLength) break;

            const rawX = view.getInt32(pointOffset + posAttr.byteOffset + 0, true);
            const rawY = view.getInt32(pointOffset + posAttr.byteOffset + 4, true);
            const rawZ = view.getInt32(pointOffset + posAttr.byteOffset + 8, true);

            const x = (rawX * scale[0]) + metaOffset[0] - bbMin[0];
            const y = (rawY * scale[1]) + metaOffset[1] - bbMin[1];
            const z = (rawZ * scale[2]) + metaOffset[2] - bbMin[2];

            positions[3 * j + 0] = x;
            positions[3 * j + 1] = y;
            positions[3 * j + 2] = z;

            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);

            // Decode RGB (uint16 × 3)
            if (rgbAttr && pointOffset + rgbAttr.byteOffset + 6 <= buffer.byteLength) {
                const rRaw = view.getUint16(pointOffset + rgbAttr.byteOffset + 0, true);
                const gRaw = view.getUint16(pointOffset + rgbAttr.byteOffset + 2, true);
                const bRaw = view.getUint16(pointOffset + rgbAttr.byteOffset + 4, true);

                const r = rRaw > 255 ? rRaw / 256 : rRaw;
                const g = gRaw > 255 ? gRaw / 256 : gRaw;
                const b = bRaw > 255 ? bRaw / 256 : bRaw;

                colors[4 * j + 0] = r / 255.0;
                colors[4 * j + 1] = g / 255.0;
                colors[4 * j + 2] = b / 255.0;
                colors[4 * j + 3] = 1.0;
            } else {
                colors[4 * j + 0] = 1.0;
                colors[4 * j + 1] = 1.0;
                colors[4 * j + 2] = 1.0;
                colors[4 * j + 3] = 1.0;
            }

            // Decode POINT_ID
            if (pointIdAttr && pointOffset + pointIdAttr.byteOffset + 4 <= buffer.byteLength) {
                // Read as Int32/Uint32 (usually 4 bytes). Using getInt32.
                pointIds[j] = view.getInt32(pointOffset + pointIdAttr.byteOffset, true);
            }
        }


        // Save original positions so they can be hidden (by assigning NaN)
        // and restored later, bypassing any bugs/limitations of BabylonJS point
        // cloud materials regarding alpha compositing.
        const originalPositions = new Float32Array(positions);

        // Save originalColors HERE — after RGB decode but BEFORE applying
        // selection or classification. This ensures originalColors always contains
        // only the real point cloud colors.
        const originalColors = new Float32Array(colors);

        const tmpVec = new BABYLON.Vector3();

        // Apply persistent selection highlight AFTER saving originalColors
        if (this.selectionHistory.length > 0) {
            for (let j = 0; j < numPoints; j++) {
                tmpVec.set(positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]);
                const inHistory = this._isPointInSelectionHistory(tmpVec);
                // If inverted: highlight points OUTSIDE the selection regions
                const shouldHighlight = this.selectionInverted ? !inHistory : inHistory;
                if (shouldHighlight) {
                    colors[4 * j + 0] = 1.0;
                    colors[4 * j + 1] = 0.0;
                    colors[4 * j + 2] = 0.0;
                }
            }
        }

        // // Debug bounds for first few nodes
        // if (this.stats.loadedNodes < 5) {
        //     console.log(`   📐 Node ${node.name} bounds: X[${minX.toFixed(2)}, ${maxX.toFixed(2)}] Y[${minY.toFixed(2)}, ${maxY.toFixed(2)}] Z[${minZ.toFixed(2)}, ${maxZ.toFixed(2)}]`);
        // }

        // Apply existing classifications using the same 2D-projection approach
        // as selectionHistory — works correctly at every LOD level.
        const classIds = new Int32Array(numPoints);        // 0 = unclassified
        const classColors = new Float32Array(numPoints * 4);  // RGBA per point

        if (this.classificationHistory.length > 0) {
            for (let j = 0; j < numPoints; j++) {
                tmpVec.set(positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]);
                const cls = this._getPointClassification(tmpVec);
                if (cls) {
                    classIds[j] = cls.classId;
                    classColors[j * 4] = cls.r;
                    classColors[j * 4 + 1] = cls.g;
                    classColors[j * 4 + 2] = cls.b;
                    classColors[j * 4 + 3] = 1.0;
                    // Paint vertex colors only if the active mode requires it.
                    // In "color" mode vertices keep their originalColors — no flash
                    // of class colors when a LOD node loads new detail.
                    if (this.colorMode === "classification") {
                        this._writeBlendedClassColorFromRGB(colors, j, originalColors, cls.r, cls.g, cls.b);
                    }
                }
            }
        }

        // Apply existing cut segments
        const segmentIds = new Int32Array(numPoints); // 0 = main cloud
        if (this.cutHistory.length > 0 || !this.mainCloudVisible) {
            const chronologicalCuts = this._getChronologicalCuts();
            for (let j = 0; j < numPoints; j++) {
                const pid = pointIds[j];
                let resolvedSegId = -1; // -1 = not resolved yet

                // 1. Canonical map lookup (source of truth for points seen at cut time).
                //    Preserved across LOD unload/reload cycles — prevents the geometry-
                //    based fallback from re-assigning overlap points to a newer segment.
                const canonical = this._readPointSegment(pid);
                if (canonical !== null) resolvedSegId = canonical;

                if (resolvedSegId !== -1) {
                    segmentIds[j] = resolvedSegId;
                    let shouldHide;
                    if (resolvedSegId === 0) {
                        shouldHide = !this.mainCloudVisible;
                    } else if (resolvedSegId === this._deletedSegmentId) {
                        shouldHide = true;
                    } else {
                        const entry = this.cutHistory.find(e => e.segmentId === resolvedSegId);
                        shouldHide = entry ? !entry.visible : false;
                    }
                    if (shouldHide) {
                        positions[3 * j] = NaN;
                        positions[3 * j + 1] = NaN;
                        positions[3 * j + 2] = NaN;
                    }
                } else {
                    // 2. Geometry fallback for points not loaded at cut time: replay the
                    //    whole history in order, honouring what was visible at each step.
                    //    The outcome is frozen into the canonical map so this point is
                    //    never re-derived again: replaying the 2D regions on a later
                    //    reload can pick a different segment on overlaps, which is what
                    //    made hidden segments reappear in blocks while navigating.
                    //    Only a positive match is frozen — points that match no region
                    //    stay SEG_UNRESOLVED so a future cut can still claim them.
                    tmpVec.set(positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]);
                    const seg = this._resolvePointSegment(tmpVec, chronologicalCuts);
                    if (seg !== null) {
                        segmentIds[j] = seg.segmentId;
                        this._writePointSegment(pid, seg.segmentId);
                        if (!seg.visible) {
                            positions[3 * j] = NaN;
                            positions[3 * j + 1] = NaN;
                            positions[3 * j + 2] = NaN;
                        }
                    } else {
                        segmentIds[j] = 0;
                        if (!this.mainCloudVisible) {
                            positions[3 * j] = NaN;
                            positions[3 * j + 1] = NaN;
                            positions[3 * j + 2] = NaN;
                        }
                    }
                }
            }
        }

        // Create BabylonJS mesh
        const mesh = new BABYLON.Mesh(`potree2_${node.name}`, this.scene);
        const vertexData = new BABYLON.VertexData();
        vertexData.positions = positions;
        vertexData.colors = colors;
        // updatable=true is REQUIRED to update visibility and colors dynamically
        vertexData.applyToMesh(mesh, true);

        const mat = new BABYLON.StandardMaterial(`mat_p2_${node.name}`, this.scene);
        mat.pointsCloud = true;
        mat.pointSize = this.pointSize;
        mat.disableLighting = true;
        mat.emissiveColor = new BABYLON.Color3(1, 1, 1);
        mat.useVertexAlpha = true; // needed for alpha=0 to hide cut segment points
        // Use ALPHA TEST to discard invisible points efficiently and avoid depth sorting issues
        mat.transparencyMode = BABYLON.Material.MATERIAL_ALPHATEST;
        mat.alphaCutOff = 0.1;

        mesh.material = mat;
        mesh.hasAlpha = true;

        mesh.parent = this.rootTransform;
        mesh.isVisible = false;
        mesh.isPickable = true;

        mesh.metadata = {
            nodeInfo: { name: node.name, level: node.level, numPoints: node.numPoints },
            originalPositions,
            originalColors,
            classIds,
            classColors,
            segmentIds,
            pointIds,
            potree2Node: true,
            // Used by update() to compute per-node projected point size (spacing-based LOD).
            nodeSpacing: node.spacing,
            nodeBoundingBox: node.boundingBox
        };

        this.loadedNodes.set(node.name, mesh);
        node.loaded = true;
        node.mesh = mesh;

        if (this.colorMode.startsWith('feature:') && this.featureBin) {
            mesh.metadata._origMaterial = mesh.material;
            this._uploadFeatureAttribute(mesh);
            mesh.material = this._getOrCreateFeatureShaderMaterial();
        }
    }

    // ========== CLEANUP ==========

    cleanup(keepCount = 200) {
        if (this.loadedNodes.size <= keepCount) return;

        const inactiveNodes = [...this.loadedNodes.keys()]
            .filter(name => !this.activeNodes.has(name));

        const toRemove = inactiveNodes.slice(0, Math.max(0, inactiveNodes.length - keepCount));

        for (const name of toRemove) {
            const mesh = this.loadedNodes.get(name);
            if (mesh.material) mesh.material.dispose();
            mesh.dispose();
            this.loadedNodes.delete(name);
            this.stats.loadedNodes--;
        }

        if (toRemove.length > 0) console.log(`🧹 Cleaned up ${toRemove.length} nodes`);
    }

    // ========== SELECTION ==========

    /**
     * Checks if a 3D point falls into any region in selectionHistory
     * AND is not excluded by deselectionHistory.
     */
    _isPointInSelectionHistory(localVector) {
        return this._matchesSelectionEntry(
            localVector,
            this.selectionHistory,
            this.deselectionHistory,
            this.selectionInverted
        );
    }

    _matchesSelectionEntry(localVector, selections, deselections = [], inverted = false) {
        const worldMatrix = this.rootTransform.getWorldMatrix();
        this._tmpProj = this._tmpProj || new BABYLON.Vector3();

        let selected = false;
        for (const sel of selections) {
            BABYLON.Vector3.ProjectToRef(localVector, worldMatrix, sel.transformMatrix, sel.viewport, this._tmpProj);
            if (sel.type === "rect") {
                if (this._tmpProj.x >= sel.area.x && this._tmpProj.x <= sel.area.x + sel.area.width &&
                    this._tmpProj.y >= sel.area.y && this._tmpProj.y <= sel.area.y + sel.area.height) {
                    selected = true;
                    break;
                }
            } else if (sel.type === "lasso" || sel.type === "polygon") {
                if (this._isPointInPoly(sel.area, [this._tmpProj.x, this._tmpProj.y])) {
                    selected = true;
                    break;
                }
            }
        }

        if (selected) {
            for (const dsel of deselections) {
                BABYLON.Vector3.ProjectToRef(localVector, worldMatrix, dsel.transformMatrix, dsel.viewport, this._tmpProj);
                if (dsel.type === "rect") {
                    if (this._tmpProj.x >= dsel.area.x && this._tmpProj.x <= dsel.area.x + dsel.area.width &&
                        this._tmpProj.y >= dsel.area.y && this._tmpProj.y <= dsel.area.y + dsel.area.height) {
                        selected = false;
                        break;
                    }
                } else if (dsel.type === "lasso" || dsel.type === "polygon") {
                    if (this._isPointInPoly(dsel.area, [this._tmpProj.x, this._tmpProj.y])) {
                        selected = false;
                        break;
                    }
                }
            }
        }

        return inverted ? !selected : selected;
    }

    /**
     * Returns the class { classId, r, g, b } for a 3D point, or null.
     * Accurate version: uses AABB for fast pruning, then projects to 2D regions.
     */
    _getPointClassification(localVector) {
        const worldMatrix = this.rootTransform.getWorldMatrix();
        const x = localVector.x, y = localVector.y, z = localVector.z;
        let finalCls = null;

        // Determine point's segment once per vector
        const pointSeg = this._resolvePointSegment(localVector);
        const pointSegId = pointSeg ? pointSeg.segmentId : 0;

        // Iterate from oldest to newest (later entries override earlier ones)
        for (const entry of this.classificationHistory) {
            // Visibility constraint: skip if point's segment was hidden at classification time,
            // BUT only if the segment existed before classification (CASE A).
            // If the segment was created after classification (CASE B), the point was in an ancestor
            // segment that was visible — apply classification if segment 0 was visible then.
            if (entry.visibleSegmentIds && !entry.visibleSegmentIds.includes(pointSegId)) {
                if (pointSegId > 0) {
                    const cutEntry = this.cutHistory.find(c => c.segmentId === pointSegId);
                    const orderAtClass = entry.cutCreationOrderAtClassification ?? 0;
                    // CASE A: segment existed before classification but was hidden → correct skip
                    if (cutEntry && cutEntry.creationOrder !== undefined && cutEntry.creationOrder < orderAtClass) {
                        continue;
                    }
                    // CASE B: segment created at/after classification → point was in a visible ancestor.
                    // Apply only if segment 0 was visible when the class was assigned.
                    if (!entry.visibleSegmentIds.includes(0)) continue;
                    // else: fall through and apply classification
                } else {
                    continue; // segment 0 was not visible → skip
                }
            }

            // 1. Fast AABB pruning
            if (x < entry.minX || x > entry.maxX ||
                y < entry.minY || y > entry.maxY ||
                z < entry.minZ || z > entry.maxZ) continue;

            // 2. Accurate region check, including deselections and inverted selection state.
            if (this._matchesSelectionEntry(localVector, entry.selections, entry.deselections, entry.inverted)) {
                finalCls = { classId: entry.classId, r: entry.r, g: entry.g, b: entry.b };
            }
        }
        return finalCls;
    }

    _isPointInPoly(poly, pt) {
        const x = pt[0], y = pt[1];
        let inside = false;
        for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            const xi = poly[i][0], yi = poly[i][1];
            const xj = poly[j][0], yj = poly[j][1];
            const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    // ========== CLASSIFICATION ==========

    /**
     * Assigns a class to all selected points across ALL LOD levels.
     *
     * How it works:
     *  1. Each selectionHistory entry (a frozen 2D region + camera state) is
     *     promoted into a classificationHistory entry with classId and color.
     *     This is the key insight: classification uses the same regions and the
     *     same projection logic as selection, so it works at every LOD level —
     *     coarse nodes seen from far away and fine detail nodes zoomed in alike.
     *  2. All currently loaded nodes are re-classified immediately using the
     *     same projection (catches both red-highlighted and non-highlighted points
     *     in the region, handling any sync issues between LOD and highlight state).
     *  3. selectionHistory is cleared and all red highlights are reset to
     *     originalColors — the selection is gone after classification.
     *
     * Future nodes loaded by the LOD system are handled automatically via
     * _getPointClassification() inside _createMeshFromBuffer().
     *
     * @param {number} classId  - integer class ID
     * @param {number} r,g,b    - RGB floats [0,1]
     * @returns {number} total points classified
     */
    applyClassToLoadedNodes(classId, r, g, b) {
        if (this.selectionHistory.length === 0) {
            console.warn("⚠️ No active selection to classify. Select points first.");
            return 0;
        }

        // 1. Classify currently loaded nodes using the 2D projection with
        //    the frozen camera state — works because these nodes were already
        //    visible at selection time and the projection is correct.
        //    Also builds the 3D AABB of all classified points.
        const worldMatrix = this.rootTransform.getWorldMatrix();
        let total = 0;

        // 3D AABB accumulated across all nodes
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        let anyClassified = false;

        const tmpVec = new BABYLON.Vector3();

        this.loadedNodes.forEach((mesh) => {
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            if (!positions) return;

            const numPoints = positions.length / 3;
            if (!mesh.metadata.classIds) mesh.metadata.classIds = new Int32Array(numPoints);
            if (!mesh.metadata.classColors) mesh.metadata.classColors = new Float32Array(numPoints * 4);

            let classified = 0;

            for (let i = 0; i < numPoints; i++) {
                // Skip invisible points (NaN-masked)
                if (isNaN(positions[i * 3])) continue;

                tmpVec.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);

                const isInside = this._matchesSelectionEntry(
                    tmpVec,
                    this.selectionHistory,
                    this.deselectionHistory,
                    this.selectionInverted
                );

                if (isInside) {
                    const pid = mesh.metadata.pointIds ? mesh.metadata.pointIds[i] : -1;
                    if (this._pointClassMap && pid >= 0 && pid < this._pointClassMap.length) {
                        this._pointClassMap[pid] = classId;
                    }

                    mesh.metadata.classIds[i] = classId;
                    mesh.metadata.classColors[i * 4] = r;
                    mesh.metadata.classColors[i * 4 + 1] = g;
                    mesh.metadata.classColors[i * 4 + 2] = b;
                    mesh.metadata.classColors[i * 4 + 3] = 1.0;
                    classified++;
                    // Update 3D AABB
                    const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
                    if (px < minX) minX = px; if (px > maxX) maxX = px;
                    if (py < minY) minY = py; if (py > maxY) maxY = py;
                    if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;
                    anyClassified = true;
                }
            }

            total += classified;

            // Rebuild the correct vertex colors for the active mode:
            // start from originalColors (removes selection red), then
            // overwrite classified points if mode is "classification".
            const finalColors = mesh.metadata.originalColors
                ? new Float32Array(mesh.metadata.originalColors)
                : mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);

            if (finalColors && this.colorMode === "classification") {
                const cIds = mesh.metadata.classIds;
                const cClrs = mesh.metadata.classColors;
                if (cIds && cClrs) {
                    for (let i = 0; i < cIds.length; i++) {
                        if (cIds[i] > 0) {
                            this._writeBlendedClassColor(finalColors, i, mesh.metadata.originalColors, cClrs);
                        }
                    }
                }
            }

            if (finalColors) {
                mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, finalColors);
            }
        });

        // 2. Update canonical color LUT and save the classification logic for future LOD nodes.
        if (anyClassified) {
            this._classColorLUT.set(classId, { r, g, b });

            const visibleSegmentIds = [];
            if (this.mainCloudVisible) visibleSegmentIds.push(0);
            this.cutHistory.forEach(c => {
                if (c.visible) visibleSegmentIds.push(c.segmentId);
            });

            const margin = this._getAabbMargin();
            this.classificationHistory.push({
                classId, r, g, b,
                inverted: this.selectionInverted,
                visibleSegmentIds,
                cutCreationOrderAtClassification: this._cutCreationCounter,
                minX: minX - margin, minY: minY - margin, minZ: minZ - margin,
                maxX: maxX + margin, maxY: maxY + margin, maxZ: maxZ + margin,
                // Copy current histories. Clone BABYLON matrices accurately.
                selections: this.selectionHistory.map(s => ({ ...s, transformMatrix: s.transformMatrix.clone() })),
                deselections: this.deselectionHistory.map(d => ({ ...d, transformMatrix: d.transformMatrix.clone() }))
            });
        }

        // 3. Clear selection state (selection and deselection)
        this.selectionHistory = [];
        this.deselectionHistory = [];

        // console.log(`✅ Classified ${total.toLocaleString()} points. classificationHistory: ${this.classificationHistory.length} region(s).`);
        return total;
    }

    /**
     * Store a selection region and highlight matching points in red.
     */
    applySelection(type, area, append = false) {
        if (!this.scene.activeCamera) return 0;

        // A new drag-selection starts a new selection session.
        // This avoids carrying old regions into invert/cut operations.
        if (!append && (this.selectionHistory.length > 0 || this.deselectionHistory.length > 0 || this.selectionInverted)) {
            this.clearSelection();
        }

        const camera = this.scene.activeCamera;
        const engine = this.scene.getEngine();
        const viewport = camera.viewport.toGlobal(engine.getRenderWidth(), engine.getRenderHeight());
        const transformMatrix = this.scene.getTransformMatrix().clone();

        this.selectionHistory.push({ type, area, viewport, transformMatrix });
        this.selectionInverted = false; // new selection resets invert state
        // console.log(`📌 Selection added to history. Total regions: ${this.selectionHistory.length}`);

        let totalSelected = 0;

        const tmpVec = new BABYLON.Vector3();
        const tmpProj = new BABYLON.Vector3();

        this.loadedNodes.forEach((mesh) => {
            const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            if (!colors || !positions) return;

            let modified = false;
            const meshWorldMatrix = mesh.getWorldMatrix();

            for (let i = 0; i < positions.length / 3; i++) {
                tmpVec.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                BABYLON.Vector3.ProjectToRef(tmpVec, meshWorldMatrix, transformMatrix, viewport, tmpProj);

                let isInside = false;
                if (type === "rect") {
                    isInside = (tmpProj.x >= area.x && tmpProj.x <= area.x + area.width &&
                        tmpProj.y >= area.y && tmpProj.y <= area.y + area.height);
                } else if (type === "lasso" || type === "polygon") {
                    isInside = this._isPointInPoly(area, [tmpProj.x, tmpProj.y]);
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

        return totalSelected;
    }

    /**
     * Updates the color of an existing class across all loaded nodes and the history.
     * @param {number} classId - The ID of the class to update.
     * @param {string} hexColor - The new hex color string.
     */
    updateClassColor(classId, hexColor) {
        // 1. Convert hex to RGB float
        const h = hexColor.replace('#', '');
        const r = parseInt(h.slice(0, 2), 16) / 255;
        const g = parseInt(h.slice(2, 4), 16) / 255;
        const b = parseInt(h.slice(4, 6), 16) / 255;

        // 2. Update canonical color LUT and Classification History
        this._classColorLUT.set(classId, { r, g, b });

        this.classificationHistory.forEach(entry => {
            if (entry.classId === classId) {
                entry.r = r; entry.g = g; entry.b = b;
            }
        });

        // 3. Update currently loaded meshes
        this.loadedNodes.forEach((mesh) => {
            const classIds = mesh.metadata?.classIds;
            const classColors = mesh.metadata?.classColors;
            const originalColors = mesh.metadata?.originalColors;
            const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
            if (!classIds || !classColors || !colors) return;

            let modified = false;
            for (let i = 0; i < classIds.length; i++) {
                if (classIds[i] === classId) {
                    classColors[i * 4] = r;
                    classColors[i * 4 + 1] = g;
                    classColors[i * 4 + 2] = b;

                    // Apply to visible colors if in classification mode
                    if (this.colorMode === "classification") {
                        this._writeBlendedClassColorFromRGB(colors, i, originalColors, r, g, b);
                        modified = true;
                    }
                }
            }
            if (modified) mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
        });

        // console.log(`🎨 Class ${classId} color updated to ${hexColor}`);
    }

    /**
     * Remove one class and recompute per-point classification/color for loaded nodes.
     * Points previously assigned to the removed class become unclassified (classId = 0)
     * unless covered by another remaining classification event.
     * @param {number} classId - Class ID to remove.
     * @returns {number} number of points affected in loaded meshes.
     */
    removeClass(classId) {
        // Remove all events of this class from history (also affects future LOD nodes).
        this.classificationHistory = this.classificationHistory.filter(entry => entry.classId !== classId);
        this._classColorLUT.delete(classId);

        // Update canonical map: points assigned to this class must be reset or recomputed.
        if (this._pointClassMap) {
            for (let pid = 0; pid < this._pointClassMap.length; pid++) {
                if (this._pointClassMap[pid] === classId) {
                    this._pointClassMap[pid] = 0; // Reset to unclassified, fallback to recomputation on next load if history exists
                }
            }
        }

        let affected = 0;

        // Rebuild classIds/classColors from the remaining history for each loaded mesh.
        const tmpVec = new BABYLON.Vector3();

        this.loadedNodes.forEach((mesh) => {
            if (!mesh.metadata) return;

            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
            const originalColors = mesh.metadata.originalColors;
            const originalPositions = mesh.metadata.originalPositions;
            if (!positions || !colors || !originalColors) return;

            const numPoints = positions.length / 3;
            const newClassIds = new Int32Array(numPoints);
            const newClassColors = new Float32Array(numPoints * 4);

            for (let i = 0; i < numPoints; i++) {
                const x = originalPositions ? originalPositions[i * 3] : positions[i * 3];
                const y = originalPositions ? originalPositions[i * 3 + 1] : positions[i * 3 + 1];
                const z = originalPositions ? originalPositions[i * 3 + 2] : positions[i * 3 + 2];

                tmpVec.set(x, y, z);
                const cls = this._getPointClassification(tmpVec);
                if (cls) {
                    newClassIds[i] = cls.classId;
                    newClassColors[i * 4] = cls.r;
                    newClassColors[i * 4 + 1] = cls.g;
                    newClassColors[i * 4 + 2] = cls.b;
                    newClassColors[i * 4 + 3] = 1.0;

                    // Sync to canonical map
                    const pid = mesh.metadata.pointIds ? mesh.metadata.pointIds[i] : -1;
                    if (this._pointClassMap && pid >= 0 && pid < this._pointClassMap.length) {
                        this._pointClassMap[pid] = cls.classId;
                    }
                }

                const hadRemovedClass = mesh.metadata.classIds && mesh.metadata.classIds[i] === classId;
                if (hadRemovedClass) affected++;

                const hasClassNow = newClassIds[i] > 0;
                if (this.colorMode === "classification" && hasClassNow) {
                    this._writeBlendedClassColor(colors, i, originalColors, newClassColors);
                } else {
                    colors[i * 4] = originalColors[i * 4];
                    colors[i * 4 + 1] = originalColors[i * 4 + 1];
                    colors[i * 4 + 2] = originalColors[i * 4 + 2];
                    colors[i * 4 + 3] = originalColors[i * 4 + 3];
                }
            }

            mesh.metadata.classIds = newClassIds;
            mesh.metadata.classColors = newClassColors;
            mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
        });

        return affected;
    }

    // ========== CUT / SEGMENTS ==========

    /**
     * Promotes the current selectionHistory into a new named cut segment.
     * All loaded points inside the selection regions are assigned the new segmentId.
     * Future LOD nodes are handled via cutHistory in _createMeshFromBuffer.
     * @returns {{ segmentId: number, count: number } | null}
     */
    cutSelection() {
        if (this.selectionHistory.length === 0) {
            console.warn("⚠️ No active selection to cut.");
            return null;
        }

        const segmentId = this._segmentIdCounter++;
        const visibleSegmentIdsAtCut = this._currentVisibleSegmentIds();
        let total = 0;
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        let anyAssigned = false;

        const tmpVec = new BABYLON.Vector3();

        this.loadedNodes.forEach((mesh) => {
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            if (!positions) return;

            if (!mesh.metadata.segmentIds) {
                mesh.metadata.segmentIds = new Int32Array(positions.length / 3);
            }
            const segmentIds = mesh.metadata.segmentIds;
            let assigned = 0;

            for (let i = 0; i < positions.length / 3; i++) {
                // Skip already-hidden points (NaN-masked by a previous cut/delete).
                // With selectionInverted=true, a NaN point projects to (NaN,NaN,NaN),
                // all area comparisons return false → selected=false → !selected=true,
                // which would wrongly capture every previously hidden point.
                if (isNaN(positions[i * 3])) continue;

                tmpVec.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                const isInside = this._matchesSelectionEntry(
                    tmpVec,
                    this.selectionHistory,
                    this.deselectionHistory,
                    this.selectionInverted
                );

                if (isInside) {
                    const pid = mesh.metadata.pointIds ? mesh.metadata.pointIds[i] : -1;
                    this._writePointSegment(pid, segmentId);

                    segmentIds[i] = segmentId;
                    assigned++;
                    const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
                    if (px < minX) minX = px; if (px > maxX) maxX = px;
                    if (py < minY) minY = py; if (py > maxY) maxY = py;
                    if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;
                    // Hide the point physically by setting its position to NaN
                    positions[i * 3] = NaN;
                    positions[i * 3 + 1] = NaN;
                    positions[i * 3 + 2] = NaN;
                    anyAssigned = true;
                }
            }
            total += assigned;

            if (assigned > 0) {
                this._setMeshPositionsAndNotify(mesh, positions);
            }

            // Restore colors (remove red highlight), keep classification colors
            if (mesh.metadata.originalColors) {
                const finalColors = new Float32Array(mesh.metadata.originalColors);
                if (this.colorMode === "classification") {
                    const cIds = mesh.metadata.classIds;
                    const cClrs = mesh.metadata.classColors;
                    if (cIds && cClrs) {
                        for (let i = 0; i < cIds.length; i++) {
                            if (cIds[i] > 0) {
                                this._writeBlendedClassColor(finalColors, i, mesh.metadata.originalColors, cClrs);
                            }
                        }
                    }
                }
                mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, finalColors);
            }
        });

        if (anyAssigned) {
            const margin = this._getAabbMargin();
            this.cutHistory.push({
                segmentId,
                visible: false, // Hidden by default upon cut
                inverted: this.selectionInverted,
                creationOrder: this._cutCreationCounter++,
                // Which segments were on screen when this cut ran — the loop above
                // only captured visible (non-NaN) points, and the LOD replay has to
                // reproduce that same restriction.
                visibleSegmentIds: visibleSegmentIdsAtCut,
                minX: minX - margin, minY: minY - margin, minZ: minZ - margin,
                maxX: maxX + margin, maxY: maxY + margin, maxZ: maxZ + margin,
                selections: this.selectionHistory.map(s => ({ ...s, transformMatrix: s.transformMatrix.clone() })),
                deselections: this.deselectionHistory.map(d => ({ ...d, transformMatrix: d.transformMatrix.clone() }))
            });
        }

        this.selectionHistory = [];
        this.deselectionHistory = [];

        // console.log(`✂️ Cut segment ${segmentId}: ${total.toLocaleString()} points assigned.`);
        return anyAssigned ? { segmentId, count: total } : null;
    }

    _setMeshPositionsAndNotify(mesh, positions) {
        mesh.setVerticesData(BABYLON.VertexBuffer.PositionKind, positions);
        if (mesh.refreshBoundingInfo) mesh.refreshBoundingInfo();
        mesh.computeWorldMatrix(true);

        if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
            window.dispatchEvent(new CustomEvent('pointcloud-positions-updated'));
        }
    }

    /**
     * Show or hide all points belonging to segmentId across all loaded nodes.
     * segmentId 0 = the main (uncut) cloud.
     */
    setSegmentVisible(segmentId, visible) {
        if (segmentId === 0) {
            this.mainCloudVisible = visible;
        }
        // Sync visibility to all history entries for this segment
        this.cutHistory.forEach(entry => {
            if (entry.segmentId === segmentId) entry.visible = visible;
        });

        // After updating the flags, recompute isVisible for every loaded node.
        // A node should be visible as long as at least one of its points is visible
        // (i.e. mainCloudVisible OR any cut segment is visible).
        const anySegmentVisible = this.mainCloudVisible || this.cutHistory.some(e => e.visible);
        // Keep root transform enabled: segment visibility is handled at point level.
        this.rootTransform.setEnabled(true);

        this.loadedNodes.forEach((mesh) => {
            const segmentIds = mesh.metadata?.segmentIds;
            const originalPositions = mesh.metadata?.originalPositions;
            if (!segmentIds || !originalPositions) return;

            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            if (!positions) return;

            const numPoints = positions.length / 3;
            let modified = false;

            for (let i = 0; i < numPoints; i++) {
                const pointSeg = segmentIds[i] || 0;
                if (pointSeg !== segmentId) continue;

                if (visible) {
                    positions[i * 3] = originalPositions[i * 3];
                    positions[i * 3 + 1] = originalPositions[i * 3 + 1];
                    positions[i * 3 + 2] = originalPositions[i * 3 + 2];
                } else {
                    positions[i * 3] = NaN;
                    positions[i * 3 + 1] = NaN;
                    positions[i * 3 + 2] = NaN;
                }
                modified = true;
            }

            if (modified) {
                this._setMeshPositionsAndNotify(mesh, positions);
            }

            // Keep mesh.isVisible in sync: hide entirely only when nothing is visible,
            // show immediately when at least one segment becomes visible again.
            if (this.activeNodes.has(mesh.metadata?.nodeInfo?.name)) {
                mesh.isVisible = anySegmentVisible;
            }
        });

        // console.log(`👁️ Segment ${segmentId} → ${visible ? "visible" : "hidden"}`);
    }

    /**
     * Slack added around the 3D AABB stored with every cut/classification entry.
     *
     * That AABB is built from the points present when the action ran, i.e. from a
     * coarse LOD sample, and is the ONLY depth bound the 2D-region replay has when
     * a finer node loads later. It must therefore cover the sampling error between
     * levels — roughly the root spacing — and nothing more.
     *
     * This used to be a hard-coded 0.05 in world units. On a cloud only 0.13 units
     * across that inflates every box by ~40% of the whole model, so the prune never
     * rejects anything and the replay degenerates into an unbounded prism; on a
     * cloud tens of metres wide the same constant is far too tight. Deriving it
     * from metadata.spacing makes it correct at any scale.
     */
    _getAabbMargin() {
        if (this._aabbMargin === undefined) {
            const spacing = this.metadata?.spacing;
            if (spacing > 0) {
                this._aabbMargin = spacing * 2;
            } else {
                const bb = this.metadata?.boundingBox;
                const diag = bb
                    ? Math.hypot(bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2])
                    : 1;
                this._aabbMargin = diag * 0.005;
            }
        }
        return this._aabbMargin;
    }

    /**
     * Reads the canonical segment of a point from _pointSegmentMap.
     * @returns {number|null} the segmentId, or null when the map is unavailable
     *          or the point has never been resolved.
     */
    _readPointSegment(pid) {
        if (!this._pointSegmentMap || pid < 0 || pid >= this._pointSegmentMap.length) return null;
        const stored = this._pointSegmentMap[pid];
        if (stored === SEG_UNRESOLVED) return null;
        return (stored === SEG_DELETED) ? this._deletedSegmentId : stored;
    }

    /**
     * Writes the canonical segment of a point into _pointSegmentMap.
     * Negative ids (the internal deleted segment) are stored as SEG_DELETED.
     */
    _writePointSegment(pid, segmentId) {
        if (!this._pointSegmentMap || pid < 0 || pid >= this._pointSegmentMap.length) return;
        this._pointSegmentMap[pid] = (segmentId === this._deletedSegmentId) ? SEG_DELETED : segmentId;
    }

    /**
     * Re-applies NaN / position-restore to a single mesh based on the current
     * mainCloudVisible and cutHistory states.  Called when a node re-enters
     * the LOD view so the outline hide/show state is honoured on fresh nodes.
     */
    _applySegmentVisibilityToMesh(mesh) {
        if (this.mainCloudVisible && this.cutHistory.every(e => e.visible)) return;

        const segmentIds = mesh.metadata?.segmentIds;
        const originalPositions = mesh.metadata?.originalPositions;
        if (!segmentIds || !originalPositions) return;

        const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
        if (!positions) return;

        const numPoints = positions.length / 3;
        let modified = false;

        for (let i = 0; i < numPoints; i++) {
            const pid = mesh.metadata?.pointIds ? mesh.metadata.pointIds[i] : -1;
            let pointSeg = 0;
            let resolved = false;

            // 1. Canonical map lookup
            const canonical = this._readPointSegment(pid);
            if (canonical !== null) {
                pointSeg = canonical;
                resolved = true;
            }

            // 2. Fallback to mesh metadata
            if (!resolved && segmentIds) {
                pointSeg = segmentIds[i] || 0;
            }

            let shouldHide;

            if (pointSeg === 0) {
                shouldHide = !this.mainCloudVisible;
            } else if (pointSeg === this._deletedSegmentId) {
                shouldHide = true;
            } else {
                const entry = this.cutHistory.find(e => e.segmentId === pointSeg);
                shouldHide = entry ? !entry.visible : false;
            }

            if (shouldHide) {
                positions[i * 3] = NaN;
                positions[i * 3 + 1] = NaN;
                positions[i * 3 + 2] = NaN;
            } else {
                positions[i * 3] = originalPositions[i * 3];
                positions[i * 3 + 1] = originalPositions[i * 3 + 1];
                positions[i * 3 + 2] = originalPositions[i * 3 + 2];
            }
            modified = true;
        }

        if (modified) {
            this._setMeshPositionsAndNotify(mesh, positions);
        }
    }

    /**
     * Snapshot of which segments are on screen right now (0 = main cloud).
     * Stored on every cut/assign entry so _resolvePointSegment can tell whether a
     * point was even selectable when that action ran. Must be taken BEFORE the new
     * entry is pushed — the segment being created does not exist yet.
     */
    _currentVisibleSegmentIds() {
        const ids = [];
        if (this.mainCloudVisible) ids.push(0);
        this.cutHistory.forEach(c => { if (c.visible) ids.push(c.segmentId); });
        return ids;
    }

    /**
     * cutHistory sorted by the order the actions actually happened.
     * cutHistory is push-ordered, but mergeSegments() can append a target entry
     * carrying an older creationOrder, so replaying the array as-is would apply
     * the actions out of sequence.
     */
    _getChronologicalCuts() {
        return this.cutHistory
            .slice()
            .sort((a, b) => (a.creationOrder ?? 0) - (b.creationOrder ?? 0));
    }

    /**
     * Replays the whole cut history for one point and returns the segment it ends
     * up in: { segmentId, visible }, or null when it never left the main cloud.
     *
     * This is a CHRONOLOGICAL SIMULATION, and it has to be, because cutSelection()
     * and assignSelectionToSegment() only ever capture points that are *currently
     * visible* (they skip NaN-masked ones). So whether a cut claims a point depends
     * on which segment the point was in at that moment and whether that segment was
     * on screen — which is exactly what entry.visibleSegmentIds records.
     *
     * The previous oldest-first scan ignored this and returned the first region that
     * geometrically contained the point. With a single cut the two agree, but from
     * the second cut on they diverge: cutting into a segment the user had toggled
     * back on moves those points to the newer (hidden) segment, while the old scan
     * kept handing them to the older — often visible — one. Every LOD node loaded
     * afterwards then resurrected that geometry in blocks.
     *
     * Entries with no visibleSegmentIds (older sessions) are treated as
     * "everything was visible", which reproduces the previous behaviour for them.
     */
    _resolvePointSegment(localVector, cuts) {
        const chronological = cuts || this._getChronologicalCuts();
        if (chronological.length === 0) return null;

        const x = localVector.x, y = localVector.y, z = localVector.z;
        let segmentId = 0;
        let entryOfSegment = null;

        for (let c = 0; c < chronological.length; c++) {
            const entry = chronological[c];

            // Could this point be picked at all when the action ran? It had to be
            // visible, i.e. sitting in a segment that was on screen at that time.
            const visibleThen = entry.visibleSegmentIds;
            if (visibleThen && !visibleThen.includes(segmentId)) continue;

            // Cheap AABB reject before the projection test.
            if (x < entry.minX || x > entry.maxX ||
                y < entry.minY || y > entry.maxY ||
                z < entry.minZ || z > entry.maxZ) continue;

            if (this._matchesSelectionEntry(localVector, entry.selections, entry.deselections, entry.inverted)) {
                segmentId = entry.segmentId;
                entryOfSegment = entry;
            }
        }

        if (segmentId === 0) return null;
        return { segmentId, visible: entryOfSegment ? !!entryOfSegment.visible : true };
    }

    /**
     * Clear selection highlight and history.
     * Does NOT touch classificationHistory — classifications are permanent until clearClassifications().
     */
    clearSelection() {
        this.selectionHistory = [];
        this.deselectionHistory = [];
        this.selectionInverted = false;
        this.loadedNodes.forEach((mesh) => {
            const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
            const originalColors = mesh.metadata?.originalColors;
            const classIds = mesh.metadata?.classIds;
            const classColors = mesh.metadata?.classColors;
            if (!colors || !originalColors) return;

            const numPoints = colors.length / 4;
            for (let i = 0; i < numPoints; i++) {
                const isRed = (
                    colors[i * 4] > 0.9 &&
                    colors[i * 4 + 1] < 0.1 &&
                    colors[i * 4 + 2] < 0.1
                );
                if (!isRed) continue;

                if (classIds && classIds[i] > 0 && classColors) {
                    this._writeBlendedClassColor(colors, i, originalColors, classColors);
                } else {
                    colors[i * 4] = originalColors[i * 4];
                    colors[i * 4 + 1] = originalColors[i * 4 + 1];
                    colors[i * 4 + 2] = originalColors[i * 4 + 2];
                    colors[i * 4 + 3] = originalColors[i * 4 + 3];
                }
            }
            mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
        });
        // console.log("🧹 Selection cleared.");
    }

    /**
     * Invert the current selection.
     * Points inside selectionHistory regions become unselected,
     * points outside become selected (red).
     * Sets a flag so newly loaded LOD nodes are colored correctly too.
     */
    invertSelection() {
        if (this.selectionHistory.length === 0) {
            console.warn("⚠️ No active selection to invert.");
            return;
        }

        this.selectionInverted = !this.selectionInverted;

        let totalNowSelected = 0;

        this.loadedNodes.forEach((mesh) => {
            if (!mesh || !mesh.isVisible) return;

            const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            const originalColors = mesh.metadata?.originalColors;
            if (!colors || !positions || !originalColors) return;

            const numPoints = colors.length / 4;
            for (let i = 0; i < numPoints; i++) {
                if (isNaN(positions[i * 3])) continue; // skip NaN-masked hidden points

                const isRed = (colors[i * 4] > 0.9 && colors[i * 4 + 1] < 0.1 && colors[i * 4 + 2] < 0.1);

                if (isRed) {
                    // Was selected → restore original color
                    colors[i * 4] = originalColors[i * 4];
                    colors[i * 4 + 1] = originalColors[i * 4 + 1];
                    colors[i * 4 + 2] = originalColors[i * 4 + 2];
                    colors[i * 4 + 3] = originalColors[i * 4 + 3];
                } else {
                    // Was not selected → mark red
                    colors[i * 4] = 1.0;
                    colors[i * 4 + 1] = 0.0;
                    colors[i * 4 + 2] = 0.0;
                    colors[i * 4 + 3] = 1.0;
                    totalNowSelected++;
                }
            }
            mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
        });

        // console.log(`🔄 Selection inverted (inverted=${this.selectionInverted}). ${totalNowSelected.toLocaleString()} points now selected.`);
    }

    /**
     * Remove selection from points inside the given region (CTRL+select).
     *
     * - Finds all loaded-node points that fall inside the region using the same
     *   2D-projection approach as applySelection.
     * - Resets their vertex color to originalColors (or class color if classified
     *   and colorMode === "classification").
     * - Also removes from selectionHistory any previously added region that
     *   overlaps — keeping the history consistent.
     */
    removeSelection(type, area) {
        if (!this.scene.activeCamera) return 0;

        const camera = this.scene.activeCamera;
        const engine = this.scene.getEngine();
        const viewport = camera.viewport.toGlobal(engine.getRenderWidth(), engine.getRenderHeight());
        const transformMatrix = this.scene.getTransformMatrix().clone();

        let totalDeselected = 0;

        const tmpVec = new BABYLON.Vector3();
        const tmpProj = new BABYLON.Vector3();

        this.loadedNodes.forEach((mesh) => {
            const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            if (!colors || !positions) return;

            const originalColors = mesh.metadata?.originalColors;
            const classIds = mesh.metadata?.classIds;
            const classColors = mesh.metadata?.classColors;
            const meshWorldMatrix = mesh.getWorldMatrix();
            let modified = false;

            for (let i = 0; i < positions.length / 3; i++) {
                // Only act on currently-selected (red) points
                if (!(colors[i * 4] > 0.9 && colors[i * 4 + 1] < 0.1 && colors[i * 4 + 2] < 0.1)) continue;

                tmpVec.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                BABYLON.Vector3.ProjectToRef(tmpVec, meshWorldMatrix, transformMatrix, viewport, tmpProj);

                let isInside = false;
                if (type === "rect") {
                    isInside = (
                        tmpProj.x >= area.x && tmpProj.x <= area.x + area.width &&
                        tmpProj.y >= area.y && tmpProj.y <= area.y + area.height
                    );
                } else if (type === "lasso" || type === "polygon") {
                    isInside = this._isPointInPoly(area, [tmpProj.x, tmpProj.y]);
                }

                if (isInside) {
                    // Restore: class color if classified + classification mode, else original
                    const hasClass = classIds && classIds[i] > 0 && classColors;
                    if (hasClass && this.colorMode === "classification") {
                        this._writeBlendedClassColor(colors, i, originalColors, classColors);
                    } else if (originalColors) {
                        colors[i * 4] = originalColors[i * 4];
                        colors[i * 4 + 1] = originalColors[i * 4 + 1];
                        colors[i * 4 + 2] = originalColors[i * 4 + 2];
                        colors[i * 4 + 3] = originalColors[i * 4 + 3];
                    }
                    totalDeselected++;
                    modified = true;
                }
            }

            if (modified) mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
        });

        // Register the deselection region in history with the current camera state.
        // Does NOT modify selectionHistory — original regions remain intact to allow
        // classification of the remaining points.
        this.deselectionHistory.push({ type, area, viewport, transformMatrix });

        // console.log(`🔴 Deselected ${totalDeselected.toLocaleString()} points. deselectionHistory: ${this.deselectionHistory.length} region(s).`);
        return totalDeselected;
    }

    /**
     * Clear all classifications. Resets classificationHistory and per-mesh class data.
     */
    clearClassifications() {
        this.classificationHistory = [];
        this._classColorLUT.clear();
        if (this._pointClassMap) this._pointClassMap.fill(0);

        this.loadedNodes.forEach((mesh) => {
            if (!mesh.metadata) return;
            const numPoints = mesh.metadata.classIds?.length || 0;
            mesh.metadata.classIds = new Int32Array(numPoints);
            mesh.metadata.classColors = new Float32Array(numPoints * 4);
            if (mesh.metadata.originalColors) {
                mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, new Float32Array(mesh.metadata.originalColors));
            }
        });
        // console.log("🗑️ All classifications cleared.");
    }

    /**
     * Assign currently selected points to an existing segment.
     * segmentId 0 = return to main cloud.
     */
    assignSelectionToSegment(segmentId) {
        if (this.selectionHistory.length === 0) return 0;

        const worldMatrix = this.rootTransform.getWorldMatrix();
        let totalAssigned = 0;
        let isVisible = (segmentId === 0) ? this.mainCloudVisible : true;

        if (segmentId > 0) {
            const entry = this.cutHistory.find(c => c.segmentId === segmentId);
            if (entry) isVisible = entry.visible;
        }

        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        let anyPoints = false;

        const tmpVec = new BABYLON.Vector3();
        const tmpProj = new BABYLON.Vector3();

        this.loadedNodes.forEach(mesh => {
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            const originalPositions = mesh.metadata.originalPositions;
            const segmentIds = mesh.metadata.segmentIds;
            if (!positions || !originalPositions || !segmentIds) return;

            let modified = false;

            for (let i = 0; i < segmentIds.length; i++) {
                // Skip points that are currently hidden, exactly like cutSelection() does.
                // A 2D selection region is a depth-unbounded prism: without this guard a
                // small on-screen rectangle also captures everything occluded behind it,
                // including the hidden main cloud, and assigning to a visible group makes
                // all of it pop back into view. You can only assign what you can see.
                if (isNaN(positions[i * 3])) continue;

                tmpVec.set(originalPositions[i * 3], originalPositions[i * 3 + 1], originalPositions[i * 3 + 2]);
                let isInside = false;
                for (const sel of this.selectionHistory) {
                    BABYLON.Vector3.ProjectToRef(tmpVec, worldMatrix, sel.transformMatrix, sel.viewport, tmpProj);
                    if (sel.type === "rect") {
                        isInside = (tmpProj.x >= sel.area.x && tmpProj.x <= sel.area.x + sel.area.width &&
                            tmpProj.y >= sel.area.y && tmpProj.y <= sel.area.y + sel.area.height);
                    } else if (sel.type === "lasso" || sel.type === "polygon") {
                        isInside = this._isPointInPoly(sel.area, [tmpProj.x, tmpProj.y]);
                    }
                    if (isInside) break;
                }

                if (isInside && this.deselectionHistory.length > 0) {
                    for (const dsel of this.deselectionHistory) {
                        BABYLON.Vector3.ProjectToRef(tmpVec, worldMatrix, dsel.transformMatrix, dsel.viewport, tmpProj);
                        let deselected = false;
                        if (dsel.type === "rect") {
                            deselected = (tmpProj.x >= dsel.area.x && tmpProj.x <= dsel.area.x + dsel.area.width &&
                                tmpProj.y >= dsel.area.y && tmpProj.y <= dsel.area.y + dsel.area.height);
                        } else if (dsel.type === "lasso" || dsel.type === "polygon") {
                            deselected = this._isPointInPoly(dsel.area, [tmpProj.x, tmpProj.y]);
                        }
                        if (deselected) { isInside = false; break; }
                    }
                }

                if (isInside) {
                    const pid = mesh.metadata.pointIds ? mesh.metadata.pointIds[i] : -1;
                    this._writePointSegment(pid, segmentId);

                    segmentIds[i] = segmentId;
                    totalAssigned++;
                    modified = true;
                    anyPoints = true;
                    const px = originalPositions[i * 3], py = originalPositions[i * 3 + 1], pz = originalPositions[i * 3 + 2];
                    if (px < minX) minX = px; if (px > maxX) maxX = px;
                    if (py < minY) minY = py; if (py > maxY) maxY = py;
                    if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;

                    if (isVisible) {
                        positions[i * 3] = px; positions[i * 3 + 1] = py; positions[i * 3 + 2] = pz;
                    } else {
                        positions[i * 3] = NaN; positions[i * 3 + 1] = NaN; positions[i * 3 + 2] = NaN;
                    }
                }
            }
            if (modified) {
                this._setMeshPositionsAndNotify(mesh, positions);
                this._resetSelectionColors(mesh);
            }
        });

        if (anyPoints) {
            const margin = this._getAabbMargin();
            const cloneRegions = (regions) => regions.map(r => ({
                ...r,
                transformMatrix: r.transformMatrix?.clone ? r.transformMatrix.clone() : r.transformMatrix
            }));

            // segmentId 0 is the main cloud and must never be represented as a cutHistory entry.
            // Instead, remove the moved area from all explicit cut segments via deselections.
            if (segmentId === 0) {
                this.cutHistory.forEach(entry => {
                    if (!entry || entry.segmentId <= 0) return;
                    if (!Array.isArray(entry.deselections)) entry.deselections = [];
                    entry.deselections.push(...cloneRegions(this.selectionHistory));
                    entry.deselections.push(...cloneRegions(this.deselectionHistory));
                });
            } else {
                let targetEntry = this.cutHistory.find(c => c.segmentId === segmentId);
                if (!targetEntry) {
                    targetEntry = {
                        segmentId,
                        visible: isVisible,
                        creationOrder: this._cutCreationCounter++,
                        minX: Infinity,
                        minY: Infinity,
                        minZ: Infinity,
                        maxX: -Infinity,
                        maxY: -Infinity,
                        maxZ: -Infinity,
                        selections: [],
                        deselections: []
                    };
                    this.cutHistory.push(targetEntry);
                }

                targetEntry.visible = isVisible;
                if (!Array.isArray(targetEntry.selections)) targetEntry.selections = [];
                if (!Array.isArray(targetEntry.deselections)) targetEntry.deselections = [];

                // An entry accumulates several assign actions, each taken under its own
                // visibility state, but carries a single snapshot. Union them: a segment
                // that was visible for any of those actions stays selectable during the
                // replay. Union is the permissive choice, so this can only fail towards
                // the old behaviour, never towards hiding something the user still has.
                const visibleNow = this._currentVisibleSegmentIds();
                targetEntry.visibleSegmentIds = Array.isArray(targetEntry.visibleSegmentIds)
                    ? [...new Set([...targetEntry.visibleSegmentIds, ...visibleNow])]
                    : visibleNow;

                targetEntry.minX = Math.min(targetEntry.minX ?? Infinity, minX - margin);
                targetEntry.minY = Math.min(targetEntry.minY ?? Infinity, minY - margin);
                targetEntry.minZ = Math.min(targetEntry.minZ ?? Infinity, minZ - margin);
                targetEntry.maxX = Math.max(targetEntry.maxX ?? -Infinity, maxX + margin);
                targetEntry.maxY = Math.max(targetEntry.maxY ?? -Infinity, maxY + margin);
                targetEntry.maxZ = Math.max(targetEntry.maxZ ?? -Infinity, maxZ + margin);

                targetEntry.selections.push(...cloneRegions(this.selectionHistory));
                targetEntry.deselections.push(...cloneRegions(this.deselectionHistory));

                // Prevent future LOD replay from assigning this moved area back to other segments.
                this.cutHistory.forEach(entry => {
                    if (!entry || entry.segmentId <= 0 || entry.segmentId === segmentId) return;
                    if (!Array.isArray(entry.deselections)) entry.deselections = [];
                    entry.deselections.push(...cloneRegions(this.selectionHistory));
                    entry.deselections.push(...cloneRegions(this.deselectionHistory));
                });
            }
        }

        this.selectionHistory = [];
        this.deselectionHistory = [];
        return totalAssigned;
    }

    /**
     * Build and return an array of { point_id, element } for all points 
     * Build and return an array of { point_id, element } for all points
     * in the entire cloud that belong to a mapped segment.
     * Traverse the octree to find all points across all levels of detail.
     */
    async exportAllTrainingData(segmentNameMap) {
        const totalPoints = this.metadata.points;
        const buffer = new Uint8Array(totalPoints * 2); // Interleaved: [segId, classId, segId, classId, ...]
        const seenIds = new Set();
        let totalProcessed = 0;
        let skippedNoId = 0;
        let duplicates = 0;

        // Identify which segments the user actually cares about (including segment 0)
        const requestedIds = Object.keys(segmentNameMap).map(id => parseInt(id, 10));
        if (requestedIds.length === 0) return null;

        const includeSegmentZero = segmentNameMap[0] !== undefined;

        // Helper: Is this node worth visiting?
        const shouldVisit = (node) => {
            if (includeSegmentZero) return true;
            return this.cutHistory.some(region => {
                if (!segmentNameMap[region.segmentId]) return false;
                return (node.boundingBox.min[0] <= region.maxX && node.boundingBox.max[0] >= region.minX &&
                    node.boundingBox.min[1] <= region.maxY && node.boundingBox.max[1] >= region.minY &&
                    node.boundingBox.min[2] <= region.maxZ && node.boundingBox.max[2] >= region.minZ);
            });
        };

        const traverse = async (node) => {
            if (!shouldVisit(node)) return;

            // Proxy node? Expand hierarchy
            if (node.nodeType === 2) {
                this._ensureHierarchyLoaded(node);
            }

            // Is this node already loaded in the scene?
            const mesh = this.loadedNodes.get(node.name);
            if (mesh && mesh.metadata && mesh.metadata.pointIds && mesh.metadata.segmentIds) {
                const segmentIds = mesh.metadata.segmentIds;
                const classIds = mesh.metadata.classIds || new Int32Array(mesh.metadata.pointIds.length);
                const pointIds = mesh.metadata.pointIds;
                for (let i = 0; i < pointIds.length; i++) {
                    const pid = pointIds[i];
                    totalProcessed++;
                    if (pid === -1 || pid >= totalPoints) {
                        skippedNoId++;
                        continue;
                    }
                    if (seenIds.has(pid)) {
                        duplicates++;
                        continue;
                    }

                    const segId = segmentIds[i] || 0;
                    if (segmentNameMap[segId] !== undefined) {
                        seenIds.add(pid);
                        buffer[pid * 2] = segId + 1;      // +1 so 0 stays "unannotated"
                        buffer[pid * 2 + 1] = classIds[i] || 0;
                    }
                }
            } else if (node.numPoints > 0) {
                // Fetch and parse points for this node
                const nodeBuffer = await this._fetchRange("octree.bin", node.byteOffset, node.byteSize);
                if (nodeBuffer) {
                    const points = this._parsePointsFromBufferDirect(node, nodeBuffer);
                    for (const p of points) {
                        totalProcessed++;
                        const pid = p.id;
                        if (pid === -1 || pid >= totalPoints) {
                            skippedNoId++;
                            continue;
                        }
                        if (seenIds.has(pid)) {
                            duplicates++;
                            continue;
                        }

                        let finalSegId = 0;
                        let finalClassId = 0;

                        // Use canonical maps if available
                        if (this._pointSegmentMap && pid < this._pointSegmentMap.length) {
                            const canonical = this._readPointSegment(pid);
                            if (canonical !== null) {
                                finalSegId = canonical;
                            } else {
                                // Fallback to projection if not yet evaluated — same
                                // chronological replay the LOD path uses, so the export
                                // agrees with what the user sees on screen.
                                const seg = this._resolvePointSegment(p.pos);
                                finalSegId = seg ? seg.segmentId : 0;
                                // Cache it
                                this._writePointSegment(pid, finalSegId);
                            }
                        }

                        if (this._pointClassMap && pid < this._pointClassMap.length) {
                            const stored = this._pointClassMap[pid];
                            if (stored !== 0 && stored !== 0xFF) {
                                finalClassId = stored;
                            } else {
                                // Fallback
                                const cls = this._getPointClassification(p.pos);
                                finalClassId = cls ? cls.classId : 0;
                                // Cache it
                                if (finalClassId > 0) this._pointClassMap[pid] = finalClassId;
                            }
                        }

                        if (segmentNameMap[finalSegId] !== undefined) {
                            seenIds.add(pid);
                            buffer[pid * 2] = finalSegId + 1; // +1 so 0 stays "unannotated"
                            buffer[pid * 2 + 1] = finalClassId;
                        }
                    }
                }
            }

            // Subdivide (processed sequentially to avoid ERR_INSUFFICIENT_RESOURCES)
            for (const child of node.children) {
                if (child) await traverse(child);
            }
        };

        // Diagnostic: check if POINT_ID attribute exists in the Potree octree
        const hasPointIdAttr = this.attributes.some(a => {
            const low = a.name.toLowerCase();
            return low === "point_id" || low === "pointid";
        });
        console.log(`[Diag] POINT_ID attribute in Potree metadata: ${hasPointIdAttr}`);
        console.log(`[Diag] Attributes: ${this.attributes.map(a => a.name).join(", ")}`);
        console.log(`[Diag] totalPoints (metadata.points): ${totalPoints}`);

        console.log("🚀 Starting global octree traversal (2-Channel Binary Mode)...");
        await traverse(this.root);
        console.log(`✅ Global traversal finished.`);
        console.log(`   - Total points in octree nodes: ${totalProcessed.toLocaleString()}`);
        console.log(`   - Unique points assigned in buffer: ${seenIds.size.toLocaleString()}`);
        console.log(`   - Points skipped: ${skippedNoId.toLocaleString()} (no ID), ${duplicates.toLocaleString()} (duplicates)`);

        // Diagnostic: sample first 20 assigned PIDs
        const samplePids = [];
        for (let k = 0; k < totalPoints && samplePids.length < 20; k++) {
            if (buffer[k * 2] !== 0) samplePids.push({ pid: k, seg: buffer[k * 2] - 1, cls: buffer[k * 2 + 1] });
        }
        console.log(`[Diag] Sample annotated PIDs:`, samplePids);

        return {
            buffer: buffer,
            segmentMap: segmentNameMap
        };
    }

    /**
     * Lightweight point parser that returns { pos: Vector3, id: number }[]
     */
    _parsePointsFromBufferDirect(node, buffer) {
        const view = new DataView(buffer);
        const numPoints = node.numPoints;
        const results = [];

        const scale = this.metadata.scale;
        const metaOffset = this.metadata.offset;
        const bbMin = this.metadata.boundingBox.min;

        let posAttr = null;
        let pointIdAttr = null;
        for (const attr of this.attributes) {
            const lowName = attr.name.toLowerCase();
            if (lowName === "position") posAttr = attr;
            if (lowName === "point_id" || lowName === "pointid") {
                pointIdAttr = attr;
            }
        }
        if (!posAttr) return [];

        for (let j = 0; j < numPoints; j++) {
            const pointOffset = j * this.bytesPerPoint;
            if (pointOffset + posAttr.byteOffset + 12 > buffer.byteLength) break;

            const rawX = view.getInt32(pointOffset + posAttr.byteOffset + 0, true);
            const rawY = view.getInt32(pointOffset + posAttr.byteOffset + 4, true);
            const rawZ = view.getInt32(pointOffset + posAttr.byteOffset + 8, true);

            const x = (rawX * scale[0]) + metaOffset[0] - bbMin[0];
            const y = (rawY * scale[1]) + metaOffset[1] - bbMin[1];
            const z = (rawZ * scale[2]) + metaOffset[2] - bbMin[2];

            let pid = -1;
            if (pointIdAttr && pointOffset + pointIdAttr.byteOffset + 4 <= buffer.byteLength) {
                pid = view.getInt32(pointOffset + pointIdAttr.byteOffset, true);
            }

            results.push({ pos: new BABYLON.Vector3(x, y, z), id: pid });
        }
        return results;
    }

    /**
     * Build and return an array of { point_id, element } for all loaded points
     * that belong to a mapped segment. (Legacy/LOD-limited version)
     */
    exportTrainingData(segmentNameMap) {
        const result = [];
        const seenIds = new Set();

        this.loadedNodes.forEach(mesh => {
            if (!mesh.metadata || !mesh.metadata.segmentIds || !mesh.metadata.pointIds) return;
            const segmentIds = mesh.metadata.segmentIds;
            const pointIds = mesh.metadata.pointIds;

            for (let i = 0; i < segmentIds.length; i++) {
                const segId = segmentIds[i];
                if (segId > 0 && segmentNameMap[segId]) {
                    const pid = pointIds[i];
                    if (pid !== undefined && pid !== -1 && !seenIds.has(pid)) {
                        seenIds.add(pid);
                        result.push({ point_id: pid, element: segmentNameMap[segId] });
                    }
                }
            }
        });

        return result;
    }

    /**
     * Export training data as binary annotations written into the .pcbin store.
     * Collects the segment/class buffer and POSTs to /api/export-mapping/,
     * which updates the server-side .pcbin file atomically.
     * Returns the pcbin_path on success.
     */
    async exportAllTrainingDataAsMapping(segmentNameMap) {
        // First collect the binary buffer (same as exportAllTrainingData)
        const exportResult = await this.exportAllTrainingData(segmentNameMap);
        if (!exportResult || !exportResult.buffer) {
            throw new Error('Failed to export training data');
        }

        // POST to /api/export-mapping/ — now updates the .pcbin store
        const formData = new FormData();
        formData.append('buffer', new Blob([exportResult.buffer], { type: 'application/octet-stream' }));
        formData.append('point_count', this.metadata.points.toString());
        formData.append('pcbin_path', (window.__APP_CONFIG?.runtimeDataPathPrefix || 'runtime_data') + '/working/features.pcbin');

        const response = await fetch('/api/export-mapping/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || 'Failed to export mapping');
        }

        const data = await response.json();
        console.log('[Potree2Loader] .pcbin annotations updated:', data.pcbin_path, `(${data.point_count} points annotated)`);

        return {
            mapping_path: data.pcbin_path,   // kept as mapping_path for caller compatibility
            pcbin_path: data.pcbin_path,
            point_count: data.point_count,
            segmentMap: exportResult.segmentMap
        };
    }

    /**
     * Remove selection from a specific segment (moves it back to main cloud 0).
     */
    removeSelectionFromSegment(segmentId) {
        if (this.selectionHistory.length === 0) return 0;

        const worldMatrix = this.rootTransform.getWorldMatrix();
        let totalRemoved = 0;

        const tmpVec = new BABYLON.Vector3();
        const tmpProj = new BABYLON.Vector3();

        this.loadedNodes.forEach(mesh => {
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            const originalPositions = mesh.metadata.originalPositions;
            const segmentIds = mesh.metadata.segmentIds;
            if (!positions || !originalPositions || !segmentIds) return;

            let modified = false;
            for (let i = 0; i < segmentIds.length; i++) {
                if (segmentIds[i] !== segmentId) continue;

                tmpVec.set(originalPositions[i * 3], originalPositions[i * 3 + 1], originalPositions[i * 3 + 2]);
                let isInside = false;
                for (const sel of this.selectionHistory) {
                    BABYLON.Vector3.ProjectToRef(tmpVec, worldMatrix, sel.transformMatrix, sel.viewport, tmpProj);
                    if (sel.type === "rect") {
                        isInside = (tmpProj.x >= sel.area.x && tmpProj.x <= sel.area.x + sel.area.width &&
                            tmpProj.y >= sel.area.y && tmpProj.y <= sel.area.y + sel.area.height);
                    } else if (sel.type === "lasso" || sel.type === "polygon") {
                        isInside = this._isPointInPoly(sel.area, [tmpProj.x, tmpProj.y]);
                    }
                    if (isInside) break;
                }

                if (isInside) {
                    const pid = mesh.metadata.pointIds ? mesh.metadata.pointIds[i] : -1;
                    this._writePointSegment(pid, 0);

                    segmentIds[i] = 0;
                    totalRemoved++;
                    modified = true;
                    if (this.mainCloudVisible) {
                        positions[i * 3] = originalPositions[i * 3];
                        positions[i * 3 + 1] = originalPositions[i * 3 + 1];
                        positions[i * 3 + 2] = originalPositions[i * 3 + 2];
                    } else {
                        positions[i * 3] = NaN; positions[i * 3 + 1] = NaN; positions[i * 3 + 2] = NaN;
                    }
                }
            }
            if (modified) {
                this._setMeshPositionsAndNotify(mesh, positions);
                this._resetSelectionColors(mesh);
            }
        });

        // Update history: add these regions as deselection to all entries of this segmentId
        this.cutHistory.forEach(entry => {
            if (entry.segmentId === segmentId) {
                this.selectionHistory.forEach(s => entry.deselections.push({ ...s, transformMatrix: s.transformMatrix.clone() }));
            }
        });

        this.selectionHistory = [];
        this.deselectionHistory = [];
        return totalRemoved;
    }

    /**
     * Move the current selection into an internal hidden "deleted" segment.
     * The segment is never exposed in the outline and remains always invisible.
     * @returns {number} number of points moved
     */
    deleteSelectedPoints() {
        if (this.selectionHistory.length === 0) return 0;

        const deletedSegmentId = this._deletedSegmentId;
        const visibleSegmentIdsAtDelete = this._currentVisibleSegmentIds();
        let total = 0;
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        let anyAssigned = false;

        const tmpVec = new BABYLON.Vector3();

        this.loadedNodes.forEach((mesh) => {
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            if (!positions) return;

            if (!mesh.metadata.segmentIds) {
                mesh.metadata.segmentIds = new Int32Array(positions.length / 3);
            }
            const segmentIds = mesh.metadata.segmentIds;

            for (let i = 0; i < positions.length / 3; i++) {
                if (isNaN(positions[i * 3])) continue; // already hidden — skip

                tmpVec.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                const isInside = this._matchesSelectionEntry(
                    tmpVec,
                    this.selectionHistory,
                    this.deselectionHistory,
                    this.selectionInverted
                );

                if (!isInside) continue;

                const pid = mesh.metadata.pointIds ? mesh.metadata.pointIds[i] : -1;
                this._writePointSegment(pid, deletedSegmentId);

                segmentIds[i] = deletedSegmentId;
                total++;

                const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
                if (px < minX) minX = px; if (px > maxX) maxX = px;
                if (py < minY) minY = py; if (py > maxY) maxY = py;
                if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;

                positions[i * 3] = NaN;
                positions[i * 3 + 1] = NaN;
                positions[i * 3 + 2] = NaN;
                anyAssigned = true;
            }

            if (anyAssigned) {
                this._setMeshPositionsAndNotify(mesh, positions);
            }

            this._resetSelectionColors(mesh);
        });

        if (anyAssigned) {
            const margin = this._getAabbMargin();
            this.cutHistory.push({
                segmentId: deletedSegmentId,
                visible: false,
                inverted: this.selectionInverted,
                creationOrder: this._cutCreationCounter++,
                visibleSegmentIds: visibleSegmentIdsAtDelete,
                minX: minX - margin, minY: minY - margin, minZ: minZ - margin,
                maxX: maxX + margin, maxY: maxY + margin, maxZ: maxZ + margin,
                selections: this.selectionHistory.map(s => ({ ...s, transformMatrix: s.transformMatrix.clone() })),
                deselections: this.deselectionHistory.map(d => ({ ...d, transformMatrix: d.transformMatrix.clone() }))
            });
        }

        this.selectionHistory = [];
        this.deselectionHistory = [];
        return total;
    }

    /**
     * Restore all points previously moved to the hidden deleted segment.
     * Restored points return to the main cloud (segment 0) and the hidden
     * deleted entry is removed from cutHistory so future LOD nodes stay visible.
     * @returns {number} number of restored points in loaded nodes
     */
    restoreDeletedPoints() {
        const deletedSegmentId = this._deletedSegmentId;
        this.cutHistory = this.cutHistory.filter(entry => entry.segmentId !== deletedSegmentId);

        if (this._pointSegmentMap) {
            for (let pid = 0; pid < this._pointSegmentMap.length; pid++) {
                if (this._pointSegmentMap[pid] === SEG_DELETED) {
                    this._pointSegmentMap[pid] = 0;
                }
            }
        }

        let restored = 0;

        this.loadedNodes.forEach((mesh) => {
            const segmentIds = mesh.metadata?.segmentIds;
            const pointIds = mesh.metadata?.pointIds;
            const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
            const originalPositions = mesh.metadata?.originalPositions;
            if (!segmentIds || !positions || !originalPositions) return;

            let modified = false;
            for (let i = 0; i < segmentIds.length; i++) {
                if (segmentIds[i] !== deletedSegmentId) continue;

                segmentIds[i] = 0;
                restored++;
                modified = true;

                if (this.mainCloudVisible) {
                    positions[i * 3] = originalPositions[i * 3];
                    positions[i * 3 + 1] = originalPositions[i * 3 + 1];
                    positions[i * 3 + 2] = originalPositions[i * 3 + 2];
                } else {
                    positions[i * 3] = NaN;
                    positions[i * 3 + 1] = NaN;
                    positions[i * 3 + 2] = NaN;
                }
            }

            if (modified) {
                this._setMeshPositionsAndNotify(mesh, positions);
                this._resetSelectionColors(mesh);
            }
        });
        return restored;
    }

    /**
     * Merge source segments into a target segment.
     * Updates loaded nodes and cut history so future LOD nodes follow the merged state.
     * @param {number} targetSegmentId
     * @param {number[]} sourceSegmentIds
     * @returns {{ movedPoints: number, mergedSegments: number }}
     */
    mergeSegments(targetSegmentId, sourceSegmentIds) {
        const sources = Array.from(new Set((sourceSegmentIds || [])
            .map(id => parseInt(id, 10))
            .filter(id => Number.isFinite(id) && id > 0 && id !== targetSegmentId)));

        if (sources.length === 0) return { movedPoints: 0, mergedSegments: 0 };

        const sourceSet = new Set(sources);
        const storeTarget = (targetSegmentId === this._deletedSegmentId) ? SEG_DELETED : targetSegmentId;

        if (this._pointSegmentMap) {
            for (let pid = 0; pid < this._pointSegmentMap.length; pid++) {
                const stored = this._pointSegmentMap[pid];
                if (stored === SEG_UNRESOLVED) continue;
                // Treat SEG_DELETED as deletedSegmentId
                const actualStored = (stored === SEG_DELETED) ? this._deletedSegmentId : stored;
                if (sourceSet.has(actualStored)) {
                    this._pointSegmentMap[pid] = storeTarget;
                }
            }
        }

        let movedPoints = 0;

        // Reassign segment ids on already loaded points.
        this.loadedNodes.forEach(mesh => {
            const segmentIds = mesh.metadata?.segmentIds;
            if (!segmentIds) return;

            let modified = false;
            for (let i = 0; i < segmentIds.length; i++) {
                if (!sourceSet.has(segmentIds[i])) continue;
                segmentIds[i] = targetSegmentId;
                movedPoints++;
                modified = true;
            }

            if (modified) {
                // Ensure visibility masks (NaN hiding) reflect the new target segment.
                this._applySegmentVisibilityToMesh(mesh);
            }
        });

        // Merge history from source segments into target history.
        const sourceEntries = this.cutHistory.filter(c => sourceSet.has(c.segmentId));
        if (sourceEntries.length > 0) {
            if (targetSegmentId > 0) {
                let targetEntry = this.cutHistory.find(c => c.segmentId === targetSegmentId);
                if (!targetEntry) {
                    targetEntry = {
                        segmentId: targetSegmentId,
                        visible: true,
                        creationOrder: Math.min(...sourceEntries.map(e => e.creationOrder ?? Infinity)),
                        selections: [],
                        deselections: [],
                        minX: Infinity,
                        minY: Infinity,
                        minZ: Infinity,
                        maxX: -Infinity,
                        maxY: -Infinity,
                        maxZ: -Infinity
                    };
                    this.cutHistory.push(targetEntry);
                }

                sourceEntries.forEach(src => {
                    targetEntry.visible = targetEntry.visible || !!src.visible;
                    if (Array.isArray(src.selections)) targetEntry.selections.push(...src.selections);
                    if (Array.isArray(src.deselections)) targetEntry.deselections.push(...src.deselections);
                    targetEntry.minX = Math.min(targetEntry.minX, src.minX ?? Infinity);
                    targetEntry.minY = Math.min(targetEntry.minY, src.minY ?? Infinity);
                    targetEntry.minZ = Math.min(targetEntry.minZ, src.minZ ?? Infinity);
                    targetEntry.maxX = Math.max(targetEntry.maxX, src.maxX ?? -Infinity);
                    targetEntry.maxY = Math.max(targetEntry.maxY, src.maxY ?? -Infinity);
                    targetEntry.maxZ = Math.max(targetEntry.maxZ, src.maxZ ?? -Infinity);
                });
            }

            this.cutHistory = this.cutHistory.filter(c => !sourceSet.has(c.segmentId));
        }

        return {
            movedPoints,
            mergedSegments: sources.length
        };
    }

    _resetSelectionColors(mesh) {
        const colors = mesh.getVerticesData(BABYLON.VertexBuffer.ColorKind);
        if (!colors) return;
        const orgClrs = mesh.metadata.originalColors;
        const clsIds = mesh.metadata.classIds;
        const clsClrs = mesh.metadata.classColors;
        let mod = false;
        for (let i = 0; i < colors.length / 4; i++) {
            if (colors[i * 4] > 0.9 && colors[i * 4 + 1] < 0.1 && colors[i * 4 + 2] < 0.1) {
                if (this.colorMode === "classification" && clsIds && clsIds[i] > 0) {
                    this._writeBlendedClassColor(colors, i, orgClrs, clsClrs);
                } else if (orgClrs) {
                    colors[i * 4] = orgClrs[i * 4];
                    colors[i * 4 + 1] = orgClrs[i * 4 + 1];
                    colors[i * 4 + 2] = orgClrs[i * 4 + 2];
                }
                colors[i * 4 + 3] = 1.0;
                mod = true;
            }
        }
        if (mod) mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
    }

    dispose() {
        if (this._cameraViewObserver && this._cameraForObserver) {
            this._cameraForObserver.onViewMatrixChangedObservable.remove(this._cameraViewObserver);
        }
        if (this._cleanupIntervalId) {
            window.clearInterval(this._cleanupIntervalId);
        }
        this._pointSegmentMap = null;
        this._pointClassMap = null;
        this._classColorLUT?.clear();
        this.loadedNodes.clear();
        this.activeNodes.clear();
        this.rootTransform.dispose();
    }

} // END class Potree2Loader
Potree2Loader._viridisLUT = Potree2Loader._buildViridisLUT();


// =====================================================================
// PUBLIC HELPER FUNCTIONS
// =====================================================================

export async function loadPotree2PointCloud(basePath, scene, options = {}) {
    // console.log("🚀 loadPotree2PointCloud:", basePath);

    const loader = new Potree2Loader(scene, basePath, options);
    await loader.load();

    const camera = scene.activeCamera;
    if (camera) {
        const bbMin = loader.metadata.boundingBox.min;
        const bbMax = loader.metadata.boundingBox.max;
        const localCenter = new BABYLON.Vector3(
            (bbMax[0] - bbMin[0]) / 2,
            (bbMax[1] - bbMin[1]) / 2,
            (bbMax[2] - bbMin[2]) / 2
        );
        const localSize = new BABYLON.Vector3(
            bbMax[0] - bbMin[0],
            bbMax[1] - bbMin[1],
            bbMax[2] - bbMin[2]
        );
        const radius = localSize.length() * 0.7;

        // console.log(`📷 Camera: target=${localCenter}, radius=${radius.toFixed(1)}`);
        camera.setTarget(localCenter);
        camera.radius = radius;
        camera.minZ = 0.1;
        camera.maxZ = radius * 10;

        let lastUpdate = 0;
        const updateThrottle = 200;

        loader._cameraForObserver = camera;
        loader._cameraViewObserver = camera.onViewMatrixChangedObservable.add(() => {
            const now = Date.now();
            if (now - lastUpdate > updateThrottle) {
                loader.update(camera);
                lastUpdate = now;
            }
        });

        loader._cleanupIntervalId = window.setInterval(() => loader.cleanup(200), 30000);
    }

    scene.potree2Loader = loader;

    // Notify main.js that attribute list is ready, so it can populate the color menu
    window.dispatchEvent(new CustomEvent('potree2-loaded', { detail: { loader } }));

    if (window.__registerPointCloudInOutline) {
        window.__registerPointCloudInOutline(loader.rootTransform, "Potree 2.0 Cloud");
    }

    if (window.__frameCameraOnMesh) {
        window.__frameCameraOnMesh(scene.activeCamera, loader.rootTransform);
    }

    console.log(`✅ Potree2Loader ready: ${loader.loadedNodes.size} nodes loaded, ${loader.activeNodes.size} visible`);
    return loader.getRoot();
}

export function getPotree2Loader(scene) {
    return scene.potree2Loader || null;
}