// =====================================================================
// CHUNKED POINT CLOUD LOADER for BabylonJS
// Format: meta.json + geom.bin + col/<name>.bin  (see docs/POINTCLOUD_FORMAT.md)
//
// The geometry is built once at import (las2pc): spatial chunks, points ordered by
// stratified levels of detail inside each chunk. Attributes (features, prediction) live
// in separate columns with the same point order, downloaded only when needed.
//
// Every request (chunk, points a..b of its level-ordered sequence) becomes one "virtual node" =
// one BabylonJS mesh in `loadedNodes` (name `c<id>_p<a>-<b>`), with the same `mesh.metadata`
// layout the editing code (selection, cut, classes, segments) has always used.
//
// Level of detail is CONTINUOUS and per point: the loader decides how many points of every chunk
// to fetch (`need`, from the projected spacing of the chunk's nearest point), and the vertex shader
// drops each point whose own spacing (attribute `lodInfo.x`, see pointcloud-lod.js) is finer than
// the pixel threshold, so the density on screen does not depend on the chunk but on the distance of
// every single point. The same screen metric (pixels per unit, perspective or orthographic) drives
// the loading decisions and the shader.
// =====================================================================

import { makeLodModel, lodCountFor, lodSpacingAt } from './pointcloud-lod.js';

/** A point is dropped when its projected spacing is below DRAW_MIN_RATIO * threshold; between that
 *  and the threshold it is drawn smaller (transition band). */
const DRAW_MIN_RATIO = 0.7;
const PREFETCH_MARGIN = 1.25;            // fetch up to need * margin
const MIN_REQUEST_GAP_POINTS = 1024;     // a shortfall smaller than this is not worth a request (each request prefetches 25 % more)
const MAX_REQUEST_POINTS = 256 * 1024;   // points per request (keeps one connection from being monopolised)
const MAX_BUDGET_FACTOR = 4096;

const FEATURE_MISSING_SENTINEL = -1e38;
const FEATURE_MISSING_COLOR = 0.8;
const GEOM_RECORD_SIZE = 20;
const COLUMN_FETCH_MAX_POINTS = 4_000_000;   // merge adjacent mesh ranges up to this size
const SCAN_BLOCK_POINTS = 400_000;           // 8 MB of geometry per export block

// ---- Sentinels for the canonical point_id → segmentId map (_pointSegmentMap) ----
// The map is a Uint16Array, so it cannot store the negative _deletedSegmentId (-1)
// nor an "unknown" state: both need out-of-band values.
//   SEG_UNRESOLVED — this point has never been resolved to a segment. The geometry
//                    replay in _createMeshFromDecoded decides (and then freezes) it.
//   SEG_DELETED    — this point lives in the internal hidden "deleted" segment.
// Uint16 leaves room for 65534 user segments, which also removes the 253-segment
// ceiling an 8-bit map would have imposed.
const SEG_UNRESOLVED = 0xFFFF;
const SEG_DELETED = 0xFFFE;

/**
 * Small pool of module workers. Each request is answered through a promise; requests are sent
 * to the worker with the fewest requests in flight.
 */
/** Distinct colour for the debug views (hue in turns). */
function lodDebugColor(h) {
    h = h - Math.floor(h);
    const f = (n) => { const k = (n + h * 6) % 6; return 0.85 - 0.75 * Math.max(0, Math.min(k, 4 - k, 1)); };
    return [f(5), f(3), f(1)];
}

class WorkerPool {
    constructor(size, cfg) {
        this._workers = [];
        this._pending = new Map();
        this._nextId = 1;
        const url = new URL('./pointcloud-worker.js', import.meta.url);
        for (let i = 0; i < size; i++) {
            const w = new Worker(url, { type: 'module' });
            w.inflight = 0;
            w.onmessage = (e) => {
                const { id, ok, result, error } = e.data;
                const p = this._pending.get(id);
                if (!p) return;
                this._pending.delete(id);
                p.worker.inflight--;
                if (ok) p.resolve(result); else p.reject(new Error(error));
            };
            w.onerror = (e) => console.error('[PointCloudLoader] worker error:', e.message || e);
            this._workers.push(w);
        }
        this.ready = Promise.all(this._workers.map(w => this._send(w, { type: 'init', cfg })));
    }

    _send(worker, msg) {
        return new Promise((resolve, reject) => {
            const id = this._nextId++;
            worker.inflight++;
            this._pending.set(id, { resolve, reject, worker });
            worker.postMessage({ ...msg, id });
        });
    }

    async request(msg) {
        await this.ready;
        let best = this._workers[0];
        for (const w of this._workers) if (w.inflight < best.inflight) best = w;
        return this._send(best, msg);
    }

    dispose() {
        for (const p of this._pending.values()) p.reject(new Error('Loader disposed'));
        this._pending.clear();
        for (const w of this._workers) w.terminate();
        this._workers = [];
    }
}

/**
 * Chunked point cloud loader for BabylonJS.
 *
 * Geometry and columns are fetched with HTTP Range requests (a Django endpoint backed by
 * sendfile), so multi-GB clouds never need to fit in browser memory: a point budget with LRU
 * eviction bounds what stays loaded.
 */
export class ChunkedPointCloudLoader {
    constructor(scene, baseUrl, options = {}) {
        this.scene = scene;
        this.baseUrl = baseUrl;
        // Cache-busting token of the geometry (appended as ?v=)
        this.version = options.version ?? null;
        // State exported by a previous loader (reload after a geometry rebuild), applied in load()
        this._initialState = options.initialState ?? null;
        this.metadata = null;
        this.chunks = [];
        this.rootTransform = new BABYLON.TransformNode("PointCloudRoot", scene);

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

        this.loadedNodes = new Map();   // name → mesh
        this.activeNodes = new Set();   // names of the meshes currently shown
        this.loadingNodes = new Set();  // names of the virtual nodes being fetched

        // Fixed point size for all nodes — same value regardless of LOD level
        this.pointSize = options.pointSize ?? 2;
        // Multiplier applied on top of the auto-computed size (controlled by the UI slider)
        this.pointSizeMultiplier = options.pointSizeMultiplier ?? 1.0;
        this.maxVisibleNodes = options.maxVisibleNodes || 4000;
        this.maxVisiblePoints = options.maxVisiblePoints || 5_000_000;
        this._maxLoadedPoints = options.maxLoadedPoints ?? null;
        this.maxConcurrentLoads = options.maxConcurrentLoads || 6;
        this.workerCount = options.workerCount || 3;
        // Continuous LOD (see the header): uniform budget factor, adaptive point size, debug view
        this.budgetFactor = 1;                       // >= 1: threshold multiplier that makes the needed points fit maxVisiblePoints
        this.adaptivePointSize = options.adaptivePointSize !== false;
        this.maxPointSizePx = options.maxPointSizePx ?? 8;
        this.lodDebug = "off";                       // off | chunk | level | spacing
        this._screenK = 1;                           // pixels per unit at depth 1 (perspective) / per unit (orthographic)
        this._persp = true;
        this._lodU = null;                           // last uniform values sent to the materials
        this._lodStats = { needTotal: 0 };

        this.stats = {
            loadedNodes: 0,
            visibleNodes: 0,
            totalPointsRendered: 0,
            loadingNodes: 0,
            loadedPoints: 0
        };
        this._loadedPoints = 0;
        this._tick = 0;

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
        // _createMeshFromDecoded uses this value to decide which colors to write
        // into the vertex data of newly loaded LOD nodes.
        this.colorMode = "classification";
        this.classColorBlendStrength = 0.75;
        // Feature attributes come from the columns of meta.json, filled by _buildFeatureAttributes()
        this.featureAttributes = new Map();
        this._featureRangeMin = null;
        this._featureRangeMax = null;
        this._featureShaderMat = null;
        this._colormapId = 0; // 0=Blue>Green>Yellow>Red (default)
        this._featureDiscreteFilter = null; // { featureName: string, value: number|null }
        this._columnFetchQueue = new Set();
        this._columnFetchTimer = null;
        this._columnEpoch = 0;

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

        // Opt-in diagnostics for tracking segment state across LOD eviction/reload.
        // Enable from the loader options or at runtime with setSegmentationDebug().
        this.debugSegmentation = options.debugSegmentation === true;
        this.debugWatchPointId = Number.isInteger(options.debugWatchPointId)
            ? options.debugWatchPointId : null;
        this._segmentRevision = 0;

        // Kept for API compatibility: eviction is now driven by the point budget (maxLoadedPoints)
        this.autoCleanupEnabled = options.autoCleanupEnabled === true;

        // Runtime handles that must be cleaned on dispose() to avoid stale callbacks
        // after Reset Scene / re-import.
        this._cameraForObserver = null;
        this._cameraViewObserver = null;
        this._cameraProjObserver = null;
        this._resizeObserver = null;
        this._renderObserver = null;
        this._throttleTimer = null;
        this._cleanupIntervalId = null;
        this._pool = null;
        this._lastCamera = null;
        this._updateTimer = null;
        this._pendingLoads = [];
        this._disposed = false;
    }

    /** Point budget kept in memory (loaded, visible or not). */
    get maxLoadedPoints() {
        return this._maxLoadedPoints ?? this.maxVisiblePoints * 3;
    }

    set maxLoadedPoints(v) {
        this._maxLoadedPoints = v;
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

    // ========== STATE TRANSFER (reload after a geometry rebuild) ==========

    /**
     * Snapshot of every instance state that is NOT tied to meshes/nodes/buffers, so a
     * new loader created on a rebuilt point cloud (same coordinates, same POINT_IDs)
     * can continue exactly where this one was. Arrays/maps are handed over by reference:
     * call it right before disposing this loader.
     */
    exportState() {
        const rt = this.rootTransform;
        return {
            boundingBoxMin: this.metadata?.boundingBox ? [...this.metadata.boundingBox.min] : null,
            points: this.metadata?.points ?? null,
            offset: this.metadata?.offset ? [...this.metadata.offset] : null,
            scale: this.metadata?.scale ? [...this.metadata.scale] : null,

            selectionHistory: this.selectionHistory,
            deselectionHistory: this.deselectionHistory,
            selectionInverted: this.selectionInverted,
            classificationHistory: this.classificationHistory,
            cutHistory: this.cutHistory,
            segmentIdCounter: this._segmentIdCounter,
            cutCreationCounter: this._cutCreationCounter,
            deletedSegmentId: this._deletedSegmentId,
            mainCloudVisible: this.mainCloudVisible,
            classColorLUT: this._classColorLUT,
            pointClassMap: this._pointClassMap,
            pointSegmentMap: this._pointSegmentMap,
            segmentRevision: this._segmentRevision,

            colorMode: this.colorMode,
            classColorBlendStrength: this.classColorBlendStrength,
            pointSize: this.pointSize,
            pointSizeMultiplier: this.pointSizeMultiplier,
            maxVisibleNodes: this.maxVisibleNodes,
            maxVisiblePoints: this.maxVisiblePoints,
            maxLoadedPoints: this._maxLoadedPoints,
            maxConcurrentLoads: this.maxConcurrentLoads,
            featureRangeMin: this._featureRangeMin,
            featureRangeMax: this._featureRangeMax,
            colormapId: this._colormapId,
            featureDiscreteFilter: this._featureDiscreteFilter,

            debugSegmentation: this.debugSegmentation,
            debugWatchPointId: this.debugWatchPointId,
            autoCleanupEnabled: this.autoCleanupEnabled,

            // Root node orientation (rotateAroundBoundingBoxCenter)
            rootRotationQuaternion: rt?.rotationQuaternion ? rt.rotationQuaternion.clone() : null,
            rootRotation: rt?.rotation ? rt.rotation.clone() : null,
            rootPivot: rt?.getPivotPoint ? rt.getPivotPoint().clone() : null
        };
    }

    /**
     * Applies a state produced by exportState() of a previous loader. Must run after
     * metadata/columns are parsed and BEFORE the first nodes are created (load() does it
     * when `options.initialState` is given), so new nodes pick up segments/classes/visibility.
     *
     * Geometric histories are expressed relative to metadata.boundingBox.min: if the new
     * point cloud has a different bounding box, the AABBs (cutHistory, classificationHistory)
     * are translated. Screen-space selections (selectionHistory/deselectionHistory) cannot
     * be translated; a warning is logged in that case.
     * Per-point maps are only transferred if the point count is identical.
     */
    importState(state) {
        if (!state) return;
        const sameCloud = state.points === null || state.points === this.metadata.points;

        // Bounding box / quantization differences
        const newMin = this.metadata.boundingBox.min;
        let delta = null;
        if (state.boundingBoxMin) {
            const d = [state.boundingBoxMin[0] - newMin[0], state.boundingBoxMin[1] - newMin[1], state.boundingBoxMin[2] - newMin[2]];
            if (Math.abs(d[0]) > 1e-9 || Math.abs(d[1]) > 1e-9 || Math.abs(d[2]) > 1e-9) delta = d;
        }
        const arrDiffers = (a, b) => !!a && !!b && a.some((v, i) => Math.abs(v - b[i]) > 1e-12);
        if (arrDiffers(state.offset, this.metadata.offset) || arrDiffers(state.scale, this.metadata.scale)) {
            console.warn('⚠️ offset/scale changed between builds; positions are re-derived from the new metadata.');
        }
        if (delta) {
            console.warn('⚠️ boundingBox changed between builds, translating AABB histories by', delta,
                '(screen-space selection history cannot be translated).');
        }
        const shiftAABB = (e) => {
            if (!delta || !e || e.minX === undefined) return e;
            return { ...e,
                minX: e.minX + delta[0], maxX: e.maxX + delta[0],
                minY: e.minY + delta[1], maxY: e.maxY + delta[1],
                minZ: e.minZ + delta[2], maxZ: e.maxZ + delta[2] };
        };

        this.selectionHistory = state.selectionHistory ?? [];
        this.deselectionHistory = state.deselectionHistory ?? [];
        this.selectionInverted = !!state.selectionInverted;
        this.classificationHistory = (state.classificationHistory ?? []).map(shiftAABB);
        this.cutHistory = (state.cutHistory ?? []).map(shiftAABB);
        this._segmentIdCounter = state.segmentIdCounter ?? this._segmentIdCounter;
        this._cutCreationCounter = state.cutCreationCounter ?? this._cutCreationCounter;
        this._deletedSegmentId = state.deletedSegmentId ?? this._deletedSegmentId;
        this.mainCloudVisible = state.mainCloudVisible ?? true;
        if (state.classColorLUT) this._classColorLUT = state.classColorLUT;
        this._segmentRevision = state.segmentRevision ?? this._segmentRevision;

        if (sameCloud) {
            if (state.pointSegmentMap && this._pointSegmentMap && state.pointSegmentMap.length === this._pointSegmentMap.length) {
                this._pointSegmentMap = state.pointSegmentMap;
            }
            if (state.pointClassMap && this._pointClassMap && state.pointClassMap.length === this._pointClassMap.length) {
                this._pointClassMap = state.pointClassMap;
            }
        } else {
            console.warn('⚠️ Point count changed between builds: per-point segment/class maps are NOT transferred.');
        }

        this.colorMode = state.colorMode ?? this.colorMode;
        this.classColorBlendStrength = state.classColorBlendStrength ?? this.classColorBlendStrength;
        this.pointSize = state.pointSize ?? this.pointSize;
        this.pointSizeMultiplier = state.pointSizeMultiplier ?? this.pointSizeMultiplier;
        this.maxVisibleNodes = state.maxVisibleNodes ?? this.maxVisibleNodes;
        this.maxVisiblePoints = state.maxVisiblePoints ?? this.maxVisiblePoints;
        this._maxLoadedPoints = state.maxLoadedPoints ?? this._maxLoadedPoints;
        this.maxConcurrentLoads = state.maxConcurrentLoads ?? this.maxConcurrentLoads;
        this._featureRangeMin = state.featureRangeMin ?? null;
        this._featureRangeMax = state.featureRangeMax ?? null;
        this._colormapId = state.colormapId ?? this._colormapId;
        this._featureDiscreteFilter = state.featureDiscreteFilter ?? null;
        this.debugSegmentation = state.debugSegmentation ?? this.debugSegmentation;
        this.debugWatchPointId = state.debugWatchPointId ?? this.debugWatchPointId;
        this.autoCleanupEnabled = state.autoCleanupEnabled ?? this.autoCleanupEnabled;

        // A feature colour mode only makes sense if the feature still exists in the new point cloud
        if (this.colorMode.startsWith('feature:') && !this.featureAttributes.has(this.colorMode.slice(8))) {
            this.colorMode = 'classification';
            this._featureRangeMin = null;
            this._featureRangeMax = null;
            this._featureDiscreteFilter = null;
        }

        const rt = this.rootTransform;
        if (state.rootPivot) rt.setPivotPoint(state.rootPivot, BABYLON.Space.LOCAL);
        if (state.rootRotationQuaternion) {
            rt.rotationQuaternion = state.rootRotationQuaternion.clone();
        } else if (state.rootRotation) {
            rt.rotation.copyFrom(state.rootRotation);
        }
        rt.computeWorldMatrix(true);
    }

    // ========== PUBLIC API ==========

    async load() {
        this.metadata = await this._fetchMeta();
        if (this.metadata.format !== 'pck') throw new Error(`Unsupported point cloud format: ${this.metadata.format}`);

        this._prepareChunks();
        this._buildFeatureAttributes();

        // Canonical point_id → segmentId map, indexed by POINT_ID (the index of the point in
        // features.las). This is the source of truth for segment membership: it survives LOD
        // unload/reload cycles, so a point keeps the segment it was assigned to even when its
        // node is evicted and fetched again. Without it every reload re-derives membership from
        // the 2D selection replay, which is not equivalent to the original assignment
        // (different tie-breaking on overlaps) and makes hidden segments pop back in blocks.
        if (this.metadata.points > 0) {
            this._pointSegmentMap = new Uint16Array(this.metadata.points).fill(SEG_UNRESOLVED);
            // Class per point (0 = none). Needed for the annotations export of points whose node
            // is not loaded: it falls back to the classification history replay when 0.
            this._pointClassMap = new Uint8Array(this.metadata.points);
        }

        // Reload after a rebuild: restore user state before any node is created
        if (this._initialState) {
            this.importState(this._initialState);
            this._initialState = null;
        }

        const pointCountDisplay = document.getElementById('point-count');
        if (pointCountDisplay) pointCountDisplay.textContent = this.metadata.points.toLocaleString();

        const bbMin = this.metadata.boundingBox.min;
        this._pool = new WorkerPool(this.workerCount, {
            scale: this.metadata.scale,
            offset: this.metadata.offset,
            qMin: this.metadata.qMin,
            bbMin: bbMin
        });

        await this._loadHead();

        // Fix LAS ↔ BabylonJS coordinate system mismatch: negate X scale to remove mirror effect
        this.rootTransform.scaling.x = -1;
        this.rootTransform.computeWorldMatrix(true);

        // The first LOD pass (detail fetching) is run by loadPointCloud() once the camera is framed on
        // the cloud: running it now would fetch for a view nobody sees.
        this._installRenderObserver();

        return this.rootTransform;
    }

    async _fetchMeta() {
        const url = `/pointcloud-data/${this.rangeBasePath}/meta.json?t=${Date.now()}`;
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Failed to load meta.json: ${response.status}`);
        return await response.json();
    }

    _withVersion(url, version = this.version) {
        if (version === null || version === undefined || version === "") return url;
        return `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(version)}`;
    }

    _geomUrl() {
        return this._withVersion(`/pointcloud-data/${this.rangeBasePath}/${this.metadata.geom.file}`,
            this.version ?? this.metadata.version);
    }

    _columnUrl(name) {
        const col = this.metadata.columns[name];
        return this._withVersion(`/pointcloud-data/${this.rangeBasePath}/${col.file}`, col.version);
    }

    /** Derives per-chunk bookkeeping from meta.json (prefix sums, AABBs, loaded-level state). */
    _prepareChunks() {
        const meta = this.metadata;
        const H = meta.levels.head;
        this.chunks = meta.chunks.map(c => {
            const lod = makeLodModel(c.size, meta.levels.base, c.levelPoints);
            const prefix = lod.prefix;                      // points of levels [0, l)
            return {
                id: c.id,
                min: c.min, max: c.max, size: c.size,
                levelPoints: c.levelPoints,
                lod,                                        // continuous LOD model (pointcloud-lod.js)
                headOffset: c.headOffset, bodyOffset: c.bodyOffset, points: c.points,
                headPoints: prefix[Math.min(H, c.levelPoints.length)],
                loaded: 0,             // points [0, loaded) of the level-ordered sequence are in memory (never a hole)
                loading: false,
                stack: [],             // meshes of the body, in sequence order (top = last)
                headMesh: null,
                need: 0,               // points the current view asks for
                visible: false,
                bb: null
            };
        });
        // Nominal spacing of the coarsest level of a typical chunk, comparable to the root
        // spacing of an octree: used to size the AABB margin of cut/classification regions.
        const sizes = this.chunks.map(c => c.size).sort((a, b) => a - b);
        const medianSize = sizes.length ? sizes[Math.floor(sizes.length / 2)] : 1;
        meta.spacing = medianSize / Math.pow(2, meta.levels.base + 1);
        this._worldMatrixKey = null;
    }

    /** Loads the overview (levels < head of every chunk) with a single Range request. */
    async _loadHead() {
        const meta = this.metadata;
        const H = meta.levels.head;
        const segments = this.chunks
            .filter(c => c.headPoints > 0)
            .map(c => ({ key: c.id, from: c.headOffset, count: c.headPoints, lod: { model: c.lod, seqStart: 0 } }));
        const result = await this._pool.request({
            type: 'geom', url: this._geomUrl(), start: 0, count: meta.head.points, segments
        });
        if (this._disposed) return;
        for (const seg of result.segments) {
            const chunk = this.chunks[seg.key];
            const n = seg.positions.length / 3;
            const vnode = this._makeVirtualNode(chunk, 0, n, chunk.headOffset);
            this._createMeshFromDecoded(vnode, seg);
            chunk.loaded = n;
            chunk.headMesh = this.loadedNodes.get(vnode.name);
        }
    }

    /** A node = the points [seqFrom, seqTo) of the level-ordered sequence of a chunk. */
    _makeVirtualNode(chunk, seqFrom, seqTo, start) {
        const numPoints = seqTo - seqFrom;
        return {
            name: `c${chunk.id}_p${seqFrom}-${seqTo}`,
            seqFrom,
            seqTo,
            chunk,
            start,
            numPoints,
            // Nominal spacing of the first point of the node (used to size cut/classification margins)
            spacing: lodSpacingAt(chunk.lod, seqFrom),
            boundingBox: { min: chunk.min, max: chunk.max }
        };
    }

    // ========== LOD SELECTION ==========

    /**
     * Rebuilds the AABB of every chunk in world space when the root transform changed
     * (rotateAroundBoundingBoxCenter). Cheap check: 16 floats.
     */
    _refreshChunkBoxes(world) {
        const m = world.m;
        let same = this._worldMatrixKey !== null;
        if (same) for (let i = 0; i < 16; i++) if (this._worldMatrixKey[i] !== m[i]) { same = false; break; }
        if (same) return;
        this._worldMatrixKey = Float32Array.from(m);
        const mn = new BABYLON.Vector3(), mx = new BABYLON.Vector3();
        for (const c of this.chunks) {
            mn.set(c.min[0], c.min[1], c.min[2]);
            mx.set(c.max[0], c.max[1], c.max[2]);
            if (!c.bb) c.bb = new BABYLON.BoundingBox(mn, mx, world);
            else c.bb.reConstruct(mn, mx, world);
        }
    }

    /**
     * Pixels per unit at depth 1 (perspective) or per unit (orthographic): the single screen metric
     * of the loader AND of the vertex shader. ppu(z) = k / z in perspective, k in orthographic.
     */
    _computeScreenK(camera) {
        const P = camera.getProjectionMatrix().m;
        return P[5] * this.scene.getEngine().getRenderHeight() / 2;
    }

    /**
     * Continuous LOD pass.
     *  1. visible chunks (frustum) and their ppu at the NEAREST point of the chunk's AABB;
     *  2. need = points of the level-ordered sequence whose spacing still projects to >= the pixel
     *     threshold (pointcloud-lod.js): the overview (head) is always counted;
     *  3. if the needs exceed maxVisiblePoints the threshold is multiplied by one common factor
     *     (bisection): the whole cloud degrades the same way instead of near = full / far = overview;
     *  4. chunks whose need exceeds what is loaded fetch the missing points (geometric growth), the
     *     most starved first.
     * Every loaded mesh of a chunk in view is shown: the vertex shader drops the points that are too
     * fine for their own distance, so there is no density border between chunks.
     */
    update(camera, { ignoreFrustum = false } = {}) {
        if (!this.chunks.length || !camera || this._disposed) return;
        this._lastCamera = camera;
        this._tick++;

        const anySegmentVisible = this.mainCloudVisible || this.cutHistory.some(e => e.visible);
        const meta = this.metadata;

        const persp = camera.mode !== BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
        const k = this._computeScreenK(camera);
        this._screenK = k;
        this._persp = persp;
        const T0 = Math.max(1, this.pointSize);

        const world = this.rootTransform.getWorldMatrix();
        this._refreshChunkBoxes(world);
        const invWorld = BABYLON.Matrix.Invert(world);
        const camWorld = camera.globalPosition || camera.position;
        const cam = BABYLON.Vector3.TransformCoordinates(camWorld, invWorld);
        const planes = BABYLON.Frustum.GetPlanes(camera.getTransformationMatrix());

        // ---- 1. visibility and screen scale of every chunk
        const visibleChunks = [];
        for (const c of this.chunks) {
            c.visible = ignoreFrustum || c.bb.isInFrustum(planes);
            if (!c.visible) { c.need = c.headPoints; continue; }
            if (persp) {
                const dx = Math.max(c.min[0] - cam.x, 0, cam.x - c.max[0]);
                const dy = Math.max(c.min[1] - cam.y, 0, cam.y - c.max[1]);
                const dz = Math.max(c.min[2] - cam.z, 0, cam.z - c.max[2]);
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                // Camera inside (or touching) the chunk: refine as much as the budget allows
                c.ppu = dist <= 1e-6 ? Infinity : k / dist;
            } else {
                c.ppu = k;
            }
            visibleChunks.push(c);
        }

        // ---- 2. points needed at pixel threshold T
        const needAt = (c, T) => {
            const sMin = c.ppu === Infinity ? 0 : T / c.ppu;
            return Math.max(c.headPoints, lodCountFor(c.lod, sMin));
        };

        // ---- 3. one budget factor for the whole cloud
        // The budget counts the points the shader DRAWS (spacing down to DRAW_MIN_RATIO x threshold): the overview of the
        // chunks out of view, plus, for every chunk in view, the points that still resolve. There is no floor at the head
        // here (the head is always LOADED, but the shader drops its fine points too), so a budget smaller than the head
        // simply raises the threshold until the drawn points fit.
        const budget = this.maxVisiblePoints;
        const base = meta.head.points;
        let headInView = 0;
        for (const c of visibleChunks) headInView += c.headPoints;
        const drawnAt = (m) => {
            let t = base - headInView;
            for (const c of visibleChunks) t += c.ppu === Infinity ? c.points : lodCountFor(c.lod, T0 * m * DRAW_MIN_RATIO / c.ppu);
            return t;
        };
        let m = 1;
        if (drawnAt(1) > budget) {
            let lo = 1, hi = MAX_BUDGET_FACTOR;
            for (let it = 0; it < 14; it++) {
                const mid = Math.sqrt(lo * hi);
                if (drawnAt(mid) > budget) lo = mid; else hi = mid;
            }
            m = hi;
        }
        this.budgetFactor = m;
        const T = T0 * m;
        let needTotal = base;   // points to fetch (at the threshold; the budget above counts the points drawn)
        for (const c of visibleChunks) {
            c.need = needAt(c, T);
            needTotal += c.need - c.headPoints;
        }

        // ---- 4. requests: only the chunks whose need exceeds what is loaded, most starved first
        this._pendingLoads = [];
        for (const c of visibleChunks) {
            if (c.loading || c.loaded >= c.points) continue;
            // Ask as soon as the view needs more than is loaded (a shortfall shows as a density hole);
            // the request itself prefetches PREFETCH_MARGIN more, so requests per chunk stay logarithmic.
            if (c.need - c.loaded >= MIN_REQUEST_GAP_POINTS || (c.need > c.loaded && c.loaded < c.headPoints + 1)) this._pendingLoads.push(c);
        }
        this._pendingLoads.sort((a, b) =>
            (b.need / Math.max(1, b.loaded)) - (a.need / Math.max(1, a.loaded)) || (a.ppu === b.ppu ? 0 : (b.ppu > a.ppu ? 1 : -1)));
        this._pump();

        // ---- 5. visibility of the loaded meshes: all of a chunk in view; only the overview otherwise
        let drawn = 0;
        for (const c of this.chunks) drawn += c.visible ? Math.min(c.loaded, c.need) : c.headPoints;
        for (const [name, mesh] of this.loadedNodes) {
            const info = mesh.metadata.nodeInfo;
            const chunk = this.chunks[info.chunkId];
            const shouldShow = chunk.visible || info.seqTo <= chunk.headPoints;
            const wasHidden = !mesh.isVisible;
            const targetVisible = shouldShow && anySegmentVisible;

            if (mesh.isVisible !== targetVisible) mesh.isVisible = targetVisible;
            if (shouldShow) mesh.metadata.lastUsedTick = this._tick;

            if (targetVisible && wasHidden) {
                this._debugSegment('node-visible', {
                    node: name,
                    meshRevision: mesh.metadata?.segmentRevision,
                    currentRevision: this._segmentRevision
                });
                this._applyColorModeToMesh(mesh);
                this._applySegmentVisibilityToMesh(mesh);
            }

            if (targetVisible) { this.activeNodes.add(name); } else { this.activeNodes.delete(name); }
        }

        this._updateLodUniforms();
        this.stats.visibleNodes = this.activeNodes.size;
        this.stats.totalPointsRendered = drawn;
        this.stats.loadingNodes = this.loadingNodes.size;
        this.stats.loadedPoints = this._loadedPoints;
        this._lodStats = { needTotal, budgetFactor: m, thresholdPx: T };
    }

    /** Re-runs update() on the next tick, collapsing bursts (used when a fetch completes). */
    _requestUpdate() {
        if (this._updateTimer !== null || this._disposed) return;
        this._updateTimer = setTimeout(() => {
            this._updateTimer = null;
            if (this._lastCamera && !this._disposed) this.update(this._lastCamera);
        }, 16);
    }

    /** Starts as many queued chunk requests as the concurrency limit allows. */
    _pump() {
        while (this.loadingNodes.size < this.maxConcurrentLoads && this._pendingLoads.length > 0) {
            const chunk = this._pendingLoads.shift();
            if (chunk.loading || chunk.loaded >= chunk.points || chunk.need <= chunk.loaded) continue;
            const from = chunk.loaded;
            const target = Math.min(chunk.points, Math.ceil(chunk.need * PREFETCH_MARGIN));
            const to = Math.min(target, from + MAX_REQUEST_POINTS);
            if (to > from) this._loadChunkRange(chunk, from, to);
        }
    }

    /** Fetches and builds the mesh of the points [from, to) of one chunk's sequence (from >= head). */
    async _loadChunkRange(chunk, from, to) {
        const H = chunk.headPoints;
        if (from < H) from = H;
        if (to <= from) return;
        const name = `c${chunk.id}_p${from}-${to}`;
        const start = chunk.bodyOffset + (from - H);
        const count = to - from;

        chunk.loading = true;
        this.loadingNodes.add(name);
        this.stats.loadingNodes = this.loadingNodes.size;
        this._debugSegment('node-load-start', { node: name, from, expectedPoints: count });

        try {
            const result = await this._pool.request({
                type: 'geom', url: this._geomUrl(), start, count,
                segments: [{ key: name, from: 0, count, lod: { model: chunk.lod, seqStart: from } }]
            });
            if (this._disposed) return;
            const seg = result.segments[0];
            const n = seg.positions.length / 3;
            const vnode = this._makeVirtualNode(chunk, from, from + n, start);
            this._createMeshFromDecoded(vnode, seg);
            chunk.stack.push(this.loadedNodes.get(name));
            chunk.loaded = from + n;
            this._debugSegment('node-load-complete', { node: name, from, points: vnode.numPoints });
            this._evictIfNeeded();
        } catch (err) {
            if (!this._disposed) console.error(`❌ Failed to load ${name}:`, err);
        } finally {
            chunk.loading = false;
            this.loadingNodes.delete(name);
            this.stats.loadingNodes = this.loadingNodes.size;
            if (!this._disposed) {
                this._pump();
                this._requestUpdate();
            }
        }
    }

    // ========== EVICTION ==========

    /**
     * Keeps the number of loaded points under maxLoadedPoints. Only the TOP mesh of a chunk (its
     * last points of the sequence) is a candidate, so a chunk never ends up with a hole; the
     * overview (head) is never evicted. Meshes shown in the last update are kept and, among the
     * others, the least recently used goes first. Chunks with a request in flight are never touched.
     * If nothing else is left, a chunk in view gives up its top mesh when what remains still covers
     * what its view needs.
     */
    _evictIfNeeded() {
        const limit = this.maxLoadedPoints;
        let guard = 100000;
        while (this._loadedPoints > limit && guard-- > 0) {
            let victim = null, victimChunk = null;
            for (const c of this.chunks) {
                const top = c.stack[c.stack.length - 1];
                if (!top || c.loading) continue;     // a request in flight extends the prefix that eviction would shorten
                if (top.metadata.lastUsedTick === this._tick && c.visible) continue;
                if (victim === null || top.metadata.lastUsedTick < victim.metadata.lastUsedTick) {
                    victim = top; victimChunk = c;
                }
            }
            if (!victim) {
                let bestSurplus = 0;
                for (const c of this.chunks) {
                    const top = c.stack[c.stack.length - 1];
                    if (!top || !c.visible || c.loading) continue;
                    // evict only if the chunk still holds what its view needs afterwards (no download/evict cycle)
                    const surplus = c.loaded - top.metadata.nodeInfo.numPoints - c.need;
                    if (surplus > bestSurplus) { bestSurplus = surplus; victim = top; victimChunk = c; }
                }
            }
            if (!victim) break;
            victimChunk.stack.pop();
            victimChunk.loaded = victim.metadata.nodeInfo.seqFrom;
            this._disposeMesh(victim);
        }
    }

    _disposeMesh(mesh) {
        const name = mesh.metadata?.nodeInfo?.name;
        if (name !== undefined) {
            this.loadedNodes.delete(name);
            this.activeNodes.delete(name);
        }
        this._loadedPoints -= mesh.metadata?.nodeInfo?.numPoints ?? 0;
        this._columnFetchQueue.delete(mesh);
        mesh.dispose();
        this.stats.loadedNodes = this.loadedNodes.size;
        this.stats.loadedPoints = this._loadedPoints;
    }

    // ========== CLEANUP (API compatibility) ==========

    /** Frees every mesh that is not currently shown (except the overview). */
    cleanup() {
        for (const c of this.chunks) {
            while (c.stack.length > 0) {
                const top = c.stack[c.stack.length - 1];
                if (this.activeNodes.has(top.metadata.nodeInfo.name)) break;
                c.stack.pop();
                c.loaded = top.metadata.nodeInfo.seqFrom;
                this._disposeMesh(top);
            }
        }
    }

    /** No-op kept for API compatibility: eviction is driven by the point budget. */
    startAutoCleanup() { }

    stopAutoCleanup() {
        if (this._cleanupIntervalId !== null) {
            window.clearInterval(this._cleanupIntervalId);
            this._cleanupIntervalId = null;
        }
    }

    /**
     * Sets the point size in pixels (UI slider). It is also the pixel threshold of the continuous LOD
     * (a point is kept while the spacing of its level projects to at least this many pixels), so
     * changing it changes how many points are needed: the LOD is re-evaluated.
     */
    setPointSize(size) {
        this.pointSize = Math.max(1, size);
        this._updateLodUniforms();
        if (this._lastCamera) this._requestUpdate();
    }

    /** Point size grows where the data is coarser than the screen (zoomed in beyond the resolution). */
    setAdaptivePointSize(enabled) {
        this.adaptivePointSize = !!enabled;
        this._updateLodUniforms();
    }

    // ========== CAMERA ==========

    /**
     * Re-evaluates the LOD when the view OR the projection changes (orthographic zoom only changes the
     * projection), the engine is resized, or the camera mode changes. Throttled with a trailing call so
     * the final position is always evaluated.
     */
    attachCamera(camera, { throttleMs = 200 } = {}) {
        this.detachCamera();
        this._cameraForObserver = camera;
        let last = 0;
        const trigger = () => {
            if (this._disposed) return;
            const now = Date.now();
            if (now - last > throttleMs) {
                last = now;
                this.update(camera);
            } else if (this._throttleTimer === null) {
                this._throttleTimer = setTimeout(() => {
                    this._throttleTimer = null;
                    last = Date.now();
                    if (!this._disposed) this.update(camera);
                }, throttleMs);
            }
        };
        this._cameraViewObserver = camera.onViewMatrixChangedObservable.add(trigger);
        this._cameraProjObserver = camera.onProjectionMatrixChangedObservable.add(trigger);
        this._resizeObserver = this.scene.getEngine().onResizeObservable.add(trigger);
    }

    detachCamera() {
        const cam = this._cameraForObserver;
        if (cam) {
            if (this._cameraViewObserver) cam.onViewMatrixChangedObservable.remove(this._cameraViewObserver);
            if (this._cameraProjObserver) cam.onProjectionMatrixChangedObservable.remove(this._cameraProjObserver);
        }
        if (this._resizeObserver) this.scene.getEngine().onResizeObservable.remove(this._resizeObserver);
        this._cameraViewObserver = null;
        this._cameraProjObserver = null;
        this._resizeObserver = null;
        if (this._throttleTimer !== null) { clearTimeout(this._throttleTimer); this._throttleTimer = null; }
    }

    // ========== CONTINUOUS LOD: UNIFORMS, PICKING, DEBUG ==========

    /** Keeps the screen metric of the shaders equal to the camera's at every frame. */
    _installRenderObserver() {
        if (this._renderObserver) return;
        this._renderObserver = this.scene.onBeforeRenderObservable.add(() => {
            const cam = this.scene.activeCamera;
            if (!cam || this._disposed) return;
            this._screenK = this._computeScreenK(cam);
            this._persp = cam.mode !== BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
            this._updateLodUniforms();
        });
    }

    _updateLodUniforms() {
        const T = Math.max(1, this.pointSize) * this.budgetFactor;
        const mode = this.lodDebug === 'chunk' || this.lodDebug === 'level' ? 1 : (this.lodDebug === 'spacing' ? 2 : 0);
        const u = this._lodU, adaptive = this.adaptivePointSize ? 1 : 0, persp = this._persp ? 1 : 0;
        if (u && u[0] === this._screenK && u[1] === persp && u[2] === T && u[3] === adaptive && u[4] === this.maxPointSizePx && u[5] === mode) return;
        this._lodU = [this._screenK, persp, T, adaptive, this.maxPointSizePx, mode];
        for (const mat of [this._baseMat, this._featureShaderMat]) {
            if (!mat) continue;
            mat.setVector4('pcLod', new BABYLON.Vector4(this._screenK, persp, T, 0));
            mat.setVector4('pcLod2', new BABYLON.Vector4(adaptive, this.maxPointSizePx, mode, 0));
        }
    }

    /** Parameters of the draw test of the shader, for CPU code (picking, probes): see isPointDrawn(). */
    getDrawParams() {
        return {
            k: this._screenK, perspective: this._persp,
            threshold: Math.max(1, this.pointSize) * this.budgetFactor,
            minRatio: DRAW_MIN_RATIO
        };
    }

    /**
     * Same test as the vertex shader: is point `i` of `mesh` drawn (not dropped by the continuous LOD)?
     * `w` is the clip-space w of the point (view depth in perspective; ignored in orthographic).
     */
    static isDrawn(lodSpacing, w, params) {
        const ppu = params.perspective ? params.k / Math.max(w, 1e-6) : params.k;
        return lodSpacing * ppu >= params.minRatio * params.threshold;
    }

    /**
     * Returns `(i) => boolean`: is point i of `mesh` drawn? The buffers, the matrix and the parameters are
     * read once, so use this (not isPointDrawn) when testing many points of one mesh.
     */
    getDrawFilter(mesh) {
        const lod = mesh.getVerticesData('lodInfo');
        if (!lod) return () => true;
        const pos = mesh.getVerticesData('position');
        const M = mesh.getWorldMatrix().multiply(this.scene.activeCamera.getTransformationMatrix()).m;
        const params = this.getDrawParams();
        const persp = this._persp;
        return (i) => {
            const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
            const w = x * M[3] + y * M[7] + z * M[11] + M[15];
            if (persp && !(w > 0)) return false;
            return ChunkedPointCloudLoader.isDrawn(lod[2 * i], w, params);
        };
    }

    /** Single-point convenience; for loops over a mesh use getDrawFilter(mesh). */
    isPointDrawn(mesh, i) {
        return this.getDrawFilter(mesh)(i);
    }

    /** 'off' | 'chunk' (colour per chunk) | 'level' (colour per mesh of a chunk) | 'spacing' (heatmap of projected spacing). */
    setLodDebug(mode) {
        this.lodDebug = ['chunk', 'level', 'spacing'].includes(mode) ? mode : 'off';
        this._updateLodUniforms();
    }

    /** Per-chunk state of the last update: need, loaded points and spacing projected on the screen. */
    getLodStats() {
        const chunks = this.chunks.map(c => ({
            id: c.id, points: c.points, need: c.need, loaded: c.loaded, visible: c.visible,
            meshes: c.stack.length + (c.headMesh ? 1 : 0),
            projectedSpacingPx: c.visible && Number.isFinite(c.ppu) ? lodSpacingAt(c.lod, Math.max(1, c.need)) * c.ppu : null
        }));
        return { ...this._lodStats, budgetFactor: this.budgetFactor, chunks };
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

    // ========== FEATURES (columns) ==========

    /**
     * Names of the columns that are NOT features (standard LAS fields and service attributes).
     * Everything else in meta.json "columns" (LAS Extra Bytes written by the feature extraction,
     * plus `prediction`) is exposed as a feature. Compared case-insensitively.
     */
    static NON_FEATURE_ATTRIBUTES = new Set([
        'position', 'rgb', 'intensity', 'return number', 'number of returns',
        'classification', 'classification flags', 'scan direction flag',
        'edge of flight line', 'scan angle rank', 'scan angle', 'scanner channel',
        'user data', 'point source id', 'gps-time', 'gps time', 'nir',
        'point_id', 'pointid'
    ]);

    _buildFeatureAttributes() {
        this.featureAttributes = new Map();
        for (const [name, col] of Object.entries(this.metadata.columns || {})) {
            if (ChunkedPointCloudLoader.NON_FEATURE_ATTRIBUTES.has(String(name).toLowerCase())) continue;
            this.featureAttributes.set(name, { name, ...col });
        }
    }

    hasFeatures() {
        return this.featureAttributes.size > 0;
    }

    getFeatureList() {
        return [...this.featureAttributes.keys()];
    }

    /**
     * Value range of a feature from meta.json. Falls back to 0..1 when min/max are missing or
     * not finite.
     */
    getFeatureRange(name) {
        const attr = this.featureAttributes.get(name);
        if (!attr) return { min: 0, max: 1 };
        const min = Number(attr.min);
        const max = Number(attr.max);
        if (attr.min === null || attr.max === null || !Number.isFinite(min) || !Number.isFinite(max)) {
            return { min: 0, max: 1 };
        }
        return { min, max };
    }

    /**
     * Re-reads meta.json after the columns changed on the server (features computed, classification,
     * backup restored) WITHOUT touching the geometry: updates the feature list, drops the cached
     * values of the columns whose version changed and re-downloads the active one.
     * Returns { reloadNeeded: true } when the geometry itself changed (different build): the
     * caller then has to reload the loader.
     */
    async refreshColumns() {
        const meta = await this._fetchMeta();
        if (meta.version !== this.metadata.version || meta.points !== this.metadata.points) {
            return { reloadNeeded: true };
        }
        this.metadata.columns = meta.columns || {};
        this._buildFeatureAttributes();

        if (this.colorMode.startsWith('feature:')) {
            const name = this.colorMode.slice(8);
            if (!this.featureAttributes.has(name)) {
                // The active feature does not exist any more (e.g. after a restore)
                this.setColorMode('classification');
                window.dispatchEvent(new CustomEvent('features-available', { detail: { names: this.getFeatureList() } }));
                return { reloadNeeded: false, activeFeatureLost: name };
            }
            // Same name, possibly new values: invalidate and fetch again
            this._columnEpoch++;
            this._applyFeatureShaderMode(true, name);
        }
        window.dispatchEvent(new CustomEvent('features-available', { detail: { names: this.getFeatureList() } }));
        return { reloadNeeded: false };
    }

    /**
     * Makes sure `mesh` has a `featureValue` vertex buffer for the active feature: a placeholder
     * filled with the "missing" sentinel is installed immediately (the shader paints it grey)
     * and the real values are fetched from the column in the background.
     */
    _ensureFeatureBuffer(mesh) {
        const md = mesh.metadata;
        const featureName = this.colorMode.slice(8);
        if (!this.featureAttributes.has(featureName)) return;
        if (md.featureName === featureName && md.featureEpoch === this._columnEpoch) return;

        const n = md.nodeInfo.numPoints;
        if (!mesh.getVertexBuffer('featureValue')) {
            const placeholder = new Float32Array(n).fill(FEATURE_MISSING_SENTINEL);
            const buf = new BABYLON.VertexBuffer(this.scene.getEngine(), placeholder, 'featureValue', true, false, 1);
            mesh.setVerticesBuffer(buf);
        }
        md.featureName = featureName;
        md.featureEpoch = this._columnEpoch;
        md.featureReady = false;
        this._columnFetchQueue.add(mesh);
        if (this._columnFetchTimer === null) {
            this._columnFetchTimer = setTimeout(() => {
                this._columnFetchTimer = null;
                this._flushColumnFetches();
            }, 0);
        }
    }

    /** Groups the pending meshes into contiguous ranges (as few requests as possible) and fetches them. */
    _flushColumnFetches() {
        if (this._disposed || this._columnFetchQueue.size === 0) return;
        const name = this.colorMode.startsWith('feature:') ? this.colorMode.slice(8) : null;
        const col = name ? this.metadata.columns?.[name] : null;
        const meshes = [...this._columnFetchQueue].filter(m => m.metadata.featureName === name);
        this._columnFetchQueue.clear();
        if (!col || meshes.length === 0) return;

        const epoch = this._columnEpoch;
        const items = meshes.map(m => ({ mesh: m, start: m.metadata.pcRange.start, count: m.metadata.pcRange.count }))
            .sort((a, b) => a.start - b.start);

        const groups = [];
        for (const it of items) {
            const g = groups[groups.length - 1];
            if (g && it.start <= g.end + 4096 && (it.start + it.count - g.start) <= COLUMN_FETCH_MAX_POINTS) {
                g.items.push(it);
                g.end = Math.max(g.end, it.start + it.count);
            } else {
                groups.push({ start: it.start, end: it.start + it.count, items: [it] });
            }
        }

        const url = this._columnUrl(name);
        for (const g of groups) {
            const segments = g.items.map(it => ({ key: it.mesh.metadata.nodeInfo.name, from: it.start - g.start, count: it.count }));
            this._pool.request({ type: 'column', url, start: g.start, count: g.end - g.start, kind: col.type, segments })
                .then(result => {
                    if (this._disposed) return;
                    for (const seg of result.segments) {
                        const mesh = this.loadedNodes.get(seg.key);
                        if (!mesh || mesh.metadata.featureName !== name || mesh.metadata.featureEpoch !== epoch) continue;
                        if (seg.values.length !== mesh.metadata.nodeInfo.numPoints) continue;
                        mesh.updateVerticesData('featureValue', seg.values);
                        mesh.metadata.featureReady = true;
                    }
                })
                .catch(err => console.error(`❌ Failed to load column ${name}:`, err));
        }
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
     * (Feature colours are drawn by the feature shader from the `featureValue` buffer, the vertex
     * colours always hold the classification / original colours.)
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

        const tmpVecColorMode = new BABYLON.Vector3();

        for (let i = 0; i < numPoints; i++) {
            // 1. Apply base color
            const hasClass = classIds && classIds[i] > 0 && classColors;
            if (this.colorMode === "classification" && hasClass) {
                this._writeBlendedClassColor(colors, i, originalColors, classColors);
            } else if (originalColors) {
                colors[i * 4] = originalColors[i * 4];
                colors[i * 4 + 1] = originalColors[i * 4 + 1];
                colors[i * 4 + 2] = originalColors[i * 4 + 2];
                colors[i * 4 + 3] = originalColors[i * 4 + 3];
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
        }
        mesh.setVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
    }

    getStats() {
        return { ...this.stats, loadedNodes: this.loadedNodes.size, loadedPoints: this._loadedPoints };
    }

    getRoot() {
        return this.rootTransform;
    }

    dispose() {
        this.detachCamera();
        this._cameraForObserver = null;
        if (this._renderObserver) { this.scene.onBeforeRenderObservable.remove(this._renderObserver); this._renderObserver = null; }
        this._disposed = true;

        if (this._cleanupIntervalId !== null) {
            clearInterval(this._cleanupIntervalId);
            this._cleanupIntervalId = null;
        }
        if (this._updateTimer !== null) { clearTimeout(this._updateTimer); this._updateTimer = null; }
        if (this._columnFetchTimer !== null) { clearTimeout(this._columnFetchTimer); this._columnFetchTimer = null; }
        this._columnFetchQueue.clear();
        this._pendingLoads = [];

        for (const mesh of this.loadedNodes.values()) {
            mesh.dispose();
        }
        this.loadedNodes.clear();
        this.activeNodes.clear();
        this.loadingNodes.clear();
        this.rootTransform.dispose();
        if (this._pool) { this._pool.dispose(); this._pool = null; }
        if (this._baseMat) { this._baseMat.dispose(); this._baseMat = null; }
        if (this._featureShaderMat) {
            this._featureShaderMat.dispose();
            this._featureShaderMat = null;
        }
    }

    // ========== SHADERS: CONTINUOUS LOD ==========

    /**
     * Shared by the colour and the feature vertex shaders. The point is kept while the spacing of its
     * level (lodInfo.x, world units) projects to >= DRAW_MIN * T pixels, T = pcLod.z (point size x
     * budget factor); between DRAW_MIN*T and T it is drawn smaller. Pixels per unit:
     *   perspective  ppu = k / w   (w = view depth),   orthographic  ppu = k       (k = P[1][1] * H / 2)
     * which is exactly what the loader uses to decide what to fetch. NaN positions (hidden points) give
     * a NaN ratio, which fails the test and is dropped.
     *   pcLod  = (k, perspective flag, threshold px, -)
     *   pcLod2 = (adaptive point size flag, max point size px, debug mode, -)
     */
    static _lodVertexCommon = `
        precision highp float;
        attribute vec3 position;
        attribute vec2 lodInfo;
        uniform mat4 worldViewProjection;
        uniform vec4 pcLod;
        uniform vec4 pcLod2;
        uniform vec3 pcMeshColor;

        // false: the point is dropped (outside the clip volume, size 0)
        bool pcLodVertex(out float ratio) {
            vec4 p = worldViewProjection * vec4(position, 1.0);
            float ppu = pcLod.y > 0.5 ? pcLod.x / max(p.w, 1e-6) : pcLod.x;
            float T = pcLod.z;
            ratio = lodInfo.x * ppu / T;
            if (!(ratio >= ${DRAW_MIN_RATIO.toFixed(2)})) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                gl_PointSize = 0.0;
                return false;
            }
            float keep = clamp((ratio - ${DRAW_MIN_RATIO.toFixed(2)}) / ${(1 - DRAW_MIN_RATIO).toFixed(2)}, 0.0, 1.0);
            float size = T;
            // Adaptive size: where even the full data is sparser than the point size, grow the points
            if (pcLod2.x > 0.5) size = max(size, lodInfo.y * ppu);
            size *= 0.5 + 0.5 * keep;
            gl_Position = p;
            gl_PointSize = clamp(size, 1.0, max(pcLod2.y, 1.0));
            return true;
        }

        vec3 pcHeat(float ratio) {
            float t = clamp(log2(max(ratio, 1.0)) / 4.0, 0.0, 1.0);
            return vec3(t, 1.0 - abs(2.0 * t - 1.0), 1.0 - t);
        }
    `;

    static _baseVertexShader = ChunkedPointCloudLoader._lodVertexCommon + `
        attribute vec4 color;
        varying vec4 vColor;
        void main() {
            float ratio;
            if (!pcLodVertex(ratio)) { vColor = vec4(0.0); return; }
            vColor = color;
            if (pcLod2.z > 0.5 && pcLod2.z < 1.5) vColor = vec4(pcMeshColor, color.a);
            else if (pcLod2.z > 1.5) vColor = vec4(pcHeat(ratio), color.a);
        }
    `;

    static _baseFragmentShader = `
        precision highp float;
        varying vec4 vColor;
        void main() {
            if (vColor.a < 0.1) discard;     // hidden segments carry alpha 0
            gl_FragColor = vColor;
        }
    `;

    // ========== FEATURE SHADER ==========

    static _featureVertexShader = ChunkedPointCloudLoader._lodVertexCommon + `
        attribute float featureValue;
        varying float vFeature;
        void main() {
            float ratio;
            vFeature = featureValue;
            pcLodVertex(ratio);
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
        BABYLON.Effect.ShadersStore['pcFeatureVertexShader'] = ChunkedPointCloudLoader._featureVertexShader;
        BABYLON.Effect.ShadersStore['pcFeatureFragmentShader'] = ChunkedPointCloudLoader._featureFragmentShader;
        const mat = new BABYLON.ShaderMaterial('pcFeatureMat', this.scene, 'pcFeature', {
            attributes: ['position', 'featureValue', 'lodInfo'],
            uniforms: ['worldViewProjection', 'pcLod', 'pcLod2', 'pcMeshColor', 'fmin', 'fmax', 'colormap', 'predictionDiscrete', 'discreteFilterEnabled', 'discreteFilterValue'],
        });
        mat.pointsCloud = true;
        mat.disableLighting = true;
        mat.setVector3('pcMeshColor', new BABYLON.Vector3(1, 1, 1));
        mat.setFloat('fmin', 0.0);
        mat.setFloat('fmax', 1.0);
        mat.setInt('colormap', this._colormapId);
        mat.setInt('predictionDiscrete', 0);
        mat.setInt('discreteFilterEnabled', 0);
        mat.setFloat('discreteFilterValue', 0.0);
        mat.transparencyMode = BABYLON.Material.MATERIAL_ALPHATEST;
        mat.alphaCutOff = 0.1;
        this._featureShaderMat = mat;
        this._lodU = null;
        this._updateLodUniforms();
        this._attachDebugBind(mat);
        return mat;
    }

    _applyFeatureShaderMode(enable, featureName) {
        const shaderMat = this._getOrCreateFeatureShaderMaterial();
        const isPredictionFeature = typeof featureName === 'string' && featureName.toLowerCase() === 'prediction';
        if (featureName && this.featureAttributes.has(featureName)) {
            const range = this.getFeatureRange(featureName);
            shaderMat.setFloat('fmin', this._featureRangeMin !== null ? this._featureRangeMin : range.min);
            shaderMat.setFloat('fmax', this._featureRangeMax !== null ? this._featureRangeMax : range.max);
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
        for (const mesh of this.loadedNodes.values()) {
            if (enable) {
                if (!mesh.metadata._origMaterial) mesh.metadata._origMaterial = mesh.material;
                this._ensureFeatureBuffer(mesh);
                mesh.material = shaderMat;
            } else {
                if (mesh.metadata._origMaterial) {
                    mesh.material = mesh.metadata._origMaterial;
                    mesh.metadata._origMaterial = null;
                }
                // Only the active column is kept in memory
                if (mesh.metadata.featureName) {
                    mesh.metadata.featureName = null;
                    mesh.metadata.featureReady = false;
                    if (mesh.getVertexBuffer('featureValue')) mesh.removeVerticesData('featureValue');
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
        if (this._featureShaderMat && this.colorMode.startsWith('feature:')) {
            const featureName = this.colorMode.slice(8);
            if (this.featureAttributes.has(featureName)) {
                const range = this.getFeatureRange(featureName);
                this._featureShaderMat.setFloat('fmin', range.min);
                this._featureShaderMat.setFloat('fmax', range.max);
            }
        }
    }

    setColormap(id) {
        this._colormapId = id;
        if (this._featureShaderMat) this._featureShaderMat.setInt('colormap', id);
    }

    // ========== MESH CREATION ==========

    /**
     * Base material shared by every node: vertex colours (alpha 0 hides cut segments) and the
     * continuous LOD of the vertex shader (_lodVertexCommon).
     */
    _getBaseMaterial() {
        if (this._baseMat) return this._baseMat;
        BABYLON.Effect.ShadersStore['pcBaseVertexShader'] = ChunkedPointCloudLoader._baseVertexShader;
        BABYLON.Effect.ShadersStore['pcBaseFragmentShader'] = ChunkedPointCloudLoader._baseFragmentShader;
        const mat = new BABYLON.ShaderMaterial('mat_pc_base', this.scene, 'pcBase', {
            attributes: ['position', 'color', 'lodInfo'],
            uniforms: ['worldViewProjection', 'pcLod', 'pcLod2', 'pcMeshColor']
        });
        mat.pointsCloud = true;
        mat.disableLighting = true;
        mat.setVector3('pcMeshColor', new BABYLON.Vector3(1, 1, 1));
        // The fragment shader discards alpha < 0.1 itself; ALPHATEST keeps the draw out of the sorted pass
        mat.transparencyMode = BABYLON.Material.MATERIAL_ALPHATEST;
        mat.alphaCutOff = 0.1;
        this._baseMat = mat;
        this._lodU = null;
        this._updateLodUniforms();
        this._attachDebugBind(mat);
        return mat;
    }

    /** Per-mesh debug colour (chunk / level views): written on the effect right before each draw. */
    _attachDebugBind(mat) {
        mat.onBindObservable.add((mesh) => {
            if (this.lodDebug !== 'chunk' && this.lodDebug !== 'level') return;
            const info = mesh.metadata?.nodeInfo;
            if (!info) return;
            const c = lodDebugColor(this.lodDebug === 'chunk' ? info.chunkId * 0.6180339887 : (info.seqFrom === 0 ? 0 : this.chunks[info.chunkId].stack.length + 1) * 0.17);
            const eff = mat.getEffect();
            if (c && eff) eff.setFloat3('pcMeshColor', c[0], c[1], c[2]);
        });
    }

    /**
     * Builds the BabylonJS mesh of a virtual node from the arrays decoded by the worker
     * (`positions` local Float32 relative to boundingBox.min, `colors` RGBA Float32, `pointIds`).
     * Persistent selection / classification / cut state is replayed on the new points exactly
     * as before: this is the only place where the data layer meets the editing logic.
     */
    _createMeshFromDecoded(node, decoded) {
        const positions = decoded.positions;
        const colors = decoded.colors;
        const pointIds = decoded.pointIds;
        const numPoints = positions.length / 3;
        node.numPoints = numPoints;


        // Save original positions so they can be hidden (by assigning NaN)
        // and restored later, bypassing any bugs/limitations of BabylonJS point
        // cloud materials regarding alpha compositing.
        const originalPositions = new Float32Array(positions);

        // Save originalColors HERE — after RGB decode but BEFORE applying
        // selection or classification. This ensures originalColors always contains
        // only the real point cloud colors.
        const originalColors = new Float32Array(colors);

        const segmentDebug = {
            canonicalRestored: 0,
            fallbackResolved: 0,
            unresolved: 0,
            invalidPointIds: 0,
            hidden: 0,
            segmentCounts: {}
        };

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

                if (this._pointSegmentMap && (pid < 0 || pid >= this._pointSegmentMap.length)) {
                    segmentDebug.invalidPointIds++;
                }

                // 1. Canonical map lookup (source of truth for points seen at cut time).
                //    Preserved across LOD unload/reload cycles — prevents the geometry-
                //    based fallback from re-assigning overlap points to a newer segment.
                const canonical = this._readPointSegment(pid);
                if (canonical !== null) {
                    resolvedSegId = canonical;
                    segmentDebug.canonicalRestored++;
                    if (pid === this.debugWatchPointId) {
                        this._debugSegment('watch-node-restore', { node: node.name, pid, segmentId: resolvedSegId, source: 'canonical' });
                    }
                }

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
                        segmentDebug.hidden++;
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
                        segmentDebug.fallbackResolved++;
                        if (pid === this.debugWatchPointId) {
                            this._debugSegment('watch-node-restore', { node: node.name, pid, segmentId: seg.segmentId, source: 'geometry-fallback' });
                        }
                        if (!seg.visible) {
                            positions[3 * j] = NaN;
                            positions[3 * j + 1] = NaN;
                            positions[3 * j + 2] = NaN;
                            segmentDebug.hidden++;
                        }
                    } else {
                        segmentIds[j] = 0;
                        segmentDebug.unresolved++;
                        if (!this.mainCloudVisible) {
                            positions[3 * j] = NaN;
                            positions[3 * j + 1] = NaN;
                            positions[3 * j + 2] = NaN;
                            segmentDebug.hidden++;
                        }
                    }
                }
                const key = String(segmentIds[j]);
                segmentDebug.segmentCounts[key] = (segmentDebug.segmentCounts[key] || 0) + 1;
            }
        }

        // With no cut history, all points are implicitly in the main segment.
        if (Object.keys(segmentDebug.segmentCounts).length === 0) {
            segmentDebug.segmentCounts['0'] = numPoints;
        }

        // Verify the CPU-side visibility mask before it is uploaded to Babylon.
        // A non-zero value proves that the inconsistency exists before rendering.
        segmentDebug.maskMismatches = 0;
        if (this.debugSegmentation) {
            for (let j = 0; j < numPoints; j++) {
                const pointSeg = segmentIds[j] || 0;
                const entry = this.cutHistory.find(e => e.segmentId === pointSeg);
                const shouldHide = pointSeg === 0 ? !this.mainCloudVisible
                    : pointSeg === this._deletedSegmentId ? true
                        : entry ? !entry.visible : false;
                const isHidden = Number.isNaN(positions[j * 3]);
                if (isHidden !== shouldHide) segmentDebug.maskMismatches++;
            }
        }

        this._debugSegment('node-ready', {
            node: node.name,
            level: node.level,
            points: numPoints,
            ...segmentDebug,
            mainCloudVisible: this.mainCloudVisible,
            visibleSegments: this._currentVisibleSegmentIds()
        });


        // Create BabylonJS mesh
        const mesh = new BABYLON.Mesh(`pc_${node.name}`, this.scene);
        const vertexData = new BABYLON.VertexData();
        vertexData.positions = positions;
        vertexData.colors = colors;
        // updatable=true is REQUIRED to update visibility and colors dynamically
        vertexData.applyToMesh(mesh, true);
        // Per-point spacing of the continuous LOD (pointcloud-lod.js): read by the vertex shader
        if (decoded.lodInfo) mesh.setVerticesData("lodInfo", decoded.lodInfo, false, 2);

        mesh.material = this._getBaseMaterial();
        mesh.hasAlpha = true;

        mesh.parent = this.rootTransform;
        mesh.isVisible = false;
        mesh.isPickable = true;

        mesh.metadata = {
            nodeInfo: {
                name: node.name, seqFrom: node.seqFrom, seqTo: node.seqTo,
                numPoints, chunkId: node.chunk.id
            },
            originalPositions,
            originalColors,
            classIds,
            classColors,
            segmentIds,
            pointIds,
            // Where this node lives in geom.bin (and in every column): offset/length in points
            pcRange: { start: node.start, count: numPoints },
            pointCloudNode: true,
            // Nominal spacing / bounds of the node (levels from..to of one chunk)
            nodeSpacing: node.spacing,
            nodeBoundingBox: node.boundingBox,
            segmentRevision: this._segmentRevision,
            lastUsedTick: this._tick,
            featureName: null,
            featureEpoch: -1,
            featureReady: false
        };

        this.loadedNodes.set(node.name, mesh);
        this._loadedPoints += numPoints;
        this.stats.loadedNodes = this.loadedNodes.size;
        this.stats.loadedPoints = this._loadedPoints;

        if (this.colorMode.startsWith('feature:') && this.featureAttributes.has(this.colorMode.slice(8))) {
            mesh.metadata._origMaterial = mesh.material;
            mesh.material = this._getOrCreateFeatureShaderMaterial();
            this._ensureFeatureBuffer(mesh);
        }
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
        if (anyAssigned) this._bumpSegmentRevision('cut', { segmentId, assigned: total });
        return anyAssigned ? { segmentId, count: total } : null;
    }

    _setMeshPositionsAndNotify(mesh, positions) {
        // VertexData.applyToMesh(mesh, true) creates an updatable position buffer.
        // Update that existing GPU buffer instead of replacing its vertex data on
        // every CUT/LOD visibility change; replacement can leave a stale buffer
        // bound for a render pass while meshes are rapidly toggled by the LOD loop.
        if (typeof mesh.updateVerticesData === 'function') {
            mesh.updateVerticesData(BABYLON.VertexBuffer.PositionKind, positions, false, false);
        } else {
            mesh.setVerticesData(BABYLON.VertexBuffer.PositionKind, positions, true);
        }
        this._debugSegment('gpu-position-upload', {
            node: mesh.metadata?.nodeInfo?.name,
            points: positions.length / 3
        });
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

        this._bumpSegmentRevision('visibility', { segmentId, visible });
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

    /** Enable concise, structured segmentation diagnostics at runtime. */
    setSegmentationDebug(enabled = true, watchPointId = null) {
        this.debugSegmentation = !!enabled;
        this.debugWatchPointId = Number.isInteger(watchPointId) ? watchPointId : null;
    }

    _debugSegment(event, details = {}) {
        if (!this.debugSegmentation) return;
        // Segmentation diagnostics are intentionally silenced by default.
        // Toggle behavior can be reintroduced here if needed in future.
        void event;
        void details;
    }

    _bumpSegmentRevision(operation, details = {}) {
        this._segmentRevision++;
        this._debugSegment('state-change', { operation, ...details });
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
        if (!this._pointSegmentMap) return;
        if (pid < 0 || pid >= this._pointSegmentMap.length) {
            this._debugSegment('invalid-point-id', { pid, segmentId, mapLength: this._pointSegmentMap?.length ?? 0 });
            return;
        }
        this._pointSegmentMap[pid] = (segmentId === this._deletedSegmentId) ? SEG_DELETED : segmentId;
        if (pid === this.debugWatchPointId) {
            this._debugSegment('watch-write', { pid, segmentId });
        }
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
        if (totalAssigned > 0) this._bumpSegmentRevision('assign', { segmentId, assigned: totalAssigned });
        return totalAssigned;
    }

    /**
     * Build and return the buffer [seg+1, class] (2 bytes per POINT_ID) for every point of the
     * entire cloud that belongs to a mapped segment.
     *
     * Points of the nodes currently loaded are read from their meshes (that is where the live
     * state is). All the others are streamed from geom.bin in contiguous blocks of ~8 MB
     * (positions + POINT_ID only, decoded in a worker) and resolved with the canonical
     * segment/class maps, falling back to the chronological geometric replay.
     */
    async exportAllTrainingData(segmentNameMap) {
        const totalPoints = this.metadata.points;
        const buffer = new Uint8Array(totalPoints * 2); // Interleaved: [segId, classId, segId, classId, ...]
        // 1 = already handled through a loaded mesh, 2 = handled from geom.bin
        const handled = new Uint8Array(totalPoints);

        // Identify which segments the user actually cares about (including segment 0)
        const requestedIds = Object.keys(segmentNameMap).map(id => parseInt(id, 10));
        if (requestedIds.length === 0) return null;

        const includeSegmentZero = segmentNameMap[0] !== undefined;

        // Helper: is this chunk worth visiting?
        const shouldVisit = (chunk) => {
            if (includeSegmentZero) return true;
            return this.cutHistory.some(region => {
                if (!segmentNameMap[region.segmentId]) return false;
                return (chunk.min[0] <= region.maxX && chunk.max[0] >= region.minX &&
                    chunk.min[1] <= region.maxY && chunk.max[1] >= region.minY &&
                    chunk.min[2] <= region.maxZ && chunk.max[2] >= region.minZ);
            });
        };

        // ---- 1. loaded nodes: live mesh state
        for (const mesh of this.loadedNodes.values()) {
            const chunk = this.chunks[mesh.metadata?.nodeInfo?.chunkId];
            if (!chunk || !shouldVisit(chunk)) continue;
            if (!mesh.metadata.pointIds || !mesh.metadata.segmentIds) continue;
            const segmentIds = mesh.metadata.segmentIds;
            const classIds = mesh.metadata.classIds || new Int32Array(mesh.metadata.pointIds.length);
            const pointIds = mesh.metadata.pointIds;
            for (let i = 0; i < pointIds.length; i++) {
                const pid = pointIds[i];
                if (pid < 0 || pid >= totalPoints || handled[pid]) continue;
                handled[pid] = 1;

                const segId = segmentIds[i] || 0;
                if (segmentNameMap[segId] !== undefined) {
                    buffer[pid * 2] = segId + 1;      // +1 so 0 stays "unannotated"
                    buffer[pid * 2 + 1] = classIds[i] || 0;
                }
            }
        }

        // ---- 2. everything else: contiguous ranges of geom.bin
        const ranges = [];
        for (const chunk of this.chunks) {
            if (!shouldVisit(chunk)) continue;
            const hp = chunk.headPoints;
            if (hp > 0) ranges.push([chunk.headOffset, chunk.headOffset + hp]);
            if (chunk.points - hp > 0) ranges.push([chunk.bodyOffset, chunk.bodyOffset + chunk.points - hp]);
        }
        ranges.sort((a, b) => a[0] - b[0]);
        const merged = [];
        for (const r of ranges) {
            const last = merged[merged.length - 1];
            if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]); else merged.push([r[0], r[1]]);
        }
        const blocks = [];
        for (const [a, b] of merged) {
            for (let s = a; s < b; s += SCAN_BLOCK_POINTS) blocks.push([s, Math.min(b, s + SCAN_BLOCK_POINTS)]);
        }

        const url = this._geomUrl();
        const pos = new BABYLON.Vector3();
        const chronologicalCuts = this._getChronologicalCuts();
        const hasClassMap = !!this._pointClassMap;
        let next = 0;
        const scanner = async () => {
            while (next < blocks.length) {
                const [s, e] = blocks[next++];
                const result = await this._pool.request({
                    type: 'geom', url, start: s, count: e - s, colors: false,
                    segments: [{ key: 0, from: 0, count: e - s }]
                });
                if (this._disposed) return;
                const { positions, pointIds } = result.segments[0];
                for (let i = 0; i < pointIds.length; i++) {
                    const pid = pointIds[i];
                    if (pid < 0 || pid >= totalPoints || handled[pid]) continue;
                    handled[pid] = 2;

                    let finalSegId = 0;
                    let finalClassId = 0;

                    // Canonical segment map first
                    if (this._pointSegmentMap && pid < this._pointSegmentMap.length) {
                        const canonical = this._readPointSegment(pid);
                        if (canonical !== null) {
                            finalSegId = canonical;
                        } else {
                            // Fallback to projection if not yet evaluated — same chronological
                            // replay the LOD path uses, so the export agrees with what the user
                            // sees on screen.
                            pos.set(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
                            const seg = this._resolvePointSegment(pos, chronologicalCuts);
                            finalSegId = seg ? seg.segmentId : 0;
                            // Cache it
                            this._writePointSegment(pid, finalSegId);
                        }
                    }

                    if (segmentNameMap[finalSegId] === undefined) continue;

                    // Class: canonical map, else the classification history replay
                    const stored = hasClassMap ? this._pointClassMap[pid] : 0;
                    if (stored !== 0 && stored !== 0xFF) {
                        finalClassId = stored;
                    } else if (this.classificationHistory.length > 0) {
                        pos.set(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
                        const cls = this._getPointClassification(pos);
                        finalClassId = cls ? cls.classId : 0;
                        if (finalClassId > 0 && hasClassMap) this._pointClassMap[pid] = finalClassId;
                    }

                    buffer[pid * 2] = finalSegId + 1; // +1 so 0 stays "unannotated"
                    buffer[pid * 2 + 1] = finalClassId;
                }
            }
        };
        await Promise.all(Array.from({ length: Math.min(3, blocks.length) }, scanner));

        return {
            buffer: buffer,
            segmentMap: segmentNameMap
        };
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
     * Export the segment/class annotations to the server (working/annotations.bin).
     * Collects the segment/class buffer (2 bytes per POINT_ID: seg+1, class) and POSTs it
     * to /api/export-mapping/ — gzip-compressed with CompressionStream when available.
     * Returns { annotations_path, point_count, segmentMap } on success.
     */
    async exportAnnotations(segmentNameMap) {
        const exportResult = await this.exportAllTrainingData(segmentNameMap);
        if (!exportResult || !exportResult.buffer) {
            throw new Error('Failed to export training data');
        }

        let body = new Blob([exportResult.buffer], { type: 'application/octet-stream' });
        let encoding = null;
        if (typeof CompressionStream !== 'undefined') {
            try {
                body = await new Response(body.stream().pipeThrough(new CompressionStream('gzip'))).blob();
                encoding = 'gzip';
            } catch (e) {
                console.warn('[PointCloudLoader] gzip compression failed, sending raw annotations:', e);
                body = new Blob([exportResult.buffer], { type: 'application/octet-stream' });
            }
        }

        const formData = new FormData();
        formData.append('buffer', body, 'annotations.bin');
        formData.append('point_count', this.metadata.points.toString());
        if (encoding) formData.append('encoding', encoding);

        const response = await fetch('/api/export-mapping/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || 'Failed to save annotations');
        }

        const data = await response.json();
        console.log('[PointCloudLoader] annotations.bin saved:', data.annotations_path, `(${data.point_count} points)`);

        return {
            annotations_path: data.annotations_path,
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
        if (totalRemoved > 0) this._bumpSegmentRevision('remove-from-segment', { segmentId, removed: totalRemoved });
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
        if (anyAssigned) this._bumpSegmentRevision('delete', { deleted: total });
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
        if (restored > 0) this._bumpSegmentRevision('restore-deleted', { restored });
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

        this._bumpSegmentRevision('merge', { targetSegmentId, sourceSegmentIds: sources, movedPoints });
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

} // END class ChunkedPointCloudLoader


// =====================================================================
// PUBLIC HELPER FUNCTIONS
// =====================================================================

/**
 * Load a chunked point cloud (runtime_data/working/pc).
 * options (besides the ChunkedPointCloudLoader ones):
 *   version      - cache-busting token appended as ?v= to the geometry requests
 *   initialState - state exported by a previous loader (see exportState()), applied before the first nodes
 *   preserveView - true when reloading: keep camera and outline entry untouched
 */
export async function loadPointCloud(basePath, scene, options = {}) {
    const loader = new ChunkedPointCloudLoader(scene, basePath, options);
    await loader.load();

    const camera = scene.activeCamera;
    if (camera) {
        if (!options.preserveView) {
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

            // The root is mirrored on X: the target has to be the centre in WORLD space
            camera.setTarget(BABYLON.Vector3.TransformCoordinates(localCenter, loader.rootTransform.getWorldMatrix()));
            camera.radius = radius;
            camera.minZ = 0.1;
            camera.maxZ = radius * 10;
        }

        loader.attachCamera(camera);
    }

    scene.pointCloudLoader = loader;

    // Notify main.js that the loader is ready, so it can populate the color menu
    window.dispatchEvent(new CustomEvent('pointcloud-loaded', { detail: { loader } }));

    // Features come from the columns: tell the UI which ones are available
    window.dispatchEvent(new CustomEvent('features-available', { detail: { names: loader.getFeatureList() } }));

    if (!options.preserveView) {
        if (window.__registerPointCloudInOutline) {
            window.__registerPointCloudInOutline(loader.rootTransform, "Point Cloud");
        }

        if (window.__frameCameraOnMesh) {
            window.__frameCameraOnMesh(scene.activeCamera, loader.rootTransform);
        }
    }

    // First LOD pass, once the camera is framed on the cloud: shows the overview and fetches the detail
    if (scene.activeCamera) loader.update(scene.activeCamera);

    console.log(`✅ PointCloudLoader ready: ${loader.loadedNodes.size} nodes loaded, ${loader.activeNodes.size} visible`);
    return loader.getRoot();
}

export function getPointCloudLoader(scene) {
    return scene.pointCloudLoader || null;
}
