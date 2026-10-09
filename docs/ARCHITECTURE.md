# Architecture

This document describes the technical architecture of the 3D Classify Viewer, covering
the system components, data flow, and integration between the frontend, backend, C++
processing pipeline, and ML subsystem.

## Table of Contents

1. [System Overview](#system-overview)
2. [Frontend](#frontend)
3. [Django Backend](#django-backend)
4. [C++ Processing Pipeline](#c-processing-pipeline)
5. [Python ↔ C++ Integration](#python--c-integration)
6. [ML Pipeline](#ml-pipeline)
7. [Out-of-core Server Pipeline](#out-of-core-server-pipeline)
8. [Browser Memory Model and Limits](#browser-memory-model-and-limits)
9. [Data Storage Layout](#data-storage-layout)

---

## System Overview

```
Browser (BabylonJS + chunked point cloud loader)
        │  REST API calls
        ▼
Django REST API (Python 3.10, Gunicorn)
        │  subprocess
        ├──────────────────────────────► C++ Binaries (/webapp/opt/)
        │  subprocess                         │
        └──────────────────────────────► ML Pipeline (RF)
                                              │
                         File System (runtime_data/) ◄──────────┘
                                 │
                    HTTP Range / serve
                                 │
                                 ▼
                    Browser (LOD streaming)
```

**Request lifecycle:**

1. A user action in the browser triggers a REST call to Django.
2. Django validates the request and delegates to `functions.py`.
3. `functions.py` launches the appropriate C++ binary or Python ML script via
   `subprocess.Popen`.
4. Stdout is streamed in real time to the Django process log.
5. On completion, the result (file path, status) is returned as JSON to the browser.
6. The browser fetches processed data files via dedicated serve endpoints with HTTP
   Range support for efficient streaming of large binary files.

---

## Frontend

**Key files:**

| File | Role |
|---|---|
| `static/viewer/js/main.js` | Application controller (~1 000 lines): initializes the BabylonJS scene, the point cloud loader, toolbar, and panels |
| `static/viewer/js/functions.js` | UI and data logic (~4 700 lines): selection tools, class registry, REST API calls, colormap management, and dynamic modals |
| `static/viewer/js/pointcloud-loader.js` | Chunked point cloud loader (LOD selection, memory budget, columns) with the selection / cut / class / segment logic |
| `static/viewer/js/pointcloud-worker.js` | Module workers that fetch and decode `geom.bin` / column ranges off the main thread |
| `templates/viewer/viewer_page.html` | Main HTML shell; injects `RUNTIME_DATA_URL` and `RUNTIME_DATA_PATH_PREFIX` from the Django template context |

**Rendering stack:**

- **BabylonJS** renders the 3D scene, manages the camera, lighting, and any additional
  mesh or annotation layers.
- The **chunked point cloud loader** streams the cloud (`geom.bin` and attribute columns)
  with HTTP Range requests against the `/pointcloud-data/` endpoint. It picks the levels
  of detail of every spatial chunk from the projected point spacing, culls chunks outside
  the view frustum, keeps a point budget in memory with LRU eviction, and decodes data in
  a pool of workers (format: [POINTCLOUD_FORMAT.md](POINTCLOUD_FORMAT.md)).
- **Selection tools** (rectangle, lasso, polygon) are implemented via mouse event
  listeners and an SVG overlay rendered on top of the 3D canvas.

**State management:** Application state (class registry, segment map, active mode,
annotation buffer) is held in the global scope and ES6 module exports. There is no
client-side persistence — all data lives on the server under `runtime_data/`.

**Operating modes:**

- **Training mode:** Enables point selection tools, class assignment, and model training
  controls. The context menu is only available in this mode.
- **Classify mode:** Enables model loading, inference triggering, and prediction
  visualization.

---

## Django Backend

**Settings** (`classifyViewer/settings.py`):

| Setting | Value | Description |
|---|---|---|
| `DEBUG` | `False` | Production mode |
| `ALLOWED_HOSTS` | `['0.0.0.0', 'localhost']` | Accepted hosts |
| `RUNTIME_DATA_ROOT` | `BASE_DIR / 'runtime_data'` | Root directory for all runtime files |
| `RUNTIME_DATA_URL` | `/runtime-data/` | URL prefix for runtime file serving |
| `DATA_UPLOAD_MAX_MEMORY_SIZE` | 5 GB | Maximum non-file request body size |
| `FILE_UPLOAD_MAX_MEMORY_SIZE` | 10 MB | Larger uploads are spooled to a temporary file instead of RAM |
| Gunicorn `timeout` | `0` (disabled) | No timeout; required for long-running C++ operations |

**Static files:** Collected at image build time via `collectstatic` and served by
WhiteNoise middleware without requiring a separate web server.

**URL structure:**

```
/                         → viewer_page.html  (main application)
/documentation/           → docs_page.html    (in-app documentation)
/pointcloud-data/<path>   → HTTP Range-capable binary file server
/runtime-data/<path>      → Runtime file server (LAS, JSON, annotations.bin)
/api/*                    → REST API endpoints
/<operation>/             → Processing pipeline endpoints
```

See [API_REFERENCE.md](API_REFERENCE.md) for full endpoint documentation.

---

## C++ Processing Pipeline

Seven pre-compiled binaries are deployed to `/webapp/opt/` inside the container. They
are invoked from Python via `subprocess.Popen`, communicate through file paths and CLI
arguments, and write all output to `runtime_data/working/`.

| Binary | Purpose | Input | Output |
|---|---|---|---|
| `feature_extraction_viewer_gpu` | GPU-accelerated per-point geometric feature extraction (tiled, Thrust) | LAS | LAS with Extra Bytes |
| `feature_extraction_viewer_cpu` | CPU-only feature extraction fallback (OpenMP) | LAS | LAS with Extra Bytes |
| `subsample_pc` | Voxel-grid downsampling | PLY / LAS | Subsampled file |
| `mesh2pc` | Surface mesh → point cloud (uniform sampling) | GLB / GLTF / OBJ | LAS |
| `ply2las` | PLY → LAS format conversion | PLY | LAS |
| `split_las_by_binary` | Split LAS into per-segment files using the `annotations.bin` store | LAS + `annotations.bin` | Multiple LAS files |
| `check_point_id` | Validate and normalize POINT_ID to canonical 0-based indexing | LAS | LAS (validated) |

**Build dependencies:** PCL 1.9, PDAL 2.7.1, GDAL 3.6.2, Open3D 0.19.0, LASzip,
CGAL, Boost, CUDA 11.8 + Thrust (GPU binaries only).

See [BINARIES.md](BINARIES.md) for full CLI reference per binary.

### `annotations.bin` format

`annotations.bin` (`runtime_data/working/annotations.bin`) stores the user annotations
(segments and classes). It is a raw buffer with **2 bytes per point, indexed by `POINT_ID`**
(no header):

```
byte[2*pid]     segment_id + 1   (0 = point not annotated)
byte[2*pid + 1] class_id         (0 = no class assigned)
```

The browser builds the buffer (`ChunkedPointCloudLoader.exportAnnotations`), gzips it with
`CompressionStream` when available and POSTs it to `/api/export-mapping/`. The server
streams it to a temp file, patches the existing store with numpy (only points with
`segment_id != 0` in the new buffer overwrite the stored ones, because the client only
exports the requested segments) and moves it atomically into place.
`split_las_by_binary` reads it with O(1) lookup per `POINT_ID`.

There is no unified feature store: the per-point features are Extra Bytes of
`features.las`, mirrored into per-attribute columns of the chunked point cloud (see
[Visualization and the commit model](#visualization-and-the-commit-model)).

### Visualization and the commit model

`features.las` is the single source of truth. For the browser it is converted **once** (at
import) by `las2pc` into the chunked point cloud `working/pc/` (full description in
[POINTCLOUD_FORMAT.md](POINTCLOUD_FORMAT.md)):

- `geom.bin` — 20 B/point (quantized XYZ, RGB, `POINT_ID`), grouped in spatial chunks of at
  most 250 000 points; inside each chunk the points are ordered by stratified levels of detail
  (a random subsample of any level prefix is uniform). The first Range request of a cloud is
  the whole overview (`head` block), then the loader asks one contiguous range per chunk for
  the missing levels.
- `col/<name>.bin` — one column per attribute (features, `prediction`) in the same point
  order; `meta.json` holds their `min`/`max` and a `version`. The loader downloads only the
  column of the feature being displayed, for the meshes it has loaded.

**Commit model.** Selections, cuts and manual classifications are client-side (history and
per-`POINT_ID` maps). The operations that change `features.las` no longer rebuild or reload
the cloud, they only rewrite columns (`pc_columns.py`, joined by `POINT_ID`):

| Operation | Server | Browser |
|---|---|---|
| Import | `las2pc` + all columns, atomic swap `pc_tmp` → `pc` | `loadPointCloud` |
| Feature calculation | `/feature_extraction/`, then the new columns (`only`, `prune`) | `refreshColumns()` |
| RF classification | `--prediction <classified.las>` → `prediction` column (255 = no prediction) | `refreshColumns()` |
| Backup restore | `--drop-all` + columns of the backup (rebuild only if the point count differs) | `refreshColumns()` |

`ChunkedPointCloudLoader.refreshColumns()` re-reads `meta.json`, updates the feature list,
invalidates the columns whose `version` changed and re-downloads the active one. The loader
is not disposed: selections, cuts, classes, colormap and camera stay as they are. The
geometry is rebuilt (and the loader reloaded with `reloadPointCloudPreservingState`) only if
its point count or build changes. `POINT_ID` is the only join key between a LAS and the
geometry — the GPU feature extractor rewrites the records tile by tile, so the position in
the file means nothing.

**Atomic updates.** The geometry is built in `working/pc_tmp/` and swapped with `pc/` only on
success (`functions.build_pointcloud`); a column is written to `col/<name>.bin.tmp` and moved
with `os.replace`, then `meta.json` is replaced the same way. The geometry `version` and every
column `version` go in the URL as `?v=<token>`, which makes the Range responses immutable and
cacheable for a year; `meta.json` is always fetched with `cache: 'no-store'`.

**Classification fallback.** If the classified LAS has no `POINT_ID` belonging to the current
cloud (the column writer answers `409`), the classified LAS is converted into its own
`working/pc_classified/` and loaded with `reloadPointCloudPreservingState(scene, version,
'pc_classified')`, as the former pipeline did.

**State preservation.** `reloadPointCloudPreservingState(scene, version)` builds the new
loader with the old loader's `exportState()` (cut/classification/selection history,
per-point segment and class maps, segment visibility, colour mode, point size, colormap,
feature range, ...), applied before the first nodes are created, and disposes the old
loader only when the new one is ready. Outline entries, class registry and camera are
untouched. Limits: if the new point cloud has a different `boundingBox` only the AABB
histories are translated (screen-space selections cannot be); if the point count differs
(classified segment or sampled cloud) the per-point maps are not transferred.

**Server throughput.** `serve_range_file` positions a `FileResponse` at the requested byte
and sets `Content-Length` to the slice length, so Gunicorn serves it with `sendfile()`
(zero copy); `gunicorn.conf.py` runs 1 worker (the `JobManager` singleton must be reachable by
`/stop_process/`) with 8 threads, because long calculation endpoints keep a thread busy while the
viewer needs up to 6 parallel Range requests.

---

## Python ↔ C++ Integration

All C++ binary invocations are managed by the `JobManager` class in `functions.py`.

**`JobManager` responsibilities:**

- Launch a subprocess with `subprocess.Popen`, capturing combined stdout/stderr.
- Store a reference to the running process so `stop_process` can terminate it at any
  time.
- On Linux, use `os.setsid` to create a new process group, allowing `SIGTERM` to
  propagate to child processes.
- Stream stdout to the Django log in real time, handling both `\n` and `\r` progress
  updates (e.g. tqdm progress bars).
- On subprocess failure (non-zero exit code, excluding `SIGTERM`), parse the last
  stdout line as an error message and raise `RuntimeError`.

**`stop_process` endpoint:** Sends `SIGTERM` to the active process group on Linux, or
calls `taskkill /F /T` on Windows. Used to cancel long-running operations such as
feature extraction or model training.

---

## ML Pipeline

### Training (`utils_functions/RF_training.py`)

1. Reads `features.las` (LAS with Extra Bytes) and the binary annotation buffer
   (`annotations.bin`, 2 bytes per point).
2. Extracts feature column names from LAS VLR Extra Bytes metadata.
3. Splits annotated points into training and validation sets by segment ID.
4. Trains a Random Forest classifier using RAPIDS `cuRF` (GPU) if available, with
   automatic fallback to scikit-learn's `RandomForestClassifier` (CPU).
5. Saves `model.pkl` (serialized classifier) and a performance report (accuracy, F1,
   confusion matrix) to `runtime_data/models/<model_name>/`.

### Inference (`utils_functions/RF_classify.py`)

1. Loads the saved `model.pkl`.
2. Reads `features.las` and extracts the same feature columns used during training,
   matched by name from VLR Extra Bytes.
3. Runs `predict` and `predict_proba` to obtain per-point class labels and confidence
   scores.
4. Writes predictions and confidence values back to the LAS file as Extra Bytes.
5. The viewer does not reload anything: the `prediction` column of the current point cloud is written from the classified LAS (joined by `POINT_ID`, `255` = no prediction) and the loader picks it up as a feature. `POINT_ID` is read and written as uint32 separately from the float32 feature matrix, so ids above 16 777 216 survive.

**GPU acceleration:** cuML's `RandomForestClassifier` is a drop-in replacement for
scikit-learn's. Both training and inference attempt GPU execution first; any import
error or CUDA exception triggers a graceful fallback to the CPU implementation.

---

## Out-of-core Server Pipeline

The server side must work with a `features.las` that is **bigger than the memory of the container**. Every tool receives
`--memory-budget MB` (default 50 % of `min(cgroup limit, MemAvailable)`, one setting: `PIPELINE_MEMORY_BUDGET_MB`) and
`--temp-dir` (`PIPELINE_TEMP_DIR`, default `/tmp/pipeline_work`, never inside `runtime_data/`); see
[INSTALLATION.md](INSTALLATION.md#memory-scratch-space-and-storage).

**Canonical order.** After the geometry is built, `features.las` is rewritten **in the order of `pc/geom.bin`** (row *r* of the
LAS is point *r* of the geometry; `POINT_ID` is unchanged) and the backup is taken afterwards. Every later tool preserves the
order of the records it reads. This is what turns the column updates into sequential copies (no join) and lets a prediction on an
extracted segment (an ordered subset) be merged with `point_order.bin` in one pass. A LAS in another order is still accepted:
`pc_columns.py` falls back to an external join through bucket files (`POINT_ID` is always the key; `annotations.bin` stays
indexed by `POINT_ID`). Nothing else depends on the order of the records.

| Step | Tool | How it stays within the budget |
|---|---|---|
| Geometry + canonical LAS | `las2pc` | in memory (28 B/point + LAS in the page cache) when it fits, otherwise out-of-core: sequential count pass, one temp file per chunk, chunks processed by threads, `pwrite` at the final position; both give byte-identical files |
| Columns | `pc_columns.py` | LAS read once in blocks; `fast` / `merge` / `join` paths |
| Feature extraction | `feature_extraction_viewer_gpu` / `_cpu` | XY tiles sized from the budget (and the free VRAM): tile files with a buffer, one tile in memory (and on the GPU) at a time, record written at its input row; a tile that fails is split and retried |
| Classification | `RF_classify.py` | blocks of points; output = input records + `prediction` |
| Training | `RF_training.py` | only the selected features are loaded; stratified sampling by class above `--max_training_points` |
| Split / segment extraction | `split_las_by_binary` | one pass, no PDAL; output records = input records (+ `labels`), same order |
| Normals (`check_point_id`, `ply2las`, `subsample_pc` on LAS) | tile pipeline with a buffer of the search radius | the normals of one tile at a time; PLY read by blocks; voxel subsampling per tile aligned to the voxel grid |
| `mesh2pc` | — | the memory needed for the requested number of points is checked before starting |

Common primitives live in `testC++/ooc/` (header-only: `LasStreamReader`, `BucketWriter`, `PositionalWriter`, `TempWorkDir`,
memory budget, tile pipeline, normals) and `viewer/utils_functions/pipeline_common.py`. Large files are read with `pread` and
`posix_fadvise(DONTNEED)` behind the cursor and written with `pwrite` + `sync_file_range` (no `mmap(MAP_SHARED)` on files bigger than
the budget: dirty pages of a mapping count in the cgroup limit and cannot be freed before they are written).

**Stopping a job.** `/stop_process/` sends `SIGTERM` to the process group; each tool removes its scratch folder and its partial
outputs before exiting (the backend also removes `pc_tmp/` on any failure), so nothing is left behind and `runtime_data/` keeps
its previous state.

## Browser Memory Model and Limits

The point cloud itself is streamed in chunks (`geom.bin` ranges) and only the chunks that are visible are resident. The
**per-point maps indexed by `POINT_ID`** are not: `ChunkedPointCloudLoader` allocates, once per cloud,

| Array | Type | Size | Used for |
|---|---|---|---|
| `_pointSegmentMap` | `Uint16Array(points)` | 2 B/point | segment of every point (with sentinels) |
| `_pointClassMap` | `Uint8Array(points)` | 1 B/point | class of every point |
| `buffer`, `handled` (annotation export) | `Uint8Array(2 × points)`, `Uint8Array(points)` | 3 B/point, transient | `annotations.bin` before it is sent to the server |

That is **3 B/point resident and about 6 B/point while exporting**: 300 MB / 600 MB for 100 M points, 3 GB / 6 GB for 1 G points
in one tab (browsers limit a tab to a few GB; a typed array is also limited to 2³² elements). Up to a few hundred million points
this is comfortable; beyond that it is the next limit of the system (the server side no longer has one).

*Proposal (not implemented here, see the follow-ups):* index the maps by **row** (position in `geom.bin`, which is also the row of the
canonical LAS) instead of `POINT_ID`, allocate them **per chunk** the first time the chunk is touched (a chunk is at most 250 000
rows, so 750 KB), and let the server convert rows → `POINT_ID` when it writes `annotations.bin` with one sequential merge on
`point_order.bin`. The memory then follows the area the user works on, not the size of the cloud.

---

## Data Storage Layout

All runtime data is stored under `classifyViewer/runtime_data/` (configurable via
`RUNTIME_DATA_ROOT` in `settings.py`):

```
runtime_data/
├── working/
│   ├── <uploaded_file>.*           Original uploaded file
│   ├── features.las                Canonical feature LAS (source of truth for the pipeline)
│   ├── annotations.bin             User annotations (2 bytes per POINT_ID: segment+1, class)
│   ├── pointcloud_backup.las       Backup copy before feature re-extraction
│   ├── pc/                         Chunked point cloud (see POINTCLOUD_FORMAT.md), swapped atomically
│   │   ├── meta.json
│   │   ├── geom.bin                20 B/point: quantized XYZ, RGB, POINT_ID
│   │   ├── point_order.bin         POINT_ID in geom.bin order
│   │   └── col/<name>.bin          One column per feature / prediction
│   └── pc_classified/              Only if a classified LAS does not match pc/ (fallback)
└── models/
    └── <model_name>/
        ├── model.pkl               Serialized Random Forest model
        └── report.txt              Training performance metrics
```
