# API Reference

All endpoints are served by the Django backend at `http://localhost:8000`.

**Conventions:**

- **POST** requests with a JSON body use `Content-Type: application/json`.
- All JSON responses carry at minimum `{"status": "success"}` or
  `{"status": "error", "message": "..."}`.
- CSRF protection is disabled on API endpoints (`@csrf_exempt`); they are
  intended for same-origin browser clients.
- File paths in request bodies are relative to `BASE_DIR`
  (`classifyViewer/`) unless marked as absolute.

## Table of Contents

- [Page Routes](#page-routes)
- [Data Management](#data-management)
- [Processing Pipeline](#processing-pipeline)
- [ML Operations](#ml-operations)
- [Annotation and Export](#annotation-and-export)
- [Model Management](#model-management)
- [File Serving](#file-serving)

---

## Page Routes

### `GET /`

Returns the main viewer HTML page (`viewer_page.html`).

### `GET /documentation/`

Returns the in-app documentation HTML page (`docs_page.html`).

---

## Data Management

### `POST /api/upload-data/`

Upload a point cloud or mesh file to the working directory
(`runtime_data/working/`).

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | Point cloud file (`.ply`, `.las`, `.laz`, `.glb`, etc.) |

**Response:**

```json
{
  "message": "File uploaded successfully",
  "filename": "input.las",
  "rel_path": "input.las"
}
```

---

### `POST /api/clear-data/`

Delete all files in the working directory (`runtime_data/working/`). The
models directory is not affected.

**Request:** Empty POST body.

**Response:**

```json
{ "message": "Working directory cleared successfully" }
```

---

### `POST /api/backup-pointcloud/`

Create a backup copy of `features.las` as `pointcloud_backup.las` in the
working directory.

**Response:**

```json
{
  "status": "success",
  "message": "Point cloud backup created",
  "backup_path": "runtime_data/working/pointcloud_backup.las"
}
```

---

### `POST /api/restore-pointcloud-backup/`

Restore `features.las` from `pointcloud_backup.las` and delete the stored `annotations.bin`.
The geometry of the chunked point cloud (`working/pc`) and the `POINT_ID`s are the same as in
the backup, so the geometry is **not** rebuilt: every column is dropped and rewritten from the
backup (normally only the normals). If the number of points differs from the current cloud
the geometry is rebuilt with `las2pc` (`rebuilt: true`).

**Response:**

```json
{
  "status": "success",
  "message": "Point cloud restored from backup",
  "las_path": "runtime_data/working/features.las",
  "version":  "1760000000000",
  "rebuilt": false
}
```

---

## Processing Pipeline

### `POST /subsample_pc/`

Downsample a point cloud using a voxel-grid filter (one representative point
per voxel cell).

**Request body:**

```json
{
  "file_path":  "runtime_data/working/input.las",
  "out_path":   "runtime_data/working/subsampled.las",
  "voxel_size": 0.05
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Subsampling completed.",
  "output_file_path": "runtime_data/working/subsampled.las"
}
```

---

### `POST /mesh2pc/`

Convert a surface mesh (GLB, GLTF, OBJ) to a point cloud by uniform surface
sampling.

**Request body:**

```json
{
  "file_path":  "runtime_data/working/model.glb",
  "out_path":   "runtime_data/working/pointcloud.las",
  "num_points": 500000
}
```

| Field | Type | Description |
|---|---|---|
| `num_points` | `int` | Number of points to sample from the mesh surface |

**Response:**

```json
{ "status": "success", "message": "Mesh to Point Cloud completed." }
```

---

### `POST /ply2las/`

Convert a PLY point cloud to LAS format.

**Request body:**

```json
{
  "file_path": "runtime_data/working/input.ply",
  "out_path":  "runtime_data/working/output.las"
}
```

**Response:**

```json
{ "status": "success", "message": "PLY to LAS completed." }
```

---

### `POST /check_point_id/`

Validate and normalize the `POINT_ID` field in a LAS file to canonical 0-based
sequential indexing. Required before annotation operations.

**Request body:**

```json
{
  "input_path":  "runtime_data/working/input.las",
  "output_path": "runtime_data/working/output.las"
}
```

**Response:**

```json
{ "status": "success", "message": "CHECK POINT ID completed." }
```

---

### `POST /inspect_las_input/`

Read and return header metadata from a LAS file.

**Request body:**

```json
{ "file_path": "runtime_data/working/features.las" }
```

**Response:**

```json
{
  "status":      "success",
  "point_count": 1500000,
  "extra_dims":  ["anisotropy_0.5", "linearity_0.5", "planarity_0.5"],
  "bounds":      { "min": [x, y, z], "max": [x, y, z] }
}
```

---

### `POST /feature_extraction/`

Extract per-point geometric features at one or more radii. Uses the GPU binary
by default; falls back to the CPU binary when `use_gpu` is `false`.

**Request body:**

```json
{
  "input_filepath":  "runtime_data/working/features.las",
  "output_filepath": "runtime_data/working/features.las",
  "feature_list":    ["anisotropy", "linearity", "planarity", "sphericity"],
  "radius_list":     [0.5, 1.0, 2.0],
  "sampling":        0,
  "use_gpu":         true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `feature_list` | `string[]` | — | Names of geometric features to compute |
| `radius_list` | `float[]` | — | Neighborhood radii in scene units |
| `sampling` | `int` | `0` | Subsampling resolution; `0` disables subsampling |
| `use_gpu` | `bool` | `true` | `false` forces the CPU binary |

**Response:**

```json
{ "status": "success", "message": "Feature extraction completed." }
```

---

### `POST /api/build-pointcloud/`

Convert a LAS file into the chunked point cloud format (see
[POINTCLOUD_FORMAT.md](POINTCLOUD_FORMAT.md)): `las2pc` builds the geometry and
`pc_columns.py` writes one column per LAS Extra Byte. Everything is written into
`<output_filepath>_tmp`; on success the folder is swapped atomically with `<output_filepath>`
(on failure or user stop the previous point cloud is left untouched). The viewer uses
`runtime_data/working/pc`. This is needed at import (and when the number of points changes);
features and classification only rewrite columns.

**Request body:**

```json
{
  "input_filepath":  "runtime_data/working/features.las",
  "output_filepath": "runtime_data/working/pc"
}
```

**Response:**

```json
{ "status": "success", "message": "Point cloud built.", "version": "1760000000000" }
```

`version` is the geometry token. The client appends it as `?v=<version>` to the `geom.bin`
URL (every column has its own version in `meta.json`) so cached Range responses are never stale.

---

### `POST /api/update-pointcloud-columns/`

Rewrite attribute columns of an existing chunked point cloud without touching its geometry
(about 0.1 s per float32 column on a 5 M point cloud). The LAS is joined to the geometry by
`POINT_ID`, so the order of its records does not matter and points missing from it keep the
"missing" value (`NaN`; `255` for `prediction`).

**Request body** (all fields optional, at least one action is required):

```json
{
  "pc_dir":              "runtime_data/working/pc",
  "las_filepath":        "runtime_data/working/features.las",
  "only":                ["planarity_0_8"],
  "prune":               true,
  "prediction_filepath": "runtime_data/working/classify/classified.las",
  "drop_all":            false
}
```

| Field | Meaning |
|---|---|
| `las_filepath` (+ `only`) | Write the Extra Bytes of the LAS: all of them, or the names in `only` (Extra Bytes unknown to the cloud are always added) |
| `prune` | With `las_filepath`: drop the columns that are no longer Extra Bytes of the LAS |
| `prediction_filepath` | Write the `prediction` (uint8) column from a classified LAS; it may be a subset of the points (the others stay `255`). RF class ids must be `< 255` |
| `drop_all` | Delete every column first (applied before the other actions) |

**Response:** `{ "status": "success", "message": "Point cloud columns updated." }`.
If the classified LAS has no `POINT_ID` that belongs to the point cloud the answer is
`409 { "status": "mismatch", "message": "..." }` and nothing is written.

---

### `POST /split_las_by_binary/`

Split a LAS file into per-segment LAS files based on annotations stored in `annotations.bin`
(2 bytes per `POINT_ID`: `segment_id + 1`, `class_id`; see ARCHITECTURE.md).

**Request body:**

```json
{
  "las_path":              "runtime_data/working/features.las",
  "annotations_path":      "runtime_data/working/annotations.bin",
  "output_dir":            "runtime_data/working/segments",
  "exclude_unclassified":  false,
  "segment_names":         { "1": "training", "2": "validation" }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `annotations_path` | `string` | `runtime_data/working/annotations.bin` | Annotation store (error if missing) |
| `exclude_unclassified` | `bool` | `false` | Omit points with no class assigned |
| `segment_names` | `object` | `null` | Map segment IDs to output file names |

**Response:**

```json
{ "status": "success", "message": "Split LAS completed." }
```

---

### `POST /get_model_voxel_size/`

Retrieve the voxel size recorded in a model's training report file.

**Request body:**

```json
{ "model_dir": "runtime_data/models/my_model" }
```

**Response:**

```json
{ "status": "success", "voxel_size": 0.05 }
```

---

### `POST /stop_process/`

Send a termination signal to the currently running C++ or ML subprocess.

**Request:** Empty POST body.

**Response:**

```json
{ "status": "success", "message": "Process stopped successfully." }
```

---

## ML Operations

### `POST /launch_RF_training/`

Train a Random Forest classifier on the annotated point cloud.

**Request body:**

```json
{
  "model_name":    "my_model",
  "features_las":  "runtime_data/working/features.las",
  "labels_bin":    "runtime_data/working/labels_20240101_120000.bin",
  "meta_json":     "runtime_data/working/meta_20240101_120000.json"
}
```

**Response:**

```json
{ "status": "success", "message": "RF training launched successfully." }
```

---

### `POST /launch_RF_classify/`

Run inference with a trained model on the current point cloud.

**Request body:**

```json
{
  "model_dir":    "runtime_data/models/my_model",
  "features_las": "runtime_data/working/features.las",
  "output_las":   "runtime_data/working/features.las"
}
```

**Response:**

```json
{ "status": "success", "message": "RF classify launched successfully." }
```

---

## Annotation and Export

### `POST /api/export-mapping/`

Persist point annotation data (segment IDs and class IDs) into
`runtime_data/working/annotations.bin`. The buffer is streamed to disk (gunzipped on the fly when
`encoding=gzip`) and merged into the stored file with numpy: only points with `segment_id != 0` in
the buffer overwrite the stored ones. The final size must be `point_count * 2` bytes.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `buffer` | Binary blob | 2 bytes per point: `segment_id` (1-based, 0 = unannotated) + `class_id` |
| `point_count` | Integer string | Total number of points |
| `encoding` | String | Optional: `gzip` if `buffer` is gzip-compressed |

**Response:**

```json
{
  "annotations_path": "runtime_data/working/annotations.bin",
  "point_count":      84230
}
```

---

### `POST /api/extract-segment-las/`

Extract all points belonging to a single segment ID into a new LAS file.

**Request body:**

```json
{
  "las_path":   "runtime_data/working/features.las",
  "annotations_path": "runtime_data/working/annotations.bin",
  "seg_id":     1,
  "out_path":   "runtime_data/working/segment_1.las"
}
```

**Response:**

```json
{ "status": "success", "message": "Segment extraction completed." }
```

---

### `POST /api/download-package/`

Assemble and stream a ZIP archive containing selected segments, point cloud
files, and trained models. Temporary files generated during packaging are
deleted after the response is sent.

**Request body:**

```json
{
  "segments": [
    { "id": 1, "label": "vegetation" },
    { "id": 2, "label": "ground" }
  ],
  "point_cloud_files": [
    { "path": "runtime_data/working/predicted.las", "label": "classified" }
  ],
  "models":   ["my_model"],
  "las_path": "runtime_data/working/features.las",
  "bin_path": "runtime_data/working/annotations.bin"
}
```

**Response:** Binary ZIP stream (`Content-Type: application/zip`).

---

### `POST /save_file/`

Save a base64-encoded payload to a path on the server. Used by the frontend to
persist annotation data and configuration files.

**Request body:**

```json
{
  "filepath": "runtime_data/working/annotations.json",
  "data":     "<base64-encoded content>"
}
```

**Response:**

```json
{ "status": "success", "filepath": "/abs/path/to/annotations.json" }
```

---

## Model Management

### `GET /api/models-list/`

List all trained models in `runtime_data/models/`.

**Response:**

```json
{
  "status": "success",
  "models": [
    {
      "name":     "my_model",
      "path":     "runtime_data/models/my_model/model.pkl",
      "created":  "2024-01-15 10:30",
      "size_mb":  12.4
    }
  ]
}
```

---

### `GET /api/model-exists/?name=<model_name>`

Check whether a model folder exists under `runtime_data/models/<name>/`.

**Response:**

```json
{ "exists": true }
```

---

### `POST /api/delete-model/`

Delete a trained model and all files in its directory.

**Request body:**

```json
{ "name": "my_model" }
```

**Response:**

```json
{ "status": "success", "message": "Model 'my_model' deleted successfully." }
```

---

### `POST /api/upload-model/`

Upload an externally trained model file to `runtime_data/models/`.

**Request:** `multipart/form-data` with a `.pkl` model file.

**Response:**

```json
{ "status": "success" }
```

---

## File Serving

### `GET /pointcloud-data/<path>`

Serve the chunked point cloud files (`pc/geom.bin`, `pc/col/*.bin`, `pc/meta.json`, ...)
with HTTP Range request support: the viewer fetches arbitrary byte ranges of multi-GB files
without downloading them in full.

- **Allowed extensions:** `.bin`, `.json`
- **Restricted to** `runtime_data/` (directory traversal is prevented)
- Returns `206 Partial Content` (with `Content-Range`) for range requests, `416` when the range is
  outside the file, `200 OK` for full-file requests; suffix ranges (`bytes=-N`) are supported
- The file is handed to the WSGI server through `FileResponse` positioned at the first byte and with
  an explicit `Content-Length`, so Gunicorn serves the slice with `sendfile()` (zero copy, no Python
  loop: about 4× the throughput of the former 64 KB generator)
- With `?v=<token>` in the query the response carries
  `Cache-Control: public, max-age=31536000, immutable`; otherwise `no-cache`

---

### `GET /runtime-data/<path>`

Serve general runtime data files without Range support (streamed with `FileResponse`).

- **Allowed extensions:** `.las`, `.bin`, `.json`, `.txt`
- **Restricted to** `runtime_data/`

---

### `GET /api/read-file/?path=<relative_path>`

Read and return the text content of a `.txt` file. If `path` points to a
directory, the first `.txt` file found inside it is returned.

**Response:**

```json
{ "status": "success", "content": "accuracy: 0.95\n..." }
```
