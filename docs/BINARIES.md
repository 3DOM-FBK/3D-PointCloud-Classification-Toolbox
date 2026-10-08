# C++ Binaries Reference

The application ships eight pre-compiled C++ binaries located at `/webapp/opt/` inside
the Docker container. They are invoked by the Django backend via `subprocess.Popen`,
communicate exclusively through file paths and CLI arguments, and write all output to
`runtime_data/working/`.

**Shared build dependencies:** PCL 1.9, PDAL 2.7.1, GDAL 3.6.2, Open3D 0.19.0,
LASzip, CGAL, Boost, CUDA 11.8 + Thrust (GPU binaries only).

---

## `feature_extraction_viewer_gpu`

**Purpose:** Compute per-point geometric features from a LAS file using GPU-accelerated
CUDA kernels with tiled processing (Thrust). This is the primary feature extraction
binary used in production.

**Input:** LAS  
**Output:** LAS with computed features stored in Extra Bytes

**CLI:**

```
feature_extraction_viewer_gpu <input.las> <output.las>
    --features <feat1,feat2,...>
    --radius <r1,r2,...>
    --sampling_resolution <int>
```

| Argument | Description |
|---|---|
| `--features` | Comma-separated list of features to compute (e.g. `anisotropy,linearity,planarity`) |
| `--radius` | Comma-separated neighborhood radii in scene units (e.g. `0.5,1.0,2.0`) |
| `--sampling_resolution` | Subsampling factor before computation; `0` disables subsampling |

**Available features:** `anisotropy`, `linearity`, `planarity`, `sphericity`,
`omnivariance`, `eigenentropy`, `change_of_curvature`, and additional neighborhood
statistics derived from local covariance.

**GPU requirements:** NVIDIA GPU with Compute Capability ≥ 7.5, CUDA 11.8 runtime,
≥ 4 GB VRAM.

**Example:**

```bash
/webapp/opt/feature_extraction_viewer_gpu \
  runtime_data/working/features.las \
  runtime_data/working/features.las \
  --features anisotropy,linearity,planarity,sphericity \
  --radius 0.5,1.0 \
  --sampling_resolution 0
```

**Common errors:**

- `CUDA out of memory` — increase `--sampling_resolution` or reduce point cloud density
  before extraction.
- `libcudart.so not found` — verify `LD_LIBRARY_PATH` includes the CUDA runtime
  libraries.

---

## `feature_extraction_viewer_cpu`

**Purpose:** CPU-only fallback for geometric feature extraction. Uses the same algorithm
as the GPU binary but runs with OpenMP multi-threading. Suitable when no CUDA GPU is
available; expect 5–15× longer processing time on large point clouds.

**Input:** LAS  
**Output:** LAS with Extra Bytes

**CLI:** Identical to `feature_extraction_viewer_gpu`.

```bash
/webapp/opt/feature_extraction_viewer_cpu \
  runtime_data/working/features.las \
  runtime_data/working/features.las \
  --features anisotropy,linearity \
  --radius 0.5,1.0 \
  --sampling_resolution 0
```

**Dependencies:** PCL, PDAL, OpenMP, Eigen (no CUDA required).

---

## `subsample_pc`

**Purpose:** Downsample a point cloud using a voxel-grid filter. Each voxel cell
retains one representative point. Used to reduce point count before feature extraction
or visualization.

**Input:** PLY or LAS  
**Output:** Subsampled PLY or LAS

**CLI:**

```
subsample_pc <input> <output> <voxel_size>
```

| Argument | Description |
|---|---|
| `voxel_size` | Side length of the voxel grid cell in scene units (e.g. `0.05` for 5 cm) |

**Example:**

```bash
/webapp/opt/subsample_pc \
  runtime_data/working/input.las \
  runtime_data/working/subsampled.las \
  0.05
```

**Dependencies:** Open3D 0.19.0.

---

## `mesh2pc`

**Purpose:** Convert a surface mesh to a point cloud by uniform surface sampling.
Supports GLB, GLTF input formats via tinygltf and Open3D.

**Input:** GLB / GLTF  
**Output:** LAS

**CLI:**

```
mesh2pc <input_mesh> <output.las> <num_points>
```

| Argument | Description |
|---|---|
| `num_points` | Number of points to sample from the mesh surface |

**Example:**

```bash
/webapp/opt/mesh2pc \
  runtime_data/working/model.glb \
  runtime_data/working/pointcloud.las \
  500000
```

**Dependencies:** Open3D 0.19.0, CGAL, tinygltf, OpenMP.

---

## `ply2las`

**Purpose:** Convert a PLY point cloud file to LAS format. Invoked as the first step
when the uploaded file is in PLY format.

**Input:** PLY  
**Output:** LAS

**CLI:**

```
ply2las <input.ply> <output.las>
```

**Example:**

```bash
/webapp/opt/ply2las \
  runtime_data/working/input.ply \
  runtime_data/working/output.las
```

**Dependencies:** Open3D 0.19.0.

---

## `split_las_by_binary`

**Purpose:** Split a feature LAS file into multiple per-segment output LAS files based
on annotation data stored in `annotations.bin`. Used to prepare training and
validation sets.

**Input:** LAS + `annotations.bin` (raw buffer, 2 bytes per `POINT_ID`: `segment_id + 1`, `class_id`; see [ARCHITECTURE.md](ARCHITECTURE.md#annotationsbin-format))  
**Output:** Multiple LAS files (`segment_1.las`, `segment_2.las`, …)

**CLI:**

```
split_las_by_binary <features.las> <annotations.bin> <output_dir> [--exclude-unclassified]
split_las_by_binary <features.las> <annotations.bin> --extract-segment <seg_id> <out_path>
```

| Argument | Description |
|---|---|
| `output_dir` | Directory where per-segment LAS files are written |
| `--exclude-unclassified` | If set, annotated points with no class (class_id = 0) are omitted from output |
| `--extract-segment <seg_id> <out_path>` | Write only the points of one segment into a single LAS (no `labels` dimension) |

**Example:**

```bash
/webapp/opt/split_las_by_binary \
  runtime_data/working/features.las \
  runtime_data/working/annotations.bin \
  runtime_data/working/segments
```

**Dependencies:** PDAL, PCL, OpenMP.

---

## `check_point_id`

**Purpose:** Validate and rewrite the `POINT_ID` attribute in a LAS file to ensure
canonical 0-based sequential indexing. Required by the annotation system to correctly
map per-point buffer indices to spatial positions.

**Input:** LAS  
**Output:** LAS (validated; may be identical to input if POINT_ID was already correct)

**CLI:**

```
check_point_id <input.las> <output.las>
```

**Example:**

```bash
/webapp/opt/check_point_id \
  runtime_data/working/input.las \
  runtime_data/working/validated.las
```

**Dependencies:** Open3D 0.19.0.


---

## `las2pc`

**Purpose:** Convert the canonical `features.las` into the chunked point cloud geometry
read by the viewer (`geom.bin`, `point_order.bin`, `meta.json`, see
[POINTCLOUD_FORMAT.md](POINTCLOUD_FORMAT.md)). Runs once at import (and again only if the
number of points changes). The attribute columns are written afterwards by
`viewer/utils_functions/pc_columns.py`.

**Input:** LAS 1.x, uncompressed, any point format; `POINT_ID` Extra Byte (uint32) expected  
**Output:** folder with `geom.bin` (20 B/point), `point_order.bin`, `meta.json` (no columns)

**CLI:**

```
las2pc --input <features.las> --output <out_dir>
    [--max-chunk 250000] [--base 3] [--levels 6]
    [--head-budget 1000000] [--seed 12345]
```

| Option | Default | Meaning |
|---|---|---|
| `--max-chunk` | 250000 | Maximum points of a spatial chunk (leaf of the implicit octree) |
| `--base` | 3 | Level 0 uses a `2^base` grid per side inside the chunk cube |
| `--levels` | 6 | Number of stratified levels (a last "remainder" level is added) |
| `--head-budget` | 1000000 | Points of the overview block that is fetched with one Range request |
| `--seed` | 12345 | Seed of the per-chunk shuffle: the output is deterministic |

**Notes:** the conversion is done in memory (about 32 bytes per point); if it does not fit
in the available RAM it stops with an explicit error (out-of-core conversion is not
implemented). The Django backend writes into `pc_tmp/` and swaps it atomically with `pc/`.
On the 5 M point test cloud it takes about 0.4 s with 8 threads.

**Dependencies:** OpenMP only (no PCL/PDAL/Open3D).
