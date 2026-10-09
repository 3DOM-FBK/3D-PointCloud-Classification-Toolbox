# C++ Binaries Reference

The application ships eight pre-compiled C++ binaries located at `/webapp/opt/` inside
the Docker container. They are invoked by the Django backend via `subprocess.Popen`,
communicate exclusively through file paths and CLI arguments, and write all output to
`runtime_data/working/`.

All of them are **out-of-core**: they never load the whole point cloud, they read the big files
sequentially and keep their memory under a budget. Shared options (see
[INSTALLATION.md](INSTALLATION.md#memory-scratch-space-and-storage)):

| Option | Default | Meaning |
|---|---|---|
| `--memory-budget MB` | 50 % of `min(cgroup limit, MemAvailable)` | peak memory the tool may use |
| `--temp-dir DIR` | `$PIPELINE_TEMP_DIR` or `/tmp/pipeline_work` | scratch folder (one sub-folder per run, removed at the end, on error and on `SIGTERM`) |

Before writing temporary files a tool checks the free space and stops with a message that says how much is needed.
On `SIGTERM` (`/stop_process/`) every tool removes its scratch folder and its partial output before exiting.

Source of the shared primitives: `testC++/ooc/` (header only). The Python tools use
`viewer/utils_functions/pipeline_common.py`.

**Build dependencies:** PCL 1.9 (CPU feature extraction), Open3D 0.19.0 (PLY branch of `subsample_pc`, `mesh2pc`, fallback of
`ply2las`), CGAL (`mesh2pc`), CUDA 11.8 + Thrust (GPU binary only), OpenMP. `las2pc`, `split_las_by_binary` and
`check_point_id` need nothing but the C++ standard library (and OpenMP).

---

## `feature_extraction_viewer_gpu`

**Purpose:** Per-point geometric features of a LAS with CUDA kernels, tile by tile. This is the primary feature
extraction binary used in production.

**Input:** LAS (the tool reads x, y, z, intensity, returns, class, scan angle, RGB and `POINT_ID`)  
**Output:** LAS (point format 3) with `POINT_ID`, one Extra Byte per feature and radius (`<feature>_<radius>`, e.g.
`planarity_1_2`) and `normal_x/y/z` (zeros). **One record per input point, in the order of the input** (the record of
input row *r* is written at output row *r*).

**CLI:**

```
feature_extraction_viewer_gpu <input.las> <output.las>
    [--features f1,f2,...] [--radius r1,r2,...] [--buffer B]
    [--tile_size S] [--target_tiles N]
    [--memory-budget MB] [--temp-dir DIR]
```

| Argument | Description |
|---|---|
| `--features` | Comma-separated features (default: all): `anisotropy, omnivariance, sphericity, planarity, linearity, verticality, surface_variation, neighbours, vertical_range, height_above, height_below` |
| `--radius` | Neighbourhood radii in metres (default `0.8,1.2,2.0,3.0`); the buffer is at least `max(radius) + 0.5` |
| `--buffer` | Buffer around every tile in metres (default 4) |
| `--tile_size` | Fixed tile size in metres (overrides the automatic one) |
| `--target_tiles` | Tiles wanted by the automatic sizing of small clouds (default 16) |

**Memory.** The cloud is distributed in XY tiles (core points plus a buffer of copies) in temporary files
(`36 bytes × points × buffer overhead`); a tile is loaded, uploaded to the GPU and computed alone, so the host memory and the
VRAM no longer depend on the size of the cloud. The tile size is the one it always was (about 16 tiles) as long as the
densest tile fits the point cap derived from the memory budget (`60 %` of it) and from the free VRAM (`70 %`); otherwise the
tiles are made smaller from an XY histogram of the cloud. A tile whose computation fails (GPU out of memory, grid too large) is
split in two and retried; if it still fails the run stops with a message (it never leaves holes in the output: the number of
rows written is checked). Points exactly on the upper edge of the grid belong to the last tile (the previous CPU version
dropped them silently; the GPU one already included them).

**GPU requirements:** NVIDIA GPU with Compute Capability ≥ 7.5, CUDA 11.8 runtime.

**Example:**

```bash
/webapp/opt/feature_extraction_viewer_gpu \
  runtime_data/working/features.las \
  runtime_data/working/features_temp.las \
  --features anisotropy,linearity,planarity,sphericity \
  --radius 0.5,1.0
```

**Common errors:**

- `no CUDA device available` — run the container with `--gpus all`, or use the CPU binary.
- `Tile ... could not be computed` — the tile stayed too big for the GPU even after being split ten times; lower the radii.
- `Not enough free space in the temporary folder` — set `PIPELINE_TEMP_DIR` to a bigger volume.

---

## `feature_extraction_viewer_cpu`

**Purpose:** CPU-only fallback of the previous binary (same features, same output layout), with PCL octree radius searches
and OpenMP inside every tile. Expect much longer processing times on large clouds.

**CLI:** the same as the GPU binary, without `--target_tiles`; `--tile_size` defaults to 50 m. The tile size is reduced
automatically when a tile would not fit in the memory budget (about 650 bytes per tile point).

```bash
/webapp/opt/feature_extraction_viewer_cpu runtime_data/working/features.las runtime_data/working/features_temp.las \
  --features anisotropy,linearity --radius 0.5,1.0
```

**Dependencies:** PCL, OpenMP, Eigen (no CUDA).

---

## `subsample_pc`

**Purpose:** Voxel-grid subsampling.

```
subsample_pc <input.ply|input.las> <output.ply|output.las> [voxel_size=0.002] [--memory-budget MB] [--temp-dir DIR]
```

* **LAS → LAS** (out-of-core, no Open3D): same result as Open3D's `VoxelDownSample` (one point per voxel: the average of its
  points and colours; the voxel grid is anchored at *minimum − voxel/2*) followed by the estimation of the normals on the
  subsampled cloud (radius 0.02 m, 30 neighbours, oriented towards the origin). The points are distributed in XY tiles
  aligned to the voxel grid (no buffer is needed), every tile is subsampled on its own, and the normals are computed tile by
  tile with a buffer of the search radius. Output: LAS 1.2, point format 3, `NormalX/Y/Z` + `POINT_ID`; the order of the points is
  deterministic (voxel grid order). Memory is bounded by `--memory-budget`.
* **PLY → PLY** (Open3D, the whole cloud is loaded): the memory it needs is checked first and the tool stops with a clear
  message when it does not fit; convert the cloud to LAS to subsample it out-of-core.

---

## `mesh2pc`

**Purpose:** Convert a surface mesh to a point cloud by uniform surface sampling (GLB / GLTF via tinygltf, Open3D, CGAL).

```
mesh2pc <input_mesh> <output.las> [num_points=5000000] [--memory-budget MB]
```

The number of points is chosen by the user; the tool needs about **200 bytes per sampled point** and checks it against the
memory budget **before doing any work**, stopping with a message that tells the largest number of points that fits.

---

## `ply2las`

**Purpose:** Convert a PLY to the canonical LAS (point format 3, `NormalX/Y/Z`, `POINT_ID`), first step of the import of a PLY.

```
ply2las <input.ply> <output.las> [--memory-budget MB] [--temp-dir DIR]
```

The PLY (binary little/big endian or ASCII, `x y z` and optionally `red green blue`) is read **by blocks** and the normals are
estimated tile by tile (radius 0.02 m, 30 neighbours, oriented towards the origin, as before), so the memory does not depend on
the size of the cloud. PLY variants the block reader does not support (list properties in the vertex element, other elements
before the vertices) fall back to Open3D, after checking that the whole cloud fits in the memory budget.

---

## `split_las_by_binary`

**Purpose:** Split a LAS by the annotations of the viewer (`annotations.bin`: 2 bytes per `POINT_ID` = `segment_id + 1`,
`class_id`; see [ARCHITECTURE.md](ARCHITECTURE.md#annotationsbin-format)). Used to prepare training/validation sets and to
extract the segment that is classified.

```
split_las_by_binary <features.las> <annotations.bin> <output_dir> [--exclude-unclassified] [--memory-budget MB]
split_las_by_binary <features.las> <annotations.bin> --extract-segment <seg_id> <out_path> [--memory-budget MB]
```

It no longer uses PDAL: the LAS is read once in blocks and every output record is the **input record byte for byte** (coordinates,
scale/offset, every attribute) plus, when splitting, a `labels` attribute (uint8: the class, 255 = unclassified). The relative
order of the points is preserved, so an extracted segment is an ordered subset of the cloud and its prediction is merged back
into the columns without a join. Segments without points are not written. Memory: `annotations.bin` + one block + a 1 MB write
buffer per segment.

---

## `check_point_id`

**Purpose:** Bring a LAS to the canonical layout (LAS 1.2, point format 3, `NormalX/Y/Z`, `POINT_ID` = index of the point; 50 bytes
per point), required by the annotation system.

```
check_point_id <input.las> <output.las> [--memory-budget MB] [--temp-dir DIR]
```

Already canonical → the input is moved to the output path. Normals present → the file is rewritten in blocks. Normals missing →
estimated tile by tile (radius 0.02 m, 30 neighbours, oriented towards the origin). Open3D is no longer needed; the last line of
the output is `Output: <path>`.

---

## `las2pc`

**Purpose:** Convert the canonical `features.las` into the chunked point cloud geometry read by the viewer (`geom.bin`,
`point_order.bin`, `meta.json`, see [POINTCLOUD_FORMAT.md](POINTCLOUD_FORMAT.md)) and, with `--ordered-las`, rewrite the LAS in the
order of the geometry (**canonical order**). Runs once at import (and again only if the number of points changes). The attribute
columns are written afterwards by `viewer/utils_functions/pc_columns.py`.

**Input:** LAS 1.x, uncompressed, any point format; `POINT_ID` Extra Byte (uint32) expected  
**Output:** folder with `geom.bin` (20 B/point), `point_order.bin`, `meta.json` (no columns) and, with `--ordered-las`, the LAS

```
las2pc --input <features.las> --output <out_dir> [--ordered-las <out.las>]
    [--max-chunk 250000] [--base 3] [--levels 6] [--head-budget 1000000] [--seed 12345]
    [--mode auto|memory|ooc] [--memory-budget MB] [--temp-dir DIR]
```

| Option | Default | Meaning |
|---|---|---|
| `--ordered-las` | none | also write the input LAS in the order of `geom.bin` (row *r* = point *r*) |
| `--max-chunk` | 250000 | Maximum points of a spatial chunk; a denser grid cell is split into octants |
| `--base` | 3 | Level 0 uses a `2^base` grid per side inside the chunk cube |
| `--levels` | 6 | Number of stratified levels (a last "remainder" level is added) |
| `--head-budget` | 1000000 | Points of the overview block that is fetched with one Range request |
| `--seed` | 12345 | Seed of the per-chunk shuffle (the output is deterministic) |
| `--mode` | `auto` | `memory` (mapped input, ~28 B/point of RAM + the LAS in the page cache), `ooc` (sequential passes, temp file per chunk) or `auto`: memory when it fits in the budget |

**The two modes give byte-identical output**, whatever the order of the records of the input (the points of every chunk are sorted
by `POINT_ID` before the deterministic shuffle). Out-of-core mode: pass 1 counts the points on a 128³ (256³/512³ for
bigger clouds) grid using the bbox of the LAS header as an estimate (checked while counting; measured again only if a point is
outside); pass 2 distributes the full records in one temporary file per chunk; every chunk is then loaded alone to compute its
levels and tight bbox, and finally written with `pwrite` at its final position in `geom.bin`, `point_order.bin` and the ordered LAS.
Scratch space: *LAS size + 4 B/point*.

---
