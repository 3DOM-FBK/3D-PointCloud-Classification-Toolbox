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

**Build dependencies:** Open3D 0.19.0 (PLY branch of `subsample_pc`, `mesh2pc`, fallback of
`ply2las`), CGAL (`mesh2pc`), CUDA 12.8 (feature extraction; the program also builds without CUDA), OpenMP. `las2pc`, `split_las_by_binary` and
`check_point_id` need nothing but the C++ standard library (and OpenMP).

---

## `feature_extraction_viewer`

**Purpose:** Per-point geometric features of a LAS, tile by tile, on the GPU (CUDA) or on the CPU. One program with both
backends (it replaces `feature_extraction_viewer_gpu` and `feature_extraction_viewer_cpu`, which no longer exist); the web
app runs it with `--backend auto` when the GPU switch is on and `--backend cpu` otherwise.

**Input:** LAS (the tool reads x, y, z, intensity, returns, class, scan angle, RGB and `POINT_ID`)  
**Output:** LAS (point format 3) with `POINT_ID`, one Extra Byte per feature and radius (`<feature>_<radius>`, e.g.
`planarity_1_2`) and `normal_x/y/z` (zeros). **One record per input point, in the order of the input** (the record of
input row *r* is written at output row *r*). The output is reproducible: same input and options, same bytes.

**CLI:**

```
feature_extraction_viewer <input.las> <output.las>
    [--features f1,f2,...] [--radius r1,r2,...] [--buffer B]
    [--tile_size S] [--target_tiles N]
    [--backend auto|gpu|cpu] [--threads N] [--slots N]
    [--memory-budget MB] [--temp-dir DIR]
feature_extraction_viewer --probe          # prints the usable CPUs and GPU; exit code 3 when there is no usable GPU
```

| Argument | Description |
|---|---|
| `--features` | Comma-separated features (default: all): `anisotropy, omnivariance, sphericity, planarity, linearity, verticality, surface_variation, neighbours, vertical_range, height_above, height_below` |
| `--radius` | Neighbourhood radii in metres (default `0.8,1.2,2.0,3.0`, at most 10); the buffer is at least `max(radius) + 0.5` |
| `--buffer` | Buffer around every tile in metres (default 4) |
| `--tile_size` | Fixed tile size in metres (overrides the automatic one) |
| `--target_tiles` | Tiles wanted by the automatic sizing of small clouds (default 16) |
| `--backend` | `auto` (default): the GPU when the machine has a usable CUDA device, the CPU otherwise; `gpu`: the GPU or exit code 3; `cpu`: never touches the GPU |
| `--threads` | Host threads (default: the CPUs this process may use: affinity mask and container quota, so `docker run --cpus 2` really uses 2) |
| `--slots` | Tiles in flight on the GPU (default 2; 1 = no overlap) |

**Algorithm (the same on both backends, `testC++/ooc/feature_math.h`).** Coordinates are integers (ticks of the smallest LAS
scale, counted from the minimum of the tile), so the radius test `d² ≤ r²` is exact and identical to the previous versions
(points exactly at distance `r` are inside). A uniform grid (cell = largest radius / 2) is sorted by cell with x fastest, so
the cells of a row are one contiguous range. The neighbours of the *largest* radius are visited **once**; every neighbour is
added to the shell of the smallest radius that contains it and the shells are summed cumulatively, so all the scales come
from one traversal. The moments (`n, Σd, Σd·dᵀ, min/max y`) are accumulated relative to the query point in `float`, the
eigenvalues come from a trigonometry-free Jacobi solver. Y is the vertical axis (verticality, height features). Omnivariance is
`(e0·e1·e2)^(1/3)` and a neighbourhood with fewer than 3 points gives zeros (only `neighbours` is reported).

**Pipeline.** A reader thread loads the next tile from disk, the GPU computes tile *k* while the host prepares tile *k+1*
and packs tile *k−1*, and a writer thread writes the rows of the previous tile. The CPU backend computes with all its threads
while the reader and writer threads overlap the I/O. Pinned staging buffers and the device workspace are allocated once.

**Memory.** The cloud is distributed in XY tiles (core points plus a buffer of copies) in temporary files
(`36 bytes × points × buffer overhead`); only the tiles in flight are in memory (and on the GPU), so the host memory and the
VRAM do not depend on the size of the cloud. The tile size is the one it always was (about 16 tiles) as long as the
densest tile fits the point cap derived from the memory budget (`60 %` of it, divided among the tiles in flight) and from the
free VRAM (`70 %`); otherwise the tiles are made smaller from an XY histogram of the cloud. A tile whose computation fails (GPU
out of memory, grid too large) is split in two and retried; if it still fails the run stops with a message (it never leaves holes in the output: the number of rows
written is checked).

**Portability.** The program is built with CUDA 12.8 for `sm_61 … sm_120` (Pascal to Blackwell) plus PTX for the newest
architecture, so it also starts on GPUs released later (the driver compiles the PTX); the driver must support CUDA 12
(minor-version compatibility starts at 525; Blackwell GPUs need a driver that knows them, 570 or newer; these limits come from
NVIDIA's compatibility rules and were not tested here beyond a driver 610 with an sm_75 GPU). The CPU code is compiled for generic
x86-64 and selects SSE2, AVX2 or AVX-512 when it starts (`target_clones`). No GPU, no driver, a GPU without code for its
architecture, or no free VRAM: with `--backend auto` it simply runs on the CPU (the message says why).

**Example:**

```bash
/webapp/opt/feature_extraction_viewer \
  runtime_data/working/features.las \
  runtime_data/working/features_temp.las \
  --features anisotropy,linearity,planarity,sphericity \
  --radius 0.5,1.0
```

**Common errors:**

- `no usable CUDA device` with `--backend gpu` (exit code 3) — run the container with `--gpus all`, or use `--backend auto`/`cpu`.
- `Tile ... could not be computed` — the tile stayed too big even after being split ten times; lower the radii.
- `Not enough free space in the temporary folder` — set `PIPELINE_TEMP_DIR` to a bigger volume.

**Reference numbers (RTX 2080 Ti, i7-9700, 3 GB budget, input and output on the same NVMe through Docker):** 5 M points,
radii 0.5/1/2: 8–12 s (previous GPU version 17 s, previous CPU/PCL version 241 s; the new CPU backend 18 s with 8 threads,
40 s with 2). 20 M points, radii 0.8/1.2/2/3: 53 s (previous GPU version 106 s). On this machine the remaining time is disk
(distribution of the tiles, writing the output): the GPU waits for the disk.

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
    [--max-chunk 250000] [--base 3] [--levels 0] [--max-levels 16] [--remainder-frac 0.05] [--remainder-min 2048]
    [--head-budget 1000000] [--seed 12345]
    [--mode auto|memory|ooc] [--memory-budget MB] [--temp-dir DIR]
```

| Option | Default | Meaning |
|---|---|---|
| `--ordered-las` | none | also write the input LAS in the order of `geom.bin` (row *r* = point *r*) |
| `--max-chunk` | 250000 | Maximum points of a spatial chunk; a denser grid cell is split into octants |
| `--base` | 3 | Level 0 uses a `2^base` grid per side inside the chunk cube |
| `--levels` | 0 | `0`: adaptive number of levels per chunk (`formatVersion` 2); `N`: exactly `N` regular levels (`formatVersion` 1). A last "remainder" level is always added |
| `--max-levels` | 16 | Cap of the adaptive levels (`base + max-levels <= 20`) |
| `--remainder-frac`, `--remainder-min` | 0.05, 2048 | Adaptive: stop when the remainder is below this fraction of the chunk or below this many points |
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
