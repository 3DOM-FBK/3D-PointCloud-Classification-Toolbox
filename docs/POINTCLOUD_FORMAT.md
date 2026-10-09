# Chunked point cloud format (`pck`)

The viewer does not read LAS files directly. At import, `features.las` is converted **once** into a
folder `runtime_data/working/pc/` with a geometry file and one file per attribute (column).
Computing features, classifying with the Random Forest or restoring the backup only rewrites the
affected columns: the geometry is never rebuilt and the browser never reloads the cloud.

`features.las` stays the single source of truth for the pipeline (RF training/classification, split,
export). `pc/` is a derived, disposable cache.

```
pc/
  meta.json            description of chunks, levels and columns
  geom.bin             20-byte records, one per point
  point_order.bin      uint32 POINT_ID in geom.bin order (compact copy, used by the backend)
  col/<name>.bin       one column per attribute, same point order as geom.bin
```

## geom.bin

20 bytes per point, little endian:

| bytes | type | content |
|---|---|---|
| 0–11 | int32 × 3 | `x, y, z` = LAS integer coordinate (`X` of the record) minus `qMin` |
| 12–14 | uint8 × 3 | `r, g, b` (16-bit LAS colours are shifted `>> 8`; 255,255,255 when the LAS has no colour) |
| 15 | uint8 | padding |
| 16–19 | uint32 | `POINT_ID` |

Real coordinate of a point: `(q + qMin) * scale + offset`. The viewer works with Float32 positions
**relative to `boundingBox.min`**: `(q + qMin) * scale + offset - boundingBox.min` (equals `q * scale`
because `boundingBox.min = qMin * scale + offset`).

## Chunks and levels

* The cubic bounding box (the bbox of the LAS header widened by one cell and verified while counting; measured
  again only if a point falls outside it) is counted on a 128³ grid (256³ above 100 M points, 512³ above 500 M),
  merged bottom-up into an implicit octree; every leaf with at most `--max-chunk` points (default 250 000) is a
  **chunk**. A grid cell that is still denser than `--max-chunk` is split into octants (recursively, at most 14
  levels below the grid, in place of the cell in the depth-first order), so the size of a chunk is bounded.
* Inside each chunk the points are first **sorted by `POINT_ID`**, then shuffled with a fixed seed (the output is
  deterministic and does **not** depend on the order of the records in the LAS, nor on the execution mode or the
  number of threads: in-memory and out-of-core runs give byte-identical files) and split
  into **levels** with stratified sampling: for `l = 0, 1, …` a grid of `2^(base+l)` cells per side is laid over the
  chunk cube; each point, in shuffled order, whose cell is still free goes to level `l`. Points that never find a
  free cell go to the last level, the **remainder** (always present, possibly empty). The occupancy of a level is a
  sparse set of cell keys (no dense `R³` array), so the grid can go down to `2^20` cells per side.
  Nominal spacing of level `l` is `size / 2^(base+l)`. Inside a level (and inside the remainder) the order stays
  random, so any prefix of a level is still a uniform subsample.
* **Adaptive number of levels (`formatVersion` 2, the default):** a chunk keeps building levels while its remainder is
  larger than `max(5 % of its points, 2048)` and the cell is still larger than twice the LAS scale, up to
  `--max-levels` (16). The number of levels therefore depends on the chunk (`levelPoints` has variable length;
  `levels.count` is the maximum), and the remainder holds ~1 % of the points instead of the 27 % of the fixed
  6 levels of `formatVersion` 1 (`--levels N` still builds exactly `N` levels and writes `formatVersion` 1; the
  loader reads both). Memory and out-of-core runs stay byte-identical.
* **Layout of the points in the file**: first the *head block* (levels `0 … H-1` of **all** chunks,
  chunk by chunk), then, for each chunk, its remaining levels contiguous. `H` is the largest value such
  that the points of levels `< H` fit in `--head-budget` (default 1 M points, minimum `H = 1`; a chunk with
  fewer than `H` levels has all of its points in the head).
  The whole overview is a single Range request; any range of consecutive points of one chunk's body is
  contiguous (one request); the head points of one chunk are contiguous inside the head block as well.

### Continuous level of detail (viewer)

The points of a chunk, read in file order (head, then body), are a sequence whose every prefix is a uniform
sample. The spacing of the sample made by the first `k` points is a continuous decreasing function of `k`, and the
viewer uses it instead of whole levels (`static/viewer/js/pointcloud-lod.js`, shared by the loader and the worker):

* inside regular level `l`, for the point at fraction `u = (k - prefix[l]) / levelPoints[l]`:
  `lodSpacing = size / 2^(base + l + u)`;
* in the remainder the spacing is extrapolated with `n(s) ∝ s^-D` (`D` from the last two regular levels, clamped to
  1…3); level-0 points are never dropped.

The worker writes `lodSpacing` and `fullSpacing` (the spacing with every point of the chunk loaded) as the vertex
attribute `lodInfo`; the vertex shader drops the points whose spacing projects to less than the pixel threshold
(point size × a budget factor) and enlarges the points where even the full data is sparser than the screen.
The loader fetches, for every chunk in view, `need` points: those whose spacing at the chunk's nearest point
still projects to the threshold. All the screen decisions use one metric, `ppu = P[1][1]·H/2 / z` (perspective) or
`P[1][1]·H/2` (orthographic), computed from the camera projection matrix.

## meta.json

```jsonc
{
  "format": "pck", "formatVersion": 2, "version": "<geometry token>",
  "points": 5000000, "scale": [..], "offset": [..], "qMin": [..],
  "boundingBox": {"min": [..], "max": [..]},
  "hasColor": true,
  "geom": {"file": "geom.bin", "recordSize": 20},
  "order": {"file": "point_order.bin", "type": "uint32"},
  "levels": {"count": 9, "base": 3, "head": 4},   // count = longest levelPoints of any chunk
  "head": {"points": 453495},
  "chunks": [{"id": 0, "min": [..], "max": [..], "cubeMin": [..], "size": 32.0,
              "levelPoints": [81, 321, 1243, 4605, 15103, 23870, 6100, 2100, 233],   // last = remainder
              "headOffset": 0, "bodyOffset": 453495, "points": 53656}],
  "columns": {"NormalX": {"file": "col/NormalX.bin", "type": "float32", "min": -1, "max": 1, "version": "<token>"},
              "prediction": {"file": "col/prediction.bin", "type": "uint8", "missing": 255, "min": 0, "max": 5, "version": "<token>"}}
}
```

* All offsets are expressed in **points** (bytes = points × element size of the file; geometry 20 B,
  float32 columns 4 B, uint8 columns 1 B).
* A chunk's head points (levels `0 … H-1`) start at `headOffset` (in order), the rest at `bodyOffset`.
  The first point of level `l ≥ H` is `bodyOffset + sum(levelPoints[H..l-1])`; the point at position `k ≥ headPoints` of the chunk sequence is at `bodyOffset + k - headPoints`.
* `min`/`max` are the tight AABB of the chunk's points; `cubeMin`/`size` the octree cube. Both are in
  metres **relative to `boundingBox.min`** (LAS orientation, before the viewer's X mirror).
* `version` changes whenever the geometry is rebuilt; every column has its own `version`, which
  changes at each rewrite. The browser appends `?v=<version>` to the URLs so that immutable caching
  is safe; `meta.json` is always fetched with `cache: 'no-store'`.

## Columns

| type | used for | missing value |
|---|---|---|
| `float32` | features and every non-standard Extra Byte of the LAS (including `NormalX/Y/Z`) | `NaN` |
| `uint8` | `prediction` (Random Forest class) | `255` – RF class ids must stay `< 255` |

The writer (`viewer/utils_functions/pc_columns.py`) never loads a column in memory: it reads the LAS **once, in
blocks**, extracting every requested column from each block, and appends the values to the column files. The join
key is the `POINT_ID`, and the strategy depends on how the LAS is ordered (the log prints `path: ...`):

| path | when | cost |
|---|---|---|
| `fast` | the LAS is in **canonical order** (see below): row *r* of the LAS is point *r* of `geom.bin` | one sequential read, values written as they are |
| `merge` | the LAS is an *ordered subset* of the cloud (e.g. a segment extracted for the classification) | sequential merge with `point_order.bin`; points without a value get the missing marker |
| `join` | any other order (external LAS, files from older versions) | external join through bucket files in the temporary folder (LAS → `(pid, values)` by pid range, `point_order.bin` → `(pid, row)` by pid range, per-range dense join, `(row, values)` by row range, per-range sequential write) |

All three give identical files. Points missing from the LAS keep the missing value.

## Canonical order of `features.las`

`las2pc --ordered-las` also writes the input LAS **in the order of `geom.bin`** (records unchanged, `POINT_ID`
unchanged, header bbox/counters fixed). The backend (`build_pointcloud`) replaces `features.las` with it before the
backup is taken, so the working LAS, the backup and the columns share the order. Every later tool preserves the
order of the records it reads (RF classification, column updates, subsets extracted by `split_las_by_binary` keep
the relative order), which is what makes the `fast` and `merge` paths possible. `meta.json` has
`"canonicalLas": true` when the build produced it. Nothing depends on the order for correctness (the `POINT_ID` is
the key and `annotations.bin` is indexed by it): a LAS out of order is simply joined by the `join` path.

## Tools

* `las2pc` (C++, `testC++/las2pc.cpp`, `/webapp/opt/las2pc`) – `features.las` → geometry (+ canonical LAS).
  Two modes with the same output: **memory** (the LAS is mapped, about 28 bytes/point of RAM, fastest) and
  **out-of-core** (sequential passes over the input, one temporary file per chunk, chunks processed by
  threads, `pwrite` at the final position; memory bounded by `--memory-budget`, scratch space of about
  *LAS size + 4 B/point* in `--temp-dir`). `--mode auto` (default) picks memory when *LAS size + 28 B/point* fits in
  the budget (default: 50% of the memory available to the container).
* `pc_columns.py` – writes/updates/drops columns (`--all`, `--only`, `--prediction`, `--drop-all`); memory O(block).
* `testC++/bench/verify_pc.py` – consistency check of a converted folder against its LAS.
* `testC++/bench/compare_pc.py` – byte-for-byte comparison of two `pc/` folders / two LAS (used to check that the
  out-of-core run equals the in-memory one).
