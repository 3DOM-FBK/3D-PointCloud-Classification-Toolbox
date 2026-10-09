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
  into **levels** with stratified sampling: for `l = 0 … L-1` a grid of `2^(base+l)` cells per side is laid over the
  chunk cube; each point, in shuffled order, whose cell is still free goes to level `l`. Points that
  never find a free cell go to the last level `L`. Defaults: `base = 3`, `L = 6`, i.e. 7 levels.
  Nominal spacing of level `l` is `size / 2^(base+l)`. Inside a level the order stays random, so any
  prefix of a level is still a uniform subsample.
* **Layout of the points in the file**: first the *head block* (levels `0 … H-1` of **all** chunks,
  chunk by chunk), then, for each chunk, its levels `H … L` contiguous. `H` is the largest value such
  that the points of levels `< H` fit in `--head-budget` (default 1 M points, minimum `H = 1`).
  The whole overview is a single Range request; any range of consecutive levels `≥ H` of one chunk is
  contiguous (one request); levels `< H` of one chunk are contiguous inside the head block as well.

## meta.json

```json
{
  "format": "pck", "formatVersion": 1, "version": "<geometry token>",
  "points": 5000000, "scale": [..], "offset": [..], "qMin": [..],
  "boundingBox": {"min": [..], "max": [..]},
  "hasColor": true,
  "geom": {"file": "geom.bin", "recordSize": 20},
  "order": {"file": "point_order.bin", "type": "uint32"},
  "levels": {"count": 7, "base": 3, "head": 4},
  "head": {"points": 453495},
  "chunks": [{"id": 0, "min": [..], "max": [..], "cubeMin": [..], "size": 32.0,
              "levelPoints": [81, 321, 1243, 4605, 15103, 23870, 8433],
              "headOffset": 0, "bodyOffset": 453495, "points": 53656}],
  "columns": {"NormalX": {"file": "col/NormalX.bin", "type": "float32", "min": -1, "max": 1, "version": "<token>"},
              "prediction": {"file": "col/prediction.bin", "type": "uint8", "missing": 255, "min": 0, "max": 5, "version": "<token>"}}
}
```

* All offsets are expressed in **points** (bytes = points × element size of the file; geometry 20 B,
  float32 columns 4 B, uint8 columns 1 B).
* A chunk's levels `0 … H-1` start at `headOffset` (in order), its levels `H … L` at `bodyOffset`.
  The first point of level `l ≥ H` is `bodyOffset + sum(levelPoints[H..l-1])`.
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
