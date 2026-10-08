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

* The cubic bounding box is counted on a 128³ grid (256³ above 100 M points), merged bottom-up into an
  implicit octree; every leaf with at most `--max-chunk` points (default 250 000) is a **chunk**.
* Inside each chunk the points are shuffled with a fixed seed (the output is deterministic) and split
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

The writer (`viewer/utils_functions/pc_columns.py`) builds `by_pid = full(N, missing)`, scatters the
values of the LAS by `POINT_ID` and writes `by_pid[point_order]`. Because the join key is always the
`POINT_ID`, the order of the records in the LAS (the GPU feature extractor rewrites the points tile
by tile) does not matter, and points missing from the LAS keep the missing value.

## Tools

* `las2pc` (C++, `testC++/las2pc.cpp`, `/webapp/opt/las2pc`) – `features.las` → geometry. In memory
  (about 32 bytes per point); out-of-core conversion is not implemented.
* `pc_columns.py` – writes/updates/drops columns (`--all`, `--only`, `--prediction`, `--drop-all`).
* `testC++/bench/verify_pc.py` – consistency check of a converted folder against its LAS.
