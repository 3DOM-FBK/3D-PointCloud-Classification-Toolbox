# Gaussian splat format (`gs`, formatVersion 2)

The viewer does not read Gaussian splat files directly. Each supported source is converted **once** into a
`gs` folder, the same idea as the point cloud `pck` format ([POINTCLOUD_FORMAT.md](POINTCLOUD_FORMAT.md)).

Status: **v2 (quantised) is the default**, about 80 bytes per splat with SH degree 3. **v1 (float32, exact)** is kept
as the reference. The renderer, LOD and editing are not implemented yet.

## Supported inputs

| Source | Kind | Notes |
|---|---|---|
| 3DGS PLY (binary little endian) | `3dgs` | Three scales. Fields are read by name, the property order is not fixed. |
| 2DGS PLY (binary little endian) | `2dgs` | Two scales (oriented discs). Normals in the file are zero and are not used. |
| SOG v2 (PlayCanvas splat-transform) | `3dgs` | ZIP with `meta.json` and WebP textures. Only version 2. |
| SPZ | — | Not supported yet. |

Code: `classifyViewer/viewer/utils_functions/gs_io.py` (readers), `gs_format.py` (folder and PLY writer),
`gs_quant.py` (quantisation and error report), `gs_convert.py` (command line).

```bash
python gs_convert.py INPUT.ply OUTPUT_DIR             # gs v2 (default)
python gs_convert.py INPUT.sog OUTPUT_DIR --exact     # gs v1, float32
python gs_convert.py INPUT.sog OUTPUT_DIR --ply       # also write roundtrip.ply for verification
```

## Values kept from the source

The import changes no value except the quaternion, which is normalised so that every splat has a unit rotation.
Every other value keeps the convention of the source file:

| Field | Source convention |
|---|---|
| `xyz` | position, source units |
| `rot` | quaternion `w, x, y, z` (`rot_0..3`), normalised at import |
| `scale` | **log** of the axis standard deviation (`scale_0..2`, or `0..1` for 2DGS) |
| `opacity` | **logit** |
| `dc` | DC colour coefficient `f_dc_0..2` (not the 0–1 colour) |
| `sh` | `f_rest_0..K-1`, **channel-major**: R coefficients, then G, then B. Degree 3 gives 45 values. |

Splats with a very small scale (log below −10) are **kept**. The converter does not prune, so the splat IDs stay valid.

## Folder layout

```
gs/
  meta.json     format version, kind, count, shDegree, chunks (ranges and quantisation ranges), columns
  xyz.bin       v2: uint8 (N, 9), 24 bit per axis   | v1: float32 (N, 3)
  rot.bin       v2: uint16 (N, 4), smallest-three   | v1: float32 (N, 4)
  scale.bin     v2: uint16 (N, S)                   | v1: float32 (N, S)     S = 3 (3DGS), 2 (2DGS)
  opacity.bin   v2: uint16 (N)                      | v1: float32 (N)
  dc.bin        v2: uint16 (N, 3)                   | v1: float32 (N, 3)
  sh.bin        v2: uint8 (N, 45)                   | v1: float32 (N, 45)    only when shDegree > 0
  id.bin        uint32 (N), source index of each row
```

* Rows are in **Morton order** (3D Morton code over the bounding box, ties by source index). The order is
  deterministic.
* Rows are grouped in **chunks** of at most 250 000 splats. `meta.json` gives each chunk's range, its tight AABB
  and its quantisation ranges. Chunks are the unit for the streaming and LOD work that comes next.
* `id.bin` is the **stable splat ID**: the index of the splat in the source. Edits, classes and history will be
  keyed by this ID, as `POINT_ID` is for point clouds. The file can be restored to source order with it.
* Columns are little-endian arrays, one row per splat.

## Quantisation (v2)

Every field is quantised **per chunk**, with the chunk's range (min, max) stored in `meta.json`:

| Field | Encoding | Why |
|---|---|---|
| position | 24 bit per axis, inside the chunk's AABB | 16 bit gave 4 % of the splat size at p99 on 3DGS and 20 % on 2DGS, because the 2DGS chunks are wide (floaters). |
| rotation | smallest-three: 3 × uint16 in [−1/√2, 1/√2] + index of the dropped component (uint16) | the largest component is always positive, so it is not stored. |
| scale | 16 bit per axis, linear in the log range of the chunk | |
| opacity | 16 bit, linear in the logit range of the chunk | |
| DC colour | 16 bit per channel, linear in the chunk's range | |
| SH | 8 bit per coefficient, linear in the chunk's range of that coefficient | 16 bit costs +45 bytes per splat and does not change anything visible at these errors. |

The chunk's SH bit depth is stored in `meta.json` (`chunks[i].sh.bits`), so it can be raised later for a
particular scene without changing the format.

### Error budget

Measured with `testC++/gs/test_gs_quant.py` against the exact values (v1). The limits are in `BUDGET`.

| Field | 3DGS PLY | 2DGS PLY | SOG | Limit |
|---|---|---|---|---|
| position, p99 relative to the splat's largest axis | 1.5e-4 | 7.9e-4 | 6e-5 | 1e-2 |
| position, max absolute | 1.5e-7 | 3.5e-4 | 2e-8 | |
| rotation, max angle | 3.1e-5 rad | 6.1e-5 rad | 3.0e-5 rad | 1e-3 rad |
| scale, max log error | 1.8e-4 | 2.4e-4 | 1.5e-4 | 1e-3 |
| opacity, max logit error | 8.4e-5 | 1.9e-4 | 7.6e-5 | 1e-3 |
| DC colour, max error | 1.0e-4 | 1.2e-4 | 7.5e-5 | 1e-3 |
| SH, rms error / rms signal | 1.5 % | 1.6 % | 1.7 % | 2 % |

The position metric uses a relative error, because it is the error that is visible on screen. The maximum relative
error is large (up to about 1e6 on 2DGS) only for a few splats with a scale near zero, whose size is far below any
visible error. The metric to watch is p99.

Position in the 2DGS file has the largest absolute error (0.35 mm), because the scene spans about 6 km in x and the
chunks are wide. That is still far below the size of the discs.

## Size

Measured on the sample files (MiB = 2^20 bytes):

| Source | Splats | Source file | `gs` v1 (float32) | `gs` v2 (default) |
|---|---|---|---|---|
| 3DGS PLY, SH3 | 958 680 | 215.8 MiB | 219.4 MiB | **73.1 MiB** (80 B/splat) |
| 2DGS PLY, SH3 | 745 202 | 173.4 MiB | 167.7 MiB | **55.4 MiB** (78 B/splat) |
| SOG v2, SH3 | 750 322 | 12.6 MiB | 171.7 MiB | **57.2 MiB** (80 B/splat) |

The v1 sizes are the sums of the column files (240 bytes per splat for 3DGS, 236 for 2DGS). The SOG source is
still much smaller (about 18 bytes per splat), because SOG uses a palette for SH and WebP compression. Entropy
coding of each column is a possible later step, but it would cost decoding time in the browser.

## Export

The export is always a standard file, never the internal format:

* **Default**: 3DGS PLY with SH degree 3 (`gs_format.write_ply`). Properties: `x y z nx ny nz f_dc_0..2 f_rest_0..44
  opacity scale_0..2 rot_0..3`, with normals written as zero, as the trainers do. The values are the decoded v2
  values, so an export differs from the source by the errors above.
* A 2DGS set keeps its two scales when written with `write_ply`. Whether a 2DGS source is exported as 2DGS PLY
  or as 3DGS PLY is still open for the export feature.
* SOG input is exported as 3DGS PLY. Its SH values are the codebook values of the palette, so they are the
  quantised version of the original, not the original trained values.

## Verification

Run inside the app image (commands in `testC++/gs/`):

```bash
python test_gs_quant.py --data DIR_WITH_SAMPLES           # error report and budget for SH 8 and 16 bit
python test_gs_import.py --data DIR_WITH_SAMPLES --out SCRATCH_DIR --ref-dir DIR_WITH_REFERENCE_PLYS
```

`test_gs_import.py` checks for each source:

1. **v1 round trip**: every column read back in source order is bit-exact.
2. **v2 round trip**: the error report is within `BUDGET`.
3. **PLY round trip of v2**: every field is bit-exact after reading, except the quaternion, which is compared
   within 1e-6 after the second normalisation.
4. **Reference**: the output of PlayCanvas splat-transform 3.9.0 (the tool named in the SOG `meta.json`). The SOG
   output agrees with our reader (max difference 0 apart from the quaternion, 6e-8). For the 3DGS and 2DGS PLY files,
   the reference also agrees field by field, which confirms the channel-major `f_rest` convention.

The reference PLY files are generated with `node bin/cli.mjs INPUT OUT.ply` from `@playcanvas/splat-transform@3.9.0`.
That package needs the `webgpu` module, which does not load on Alpine: a stub is enough for CPU-only reading and writing.

## Integration in the main viewer

Import: **File > Load Data**, then **Upload & Process**, accepts `.ply`, `.sog`, `.las`, `.glb`.

1. A `.ply` with the 3DGS/2DGS properties, or a `.sog`, goes to `POST /api/gs-import/` (`viewer/gs_import.py`). The file
   is converted to the `gs` v2 folder under `runtime_data/working/gs`. Any other PLY goes through the point cloud path,
   unchanged.
2. The frontend (`static/viewer/js/functions.js`) skips the point cloud steps and calls `loadGaussianSplats`
   (`static/viewer/js/gs-splat.js`). The splats are a Babylon mesh with one `ShaderMaterial`, drawn with the same shaders
   as the test renderer. The depth order is computed in `gs-sort-worker.js`.
3. The camera is framed on the splats (target = median, distance from the 90th percentile of the radius).

Checked in the app (software WebGL2 in headless Chromium, 900×600-ish window, full scenes):

| File | Result |
|---|---|
| `3DGS.ply` (958 680 splats, SH 3) | converted, registered, rendered (the insect) |
| `2dgs_cardboard_point_cloud.ply` (745 202 splats) | converted, registered, rendered (box and discs) |
| `The Fruit Bowl.sog` (750 322 splats) | converted, registered, rendered |
| `Paestum.ply` (normal point cloud) | unchanged: point cloud loader registered |

Limits of this integration (not yet implemented):

* **Orthographic camera:** the splat layer is hidden. Perspective only.
* **No LOD:** all splats are drawn. A scene of millions of splats will need LOD (phase 3).
* **No editing:** selection, cut, classes, RF features, export and the other point cloud tools do not work on splats.
  The tool panels stay visible.
* **Transform:** the splats are not moved by the point cloud root transform, and there is no X mirror.
* **Import time:** the conversion runs synchronously in the request (a few seconds for the samples).

The check in the app is `testC++/gs/viewer/gs_import_e2e.js` (Playwright, shared network namespace with the app
container, because `ALLOWED_HOSTS` only accepts localhost).

## Test renderer (not the production renderer)

`testC++/gs/viewer/index.html` is a standalone WebGL2 page that opens a `gs` v2 folder under `/data/<name>`.
It exists to check the format and the shaders, not to be the viewer:

* **Decoding** (`loadFolder`): the same arithmetic as `gs_quant.decode`, on the main thread. The JavaScript and
  Python decoders agree on sums over the whole scene to about 3e-9 relative (e.g. 3DGS alpha sum
  240079.9906 in JavaScript, 240079.9900 in Python).
* **3DGS**: one instanced quad per splat. The vertex shader projects the 3D covariance (EWA, with the usual
  0.3 px low-pass term and the 1.3 frustum clamp), the quad covers 3 sigma of the 2D footprint, and the fragment
  shader applies the Gaussian with premultiplied alpha, drawn back to front.
* **2DGS**: each splat is a disc. The quad covers 3 sigma along the two axes, and the fragment shader intersects
  the view ray with the disc plane (the fragment's world position lies on the plane), then applies the weight.
* **Colour**: SH of degree 3 evaluated per splat with the view direction from the camera, as the INRIA code does.
* **Sorting** (`sort-worker.js`): a Web Worker holds the positions (copied once) and returns the splat order for a
  view matrix, by a two-pass 16-bit radix sort of the view-space depth. The order comes back as a transferred
  buffer. At most one request is in flight, and a new one is sent only when the view changed. The draw uses the
  newest order it has, so after a camera move the image can show the previous order for one sort time.

Run it with `testC++/gs/viewer/run.sh <folder> [png]` (Playwright 1.49.1 in the official container, software WebGL2 via
SwiftShader). Options: `STRIDE=8` draws every 8th splat (a uniform thinning, needed for the software renderer),
`NODRAW=1` only decodes, `YAW`, `PITCH`, `DIST` set the camera.

Results (STRIDE=8, 900×600): the three samples are recognisable (the insect, the fruit bowl, and the cardboard
object in the middle of the 2DGS floaters). Edge-on discs show as thin lines, as expected for discs.
Full-scene decode on the main thread, single run in the container: 3DGS 2.1 s, 2DGS 1.5 s, SOG 1.6 s.

### Comparison with a reference render

`testC++/gs/viewer/reference_render.py` is a numpy renderer written from the formulas: EWA for 3DGS
(`Sigma' = J W Sigma W^T J^T`, 1.3 frustum clamp, 0.3 px low-pass, 3 sigma footprint) and an exact ray/disc
intersection for 2DGS. It reads the same gs v2 folder, the same subset of splats (every 8th row), and the
same camera as the page. `compare_renders.py` compares the two images, excluding the HUD box.

The comparison found a real bug: the shader computed `Tᵀ Sigma T` with `T = W J`, which is `Jᵀ Wᵀ Sigma W J`, not the
standard `J W Sigma Wᵀ Jᵀ`. The footprints were wrong whenever the camera was rotated. It is fixed in `index.html`.

| Case (900×600, same camera) | pixels with diff > 16 | mean diff over splat-covered pixels | max |
|---|---|---|---|
| 3DGS, before the fix | 8.0 % | 15.3 | 174 |
| 3DGS, after the fix | 0.17 % | 1.8 | 112 |
| 2DGS (disc ray intersection) | 0.20 % | 1.0 | 129 |

The 2DGS disc rendering agrees with the independent ray intersection, with no systematic pattern in the residual.
The remaining differences are sparse points, mostly on bright specular splats. I did not find their cause. They could be
the order of splats whose depths differ by less than 1e-4, or float32 against float64 arithmetic. I did not test either.

Limit of this check: the reference is my own code, written from the formulas. It is independent of the shader code, but
not of my understanding of the formulas. An external renderer (for example PlayCanvas on the same file) would be a stronger
check of colour, SH and orientation.

What is **not** verified yet:

* **Colour and disc orientation against an external renderer.** The page agrees with the numpy reference (above),
  but that reference is my own code. An external renderer on the same file and camera is still needed.
* **Performance.** SwiftShader is a CPU rasteriser, so frame times here mean nothing for a GPU. Measured sort in the
  worker, full 3DGS scene (958 680 splats), 3 runs in the container: median 59 ms (55–72 ms). The main-thread
  version was not measured at full size, so the worker's advantage is not quantified; the earlier 130 ms estimate
  for 1 M on the main thread was an extrapolation and should not be quoted.
* **Sort correctness.** Checked in the page (`?bench=N`): the order returned by the worker never decreases in depth
  by more than 1e-4 (0 violations in 3 runs, 958 k splats). Smaller "inversions" are float32 rounding of the depth
  and are not errors. The image changed by about 13 000 pixels (max 198) compared with the earlier main-thread
  render, and two worker renders differ by 61 pixels. I attribute the difference to the order of splats whose depths
  differ by less than 1e-4, but I have not confirmed it bit by bit, because the earlier code no longer exists.

## GPU measurement protocol

The numbers above come from a software renderer and do not describe a GPU. To measure on a real GPU:

1. Start the server: `testC++/gs/viewer/serve.sh` (needs Docker, uses the same image as the tests).
2. Open the three pages in a Chrome or Edge window at 1280×720. Full scenes, no subsampling:
   * `http://localhost:8080/viewer/index.html?set=3dgs_gs&w=1280&h=720&spin=300`
   * `http://localhost:8080/viewer/index.html?set=2dgs_gs&w=1280&h=720&spin=300`
   * `http://localhost:8080/viewer/index.html?set=fruit_sog_gs&w=1280&h=720&spin=300`
3. Each page turns the camera for 300 frames and logs `spinResult` in the console (F12). It reports the frame
   interval (mean, p95, worst), the draw time (CPU side, includes `gl.finish`), and the number of sorts.
4. For the sort alone, add `&nodraw=1&bench=5`: the page logs `benchResult` with the worker sort time for five
   views on the full scene.
5. The HUD shows the load time (JavaScript decode), the sort time and latency, and the draw time.

Report: GPU model, browser and version, the `spinResult` and `benchResult` lines for each scene, and the load time.

## Limits

* The whole source is loaded in memory during the conversion (about 1.2 GB for 5 M splats with SH3). Streaming
  conversion is needed for larger files.
* No LOD yet: all splats of a chunk are present.
* No renderer for 2DGS discs yet. Babylon's `GaussianSplattingMesh` does not render them.
* SPZ input is not supported.
