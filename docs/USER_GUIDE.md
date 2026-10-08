# User Guide

This guide walks through the complete workflow: loading 3D data, annotating training
points, training a classifier, running inference, and exporting results.

## Table of Contents

1. [Supported Input Formats](#supported-input-formats)
2. [Loading Data](#loading-data)
3. [3D Viewport Navigation](#3d-viewport-navigation)
4. [Training Mode](#training-mode)
   - [Adding Classes](#adding-classes)
   - [Selecting Points](#selecting-points)
   - [Assigning Labels](#assigning-labels)
   - [Feature Extraction](#feature-extraction)
   - [Training a Model](#training-a-model)
5. [Classification Mode](#classification-mode)
6. [Visualizing Results](#visualizing-results)
7. [Export and Download](#export-and-download)

---

## Supported Input Formats

| Format | Extension | Notes |
|---|---|---|
| LAS / LAZ | `.las`, `.laz` | Native point cloud format; recommended input |
| PLY | `.ply` | Converted to LAS automatically on upload |
| GLB / GLTF | `.glb`, `.gltf` | Mesh sampled to a point cloud |

> **File size limit:** The maximum upload size is 5 GB. For very large files, it is
> recommended to pre-subsample the point cloud externally before uploading.

---

## Loading Data

1. Click **File → Load Data** in the top menu bar.
2. Select your file. The application detects the format and applies the appropriate
   conversion pipeline automatically:
   - **PLY** → converted to LAS via `ply2las`
   - **GLB / GLTF** → surface-sampled to a point cloud via `mesh2pc`, then
     converted to LAS
   - **LAS / LAZ** → loaded directly
3. **Subsampling (optional but recommended):** If the point cloud is very dense, a
   voxel-based subsampling step is offered. A larger voxel size reduces point count
   and speeds up all subsequent operations.
4. **Point ID check:** The pipeline validates and normalizes POINT_ID values to ensure
   canonical 0-based sequential indexing, which is required by the annotation system.
5. **Feature extraction:** Geometric features are computed at one or more radii (see
   [Feature Extraction](#feature-extraction)).
6. **Point cloud conversion:** The processed LAS is converted once into the chunked point cloud (multi-resolution geometry plus one column per attribute) that is streamed to the 3D viewport.

Once complete, the point cloud is displayed and ready for interaction.

---

## 3D Viewport Navigation

| Action | Control |
|---|---|
| Orbit (rotate view) | Left mouse button drag |
| Pan | Middle mouse button drag (or Shift + left drag) |
| Zoom | Mouse wheel |
| Reset camera | Double-click on the viewport |

Point size and rendering quality can be adjusted via controls in the right-side panel.

---

## Training Mode

Switch to **Training Mode** using the toggle in the navigation bar. This mode enables
point selection and class labeling.

### Adding Classes

1. Open the **Classes** panel on the right side.
2. Click **Add Class**, then enter a name and pick a display color.
3. Repeat for each semantic class in your classification scheme.

### Selecting Points

Three selection tools are available in the toolbar:

| Tool | Description |
|---|---|
| **Rectangle selection** | Draw a 2D bounding rectangle over the viewport |
| **Lasso selection** | Draw a freehand closed shape |
| **Polygon selection** | Click to place vertices defining a polygonal selection area |

Selected points are highlighted in the viewport.

### Assigning Labels

After making a selection:

1. Right-click within the selection to open the context menu (Training Mode only).
2. Assign the selected points to a named **segment** (a spatial region) and optionally
   to a **class** (semantic label).
3. Annotations are kept in the viewer and saved on the server as `annotations.bin` (2 bytes per point) when needed (training, classification of a segment).

> **Segments and classes** are distinct concepts. A segment is a named spatial region
> defined by a selection; a class is the semantic label (e.g. *ground*, *vegetation*)
> assigned to points within that segment. Multiple segments can share the same class.

### Feature Extraction

Feature extraction computes per-point geometric descriptors used as input features for
the Random Forest classifier. It is the most compute-intensive step in the pipeline.

**Available features** include: anisotropy, linearity, planarity, sphericity,
omnivariance, eigenentropy, change of curvature, and additional neighborhood statistics.

**Configuration parameters:**

| Parameter | Description |
|---|---|
| **Feature list** | Names of geometric features to compute (e.g. `anisotropy`, `linearity`, `planarity`) |
| **Radius list** | One or more neighborhood radii in scene units. Using multiple radii captures multi-scale structure (e.g. `0.5, 1.0, 2.0`) |
| **Sampling resolution** | Subsampling factor applied before computation. Higher values reduce computation time at the cost of accuracy. Set to `0` to use the full point cloud |
| **GPU / CPU** | GPU binary is selected by default when a CUDA device is available; fallback to CPU otherwise |

Real-time progress is reported in the status panel during extraction.

When extraction finishes only the new feature columns are written (the geometry is not
converted again) and the viewer picks them up without reloading the cloud. Your segments,
classes, selections, camera and view settings are untouched. The same happens after a
classification (the `prediction` column is written; points outside the classified segment have
no prediction and are shown grey) and after restoring the backup. The features are then
available in the colour menu, with their value range taken from the point cloud metadata.

### Training a Model

Once sufficient annotated points exist for at least two distinct classes:

1. Click **Train Model** in the Training panel.
2. Enter a name for the model.
3. The backend prepares the training data from annotated segments, trains a Random
   Forest classifier, and saves the model with a performance report to
   `runtime_data/models/<model_name>/`.
4. Training metrics (accuracy, F1 score, confusion matrix) are displayed on completion.

---

## Classification Mode

Switch to **Classify Mode** using the navigation bar toggle.

1. Select a trained model from the **Models** list in the right panel.
2. Click **Run Classification**.
3. The RF classifier runs inference on all points in the current point cloud using
   the extracted features. Feature names and radii must match those used during training.
4. Predictions are written back to the point cloud; the viewport updates to show
   per-class coloring.

**Confidence:** Each point receives a prediction confidence score (0–1). A confidence
overlay can be toggled in the visualization panel.

> If no features have been extracted yet, or if the feature set does not match the
> model's training configuration, run feature extraction first using the same
> parameters (feature names and radii) as when the model was trained.

---

## Visualizing Results

The coloring mode can be switched in the right panel:

| Mode | Description |
|---|---|
| **Class color** | Each point colored by its predicted class using the class registry colors |
| **Confidence** | Gradient coloring from low (cold tones) to high (warm tones) confidence |
| **Feature** | Value of any extracted feature dimension mapped to a color gradient |
| **Original** | Point cloud rendered with its original RGB values or elevation-based coloring |

---

## Export and Download

Click **File → Download Data** to open the export dialog.

**Available exports:**

| Item | Format | Description |
|---|---|---|
| Classified segments | `.las` | One LAS file per labeled segment |
| Full point cloud | `.las` | Complete point cloud with prediction labels |
| Trained model | `.pkl` | Serialized Random Forest model for external use |
| Package | `.zip` | All selected segments, models, and metadata in a single archive |

The ZIP package is assembled on the server and streamed to the browser on completion.
Temporary segment files are deleted from the server after the download is prepared.
