# 3D PointCloud Classification Toolbox

A web-based application for interactive visualization and supervised classification of
3D point clouds. It combines a Django REST backend, a BabylonJS + Potree 2.0 frontend,
and a GPU-accelerated C++ processing pipeline to support the full workflow — from raw
3D data upload to labeling, feature extraction, Random Forest training, and classified
output export.

## Key Features

- **Multi-format input** — PLY, LAS, GLB, GLTF meshes are automatically
  sampled to point clouds
- **GPU-accelerated feature extraction** — per-point geometric descriptors (anisotropy,
  linearity, planarity, etc.) computed at multiple radius using CUDA; CPU fallback
  included
- **Interactive point selection and labeling** — rectangular, lasso, and polygon
  selection tools for annotating training regions directly in the 3D viewport
- **Random Forest classification** — GPU-accelerated training and inference via RAPIDS
  cuML; scikit-learn CPU fallback for non-GPU environments
- **LOD streaming** — large point clouds streamed with adaptive level-of-detail via
  Potree 2.0
- **Prediction visualization** — discrete per-class coloring and confidence overlay
- **Export** — download classified segments and trained models as a single ZIP package

## Architecture Overview

The application is structured in three layers:

1. **Frontend** — BabylonJS 3D scene with Potree 2.0 LOD streaming. All user
   interaction (selection tools, class registry, colormap, segment management) runs
   in the browser.
2. **Django backend** — REST API with 30+ endpoints that orchestrate file uploads,
   C++ pipeline invocation, model management, and file serving (including HTTP Range
   support for large binary files).
3. **C++ processing pipeline** — eight pre-compiled binaries at `/webapp/opt/` inside
   the container, covering format conversion, subsampling, GPU/CPU feature extraction,
   Potree conversion, and `.pcbin` binary export. Invoked from Python via `subprocess`.

For a detailed breakdown, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Prerequisites

| Requirement | Version |
|---|---|
| Docker | ≥ 20.10 |
| NVIDIA GPU | Compute Capability ≥ 7.5 (Turing / RTX 20xx or newer) |
| NVIDIA Driver | ≥ 520 |
| VRAM | ≥ 8 GB recommended for large point clouds |
| nvidia-container-toolkit | any current release |

> The application falls back to CPU for feature extraction and RF inference when no
> CUDA-capable GPU is detected, but processing times will be significantly longer.

## Quick Start

A pre-built Docker image is available on Docker Hub and can be used directly without
compiling from source:

```bash
docker pull 3domfbk/3d-classify-viewer:v1.0.0-beta.9
```

### 1. Build the Docker image

Skip this step if you are using the pre-built image from Docker Hub.

```bash
docker build -t 3d-classify-viewer .
```

To speed up the C++ compilation stage, increase the number of parallel build threads:

```bash
docker build --build-arg NUM_THREADS=16 -t 3d-classify-viewer .
```

### 2. Run in production

Replace `3d-classify-viewer` with `3domfbk/3d-classify-viewer:v1.0.0-beta.9` if using
the pre-built image.

```bash
docker run -d -p 8000:8000 --gpus all 3d-classify-viewer
```

The application is served by Gunicorn on port **8000**. Open `http://localhost:8000`
in a browser.

### 3. Run in development

```bash
docker run -it -p 8000:8000 --gpus all \
  -v "$(pwd)/classifyViewer:/webapp/classifyViewer" \
  3d-classify-viewer \
  python classifyViewer/manage.py runserver 0.0.0.0:8000
```

Mounting the source directory enables live code reloading without rebuilding the image.

## Project Structure

```
3d-classify-viewer/
├── Dockerfile                      # Multi-stage build: CUDA 11.8 builder → runtime image
├── requirements.txt                # Python dependencies
├── classifyViewer/
│   ├── manage.py
│   ├── config/
│   │   └── gunicorn.conf.py        # Gunicorn production server configuration
│   ├── classifyViewer/             # Django project configuration
│   │   ├── settings.py
│   │   └── urls.py
│   └── viewer/                     # Main Django application
│       ├── views.py                # Page views (viewer and documentation)
│       ├── urls.py                 # URL patterns and API routes
│       ├── request_functions.py    # REST API request handlers
│       ├── functions.py            # JobManager and C++ subprocess wrappers
│       ├── utils_functions/
│       │   ├── RF_training.py      # Random Forest training
│       │   └── RF_classify.py      # Random Forest inference (CPU/GPU)
│       ├── static/viewer/
│       │   ├── js/                 # BabylonJS app, Potree 2.0 loader, UI logic
│       │   ├── css/                # Application and documentation styles
│       │   └── images/             # Workflow diagrams
│       └── templates/viewer/
│           ├── viewer_page.html    # Main 3D viewer page
│           └── docs_page.html      # In-app documentation page
└── opt/                            # Pre-compiled C++ processing binaries
    ├── feature_extraction_viewer_gpu
    ├── feature_extraction_viewer_cpu
    ├── subsample_pc
    ├── mesh2pc
    ├── ply2las
    ├── split_las_by_binary
    ├── las_to_feature_bin
    └── check_point_id
```

## Documentation

| Document | Description |
|---|---|
| [Installation Guide](docs/INSTALLATION.md) | Hardware and software requirements, Docker setup, environment validation |
| [User Guide](docs/USER_GUIDE.md) | Full usage walkthrough: data loading, labeling, training, classification, export |
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, Python ↔ C++ integration, ML pipeline |
| [API Reference](docs/API_REFERENCE.md) | All REST endpoints with request and response schemas |
| [C++ Binaries Reference](docs/BINARIES.md) | CLI reference for all processing binaries in `opt/` |

## Python Dependencies

Key Python packages (`requirements.txt` for the full list):

| Package | Purpose |
|---|---|
| Django | Web framework |
| Gunicorn + Whitenoise | Production WSGI server and static file serving |
| scikit-learn | Random Forest training and inference (CPU) |
| numpy, pandas | Numerical computation and data handling |
| laspy | LAS/LAZ point cloud I/O |
| open3d | Point cloud processing utilities |
| trimesh, pygltflib | Mesh file I/O (GLB/GLTF/OBJ) |
| torch (CUDA 11.8) | GPU compute backend |
| cupy-cuda11x, cuml-cu11 | GPU-accelerated ML via RAPIDS |
