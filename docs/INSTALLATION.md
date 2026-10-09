# Installation Guide

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Requirements](#software-requirements)
3. [Building the Docker Image](#building-the-docker-image)
4. [Running the Application](#running-the-application)
   - [Production Mode (Gunicorn)](#production-mode-gunicorn)
   - [Development Mode (Django Dev Server)](#development-mode-django-dev-server)
5. [Memory, Scratch Space and Storage](#memory-scratch-space-and-storage)
6. [Environment Validation](#environment-validation)
7. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA with Compute Capability ≥ 7.5 (Turing / RTX 20xx series or newer) |
| VRAM | ≥ 8 GB recommended; minimum 4 GB for small point clouds |
| RAM | The pipeline is out-of-core: the point cloud does not have to fit in memory (see [Memory, Scratch Space and Storage](#memory-scratch-space-and-storage)); more RAM only makes the in-memory fast paths available |
| CPU | Multi-core (≥ 8 cores recommended to speed up the C++ build stage) |
| Disk | ≥ 20 GB free for the Docker image plus runtime data |

> **CPU-only mode:** The application automatically falls back to the CPU binary
> (`feature_extraction_viewer_cpu`) when no CUDA-capable GPU is detected. Feature
> extraction and Random Forest inference will be significantly slower.

---

## Software Requirements

| Software | Version | Notes |
|---|---|---|
| Docker | ≥ 20.10 | |
| NVIDIA Driver | ≥ 520 | Required for CUDA 11.8 support |
| nvidia-container-toolkit | Current | Enables GPU passthrough inside Docker containers |

### Installing nvidia-container-toolkit (Ubuntu/Debian)

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo apt-key add -

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Building the Docker Image

The `Dockerfile` uses a **multi-stage build**:

- **Stage 1 — builder** (`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`): compiles
  LASzip, laz-perf, GDAL 3.6.2, and PDAL 2.7.1 from source, then builds all eight
  C++ processing binaries.
- **Stage 2 — runtime** (`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`): installs
  Python 3.10, runtime system libraries, and Python packages (including PyTorch 2.3.0
  + CUDA 11.8 and RAPIDS cuML), then copies the compiled binaries from Stage 1.

### Standard build

```bash
docker build -t 3d-classify-viewer .
```

### Build with parallel compilation (recommended)

Set `NUM_THREADS` to the number of available CPU cores to speed up the C++ compilation:

```bash
docker build --build-arg NUM_THREADS=16 -t 3d-classify-viewer .
```

> **Note:** The first build takes 20–40 minutes depending on network speed and CPU
> core count, as it compiles GDAL, PDAL, and multiple C++ binaries from source.
> Subsequent builds benefit from Docker layer cache and complete significantly faster
> when only Python or Django files are changed.

---

## Running the Application

### Production Mode (Gunicorn)

```bash
docker run -d \
  -p 8000:8000 \
  --gpus all \
  --name classify-viewer \
  3d-classify-viewer
```

The application is served by Gunicorn (1 worker, 8 threads, no timeout) on port **8000**.
Open `http://localhost:8000` in a browser.

To **persist runtime data** (uploaded point clouds, trained models) between container
restarts, mount a host directory:

```bash
docker run -d \
  -p 8000:8000 \
  --gpus all \
  -v "$(pwd)/runtime_data:/webapp/classifyViewer/runtime_data" \
  --name classify-viewer \
  3d-classify-viewer
```

### Development Mode (Django Dev Server)

```bash
docker run -it \
  -p 8000:8000 \
  --gpus all \
  -v "$(pwd)/classifyViewer:/webapp/classifyViewer" \
  3d-classify-viewer \
  python classifyViewer/manage.py runserver 0.0.0.0:8000
```

Mounting the source directory enables live code reloading. Changes to Python files
and templates take effect immediately without rebuilding the image.

---

---

## Memory, Scratch Space and Storage

The server-side pipeline (point cloud build, columns, Random Forest, split, feature extraction, subsampling)
does not need the point cloud to fit in memory: every tool works in blocks with a **memory budget** and only
reads/writes the big files sequentially. Large clouds therefore run in a container with much less memory than the
size of `features.las`.

### Memory budget

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `PIPELINE_MEMORY_BUDGET_MB` | environment variable (read by `settings.py`) | unset | Peak memory a job may use. Unset = **50 % of `min(container memory limit, MemAvailable)`**. |
| `--memory-budget MB` | every tool (`las2pc`, `pc_columns.py`, `RF_classify.py`, `RF_training.py`, `feature_extraction_*`, ...) | from the setting above | Same, for one run. |

Limit the container memory with Docker (the page cache of the files read and written is counted in the limit,
but it is reclaimable once clean; the tools drop the pages they have finished with):

```bash
docker run -d -p 8000:8000 --gpus all \
  --memory 8g --memory-swap 8g \
  -e PIPELINE_MEMORY_BUDGET_MB=4096 \
  -v classify_runtime:/webapp/classifyViewer/runtime_data \
  --name classify-viewer 3d-classify-viewer
```

With the default budget `las2pc` picks its **in-memory** mode (fast, about 28 bytes/point + the LAS in the page
cache) when it fits and its **out-of-core** mode otherwise (the output is identical byte for byte). The log says
which one was used (`mode=memory|ooc`).

### Scratch folder

| Setting | Default |
|---|---|
| `PIPELINE_TEMP_DIR` (environment) / `--temp-dir` (tools) | `/tmp/pipeline_work` |

Out-of-core steps write temporary files there (one sub-folder per job): `las2pc` needs about *LAS size + 4 bytes/point*,
the feature extractors about *(1 + buffer overhead) × 36 bytes/point*, `pc_columns.py` (only when the LAS is not in
canonical order) a few tens of bytes/point. The folder is removed when the job ends, fails or is stopped
(`/stop_process/` sends `SIGTERM` to the job, which cleans up before exiting). Before starting, every tool checks the
free space and stops with a message that says how much is needed.

**Keep the scratch folder on the container filesystem (or another fast local disk), never inside `runtime_data/`.**

### Where to keep `runtime_data/`

`runtime_data/` holds `features.las`, `pc/` (the chunked point cloud), the models and the annotations. It is
read and written sequentially in big blocks, but the viewer also reads `pc/` with HTTP Range requests, so the storage
must have decent random access.

* **Do not put `runtime_data/` on a Windows bind mount (WSL2/Docker Desktop `-v C:\...` or `-v D:\...`) or on a
  OneDrive/Dropbox folder in production.** Measured on Docker Desktop with the data on a OneDrive folder: ~190 MB/s
  sequential and only ~6 000 random reads of 50 bytes per second (every access crosses the 9p/drvfs filesystem), against
  ~460 MB/s cold sequential reads on a Docker volume; the synchronisation client also tries to upload files of several
  GB while they are being written.
* In development use a Docker **named volume** (or a path inside the WSL2 filesystem):

  ```bash
  docker volume create classify_runtime
  docker run -it -p 8000:8000 --gpus all \
    -v "$(pwd)/classifyViewer:/webapp/classifyViewer" \
    -v classify_runtime:/webapp/classifyViewer/runtime_data \
    3d-classify-viewer python classifyViewer/manage.py runserver 0.0.0.0:8000
  ```

* Disk space: besides `features.las` (and its backup, same size) the build needs the same amount again for a moment
  (`features_ordered.las` replaces `features.las` at the end), plus the scratch folder.

---
## Environment Validation

After starting the container, run the following checks to verify the environment.

### Check GPU availability

```bash
docker exec classify-viewer nvidia-smi
```

Expected: the NVIDIA GPU, driver version, and CUDA version are listed.

### Check runtime libraries

```bash
docker exec classify-viewer ldconfig -p | grep -E 'gdal|pdal|laszip|pcl'
```

Expected: entries for `libgdal`, `libpdal_base`, `liblaszip`, and `libpcl_*`.

### Check C++ binaries

```bash
docker exec classify-viewer ls -la /webapp/opt/
```

Expected: seven executable files — `feature_extraction_viewer_gpu`,
`feature_extraction_viewer_cpu`, `subsample_pc`, `mesh2pc`, `ply2las`,
`split_las_by_binary`, `check_point_id`.

### Check Python / CUDA environment

```bash
docker exec classify-viewer python3 -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True` when a CUDA-capable GPU is present and the driver is correctly
configured.

---

## Troubleshooting

### `nvidia-smi` not found inside the container

Ensure `nvidia-container-toolkit` is installed on the host and Docker was restarted
after installation. The `--gpus all` flag must be passed to `docker run`.

### `cannot open shared object file: libgdal.so`

Run `ldconfig` inside the container to refresh the dynamic linker cache, then verify
the library path:

```bash
docker exec classify-viewer sh -c 'echo $LD_LIBRARY_PATH'
```

The path must include `/usr/local/lib`.

### `CUDA out of memory` during feature extraction

Increase the subsampling voxel size before running feature extraction to reduce point
density, or set `use_gpu: false` in the feature extraction request to use the CPU
fallback binary.

### Port 8000 already in use

Map to a different host port:

```bash
docker run -d -p 9000:8000 --gpus all 3d-classify-viewer
```

### Large file uploads failing

The default upload limit is 5 GB (`DATA_UPLOAD_MAX_MEMORY_SIZE` in `settings.py`).
For files exceeding this limit, increase the value and rebuild the image.
