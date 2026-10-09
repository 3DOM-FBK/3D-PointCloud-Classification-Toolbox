# syntax=docker/dockerfile:1.7
# =============================================================================
# 3D Classify Viewer — lean runtime image (linux/amd64)
#
# What the code really needs at runtime:
#   * Python 3.10 + Django/gunicorn/whitenoise + numpy/scipy/scikit-learn/laspy/tqdm
#   * opt/ binaries (prebuilt, x86-64):
#       las2pc, split_las_by_binary, check_point_id   libgomp only (no PDAL, no Open3D)
#       feature_extraction_viewer_cpu                 libpcl_common 1.12 (+ libgomp)
#       feature_extraction_viewer_gpu                 libgomp; the CUDA runtime is linked statically and
#                                                     the driver (libcuda) is injected by the NVIDIA
#                                                     container toolkit (docker run --gpus all)
#       ply2las, subsample_pc, mesh2pc                libOpen3D.so (+ libtbb, libc++, libGL, libX11);
#                                                     mesh2pc also GMP/MPFR/libomp
#   * the three BabylonJS bundles the page loads (the rest of the BabylonJS distribution is dev tooling)
# Nothing is compiled here: no CUDA toolkit, GDAL, PDAL, LASzip, PCL/CGAL/Qt/VTK development packages,
# torch or RAPIDS. GPU Random Forest (cuML) is an opt-in:  --build-arg WITH_RAPIDS=1
# =============================================================================
ARG BASE=ubuntu:22.04

# ── Stage 1: Python environment (a venv copied as a whole into the later stages) ──
FROM ${BASE} AS pydeps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
ARG WITH_RAPIDS=0
RUN python3.10 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    # opt-in GPU Random Forest. RAPIDS 25.06 is the last release with CUDA 11 wheels (dropped in 25.08);
    # the old nightly index (pypi.anaconda.org/rapidsai-wheels-nightly) is no longer used.
    && if [ "$WITH_RAPIDS" = "1" ]; then \
         /opt/venv/bin/pip install --no-cache-dir cupy-cuda11x "cuml-cu11==25.6.*" --extra-index-url https://pypi.nvidia.com; \
       fi \
    # runtime only: no pip/setuptools, no bytecode caches, no test suites (numpy.testing imports
    # numpy/_core/tests at runtime, so numpy keeps its tests)
    && cd /opt/venv/lib/python3.10/site-packages \
    && rm -rf pip pip-* setuptools setuptools-* pkg_resources _distutils_hack distutils-precedence.pth \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && find scipy sklearn joblib laspy -type d -name tests -prune -exec rm -rf {}  +

# ── Stage 2: Open3D shared library only (the "devel" tarball is ~1 GB with headers/cmake/static libs) ──
FROM ${BASE} AS open3d
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget xz-utils \
    && rm -rf /var/lib/apt/lists/*
ARG OPEN3D_VERSION=0.19.0
ARG OPEN3D_SHA256=2e525fd2afe7e80907d2b6e3c66b69e2ef1481dbceed5727b749a6e65cba5720
RUN set -eux; \
    T=open3d-devel-linux-x86_64-cxx11-abi-${OPEN3D_VERSION}.tar.xz; \
    wget -q -O /tmp/$T https://github.com/isl-org/Open3D/releases/download/v${OPEN3D_VERSION}/$T; \
    echo "$OPEN3D_SHA256  /tmp/$T" | sha256sum -c -; \
    L=/o3d/lib; mkdir -p $L; \
    tar -xJf /tmp/$T -C $L --strip-components=2 --wildcards '*/lib/libOpen3D.so.0.19.0' '*/lib/libtbb.so.12.12'; \
    ln -s libOpen3D.so.0.19.0 $L/libOpen3D.so.0.19; ln -s libOpen3D.so.0.19 $L/libOpen3D.so; \
    ln -s libtbb.so.12.12 $L/libtbb.so.12; \
    rm -f /tmp/$T

# ── Stage 3: application tree with the static files already collected ──
# (done in its own stage so that the BabylonJS distribution, its source maps and the uncollected
#  copy of the static files never become layers of the final image)
FROM ${BASE} AS app
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=pydeps /opt/venv /opt/venv
COPY classifyViewer/ /webapp/classifyViewer/
WORKDIR /webapp/classifyViewer
RUN set -eux; \
    B=viewer/static/viewer/js/babylon_js; \
    # keep only what viewer_page.html loads: babylon.js, materialsLibrary/, loaders/ (no editors, inspector, physics, codecs, maps, typings)
    find $B -mindepth 1 -maxdepth 1 ! -name babylon.js ! -name materialsLibrary ! -name loaders -exec rm -rf {} +; \
    find $B \( -name '*.map' -o -name '*.d.ts' \) -delete; \
    python manage.py collectstatic --noinput; \
    # whitenoise serves STATIC_ROOT only
    rm -rf viewer/static

# ── Stage 4: runtime ──
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    # GPU feature extraction: the NVIDIA container toolkit mounts the driver (libcuda) for these
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Shared libraries only (no -dev packages, no compilers).
# libGL.so.1 is only needed to LOAD libOpen3D (the tools never open a GL context). In jammy libglx0 hard-depends
# on Mesa (libglx-mesa0 -> libgl1-mesa-dri -> LLVM 15, ~190 MB), so the three GLVND packages are installed
# with dpkg --force-depends instead; apt is not used afterwards.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        libgomp1 \
        libpcl-common1.12 \
        libomp5 libc++1 libc++abi1 \
        libgmp10 libgmpxx4ldbl libmpfr6 \
        libx11-6 \
    && cd /tmp && apt-get download libglvnd0 libgl1 libglx0 \
    && dpkg -i --force-depends ./libglvnd0_*.deb ./libgl1_*.deb ./libglx0_*.deb \
    && rm -rf /tmp/*.deb /var/lib/apt/lists/*

COPY --from=pydeps /opt/venv /opt/venv
# Same path the binaries were linked with (RPATH), so no LD_LIBRARY_PATH is needed
COPY --from=open3d /o3d/lib /app/open3d-devel-linux-x86_64-cxx11-abi-0.19.0/lib

# Application (last: it is what changes most often)
COPY --chmod=755 opt/ /webapp/opt/
COPY --from=app /webapp/classifyViewer /webapp/classifyViewer
WORKDIR /webapp/classifyViewer

EXPOSE 8000
ENTRYPOINT ["gunicorn", "classifyViewer.wsgi:application", "--config", "config/gunicorn.conf.py"]
