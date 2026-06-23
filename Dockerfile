# ── Seleziona base image in base all'architettura target ───
# amd64 → CUDA 11.8 full stack  |  arm64 → Ubuntu base (no CUDA)
# Docker buildx imposta automaticamente TARGETARCH (amd64 / arm64).
# Per build normali senza --platform il default è amd64.
ARG TARGETARCH=amd64
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder-amd64
FROM ubuntu:22.04                                 AS builder-arm64

# ============================================================
# STAGE 1 — builder
# Compila GDAL, PDAL, LASzip, laz-perf da sorgente.
# Questa immagine NON finisce in produzione.
# ============================================================
FROM builder-${TARGETARCH} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ARG NUM_THREADS=8

# Dipendenze di compilazione (solo quanto serve per build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates gnupg && \
    add-apt-repository universe && \
    apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    libflann-dev libjpeg-dev libpng-dev libtiff-dev \
    libpcl-dev libpq-dev \
    libx11-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    wget curl unzip \
    libgomp1 libomp-dev liblaszip-dev \
    libtbb-dev \
    # Solo i moduli Boost necessari a GDAL/PDAL
    libboost-filesystem-dev libboost-iostreams-dev \
    libboost-program-options-dev libboost-system-dev \
    libboost-thread-dev libboost-regex-dev \
    libcgal-dev \
    python3.10 python3.10-dev python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# LASzip
RUN git clone --depth 1 https://github.com/LASzip/LASzip.git /tmp/LASzip && \
    cmake -S /tmp/LASzip -B /tmp/LASzip/build -DCMAKE_BUILD_TYPE=Release && \
    make -C /tmp/LASzip/build -j${NUM_THREADS} install && \
    rm -rf /tmp/LASzip

# laz-perf
RUN git clone --depth 1 https://github.com/hobu/laz-perf.git /tmp/laz-perf && \
    cmake -S /tmp/laz-perf -B /tmp/laz-perf/build -DCMAKE_BUILD_TYPE=Release && \
    make -C /tmp/laz-perf/build -j${NUM_THREADS} install && \
    rm -rf /tmp/laz-perf

# GDAL 3.6.2
RUN git clone --depth 1 --branch v3.6.2 https://github.com/OSGeo/gdal.git /tmp/gdal && \
    cmake -S /tmp/gdal -B /tmp/gdal/build -DCMAKE_BUILD_TYPE=Release && \
    make -C /tmp/gdal/build -j${NUM_THREADS} install && \
    rm -rf /tmp/gdal

# PDAL 2.7.1
RUN git clone --depth 1 --branch 2.7.1 https://github.com/PDAL/PDAL.git /tmp/PDAL && \
    cmake -S /tmp/PDAL -B /tmp/PDAL/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_PLUGIN_PCL=ON \
    -DBUILD_PLUGIN_PYTHON=OFF \
    -DBUILD_PLUGIN_PGPOINTCLOUD=OFF \
    -DBUILD_PLUGIN_GREYHOUND=OFF \
    -DBUILD_PLUGIN_ICEBRIDGE=OFF \
    -DWITH_TESTS=OFF && \
    ninja -C /tmp/PDAL/build -j${NUM_THREADS} install && \
    rm -rf /tmp/PDAL

# PotreeConverter 2.1.1 — compilato da sorgente (unico modo per arm64)
# Il risultato viene copiato nel runtime stage tramite COPY --from=builder.
RUN git clone --depth 1 --branch 2.1.1 https://github.com/potree/PotreeConverter.git /tmp/PotreeConverter && \
    cmake -S /tmp/PotreeConverter -B /tmp/PotreeConverter/build \
        -DCMAKE_BUILD_TYPE=Release && \
    make -C /tmp/PotreeConverter/build -j${NUM_THREADS} && \
    mkdir -p /opt/potree && \
    cp /tmp/PotreeConverter/build/PotreeConverter /opt/potree/ && \
    rm -rf /tmp/PotreeConverter

# ============================================================
# STAGE 2 — runtime finale
# Base runtime (no compiler), copia solo i binari compilati.
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime-amd64
FROM ubuntu:22.04                                   AS runtime-arm64
FROM runtime-${TARGETARCH}

# ARG deve essere re-dichiarato dentro ogni stage per essere usabile nei RUN
ARG TARGETARCH=amd64

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUPY_CACHE_DIR=/tmp/.cupy \
    LD_LIBRARY_PATH=/app/open3d-devel-linux-x86_64-cxx11-abi-0.19.0/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /app

# Solo runtime di sistema — niente *-dev, niente compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates gnupg && \
    add-apt-repository universe && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    wget curl unzip xz-utils \
    libgomp1 libomp5 \
    libgl1 libgl1-mesa-glx libglu1-mesa \
    libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 \
    libstdc++6 libgcc-s1 \
    # Runtime Boost (solo .so, non gli header -dev)
    libboost-filesystem1.74.0 libboost-iostreams1.74.0 \
    libboost-program-options1.74.0 libboost-system1.74.0 \
    libboost-thread1.74.0 libboost-regex1.74.0 \
    # Runtime PCL esteso + Librerie Matematiche
    libpcl-common1.12 libpcl-io1.12 libpcl-filters1.12 libpcl-dev \
    libqhull-r8.0 libarmadillo10 \
    # Runtime VTK e Qt5 (richiesti da libpcl_visualization)
    libqt5opengl5 libqt5widgets5 libqt5gui5 libqt5core5a \
    # GDAL runtime avanzato (Formati vari, Database, GIS)
    libproj22 libcurl4 libsqlite3-0 libxml2 zlib1g \
    libgeotiff5 unixodbc libblosc1 libxerces-c3.2 libheif1 libpoppler118 \
    libgif7 libcfitsio9 libopenjp2-7 libkmlbase1 libkmldom1 libkmlengine1 \
    libfyba0 libspatialite7 libmysqlclient21 libfreexl1 \
    libgeos-c1v5 libhdf4-0-alt \
    # LASzip runtime (per PDAL)
    liblaszip8 \
    # PotreeConverter runtime
    libtbb2 \
    # Runtime GMP/MPFR/CGAL — richiesti dal binario mesh2pc
    libgmp10 libgmpxx4ldbl \
    libmpfr6 \
    libcgal-dev \
    # LLVM libc++ — richiesto dal binario mesh2pc
    libc++1 libc++abi1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python di default e upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip --no-cache-dir

# ── Copia binari compilati dallo stage builder ─────────────
COPY --from=builder /usr/local/lib     /usr/local/lib
COPY --from=builder /usr/local/bin     /usr/local/bin
COPY --from=builder /usr/local/share   /usr/local/share
COPY --from=builder /usr/local/include /usr/local/include

COPY requirements.txt /app/

# ── STEP 1: Pacchetti standard da PyPI ────────────────────
# --ignore-installed evita il conflitto con pacchetti di sistema installati
# via apt/distutils (es. blinker 1.4) che pip non può disinstallare.
RUN python3 -m pip install --no-cache-dir --ignore-installed -r /app/requirements.txt

# ── STEP 2: PyTorch (CUDA su amd64, CPU-only su arm64) ────
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        python3 -m pip install --no-cache-dir \
            torch==2.3.0+cu118 \
            torchvision==0.18.0+cu118 \
            --index-url https://download.pytorch.org/whl/cu118; \
    else \
        python3 -m pip install --no-cache-dir \
            torch==2.3.0 \
            torchvision==0.18.0; \
    fi

# ── STEP 3: CuPy e RAPIDS (solo amd64/CUDA, non disponibili su arm64) ──
# Separato da Torch per evitare conflitti di risoluzione delle dipendenze.
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        python3 -m pip install --no-cache-dir \
            cupy-cuda11x \
            cuml-cu11 \
            --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple; \
    fi

# Pulisce cache bytecode
RUN find /usr/local/lib/python3.10 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Aggiorna cache librerie dinamiche
# Rimuoviamo la liblaszip custom (manda in crash Potree) e usiamo quella di sistema
RUN rm -f /usr/local/lib/liblaszip* && \
    if [ "$TARGETARCH" = "amd64" ]; then \
        ln -s /usr/lib/x86_64-linux-gnu/liblaszip.so.8 /usr/local/lib/liblaszip.so; \
    else \
        ln -s /usr/lib/aarch64-linux-gnu/liblaszip.so.8 /usr/local/lib/liblaszip.so; \
    fi && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig

# ── Open3D precompilata (solo amd64, nessun prebuilt ufficiale per arm64) ──
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        wget -q https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz && \
        tar -xf open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz && \
        rm    open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz; \
    fi

# ── tinygltf / stb headers ─────────────────────────────────
RUN mkdir -p /app/tinygltf && \
    wget -q https://raw.githubusercontent.com/syoyo/tinygltf/master/tiny_gltf.h     -O /app/tinygltf/tiny_gltf.h && \
    wget -q https://raw.githubusercontent.com/syoyo/tinygltf/master/json.hpp        -O /app/tinygltf/json.hpp && \
    wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image.h       -O /app/tinygltf/stb_image.h && \
    wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -O /app/tinygltf/stb_image_write.h

# ── PotreeConverter — copiato dal builder (compilato da sorgente per entrambe le arch)
# Il percorso finale replica quello del prebuilt x64 per compatibilità con il codice Python.
COPY --from=builder /opt/potree/PotreeConverter /app/PotreeConverter_linux_x64/PotreeConverter
RUN chmod +x /app/PotreeConverter_linux_x64/PotreeConverter

# Aggiorna cache librerie dinamiche
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig

# ── Progetto Django e cartella opt ────────────────────────
COPY classifyViewer/ /webapp/classifyViewer/
COPY opt/            /webapp/opt/

WORKDIR /webapp/classifyViewer

RUN python3 manage.py collectstatic --noinput

EXPOSE 8000

# Run Django development server
# ENTRYPOINT [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]

# Run Django production server
ENTRYPOINT [ "gunicorn", "classifyViewer.wsgi:application", "--config", "config/gunicorn.conf.py" ]