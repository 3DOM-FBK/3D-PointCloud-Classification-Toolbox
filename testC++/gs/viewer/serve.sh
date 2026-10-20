#!/usr/bin/env bash
# Serves the test viewer and the gs folders on http://localhost:8080 for a browser on this machine.
# The page loads a Web Worker, so it must be served over http (not opened as a file).
# usage: serve.sh            env: DATA (directory holding the gs folders, default C:/gs_scratch/out2)
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "$0")" && pwd -W)"
DATA="${DATA:-C:/gs_scratch/out2}"
echo "open: http://localhost:8080/viewer/index.html?set=3dgs_gs&w=1280&h=720&spin=300"
docker run --rm -p 8080:8080 -v "$HERE:/srv/viewer:ro" -v "$DATA:/srv/data:ro" \
  --entrypoint python3 mcr.microsoft.com/playwright:v1.49.1-jammy -m http.server 8080 --directory /srv --bind 0.0.0.0
