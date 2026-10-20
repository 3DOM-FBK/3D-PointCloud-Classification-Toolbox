#!/usr/bin/env bash
# Renders a gs v2 folder with the test renderer in headless Chromium (software WebGL2).
# usage: run.sh <folder under DATA> [png under SHOTS]     env: STRIDE, W, H, YAW, PITCH, DIST, NOSORT, NODRAW
# DATA  = directory holding the gs folders   (default: C:/gs_scratch/out2)
# SHOTS = directory for the screenshots      (default: C:/gs_scratch/shots)
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "$0")" && pwd -W)"
DATA="${DATA:-C:/gs_scratch/out2}"
SHOTS="${SHOTS:-C:/gs_scratch/shots}"
mkdir -p "$SHOTS"
SHOT_ARG=""
if [ -n "$2" ]; then SHOT_ARG="/shots/$2"; fi
docker run --rm -e PAGE_URL=http://127.0.0.1:8080/viewer/index.html -e SET="$1" -e SHOT="$SHOT_ARG" \
  -e W="${W:-900}" -e H="${H:-600}" -e STRIDE -e YAW -e PITCH -e DIST -e NOSORT -e NODRAW -e BENCH -e SPIN -e TIMEOUT_MS=900000 \
  --entrypoint bash -v "$HERE:/srv/viewer:ro" -v "$DATA:/srv/data:ro" -v "$SHOTS:/shots" -v pwmods:/mods \
  mcr.microsoft.com/playwright:v1.49.1-jammy -c "cd /mods && (ls node_modules/playwright >/dev/null 2>&1 || (npm init -y >/dev/null && npm i playwright@1.49.1 2>&1 | tail -1)); (python3 -m http.server 8080 --directory /srv >/tmp/http.log 2>&1 &); sleep 1; NODE_PATH=/mods/node_modules node /srv/viewer/shot.js 2>&1 | tail -30"
