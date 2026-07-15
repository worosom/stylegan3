#!/usr/bin/env bash

set -euo pipefail

# Full per-tile W+ animation with LCS noise. Start with a small grid/short
# transition; an 8x6 2K-model animation is an 8192x6144 render on every frame.

NETWORK="/Users/alex/Downloads/moma-v3-2k_network-snapshot-034000.pkl"
OUTPUT="out/tiled-animation-prores.mov"

GRID_COLS=3
GRID_ROWS=3
SPLIT_RESOLUTION=32
CORE_SIZE=16

BLEND_WIDTH=6
CHUNK_SIZE=16
HALO=8

# Each transition excludes its endpoint. Repeating the first seed at the end
# closes the latent path. For an exact LCS noise loop, make the total across
# all transitions 512 frames (for this three-keyframe loop: 256 per transition).
KEYFRAME_SEEDS="48,589,1001,48"
TRANSITION_FRAMES="600"    # One value for all transitions, or CSV per transition.
FPS=60
TAIL_SEED_OFFSET=1000000
TRUNCATION_PSI=1.0

CANDIDATES=3              # Keep at 1 for previews; every candidate runs every frame.
TILE_BATCH=4
MATCH_WIDTH=4
SELECTION="lock"          # lock avoids candidate switching/flicker; dynamic reselects.
PREFIX_RGB="drop"

NOISE_SEED=1
NOISE_FRAME_START=0
NOISE_LEVEL=1.0

CRF=18
PRESET="medium"           # Use slow for a final encode, ultrafast for tests.
ENCODER="prores_ks"       # ProRes MOV plays back reliably in macOS apps.
PRORES_PROFILE=1           # 4 = ProRes 4444. Use 1 for the smaller ProRes LT.
BITRATE_MBPS=80            # Used by VideoToolbox; ignored by libx264.
DETAILED_PROGRESS=0        # 1 restores per-chunk and per-noise-plane logging.
DEVICE="auto"
PYTHON_BIN="python"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if (( DETAILED_PROGRESS )); then
  PROGRESS_ARG="--detailed-progress"
else
  PROGRESS_ARG="--no-detailed-progress"
fi

"$PYTHON_BIN" tile_stylegan2_video.py \
  --network="$NETWORK" \
  --output="$OUTPUT" \
  --grid="${GRID_COLS}x${GRID_ROWS}" \
  --split-resolution="$SPLIT_RESOLUTION" \
  --core-size="$CORE_SIZE" \
  --blend-width="$BLEND_WIDTH" \
  --chunk-size="$CHUNK_SIZE" \
  --halo="$HALO" \
  --keyframe-seeds="$KEYFRAME_SEEDS" \
  --transition-frames="$TRANSITION_FRAMES" \
  --fps="$FPS" \
  --tail-seed-offset="$TAIL_SEED_OFFSET" \
  --truncation-psi="$TRUNCATION_PSI" \
  --candidates="$CANDIDATES" \
  --tile-batch="$TILE_BATCH" \
  --match-width="$MATCH_WIDTH" \
  --selection="$SELECTION" \
  --prefix-rgb="$PREFIX_RGB" \
  --noise-seed="$NOISE_SEED" \
  --noise-frame-start="$NOISE_FRAME_START" \
  --noise-level="$NOISE_LEVEL" \
  --crf="$CRF" \
  --preset="$PRESET" \
  --encoder="$ENCODER" \
  --prores-profile="$PRORES_PROFILE" \
  --bitrate-mbps="$BITRATE_MBPS" \
  --device="$DEVICE" \
  "$PROGRESS_ARG" \
  "$@"
