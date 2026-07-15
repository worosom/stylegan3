#!/usr/bin/env bash

set -euo pipefail

# -----------------------------------------------------------------------------
# Experiment settings — edit these.

# NETWORK="/Users/alex/Downloads/network-snapshot-005443.pkl"
NETWORK="/Users/alex/Downloads/moma-v3-2k_network-snapshot-034000.pkl"
OUTPUT="out/large-texture.png"

GRID_COLS=8
GRID_ROWS=6
SPLIT_RESOLUTION=32
CORE_SIZE=16
BLEND_WIDTH=8             # Feature pixels feathered on each side of tile edges.
# These are measured at SPLIT_RESOLUTION, then scaled up by the network.
# For a 2K model split at 32px, 16px becomes 1024 output pixels. Keeping
# chunks modest avoids asking MPS to render a multi-gigabyte 5K work area.
CHUNK_SIZE=16
HALO=8

# Early per-tile latent search.
SEED=57
TRUNCATION_PSI=1.0
TILE_BATCH=4
CANDIDATES=8
MATCH_WIDTH=4
PREFIX_RGB="drop"          # drop or keep

# One shared latent style for all post-split layers.
TAIL_SEED=$SEED

# Noise. LCS uses one continuous coordinate field per synthesis layer.
PREFIX_NOISE="lcs"         # lcs, none, const, or random
TAIL_NOISE="lcs"           # lcs, coordinate, or none
NOISE_SEED=1
NOISE_FRAME=0
NOISE_LEVEL=1.0

DEVICE="auto"              # auto, cpu, mps, cuda, or cuda:N
PYTHON_BIN="python"

# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

"$PYTHON_BIN" tile_stylegan2.py \
  --network="$NETWORK" \
  --output="$OUTPUT" \
  --grid="${GRID_COLS}x${GRID_ROWS}" \
  --split-resolution="$SPLIT_RESOLUTION" \
  --core-size="$CORE_SIZE" \
  --blend-width="$BLEND_WIDTH" \
  --chunk-size="$CHUNK_SIZE" \
  --halo="$HALO" \
  --seed="$SEED" \
  --tail-seed="$TAIL_SEED" \
  --truncation-psi="$TRUNCATION_PSI" \
  --tile-batch="$TILE_BATCH" \
  --prefix-noise="$PREFIX_NOISE" \
  --prefix-rgb="$PREFIX_RGB" \
  --candidates="$CANDIDATES" \
  --match-width="$MATCH_WIDTH" \
  --tail-noise="$TAIL_NOISE" \
  --noise-seed="$NOISE_SEED" \
  --noise-frame="$NOISE_FRAME" \
  --noise-level="$NOISE_LEVEL" \
  --device="$DEVICE" \
  "$@"
