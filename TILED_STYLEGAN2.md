# Large non-repeating StyleGAN2 canvases

`tile_stylegan2.py` adapts the core TileGAN technique to the StyleGAN2
implementation included in this repository. It runs independent latent codes
through the early synthesis blocks, crops their intermediate feature maps,
stitches those maps, and runs the remaining synthesis blocks over the larger
canvas.

This is different from making one seamless repeating tile. Neighboring regions
receive different early latent codes, so the output can continue without a
short repetition period.

## Example

Run from the `stylegan3` directory:

```bash
python tile_stylegan2.py \
  --network=/path/to/stylegan2-texture.pkl \
  --output=out/large-texture.png \
  --grid=8x6 \
  --split-resolution=16 \
  --core-size=8 \
  --chunk-size=64 \
  --halo=16 \
  --prefix-noise=lcs \
  --prefix-rgb=drop \
  --candidates=8 \
  --match-width=2 \
  --tail-noise=lcs \
  --noise-seed=1 \
  --noise-frame=0 \
  --noise-level=1 \
  --device=auto
```

For a 1024px network, these settings produce a 4096x3072 image: each retained
8px feature tile is enlarged by the post-split scale factor of 64.

The renderer accepts original StyleGAN2 TensorFlow pickles that `legacy.py` can
convert, as well as StyleGAN2 PyTorch pickles compatible with this repository.
It rejects StyleGAN3 generators.

## Important controls

- `--split-resolution` chooses where independent tiles are merged. Earlier
  merges provide smoother large structures; later merges preserve more
  independent detail but make joins harder to hide.
- `--core-size` retains only the center of each prefix feature map. Values near
  half the split resolution are a useful starting point.
- `--blend-width` retains an additional feature margin around each selected
  prefix tile and crossfades overlapping activations before the shared tail.
  It is a true feathered blend, measured at the split resolution, rather than
  just a candidate-selection cost. It cannot exceed either the available
  decrust margin or half the core size.
- `--prefix-rgb=drop` prevents independently generated low-resolution RGB skip
  images from bypassing the shared synthesis tail. Keeping them usually exposes
  the latent grid as a visible montage with `architecture=skip` checkpoints.
- `--candidates` generates several latent candidates for each grid cell and
  greedily chooses the candidate whose feature edges best match its selected
  left and upper neighbors. This is a lightweight version of TileGAN's
  neighbor optimization. Higher values improve the search but increase prefix
  generation time proportionally.
- `--match-width` controls how many intermediate feature samples participate in
  the left/top compatibility cost.
- `--tail-seed` supplies one shared set of late styles for the whole canvas.
  This keeps color and material appearance coherent across tile boundaries.
- `--halo` is overlap used only while rendering chunks. Increase it if chunk
  boundaries are visible. It does not blend the independently generated latent
  tiles.
- `--prefix-noise=lcs` samples the 4D OpenSimplex slices used by Latent Cinema
  Suite at global feature coordinates. Each prefix tile receives the rectangle
  belonging to its grid position, including negative coordinates for the
  top/left decrust halo. Adjacent tiles therefore use coherent windows from one
  continuous field instead of restarting the same fixed noise buffer. `const`,
  `random`, and `none` remain available for comparison.
- `--tail-noise=lcs` continues the coordinate-addressed LCS approach through
  the shared tail. Unlike LCS's CUDA-only precomputed square volumes, the
  renderer evaluates only each requested rectangle and then moves it to CPU,
  MPS, or CUDA. Overlapping chunks receive identical values. As in LCS, every
  noise-enabled synthesis layer has its own seed/field; the field is shared
  spatially across all tiles and chunks within that layer. The older white
  Gaussian provider remains available as `--tail-noise=coordinate`.
- `--noise-frame` selects the animated LCS slice. Consecutive values move around
  the same circular 4D path used by LCS. `--noise-seed` changes the OpenSimplex
  fields, and `--noise-level` corresponds to LCS's per-layer level before its
  built-in 5x multiplier.

## Limitations

Pretrained StyleGAN2 networks were not trained on stitched intermediate feature
maps. A texture-focused model may work directly, but fine-tuning with stitched
canvases is expected to improve transitions substantially. The current renderer
uses a shared late style rather than spatially varying modulation. It keeps the
entire intermediate feature canvas and final encoded image in memory, while the
expensive high-resolution tail is evaluated in bounded chunks.

PNG and JPEG still require a finite output allocation. For hundred-megapixel
outputs, connect `iter_tail_chunks()` to a tiled TIFF, Zarr, or Deep Zoom writer
instead of assembling a single image array.

## Full animation

`tile_stylegan2_video.py` follows the offline `LCS/encoder` design while using
the tiled feature pipeline and rectangular FFmpeg input. For every frame it:

1. spherically interpolates the complete W+ tensor for every tile candidate;
2. rebuilds and feather-blends the prefix feature canvas;
3. interpolates the shared tail W+ tensor;
4. advances the globally addressed LCS noise field; and
5. renders the tail in bounded chunks and writes RGB frames to FFmpeg.

Use `tile_stylegan2_video.sh` as the editable preset. `KEYFRAME_SEEDS` defines
the latent path and `TRANSITION_FRAMES` controls each segment. Repeating the
first keyframe seed at the end closes the latent loop. LCS layer noise volumes
all repeat over a common 512-frame period, so an exact latent-and-noise loop
must contain 512 frames (or a multiple of 512).

Candidate selection defaults to `lock`: the first frame chooses the best
candidate sequence for every grid cell and later frames retain those indices.
This avoids discrete candidate switches during interpolation. `dynamic`
selection is available for experiments but may visibly flicker.

Full animation is substantially more expensive than a static prefix because
every prefix tile and every tail chunk is synthesized on every frame. Begin
with the preset's 2x1 grid, one candidate, and a short transition. The encoder
defaults to a 16-bit RGB input pipe and `prores_ks` profile 4 (ProRes 4444) in
a `.mov` container, which plays reliably in macOS applications. Set
`ENCODER=libx265` in the preset for a much smaller HEVC `.mp4`, or set
`PRORES_PROFILE=1` for ProRes LT. Apple VideoToolbox can be requested
explicitly with `--encoder=h264_videotoolbox`.

Video progress is displayed as a single `tqdm` frame bar with rate, ETA,
segment, and interpolation position. Set `DETAILED_PROGRESS=1` in the video
preset, or pass `--detailed-progress`, to restore per-chunk and large-noise
timing messages for diagnosis.
