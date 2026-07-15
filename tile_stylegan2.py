#!/usr/bin/env python3
"""Render large, non-repeating texture canvases with a StyleGAN2 generator.

The renderer follows the central TileGAN idea:

1. Run independent latent codes through the early synthesis blocks.
2. Crop the center ("decrust") of each intermediate activation and RGB tile.
3. Stitch the cropped tiles into one spatial feature canvas.
4. Run the remaining StyleGAN2 blocks convolutionally over that canvas.

Late synthesis styles are shared across the complete canvas.  This keeps the
material appearance coherent while the early, per-tile styles provide local
variation.  Tail noise is disabled so independently rendered chunks agree.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

# OpenSimplex enables Numba's on-disk cache inside its installed package by
# default. Keep generated machine code in this writable project cache instead.
# This must be set before any dependency has a chance to import Numba.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).parent / ".numba_cache"))

import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy
from torch_utils.ops import bias_act
from torch_utils.ops import upfirdn2d
from training import networks_stylegan2


_DETAILED_PROGRESS = True


def set_detailed_progress(enabled: bool) -> None:
    """Enable or suppress per-chunk and large-noise-plane diagnostics."""
    global _DETAILED_PROGRESS
    _DETAILED_PROGRESS = bool(enabled)


try:
    from numba import njit, prange
    from opensimplex.internals import _noise4

    @njit(cache=True, parallel=True)
    def _lcs_noise4_plane(
        row_coords: np.ndarray,
        col_coords: np.ndarray,
        z: float,
        w: float,
        perm: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one OpenSimplex plane in parallel across image rows."""
        values = np.empty((row_coords.size, col_coords.size), dtype=np.float64)
        for row_idx in prange(row_coords.size):
            for col_idx in range(col_coords.size):
                values[row_idx, col_idx] = _noise4(
                    row_coords[row_idx], col_coords[col_idx], z, w, perm
                )
        return values

except ImportError:
    _lcs_noise4_plane = None


@dataclass
class FeatureCanvas:
    """Intermediate StyleGAN2 state assembled from independently styled tiles."""

    x: torch.Tensor
    img: Optional[torch.Tensor]
    split_resolution: int
    core_size: int
    rows: int
    cols: int
    seeds: Tuple[int, ...]
    candidate_indices: Tuple[int, ...] = ()

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.x.shape[-2]), int(self.x.shape[-1])


def _parse_grid(value: str) -> Tuple[int, int]:
    try:
        cols, rows = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("grid must use COLSxROWS, for example 8x6") from exc
    if cols < 1 or rows < 1:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return cols, rows


def _select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seeded_z(seeds: Sequence[int], z_dim: int, device: torch.device) -> torch.Tensor:
    values = np.stack([np.random.RandomState(seed).randn(z_dim) for seed in seeds])
    return torch.from_numpy(values.astype(np.float32)).to(device)


def _mix_uint64(value: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer used by the coordinate-addressed noise generator."""
    value = value ^ (value >> np.uint64(30))
    value = value * np.uint64(0xBF58476D1CE4E5B9)
    value = value ^ (value >> np.uint64(27))
    value = value * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def coordinate_noise(
    height: int,
    width: int,
    origin_y: int,
    origin_x: int,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Return deterministic N(0, 1) noise for a global coordinate rectangle.

    Values depend only on ``(x, y, seed, stream)`` rather than tensor shape or
    evaluation order. Overlapping render chunks therefore receive bit-identical
    noise in their shared halo.
    """
    if height < 1 or width < 1:
        raise ValueError("noise dimensions must be positive")
    if origin_y < 0 or origin_x < 0:
        raise ValueError("noise origins must be non-negative")

    y = np.arange(origin_y, origin_y + height, dtype=np.uint64)[:, None]
    x = np.arange(origin_x, origin_x + width, dtype=np.uint64)[None, :]
    mask = 0xFFFFFFFFFFFFFFFF
    seed_offset = np.uint64(((seed & mask) * 0x9E3779B97F4A7C15) & mask)
    stream_offset = np.uint64(((stream & mask) * 0xDB4F0B9175AE2165) & mask)
    counter = (
        x * np.uint64(0xD2B74407B1CE6E93)
        + y * np.uint64(0xCA5A826395121157)
        + seed_offset
        + stream_offset
    )
    bits_a = _mix_uint64(counter + np.uint64(0x9E3779B97F4A7C15))
    bits_b = _mix_uint64(counter + np.uint64(0x3C79AC492BA7B653))
    uniform_a = ((bits_a >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))
    uniform_b = ((bits_b >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))
    values = np.sqrt(-2.0 * np.log(uniform_a)) * np.cos(2.0 * np.pi * uniform_b)
    return torch.from_numpy(values.astype(np.float32))[None, None]


@functools.lru_cache(maxsize=64)
def _get_open_simplex(seed: int):
    try:
        from opensimplex import OpenSimplex
    except ImportError as exc:
        raise RuntimeError(
            "LCS noise requires OpenSimplex; install it with `pip install opensimplex`"
        ) from exc
    return OpenSimplex(seed)


def lcs_noise(
    height: int,
    width: int,
    origin_y: int,
    origin_x: int,
    seed: int,
    frame: int,
    generator_resolution: int,
    layer_resolution: int,
) -> torch.Tensor:
    """Generate one coordinate-addressed slice using LCS's 4D OpenSimplex path.

    LCS precomputes ``[num_slices, res, res]`` volumes and indexes one slice per
    video frame. This version evaluates only the requested global rectangle, so
    it works for expanded canvases and independently rendered chunks.
    """
    if height < 1 or width < 1:
        raise ValueError("noise dimensions must be positive")
    if generator_resolution < 1 or layer_resolution < 1:
        raise ValueError("generator and layer resolutions must be positive")

    num_slices = min(
        512,
        generator_resolution * generator_resolution * 16 // (layer_resolution * layer_resolution),
    )
    num_slices = max(1, num_slices)
    slice_id = frame % num_slices
    radius = min(1.0, max(0.1, 64.0 / layer_resolution))
    phase = slice_id / num_slices * np.pi * 2
    z = np.sin(phase) * radius
    w = np.cos(phase) * radius
    simplex = _get_open_simplex(seed)

    # LCS stores output[x, y] = noise4(x, y, z, w). OpenSimplex's array API
    # returns [w, z, y, x], so transpose the final plane to preserve that exact
    # first-index/second-index convention while extending global coordinates.
    row_coords = np.arange(origin_y, origin_y + height, dtype=np.float64)
    col_coords = np.arange(origin_x, origin_x + width, dtype=np.float64)
    area = height * width
    log_progress = _DETAILED_PROGRESS and area >= 1_000_000
    if log_progress:
        backend = "Numba row-parallel" if _lcs_noise4_plane is not None else "pure Python"
        print(
            f"    LCS noise b{layer_resolution} {width}x{height} "
            f"(seed {seed}, {backend})...",
            end="",
            flush=True,
        )
    started = time.perf_counter()
    if _lcs_noise4_plane is not None:
        values = _lcs_noise4_plane(row_coords, col_coords, z, w, simplex._perm)
    elif area >= 262_144:
        raise RuntimeError(
            "Large LCS noise planes require Numba; OpenSimplex otherwise falls "
            "back to extremely slow pure-Python loops. Install it with "
            "`python -m pip install 'numba>=0.63'`."
        )
    elif hasattr(simplex, "noise4array"):
        values = simplex.noise4array(
            row_coords,
            col_coords,
            np.asarray([z], dtype=np.float64),
            np.asarray([w], dtype=np.float64),
        )[0, 0].T
    else:
        values = np.empty([height, width], dtype=np.float64)
        for row_idx, row in enumerate(row_coords):
            for col_idx, col in enumerate(col_coords):
                values[row_idx, col_idx] = simplex.noise4(row, col, z, w)
    if log_progress:
        print(f" {time.perf_counter() - started:.2f}s", flush=True)
    # LCS/renderer/misc.py defaults get_noise_volume_cuda() to bfloat16 before
    # assigning a slice back into StyleGAN2's float32 noise_const buffer.
    values_tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    return values_tensor.to(torch.bfloat16).to(torch.float32)


class TiledStyleGAN2Renderer:
    """Split and spatially extend a StyleGAN2 synthesis network."""

    def __init__(self, G: torch.nn.Module, device: torch.device | str = "cpu") -> None:
        self.G = G.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.G = self.G.to(self.device)

        synthesis = getattr(self.G, "synthesis", None)
        if synthesis is None or not hasattr(synthesis, "block_resolutions"):
            raise TypeError("network is not a StyleGAN2 generator with resolution blocks")
        self.resolutions = tuple(int(res) for res in synthesis.block_resolutions)
        if not self.resolutions or self.resolutions[0] != 4:
            raise TypeError("unsupported StyleGAN2 synthesis layout")

    def _validate_split(self, split_resolution: int) -> None:
        if split_resolution not in self.resolutions:
            choices = ", ".join(str(value) for value in self.resolutions)
            raise ValueError(f"split resolution must be one of: {choices}")

    def _labels(self, count: int, class_idx: Optional[int]) -> torch.Tensor:
        labels = torch.zeros([count, self.G.c_dim], device=self.device)
        if self.G.c_dim == 0:
            if class_idx is not None:
                raise ValueError("--class cannot be used with an unconditional network")
            return labels
        if class_idx is None:
            raise ValueError("a class index is required for this conditional network")
        if not 0 <= class_idx < self.G.c_dim:
            raise ValueError(f"class index must be in [0, {self.G.c_dim - 1}]")
        labels[:, class_idx] = 1
        return labels

    def map_seeds(
        self,
        seeds: Sequence[int],
        truncation_psi: float = 1.0,
        class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        z = _seeded_z(seeds, self.G.z_dim, self.device)
        labels = self._labels(len(seeds), class_idx)
        return self.G.mapping(z, labels, truncation_psi=truncation_psi)

    def split_ws(self, ws: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Match SynthesisNetwork.forward(), including its overlapping ToRGB styles."""
        result: Dict[int, torch.Tensor] = {}
        w_idx = 0
        for resolution in self.resolutions:
            block = getattr(self.G.synthesis, f"b{resolution}")
            count = block.num_conv + block.num_torgb
            result[resolution] = ws.narrow(1, w_idx, count)
            w_idx += block.num_conv
        return result

    def run_prefix(
        self,
        ws: torch.Tensor,
        split_resolution: int,
        noise_mode: str = "none",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run normal fixed-size blocks through ``split_resolution``."""
        self._validate_split(split_resolution)
        block_ws = self.split_ws(ws)
        x = img = None
        for resolution in self.resolutions:
            if resolution > split_resolution:
                break
            block = getattr(self.G.synthesis, f"b{resolution}")
            x, img = block(
                x,
                img,
                block_ws[resolution],
                force_fp32=True,
                fused_modconv=False,
                noise_mode=noise_mode,
            )
        if x is None:
            raise RuntimeError("prefix produced no feature tensor")
        return x, img

    @staticmethod
    def _noise_stream(resolution: int, layer_idx: int) -> int:
        """Match LCS's sequential seeds: b8.conv0=0, b8.conv1=1, ..."""
        return max(0, 2 * (int(math.log2(resolution)) - 3)) + layer_idx

    def run_lcs_prefix(
        self,
        ws: torch.Tensor,
        split_resolution: int,
        origin: Tuple[int, int],
        noise_seed: int,
        noise_frame: int,
        noise_level: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a prefix tile against one global, coordinate-addressed LCS field."""
        self._validate_split(split_resolution)
        block_ws = self.split_ws(ws)
        x = img = None
        for resolution in self.resolutions:
            if resolution > split_resolution:
                break
            block = getattr(self.G.synthesis, f"b{resolution}")
            if block.in_channels == 0:
                # LCS starts replacing noise at b8, leaving the learned-constant
                # b4 block on the checkpoint's fixed noise buffer.
                x, img = block(
                    x,
                    img,
                    block_ws[resolution],
                    force_fp32=True,
                    fused_modconv=False,
                    noise_mode="const",
                )
                continue

            scaled_y = origin[0] * resolution
            scaled_x = origin[1] * resolution
            if scaled_y % split_resolution or scaled_x % split_resolution:
                raise ValueError(
                    "LCS prefix coordinates do not align at "
                    f"resolution {resolution}; choose core/split sizes whose "
                    "decrust crop scales to integer layer coordinates"
                )
            layer_origin = (
                scaled_y // split_resolution,
                scaled_x // split_resolution,
            )
            x, img = self._run_spatial_block(
                block,
                x,
                img,
                block_ws[resolution],
                noise_seed=noise_seed,
                noise_origin=layer_origin,
                noise_mode="lcs",
                noise_frame=noise_frame,
                noise_level=noise_level,
            )
        if x is None:
            raise RuntimeError("prefix produced no feature tensor")
        return x, img

    @staticmethod
    def _run_spatial_layer(
        layer: networks_stylegan2.SynthesisLayer,
        x: torch.Tensor,
        w: torch.Tensor,
        gain: float = 1.0,
        noise_seed: Optional[int] = None,
        noise_origin: Tuple[int, int] = (0, 0),
        noise_stream: int = 0,
        noise_mode: str = "coordinate",
        noise_frame: int = 0,
        noise_level: float = 1.0,
        generator_resolution: int = 0,
        layer_resolution: int = 0,
    ) -> torch.Tensor:
        """Run a synthesis layer without its fixed-resolution shape assertion."""
        styles = layer.affine(w)
        noise = None
        if layer.use_noise and noise_seed is not None:
            noise_height = int(x.shape[-2]) * layer.up
            noise_width = int(x.shape[-1]) * layer.up
            if noise_mode == "coordinate":
                noise = coordinate_noise(
                    noise_height,
                    noise_width,
                    noise_origin[0],
                    noise_origin[1],
                    noise_seed,
                    noise_stream,
                )
            elif noise_mode == "lcs":
                noise = lcs_noise(
                    noise_height,
                    noise_width,
                    noise_origin[0],
                    noise_origin[1],
                    noise_seed + noise_stream,
                    noise_frame,
                    generator_resolution,
                    layer_resolution,
                ) * (noise_level * 5)
            else:
                raise ValueError(f"unsupported spatial noise mode: {noise_mode}")
            noise = noise.to(device=x.device, dtype=x.dtype)
            noise = noise * layer.noise_strength.to(x.dtype)
        flip_weight = layer.up == 1
        x = networks_stylegan2.modulated_conv2d(
            x=x,
            weight=layer.weight,
            styles=styles,
            noise=noise,
            up=layer.up,
            padding=layer.padding,
            resample_filter=layer.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=False,
        )
        act_gain = layer.act_gain * gain
        act_clamp = layer.conv_clamp * gain if layer.conv_clamp is not None else None
        return bias_act.bias_act(
            x,
            layer.bias.to(x.dtype),
            act=layer.activation,
            gain=act_gain,
            clamp=act_clamp,
        )

    def _run_spatial_block(
        self,
        block: networks_stylegan2.SynthesisBlock,
        x: torch.Tensor,
        img: Optional[torch.Tensor],
        ws: torch.Tensor,
        noise_seed: Optional[int],
        noise_origin: Tuple[int, int],
        noise_mode: str,
        noise_frame: int,
        noise_level: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a non-initial block on an arbitrary HxW feature field."""
        if block.in_channels == 0:
            raise ValueError("the spatial tail cannot start with the learned-constant block")

        x = x.to(torch.float32)
        img = img.to(torch.float32) if img is not None else None
        w_iter = iter(ws.unbind(dim=1))

        if block.architecture == "resnet":
            y = block.skip(x, gain=np.sqrt(0.5))
            x = self._run_spatial_layer(
                block.conv0,
                x,
                next(w_iter),
                noise_seed=noise_seed,
                noise_origin=noise_origin,
                noise_stream=self._noise_stream(block.resolution, 0),
                noise_mode=noise_mode,
                noise_frame=noise_frame,
                noise_level=noise_level,
                generator_resolution=self.G.img_resolution,
                layer_resolution=block.resolution,
            )
            x = self._run_spatial_layer(
                block.conv1,
                x,
                next(w_iter),
                gain=np.sqrt(0.5),
                noise_seed=noise_seed,
                noise_origin=noise_origin,
                noise_stream=self._noise_stream(block.resolution, 1),
                noise_mode=noise_mode,
                noise_frame=noise_frame,
                noise_level=noise_level,
                generator_resolution=self.G.img_resolution,
                layer_resolution=block.resolution,
            )
            x = y.add_(x)
        else:
            x = self._run_spatial_layer(
                block.conv0,
                x,
                next(w_iter),
                noise_seed=noise_seed,
                noise_origin=noise_origin,
                noise_stream=self._noise_stream(block.resolution, 0),
                noise_mode=noise_mode,
                noise_frame=noise_frame,
                noise_level=noise_level,
                generator_resolution=self.G.img_resolution,
                layer_resolution=block.resolution,
            )
            x = self._run_spatial_layer(
                block.conv1,
                x,
                next(w_iter),
                noise_seed=noise_seed,
                noise_origin=noise_origin,
                noise_stream=self._noise_stream(block.resolution, 1),
                noise_mode=noise_mode,
                noise_frame=noise_frame,
                noise_level=noise_level,
                generator_resolution=self.G.img_resolution,
                layer_resolution=block.resolution,
            )

        if img is not None:
            img = upfirdn2d.upsample2d(img, block.resample_filter)
        if block.is_last or block.architecture == "skip":
            y = block.torgb(x, next(w_iter), fused_modconv=False)
            y = y.to(torch.float32)
            img = img.add_(y) if img is not None else y
        return x, img

    def run_tail(
        self,
        x: torch.Tensor,
        img: Optional[torch.Tensor],
        ws: torch.Tensor,
        split_resolution: int,
        noise_seed: Optional[int] = None,
        origin: Tuple[int, int] = (0, 0),
        noise_mode: str = "coordinate",
        noise_frame: int = 0,
        noise_level: float = 1.0,
    ) -> torch.Tensor:
        """Run all post-split blocks over an arbitrary spatial feature field."""
        self._validate_split(split_resolution)
        block_ws = self.split_ws(ws)
        origin_y, origin_x = origin
        if origin_y < 0 or origin_x < 0:
            raise ValueError("tail origin must be non-negative")
        for resolution in self.resolutions:
            if resolution <= split_resolution:
                continue
            block = getattr(self.G.synthesis, f"b{resolution}")
            origin_y *= 2
            origin_x *= 2
            x, img = self._run_spatial_block(
                block,
                x,
                img,
                block_ws[resolution],
                noise_seed=noise_seed,
                noise_origin=(origin_y, origin_x),
                noise_mode=noise_mode,
                noise_frame=noise_frame,
                noise_level=noise_level,
            )
        if img is None:
            raise RuntimeError("tail produced no RGB image")
        return img

    def build_feature_canvas(
        self,
        rows: int,
        cols: int,
        split_resolution: int,
        core_size: int,
        seed: int,
        blend_width: int = 0,
        truncation_psi: float = 1.0,
        class_idx: Optional[int] = None,
        batch_size: int = 4,
        prefix_noise_mode: str = "const",
        keep_prefix_rgb: bool = False,
        candidates_per_tile: int = 1,
        match_width: int = 2,
        noise_seed: int = 1,
        noise_frame: int = 0,
        noise_level: float = 1.0,
        candidate_ws: Optional[torch.Tensor] = None,
        selected_candidate_indices: Optional[Sequence[int]] = None,
    ) -> FeatureCanvas:
        """Generate, decrust, select, and stitch independently styled prefix tiles."""
        self._validate_split(split_resolution)
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        if not 1 <= core_size <= split_resolution:
            raise ValueError("core size must be between 1 and the split resolution")
        crop_lo = (split_resolution - core_size) // 2
        crop_hi = crop_lo + core_size
        available_right = split_resolution - crop_hi
        max_blend = min(crop_lo, available_right, core_size // 2)
        if not 0 <= blend_width <= max_blend:
            raise ValueError(
                f"blend width must be between 0 and {max_blend} for this "
                "split/core combination"
            )
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if candidates_per_tile < 1:
            raise ValueError("candidates per tile must be positive")
        if not 1 <= match_width <= core_size:
            raise ValueError("match width must be between 1 and the core size")
        if prefix_noise_mode not in {"none", "const", "random", "lcs"}:
            raise ValueError("prefix noise mode must be none, const, random, or lcs")

        count = rows * cols
        if candidate_ws is not None:
            if candidate_ws.ndim != 4:
                raise ValueError(
                    "candidate_ws must have shape [tiles, candidates, num_ws, w_dim]"
                )
            expected = (count, candidates_per_tile, self.G.mapping.num_ws, self.G.w_dim)
            if tuple(candidate_ws.shape) != expected:
                raise ValueError(
                    f"candidate_ws has shape {tuple(candidate_ws.shape)}, expected {expected}"
                )
        if selected_candidate_indices is not None:
            if len(selected_candidate_indices) != count:
                raise ValueError("selected candidate index count must match the tile count")
            if any(not 0 <= int(index) < candidates_per_tile for index in selected_candidate_indices):
                raise ValueError("selected candidate indices are out of range")
        blend_lo = crop_lo - blend_width
        blend_hi = crop_hi + blend_width
        x_canvas: Optional[torch.Tensor] = None
        img_canvas: Optional[torch.Tensor] = None
        x_blend_canvas: Optional[torch.Tensor] = None
        img_blend_canvas: Optional[torch.Tensor] = None
        blend_weights: Optional[torch.Tensor] = None
        selected_seeds = []
        selected_indices = []

        with torch.inference_mode():
            for tile_idx in range(count):
                row, col = divmod(tile_idx, cols)
                prefix_origin = (
                    row * core_size - crop_lo,
                    col * core_size - crop_lo,
                )
                candidate_seeds = tuple(
                    seed + tile_idx * candidates_per_tile + candidate_idx
                    for candidate_idx in range(candidates_per_tile)
                )
                x_parts = []
                x_blend_parts = []
                img_parts = []
                img_blend_parts = []
                for start in range(0, candidates_per_tile, batch_size):
                    batch_seeds = candidate_seeds[start : start + batch_size]
                    if candidate_ws is None:
                        ws = self.map_seeds(batch_seeds, truncation_psi, class_idx)
                    else:
                        ws = candidate_ws[
                            tile_idx, start : start + len(batch_seeds)
                        ].to(self.device)
                    if prefix_noise_mode == "lcs":
                        x, img = self.run_lcs_prefix(
                            ws,
                            split_resolution,
                            origin=prefix_origin,
                            noise_seed=noise_seed,
                            noise_frame=noise_frame,
                            noise_level=noise_level,
                        )
                    else:
                        x, img = self.run_prefix(ws, split_resolution, prefix_noise_mode)
                    x_parts.append(x[:, :, crop_lo:crop_hi, crop_lo:crop_hi].to("cpu"))
                    if blend_width:
                        x_blend_parts.append(
                            x[:, :, blend_lo:blend_hi, blend_lo:blend_hi].to("cpu")
                        )
                    if keep_prefix_rgb and img is not None:
                        img_parts.append(img[:, :, crop_lo:crop_hi, crop_lo:crop_hi].to("cpu"))
                        if blend_width:
                            img_blend_parts.append(
                                img[:, :, blend_lo:blend_hi, blend_lo:blend_hi].to("cpu")
                            )

                x_candidates = torch.cat(x_parts, dim=0)
                x_blend_candidates = (
                    torch.cat(x_blend_parts, dim=0) if x_blend_parts else None
                )
                img_candidates = torch.cat(img_parts, dim=0) if img_parts else None
                img_blend_candidates = (
                    torch.cat(img_blend_parts, dim=0) if img_blend_parts else None
                )
                if x_canvas is None:
                    x_canvas = torch.empty(
                        [1, x_candidates.shape[1], rows * core_size, cols * core_size],
                        dtype=x_candidates.dtype,
                        device="cpu",
                    )
                    if blend_width:
                        x_blend_canvas = torch.zeros_like(x_canvas)
                        blend_weights = torch.zeros(
                            [1, 1, rows * core_size, cols * core_size],
                            dtype=x_candidates.dtype,
                            device="cpu",
                        )
                if img_candidates is not None and img_canvas is None:
                    img_canvas = torch.empty(
                        [1, img_candidates.shape[1], rows * core_size, cols * core_size],
                        dtype=img_candidates.dtype,
                        device="cpu",
                    )
                    if blend_width:
                        img_blend_canvas = torch.zeros_like(img_canvas)

                costs = torch.zeros([candidates_per_tile], dtype=torch.float64)
                if col > 0:
                    left_edge = x_canvas[
                        0,
                        :,
                        row * core_size : (row + 1) * core_size,
                        col * core_size - match_width : col * core_size,
                    ]
                    costs += (
                        x_candidates[:, :, :, :match_width] - left_edge.unsqueeze(0)
                    ).to(torch.float64).square().mean(dim=(1, 2, 3))
                if row > 0:
                    top_edge = x_canvas[
                        0,
                        :,
                        row * core_size - match_width : row * core_size,
                        col * core_size : (col + 1) * core_size,
                    ]
                    costs += (
                        x_candidates[:, :, :match_width, :] - top_edge.unsqueeze(0)
                    ).to(torch.float64).square().mean(dim=(1, 2, 3))

                selected_idx = (
                    int(selected_candidate_indices[tile_idx])
                    if selected_candidate_indices is not None
                    else int(costs.argmin())
                )
                selected_seeds.append(candidate_seeds[selected_idx])
                selected_indices.append(selected_idx)
                ys = slice(row * core_size, (row + 1) * core_size)
                xs = slice(col * core_size, (col + 1) * core_size)
                x_canvas[0, :, ys, xs] = x_candidates[selected_idx]
                if img_canvas is not None and img_candidates is not None:
                    img_canvas[0, :, ys, xs] = img_candidates[selected_idx]

                if blend_width:
                    if x_blend_candidates is None or x_blend_canvas is None or blend_weights is None:
                        raise RuntimeError("feature blend buffers were not initialized")
                    blend_extent = core_size + 2 * blend_width
                    axis = torch.ones([blend_extent], dtype=x_candidates.dtype)
                    ramp = torch.arange(
                        2 * blend_width, dtype=x_candidates.dtype
                    ) / (2 * blend_width)
                    weight_y = axis.clone()
                    weight_x = axis.clone()
                    if row > 0:
                        weight_y[: 2 * blend_width] = ramp
                    if row < rows - 1:
                        weight_y[core_size:] = 1 - ramp
                    if col > 0:
                        weight_x[: 2 * blend_width] = ramp
                    if col < cols - 1:
                        weight_x[core_size:] = 1 - ramp
                    tile_weights = weight_y[:, None] * weight_x[None, :]

                    global_y0 = row * core_size - blend_width
                    global_x0 = col * core_size - blend_width
                    global_y1 = global_y0 + blend_extent
                    global_x1 = global_x0 + blend_extent
                    dest_y0 = max(0, global_y0)
                    dest_x0 = max(0, global_x0)
                    dest_y1 = min(rows * core_size, global_y1)
                    dest_x1 = min(cols * core_size, global_x1)
                    src_y0 = dest_y0 - global_y0
                    src_x0 = dest_x0 - global_x0
                    src_y1 = src_y0 + dest_y1 - dest_y0
                    src_x1 = src_x0 + dest_x1 - dest_x0
                    dest_ys = slice(dest_y0, dest_y1)
                    dest_xs = slice(dest_x0, dest_x1)
                    src_ys = slice(src_y0, src_y1)
                    src_xs = slice(src_x0, src_x1)
                    weights = tile_weights[src_ys, src_xs]

                    x_blend_canvas[:, :, dest_ys, dest_xs] += (
                        x_blend_candidates[selected_idx, :, src_ys, src_xs][None]
                        * weights[None, None]
                    )
                    blend_weights[:, :, dest_ys, dest_xs] += weights[None, None]
                    if img_blend_canvas is not None and img_blend_candidates is not None:
                        img_blend_canvas[:, :, dest_ys, dest_xs] += (
                            img_blend_candidates[selected_idx, :, src_ys, src_xs][None]
                            * weights[None, None]
                        )

        if x_canvas is None:
            raise RuntimeError("feature canvas construction failed")
        if blend_width:
            if x_blend_canvas is None or blend_weights is None:
                raise RuntimeError("feature blending failed")
            if bool((blend_weights <= 0).any()):
                raise RuntimeError("feature blending left uncovered canvas pixels")
            x_canvas = x_blend_canvas / blend_weights
            if img_blend_canvas is not None:
                img_canvas = img_blend_canvas / blend_weights
        return FeatureCanvas(
            x=x_canvas,
            img=img_canvas,
            split_resolution=split_resolution,
            core_size=core_size,
            rows=rows,
            cols=cols,
            seeds=tuple(selected_seeds),
            candidate_indices=tuple(selected_indices),
        )

    def iter_tail_chunks(
        self,
        canvas: FeatureCanvas,
        tail_ws: torch.Tensor,
        chunk_size: int,
        halo: int,
        noise_seed: Optional[int] = None,
        noise_mode: str = "coordinate",
        noise_frame: int = 0,
        noise_level: float = 1.0,
    ) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """Yield ``(output_y, output_x, RGB_chunk)`` without chunk-edge seams."""
        if chunk_size < 1:
            raise ValueError("chunk size must be positive")
        if halo < 0:
            raise ValueError("halo must be non-negative")
        scale = self.G.img_resolution // canvas.split_resolution
        height, width = canvas.shape
        tail_ws = tail_ws.to(self.device)
        chunks_y = (height + chunk_size - 1) // chunk_size
        chunks_x = (width + chunk_size - 1) // chunk_size
        total_chunks = chunks_y * chunks_x
        chunk_index = 0

        with torch.inference_mode():
            for y0 in range(0, height, chunk_size):
                y1 = min(y0 + chunk_size, height)
                ey0 = max(0, y0 - halo)
                ey1 = min(height, y1 + halo)
                for x0 in range(0, width, chunk_size):
                    chunk_index += 1
                    x1 = min(x0 + chunk_size, width)
                    ex0 = max(0, x0 - halo)
                    ex1 = min(width, x1 + halo)

                    work_width = (ex1 - ex0) * scale
                    work_height = (ey1 - ey0) * scale
                    core_width = (x1 - x0) * scale
                    core_height = (y1 - y0) * scale
                    if _DETAILED_PROGRESS:
                        print(
                            f"  Tail chunk {chunk_index}/{total_chunks}: "
                            f"{core_width}x{core_height} core, "
                            f"{work_width}x{work_height} with halo",
                            flush=True,
                        )

                    x = canvas.x[:, :, ey0:ey1, ex0:ex1].to(self.device)
                    img = None
                    if canvas.img is not None:
                        img = canvas.img[:, :, ey0:ey1, ex0:ex1].to(self.device)
                    output = self.run_tail(
                        x,
                        img,
                        tail_ws,
                        canvas.split_resolution,
                        noise_seed=noise_seed,
                        origin=(ey0, ex0),
                        noise_mode=noise_mode,
                        noise_frame=noise_frame,
                        noise_level=noise_level,
                    )

                    crop_y0 = (y0 - ey0) * scale
                    crop_y1 = crop_y0 + (y1 - y0) * scale
                    crop_x0 = (x0 - ex0) * scale
                    crop_x1 = crop_x0 + (x1 - x0) * scale
                    core = output[:, :, crop_y0:crop_y1, crop_x0:crop_x1].to("cpu")
                    yield y0 * scale, x0 * scale, core

    def render_tail_chunked(
        self,
        canvas: FeatureCanvas,
        tail_ws: torch.Tensor,
        chunk_size: int,
        halo: int,
        noise_seed: Optional[int] = None,
        noise_mode: str = "coordinate",
        noise_frame: int = 0,
        noise_level: float = 1.0,
    ) -> torch.Tensor:
        """Assemble all tail chunks into one CPU float32 tensor."""
        scale = self.G.img_resolution // canvas.split_resolution
        height, width = canvas.shape
        output = torch.empty(
            [1, self.G.img_channels, height * scale, width * scale],
            dtype=torch.float32,
            device="cpu",
        )
        for y, x, chunk in self.iter_tail_chunks(
            canvas,
            tail_ws,
            chunk_size,
            halo,
            noise_seed=noise_seed,
            noise_mode=noise_mode,
            noise_frame=noise_frame,
            noise_level=noise_level,
        ):
            output[:, :, y : y + chunk.shape[-2], x : x + chunk.shape[-1]] = chunk
        return output


def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    image = (image[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255)
    pixels = image.to(torch.uint8).cpu().numpy()
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    return pixels


def _to_uint16_hwc(image: torch.Tensor) -> np.ndarray:
    """Convert a StyleGAN image in [-1, 1] to full-range 16-bit RGB."""
    image = (image[0].permute(1, 2, 0) * 32767.5 + 32767.5).clamp(0, 65535)
    pixels = image.to(torch.uint16).cpu().numpy()
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    return pixels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a large non-repeating canvas from a StyleGAN2 network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--network", required=True, help="StyleGAN2 network pickle or URL")
    parser.add_argument("--output", required=True, help="Output PNG/JPEG path")
    parser.add_argument("--grid", type=_parse_grid, default=(8, 8), metavar="COLSxROWS")
    parser.add_argument("--split-resolution", type=int, default=32)
    parser.add_argument("--core-size", type=int, default=16, help="Center retained from each prefix tile")
    parser.add_argument(
        "--blend-width",
        type=int,
        default=0,
        help="Feature-space feather radius on each side of a latent-tile edge",
    )
    parser.add_argument("--chunk-size", type=int, default=64, help="Tail chunk size at split resolution")
    parser.add_argument("--halo", type=int, default=16, help="Tail overlap at split resolution")
    parser.add_argument("--seed", type=int, default=0, help="First per-tile seed")
    parser.add_argument("--tail-seed", type=int, help="Shared late-style seed; defaults to seed + 1000000")
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument("--class", dest="class_idx", type=int)
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--prefix-noise", choices=["lcs", "none", "const", "random"], default="lcs")
    parser.add_argument("--prefix-rgb", choices=["drop", "keep"], default="drop")
    parser.add_argument("--candidates", type=int, default=8, help="Latent candidates tested per grid cell")
    parser.add_argument("--match-width", type=int, default=2, help="Feature-edge width used to rank candidates")
    parser.add_argument("--tail-noise", choices=["lcs", "coordinate", "none"], default="lcs")
    parser.add_argument("--noise-seed", type=int, default=1, help="Base seed for injected noise")
    parser.add_argument("--noise-frame", type=int, default=0, help="LCS animation frame/slice index")
    parser.add_argument("--noise-level", type=float, default=1.0, help="LCS noise multiplier before the built-in 5x gain")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    cols, rows = args.grid
    device = _select_device(args.device)
    print(f"Loading {args.network!r} on {device}...")
    with dnnlib.util.open_url(args.network) as network_file:
        G = legacy.load_network_pkl(network_file)["G_ema"]
    renderer = TiledStyleGAN2Renderer(G, device)

    scale = G.img_resolution // args.split_resolution
    output_width = cols * args.core_size * scale
    output_height = rows * args.core_size * scale
    print(
        f"Output geometry: {output_width}x{output_height}px "
        f"(grid {cols}x{rows}, core {args.core_size}px, "
        f"split {args.split_resolution}px, model {G.img_resolution}px, scale {scale}x)",
        flush=True,
    )

    tail_seed = args.tail_seed if args.tail_seed is not None else args.seed + 1_000_000
    noise_seed = args.noise_seed
    print(f"Building {cols}x{rows} feature canvas at {args.split_resolution}px...")
    canvas = renderer.build_feature_canvas(
        rows=rows,
        cols=cols,
        split_resolution=args.split_resolution,
        core_size=args.core_size,
        blend_width=args.blend_width,
        seed=args.seed,
        truncation_psi=args.truncation_psi,
        class_idx=args.class_idx,
        batch_size=args.tile_batch,
        prefix_noise_mode=args.prefix_noise,
        keep_prefix_rgb=(args.prefix_rgb == "keep"),
        candidates_per_tile=args.candidates,
        match_width=args.match_width,
        noise_seed=noise_seed,
        noise_frame=args.noise_frame,
        noise_level=args.noise_level,
    )
    tail_ws = renderer.map_seeds([tail_seed], args.truncation_psi, args.class_idx)

    out_height, out_width = (value * scale for value in canvas.shape)
    print(f"Rendering {out_width}x{out_height} output in chunks...")
    output_hwc = np.empty([out_height, out_width, G.img_channels], dtype=np.uint8)
    for y, x, chunk in renderer.iter_tail_chunks(
        canvas,
        tail_ws,
        args.chunk_size,
        args.halo,
        noise_seed=(noise_seed if args.tail_noise != "none" else None),
        noise_mode=args.tail_noise,
        noise_frame=args.noise_frame,
        noise_level=args.noise_level,
    ):
        pixels = _to_uint8_hwc(chunk)
        output_hwc[y : y + pixels.shape[0], x : x + pixels.shape[1]] = pixels

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGB" if G.img_channels == 3 else ("L" if G.img_channels == 1 else None)
    PIL.Image.fromarray(output_hwc, mode=mode).save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
