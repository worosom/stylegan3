#!/usr/bin/env python3
"""Encode a fully animated tiled StyleGAN2 canvas using LCS-style motion."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

import dnnlib
import legacy
from tile_stylegan2 import (
    TiledStyleGAN2Renderer,
    _parse_grid,
    _select_device,
    _to_uint8_hwc,
    _to_uint16_hwc,
    set_detailed_progress,
)


def parse_int_list(value: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("the integer list cannot be empty")
    return values


def quadratic_ease_in_out(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    if value < 0.5:
        return 2.0 * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 2) / 2.0


def slerp_ws(first: torch.Tensor, second: torch.Tensor, amount: float) -> torch.Tensor:
    """Apply LCS-style eased spherical interpolation to every W+ vector."""
    if first.shape != second.shape:
        raise ValueError("latent keyframes must have matching shapes")
    amount = quadratic_ease_in_out(amount)
    first = first.to(dtype=torch.float32)
    second = second.to(dtype=torch.float32)
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(1e-12)
    second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True).clamp_min(1e-12)
    cosine = ((first / first_norm) * (second / second_norm)).sum(dim=-1, keepdim=True)
    cosine = cosine.clamp(-1.0, 1.0)
    omega = torch.acos(cosine)
    sine = torch.sin(omega)
    spherical = (
        torch.sin((1.0 - amount) * omega) / sine.clamp_min(1e-7) * first
        + torch.sin(amount * omega) / sine.clamp_min(1e-7) * second
    )
    linear = torch.lerp(first, second, amount)
    return torch.where(sine.abs() < 1e-6, linear, spherical)


def expand_transition_frames(values: Tuple[int, ...], transition_count: int) -> Tuple[int, ...]:
    if len(values) == 1:
        values = values * transition_count
    if len(values) != transition_count:
        raise ValueError(
            "transition frame count must be one value or one value per keyframe transition"
        )
    if any(value < 1 for value in values):
        raise ValueError("transition frame counts must be positive")
    return values


def map_seed_keyframes(
    renderer: TiledStyleGAN2Renderer,
    base_seeds: Sequence[int],
    values_per_keyframe: int,
    batch_size: int,
    truncation_psi: float,
    class_idx: Optional[int],
) -> torch.Tensor:
    """Map sequential seed grids and retain the keyframes on CPU."""
    keyframes = []
    for keyframe_index, base_seed in enumerate(base_seeds, start=1):
        print(
            f"Mapping latent keyframe {keyframe_index}/{len(base_seeds)} "
            f"from seed {base_seed}...",
            flush=True,
        )
        parts = []
        for start in range(0, values_per_keyframe, batch_size):
            stop = min(start + batch_size, values_per_keyframe)
            seeds = range(base_seed + start, base_seed + stop)
            parts.append(renderer.map_seeds(seeds, truncation_psi, class_idx).cpu())
        keyframes.append(torch.cat(parts, dim=0))
    return torch.stack(keyframes, dim=0)


class RawVideoWriter:
    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
        encoder: str,
        bitrate_mbps: int,
        prores_profile: int,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required; install it with `brew install ffmpeg`")
        output.parent.mkdir(parents=True, exist_ok=True)
        available_encoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if encoder == "auto":
            if " prores_ks " in available_encoders:
                encoder = "prores_ks"
            elif " libx264 " in available_encoders:
                encoder = "libx264"
            elif " libx265 " in available_encoders:
                encoder = "libx265"
            else:
                encoder = "h264_videotoolbox"
        if f" {encoder} " not in available_encoders:
            raise RuntimeError(f"FFmpeg encoder {encoder!r} is not available")
        is_prores = encoder == "prores_ks"
        if not is_prores and (width % 2 or height % 2):
            raise ValueError("H.264/HEVC yuv420p output requires even image dimensions")
        self.pixel_format = "rgb48le" if is_prores else "rgb24"
        self.dtype = np.dtype("<u2") if is_prores else np.dtype(np.uint8)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            self.pixel_format,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            encoder,
        ]
        if encoder in {"libx264", "libx265"}:
            command += ["-crf", str(crf), "-preset", preset]
            if encoder == "libx264":
                command += ["-tune", "animation"]
            else:
                command += ["-tag:v", "hvc1"]
        elif encoder == "h264_videotoolbox":
            command += [
                "-b:v",
                f"{bitrate_mbps}M",
                "-profile:v",
                "high",
                "-allow_sw",
                "1",
            ]
        elif is_prores:
            command += ["-profile:v", str(prores_profile)]
        command += [
            "-pix_fmt",
            "yuv444p10le" if is_prores else "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, pixels: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is unavailable")
        self.process.stdin.write(np.ascontiguousarray(pixels).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


def render_frame(
    renderer: TiledStyleGAN2Renderer,
    canvas,
    tail_ws: torch.Tensor,
    chunk_size: int,
    halo: int,
    noise_seed: int,
    noise_frame: int,
    noise_level: float,
    output_dtype: np.dtype,
) -> np.ndarray:
    scale = renderer.G.img_resolution // canvas.split_resolution
    height, width = canvas.shape
    output = np.empty(
        [height * scale, width * scale, renderer.G.img_channels], dtype=output_dtype
    )
    for y, x, chunk in renderer.iter_tail_chunks(
        canvas,
        tail_ws,
        chunk_size,
        halo,
        noise_seed=noise_seed,
        noise_mode="lcs",
        noise_frame=noise_frame,
        noise_level=noise_level,
    ):
        pixels = _to_uint16_hwc(chunk) if output_dtype == np.dtype("<u2") else _to_uint8_hwc(chunk)
        output[y : y + pixels.shape[0], x : x + pixels.shape[1]] = pixels
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode fully animated tiled StyleGAN2 video with LCS noise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--network", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid", type=_parse_grid, default=(2, 1), metavar="COLSxROWS")
    parser.add_argument("--split-resolution", type=int, default=32)
    parser.add_argument("--core-size", type=int, default=16)
    parser.add_argument("--blend-width", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--halo", type=int, default=8)
    parser.add_argument("--keyframe-seeds", type=parse_int_list, default=(0, 10000, 0))
    parser.add_argument("--tail-seed-offset", type=int, default=1000000)
    parser.add_argument("--transition-frames", type=parse_int_list, default=(60,))
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument("--class", dest="class_idx", type=int)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--match-width", type=int, default=4)
    parser.add_argument("--selection", choices=["lock", "dynamic"], default="lock")
    parser.add_argument("--prefix-rgb", choices=["drop", "keep"], default="drop")
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--noise-frame-start", type=int, default=0)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    parser.add_argument(
        "--encoder",
        choices=["auto", "prores_ks", "libx264", "libx265", "h264_videotoolbox"],
        default="prores_ks",
    )
    parser.add_argument(
        "--bitrate-mbps",
        type=int,
        default=80,
        help="VideoToolbox bitrate; ignored by libx264",
    )
    parser.add_argument(
        "--prores-profile",
        type=int,
        choices=range(6),
        default=4,
        help="ProRes profile: 1=LT, 3=HQ, 4=4444, 5=4444 XQ",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--detailed-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show per-chunk and large LCS-noise timing diagnostics",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if len(args.keyframe_seeds) < 2:
        raise ValueError("at least two keyframe seeds are required")
    transition_frames = expand_transition_frames(
        args.transition_frames, len(args.keyframe_seeds) - 1
    )
    cols, rows = args.grid
    tile_count = rows * cols
    device = _select_device(args.device)
    set_detailed_progress(args.detailed_progress)
    print(f"Loading {args.network!r} on {device}...", flush=True)
    with dnnlib.util.open_url(args.network) as network_file:
        G = legacy.load_network_pkl(network_file)["G_ema"]
    renderer = TiledStyleGAN2Renderer(G, device)

    scale = G.img_resolution // args.split_resolution
    output_width = cols * args.core_size * scale
    output_height = rows * args.core_size * scale
    total_frames = sum(transition_frames)
    print(
        f"Output geometry: {output_width}x{output_height}px, {total_frames} frames "
        f"at {args.fps:g} fps (grid {cols}x{rows}, core {args.core_size}px, "
        f"split {args.split_resolution}px, model {G.img_resolution}px, scale {scale}x)",
        flush=True,
    )

    values_per_keyframe = tile_count * args.candidates
    tile_keyframes = map_seed_keyframes(
        renderer,
        args.keyframe_seeds,
        values_per_keyframe,
        args.tile_batch,
        args.truncation_psi,
        args.class_idx,
    ).reshape(
        len(args.keyframe_seeds),
        tile_count,
        args.candidates,
        G.mapping.num_ws,
        G.w_dim,
    )
    tail_keyframe_seeds = tuple(
        seed + args.tail_seed_offset for seed in args.keyframe_seeds
    )
    tail_keyframes = map_seed_keyframes(
        renderer,
        tail_keyframe_seeds,
        1,
        1,
        args.truncation_psi,
        args.class_idx,
    )[:, 0]

    writer = RawVideoWriter(
        Path(args.output),
        output_width,
        output_height,
        args.fps,
        args.crf,
        args.preset,
        args.encoder,
        args.bitrate_mbps,
        args.prores_profile,
    )
    locked_indices: Optional[Tuple[int, ...]] = None
    frame_index = 0
    try:
        with tqdm(
            total=total_frames,
            desc="Encoding",
            unit="frame",
            dynamic_ncols=True,
            smoothing=0.1,
        ) as progress:
            for segment_index, segment_frames in enumerate(transition_frames):
                for segment_frame in range(segment_frames):
                    amount = segment_frame / segment_frames
                    progress.set_postfix(
                        segment=f"{segment_index + 1}/{len(transition_frames)}",
                        t=f"{amount:.3f}",
                        refresh=False,
                    )
                    candidate_ws = slerp_ws(
                        tile_keyframes[segment_index],
                        tile_keyframes[segment_index + 1],
                        amount,
                    )
                    tail_ws = slerp_ws(
                        tail_keyframes[segment_index],
                        tail_keyframes[segment_index + 1],
                        amount,
                    )[None]
                    canvas = renderer.build_feature_canvas(
                        rows=rows,
                        cols=cols,
                        split_resolution=args.split_resolution,
                        core_size=args.core_size,
                        blend_width=args.blend_width,
                        seed=args.keyframe_seeds[segment_index],
                        truncation_psi=args.truncation_psi,
                        class_idx=args.class_idx,
                        batch_size=args.tile_batch,
                        prefix_noise_mode="lcs",
                        keep_prefix_rgb=(args.prefix_rgb == "keep"),
                        candidates_per_tile=args.candidates,
                        match_width=args.match_width,
                        noise_seed=args.noise_seed,
                        noise_frame=args.noise_frame_start + frame_index,
                        noise_level=args.noise_level,
                        candidate_ws=candidate_ws,
                        selected_candidate_indices=(
                            locked_indices if args.selection == "lock" else None
                        ),
                    )
                    if args.selection == "lock" and locked_indices is None:
                        locked_indices = canvas.candidate_indices
                    pixels = render_frame(
                        renderer,
                        canvas,
                        tail_ws,
                        args.chunk_size,
                        args.halo,
                        args.noise_seed,
                        args.noise_frame_start + frame_index,
                        args.noise_level,
                        writer.dtype,
                    )
                    writer.write(pixels)
                    frame_index += 1
                    progress.update(1)
                    del canvas, candidate_ws, tail_ws, pixels
                    if device.type == "mps":
                        torch.mps.empty_cache()
        writer.close()
    except BaseException:
        writer.abort()
        raise
    print(f"Saved {args.output} ({total_frames} frames at {args.fps:g} fps)")


if __name__ == "__main__":
    main()
