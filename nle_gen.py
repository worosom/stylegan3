# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import copy
import os
import re
from typing import List, Optional, Tuple, Union
import json

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
from easing_functions import QuadEaseInOut as Easing

import legacy

from nle_renderer.renderer import AsyncRenderer

#----------------------------------------------------------------------------
def frames_from_scene(
        renderer,
        G,
        w0_seeds: list,
        transforms: list,
        segment: dict,
        kind='cubic',
        fps=60,
        interpolation_space='w'):
    scene_duration = segment['scene_duration']
    num_frames = fps * scene_duration

    _w0_seeds = []
    _, all_zs, all_cs = zip(*[renderer._renderer_obj.get_latents(G, [[seed, 1] for seed in w0_seeds])])
    (a,), (b,) = zip(*[[renderer._renderer_obj.map_latents(G, [z], [c]) for z, c in zip(zs, cs)] for zs, cs in zip(all_zs, all_cs)])
    w_a = a[1].double()
    w_b = b[1].double()
    w_avg = a[0]

    input_transforms = []
    ws = []
    ease = Easing(start=0, end=1)
    for i in range(num_frames):
        _r = float(i)/num_frames
        r = ease(_r)
        if interpolation_space == 'linear':
            _w0_seeds.append([[w0_seeds[0], 1-r], [w0_seeds[1], r]])
        elif interpolation_space == 'w':
            ws.append(
                    (w_a * (1-r) + w_b * r + w_avg).tolist()
                    )
        rotate = transforms[0]['rotate'] + (transforms[1]['rotate'] - transforms[0]['rotate']) * r
        translate_a = transforms[0]['translate']
        translate_b = transforms[1]['translate']
        translate = dict(
                x=translate_a['x'] * (1-r) + translate_b['x'] * r,
                y=translate_a['y'] * (1-r) + translate_b['y'] * r
                )
        input_transforms.append([
                [np.cos(rotate), np.sin(rotate), translate['x']],
                [-np.sin(rotate), np.cos(rotate), translate['y']],
                [0., 0., 1.]
                ])
    return zip(ws, input_transforms)
    

def gen_interp_video(
        pkl: str,
        latents: dict,
        mp4: str,
        fps=60,
        stylemix_idx=[],
        noise_mode='static',
        force_fp32=True,
        random_seed=0,
        stylemix_seed=0,
        layer_name=None,
        trunc_psi=1,
        trunc_cutoff=0,
        kind='cubic',
        psi=1,
        device=torch.device('cuda'),
        **video_kwargs):
    renderer = AsyncRenderer()
    renderer.set_args(pkl=pkl)
    G = renderer._renderer_obj.get_network(renderer._cur_args['pkl'], 'G_ema')

    scenes = []
    total_duration = 0
    for i in range(1, len(latents)):
        a = latents[i-1]
        b = latents[i]
        scene_duration = b['interpolation']['duration']
        scenes.append(dict(
            w0_seeds=[a['w0_seeds'][0][0], b['w0_seeds'][0][0]],
            transforms=[a['transform'], b['transform']],
            segment=dict(
                start=total_duration,
                end=total_duration+scene_duration,
                scene_duration=scene_duration
                )
            ))
        total_duration += scene_duration

    frames = []
    for scene in scenes[:1]:
        frames.extend(frames_from_scene(renderer, G, fps=fps, **scene))

    # Render video.
    video_out = imageio.get_writer(
            mp4,
            mode='I',
            fps=fps,
            codec='libx264',
            output_params=['-movflags', '+faststart'],
            **video_kwargs)
    for frame_idx, [w, input_transform] in enumerate(tqdm(frames)):
        imgs = []
        renderer.set_args(pkl=pkl, w=w, input_transform=input_transform)
        result = renderer.get_result()
        try:
            img = result.image
            video_out.append_data(img)
            imgs.append(img)
        except:
            print(result.error)
            video_out.close()
            exit()
    video_out.close()


def get_network(pkl, key, **tweak_kwargs):
    print(f'Loading "{pkl}"... ', end='', flush=True)
    try:
        with dnnlib.util.open_url(pkl, verbose=False) as f:
            data = legacy.load_network_pkl(f)
        print('Done.')
    except:
        data = CapturedException()
        print('Failed!')
    if isinstance(data, CapturedException):
        raise data

    orig_net = data[key]
    cache_key = (orig_net, _device, tuple(sorted(tweak_kwargs.items())))
    try:
        net = copy.deepcopy(orig_net)
        net.to(_device)
    except:
        net = CapturedException()
    if isinstance(net, CapturedException):
        raise net
    return net

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--latents', type=str, help='JSON file containing w0_seeds and transformations', required=True, metavar='FILE')
@click.option('--fps', type=int, default=60, help='FPS of the resulting sequence')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
@click.option('--video_bitrate', default='12M', type=str)
def generate_images(
    network_pkl: str,
    latents: str,
    fps: int,
    truncation_psi: float,
    output: str,
    video_bitrate: str
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    with open(latents) as f:
        latents_obj = json.loads(f.read())

    gen_interp_video(pkl=network_pkl, latents=latents_obj, mp4=output, bitrate=video_bitrate, psi=truncation_psi, fps=fps)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
