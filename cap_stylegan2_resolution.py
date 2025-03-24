#!/usr/bin/env python3
"""
Script to create a StyleGAN2 generator with a capped resolution.
This works with the StyleGAN2 implementation from the StyleGAN3 repository.
"""

import re
import copy
import torch
import dnnlib
import legacy
from torch_utils import misc

def create_capped_stylegan2_generator(G_original, target_resolution):
    """
    Create a new StyleGAN2 generator with a capped resolution based on the original generator.
    This works specifically with the StyleGAN2 implementation from the StyleGAN3 repository.
    
    Args:
        G_original: The original StyleGAN2 generator model
        target_resolution: The target resolution to cap at (e.g., 512)
        
    Returns:
        A new generator with the same weights as the original but capped at the target resolution
    """
    # Ensure target resolution is a power of 2
    assert (target_resolution & (target_resolution - 1) == 0) and target_resolution > 0, \
        f"Target resolution must be a power of 2, got {target_resolution}"
    
    # Ensure target resolution is not larger than the original
    assert target_resolution <= G_original.img_resolution, \
        f"Target resolution ({target_resolution}) cannot be larger than original ({G_original.img_resolution})"
    
    # Calculate the number of blocks needed for the target resolution
    # StyleGAN2 starts at 4x4 and doubles resolution with each block
    num_blocks = (target_resolution // 4).bit_length()
    
    # Create a copy of the original generator
    G_capped = copy.deepcopy(G_original)
    
    # Modify the synthesis network to cap at the target resolution
    G_capped.synthesis.img_resolution = target_resolution
    G_capped.img_resolution = target_resolution
    
    # Get the original and new synthesis blocks
    orig_blocks = G_original.synthesis.block_resolutions
    target_blocks = [res for res in orig_blocks if res <= target_resolution]
    
    # Update the block_resolutions attribute
    G_capped.synthesis.block_resolutions = target_blocks
    
    # Remove blocks that exceed the target resolution
    for res in orig_blocks:
        if res > target_resolution:
            block_name = f'b{res}'
            if hasattr(G_capped.synthesis, block_name):
                delattr(G_capped.synthesis, block_name)
    
    # Update the output layer to match the new resolution
    if hasattr(G_capped.synthesis, 'img_output'):
        delattr(G_capped.synthesis, 'img_output')
        
    # Create a new output layer for the target resolution
    output_block_name = f'b{target_resolution}'
    if hasattr(G_capped.synthesis, output_block_name):
        output_block = getattr(G_capped.synthesis, output_block_name)
        G_capped.synthesis.img_output = output_block.torgb
    
    # Update the num_ws attribute if needed
    if hasattr(G_capped.synthesis, 'num_ws'):
        # Calculate the new num_ws based on the number of remaining blocks
        G_capped.synthesis.num_ws = G_capped.mapping.num_ws = 2 * len(target_blocks)
    
    return G_capped

def main():
    """Example usage of the capped generator function."""
    import argparse
    import os
    import numpy as np
    import PIL.Image
    
    parser = argparse.ArgumentParser(description="Create a StyleGAN2 generator with capped resolution")
    parser.add_argument("--network", type=str, help="Path to the .pkl file containing the StyleGAN2 model")
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution to cap at (default: 512)")
    parser.add_argument("--output", type=str, default="capped_stylegan2.pkl", help="Output .pkl file path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image generation")
    parser.add_argument("--outdir", type=str, default="out", help="Directory to save output images")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the network
    print(f"Loading network from {args.network}...")
    with dnnlib.util.open_url(args.network) as f:
        network_data = legacy.load_network_pkl(f)
        G_original = network_data['G_ema'].to(device)
    
    print(f"Original generator resolution: {G_original.img_resolution}")
    
    # Create a capped generator
    print(f"Creating capped generator with resolution {args.resolution}...")
    G_capped = create_capped_stylegan2_generator(G_original, args.resolution)
    
    print(f"Capped generator resolution: {G_capped.img_resolution}")
    
    # Generate a sample image with both generators
    os.makedirs(args.outdir, exist_ok=True)
    
    # Create a random latent vector
    z = torch.from_numpy(np.random.RandomState(args.seed).randn(1, G_original.z_dim)).to(device)
    
    # Create a label tensor (zeros for unconditional models)
    c = torch.zeros([1, G_original.c_dim], device=device)
    
    # Generate images
    print("Generating images...")
    
    # Original generator
    img_original = G_original(z, c, truncation_psi=0.7, noise_mode='const')
    img_original = (img_original.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    
    # Capped generator
    img_capped = G_capped(z, c, truncation_psi=0.7, noise_mode='const')
    img_capped = (img_capped.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    
    # Save images
    PIL.Image.fromarray(img_original[0].cpu().numpy(), 'RGB').save(f"{args.outdir}/original_{G_original.img_resolution}.png")
    PIL.Image.fromarray(img_capped[0].cpu().numpy(), 'RGB').save(f"{args.outdir}/capped_{args.resolution}.png")
    
    print(f"Images saved to {args.outdir}")
    
    # Save the capped generator
    print(f"Saving capped generator to {args.output}...")
    capped_network_data = {k: v for k, v in network_data.items()}
    capped_network_data['G_ema'] = G_capped
    
    with open(args.output, 'wb') as f:
        torch.save(capped_network_data, f)
    
    print(f"Capped generator saved to {args.output}")

if __name__ == "__main__":
    main()