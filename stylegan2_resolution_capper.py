#!/usr/bin/env python3
"""
A more detailed implementation for capping the resolution of a StyleGAN2 generator
from the StyleGAN3 repository.
"""

import os
import re
import copy
import torch
import numpy as np
import PIL.Image
import dnnlib
import legacy
from torch_utils import misc

def inspect_generator(G):
    """Print detailed information about a generator to help with debugging."""
    print(f"Generator type: {type(G).__name__}")
    print(f"Generator resolution: {G.img_resolution}")
    print(f"Generator z_dim: {G.z_dim}")
    print(f"Generator c_dim: {G.c_dim}")
    
    if hasattr(G, 'synthesis'):
        print("\nSynthesis network attributes:")
        for attr_name in dir(G.synthesis):
            if not attr_name.startswith('_') and not callable(getattr(G.synthesis, attr_name)):
                attr_value = getattr(G.synthesis, attr_name)
                print(f"  {attr_name}: {attr_value}")
        
        print("\nSynthesis network blocks:")
        for block_name in dir(G.synthesis):
            if block_name.startswith('b') and block_name[1:].isdigit():
                block = getattr(G.synthesis, block_name)
                print(f"  {block_name}: {type(block).__name__}")
                
                # Print block attributes
                for attr_name in dir(block):
                    if not attr_name.startswith('_') and not callable(getattr(block, attr_name)):
                        try:
                            attr_value = getattr(block, attr_name)
                            if not isinstance(attr_value, torch.nn.Module):
                                print(f"    {attr_name}: {attr_value}")
                        except:
                            pass

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
    
    # Create a new generator with the same architecture but different resolution
    # First, get the original kwargs
    synthesis_kwargs = copy.deepcopy(G_original.synthesis.init_kwargs)
    mapping_kwargs = copy.deepcopy(G_original.mapping.init_kwargs)
    
    # Update the resolution in the synthesis kwargs
    synthesis_kwargs['img_resolution'] = target_resolution
    
    # Get the original block resolutions
    orig_block_resolutions = G_original.synthesis.block_resolutions
    
    # Filter block resolutions to only include those up to the target resolution
    new_block_resolutions = [res for res in orig_block_resolutions if res <= target_resolution]
    synthesis_kwargs['block_resolutions'] = new_block_resolutions
    
    # Adjust the number of layers in the mapping network if needed
    if 'num_layers' in mapping_kwargs:
        # Keep the mapping network the same size
        pass
    
    # Create a new generator with the updated kwargs
    from training.networks_stylegan2 import Generator
    G_capped = Generator(
        z_dim=G_original.z_dim,
        c_dim=G_original.c_dim,
        w_dim=G_original.w_dim,
        img_resolution=target_resolution,
        img_channels=G_original.img_channels,
        mapping_kwargs=mapping_kwargs,
        synthesis_kwargs=synthesis_kwargs
    ).eval().requires_grad_(False)
    
    # Copy the mapping network weights (these are independent of resolution)
    misc.copy_params_and_buffers(G_original.mapping, G_capped.mapping, require_all=True)
    
    # Copy the synthesis network weights up to the target resolution
    # For each block in the capped generator, copy the corresponding weights from the original
    for res in new_block_resolutions:
        block_name = f'b{res}'
        if hasattr(G_original.synthesis, block_name) and hasattr(G_capped.synthesis, block_name):
            orig_block = getattr(G_original.synthesis, block_name)
            capped_block = getattr(G_capped.synthesis, block_name)
            
            # Copy all parameters and buffers for this block
            misc.copy_params_and_buffers(orig_block, capped_block, require_all=True)
    
    # Copy the ToRGB layer for the highest resolution
    highest_res_block = f'b{new_block_resolutions[-1]}'
    if hasattr(G_original.synthesis, highest_res_block) and hasattr(G_capped.synthesis, highest_res_block):
        orig_block = getattr(G_original.synthesis, highest_res_block)
        capped_block = getattr(G_capped.synthesis, highest_res_block)
        
        if hasattr(orig_block, 'torgb') and hasattr(capped_block, 'torgb'):
            misc.copy_params_and_buffers(orig_block.torgb, capped_block.torgb, require_all=True)
    
    return G_capped

def main():
    """Example usage of the capped generator function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a StyleGAN2 generator with capped resolution")
    parser.add_argument("--network", type=str, required=True, help="Path to the .pkl file containing the StyleGAN2 model")
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution to cap at (default: 512)")
    parser.add_argument("--output", type=str, default="capped_stylegan2.pkl", help="Output .pkl file path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image generation")
    parser.add_argument("--outdir", type=str, default="out", help="Directory to save output images")
    parser.add_argument("--inspect", action="store_true", help="Inspect the generator structure")
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
    
    # Inspect the generator if requested
    if args.inspect:
        print("\nInspecting original generator:")
        inspect_generator(G_original)
    
    # Create a capped generator
    print(f"\nCreating capped generator with resolution {args.resolution}...")
    G_capped = create_capped_stylegan2_generator(G_original, args.resolution)
    
    print(f"Capped generator resolution: {G_capped.img_resolution}")
    
    # Inspect the capped generator if requested
    if args.inspect:
        print("\nInspecting capped generator:")
        inspect_generator(G_capped)
    
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