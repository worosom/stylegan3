#!/usr/bin/env python3
"""
Example script demonstrating how to load a pre-trained StyleGAN2 model from the StyleGAN3 repository,
create a generator with a capped resolution, and generate images with it.
"""

import os
import torch
import numpy as np
import PIL.Image
import dnnlib
import legacy
from cap_stylegan2_resolution import create_capped_stylegan2_generator

def main():
    # Configuration
    network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl"  # Pre-trained model URL
    target_resolution = 512  # Target resolution to cap at
    output_dir = "stylegan2_capped_output"  # Directory to save output images
    num_samples = 3  # Number of sample images to generate
    seeds = [100, 200, 300]  # Random seeds for reproducibility
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the pre-trained model
    print(f"Loading network from {network_pkl}...")
    with dnnlib.util.open_url(network_pkl) as f:
        network_data = legacy.load_network_pkl(f)
        G_original = network_data['G_ema'].to(device)
    
    print(f"Original generator resolution: {G_original.img_resolution}")
    print(f"Original generator type: {type(G_original).__name__}")
    
    # Print the synthesis network structure to understand the block naming
    print("\nOriginal synthesis network structure:")
    for name, _ in G_original.synthesis.named_parameters():
        if 'weight' in name and 'torgb' not in name:
            print(f"  {name}")
    
    # Create a capped generator
    print(f"\nCreating capped generator with resolution {target_resolution}...")
    G_capped = create_capped_stylegan2_generator(G_original, target_resolution)
    
    print(f"Capped generator resolution: {G_capped.img_resolution}")
    
    # Print the capped synthesis network structure
    print("\nCapped synthesis network structure:")
    for name, _ in G_capped.synthesis.named_parameters():
        if 'weight' in name and 'torgb' not in name:
            print(f"  {name}")
    
    # Generate and save sample images using both generators for comparison
    for i, seed in enumerate(seeds):
        if i >= num_samples:
            break
            
        # Create a random latent vector
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G_original.z_dim)).to(device)
        
        # Create a label tensor (zeros for unconditional models)
        c = torch.zeros([1, G_original.c_dim], device=device)
        
        # Generate images with both generators
        print(f"Generating images for seed {seed}...")
        
        # Original generator
        img_original = G_original(z, c, truncation_psi=0.7, noise_mode='const')
        img_original = (img_original.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        # Capped generator
        img_capped = G_capped(z, c, truncation_psi=0.7, noise_mode='const')
        img_capped = (img_capped.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        # Save images
        PIL.Image.fromarray(img_original[0].cpu().numpy(), 'RGB').save(
            f"{output_dir}/seed{seed:04d}_original_{G_original.img_resolution}.png")
        PIL.Image.fromarray(img_capped[0].cpu().numpy(), 'RGB').save(
            f"{output_dir}/seed{seed:04d}_capped_{target_resolution}.png")
    
    print(f"Generated {min(num_samples, len(seeds))} sample images in {output_dir}")
    
    # Save the capped generator
    output_pkl = f"stylegan2_capped_{target_resolution}.pkl"
    print(f"Saving capped generator to {output_pkl}...")
    
    # Create a new network data dictionary with the capped generator
    capped_network_data = {k: v for k, v in network_data.items()}
    capped_network_data['G_ema'] = G_capped
    
    with open(output_pkl, 'wb') as f:
        torch.save(capped_network_data, f)
    
    print(f"Capped generator saved to {output_pkl}")

if __name__ == "__main__":
    main()