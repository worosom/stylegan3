"""
Utility functions for capping the resolution of StyleGAN2 and StyleGAN3 generators.
"""

import re
import copy
import torch
from torch_utils import misc

def create_capped_stylegan3_generator(G_original, target_resolution):
    """
    Create a new StyleGAN3 generator with a capped resolution based on the original generator.
    
    Args:
        G_original: The original StyleGAN3 generator model
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
    
    # Create a copy of the original generator's kwargs
    kwargs = copy.deepcopy(G_original.init_kwargs)
    
    # Update the resolution in the kwargs
    kwargs['img_resolution'] = target_resolution
    
    # Adjust synthesis kwargs if needed
    if 'synthesis_kwargs' in kwargs:
        kwargs['synthesis_kwargs']['img_resolution'] = target_resolution
    
    # Create a new generator with the updated kwargs
    G_capped = type(G_original)(**kwargs).eval().requires_grad_(False)
    
    # Copy the mapping network weights (these are independent of resolution)
    misc.copy_params_and_buffers(G_original.mapping, G_capped.mapping, require_all=True)
    
    # Copy the synthesis network weights up to the target resolution
    # First, get all named parameters and buffers from both networks
    orig_params = dict(G_original.synthesis.named_parameters())
    orig_buffers = dict(G_original.synthesis.named_buffers())
    capped_params = dict(G_capped.synthesis.named_parameters())
    capped_buffers = dict(G_capped.synthesis.named_buffers())
    
    # Copy input layer parameters (these are always needed)
    for name in capped_params:
        if name.startswith('input.'):
            if name in orig_params:
                capped_params[name].copy_(orig_params[name])
    
    for name in capped_buffers:
        if name.startswith('input.'):
            if name in orig_buffers:
                capped_buffers[name].copy_(orig_buffers[name])
    
    # Copy synthesis layer parameters up to the target resolution
    for name in list(capped_params.keys()):
        # Extract resolution from layer name (e.g., 'L2_32_512' -> 32)
        match = re.match(r'L\d+_(\d+)_\d+', name.split('.')[0])
        if match:
            layer_res = int(match.group(1))
            # Only copy if the layer resolution is less than or equal to target
            if layer_res <= target_resolution:
                if name in orig_params:
                    capped_params[name].copy_(orig_params[name])
    
    for name in list(capped_buffers.keys()):
        match = re.match(r'L\d+_(\d+)_\d+', name.split('.')[0])
        if match:
            layer_res = int(match.group(1))
            if layer_res <= target_resolution:
                if name in orig_buffers:
                    capped_buffers[name].copy_(orig_buffers[name])
    
    return G_capped

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
    
    return G_capped

def load_and_cap_generator(network_pkl, target_resolution, is_stylegan3=True):
    """
    Load a pre-trained StyleGAN model and create a generator with a capped resolution.
    
    Args:
        network_pkl: Path or URL to the .pkl file containing the StyleGAN model
        target_resolution: The target resolution to cap at (e.g., 512)
        is_stylegan3: Whether the model is StyleGAN3 (True) or StyleGAN2 (False)
        
    Returns:
        A tuple containing (original_generator, capped_generator)
    """
    import dnnlib
    import legacy
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the network
    with dnnlib.util.open_url(network_pkl) as f:
        network_data = legacy.load_network_pkl(f)
        G_original = network_data['G_ema'].to(device)
    
    # Create a capped generator
    if is_stylegan3:
        G_capped = create_capped_stylegan3_generator(G_original, target_resolution)
    else:
        G_capped = create_capped_stylegan2_generator(G_original, target_resolution)
    
    return G_original, G_capped

def generate_images(G, z, c=None, truncation_psi=0.7, noise_mode='const'):
    """
    Generate images using a StyleGAN generator.
    
    Args:
        G: The generator model
        z: The latent vectors (batch_size x z_dim)
        c: The class labels (batch_size x c_dim), or None for unconditional models
        truncation_psi: Truncation psi value (lower = more average/less variation)
        noise_mode: Noise mode ('const', 'random', 'none')
        
    Returns:
        Generated images as a torch tensor (batch_size x 3 x resolution x resolution)
    """
    if c is None:
        c = torch.zeros([z.shape[0], G.c_dim], device=z.device)
    
    # Generate images
    img = G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
    
    return img

def save_generator(G, output_pkl, network_data=None):
    """
    Save a generator model to a .pkl file.
    
    Args:
        G: The generator model to save
        output_pkl: Path to the output .pkl file
        network_data: Optional dictionary containing additional network data
        
    Returns:
        None
    """
    if network_data is None:
        network_data = {'G_ema': G}
    else:
        network_data = {k: v for k, v in network_data.items()}
        network_data['G_ema'] = G
    
    with open(output_pkl, 'wb') as f:
        torch.save(network_data, f)