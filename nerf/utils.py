import os
import glob
import tqdm
import math

import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.experimental.maps import mesh
from jax.experimental import stax


from torch.utils.data import Dataset, DataLoader


def custom_meshgrid(*args):
    """Custom meshgrid implementation that works with JAX."""
    return jnp.meshgrid(*args, indexing='ij')

@jit
def linear_to_srgb(x):
    """Convert linear RGB to sRGB."""
    return jnp.where(x <= 0.0031308, 12.92 * x, 1.055 * jnp.power(x, 1 / 2.4) - 0.055)

@jit
def srgb_to_linear(x):
    """Convert sRGB to linear RGB."""
    return jnp.where(x <= 0.04045, x / 12.92, jnp.power((x + 0.055) / 1.055, 2.4))

def get_rays(poses, intrinsics, H, W, N=-1, error_map = None, patch_size = 1):
    '''
    Args:
        poses: (B, 4, 4)
        intrinsics: (4)
        H: int
        W: int
        N: int
        error_map: (H, 128*128)
        patch_size: int
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    
    '''
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    i, j = custom_meshgrid(jnp.linspace(0, W - 1, W, device = device), jnp.linspace(0, H - 1, H, device = device))
    i = i.T.reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.T.reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if patch_size > 1:

            num_patch = N // (patch_size * patch_size)
            inds_x = jnp.random.randint(0, H - patch_size, [num_patch], device = device)
            inds_y = jnp.random.randint(0, W - patch_size, [num_patch], device = device)
            inds = jnp.stack([inds_x, inds_y], dim = -1) # [num_patch, 2]

            pi, pj = custom_meshgrid(jnp.arrange(patch_size, device = device), jnp.arange(patch_size, device = device))
            offsets = jnp.stack([pi.reshape(-1), pj.reshape(-1)], dim = -1)

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [num_patch, patch_size*patch_size, 2]
            inds = inds.reshape(-1, 2) # [num_patch*patch_size*patch_size, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [num_patch*patch_size*patch_size]

            inds = inds.expand([B, N]) # [B, num_patch*patch_size*patch_size]
        
        elif error_map is None:
            inds = jnp.random.randint(0, H*W, [N], device = device)
            inds = inds.expand([B, N])
        else:
            


