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


def linear_to_srgb(x):
    """Convert linear RGB to sRGB."""
    return jnp.where(x <= 0.0031308, 12.92 * x, 1.055 * jnp.power(x, 1 / 2.4) - 0.055)