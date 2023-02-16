import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.experimental.maps import mesh
from jax.experimental import stax

from jax.config import config
config.update("jax_enable_x64", True)

from .utils import get_rays