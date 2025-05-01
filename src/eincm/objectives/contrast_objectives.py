from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array as JaxArray
from jax.typing import ArrayLike as JaxArrayLike

from utils.img_utils import extract_tiles
from utils.img_utils import sobel_scharr_optimized_image_grads



def compute_mean_gradient_magnitude(arr: JaxArray) -> jnp.float64:
    """Computes and returns mean of gradient magnitude of 2D input array

    Args:
        arr (JaxArray): A 2D image-like JAX array

    Returns:
        jnp.float64: The mean gradient magnitude of `arr`
    """
    arr_grad = sobel_scharr_optimized_image_grads(arr.astype(jnp.float64))
    arr_grad_mag = arr_grad[..., 0]**2 + arr_grad[..., 1]**2 # note: no sqrt, so its mag**2
    # arr_grad_mag = jnp.sqrt(arr_grad_mag)
    mean_arr_grad_mag = arr_grad_mag.mean()
    return mean_arr_grad_mag


def compute_variance(arr: JaxArrayLike) -> jnp.float64:
    """Computes and returns variance of 2D input array

    Args:
        arr (JaxArrayLike): A 2D image-like JAX array

    Returns:
        jnp.float64: The variance of `arr`
    """
    arr_variance = jnp.var(arr.astype(jnp.float64))
    return arr_variance


def compute_adaptive_mean_gradient_magnitude(
        arr: JaxArray,
        tile_size: Tuple[int, int] = None
) -> jnp.float64:
    """Computes mean magnitudes of the gradient of tiles within a 2D input array and returns the sum

    Args:
        arr (JaxArray): A 2D image-like JAX array
        tile_size (Tuple[int, int], optional): Size of each tile (height, width). Defaults to None.

    Returns:
        jnp.float64: The sum of mean gradient magnitudes across all tiles
    """
    if tile_size is None:
        tile_h, tile_w = 32, 42
    else:
        tile_h, tile_w = tile_size
        
    tiles = extract_tiles(arr, tile_h, tile_w)
    tile_mgms = jax.vmap(compute_mean_gradient_magnitude)(tiles)
    tot_tile_mgm = tile_mgms.sum()
    return tot_tile_mgm


def compute_adaptive_variance(
        arr: JaxArrayLike,
        tile_size: Tuple[int, int] = None
) -> jnp.float64:
    """Computes variances of tiles within a 2D input array and returns the sum

    Args:
        arr (JaxArrayLike): A 2D image-like JAX array
        tile_size (Tuple[int, int], optional): Size of each tile (height, width). Defaults to None.

    Returns:
        jnp.float64: The sum of variances across all tiles
    """
    if tile_size is None:
        tile_h, tile_w = 32, 42
    else:
        tile_h, tile_w = tile_size
        
    tiles = extract_tiles(arr, tile_h, tile_w)
    tile_variances = jax.vmap(compute_variance)(tiles)
    tot_tile_variance = tile_variances.sum()
    return tot_tile_variance
