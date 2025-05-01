from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array as JaxArray

from utils.img_utils import extract_tiles
from eincm.objectives.contrast_objectives import compute_mean_gradient_magnitude



def compute_mean_squared_error(
        arr_1: JaxArray, 
        arr_2: JaxArray
) -> jnp.float64:
    """Computes and returns mean of squared errors between two input arrays

    Args:
        arr_1 (JaxArray): A 2D image-like JAX array
        arr_2 (JaxArray): A 2D image-like JAX array

    Returns:
        jnp.float64: The mean squared error
    """
    squared_error = (arr_1 - arr_2)**2
    mean_squared_error = squared_error.mean()
    return mean_squared_error


def compute_sum_squared_error(
        arr_1: JaxArray, 
        arr_2: JaxArray
) -> jnp.float64:
    """Computes and returns sum of squared errors between two input arrays

    Args:
        arr_1 (JaxArray): A 2D image-like JAX array
        arr_2 (JaxArray): A 2D image-like JAX array

    Returns:
        jnp.float64: The Sum of Squared Errors
    """
    squared_error = (arr_1 - arr_2)**2
    sum_squared_error = squared_error.sum()
    return sum_squared_error


def compute_mean_hadamard_product(
        arr_1: JaxArray, 
        arr_2: JaxArray
) -> jnp.float64:
    """Computes the hadamard (element-wise) product between two input arrays 
    and returns mean of the 2D product array

    Args:
        arr_1 (JaxArray): A 2D image-like JAX array
        arr_2 (JaxArray): A 2D image-like JAX array

    Returns:
        jnp.float64: The mean of the hadamard product
    """
    hadamard_product = arr_1 * arr_2
    mean_hadamard_product = hadamard_product.mean()
    return mean_hadamard_product


def compute_sum_hadamard_product(
        arr_1: JaxArray, 
        arr_2: JaxArray
) -> jnp.float64:
    """Computes the hadamard (element-wise) product between two input arrays 
    and returns sum of the 2D product array

    Args:
        arr_1 (JaxArray): A 2D image-like JAX array
        arr_2 (JaxArray): A 2D image-like JAX array

    Returns:
        jnp.float64: The mean of the hadamard product
    """
    hadamard_product = arr_1 * arr_2
    sum_hadamard_product = hadamard_product.sum()
    return sum_hadamard_product


def compute_joint_contrast(
        arr_1: JaxArray, 
        arr_2: JaxArray
) -> jnp.float64:
    """Computes correlation in a joint filtering fashion as suggested in 
    "Joint Filtering of Intensity Images and Neuromorphic Events for 
    High-Resolution Noise-Robust Imaging"

    Args:
        arr_1 (JaxArray): A 2D image-like JAX array (edge)
        arr_2 (JaxArray): A 2D image-like JAX array (iwe)

    Returns:
        jnp.float64: The joint contrast
    """
    joint_contrast = compute_mean_gradient_magnitude(arr_1 + arr_2)
    return joint_contrast


def compute_adaptive_mean_squared_error(
        arr_1: JaxArray, 
        arr_2: JaxArray,
        tile_size: Tuple[int, int] = None
) -> jnp.float64:
    """Computes the mean squared error (MSE) between corresponding tile pairs from the two 2D input arrays
    and returns the sum of the MSEs of each tile pair

    Args:
        arr_1 (JaxArray): A 2D image-like JAX array
        arr_2 (JaxArray): A 2D image-like JAX array
        tile_size (Tuple[int, int], optional): Size of each tile (height, width). Defaults to None.

    Returns:
        jnp.float64: The total MSE across all tile pairs
    """
    if tile_size is None:
        tile_h, tile_w = 32, 42
    else:
        tile_h, tile_w = tile_size
        
    arr1_tiles = extract_tiles(arr_1, tile_h, tile_w)
    arr2_tiles = extract_tiles(arr_2, tile_h, tile_w)
    tile_mses = jax.vmap(compute_mean_squared_error, (0, 0))(arr1_tiles, arr2_tiles)
    tot_tile_mse = tile_mses.sum()
    return tot_tile_mse