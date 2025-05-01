import jax.numpy as jnp
from jax.typing import ArrayLike as JaxArrayLike



def compute_fwl(iwe: JaxArrayLike, zero_iwe: JaxArrayLike) -> jnp.float64:
    """Computes Flow Warp Loss (FWL), given the image of warped events (IWE) 
    and the image of unwarped (zero-warp) events IUE

    Args:
        iwe (JaxArrayLike): A 2D image-like JAX array
        zero_iwe (JaxArrayLike): A 2D image-like JAX array

    Returns:
        jnp.float64: The FWL
    """
    fwl = jnp.var(iwe) / jnp.var(zero_iwe)
    return fwl