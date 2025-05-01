from functools import partial
from typing import Tuple

import jax
import jax.image as jim
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array as JaxArray
from jax.typing import ArrayLike as JaxArrayLike



@partial(jax.jit, static_argnames=['sensor_size', 'window_size'])
def events_to_pdf_frame(
        xs: JaxArrayLike, 
        ys: JaxArrayLike, 
        sensor_size: Tuple[int, int] = (260, 346), 
        window_size: int = 3
) -> JaxArray:
    """Lays a pdf on a 2D grid for each event (x,y) coordinate.

    Args:
        xs (JaxArrayLike): X coords of warped/unwarped events (float/int)
        ys (JaxArrayLike): Y coords of warped/unwarped events (float/int)
        sensor_size (Tuple[int, int], optional): Size of the sensor (height, width). Defaults to (260, 346).
        window_size (int, optional): Kernel size for Gaussian. Defaults to 3.

    Returns:
        JaxArray: Image of Warped/Unwarped Events
    """
    # compute rounded (integer) event coords
    Xs_warped = jnp.array([xs, ys]).astype(jnp.float64)   # (2, num_events)
    Xs_rounded = jnp.round(Xs_warped).astype(jnp.int32)   # (2, num_events)

    # initialize
    frame = jnp.zeros(sensor_size) # (H, W)
    mu = jnp.array([0, 0])
    covar = jnp.array([[1, 0],
                       [0, 1]]) * 1.0

    w = window_size//2
    for dx in range(-w, w + 1):
        for dy in range(-w, w + 1):
            dX = jnp.array([[dx], [dy]])
            # cols, rows = xs, ys = shifted_coords = Xs_rounded + dXs
            cs, rs = shifted_coords = Xs_rounded + dX   # (2, num_events) 
            
            # Note:
            # -----
            # - `dX` will get broadcasted
            # - `shifted_coords` (like `X_rounded`) are integer type, 
            #   unlike `Xs_warped` which are floats

            # vectors from Xs_warped to shifted_coords for each event
            quantiles = shifted_coords - Xs_warped                                          # (2, num_events) 
            pdf_val = jsp.stats.multivariate_normal.pdf(quantiles.T, mean=mu, cov=covar)    # (num_events, )
            
            # do 'frame[rs, cs] += value' using jax.numpy (JAX: immutable arrays)
            frame = frame.at[rs, cs].add(pdf_val, mode='drop')

    return frame # (H, W)


def make_event_mask(xs: JaxArray, ys: JaxArray, sensor_size: Tuple[int, int]) -> JaxArray:
    """Create mask from events indicating presence of events

    Args:
        xs (JaxArray): A 1D array of X coords of events
        ys (JaxArray): A 1D array of X coords of events
        sensor_size (Tuple[int, int]): Size of the sensor (H, W)

    Returns:
        JaxArray: Boolean mask indicating presence of events
    """
    event_mask = jnp.zeros(sensor_size, dtype=bool)
    event_mask = event_mask.at[ys.astype(jnp.int16), xs.astype(jnp.int16)].set(1, mode='drop')
    return event_mask