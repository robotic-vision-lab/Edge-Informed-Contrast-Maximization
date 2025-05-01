from typing import Tuple

import jax
import jax.numpy as jnp
import jax.image as jim
from jax import Array as JaxArray



def scale_theta_to_sensor_size(
        theta: JaxArray, 
        sensor_size: Tuple[int, int], 
        method: str = 'bilinear'
) -> JaxArray:
    """Scales theta (2D velocity field) to theta_array at given `sensor_size` 

    Args:
        theta (JaxArray): Velocity field [shape: (h, w, 2)]
        sensor_size (Tuple[int, int]): Sensor size (H, W)
        method (str, optional): Method to use for interpolation. Defaults to 'bilinear'.

    Returns:
        scaled_theta (JaxArray): Per pixel theta array (H, W, 2)
    """
    scaled_theta = jim.scale_and_translate(
        image=theta,
        shape=(sensor_size[0], sensor_size[1], 2),
        spatial_dims=(0, 1, 2),
        scale=jnp.array([sensor_size[0]/theta.shape[0], 
                         sensor_size[1]/theta.shape[1], 
                         1.0]
                         ).astype(jnp.float64),
        translation=jnp.array([0.0, 0.0, 0.0]).astype(jnp.float64),
        method=method
    )

    return scaled_theta


@jax.jit
def per_pix_theta_to_flow(
        theta: JaxArray,
        xs: JaxArray,
        ys: JaxArray,
        ts: JaxArray
) -> JaxArray:
    """Theta is per-pixel 2D velocity. Convert per-pixel velocity 
    to (event) flow displacement at event location.

    Args:
        theta (JaxArray): Velocity field [shape: (H, W, 2)]
        xs (JaxArray): X coords of events
        ys (JaxArray): Y coords of events
        ts (JaxArray): Timestamps of events

    Returns:
        JaxArray: 2D flow (displacement) field (H, W, 2)
    """
    xs = xs.astype(jnp.int16) # (n_events,)
    ys = ys.astype(jnp.int16) # (n_events,)
    dt = jnp.ones_like(ts) # (n_events,)
    # Note: The estimated flow is to be compared with groundtruth which is provided on per-pixel basis.
    # Therefore, the theta to flow conversion ignores the timestamp (setting all timestamps to 1.0, to indicate 
    # event motion from beginning to end of event volume) of events and only considers pixel location.

    # start with zero flow array
    flow = jnp.zeros((theta.shape))

    # update flow with theta*dt at event locations,  
    flow = flow.at[ys, xs, 0].set(theta[ys, xs, 0]*dt, mode='drop')
    flow = flow.at[ys, xs, 1].set(theta[ys, xs, 1]*dt, mode='drop')

    return flow # (H, W, 2)