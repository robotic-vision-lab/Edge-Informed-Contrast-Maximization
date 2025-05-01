import jax
import jax.numpy as jnp



@jax.jit
def per_pix_warp(per_pix_theta, xs, ys, ts, t_ref, delta_time):
    """Per pixel warper
    per_pix_theta   (H, W, 2)       float64
    xs              (n_events,)     int16
    ys              (n_events,)     int16
    ts              (n_events,)     float64
    t_ref           scalar          float64

    Let:
        t_0 = ts[0]
        t_n = ts[-1]

    Assume that between t_0 and ts, the visual object generating 
    the events moved to X (x,y). Thereby generating the events zip(xs, ys, ts).
    If we undo the flow (t_0 through ts) the spatial coordinates 
    zip(xs, ys) should move back to zip(warped_xs, warped_ys) representing the 
    visual object's position at t_0.

    Unless specified otherwise, t_ref = t_0. Therefore, warping will cause
    displacing the events back in time to t_0.
    """
    xs = jnp.round(xs).astype(jnp.int16)
    ys = jnp.round(ys).astype(jnp.int16)
    dts = ts - t_ref

    # per_event_theta_x = per_pix_theta[ys, xs, 0] # (n_events, 2)
    # per_event_theta_y = per_pix_theta[ys, xs, 1] # (n_events, 2)
    warped_xs = xs - per_pix_theta[ys, xs, 0]*dts*delta_time
    warped_ys = ys - per_pix_theta[ys, xs, 1]*dts*delta_time

    return warped_xs, warped_ys
