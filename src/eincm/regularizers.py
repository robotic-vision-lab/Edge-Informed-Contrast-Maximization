import sys 

import jax
import jax.numpy as jnp

from utils.img_utils import sobel_scharr_optimized_image_grads
from utils.theta_utils import per_pix_theta_to_flow



EPSN = sys.float_info.epsilon


@jax.jit
def per_pix_total_variation(theta, xs, ys, ts):
    flow = per_pix_theta_to_flow(theta, xs, ys, ts) # (H, W, 2)
    # flow = theta
    flow_x, flow_y = flow.transpose(2, 0, 1) # (2, H, W)
    
    flow_x_grad = sobel_scharr_optimized_image_grads(flow_x) # (H, W, 2)
    flow_y_grad = sobel_scharr_optimized_image_grads(flow_y) # (H, W, 2)

    flow_x_grad_x, flow_x_grad_y = flow_x_grad.transpose(2, 0, 1) # (2, H, W)
    flow_y_grad_x, flow_y_grad_y = flow_y_grad.transpose(2, 0, 1) # (2, H, W)

    nz_grad_mask = ((jnp.abs(flow_x_grad_x) > 0) 
                  | (jnp.abs(flow_x_grad_y) > 0)
                  | (jnp.abs(flow_y_grad_x) > 0)
                  | (jnp.abs(flow_y_grad_y) > 0))

    tot_var = jnp.sum(
        (
            (jnp.abs(flow_x_grad_x)*0.25 + jnp.abs(flow_x_grad_y)*0.25)
            + (jnp.abs(flow_y_grad_x)*0.25 + jnp.abs(flow_y_grad_y)*0.25)
        ) 
    ) / (nz_grad_mask.sum() + EPSN) # L1 version

    return tot_var


@jax.jit
def per_pix_theta_divergence(theta):
    theta_x, theta_y = theta.transpose(2, 0, 1) # (2, H, W)
    theta_x_grad = sobel_scharr_optimized_image_grads(theta_x) # (H, W, 2)
    theta_y_grad = sobel_scharr_optimized_image_grads(theta_y) # (H, W, 2)

    theta_x_grad_x, theta_x_grad_y = theta_x_grad.transpose(2, 0, 1) # (2, H, W)
    theta_y_grad_x, theta_y_grad_y = theta_y_grad.transpose(2, 0, 1) # (2, H, W)

    div_kern = jnp.array([[1/12, 1/6, 1/12], [1/6, 0.0, 1/6], [1/12, 1/6, 1/12]])
    theta_div_xx = jax.scipy.signal.convolve(theta_x_grad_x, div_kern, mode='same')
    theta_div_xy = jax.scipy.signal.convolve(theta_x_grad_y, div_kern, mode='same')
    theta_div_yx = jax.scipy.signal.convolve(theta_y_grad_x, div_kern, mode='same')
    theta_div_yy = jax.scipy.signal.convolve(theta_y_grad_y, div_kern, mode='same')

    theta_div = jnp.abs(theta_div_xx + theta_div_xy + theta_div_yx + theta_div_yy) # |divergence of theta|

    return theta_div.mean() # to be minimized