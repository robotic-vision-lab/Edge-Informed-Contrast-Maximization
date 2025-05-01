import jax
import jax.numpy as jnp

from utils.img_utils import sobel_scharr_optimized_image_grads



@jax.jit
def iwe_divergence(iwe):
    iwe_grad = sobel_scharr_optimized_image_grads(iwe) # (H, W, 2)

    iwe_grad_x, iwe_grad_y = iwe_grad.transpose(2, 0, 1) # (2, H, W)

    div_kern = jnp.array([[1/12, 1/6, 1/12], [1/6, 0.0, 1/6], [1/12, 1/6, 1/12]])
    iwe_div_x = jax.scipy.signal.convolve(iwe_grad_x, div_kern, mode='same')
    iwe_div_y = jax.scipy.signal.convolve(iwe_grad_y, div_kern, mode='same')

    iwe_div = jnp.abs(iwe_div_x + iwe_div_y) # |divergence of iwe|

    return iwe_div.mean() # needs to be minimized