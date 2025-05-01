import sys
from typing import Dict, Tuple

import jax
import math
import numpy as np
import scipy.stats as stats
from jax import Array as JaxArray

from eincm.contrast_metrics import compute_fwl
from eincm.event_warpers import per_pix_warp
from eincm.objectives.contrast_objectives import compute_mean_gradient_magnitude
from eincm.objectives.correlation_objectives import compute_joint_contrast
from eincm.objectives.correlation_objectives import compute_mean_squared_error
from eincm.objectives.event_collapse_objectives import iwe_divergence
from eincm.regularizers import per_pix_theta_divergence
from eincm.regularizers import per_pix_total_variation
from utils.event_utils import events_to_pdf_frame
from utils.img_utils import normalize_to_unit_range
from utils.theta_utils import scale_theta_to_sensor_size



EPSN = sys.float_info.epsilon

vmapped_per_pix_warp = jax.vmap(per_pix_warp, (None, None, None, None, 0, None))
vmapped_events_to_pdf_frame = jax.vmap(events_to_pdf_frame, (0, 0, None))
vmapped_normalize_to_unit_range = jax.vmap(normalize_to_unit_range)
vmapped_iwe_divergence = jax.vmap(iwe_divergence)
vmapped_compute_fwl = jax.vmap(compute_fwl, (0, None))

vmapped_compute_joint_contrast = jax.vmap(compute_joint_contrast, (0, 0))
vmapped_compute_joint_contrast_zero = jax.vmap(compute_joint_contrast, (0, None))
vmapped_compute_mean_squared_error = jax.vmap(compute_mean_squared_error, (0, 0))
vmapped_compute_mean_squared_error_zero = jax.vmap(compute_mean_squared_error, (0, None))
vmapped_compute_mean_gradient_magnitude = jax.vmap(compute_mean_gradient_magnitude)


def compute_weights_for_multi_reference(n_refs, n_sigma=1.5):
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)

    w = stats.norm.pdf(np.linspace(mu - n_sigma*sigma, mu + n_sigma*sigma, n_refs), mu, sigma)

    return w / w.sum()


def compute_loss_objectives(theta, 
                            xs, ys, ts,
                            edges, edge_ts,
                            sensor_size):
    # construct zero IWE (or IUE)
    zero_iwe = events_to_pdf_frame(xs, ys, sensor_size)     # (H, W)
    normalized_zero_iwe = normalize_to_unit_range(zero_iwe) # (H, W)

    # warp events to reference times t0, tmid, t1, etc 
    warped_xs, warped_ys = vmapped_per_pix_warp(theta, xs, ys, ts, edge_ts, 1.0) # (n_imgs | n_refs, n_events)
    
    # construct corresponding IWEs
    iwes = vmapped_events_to_pdf_frame(warped_xs, warped_ys, sensor_size) # (n_imgs | n_refs, H, W)
    normalized_iwes = vmapped_normalize_to_unit_range(iwes)               # (n_imgs | n_refs, H, W)

    # compute edge correlations for IWEs and IUE (negative MSE)
    corrs = vmapped_compute_mean_squared_error(edges, normalized_iwes)*(-1)               # (n_imgs,)
    zero_corrs = vmapped_compute_mean_squared_error_zero(edges, normalized_zero_iwe)*(-1) # (n_imgs,)
    rel_corrs = corrs / ((zero_corrs) + EPSN)

    # compute contrast for IWEs and IUE (mean grad IWE)
    contrasts = vmapped_compute_mean_gradient_magnitude(iwes) # (n_imgs | n_refs,)
    zero_contrast = compute_mean_gradient_magnitude(zero_iwe) # ()
    rel_contrasts = contrasts / ((zero_contrast) + EPSN)      # (n_imgs | n_refs,)

    # compute total variation (regularizer), and theta divergence
    theta_total_variation = per_pix_total_variation(theta, xs, ys, ts) # ()
    theta_divergence = per_pix_theta_divergence(theta)                 # ()

    # compute divergences for IWEs and IUE
    iwe_divergences = vmapped_iwe_divergence(normalized_iwes)              # (n_imgs | n_refs,)
    zero_iwe_divergence = iwe_divergence(normalized_zero_iwe)              # ()
    rel_iwe_divergences = iwe_divergences / ((zero_iwe_divergence) + EPSN) # (n_imgs | n_refs,)

    # compute FWL for each IWE
    flow_warp_losses = vmapped_compute_fwl(iwes, zero_iwe) # (n_imgs | n_refs,)

    # compute multiple reference weights
    multi_ref_weights = compute_weights_for_multi_reference(n_refs=len(edge_ts))
    
    return {
        'warped_xs': warped_xs,                         # float64 (n_events,)
        'warped_ys': warped_ys,                         # float64 (n_events,)
        'correlations': corrs,                          # float64 (n_imgs,)
        'zero_correlations': zero_corrs,                # float64 (n_imgs,)
        'rel_correlations': rel_corrs,                  # float64 (n_imgs,)
        'contrasts': contrasts,                         # float64 (n_imgs | n_refs,)
        'zero_contrast': zero_contrast,                 # float64 ()
        'rel_contrasts': rel_contrasts,                 # float64 (n_imgs | n_refs,)
        'theta_total_variation': theta_total_variation, # float64 ()
        'theta_divergence': theta_divergence,           # float64 ()
        'iwe_divergences': iwe_divergences,             # float64 (n_imgs | n_refs,)
        'zero_iwe_divergence': zero_iwe_divergence,     # float64 ()
        'rel_iwe_divergences': rel_iwe_divergences,     # float64 (n_imgs | n_refs,)
        'flow_warp_losses': flow_warp_losses,           # float64 (n_imgs | n_refs,)
        'multi_ref_weights': multi_ref_weights          # float64 (n_imgs | n_refs,)
    }


def loss_func(
        theta: JaxArray,
        xs: JaxArray,
        ys: JaxArray,
        ts: JaxArray,
        edges: JaxArray, 
        edge_ts: JaxArray,
        alpha: int,
        beta: int,
        gamma: int,
        delta: int,
        cur_pyr_lvl: int,
        n_pyr_lvls: int,
        sensor_size: Tuple[int, int],
        scale_to_sensor_size_method: str
) -> Tuple[int, Dict]:
    """Computes the loss comprising weighting an assortment of objectives and returns final loss along with auxilliary computation results.
    
    Steps:
        1. Scale current theta to full size
        2. Form IWEs using warped events that are warped to multiple timestamps (according to available timestamps of images)
        3a. Compute corresponding correlations between Image edges and IWEs (MSE)
        3b. Compute contrast of each IWE (magnitude of gradient of IWE)
        4. Construct final loss function 

    Args:
        theta (JaxArray): Per pixel 2D velocity field [shape: (h, w, 2)]
        xs (JaxArray): X coords of events in a sample [shape: (n_events,)]
        ys (JaxArray): Y coords of events in a sample [shape: (n_events,)]
        ts (JaxArray): Timestamps of events (normalized) [shape: (n_events,)]
        edges (JaxArray): Edges of images within a sample [shape: (n_imgs, H, W)]
        edge_ts (JaxArray): Timestamps of the input edges [shape: (n_imgs,)]
        alpha (int): Weight for contrast objective
        beta (int): Weight for correlation objective
        gamma (int): Weight for regularizer (total_variation)
        delta (int): Weight for event collapse objective (IWE divergence)
        cur_pyr_lvl (int): Pyramid level associated with the input theta
        n_pyr_lvls (int): Total number of pyramid levels
        sensor_size (Tuple[int, int]): Size of sensor (H, W)
        scale_to_sensor_size_method (str): Upscale method to upscale theta to sensor array size

    Returns:
        Tuple[int, Dict]: Final loss and a dictionary of auxiliary information as a tuple
    """
    # using theta, undo the flow (displacement) by warping events
    # from ts -> t_ref, then construct iwe

    # --------------------------------------------------------------------------------
    # scale theta to original sensor size, construct IWEs and IUE 
    # --------------------------------------------------------------------------------
    scaled_theta = scale_theta_to_sensor_size(theta, 
                                              sensor_size, 
                                              scale_to_sensor_size_method) # (H, W, 2)

    loss_objectives = compute_loss_objectives(scaled_theta,
                                              xs, ys, ts,
                                              edges, edge_ts,
                                              sensor_size)
    
    corrs                 = loss_objectives['correlations']        # (n_imgs,)
    zero_corrs            = loss_objectives['zero_correlations']   # (n_imgs,)
    contrasts             = loss_objectives['contrasts']           # (n_imgs | n_refs)
    zero_contrast         = loss_objectives['zero_contrast']       # ()
    theta_total_variation = loss_objectives['theta_total_variation'] if cur_pyr_lvl <=0 else 0.0 # ()
    iwe_divergences       = loss_objectives['iwe_divergences']     # (n_refs,)
    zero_iwe_divergence   = loss_objectives['zero_iwe_divergence'] # (n_refs,)  
    multi_ref_weights     = loss_objectives['multi_ref_weights']   # (n_imgs | n_refs)

    rel_corrs           = (multi_ref_weights*corrs) / ((zero_corrs) + EPSN)                     # (n_imgs)
    rel_contrasts       = (multi_ref_weights*contrasts) / ((zero_contrast) + EPSN)              # (n_imgs | n_refs)
    rel_iwe_divergences = (multi_ref_weights*iwe_divergences) / ((zero_iwe_divergence) + EPSN)  # (n_imgs | n_refs)

    mean_rel_corr           = rel_corrs.mean()
    mean_rel_contrast       = rel_contrasts.mean()
    mean_rel_iwe_divergence = rel_iwe_divergences.mean()

    # --------------------------------------------------------------------------------
    # construct final loss by combining objectives to minimize
    # --------------------------------------------------------------------------------
    contrast_loss = mean_rel_contrast*(-1)  # minimize the additive inverse (*(-1)) or multiplicative inverse (^(-1))
    correlation_loss = mean_rel_corr*(-1)   # minimize the additive inverse (*(-1)) or multiplicative inverse (^(-1))
    
    contrast_correlation_loss = (alpha*contrast_loss + beta*correlation_loss)**1 # Our C^2M hybrid objective
    regularization_loss = gamma*theta_total_variation + delta*mean_rel_iwe_divergence
    
    final_loss = contrast_correlation_loss + regularization_loss
    
    aux_info = {
        'final_loss': final_loss, 
        'scaled_theta': scaled_theta,
        'mean_rel_corr': mean_rel_corr, 
        'mean_rel_contrast': mean_rel_contrast, 
        'mean_rel_iwe_divergence': mean_rel_iwe_divergence,
        'theta_total_variation': theta_total_variation,
        'multi_ref_weights': multi_ref_weights,
    } # can be useful with certain solvers only

    return final_loss, aux_info


def handover_loss_func(
        alpha_handover:float,
        prev_theta: JaxArray,
        theta: JaxArray,
        xs: JaxArray,
        ys: JaxArray,
        ts: JaxArray,
        edges: JaxArray, 
        edge_ts: JaxArray,
        alpha: int,
        beta: int,
        gamma: int,
        delta: int,
        cur_pyr_lvl: int,
        n_pyr_lvls: int,
        sensor_size: Tuple[int, int],
        scale_to_sensor_size_method: str,
) -> Tuple[int, Dict]:
    """Computes the hand over loss comprising of 'hand over' and weighting 
    an assortment of objectives and returns final loss along with 
    auxilliary computation results.
    
    Steps:
        1. Scale current theta to full size
        2. Form IWEs using warped events that are warped to multiple timestamps (according to available timestamps of images)
        3a. Compute corresponding correlations between Image edges and IWEs (MSE)
        3b. Compute contrast of each IWE (magnitude of gradient of IWE)
        4. Construct final loss function 

    Args:
        alpha_handover (float): Weight to be used for linearly combining prev_theta and theta
        prev_theta (JaxArray): Per pixel 2D velocity field from previous iteration (one level finer unless finest) [shape: (h, w, 2)]
        theta (JaxArray): Per pixel 2D velocity field [shape: (h, w, 2)]
        xs (JaxArray): X coords of events in a sample [shape: (n_events,)]
        ys (JaxArray): Y coords of events in a sample [shape: (n_events,)]
        ts (JaxArray): Timestamps of events (normalized) [shape: (n_events,)]
        edges (JaxArray): Edges of images within a sample [shape: (n_imgs, H, W)]
        edge_ts (JaxArray): Timestamps of the input edges [shape: (n_imgs,)]
        alpha (int): Weight for contrast objective
        beta (int): Weight for correlation objective
        gamma (int): Weight for regularizer (total_variation)
        delta (int): Weight for event collapse objective (IWE divergence)
        cur_pyr_lvl (int): Pyramid level associated with the input theta
        n_pyr_lvls (int): Total number of pyramid levels
        sensor_size (Tuple[int, int]): Size of sensor (H, W)
        scale_to_sensor_size_method (str): Upscale method to upscale theta to sensor array size

    Returns:
        Tuple[int, Dict]: Final loss and a dictionary of auxiliary information as a tuple
    """

    # add upscale awareness
    # loss objective computation scales theta to array from cur_lvl
    # however, for middle levels an upscale operation is performed for all middle levels
    # the computed loss objective can be different when scaling theta to array size from the upscaled theta
    # therefore, the loss objective computation should be made aware of the additional prior upscaling for the middle 
    # levels to ensure meaningful optimization

    # if cur_pyr_lvl > 0:
    #     upcaled_theta = theta_lvl_upscaler()
    # else: # level 0
    theta_ho = alpha_handover*prev_theta + (1-alpha_handover)*theta
    loss, _ = loss_func(theta_ho, 
                        xs, ys, ts, 
                        edges, edge_ts,
                        alpha, beta, gamma, delta,
                        cur_pyr_lvl, n_pyr_lvls, sensor_size, scale_to_sensor_size_method)
    
    return loss
