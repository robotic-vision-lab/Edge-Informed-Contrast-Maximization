import sys
from typing import Dict

import jax.numpy as jnp
from jax.typing import ArrayLike as JaxArrayLike


__all__ = ['sparse_flow_error']


EPSN = sys.float_info.epsilon


def sparse_flow_error(
        pred_flow: JaxArrayLike, 
        gt_flow: JaxArrayLike, 
        event_mask: JaxArrayLike = None
) -> Dict:
    """Evaluate flow and return End-point Errors, N-pixel error percentages, and counts of evaluations

    Args:
        pred_flow (JaxArrayLike): Predicted flow (pixel displacements) [shape: (H, W, 2)]
        gt_flow (JaxArrayLike): Groundtruth flow (pixel displacements) [shape: (H, W, 2)]
        event_mask (JaxArrayLike, optional): Event mask indicating validity of events [shape: (H, W)]. Defaults to None.

    Returns:
        Dict: _description_
    """

    # construct mask to discard invalid ('0' or 'inf') predicted flow values
    mask_pred = jnp.logical_and(
        jnp.logical_and(~jnp.isinf(pred_flow[..., 0]), 
                        ~jnp.isinf(pred_flow[..., 1])),
        jnp.linalg.norm(pred_flow, axis=-1) > 0
    ) # (H, W)

    if event_mask is not None:
        mask_pred = jnp.logical_and(mask_pred, event_mask) # (H, W)
    
    # construct mask to discard invalid ('0' or 'inf') groundtruth flow values
    mask_gt = jnp.logical_and(
        jnp.logical_and(~jnp.isinf(gt_flow[..., 0]), 
                        ~jnp.isinf(gt_flow[..., 1])),
        jnp.linalg.norm(gt_flow, axis=-1) > 0
    ) # (H, W)

    # generate combined mask to select coord locations with valid pred and gt flow values
    mask_intersection = jnp.logical_and(mask_pred, mask_gt) # (H, W)

    # use mask to gather valid pred_flow and gt_flow values
    pred_masked = pred_flow[mask_intersection] # (n_ee, 2)
    gt_masked = gt_flow[mask_intersection]     # (n_ee, 2)

    # compute Average End-Point Error (Average EPE or Average EE or AEE)
    endpoint_err = jnp.linalg.norm(pred_masked - gt_masked, axis=-1) # (n_ee, )

    # compute Average Relative End-Point Error 
    rel_endpoint_err = endpoint_err / (jnp.linalg.norm(gt_masked, axis=-1) + EPSN) # (n_ee, )

    # init errors and counts
    errs, cnts = {}, {}

    # update counts
    cnts['n_ee'] = endpoint_err.shape[0]
    cnts['n_pred'] = mask_pred.sum()
    cnts['n_gt'] = mask_gt.sum()

    # update errors
    errs['AEE'] = endpoint_err.mean()
    errs['AREE'] = rel_endpoint_err.mean()

    # Average N-pixel error (Average N-PE or ANPE) percentages
    for N in [1, 2, 3, 5, 10, 20]:
        errs[f'A{N}PE'] = (endpoint_err > N).sum() * 100 / (cnts['n_ee'] + EPSN)

    return {'errors': errs, 'counts': cnts}