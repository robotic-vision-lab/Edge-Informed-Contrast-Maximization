import time

import jax.numpy as jnp
from easydict import EasyDict as edict

from eincm.losses import compute_loss_objectives
from utils.event_utils import events_to_pdf_frame
from utils.theta_utils import per_pix_theta_to_flow

from .flow_eval import sparse_flow_error



def evaluate_theta_array(theta_array, 
                         eval_xs, eval_ys, eval_ts,
                         edges, edge_ts,
                         gt_flow, 
                         alpha, beta, gamma, delta,
                         sensor_size, err_eval_event_mask=None):
    # compute all things you want to display or plot
    loss_obj = compute_loss_objectives(theta_array, 
                                       eval_xs, eval_ys, eval_ts,
                                       edges, edge_ts,
                                       sensor_size)

    # locals().update(l_obj)
    mean_rel_contrast = loss_obj['rel_contrasts'].mean()
    mean_rel_corr     = loss_obj['rel_correlations'].mean()
    mean_rel_iwe_div  = loss_obj['rel_iwe_divergences'].mean()
    tot_var           = loss_obj['theta_total_variation']
    theta_div         = loss_obj['theta_divergence']
    flow_warp_loss    = loss_obj['flow_warp_losses'][0]
    l_xs_warped       = loss_obj['warped_xs'][0]
    l_ys_warped       = loss_obj['warped_ys'][0]
    
    l_iwe = events_to_pdf_frame(l_xs_warped, l_ys_warped, sensor_size)
    loss = (
        alpha * (-mean_rel_contrast)
        + beta * (-mean_rel_corr)
        + gamma * tot_var
        + delta * mean_rel_iwe_div
    )

    evals = {}
    acc_eval_str = ''
    if gt_flow is not None:
        pred_flow = per_pix_theta_to_flow(theta_array, eval_xs, eval_ys, eval_ts)
        flow_evals = edict(sparse_flow_error(pred_flow, gt_flow, err_eval_event_mask))
        AEE      = flow_evals.errors.AEE
        AREE     = flow_evals.errors.AREE
        p1_AEE   = flow_evals.errors.A1PE
        p2_AEE   = flow_evals.errors.A2PE
        p3_AEE   = flow_evals.errors.A3PE
        p5_AEE   = flow_evals.errors.A5PE
        p10_AEE  = flow_evals.errors.A10PE
        p20_AEE  = flow_evals.errors.A20PE
        n_points = flow_evals.counts.n_ee
        n_pred   = flow_evals.counts.n_pred
        n_gt     = flow_evals.counts.n_gt
        n_pixels = sensor_size[0]*sensor_size[1]

        evals.update(flow_evals.errors)
        evals.update(flow_evals.counts)
        evals.update({'n_pixels': n_pixels})

        acc_eval_str = (f', AEE(↓): {f"{AEE:8.6f}"}, AREE(↓): {f"{AREE:8.6f}"}, '
            + f'A1PE(↓): {p1_AEE:8.6f}, A2PE(↓): {p2_AEE:8.6f}, A3PE(↓): {p3_AEE:8.6f}, A5PE(↓): {p5_AEE:8.6f}, '
            + f'A10PE(↓): {p10_AEE:8.6f}, A20PE(↓): {p20_AEE:8.6f}, '
            + f'| n_pixels:{n_pixels:,}, n_gt_mask:{n_gt:,}, n_event_mask:{n_pred:,}, n_ee: {n_points:,}\n')

    time_str = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}]'
    eval_str =  (f'total_loss(↓): {loss:8.6f}, iwe_var(↑): {f"{jnp.var(l_iwe):8.6f}"}, '
            + f'mean_rel_contrast(↑): {f"{mean_rel_contrast:8.6f}"}, mean_rel_corr(↑): {f"{mean_rel_corr:8.6f}"}, '
            + f'theta_tot_var(↓): {f"{tot_var:8.6f}"}, theta_div(↓): {f"{theta_div:8.6f}"}, '
            + f'mean_rel_iwe_div(↓): {f"{mean_rel_iwe_div:8.6f}"}, '
            + f'FWL(↑): {f"{flow_warp_loss:8.6f}"}'
            + f'{acc_eval_str}')
    
    evals.update({
        'loss': loss,
        'iwe_var': jnp.var(l_iwe),
        'mean_rel_contrast': mean_rel_contrast,
        'mean_rel_corr': mean_rel_corr,
        'theta_tot_var': tot_var,
        'theta_div': theta_div,
        'fwl': flow_warp_loss,
        'mean_rel_iwe_div': mean_rel_iwe_div,
        'rel_iwe_divergences': loss_obj['rel_iwe_divergences'],
        'rel_contrasts': loss_obj['rel_contrasts'],
        'rel_correlations': loss_obj['rel_correlations'],
        'flow_warp_losses': loss_obj['flow_warp_losses'],
        'multi_ref_weights': loss_obj['multi_ref_weights'],
    })

    return time_str, eval_str, evals, loss_obj


