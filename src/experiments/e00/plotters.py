import sys

import cv2 as cv
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from eincm.event_warpers import per_pix_warp
from utils.event_utils import events_to_pdf_frame
from utils.img_utils import blend_two_imgs
from utils.img_utils import jnp_to_ocv_n255
from utils.img_utils import normalize_to_unit_range
from utils.img_utils import ocv_normalize_to_255_range as normalize_to_255_range
from utils.flow_utils import flow_uv_to_img
from utils.flow_utils import embed_flow_code_in_img
from utils.theta_utils import per_pix_theta_to_flow



class EINCMExperimentPlotter:
    def __init__(self, cfg):
        self.cfg = cfg

        self.datasample_idx = -1
        self.datasample = None
        self.staged_sample = None

        plt.rcParams.update(self.cfg.mpl_rcparams)


    def update_datasample(self, datasample_idx, datasample, is_staged=False):
        self.set_datasample_idx(datasample_idx)
        self.set_datasample(datasample, is_staged=is_staged)


    def set_datasample(self, datasample, is_staged=False):
        if is_staged:
            self.datasample = datasample
        else:
            self.datasample = self.stage_sample(datasample)


    def stage_sample(self, datasample):
        # --------------------------------------------------------------------------------------------------------#
        #                                             LOAD DATASAMPLE                                             #
        # --------------------------------------------------------------------------------------------------------#
        # load data sample at index
        xs                      = jnp.array(datasample['events']['x']) # (n_events,) int16
        ys                      = jnp.array(datasample['events']['y']) # (n_events,) int16
        ts                      = jnp.array(datasample['events']['t']) # (n_events,) uint64
        ps                      = jnp.array(datasample['events']['p']) # (n_events,) bool
        images                  = jnp.array(datasample['images']) # (n_images, H, W) uint8
        image_ts                = jnp.array(datasample['image_ts']) # (n_images,) int64
        gt_flow_x, gt_flow_y    = jnp.array(datasample['flow_gt']).transpose(2,0,1) if 'flow_gt' in datasample else (None, None) # (H, W), (H, W) int64
        start_time, end_time    = jnp.array(datasample['eval_ts_us'] if 'eval_ts_us' in datasample else datasample['eval_ts']) # (), ()
        file_idx                = jnp.array(datasample['file_idx']) if 'file_idx' in datasample else None # () int64
        n_event_deficiency      = jnp.array(datasample['n_event_deficiency']) if 'n_event_deficiency' in datasample else None # () int64

        # stack groundtruth
        if 'flow_gt' in datasample:
            gt_flow = jnp.stack([gt_flow_x, gt_flow_y], axis=-1) # (H, W, 2)
        else:
            gt_flow = None 
    

        # loaded events were adjusted to have a count of des_n_events
        # but for evaluation we consider events within the eval_ts or 
        # optimization range appropriately for consistency
        if n_event_deficiency > 0: # event scarcity (optimization using superset)
            evt_start_idx, evt_end_idx = np.searchsorted(ts, np.array([start_time, end_time]))
            eval_events_slice = slice(max(0, evt_start_idx+1), min(len(xs), evt_end_idx-1))
            xs_eval = xs[eval_events_slice]
            ys_eval = ys[eval_events_slice]
            ts_eval = ts[eval_events_slice]
            ps_eval = ps[eval_events_slice]
        else: # event surplus (optimization using subset)
            xs_eval = xs
            ys_eval = ys
            ts_eval = ts
            ps_eval = ps


        # normalize datasample eval start and end times to unit range 
        # and normalize other timestamps w.r.t normalized eval_ts
        # delta_time = 1.0 # end_time - start_time
        time_scaler = 1.0
        ts         = ((ts       - start_time)/(end_time - start_time + sys.float_info.epsilon)) * time_scaler # (des_n_events,)
        image_ts   = ((image_ts - start_time)/(end_time - start_time + sys.float_info.epsilon)) * time_scaler # (n_imgs,)
        ts_eval    = ((ts_eval  - start_time)/(end_time - start_time + sys.float_info.epsilon)) * time_scaler # (n_events,)
        # start_time = 0.0
        # end_time   = 1.0 * time_scaler
        t_ref      = 0.0 # normalized start time
        ts_units = 'us' if 'eval_ts_us' in datasample else 's'


        # --------------------------------------------------------------------------------------------------------#
        #                                             EDGE EXTRACTION                                             #
        # --------------------------------------------------------------------------------------------------------#
        # func:           utils.img_utils.preprocess_image
        # remaining args: img
        # injected args:  denoise_h, denoise_template_win_size, denoise_search_win_size, 
        #                 clahe_clip_limit, clahe_tile_grid_size, 
        #                 sharpen_kernel_size, sharpen_sigma_x, sharpen_alpha, sharpen_beta, 
        #                 bilateral_filter_neigh_diameter, bilateral_filter_sigma_color, bilateral_filter_sigma_space,
        self.preprocess_image_pfunc = (hydra.utils.instantiate(self.cfg.edge_extraction.preprocess_image_func)
                                        if self.cfg.enable_image_preprocessing
                                        else lambda x: x)
        
        # func:           utils.img_utils.image_to_edge
        # remaining args: img
        # injected args:  apert_size, th1, th2
        self.image_to_edge_pfunc  = hydra.utils.instantiate(self.cfg.edge_extraction.image_to_edge_func)
        
        # func:           utils.img_utils.smoothen_edges or utils.img_utils.edges_to_inv_exp_dist
        # remaining_args: edge_img
        # injected args:  k_size, sigma or dist_surf_saturation_distance, alpha_iedt, formulation
        self.smoothen_edges_pfunc = hydra.utils.instantiate(self.cfg.edge_extraction.smoothen.smoothen_edges_func)
        
        # preprocess sampled images and normalize to unit range
        images = [
            normalize_to_unit_range(
                self.preprocess_image_pfunc(image).astype(jnp.float64)
            )
            for image in images
        ]

        # construct edge images from grayscale images
        edges = jnp.stack([
            normalize_to_unit_range(
                self.smoothen_edges_pfunc(
                    self.image_to_edge_pfunc(jnp_to_ocv_n255(image))
                )
            )
            for image in images
        ])

        staged_sample = {
            'events': {
                'x': xs,      # (n_events,) int16
                'y': ys,      # (n_events,) int16
                't': ts,      # (n_events,) float64
                'p': ps,      # (n_events,) bool
            },
            'eval_events': {
                'x': xs_eval, # (n_events,) int16
                'y': ys_eval, # (n_events,) int16
                't': ts_eval, # (n_events,) uint64
                'p': ps_eval, # (n_events,) bool
            },
            'images': images,                         # (n_images, H, W) float64
            'edges': edges,                           # (n_images, H, W) float64
            'image_ts': jnp.array(image_ts),          # (n_images,) float64
            'gt_flow': gt_flow,                       # (H, W, 2) float64
            't_ref': t_ref,                           # () float64
            'file_idx': file_idx,                     # () int64
            'n_event_deficiency': n_event_deficiency, # () int64
            'eval_ts': (start_time, end_time),        # (2,) float64
            'eval_ts_units': ts_units,                # str
        }

        return staged_sample


    def set_datasample_idx(self, datasample_idx):
        self.datasample_idx = datasample_idx


    def construct_event_frame(self, events=None, use_eval_events=False, use_polarities=False, norm_type=None, empty_sentinel=-1):
        """Accumulates all events in the datasample event (x-y-t) volume to construct a 2D (x-y) event frame"""
        norm_type = self.cfg.range_normalization.from_0_to_255 if norm_type is None else norm_type
        sensor_size = self.cfg.dataset.sensor_size
        polarity_map = (
            {
                1:  1.0, 
                0:  -1.0, 
                -1: -1.0,
            } if use_polarities
            else 
            {
                1:  1.0, 
                0:  1.0, 
                -1: 1.0,
            }
        )

        if events is None:
            events  = self.datasample['eval_events'] if use_eval_events else self.datasample['events']
        xs = np.array(events['x'])
        ys = np.array(events['y'])
        ps = np.array(events['p'])

        # initialize event frame and event mask
        ev_frame = np.zeros(sensor_size, dtype=float)
        ev_frame_mask = np.zeros(sensor_size, dtype=np.bool_)

        # accumulate events into event frame and mark event mask
        for x, y, p in zip(xs, ys, ps):
            ev_frame[int(y), int(x)] += polarity_map[int(p)] # 1.0 -> 1.0, 0.0 -> -1.0
            ev_frame_mask[int(y), int(x)] |= 1               # 1.0 -> True, 0.0 -> False

        # normalize event frame
        _min_ev_frame = np.min(ev_frame)
        _max_ev_frame = np.max(ev_frame)
        ev_frame_norm = (ev_frame - _min_ev_frame) / (_max_ev_frame - _min_ev_frame + sys.float_info.epsilon)
        if norm_type == self.cfg.range_normalization.from_0_to_255:
            ev_frame_norm = normalize_to_255_range(ev_frame_norm)
        
        # Note: with summing (+/-) polarities, zero-summed accumulation can become indistinguishable 
        # from zero accumulation. Therefore, construct extended event frame where, in the normalized event frame,
        # empty_sentinel value is inserted at locations with no events (zero accumulation) 
        ev_frame_ext = np.where(ev_frame_mask, ev_frame_norm, empty_sentinel)

        return {
            'event_frame': ev_frame, 
            'normalized_event_frame': ev_frame_norm, 
            'extended_normalized_event_frame': ev_frame_ext, 
            'event_frame_mask': ev_frame_mask,
            'range_event_frame': _max_ev_frame - _min_ev_frame
        }


    def overlay_events_on_image(self, events, image, use_eval_events=False, event_color=np.array([213, 0, 50]), mask_gamma=0.382):
        assert image.shape[-1] == 3 and image.dtype == np.uint8, 'image needs to be 3-channel and np.uint8'

        if events is None:
            events  = self.datasample['eval_events'] if use_eval_events else self.datasample['events']

        ev_frame_res = self.construct_event_frame(events)
        ev_frame_norm = ev_frame_res['normalized_event_frame']

        # create alpha (non-binary) mask using the normalized event frame 
        efn =  (ev_frame_norm - ev_frame_norm.min()) / (ev_frame_norm.max() - ev_frame_norm.min() + sys.float_info.epsilon) # 255 norm to 1.0 norm
        efn = efn**mask_gamma # to lift lower event counts for better visualization

        # added to image colors weighted by event counts
        final_img = np.zeros_like(image)
        final_img[..., 0] = (event_color[0]*(efn) + image[..., 0]*(1.0-efn)).astype('uint8')
        final_img[..., 1] = (event_color[1]*(efn) + image[..., 1]*(1.0-efn)).astype('uint8')
        final_img[..., 2] = (event_color[2]*(efn) + image[..., 2]*(1.0-efn)).astype('uint8')
        
        return final_img
    

    def blend_image_events_and_gt_flow(self, image, events, gt_flow, triple_blend=False, use_eval_events=False):
        # convert image to (H, W, 3)
        img = jnp_to_ocv_n255(image)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        # obtain encoded RGB image from  ground-truth flow (H, W, 3)
        gt_flow_img = flow_uv_to_img(gt_flow)
        
        # construct event frame
        if events is None:
            events  = self.datasample['eval_events'] if use_eval_events else self.datasample['events']   
        ev_frame_res = self.construct_event_frame(events)    
        ev_frame_norm = ev_frame_res['normalized_event_frame']
        ev_frame_mask = ev_frame_res['event_frame_mask']

        # make event_frame_norm the red channel of a new (H, W, 3) array
        red_ev_frame_norm = np.zeros_like(img)
        red_ev_frame_norm[..., 0] = ev_frame_norm

        # broadcast 1-channel event frame mask to 3-channels
        ev_frame_mask = np.broadcast_to(ev_frame_mask[..., np.newaxis], (*ev_frame_mask.shape, 3))
        
        # use 3-channel ev_frame_mask to add red_ev_frame_norm to gt_flow_img
        evt_gt = np.where(ev_frame_mask, red_ev_frame_norm, gt_flow_img)
        
        # embed flow_code
        evt_gt = embed_flow_code_in_img(evt_gt)
        if triple_blend:
            _event_alpha = 0.8
            evt_gt = blend_two_imgs(evt_gt, gt_flow_img, _event_alpha)

        # final blend
        _img_alpha = 0.35
        img_evt_gt_blend = blend_two_imgs(img, evt_gt, _img_alpha)

        return img_evt_gt_blend
    

    def get_event_flow_and_image_blend(self, event_flow, image, event_mask):
        if len(event_mask.shape) == 2:
            event_mask = np.broadcast_to(event_mask[..., np.newaxis], (*event_mask.shape, 3))
        
        event_flow_img = flow_uv_to_img(event_flow, embed_code=False)

        final_img = np.where(event_mask, event_flow_img, image)
        final_img = embed_flow_code_in_img(final_img)
        return final_img


    def construct_theta_nans_image(self, image, theta):
        # compute theta nan mask 
        theta_nan_mask = jnp.isnan(theta[..., 0]) | jnp.isnan(theta[..., 1])
        
        # make uint8 
        tnm = theta_nan_mask.astype(jnp.float64)
        tnm_norm = ((tnm - tnm.min()) / (tnm.max() - tnm.min() + sys.float_info.epsilon) * 255).astype('uint8')

        # create empty 3-channel and add nans to blue channel
        blue_tnm = jnp.stack(
            [
                jnp.zeros((tnm.shape), dtype='uint8'), 
                jnp.zeros((tnm.shape), dtype='uint8'), 
                tnm_norm
            ], axis=-1
        )
        
        # white blend img and make 3-channel
        img_255 = ((image - image.min()) / (image.max() - image.min() + sys.float_info.epsilon) * 255).astype('uint8')
        img_light = blend_two_imgs(np.array(image), np.ones_like(img_255) * 255, 0.413)
        img_light_3ch  = jnp.broadcast_to(img_light[...,jnp.newaxis], (*img_light.shape, 3))

        # 3-channel mask 
        tnm_3ch  = jnp.broadcast_to(tnm[...,jnp.newaxis], (*tnm.shape, 3))

        # use 3-channel mask to add blue_nan_frm to img_light_3ch
        final_img = np.where(tnm_3ch, blue_tnm, img_light_3ch)

        return final_img
    

    def construct_neg_grad_loss_image(self, grad_loss, scale=10):
        return flow_uv_to_img(-grad_loss * scale)


    def construct_gt_event_flow(self, gt_flow, events=None, use_eval_events=False):
        if events is None:
            events  = self.datasample['eval_events'] if use_eval_events else self.datasample['events'] 
        xs = events['x']
        ys = events['y']

        # round xs, ys to nearest integer
        xs, ys = jnp.round(xs).astype(int), jnp.round(ys).astype(int)

        # create event gt flow mask
        gt_flow_event_mask = jnp.zeros_like(gt_flow, dtype=jnp.bool_) # initialize (H, W, 2)
        gt_flow_event_mask = (gt_flow_event_mask
                              .at[ys, xs]
                              .set(True))

        # filter out gt_flow using ev_gt_flow_mask to obtain ev_gt_flow 
        gt_event_flow = jnp.where(gt_flow_event_mask, gt_flow, np.zeros_like(gt_flow))
        
        # convert flow to flow encoded image (u,v -> r,g,b)
        gt_event_flow_img = flow_uv_to_img(gt_event_flow)

        return gt_event_flow


    def plot_theta_nans_image(self, image, theta, 
                              save=False, show=True, save_format='png', path=None, idx='', itr='', pyr=''):
        if not (save or show): return 
        theta_nan_img = self.construct_theta_nans_image(image, theta)

        fig, axs = plt.subplots()
        axs.imshow(theta_nan_img)
        fig.canvas.manager.set_window_title(f'Theta NaNs i={itr}')
        # fig.canvas.manager.window.move(0, 0)
        axs.grid(which='minor', color='#EEF0F2', linewidth=1.25)
        axs.minorticks_on()
        axs.tick_params(which='minor', bottom=False, top=False)
        axs.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        axs.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        if save:
            plt.savefig(f'{path}/theta_nans_idx{idx}_pyr{pyr}_itr{itr}.{save_format.strip(".")}')
        
        if show:
            plt.show()
        else:
            plt.close()


    def plot_neg_grad_loss(self, grad_loss, 
                           save=False, show=True, save_format='png', path=None, idx='', itr='', pyr=''):
        if not (save or show): return 
        scale = max(jnp.median(1/jnp.linalg.norm(-grad_loss)),
                    jnp.mean(1/jnp.linalg.norm(-grad_loss)))

        fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
        # fig.canvas.manager.window.move(0, 0)
        fig.canvas.manager.set_window_title(f'Negative Gradient of Loss [idx={idx}, pyr={pyr}, itr={itr}]')
        fig.suptitle('Negative Gradient of Loss ' + r'$\left(-\nabla_{\theta} \mathcal{L}\right)$' + f' [idx={idx}, pyr={pyr}, itr={itr}]')

        im00 = axs[0, 0].imshow(-grad_loss[..., 0], cmap='jet')
        axs[0, 0].set_title('-grad_loss_x ' + r'$\left(-\nabla_{\theta_x} \mathcal{L}\right)$')
        axs[0, 0].grid(which='minor', color='#EEF0F2', linewidth=1.25)
        axs[0, 0].minorticks_on()
        axs[0, 0].tick_params(which='minor', bottom=False, top=False)
        axs[0, 0].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[0, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        axs[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        fig.colorbar(im00, cax=cax, orientation='vertical')

        im01 = axs[0, 1].imshow(-grad_loss[..., 1], cmap='jet')
        axs[0, 1].set_title('-grad_loss_y ' + r'$\left(-\nabla_{\theta_y} \mathcal{L}\right)$')
        axs[0, 1].grid(which='minor', color='#EEF0F2', linewidth=1.25)
        axs[0, 1].minorticks_on()
        axs[0, 1].tick_params(which='minor', bottom=False, top=False)
        axs[0, 1].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[0, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        axs[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        fig.colorbar(im01, cax=cax, orientation='vertical')

        axs[1, 0].imshow(self.construct_neg_grad_loss_image(grad_loss, scale=1))
        axs[1, 0].set_title('Flow image of (-grad_loss)')
        axs[1, 0].grid(which='minor', color='#EEF0F2', linewidth=1.25)
        axs[1, 0].minorticks_on()
        axs[1, 0].tick_params(which='minor', bottom=False, top=False)
        axs[1, 0].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[1, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        axs[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))

        axs[1, 1].imshow(self.construct_neg_grad_loss_image(grad_loss, scale=scale))
        axs[1, 1].set_title(f'Flow image of scaled (-grad_loss) [scale={scale:6.4f}]')
        axs[1, 1].grid(which='minor', color='#EEF0F2', linewidth=1.25)
        axs[1, 1].minorticks_on()
        axs[1, 1].tick_params(which='minor', bottom=False, top=False)
        axs[1, 1].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[1, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        axs[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))

        if save:
            plt.savefig(f'{path}/neg_grad_loss_idx{idx}_pyr{pyr}_itr{itr}.{save_format}')
        if show:
            plt.show()
        else:
            plt.close()

    
    def plot_handover(self, theta, prior_theta, post_handover_theta, alpha_ho=0.0, 
                      save=False, show=True, save_format='png', path=None, idx='', itr='', pyr='', ):
    
        if not (save or show): return
        fig, axs = plt.subplots(1, 3, figsize=(30,12))
        # fig.canvas.manager.window.move(0, 0)
        fig.canvas.manager.set_window_title(f'Upscaling Theta from pyr {pyr} to {pyr-1} [idx={idx}, pyr={pyr}, itr={itr}]')
        fig.suptitle(f'Upscaling Theta from pyr {pyr} to {max(0, pyr-1)} [idx={idx}, pyr={pyr}, itr={itr}]')

        axs[0].set_title(f'Pre-Handover Theta')
        axs[0] = self.draw_on_axs_imshow_theta(axs[0], theta)

        
        axs[1].set_title(f'Downscaled Preceding Theta')
        axs[1] = self.draw_on_axs_imshow_theta(axs[1], prior_theta)

        axs[2].set_title(f'Post-Handover Theta with alpha_handover={alpha_ho:4.2f}')
        axs[2] = self.draw_on_axs_imshow_theta(axs[2], post_handover_theta)

        if save:
            plt.savefig(f'{path}/upscale_theta_idx{idx}_pyr{pyr}_itr{itr}.{save_format}')
        if show:
            plt.show()
        else:
            plt.close()


    def draw_on_axs_imshow_theta(self, axs, theta):
        theta_flow_img = flow_uv_to_img(theta)
        axs = self.draw_on_axs_imshow_pixel_mesh(axs, theta_flow_img)
        return axs 
    

    def draw_on_axs_imshow_pixel_mesh(self, axs, img):
        axs.imshow(img)
        axs.grid(which='minor', color='#EEF0F2', linewidth=1.25)
        axs.minorticks_on()
        axs.tick_params(which='minor', bottom=False, top=False)
        axs.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        axs.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        axs.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        return axs
    

    def plot_step_results(self, 
                          theta_array, # (H, W, 2)
                          prev_theta_array, # (H, W, 2)
                          use_eval_events=True,
                          show=True, save=False, save_format='png', path=None, idx='', itr='', pyr=''):
    
        if not (save or show): return 
        
        idx = self.datasample_idx if idx == '' else idx
        itr = '--' if itr == '' else itr
        pyr = 0 if pyr == '' else pyr


        # choose between either the "eval events" or "the desired N events" for visualization
        events = self.datasample['eval_events'] if use_eval_events else self.datasample['events']

        # create vmapped functions for quicker computations
        vmapped_per_pix_warp = jax.vmap(per_pix_warp, (None, None, None, None, 0, None))
        vmapped_events_to_pdf_frame = jax.vmap(events_to_pdf_frame, (0, 0, None))
        vmapped_normalize_to_unit_range = jax.vmap(normalize_to_unit_range)

        # obtain warped events at multiple reference times
        t_refs = self.datasample['image_ts']
        multi_ref_warped_xs, multi_ref_warped_ys = vmapped_per_pix_warp(theta_array, 
                                                                        events['x'], 
                                                                        events['y'], 
                                                                        events['t'], 
                                                                        t_refs, 
                                                                        1.0) # (n_refs, n_events)
        
        # obtain normalized iwes for each reference time
        normalized_iwes = vmapped_normalize_to_unit_range(
            vmapped_events_to_pdf_frame(multi_ref_warped_xs, multi_ref_warped_ys, self.cfg.dataset.sensor_size)
        ) # (n_refs, H, W)

        # pick out the left (t_ref=0) and the right (t_ref=1) iwes
        l_normalized_iwe = normalized_iwes[0]   # (H, W)
        r_normalized_iwe = normalized_iwes[-1]  # (H, W)

        # warped events corresponding to t_ref=0 and t_ref=1
        l_warped_xs, l_warped_ys = multi_ref_warped_xs[0], multi_ref_warped_ys[0]     # (n_events,), (n_events)
        r_warped_xs, r_warped_ys = multi_ref_warped_xs[-1], multi_ref_warped_ys[-1]   # (n_events,), (n_events)

        l_in_array_mask = np.logical_and(
            np.logical_and(np.round(l_warped_xs) >= 0, np.round(l_warped_xs) < self.cfg.dataset.width), 
            np.logical_and(np.round(l_warped_ys) >= 0, np.round(l_warped_ys) < self.cfg.dataset.height)
        )

        l_filter_msg = (f'Filtered {np.sum(~l_in_array_mask):,} out-array events. '
                      + f'Keeping {np.sum(l_in_array_mask):,} in-array events out of total {len(l_in_array_mask):,} events.')
        print(l_filter_msg)

        # apply in-array mask to both orig and warped events
        l_warped_xs = l_warped_xs[l_in_array_mask]
        l_warped_ys = l_warped_ys[l_in_array_mask]
        l_warped_ts = events['t'][l_in_array_mask]
        l_warped_ps = events['p'][l_in_array_mask]
        for tup_elem in ['x', 'y', 't', 'p']:
            events[tup_elem] = events[tup_elem][l_in_array_mask]
        

        if 'gt_flow' in self.datasample and self.datasample['gt_flow'] is not None:
            gt_event_flow = self.construct_gt_event_flow(self.datasample['gt_flow'], use_eval_events=use_eval_events) # (H, W, 2)
        else:
            gt_event_flow = None
        
        pred_event_flow = per_pix_theta_to_flow(theta_array, 
                                                events['x'],
                                                events['y'],
                                                jnp.ones_like(events['t'])) # (H, W, 2)
        # Note: The estimated flow is to be compared with groundtruth which is provided on per-pixels basis.
        # Therefore, the theta to flow conversion above ignores the timestamp (setting all timestamps to 1.0, to indicate 
        # event motion from beginning to end of event volume) of events and only considers pixel location. 
        
        if 'gt_flow' in self.datasample and self.datasample['gt_flow'] is not None:
            gt_event_flow_1d = self.datasample['gt_flow'][
                jnp.round(events['y']).astype(int),
                jnp.round(events['x']).astype(int)
            ]
        else:
            gt_event_flow_1d = None

        dt = jnp.ones_like(events['t'])
        prev_pred_event_flow_1d = (
            prev_theta_array[jnp.round(events['y']).astype(int),
                             jnp.round(events['x']).astype(int),]
        ).astype(jnp.float64) * jnp.array([dt, dt]).T
        cur_pred_event_flow_1d = (
            theta_array[jnp.round(events['y']).astype(int),
                        jnp.round(events['x']).astype(int),]
        ).astype(jnp.float64) * jnp.array([dt, dt]).T


        fig, axs = plt.subplots(2,4, figsize=(40,16))
        # fig.canvas.manager.window.move(0, 0)
        fig.canvas.manager.set_window_title(f'Visualize Optimization Step Results [idx={self.datasample_idx}, pyr={pyr}, itr={itr}]')
        fig.suptitle(f'{self.cfg.sequence_name} Optimization Step Results [idx={self.datasample_idx}, pyr={pyr}, itr={itr}]')

        if gt_event_flow_1d is not None:
            axs[0,0].scatter(gt_event_flow_1d[:, 0], gt_event_flow_1d[:,1], label='Groundtruth Event Flow',
                             edgecolor='forestgreen', facecolor='forestgreen', marker='H', lw=0, s=21, alpha=0.4)
        axs[0,0].scatter(prev_pred_event_flow_1d[:,0], prev_pred_event_flow_1d[:,1], label='Predicted Event Flow (Prev.)', 
                         edgecolor='cornflowerblue', facecolor='cornflowerblue', marker='H', lw=0, s=27, alpha=0.35)
        axs[0,0].scatter(cur_pred_event_flow_1d[:,0], cur_pred_event_flow_1d[:,1], label='Predicted Event Flow (Curr.)', 
                         edgecolor='crimson', facecolor='crimson', marker='H', lw=0, s=21, alpha=0.25)
        axs[0,0].grid(True)
        axs[0,0].axis('equal')
        axs[0,0].legend(markerscale=6)
        axs[0,0].set_title('Event Flow Distribution Trend')

        if gt_event_flow is not None:
            axs[1,0].imshow(flow_uv_to_img(gt_event_flow))
        else:
            axs[1,0].annotate(f'N/A', (0.5,0.5), transform=axs[1,0].transAxes,
                              ha='center', va='center', fontsize=75, color='darkgray')
        axs[1,0].set_title('Groundtruth Event Flow')

        axs[0,1].imshow(flow_uv_to_img(theta_array))
        axs[0,1].set_title('Estimated 2D Velocity Field (Theta)')
        
        axs[1,1].imshow(flow_uv_to_img(pred_event_flow))
        axs[1,1].set_title('Predicted Event Flow')
        
        im02 = axs[0,2].imshow(l_normalized_iwe**0.382, cmap='binary')
        axs[0,2].set_title(r'Image of Warped Events at t$_0$')
        divider = make_axes_locatable(axs[0, 2])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        fig.colorbar(im02, cax=cax, orientation='vertical')
        
        im12 = axs[1,2].imshow(r_normalized_iwe**0.382, cmap='binary')
        axs[1,2].set_title(r'Image of Warped Events at t$_1$')
        divider = make_axes_locatable(axs[1, 2])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        fig.colorbar(im12, cax=cax, orientation='vertical')

        im03 = axs[0,3].imshow(self.datasample['edges'][0], cmap='binary')
        axs[0,3].set_title(r'Edge at t$_0$')
        divider = make_axes_locatable(axs[0, 3])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        fig.colorbar(im03, cax=cax, orientation='vertical')
        
        im13 = axs[1,3].imshow(self.datasample['edges'][-1], cmap='binary')
        axs[1,3].set_title(r'Edge at t$_1$')
        divider = make_axes_locatable(axs[1, 3])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        fig.colorbar(im13, cax=cax, orientation='vertical')
        
        if save:
            plt.savefig(f'{path}/plot_step_result_idx{self.datasample_idx}_pyr{pyr}_itr{itr}.{save_format}')
        if show:
            plt.show()
        else:
            plt.close()


    def plot_end_results(self,
                         theta_array, 
                         events=None, 
                         use_eval_events=True,
                         draw_events_every=1,
                         event_color=np.array([213, 0, 50]),
                         show=True, save=False, save_format='png', path=None, idx='', add_title=''):
        if not (save or show): return 

        if idx == '':
            idx = self.datasample_idx

        if events is None:
            events = self.datasample['eval_events'] if use_eval_events else self.datasample['events']

        # import pdb; pdb.set_trace()
        # obtain warped events at multiple reference times
        vmapped_per_pix_warp = jax.vmap(per_pix_warp, (None, None, None, None, 0, None))
        t_refs = self.datasample['image_ts']
        multi_ref_warped_xs, multi_ref_warped_ys = vmapped_per_pix_warp(theta_array, 
                                                                        events['x'], 
                                                                        events['y'], 
                                                                        events['t'], 
                                                                        t_refs, 
                                                                        1.0) # (n_refs, n_events)
        

        # warped events corresponding to t_ref=0 and t_ref=1
        l_warped_xs, l_warped_ys = multi_ref_warped_xs[0], multi_ref_warped_ys[0]
        r_warped_xs, r_warped_ys = multi_ref_warped_xs[-1], multi_ref_warped_ys[-1]

        l_in_array_mask = np.logical_and(
            np.logical_and(np.round(l_warped_xs) >= 0, np.round(l_warped_xs) < self.cfg.dataset.width), 
            np.logical_and(np.round(l_warped_ys) >= 0, np.round(l_warped_ys) < self.cfg.dataset.height)
        )

        l_filter_msg = (f'Filtered {np.sum(~l_in_array_mask):,} out-array events. '
                      + f'Keeping {np.sum(l_in_array_mask):,} in-array events out of total {len(l_in_array_mask):,} events.')
        # print(l_filter_msg)

        # filter for in-array
        l_warped_xs = l_warped_xs[l_in_array_mask]
        l_warped_ys = l_warped_ys[l_in_array_mask]
        l_warped_ts = events['t'][l_in_array_mask]
        l_warped_ps = events['p'][l_in_array_mask]
        for tup_elem in ['x', 'y', 't', 'p']:
            events[tup_elem] = events[tup_elem][l_in_array_mask]
        l_warped_events = {
            'x': l_warped_xs,
            'y': l_warped_ys,
            't': l_warped_ts,
            'p': l_warped_ps,
        }
        pred_event_flow = per_pix_theta_to_flow(theta_array, 
                                                events['x'],
                                                events['y'],
                                                jnp.ones_like(events['t'])) # (H, W, 2)
        # Note: The estimated flow is to be compared with groundtruth which is provided on per-pixels basis.
        # Therefore, the theta to flow conversion above ignores the timestamp (setting all timestamps to 1.0, to indicate 
        # event motion from beginning to end of event volume) of events and only considers pixel location. 
        
        

        # show gt comparison, event_image blend comparison, scatter plot of warped vs non-warped
        # image_left could be jax and float64, make it numpy float32/uint8 to work in opencv
        # l_img_ = l_img.copy()
        l_img = self.datasample['images'][0]
        l_img = np.array(l_img).astype('float32')
        l_img_255 = cv.normalize(l_img, None, 0, 255, norm_type=cv.NORM_MINMAX).astype('uint8')
        l_img_255_3ch = cv.cvtColor(np.array(l_img_255), cv.COLOR_GRAY2RGB)
        l_img_255_3ch_whitened = blend_two_imgs(l_img_255_3ch, np.ones_like(l_img_255_3ch)*255, 0.5)

        ev_frame_res_bef = self.construct_event_frame(events)
        ev_frame_norm_bef = ev_frame_res_bef['normalized_event_frame']
        ev_frame_mask_bef = ev_frame_res_bef['event_frame_mask']
        ev_frame_res_aft = self.construct_event_frame(l_warped_events)
        ev_frame_norm_aft = ev_frame_res_aft['normalized_event_frame']
        ev_frame_mask_aft = ev_frame_res_aft['event_frame_mask']

        if 'gt_flow' in self.datasample and self.datasample['gt_flow'] is not None:
            gt_flow = self.datasample['gt_flow']
            gt_mask = jnp.logical_and(
                jnp.logical_and(~jnp.isinf(gt_flow[..., 0]), 
                                ~jnp.isinf(gt_flow[..., 1])),
                jnp.linalg.norm(gt_flow, axis=-1) > 0
            ) # (H, W)
            gt_event_mask = jnp.logical_and(gt_mask, ev_frame_mask_bef)
            gt_flow_img = flow_uv_to_img(gt_flow)
            gt_event_flow = self.construct_gt_event_flow(gt_flow, events=events)
            gt_event_flow_img_blend = self.get_event_flow_and_image_blend(gt_event_flow, l_img_255_3ch_whitened, gt_event_mask)
        else:
            gt_flow = None
            gt_mask = None
            gt_event_mask = None
            gt_flow_img = None
            gt_event_flow = None
            gt_event_flow_img_blend = None

        ize = self.overlay_events_on_image(events, np.ones_like(l_img_255_3ch)*255, event_color=event_color)
        l_edge = self.datasample['edges'][0]
        pred_event_flow_img_blend = self.get_event_flow_and_image_blend(pred_event_flow, l_img_255_3ch_whitened, ev_frame_mask_bef)
        orig_events_image_overlay = self.overlay_events_on_image(events, l_img_255_3ch_whitened, event_color=event_color)
        warped_events_image_overlay = self.overlay_events_on_image(l_warped_events, l_img_255_3ch_whitened, event_color=event_color)
       
        # --------------------------------------------------------------------------------
        # plot
        fig, axs = plt.subplots(3, 4, figsize=(31, 19))
        # fig.canvas.manager.window.move(0, 0)
        fig.canvas.manager.set_window_title(f'Visualize Optimization End Results [idx={idx}] {add_title}')
        fig.suptitle(f'{self.cfg.sequence_name} Optimization End Results [idx={idx}] {add_title}')

        if gt_flow is not None:
            axs[0,0].imshow(gt_flow_img)
        else:
            axs[0,0].annotate(f'N/A', (0.5,0.5), transform=axs[0,0].transAxes,
                            ha='center', va='center', fontsize=75, color='darkgray')
        axs[0,0].set_title('GT Flow')
        
        axs[0,1].imshow(ize)
        axs[0,1].set_title('Image of Zero-Warped Events')
        
        axs[0,2].imshow(l_img_255, cmap='gray')
        axs[0,2].set_title(r'Image at t$_0$')
        
        im03 = axs[0,3].imshow(l_edge, cmap='binary')
        axs[0,3].set_title(r'Edge at t$_0$')
        divider = make_axes_locatable(axs[0, 3])
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        fig.colorbar(im03, cax=cax, orientation='vertical')

        if gt_flow is not None:
            axs[1,0].imshow(gt_event_flow_img_blend)
            axs[2,0].imshow(pred_event_flow_img_blend)
        else:
            axs[1,0].annotate(f'N/A', (0.5,0.5), transform=axs[1,0].transAxes,
                            ha='center', va='center', fontsize=75, color='darkgray')
            axs[2,0].annotate(f'N/A', (0.5,0.5), transform=axs[2,0].transAxes,
                            ha='center', va='center', fontsize=75, color='darkgray')
        axs[1,0].set_title('GT Event Flow and Image Blend (before)')
        axs[2,0].set_title('Predicted Event Flow and Image Blend (after)')

        axs[1,1].imshow(orig_events_image_overlay)
        axs[1,1].set_title('Events+Image Overlay (before)')
        axs[2,1].imshow(warped_events_image_overlay)
        axs[2,1].set_title('Events+Image Overlay (after)')
        
        if gt_flow is not None:
            axs[1,2].imshow(flow_uv_to_img(gt_event_flow))
        else:
            axs[1,2].annotate(f'N/A', (0.5,0.5), transform=axs[1,2].transAxes,
                            ha='center', va='center', fontsize=75, color='darkgray')
        axs[1,2].set_title('Estimated GT Event Flow')
        axs[2,2].imshow(flow_uv_to_img(pred_event_flow))
        axs[2,2].set_title('Predicted Event Flow')
        
        # apply selection mask on both orig and warped events to cut down scatter plot render time
        selection_mask = np.arange(len(l_warped_xs))%draw_events_every == 0
        l_warped_xs = l_warped_xs[selection_mask]
        l_warped_ys = l_warped_ys[selection_mask]
        l_warped_ts = l_warped_ts[selection_mask]
        l_warped_ps = l_warped_ps[selection_mask]
        for tup_elem in ['x', 'y', 't', 'p']:
            events[tup_elem] = events[tup_elem][selection_mask]

        l_warped_events = {
            'x': l_warped_xs,
            'y': l_warped_ys,
            't': l_warped_ts,
            'p': l_warped_ps,
        }
        if draw_events_every > 1:
            l_select_msg = (f'For scatter plots, using every {draw_events_every} events. '
                        + f'Selecting {np.sum(selection_mask):,} out of total {len(selection_mask):,} events.')
            # print(l_select_msg)

        axs[1,3].invert_yaxis()
        axs[1,3].scatter(events['x'], events['y'], marker='o', lw=0, alpha=0.2,
                        s=2, facecolor='black', edgecolors='none')
        axs[1,3].set_title(r'Scatter Plot of Zero-Warped Events at t$_0$ (before)')
        axs[1,3].set(xlim=(0, l_img.shape[1]-1), ylim=(0, l_img.shape[0]-1))
        axs[1,3].invert_yaxis()
        axs[1,3].set_aspect('equal', 'box')
        
        axs[2,3].scatter(l_warped_events['x'], l_warped_events['y'], marker='o', lw=0, alpha=0.2, 
                        s=2, facecolor='black', edgecolors='none')
        axs[2,3].set_title(r'Scatter Plot of Warped Events at t$_0$ (after)')
        axs[2,3].set(xlim=(0, l_img.shape[1]-1), ylim=(0, l_img.shape[0]-1))
        axs[2,3].invert_yaxis()
        axs[2,3].set_aspect('equal', 'box')

        fig.tight_layout()
        if save:
            plt.savefig(f'{path}/plot_end_result_idx_{idx}.{save_format}')
            if 0:
                plt.savefig(f'{path}/gt_flow_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[0,0]))
                plt.savefig(f'{path}/iue_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[0,1]))
                plt.savefig(f'{path}/img0_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[0,2]))
                plt.savefig(f'{path}/edge0_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[0,3]))
                plt.savefig(f'{path}/gt_event_flow_img_blend_bef_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[1,0]))
                plt.savefig(f'{path}/events_img_overlay_bef_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[1,1]))
                plt.savefig(f'{path}/gt_event_flow_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[1,2]))
                plt.savefig(f'{path}/iwe0_bef_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[1,3]))
                plt.savefig(f'{path}/gt_event_flow_img_blend_aft_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[2,0]))
                plt.savefig(f'{path}/events_img_overlay_aft_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[2,1]))
                plt.savefig(f'{path}/pred_event_flow_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[2,2]))
                plt.savefig(f'{path}/iwe0_aft_idx_{idx}.{save_format}', bbox_inches=self.get_axes_bbox_inches(axs[2,3]))
        if show:
            plt.show()
        else:
            plt.close()
        

    def plot_fwls(self, fwls, path=None, show=True, save=False, save_format='pdf'):
        fig, axs = plt.subplots(figsize=(20, 4))

        if (str(self.cfg.sequence_name).split('_')[0] in ['interlaken', 'thun', 'zurich']
            and self.cfg.dataset.loader.extended == True):
            axs.plot(fwls, 
                     'o--', lw=2, ms=8, color='gainsboro',
                     alpha=0.6, label=f"FWL (extd): {fwls.mean():.2f} +/- {fwls.std():.2f}")
            axs.plot(np.array(range(len(fwls)))[::5][1:], fwls[::5][1:], 
                     'o--', lw=1.5, ms=6, color='forestgreen',
                     alpha=0.9, label=f"FWL (orig): {fwls[::5][1:].mean():.2f} +/- {fwls[::5][1:].std():.2f}")
        else:
            axs.plot(fwls, 
                     'o--', lw=1.5, ms=6, color='forestgreen',
                     alpha=0.9, label=f"FWL: {fwls.mean():.2f} +/- {fwls.std():.2f}")
        
        axs.set_title('Flow Warp Losses')
        axs.set_xlabel('Datasample Indices')
        axs.set_ylabel('FWL scores')
        axs.xaxis.set_major_locator(ticker.MultipleLocator(base=int(len(fwls)/10)))
        axs.set(xlim=(0, len(fwls)-1))
        axs.grid(True)
        axs.legend()

        fig.tight_layout()

        if save:
            plt.savefig(f'{path}/fwls.{save_format}')
        if show:
            plt.show()
        else:
            plt.close()


    def plot_aees(self, aees, path=None, show=True, save=False, save_format='pdf'):
        fig, axs = plt.subplots(figsize=(20, 4))

        axs.plot(aees, 'o--', lw=1.5, ms=6, color='crimson', alpha=0.8, label=f"AEE: {aees.mean():.2f} +/- {aees.std():.2f}")
        
        axs.set_title('Average Endpoint Errors')
        axs.set_xlabel('Datasample Indices')
        axs.set_ylabel('AEE scores')
        axs.xaxis.set_major_locator(ticker.MultipleLocator(base=int(len(aees)/10)))
        axs.set(xlim=(0, len(aees)-1))
        axs.grid(True)
        axs.legend()

        fig.tight_layout()

        if save:
            plt.savefig(f'{path}/aees.{save_format}')
        if show:
            plt.show()
        else:
            plt.close()


    def plot_anpes(self, scores, path=None, show=True, save=False, save_format='pdf'):
        sorted_sample_idxs = [i for i in range(len(scores['A1PE']))]

        cp = [
            '#72CE27',
            '#95E214',
            '#B8F500',
            '#FFF700',
            '#FFE000',
            '#FFC800',
            '#FF9900',
        ]

        fill_alpha=1
        plot_alpha=0.8

        fig, axs = plt.subplots(figsize=(20, 5))

        axs.fill_between(sorted_sample_idxs, 0, 100-scores['A1PE'], label='0 '.ljust(2, ' ') + r' $\leq$ EE $<$ 1 ', 
                         color=cp[0], alpha=fill_alpha, zorder=0)
        axs.fill_between(sorted_sample_idxs, 100-scores['A1PE'],  100-scores['A2PE'], label='1 '.ljust(2, ' ') + r' $\leq$ EE $<$ 2 ',
                         color=cp[1], alpha=fill_alpha, zorder=-2)
        axs.fill_between(sorted_sample_idxs, 100-scores['A2PE'],  100-scores['A3PE'], label='2 '.ljust(2, ' ') + r' $\leq$ EE $<$ 3 ',
                         color=cp[2], alpha=fill_alpha, zorder=-4)
        axs.fill_between(sorted_sample_idxs, 100-scores['A3PE'],  100-scores['A5PE'], label='3 '.ljust(2, ' ') + r' $\leq$ EE $<$ 5 ',
                         color=cp[3], alpha=fill_alpha, zorder=-6)
        axs.fill_between(sorted_sample_idxs, 100-scores['A5PE'],  100-scores['A10PE'], label='5 '.ljust(2, ' ') + r' $\leq$ EE $<$ 10',
                         color=cp[4], alpha=fill_alpha, zorder=-8)
        axs.fill_between(sorted_sample_idxs, 100-scores['A10PE'], 100-scores['A20PE'], label='10'.ljust(2, ' ') + r' $\leq$ EE $<$ 20',
                         color=cp[5], alpha=fill_alpha, zorder=-10)
        axs.fill_between(sorted_sample_idxs, 100-scores['A20PE'], 100, label='20'.ljust(2, ' ') + r' $\leq$ EE $<$ $\infty$',
                         color=cp[6], alpha=fill_alpha, zorder=-12)
        
        axs.plot(sorted_sample_idxs, 100-scores['A1PE'], color=cp[0], lw=1.25, alpha=plot_alpha, zorder=-1)
        axs.plot(sorted_sample_idxs, 100-scores['A2PE'], color=cp[1], lw=1.25, alpha=plot_alpha, zorder=-3)
        axs.plot(sorted_sample_idxs, 100-scores['A3PE'], color=cp[2], lw=1.25, alpha=plot_alpha, zorder=-5)
        axs.plot(sorted_sample_idxs, 100-scores['A5PE'], color=cp[3], lw=1.25, alpha=plot_alpha, zorder=-7)
        axs.plot(sorted_sample_idxs, 100-scores['A10PE'], color=cp[4], lw=1.25, alpha=plot_alpha, zorder=-9)
        axs.plot(sorted_sample_idxs, 100-scores['A20PE'], color=cp[5], lw=1.25, alpha=plot_alpha, zorder=-11)

        axs.xaxis.set_major_locator(ticker.MultipleLocator(base=int(len(sorted_sample_idxs)/10)))
        axs.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
        axs.set_yticklabels([])
        axs.tick_params(axis='y', length=0)
        axs.set(xlim=(0, len(sorted_sample_idxs)-1), ylim=(100.0, 0.0))
        axs.set_title('Average N-Pixel Error Percentage Distribution')
        axs.set_xlabel('Datasample Indices')
        axs.set_ylabel(r'$\longleftarrow$ 100% $\longrightarrow$')
        axs.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        axs.grid(True)
        fig.tight_layout()

        if save:
            plt.savefig(f'{path}/anpes.{save_format}')
        if show:
            plt.show()
        else:
            plt.close()

                    
    def get_axes_bbox_inches(self, axs, expand_factor=1.025):

        fig = axs.get_figure()

        # get tight bounding box in axes coordinates
        tight_bbox_axcoords = axs.get_tightbbox(fig.canvas.get_renderer())

        # transform to inches
        tight_bbox_inches = transforms.TransformedBbox(tight_bbox_axcoords, transforms.Affine2D().scale(1.0/fig.dpi))

        # expand bbox by expand_factor accounting for width and height inversely
        expand_factor_width = 1 + (tight_bbox_inches.height / fig.get_size_inches()[1]) * (expand_factor - 1.0)
        expand_factor_height = 1 + (tight_bbox_inches.width / fig.get_size_inches()[0]) * (expand_factor - 1.0)
        expanded_bbox_inches = tight_bbox_inches.expanded(expand_factor_width, expand_factor_height)

        return expanded_bbox_inches



