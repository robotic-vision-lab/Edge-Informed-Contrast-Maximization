import os
import shutil
import time
import sys
from functools import partial
from pathlib import Path
from tqdm import tqdm

import cv2 as cv
import hydra
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from eincm.callbacks import EmptyHandoverSolverCallback
from eincm.callbacks import EmptyThetaSolverCallback
from eincm.callbacks import EINCMHandoverSolverCallback
from eincm.callbacks import EINCMThetaSolverCallback
from eincm.solver import MultipleLevelEINCMSolver
from evaluations.theta_eval import evaluate_theta_array
from utils.event_utils import make_event_mask
from utils.img_utils import jnp_to_ocv_n255
from utils.img_utils import normalize_to_unit_range
from utils.theta_utils import scale_theta_to_sensor_size
from .outputs_loader import EINCMOutputLoader
from .plotters import EINCMExperimentPlotter



class EINCMExperiment:
    """Duties:
        - Manage Callback
            - create
            - update datasample

        - Manage EINCMSolver
    """
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.prepare_output_dirs()
        self.prepare_partial_callables()
        self.prepare_callbacks()
        self.prepare_maxiters()
        self.prepare_solver()

        self.opt_results = {}
        self.eval_results = {}
        self.scores = {}

        self._LOADED_CHECKPOINT = False
        self.ckpt_idx = -1

        self.output_loader = EINCMOutputLoader()


    def prepare_output_dirs(self):
        self.out_dir = None
        self.checkpoint_dir = None
        self.plot_dir = None
        self.plot_end_results_dir = None
        self.plot_evals_dir = None

        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        if self.cfg.experiment_settings.solver.enable and self.cfg.experiment_settings.solver.checkpoints.enable:
            self.checkpoint_dir = self.out_dir / 'checkpoints'
            if not self.checkpoint_dir.exists():
                os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.cfg.experiment_settings.plot.enable:
            self.plot_dir = self.out_dir / 'plots'
            if not self.plot_dir.exists():
                os.makedirs(self.plot_dir, exist_ok=True)
            if self.cfg.experiment_settings.plot.end_result.save:
                self.plot_end_results_dir = self.plot_dir / 'end_results'
                if not self.plot_end_results_dir.exists():
                    os.makedirs(self.plot_end_results_dir, exist_ok=True)
            if self.cfg.experiment_settings.plot.evals.save:
                self.plot_evals_dir = self.plot_dir / 'evals'
                if not self.plot_evals_dir.exists():
                    os.makedirs(self.plot_evals_dir, exist_ok=True)


    def prepare_partial_callables(self):
        """Callables to prepare:
            - evaluate_theta
            - scale_theta_to_sensor_size
            - preprocess_image
            - image_to_edge
            - smoothen_edges
            - c2max_theta_loss
            - c2max_handover_loss
        """
        #-------------------------------------------------------------------------------------------------------------#
        #                                      edge extraction partial functions                                      #
        #-------------------------------------------------------------------------------------------------------------#
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

        #-------------------------------------------------------------------------------------------------------------#
        #                                           loss partial functions                                            #
        #-------------------------------------------------------------------------------------------------------------#
        # func:           eincm.losses.loss_func
        # remaining args: theta, xs, ys, ts, edges, edge_ts, cur_pyr_lvl
        # injected args:  alpha, beta, gamma, delta, n_pyr_lvls, sensor_size
        self.c2max_theta_loss_pfunc = hydra.utils.instantiate(self.cfg.theta_loss_func)

        # func:           eincm.losses.handover_loss_func
        # remaining args: alpha_handover, prev_theta, theta, xs, ys, ts, edges, edge_ts, cur_pyr_lvl
        # injected args:  alpha, beta, gamma, delta, n_pyr_lvls, sensor_size
        self.c2max_handover_loss_pfunc = hydra.utils.instantiate(self.cfg.handover_loss_func)

        #-------------------------------------------------------------------------------------------------------------#
        #                                      theta evaluation partial functions                                     #
        #-------------------------------------------------------------------------------------------------------------#
        # remaining args: theta, xs, ys, ts, edges, edge_ts, gt_flow
        self.evaluate_theta_array_pfunc = partial(
            evaluate_theta_array, 
            alpha=self.cfg.alpha, beta=self.cfg.beta, gamma=self.cfg.gamma, delta=self.cfg.delta,
            sensor_size=self.cfg.dataset.sensor_size
        )
        
        # remaining args: theta, cur_pyr_lvl
        self.scale_theta_to_sensor_size_pfunc = partial(
            scale_theta_to_sensor_size,
            sensor_size=self.cfg.dataset.sensor_size,
            method=self.cfg.scale_theta_to_sensor_size_method
        )
             

    def prepare_callbacks(self):
        # create callback object instance
        if self.cfg.callback_options.theta_opt.enable:
            self.theta_opt_callback = EINCMThetaSolverCallback(n_pyr_lvls=self.cfg.n_pyr_lvls,
                                                               scale_theta_to_sensor_size_func=self.scale_theta_to_sensor_size_pfunc,
                                                               evaluate_theta_func=self.evaluate_theta_array_pfunc,
                                                               callback_options=self.cfg.callback_options.theta_opt)
        else:
            self.theta_opt_callback = EmptyThetaSolverCallback()

        if self.cfg.callback_options.handover_opt.enable:
            self.handover_opt_callback = EINCMHandoverSolverCallback(n_pyr_lvls=self.cfg.n_pyr_lvls, 
                                                                     scale_theta_to_sensor_size_func=self.scale_theta_to_sensor_size_pfunc,
                                                                     evaluate_theta_func=self.evaluate_theta_array_pfunc,
                                                                     callback_options=self.cfg.callback_options.handover_opt)
        else:
            self.handover_opt_callback = EmptyHandoverSolverCallback()


    def prepare_maxiters(self):
        self.theta_opt_maxiters, self.handover_opt_maxiters = {}, {}

        theta_miniter = self.cfg.solver_params.theta_opt.miniter
        theta_maxiter = self.cfg.solver_params.theta_opt.maxiter
        handover_miniter = self.cfg.solver_params.handover_opt.miniter
        handover_maxiter = self.cfg.solver_params.handover_opt.maxiter
        
        for pyr_lvl in range(self.cfg.n_pyr_lvls):
            key = f'pyr_lvl_{pyr_lvl}'
            p = pyr_lvl / (self.cfg.n_pyr_lvls-1)
            ordr = self.cfg.maxiters_grow_order
            # construct maxiters
            if self.cfg.use_growing_maxiters:
                self.theta_opt_maxiters[key] = int(np.ceil(theta_miniter*p**ordr + theta_maxiter*(1-p)**ordr)) 
                self.handover_opt_maxiters[key] = int(np.ceil(handover_miniter*p**ordr + handover_maxiter*(1-p)**ordr))
            else:
                self.theta_opt_maxiters[key] = self.cfg.solver_params.theta_opt.maxiter
                self.handover_opt_maxiters[key] = self.cfg.solver_params.handover_opt.maxiter
            
    
    def prepare_solver(self):
        multi_level_solver_kwargs = {
            'n_pyr_lvls':                 self.cfg.n_pyr_lvls,
            'theta_opt_maxiters':         self.theta_opt_maxiters,
            'theta_loss_pfunc':           self.c2max_theta_loss_pfunc,
            'theta_opt_solver_params':    self.cfg.solver_params.theta_opt,
            'handover_opt_maxiters':      self.handover_opt_maxiters,
            'handover_loss_pfunc':        self.c2max_handover_loss_pfunc,
            'handover_opt_solver_params': self.cfg.solver_params.handover_opt,
            'handover_settings':          self.cfg.handover_settings,
            'pyramid_downscale_method':   self.cfg.pyramid_downscale_method,
            'pyramid_upscale_method':     self.cfg.pyramid_upscale_method,
            'pyramid_bases':              list(self.cfg.pyramid_bases),
            'theta_solver_callback':      self.theta_opt_callback,
            'handover_solver_callback':   self.handover_opt_callback
        }
        self.multi_level_solver = MultipleLevelEINCMSolver(**multi_level_solver_kwargs)

    
    def prepare_dataloader(self):
        # func:           dataloaders.<dset>_loader.<DSET>DataLoader
        # remaining args: root_dir, sequence_name
        # import pdb; pdb.set_trace()
        self.dataloader = hydra.utils.instantiate(self.cfg.dataset.loader)
        self.dataloader.get_ready()


    def prepare_plotter(self):
        self.plotter = EINCMExperimentPlotter(self.cfg)


    def _prerun(self):
        self.exp_begin_time = None
        
        self.prepare_dataloader()
        self.total_datasamples = len(self.dataloader)
        
        # handle loading from checkpoint case
        if (self.cfg.experiment_settings.solver.enable
            and self.cfg.experiment_settings.solver.run_from_checkpoint
            and self._LOADED_CHECKPOINT is False):
            checkpoint_path = Path(self.cfg.experiment_settings.solver.run_from_checkpoint_path)
            assert checkpoint_path.exists(), f'Checkpoint path {checkpoint_path} does not exist!'

            # load opt_results
            opt_results = np.load(checkpoint_path, allow_pickle=True)
            self.opt_results = opt_results['opt_results'].item()
            loaded_cfg = opt_results['cfg'].item()
            self.cfg = loaded_cfg.copy()

            # update solver prior theta
            sort_key = lambda k: int(k.replace('datasample_idx_', ''))
            self.ckpt_idx = int((sorted(self.opt_results.keys(), key=sort_key)[-1]).replace('datasample_idx_', ''))
            last_final_theta_pyr = self.opt_results[f'datasample_idx_{self.ckpt_idx}']['solver_final_results']['final_theta_pyr']
            self.multi_level_solver.prior_theta_pyr = last_final_theta_pyr.copy()
            self.multi_level_solver.not_first_sample()
            self._LOADED_CHECKPOINT = True

        if self.cfg.experiment_settings.plot.enable: 
            self.prepare_plotter()


    def _skip_datasample_idx(self, datasample_idx):
        if self.cfg.experiment_settings.solver.run_from_checkpoint:
            if datasample_idx <= self.ckpt_idx:
                return True
            
        if not self.cfg.run_full_sequence:
            if not (self.cfg.run_idx_range[0] <= datasample_idx < self.cfg.run_idx_range[1]):
                return True
            if (self.dataloader.sequence_name == 'outdoor_day1'
                and self.cfg.outdoor_day1_run_idx_range.type == 'continuous'
                and not (self.cfg.outdoor_day1_run_idx_range.start <= datasample_idx < self.cfg.outdoor_day1_run_idx_range.end)):
                return True
            if (self.dataloader.sequence_name == 'outdoor_day1' 
                and self.cfg.outdoor_day1_run_idx_range.type == 'split' 
                and not (self.cfg.outdoor_day1_run_idx_range.start_1 <= datasample_idx < self.cfg.outdoor_day1_run_idx_range.end_1
                     or self.cfg.outdoor_day1_run_idx_range.start_2 <= datasample_idx < self.cfg.outdoor_day1_run_idx_range.end_2)):
                return True
        
        return False


    def _set_exp_begin_time(self):
        if self.exp_begin_time is None:
            self.exp_begin_time = time.perf_counter()


    def stage_datasample(self, datasample):
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
        if self.dataloader.n_event_deficiency > 0: # event scarcity (optimization using superset)
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
        t_ref      = 0.0 # start
        ts_units = 'us' if 'eval_ts_us' in datasample else 's'


        # --------------------------------------------------------------------------------------------------------#
        #                                             EDGE EXTRACTION                                             #
        # --------------------------------------------------------------------------------------------------------#
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
    

    def feed_staged_sample(self, datasample_idx, staged_sample):
        datasample_kwargs = {
            'xs':      staged_sample['events']['x'], 
            'ys':      staged_sample['events']['y'], 
            'ts':      staged_sample['events']['t'], 
            'edges':   staged_sample['edges'], 
            'edge_ts': staged_sample['image_ts'],
        }
        # feed datasample without groundtruth to solver 
        self.multi_level_solver.set_datasample(**datasample_kwargs)

        # feed datasample with groundtruth (used for eval) to solver callback functions
        self.theta_opt_callback.set_datasample(**datasample_kwargs, gt_flow=staged_sample['gt_flow'])
        self.handover_opt_callback.set_datasample(**datasample_kwargs, gt_flow=staged_sample['gt_flow'])
        
        # feed staged sample to plotter
        if self.cfg.experiment_settings.plot.enable:
            self.plotter.update_datasample(datasample_idx, staged_sample, is_staged=True)


    def fetch_collected_optimization_intermediate_results(self):
        self.intermediate_results = {'theta_opt': {}, 'handover_opt': {}}

        # collect number of iterations from both theta and handover optimizations
        self.intermediate_results['theta_opt']['n_iters'] = self.theta_opt_callback.get_iters()
        self.intermediate_results['handover_opt']['n_iters'] = self.handover_opt_callback.get_iters()
        
        # collect intermediate thetas and losses from theta optimization
        if self.cfg.experiment_settings.solver.intermediate_result_collection.theta_opt.thetas:
            self.intermediate_results['theta_opt']['thetas'] = self.theta_opt_callback.get_thetas()
        if self.cfg.experiment_settings.solver.intermediate_result_collection.theta_opt.losses:
            self.intermediate_results['theta_opt']['losses'] = self.theta_opt_callback.get_losses()

        # collect intermediate handover weights, losses, and thetas from handover optimization
        if self.cfg.experiment_settings.solver.intermediate_result_collection.handover_opt.handover_weights:
            self.intermediate_results['handover_opt']['handover_weights'] = self.handover_opt_callback.get_handover_weights()
        if self.cfg.experiment_settings.solver.intermediate_result_collection.handover_opt.losses:
            self.intermediate_results['handover_opt']['losses'] = self.handover_opt_callback.get_losses()
        if self.cfg.experiment_settings.solver.intermediate_result_collection.handover_opt.thetas:
            self.intermediate_results['handover_opt']['thetas'] = self.handover_opt_callback.get_thetas()


    def evaluate_opt_results_at_datasample_idx_and_collect(self, datasample_idx, staged_sample):
        sample_key = f'datasample_idx_{datasample_idx}'
        if not sample_key in self.opt_results:
            print(f'Key {sample_key} not found in opt_results')
            return None
        
        # for outdoor_day1 evaluations, set event mask to ignore lower pixel rows containing car
        # this is to compare other works which followed the same process while reporting their metrics 
        if self.cfg.sequence_name == 'outdoor_day1':
            event_mask = make_event_mask(staged_sample['events']['x'], staged_sample['events']['y'],self.cfg.dataset.sensor_size)
            event_mask = event_mask.at[190:, :].set(0)

        # extract final-optimized theta and scale to sensor array size
        final_theta = self.opt_results[sample_key]['solver_final_results']['final_theta_pyr']['pyr_lvl_0']
        final_theta_array = self.scale_theta_to_sensor_size_pfunc(theta=final_theta)

        # evaluate theta and print results
        final_theta_eval_results = self.evaluate_theta_array_pfunc(theta_array=final_theta_array,
                                                                   eval_xs=staged_sample['eval_events']['x'], 
                                                                   eval_ys=staged_sample['eval_events']['y'], 
                                                                   eval_ts=staged_sample['eval_events']['t'], 
                                                                   edges=staged_sample['edges'], 
                                                                   edge_ts=staged_sample['image_ts'], 
                                                                   gt_flow=staged_sample['gt_flow'])
        time_str, eval_str, evals, _ = final_theta_eval_results
        if self.cfg.experiment_settings.theta_evaluation.print_eval_results_at_sample:
            print(f'{time_str} | {eval_str}')

        # collect evaluation results at current datasample index
        self.eval_results.update({
            sample_key: {
                'evals': evals,
                'eval_ts': staged_sample['eval_ts'],
                'eval_ts_units': staged_sample['eval_ts_units'],
            }
        })
        
        return final_theta_eval_results
        

    def plot_end_results_at_datasample_idx(self, datasample_idx):
        sample_key = f'datasample_idx_{datasample_idx}'
        if not sample_key in self.opt_results:
            print(f'Key {sample_key} not found in opt_results')
            return None
        
        theta = self.opt_results[sample_key]['solver_final_results']['final_theta_pyr']['pyr_lvl_0']
        theta_array = self.scale_theta_to_sensor_size_pfunc(theta)
        self.plotter.plot_end_results(theta_array, 
                                      draw_events_every=1, 
                                      show=self.cfg.experiment_settings.plot.end_result.show, 
                                      save=self.cfg.experiment_settings.plot.end_result.save, 
                                      save_format=self.cfg.experiment_settings.plot.end_result.save_format,
                                      path=self.plot_end_results_dir, 
                                      idx=datasample_idx)

        # prior_theta = self.opt_results[sample_key]['solver_final_results']['prior_theta_pyr']['pyr_lvl_0']
        # prior_theta_array = self.scale_theta_to_sensor_size_pfunc(prior_theta)
        # self.plotter.plot_step_results(theta_array, prior_theta_array)


    def _display_exp_time_progress(self, datasample_idx):
        tot_elapsed_time = time.perf_counter() - self.exp_begin_time # seconds
        est_tot_runtime = tot_elapsed_time * (self.total_datasamples/(datasample_idx+1)) # seconds
        remaining_time = est_tot_runtime - tot_elapsed_time # seconds
        tot_elapsed_hms = self._seconds_to_hms(tot_elapsed_time)
        est_tot_hms = self._seconds_to_hms(est_tot_runtime)
        remaining_hms = self._seconds_to_hms(remaining_time)
        time_progress_str = (
            f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] ⌛ ...\n '
            + f'Est. Total: {est_tot_hms["hrs"]}hrs, {est_tot_hms["mins"]}mins, and {est_tot_hms["secs"]:.2f}secs '
            + f'| Elapsed: {tot_elapsed_hms["hrs"]}hrs, {tot_elapsed_hms["mins"]}mins, and {tot_elapsed_hms["secs"]:.2f}secs '
            + f'| Remaining: {remaining_hms["hrs"]}hrs, {remaining_hms["mins"]}mins, and {remaining_hms["secs"]:.2f}secs\n'
            + f'{"":=^120}\n'
        )
        print(time_progress_str)


    def _seconds_to_hms(self, seconds):
        _mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(_mins, 60)
        return {
            'hrs': int(hrs), 
            'mins': int(mins), 
            'secs': secs
        }


    def save_checkpoint(self, datasample_idx):
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{datasample_idx+1}_{self.total_datasamples}.npz'
        np.savez(checkpoint_path, opt_results=self.opt_results, cfg=OmegaConf.to_object(self.cfg))
        print(f'Checkpoint saved to {checkpoint_path}')


    def delete_checkpoints(self):
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)


    def collect_optimization_results_at_datasample_idx(self, datasample_idx):
        # fetch intermediate results collected by callbacks
        self.fetch_collected_optimization_intermediate_results()

        # collect both solver final and solver intermediate optimization results
        self.opt_results.update({
            f'datasample_idx_{datasample_idx}': {
                'solver_final_results': self.solver_results_on_datasample.copy(),
                'solver_intermediate_results': self.intermediate_results.copy()
            }
        })


    def run(self):
        if self.cfg.experiment_settings.solver.enable:
            # ---------------------------------------------------------------------------------------------------------#
            #                                                  SOLVE                                                   #
            # ---------------------------------------------------------------------------------------------------------#
            self.run_solver()

            # STORE optimization results to disk
            if self.cfg.experiment_settings.store_to_disk.opt_results:
                np.savez(f'{self.out_dir}/opt_results.npz', 
                         opt_results=self.opt_results, 
                         cfg=OmegaConf.to_object(self.cfg))
                
                # delete checkpoints
                if (self.cfg.experiment_settings.solver.checkpoints.enable
                    and self.cfg.experiment_settings.solver.checkpoints.delete_after_final_save):
                    self.delete_checkpoints()
            
            # ---------------------------------------------------------------------------------------------------------#
            #                                                 EVALUATE                                                 #
            # ---------------------------------------------------------------------------------------------------------#
            if (self.cfg.experiment_settings.theta_evaluation.enable
                and (not self.cfg.experiment_settings.theta_evaluation.eager)):
                # EVALUATE theta (patient)
                self.run_eval(opt_results_from_mem=True)

            # STORE evaluation results to disk
            if (self.cfg.experiment_settings.theta_evaluation.enable
                and self.cfg.experiment_settings.store_to_disk.eval_results):
                np.savez(f'{self.out_dir}/eval_results.npz', 
                         eval_results=self.eval_results, 
                         cfg=OmegaConf.to_object(self.cfg))


            # ---------------------------------------------------------------------------------------------------------#
            #                                                   PLOT                                                   #
            # ---------------------------------------------------------------------------------------------------------#
            if (self.cfg.experiment_settings.plot.enable
                and (not self.cfg.experiment_settings.plot.eager)):
                self.run_plot(opt_results_from_mem=True, eval_results_from_mem=True)


        else:
            # ---------------------------------------------------------------------------------------------------------#
            #                                                 EVALUATE                                                 #
            # ---------------------------------------------------------------------------------------------------------#
            if self.cfg.experiment_settings.theta_evaluation.enable:
                self.run_eval(opt_results_from_mem=False, 
                              load_cfg_opt=self.cfg.experiment_settings.load_cfg_from_opt_results)

                # STORE optimization results loaded from disk back to disk in current output dir
                if self.cfg.experiment_settings.store_to_disk.opt_results:
                    np.savez(f'{self.out_dir}/opt_results.npz', 
                            opt_results=self.opt_results, 
                            cfg=OmegaConf.to_object(self.cfg))

                # STORE evaluation results to disk in current output dir
                if self.cfg.experiment_settings.store_to_disk.eval_results:
                    np.savez(f'{self.out_dir}/eval_results.npz', 
                            eval_results=self.eval_results, 
                            cfg=OmegaConf.to_object(self.cfg))

            # ---------------------------------------------------------------------------------------------------------#
            #                                                   PLOT                                                   #
            # ---------------------------------------------------------------------------------------------------------#
            if self.cfg.experiment_settings.plot.enable:
                if self.cfg.experiment_settings.theta_evaluation.enable:
                    self.run_plot(opt_results_from_mem=True, eval_results_from_mem=True)
                else:
                    if not (self.cfg.experiment_settings.loading_paths.opt_results is None
                            and self.cfg.experiment_settings.loading_paths.eval_results is None):
                        self.run_plot(opt_results_from_mem=False, 
                                      load_cfg_opt=self.cfg.experiment_settings.load_cfg_from_opt_results,
                                      eval_results_from_mem=False,
                                      load_cfg_eval=self.cfg.experiment_settings.load_cfg_from_eval_results)
                    else:
                        raise ValueError('No opt_results or eval_results paths to plot from!')



    def run_solver(self):

        self._prerun()
        self._set_exp_begin_time()
        
        for datasample_idx, datasample in enumerate(self.dataloader):
            if self._skip_datasample_idx(datasample_idx):
                continue

            print(f'\n{"":-^120}\nWorking datasample index: {datasample_idx+1}/{self.total_datasamples}\n{"":-^120}')
        
            # stage datasample
            staged_sample = self.stage_datasample(datasample)
            
            # feed datasample (to solver, callbacks and plotter)
            self.feed_staged_sample(datasample_idx, staged_sample)

            # SOLVE theta for datasample and fetch results
            for i in range(self.cfg.solver_params.n_repeat_solve):
                self.solver_results_on_datasample = self.multi_level_solver.solve()

            # collect (final and intermediate) optimization results at current datasample index
            self.collect_optimization_results_at_datasample_idx(datasample_idx)

            # save checkpoints
            if (self.cfg.experiment_settings.solver.checkpoints.enable
                and datasample_idx > 1
                and datasample_idx % int(self.total_datasamples * self.cfg.experiment_settings.solver.checkpoints.at_percentage / 100) == 0):
                self.save_checkpoint(datasample_idx)

            # EVALUATE theta (eager)
            if (self.cfg.experiment_settings.theta_evaluation.enable
                and self.cfg.experiment_settings.theta_evaluation.eager
                and datasample_idx % self.cfg.experiment_settings.theta_evaluation.eval_results_on_sample_every == 0):
                # perform evaluation computations
                self.evaluate_opt_results_at_datasample_idx_and_collect(datasample_idx, staged_sample)

            # PLOT results (eager)
            if (self.cfg.experiment_settings.plot.enable
                and self.cfg.experiment_settings.plot.eager
                and datasample_idx % self.cfg.experiment_settings.plot.plot_end_results_on_sample_every == 0):
                self.plot_end_results_at_datasample_idx(datasample_idx)

            print(f'\n{"":=^120}\nFinished datasample index: {datasample_idx+1}/{self.total_datasamples}\n{"":-^120}')
            self._display_exp_time_progress(datasample_idx)

    
    def run_eval(self, 
                 opt_results_path=None, 
                 opt_results_from_mem=False,
                 load_cfg_opt=False,):
        if not opt_results_from_mem:
            self.load_opt_results_from_disk(opt_results_path, load_cfg=load_cfg_opt)
        
        self._prerun()

        print(f'\n{"":-^120}\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] Evaluating Thetas\n{"":-^120}')
        tqdm_kwargs = {
            'bar_format': 'Evaluating {n_fmt}/{total_fmt} | {l_bar}{bar} [{rate_fmt} {elapsed}<{remaining}{postfix}] ',
            'colour': 'green',
            'ncols': 120,
            'total': self.total_datasamples,
        }
        for datasample_idx, datasample in tqdm(enumerate(self.dataloader), **tqdm_kwargs):
            if self._skip_datasample_idx(datasample_idx):
                continue
        
            # stage datasample
            staged_sample = self.stage_datasample(datasample)
            
            # perform evaluation computations
            if  datasample_idx % self.cfg.experiment_settings.theta_evaluation.eval_results_on_sample_every == 0:
                self.evaluate_opt_results_at_datasample_idx_and_collect(datasample_idx, staged_sample)

        print(f'\n{"":=^120}\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] Finished Evaluating Thetas\n{"":=^120}')

        # extract scores
        self.extract_scores()
        # write scores to file
        with open(f'{self.out_dir}/scores.txt', 'w') as f:
            for metric in ['fwl', 'AEE', 'A1PE', 'A2PE', 'A3PE', 'A5PE', 'A10PE', 'A20PE', 'AREE']:
                if metric in self.scores:
                    score_stats_str = (
                        f'{metric.upper()}: '.ljust(7, ' ')
                        + f'[{self.scores[metric].min():.4f} '.ljust(8, ' ')
                        + f'| {self.scores[metric].mean():.4f} ± {self.scores[metric].std():.4f} '.ljust(24, ' ')
                        + f'| {self.scores[metric].max():.4f}]\n'
                    )
                    f.write(score_stats_str)
                    # if DSEC extended samples were used for optimization, report the orig sample scores
                    if ((self.cfg.sequence_name).split('_')[0] in ['interlaken', 'thun', 'zurich']
                        and self.cfg.dataset.loader.extended == True):
                        orig_score_stats_str = (
                            f'(orig) {metric.upper()}: '.ljust(7, ' ')
                            + f'[{self.scores[metric][::5][1:].min():.4f} '.ljust(8, ' ')
                            + f'| {self.scores[metric][::5][1:].mean():.4f} ± {self.scores[metric][::5][1:].std():.4f} '.ljust(20, ' ')
                            + f'| {self.scores[metric][::5][1:].max():.4f}'.ljust(10, ' ') + ']\n'
                        )
                        f.write(orig_score_stats_str)



    def run_plot(self, 
                 opt_results_path=None, 
                 opt_results_from_mem=False, 
                 load_cfg_opt=False,
                 eval_results_path=None, 
                 eval_results_from_mem=False,
                 load_cfg_eval=False):
        if not opt_results_from_mem and self.cfg.experiment_settings.loading_paths.opt_results is not None:
            self.load_opt_results_from_disk(opt_results_path, load_cfg=load_cfg_opt)

        if not eval_results_from_mem and self.cfg.experiment_settings.loading_paths.eval_results is not None:
            self.load_eval_results_from_disk(eval_results_path, load_cfg=load_cfg_eval)
        
        self._prerun()

        if (self.cfg.experiment_settings.plot.enable
            and (self.cfg.experiment_settings.plot.end_result.show
                 or self.cfg.experiment_settings.plot.end_result.save)):
            print(f'\n{"":-^120}\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] Plotting Qualitative Results\n{"":-^120}')
            tqdm_kwargs = {
                'bar_format': 'Plotting {n_fmt}/{total_fmt} | {l_bar}{bar} [{rate_fmt} {elapsed}<{remaining}{postfix}] ',
                'colour': 'green',
                'ncols': 120,
                'total': self.total_datasamples,
            }
            for datasample_idx, datasample in tqdm(enumerate(self.dataloader), **tqdm_kwargs):
                if self._skip_datasample_idx(datasample_idx):
                    continue
            
                # stage datasample
                staged_sample = self.stage_datasample(datasample)
                self.plotter.update_datasample(datasample_idx, staged_sample, is_staged=True)
                
                # PLOT results
                if datasample_idx % self.cfg.experiment_settings.plot.plot_end_results_on_sample_every == 0:
                    self.plot_end_results_at_datasample_idx(datasample_idx)

            print(f'\n{"":=^120}\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] Finished Plotting Qualitative Results\n{"":=^120}')

        if (self.cfg.experiment_settings.plot.enable
            and self.cfg.experiment_settings.plot.end_result.save
            and self.cfg.experiment_settings.plot.end_result.save_format=='png'
            and self.cfg.experiment_settings.plot.end_result.make_vid):
            
            self.vid_dir = self.out_dir / 'vids'
            os.makedirs(self.vid_dir, exist_ok=True)
            prefix = 'plot_end_result_idx_'
            fourcc_codec='DIVX'
            fps = 2

            img_paths = sorted([str(p) for p in self.plot_end_results_dir.iterdir() if str(p).endswith('.png') and prefix in str(p)], 
                               key = lambda x: int(Path(x).name.split('.')[0].replace(prefix, '')))

            # extract size information from 1st image, to initialize a video writer
            im = cv.imread(img_paths[0])
            im_size = (im.shape[1], im.shape[0])

            # initialize a video writer
            vid_name = self.vid_dir / f'{self.cfg.sequence_name}_{prefix[:-5]}.avi'
            vid_out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*fourcc_codec), fps, im_size)

            # read each frame and write into video
            tqdm_kwargs = {
            'bar_format': 'plot_end_results to vid {n_fmt}/{total_fmt} | {l_bar}{bar} [{rate_fmt} {elapsed}<{remaining}{postfix}] ',
            'colour': 'green',
            'ncols': 120,
            'total': self.total_datasamples,
            }
            for img_path in tqdm(img_paths, **tqdm_kwargs):
                img = cv.imread(img_path)
                vid_out.write(img)

            # release the video writer
            vid_out.release()



        if self.eval_results:
            # extract scores
            self.extract_scores()

            # plot fwls
            self.plotter.plot_fwls(self.scores['fwl'], 
                                   path=self.plot_evals_dir, 
                                   show=self.cfg.experiment_settings.plot.evals.show, 
                                   save=self.cfg.experiment_settings.plot.evals.save,
                                   save_format=self.cfg.experiment_settings.plot.evals.save_format)

            # plots aees 
            if 'AEE' in self.scores:
                self.plotter.plot_aees(self.scores['AEE'], 
                                   path=self.plot_evals_dir, 
                                   show=self.cfg.experiment_settings.plot.evals.show, 
                                   save=self.cfg.experiment_settings.plot.evals.save,
                                   save_format=self.cfg.experiment_settings.plot.evals.save_format)
                self.plotter.plot_anpes(self.scores, 
                                   path=self.plot_evals_dir, 
                                   show=self.cfg.experiment_settings.plot.evals.show, 
                                   save=self.cfg.experiment_settings.plot.evals.save,
                                   save_format=self.cfg.experiment_settings.plot.evals.save_format)


    def extract_scores(self):
        sorted_sample_keys = sorted(self.eval_results.keys(), key=lambda k: int(k.replace('datasample_idx_', '')))
        for metric in ['fwl', 'AEE', 'A1PE', 'A2PE', 'A3PE', 'A5PE', 'A10PE', 'A20PE', 'AREE']:
            self.scores[metric] = np.array([
                self.eval_results[sample_key]['evals'][metric]
                for sample_key in sorted_sample_keys if metric in self.eval_results[sample_key]['evals']
            ])

        empty_metric_keys = [metric_key for metric_key in self.scores.keys() if len(self.scores[metric_key]) == 0]
        for key in empty_metric_keys:
            self.scores.pop(key)

        return self.scores


    def load_opt_results_from_disk(self, opt_results_path=None, load_cfg=True):
        # load optimization results from disk (opt_results + cfg)
        if opt_results_path is None:
            opt_results_path = self.cfg.experiment_settings.loading_paths.opt_results
        assert (
            opt_results_path is not None
            and os.path.exists(opt_results_path)
        ), "Please provide a valid path to load opt_results.npz from disk."
        
        self.output_loader.load_opt_results(opt_results_path, load_cfg=load_cfg)
        self.opt_results = self.output_loader.get_opt_results()
        if load_cfg:
            self.cfg = self.output_loader.get_cfg()


    def load_eval_results_from_disk(self, eval_results_path=None, load_cfg=False):
        # load eval results from disk (eval_results + cfg)
        if eval_results_path is None:
            eval_results_path = self.cfg.experiment_settings.loading_paths.eval_results
        assert (
            eval_results_path is not None
            and os.path.exists(eval_results_path)
        ), "Please provide a valid path to load eval_results.npz from disk."

        self.output_loader.load_eval_results(eval_results_path, load_cfg=load_cfg)
        self.eval_results = self.output_loader.get_eval_results()
        if load_cfg:
            self.cfg = self.output_loader.get_cfg()




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   





    def get_loss_pyramids_for_sequence(self):
        loss_pyramids = []
        for key in self.opt_results.keys():
            loss_pyramids.append(self.opt_results[key]['solver_intermediate_results']['theta_opt']['losses'])
            loss_pyramids.append(self.opt_results[key]['solver_intermediate_results']['handover_opt']['losses'])
        return np.array(loss_pyramids)


    def get_fwls_for_sequence(self):
        fwls = []
        for key in self.eval_results.keys():
            fwls.append(self.eval_results[key]['evals']['fwl'])
        return np.array(fwls)
    
    def get_contrasts_for_sequence(self):
        contrasts = []
        for key in self.eval_results.keys():
            contrasts.append(self.eval_results[key]['evals']['mean_rel_contrast'])
        return np.array(contrasts)
    
    def get_correlations_for_sequence(self):
        correlations = []
        for key in self.eval_results.keys():
            correlations.append(self.eval_results[key]['evals']['mean_rel_corr'])
        return np.array(correlations)
    
    def get_total_variations_for_sequence(self):
        total_variations = []
        for key in self.eval_results.keys():
            total_variations.append(self.eval_results[key]['evals']['theta_tot_var'])
        return np.array(total_variations)

          

    def plot_loss_pyr(self, loss_pyr):
        UTA_GRAY_3          = [ 91/255, 103/255, 112/255]
        UTA_GREEN_1         = [108/255, 194/255,  74/255]
        n_pyr_lvls = len(loss_pyr.keys())
        pkey = lambda p: f'pyr_lvl_{p}'
        losses  = []
        for p in reversed(range(n_pyr_lvls)):
            losses += loss_pyr[pkey(p)]
        losses = np.array(losses)
        iter_idx = np.array([i for i in range(len(losses))])
        iter_lens = np.array([len(loss_pyr[pkey(pyr_lvl)]) for pyr_lvl in reversed(range(n_pyr_lvls))])
        iter_lens_cum = [0]
        for elem in iter_lens:
            iter_lens_cum.append(iter_lens_cum[-1] + elem)
        iter_lens_cum = np.array(iter_lens_cum)
        fig, axs = plt.subplots()

        fig.suptitle('Loss Pyramid at Final Datasample')

        axs.plot(losses, label='Intermediate Losses', alpha=0.8, linewidth=3, color=UTA_GREEN_1, solid_joinstyle='round', solid_capstyle='round')
        axs.plot(iter_idx[iter_lens_cum[:-1]], losses[iter_lens_cum[:-1]], label='Pyramid Level Begin (Coarse-to-Fine)', linewidth=0, color=UTA_GRAY_3, marker="^", markersize=10, alpha=0.9)
        axs.legend()
        axs.grid(which='major', color='#DDDDDD', linewidth=0.8)
        axs.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        axs.minorticks_on()

        plt.show()
            
        














