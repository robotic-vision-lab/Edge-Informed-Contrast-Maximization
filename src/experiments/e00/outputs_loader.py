from pathlib import Path

import numpy as np
from omegaconf import OmegaConf



class EINCMOutputLoader:
    def __init__(self):
        self.opt_path = None
        self.eval_path = None

        self.opt_results = None
        self.eval_results = None
        self.cfg = None


    def load_opt_results(self, opt_path, run_validation=True, load_cfg=True):
        """
        Loads EINCM optimization output results 
        """
        self.opt_path = Path(opt_path)
        assert self.opt_path.exists(), f'Output file path {self.opt_path} does not exist.'
        assert self.opt_path.suffix == '.npz', f'Output file {self.opt_path} is not a .npz file.'

        # load optimization results from disk (opt_results + cfg)
        try:
            opt_results_npz = np.load(self.opt_path, allow_pickle=True)
            self.opt_results = opt_results_npz['opt_results'].item()
            if load_cfg:
                self.cfg = opt_results_npz['cfg'].item()
                self.cfg = OmegaConf.create(self.cfg)
        except Exception as e:
            print(f'Failed to load optimization results from {self.opt_path}. \nError:\n{e}')
        
        if run_validation:
            self.validate_opt_results()

        return self.opt_results


    def validate_opt_results(self):
        """
        Validates the dictionary key structure of loaded optimization results
        Expected key structure:

            opt_results
            ├── datasample_idx_{index}
            │   ├── solver_final_results
            │   │   ├── prior_theta_pyr
            │   │   │   ├── pyr_lvl_0
            │   │   │   └── ...
            │   │   ├── pre_opt_theta_pyr
            │   │   │   ├── pyr_lvl_0
            │   │   │   └── ...
            │   │   ├── theta_opt_state_pyr
            │   │   │   ├── pyr_lvl_0
            │   │   │   └── ...
            │   │   ├── pre_handover_theta_pyr
            │   │   │   ├── pyr_lvl_0
            │   │   │   └── ...
            │   │   ├── ho_opt_state_pyr
            │   │   │   ├── pyr_lvl_0
            │   │   │   └── ...
            │   │   ├── final_handover_weight_pyr
            │   │   │   ├── pyr_lvl_0
            │   │   │   └── ...
            │   │   └── final_theta_pyr
            │   │       ├── pyr_lvl_0
            │   │       └── ...
            │   └── solver_intermediate_results
            │       ├── theta_opt
            │       │   ├── n_iters
            │       │   │   ├── pyr_lvl_0
            │       │   │   └── ...
            │       │   ├── thetas (optional)
            │       │   │   ├── pyr_lvl_0
            │       │   │   └── ...
            │       │   └── losses (optional)
            │       │       ├── pyr_lvl_0
            │       │       └── ...
            │       └── handover_opt
            │           ├── n_iters
            │           │   ├── pyr_lvl_0
            │           │   └── ...
            │           ├── handover_weights (optional)
            │           │   ├── pyr_lvl_0
            │           │   └── ...
            │           ├── losses (optional)
            │           │   ├── pyr_lvl_0
            │           │   └── ...
            │           └── thetas (optional)
            │               ├── pyr_lvl_0
            │               └── ...
            └── ...
        """
        assert isinstance(self.opt_results, dict), "opt_results must be a dictionary."

        assert all (
            (k0.startswith('datasample_idx_') 
             and  k1 in self.opt_results[k0] for k1 in ['solver_final_results', 'solver_intermediate_results']) 
                for k0 in self.opt_results.keys()
        ), (
            """
            opt_results must have the following top-level structure:

                opt_results
                ├── datasample_idx_{index}
                │   ├── solver_final_results
                │   └── solver_intermediate_results
                └── ...
            
            """
        )

        assert all (
            ((k2 in self.opt_results[k0]['solver_final_results'] 
              and len(self.opt_results[k0]['solver_final_results'][k2]) == self.cfg.n_pyr_lvls
              and k3.startswith('pyr_lvl_') for k3 in self.opt_results[k0]['solver_final_results'][k2])
                for k2 in ['prior_theta_pyr',
                           'pre_opt_theta_pyr',
                           'theta_opt_state_pyr',
                           'pre_handover_theta_pyr',
                           'ho_opt_state_pyr',
                           'final_handover_weight_pyr',
                           'final_theta_pyr'])
                    for k0 in self.opt_results.keys()
        ), """
        opt_results['datasample_idx_{index}']['solver_final_results'] must have the following structure,

            ├── solver_final_results
            │   ├── prior_theta_pyr
            │   │   ├── pyr_lvl_0
            │   │   └── ...
            │   ├── pre_opt_theta_pyr
            │   │   ├── pyr_lvl_0
            │   │   └── ...
            │   ├── theta_opt_state_pyr
            │   │   ├── pyr_lvl_0
            │   │   └── ...
            │   ├── pre_handover_theta_pyr
            │   │   ├── pyr_lvl_0
            │   │   └── ...
            │   ├── ho_opt_state_pyr
            │   │   ├── pyr_lvl_0
            │   │   └── ...
            │   ├── final_handover_weight_pyr
            │   │   ├── pyr_lvl_0
            │   │   └── ...
            │   └── final_theta_pyr
            │       ├── pyr_lvl_0
            │       └── ...

        where each item within must have keys for every pyramid level starting with prefix "pyr_lvl_".
        """


    def load_eval_results(self, eval_path, run_validation=True, load_cfg=False):
        """
        Loads EINCM evaluation output results
        """
        self.eval_path = Path(eval_path)
        assert self.eval_path.exists(), f'Output file path {self.eval_path} does not exist.'
        assert self.eval_path.suffix == '.npz', f'Output file {self.eval_path} is not a .npz file.'

        # load evaluation results from disk (eval_results + cfg)
        try:
            eval_results_npz = np.load(self.eval_path, allow_pickle=True)
            self.eval_results = eval_results_npz['eval_results'].item()
            if load_cfg:
                self.cfg = eval_results_npz['cfg'].item()
                self.cfg = OmegaConf.create(self.cfg)
        except Exception as e:
            print(f'Failed to load evaluation results from {self.eval_path}. \nError:\n{e}')

        if run_validation:
            self.validate_eval_results()

        return self.eval_results


    def validate_eval_results(self):
        """
        Validates dictionary key structure of loaded evaluation results.
        Expected key structure:
            eval_results
            ├── datasample_idx_{index}
            │   ├── evals
            │   │   ├── AEE
            │   │   ├── AREE
            │   │   ├── A1PE
            │   │   ├── A2PE
            │   │   ├── A3PE
            │   │   ├── A5PE
            │   │   ├── A10PE
            │   │   ├── A20PE
            │   │   ├── n_ee
            │   │   ├── n_pred
            │   │   ├── n_gt
            │   │   ├── n_pixels
            │   │   ├── loss
            │   │   ├── iwe_var
            │   │   ├── mean_rel_contrast
            │   │   ├── mean_rel_corr
            │   │   ├── theta_tot_var
            │   │   ├── theta_div
            │   │   ├── fwl
            │   │   ├── mean_rel_iwe_div
            │   │   ├── rel_contrasts
            │   │   ├── rel_correlations
            │   │   ├── flow_warp_losses
            │   │   └── multi_ref_weights
            │   ├── eval_ts
            │   └── eval_ts_units
            └── ...
        """

        assert isinstance(self.eval_results, dict), "eval_results must be a dictionary."

        assert all (
            (k0.startswith('datasample_idx_') 
             and k1 in self.eval_results[k0] for k1 in ['evals', 'eval_ts', 'eval_ts_units']) 
                for k0 in self.eval_results.keys()
        ), """
        eval_results must have the following top-level structure:

            eval_results
            ├── datasample_idx_{index}
            │   ├── evals
            │   ├── eval_ts
            │   └── eval_ts_units
            └── ...
        
        """

        assert (
            all (k2 in self.eval_results[k0]['evals'] for k2 in ['loss',
                                                                 'iwe_var',
                                                                 'mean_rel_contrast',
                                                                 'mean_rel_corr',
                                                                 'theta_tot_var',
                                                                 'theta_div',
                                                                 'fwl',
                                                                 'mean_rel_iwe_div',
                                                                 'rel_iwe_divergences',
                                                                 'rel_contrasts',
                                                                 'rel_correlations',
                                                                 'flow_warp_losses',
                                                                 'multi_ref_weights',]
                for k0 in self.eval_results.keys()) 
            or all (k2 in self.eval_results[k0]['evals'] for k2 in ['AEE',
                                                                    'AREE',
                                                                    'p1_AEE',
                                                                    'p2_AEE',
                                                                    'p3_AEE',
                                                                    'p5_AEE',
                                                                    'p10_AEE',
                                                                    'p20_AEE',
                                                                    'n_points',
                                                                    'n_pred',
                                                                    'n_gt',
                                                                    'n_pixels',
                                                                    'loss',
                                                                    'iwe_var',
                                                                    'mean_rel_contrast',
                                                                    'mean_rel_corr',
                                                                    'theta_tot_var',
                                                                    'theta_div',
                                                                    'fwl',
                                                                    'mean_rel_iwe_div',
                                                                    'rel_iwe_divergences',
                                                                    'rel_contrasts',
                                                                    'rel_correlations',
                                                                    'flow_warp_losses',
                                                                    'multi_ref_weights',]
                for k0 in self.eval_results.keys())
        ), """
        eval_results['datasample_idx_{index}']['evals'] must have the following structure,

            evals
            ├── AEE (optional)
            ├── AREE (optional)
            ├── A1PE (optional)
            ├── A2PE (optional)
            ├── A3PE (optional)
            ├── A5PE (optional)
            ├── A10PE (optional)
            ├── A20PE (optional)
            ├── n_ee (optional)
            ├── n_pred (optional)
            ├── n_gt (optional)
            ├── n_pixels (optional)
            ├── loss
            ├── iwe_var
            ├── mean_rel_contrast
            ├── mean_rel_corr
            ├── theta_tot_var
            ├── theta_div
            ├── fwl
            ├── mean_rel_iwe_div
            ├── rel_contrasts
            ├── rel_correlations
            ├── flow_warp_losses
            └── multi_ref_weights

        where each item within must have keys for every pyramid level starting with prefix "pyr_lvl_".
        """


    def get_opt_results(self):
        return self.opt_results
    

    def get_eval_results(self):
        return self.eval_results
    

    def get_cfg(self):
        return self.cfg