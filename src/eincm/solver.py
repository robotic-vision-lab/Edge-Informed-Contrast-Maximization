from functools import partial

import jax.numpy as jnp
import jax.image as jim
from jaxopt import ScipyMinimize
from jaxopt import ScipyBoundedMinimize



class MultipleLevelEINCMSolver:
    """
    Will be stateful with theta_pyramids


    """
    def __init__(self, 
                 n_pyr_lvls, 
                 theta_opt_maxiters, 
                 theta_loss_pfunc, 
                 theta_opt_solver_params, 
                 handover_opt_maxiters=None,
                 handover_loss_pfunc=None, 
                 handover_opt_solver_params=None,
                 handover_settings=None,
                 pyramid_downscale_method='bilinear',
                 pyramid_upscale_method='repeat',
                 pyramid_bases=None,
                 theta_solver_callback=None,
                 handover_solver_callback=None):
        assert (
            handover_settings is None
            or ('use_handover' in handover_settings
                and 'solve_handover_for_levels' in handover_settings
                and 'use_downscaled_finest_priors' in handover_settings
                and 'clip_solved_handover' in handover_settings
                and 'alpha_handover' in handover_settings)
        ), (
            ('handover_settings should contain the following keys: '
             + 'use_handover, solve_handover_for_levels, use_downscaled_finest_priors, clip_solved_handover, alpha_handover')
        )
        assert (
            handover_settings is None
            or (
                handover_settings['clip_solved_handover'] == False
                or (
                    'clip_solved_handover_limits' in handover_settings
                    and len(handover_settings['clip_solved_handover_limits']) == 2
                )
            )
        ), (
            'valid clip_solved_handover_limits should be provided if clip_solved_handover is set to True'
        )
        assert (
            theta_opt_maxiters is not None 
            and len(theta_opt_maxiters) == n_pyr_lvls
        ), ( 
            'theta_opt_maxiters should be provided for each pyramid level'
        )
        assert (
            handover_opt_maxiters is None 
            or len(handover_opt_maxiters) == n_pyr_lvls
        ), ( 
            'handover_opt_maxiters should be provided for each pyramid level'
        )
        assert (
            pyramid_upscale_method in ['repeat', 'linear', 'bilinear', 'trilinear',
                                       'cubic', 'bicubic', 'tricubic', 'lanczos3', 'lanczos5']
        ), (
            f'Invalid pyramid_upscale_method: "{pyramid_upscale_method}"'
        )
        assert (
            pyramid_downscale_method in ['linear', 'bilinear', 'trilinear', 
                                         'cubic', 'bicubic', 'tricubic', 'lanczos3', 'lanczos5']
        ), (
            f'Invalid pyramid_downscale_method: "{pyramid_downscale_method}"'
        )
        
        self.n_pyr_lvls = n_pyr_lvls

        self.theta_opt_maxiters = theta_opt_maxiters
        self.theta_opt_solver_params = theta_opt_solver_params
        self.theta_loss_pfunc = theta_loss_pfunc
        self.handover_opt_maxiters = handover_opt_maxiters if handover_opt_maxiters is not None else {}
        self.handover_opt_solver_params = handover_opt_solver_params
        self.handover_loss_pfunc = handover_loss_pfunc   

        self.handover_settings = handover_settings
        self.use_handover = handover_settings['use_handover']
        self.solve_handover_switch_per_level = {}
        for pyr_lvl in range(n_pyr_lvls):
            key = f'pyr_lvl_{pyr_lvl}'
            self.solve_handover_switch_per_level[key] = (
                True if pyr_lvl in handover_settings['solve_handover_for_levels']
                else False
            )
        self.use_downscaled_finest_priors = handover_settings['use_downscaled_finest_priors']
        self.clip_solved_handover = handover_settings['clip_solved_handover']
        self.clip_solved_handover_limits = (handover_settings['clip_solved_handover_limits'] 
                                            if self.clip_solved_handover 
                                            else None)
        self.alpha_handover = handover_settings['alpha_handover']

        self.pyramid_downscale_method = pyramid_downscale_method
        self.pyramid_upscale_method = pyramid_upscale_method
        self.pyramid_bases = pyramid_bases if pyramid_bases is not None else [2]*(self.n_pyr_lvls-1)

        self.theta_solver_callback = theta_solver_callback       # solver callback for theta
        self.handover_solver_callback = handover_solver_callback # solver callback for handover weight ('solve-handover' case)

        self.pre_opt_theta_pyr = {}      # initial theta
        self.opt_theta_pyr = {}          # optimized theta
        self.handover_opt_theta_pyr = {} # handovered + optimized theta (theta for current datasample)
        self.prior_theta_pyr = {}        # handovered + optimized theta for previous datasample
        self._initialize_theta_pyramids()

        self.init_handover_weight_pyr = {}  # initial handover weights used for the 'solved-handover' case
        self.final_handover_weight_pyr = {} # determined after either 'solving' or 'fixing' the handover weights
        self._initialize_handover_weights()

        self.single_lvl_theta_solvers = {}     # solvers for optimizing theta at each pyramid level
        self.single_lvl_handover_solvers = {}  # solvers for optimizing handover weights at each pyramid level
        self._construct_solver_for_each_pyramid_level()
        
        self.datasample = {}
        self._IS_FIRST_SAMPLE = True

    
    def not_first_sample(self):
        self._IS_FIRST_SAMPLE = False


    def _initialize_theta_pyramids(self, theta_pyr_init=None):
        self.pre_opt_theta_pyr[f'pyr_lvl_{self.n_pyr_lvls-1}'] = jnp.zeros((1,1,2))
        self.opt_theta_pyr[f'pyr_lvl_{self.n_pyr_lvls-1}'] = jnp.zeros((1,1,2))
        self.handover_opt_theta_pyr[f'pyr_lvl_{self.n_pyr_lvls-1}'] = jnp.zeros((1,1,2))
        if theta_pyr_init is not None:
                self.prior_theta_pyr = theta_pyr_init
        else:
            self.prior_theta_pyr[f'pyr_lvl_{self.n_pyr_lvls-1}'] = jnp.zeros((1,1,2))

        for pyr_lvl in reversed(range(self.n_pyr_lvls-1)):
            key, key_coarser = f'pyr_lvl_{pyr_lvl}', f'pyr_lvl_{pyr_lvl+1}'
            self.pre_opt_theta_pyr[key] = self._upscale_theta(self.pre_opt_theta_pyr[key_coarser],
                                                              base=self.pyramid_bases[-pyr_lvl-1])
            self.opt_theta_pyr[key] = self._upscale_theta(self.opt_theta_pyr[key_coarser],
                                                          base=self.pyramid_bases[-pyr_lvl-1])
            self.handover_opt_theta_pyr[key] = self._upscale_theta(self.handover_opt_theta_pyr[key_coarser],
                                                                   base=self.pyramid_bases[-pyr_lvl-1])
            if theta_pyr_init is None:
                self.prior_theta_pyr[key] = self._upscale_theta(self.prior_theta_pyr[key_coarser],
                                                                base=self.pyramid_bases[-pyr_lvl-1])
        

    def _initialize_handover_weights(self):
        for pyr_lvl in range(self.n_pyr_lvls):
            key = f'pyr_lvl_{pyr_lvl}'
            self.init_handover_weight_pyr[key] = 0.5
            self.final_handover_weight_pyr[key] = 0.5  


    def _construct_solver_for_each_pyramid_level(self):
        for pyr_lvl in range(self.n_pyr_lvls):
            key = f'pyr_lvl_{pyr_lvl}'
            
            self.single_lvl_theta_solvers[key] = ScipyMinimize(
                fun=partial(self.theta_loss_pfunc, cur_pyr_lvl=pyr_lvl), 
                method=self.theta_opt_solver_params['method'],
                maxiter=self.theta_opt_maxiters[key],
                jit=True,
                has_aux=True,
                options={'gtol': self.theta_opt_solver_params['options']['gtol'], 'return_all': True},
                callback=self.theta_solver_callback
            ) 

            self.single_lvl_handover_solvers[key] = ScipyBoundedMinimize(
                fun=partial(self.handover_loss_pfunc, cur_pyr_lvl=pyr_lvl), 
                method=self.handover_opt_solver_params['method'],
                maxiter=self.handover_opt_maxiters[key],
                jit=True,
                has_aux=False,
                options={'gtol': self.handover_opt_solver_params['options']['gtol'], 'return_all': True},
                callback=self.handover_solver_callback
            ) 

    def set_datasample(self, xs, ys, ts, edges, edge_ts):
        self.datasample = {
            'events': {
                'x': xs,
                'y': ys,
                't': ts,
            },
            'edges': edges,
            'edge_ts': edge_ts
        }


    def solve(self):
        """Solves theta, for the current datasample, through multiple pyramid levels.
        """
        self._pre_solve()
        
        for pyr_lvl in reversed(range(self.n_pyr_lvls)): # n_pyr_lvls-1, n_pyr_lvls-2, ..., 0
            key, next_key = f'pyr_lvl_{pyr_lvl}', f'pyr_lvl_{pyr_lvl-1}'
            print(f'\nPyramid Level: {pyr_lvl}')
            self._update_callback_pyr_lvl(pyr_lvl)            

            _n_extra_attempts = 0
            # optimize theta (pre-opt -> opt) at current pyramid level
            self.opt_theta_pyr[key], self.theta_opt_state_pyr[key] = (
                self.single_lvl_theta_solvers[key].run(self.pre_opt_theta_pyr[key],
                                                       self.datasample['events']['x'], 
                                                       self.datasample['events']['y'], 
                                                       self.datasample['events']['t'],
                                                       self.datasample['edges'], 
                                                       self.datasample['edge_ts'])
            )

            while ((not self.theta_opt_state_pyr[key].success) 
                   and self.theta_opt_state_pyr[key].iter_num > 0
                   and f'pyr_lvl_{pyr_lvl}' in self.theta_opt_solver_params['n_extra_attempts']
                   and _n_extra_attempts < self.theta_opt_solver_params['n_extra_attempts'][f'pyr_lvl_{pyr_lvl}']
                   ):
                print(f'  :  {key} optimization failed to converge with state:\n'
                      + f'  :      {self._formatted_opt_state(self.theta_opt_state_pyr[key], total_iters=self.theta_solver_callback.get_iters()[key])}\n'
                      + f'  continuing c2max (extra attempt: {_n_extra_attempts+1}/{self.theta_opt_solver_params["n_extra_attempts"][f"pyr_lvl_{pyr_lvl}"]})...')
                _n_extra_attempts += 1
                self.opt_theta_pyr[key], self.theta_opt_state_pyr[key] = (
                self.single_lvl_theta_solvers[key].run(self.opt_theta_pyr[key],
                                                       self.datasample['events']['x'], 
                                                       self.datasample['events']['y'], 
                                                       self.datasample['events']['t'],
                                                       self.datasample['edges'], 
                                                       self.datasample['edge_ts'])
                )
                if (_n_extra_attempts == self.theta_opt_solver_params['n_extra_attempts'][f'pyr_lvl_{pyr_lvl}'] 
                    or self.theta_opt_state_pyr[key].iter_num == 0):
                    print(f'  :  performed max attempts at {key} with state:\n'
                          + f'  :      {self._formatted_opt_state(self.theta_opt_state_pyr[key], total_iters=self.theta_solver_callback.get_iters()[key])}\n'
                          + f'  End c2max.')
                    

            self.handover_opt_theta_pyr[key] = self._perform_handover_at_level(pyr_lvl)

            # if not finest level, upscale and initialize next pyramid level
            if not pyr_lvl == 0:
                # upscale theta to initialize pre-opt theta at finer pyramid level
                self.pre_opt_theta_pyr[next_key] = self._upscale_theta(self.handover_opt_theta_pyr[key],
                                                                       base=self.pyramid_bases[-pyr_lvl])
            
            print(f'{key} done. | theta_opt_state: {self._formatted_opt_state(self.theta_opt_state_pyr[key], total_iters=self.theta_solver_callback.get_iters()[key])}')
            if key in self.ho_opt_state_pyr:
                print(f'{key} done. | ho_opt_state:    {self._formatted_opt_state(self.ho_opt_state_pyr[key], total_iters=self.handover_solver_callback.get_iters()[key])} | handover_weight: {self.final_handover_weight_pyr[key]:4.6f}')

        old_prior_theta_pyr = self.prior_theta_pyr.copy()
        self.prior_theta_pyr = self.handover_opt_theta_pyr.copy()
        self._IS_FIRST_SAMPLE = False


        return {
            'prior_theta_pyr': old_prior_theta_pyr.copy(),                      # theta pyramid from preceding iteration
            'pre_opt_theta_pyr': self.pre_opt_theta_pyr.copy(),                 # theta pyramid initial pre-c2max
            'theta_opt_state_pyr': self.theta_opt_state_pyr.copy(),             # theta optimization state per pyr_lvl
            'pre_handover_theta_pyr': self.opt_theta_pyr.copy(),                # theta pyramid post-c2max
            'ho_opt_state_pyr': self.ho_opt_state_pyr.copy(),                   # handover optimization state per pyr_lvl
            'final_handover_weight_pyr': self.final_handover_weight_pyr.copy(), # fixed/solved handover weights per pyr_lvl
            'final_theta_pyr': self.handover_opt_theta_pyr.copy(),              # theta pyramid post handover operation
        }


    def _pre_solve(self):
        self._stage_prior_theta_pyr()
        # initialize the coarsest pre_opt_theta
        key_coarsest = f'pyr_lvl_{self.n_pyr_lvls-1}'
        self.pre_opt_theta_pyr[key_coarsest] = self.prior_theta_pyr[key_coarsest]     

        self.theta_solver_callback.reset()
        self.handover_solver_callback.reset()

        self.theta_opt_state_pyr = {}
        self.ho_opt_state_pyr = {}


    def _stage_prior_theta_pyr(self):
        if self.use_downscaled_finest_priors:
            # for coarser levels (> 0), use downscaled finest prior theta
            for pyr_lvl in range(1, self.n_pyr_lvls):
                key, key_finer = f'pyr_lvl_{pyr_lvl}', f'pyr_lvl_{pyr_lvl-1}'  
                self.prior_theta_pyr[key] = self._downscale_theta(self.prior_theta_pyr[key_finer], 
                                                                  base=self.pyramid_bases[-(pyr_lvl-1)-1])


    def _update_callback_pyr_lvl(self, pyr_lvl): 
        # set current pyramid level for callbacks
        self.theta_solver_callback.set_cur_pyr_lvl(pyr_lvl)
        self.handover_solver_callback.set_cur_pyr_lvl(pyr_lvl)

        # every time we move to a new pyramid level, reset opt_iter for better accounting
        self.theta_solver_callback.reset_opt_iter()
        self.handover_solver_callback.reset_opt_iter()


    def _perform_handover_at_level(self, cur_pyr_lvl):
        key, key_finer = f'pyr_lvl_{cur_pyr_lvl}', f'pyr_lvl_{cur_pyr_lvl-1}'
        self.handover_solver_callback.set_prior_and_current_thetas(self.prior_theta_pyr[key], self.opt_theta_pyr[key])
        if self._IS_FIRST_SAMPLE:       
            return self.opt_theta_pyr[key]
        
        if not self.use_handover:
            return self.opt_theta_pyr[key]

        if self.solve_handover_switch_per_level[key]:
            # optimize handover weight at current pyramid level
            # since upscale is to be carried out after handover, the handover optimization is done for the finer
            # scale, unless already finest
            if cur_pyr_lvl > 0:
                prior_theta = self.prior_theta_pyr[key_finer]
                theta = self._upscale_theta(self.opt_theta_pyr[key], self.pyramid_bases[-cur_pyr_lvl])
                single_lvl_ho_solver = self.single_lvl_handover_solvers[key_finer]
            else:
                prior_theta = self.prior_theta_pyr[key]
                theta = self.opt_theta_pyr[key]
                single_lvl_ho_solver = self.single_lvl_handover_solvers[key]
            
            # solve handover
            solved_handover_weight, self.ho_opt_state_pyr[key] = (
                single_lvl_ho_solver.run(self.init_handover_weight_pyr[key],
                                         tuple(self.handover_settings.handover_limits),
                                         prior_theta,
                                         theta,
                                         self.datasample['events']['x'],
                                         self.datasample['events']['y'],
                                         self.datasample['events']['t'],
                                         self.datasample['edges'],
                                         self.datasample['edge_ts'])
            )
            if self.clip_solved_handover:
                solved_handover_weight = jnp.clip(solved_handover_weight, *self.clip_solved_handover_limits)
            self.final_handover_weight_pyr[key] = solved_handover_weight
        else:
            # use the fixed handover weight
            self.final_handover_weight_pyr[key] = self.alpha_handover

        # perform handover operation to obtain the final theta at current pyramid level
        post_handover_theta =  self.final_handover_weight_pyr[key] * self.prior_theta_pyr[key] \
                                + (1 - self.final_handover_weight_pyr[key]) * self.opt_theta_pyr[key]

        return post_handover_theta


    def _upscale_theta(self, theta, base=2):
        if self.pyramid_upscale_method == 'repeat':
            return jnp.repeat(jnp.repeat(theta, base, axis=0), base, axis=1)
        elif (self.pyramid_upscale_method in ['linear', 'bilinear', 'trilinear', 
                                              'cubic', 'bicubic', 'tricubic', 
                                              'lanczos3', 'lanczos5']):
            return jim.scale_and_translate(image=theta,
                                           shape=(int(theta.shape[0]*base), int(theta.shape[1]*base), 2),
                                           spatial_dims=(0, 1, 2),
                                           scale=jnp.array([base, base, 1.0]).astype(jnp.float64),
                                           translation=jnp.array([0.0, 0.0, 0.0]).astype(jnp.float64),
                                           method=self.pyramid_upscale_method)
        else:
            raise NotImplementedError(f'Upscale_method: "{self.pyramid_upscale_method}" is not implemented')


    def _downscale_theta(self, theta, base=2):
        if (self.pyramid_downscale_method in ['linear', 'bilinear', 'trilinear', 
                                              'cubic', 'bicubic', 'tricubic', 
                                              'lanczos3', 'lanczos5']):
            return jim.scale_and_translate(image=theta,
                                           shape=(int(theta.shape[0]/base), int(theta.shape[1]/base), 2),
                                           spatial_dims=(0, 1, 2),
                                           scale=jnp.array([1/base, 1/base, 1.0]).astype(jnp.float64),
                                           translation=jnp.array([0.0, 0.0, 0.0]).astype(jnp.float64),
                                           method=self.pyramid_downscale_method)
        else:     
            raise NotImplementedError(f'Downscale_method: "{self.pyramid_downscale_method}" is not implemented')

    def _formatted_opt_state(self, opt_state, total_iters='--'):
        return (f'loss={opt_state.fun_val:8.4f},'.ljust(18, ' ')
                # + f'success={opt_state.success},'.ljust(15, ' ')
                + f'status={opt_state.status}, '.ljust(11, ' ')
                + f'n_iters={opt_state.iter_num}, '.ljust(12, ' ')
                + f'tot_iters={total_iters}'.ljust(13, ' '))