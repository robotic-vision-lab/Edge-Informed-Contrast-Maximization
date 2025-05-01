from typing import Callable

from jax import Array as JaxArray
from jax import jit



class EmptyThetaSolverCallback:
    def __init__(self):
        pass


    def __call__(self, param_k):
        return None


    def set_cur_pyr_lvl(self, *args) -> None:
        pass


    def set_datasample(self, *args, **kwargs):
        pass


    def get_iters(self):
        pass


    def get_thetas(self):
        pass


    def get_losses(self):
        pass


    def get_eval_results(self):
        pass


    def reset_opt_iter(self):
        pass


    def reset(self):
        pass



class EmptyHandoverSolverCallback:
    def __init__(self):
        pass


    def __call__(self, param_k):
        return None


    def set_cur_pyr_lvl(self, *args) -> None:
        pass


    def set_datasample(self, *args, **kwargs):
        pass

 
    def set_prior_and_current_thetas(self, *args):
        pass


    def get_iters(self):
        pass


    def get_handover_weights(self):
        pass


    def get_losses(self):
        pass


    def get_eval_results(self):
        pass


    def get_thetas(self):
        pass


    def reset_opt_iter(self):
        pass


    def reset(self):
        pass



class EINCMThetaSolverCallback:
    def __init__(self, 
                 n_pyr_lvls: int,
                 scale_theta_to_sensor_size_func: Callable, 
                 evaluate_theta_func: Callable,
                 callback_options: dict,
                 datasample={}):
        self.n_pyr_lvls = n_pyr_lvls
        self.scale_theta_to_sensor_size_func = scale_theta_to_sensor_size_func
        self.evaluate_theta_func = jit(evaluate_theta_func) if callback_options.eval_thetas else evaluate_theta_func
        
        assert all(
            key in callback_options for key in ['collect_thetas_and_losses', 
                                                'eval_thetas', 
                                                'collect_eval_results', 
                                                'print_intermediate_loss',
                                                'print_eval_results']
        ), 'callback_options must contain keys: collect_thetas_and_losses, eval_thetas, collect_eval_results, print_eval_results'

        self.callback_options = callback_options        
        self.datasample = datasample

        self.cur_pyr_lvl = None
        self.eval_results = {}
        self.thetas = {}
        self.losses = {}
        self._opt_iters = {}

        self.reset()


    def __call__(self, intermediate_result):
        theta_k = intermediate_result.x
        loss_k = intermediate_result.fun
        if self.callback_options.print_intermediate_loss:
            print(f'  ├─ c2max itr: {self._opt_iters[f"pyr_lvl_{self.cur_pyr_lvl}"]}, loss: {loss_k:8.8f}')
        if self.callback_options.collect_thetas_and_losses:
            self._collect_theta_k(theta_k)
            self._collect_loss_k(loss_k)

        if self.callback_options.eval_thetas:
            time_str, eval_str, evals, _ = self._evaluate_theta(theta_k)
            if self.callback_options.collect_eval_results:
                self.eval_results[f'pyr_lvl_{self.cur_pyr_lvl}'].append({
                    'time_str': time_str,
                    'eval_str': eval_str,
                    'evals': evals,
                })
            if self.callback_options.print_eval_results:
                print(f'{time_str} | {eval_str}')
        
        self._opt_iters[f'pyr_lvl_{self.cur_pyr_lvl}'] += 1
 

    def _collect_theta_k(self, theta_k):
        self.thetas[f'pyr_lvl_{self.cur_pyr_lvl}'].append(theta_k) 


    def _collect_loss_k(self, loss_k):
        self.losses[f'pyr_lvl_{self.cur_pyr_lvl}'].append(loss_k) 


    def _evaluate_theta(self, theta):
        theta_array = self.scale_theta_to_sensor_size_func(theta)

        eval_results = self.evaluate_theta_func(theta_array, 
                                                self.datasample['xs'], self.datasample['ys'], self.datasample['ts'],
                                                self.datasample['edges'], self.datasample['edge_ts'],
                                                self.datasample['gt_flow'])
        
        return eval_results # (time_str, eval_str, evals, loss_computation_aux)

    
    def set_cur_pyr_lvl(self, cur_pyr_lvl: int) -> None:
        self.cur_pyr_lvl = cur_pyr_lvl


    def set_datasample(self, 
                       xs: JaxArray, 
                       ys: JaxArray, 
                       ts: JaxArray, 
                       edges: JaxArray, 
                       edge_ts: JaxArray, 
                       gt_flow: JaxArray=None) -> None:
        self.datasample['xs'] = xs
        self.datasample['ys'] = ys
        self.datasample['ts'] = ts
        self.datasample['edges'] = edges
        self.datasample['edge_ts'] = edge_ts
        self.datasample['gt_flow'] = gt_flow


    def get_iters(self):
        return self._opt_iters.copy()


    def get_thetas(self):
        return self.thetas.copy()


    def get_losses(self):
        return self.losses.copy()


    def get_eval_results(self):
        return self.eval_results.copy()

    
    def reset_opt_iter(self):
        self._opt_iters[f'pyr_lvl_{self.cur_pyr_lvl}'] = 0


    def reset(self):
        # self.reset_opt_iter()
        for pyr_lvl in range(self.n_pyr_lvls):          
            key = f'pyr_lvl_{pyr_lvl}'
            self.eval_results[key] = []
            self.thetas[key] = []
            self.losses[key] = []
            self._opt_iters[key] = 0



class EINCMHandoverSolverCallback:
    def __init__(self,
                 n_pyr_lvls: int,
                 scale_theta_to_sensor_size_func: Callable, 
                 evaluate_theta_func: Callable,
                 callback_options: dict,
                 datasample={}):
        self.n_pyr_lvls = n_pyr_lvls
        self.scale_theta_to_sensor_size_func = scale_theta_to_sensor_size_func
        self.evaluate_theta_func = jit(evaluate_theta_func) if callback_options.eval_ho_weights else evaluate_theta_func
        
        assert all(
            key in callback_options for key in ['collect_ho_weights_and_losses', 
                                                'collect_thetas',
                                                'print_intermediate_loss',
                                                'eval_ho_weights', 
                                                'collect_eval_results', 
                                                'print_eval_results']
        ), 'callback_options must contain keys: collect_ho_weights_and_losses, eval_ho_weights, collect_eval_results, print_eval_results'
        
        self.callback_options = callback_options        
        self.datasample = datasample

        self.cur_pyr_lvl = None
        self.eval_results = {}
        self.handover_weights = {}
        self.losses = {}
        self.thetas = {}
        self._opt_iters = {}

        self.reset()

        self.prior_theta = 0
        self.cur_theta = 0


    def __call__(self, intermediate_result):
        handover_weight_k = intermediate_result.x
        loss_k = intermediate_result.fun
        if self.callback_options.print_intermediate_loss:
            print(f'  ├─ hndvr itr: {self._opt_iters[f"pyr_lvl_{self.cur_pyr_lvl}"]}, loss: {loss_k:8.8f}')
        if self.callback_options.collect_ho_weights_and_losses:
            self._collect_handover_weight_k(handover_weight_k)
            self._collect_loss_k(loss_k)

        # collect thetas
        if self.callback_options.collect_thetas:
            self.thetas[f'pyr_lvl_{self.cur_pyr_lvl}'].append(
                handover_weight_k * self.prior_theta + (1 - handover_weight_k) * self.cur_theta
            )

        if self.callback_options.eval_ho_weights:
            time_str, eval_str, evals, _ = self._evaluate_handover_weight(handover_weight_k)
            if self.callback_options.collect_eval_results:
                self.eval_results[f'pyr_lvl_{self.cur_pyr_lvl}'].append({
                    'time_str': time_str,
                    'eval_str': eval_str,
                    'evals': evals,
                })
            if self.callback_options.print_eval_results:
                print(f'{time_str} | {eval_str}')

        self._opt_iters[f'pyr_lvl_{self.cur_pyr_lvl}'] += 1


    def _collect_handover_weight_k(self, handover_weight_k):
        self.handover_weights[f'pyr_lvl_{self.cur_pyr_lvl}'].append(handover_weight_k)


    def _collect_loss_k(self, loss_k):
        self.losses[f'pyr_lvl_{self.cur_pyr_lvl}'].append(loss_k) 


    def _evaluate_handover_weight(self, handover_weight):
        theta = handover_weight * self.prior_theta + (1 - handover_weight) * self.cur_theta
        theta_array = self.scale_theta_to_sensor_size_func(theta)
        
        eval_results = self.evaluate_theta_func(theta_array, 
                                                self.datasample['xs'], self.datasample['ys'], self.datasample['ts'],
                                                self.datasample['edges'], self.datasample['edge_ts'],
                                                self.datasample['gt_flow'])
        
        return eval_results  # (time_str, eval_str, evals, loss_obj)


    def set_cur_pyr_lvl(self, cur_pyr_lvl: int) -> None:
        self.cur_pyr_lvl = cur_pyr_lvl


    def set_datasample(self, 
                       xs: JaxArray, 
                       ys: JaxArray, 
                       ts: JaxArray, 
                       edges: JaxArray, 
                       edge_ts: JaxArray, 
                       gt_flow: JaxArray=None) -> None:
        self.datasample['xs'] = xs
        self.datasample['ys'] = ys
        self.datasample['ts'] = ts
        self.datasample['edges'] = edges
        self.datasample['edge_ts'] = edge_ts
        self.datasample['gt_flow'] = gt_flow


    def set_prior_and_current_thetas(self, prior_theta, cur_theta):
        self.prior_theta = prior_theta
        self.cur_theta = cur_theta


    def get_iters(self):
        return self._opt_iters.copy()


    def get_handover_weights(self): 
        return self.handover_weights.copy()


    def get_losses(self):
        return self.losses.copy()


    def get_eval_results(self):
        return self.eval_results.copy()


    def get_thetas(self):
        return self.thetas.copy()


    def reset_opt_iter(self):
        self._opt_iters[f'pyr_lvl_{self.cur_pyr_lvl}'] = 0


    def reset(self):
        # self.reset_opt_iter()        
        for pyr_lvl in range(self.n_pyr_lvls):
            key = f'pyr_lvl_{pyr_lvl}'
            self.eval_results[key] = []
            self.handover_weights[key] = []
            self.thetas[key] = []
            self.losses[key] = []
            self._opt_iters[key] = 0