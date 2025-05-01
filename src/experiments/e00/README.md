# Experiment `e00`

## TOC
>1. [About Experiment](#1-about-experiment) <br/>
>2. [Configs](#2-configs) <br/>
>├── 2.1. [Basic run with `main.yaml`](#21-basic-run-with-mainyaml) <br/>
>├── 2.2. [Add configs on command line with `main.yaml`](#22-add-configs-on-command-line-with-mainyaml) <br/>
>│   ├── 2.2.1. [Run `SOLVE` with MVSEC `indoor_flying1`](#221-run-solve-with-mvsec-indoor_flying1) <br/>
>│   ├── 2.2.2. [Run `SOLVE` + `EVAL` with MVSEC `indoor_flying1`](#222-run-solve--eval-with-mvsec-indoor_flying1) <br/>
>│   └── 2.2.3. [Run `SOLVE` + `EVAL` + `PLOT` with MVSEC `indoor_flying1`](#223-run-solve--eval--plot-with-mvsec-indoor_flying1) <br/>
>└── 2.3. [Features](#23-features) <br/>
>    ├── 2.3.1. [Edge Extraction Pipeline](#231-edge-extraction-pipeline)  <br/>
>    │   ├── 2.3.1.1 [Image Pre-processing](#2311-image-pre-processing) <br/>
>    │   └── 2.3.1.2. [Image to Edge](#2312-image-to-edge) <br/>
>    ├── 2.3.2 [Pyramid Tweaking](#232-pyramid-tweaking) <br/>
>    │   ├── 2.3.2.1 [Changing Pyramid Levels and Bases](#2321-changing-pyramid-levels-and-bases) <br/>
>    │   └── 2.3.2.2 [Scaling Methods](#2322-scaling-methods) <br/>
>    ├── 2.3.3 [Solver (Optimizer) Tweaking](#233-solver-optimizer-tweaking) <br/>
>    │   ├── 2.3.3.1 [Growing `maxiter`](#2331-growing-maxiter) <br/>
>    │   ├── 2.3.3.2 [Multiple Attempts](#2332-multiple-attempts) <br/>
>    │   └── 2.3.3.3 [Repeat Solve](#2333-repeat-solve) <br/>
>    ├── 2.3.4 [Handovers](#234-handovers) <br/>
>    ├── 2.3.5 [Callbacks](#235-callbacks) <br/>
>    ├── 2.3.6 [Checkpoints](#236-checkpoints) <br/>
>    ├── 2.3.7 [Running in Phases](#237-running-in-phases) <br/>
>    │   ├── 2.3.7.1 [Run `EVAL` from previously run `SOLVE`](#2371-run-eval-from-previously-run-solve) <br/>
>    │   └── 2.3.7.2 [Run `PLOT` from previously run `SOLVE`, `EVAL`](#2372-run-plot-from-previously-run-solve-eval) <br/>
>    └── 2.3.8 [Advanced Config Overrides](#238-advanced-config-overrides)  <br/>


## 1. About Experiment 
Experiment `e00` contains Python implementation of the _contrast_&ndash;_correlation_ maximization (c2max) framework proposed in our work, 
"[Secrets of Edge-Informed Contrast Maximization](https://openaccess.thecvf.com/content/WACV2025/papers/Karmokar_Secrets_of_Edge-Informed_Contrast_Maximization_for_Event-Based_Vision_WACV_2025_paper.pdf)" (EINCM).

A full experiment run includes three phases:
1. `SOLVE` (optimization), 
2. `EVAL` (evaluation), and 
3. `PLOT` (plotting).

Computations for the three phases `SOLVE`, `EVAL`, and `PLOT` are performed inside the `run_solve`, the `run_eval`, and 
the `run_plot` methods, respectively. Broadly, each of these methods contain purposeful implementations with defined input&ndash;output mapping as follows.
1. `run_solve`
   - Input: Datasamples $\{\mathcal{D}^{(i)}\}_{i=0}^{n_{\text{samples}}}$
   - Output: Thetas $\{\Theta^{(i)}\}_{i=0}^{n_{\text{samples}}}$ [stored to disk in `opt_results.npz`]
1. `run_eval`
   - Input: Datasamples $\{\mathcal{D}^{(i)}\}_{i=0}^{n_{\text{samples}}}$ and Thetas $\{\Theta^{(i)}\}_{i=0}^{n_{\text{samples}}}$
   - Output: Evaluation Metrics (e.g., FWLs, AEEs, ANPEs, Losses, etc.) [stored to disk in `eval_results.npz` and `scores.txt`]
1. `run_plot`
   - Input: Datasamples $\{\mathcal{D}^{(i)}\}_{i=0}^{n_{\text{samples}}}$, Thetas $\{\Theta^{(i)}\}_{i=0}^{n_{\text{samples}}}$ and Evaluation Metrics
   - Output: Plots i.e., renders, `.png`, `.pdf`, `.mp4`, etc

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

## 2. Configs

Config handling is performed using [`hydra`](https://hydra.cc/docs/1.3/intro/). The main config file resides in 
[`./configs/main.yaml`](./configs/main.yaml). 

### 2.1. Basic Run With `main.yaml`
To run `e00` with this config, navigate to `src` 
```bash
cd /path/to/src
```
and run the following in commandline
```bash
python -m experiments.e00 --config-path="./configs" --config-name=main
```
To run on various dataset and various configs the user may edit the `main.yaml` file and run the above command. 
Alternatively, the use may create multiple YAML files (e.g., `main_1.yaml`, `main_ecd.yaml`, etc) and run through 
command line with appropriate argument `config-name` (e.g., `--config-name=main_1`, `--config-name=main_ecd`, etc).

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.2. Add Configs On Command Line With `main.yaml`
The codebase allows the user to use our framework with several options and designs. Therefore, the authors only provide 
a single config file (`main.yaml`) and encourage users to add hydra config overrides through command line arguments as follows.
> Author recommendation: Creating a bash script file and editing and running it would be easier.

#### 2.2.1. Run `SOLVE` with MVSEC `indoor_flying1`

```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   dataset=mvsec\
   des_n_events=30000\
   root_dir="/path/to/mvsec/root/dir"\
   sequence_name=indoor_flying1\
   dt=4\
   alpha=20\
   beta=35\
   gamma=0.0\
   run_full_sequence=True\
   solver_params.theta_opt.maxiter=25\
   solver_params.ho_opt.maxiter=15\
   callback_options.theta_opt.print_intermediate_loss=True\
   callback_options.handover_opt.print_intermediate_loss=True\
   n_pyr_lvls=5\
   pyramid_bases=[2,2,2,2]\
   edge_extraction.canny.threshold_1=100\
   edge_extraction.canny.threshold_2=200\
   experiment_settings.solver.enable=True\
   experiment_settings.theta_evaluation.enable=False\
   experiment_settings.plot.enable=False\
```


#### 2.2.2. Run `SOLVE` + `EVAL` With MVSEC `indoor_flying1`

```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   dataset=mvsec\
   des_n_events=30000\
   root_dir="/path/to/mvsec/root/dir"\
   sequence_name=indoor_flying1\
   dt=4\
   alpha=20\
   beta=35\
   gamma=0.0\
   run_full_sequence=True\
   solver_params.theta_opt.maxiter=25\
   solver_params.ho_opt.maxiter=15\
   callback_options.theta_opt.print_intermediate_loss=True\
   callback_options.handover_opt.print_intermediate_loss=True\
   n_pyr_lvls=5\
   pyramid_bases=[2,2,2,2]\
   edge_extraction.canny.threshold_1=100\
   edge_extraction.canny.threshold_2=200\
   experiment_settings.solver.enable=True\
   experiment_settings.theta_evaluation.enable=True\
   experiment_settings.theta_evaluation.print_eval_results_at_sample=False\
   experiment_settings.plot.enable=False\
```

#### 2.2.3. Run `SOLVE` + `EVAL` + `PLOT` With MVSEC `indoor_flying1`

```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   dataset=mvsec\
   des_n_events=30000\
   root_dir="/path/to/mvsec/root/dir"\
   sequence_name=indoor_flying1\
   dt=4\
   alpha=20\
   beta=35\
   gamma=0.0\
   run_full_sequence=True\
   solver_params.theta_opt.maxiter=25\
   solver_params.ho_opt.maxiter=15\
   callback_options.theta_opt.print_intermediate_loss=True\
   callback_options.handover_opt.print_intermediate_loss=True\
   n_pyr_lvls=5\
   pyramid_bases=[2,2,2,2]\
   edge_extraction.canny.threshold_1=100\
   edge_extraction.canny.threshold_2=200\
   experiment_settings.solver.enable=True\
   experiment_settings.solver.checkpoints.enable=True\
   experiment_settings.theta_evaluation.enable=True\
   experiment_settings.theta_evaluation.print_eval_results_at_sample=False\
   experiment_settings.plot.enable=True\
   experiment_settings.plot.end_result.show=False\
   experiment_settings.plot.end_result.save=True\
   experiment_settings.plot.end_result.save_format=png\
   experiment_settings.plot.evals.show=False\
   experiment_settings.plot.evals.save=True\
   experiment_settings.plot.evals.save_format=pdf\
```

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

## 2.3. Features
 
### 2.3.1. Edge Extraction Pipeline 
The stages in edge extraction pipeline are highly configurable. They are divided in two parts, (1) image preprocessing and
(2) image to edge.

#### 2.3.1.1 Image Pre-Processing
   - Denoise 
      ```bash
      python -m experiments.e00 --config-path=./configs --config-name=main\
         ...
         edge_extraction.denoise.smoothness_factor=4\
         edge_extraction.denoise.template_win_size=3\
         edge_extraction.denoise.search_win_size=11\
         ...
     ```
   - Clip Limited Adaptive Histogram Equalization (CLAHE) 
      ```bash
      python -m experiments.e00 --config-path=./configs --config-name=main\
         ...
         edge_extraction.clahe.clip_limit=5\
         edge_extraction.clahe.tile_grid_size=[10,10]\
         ...
     ```
   - Sharpening
      ```bash
      python -m experiments.e00 --config-path=./configs --config-name=main\
         ...
         edge_extraction.sharpen.neg_blur_kernel_size=3\
         edge_extraction.sharpen.sigma=2\
         edge_extraction.sharpen.pos_weight=1.5\
         edge_extraction.sharpen.neg_weight=-0.5\
         ...
     ```
   - Bilateral Filtering
      ```bash
      python -m experiments.e00 --config-path=./configs --config-name=main\
         ...
         edge_extraction.bilateral_filter.pix_neigh_diameter=5\
         edge_extraction.bilateral_filter.sigma_color=15\
         edge_extraction.bilateral_filter.sigma_space=15\
         ...
     ```
#### 2.3.1.2. Image To Edge 
   - Canny
      ```bash
      python -m experiments.e00 --config-path=./configs --config-name=main\
         ...
         edge_extraction.canny.aperture_size=3\
         edge_extraction.canny.threshold_1=100\
         edge_extraction.canny.threshold_2=200\
         ...
     ```
   - Smoothen Edges
      ```bash
      python -m experiments.e00 --config-path=./configs --config-name=main\
         ...
         edge_extraction.smoothen=gaussian
         edge_extraction.smoothen.k_size=1\
         edge_extraction.smoothen.sigma=1\
         ...
     ```
     This codebase additionally features a Python re-implementation of the Inverse Exponential Distance Transform (IEDT) from 
     (https://github.com/heudiasyc/rt_of_low_high_res_event_cameras/blob/master/src/distance_surface_cpu.cpp). As well as,
     an easier implementation of the IEDT using `scipy.ndimage.distance_transform_edt`.

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.3.2 Pyramid Tweaking

#### 2.3.2.1 Changing Pyramid Levels And Bases
To change the pyramid scheme users needs to edit two config elements, (1) `n_pyr_lvls` and (2) `pyramid_bases`, consistently.
The coarsest resolution is always $1\times1$. Then, pyramid bases are used to scale up. Therefore, the number of elements 
in `pyramid_bases` needs to be equal to (`n_pyr_lvls` $-1$).
```bash
# config#1
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   n_pyr_lvls=5\
   pyramid_bases=[2,2,2,2]\
   ...

# config#2
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   n_pyr_lvls=4\
   pyramid_bases=[4,2,2]\
   ...
```
In the above snippet, `config#1` will configure the pyramid resolutions to be $1\times1$, $2\times2$, $4\times4$, 
$8\times8$, and $16\times16$. <br/>
Whereas, `config#2` will configure the pyramid resolutions to be $1\times1$, $4\times4$, 
and $16\times16$.

#### 2.3.2.2 Scaling Methods
Scaling algorithms are used while upscaling or downscaling within pyramid levels, as well as scaling from theta to 
`sensor_size`. These can be configured as follows.
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   scale_theta_to_sensor_size_method=bilinear\
   pyramid_downscale_method=lanczos3\
   pyramid_upscale_method=repeat\
   ...
```

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.3.3 Solver (Optimizer) Tweaking
#### 2.3.3.1 Growing `maxiter`
While the solve method (e.g., `"BFGS"`, `"L-BFGS-B"`, `"Newton-CG"`, etc.) can be configured through `solver_params.theta_opt.method` and `solver_params.handover_opt.method`, the user can also enable maxiter schemes. The frameworks solves theta at each pyramid level, wherein 
the `maxiter` option limits the maximum optimization iterations performed in the solver algorithm. Since, optimization at 
finer pyramid levels are more significant, the code features application of a growing maxiter option.
`maxiter` can be varied between two numbers, `miniter` ($i_{\text{low}}$) and `maxiter` ($i_{\text{high}}$), as per 
$i_{\text{low}} \cdot t^{\text{order}} + i_{\text{high}} \cdot (1-t)^{\text{order}}$, where the parameter $t$ is the 
pyramid level.

```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   use_growing_maxiters=True\
   maxiters_grow_order=1.413\
   ...
```

#### 2.3.3.2 Multiple Attempts
Alternatively or additionally to growing maxiters, multiple attempts can be configured at specific levels for solving theta.
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   solver_params.theta_opt.n_extra_attempts.pyr_lvl_0=1\
   solver_params.theta_opt.n_extra_attempts.pyr_lvl_1=1\
   ...
```
Above config will run one extra attempts each for the finest two levels. For example, if the `maxiter` for the two levels 
where 30 and 25, it would instead run for 60 and 50.

#### 2.3.3.3 Repeat Solve
The code framework can also repeat the solve on current datasample using `solver_params.n_repeat_params`.
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   solver_params.n_repeat_solve=2\
   ...
```
With the above config, now the code will repeat the solve on the sample one more time. This is feature can help experimenting 
with the handover parameter selection, since the repeat solve preserves history and moves forward after performing handover 
between the solves.

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.3.4 Handovers
Handovers can be switched on/off using `handover_settings.use_handover`. Which levels use handover solves can be controlled 
using `handover_settings.solve_handover_for_levels`. 
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   handover_settings.use_handover=True\
   handover_settings.solve_handover_for_levels=[0,1]\
   ...
```
Above config will solve handovers for the two finest levels, 0 and 1.

The algorithm uses preceding theta pyramid and handovers to solve current theta pyramid. For handover, the preceding theta 
at corresponding levels can be used or it can use the finest levels downscaled for each handover. 
This can be configured using `handover_settings.use_downscaled_finest_prior`.
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   handover_settings.use_downscaled_finest_prior=True\
   ...
```
Above config will enforced use of finest preceding theta through downscaling for current handover.

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.3.5 Callbacks
Callback functions architectured for both theta and handover solvers within the EINCM code framework.
Enabling callbacks will allow the code to collect intermediate thetas (and/or solved handover weights) through the iterations.
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   callback_options.theta_opt.enable=True\
   callback_options.theta_opt.print_intermediate_loss=True\
   callback_options.theta_opt.collect_thetas_and_losses=True\
   callback_options.handover_opt.enable=True\
   callback_options.handover_opt.print_intermediate_loss=True\
   callback_options.handover_opt.collect_ho_weights_and_losses=True\
   ...
```
Above config will enable callbacks for both theta and handover weight solver, enable printing at every iteration within 
optimization, and enable collecting intermediate thetas, handovers, and corresponding losses.

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.3.6 Checkpoints
To avoid losing experiment `SOLVE` progress due to arbitrary failures, checkpointing can be enabled as follows.
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   experiment_settings.solver.checkpoints.enable=True\
   experiment_settings.solver.checkpoints.at_percentage=20\
   experiment_settings.solver.checkpoints.delete_after_final_save=True\
   ...
```
Above config will perform checkpointing after every $20\%$ completion of solve. Finally, after the completion and saving 
the `opt_results.npz` file, it will delete the checkpoints (to save disk space).

To continue running from a checkpoint, do the following.
```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   experiment_settings.solver.run_from_checkpoint=True\
   experiment_settings.solver.run_from_checkpoint="/path/to/checkpoint/npz/file"\
   ...
```
Above config will run code from the checkpoint and use configs stored inside the checkpoint NPZ file.

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.3.7 Running In Phases
As described in [Section 2.2](#22-add-configs-on-command-line-with-mainyaml), the EINCM codebase can be run in phases. 
Therefore, to continue with next phase, do the following.

#### 2.3.7.1 Run `EVAL` From Previously Run `SOLVE`
Successful `SOLVE` ends with storing `opt_results.npz` on disk. This can be used to continue to next phases.

```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   experiment_settings.solver.enable=False\
   experiment_settings.theta_evaluation.enable=True\
   experiment_settings.plot.enable=False\
   experiment_settings.load_cfg_from_opt_results=True\
   experiment_settings.load_cfg_from_eval_results=False\
   experiment_settings.loading_paths.opt_results="/path/to/opt_results.npz"\
   experiment_settings.store_to_disk.opt_results=True\
   experiment_settings.store_to_disk.eval_results=True\
   ...
```
Above config will run `EVAL` from a previously run `SOLVE` using the previous run's configs and store both `opt_results.npz`
and `eval_results.npz` (and `scores.txt`) inside current run's output directory.

#### 2.3.7.2 Run `PLOT` From Previously Run `SOLVE`, `EVAL`
Successful `SOLVE` ends with storing `opt_results.npz` on disk. Likewise, successful completion of `EVAL` results in storing 
`eval_results.npz`. Either or both these NPZ files can then be used to plot the optimization results and/or evaluations results.

```bash
python -m experiments.e00 --config-path=./configs --config-name=main\
   ...
   experiment_settings.solver.enable=False\
   experiment_settings.theta_evaluation.enable=False\
   experiment_settings.plot.enable=True\
   experiment_settings.load_cfg_from_opt_results=True\
   experiment_settings.load_cfg_from_eval_results=False\
   experiment_settings.loading_paths.opt_results="/path/to/opt_results.npz"\
   experiment_settings.loading_paths.eval_results="/path/to/eval_results.npz"\
   experiment_settings.store_to_disk.opt_results=True\
   experiment_settings.store_to_disk.eval_results=True\
   experiment_settings.plot.end_result.show=False\
   experiment_settings.plot.end_result.save=True\
   experiment_settings.plot.end_result.make_vid=True\
   experiment_settings.plot.end_result.save_format=png\
   experiment_settings.plot.evals.show=False\
   experiment_settings.plot.evals.save=True\
   experiment_settings.plot.evals.save_format=pdf\
   ...
```
Above config will run `PLOT` from a previously run `SOLVE` and/or `EVAL` using the previous `SOLVE` run's configs.
Addtionally, overrides plot parameters in current run. This will results in creation of end result plots, video and \
eval metric plots inside current run's output directory.

<p align="right">[<a href="#toc">Go to TOC</a>]</p>

### 2.3.8 Advanced Config Overrides 
```bash
python -m experiments.e00 --config-dir="/path/to/.hydra/directory" --config-name=config\
   ...


# for example
# python -m experiments.e00\
#    --config-dir="/Edge-Informed-Contrast-Maximization/src/outputs/2025-03-01/12-07-01/.hydra"\
#    --config-name=config\
#    ...
```

<p align="right">[<a href="#toc">Go to TOC</a>]</p>