#! /usr/bin/env bash


eval "$(conda shell.bash hook)"
conda deactivate 
conda deactivate 
conda activate eincm-env

cd src 

# =====================================================================================================================#
#                                                   RUN SOLVE + EVAL                                                   #
# =====================================================================================================================#
datasetsequence="mvsec-indoor"

case $datasetsequence in 
    "ecd-slider_depth")
        # ECD slider_depth
        python -m experiments.e00\
            --config-path=./configs --config-name=main\
            dataset=ecd\
            des_n_events=30000\
            root_dir="/path/to/ecd/root/dir"\
            sequence_name=slider_depth\
            alpha=60\
            beta=60\
            gamma=0.0\
            run_full_sequence=True\
            solver_params.theta_opt.maxiter=25\
            solver_params.ho_opt.maxiter=15\
            n_pyr_lvls=5\
            pyramid_bases=[2,2,2,2]\
            edge_extraction.canny.threshold_1=100\
            edge_extraction.canny.threshold_2=200\
            experiment_settings.theta_evaluation.enable=True\
            experiment_settings.plot.enable=False\
            experiment_settings.theta_evaluation.print_eval_results_at_sample=True

        ;;

    "mvsec-indoor")
        # MVSEC Indoor
        python -m experiments.e00\
            --config-path=./configs --config-name=main\
            dataset=mvsec\
            des_n_events=30000\
            root_dir="/path/to/mvsec/root/dir"\
            sequence_name=indoor_flying1\
            dt=4\
            alpha=20\
            beta=35\
            gamma=0.0\
            n_pyr_lvls=5\
            pyramid_bases=[2,2,2,2]\
            run_full_sequence=True\
            solver_params.theta_opt.maxiter=40\
            solver_params.ho_opt.maxiter=15\
            solver_params.theta_opt.n_extra_attempts.pyr_lvl_0=1\
            solver_params.theta_opt.n_extra_attempts.pyr_lvl_1=1\
            callback_options.theta_opt.enable=True\
            callback_options.theta_opt.print_intermediate_loss=True\
            callback_options.theta_opt.collect_thetas_and_losses=True\
            callback_options.handover_opt.enable=True\
            callback_options.handover_opt.print_intermediate_loss=True\
            callback_options.handover_opt.collect_ho_weights_and_losses=True\
            edge_extraction.canny.threshold_1=100\
            edge_extraction.canny.threshold_2=200\
            experiment_settings.theta_evaluation.enable=True\
            experiment_settings.theta_evaluation.print_eval_results_at_sample=False\
            experiment_settings.plot.enable=False

        ;;
    "mvsec-outdoor")
        # MVSEC Outdoor
        python -m experiments.e00\
            --config-path=./configs --config-name=main\
            dataset=mvsec\
            outdoor_day1_run_idx_range=continuous\
            des_n_events=40000\
            root_dir="/path/to/mvsec/root/dir"\
            sequence_name=outdoor_day1\
            dt=4\
            alpha=20\
            beta=35\
            gamma=0.0025\
            run_full_sequence=True\
            solver_params.theta_opt.maxiter=25\
            solver_params.ho_opt.maxiter=15\
            n_pyr_lvls=5\
            pyramid_bases=[2,2,2,2]\
            edge_extraction.canny.threshold_1=30\
            edge_extraction.canny.threshold_2=80\
            experiment_settings.theta_evaluation.enable=True\
            experiment_settings.plot.enable=False\
            experiment_settings.theta_evaluation.print_eval_results_at_sample=False

        ;;

    "dsec")
        # DSEC
        python -m experiments.e00\
            --config-path=./configs --config-name=main\
            dataset=dsec\
            des_n_events=1500000\
            root_dir="/path/to/DSEC/root/dir"\
            sequence_name=thun_01_a\
            alpha=2000\
            beta=4000\
            gamma=0.0\
            run_full_sequence=True\
            solver_params.theta_opt.maxiter=40\
            solver_params.ho_opt.maxiter=15\
            solver_params.theta_opt.n_extra_attempts_per_level=2\
            n_pyr_lvls=5\
            pyramid_bases=[2,2,2,2]\
            edge_extraction.canny.threshold_1=30\
            edge_extraction.canny.threshold_2=80\
            experiment_settings.theta_evaluation.enable=True\
            experiment_settings.plot.enable=False\
            experiment_settings.theta_evaluation.print_eval_results_at_sample=False
        ;;

    *)
        echo "Not running solve or eval."
        ;;

esac

# ---------------------------------------------------------------------------------------------------------------------#


# =====================================================================================================================#
#                                                   PLOT END RESULTS                                                   #
# =====================================================================================================================#
if false; then 
# if true; then 
    python -m experiments.e00\
        --config-dir="/media/pritam/Extreme Pro/Git/Edge-Informed-Contrast-Maximization/src/outputs/2025-03-01/12-07-01/.hydra" --config-name=config\
        experiment_settings.load_cfg_from_opt_results=False\
        experiment_settings.load_cfg_from_eval_results=False\
        experiment_settings.solver.enable=False\
        experiment_settings.theta_evaluation.enable=False\
        experiment_settings.plot.enable=True\
        experiment_settings.plot.end_result.show=False\
        experiment_settings.plot.end_result.save=True\
        experiment_settings.plot.end_result.save_format=png\
        experiment_settings.plot.evals.show=False\
        experiment_settings.plot.evals.save=True\
        experiment_settings.plot.evals.save_format=pdf\
        experiment_settings.loading_paths.eval_results="/media/pritam/Extreme Pro/Git/Edge-Informed-Contrast-Maximization/src/outputs/2025-03-01/12-07-01/eval_results.npz"\
        experiment_settings.loading_paths.opt_results="/media/pritam/Extreme Pro/Git/Edge-Informed-Contrast-Maximization/src/outputs/2025-03-01/12-07-01/opt_results.npz"

fi

# =====================================================================================================================#
