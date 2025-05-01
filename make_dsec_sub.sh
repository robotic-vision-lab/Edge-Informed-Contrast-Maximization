#! /usr/bin/env bash

eval "$(conda shell.bash hook)"
conda deactivate 
conda deactivate 
conda activate eincm-env

cd src 

python dsec_npz_to_png.py --skip_count 0  --sequence_name interlaken_00_b\
    --eval_ts_path "/media/pritam/Extreme Pro/Datasets/DSEC/Evaluation/test_forward_optical_flow_timestamps/interlaken_00_b.csv"\
    --opt_results_path "/media/pritam/Extreme Pro/Git/Event-Vision-Plane-Fitting-And-Contrast-Maximization/src/outputs/2025-04-16/15-43-04/opt_results.npz"\
    --out_dir "/media/pritam/Extreme Pro/Git/dsec_results/dsec_submission"

