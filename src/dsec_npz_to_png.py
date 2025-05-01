import os
import warnings
from pathlib import Path
import argparse
import jax
import jax.image as jim
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import imageio.v2 as imageio


warnings.filterwarnings('ignore')
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)
jax.config.update('jax_traceback_filtering', 'off')


# Usage:
# python dsec_npz_to_png.py --skip_count 4 --jump_first 1 --sequence_name zurich_city_14_c\
#     --eval_ts_path "/media/pritam/Extreme Pro/Datasets/DSEC/Evaluation/test_forward_optical_flow_timestamps/zurich_city_14_c.csv"\
#     --opt_results_path "./src/outputs/2025-04-09/18-22-22/opt_results.npz"\
#     --out_dir "../../dsec_results/dsec_submission"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sequence_name', type=str, help='Name of the DSEC sequence')
    parser.add_argument('--opt_results_path', type=str, help=f'Path to the opt_results.npz.')
    parser.add_argument('--eval_ts_path', type=str, help=f'Path to Evaluation timestamps .csv file.')
    parser.add_argument('--out_dir', type=str, help=f'Path to the output directory.')
    parser.add_argument('--skip_count', type=str, help='Number of Thetas to skip while converting to png')
    parser.add_argument('--jump_first', type=bool, help='True/False indicating whether to jump the first Theta')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    DSEC_H, DSEC_W = 480, 640

    args = parse_args()
    print(f'Converting for sequence {args.sequence_name}...')

    out_dir_path = Path(f'{args.out_dir}') / f'{args.sequence_name}'
    os.makedirs(out_dir_path, exist_ok=True)

    eval_ts_path = Path(f'{args.eval_ts_path}')
    eval_ts = np.loadtxt(eval_ts_path, delimiter=',', skiprows=1, dtype='int64') # ndarray with 3 columns: from_ts, to_ts, file_idx
    eval_file_idxs = eval_ts[:, 2]
    print(f'Read {len(eval_file_idxs)} eval points from {eval_ts_path}')

    # extract thetas
    opt_results_npz = np.load(Path(args.opt_results_path).absolute(), allow_pickle=True)
    opt_results = opt_results_npz['opt_results'].item()
    thetas = np.array([opt_results[k]['solver_final_results']['final_theta_pyr']['pyr_lvl_0'] for k in opt_results])

    # first perform the skips
    if args.skip_count is not None:
        thetas = thetas[::int(args.skip_count)+1]

    print(f'Input NPZ file: {opt_results_npz}')
    print(f'Output directory: {out_dir_path}')
    print(f'Loaded thetas. Thetas shape: {thetas.shape}')

    # then perform the jump-first if needed  
    if args.jump_first is not None and args.jump_first == True:
        thetas = thetas[1:, ...]
        print(f'Jumped first theta. Thetas shape: {thetas.shape}')

    # scaled theta to sensor size and convert to png
    THETA_H, THETA_W = thetas[0].shape[:2]
    tqdm_kwargs = {
            'bar_format': f'Converting ({args.sequence_name}): ' + '{n_fmt}/{total_fmt} | {l_bar}{bar} [{rate_fmt} {elapsed}<{remaining}{postfix}] ',
            'colour': 'green',
            'ncols': 80,
            'total': thetas.shape[0],
        }
    for i, theta in tqdm(enumerate(thetas), **tqdm_kwargs):
        scaled_theta = jim.scale_and_translate(
            image=theta,
            shape=(DSEC_H, DSEC_W, 2),
            spatial_dims=(0, 1, 2),
            scale=jnp.array([DSEC_H/THETA_H, DSEC_W/THETA_W, 1.0]).astype(jnp.float64),
            translation=jnp.array([0.0, 0.0, 0.0]).astype(jnp.float64),
            method='bilinear'
        )

        flow_map = np.zeros((DSEC_H, DSEC_W, 3), dtype='uint16')
        flow_map[..., 0] = (scaled_theta[..., 0] * 128 + 2**15).astype('uint16')
        flow_map[..., 1] = (scaled_theta[..., 1] * 128 + 2**15).astype('uint16')

        out_filename = str(eval_file_idxs[i]).zfill(6) + '.png'
        out_path = out_dir_path / f'{out_filename}'

        imageio.imwrite(out_path, flow_map, format='PNG-FI')





