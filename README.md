## Secrets of Edge-Informed Contrast Maximization (EINCM) for Event-Based Vision

[[Paper - CVF](https://openaccess.thecvf.com/content/WACV2025/papers/Karmokar_Secrets_of_Edge-Informed_Contrast_Maximization_for_Event-Based_Vision_WACV_2025_paper.pdf)] 
[[Paper + Supplementary Material (preprint)- arXiv](https://arxiv.org/pdf/2409.14611)] 
[[Oral Presentation](./docs/assets/oral_presentation/EINCM_Presentation_WACV_2025_animateless.pdf)] 
[[Poster](./docs/assets/poster/EINCM_Poster_WACV_2025.pdf)] 

<p align=center>
  <img src="./docs/assets/images/eincm_intro_figure.png" style="width: 1280px; height: auto; border: 2px solid white; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);"> <br/>
</p>

### Overview

This repository provides source code for our 2025 WACV paper titled "Secrets of
Edge-Informed Contrast Maximization for Event-Based Vision." Our paper extends
the uni-modal _contrast_ maximization to a bi-modal
_contrast_&ndash;_correlation_ maximization. EINCM produces superior sharpness
scores and establishes state-of-the-art event optical flow benchmarks on
publicly available datasets.

### Authors

- Pritam P. Karmokar [<img src="./docs/assets/google_scholar_logo/google_scholar_logo.svg" width=14pix>](https://scholar.google.com/citations?hl=en&user=9nBwKG4AAAAJ)
- Quan H. Nguyen [<img src="./docs/assets/google_scholar_logo/google_scholar_logo.svg" width=14pix>](https://scholar.google.com/citations?user=ewEzQiYAAAAJ&hl=en)
- William J. Beksi [<img src="./docs/assets/google_scholar_logo/google_scholar_logo.svg" width=14pix>](https://scholar.google.com/citations?user=lU2Z7MMAAAAJ&hl=en)

### Citation

If you find this project useful, then please consider citing our work.

```
@inproceedings{karmokar2025secrets,
    author={Karmokar, Pritam P. and Nguyen, Quan H. and Beksi, William J.},
    title={Secrets of Edge-Informed Contrast Maximization for Event-Based Vision},
    booktitle={Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    year={2025},
    pages={630--639},
    doi={10.1109/WACV61041.2025.00071}
}
```

### Code Setup

#### Conda Environment 

Create a virtual environment using [`conda`](https://anaconda.org/anaconda/conda):

```bash
# create the environment 
conda create -n eincm-env python=3.11

# activate the environment
conda activate eincm-env

# make sure pip is installed within the environment
conda install pip
```

With the virtual environment activated, move on to installing the pip packages.

#### Pip Packages 

Linux would be the best option for working with JAX! At the time of
development, JAX GPU was not supported on Windows natively (only through WSL).
We strongly recommend using JAX GPU.

1. Install [JAX](https://docs.jax.dev/en/latest/installation.html) (please follow the [instructions](https://docs.jax.dev/en/latest/installation.html) to install it successfully):
   ```bash
   pip install -U "jax[cuda12]"
   ```
2. Install [JAXopt](https://pypi.org/project/jaxopt) and read the next section about adding a patch to the jaxopt source code locally:
   ```bash
   pip install jaxopt
   ```
3. Install [`h5py`](https://pypi.org/project/h5py) and [`hdf5plugin`](https://pypi.org/project/hdf5plugin):
   ```bash
   pip install h5py hdf5plugin
   ```
4. Install [OpenCV](https://pypi.org/project/opencv-python):
   ```bash
   pip install opencv-python opencv-contrib-python
   ```
5. Install [`omegaconf`](https://pypi.org/project/omegaconf) and [`hydra`](https://pypi.org/project/hydra-core):
   ```bash
   pip install omegaconf hydra-core
   ```
6. Install additional packages:
   ```bash
   pip install easydict flow-vis imageio matplotlib numpy scikit-image scipy rich termcolor tqdm
   ```

#### JAX Source Code Edits

`scipy_callback` is the callback function that `Scipy` will receive. If
`Scipy`'s call wrapper sees the attribute `intermediate_result` in the
signature of the callback function `callback`, then it will preserve the
`OptimizeResult` wrapper around `callback`'s argument, `res`, and call the
callback function in keyword argument form `callback(intermediate_result=res)`.
Otherwise, it calls `callback` as `callback(np.copy(res.x))`. 

Most minimizers, including BFGS, pass `callback`, an `OptimizeResult` object
with attributes `x` and `fun` assigned with the intermediate (k-th iteration)
param and objective value, respectively.  The `jaxopt` ScipyWrapper may not
account for this detail. Therefore, a small patch in the jaxopt source code is
needed to accommodate this function to make the intermediate loss values
available in EINCM's callbacks.

Go to `jaxopt._src.scipywrapper.py` around line 325-328 (might be version
dependent), where you see the `scipy_callback` function:

```python
325 def scipy_callback(x_onp: onp.ndarray):
326   x_jnp = onp_to_jnp(x_onp)
327   return self.callback(x_jnp)
```

and replace it with the following:

```python
325 def scipy_callback(intermediate_result: osp.optimize.OptimizeResult):
326   intermediate_result.x = onp_to_jnp(intermediate_result.x)
327   return self.callback(intermediate_result)
328 # def scipy_callback(x_onp: onp.ndarray):
329 #   x_jnp = onp_to_jnp(x_onp)
330 #   return self.callback(x_jnp)
```

### Running Experiments

EINCM experiments are designed as Python packages. To run an experiment,
navigate to `src/` directory. Next, specify the package using namespace scoping
(e.g., `experiments.e00`) and run it as a module (with `python -m`). For
instance:

```bash
cd /path/to/src
python -m experiments.e00
```

This will run the `__main__.py` module within `experiments.e00`.  To enable
flexibility in how configs are input, the experiment expects configs to be
provided explicitly through command line.

```bash
cd /path/to/src
python -m experiments.e00 --config-dir="path/to/config/direcory" --config-name="<name-of-config-yaml-file>"

# For example:
#  python -m experiments.e00 --config-dir="./experiments/e00/config" --config-name=main
#  OR
#  python -m experiments.e00 --config-path="./configs" --config-name=main
# 
# Note:
# -----
# If relative paths are used, the args config-dir and config-path presume different current working directories.

```
Alternatively, the user may make use of the bash script `run.sh` (needs execute
permissions `chmod +x run.sh`). 

For more details on running the experiment under different configurations go to
[experiments/e00/](./src/experiments/e00/)

### License 

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
