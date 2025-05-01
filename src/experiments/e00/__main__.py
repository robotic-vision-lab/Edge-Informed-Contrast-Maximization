import logging 
import warnings

import hydra
import matplotlib as mpl
from omegaconf import DictConfig, OmegaConf

from .exp_mgr import EINCMExperiment
from .jax_helpers import delete_on_device_buffers
from .jax_helpers import print_jax_info
from .jax_helpers import update_jax_config



warnings.filterwarnings('ignore')


log = logging.getLogger("main")

OmegaConf.register_new_resolver(
    "divide", lambda num1, num2: num1/num2
)


@hydra.main(version_base=None)
def run(cfg: DictConfig) -> None:
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    update_jax_config(cfg.jax_config)
    delete_on_device_buffers()
    print_jax_info()
    mpl.rcParams.update(cfg.mpl_rcparams)
    
    exp = EINCMExperiment(cfg)
    exp.run()


if __name__ == '__main__':
    run()
