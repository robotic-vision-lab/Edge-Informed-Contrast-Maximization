import jax

from utils.console_utils import bf

def print_jax_info():
    print(f'\n{"":-^80}')
    # check JAX GPU (https://github.com/google/jax/issues/971)
    print(f'JAX platform:    {bf(str(jax.lib.xla_bridge.get_backend().platform).upper())}')

    # config JAX x64 (https://github.com/google/jax/issues/756)
    print(f'JAX x64 enabled: {bf(jax.config.x64_enabled)}')
    print(f'{"":-^80}\n')


def delete_on_device_buffers():
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers(): 
        buf.delete()


def update_jax_config(cfg):
    for key, val in cfg.items():
        jax.config.update(key, val)