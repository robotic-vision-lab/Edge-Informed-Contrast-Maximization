from functools import lru_cache

import flow_vis
import numpy as np


@lru_cache(maxsize=100)
def get_flow_color_code(img_size=(51,51), padding=1):
    H, W = img_size
    H -= padding*2
    W -= padding*2

    u = np.zeros((H, W), dtype=float)
    v = np.zeros((H, W), dtype=float)

    for row in range(H):
        for col in range(W):
            x, y = col, row
            u[row][col] = x - W//2
            v[row][col] = y - H//2
    
    # normalize [-1, 1]
    u = ((u - u.min()) / (u.max() - u.min()))*2 - 1
    v = ((v - v.min()) / (v.max() - v.min()))*2 - 1

    flow_code = flow_vis.flow_uv_to_colors(u,v)

    flow_code_pad = np.ones((H+2*padding, W+2*padding, 3), dtype=np.uint8) * 255
    flow_code_pad[padding:-padding, padding:-padding, :] = flow_code

    return flow_code_pad


def embed_flow_code_in_img(img, code_size=None):
    """Embeds the flow color encoding patch at the right-bottom of image"""
    if img.shape[0] < 65: return img

    # default code size 1/6-th img height
    if code_size is None:
        code_size = (img.shape[0]//6, img.shape[0]//6)

    # flow code should not be bigger than 1/6-th img width
    if code_size is not None and code_size[0] > img.shape[1]//6:
        code_size = (img.shape[1]//6, img.shape[1]//6)

    flow_code = get_flow_color_code(code_size)
    fcH, fcW, _ = flow_code.shape
    img[-fcH:, -fcW:, :] = flow_code
    
    return img


def flow_uv_to_img(uv, embed_code=True):
    """ obtain flow rgb image (H, W, 3) from uv (H, W, 2)"""
    flow_img = flow_vis.flow_uv_to_colors(uv[..., 0], uv[..., 1])
    if embed_code:
        flow_img = embed_flow_code_in_img(flow_img)

    return flow_img
