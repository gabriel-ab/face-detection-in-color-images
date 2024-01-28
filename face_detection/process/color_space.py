import cv2 as cv
import numpy as np
from functools import lru_cache
import numba as nb

KL = 125
KH = 188
Y_MIN = 16
Y_MAX = 235

@nb.jit(nopython=True)
def transform_cb(
    y: float,
    c: float,
) -> float:

    w = 46.97
    wl = 23
    wh = 14
    threshold = 108 # center(KH) -> 108 + (KH - KH) * (118 - 108) / (Y_MAX - KH) -> 108

    if y < KL:
        y_center = 108 + (KL - y) * (118 - 108) / (KL - Y_MIN)
        y_cluster = wl + (y - Y_MIN) * (w - wl) / (KL - Y_MIN)
    elif y < KH:
        return c
    else:
        y_center = 108 + (y - KH) * (118 - 108) / (Y_MAX - KH)
        y_cluster = wh + (Y_MAX - y) * (w - wh) / (Y_MAX - KH)

    return (c - y_center) * w / y_cluster + threshold


@nb.jit(nopython=True)
def transform_cr(
    y: float,
    c: float,
) -> float:

    w = 38.76
    wl = 20
    wh = 10
    threshold = 154 # center(KH) -> 154 - (KH - KH) * (154 - 132) / (Y_MAX - KH) -> 154

    if y < KL:
        y_center = 154 - (KL - y) * (154 - 144) / (KL - Y_MIN)
        y_cluster = wl + (y - Y_MIN) * (w - wl) / (KL - Y_MIN)
    elif y < KH:
        return c
    else:
        y_center = 154 - (y - KH) * (154 - 132) / (Y_MAX - KH)
        y_cluster = wh + (Y_MAX - y) * (w - wh) / (Y_MAX - KH)

    return (c - y_center) * w / y_cluster + threshold


@nb.jit(nopython=True)
def transform(ycbcr: np.ndarray):
    res = ycbcr.copy()
    for i, line in enumerate(res):
        for j, (y, cr, cb) in enumerate(line):
            res[i, j, 1] = transform_cr(y, cr)
            res[i, j, 2] = transform_cr(y, cb)
    return res
