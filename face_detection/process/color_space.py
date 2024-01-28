import cv2 as cv
import numpy as np
from functools import lru_cache
import numba as nb

KL = 125
KH = 188
Y_MIN = 16
Y_MAX = 235

@nb.jit
def transform_cb(
    y: float,
    c: float,
    w: float = 46.97,
    wl: float = 23,
    wh: float = 14,
) -> float:
    if y < KL:
        y_center = 108 - (KL - y) * (118 - 108) / (KL - Y_MIN)
        y_cluster = wl + (w - Y_MIN) * (w - wl) / (KL - Y_MIN)
    elif y < KH:
        return c
    else:
        y_center = 108 - (y - KH) * (118 - 108) / (Y_MAX - KH)
        y_cluster = wh + (Y_MAX - y) * (w - wh) / (Y_MAX - KH)

    threshold = wh + (Y_MAX - KH) * (w - wh) / (Y_MAX - KH)
    return (c - y_center) * w / y_cluster + threshold


@nb.jit
def transform_cr(
    y: float,
    c: float,
    w: float = 38.76,
    wl: float = 20,
    wh: float = 10
) -> float:
    if y < KL:
        y_center = 154 - (KL - y) * (154 - 144) / (KL - Y_MIN)
        y_cluster = wl + (w - Y_MIN) * (w - wl) / (KL - Y_MIN)
    elif y < KH:
        return c
    else:
        y_center = 154 - (y - KH) * (154 - 132) / (Y_MAX - KH)
        y_cluster = wh + (Y_MAX - y) * (w - wh) / (Y_MAX - KH)

    threshold = wh + (Y_MAX - KH) * (w - wh) / (Y_MAX - KH)
    return (c - y_center) * w / y_cluster + threshold


def transform(ycbcr: np.ndarray):
    isfloat = ycbcr.dtype in (np.float32, np.float64)

    if isfloat:
        res = (ycbcr * 255).astype(np.uint8)
    else:
        res = ycbcr.copy()

    for i, line in enumerate(res):
        for j, (y, cr, cb) in enumerate(line):
            res[i, j, 1] = transform_cr(y, cr)
            res[i, j, 2] = transform_cr(y, cb)

    if isfloat:
        return res.astype(ycbcr.dtype)
    return res
