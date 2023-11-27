import cv2 as cv
import numpy as np
from functools import lru_cache


class ChromaNonLinearTransformation:
    w: float
    wl: float
    wh: float

    kl = 125
    kh = 188
    ymin = 16
    ymax = 235

    @classmethod
    def transform(cls, y: float, c: float):
        is_bright = y > cls.kh
        is_dark =  y < cls.kl

        if is_bright:
            y_center = cls.bright_center(y)
            y_cluster = cls.bright_cluster(y)
        elif is_dark:
            y_center = cls.dark_center(y)
            y_cluster = cls.dark_cluster(y)
        else: # kl < y < kh
            return c

        return (c - y_center) * cls.w / y_cluster + cls.bright_center(cls.kh)

    @classmethod
    def cluster(cls, y: float):
        if y < cls.kl:
            return cls.bright_cluster(y)
        if cls.kh < y:
            return cls.dark_cluster(y)

    @classmethod
    @lru_cache
    def center(cls, y: float) -> float:
        if y < cls.kl:
            return cls.dark_center(y)
        if cls.kh < y:
            return cls.bright_center(y)

    @classmethod
    def bright_cluster(cls, y: float):
        return cls.wh + (cls.ymax - y) * (cls.w - cls.wh) / (cls.ymax - cls.kh)

    @classmethod
    def dark_cluster(cls, y: float):
        return cls.wl + (y - cls.ymin) * (cls.w - cls.wl) / (cls.kl - cls.ymin)

    @classmethod
    def bright_center(cls, y: float):
        raise NotImplementedError

    @classmethod
    def dark_center(cls, y: float):
        raise NotImplementedError


class CR(ChromaNonLinearTransformation):
    w = 38.76
    wl = 20
    wh = 10

    @classmethod
    def dark_center(cls, y: float) -> float:
        return 154 - (cls.kl - y) * (154 - 144) / (cls.kl - cls.ymin)

    @classmethod
    def bright_center(cls, y: float) -> float:
        return 154 - (y - cls.kh) * (154 - 132) / (cls.ymax - cls.kh)


class CB(ChromaNonLinearTransformation):
    w = 46.97
    wl = 23
    wh = 14

    @classmethod
    def dark_center(cls, y: float) -> float:
        return 108 + (cls.kl - y) * (118 - 108) / (cls.kl - cls.ymin)

    @classmethod
    def bright_center(cls, y: float) -> float:
        return 108 + (y - cls.kh) * (118 - 108) / (cls.ymax - cls.kh)


def color_space_transformation(image: np.ndarray):
    ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
    if ycc.mean() < 1:
        ycc = (ycc * 255).astype(np.uint8)

    ycc = ycc.copy()
    for i, line in enumerate(ycc):
        for j, (y, cr, cb) in enumerate(line):
            ycc[i, j, 1] = CR.transform(y, cr)
            ycc[i, j, 2] = CB.transform(y, cb)

    return cv.cvtColor(ycc, cv.COLOR_YCR_CB2RGB)
