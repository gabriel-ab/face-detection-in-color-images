import cv2 as cv
import numpy as np


def minmax_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)



def eye_map_c(cb, cr):
    cb2 = minmax_scale(cb**2)
    cr_neg = (1 - minmax_scale(cr)) ** 2
    cb_by_cr = minmax_scale(cb / cr)
    pre_eye_map_c = np.mean([cb2, cr_neg, cb_by_cr], axis=0)
    eye_map_c = cv.equalizeHist(np.uint8(pre_eye_map_c * 255)).astype(np.float32) / 255
    return eye_map_c

def eye_map_l(y):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12,12))
    y_dilated = cv.dilate(y, kernel, iterations=2)
    y_eroded = cv.erode(y, kernel, iterations=2)
    eye_map_l = y_dilated - y_eroded
    return eye_map_l

def eye_map(ycbcr):
    y,cb,cr = np.moveaxis(ycbcr, -1, 0)
    return eye_map_l(y) * eye_map_c(cb,cr)


def mouth_map(ycbcr):
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]
    cr_by_cb = minmax_scale(cr / cb)
    cr2 = minmax_scale(cr**2)
    n = 0.95 * cr2.mean() / cr_by_cb.mean()
    mouth_map = minmax_scale(cr2 * (cr2 - n*cr_by_cb)**2)
    return mouth_map