import numpy as np
import cv2 as cv


a = np.sin(2.53)
b = np.cos(2.53)
skin_model = np.array([[ b, a],[-a, b]])
cx = 109.38
cy = 152.02


def skin_color_detection(ycc: np.ndarray):
    cb = ycc[..., 2] - cx
    cr = ycc[..., 1] - cy
    xy = skin_model @ [cb,cr]
    return xy



def my_skin_color_detection(image: np.ndarray, ycc: np.ndarray = None, inplace: bool = False):
    ymin = 16.0
    ymax = 235.0
    crmin = 142.0
    crmax = 162.0
    cbmin = 100.0
    cbmax = 119.0

    if image.dtype == np.float32:
        image = np.uint8(image * 255)

    if ycc is None:
        ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)

    y, cr, cb = np.moveaxis(ycc, 2, 0)

    valid_pixels = (
        ((ymin < y) & (y < ymax))
        & ((crmin < cr) & (cr < crmax))
        & ((cbmin < cb) & (cb < cbmax))
    )
    result = image if inplace else image.copy()
    result[~valid_pixels] = 0
    return result


def aproximation_skin_color(image: np.ndarray, ycc: np.ndarray = None, inplace: bool = False):
    ymin = 16.0
    ymax = 235.0
    crmin = 142.0
    crmax = 162.0
    cbmin = 100.0
    cbmax = 119.0

    if image.dtype == np.float32:
        image = np.uint8(image * 255)

    if ycc is None:
        ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)

    y, cr, cb = np.moveaxis(ycc, 2, 0)

    ylim = ((ymin < y) & (y < ymax))
    crlim = ((crmin < cr) & (cr < crmax))
    cblim = ((ymin < y) & (y < ymax))