from functools import lru_cache
import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)
QUIT_KEY = ord('q')

def light_compensation(image: np.ndarray):
    ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
    reference_white_pixels = ycc[..., 0] > (0.95 if isinstance(ycc, np.floating) else 0.95 * 255)
    # cv.imshow('Reference White', np.uint8(reference_white_pixels) * 255)

    if reference_white_pixels.sum() <= 0.02 * ycc.size / 3:
        return image

    reference_white = image[reference_white_pixels].mean(axis=0)
    lux_corrected_image = (image * (1 + (1 - reference_white))).clip(0, 1)
    return lux_corrected_image


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
        return 154 + (y - cls.kh) * (154 - 132) / (cls.ymax - cls.kh)

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
    ycc = cv.cvtColor(image, cv.COLOR_YCR_CB2RGB)
    if image.mean() < 1:
        ycc = np.uint8(ycc * 255)

    for i, line in enumerate(ycc):
        for j, (y, cr, cb) in enumerate(line):
            ycc[i, j, 1] = CR.transform(y, cr)
            ycc[i, j, 2] = CB.transform(y, cb)

    return cv.cvtColor(ycc, cv.COLOR_YCR_CB2RGB)

def skin_color_detection(image: np.ndarray): ...
def variance_based_segmentation(image: np.ndarray): ...
def connected_component_and_grouping(image: np.ndarray): ...

def eye_mouth_detection(image: np.ndarray): ...
def face_boundary_detection(image: np.ndarray): ...
def verifying_eye_mouth_triangles(image: np.ndarray): ...


def stream(camera: cv.VideoCapture):
    ok, image = camera.read()
    try:
        while ok:
            yield image
            ok, image = camera.read()
        else:
            print('Unable to start read video input')
    finally:
        camera.release()

def is_window_closed(winname: str) -> bool:
    return cv.getWindowProperty(winname, cv.WND_PROP_VISIBLE) < 1

def main():
    window = 'Image'
    window2 = 'Result'
    cv.namedWindow(window, cv.WINDOW_GUI_NORMAL)
    cv.namedWindow(window2, cv.WINDOW_GUI_NORMAL)

    for image in stream(camera):
        image = np.float32(image) / 255

        key = cv.waitKey(1)
        if key == QUIT_KEY or is_window_closed(window): break

        cv.imshow(window, image)
        # result = light_compensation(image)
        cv.imshow(window2, result)
    # image = cv.imread('project/image.png')
    # cv.imshow(window, image)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
