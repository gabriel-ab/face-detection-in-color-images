import cv2 as cv
import numpy as np
import dataclasses as dc


detector = cv.SimpleBlobDetector.create()
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12,12))


@dc.dataclass
class EyesMouth:
    eyes: tuple[cv.KeyPoint, cv.KeyPoint]
    mouth: cv.KeyPoint
    def draw(self, image: np.ndarray):
        cv.circle(image, tuple(map(int, self.eyes[0].pt)), int(self.eyes[0].size), (0, 255, 0), 5)
        cv.circle(image, tuple(map(int, self.eyes[1].pt)), int(self.eyes[1].size), (0, 255, 0), 5)
        cv.circle(image, tuple(map(int, self.mouth.pt)), int(self.mouth.size), (0, 0, 255), 5)

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
    y_dilated = cv.dilate(y, kernel, iterations=2)
    y_eroded = cv.erode(y, kernel, iterations=2)
    eye_map_l = minmax_scale(y_dilated - y_eroded)
    return eye_map_l


def eye_map(ycbcr):
    y,cb,cr = np.moveaxis(ycbcr, -1, 0)
    eye_map = minmax_scale(eye_map_l(y) * eye_map_c(cb,cr))
    return eye_map


def mouth_map(ycbcr):
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]
    cr_by_cb = minmax_scale(cr / cb)
    cr2 = minmax_scale(cr**2)
    n = 0.95 * cr2.mean() / cr_by_cb.mean()
    mouth_map = minmax_scale(cr2 * (cr2 - n*cr_by_cb)**2)
    return mouth_map


def simplify(gray):
    gray = np.uint8(gray * 255)
    gray = cv.pyrDown(gray)
    gray = cv.pyrDown(gray)
    gray = cv.pyrUp(gray)
    gray = cv.pyrUp(gray)
    cv.threshold(gray, 127, 255, cv.THRESH_BINARY, gray)
    return gray


def detect(image):
    ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
    ycc = minmax_scale(ycc.astype(np.float32))

    eyes = eye_map(ycc)
    mouth = mouth_map(ycc)

    eyes = simplify(eyes)
    mouth = cv.dilate(np.float32(mouth > 0.3), kernel, iterations=2)
    mouth = simplify(mouth)

    eyes = detector.detect(~eyes) + (None,None)
    mouth = detector.detect(~mouth) + (None,)
    return EyesMouth(eyes[:2], mouth[0])


