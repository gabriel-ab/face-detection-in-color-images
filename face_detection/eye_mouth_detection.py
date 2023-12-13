import itertools
import math
import cv2 as cv
import numpy as np
import dataclasses as dc


detector = cv.SimpleBlobDetector.create()
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12,12))


@dc.dataclass
class EyesMouth:
    left_eye: cv.KeyPoint
    right_eye: cv.KeyPoint
    mouth: cv.KeyPoint

    def center(self) -> tuple[int, int]:
        xy = np.mean([
            self.left_eye.pt,
            self.right_eye.pt,
            self.mouth.pt,
        ], axis=0)
        return xy

    def draw(self, image: np.ndarray):
        left_eye = np.array(self.left_eye.pt).astype(int)
        right_eye = np.array(self.right_eye.pt).astype(int)
        mouth = np.array(self.mouth.pt).astype(int)
        forehead = np.mean([left_eye, right_eye], axis=0)
        angle = math.atan2( forehead[1] - mouth[1], forehead[0] - mouth[0] ) * ( 180 / math.pi )
        cv.line(image, left_eye, right_eye, (255,255,255), 2)
        cv.line(image, right_eye, mouth, (255,255,255), 2)
        cv.line(image, mouth, left_eye, (255,255,255), 2)
        cv.ellipse(image, self.center().astype(int), (180,120), angle, 0, 360, (0,255,0), 2)


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
    # cv.GaussianBlur(gray, (5,5), 1, gray)
    gray = cv.pyrDown(gray)
    gray = cv.pyrDown(gray)
    gray = cv.pyrUp(gray)
    gray = cv.pyrUp(gray)
    cv.threshold(gray, 127, 255, cv.THRESH_BINARY, gray)
    return gray

def filter_detections(image: np.ndarray, eyes: tuple[cv.KeyPoint, ...], mouth: tuple[cv.KeyPoint, ...]) -> list[EyesMouth]:
    height, width = image.shape[:2]

    def is_good_eye_combination(left_eye: cv.KeyPoint, right_eye: cv.KeyPoint) -> bool:
        # is ordered? (is left on left? and right on right?)
        is_ordered = left_eye.pt[0] < right_eye.pt[0]

        # is horizontaly aligned? (tolerance: 10% of the image height)
        is_horizontaly_aligned = (abs(left_eye.pt[1] - right_eye.pt[1]) / height) < 0.1

        return is_ordered and is_horizontaly_aligned

    def is_good_eye_mouth_combination(left_eye: cv.KeyPoint, right_eye: cv.KeyPoint, mouth: cv.KeyPoint) -> bool:
        eye_to_eye_distance = math.dist(left_eye.pt, right_eye.pt)
        eye_to_mouth_distance = (math.dist(left_eye.pt, mouth.pt) + math.dist(right_eye.pt, mouth.pt)) / 2
        ratio = eye_to_mouth_distance / eye_to_eye_distance

        # the distance of eye to mouth must be equal to or, at maximum, twice as big as eye to eye
        return 1 < ratio < 2

    good_eyes_candidates = [eyes for eyes in itertools.combinations(eyes, 2) if is_good_eye_combination(*eyes)]

    return [
        EyesMouth(l, r, m)
        for (l, r), m in itertools.product(good_eyes_candidates, mouth)
        if is_good_eye_mouth_combination(l, r, m)
    ]


def detect(image: np.ndarray) -> list[EyesMouth]:
    ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
    ycc = minmax_scale(ycc.astype(np.float32))

    eyes = eye_map(ycc)
    mouth = mouth_map(ycc)

    eyes = simplify(eyes)
    mouth = cv.dilate(np.float32(mouth > 0.3), kernel, iterations=2)
    mouth = simplify(mouth)

    eyes = detector.detect(~eyes)
    mouth = detector.detect(~mouth)

    return filter_detections(image, eyes, mouth)

