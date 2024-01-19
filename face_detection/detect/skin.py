from typing import Sequence
import cv2 as cv
import numpy as np

def load():
    params = cv.SimpleBlobDetector.Params()
    params.blobColor = 255
    return cv.SimpleBlobDetector.create(params)

detector = load()

def detect(skin_mask: np.ndarray) -> Sequence[cv.KeyPoint]:
    return detector.detect(skin_mask)