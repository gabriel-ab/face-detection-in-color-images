from typing import Sequence, NamedTuple
import cv2 as cv
import numpy as np


def load():
    params = cv.SimpleBlobDetector.Params()
    params.blobColor = 255
    return cv.SimpleBlobDetector.create(params)

detector = load()

class Rect(NamedTuple):
    left: int
    top: int
    width: int
    height: int


def detect(skin_mask: np.ndarray):
    # Find the connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(skin_mask)

    # Find the index of the largest component
    largest_component_index = 0
    largest_area = 0
    largest_component_rect = Rect(0,0,0,0)

    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > largest_area:
            largest_component_index = i
            largest_area = area
            largest_component_rect = Rect(
                left = int(stats[i, cv.CC_STAT_LEFT]),
                top = int(stats[i, cv.CC_STAT_TOP]),
                width = int(stats[i, cv.CC_STAT_WIDTH]),
                height = int(stats[i, cv.CC_STAT_HEIGHT]),
            )

    # Get the coordinates of the largest component
    return largest_component_rect

