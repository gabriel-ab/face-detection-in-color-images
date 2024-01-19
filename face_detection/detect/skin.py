from typing import Sequence
import cv2 as cv
import numpy as np
import dataclasses as dc

def load():
    params = cv.SimpleBlobDetector.Params()
    params.blobColor = 255
    return cv.SimpleBlobDetector.create(params)

detector = load()

@dc.dataclass(slots=True)
class Rect:
    left: int
    top: int
    width: int
    height: int
    centroid: tuple[float, float]


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
                centroid = tuple(centroids[i, :]),
            )

    # Get the coordinates of the largest component
    return largest_component_rect

