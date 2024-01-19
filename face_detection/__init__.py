import numpy as np
import cv2 as cv

from .eye_mouth_detection import detect as detect_eye_mouth, EyesMouth
from .light_compensation import compensate_light
from .skin_color_detection import segment_skin
from . import utils

def detect(image: np.ndarray):
    
    return image


if __name__ == '__main__':
    # img = cv.imread('visualization/reference.png')
    # img = detect(img)
    # cv.imshow('debug', image)
    # cv.waitKey(-1)
    
    for raw in utils.stream(cv.VideoCapture(0)):
        compensated = compensate_light(raw)
        skin_mask = segment_skin(cv.cvtColor(compensated, cv.COLOR_BGR2HSV))
        segmented = cv.bitwise_and(compensated, compensated, mask=skin_mask)
        detect_eye_mouth()

        
        cv.imshow('raw', raw)
        cv.imshow('segmented', segmented)
        if utils.is_window_closed('raw') or cv.waitKey(1) == utils.Q:
            break


