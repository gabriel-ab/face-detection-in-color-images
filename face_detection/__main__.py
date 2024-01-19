import numpy as np
import cv2 as cv

from .process.light import compensate_light
from .segment.skin import segment_skin
from .detect.skin import detect as detect_skin
from . import utils

def cvstream():
    for raw in utils.stream(cv.VideoCapture(0)):
        if utils.is_window_closed('raw') or cv.waitKey(1) == utils.Q:
            break
        yield raw



if 1:
    raw = cv.imread('temp/example.png')
    compensated = compensate_light(raw)
    skin_mask = segment_skin(cv.cvtColor(compensated, cv.COLOR_BGR2HSV))
    segmented = cv.bitwise_and(compensated, compensated, mask=skin_mask)
    keypoints = detect_skin(skin_mask)
    canvas = cv.drawKeypoints(skin_mask, keypoints, None, (255,0,0))
    print(keypoints)

    
    cv.imshow('raw', raw)
    cv.imshow('segmented', segmented)
    cv.imshow('canvas', canvas)
    cv.waitKey()
    
