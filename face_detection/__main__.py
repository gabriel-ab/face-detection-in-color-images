import numpy as np
import cv2 as cv

from .detect.eye_mouth import detect as detect_eye_mouth, EyesMouth
from .process.light import compensate_light
from .segment.skin_color import segment_skin
from . import utils

blobdet = cv.SimpleBlobDetector()

for raw in utils.stream(cv.VideoCapture(0)):
    compensated = compensate_light(raw)
    skin_mask = segment_skin(cv.cvtColor(compensated, cv.COLOR_BGR2HSV))
    segmented = cv.bitwise_and(compensated, compensated, mask=skin_mask)
    keypoints = blobdet.detect(skin_mask)
    
    cv.imshow('raw', raw)
    cv.imshow('segmented', segmented)
    if utils.is_window_closed('raw') or cv.waitKey(1) == utils.Q:
        break
