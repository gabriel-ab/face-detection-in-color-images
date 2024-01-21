import numpy as np
import cv2 as cv

from .process.light import compensate_light
from .segment.skin import segment_skin
from .detect.skin import detect as detect_skin
from .detect.eye_mouth import detect as detect_face
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
    skin_rect = detect_skin(skin_mask)
    print(skin_rect)
    canvas = cv.rectangle(skin_mask.copy(), skin_rect, (255,0,0))
    segmented = cv.bitwise_and(compensated, compensated, mask=skin_mask)
    l,t,w,h = skin_rect
    face_blob = cv.cvtColor(segmented, cv.COLOR_BGR2YCR_CB)[t:t+h, l:l+w]
    face_candidates = detect_face(face_blob)
    print(face_candidates)


    cv.imshow('raw', raw)
    cv.imshow('segmented', segmented)
    cv.imshow('face blob', face_blob)
    cv.imshow('canvas', canvas)
    cv.waitKey()
    
