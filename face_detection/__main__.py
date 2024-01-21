import numpy as np
import cv2 as cv

from .process.light import compensate_light
from .segment.skin import segment_skin
from .detect.skin import detect as detect_skin
from .detect.eye_mouth import detect as detect_face
from . import utils

def cvstream(*observe: str):
    for winname in observe:
        cv.namedWindow(winname)
    for raw in utils.stream(cv.VideoCapture(0)):
        if any(map(utils.is_window_closed, observe)) or cv.waitKey(1) == utils.Q:
            break
        yield raw



# if 1:
for raw in cvstream('raw', 'face', 'detections'):
    # raw = cv.imread('temp/example.png')
    # raw = cv.imread('visualization/face.png')
    compensated = compensate_light(raw)
    skin_mask = segment_skin(cv.cvtColor(compensated, cv.COLOR_BGR2HSV))
    # skin_mask = segment_skin(cv.cvtColor(raw, cv.COLOR_BGR2HSV))
    l,t,w,h = skin_rect = detect_skin(skin_mask)
    print(skin_rect)
    canvas = cv.rectangle(skin_mask.copy(), skin_rect, (255,0,0))
    segmented = cv.bitwise_and(compensated, compensated, mask=skin_mask)
    # segmented = cv.bitwise_and(raw, raw, mask=skin_mask)
    segmented[segmented==0] = 127
    face_blob = cv.cvtColor(segmented, cv.COLOR_BGR2YCR_CB)[t:t+h, l:l+w]
    # # face_blob = segmented[t:t+h, l:l+w]
    # print(face_blob.min(), face_blob.mean(), face_blob.max())
    # face_candidates = detect_face(face_blob)
    detections = segmented.copy()
    # detections = raw.copy()
    # face_blob = cv.cvtColor(raw, cv.COLOR_BGR2YCR_CB)
    for face in detect_face(face_blob):
        face.draw(detections)
        print(face)


    cv.imshow('raw', raw)
    cv.imshow('compensated', compensated)
    cv.imshow('segmented', segmented)
    cv.imshow('face', canvas)
    cv.imshow('detections', detections)
    # cv.waitKey()
    
