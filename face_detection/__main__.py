import numpy as np
import cv2 as cv

from .process.light import compensate_light
from .process.color_space import transform as transform_color
from .segment.skin import segment_skin
from .detect.skin import detect as detect_skin
from .detect.eye_mouth import detect as detect_face
from .detect import eye_mouth
from . import utils

def cvstream(*observe: str):
    for winname in observe:
        cv.namedWindow(winname)
    for raw in utils.stream(cv.VideoCapture(0)):
        if any(map(utils.is_window_closed, observe)) or cv.waitKey(1) == utils.Q:
            break
        yield raw



# if 1:
for raw in cvstream('raw'):
    # raw = cv.imread('temp/example.png')
    # raw = cv.imread('visualization/face.png')
    compensated = compensate_light(raw)

    skin_mask = segment_skin(cv.cvtColor(compensated, cv.COLOR_BGR2HSV))

    l,t,w,h = skin_rect = detect_skin(skin_mask)

    face = cv.rectangle(skin_mask.copy(), skin_rect, (255,0,0))

    segmented = cv.bitwise_and(compensated, compensated, mask=skin_mask)
    segmented[segmented==0] = 127

    face_ycbcr = cv.cvtColor(segmented, cv.COLOR_BGR2YCR_CB)[t:t+h, l:l+w]
    # face_ycbcr_color = transform_color(face_ycbcr)

    detections = segmented.copy()

    eyes = eye_mouth.eye_map(face_ycbcr)
    mouth = eye_mouth.mouth_map(face_ycbcr)

    # eyes = simplify(eyes)
    # mouth = cv.dilate(np.float32(mouth > 0.3), kernel, iterations=2)
    # mouth = simplify(mouth)

    # eyes = detector.detect(~eyes)
    # mouth = detector.detect(~mouth)

    # return filter_detections(ycbcr_image, eyes, mouth)
    # for face in detect_face(face_blob):
    #     face.draw(detections)
    #     print(face)


    cv.imshow('raw', raw)
    cv.imshow('face', face)
    cv.imshow('eyes', eyes)
    cv.imshow('mouth', mouth)

    # cv.imshow('compensated', compensated)
    # cv.imshow('segmented', segmented)
    # cv.imshow('face', canvas)
    # cv.imshow('detections', detections)
    # cv.waitKey()
    
