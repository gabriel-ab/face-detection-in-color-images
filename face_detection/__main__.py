import argparse
from skimage import color
import numpy as np
import cv2 as cv

from .process.light import compensate_light
from .process.color_space import transform as transform_color
from .segment.skin import hsv_segment_skin, ycbcr_segment_skin
from .detect.skin import detect as detect_skin
from .detect.eye_mouth import detect as detect_face
from .detect import eye_mouth
from . import utils

p = argparse.ArgumentParser()
p.add_argument('--example', action='store_true')
args = p.parse_args()

def cvstream(*observe: str):
    camera = cv.VideoCapture(0)
    camera.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
    camera.set(cv.CAP_PROP_AUTO_WB, 0)
    for winname in observe:
        cv.namedWindow(winname)
    for raw in utils.stream(camera):
        if any(map(utils.is_window_closed, observe)) or cv.waitKey(1) == utils.Q:
            break
        yield raw


def imshow_rgb(name: str, rgb: np.ndarray):
    cv.imshow(name, cv.cvtColor(rgb, cv.COLOR_RGB2BGR))


def example_stream():
    yield cv.imread('visualization/face.png')
    cv.waitKey()

image_stream = example_stream() if args.example else cvstream('compensated_rgb', 'skin_mask', 'eye_mouth_blobs', 'skin_rect', 'face_detection')

for bgr in image_stream:
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    compensated_rgb = compensate_light(rgb)

    ycbcr = color.rgb2ycbcr(compensated_rgb)
    transformed_ycbcr = transform_color(ycbcr)

    skin_mask = ycbcr_segment_skin(transformed_ycbcr)
    l,t,w,h = skin_rect = detect_skin(skin_mask)

    face_rgb = compensated_rgb[t:t+h, l:l+w].copy()
    face_ycbcr = transformed_ycbcr[t:t+h, l:l+w].copy()
    face_mask = skin_mask[t:t+h, l:l+w]

    eyes: np.ndarray = eye_mouth.eye_map(face_ycbcr)
    eyes[face_mask == 0] = 0
    eyes = eye_mouth.simplify_eye_map(eyes)

    mouth: np.ndarray = eye_mouth.mouth_map(face_ycbcr)
    mouth[face_mask == 0] = 0
    mouth = eye_mouth.simplify_mouth_map(mouth)

    eyes_det: list[cv.KeyPoint] = eye_mouth.detector.detect(eyes)
    mouth_det: list[cv.KeyPoint] = eye_mouth.detector.detect(mouth)

    faces: list[eye_mouth.EyesMouth] = eye_mouth.filter_detections(face_rgb, eyes_det, mouth_det)

    for face in faces:
        face.draw(face_rgb)

    imshow_rgb('compensated_rgb', compensated_rgb)
    imshow_rgb('skin_mask', np.uint8(skin_mask * 255))
    imshow_rgb('eye_mouth_blobs', np.stack([eyes, mouth, np.zeros_like(eyes)], axis=-1))
    imshow_rgb('skin_rect', cv.rectangle(rgb.copy(), skin_rect, (0,255,0), 2))
    imshow_rgb('face_detection', face_rgb)

    if faces:
        print(faces)

