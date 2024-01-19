from face_detection.detect import (
    eye_mouth
)
import cv2 as cv
import numpy as np

from face_detection.process import color_space, light
from face_detection.segment import skin_color

def process(image: np.ndarray):
    image = light.compensate_light(image)
    image = color_space.color_space_transformation(image)
    skin_color.ycbcr_skin_detection()



if __name__ == '__main__':

    video = cv.VideoCapture(0)
    ok, image, = video.read()
    while ok:
        for face in module.detect(image):
            face.draw(image)
        cv.imshow('Final Detection', image)
        cv.waitKey(1)
        ok, image, = video.read()