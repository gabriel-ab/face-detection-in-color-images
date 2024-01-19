from face_detection import (
    eye_mouth_detection, color_space_transformation,
    light_compensation, skin_color_detection
)
import cv2 as cv
import numpy as np

def process(image: np.ndarray):
    image = light_compensation.compensate_light(image)
    image = color_space_transformation.color_space_transformation(image)
    skin_color_detection.ycbcr_skin_detection()



if __name__ == '__main__':

    video = cv.VideoCapture(0)
    ok, image, = video.read()
    while ok:
        for face in module.detect(image):
            face.draw(image)
        cv.imshow('Final Detection', image)
        cv.waitKey(1)
        ok, image, = video.read()