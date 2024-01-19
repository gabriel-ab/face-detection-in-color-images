from face_detection.detect import eye_mouth as module
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    image = cv.imread('visualization/face.png')
    for face in module.detect(image):
        face.draw(image)
    cv.imshow('Final Detection', image)
    cv.waitKey()

    # video = cv.VideoCapture(0)
    # ok, image, = video.read()
    # while ok:
    #     for face in module.detect(image):
    #         face.draw(image)
    #     cv.imshow('Final Detection', image)
    #     cv.waitKey(1)
    #     ok, image, = video.read()
    

