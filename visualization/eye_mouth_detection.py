from face_detection import eye_mouth_detection as module
import cv2 as cv
import numpy as np

def main(image):
    
    ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
    ycc = module.minmax_scale(ycc.astype(np.float32))

    eyes = module.eye_map(ycc)
    mouth = module.mouth_map(ycc)
    cv.imshow('Eye Detection', eyes)
    cv.imshow('Mouth Detection', mouth)

    eyes = module.simplify(eyes)
    mouth = cv.dilate(np.float32(mouth > 0.3), module.kernel, iterations=2)
    mouth = module.simplify(mouth)
    cv.imshow('Eyes Simplified', eyes)
    cv.imshow('Mouth Simplified', mouth)

    eyes = module.detector.detect(~eyes) + (None,None)
    mouth = module.detector.detect(~mouth) + (None,)
    module.EyesMouth(eyes, mouth[0]).draw(image)
    cv.imshow('Final Detection', image)
    cv.waitKey()

if __name__ == '__main__':
    image = cv.imread('visualization/face.png')
    main(image)
