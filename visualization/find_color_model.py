import cv2 as cv
import numpy as np
import skimage.color 

win = 'Find Color Model'
cv.namedWindow(win)

point = np.array(([157], [100])) / 255

cv.createTrackBar('CR', win, point[0], 235, lambda x: point.__setitem__(0, x))
cv.createTrackBar('CR', win, point[1], 235, lambda x: point.__setitem__(1, x))

def skin_color_detection(ycrcb):
    points = ycrcb.transpose(2, 0, 1).reshape(3, -1)
    indices = np.linalg.norm(point - points[1:3], axis=0) < 0.07
    indices = indices.reshape(*ycrcb.shape[:2])
    return indices

image = cv.imread('visualization/reference.png')
ycbcr = skimage.color.rgb2ycbcr(image)
