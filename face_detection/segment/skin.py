import numpy as np
import cv2 as cv


ycbcr_points = np.array(([150], [110]))

def ycbcr_skin_detection(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)

    # (height, width, 2) -> (2, height, width) -> (2, height * width) = ([cr,cb], num_pixels)
    crcb_points = np.swapaxes(ycrcb, 0, -1).reshape(3, -1)

    # Calculating if is inside cluster of skin tones
    indices = np.linalg.norm(ycbcr_points - crcb_points[1:3], axis=0) < 40
    indices = indices.reshape(*ycrcb.shape[:2])
    return indices


hsv_lower = np.array([0, 48, 80], dtype=np.uint8)
hsv_upper = np.array([20, 255, 255], dtype=np.uint8)
hsv_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 13))

def segment_skin(hsv_image: np.ndarray):
    skin_mask = cv.inRange(hsv_image, hsv_lower, hsv_upper)
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, hsv_kernel, iterations=5)
    # skin_mask = cv.dilate(skin_mask, hsv_kernel, iterations = 4)
    # skin_mask = cv.erode(skin_mask, hsv_kernel, iterations = 2)
    skin_mask = cv.GaussianBlur(skin_mask, (3, 3), 0)
    return skin_mask