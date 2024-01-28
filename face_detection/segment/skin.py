import numpy as np
import cv2 as cv

ycbcr_points = np.array(([150], [110]))
ycbcr_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (16,16))

def ycbcr_segment_skin(ycbcr_image: np.ndarray):
    # (height, width, 3) -> (3, height, width) -> (3, height * width) = ([y,cb,cr], num_pixels)
    cbcr_points = ycbcr_image[..., [1,2]].transpose(2, 0, 1).reshape(2, -1)

    # Calculating if is inside cluster of skin tones
    mask = np.linalg.norm(ycbcr_points - cbcr_points, axis=0) > 40
    mask = mask.reshape(*ycbcr_image.shape[:2]).astype(np.uint8)
    cv.morphologyEx(mask, cv.MORPH_CLOSE, ycbcr_kernel, mask, iterations=2)
    cv.erode(mask, ycbcr_kernel, mask)
    return mask


hsv_lower = np.array([0, 48, 80], dtype=np.uint8)
hsv_upper = np.array([20, 255, 255], dtype=np.uint8)
hsv_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 13))

def hsv_segment_skin(hsv_image: np.ndarray):
    skin_mask = cv.inRange(hsv_image, hsv_lower, hsv_upper)
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, hsv_kernel, iterations=5)
    # skin_mask = cv.dilate(skin_mask, hsv_kernel, iterations = 4)
    # skin_mask = cv.erode(skin_mask, hsv_kernel, iterations = 2)
    skin_mask = cv.GaussianBlur(skin_mask, (3, 3), 0)
    return skin_mask