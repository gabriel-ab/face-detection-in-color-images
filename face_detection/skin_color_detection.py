import numpy as np
import cv2 as cv



point = np.array(([150], [110]))

def skin_color_detection(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)

    # (height, width, 2) -> (2, height, width) -> (2, height * width) = ([cr,cb], num_pixels)
    crcb_points = ycrcb[..., 1:3].transpose(2, 0, 1).reshape(2, -1)

    # Calculating if is inside cluster of skin tones
    indices = np.linalg.norm(point - crcb_points, axis=0) < 40
    indices = indices.reshape(*ycrcb.shape[:2])

    result = image.copy()
    result[~indices] = 0
    return result