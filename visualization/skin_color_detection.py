"""Teste visual da detecção de Pele"""

from matplotlib import pyplot as plt
from matplotlib import widgets as wid
from pathlib import Path
import cv2 as cv
import numpy as np
from face_detection.light_compensation import light_compensation
from face_detection.color_space_transformation import color_space_transformation

image_path = Path(__file__).parent / 'reference.png'
print(image_path)

point = np.array(([150], [110]))
def skin_color_detection(ycc):
    crcb_points = ycc[..., 1:3].transpose(2, 0, 1).reshape(2, -1)
    result = np.linalg.norm(point - crcb_points, axis=0) > 40
    result = result.reshape(*ycc.shape[:2])
    return result


image = plt.imread(image_path)
image2 = light_compensation(image)
image3 = cv.cvtColor(image2, cv.COLOR_RGB2YCR_CB)
image4 = color_space_transformation(image3)
image5 = image2.copy()
image5[~skin_color_detection(image4)] = 0

plt.figure(figsize=(14,10))
plt.subplot(151, title='1').imshow(image).axes.axis(False)
plt.subplot(152, title='2').imshow(image2).axes.axis(False)
plt.subplot(153, title='3').imshow(image3).axes.axis(False)
plt.subplot(154, title='4').imshow(image4).axes.axis(False)
plt.subplot(155, title='5').imshow(image5).axes.axis(False)
plt.show()