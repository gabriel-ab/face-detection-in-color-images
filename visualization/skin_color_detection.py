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

def skin_color_detection(ycc):
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    cr_selection = (130 < cr) & (cr < 155)
    cb_selection = (80 < cb) & (cb < 140)
    return cr_selection | cb_selection


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