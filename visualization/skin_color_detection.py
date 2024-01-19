"""Teste visual da detecção de Pele"""

from matplotlib import pyplot as plt
from matplotlib import widgets as wid
from pathlib import Path
import cv2 as cv
import numpy as np
from face_detection.light_compensation import compensate_light
from face_detection.color_space_transformation import color_space_transformation
from face_detection.skin_color_detection import ycbcr_skin_detection

image_path = Path(__file__).parent / 'reference.png'


def pipeline(image: np.ndarray):
    yield image
    image = compensate_light(image)
    # yield image
    # image = color_space_transformation(image)
    yield image
    image = ycbcr_skin_detection(image)
    yield image


image = plt.imread(image_path)
images = tuple(pipeline(image))
plt.figure(figsize=(14,6), facecolor='black')
num_images = len(images)
for i, curr in enumerate(images, 1):
    plt.subplot(1, num_images, i, title=str(i)).imshow(curr).axes.axis(False)
plt.show()