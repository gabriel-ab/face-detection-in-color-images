"""Teste visual da detecção de Pele"""

from matplotlib import pyplot as plt
from pathlib import Path
import cv2 as cv

from face_detection.skin_color_detection import my_skin_color_detection

image_path = Path(__file__).parent / 'reference.png'
print(image_path)

image = plt.imread(image_path)
result = my_skin_color_detection(image)

plt.figure(figsize=(14,10))
plt.subplot(121, title='Original').imshow(image).axes.axis(False)
plt.subplot(122, title='After').imshow(result).axes.axis(False)
plt.show()