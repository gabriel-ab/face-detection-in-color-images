"""Teste da correção de iluminação"""

from matplotlib import pyplot as plt
from pathlib import Path
from face_detection.process.light import compensate_light

image_path = Path(__file__).parent / 'reference.png'

image = plt.imread(image_path)
result = compensate_light(image)

plt.figure(figsize=(14,10))
plt.subplot(121, title='Original').imshow(image).axes.axis(False)
plt.subplot(122, title='Corrected').imshow(result).axes.axis(False)
plt.show()