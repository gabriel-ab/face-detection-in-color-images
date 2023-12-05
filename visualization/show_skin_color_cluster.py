"""Teste visual da detecção de Pele"""

from matplotlib import pyplot as plt

from pathlib import Path
import cv2 as cv
import numpy as np
from skimage.color import rgb2ycbcr

from face_detection.light_compensation import light_compensation
from face_detection.color_space_transformation import color_space_transformation, CR, CB


transform_cr = np.vectorize(CR.transform)
transform_cb = np.vectorize(CB.transform)

def pipeline(image: np.ndarray):
    yield image
    image = light_compensation(image)
    yield image
    image = color_space_transformation(image)
    yield image

rgb = np.mgrid[:10, :10, :10] / 9
rgb = rgb.transpose(1,2,3,0).astype(np.float32)
ycrcb = rgb2ycbcr(rgb)
ycrcb[..., 1] = transform_cr(ycrcb[..., 0], ycrcb[..., 1])
ycrcb[..., 2] = transform_cb(ycrcb[..., 0], ycrcb[..., 2])
z,x,y = ycrcb.transpose(3, 0, 1, 2)

point = np.array(([150], [110]))
ycrcb_points = np.swapaxes(ycrcb, 0, -1).reshape(3, -1)
indices = np.linalg.norm(point - ycrcb_points[1:3], axis=0) < 40
# indices = indices.reshape(*ycbcr.shape[:3])

colors = np.empty(indices.shape + (4,))
colors[indices] = (0.9, 0.2, 0.2, 0.8)
colors[~indices] = (0.2, 0.2, 0.9, 0.8)

ax = plt.axes(projection='3d')
ax.scatter(x,y,z, c=colors)
ax.set(xlabel='CB', ylabel='CR', zlabel='Y')
plt.show()
