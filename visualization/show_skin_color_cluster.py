"""Teste visual da detecção de Pele"""

from matplotlib import pyplot as plt

from pathlib import Path
import cv2 as cv
import numpy as np
from skimage.color import rgb2ycbcr

from face_detection.process.light import compensate_light
from face_detection.process.color_space import transform_cr, transform_cb

transform_cb = np.vectorize(transform_cb)
transform_cr = np.vectorize(transform_cr)



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
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1, projection='3d')
plt.gca().scatter(x,y,z, c=colors)
plt.gca().set(xlabel='CB', ylabel='CR', zlabel='Y')
plt.subplot(1, 2, 2)
plt.grid()
plt.scatter(x, y, c=colors)
plt.show()
