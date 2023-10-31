import numpy as np


a = np.sin(2.53)
b = np.cos(2.53)
skin_model = np.array([[ b, a],[-a, b]])

def skin_color_detection(ycc: np.ndarray):
    xy = skin_model