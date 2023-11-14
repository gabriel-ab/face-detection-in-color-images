import cv2 as cv
import numpy as np


def light_compensation(image: np.ndarray, ycc: np.ndarray | None = None):
    if ycc is None:
        ycc = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)

    scale = 1.0 if image.dtype == np.float32 else 255

    # Obtém os 5% de pixels com maior luma
    reference_white_pixels = ycc[..., 0] > 0.95 * scale

    # Caso não haja mais que 2% de pixels de referência
    if reference_white_pixels.sum() <= 0.02 * ycc.size / 3:
        return image

    # Obtém a cor média dos pixels em RGB
    reference_white = image[reference_white_pixels].mean(axis=0)

    # Escalona cada canal de cor para que o branco de referência (top 5%) seja (255, 255, 255)
    lux_corrected_image = (image * (scale + (scale - reference_white))).clip(0, scale).astype(image.dtype)

    return lux_corrected_image