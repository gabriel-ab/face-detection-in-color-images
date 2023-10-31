from typing import Any
import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)
QUIT_KEY = ord('q')

def light_compensation(image: np.ndarray):
    ycc = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    reference_white_pixels = ycc[..., 0] > 0.95
    cv.imshow('Reference White', np.uint8(reference_white_pixels) * 255)

    if reference_white_pixels.sum() <= 0.02 * ycc.size / 3:
        return image

    reference_white = image[reference_white_pixels].mean(axis=0)
    lux_corrected_image = (image * (1 + (1 - reference_white))).clip(0, 1)
    return lux_corrected_image
    
def color_space_transformation(image: np.ndarray):
    ...
def skin_color_detection(image: np.ndarray): ...
def variance_based_segmentation(image: np.ndarray): ...
def connected_component_and_grouping(image: np.ndarray): ...

def eye_mouth_detection(image: np.ndarray): ...
def face_boundary_detection(image: np.ndarray): ...
def verifying_eye_mouth_triangles(image: np.ndarray): ...


def stream(camera: cv.VideoCapture):
    ok, image = camera.read()
    try:
        while ok:
            yield image
            ok, image = camera.read()
        else:
            print('Unable to start read video input')
    finally:
        camera.release()

def is_window_closed(winname: str) -> bool:
    return cv.getWindowProperty(winname, cv.WND_PROP_VISIBLE) < 1

def main():
    window = 'Image'
    window2 = 'Result'
    cv.namedWindow(window, cv.WINDOW_GUI_NORMAL)
    cv.namedWindow(window2, cv.WINDOW_GUI_NORMAL)

    for image in stream(camera):
        image = np.float32(image) / 255

        key = cv.waitKey(1)
        if key == QUIT_KEY or is_window_closed(window): break

        cv.imshow(window, image)
        # result = light_compensation(image)
        cv.imshow(window2, result)
    # image = cv.imread('project/image.png')
    # cv.imshow(window, image)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
