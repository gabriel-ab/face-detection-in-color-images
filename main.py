from functools import lru_cache
import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)
QUIT_KEY = ord('q')



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
