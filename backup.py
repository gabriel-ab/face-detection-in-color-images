from typing import Any
import cv2 as cv

camera = cv.VideoCapture(0)
QUIT_KEY = ord('q')

class FaceLocalization:
    def lighting_compensation(image): ...
    def color_space_transformation(image): ...
    def skin_color_detection(image): ...
    def variance_based_segmentation(image): ...
    def connected_component_and_grouping(image): ...

class FacialFeatureDetection:
    def eye_mouth_detection(image): ...
    def face_boundary_detection(image): ...
    def verifying_eye_mouth_triangles(image): ...

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
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    for image in stream(camera):
        key = cv.waitKey(1)
        if key == QUIT_KEY or is_window_closed(window): break
        cv.imshow(window, image)

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
