import cv2 as cv

def stream(video: cv.VideoCapture):
    ok, image = video.read()
    try:
        while ok:
            yield image
            ok, image = video.read()
        else:
            print('Unable to start read video input')
    finally:
        video.release()


def is_window_closed(winname: str) -> bool:
    return cv.getWindowProperty(winname, cv.WND_PROP_VISIBLE) < 1

Q = ord('q')
