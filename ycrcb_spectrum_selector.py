from functools import lru_cache
import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)
QUIT_KEY = ord('q')

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
    ycrcb = np.zeros(6)
    cv.createTrackbar('> Y', window2, 000, 255, lambda x: ycrcb.__setitem__(0, x))
    cv.createTrackbar('< Y', window2, 255, 255, lambda x: ycrcb.__setitem__(1, x))
    cv.createTrackbar('> Cr', window2, 000, 255, lambda x: ycrcb.__setitem__(2, x))
    cv.createTrackbar('< Cr', window2, 255, 255, lambda x: ycrcb.__setitem__(3, x))
    cv.createTrackbar('> Cb', window2, 000, 255, lambda x: ycrcb.__setitem__(4, x))
    cv.createTrackbar('< Cb', window2, 255, 255, lambda x: ycrcb.__setitem__(5, x))

    def process(image):
        ycrcb_image = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
        ymin, ymax, crmin, crmax, cbmin, cbmax = ycrcb / 255
        y, cr, cb = np.moveaxis(ycrcb_image, 2, 0)
        # ycrcb_image[..., 0][~((ymin < y) & (y < ymax))] = 0
        # ycrcb_image[..., 1][~((crmin < cr) & (cr < crmax))] = 0
        # ycrcb_image[..., 2][~((cbmin < cb) & (cb < cbmax))] = 0
        # result = cv.cvtColor(ycrcb_image, cv.COLOR_YCR_CB2BGR)
        selected_pixels = ~((ymin < y) & (y < ymax)) | ~((crmin < cr) & (cr < crmax)) | ~((cbmin < cb) & (cb < cbmax))
        image[selected_pixels] = 0
        result = image
        return result


    for image in stream(camera):
        image = np.float32(image) / 255

        key = cv.waitKey(1)
        if key == QUIT_KEY or is_window_closed(window):
            ymin, ymax, crmin, crmax, cbmin, cbmax = ycrcb

            with open('result.txt', 'w') as f:
                f.write(
                    f'{ymin = }\n'
                    f'{ymax = }\n'
                    f'{crmin = }\n'
                    f'{crmax = }\n'
                    f'{cbmin = }\n'
                    f'{cbmax = }\n'
                )
            break

        cv.imshow(window, image)
        result = process(image)
        # result = light_compensation(image)
        cv.imshow(window2, result)
    # image = cv.imread('project/image.png')
    # cv.imshow(window, image)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
