import cv2 as cv

video = cv.VideoCapture(0)

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


def main():
    QUIT = ord('q')
    TAKE_PHOTO = ord(' ')

    for image in stream(video):
        cv.imshow('Test', image)
        key = cv.waitKey(1)
        if key == QUIT:
            break
        elif key == TAKE_PHOTO:
            print('photo taken')
            cv.imwrite('image.png', image)

    cv.destroyAllWindows()



if __name__ == '__main__':
    main()