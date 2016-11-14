import numpy as np
import cv2


def startVid():  # Program starts a video stream

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.flip(gray,1)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def showim(img):
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def takepic(x=0):
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cap.release()
    if x == 1:
        showim(frame)

    return frame

startVid()


