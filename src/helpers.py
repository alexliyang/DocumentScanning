import cv2

def show_image(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,img)


def start_vid():  # Program starts a video stream

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


def show_im(img):
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def take_pic(x=0):
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cap.release()
    if x == 1:
        show_im(frame)

    return frame

start_vid()


