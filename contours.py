from transform import *

def largest_contour(contour_list):
    """Returns the contour with the largest area within a list of contours"""

    l = len(contour_list)
    x = np.zeros([1, l])

    for i in range(0, l):
        x[0][i] = cv2.contourArea(contour_list[i])

    max_index = x.argmax()
    return contour_list[max_index]

# Read image
image = cv2.imread('4point.jpg')
# image = cv2.imread('receipt.jpg')
# image = cv2.imread('note2.jpg')
# image = cv2.imread('angle.jpg')
# image = cv2.imread('keycard.jpg')
orig = image.copy()

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

ret, thresh = cv2.threshold(gray, 127, 255, 0)
cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh',thresh)

(_, contours, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = largest_contour(contours)

# Anton PLS MAKE A FUNCTION THAT FITS 4 LINES to points in variable 'contours', and then returns the corners


# show the original image and the edge detected image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.namedWindow("Edged", cv2.WINDOW_NORMAL)
cv2.imshow("Edged", edged)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

