import cv2
import numpy as np
from transform import *
from adaptive_thresholding import adaptive_threshold

# image = cv2.imread('4point.jpg')
# image = cv2.imread('receipt.jpg')
# image = cv2.imread('note2.jpg')
# image = cv2.imread('angle.jpg')
image = cv2.imread('keycard.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.namedWindow('Edged', cv2.WINDOW_NORMAL)
cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

ret, thresh = cv2.threshold(gray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contours = sorted(contours,cv2.contourArea,reverse=True)[:5]

l = len(contours)

x = np.zeros([1, l])

for i in range(0, l):
    x[0][i] = cv2.contourArea(contours[i])

max_index = x.argmax()
page_contour = contours[max_index]

epsilon = 0.1 * cv2.arcLength(page_contour, True)
page_approx = cv2.approxPolyDP(page_contour, epsilon, True)
points = page_approx.sum(axis=1)
warped = four_point_transform(image,points)
#
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
thresh, warped = adaptive_threshold(gray, type='adaptive')
#
#
# warped = four_point_transform(image,points)

cv2.drawContours(image, page_approx, -1, (0, 0, 255), 20)
cv2.namedWindow('Contour', cv2.WINDOW_NORMAL)
cv2.imshow('Contour', image)
cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
cv2.imshow('Warped',warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
