import cv2
import numpy as np
import math


def order_points(pts):
    """Take the four corner points and return a rectangle that fits the point values in order
        from top left, top right, bottom right, bottom left"""

    rect = np.zeros((4, 2), dtype="float32")

    point_sum = pts.sum(axis=1)

    rect[0] = pts[np.argmin(point_sum)]     # Point with smallest sum is the top left point
    rect[2] = pts[np.argmax(point_sum)]     # Point with largest sum is the bottom right point

    point_diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(point_diff)]    # Top write point will have smallest difference
    rect[3] = pts[np.argmax(point_diff)]    # Bottom left point will have largest difference

    return rect


def four_point_transform(image, pts):
    """Compute the rectangle dimensions of a birds eye view of the document and do a perspective transform
        on the image to cut out everything outside the document boundaries."""

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image, where we compare the distance between tl/tr and br/bl and choose the max
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    max_width = int(max_width + 0.5)
    # Compute height of new image, where we compare the distance between tl/bl and tr/br and choose the max
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    max_height = int(max_height + 0.5)
    # Create the dimension of a new image with the points specified top left, top right, bottom right, bottom left
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")

    # Compute the perspective transform matrix and then apply it
    perspective_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, perspective_matrix, (max_width, max_height))
    # warped = warpPerspective(image, perspective_matrix, max_width, max_height)
    return warped


def warpPerspective(image, perspective_matrix, max_width, max_height):
    new_img = np.zeros((max_height, max_width, 3), np.uint8)
    for y in range(0, max_height):
        for x in range(0, max_width):
            xx = (perspective_matrix[0,0]*x + perspective_matrix[0, 1]*y + perspective_matrix[0, 2])/(perspective_matrix[2,0]*x + perspective_matrix[2, 1]*y + perspective_matrix[2, 2])
            yy = (perspective_matrix[1, 0] * x + perspective_matrix[1, 1] * y + perspective_matrix[1, 2]) / (
            perspective_matrix[2, 0] * x + perspective_matrix[2, 1] * y + perspective_matrix[2, 2])
            new_img[y, x] = image[yy, xx]
    return new_img
