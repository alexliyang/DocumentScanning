from adaptive_thresholding import adaptive_threshold
import cv2
import numpy as np
import transform
from rdp import rdp
from contour_corner_detection import draw_four_lines, find_intersections


def create_edge_image(image):
    """Take in an image and return a gray scale and edge image. Return an image with the most prominent edges"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # Convert to grayscale
    gray = cv2.GaussianBlur(gray, (15, 15), 0)          # Apply gaussian to remove noise
    edged = cv2.Canny(gray, 75, 200)                    # Use Canny edge detection to find the edges

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.namedWindow('Edged', cv2.WINDOW_NORMAL)
    cv2.imshow("Edged", edged)

    return gray, edged


def find_contours_from_threshold(gray):
    """Use a binary thresholding approach to find the contours of an image.
        In this application, the most important contours are the page edges"""

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('thresh', thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def largest_contour(contour_list):
    """Returns the contour with the largest area within a list of contours"""

    l = len(contour_list)
    x = np.zeros([1, l])

    for i in range(0, l):
        x[0][i] = cv2.contourArea(contour_list[i])

    max_index = x.argmax()
    return contour_list[max_index]


def find_corners_from_contours(page_contour):
    """Analyze the largest contour of the image and return the four corners of the document in the image"""

    # epsilon = 0.1 * cv2.arcLength(page_contour, True)
    epsilon = 0.00001 * cv2.arcLength(page_contour, True)
    page_approx = rdp(page_contour, epsilon)
    # page_approx = cv2.approxPolyDP(page_contour, epsilon, True)
    return page_approx.sum(axis=1), page_approx


def scan_page(image):
    """Takes an image input that has a document in it and return a birds eye view, high contrast scan of the document"""

    gray, edged = create_edge_image(image)

    contours = find_contours_from_threshold(gray)

    # This code assumes that the largest contour will contain the document of interest
    page_contour = largest_contour(contours)

    # Approximate corners, perspective project and threshold the image
    points, page_approx = find_corners_from_contours(page_contour)

    img, lines = draw_four_lines(image.copy(), page_contour)
    corners = find_intersections(lines, img.shape)

    # Add the contour onto original image and show it
    # cv2.drawContours(img, page_approx, -1, (0, 255, 0), 20)
    # cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
    for pt in corners:
        cv2.circle(img, (pt[0], pt[1]), 15, (0, 255, 0), -1)
        cv2.namedWindow('Corners2', cv2.WINDOW_NORMAL)

    cv2.imshow('Corners2', cv2.resize(img, (560, 710)))
    cv2.imwrite('../../Desktop/bad_corners.jpg', img)
    # cv2.imshow('Corners', cv2.resize(img.copy(), (560, 710)))

    # Apply a perspective transform to the document
    warped = transform.four_point_transform(image, np.array(corners))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
    cv2.imshow('warped', warped)

    # Apply an adaptive threshold on the image to remove contrasting shadows
    scanned_doc = adaptive_threshold(gray, type='adaptive')
    cv2.namedWindow('Scanned Document', cv2.WINDOW_NORMAL)
    # cv2.imshow('Scanned Document', scanned_doc)
    cv2.imshow('Scanned Document', cv2.resize(scanned_doc.copy(), (560, 710)))

    # Press "q" to close windows and end program
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    return scanned_doc


def main():
    # Read image
    # image = cv2.imread('../images/4point.jpg')
    # image = cv2.imread('../images/receipt.jpg')
    # image = cv2.imread('../images/note2.jpg')
    # image = cv2.imread('../images/angle.jpg')
    # image = cv2.imread('../images/keycard.jpg')
    # image = cv2.imread('../images/notes.jpg')

    # image = cv2.imread('../images/landscape.jpg')
    # scanned_doc = scan_page(image)

    cap = cv2.VideoCapture(1)
    # cap.set(3, 640)
    # cap.set(4, 480)
    # cap.set(15, 0.1)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('New', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    scanned_doc = scan_page(frame)

    cv2.imwrite('../images/scanned_notes.jpg', scanned_doc)

if __name__ == "__main__":
    main()
