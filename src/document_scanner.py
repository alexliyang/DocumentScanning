from adaptive_thresholding import adaptive_threshold
import cv2
import numpy as np
import transform

def largest_contour(contour_list):
    """Returns the contour with the largest area within a list of contours"""

    l = len(contour_list)
    x = np.zeros([1, l])

    for i in range(0, l):
        x[0][i] = cv2.contourArea(contour_list[i])

    max_index = x.argmax()
    return contour_list[max_index]


def main():
    # Read image
    # image = cv2.imread('../images/4point.jpg')
    # image = cv2.imread('../images/receipt.jpg')
    # image = cv2.imread('../images/note2.jpg')
    # image = cv2.imread('../images/angle.jpg')
    image = cv2.imread('../images/keycard.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  Apply gaussian three times to make sure to get rid of noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection to find the edges
    edged = cv2.Canny(gray, 75, 200)

    # Show the Original Image
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.namedWindow('Edged', cv2.WINDOW_NORMAL)
    cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('thresh',thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # This code assumes that the largest contour will contain the document of interest
    page_contour = largest_contour(contours)

    # Approximate corners, perspective project and threshold the image
    epsilon = 0.1 * cv2.arcLength(page_contour, True)
    page_approx = cv2.approxPolyDP(page_contour, epsilon, True)
    points = page_approx.sum(axis=1)
    warped = transform.four_point_transform(image,points)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warped = adaptive_threshold(gray, type='adaptive')

    # Add the contour onto original image and show it
    cv2.drawContours(image, page_approx, -1, (0, 0, 255), 20)
    cv2.imshow('Image', image)

    # Show the perspective transformed image
    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
    cv2.imshow('Warped',warped)

    # Press "q" to close windows and end program
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
