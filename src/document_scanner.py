from adaptive_thresholding import adaptive_threshold
import cv2
import numpy as np
import transform
import math


# initialize the list of reference points. This list will be populated by the points picked by the user
# refPt = set([])

def hough_transform(edge):

    import numpy as np
    import scipy.misc

    height, width = edge.shape
    max_distance = np.sqrt(height ** 2 + width ** 2)
    rhos = np.arange(-max_distance, max_distance, 1)
    thetas = np.deg2rad(np.arange(-90,90))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_coord, x_coord = np.nonzero(edge)

    for i in range(len(y_coord)):
        x = x_coord[i]
        y = y_coord[i]

        for t_coord in range(num_thetas):
            rho = int(round(x * cos_t[t_coord] + y * sin_t[t_coord]) + max_distance)
            accumulator[rho, t_coord] += 1

    accumulator = np.log(accumulator + 1)
    accumulator *= 255.0/accumulator.max()
    accumulator = scipy.misc.imresize(accumulator, (500, 500))

    return accumulator

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

def perpendicular_distance(point1, point2, point3):
    run = point2[0] - point1[0]
    rise = point2[1] - point1[1]

    dist = run**2 + rise**2
    u = ((point3[1] - point1[0]) * run + (point3[1] - point1[1]) * rise)/ float(dist)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = point1[0] + u * run
    y = point1[1] + u * rise
    dx = x - point3[0]
    dy = y - point3[1]

    return math.sqrt(dx**2 + dy**2)


def douglas_peucker(contour, epsilon):
    dmax = 0
    index = 0
    end = len(contour)
    for i in range(2, end -1):
        d = perpendicular_distance(contour[0][0], contour[end -1][0], contour[i][0])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        recursive_res1 = douglas_peucker(contour[0:index+1], epsilon)
        recursive_res2 = douglas_peucker(contour[index+1: end], epsilon)

        result_list = np.concatenate((recursive_res1[:len(recursive_res1)], recursive_res2), axis=0)
    else:
        result_list = np.array([contour[0], contour[end-1]])

    return result_list



def find_corners_from_contours(page_contour):
    """Analyze the largest contour of the image and return the four corners of the document in the image"""

    epsilon = 0.15 * cv2.arcLength(page_contour, True)
    # page_approx = douglas_peucker(page_contour, epsilon)
    page_approx = cv2.approxPolyDP(page_contour, epsilon, True)
    return page_approx.sum(axis=1), page_approx


def scan_page(image):
    """Takes an image input that has a document in it and return a birds eye view, high contrast scan of the document"""

    gray, edged = create_edge_image(image)

    contours = find_contours_from_threshold(gray)

    # This code assumes that the largest contour will contain the document of interest
    page_contour = largest_contour(contours)

    # Approximate corners, perspective project and threshold the image
    points, page_approx = find_corners_from_contours(page_contour)

    # Apply a perspective transform to the document
    warped = transform.four_point_transform(image, points)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Add the contour onto original image and show it
    cv2.drawContours(image, page_approx, -1, (0, 0, 255), 20)
    cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
    cv2.imshow('Corners', image)

    # Apply an adaptive threshold on the image to remove contrasting shadows
    scanned_doc = adaptive_threshold(gray, type='adaptive')
    cv2.namedWindow('Scanned Document', cv2.WINDOW_NORMAL)
    cv2.imshow('Scanned Document', scanned_doc)

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
    image = cv2.imread('../images/resume_low_res.jpg')

    # image_cp = image.copy()
    #
    # def click_point(event, x, y, flags, param):
    #     # grab references to the global variables
    #     global refPt
    #
    #     # if the left mouse button was clicked, record the starting (x, y) coordinates
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         refPt.add((x, y))
    #         print("Added point {}, {} to the data set.".format(x, y))
    #         # draw a circle around the region of interest
    #         cv2.circle(image_cp, (x, y), 3, (0, 0, 255), -1)
    #         cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #         cv2.imshow("image", image_cp)
    #
    # # cv2.namedWindow("image")
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("image", click_point)
    #
    # cv2.imshow("image", image_cp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    scanned_doc = scan_page(image)
    cv2.imwrite('../images/scanned_notes.jpg', scanned_doc)

if __name__ == "__main__":
    main()
