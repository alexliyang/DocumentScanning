import cv2
import numpy as np
from sort_points import find_intersections


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


def hough_transform(image, contour):
    import numpy as np

    gray, edge = create_edge_image(image)
    contour = np.squeeze(contour)

    height, width = edge.shape
    max_distance = int(np.sqrt(height ** 2 + width ** 2))
    rhos = np.arange(-max_distance, max_distance)
    thetas = np.deg2rad(np.arange(-90+25, 90+25))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((len(rhos), len(thetas)))
    x_coord, y_coord = np.split(contour, 2, 1)   # np.nonzero(edge)

    for i in range(len(y_coord)):
        x = x_coord[i]
        y = y_coord[i]

        for t_coord in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(x * cos_t[t_coord] + y * sin_t[t_coord]) + max_distance
            accumulator[rho, t_coord] += 1

    # accumulator = np.log(accumulator + 1)
    # accumulator *= 255.0 / accumulator.max()
    # accumulator = scipy.misc.imresize(accumulator, (500, 500))

    return accumulator, thetas, rhos


def erase_max(h, c):

    x_1 = 70
    y_1 = 150

    for i in range(-y_1 + c[0], y_1 + c[0]):
        for j in range(-x_1 + c[1], x_1 + c[1]):
            if i >= 0 and i < h.shape[0] and j >= 0 and j < h.shape[1]:
                h[i][j] = 0

    return h


def draw_four_lines(img, contour):
    from numpy import sin, cos

    height, width, _ = img.shape
    h, thetas, rhos = hough_transform(img, contour)

    lines = []
    for i in range(0,4):
        c = np.squeeze(np.where(h == h.max()))

        if len(c.shape) > 1:
            b = np.array((1,2))
            b[0] = c[0][0]
            b[1] = c[1][0]
            c = b
        rho = rhos[c[0]]
        theta = thetas[c[1]]

        x1 = 0
        y1 = int(rho / sin(theta))
        y2 = 0
        x2 = int(rho / cos(theta))

        if y1 >= height or y1 < 0:
            x1 = width - 1
            y1 = int((rho - x1 * cos(theta)) / sin(theta))
        if x2 >= width or x2 < 0:
            y2 = height - 1
            x2 = int((rho - y2 * sin(theta)) / cos(theta))
        if y1 >= height or y1 < 0:
            y1 = height - 1
            x1 = int((rho - y1 * sin(theta)) / cos(theta))
        if x2 >= width or x2 < 0:
            x2 = width - 1
            y2 = int((rho - x2 * cos(theta)) / sin(theta))
        lines.append([x1, y1, x2, y2])
        print x1, y1, x2, y2, 'i\'ve worked', i+1, 'times'
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        h = erase_max(h, c)

    return img, lines

def main():
    import timeit
    from numpy import cos, sin
    import scipy.misc

    # start = timeit.default_timer()

    # img = cv2.imread('../images/paper.jpg')
    # img = draw_four_lines(img)
    img = cv2.imread('../images/resume.jpg')
    img, lines = draw_four_lines(img)
    # cv2.namedWindow('Hough')
    # cv2.imshow('Hough', cv2.resize(img, (800, 600)))
    corners = find_intersections(lines, img.shape)
    for pt in corners:
        cv2.circle(img, (pt[0], pt[1]), 15, (0, 255, 0), -1)
        cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)

    cv2.imshow('Corners', cv2.resize(img, (900, 600)))

    # show_image('im', img)

    # stop = timeit.default_timer()
    # print stop - start

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # p.imshow(i1mg)
    # p.figure()

if __name__ == "__main__":
    main()