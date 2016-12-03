import cv2
import numpy as np
import matplotlib.pyplot as p
from document_scanner import create_edge_image
from sort_points import find_intersections
from src.helpers import show_image

def gradient_direction(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    Gx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)

    theta = np.arctan2(Gy, Gx)
    return theta


def hough_transform(image):
    import numpy as np
    import scipy.misc

    _, edge = create_edge_image(image)
    cv2.destroyAllWindows()
    points = np.transpose(np.where(edge == 255))

    hull = np.squeeze(cv2.convexHull(points)).T

    # edge *= 0
    # edge[hull[0],hull[1]] = 255

    height, width = edge.shape
    max_distance = int(np.sqrt(height ** 2 + width ** 2))
    rhos = np.arange(-max_distance, max_distance)
    # thetas = np.deg2rad(np.linspace(-90, 90,500))
    thetas = np.deg2rad(np.arange(-90+25, 90+25))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((len(rhos), len(thetas)))
    y_coord, x_coord = np.nonzero(edge)

    for i in range(len(y_coord)):
        x = x_coord[i]
        y = y_coord[i]

        for t_coord in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(x * cos_t[t_coord] + y * sin_t[t_coord]) + max_distance
            accumulator[rho, t_coord] += 1.0

    # accumulator = np.log(accumulator + 1)
    # accumulator *= 255.0 / accumulator.max()
    # accumulator = scipy.misc.imresize(accumulator, (500, 500))

    return accumulator, thetas, rhos


def erase_max(h, c):
    for i in range(-5,5)+c[1]:
        for j in range(-50, 50)+c[0]:
            h[j][i] = 0

    return h

def draw_four_lines(img):
    from numpy import sin, cos

    height, width, _ = img.shape
    h, thetas, rhos = hough_transform(img)
    lines = []
    for i in range(0,4):
        c = np.squeeze(np.where(h == h.max()))
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
        print x1, y1, x2, y2
        lines.append([x1, y1, x2, y2])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        h = erase_max(h, c)

    return img, lines

def main():
    import timeit
    from numpy import cos, sin
    import scipy.misc

    # start = timeit.default_timer()

    img = cv2.imread('../images/notes.jpg')
    img, lines = draw_four_lines(img)
    corners = find_intersections(lines, img.shape)
    for pt in corners:
        cv2.circle(img, (pt[0], pt[1]), 15, (0, 255, 0), -1)
        cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)

    cv2.imshow('Corners', cv2.resize(img, (560, 710)))
    # show_image('im', img)

    # stop = timeit.default_timer()
    # print stop - start


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # p.imshow(i1mg)
    # p.figure()

if __name__ == "__main__":
    main()