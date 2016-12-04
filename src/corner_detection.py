import cv2
import numpy as np
import matplotlib.pyplot as p
from document_scanner import create_edge_image
from document_scanner import find_contours_from_threshold
from document_scanner import largest_contour
from sort_points import find_intersections
from sort_points import SortPoints
from src.helpers import show_image

def hough_transform(image):
    import numpy as np
    import scipy.misc

    gray, edge = create_edge_image(image)
    cv2.destroyAllWindows()
    # print 'type: ', type(edge), 'shape: ', edge.shape
    # edge = np.squeeze(np.transpose(largest_contour(find_contours_from_threshold(gray))))
    # print 'type: ', type(edge), 'shape: ', edge[0].shape
    # y_coord = edge[0]
    # x_coord = edge[1]


    # points = np.transpose(np.where(edge == 255))
    # hull = np.squeeze(cv2.convexHull(points)).T

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

# def detect_peaks(image):
#     """
#     Takes an image and detect the peaks usingthe local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """
#
#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,2)
#
#     # apply the local maximum filter; all pixel of maximal value
#     # in their neighborhood are set to 1
#     local_max = maximum_filter(image, footprint=neighborhood)==image
#     # local_max is a mask that contains the peaks we are
#     # looking for, but also the background.
#     # In order to isolate the peaks we must remove the background from the mask.
#
#     # we create the mask of the background
#     background = (image == 0)
#
#     # a little technicality: we must erode the background in order to
#     # successfully subtract it form local_max, otherwise a line will
#     # appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
#
#     # we obtain the final mask, containing only peaks,
#     # by removing the background from the local_max mask (xor operation)
#     detected_peaks = local_max ^ eroded_background
#
#     return detected_peaks

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
        lines.append([x1, y1, x2, y2])
        print x1, y1, x2, y2
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        h = erase_max(h, c)

    return img, lines

def main():
    import timeit
    from numpy import cos, sin
    import scipy.misc

    # start = timeit.default_timer()

    # img = cv2.imread('../images/paper.jpg')
    # img = draw_four_lines(img)
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