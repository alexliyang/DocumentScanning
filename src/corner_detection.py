import cv2
import numpy as np
import matplotlib.pyplot as p
from document_scanner import create_edge_image
# from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from sort_points import SortPoints

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
    max_distance = np.sqrt(height ** 2 + width ** 2)
    rhos = np.arange(-max_distance, max_distance)
    # thetas = np.deg2rad(np.linspace(-90, 90,500))
    thetas = np.deg2rad(np.arange(-90, 90))

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
            rho = round(x * cos_t[t_coord] + y * sin_t[t_coord]) + max_distance
            accumulator[rho, t_coord] += 1.0

    # accumulator = np.log(accumulator + 1)
    # accumulator *= 255.0 / accumulator.max()
    # accumulator = scipy.misc.imresize(accumulator, (500, 500))

    return accumulator, thetas, rhos, width

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


def main():
    import timeit
    from numpy import cos, sin
    import scipy.misc

    # start = timeit.default_timer()

    img = cv2.imread('/home/hikmet/Python/DocumentScanning/images/paper.jpg')

    h, thetas, rhos, x2 = hough_transform(img)

    print detect_peaks(h)

    sortedpts = SortPoints(img)
    x = 0
    cv2.circle(img, (sortedpts.sorted_points[0][1], sortedpts.sorted_points[0][0]), 10, (0, 0, 255), -1)
    # for pt in sortedpts.sorted_points:
    #     cv2.circle(img, (pt[1], pt[0]), 1, (0, int(255 * x / 3458.0), int(255 * x / 3458.0)), 5)
    #     x += 1
    cv2.circle(img, (sortedpts.sorted_points[x - 1][1], sortedpts.sorted_points[x - 1][0]), 10, (255, 0, 0), -1)

    # show_image('image', cv2.resize(img.copy(), (560, 710)))

    # c = np.squeeze(np.where(h == h.max()))
    #
    # rho = rhos[c[0]]
    # theta = thetas[c[1]]
    #
    # x1 = 0
    # # if theta ==
    # y1 = int(rho/sin(theta))
    # y2 = int(x2*cos(theta) + y1)
    #
    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    stop = timeit.default_timer()
    # print stop - start

    # h = np.log(h + 1)
    # h *= 255.0 / h.max()
    # h = cv2.GaussianBlur(h, (5, 5), 0)
    # h = scipy.misc.imresize(h, (500, 500))
    p.imshow(img)
    p.figure()
    p.imshow(h)
    p.show()
    np.savetxt('hough.txt', h)


if __name__ == "__main__":
    main()