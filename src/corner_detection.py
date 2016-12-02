import cv2
import matplotlib.pyplot as p
from helpers import show_image

def hough_transform(edge):
    import numpy as np
    import scipy.misc

    height, width = edge.shape
    max_distance = np.sqrt(height ** 2 + width ** 2)
    rhos = np.arange(-max_distance, max_distance, 1)
    thetas = np.deg2rad(np.arange(-90, 90))

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
    accumulator *= 255.0 / accumulator.max()
    accumulator = scipy.misc.imresize(accumulator, (500, 500))

    return accumulator

def hough_lines(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def main():

    img = cv2.imread('/home/hikmet/Python/DocumentScanning/images/4point.jpg')
    # p.imshow(img)
    show_image('gray', img)

if __name__ == "__main__":
    main()