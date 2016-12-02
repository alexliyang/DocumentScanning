import cv2
import numpy as np
import matplotlib.pyplot as p
from helpers import show_image, wait_for_q_press
from document_scanner import create_edge_image
from scipy.signal import convolve2d

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

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
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

    return accumulator


def main():
    import timeit

    start = timeit.default_timer()

    img = cv2.imread('/home/hikmet/Python/DocumentScanning/images/notes.jpg')

    h = hough_transform(img)

    # _, edge = create_edge_image(img)
    # cv2.destroyAllWindows()
    # points = np.transpose(np.where(edge == 255))
    #
    # hull = np.squeeze(cv2.convexHull(points)).T
    #
    # edge *= 0
    #
    # edge[hull[0],hull[1]] = 255

    # cv2.HoughLinesP(img, )

    stop = timeit.default_timer()
    print stop - start
    p.imshow(h)
    p.show()
    # p.imshow(h, 'gray')
    # p.show()

    # for i in range(len(hull[0])):
    #
    #     cv2.circle(img,(hull[1][i],hull[0][i]),1,(0,0,255),1)
    #
    # p.imshow(img)
    # p.show()



    # coordinates = np.asarray(zip(x,y))
    #
    # coordinates = cv2.convexHull(coordinates)
    #
    # edge = np.zeros(img.shape)
    #
    # for i in coordinates:
    #     edge[i[0][1]][i[0][0]] = 255
    #
    # print edge[531][304]
    # h = hough_line(coordinates[0])
    #
    # p.imshow(edge,'gray')
    # p.show()



    # center = coordinates.sum(0)/len(coordinates)
    # cv2.circle(img,center,1,(0,0,255),5)
    # show_image('image',img)
    # sort = np.argsort(np.arctan2(y - center[0], x - center[1]))
    #
    # sorted = coordinates.copy()
    # x = 0
    #
    # for i in range(0,len(coordinates)):
    #     sorted[i] = coordinates[sort[i]]
    # for i in sorted:
    #
    #     cv2.circle(img,(i[0],i[1]),1,(0,int(255*x/3458.0),int(255*x/3458.0)),5)
    #     show_image('image',img)
    #     x += 1

    # wait_for_q_press()
if __name__ == "__main__":
    main()