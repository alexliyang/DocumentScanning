import math
import numpy as np


# def perpendicular_distance(point1, point2, point3):
#     run = point2[0] - point1[0]
#     rise = point2[1] - point1[1]
#
#     dist = run**2 + rise**2
#     u = ((point3[1] - point1[0]) * run + (point3[1] - point1[1]) * rise)/ float(dist)
#
#     if u > 1:
#         u = 1
#     elif u < 0:
#         u = 0
#
#     x = point1[0] + u * run
#     y = point1[1] + u * rise
#     dx = x - point3[0]
#     dy = y - point3[1]
#
#     return math.sqrt(dx**2 + dy**2)


def perpendicular_distance(p, p1, p2):
    if p1[0] == p2[0]:
        result = abs(p[0] - p1[0])
    else:
        slope = (p2[1] - p1[1])/(p2[0] - p1[0])
        intercept = p1[1] - (slope * p1[0])
        result = abs(slope * p[0] - p[1] + intercept)/ math.sqrt(math.pow(slope, 2) + 1)
    return result


# def douglas_peucker(contour, epsilon):
#     dmax = 0
#     index = 0
#     end = len(contour)
#     for i in range(2, end -1):
#         d = perpendicular_distance(contour[0][0], contour[end -1][0], contour[i][0])
#         if d > dmax:
#             index = i
#             dmax = d
#     if dmax > epsilon:
#         recursive_res1 = douglas_peucker(contour[0:index+1], epsilon)
#         recursive_res2 = douglas_peucker(contour[index+1: end], epsilon)
#
#         result_list = np.concatenate((recursive_res1[:len(recursive_res1)], recursive_res2), axis=0)
#     else:
#         result_list = np.array([contour[0], contour[end-1]])
#
#     return result_list


def find_farthest(contour, epsilon):
    le_eps = False
    right_start = 0
    slice_start = 0
    pos = 0
    count = len(contour)
    for i in range(0, 3):
        dist = max_dist = 0
        pos = (pos + right_start) % count
        start_pt = contour[pos][0]

        for j in range(1, count):
            pt = contour[j][0]
            dx = pt[0] - start_pt[0]
            dy = pt[1] - start_pt[1]

            dist = dx * dx + dy * dy
            if dist > max_dist:
                max_dist = dist
                right_start = j
        le_eps = max_dist <= epsilon
    if not le_eps:
        right_end = slice_start = pos % count
        slice_end = right_start = (right_start + slice_start) % count
        return slice_start



def rdp(contour, epsilon):
    first = contour[0][0]
    # largest = find_farthest(contour, epsilon)
    # last = contour[largest][0]
    last = contour[len(contour) -1][0]
    if len(contour) < 3:
        return contour
    index = -1
    dist = 0
    for i in range(1, len(contour) -1):
        c_dist = perpendicular_distance(contour[i][0], first, last)
        if c_dist > dist:
            dist = c_dist
            index = i
    if dist > epsilon:
        sub1 = contour[:index+1]
        sub2 = contour[index:]
        res1 = rdp(sub1, epsilon)
        res2 = rdp(sub2, epsilon)
        result = np.concatenate((res1[:len(res1)], res2), axis=0)
        return result
    else:
        return np.array([contour[0], contour[len(contour)-1]])