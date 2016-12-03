import math

class SortPoints:

    def __init__(self, points):
        self.points = points        # self.find_points(image)
        self.centre_point = None
        self.sorted_points = self.sort_clockwise(self.points)

    # def find_points(self, image):
    #     _, edge = create_edge_image(image)
    #     x, y = np.where(edge == 255)
    #     return zip(x, y)

    def sort_clockwise(self, points):
        self.centre_point = self.get_center_of_points(points)
        return sorted(points, cmp=self.is_less)

    def is_less(self, a, b):
        if b[0][0] < 0 <= a[0][0]:
            return 1
        elif a[0][0] == 0 and b[0][0] == 0:
            if a[0][1] > b[0][1]:
                return 1
            else:
                return -1

        det = (a[0][0] - self.centre_point[0]) * (b[0][1] - self.centre_point[1]) - (b[0][0] - self.centre_point[0]) * (a[0][1] - self.centre_point[1])
        if det < 0:
            return 1
        elif det > 0:
            return -1
        d1 = (a[0][0] - self.centre_point[0]) * (a[0][0] - self.centre_point[0]) + (a[0][1] - self.centre_point[1]) * (a[0][1] - self.centre_point[1])
        d2 = (b[0][0] - self.centre_point[0]) * (b[0][0] - self.centre_point[0]) + (b[0][1] - self.centre_point[1]) * (b[0][1] - self.centre_point[1])
        if d1 > d2:
            return 1
        else:
            return -1

    def get_center_of_points(self, points):
        sum_x = sum_y = 0
        for i in range(0, len(points)):
            sum_x += points[i][0][0]
            sum_y += points[i][0][1]
        return [sum_x/len(points), sum_y/len(points)]


def find_perpendicular(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    if x != 0:
        slope = y/x
    else:
        slope = 0
    slope_perp = -1/slope
    b_perp = -slope_perp * x + y
    return slope_perp, b_perp


