from queue import Queue
import numpy as np
import matplotlib.pyplot as plt


def circle_intersection_area(c1, r1, c2, r2):
    # Distance between the centers of the circles
    d = np.linalg.norm(c1 - c2, axis=-1)

    # Check if the circles are completely separate
    separate_mask = d >= r1 + r2
    intersection_area = np.zeros_like(d)

    # Check if one circle is completely inside the other
    inside_mask = d <= np.abs(r1 - r2)
    r_min = np.minimum(r1, r2)
    intersection_area[inside_mask] = np.pi * r_min[inside_mask] * r_min[inside_mask]

    # Calculate the intersection area for other cases
    alpha = np.arccos((r1 * r1 + d * d - r2 * r2) / (2 * r1 * d))
    beta = np.arccos((r2 * r2 + d * d - r1 * r1) / (2 * r2 * d))
    intersection_area[~(inside_mask | separate_mask)] = r1[~(inside_mask | separate_mask)] * r1[
        ~(inside_mask | separate_mask)] * alpha[~(inside_mask | separate_mask)] + \
                                                        r2[~(inside_mask | separate_mask)] * r2[
                                                            ~(inside_mask | separate_mask)] * beta[
                                                            ~(inside_mask | separate_mask)] - \
                                                        r1[~(inside_mask | separate_mask)] * d[
                                                            ~(inside_mask | separate_mask)] * np.sin(
        alpha[~(inside_mask | separate_mask)])

    intersection_area[separate_mask] = 0
    return intersection_area


def circle_union_area(c1, r1, c2, r2):
    area1 = np.pi * r1 * r1
    area2 = np.pi * r2 * r2
    intersection = circle_intersection_area(c1, r1, c2, r2)
    union = area1 + area2 - intersection
    return union


def intersection_over_union(c, r, c_list, r_list):
    c = np.array([c] * len(c_list))  # Repeat c to match the length of c_list
    r = np.array([r] * len(r_list))  # Repeat r to match the length of r_list
    intersection = circle_intersection_area(c, r, c_list, r_list)
    union = circle_union_area(c, r, c_list, r_list)
    iou = intersection / union
    return iou


def distance_of_2Dpoint(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class LDBSCAN:
    def __init__(self, min_samples=1, r=0.1):
        self.min_samples = min_samples
        self.r = r
        self.X = None
        self.Y = None
        self.label = None
        self.n_class = 0

    def calc_lines_center(self):
        self.middles_x = []
        self.middles_y = []
        self.line_r = []
        # calculate lines center
        for i in range(self.X.shape[0]):
            start_x = self.X[i][0]
            start_y = self.X[i][1]
            end_x = self.Y[i][0]
            end_y = self.Y[i][1]
            self.middles_x.append((start_x + end_x) / 2)
            self.middles_y.append((start_y + end_y) / 2)
            self.line_r.append(distance_of_2Dpoint((start_x, start_y), (end_x, end_y)) / 2)
        self.line_c = np.array([self.middles_x, self.middles_y]).T
        self.line_r = np.array(self.line_r)

    def calc_ious(self, obj):
        # anchor infomation
        start_x = obj[0][0]
        start_y = obj[0][1]
        end_x = obj[1][0]
        end_y = obj[1][1]
        middle_x = (start_x + end_x) / 2
        middle_y = (start_y + end_y) / 2
        ridus = distance_of_2Dpoint((start_x, start_y), (end_x, end_y)) / 2
        c1 = (middle_x, middle_y)
        # Calculate intersection over union for circle C and set L
        ious = intersection_over_union(c1, ridus, self.line_c, self.line_r)
        return ious

    def fit(self, data):
        self.X = data[:, :2]
        self.Y = data[:, 2:]
        self.calc_lines_center()

        self.label = np.zeros(self.X.shape[0])
        q = Queue()
        for i in range(len(self.X)):
            if self.label[i] == 0:
                cur_line = (self.X[i], self.Y[i])
                q.put(cur_line)

                ious = self.calc_ious(cur_line)
                # e_dists = np.sqrt(np.sum((self.X - self.X[i]) ** 2, axis=1))  # calculate euclidean distance list
                if self.X[(ious >= self.r) & (self.label == 0)].shape[0] >= self.min_samples:  # if current point's neighbors have more than min_samples, it is a core point
                    self.n_class += 1  # n_class +1
                # update queue
                while not q.empty():
                    # get queue element
                    p = q.get()
                    # get current point's neighbors
                    p_ious = self.calc_ious(p)
                    neighbors_index = (p_ious >= self.r) & (self.label == 0)
                    neighbors_x = self.X[neighbors_index]
                    # if current point's neighbors have more than min_samples, it is a core point
                    if neighbors_x.shape[0] >= self.min_samples:
                        # get current point's neighbors index
                        mark = ( p_ious>= self.r )
                        # set neighbors label to n_class
                        self.label[mark] = np.ones(self.label[mark].shape) * self.n_class
                        # print(self.label)
                        for k, x in enumerate(neighbors_index):
                            if x:
                                p_neighbors = (self.X[k], self.Y[k])
                                q.put(p_neighbors)

    def plot_dbscan_2D(self):
        plt.rcParams['font.sans-serif'] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False
        for i in range(self.n_class + 1):
            if i == 0:
                label = '异常数据'
            else:
                label = '第' + str(i) + '类数据'
            plt.scatter(self.X[self.label == i, 0], self.X[self.label == i, 1], label=label)
        plt.legend()
        plt.show()


data = np.array(
    [[-0.38498337, -0.05727393, 0.03070642, -0.04724828],
     [1.41703234, 0.07090077, 0.1022308, 0.27860864],
     [0.0531518, -0.05864481, 0.04940034, -0.20989687],
     [-0.1425823, 0.11900064, -1.06884148, 0.10338414],
     [0.11539587, -0.20785266, 1.39888236, 0.05953784],
     [0.13107365, -0.20565035, 1.40465788, 0.05676262],
     [-0.17958292, 0.12696127, -1.93732779, 0.0836032],
     [-0.93693045, -0.06582085, -0.05228316, -0.04637137]])

db = LDBSCAN(1, 0.3)
db.fit(data)
print(db.label)
# db.plot_dbscan_2D()
