from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
import math





def union_length(line1, line2):
    """
    计算两条线段的并集长度
    因为并集是建立在有交集之后，所以必定有交集才会调用这个函数
    """
    def find_union(range1, range2):
        x1, x2 = range1
        x3, x4 = range2
        # 计算并集的起始和结束
        union_start = min(x1, x3)
        union_end = max(x2, x4)
        # 返回并集范围
        return union_start, union_end
    
    xx1, yy1, xx2, yy2 = line1
    xx3, yy3, xx4, yy4 = line2
        
    if xx1 == xx2 and xx1 == xx3 and xx3 == xx4:
        x1, x2, x3, x4 = yy1, yy2, yy3, yy4
        y1, y2, y3, y4 = xx1, xx2, xx3, xx4
    else:
        x1, x2, x3, x4 = xx1, xx2, xx3, xx4
        y1, y2, y3, y4 = yy1, yy2, yy3, yy4 
    range1 = [x1, x2]
    range2 = [x3, x4]
    range1.sort()
    range2.sort()
    union = find_union(range1, range2)
    
    kk = (y2-y1)/(x2-x1)
    if kk == 0:
        length = abs(union[1]-union[0])
    else:
        length = math.sqrt((1+kk**2)*abs(union[1]-union[0])**2)
    return length
    
    
    
def overlap_length(line1, line2):
    """
    计算两条线段的重合长度
    """
    def find_intersection_interval(range1, range2):
        '''
            寻找两个区间的相交区间
        '''
        A, B = range1
        C, D = range2
        # 如果两个范围没有交集，则返回空范围
        if B < C or D < A:
            return None
        # 否则，返回交集范围
        return max(A, C), min(B, D)
    
    
    xx1, yy1, xx2, yy2 = line1
    xx3, yy3, xx4, yy4 = line2
        
    if xx1 == xx2 and xx1 == xx3 and xx3 == xx4:
        x1, x2, x3, x4 = yy1, yy2, yy3, yy4
        y1, y2, y3, y4 = xx1, xx2, xx3, xx4
    else:
        x1, x2, x3, x4 = xx1, xx2, xx3, xx4
        y1, y2, y3, y4 = yy1, yy2, yy3, yy4  
        
    range1 = [x1, x2]
    range2 = [x3, x4]

    range1.sort()
    range2.sort()
    intersection = find_intersection_interval(range1, range2)
    if intersection is None:
        return 0

    kk = (y2-y1)/(x2-x1)
    if kk == 0:
        length = abs(intersection[1]-intersection[0])
    else:
        length = math.sqrt((1+kk**2)*abs(intersection[1]-intersection[0])**2)
    return length


def find_intersection(p1, p2, q1, q2):
    """
        获取共面直线的交点, p1,p2是第一条直线的端点, q1,12是第二条直线的断点
    :param p1: 直线1的端点1
    :param p2: 直线1的端点2
    :param q1: 直线2的端点1
    :param q2: 直线2的端点2
    :return: 两直线的交点坐标
    """
    # 计算直线P1P2的斜率和截距
    m1 = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float("inf")
    b1 = p1[1] - m1 * p1[0] if m1 != float("inf") else float("inf")

    # 计算直线Q1Q2的斜率和截距
    m2 = (q2[1] - q1[1]) / (q2[0] - q1[0]) if q2[0] != q1[0] else float("inf")
    b2 = q1[1] - m2 * q1[0] if m2 != float("inf") else float("inf")

    # 计算交点的坐标
    if m1 != m2:  # 如果斜率不相等，直线有交点
        if m1 == float("inf"):  # 如果直线P1P2是竖直线
            x = p1[0]
            y = m2 * x + b2
        elif m2 == float("inf"):  # 如果直线Q1Q2是竖直线
            x = q1[0]
            y = m1 * x + b1
        else:
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
        return round(x, 4), round(y, 4)
    else:
        return None  # 直线平行，没有交点


def line_equation(line):
    """
    求线段所在的直线方程: Ax+By=C
    """
    x1, y1, x2, y2 = line
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return A, B, -C


def intersection_point(line1, line2):
    """
    求两条直线的交点
    :param line1: 直线1: (A, B, C)
    :param line2: 直线2: (A, B, C)
    return: 交点坐标: (x, y)
    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # 平行线段没有交点
    x = (C1 * B2 - C2 * B1) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return x, y


def intersection_point(line1, line2):
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # 平行线段没有交点
    x = (C1 * B2 - C2 * B1) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return x, y


def plot_lines(lines, title="Segments"):
    """
    画出每条线段
    :param lines: (n, 4)
    :title: 画图标题
    """
    plt.figure()

    # 画出每条线段
    for line in lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], color="blue")
    # 设置图形的标题和坐标轴标签
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    # 显示图形
    plt.grid(True)
    plt.show()


def rectangle_area(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = (
        box[0],
        box[1],
        box[2],
        box[3],
        box[4],
        box[5],
        box[6],
        box[7],
    )
    # 计算矩形的宽度和高度
    width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
    height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

    # 计算矩形的面积
    area = width * height
    return area


def calc_box(obj, cur_sin_theta, cur_cos_theta, r):
    x11 = obj[:, 0] - r * cur_sin_theta
    y11 = obj[:, 1] + r * cur_cos_theta
    x12 = obj[:, 0] + r * cur_sin_theta
    y12 = obj[:, 1] - r * cur_cos_theta

    x21 = obj[:, 2] - r * cur_sin_theta
    y21 = obj[:, 3] + r * cur_cos_theta
    x22 = obj[:, 2] + r * cur_sin_theta
    y22 = obj[:, 3] - r * cur_cos_theta
    return np.array([x11, y11, x12, y12, x21, y21, x22, y22]).T


def intersection_area(b1, b2):
    x11, y11, x12, y12, x13, y13, x14, y14 = (
        b1[0],
        b1[1],
        b1[2],
        b1[3],
        b1[4],
        b1[5],
        b1[6],
        b1[7],
    )
    x21, y21, x22, y22, x23, y23, x24, y24 = (
        b2[0],
        b2[1],
        b2[2],
        b2[3],
        b2[4],
        b2[5],
        b2[6],
        b2[7],
    )
    left_x = max(min(x11, x12, x13, x14), min(x21, x22, x23, x24))
    left_y = max(min(y11, y12, y13, y14), min(y21, y22, y23, y24))
    right_x = min(max(x11, x12, x13, x14), max(x21, x22, x23, x24))
    right_y = min(max(y11, y12, y13, y14), max(y21, y22, y23, y24))

    # 计算相交矩形的宽度和高度
    width = max(0, right_x - left_x)
    height = max(0, right_y - left_y)

    # 计算相交矩形的面积
    area = width * height
    return area


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
    intersection_area[~(inside_mask | separate_mask)] = (
        r1[~(inside_mask | separate_mask)]
        * r1[~(inside_mask | separate_mask)]
        * alpha[~(inside_mask | separate_mask)]
        + r2[~(inside_mask | separate_mask)]
        * r2[~(inside_mask | separate_mask)]
        * beta[~(inside_mask | separate_mask)]
        - r1[~(inside_mask | separate_mask)]
        * d[~(inside_mask | separate_mask)]
        * np.sin(alpha[~(inside_mask | separate_mask)])
    )

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
    intersection = circle_intersection_area(c, r, c_list, r_list) + 1e-16  # 交集
    # union = circle_union_area(c, r, c_list, r_list)  # 并集
    cur_area = math.pi * r**2
    # 覆盖率
    self_coverage = intersection / cur_area
    # iou = intersection / union
    return self_coverage


def distance_of_2Dpoint(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angles(segments):
    # 计算方向向量
    vectors = segments[:, 2:] - segments[:, :2]

    # 计算方向向量与x轴的夹角的余弦值
    cos_theta = vectors[:, 0] / np.linalg.norm(vectors, axis=1)

    # 计算夹角的正弦值
    sin_theta = np.sqrt(1 - cos_theta**2)

    return sin_theta, cos_theta


class Line_Clustering:
    def __init__(self, name="Line_Clustering"):
        self.name = name
        self.data = None

        # PCA
        self.PCA_data = None

        # DBSCAN
        self.center_labels = None
        self.slope_labels = None
        self.circle_ldbscan_labels = None
        self.box_ldbscan_labels = None

        # circle_LDBSCAN
        self.line_c = None
        self.line_r = None
        self.start_points = None
        self.end_points = None

        # box_LDBSCAN
        self.lines_sin_theta = None
        self.lines_cos_theta = None
        self.box_r = 0.05

    def data_init(self, lines):
        self.data = lines
        self.start_points = lines[:, :2]
        self.end_points = lines[:, 2:]
        self.calc_lines_center()
        self.lines_sin_theta, self.lines_cos_theta = calculate_angles(self.data)
        self.boxes = calc_box(
            self.data, self.lines_sin_theta, self.lines_cos_theta, self.box_r
        )

    def calc_lines_center(self):
        middles_x = []
        middles_y = []
        self.line_r = []
        # calculate lines center
        for i in range(self.data.shape[0]):
            start_x = self.data[i][0]
            start_y = self.data[i][1]
            end_x = self.data[i][2]
            end_y = self.data[i][3]
            middles_x.append((start_x + end_x) / 2)
            middles_y.append((start_y + end_y) / 2)
            self.line_r.append(
                distance_of_2Dpoint((start_x, start_y), (end_x, end_y)) / 2
            )
        self.line_c = np.array([middles_x, middles_y]).T
        self.line_r = np.array(self.line_r)

    def calc_circle_coverages(self, obj):
        # anchor infomation
        start_x = obj[0]
        start_y = obj[1]
        end_x = obj[2]
        end_y = obj[3]
        middle_x = (start_x + end_x) / 2
        middle_y = (start_y + end_y) / 2
        ridus = distance_of_2Dpoint((start_x, start_y), (end_x, end_y)) / 2
        c1 = (middle_x, middle_y)
        # Calculate intersection over union for circle C and set L
        coverages = intersection_over_union(c1, ridus, self.line_c, self.line_r)
        return coverages

    def circle_LDBSCAN(self, lines, r=0.7, min_samples=1):
        n_class = 0
        self.circle_ldbscan_labels = np.zeros(lines.shape[0]).astype(np.int32)
        q = Queue()
        for i in range(lines.shape[0]):
            if self.circle_ldbscan_labels[i] == 0:
                q.put(lines[i])
                coverages = self.calc_circle_coverages(lines[i])

                if (
                    lines[(coverages >= r) & (self.circle_ldbscan_labels == 0.0)].shape[
                        0
                    ]
                    >= min_samples
                ):  # if current point's neighbors have more than min_samples, it is a core point
                    n_class += 1  # n_class +1
                # update queue
                while not q.empty():
                    # get queue element
                    p = q.get()
                    # get current point's neighbors
                    p_coverages = self.calc_circle_coverages(p)
                    neighbors_index = (p_coverages >= r) & (
                        self.circle_ldbscan_labels == 0.0
                    )
                    neighbors = lines[neighbors_index]
                    # if current point's neighbors have more than min_samples, it is a core point
                    if neighbors.shape[0] >= min_samples:
                        # get current point's neighbors index
                        mark = p_coverages >= r
                        # set neighbors label to n_class
                        self.circle_ldbscan_labels[mark] = (
                            np.ones(self.circle_ldbscan_labels[mark].shape).astype(
                                np.int32
                            )
                            * n_class
                        )
                        # print(self.label)
                        for x in neighbors:
                            q.put(x)
        return self.circle_ldbscan_labels

    def calc_box_coverages(self, obj):
        cur_sin_theta, cur_cos_theta = calculate_angles(obj[np.newaxis, :])
        box = calc_box(obj[np.newaxis, :], cur_sin_theta, cur_cos_theta, self.box_r)
        intersections = []
        for i in range(self.data.shape[0]):
            target_box = self.boxes[i]
            intersection = intersection_area(box[0], target_box)
            intersections.append(intersection)
        cur_box_area = rectangle_area(box[0])
        self_coverage = np.array(intersections) / cur_box_area
        return self_coverage

    def box_LDBSCAN(self, lines, r=0.7, min_samples=1):
        n_class = 0
        self.box_ldbscan_labels = np.zeros(lines.shape[0]).astype(np.int32)
        q = Queue()
        for i in range(lines.shape[0]):
            if self.box_ldbscan_labels[i] == 0:
                q.put(lines[i])
                coverages = self.calc_box_coverages(lines[i])

                if (
                    lines[(coverages >= r) & (self.box_ldbscan_labels == 0.0)].shape[0]
                    >= min_samples
                ):  # if current point's neighbors have more than min_samples, it is a core point
                    n_class += 1  # n_class +1
                # update queue
                while not q.empty():
                    # get queue element
                    p = q.get()
                    # get current point's neighbors
                    p_coverages = self.calc_box_coverages(p)
                    neighbors_index = (p_coverages >= r) & (
                        self.box_ldbscan_labels == 0.0
                    )
                    neighbors = lines[neighbors_index]
                    # if current point's neighbors have more than min_samples, it is a core point
                    if neighbors.shape[0] >= min_samples:
                        # get current point's neighbors index
                        mark = p_coverages >= r
                        # set neighbors label to n_class
                        self.box_ldbscan_labels[mark] = (
                            np.ones(self.box_ldbscan_labels[mark].shape).astype(
                                np.int32
                            )
                            * n_class
                        )
                        # print(self.label)
                        for x in neighbors:
                            q.put(x)
        return self.box_ldbscan_labels
        return

    def slope_DBSCAN(self, lines):
        # 斜率
        slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])

        data = slopes.reshape(-1, 1)

        # 定义 DBSCAN 参数
        epsilon = 0.1  # 邻域半径
        min_samples = 1  # 最小样本数

        # 使用 DBSCAN 算法进行聚类
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        dbscan.fit(data)
        labels = dbscan.labels_
        labels += 1
        self.slope_labels = labels
        return labels

    def center_DBSCAN(self, lines):
        # 线段中点
        centers = (lines[:, :2] + lines[:, 2:]) / 2

        # 定义 DBSCAN 参数
        epsilon = 0.1  # 邻域半径
        min_samples = 1  # 最小样本数

        # 使用 DBSCAN 算法进行聚类
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(centers)
        labels += 1
        self.center_labels = labels
        return labels

    def plot_lines_class(self, data, labels, title="Lines Clustering Result"):
        """
        :param data: (n, 4)
        :param labels: (n,)
        :return:
        """

        # 绘制聚类结果
        num_clusters = len(np.unique(labels))  # 去除噪声点
        num_lines = data.shape[0]
        colors = plt.cm.jet(
            np.linspace(0, 1, num_clusters)
        )  # 使用jet colormap生成不同颜色
        lines_colors = plt.cm.jet(
            np.linspace(0, 1, num_lines)
        )  # 使用jet colormap生成不同颜色
        for label, color in zip(np.unique(labels), colors):
            if label == 0:
                plt.scatter(
                    data[labels == label, 0],
                    data[labels == label, 1],
                    color="k",
                    marker="x",
                    label="Noise",
                )
            else:
                plt.scatter(
                    data[labels == label, 0],
                    data[labels == label, 1],
                    color=color,
                    marker="o",
                    label="Cluster " + str(label),
                )

        # 绘制线段
        for i, color in zip(range(len(data)), lines_colors):
            start_x, start_y = data[i][0], data[i][1]
            end_x, end_y = data[i][2], data[i][3]
            plt.plot(
                [start_x, end_x],
                [start_y, end_y],
                color="gray",
                linestyle="--",
                linewidth=1,
            )

        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.show()

    def PCA(self, points_np, n):
        """
        主成分分析
        :param points_np: 数据
        :param n: 目标维度
        :return:
        """
        col_mean_val = np.mean(points_np, axis=0)  # 计算均值
        normalized_points = points_np - col_mean_val  # 去中心化（归一）
        cov_matrix = np.cov(normalized_points, rowvar=0)  # 计算协方差矩阵
        eig_val, eig_vector = np.linalg.eig(
            cov_matrix
        )  # 计算协方差矩阵的特征值特征向量
        origin_eig_val_index = np.argsort(eig_val)[::-1]  # 按升序排列特征值的索引数组
        origin_max_n_eig_val_index = origin_eig_val_index[:n]  # 取前 n 维元素
        n_vector = eig_vector[
            :, origin_max_n_eig_val_index
        ]  # 从特征向量矩阵 eig_vector 中选取对应的前n列
        # low_dimensional_data = normalized_points * np.mat(n_vector)   # 与特征向量相乘，可以将数据投影到一个低维度的空间中
        low_dimensional_data = np.dot(
            normalized_points, np.array(n_vector)
        )  # 与特征向量相乘，可以将数据投影到一个低维度的空间中

        self.PCA_data = low_dimensional_data  # 保存 PCA 降维后的数据

        return low_dimensional_data
