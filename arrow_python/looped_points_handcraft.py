# Author: LPT

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import math

line_order = []
line_intersections = []
points_order = []

# 定义线: 2D端点组成
# lines = [
#     [1.3, 3.1, 2.8, 4.8],
#     [3.2, 4.8, 4.8, 3.2],
#     [4.9, 3.0, 4.1, 3.0],
#     [4.0, 2.9, 4.0, 0.1],
#     [3.9, 0.0, 2.1, 0.0],
#     [2.0, 0.1, 2.0, 2.9],
#     [1.9, 3.0, 1.1, 3.0],
# ]


lines = [
    [1.3, 3.1, 2.8, 4.8],
    [3.2, 4.8, 4.8, 3.2],
    [4.9, 2.8, 4.1, 3.0],
    [4.2, 2.9, 4.0, 0.1],
    [3.9, -0.1, 2.1, 0.1],
    [2.3, 0.1, 2.0, 2.9],
    [1.9, 3.1, 1.1, 2.8],
]


# 2D平面上的两点间距
def dist_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_intersection(p1, p2, q1, q2):
    '''
    # 获取共面直线的交点, p1,p2是第一条直线的端点, q1,12是第二条直线的断点
    :param p1: 直线1的端点1
    :param p2: 直线1的端点2
    :param q1: 直线2的端点1
    :param q2: 直线2的端点2
    :return: 两直线的交点坐标
    '''
    # 计算直线P1P2的斜率和截距
    m1 = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')
    b1 = p1[1] - m1 * p1[0] if m1 != float('inf') else float('inf')

    # 计算直线Q1Q2的斜率和截距
    m2 = (q2[1] - q1[1]) / (q2[0] - q1[0]) if q2[0] != q1[0] else float('inf')
    b2 = q1[1] - m2 * q1[0] if m2 != float('inf') else float('inf')

    # 计算交点的坐标
    if m1 != m2:  # 如果斜率不相等，直线有交点
        if m1 == float('inf'):  # 如果直线P1P2是竖直线
            x = p1[0]
            y = m2 * x + b2
        elif m2 == float('inf'):  # 如果直线Q1Q2是竖直线
            x = q1[0]
            y = m1 * x + b1
        else:
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
        return x, y
    else:
        return None  # 直线平行，没有交点


def generate_closed_graph_lines_order():
    '''
    从所有线的交点中选择要保留的关键交点，因为如果有7条线，每条线都会和其他6条线产生交点（除平行）
    最终生成一个能够形成闭合图形的，线的连接顺序列表：line_order
    :return:
    '''

    # 遍历每条线，计算当前线和其他线的角点，存入line_intersections
    for i in range(len(lines)):
        # 当前直线的两个端点
        p1 = (lines[i][0], lines[i][1])
        p2 = (lines[i][2], lines[i][3])
        for j in range(len(lines)):
            if i == j: continue  # 若当前线是自己则跳过
            # 求相交的另一条直线的端点
            q1 = (lines[j][0], lines[j][1])
            q2 = (lines[j][2], lines[j][3])
            # 计算角点
            p_intersection = find_intersection(p1, p2, q1, q2)
            # 判断是否有交点，存入变量line_intersections
            if p_intersection is not None:
                line_intersections.append({
                    "curLineID": i,
                    "InterlineID": j,
                    "p_intersection": p_intersection
                })
    # line_intersection中存的是每个线和其他线的相交情况
    # 需要继续从中挑选需要保留的交点
    for i in range(len(lines)):
        cur_line_dist_list = []
        dist_list = []
        # 从p_intersection中挑选当前线和距离他最近的两条交线
        for line_intersection in line_intersections:
            curLineID = 0
            # 获取锚线：第一条线或按顺序排列表的最后一条线
            if i == 0:
                line1 = lines[curLineID]
            else:
                curLineID = line_order[-1]
                line1 = lines[curLineID]
            # 若碰到自己和自己的相交则跳过
            if line_intersection["curLineID"] != curLineID: continue
            # 获取到相交的线ID和线端点
            line2 = lines[line_intersection["InterlineID"]]
            a1 = (line1[0], line1[1])
            a2 = (line1[2], line1[3])
            b1 = (line2[0], line2[1])
            b2 = (line2[2], line2[3])
            # 计算端点间的两两距离，获取这条线距我们当前锚线最近的距离来判断是否相临近。
            dist1 = dist_points(a1, b1)
            dist2 = dist_points(a1, b2)
            dist3 = dist_points(a2, b1)
            dist4 = dist_points(a2, b2)
            cur_line_dist_list.append({
                "ID": line_intersection["InterlineID"],
                "dist": min(dist1, dist2, dist3, dist4)
            })
            dist_list.append(min(dist1, dist2, dist3, dist4))

        # 对当前线的所有交线
        sorted_list = sorted(cur_line_dist_list, key=lambda x: x['dist'])
        if i == 0:
            # 按照“闭合图形的一根线有且仅有两个邻居”的原则添加最近的两条线ID到排序列表
            # 排序列表的最后一位即为下一条锚线
            line_order.append(sorted_list[0]["ID"])
            line_order.append(i)
            line_order.append(sorted_list[1]["ID"])
        else:
            # 若通过距离选择的交线已经存在于排序列表中，就不再添加
            if sorted_list[0]["ID"] not in line_order:
                line_order.append(sorted_list[0]["ID"])
            if sorted_list[1]["ID"] not in line_order:
                line_order.append(sorted_list[1]["ID"])

    # 因为最终要根据line_order画一个闭合图形，所以在顺序列表中加入第一个元素，形成闭环
    line_order.append(line_order[0])
    print(line_order)


def generate_points_sort():
    '''
    # 根据顺序列表产生点的连接顺序列表：points_order
    :return:
    '''
    # 通过line_order的顺序, 检查当前线和交线的交点, 产生交点的连接顺序, 得到points_order
    for i in range(len(lines)):
        curLineID = line_order[i]
        for line_intersection in line_intersections:
            if line_intersection["curLineID"] != curLineID: continue
            if line_intersection["InterlineID"] != line_order[i + 1]: continue
            points_order.append(line_intersection["p_intersection"])
    points_order.append(points_order[0])
    print(points_order)


def plot_arrow():
    # 创建图形对象
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # 遍历线段集合并绘制每条线段
    for line in lines:
        x1, y1, x2, y2 = line
        ax1.add_line(mlines.Line2D([x1, x2], [y1, y2]))

    # 设置第一个子图的标题和坐标轴标签
    ax1.set_title('Original Lines')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.axis('equal')  # 确保x和y轴相等刻度

    ax2.plot(*zip(*points_order))
    # 设置第二个子图的标题和坐标轴标签
    ax2.set_title('Optimized closed graph')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()
    # 显示图形
    plt.show()


def do_somthing():
    # get line order, save to 'line_order'
    generate_closed_graph_lines_order()   # 生成线的排序
    generate_points_sort()   # 生成点的排序
    plot_arrow()   # 画图


def main():
    do_somthing()


if __name__ == '__main__':
    main()
