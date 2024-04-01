# Author: LPT

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = './'
floder_name = "arrow_lines_first"
floder_all_name = "arrow_lines_all"
txts_path = os.path.join(ROOT, floder_all_name)


# 点到直线的距离
def distance_to_line(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    # 直线上的向量
    line_vector = np.array([x3 - x2, y3 - y2, z3 - z2])

    # 目标点到直线起点的向量
    target_vector = np.array([x1 - x2, y1 - y2, z1 - z2])

    # 直线向量的模的平方
    line_length_sq = np.dot(line_vector, line_vector)

    # 目标点到直线的投影参数 t
    t = np.dot(target_vector, line_vector) / line_length_sq

    # 直线上距离目标点最近的点的坐标
    closest_point = np.array([x2, y2, z2]) + t * line_vector

    # 计算距离
    distance = np.linalg.norm(np.array([x1, y1, z1]) - closest_point)

    return distance

# 共面函数
def check_coplanar(u, v):
    # 判断两个向量是否成比例
    if np.allclose(np.cross(u, v), np.zeros(3)):
        return True
    else:
        return False

# 判断是否共面
def is_coplaner(frame_lines):
    for i in range(len(frame_lines)):
        for j in range(i + 1, len(frame_lines)):
            line1 = frame_lines[i]
            line2 = frame_lines[j]

            P1 = np.array(list((line1[0], line1[1], line1[2])))
            Q1 = np.array(list((line1[3], line1[4], line1[5])))
            P2 = np.array(list((line2[0], line2[1], line2[2])))
            Q2 = np.array(list((line2[3], line2[4], line2[5])))
            # P1 = np.array([0, 0, 0])
            # Q1 = np.array([0, 0, 1])
            # P2 = np.array([1, 0, 0])
            # Q2 = np.array([1, 0, 1])
            # 计算方向向量
            u = Q1 - P1
            v = Q2 - P2

            print(f"{i + 1} - {j + 1}: ", end="")

            if check_coplanar(u, v):
                print("两条直线共面")
            else:
                print("两条直线不共面")


def plot_3D_line(frame_lines, eliminate_list=[]):
    line_list = []
    for i, lines in enumerate(frame_lines):
        if i in eliminate_list: continue
        line_list.append(np.array([[lines[0], lines[1], lines[2]], [lines[3], lines[4], lines[5]]]))
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 遍历每条直线的端点数据，绘制直线
    for line in line_list:
        ax.plot(line[:, 0], line[:, 1], line[:, 2])

    # 设置图形的标题和坐标轴标签
    if len(eliminate_list) == 0:
        ax.set_title('Before eliminating duplicate')
    else:
        ax.set_title('After eliminating duplicate')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.gca().set_aspect('auto')

    # 启用交互式绘图
    plt.show()


# 3D点的两点距离
def getDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# 判断重叠线，获取剔除的线段索引
def eliminate_overlap_lines(frame_lines):
    """
    # 判断重叠线，获取剔除的线段索引
    :param frame_lines: 这一帧的所有线列表[ [x1,y1,z1,x2,y2,z2], ..., [x1,y1,z1,x2,y2,z2]  ]
    :return: eliminate_index: 因重叠需要剔除的线段索引 [i1, i2, ..., in]
    """

    # 保存到csv的变量
    dist_table = []
    # 如果两条线重叠，用来存储索引对
    eliminate_pair = []
    # 遍历
    for i in range(len(frame_lines)):
        tmp = []
        for j in range(len(frame_lines)):
            # 如果遇到自己和自己的比较
            if i == j:
                tmp.append(0.0)
                continue
            line1 = frame_lines[i]
            line2 = frame_lines[j]
            # 当前直线的两个端点分别到直线line2的距离
            dist1 = distance_to_line(line1[0], line1[1], line1[2], line2[0], line2[1], line2[2], line2[3], line2[4],
                                     line2[5])
            dist2 = distance_to_line(line1[3], line1[4], line1[5], line2[0], line2[1], line2[2], line2[3], line2[4],
                                     line2[5])
            tmp.append(max(dist1, dist2))
            # 两个短点的最大距离如果小于某阈值，证明这两条线特别贴近
            if 0.01 > max(dist1, dist2):
                eliminate_pair.append((i, j))
                print(f"{i} - {j}  distance = {max(dist1, dist2)}")
        dist_table.append(tmp)

    # 保存到csv
    # np_dist_table = np.array(dist_table)
    # df = pd.DataFrame(np_dist_table)
    # df.columns = [str(i) for i in range(len(frame_lines))]
    # df.to_csv('./output.csv', index=True)

    print("距离较近的索引对：", eliminate_pair)
    # 需要剔除的线段索引
    eliminate_index = []
    # 遍历剔除的索引对
    for indexPair in eliminate_pair:
        if indexPair[0] in eliminate_index or indexPair[1] in eliminate_index: continue
        line1 = frame_lines[indexPair[0]]
        line2 = frame_lines[indexPair[1]]
        # 比较线的两个端点距离，即线段长度
        d1 = getDistance((line1[0], line1[1], line1[2]), (line1[3], line1[4], line1[5]))
        d2 = getDistance((line2[0], line2[1], line2[2]), (line2[3], line2[4], line2[5]))
        # 剔除较短的
        print(f"{d1}({indexPair[0]})----{d2}({indexPair[1]})")
        eliminate_index.append(indexPair[0] if d1 < d2 else indexPair[1])
    print(f"需要剔除的线索引：{eliminate_index}")
    return eliminate_index



def do_somthing(lines):
    # 判断各个线是否共面
    # is_coplaner(lines)

    # 剔除重叠的线
    eliminate_index = eliminate_overlap_lines(lines)

    # 画图
    plot_3D_line(lines, eliminate_list=eliminate_index)
    plot_3D_line(lines)


def main():
    # 加载线段端点
    txts_name_list = os.listdir(txts_path)
    frame_list = []
    for i, name in enumerate(txts_name_list):
        # TODO
        # if name != "3205.txt": continue  # 乱
        if name != "3000.txt": continue  # 好
        file_path = os.path.join(txts_path, name)
        curr_frame_lines = []
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            for l_index, line in enumerate(file):
                lines = line.split(' ')[:-1]
                lines = [float(a) for a in lines]
                curr_frame_lines.append(lines)

            do_somthing(curr_frame_lines)


if __name__ == '__main__':
    main()
