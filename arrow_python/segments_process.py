import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from utils.lines_tools import (
    Line_Clustering,
    distance_of_2D_point,
    plot_lines,
    line_equation,
    intersection_point,
    find_intersection,
    overlap_length,
    union_length,
)

# 路径处理
ROOT = "./"
folder_name = "arrow_lines_first"
file_name = "950.txt"
file_path = os.path.join(ROOT, folder_name, file_name)

# 线段聚类工具
line_cluster = Line_Clustering(name="Line_Clustering")


def data_loader(visual=False):
    """
    数据加载
    """
    points = np.loadtxt(file_path, delimiter=" ")
    start_points = points[:, :3]
    end_points = points[:, 3:]
    points = np.concatenate((start_points, end_points), axis=0)

    # test rabbit: 可视化原始点云
    if visual:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])
    return points


def plot_lines_after_PCA(pca_data, origin_data, visual=False):
    """
    画出pca后的图像
    :param pca_data: (n, 2)
    :param origin_data: (n, 2)
    :visual: 是否可视化, default False
    """
    if pca_data.shape[0] > 1000:  # rabbit test
        PCA_point = np.asarray(pca_data)
        pcd_pca = o3d.geometry.PointCloud()
        tmp = np.zeros(np.shape(PCA_point)[0])
        PCA_3d_point = np.column_stack((PCA_point, tmp))
        pcd_pca.points = o3d.utility.Vector3dVector(PCA_3d_point)
        o3d.visualization.draw_geometries([pcd_pca])
    else:  # arrow
        num_points = pca_data.shape[0]
        start_points = pca_data[:num_points // 2, :]
        end_points = pca_data[num_points // 2:, :]
        origin_start_points = origin_data[:num_points // 2, :]
        origin_end_points = origin_data[num_points // 2:, :]

        if visual:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0] = plt.subplot(121, projection="3d")

            for i in range(num_points // 2 - 1):
                x_values = [start_points[i][0], end_points[i][0]]
                y_values = [start_points[i][1], end_points[i][1]]

                ox_values = [
                    origin_start_points[i][0], origin_end_points[i][0]
                ]
                oy_values = [
                    origin_start_points[i][1], origin_end_points[i][1]
                ]
                oz_values = [
                    origin_start_points[i][2], origin_end_points[i][2]
                ]

                axs[0].plot(ox_values, oy_values, oz_values)
                axs[1].plot(x_values, y_values)

            axs[0].set_title("Original Lines")
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")
            axs[0].grid(True)
            axs[1].set_xlim(-2.0, 1.5)  # 设置x轴范围
            axs[1].set_ylim(-0.5, 0.5)  # 设置y轴范围
            # axs[1].set_zlim(-0.5, 0.5)  # 设置z轴范围

            axs[1].set_title("PCA Lines")
            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Y")
            axs[1].grid(True)
            axs[1].set_xlim(-2.0, 1.5)  # 设置x轴范围
            axs[1].set_ylim(-0.5, 0.5)  # 设置y轴范围

            # 调整子图间距
            plt.tight_layout()
            plt.show()


def cluster_line(lines_pca, visual=False, class_type="box"):
    """
    线段聚类
    :param lines_pca: (n, 2), pca之后的数据
    :param visual: 是否可视化, default False
    :param type: 聚类类型, default box
    """
    # 数据形状处理
    num_points = lines_pca.shape[0]
    start_points = lines_pca[:num_points // 2, :]
    end_points = lines_pca[num_points // 2:, :]
    lines = np.concatenate((start_points, end_points), axis=1)

    # line_cluster数据初始化
    line_cluster.data_init(lines)

    if class_type == "slope":
        # By slope
        labels = line_cluster.slope_DBSCAN(lines)
    elif class_type == "center":
        # By segments center
        labels = line_cluster.center_DBSCAN(lines)
    elif class_type == "circle":
        # By circle
        labels = line_cluster.circle_LDBSCAN(lines, r=0.8, min_samples=1)
    elif class_type == "box":
        # By Box
        labels = line_cluster.box_LDBSCAN(lines, r=0.3,
                                          min_samples=1)  # r = 0.278 是界限

    if visual:
        line_cluster.plot_lines_class(lines,
                                      labels,
                                      title=f"{class_type} cluster results")

    return labels

def remove_duplicates(lines, labels):
    """
    去重
    :param lines: 原始数据
    :param labels: 原始标签
    :return: 去重后的数据
    """
    def segment_length(segment):
        # 计算线段的长度
        x1, y1, x2, y2 = segment
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return length

    # 数据形状处理
    num_points = lines.shape[0]
    start_points = lines[:num_points // 2, :]
    end_points = lines[num_points // 2:, :]
    lines = np.concatenate((start_points, end_points), axis=1)

    # 计算每条线段长度
    lengths = np.apply_along_axis(segment_length, 1, lines)

    # 长度从大到小排序，每一类只保留最长的一条线段
    sorted_lines_index = np.argsort(-lengths)
    final_lines_index = []
    final_labels = []
    for cur_index in sorted_lines_index:
        if labels[cur_index] not in final_labels:
            final_lines_index.append(cur_index)
            final_labels.append(labels[cur_index])
    print("最终线段的索引:", final_lines_index)
    print("最终标签：", final_labels)
    return lines[final_lines_index, :]


def close_lines(lines):
    """
    # 用交点闭合所有线段
    """
    # plot_lines(lines, title="The Original Segments ")

    # 遍历每条线，计算当前线和其他线的角点，存入line_intersections
    intersections = []
    anchor_index = []
    target_index = []
    for i in range(len(lines)):
        # 当前直线的两个端点
        p1 = (lines[i][0], lines[i][1])
        p2 = (lines[i][2], lines[i][3])
        for j in range(len(lines)):
            if i == j:
                continue  # 若当前线是自己则跳过
            # 获取另一条直线的端点
            q1 = (lines[j][0], lines[j][1])
            q2 = (lines[j][2], lines[j][3])
            # 计算角点
            p_intersection = find_intersection(p1, p2, q1, q2)
            # 判断是否有交点，存入变量line_intersections
            if p_intersection is not None:
                intersections.append(p_intersection)
                anchor_index.append(i)
                target_index.append(j)

    intersections = np.array(intersections)
    anchor_index = np.array(anchor_index)
    target_index = np.array(target_index)

    final_lines = []
    final_lines_index = []

    for i in range(len(lines)):
        # FIXME: 可以利用2维表来维护下一条直线
        if len(final_lines) != 0:
            # 为了形成有顺序的线段，需要将选出来的最后一个线段的终止点作为下一条线的起始点
            last_end_point = final_lines[-1][-1]

            # 找到与上一根线终止点相同的线段
            bool_same_as_to_last_end_point = intersections == last_end_point
            same_last_end_index = []
            for z in range(bool_same_as_to_last_end_point.shape[0]):
                b_x = bool_same_as_to_last_end_point[z][0]
                b_y = bool_same_as_to_last_end_point[z][1]
                if b_x and b_y and anchor_index[z] not in final_lines_index:
                    same_last_end_index.append(z)
            have_same_point_with_last_end_point = anchor_index[same_last_end_index]
            if (have_same_point_with_last_end_point.shape[0] > 1):  # 这里可能找到多条共线的线段，需要选择离起始点最近的一根
                p = last_end_point
                minimum_distance = 0
                cur_target_index = -1
                for h in range(have_same_point_with_last_end_point.shape[0]):
                    q1 = lines[have_same_point_with_last_end_point[h]][:2]
                    q2 = lines[have_same_point_with_last_end_point[h]][2:]
                    min_distance = min(distance_of_2D_point(p, q1), distance_of_2D_point(p, q2))
                    if h == 0 or min_distance < minimum_distance:
                        minimum_distance = min_distance
                        cur_target_index = have_same_point_with_last_end_point[h]
                if cur_target_index == -1:
                    raise ValueError("calculate distance error")
            else:
                cur_target_index = have_same_point_with_last_end_point[0]
        else:
            cur_target_index = i
            last_end_point = None
        # 获取当前直线产生的所有交点
        cur_indexs = anchor_index == cur_target_index
        cur_points = intersections[cur_indexs]

        # 从当前线产生的所有交点构成的直线中选择出与目标线段匹配度最大的两个交点
        target_line = lines[cur_target_index]
        max_iou = 0
        selected_index_s = None  # iou最大的在cur_points中的起始点索引
        selected_index_e = None  # iou最大的在cur_points中的终止点索引
        for start_index in range(cur_points.shape[0]):
            if (last_end_point is not None) and (
                (cur_points[start_index, 0] != last_end_point[0]) or
                    (cur_points[start_index, 1] != last_end_point[1])):
                continue
            for end_index in range(cur_points.shape[0]):
                if start_index == end_index:
                    continue
                # 获取两个交点构成的线段
                cur_line = [
                    cur_points[start_index, 0],
                    cur_points[start_index, 1],
                    cur_points[end_index, 0],
                    cur_points[end_index, 1],
                ]
                
                curL_2_targetL_overlap_length = overlap_length(cur_line, target_line)
                if curL_2_targetL_overlap_length == 0:
                    continue  # 没有交集直接跳过
                curL_2_targetL_union_length = union_length(cur_line, target_line)
                iou = curL_2_targetL_overlap_length / curL_2_targetL_union_length
                # 保存最大的交并比线段索引
                if iou > max_iou:
                    max_iou = iou
                    selected_index_s = start_index
                    selected_index_e = end_index
        # 选出最匹配的边
        if selected_index_s is not None and selected_index_e is not None:
            final_lines.append(
                (cur_points[selected_index_s], cur_points[selected_index_e]))
            final_lines_index.append(cur_target_index)
        else:
            raise ValueError("No match!!!")
    final_lines = np.array(final_lines).reshape(len(lines), -1)
    plot_lines(final_lines, title="The Closed Arrow!")


if __name__ == "__main__":
    # rabbit data
    # point_data = o3d.io.read_point_cloud("rabbit.pcd")
    # o3d.visualization.draw_geometries([point_data])
    # point_data = np.asarray(point_data.points)

    if 0:
        # our data
        point_data = data_loader()  # 加载数据
        PCA_point = line_cluster.PCA(point_data, 2)  # PCA降维
        plot_lines_after_PCA(PCA_point, point_data, visual=False)  # 绘制PCA降维数据
        labels = cluster_line(PCA_point, visual=True)  # 线段聚类
        final_lines = remove_duplicates(PCA_point, labels)  # 去重复
        plot_lines(final_lines,
                   title="The Segments after Remove Duplicates")  # 绘制聚类后的线段
    if 1:
        # 闭合线段
        # 定义线: 2D端点组成
        lines = np.array([
            # [1.3, 3.1, 2.8, 4.8],
            # [3.2, 4.8, 4.8, 3.2],
            # [4.9, 3.0, 4.1, 3.0],
            # [4.0, 2.9, 4.0, 0.1],
            # [3.9, 0.0, 2.1, 0.0],
            # [2.0, 0.1, 2.0, 2.9],
            # [1.9, 3.0, 1.1, 3.0],
            [1.3, 3.1, 2.8, 4.8],
            [3.2, 4.8, 4.8, 3.2],
            [4.9, 2.8, 4.1, 3.0],
            [4.2, 2.9, 4.0, 0.1],
            [3.9, -0.1, 2.1, 0.1],
            [2.3, 0.1, 2.0, 2.9],
            [1.9, 3.1, 1.1, 2.8],
        ])
        plot_lines(lines, title="The Original Segments ")
        close_lines(lines)
