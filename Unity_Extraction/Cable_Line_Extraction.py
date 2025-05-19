import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

import os

def show_points(points, label, k=0, saved=False):
    points_array = np.array(points)
    x = points_array[:, 0]
    y = points_array[:, 1]
    z = points_array[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o', zorder=1, s=1)
    ax.scatter(x[k], y[k], z[k], c='b', marker='o', zorder=2, s=2)
    plt.show()
    if saved:
        plt.savefig(f'Cable_Line_Sorted/{label}_plot_{k}.png')

def merge_points(points):
    points_array = np.array(points)
    points_count = len(points)
    k = 100
    kmeans = KMeans(n_clusters=k, random_state=0).fit(points_array)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    fused_points = centroids
    return fused_points

def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def sort_cable_points(Terminal_1, Terminal_2, route_points):
    # 无序点列排序
    route_points = list(route_points)
    route_points_ret = [Terminal_1]
    n = len(route_points)
    # 从terminal_1开始排序
    target_point = Terminal_1
    while len(route_points) > 0:
        dist_min = float('inf')
        next_point = [0, 0, 0]
        next_idx = -1
        for i, route_point in enumerate(route_points):
            dist = calculate_distance(target_point, route_point)
            if dist < dist_min:
                dist_min = dist
                next_point = route_point
                next_idx = i
        del route_points[next_idx]
        route_points_ret.append(list(next_point))
        target_point = next_point
    route_points_ret.append(Terminal_2)
    return route_points_ret

def uniform_resample(points, num_samples=1000):
    points = np.array(points)

    # 计算相邻点之间的距离
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

    total_length = np.sum(distances)

    # 计算每个采样点的理想弧长位置
    sample_distances = np.linspace(0, total_length, num_samples)

    # 初始化采样点
    sampled_points = [points[0].tolist()]

    # 当前累积距离
    current_length = 0.0
    current_index = 0

    for sample_distance in sample_distances[1:]:
        while current_length + distances[current_index] < sample_distance:
            current_length += distances[current_index]
            current_index += 1
            if current_index > 100:
                current_index = 100
                break

        # 插值找到采样点
        t = (sample_distance - current_length) / distances[current_index]
        next_point = points[current_index] + t * (points[current_index + 1] - points[current_index])
        sampled_points.append(next_point.tolist())

    return sampled_points








if __name__ == "__main__":

    in_dir = '../data/LAB_imgs_design_DLO/LAB_CABIN_CABLES_NEW/route3D'
    out_dir = '../data/LAB_imgs_design_DLO/LAB_CABIN_CABLES_NEW/route3D_extracted'
    cable_list = os.listdir(in_dir)
    os.makedirs(out_dir, exist_ok=True)

    for cable_name in cable_list:

        # cable_name = 'W05_1.json'

        path = in_dir + '/{}.json'.format(cable_name[:-5])

        with open(path, 'r', encoding='utf-8-sig') as file:
            Cable_Dict = json.load(file)

        Terminal_1 = Cable_Dict['Terminal_1']
        Terminal_2 = Cable_Dict['Terminal_2']
        Vertices = Cable_Dict['Cable']
        # show_points([Terminal_1] + Vertices + [Terminal_2], f'origin_{cable_name[:-5]}', 0, True)

        fused_Vertices = merge_points(Vertices)

        # show_points(fused_Vertices, f'fused_{cable_name[:-5]}', 0, True)

        route_points_ret = sort_cable_points(Terminal_1, Terminal_2, fused_Vertices)

        # route_points_ret[0], route_points_ret[2] = route_points_ret[2], route_points_ret[0]

        # for k in range(len(route_points_ret)):
        #     show_points(route_points_ret, cable_name[:-5], k, True)

        route_points_ret = uniform_resample(route_points_ret, 1000)

        new_path = out_dir + '/{}_extracted.json'.format(cable_name[:-5])

        fused_Vertices_str = json.dumps(route_points_ret)
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(fused_Vertices_str)