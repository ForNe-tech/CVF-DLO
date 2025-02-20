import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import json
import os

def calcdistance3D(point1, point2):
    dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return dist

def calcdistanceZ(point1, point2):
    dist_z = abs(point1[2] - point2[2])
    return dist_z

def smoothPath3D():
    pathdir = '../../data/LAB_imgs_1028_DLO/path3D_sp_cam/'

    scene_list = [''] + [str(i) for i in range(21)]

    for scene_label in scene_list:

        scene_label = '20'

        jsonpath = pathdir + 'temp{}.json'.format(scene_label)

        if not os.path.exists(jsonpath):
            continue

        with open(jsonpath, 'r', encoding='utf-8') as file:
            path_dict = json.load(file)

        for label, path_ in zip(path_dict.keys(), path_dict.values()):

            path_list = []
            for point_str in path_:
                point = list(map(float, point_str[1:-1].split(',')))
                path_list.append(point)

            n = len(path_list)

            for i in range(n - 1):
                dist = calcdistanceZ(path_list[i], path_list[i+1])
                print(dist)

            path = np.array(path_list)
            path_x = path[:, 0]
            path_z = path[:, 2]
            fig = plt.figure()
            plt.scatter(path_x, path_z)
            plt.show()

            print(1)

if __name__ == "__main__":
    smoothPath3D()