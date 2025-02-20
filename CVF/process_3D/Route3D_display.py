import os
import random

import matplotlib.pyplot as plt
import json

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import math

from scipy.optimize import least_squares

X_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
alpha = (math.pi*230)/180
Z_rot = np.array([[math.cos(alpha), math.sin(alpha), 0], [math.sin(alpha), -math.cos(alpha), 0], [0, 0, 1]])

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    route3D_dir = "../../data/LAB_imgs_1028_DLO/route3D"
    route3D_list = os.listdir(route3D_dir)

    xx_list = []
    yy_list = []

    axis_x = 0
    axis_y = 0
    axis_z = 0
    cnt = 0

    for route3D_name in route3D_list:
        route3D_path = os.path.join(route3D_dir, route3D_name)
        with open(route3D_path, 'r', encoding='utf-8') as file:
            route3D = json.load(file)
        l = len(route3D)
        route3D = np.array(route3D)
        route3D = np.dot(np.dot(Z_rot, X_rot), route3D.T).T
        xx = route3D[:, 0]
        yy = route3D[:, 1]
        zz = route3D[:, 2]

        xx_list += list(xx)
        yy_list += list(yy)

        color = [(random.random(), random.random(), random.random())]
        colors = color * l
        ax.scatter(xx, yy, zz, c=colors)

    ax.view_init(elev=20, azim=45)
    plt.show()

    # 圆上的点的坐标
    x_data = np.array(xx_list)
    y_data = np.array(yy_list)

    # 定义一个函数，计算给定参数下的圆和数据点之间的误差
    def circle_residuals(params, x, y):
        xc, yc, r = params
        # 计算每个点到圆心的距离与半径之差的平方
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r


    # 初始参数猜测：圆心在(3, 3)，半径为2
    initial_params = [3, 3, 2]

    # 使用最小二乘法进行拟合
    result = least_squares(circle_residuals, initial_params, args=(x_data, y_data))

    # 输出拟合结果
    xc, yc, r = result.x
    print(xc, yc, r)

    fig2 = plt.figure()

    plt.scatter(x_data, y_data, label='Data')
    plt.plot([xc], [yc], label='Fit')
    plt.legend()
    plt.show()

    fig3 = plt.figure(figsize=(30, 10))

    for route3D_name in route3D_list:
        route3D_path = os.path.join(route3D_dir, route3D_name)
        with open(route3D_path, 'r', encoding='utf-8') as file:
            route3D = json.load(file)
        l = len(route3D)
        route3D = np.array(route3D)
        route3D = np.dot(np.dot(Z_rot, X_rot), route3D.T).T
        xx = route3D[:, 0]
        yy = route3D[:, 1]
        zz = route3D[:, 2]

        pp = []
        for i in range(l):
            x, y = xx[i]-xc, yy[i]-yc
            # 与Y轴的夹角
            alpha_y = math.atan((abs(x/y)))
            if x > 0 and y < 0:
                alpha_y = math.pi - alpha_y
            elif x < 0 and y < 0:
                alpha_y = math.pi + alpha_y
            elif x < 0 and y > 0:
                alpha_y = 2 * math.pi - alpha_y
            p = r * alpha_y
            pp.append(p)

        pp = np.array(pp)

        color = [(random.random(), random.random(), random.random())]
        colors = color * l
        plt.scatter(pp, zz, c=colors)

    plt.axis('off')
    plt.savefig("route.png")