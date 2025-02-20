import os
import cv2
import random
import math
import numpy as np


def calcEllipseFromEnds(end_pair, ends_dict):
    p1 = ends_dict[end_pair[0]]['point']
    p2 = ends_dict[end_pair[1]]['point']
    dir_p1 = ends_dict[end_pair[0]]['dir']
    dir_p2 = ends_dict[end_pair[1]]['dir']
    x1 = p1[1]
    y1 = p1[0]
    x2 = p2[1]
    y2 = p2[0]
    a1 = dir_p1[0]
    a2 = dir_p2[0]
    b1 = dir_p1[1]
    b2 = dir_p2[1]
    x0 = (a1 * b2 * x1 + b2 * b1 * y1 - a2 * b1 * x2 - b1 * b2 * y2) / (a1 * b2 - a2 * b1)
    y0 = (a1 * a2 * x1 + a2 * b1 * y1 - a1 * a2 * x2 - a1 * b2 * y2) / (a2 * b1 - a1 * b2)
    R = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
    cos_angle1 = (x1 - x0) / R
    cos_angle2 = (x2 - x0) / R
    angle1 = (180 * (math.acos(cos_angle1)/math.pi)) if y1 > y0 else (360 - 180 * (math.acos(cos_angle1)/math.pi))
    angle2 = (180 * (math.acos(cos_angle2)/math.pi)) if y2 > y0 else (360 - 180 * (math.acos(cos_angle2)/math.pi))
    start_angle = min(angle1, angle2)
    end_angle = max(angle1, angle2)
    return (round(x0), round(y0)), round(R), start_angle, end_angle

def draw_drac(img, center, R, color, start_angle, end_angle, thickness):
    '''
    img:	2d numpy such as (512,512)
    todo:   draw drac  2d img randomly
    '''
    '''
    img表示输入的图像
    center表示椭圆圆心坐标
    axes表示椭圆长轴和短轴的长度(为半轴长），输入参数时如此表示：(long，shor)
    angle表示主轴（长轴）偏转角度
    start_angle表示圆弧起始角度
    end_angle表示圆弧终结角度
    color表示线条颜色，为BGR形式，如蓝色为（255，0，0）
    thickness为非负数时表示线条的粗细程度，否则表示椭圆被填充
    lineType表示线条的类型，默认为LINE_8,可直接用8表示，另外还有LINE_4和LINE_AA
    shift表示圆心坐标点和数轴的精度，默认为0
    '''
    rot_angle_degree = 0
    axes = (R, R)  # long & short
    shift = 3
    cv2.ellipse(img, center, axes, rot_angle_degree, start_angle, end_angle, color, thickness=thickness, shift=shift)
    cv2.imwrite('data/debug_results/route_test/temp_cross.jpg', img)
    return img


if __name__ == "__main__":

    img = np.zeros((360, 640))
    img = draw_drac(img)
    cv2.imshow('drac', img)
    cv2.waitKey(0)

