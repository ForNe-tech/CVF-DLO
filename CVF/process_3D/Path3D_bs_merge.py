import ast
import math
import os
import sys
sys.path.append('../../..')
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import json
from scipy.interpolate import UnivariateSpline, splprep, splrep, splev, interp1d
import matplotlib.pyplot as plt

def read_path3D_file(dir, file):
    jsonpath = os.path.join(dir, file)
    f1 = open(jsonpath, 'r', encoding='utf-8')
    content = f1.read()
    if content.startswith(u'\ufeff'):
        content = content.encode('utf8')[3:].decode('utf8')
    path3D_dict = json.loads(content)
    return path3D_dict

def read_path_ends_file(dir, file):
    jsonpath = os.path.join(dir, file[:-5] + '_ends2D.json')
    f1 = open(jsonpath, 'r', encoding='utf-8')
    content = f1.read()
    if content.startswith(u'\ufeff'):
        content = content.encode('utf8')[3:].decode('utf8')
    ends_dict = json.loads(content)
    return ends_dict

def StringListToIntList(StringList):
    IntList = []
    for label in StringList:
        IntList.append(int(label))
    return IntList

def read_pose_file(dir, file):
    jsonpath = os.path.join(dir, file)
    f1 = open(jsonpath, 'r', encoding='utf-8')
    content = f1.read()
    if content.startswith(u'\ufeff'):
        content = content.encode('utf8')[3:].decode('utf8')
    pose_dict = json.loads(content)
    cameraToWorldMatrix = pose_dict["cameraToWorldMatrix"]
    pose_para_list = cameraToWorldMatrix.replace('\n', '\t').split('\t')[:-1]
    viewpoint = [float(pose_para_list[2]), float(pose_para_list[6]), float(pose_para_list[10])]
    vM = np.array(list(map(float, pose_para_list))).reshape((4, 4))
    projectionMatrix = pose_dict["projectionMatrix"]
    pM = np.array(list(map(float, projectionMatrix.replace('\n', '\t').split('\t')[:-1]))).reshape((4, 4))
    clip_points = [[-2 / pM[0, 0], -2 / pM[1, 1], -2, 1],
                   [-2 / pM[0, 0], 2 / pM[1, 1], -2, 1],
                   [2 / pM[0, 0], 2 / pM[1, 1], -2, 1],
                   [2 / pM[0, 0], -2 / pM[1, 1], -2, 1]]
    clip_points_wl = []
    for clip_point in clip_points:
        clip_point_wl = np.dot(vM, clip_point)
        clip_points_wl.append(clip_point_wl[:-1])
    return viewpoint, vM, clip_points_wl


def ConstructGlobalDict(Path_Dir, Path3D_Dir):

    path3D_files = os.listdir(Path3D_Dir)

    global_path3D_dict = {}
    global_path3D_label = 0
    global_ends_dict = {}
    global_ends_label = 0
    global_view_dict = {}

    for path3D_file in path3D_files:

        path3D_names = path3D_file.split('_')
        if path3D_names[1] == "bs3D.json":
            continue

        global_view_dict[path3D_names[0]] = []

        path3D_dict = read_path3D_file(Path3D_Dir, path3D_file)
        ends_dict = read_path_ends_file(Path_Dir, path3D_names[0] + '.json')

        path3D_label_list = StringListToIntList(list(path3D_dict.keys()))
        ends_label_list = StringListToIntList(list(ends_dict.keys()))
        global_path3D_label_list = [u for u in range(global_path3D_label, global_path3D_label + len(path3D_label_list))]
        global_ends_label_list = [v for v in range(global_ends_label, global_ends_label + len(ends_label_list))]
        global_path3D_label += len(path3D_label_list)
        global_ends_label += len(ends_label_list)

        global_path3D_convert = {}
        for i, path3D_label in enumerate(path3D_label_list):
            global_path3D_convert[path3D_label] = global_path3D_label_list[i]
        global_ends_convert = {}
        for j, ends_label in enumerate(ends_label_list):
            global_ends_convert[ends_label] = global_ends_label_list[j]

        for key, value in path3D_dict.items():
            global_key = global_path3D_convert[int(key)]
            global_path3D_dict[global_key] = {
                "route_ends": [],
                "border": False,
                "view": path3D_names[0],
                "path3D": value
            }
            global_view_dict[path3D_names[0]].append(global_key)

        for key, value in ends_dict.items():
            global_key = global_ends_convert[int(key)]
            global_pair_ends = []
            for pair_end in value["pair_ends"]:
                global_pair_ends.append(global_ends_convert[pair_end])
            global_ends_dict[global_key] = {
                "type": value["type"],
                "border": value["border"],
                "route_label": global_path3D_convert[value["route_label"]],
                "route_end": global_ends_convert[value["route_end"]],
                "pair_ends": global_pair_ends,
                "view": path3D_names[0]
            }
            global_path3D_dict[global_path3D_convert[value["route_label"]]]["route_ends"].append(global_key)
            if value["border"]:
                global_path3D_dict[global_path3D_convert[value["route_label"]]]["border"] = True

    return global_path3D_dict, global_ends_dict, global_view_dict

def close_view_set(Path_dir, Pose_Dir):
    pose_files = os.listdir(Pose_Dir)
    num = len(pose_files)
    viewpoints_dict = {}

    for i, pose_file in enumerate(pose_files):
        viewpoint, vM, clip_points = read_pose_file(Pose_Dir, pose_file)
        viewpoints_dict[i] = {
            "clip_points": clip_points,
            "vM": vM,
            "viewpoint": viewpoint,
            "view": pose_file[:-5]
        }

    distance_matrix = [[0 for i in range(num)] for j in range(num)]

    for i in range(num):
        viewpoint_i = viewpoints_dict[i]["viewpoint"]
        view_i = viewpoints_dict[i]["view"]
        for j in range(i+1, num):
            viewpoint_j = viewpoints_dict[j]["viewpoint"]
            view_j = viewpoints_dict[j]["view"]
            distance = calcdistance(viewpoint_i, viewpoint_j)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    close_view_dict = {}
    for i in range(num):
        distance_list = distance_matrix[i]
        indexed_list = list(enumerate(distance_list))
        indexed_list.sort(key=lambda x : x[1])
        samllest_indices = [index for index, value in indexed_list[1:7]]
        close_view_dict[viewpoints_dict[i]["view"]] = [viewpoints_dict[j]["view"] for j in samllest_indices]

    return close_view_dict

def cross_view_set(Pose_dir):
    pose_files = os.listdir(Pose_Dir)
    num = len(pose_files)
    viewpoints_dict = {}
    cross_view_dict = {}

    for i, pose_file in enumerate(pose_files):
        viewpoint, vM, clip_points = read_pose_file(Pose_Dir, pose_file)
        viewpoints_dict[i] = {
            "clip_points": clip_points,
            "vM": vM,
            "viewpoint": viewpoint,
            "view": pose_file[:-5]
        }
        cross_view_dict[pose_file[:-5]] = []

    for i in range(num):
        viewpoints_dict_i = viewpoints_dict[i]
        for j in range(i+1, num):
            viewpoints_dict_j = viewpoints_dict[j]
            if EverCross(viewpoints_dict_i, viewpoints_dict_j):
                cross_view_dict[viewpoints_dict_i["view"]].append(viewpoints_dict_j["view"])
                cross_view_dict[viewpoints_dict_j["view"]].append(viewpoints_dict_i["view"])

    print(1)



def EverCross(dict_i, dict_j):
    # Z方向的投影
    sta_i = dict_i["viewpoint"][:2]
    ray_i = []
    for k in range(4):
        end_i = dict_i["clip_points"][k][:2]
        ray_i.append(end_i - sta_i)
    sta_j = dict_j["viewpoint"][:2]
    ray_j = []
    for k in range(4):
        end_j = dict_j["clip_points"][k][:2]
        ray_j.append(end_j - sta_j)
    for i in range(4):
        for j in range(4):
            P1, D1, P2, D2 = sta_i, ray_i[i], sta_j, ray_j[j]
            if ray_intersection(P1, D1, P2, D2):
                return True
    # X方向上的投影
    sta_i = dict_i["viewpoint"][1:]
    ray_i = []
    for k in range(4):
        end_i = dict_i["clip_points"][k][1:]
        ray_i.append(end_i - sta_i)
    sta_j = dict_j["viewpoint"][1:]
    ray_j = []
    for k in range(4):
        end_j = dict_j["clip_points"][k][1:]
        ray_j.append(end_j - sta_j)
    for i in range(4):
        for j in range(4):
            P1, D1, P2, D2 = sta_i, ray_i[i], sta_j, ray_j[j]
            if ray_intersection(P1, D1, P2, D2):
                return True
    # Y方向上的投影
    sta_i = np.array([dict_i["viewpoint"][0], dict_i["viewpoint"][2]])
    ray_i = []
    for k in range(4):
        end_i = np.array([dict_i["clip_points"][k][0], dict_i["clip_points"][k][2]])
        ray_i.append(end_i - sta_i)
    sta_j = np.array([dict_j["viewpoint"][0], dict_j["viewpoint"][2]])
    ray_j = []
    for k in range(4):
        end_j = np.array([dict_j["clip_points"][k][0], dict_j["clip_points"][k][2]])
        ray_j.append(end_j - sta_j)
    for i in range(4):
        for j in range(4):
            P1, D1, P2, D2 = sta_i, ray_i[i], sta_j, ray_j[j]
            if ray_intersection(P1, D1, P2, D2):
                return True
    return False




def ray_intersection(P1, D1, P2, D2):
    determinant = D1[0] * D2[1] - D1[1] * D2[0]
    if determinant == 0:
        return False
    # 计算交点的参数t1和t2
    t1 = (P2[0] - P1[0]) * D2[1] - (P2[1] - P1[1]) * D2[0]
    t1 /= determinant
    t2 = (P2[0] - P1[0]) * D1[1] - (P2[1] - P1[1]) * D1[0]
    t2 /= -determinant
    if t1 >= 0 and t2 >= 0:
        return True
    else:
        return False


def calcdistance(point1, point2):
    dis = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
    return dis

def global_merge(global_path3D_dict, global_view_dict, close_view_dict, threshold=0.02):
    merge_pair_list = []
    view_pair_done = []
    view_list = list(global_view_dict.keys())
    for k, view_i in enumerate(view_list):
        for i in global_view_dict[view_i]:
            path3D_dict_i = global_path3D_dict[i]
            path3D_i = path3D_dict_i["path3D"]
            tree = KDTree(path3D_i)
            for view in close_view_dict[view_i]:
                if [view_i, view] not in view_pair_done:
                    for j in global_view_dict[view]:
                        path3D_dict_j = global_path3D_dict[j]
                        path3D_j = path3D_dict_j["path3D"]
                        distances, indices = tree.query(path3D_j)
                        close_points = list(np.where(distances < threshold)[0])
                        close_points_num = len(close_points)
                        if close_points_num > 6:
                            merge_pair_list.append([i, j])
        for view in close_view_dict[view_i]:
            view_pair_done.append([view_i, view])
            view_pair_done.append([view, view_i])

    # merge_group_list = []
    # for merge_pair in merge_pair_list:
    #     append_flag = True
    #     for merge_group in merge_group_list:
    #         if merge_pair[0] in merge_group:
    #             append_flag = False
    #             if merge_pair[1] not in merge_group:
    #                 merge_group.append(merge_pair[1])
    #             break
    #         elif merge_pair[1] in merge_group:
    #             append_flag = False
    #             if merge_pair[0] not in merge_group:
    #                 merge_group.append(merge_pair[0])
    #             break
    #     if append_flag:
    #         merge_group_list.append([merge_pair[0], merge_pair[1]])

    return merge_pair_list



def global_path3D_merge_old(global_path3D_dict, global_ends_dict, merge_group_list):
    for merge_group in merge_group_list:
        reserve_label = merge_group[0]
        remove_labels = merge_group[1:]
        for remove_label in remove_labels:
            merge_two_paths(reserve_label, remove_label, global_path3D_dict, global_ends_dict, -1)
    print(3)

def global_path3D_merge(global_path3D_dict, global_ends_dict, merge_pair_list):
    for i in range(len(merge_pair_list)):
        reserve_label = min(merge_pair_list[i])
        remove_label = max(merge_pair_list[i])
        if reserve_label == remove_label:
            continue
        merge_two_paths(reserve_label, remove_label, global_path3D_dict, global_ends_dict, i)
        for j in range(i+1, len(merge_pair_list)):
            if merge_pair_list[j][0] == remove_label:
                merge_pair_list[j][0] = reserve_label
            if merge_pair_list[j][1] == remove_label:
                merge_pair_list[j][1] = reserve_label
        print(2)


def merge_two_paths(label_1, label_2, global_path3D_dict, global_ends_dict, idx, threshold=0.02):
    path_dict_1 = global_path3D_dict[label_1]
    path_dict_2 = global_path3D_dict[label_2]
    path1 = path_dict_1["path3D"]
    path2 = path_dict_2["path3D"]
    route_ends_1 = path_dict_1["route_ends"]
    route_ends_2 = path_dict_2["route_ends"]
    if 138 in route_ends_1 or 138 in route_ends_2:
        print(1)
    route_ends_all = route_ends_1 + route_ends_2
    tree = KDTree(path1)
    distances, indices = tree.query(path2)
    close_indices = list(np.where(distances < threshold)[0])
    path2_start_point_close = close_indices[0]
    path2_finish_point_close = close_indices[-1]
    path1_start_point_close = indices[close_indices[0]][0]
    closest_indice_path2 = np.where(distances == min(distances))[0][0]
    closest_indice_path1 = indices[closest_indice_path2][0]

    path_merge = []
    route_ends = []
    end_pair_mode = 0
    if indices[path2_start_point_close][0] < indices[path2_finish_point_close][0]:
        end_pair_mode = 'start_to_start'
        if close_indices[0] > 0:
            route_ends.append(route_ends_2[0])
            if close_indices[-1] < len(path2) - 1:
                # end_pair_mode = 1
                route_ends.append(route_ends_1[1])
                path_merge = path2
            else:
                # end_pair_mode = 2
                route_ends.append(route_ends_1[1])
                path_merge = path2[:closest_indice_path2] + path1[closest_indice_path1:]
        else:
            route_ends.append(route_ends_1[0])
            if close_indices[-1] < len(path2) - 1:
                # end_pair_mode = 3
                route_ends.append(route_ends_2[1])
                path_merge = path1[:closest_indice_path1] + path2[closest_indice_path2:]
            else:
                # end_pair_mode = 4
                route_ends.append(route_ends_2[1])
                path_merge = path1
    else:
        end_pair_mode = 'start_to_end'
        if close_indices[-1] < len(path2) - 1:
            route_ends.append(route_ends_2[1])
            if close_indices[0] > 0:
                # end_pair_mode = 5
                route_ends.append(route_ends_1[1])
                path_merge = list(reversed(path2))
            else:
                # end_pair_mode = 6
                route_ends.append(route_ends_1[1])
                path_merge = list(reversed(path2[closest_indice_path2:])) + path1[closest_indice_path1:]
        else:
            route_ends.append(route_ends_1[0])
            if close_indices[0] > 0:
                # end_pair_mode = 7
                route_ends.append(route_ends_2[0])
                path_merge = path1[:closest_indice_path1] + list(reversed(path2[:closest_indice_path2]))
            else:
                # end_pair_mode = 8
                route_ends.append(route_ends_2[0])
                path_merge = path1

    show_path3D(path1, path2, path_merge, idx)
    print(1)

    global_path3D_dict[label_1]["path3D"] = path_merge
    global_path3D_dict[label_1]["route_ends"] = route_ends

    global_ends_dict[route_ends[0]]["route_label"] = label_1
    global_ends_dict[route_ends[0]]["route_end"] = route_ends[1]
    global_ends_dict[route_ends[1]]["route_label"] = label_1
    global_ends_dict[route_ends[1]]["route_end"] = route_ends[0]


    for end in route_ends_all:
        if end not in route_ends:
            del global_ends_dict[end]

    del global_path3D_dict[label_2]

    print(3)



def show_path3D(path1, path2, path3, idx):
    fig, axs = plt.subplots(1, 3)

    axs[2] = plt.subplot(1, 3, 3, projection='3d')
    axs[2].set_facecolor('white')
    # axs[2].set_axis_off()
    point_numpy = np.array(path3).T
    x_data = point_numpy[0]
    y_data = point_numpy[1]
    z_data = point_numpy[2]

    x_lim_len = max(x_data) - min(x_data)
    y_lim_len = max(y_data) - min(y_data)
    z_lim_len = max(z_data) - min(z_data)
    x_lim_min, x_lim_max = min(x_data) - x_lim_len / 10, max(x_data) + x_lim_len / 10
    y_lim_min, y_lim_max = min(y_data) - y_lim_len / 10, max(y_data) + y_lim_len / 10
    z_lim_min, z_lim_max = min(z_data) - z_lim_len / 10, max(z_data) + z_lim_len / 10
    axs[2].set_xlim3d(x_lim_min, x_lim_max)
    axs[2].set_ylim3d(y_lim_min, y_lim_max)
    axs[2].set_zlim3d(z_lim_min, z_lim_max)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_zticks([])

    axs[2].scatter(x_data, y_data, z_data, s=1, c='r')

    axs[0] = plt.subplot(1, 3, 1, projection='3d')
    axs[0].set_facecolor('white')
    # axs[0].set_axis_off()
    point_numpy_1 = np.array(path1).T
    x_data_1 = point_numpy_1[0]
    y_data_1 = point_numpy_1[1]
    z_data_1 = point_numpy_1[2]
    axs[0].set_xlim3d(x_lim_min, x_lim_max)
    axs[0].set_ylim3d(y_lim_min, y_lim_max)
    axs[0].set_zlim3d(z_lim_min, z_lim_max)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_zticks([])
    axs[0].scatter(x_data_1, y_data_1, z_data_1, s=1, c='b')

    axs[1] = plt.subplot(1, 3, 2, projection='3d')
    axs[1].set_facecolor('white')
    # axs[1].set_axis_off()
    point_numpy_2 = np.array(path2).T
    x_data_2 = point_numpy_2[0]
    y_data_2 = point_numpy_2[1]
    z_data_2 = point_numpy_2[2]
    axs[1].set_xlim3d(x_lim_min, x_lim_max)
    axs[1].set_ylim3d(y_lim_min, y_lim_max)
    axs[1].set_zlim3d(z_lim_min, z_lim_max)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_zticks([])
    axs[1].scatter(x_data_2, y_data_2, z_data_2, s=1, c='g')

    plt.savefig('merged_path3D/plot{}.png'.format(idx))
    print(1)

def SpaceCurveFromPoints(path1, path2, direction):

    path = []

    points = path1 + path2
    point_numpy = np.array(points).T
    x_data = point_numpy[0]
    y_data = point_numpy[1]
    z_data = point_numpy[2]

    xy_data = np.array([x_data, y_data]).T
    xy_data = np.array(sorted(xy_data, key=lambda x:x[0])).T

    x_unique = np.unique(xy_data[0])
    y_unique = np.array([xy_data[1][i] for i in np.searchsorted(x_unique, xy_data[0])])

    x_new = np.linspace(x_data.min(), x_data.max(), int(len(x_data) / 2))

    y_new = interp1d(x_unique, y_unique, kind='cubic')(x_new)

    plt.title("path2D")
    plt.scatter(xy_data[0], xy_data[1], 5, 'b', 'o')
    plt.scatter(x_new, y_new, 1, 'r', '^')
    plt.show()
    print(4)

    return path

def SpaceCurveFromPoints_Spline(path1, path2, direction):

    points = path1 + path2
    point_numpy = np.array(points).T
    x_data = point_numpy[0]
    y_data = point_numpy[1]
    z_data = point_numpy[2]

    xy_data = np.array([x_data, y_data]).T
    xy_data = np.array(sorted(xy_data, key=lambda x:x[0])).T
    spl_xy = UnivariateSpline(xy_data[0], xy_data[1], s=0.1)

    xz_data = np.array([x_data, z_data]).T
    xz_data = np.array(sorted(xz_data, key=lambda x: x[0])).T
    spl_xz = UnivariateSpline(xz_data[0], xz_data[1], s=0.1)

    x_new = np.linspace(x_data.min(), x_data.max(), int(len(points) / 2))
    y_new = spl_xy(x_new).tolist()
    z_new = spl_xz(x_new).tolist()
    x_new = x_new.tolist()

    path = []

    if direction:
        for i in range(len(x_new)):
            path.append((x_new[i], y_new[i], z_new[i]))
    else:
        for i in range(len(x_new)-1, 0, -1):
            path.append((x_new[i], y_new[i], z_new[i]))

    plt.title("path2D")
    plt.xlabel('X')
    plt.ylabel('YZ')
    plt.scatter(x_data, z_data, 1, 'b')
    plt.scatter(x_new, z_new, 1, 'r')
    plt.scatter(x_data, y_data, 1, 'g')
    plt.scatter(x_new, y_new, 1, 'y')
    plt.show()
    print(4)

    return path

def combineEndsRoutes_NoCircle(ends_dict):
    combEnds_list = []
    for i, end_dict in ends_dict.items():
        # 从没有配对端点的孤立端点出发
        if end_dict["type"] == 'iso':
            first_label = i

            def addMultiItemsList(baseList, addList):
                large_baseList = []
                for baselist in baseList:
                    for addlist in addList:
                        large_baseList.append(baselist + addlist)
                return large_baseList

            def getRouteEnd(end_label):
                next_label = ends_dict[end_label]['route_end']
                if ends_dict[next_label]['type'] == 'iso':
                    return [[next_label]]
                else:
                    return addMultiItemsList([[next_label]], getPairEnd(next_label))

            def getPairEnd(end_label):
                next_labels = ends_dict[end_label]['pair_ends']
                re_baseList = []
                for next_label in next_labels:
                    re_baseList += addMultiItemsList([[next_label]], getRouteEnd(next_label))
                return re_baseList

            combEnds_list += addMultiItemsList([[first_label]], getRouteEnd(first_label))

    return combEnds_list

def deleteCloseRoutes(ends_dict, combEnds_list):
    del_list = []
    have_existed = []
    for i, singleRoute in enumerate(combEnds_list):
        k = 0
        while k < len(singleRoute):
            k += 2
        if singleRoute[0] > singleRoute[-1]:
            singleRoute.reverse()
        if ends_dict[singleRoute[-1]]['type'] != 'iso' or ends_dict[singleRoute[0]]['type'] != 'iso':
            del_list.append(i)
        else:
            if singleRoute in have_existed:
                del_list.append(i)
            else:
                have_existed.append(singleRoute)
    del_list.reverse()
    for del_index in del_list:
        del combEnds_list[del_index]

    return combEnds_list

def drawRoutes(global_path3D_dict, global_ends_dict, combEnds_list):
    global_routes_dict = {}
    for i, singleRoute in enumerate(combEnds_list):
        global_route = []
        turn = True
        for k in range(len(singleRoute)-1):
            point_label = singleRoute[k]
            if turn:
                turn = False
                path_label = global_ends_dict[point_label]['route_label']
                path_ends = global_path3D_dict[path_label]['route_ends']
                if point_label == path_ends[0]:
                    global_route += global_path3D_dict[path_label]['path3D']
                else:
                    global_route += list(reversed(global_path3D_dict[path_label]['path3D']))
            else:
                turn = True
                next_label = singleRoute[k+1]
                last_path_label = global_ends_dict[point_label]['route_label']
                last_path_ends = global_path3D_dict[last_path_label]['route_ends']
                if point_label == last_path_ends[0]:
                    end_p1 = global_path3D_dict[last_path_label]['path3D'][0]
                else:
                    end_p1 = global_path3D_dict[last_path_label]['path3D'][-1]
                next_path_label = global_ends_dict[next_label]['route_label']
                next_path_ends = global_path3D_dict[next_path_label]['route_ends']
                if next_label == next_path_ends[0]:
                    end_p2 = global_path3D_dict[next_path_label]['path3D'][0]
                else:
                    end_p2 = global_path3D_dict[next_path_label]['path3D'][-1]
                connect_path = connect_two_points(end_p1, end_p2)
                global_route += connect_path
        global_routes_dict[i] = global_route
    return global_routes_dict

def connect_two_points(p1, p2):
    point1 = np.array([p1[0], p1[1], p1[2]])
    point2 = np.array([p2[0], p2[1], p2[2]])
    num_samples = 10
    t_values = np.linspace(0, 1, num=num_samples)
    connect_path = []
    for t_value in t_values:
        point = (1 - t_value) * point1 + t_value * point2
        connect_path.append((point[0], point[1], point[2]))
    return connect_path



if __name__ == "__main__":

    date = '1028'

    Path_Dir = '../../EWD/LAB_imgs_{}_DLO/path2D'.format(date)
    Path3D_Dir = '../../EWD/LAB_imgs_{}_DLO/path3D_sp_bs'.format(date)
    Pose_Dir = '../../EWD/LAB_imgs_{}_DLO/json'.format(date)
    Path3D_Merge_Dir = '../../EWD/LAB_imgs_{}_DLO/path3D_bs_merge'.format(date)

    os.makedirs(Path3D_Merge_Dir, exist_ok=True)

    # 读取三维path以及相应的端点字典
    global_path3D_dict, global_ends_dict, global_view_dict = ConstructGlobalDict(Path_Dir, Path3D_Dir)

    # 对于每个视角,寻找距离最近的N个视角寻找匹配对象
    close_view_dict = close_view_set(Path_Dir, Pose_Dir)

    # 计算需要合并的path对
    merge_pair_list = global_merge(global_path3D_dict, global_view_dict, close_view_dict)

    # path对合并
    global_path3D_merge(global_path3D_dict, global_ends_dict, merge_pair_list)

    # 合并后的三维Path散点保存
    for key, value in global_path3D_dict.items():
        file = open(os.path.join(Path3D_Merge_Dir, str(key) + '.json'), 'w')
        jsonstring = json.dumps(value["path3D"], ensure_ascii=False)
        file.write(jsonstring)
        file.close()


    # 路径整合
    combEnds_list = combineEndsRoutes_NoCircle(global_ends_dict)
    combEnds_list = deleteCloseRoutes(global_ends_dict, combEnds_list)

    # 三维路径绘制(端点对之间通过直线连接)
    global_routes_dict = drawRoutes(global_path3D_dict, global_ends_dict, combEnds_list)

    # 三维Route散点保存
    Route3D_Dir = '../../EWD/LAB_imgs_{}_DLO/route3D_bs'.format(date)
    os.makedirs(Route3D_Dir, exist_ok=True)

    for key, value in global_routes_dict.items():
        file = open(os.path.join(Route3D_Dir, str(key) + '.json'), 'w')
        jsonstring = json.dumps(value, ensure_ascii=False)
        file.write(jsonstring)
        file.close()

    print(2)
