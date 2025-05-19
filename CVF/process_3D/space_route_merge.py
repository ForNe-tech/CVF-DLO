import ast
import os
import sys
sys.path.append('../../..')
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

def read_sapce_route_file(dir, file):
    path = os.path.join(dir, file)
    f1 = open(path, 'r')
    pString = f1.read()
    pList = ast.literal_eval(pString)
    label_ = file[:-5]
    class_ = file.split('_')[0]
    return label_, class_, np.array(pList)

def search_for_connected_routes(routes_dict, route_label):

    routes_List = []
    existed_roots = []

    def addMultiItemsList(baseList, addList):
        large_baseList = []
        for baselist in baseList:
            for addlist in addList:
                large_baseList.append(baselist + addlist)
        return large_baseList

    def getRoutesList(route_label):
        route_root = route_label.split('_')[0]
        existed_roots.append(route_root)
        connnected_route_labels = routes_dict[route_root][route_label]["pair_routes"]
        connnected_route_labels_re = [item for item in connnected_route_labels if item.split('_')[0] not in existed_roots]
        re_List = []
        if len(connnected_route_labels_re) == 0:
            re_List = [[route_label]]
        for connnected_route_label in connnected_route_labels_re:
            re_List += addMultiItemsList([[route_label]], getRoutesList(connnected_route_label))
        return re_List

    routes_List = getRoutesList(route_label)
    routes_List = clean_short_routes(routes_List)
    return routes_List

def clean_short_routes(routes_List):
    routes_List_clean = []
    for route in tqdm(routes_List):
        set_route = set(route)
        if len(routes_List_clean) == 0:
            routes_List_clean.append(route)
            continue
        routes_List_clean_temp = routes_List_clean.copy()
        append_flag = True
        routes_List_remove = []
        for route_clean in routes_List_clean_temp:
            set_route_clean = set(route_clean)
            if set_route.issubset(set_route_clean):
                append_flag = False
                break
            elif set_route_clean.issubset(set_route):
                routes_List_remove.append(route_clean)
        if append_flag:
            for route_remove in routes_List_remove:
                routes_List_clean.remove(route_remove)
            routes_List_clean.append(route)
    return routes_List_clean



if __name__ == "__main__":

    Space_Route_Dir = '../../EWD/LAB_imgs_1028_DLO/route3D'
    space_route_files = os.listdir(Space_Route_Dir)

    space_route_dict = {}

    for space_route_file in space_route_files:
        space_route_label, space_route_root, space_route_points = read_sapce_route_file(Space_Route_Dir, space_route_file)  # label是线缆的编号,root是图像的编号
        if space_route_root not in space_route_dict:
            space_route_dict[space_route_root] = {}
        space_route_dict[space_route_root][space_route_label] = {"points": space_route_points, "pair_routes": []}

    print(1)

    threshold = 0.01

    for space_route_root_1 in tqdm(space_route_dict.keys()):
        for space_route_label_1 in space_route_dict[space_route_root_1].keys():
            point_set_1 = space_route_dict[space_route_root_1][space_route_label_1]["points"]
            # print("For match: " + space_route_label_1)
            tree = KDTree(point_set_1)
            for space_route_root_2 in space_route_dict.keys():
                pair_route_label = ''
                close_points_num_max = 0
                if space_route_root_2 == space_route_root_1:
                    continue
                for space_route_label_2 in space_route_dict[space_route_root_2].keys():
                    if space_route_label_2 not in space_route_dict[space_route_root_1][space_route_label_1]["pair_routes"] and \
                       space_route_label_1 not in space_route_dict[space_route_root_2][space_route_label_2]["pair_routes"]:
                        point_set_2 = space_route_dict[space_route_root_2][space_route_label_2]["points"]
                        distances, indices = tree.query(point_set_2)
                        close_points = list(np.where(distances < threshold)[0])
                        close_points_num = len(close_points)
                        if close_points_num > 100 and close_points_num > close_points_num_max:
                            # print("match: " + space_route_label_2)
                            pair_route_label = space_route_label_2
                            close_points_num_max = close_points_num
                if pair_route_label != '':
                    space_route_dict[space_route_root_1][space_route_label_1]["pair_routes"].append(pair_route_label)
                    space_route_dict[space_route_root_2][pair_route_label]["pair_routes"].append(space_route_label_1)

    possible_space_routes = []
    possible_space_routes = search_for_connected_routes(space_route_dict, "temp0_0")

    print(2)